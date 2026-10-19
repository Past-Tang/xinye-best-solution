from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, WhisperFeatureExtractor, WhisperModel


class AudioEncoder(nn.Module):
    def __init__(self, sample_rate: int, n_mels: int, conv_channels: List[int], dropout: float):
        super().__init__()
        self.register_buffer("_log_clamp_min", torch.tensor(1e-4), persistent=False)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self._mel_transform = None

        c1, c2, c3 = conv_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(2, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.out_dim = c3

    def _ensure_mel(self, device: torch.device):
        if self._mel_transform is None:
            import torchaudio
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_mels=self.n_mels,
                n_fft=1024, hop_length=320, win_length=1024,
            )
        self._mel_transform = self._mel_transform.to(device)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        self._ensure_mel(wave.device)
        bsz, chans, _ = wave.shape
        mel_list = []
        for c in range(chans):
            with torch.cuda.amp.autocast(enabled=False):
                m = self._mel_transform(wave[:, c, :].float())
                m = torch.clamp(m, min=float(self._log_clamp_min.item()))
                m = torch.log(m)
            mel_list.append(m)
        mel = torch.stack(mel_list, dim=1)
        return self.encoder(mel)


# ---------------------------------------------------------------------------
# Learnable attention pooling: attend to a subset of time steps
# ---------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """Single-head attention pooling over a sequence dimension."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.scale = hidden_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        scores = (self.query * x).sum(dim=-1) * self.scale  # [B, T]
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        return (x * weights).sum(dim=1)  # [B, D]


class WhisperAudioEncoder(nn.Module):
    def __init__(
        self, model_name: str, sample_rate: int, proj_dim: int,
        freeze: bool = True, tail_ratio: float = 0.2,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.freeze = freeze
        self.tail_ratio = tail_ratio
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.encoder = WhisperModel.from_pretrained(model_name).encoder
        if self.freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        hidden_size = int(self.encoder.config.d_model)
        self.attn_pool = AttentionPooling(hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.out_dim = proj_dim

    def _build_input_features(self, wave: torch.Tensor) -> torch.Tensor:
        mono = wave.mean(dim=1)
        mono_np = mono.detach().float().cpu().numpy()
        inputs = self.feature_extractor(
            [x for x in mono_np],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        return inputs["input_features"]

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            input_features = self._build_input_features(wave).to(wave.device)

        if self.freeze:
            with torch.no_grad():
                hidden = self.encoder(input_features=input_features).last_hidden_state
        else:
            hidden = self.encoder(input_features=input_features).last_hidden_state

        # Only attend to the tail portion of the time axis
        T = hidden.shape[1]
        tail_start = max(0, T - int(T * self.tail_ratio))
        tail_hidden = hidden[:, tail_start:, :]  # [B, tail_T, D]
        pooled = self.attn_pool(tail_hidden)
        return self.proj(pooled)


class ContextLabelEncoder(nn.Module):
    """Encode context label sequence with strong tail-awareness."""
    def __init__(self, vocab_size: int, embed_dim: int, channels: List[int],
                 tail_k: int = 50):
        super().__init__()
        c1, c2 = channels
        self.tail_k = tail_k
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Tail branch: only last K chunks → richer conv + flatten (no global pool)
        self.tail_conv = nn.Sequential(
            nn.Conv1d(embed_dim, c1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.tail_proj = nn.Linear(c2 * tail_k, c2)

        # Full branch: whole sequence → conv + attention pool
        self.full_conv = nn.Sequential(
            nn.Conv1d(embed_dim, c1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.full_attn_pool = AttentionPooling(c2)

        self.out_dim = c2 * 2  # tail + full concatenated

    def forward(self, context_labels: torch.Tensor) -> torch.Tensor:
        x = self.embedding(context_labels).transpose(1, 2)  # [B, E, L]

        # Tail branch
        tail_x = x[:, :, -self.tail_k:]  # [B, E, K]
        tail_feat = self.tail_conv(tail_x)  # [B, c2, K]
        tail_feat = self.tail_proj(tail_feat.flatten(1))  # [B, c2]

        # Full branch with attention pooling
        full_feat = self.full_conv(x)  # [B, c2, L]
        full_feat = self.full_attn_pool(full_feat.transpose(1, 2))  # [B, c2]

        return torch.cat([tail_feat, full_feat], dim=-1)  # [B, c2*2]


class HandcraftedFeatures(nn.Module):
    """Compute hand-crafted statistics from context labels."""
    def __init__(self, num_labels: int = 5, context_chunks: int = 375):
        super().__init__()
        self.num_labels = num_labels
        self.context_chunks = context_chunks
        self.out_dim = num_labels * 3 + 4  # 3 windows * 5 ratios + 4 extra

    def forward(self, context_labels: torch.Tensor) -> torch.Tensor:
        B, L = context_labels.shape
        device = context_labels.device
        one_hot = F.one_hot(context_labels.long(), self.num_labels).float()  # [B, L, 5]

        tail25 = one_hot[:, -25:, :].mean(dim=1)    # [B, 5]
        tail50 = one_hot[:, -50:, :].mean(dim=1)    # [B, 5]
        tail100 = one_hot[:, -100:, :].mean(dim=1)  # [B, 5]

        # Distance to last event (T=1, BC=2, I=3)
        event_mask = (context_labels == 1) | (context_labels == 2) | (context_labels == 3)
        indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        event_positions = torch.where(event_mask, indices, torch.zeros_like(indices))
        last_event_pos = event_positions.max(dim=1).values  # [B]
        has_event = event_mask.any(dim=1).float()
        dist_to_last = ((L - 1 - last_event_pos).float() / L) * has_event + (1.0 - has_event)

        # Last 3 raw labels normalized
        last1 = context_labels[:, -1].float() / (self.num_labels - 1)
        last2 = context_labels[:, -2].float() / (self.num_labels - 1) if L > 1 else torch.zeros(B, device=device)
        last3 = context_labels[:, -3].float() / (self.num_labels - 1) if L > 2 else torch.zeros(B, device=device)

        return torch.cat([
            tail25, tail50, tail100,
            dist_to_last.unsqueeze(1),
            last1.unsqueeze(1), last2.unsqueeze(1), last3.unsqueeze(1),
        ], dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, freeze_backbone: bool = True,
                 tail_ratio: float = 0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.out_dim = int(self.backbone.config.hidden_size)
        self.tail_ratio = tail_ratio
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.attn_pool = AttentionPooling(self.out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, L, H]

        # Focus on the tail portion of the sequence (later utterances)
        L = hidden.shape[1]
        tail_start = max(0, L - int(L * self.tail_ratio))
        tail_hidden = hidden[:, tail_start:, :]
        tail_mask = attention_mask[:, tail_start:].unsqueeze(-1).to(hidden.dtype)
        masked_hidden = tail_hidden * tail_mask
        pooled = self.attn_pool(masked_hidden)
        return pooled


class MultimodalTurnTakingModel(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        # Baseline 固化为 event-level 多标签（未来 2s 窗口内各标签是否出现）
        audio_type = str(cfg["audio_encoder"].get("type", "cnn")).lower()
        if audio_type == "whisper":
            self.audio_encoder = WhisperAudioEncoder(
                model_name=cfg["audio_encoder"]["model_name"],
                sample_rate=cfg["sample_rate"],
                proj_dim=int(cfg["audio_encoder"]["proj_dim"]),
                freeze=bool(cfg["audio_encoder"].get("freeze", True)),
                tail_ratio=float(cfg["audio_encoder"].get("tail_ratio", 0.2)),
            )
        else:
            self.audio_encoder = AudioEncoder(
                sample_rate=cfg["sample_rate"],
                n_mels=cfg["audio_encoder"]["n_mels"],
                conv_channels=cfg["audio_encoder"]["conv_channels"],
                dropout=cfg["audio_encoder"]["dropout"],
            )
        self.text_encoder = TextEncoder(
            model_name=cfg["text_encoder"]["model_name"],
            freeze_backbone=bool(cfg["text_encoder"].get("freeze_backbone", True)),
            tail_ratio=float(cfg["text_encoder"].get("tail_ratio", 0.3)),
        )

        ctx_cfg = cfg["context_encoder"]
        self.context_encoder = ContextLabelEncoder(
            vocab_size=ctx_cfg["vocab_size"],
            embed_dim=ctx_cfg["embed_dim"],
            channels=ctx_cfg["channels"],
            tail_k=int(ctx_cfg.get("tail_k", 50)),
        )

        self.hand_features = HandcraftedFeatures(
            num_labels=ctx_cfg["vocab_size"],
            context_chunks=int(cfg["context_chunks"]),
        )

        fusion_in = (
            self.audio_encoder.out_dim
            + self.text_encoder.out_dim
            + self.context_encoder.out_dim
            + self.hand_features.out_dim
        )
        h1, h2 = cfg["fusion_head"]["hidden_dims"]
        dropout = cfg["fusion_head"]["dropout"]
        num_targets = len(cfg.get("labels", {}).get("multi_targets", []))
        self.num_targets = num_targets if num_targets > 0 else 1
        self.head = nn.Sequential(
            nn.Linear(fusion_in, h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, self.num_targets),
        )

    def forward(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_labels: torch.Tensor,
    ) -> torch.Tensor:
        audio_feat = self.audio_encoder(waveform)
        text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        context_feat = self.context_encoder(context_labels=context_labels)
        hand_feat = self.hand_features(context_labels)
        fusion = torch.cat([audio_feat, text_feat, context_feat, hand_feat], dim=-1)
        logits = self.head(fusion)
        if self.num_targets == 1:
            return logits.squeeze(-1)
        return logits
