"""Dual-channel Multimodal Turn-Taking Model.

与 `multimodal_baseline.py` 同级、独立、不互相依赖。

核心架构改动 (vs baseline):
1. WhisperAudioEncoder 接收 wave[B, 2, T]，对每个 channel 独立做 feature_extractor + Whisper encoder
   （Whisper encoder 权重共享，default frozen）。
2. 每个 channel 独立 tail attention pool 得 [B, D]。
3. 轻量 cross-attention：q=ch0_pool, k=v=ch1_tail_hidden -> ch0_cross (和对称 ch1_cross)
4. concat([ch0_pool, ch1_pool, ch0_cross, ch1_cross]) -> Linear(proj_dim)
5. 默认去掉 HandcraftedFeatures (Phase A 证明 -0.0008 退化)

其他模块复用 baseline 的 TextEncoder / ContextLabelEncoder。

Author: Cascade (Phase B1)
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, WhisperFeatureExtractor, WhisperModel


class AttentionPooling(nn.Module):
    """Single-head attention pooling over a sequence dimension."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.scale = hidden_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = (self.query * x).sum(dim=-1) * self.scale  # [B, T]
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        return (x * weights).sum(dim=1)  # [B, D]


class DualChannelWhisperAudioEncoder(nn.Module):
    """Dual-channel Whisper encoder with shared backbone and cross-attention.

    Forward 流程：
        wave [B,2,T] -> (feature_extractor per channel) -> mel [B,80,3000] x2
        -> shared WhisperModel.encoder -> hidden [B, T', D] x2
        -> tail slice & symmetric cross-attention -> 4 vectors of size D
        -> concat & proj -> [B, proj_dim]
    """

    def __init__(
        self,
        model_name: str,
        sample_rate: int,
        proj_dim: int,
        freeze: bool = True,
        tail_ratio: float = 0.2,
        cross_attn_heads: int = 4,
        use_gpu_mel: bool = True,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.freeze = freeze
        self.tail_ratio = tail_ratio
        self.use_gpu_mel = use_gpu_mel
        self.use_cross_attn = use_cross_attn
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.encoder = WhisperModel.from_pretrained(model_name).encoder
        if self.freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        hidden_size = int(self.encoder.config.d_model)
        self.hidden_size = hidden_size

        # GPU-side mel filterbank (aligned with HuggingFace WhisperFeatureExtractor)
        # mel_filters: [n_freqs(=n_fft//2+1), n_mels] -> register as buffer
        import numpy as _np
        mel_filters = torch.from_numpy(_np.asarray(self.feature_extractor.mel_filters, dtype=_np.float32))
        # transpose to [n_mels, n_freqs] for matmul: filters @ magnitudes
        self.register_buffer("_mel_filters", mel_filters.T.contiguous(), persistent=False)
        self._n_fft = int(self.feature_extractor.n_fft)
        self._hop_length = int(self.feature_extractor.hop_length)
        self._chunk_length = int(self.feature_extractor.chunk_length)  # 30s
        self._n_samples = self._chunk_length * sample_rate  # 480000
        self._nb_max_frames = int(self.feature_extractor.nb_max_frames)  # 3000
        # Hann window
        self.register_buffer("_window", torch.hann_window(self._n_fft), persistent=False)

        # 独立 attn-pool（两个 channel 用不同 query 学习"主动 vs 被动"）
        self.attn_pool_ch0 = AttentionPooling(hidden_size)
        self.attn_pool_ch1 = AttentionPooling(hidden_size)

        if self.use_cross_attn:
            # 对称的 cross-attention：每个方向独立
            self.cross_attn_0to1 = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=cross_attn_heads,
                batch_first=True,
            )
            self.cross_attn_1to0 = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=cross_attn_heads,
                batch_first=True,
            )
            self.cross_norm_0 = nn.LayerNorm(hidden_size)
            self.cross_norm_1 = nn.LayerNorm(hidden_size)
            proj_in = hidden_size * 4  # pool0, pool1, cross_0to1, cross_1to0
        else:
            proj_in = hidden_size * 2  # 仅 pool0, pool1

        self.proj = nn.Sequential(
            nn.Linear(proj_in, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.out_dim = proj_dim

    def _build_input_features_dual(self, wave: torch.Tensor) -> torch.Tensor:
        """wave: [B, 2, T] -> mel features [2B, n_mels, n_frames].

        - GPU path (default): all on GPU using torch.stft + cached mel filters.
        - CPU fallback: HuggingFace WhisperFeatureExtractor (slower).
        """
        B, _, T = wave.shape
        flat = wave.reshape(B * 2, T)  # [2B, T]
        if self.use_gpu_mel:
            return self._gpu_log_mel(flat)
        flat_np = flat.detach().float().cpu().numpy()
        inputs = self.feature_extractor(
            [x for x in flat_np],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        return inputs["input_features"]  # [2B, n_mels, 3000]

    def _gpu_log_mel(self, wave: torch.Tensor) -> torch.Tensor:
        """GPU-native Whisper-style log-mel spectrogram.

        Args:
            wave: [N, T] (length T = 30s * sr or shorter; will be padded/truncated)
        Returns:
            log_mel: [N, n_mels, nb_max_frames]
        """
        x = wave.float()
        N, T = x.shape
        # Pad / truncate to exactly chunk_length seconds
        if T < self._n_samples:
            x = F.pad(x, (0, self._n_samples - T))
        elif T > self._n_samples:
            x = x[:, : self._n_samples]
        # STFT (no center, matches Whisper)
        # HuggingFace WhisperFeatureExtractor uses center=True by default - check:
        # In HF impl, _np_extract_fbank_features pads waveform with N_FFT//2 each side then no centering.
        # Equivalent to torch.stft with center=True.
        stft = torch.stft(
            x, n_fft=self._n_fft, hop_length=self._hop_length,
            window=self._window.to(x.device), return_complex=True, center=True,
        )  # [N, n_freqs, frames]
        magnitudes = stft[..., :-1].abs() ** 2  # drop last frame to match Whisper (-> 3000 frames)
        mel_spec = self._mel_filters.to(magnitudes.device).to(magnitudes.dtype) @ magnitudes  # [N, n_mels, frames]
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        # Whisper-style normalization: cap max-min spread, then rescale to [0,1]-ish range
        log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-1, -2), keepdim=True) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        # Pad/truncate frames to nb_max_frames
        F_t = log_spec.shape[-1]
        if F_t < self._nb_max_frames:
            log_spec = F.pad(log_spec, (0, self._nb_max_frames - F_t))
        elif F_t > self._nb_max_frames:
            log_spec = log_spec[..., : self._nb_max_frames]
        return log_spec

    def _encode_dual(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One-shot dual-channel encode.

        Args:
            wave: [B, 2, T]
        Returns:
            (hidden_ch0, hidden_ch1) each [B, T', D]
        """
        B = wave.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            input_features = self._build_input_features_dual(wave).to(wave.device)
        if self.freeze:
            with torch.no_grad():
                hidden = self.encoder(input_features=input_features).last_hidden_state
        else:
            hidden = self.encoder(input_features=input_features).last_hidden_state
        # hidden: [2B, T', D] -> [B, 2, T', D]
        hidden = hidden.reshape(B, 2, hidden.shape[1], hidden.shape[2])
        return hidden[:, 0], hidden[:, 1]

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        # wave: [B, 2, T]
        assert wave.dim() == 3 and wave.shape[1] == 2, (
            f"DualChannelWhisperAudioEncoder expects wave [B,2,T], got {wave.shape}"
        )
        hidden_ch0, hidden_ch1 = self._encode_dual(wave)  # [B, T', D] x 2

        T = hidden_ch0.shape[1]
        tail_start = max(0, T - int(T * self.tail_ratio))
        tail_ch0 = hidden_ch0[:, tail_start:, :]  # [B, tT, D]
        tail_ch1 = hidden_ch1[:, tail_start:, :]

        pool_ch0 = self.attn_pool_ch0(tail_ch0)  # [B, D]
        pool_ch1 = self.attn_pool_ch1(tail_ch1)  # [B, D]

        if self.use_cross_attn:
            # 对称 cross-attention: query = self pool, kv = other channel tail
            q0 = pool_ch0.unsqueeze(1)  # [B, 1, D]
            q1 = pool_ch1.unsqueeze(1)
            cross_0to1, _ = self.cross_attn_0to1(q0, tail_ch1, tail_ch1)  # [B, 1, D]
            cross_1to0, _ = self.cross_attn_1to0(q1, tail_ch0, tail_ch0)
            cross_0to1 = self.cross_norm_0(cross_0to1.squeeze(1) + pool_ch0)
            cross_1to0 = self.cross_norm_1(cross_1to0.squeeze(1) + pool_ch1)
            feat = torch.cat([pool_ch0, pool_ch1, cross_0to1, cross_1to0], dim=-1)
        else:
            feat = torch.cat([pool_ch0, pool_ch1], dim=-1)
        return self.proj(feat)


class ContextLabelEncoder(nn.Module):
    """与 baseline 完全一致的 context encoder。"""

    def __init__(self, vocab_size: int, embed_dim: int, channels: List[int],
                 tail_k: int = 50):
        super().__init__()
        c1, c2 = channels
        self.tail_k = tail_k
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.tail_conv = nn.Sequential(
            nn.Conv1d(embed_dim, c1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.tail_proj = nn.Linear(c2 * tail_k, c2)
        self.full_conv = nn.Sequential(
            nn.Conv1d(embed_dim, c1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.full_attn_pool = AttentionPooling(c2)
        self.out_dim = c2 * 2

    def forward(self, context_labels: torch.Tensor) -> torch.Tensor:
        x = self.embedding(context_labels).transpose(1, 2)
        tail_x = x[:, :, -self.tail_k:]
        tail_feat = self.tail_conv(tail_x)
        tail_feat = self.tail_proj(tail_feat.flatten(1))
        full_feat = self.full_conv(x)
        full_feat = self.full_attn_pool(full_feat.transpose(1, 2))
        return torch.cat([tail_feat, full_feat], dim=-1)


class TextEncoder(nn.Module):
    """与 baseline 完全一致的 text encoder。"""

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
        hidden = out.last_hidden_state
        L = hidden.shape[1]
        tail_start = max(0, L - int(L * self.tail_ratio))
        tail_hidden = hidden[:, tail_start:, :]
        tail_mask = attention_mask[:, tail_start:].unsqueeze(-1).to(hidden.dtype)
        masked_hidden = tail_hidden * tail_mask
        pooled = self.attn_pool(masked_hidden)
        return pooled


class MultimodalTurnTakingModelDualCh(nn.Module):
    """Dual-channel multimodal model. Drops handcrafted features by default."""

    def __init__(self, cfg: Dict):
        super().__init__()
        audio_cfg = cfg["audio_encoder"]
        self.audio_encoder = DualChannelWhisperAudioEncoder(
            model_name=audio_cfg["model_name"],
            sample_rate=cfg["sample_rate"],
            proj_dim=int(audio_cfg["proj_dim"]),
            freeze=bool(audio_cfg.get("freeze", True)),
            tail_ratio=float(audio_cfg.get("tail_ratio", 0.2)),
            cross_attn_heads=int(audio_cfg.get("cross_attn_heads", 4)),
            use_cross_attn=bool(audio_cfg.get("use_cross_attn", True)),
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
        # Phase A 证明 handcrafted ≈ 死参数，默认去掉
        self.use_hand = bool(cfg.get("use_hand_features", False))
        if self.use_hand:
            from .multimodal_baseline import HandcraftedFeatures  # 仅作为可选
            self.hand_features = HandcraftedFeatures(
                num_labels=ctx_cfg["vocab_size"],
                context_chunks=int(cfg["context_chunks"]),
            )
            hand_dim = self.hand_features.out_dim
        else:
            self.hand_features = None
            hand_dim = 0

        fusion_in = (
            self.audio_encoder.out_dim
            + self.text_encoder.out_dim
            + self.context_encoder.out_dim
            + hand_dim
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
        feats = [audio_feat, text_feat, context_feat]
        if self.use_hand:
            feats.append(self.hand_features(context_labels))
        fusion = torch.cat(feats, dim=-1)
        logits = self.head(fusion)
        if self.num_targets == 1:
            return logits.squeeze(-1)
        return logits
