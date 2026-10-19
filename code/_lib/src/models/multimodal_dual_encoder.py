"""Dual Encoder Temporal Model: WavLM (paralinguistic) + Qwen2AudioEncoder (semantic/emotion).

Replaces Whisper in TemporalAlignedModel with two complementary encoders:
- WavLM Large: speaker diarization, prosody, emotion (317M, frozen)
- Qwen2AudioEncoder: semantic understanding, emotion, sound events (637M, frozen)

Both encoders process dual-channel audio independently, then features are
concatenated and fed into a GRU temporal engine.

Author: Cascade (2026-05-26)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, Wav2Vec2FeatureExtractor


class WavLMDualChannelEncoder(nn.Module):
    """WavLM encoder processing both audio channels independently, then concatenating.

    Unlike Whisper, WavLM directly accepts raw waveform (16kHz).
    Output: [B, 1500, 1024] for 30s audio (50Hz frame rate).
    Dual-channel output: [B, 1500, 2048].
    """

    def __init__(self, model_path: str, freeze: bool = True):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_path)
        self.hidden_size = self.wavlm.config.hidden_size  # 1024
        if freeze:
            for p in self.wavlm.parameters():
                p.requires_grad = False

    def _encode_channel(self, wave_ch: torch.Tensor) -> torch.Tensor:
        """wave_ch: [B, T_wav] mono 16kHz → hidden [B, T_frames, 1024]"""
        out = self.wavlm(wave_ch)
        hidden = out.last_hidden_state  # [B, ~1499, 1024]
        # Ensure frame count is divisible by 4 for Conv1d(stride=4)
        T = hidden.shape[1]
        remainder = T % 4
        if remainder != 0:
            pad_frames = 4 - remainder
            hidden = F.pad(hidden, (0, 0, 0, pad_frames))
        return hidden

    def forward(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wave: [B, 2, T] → (ch0_hidden, ch1_hidden) each [B, T_frames, 1024]"""
        B = wave.shape[0]
        with torch.no_grad():
            h0 = self._encode_channel(wave[:, 0, :])  # [B, T0, 1024]
            h1 = self._encode_channel(wave[:, 1, :])  # [B, T1, 1024]
        # Pad to same length
        max_frames = max(h0.shape[1], h1.shape[1])
        if h0.shape[1] < max_frames:
            h0 = F.pad(h0, (0, 0, 0, max_frames - h0.shape[1]))
        if h1.shape[1] < max_frames:
            h1 = F.pad(h1, (0, 0, 0, max_frames - h1.shape[1]))
        return h0, h1


class Qwen2AudioDualChannelEncoder(nn.Module):
    """Qwen2Audio encoder processing both audio channels.

    Uses the standalone Qwen2AudioEncoder weights extracted from Qwen2-Audio-7B.
    Mel spectrogram computed via HuggingFace Qwen2AudioProcessor (128 bins, correct params).
    Output: [B, ~750, 1280] for 30s audio (25Hz frame rate, Whisper-based).
    Dual-channel output: [B, 750, 2560].
    """

    def __init__(self, encoder_weights_path: str, config_dir: str, freeze: bool = True):
        super().__init__()
        from transformers import Qwen2AudioEncoder, Qwen2AudioEncoderConfig
        from transformers import AutoProcessor

        self.config = Qwen2AudioEncoderConfig.from_pretrained(config_dir)
        self.encoder = Qwen2AudioEncoder(self.config)
        # Load clean encoder weights
        from safetensors.torch import load_file
        state_dict = load_file(encoder_weights_path)
        self.encoder.load_state_dict(state_dict, strict=False)
        self.hidden_size = self.config.d_model  # 1280

        # Use HF processor for correct mel extraction
        try:
            self.processor = AutoProcessor.from_pretrained(config_dir, trust_remote_code=True)
        except Exception:
            self.processor = None

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _build_mel(self, wave: torch.Tensor) -> torch.Tensor:
        """wave: [B, T] mono → mel spectrogram matching Qwen2-Audio's native preprocessor."""
        if self.processor is not None:
            import torchaudio.functional as F
            # Resample to 16kHz if needed (dataset already does this, but ensure)
            # Use processor's feature extractor directly
            wave_np = wave.detach().float().cpu().numpy()
            inputs = self.processor.feature_extractor(
                [w for w in wave_np],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            mel = inputs["input_features"]  # [B, 128, 3000]
            return mel.to(wave.device)
        else:
            # Fallback: manual mel spectrogram (may be imperfect)
            import torchaudio
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=128,
                n_fft=400,
                hop_length=160,
                win_length=400,
                power=2,
            ).to(wave.device)
            with torch.cuda.amp.autocast(enabled=False):
                mel = mel_transform(wave.float())
                mel = torch.clamp(mel, min=1e-10)
                mel = torch.log(mel)
            F_t = mel.shape[-1]
            if F_t < 3000:
                mel = torch.nn.functional.pad(mel, (0, 3000 - F_t))
            elif F_t > 3000:
                mel = mel[..., :3000]
            return mel

    def forward(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wave: [B, 2, T] → (ch0_hidden, ch1_hidden) each [B, ~750, 1280]"""
        with torch.no_grad():
            mel_ch0 = self._build_mel(wave[:, 0, :])
            mel_ch1 = self._build_mel(wave[:, 1, :])
            h0 = self.encoder(mel_ch0).last_hidden_state  # [B, ~748, 1280]
            h1 = self.encoder(mel_ch1).last_hidden_state
        # Ensure frame count divisible by 2 for Conv1d(stride=2)
        T0, T1 = h0.shape[1], h1.shape[1]
        if T0 % 2 != 0:
            h0 = F.pad(h0, (0, 0, 0, 2 - T0 % 2))
        if T1 % 2 != 0:
            h1 = F.pad(h1, (0, 0, 0, 2 - T1 % 2))
        return h0, h1


class DualEncoderTemporalModel(nn.Module):
    """WavLM + Qwen2Audio dual encoder → GRU temporal engine → multi-label turn-taking.

    Architecture:
        Dual-channel audio [B, 2, T=480000] @ 16kHz
          ├── WavLM(ch0, ch1) → concat [B, 1500, 2048] → Conv1d → [B, 375, 512]
          └── Qwen2Audio(ch0, ch1) → concat [B, 750, 2560] → Conv1d → [B, 375, 512]
        Concat([wavlm_feat, qwen_feat]) → [B, 375, 1024]
          └── Concat Context Embedding(16) → [B, 375, 1040]
        GRU(256, 1-layer, causal) → last_frame_state
        MLP(256→128→5)

    Args:
        cfg: configuration dict with keys:
            - audio_encoder.wavlm_model: path to WavLM model
            - audio_encoder.qwen2audio_encoder_weights: path to encoder_only.safetensors
            - audio_encoder.qwen2audio_config_dir: path to encoder config
            - context_encoder, labels, etc. (same as TemporalAlignedModel)
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        audio_cfg = cfg["audio_encoder"]

        # WavLM encoder
        self.wavlm_encoder = WavLMDualChannelEncoder(
            model_path=audio_cfg["wavlm_model"],
            freeze=bool(audio_cfg.get("freeze_wavlm", True)),
        )
        wavlm_dim = self.wavlm_encoder.hidden_size  # 1024
        self.wavlm_conv = nn.Conv1d(wavlm_dim * 2, 512, kernel_size=4, stride=4)
        self.wavlm_norm = nn.LayerNorm(512)
        # Adapter: learnable alignment for frozen encoder features
        self.wavlm_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
        )

        # Qwen2Audio encoder
        self.qwen_encoder = Qwen2AudioDualChannelEncoder(
            encoder_weights_path=audio_cfg["qwen2audio_encoder_weights"],
            config_dir=audio_cfg["qwen2audio_config_dir"],
            freeze=bool(audio_cfg.get("freeze_qwen", True)),
        )
        qwen_dim = self.qwen_encoder.hidden_size  # 1280
        self.qwen_conv = nn.Conv1d(qwen_dim * 2, 512, kernel_size=2, stride=2)
        self.qwen_norm = nn.LayerNorm(512)
        # Adapter: learnable alignment for frozen encoder features
        self.qwen_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
        )

        # Context encoder
        ctx_cfg = cfg["context_encoder"]
        self.ctx_embed = nn.Embedding(ctx_cfg["vocab_size"], ctx_cfg["embed_dim"])

        # GRU temporal engine
        fusion_dim = 512 + 512 + ctx_cfg["embed_dim"]  # 1040
        gru_hidden = 256
        self.temporal_engine = nn.GRU(
            input_size=fusion_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # MLP head
        num_targets = len(cfg.get("labels", {}).get("multi_targets", []))
        self.num_targets = num_targets if num_targets > 0 else 1
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_targets),
        )

    def forward(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        context_labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            waveform: [B, 2, T] dual-channel audio
            input_ids: ignored (no text encoder)
            attention_mask: ignored
            context_labels: [B, 375] chunk-level labels

        Returns:
            logits: [B, 5] per-class logits
        """
        B = waveform.shape[0]

        # WavLM dual-channel
        wav_h0, wav_h1 = self.wavlm_encoder(waveform)
        wav_cat = torch.cat([wav_h0, wav_h1], dim=-1).transpose(1, 2)  # [B, 2048, T]
        wav_feat = self.wavlm_conv(wav_cat).transpose(1, 2)  # [B, T', 512]
        wav_feat = self.wavlm_norm(wav_feat)
        wav_feat = self.wavlm_adapter(wav_feat)

        # Qwen2Audio dual-channel
        qwen_h0, qwen_h1 = self.qwen_encoder(waveform)
        qwen_cat = torch.cat([qwen_h0, qwen_h1], dim=-1).transpose(1, 2)  # [B, 2560, T]
        qwen_feat = self.qwen_conv(qwen_cat).transpose(1, 2)  # [B, T', 512]
        qwen_feat = self.qwen_norm(qwen_feat)
        qwen_feat = self.qwen_adapter(qwen_feat)

        # Align to context length
        expected_len = context_labels.shape[1]
        if wav_feat.shape[1] < expected_len:
            wav_feat = F.pad(wav_feat, (0, 0, 0, expected_len - wav_feat.shape[1]))
        elif wav_feat.shape[1] > expected_len:
            wav_feat = wav_feat[:, :expected_len, :]
        if qwen_feat.shape[1] < expected_len:
            qwen_feat = F.pad(qwen_feat, (0, 0, 0, expected_len - qwen_feat.shape[1]))
        elif qwen_feat.shape[1] > expected_len:
            qwen_feat = qwen_feat[:, :expected_len, :]

        # Concatenate audio features
        audio_feat = torch.cat([wav_feat, qwen_feat], dim=-1)  # [B, 375, 1024]

        # Context embedding
        ctx_feat = self.ctx_embed(context_labels.long())  # [B, 375, 16]
        fused_seq = torch.cat([audio_feat, ctx_feat], dim=-1)  # [B, 375, 1040]

        # GRU temporal scan
        gru_out, _ = self.temporal_engine(fused_seq)
        last_frame_state = gru_out[:, -1, :]  # [B, 256]

        logits = self.head(last_frame_state)

        if self.num_targets == 1:
            return logits.squeeze(-1)
        return logits