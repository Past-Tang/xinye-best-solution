"""MultimodalLoRAChunkAuxModel.

Architecture:

  waveform [B, 2, 480000]
      |
      | (per-channel) GPU log-mel STFT (Whisper-aligned)
      v
  mel [2*B, 80, 3000]
      |
      | LoRA-wrapped Whisper encoder (gradient_checkpointing on training)
      v
  hidden [2*B, 1500, 1280]
      |
      | reshape -> [B, 2, 1500, 1280]
      | mean over channel dim -> audio_hidden [B, 1500, 1280]
      |
      +---> ChunkAuxHead -> aux_logits [B, 5, 375]   (CE vs context_labels)
      |
      | tail attention pool (last 300 frames = 6s) -> audio_pool [B, 1280]
      v

  text  [B, L]  --frozen Qwen3-0.6B + tail attn-pool-->  text_pool [B, 1024]
  ctx   [B, 375]                ContextLabelEncoder    -> ctx_pool   [B, 128]

  fusion [B, 1280 + 1024 + 128 = 2432]
      |
      | (Mix2Selector if training) -> mixed fusion + mixed targets
      v
  3-layer MLP head -> main_logits [B, 5]   (RobustASL vs targets)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor, WhisperModel

# Reuse baseline encoder building blocks (do NOT modify baseline)
_BASELINE_ROOT = Path(__file__).resolve().parent.parent / "_lib"
if str(_BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINE_ROOT))

from src.models.multimodal_baseline import (  # noqa: E402  (after sys.path injection)
    AttentionPooling,
    ContextLabelEncoder,
    TextEncoder,
)

from .chunk_aux_head import ChunkAuxHead
from .lora_wrapper import wrap_whisper_lora
from .text_lora import LoRATextEncoder


# ---------------------------------------------------------------------------
# GPU log-mel filterbank, aligned with HF WhisperFeatureExtractor.
# Adapted from baseline `multimodal_dualch.DualChannelWhisperAudioEncoder._gpu_log_mel`
# (already validated against feature_extractor in dualch experiments).
# ---------------------------------------------------------------------------

class WhisperGpuMel(nn.Module):
    """Reproduces HF WhisperFeatureExtractor on-GPU.

    Saves CPU bottleneck and avoids list-of-tensor copy each step.
    """

    def __init__(self, feature_extractor_path: str, sample_rate: int = 16000) -> None:
        super().__init__()
        self.fe = WhisperFeatureExtractor.from_pretrained(feature_extractor_path)
        # mel filters [n_mels, n_fft//2+1]
        mel_filters = torch.from_numpy(np.asarray(self.fe.mel_filters, dtype=np.float32))
        self.register_buffer("_mel_filters", mel_filters.T.contiguous(), persistent=False)
        self._n_fft = int(self.fe.n_fft)
        self._hop_length = int(self.fe.hop_length)
        self._chunk_length = int(self.fe.chunk_length)
        self._n_samples = self._chunk_length * int(sample_rate)  # 480000 for 30s
        self._nb_max_frames = int(self.fe.nb_max_frames)         # 3000
        self.register_buffer("_window", torch.hann_window(self._n_fft), persistent=False)
        self.sample_rate = int(sample_rate)

    def forward(self, wave_mono: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wave_mono: [N, T] float32 (any dtype, will cast)
        Returns:
            mel: [N, n_mels=80, nb_max_frames=3000] float32
        """
        if wave_mono.dim() != 2:
            raise ValueError(f"expected [N, T], got {wave_mono.shape}")
        x = wave_mono.float()
        # Pad/truncate to n_samples
        T = x.shape[-1]
        if T < self._n_samples:
            x = F.pad(x, (0, self._n_samples - T))
        elif T > self._n_samples:
            x = x[..., : self._n_samples]
        # STFT
        with torch.cuda.amp.autocast(enabled=False):
            stft = torch.stft(
                x,
                n_fft=self._n_fft,
                hop_length=self._hop_length,
                window=self._window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
            magnitudes = stft[..., :-1].abs() ** 2  # drop last frame -> 3000
            mel_spec = torch.matmul(self._mel_filters.to(x.device), magnitudes)
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max(dim=-1, keepdim=True).values - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
        # Ensure exact 3000 frames
        if log_spec.shape[-1] < self._nb_max_frames:
            log_spec = F.pad(log_spec, (0, self._nb_max_frames - log_spec.shape[-1]))
        elif log_spec.shape[-1] > self._nb_max_frames:
            log_spec = log_spec[..., : self._nb_max_frames]
        return log_spec.contiguous()


# ---------------------------------------------------------------------------
# Future-oriented auxiliary head: predicts chunk labels at multiple future
# horizons from the tail of per-channel audio hidden states.  Designed to
# force the encoder to develop forward-looking representations, directly
# addressing the finding that T/I/BC events develop 400-900ms ahead.
# ---------------------------------------------------------------------------

class FutureAuxHead(nn.Module):
    """Multi-horizon future chunk label predictor.

    Takes the last `tail_frames` of per-channel Whisper hidden states,
    pools each channel separately, fuses them, and predicts per-class
    logits at each future horizon.

    Horizons are in chunk units (80 ms each).  Typical values:
      [5, 10, 15, 25]   (400 ms, 800 ms, 1200 ms, 2000 ms)
    These align with the empirical first-occurrence times of I (~680 ms),
    T (~900 ms), BC (~850 ms).

    Args:
        in_dim:      Whisper hidden dimension (1280).
        mid_dim:     internal MLP dimension.
        num_classes: number of label classes (5).
        horizons:    list of int, future chunk offsets to predict.
        tail_frames: number of Whisper frames (50 Hz) to pool.
    """

    def __init__(
        self,
        in_dim: int = 1280,
        mid_dim: int = 256,
        num_classes: int = 5,
        horizons: tuple[int, ...] = (5, 10, 15, 25),
        tail_frames: int = 150,   # last 3 s at 50 Hz
    ) -> None:
        super().__init__()
        self.horizons = list(horizons)
        self.n_horizons = len(self.horizons)
        self.tail_frames = int(tail_frames)
        self.num_classes = int(num_classes)

        self.ch_pool = AttentionPooling(int(in_dim))
        # cat([pool0, pool1, pool0-pool1]) = in_dim * 3
        self.fuse = nn.Sequential(
            nn.Linear(in_dim * 3, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, num_classes * self.n_horizons),
        )

    def forward(
        self,
        ch0_hidden: torch.Tensor,
        ch1_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """ch0_hidden, ch1_hidden: [B, T, D]  →  logits: [B, H, C]."""
        T = ch0_hidden.shape[1]
        tail0 = ch0_hidden[:, max(0, T - self.tail_frames):, :]
        tail1 = ch1_hidden[:, max(0, T - self.tail_frames):, :]
        pool0 = self.ch_pool(tail0)   # [B, D]
        pool1 = self.ch_pool(tail1)   # [B, D]
        pool_diff = torch.abs(pool0 - pool1)  # channel mismatch magnitude
        fused = self.fuse(torch.cat([pool0, pool1, pool_diff], dim=-1))  # [B, D]
        logits = self.head(fused)     # [B, C * H]
        return logits.view(-1, self.n_horizons, self.num_classes).contiguous()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MultimodalLoRAChunkAuxModel(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        audio_cfg = cfg["audio_encoder"]
        text_cfg = cfg["text_encoder"]
        ctx_cfg = cfg["context_encoder"]
        head_cfg = cfg["fusion_head"]
        lora_cfg = cfg.get("lora", {})
        chunk_cfg = cfg.get("chunk_aux", {})

        # ---- Audio: Whisper encoder + GPU mel + LoRA ----
        whisper_path = audio_cfg["model_name"]
        self.audio_mel = WhisperGpuMel(whisper_path, sample_rate=int(cfg["sample_rate"]))
        self.channel_pool_mode = str(audio_cfg.get("channel_pool_mode", "mean")).lower()
        # attn_implementation: default None keeps legacy eager behaviour (back-compat);
        # set audio_encoder.attn_implementation: sdpa (or flash_attention_2) in config to
        # accelerate the Whisper encoder. SDPA is numerically equivalent to eager softmax
        # attention, so it does NOT change the stride ablation attribution.
        _attn_impl = audio_cfg.get("attn_implementation", None)
        _wm_kwargs = {}
        if _attn_impl:
            _wm_kwargs["attn_implementation"] = str(_attn_impl)
        whisper_full = WhisperModel.from_pretrained(whisper_path, **_wm_kwargs)
        encoder = whisper_full.encoder
        # Drop decoder + reset whisper_full to free memory
        del whisper_full
        self.hidden_size = int(encoder.config.d_model)  # 1280
        # Wrap with LoRA. By default PEFT freezes base model; only LoRA learns.
        self.audio_encoder = wrap_whisper_lora(
            encoder,
            r=int(lora_cfg.get("r", 16)),
            alpha=int(lora_cfg.get("alpha", 32)),
            dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=tuple(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"])),
            enable_gradient_checkpointing=bool(lora_cfg.get("gradient_checkpointing", True)),
        )

        # Tail attention pool over last `tail_frames` of Whisper hidden
        self.audio_tail_frames = int(audio_cfg.get("tail_frames", 300))  # last 6s at 50Hz
        self.audio_pool = AttentionPooling(self.hidden_size)
        if self.channel_pool_mode == "mean_diff":
            self.audio_merge = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
            )
            self.audio_out_dim = self.hidden_size
        else:
            self.audio_merge = None
            self.audio_out_dim = self.hidden_size  # 1280, no extra projection

        # ---- Text: frozen Qwen3-0.6B reuse from baseline (or LoRA-wrapped) ----
        text_lora_cfg = text_cfg.get("lora", {}) or {}
        if bool(text_lora_cfg.get("enabled", False)):
            self.text_encoder = LoRATextEncoder(
                model_name=text_cfg["model_name"],
                r=int(text_lora_cfg.get("r", 8)),
                alpha=int(text_lora_cfg.get("alpha", 16)),
                dropout=float(text_lora_cfg.get("dropout", 0.05)),
                target_modules=tuple(text_lora_cfg.get(
                    "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
                gradient_checkpointing=bool(text_lora_cfg.get("gradient_checkpointing", True)),
                tail_ratio=float(text_cfg.get("tail_ratio", 0.3)),
            )
        else:
            self.text_encoder = TextEncoder(
                model_name=text_cfg["model_name"],
                freeze_backbone=bool(text_cfg.get("freeze_backbone", True)),
                tail_ratio=float(text_cfg.get("tail_ratio", 0.3)),
            )

        # ---- Context labels ----
        self.context_encoder = ContextLabelEncoder(
            vocab_size=int(ctx_cfg["vocab_size"]),
            embed_dim=int(ctx_cfg["embed_dim"]),
            channels=tuple(ctx_cfg["channels"]),
            tail_k=int(ctx_cfg.get("tail_k", 50)),
        )

        # ---- Chunk auxiliary head ----
        self.use_aux = bool(chunk_cfg.get("enabled", True))
        if self.use_aux:
            self.chunk_aux_head = ChunkAuxHead(
                in_dim=self.hidden_size,
                mid_dim=int(chunk_cfg.get("mid_dim", 256)),
                num_classes=int(chunk_cfg.get("num_classes", 5)),
                whisper_frames=int(chunk_cfg.get("whisper_frames", 1500)),
                target_frames=int(chunk_cfg.get("target_frames", 375)),
            )

        # ---- Future auxiliary head (optional) ----
        future_cfg = cfg.get("future_aux", {})
        self.use_future_aux = bool(future_cfg.get("enabled", False))
        if self.use_future_aux:
            self.future_aux_head = FutureAuxHead(
                in_dim=self.hidden_size,
                mid_dim=int(future_cfg.get("mid_dim", 256)),
                num_classes=int(chunk_cfg.get("num_classes", 5)),
                horizons=tuple(future_cfg.get("horizons", [5, 10, 15, 25])),
                tail_frames=int(future_cfg.get("tail_frames", 150)),
            )

        # ---- Fusion + main head ----
        fusion_in = (
            self.audio_out_dim
            + self.text_encoder.out_dim
            + self.context_encoder.out_dim
        )
        h1, h2 = head_cfg["hidden_dims"]
        dropout = float(head_cfg.get("dropout", 0.3))
        num_targets = len(cfg.get("labels", {}).get("multi_targets", []))
        self.num_targets = num_targets if num_targets > 0 else 5
        self.head = nn.Sequential(
            nn.Linear(fusion_in, h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, self.num_targets),
        )

    # ---------- audio forward ----------
    def _audio_hidden(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """waveform [B, 2, T] -> (merged_hidden, ch0_hidden, ch1_hidden)."""
        B, C, T = waveform.shape
        if C != 2:
            raise ValueError(f"expected dual-channel waveform, got C={C}")
        wav = waveform.reshape(B * 2, T)
        mel = self.audio_mel(wav)
        out = self.audio_encoder(input_features=mel)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hidden = hidden.view(B, 2, hidden.shape[1], hidden.shape[2])
        ch0_hidden = hidden[:, 0]
        ch1_hidden = hidden[:, 1]
        if self.channel_pool_mode == "mean_diff":
            hidden_mean = 0.5 * (ch0_hidden + ch1_hidden)
            hidden_diff = torch.abs(ch0_hidden - ch1_hidden)  # channel mismatch magnitude
            merged_hidden = self.audio_merge(torch.cat([hidden_mean, hidden_diff], dim=-1))
        else:
            merged_hidden = hidden.mean(dim=1)
        return merged_hidden, ch0_hidden, ch1_hidden

    def _audio_pool(self, merged_hidden: torch.Tensor, ch0_hidden: torch.Tensor, ch1_hidden: torch.Tensor) -> torch.Tensor:
        if self.channel_pool_mode == "mean_diff":
            T0 = ch0_hidden.shape[1]
            tail0 = ch0_hidden[:, max(0, T0 - self.audio_tail_frames):, :]
            tail1 = ch1_hidden[:, max(0, T0 - self.audio_tail_frames):, :]
            pool0 = self.audio_pool(tail0)
            pool1 = self.audio_pool(tail1)
            pool_mean = 0.5 * (pool0 + pool1)
            pool_diff = torch.abs(pool0 - pool1)  # channel mismatch magnitude
            return self.audio_merge(torch.cat([pool_mean, pool_diff], dim=-1))
        T = merged_hidden.shape[1]
        tail = merged_hidden[:, max(0, T - self.audio_tail_frames):, :]
        return self.audio_pool(tail)

    # ---------- main forward ----------
    def forward(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_labels: torch.Tensor,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            main_logits     : [B, num_targets]
            aux_logits      : [B, num_classes=5, target_frames=375] or None
            fusion          : [B, fusion_in]   (returned for Mix2 application before head)
            future_aux_logits: [B, H, C] or None  (H = len(future_aux.horizons))
        """
        audio_hidden, ch0_hidden, ch1_hidden = self._audio_hidden(waveform)

        if return_aux and self.use_aux:
            aux_logits = self.chunk_aux_head(audio_hidden)
        else:
            aux_logits = None

        if return_aux and self.use_future_aux:
            future_aux_logits = self.future_aux_head(ch0_hidden, ch1_hidden)
        else:
            future_aux_logits = None

        audio_pool = self._audio_pool(audio_hidden, ch0_hidden, ch1_hidden)
        text_pool = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )                                                          # [B, 1024]
        ctx_pool = self.context_encoder(context_labels=context_labels)  # [B, 128]
        fusion = torch.cat([audio_pool, text_pool, ctx_pool], dim=-1)   # [B, 2432]
        main_logits = self.head(fusion)                                  # [B, 5]
        return main_logits, aux_logits, fusion, future_aux_logits

    def head_only(self, fusion: torch.Tensor) -> torch.Tensor:
        """Re-run only the MLP head on (possibly mix2-mixed) fusion features."""
        return self.head(fusion)