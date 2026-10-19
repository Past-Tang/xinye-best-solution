"""Frame-level (chunk-level) auxiliary head.

Whisper-large-v3 encoder outputs hidden [B, 1500, 1280] at 50 Hz frame rate
for a 30s input. The chunk grid we want to supervise is 375 frames at 80ms
(=12.5 Hz). Ratio is 1500 / 375 = 4.

This head:
  Conv1d(1280 -> mid, kernel=4, stride=4)  -> [B, mid, 375]
  LayerNorm(mid) on transposed [B, 375, mid]
  GELU
  Linear(mid, num_classes)                 -> [B, 375, num_classes]
  transpose -> [B, num_classes, 375]   (suitable for F.cross_entropy over C dim)

Supervision target = past 30s context_labels (already provided to the model
as input). This is a *self-distillation-from-input* signal that forces the
audio encoder to learn frame-level class discrimination.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChunkAuxHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 1280,
        mid_dim: int = 256,
        num_classes: int = 5,
        whisper_frames: int = 1500,
        target_frames: int = 375,
    ) -> None:
        super().__init__()
        if whisper_frames % target_frames != 0:
            raise ValueError(
                f"whisper_frames {whisper_frames} must be divisible by target_frames {target_frames}"
            )
        stride = whisper_frames // target_frames  # 4
        self.in_dim = int(in_dim)
        self.mid_dim = int(mid_dim)
        self.num_classes = int(num_classes)
        self.target_frames = int(target_frames)
        self._stride = int(stride)

        self.down = nn.Conv1d(in_dim, mid_dim, kernel_size=stride, stride=stride)
        self.norm = nn.LayerNorm(mid_dim)
        self.act = nn.GELU()
        self.head = nn.Linear(mid_dim, num_classes)

    def forward(self, audio_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_hidden: [B, whisper_frames, in_dim]
        Returns:
            logits: [B, num_classes, target_frames]  (channel-first for CE)
        """
        if audio_hidden.dim() != 3:
            raise ValueError(f"expected 3D [B,T,D], got {audio_hidden.shape}")
        x = audio_hidden.transpose(1, 2)         # [B, D, T]
        x = self.down(x)                          # [B, mid, target_frames]
        # If T is slightly off (e.g. 1499 instead of 1500), pad/truncate
        if x.shape[-1] != self.target_frames:
            if x.shape[-1] < self.target_frames:
                pad = self.target_frames - x.shape[-1]
                x = torch.nn.functional.pad(x, (0, pad))
            else:
                x = x[..., : self.target_frames]
        x = self.norm(x.transpose(1, 2))          # [B, T, mid]
        x = self.act(x)
        logits = self.head(x)                     # [B, T, C]
        return logits.transpose(1, 2).contiguous()  # [B, C, T]