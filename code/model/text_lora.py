"""LoRA-wrapped text encoder for Qwen3-style HF transformers.

Drop-in replacement for `multimodal_baseline.TextEncoder`:
  forward signature: (input_ids, attention_mask) -> [B, hidden_size]
  attribute `out_dim` exposed.

Differences from frozen TextEncoder:
  - PEFT LoRA on attention projections (q/k/v/o)
  - gradient_checkpointing + enable_input_require_grads()
  - tail attention pool over last `tail_ratio` tokens (same as baseline)

Trainable params at r=8: ~1.5M for Qwen3-0.6B; r=16: ~3M.

Why a dedicated module instead of forking baseline TextEncoder:
  - keeps `_lib/src/models/multimodal_baseline.py` untouched (frozen contract)
  - can evolve LoRA-specific configs (rank/dropout/targets) independently
  - PEFT model needs `enable_input_require_grads()` for gradient_checkpointing
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model
from transformers import AutoModel

# Reuse baseline AttentionPooling via sys.path injection done by model.py
from src.models.multimodal_baseline import AttentionPooling


QWEN_TEXT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRATextEncoder(nn.Module):
    """Qwen3-0.6B + LoRA on attention projections + tail attention pool.

    Args:
        model_name: HF path or local cache for AutoModel.
        r:          LoRA rank.
        alpha:      LoRA scaling = alpha / r.
        dropout:    LoRA dropout.
        target_modules: list of attention proj names to wrap.
        gradient_checkpointing: enable activation re-computation on backbone.
        tail_ratio: fraction of trailing tokens to attention-pool.
    """

    def __init__(
        self,
        model_name: str,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        target_modules: Sequence[str] = QWEN_TEXT_TARGETS,
        gradient_checkpointing: bool = True,
        tail_ratio: float = 0.3,
    ) -> None:
        super().__init__()
        backbone = AutoModel.from_pretrained(model_name)
        # Freeze backbone explicitly; PEFT will add LoRA adapters and only those learn.
        for p in backbone.parameters():
            p.requires_grad = False

        config = LoraConfig(
            r=int(r),
            lora_alpha=int(alpha),
            target_modules=list(target_modules),
            lora_dropout=float(dropout),
            bias="none",
            task_type=None,
        )
        self.backbone = get_peft_model(backbone, config)

        if gradient_checkpointing:
            inner = (self.backbone.base_model.model
                     if hasattr(self.backbone, "base_model") else self.backbone)
            if hasattr(inner, "gradient_checkpointing_enable"):
                inner.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            # PEFT + gradient_checkpointing requires input grads; without this the
            # checkpointed path silently drops gradients into the embedding.
            if hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()

        # hidden_size from the wrapped config
        cfg_obj = (self.backbone.base_model.model.config
                   if hasattr(self.backbone, "base_model")
                   else self.backbone.config)
        self.out_dim = int(cfg_obj.hidden_size)
        self.tail_ratio = float(tail_ratio)
        self.attn_pool = AttentionPooling(self.out_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, L, H]

        # Tail attention pool: focus on the most recent utterances.
        L = hidden.shape[1]
        tail_start = max(0, L - int(L * self.tail_ratio))
        tail_hidden = hidden[:, tail_start:, :]
        tail_mask = attention_mask[:, tail_start:].unsqueeze(-1).to(hidden.dtype)
        masked_hidden = tail_hidden * tail_mask
        pooled = self.attn_pool(masked_hidden)
        return pooled