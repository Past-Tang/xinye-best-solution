"""Whisper encoder LoRA wrapper.

Validated module names (whisper-large-v3, transformers 4.57):
    encoder.layers[i].self_attn.q_proj  Linear(1280, 1280)
    encoder.layers[i].self_attn.k_proj  Linear(1280, 1280)
    encoder.layers[i].self_attn.v_proj  Linear(1280, 1280)
    encoder.layers[i].self_attn.out_proj Linear(1280, 1280)

We attach LoRA only to attention projections (not FFN), trainable params
~5.2M (= 32 layers * 4 projs * 2 * 1280 * 16 / 1e6).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model
from peft.utils.other import _get_submodules


WHISPER_ENC_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")


def wrap_whisper_lora(
    encoder: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Sequence[str] = WHISPER_ENC_TARGETS,
    enable_gradient_checkpointing: bool = True,
):
    """Wrap a Whisper encoder with LoRA + optional gradient checkpointing.

    Important:
        * The base encoder weights remain frozen by PEFT (default behaviour);
          only LoRA adapter matrices receive gradients.
        * `enable_input_require_grads()` is required when combining
          gradient_checkpointing with PEFT, otherwise the input embedding
          tensor has requires_grad=False and the checkpointed path silently
          drops gradients.

    Args:
        encoder: Whisper encoder (HF WhisperModel.encoder).
        r: LoRA rank. r=16 -> 5.2M trainable for attention-only on whisper-large.
        alpha: LoRA scaling. Effective scale = alpha / r = 2.0 here.
        dropout: LoRA dropout.
        target_modules: attention projection names to attach LoRA.
        enable_gradient_checkpointing: turn on activation re-computation,
            cuts encoder activation memory ~6x at the cost of ~25% extra time.

    Returns:
        peft_model: a PeftModel wrapping the encoder. Forward signature is
            unchanged: ``peft_model(input_features=...)``.
    """
    config = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        target_modules=list(target_modules),
        lora_dropout=float(dropout),
        bias="none",
        task_type=None,  # encoder-only, no specific PEFT task type
    )

    # peft_model frozen-base by default; only LoRA layers learn
    peft_model = get_peft_model(encoder, config)

    if enable_gradient_checkpointing:
        # Whisper encoder supports HF gradient_checkpointing API
        if hasattr(peft_model, "gradient_checkpointing_enable"):
            peft_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        elif hasattr(peft_model.base_model.model, "gradient_checkpointing_enable"):
            peft_model.base_model.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            # Fallback: directly toggle attribute
            for mod in peft_model.modules():
                if hasattr(mod, "gradient_checkpointing"):
                    mod.gradient_checkpointing = True

        # Critical: PEFT + gradient_checkpointing requires input grads.
        # Without this, recomputed activations have no upstream grad.
        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()

    return peft_model


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Returns (trainable, total) parameter counts."""
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total