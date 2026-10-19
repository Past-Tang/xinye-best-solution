"""Curriculum scheduler with two LR groups (LoRA vs head).

Stages (epochs are continuous, ramps are linear within a 1-epoch window):
  epoch 0  to  e_warm     : LoRA LR scale = 0.0
  epoch e_warm-1 to e_warm: LoRA LR scale ramps 0.0 -> 1.0  (1-epoch ramp)
  epoch e_warm to e_full  : LoRA LR scale = 1.0  (LoRA active)
  epoch e_full-1 to e_full: head LR scale ramps 1.0 -> 0.5
  epoch e_full to inf     : LoRA LR scale = 1.0, head LR scale = 0.5

In addition to the per-group scale, we wrap a global cosine schedule with
warmup (transformers.get_cosine_schedule_with_warmup-equivalent), so the
final per-group LR at update step `s` is:
    lr_group(s) = base_lr_group * group_scale(epoch) * cosine_factor(s)

Implementation: torch LambdaLR is per-group already, so we register one lambda
per group that combines cosine_factor(s) * group_scale(epoch_for_step(s)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List

import torch


@dataclass
class CurriculumConfig:
    e_warm: int = 5    # LoRA starts being trained after this epoch (inclusive ramp)
    e_full: int = 15   # head LR starts halving after this epoch
    head_lr_late_scale: float = 0.5
    lora_lr_full_scale: float = 1.0


def _cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _epoch_for_step(step: int, steps_per_epoch: int) -> float:
    return float(step) / float(max(1, steps_per_epoch))


def _lora_scale(epoch_f: float, cfg: CurriculumConfig) -> float:
    e_warm = cfg.e_warm
    if epoch_f <= e_warm - 1:
        return 0.0
    if epoch_f >= e_warm:
        return float(cfg.lora_lr_full_scale)
    # 1-epoch linear ramp: epoch_f in (e_warm-1, e_warm)
    t = epoch_f - (e_warm - 1)
    return float(cfg.lora_lr_full_scale) * float(t)


def _head_scale(epoch_f: float, cfg: CurriculumConfig) -> float:
    e_full = cfg.e_full
    if epoch_f <= e_full - 1:
        return 1.0
    if epoch_f >= e_full:
        return float(cfg.head_lr_late_scale)
    t = epoch_f - (e_full - 1)
    return 1.0 - (1.0 - float(cfg.head_lr_late_scale)) * float(t)


def make_curriculum_lambdas(
    total_steps: int,
    warmup_steps: int,
    steps_per_epoch: int,
    cfg: CurriculumConfig,
    group_names: List[str],
) -> List[Callable[[int], float]]:
    """Build one LambdaLR fn per param group, in the order `group_names`.

    Each group is identified by name in {'lora', 'head'}. Unknown names get
    plain cosine-with-warmup (scale=1).
    """
    lambdas: List[Callable[[int], float]] = []
    for name in group_names:
        nm = str(name).lower()

        def fn(step: int, _name: str = nm) -> float:
            cos = _cosine_with_warmup(step, total_steps, warmup_steps)
            ef = _epoch_for_step(step, steps_per_epoch)
            if _name == "lora":
                return cos * _lora_scale(ef, cfg)
            if _name == "head":
                return cos * _head_scale(ef, cfg)
            return cos

        lambdas.append(fn)
    return lambdas


def split_param_groups(
    model: torch.nn.Module,
    lora_marker: str = "lora_",
    base_lr: float = 3e-4,
    weight_decay: float = 0.01,
):
    """Return AdamW-ready param groups [lora, head].

    A parameter is assigned to the 'lora' group iff its qualified name
    contains `lora_marker`. Everything else trainable goes to 'head'.

    No-decay params (LayerNorm, biases, embeddings) get weight_decay=0.

    Returns:
        groups, group_names
    """
    decay_lora, nodecay_lora = [], []
    decay_head, nodecay_head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_lora = lora_marker in n
        is_nodecay = (
            n.endswith(".bias")
            or "layer_norm" in n.lower()
            or "layernorm" in n.lower()
            or ".norm" in n.lower()
            or "embedding" in n.lower()
        )
        if is_lora:
            (nodecay_lora if is_nodecay else decay_lora).append(p)
        else:
            (nodecay_head if is_nodecay else decay_head).append(p)

    groups = []
    names = []
    if decay_lora:
        groups.append({"params": decay_lora, "lr": base_lr, "weight_decay": weight_decay, "name": "lora"})
        names.append("lora")
    if nodecay_lora:
        groups.append({"params": nodecay_lora, "lr": base_lr, "weight_decay": 0.0, "name": "lora"})
        names.append("lora")
    if decay_head:
        groups.append({"params": decay_head, "lr": base_lr, "weight_decay": weight_decay, "name": "head"})
        names.append("head")
    if nodecay_head:
        groups.append({"params": nodecay_head, "lr": base_lr, "weight_decay": 0.0, "name": "head"})
        names.append("head")

    if not groups:
        raise RuntimeError("No trainable parameters found.")
    return groups, names