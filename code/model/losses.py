"""Loss functions for Phase B.

Two losses:
1. RobustASL: Asymmetric Loss (Ridnik 2021, ICCV) + Hill regularization (Park 2023).
   Used for the main event-level multi-label head (5 sigmoid outputs).
   - Numerically stable form via log_sigmoid (avoids log(sigmoid(x)) overflow in bf16).
   - Asymmetric focusing: gamma_neg >> gamma_pos suppresses easy negatives.
   - Negative margin m: shifts negative probabilities before focusing,
     effectively discarding the easiest negatives.
   - Hill weight on negative loss to bound positive grad explosion.

2. chunk_aux_loss: frame-level cross-entropy supervising past 30s
   context_labels (single-label per frame, 5 classes), label smoothing.

Both losses accept (logits, targets) where logits are raw (pre-sigmoid/softmax)
to keep AMP-bf16 numerically stable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Robust Asymmetric Loss (multi-label, sigmoid)
# ---------------------------------------------------------------------------

class RobustASL(nn.Module):
    """Robust Asymmetric Loss for multi-label classification.

    Args:
        gamma_neg: focusing exponent for negatives (default 4.0).
        gamma_pos: focusing exponent for positives (default 0.0 = no focusing).
        margin:    negative probability margin in [0, 0.2]. Negatives with
                   p < margin contribute zero gradient.
        hill_lambda: Hill weight for negative loss term (Park 2023).
        eps: numerical stability constant.

    Forward:
        logits  : [B, C]  raw logits (pre-sigmoid)
        targets : [B, C]  float in [0, 1] (supports soft labels from Mix2)

    Returns scalar mean loss.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        margin: float = 0.05,
        hill_lambda: float = 1.5,
        eps: float = 1e-8,
        pos_weight: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.margin = float(margin)
        self.hill_lambda = float(hill_lambda)
        self.eps = float(eps)
        # Per-class positive weight: upweights rare classes (e.g. BC at 3.9%).
        # If provided, must be length C matching logits.shape[-1].
        if pos_weight is not None:
            self.register_buffer(
                "_pos_weight",
                torch.tensor(pos_weight, dtype=torch.float32).unsqueeze(0),  # [1, C]
            )
        else:
            self._pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                class_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Numerically stable log(sigmoid(x)) and log(1 - sigmoid(x))
        # via log_sigmoid(x) and log_sigmoid(-x).
        log_pos = F.logsigmoid(logits)            # log(p)
        log_neg = F.logsigmoid(-logits)           # log(1 - p)
        p = torch.sigmoid(logits.float())         # use fp32 for focusing weight only

        # Asymmetric negative margin: shift p by margin then clamp
        if self.margin > 0.0:
            p_neg = (p - self.margin).clamp(min=0.0)
        else:
            p_neg = p

        # Focusing weights (no gradient on the modulating factor itself)
        with torch.no_grad():
            w_pos = (1.0 - p).clamp(min=self.eps).pow(self.gamma_pos)
            w_neg = p_neg.clamp(min=self.eps).pow(self.gamma_neg)

        # Cross-entropy in log-space then weight
        # pos_loss = -y * w_pos * pos_weight * log(p)
        # neg_loss = -(1-y) * w_neg * log(1-p)
        pw = self._pos_weight
        if pw is not None:
            pw = pw.to(logits.device)  # [1, C]
        pos_loss = -targets * w_pos.to(log_pos.dtype) * log_pos
        if pw is not None:
            pos_loss = pos_loss * pw.to(pos_loss.dtype)
        neg_loss = -(1.0 - targets) * w_neg.to(log_neg.dtype) * log_neg

        # Hill: scale neg loss by hill_lambda; bounds positive-class gradient
        # explosion when many easy negatives dominate.
        loss = pos_loss + self.hill_lambda * neg_loss

        if class_mask is not None:
            m = class_mask.to(loss.dtype)
            return (loss * m).sum() / m.sum().clamp(min=1.0)
        # Mean over batch * classes
        return loss.mean()


# ---------------------------------------------------------------------------
# Frame-level chunk auxiliary loss
# ---------------------------------------------------------------------------

def chunk_aux_loss(
    frame_logits: torch.Tensor,
    frame_targets: torch.Tensor,
    label_smoothing: float = 0.1,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy over [B, C, T] frame-level logits vs [B, T] int64 targets.

    Args:
        frame_logits  : [B, C=5, T=375] raw logits
        frame_targets : [B, T=375]      int64 class ids (0..C-1) or ignore_index
        label_smoothing : float in [0, 1)
        ignore_index : int label to ignore (default -100, no padding here).
    Returns scalar mean loss.
    """
    return F.cross_entropy(
        frame_logits,
        frame_targets,
        label_smoothing=float(label_smoothing),
        ignore_index=int(ignore_index),
        reduction="mean",
    )


# ---------------------------------------------------------------------------
# Future auxiliary loss: multi-horizon chunk label prediction
# ---------------------------------------------------------------------------

def future_aux_loss(
    future_logits: torch.Tensor,
    future_targets: torch.Tensor,
    label_smoothing: float = 0.1,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy over future horizon predictions.

    Args:
        future_logits  : [B, H, C]   raw logits for H future horizons
        future_targets : [B, H]      int64 label ids at each horizon (may contain ignore_index)
        label_smoothing: float in [0, 1)
        ignore_index   : int, positions where future exceeds conversation length
    Returns scalar mean loss.
    """
    B, H, C = future_logits.shape
    flat_logits = future_logits.reshape(B * H, C)
    flat_targets = future_targets.reshape(B * H)
    return F.cross_entropy(
        flat_logits,
        flat_targets,
        label_smoothing=float(label_smoothing),
        ignore_index=int(ignore_index),
        reduction="mean",
    )


# ---------------------------------------------------------------------------
# Total loss combining main ASL + chunk aux with epoch-aware weight schedule
# ---------------------------------------------------------------------------

@dataclass
class AuxWeightSchedule:
    """Piecewise constant aux weight schedule.

    Default:
      epoch  0 -  4 : 0.5  (strong frame supervision early)
      epoch  5 - 14 : 0.3
      epoch 15 - inf: 0.1  (let main head dominate late stage)
    """
    e_strong: int = 5
    e_mid: int = 15
    w_strong: float = 0.5
    w_mid: float = 0.3
    w_late: float = 0.1

    def __call__(self, epoch: int) -> float:
        if epoch < self.e_strong:
            return float(self.w_strong)
        if epoch < self.e_mid:
            return float(self.w_mid)
        return float(self.w_late)


class TotalLoss(nn.Module):
    def __init__(
        self,
        asl_kwargs: Dict | None = None,
        aux_label_smoothing: float = 0.1,
        aux_schedule: AuxWeightSchedule | None = None,
        future_aux_label_smoothing: float = 0.1,
        future_aux_weight: float = 0.3,
        pos_weight: list[float] | None = None,
        mutex_weight: float = 0.0,
        bc_bce_weight: float = 0.0,
        bc_bce_pos_weight: float = 25.0,
        bc_class_index: int = 3,
    ) -> None:
        super().__init__()
        _asl_kw = dict(asl_kwargs or {})
        if pos_weight is not None:
            _asl_kw["pos_weight"] = pos_weight
        self.asl = RobustASL(**_asl_kw)
        self.aux_label_smoothing = float(aux_label_smoothing)
        self.aux_schedule = aux_schedule or AuxWeightSchedule()
        self.future_aux_label_smoothing = float(future_aux_label_smoothing)
        self.future_aux_weight = float(future_aux_weight)
        # Mutual-exclusivity regularizer (defect 3): softmax-CE on main_logits
        # toward the single active class. Only applied to "clean" rows where
        # exactly one label is 1 (skips Mix2 soft-label rows). Default 0 = off,
        # so existing models are numerically unchanged.
        self.mutex_weight = float(mutex_weight)
        # BC-specific BCE booster (Wang 2024 multi-task spirit, single-class):
        # the rare BC class (3.7%) under ASL gamma_neg=4 still gets crushed.
        # Adding a binary cross-entropy on the BC column with a strong
        # pos_weight gives it gradient magnitude comparable to common classes.
        # Weight 0 = off (default), preserves old behaviour bit-for-bit.
        self.bc_bce_weight = float(bc_bce_weight)
        self.bc_bce_pos_weight = float(bc_bce_pos_weight)
        self.bc_class_index = int(bc_class_index)

    def forward(
        self,
        main_logits: torch.Tensor,
        main_targets: torch.Tensor,
        aux_logits: torch.Tensor | None,
        aux_targets: torch.Tensor | None,
        epoch: int,
        future_aux_logits: torch.Tensor | None = None,
        future_aux_targets: torch.Tensor | None = None,
        class_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        L_main = self.asl(main_logits, main_targets, class_mask=class_mask)
        parts: Dict[str, float] = {
            "main": float(L_main.detach().item()),
            "aux": 0.0,
            "aux_w": 0.0,
            "fut_aux": 0.0,
            "mutex": 0.0,
            "bc_bce": 0.0,
        }
        total = L_main

        if aux_logits is not None and aux_targets is not None:
            L_aux = chunk_aux_loss(aux_logits, aux_targets, label_smoothing=self.aux_label_smoothing)
            w = float(self.aux_schedule(int(epoch)))
            total = total + w * L_aux
            parts["aux"] = float(L_aux.detach().item())
            parts["aux_w"] = w

        if future_aux_logits is not None and future_aux_targets is not None:
            L_fut = future_aux_loss(
                future_aux_logits, future_aux_targets,
                label_smoothing=self.future_aux_label_smoothing,
            )
            total = total + self.future_aux_weight * L_fut
            parts["fut_aux"] = float(L_fut.detach().item())

        if self.mutex_weight > 0.0:
            # Clean rows: exactly one positive label (skip Mix2 soft rows & all-neg rows).
            row_sum = main_targets.sum(dim=-1)
            is_binary = ((main_targets == 0) | (main_targets == 1)).all(dim=-1)
            clean = is_binary & (row_sum == 1.0)
            if clean.any():
                cls_idx = main_targets[clean].argmax(dim=-1)
                L_mutex = F.cross_entropy(main_logits[clean], cls_idx, reduction="mean")
                total = total + self.mutex_weight * L_mutex
                parts["mutex"] = float(L_mutex.detach().item())

        if self.bc_bce_weight > 0.0:
            # BC-only BCE booster: extract BC column, apply BCE-with-logits with
            # large pos_weight. Targets remain soft (Mix2-compatible). This is
            # additive on top of ASL's existing BC contribution; weight controls
            # how much extra gradient the BC class gets.
            bc_idx = self.bc_class_index
            if bc_idx < main_logits.shape[-1]:
                bc_logits = main_logits[..., bc_idx]
                bc_targets = main_targets[..., bc_idx].to(bc_logits.dtype)
                pw = torch.tensor(self.bc_bce_pos_weight, device=bc_logits.device, dtype=bc_logits.dtype)
                L_bc = F.binary_cross_entropy_with_logits(
                    bc_logits, bc_targets, pos_weight=pw, reduction="mean"
                )
                total = total + self.bc_bce_weight * L_bc
                parts["bc_bce"] = float(L_bc.detach().item())

        return total, parts