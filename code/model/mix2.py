"""Mix2: stochastic Mixup / Manifold Mixup / MultiMix selector.

Reference:
  - Mixup (Zhang 2018): x' = lam * x_a + (1-lam) * x_b
  - Manifold Mixup (Verma 2019): mixup applied at intermediate hidden layers
  - MultiMix (Husain 2020) / Mix2 (Bouattour 2024 EUSIPCO): k-way mixing with
    Dirichlet sampling for multi-label rare classes.

In this project Mix2 is applied at the *fusion features* level (post audio/text/
context pooling, pre main head). Chunk auxiliary loss uses unmixed audio_hidden
and unmixed frame_targets (independent path), so frame-level supervision is not
affected by mixup.

API:
    selector = Mix2Selector(prob=0.5, alpha=0.4, k_max=3)
    feats_m, targets_m, info = selector(feats, targets, training=True)

If training=False or prob fails, returns (feats, targets, {'mode': 'none'}).
"""
from __future__ import annotations

import random
from typing import Dict, Tuple

import torch


class Mix2Selector:
    def __init__(
        self,
        prob: float = 0.5,
        alpha: float = 0.4,
        k_max: int = 3,
        modes: tuple[str, ...] = ("mixup", "multimix"),
        seed: int | None = None,
    ) -> None:
        """
        Args:
            prob: probability of triggering a mix on a given step.
            alpha: Beta/Dirichlet concentration parameter; smaller -> more
                   skewed (closer to original sample); 0.4 is a typical value
                   from the Mixup paper for vision classification.
            k_max: max number of samples for MultiMix (k=3 default).
            modes: subset of {'mixup', 'manifold', 'multimix'}. ManifoldMixup
                   uses the same math as Mixup at this layer; we therefore
                   merge them and keep the 'multimix' branch for k=3.
            seed: reproducibility seed for the internal RNG.
        """
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if k_max < 2:
            raise ValueError(f"k_max must be >= 2, got {k_max}")
        valid = {"mixup", "manifold", "multimix"}
        if not set(modes).issubset(valid):
            raise ValueError(f"modes must be subset of {valid}, got {modes}")
        self.prob = float(prob)
        self.alpha = float(alpha)
        self.k_max = int(k_max)
        self.modes = tuple(modes)
        self._rng = random.Random(seed)

    @torch.no_grad()
    def _sample_lam_2way(self, device, dtype) -> torch.Tensor:
        """Draw lam ~ Beta(alpha, alpha)."""
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().to(device=device, dtype=dtype)
        # Numerical: clamp to avoid degenerate 0/1 mixes
        return lam.clamp(min=0.05, max=0.95)

    @torch.no_grad()
    def _sample_lam_kway(self, k: int, device, dtype) -> torch.Tensor:
        """Draw lam ~ Dirichlet([alpha]*k)."""
        conc = torch.full((k,), self.alpha, device=device, dtype=dtype)
        lam = torch.distributions.Dirichlet(conc).sample()
        return lam

    def __call__(
        self,
        feats: torch.Tensor,
        targets: torch.Tensor,
        training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            feats   : [B, D] fusion features
            targets : [B, C] multi-label targets (float; 0/1 ok, soft labels too)
            training: only mix when True
        Returns:
            mixed_feats   : [B, D]
            mixed_targets : [B, C]  (soft labels after mix)
            info          : dict with 'mode', 'lam', 'k'
        """
        if not training or self.prob <= 0.0 or self._rng.random() > self.prob:
            return feats, targets, {"mode": "none"}

        mode = self._rng.choice(self.modes)
        if mode == "multimix":
            k = self.k_max
        else:
            k = 2

        device = feats.device
        # use fp32 for mixing weights for stability under bf16/fp16 amp
        mix_dtype = torch.float32

        if k == 2:
            lam = self._sample_lam_2way(device, mix_dtype)
            perm = torch.randperm(feats.shape[0], device=device)
            f_b = feats[perm]
            t_b = targets[perm]
            mixed_feats = (lam * feats.float() + (1.0 - lam) * f_b.float()).to(feats.dtype)
            mixed_targets = (lam * targets.float() + (1.0 - lam) * t_b.float()).to(targets.dtype)
            return mixed_feats, mixed_targets, {"mode": mode, "lam": float(lam.item()), "k": 2}

        # k-way Dirichlet mix
        lam = self._sample_lam_kway(k, device, mix_dtype)  # [k]
        B = feats.shape[0]
        mixed_feats = lam[0] * feats.float()
        mixed_targets = lam[0] * targets.float()
        for i in range(1, k):
            perm = torch.randperm(B, device=device)
            mixed_feats = mixed_feats + lam[i] * feats[perm].float()
            mixed_targets = mixed_targets + lam[i] * targets[perm].float()
        return (
            mixed_feats.to(feats.dtype),
            mixed_targets.to(targets.dtype),
            {"mode": mode, "lam": [float(x) for x in lam.tolist()], "k": k},
        )