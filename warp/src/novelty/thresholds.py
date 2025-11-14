"""Novelty thresholds and simple perturbations utilities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import torch


@dataclass
class EntropyHistory:
    values: list[float]

    def quantile(self, q: float) -> float:
        if not self.values:
            return 0.0
        vs = sorted(self.values)
        k = max(0, min(len(vs)-1, int(q * (len(vs)-1))))
        return vs[k]


def posterior_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs = logits.softmax(dim=dim)
    logp = logits.log_softmax(dim=dim)
    return -(probs * logp).sum(dim=dim).mean()


def gaussian_perturb(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x + scale * torch.randn_like(x)
