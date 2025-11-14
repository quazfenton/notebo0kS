"""Loss functions and objective components.
Includes: reconstruction, physics consistency, KL, MDL bits penalties, EIG bonus.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    recon: float = 1.0
    phys: float = 0.0
    kl: float = 0.0
    mdL_bits: float = 0.0
    info_gain: float = 0.0


class ReconLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(pred, target, reduction="none")
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class PhysicsConsistency(nn.Module):
    """Penalty for deviation from symplectic/Hamiltonian dynamics residuals.
    Expects residuals already computed by dynamics module.
    """

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return (residual ** 2).mean()


class KLLoss(nn.Module):
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def mdl_bits_penalty(bits_estimate: torch.Tensor) -> torch.Tensor:
    return bits_estimate.mean()


def eig_bonus(eig_estimate: torch.Tensor) -> torch.Tensor:
    # Higher EIG reduces free energy → subtract as a bonus in loss, but return magnitude here
    return -eig_estimate.mean()


def aggregate_losses(parts: Dict[str, torch.Tensor], w: LossWeights) -> Tuple[torch.Tensor, Dict[str, float]]:
    total = (
        w.recon * parts.get("recon", 0.0)
        + w.phys * parts.get("phys", 0.0)
        + w.kl * parts.get("kl", 0.0)
        + w.mdL_bits * parts.get("mdl", 0.0)
        + w.info_gain * parts.get("eig", 0.0)
    )
    scalars = {k: float(v.detach().cpu().item()) for k, v in parts.items()}
    scalars["total"] = float(total.detach().cpu().item())
    return total, scalars
