"""Metrics and information-theoretic diagnostics.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def mse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.mse_loss(x, y).detach().cpu().item())


def entropy_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs = logits.softmax(dim=dim)
    logp = logits.log_softmax(dim=dim)
    return -(probs * logp).sum(dim=dim)


def ensemble_variance(preds: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return preds.var(dim=dim)


def calibration_nll(mu: torch.Tensor, logvar: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Gaussian NLL
    return 0.5 * ((x - mu) ** 2 / logvar.exp() + logvar + torch.log(torch.tensor(2 * 3.1415926535))).mean()


def bits_per_dim(nll: torch.Tensor, n_dim: int) -> float:
    # Convert nats to bits per dimension
    return float((nll / (n_dim * torch.log(torch.tensor(2.0)))).detach().cpu().item())
