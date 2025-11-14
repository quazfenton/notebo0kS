def _set_dropout_p(module: torch.nn.Module, p: float):
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p


def eig_dropout_disagreement(dyn: torch.nn.Module, z_seq: torch.Tensor, dt: float, T: int = 8, dropout_p: float = 0.1) -> torch.Tensor:
    """Compute variance across T stochastic forward passes with dropout active.
    Assumes dyn contains Dropout layers; temporarily forces train mode and sets p.
    """
    was_training = dyn.training
    # Snapshot original ps
    original_ps = []
    for m in dyn.modules():
        if isinstance(m, torch.nn.Dropout):
            original_ps.append((m, m.p))
    try:
        dyn.train(True)
        _set_dropout_p(dyn, dropout_p)
        z_last = z_seq[:, -1, :]
        preds = []
        for _ in range(T):
            z_next, _ = dyn.step(z_last, dt)
            preds.append(z_next)
        P = torch.stack(preds, dim=0)
        return P.var(dim=0).mean()
    finally:
        # restore
        for m, p in original_ps:
            m.p = p
        dyn.train(was_training)

"""Ensemble-based EIG surrogate for world-models.
This computes disagreement across perturbed parameter clones as an EIG proxy.
"""
from __future__ import annotations

from typing import Callable

import copy
import torch


def _perturb_weights(module: torch.nn.Module, scale: float) -> torch.nn.Module:
    m2 = copy.deepcopy(module)
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(scale * torch.randn_like(p))
    return m2


def eig_ensemble_disagreement(dyn: torch.nn.Module, z_seq: torch.Tensor, dt: float, K: int = 5, weight_noise_scale: float = 0.01) -> torch.Tensor:
    """Compute variance across next-step predictions under K perturbed clones.
    z_seq: [B,T,Z]
    Returns: scalar tensor proxy for EIG.
    """
    B, T, Z = z_seq.shape
    z_last = z_seq[:, -1, :]  # next-step prediction focus
    preds = []
    for _ in range(K):
        dcopy = _perturb_weights(dyn, weight_noise_scale)
        z_next, _ = dcopy.step(z_last, dt)
        preds.append(z_next)
    P = torch.stack(preds, dim=0)  # [K,B,Z]
    return P.var(dim=0).mean()