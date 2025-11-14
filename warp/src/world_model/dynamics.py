"""Hamiltonian dynamics, symplectic integration, and residuals for physics losses.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DynCfg:
    z_dim: int
    hidden: int = 128
    symplectic: bool = True


class LearnedHamiltonian(nn.Module):
    def __init__(self, z_dim: int, hidden: int = 128, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(z_dim, hidden), nn.SiLU(),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(hidden, 1)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def symplectic_step(z: torch.Tensor, H: LearnedHamiltonian, dt: float) -> torch.Tensor:
    """One leapfrog step assuming canonical split z=(q,p)."""
    q, p = torch.chunk(z, 2, dim=-1)
    q.requires_grad_(True)
    p.requires_grad_(True)

    # Half step p
    Hq = H(torch.cat([q, p], dim=-1)).sum()
    dH_dq, dH_dp = torch.autograd.grad(Hq, [q, p], create_graph=True)
    p_half = p - 0.5 * dt * dH_dq

    # Full step q
    Hq2 = H(torch.cat([q, p_half], dim=-1)).sum()
    dH_dq2, dH_dp2 = torch.autograd.grad(Hq2, [q, p_half], create_graph=True)
    q_new = q + dt * dH_dp2

    # Final half step p
    Hq3 = H(torch.cat([q_new, p_half], dim=-1)).sum()
    dH_dq3, dH_dp3 = torch.autograd.grad(Hq3, [q_new, p_half], create_graph=True)
    p_new = p_half - 0.5 * dt * dH_dq3

    return torch.cat([q_new, p_new], dim=-1).detach()  # detach to avoid graph blow-up


class ResidualDyn(nn.Module):
    """Residual dynamics term f_theta(z,a) added to Hamiltonian flow (here no actions)."""
    def __init__(self, z_dim: int, hidden: int = 128, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(z_dim, hidden), nn.SiLU(),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(hidden, z_dim)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class WorldModel(nn.Module):
    def __init__(self, z_dim: int, hidden: int = 128, symplectic: bool = True, dropout_p: float = 0.0):
        super().__init__()
        assert z_dim % 2 == 0, "z_dim must be even to split (q,p)"
        self.H = LearnedHamiltonian(z_dim, hidden, dropout_p=dropout_p)
        self.f = ResidualDyn(z_dim, hidden, dropout_p=dropout_p)
        self.symplectic = symplectic

    def step(self, z: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.symplectic:
            z_flow = symplectic_step(z, self.H, dt)
        else:
            z_flow = z
        z_res = z_flow + dt * self.f(z_flow)
        residual = z_res - z  # simple residual signal for physics loss
        return z_res, residual
