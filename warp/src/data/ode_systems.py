"""Classical ODE systems and utilities with torchdiffeq wrappers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch


@dataclass
class LorenzParams:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0


def lorenz_rhs(t: torch.Tensor, x: torch.Tensor, p: LorenzParams) -> torch.Tensor:
    # x = [x, y, z]
    dx = p.sigma * (x[:, 1] - x[:, 0])
    dy = x[:, 0] * (p.rho - x[:, 2]) - x[:, 1]
    dz = x[:, 0] * x[:, 1] - p.beta * x[:, 2]
    return torch.stack([dx, dy, dz], dim=1)


def integrate_ode(rhs: Callable, x0: torch.Tensor, t: torch.Tensor, p) -> torch.Tensor:
    """Simple fixed-step RK4 integrator (no torchdiffeq dependency for portability)."""
    x = x0
    xs = [x0]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        k1 = rhs(t[i - 1], x, p)
        k2 = rhs(t[i - 1] + 0.5 * dt, x + 0.5 * dt * k1, p)
        k3 = rhs(t[i - 1] + 0.5 * dt, x + 0.5 * dt * k2, p)
        k4 = rhs(t[i], x + dt * k3, p)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x)
    return torch.stack(xs, dim=1)  # [B, T, D]


def sample_lorenz_batch(batch: int, horizon: int, dt: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    p = LorenzParams()
    t = torch.linspace(0, (horizon - 1) * dt, horizon, device=device)
    x0 = torch.randn(batch, 3, device=device) * 0.5 + torch.tensor([1.0, 1.0, 1.0], device=device)
    xs = integrate_ode(lambda tt, xx, pp: lorenz_rhs(tt, xx, pp), x0, t, p)
    return t, xs
