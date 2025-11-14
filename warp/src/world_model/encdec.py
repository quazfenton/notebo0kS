"""Encoders/decoders for low-D physics states (toy), and image variants (stub).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPEnc(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPDec(nn.Module):
    def __init__(self, z_dim: int, x_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
