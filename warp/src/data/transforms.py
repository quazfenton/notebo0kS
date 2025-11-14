# Transform helpers (normalization, tokenization stubs)
from __future__ import annotations

import torch


def normalize_img(x: torch.Tensor) -> torch.Tensor:
    # x: [B,C,H,W] in [0,1]
    return (x - 0.5) / 0.5


def denormalize_img(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 + 0.5
