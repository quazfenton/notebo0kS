"""Minimal Slot Attention module (image inputs assumed as flattened tokens).
Note: This is a simplified variant sufficient for toy CLEVR.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    def __init__(self, num_slots: int, dim: int, iters: int = 3):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.scale = dim ** -0.5
        self.slot_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slot_sigma = nn.Parameter(torch.ones(1, 1, dim))
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        mu, sigma = self.slot_mu, F.softplus(self.slot_sigma)
        slots = mu + sigma * torch.randn(B, self.num_slots, D, device=x.device, dtype=x.dtype)

        x = self.norm_inputs(x)
        k, v = self.to_k(x), self.to_v(x)

        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))
            attn_logits = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = F.softmax(attn_logits, dim=1)
            updates = torch.einsum('bij,bjd->bid', attn, v)
            slots = self.gru(updates.reshape(-1, D), slots_prev.reshape(-1, D)).view(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots, attn  # return last attention map
