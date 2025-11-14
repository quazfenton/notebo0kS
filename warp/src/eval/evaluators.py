"""Evaluation helpers: energy drift, bitrate/distortion proxies, program complexity.
"""
from __future__ import annotations

from typing import Dict

import torch


def energy_drift(z_seq: torch.Tensor, H_fn) -> float:
    """Compute avg |H_t - H_0| across trajectory (proxy for symplectic stability)."""
    with torch.no_grad():
        H0 = H_fn(z_seq[:, 0, :]).mean()
        Ht = H_fn(z_seq.reshape(-1, z_seq.shape[-1])).view(z_seq.shape[0], z_seq.shape[1]).mean(dim=0)
        drift = (Ht - H0).abs().mean().item()
        return float(drift)


def bitrate_proxy(logits: torch.Tensor) -> float:
    """Negative entropy as bits proxy (lower entropy → lower bitrate)."""
    with torch.no_grad():
        probs = logits.softmax(dim=-1)
        logp = logits.log_softmax(dim=-1)
        H = -(probs * logp).sum(dim=-1).mean()
        bits = (H / torch.log(torch.tensor(2.0))).item()
        return float(bits)


def program_length(tokens: torch.Tensor) -> float:
    with torch.no_grad():
        return float((tokens != 0).sum(dim=-1).float().mean().item())
