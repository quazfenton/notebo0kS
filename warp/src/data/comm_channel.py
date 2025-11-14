"""Sender-Receiver communication games with noisy channel.
This provides a simple API to map discrete program tokens to channel codes and back.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


@dataclass
class ChannelCfg:
    code_len: int = 16
    snr_db: float = 20.0


class NoisyChannel(nn.Module):
    def __init__(self, code_len: int, snr_db: float):
        super().__init__()
        self.code_len = code_len
        self.snr_db = snr_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # AWGN channel
        snr_lin = 10 ** (self.snr_db / 10.0)
        sig_power = x.pow(2).mean().clamp(min=1e-6)
        noise_power = sig_power / snr_lin
        noise = torch.randn_like(x) * math.sqrt(noise_power.item())
        return x + noise


class Sender(nn.Module):
    def __init__(self, vocab: int, code_len: int, dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, code_len)

    def forward(self, toks: torch.Tensor) -> torch.Tensor:
        z = self.emb(toks).mean(dim=1)
        return torch.tanh(self.proj(z))


class Receiver(nn.Module):
    def __init__(self, vocab: int, code_len: int, dim: int = 64):
        super().__init__()
        self.inp = nn.Linear(code_len, dim)
        self.cls = nn.Linear(dim, vocab)

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.inp(code))
        return self.cls(z)
