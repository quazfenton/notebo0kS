"""Grammar, program tokens, and simple differentiable interpreter.
This is a minimal placeholder that supports a small vocabulary of compositional ops
(bind, move, recolor) over slot latents; execution is differentiable via MLPs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class GrammarCfg:
    vocab: int = 64
    slot_dim: int = 64


class ProgramHead(nn.Module):
    """Autoregressive logits over program tokens (no teacher-forcing here for brevity)."""
    def __init__(self, vocab: int, slot_dim: int, hidden: int = 128, max_len: int = 32):
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.embed = nn.Embedding(vocab, slot_dim)
        self.rnn = nn.GRU(slot_dim, hidden, batch_first=True)
        self.to_logits = nn.Linear(hidden, vocab)

    def forward(self, B: int, tau: float = 1.0) -> torch.Tensor:
        # Produce a sequence of logits [B, L, V]
        h = None
        toks = []
        x = torch.zeros(B, 1, dtype=torch.long, device=self.embed.weight.device)
        for _ in range(self.max_len):
            emb = self.embed(x)
            out, h = self.rnn(emb, h)
            logits = self.to_logits(out)  # [B,1,V]
            toks.append(logits)
            x = logits.argmax(dim=-1)  # greedy
        return torch.cat(toks, dim=1)


class Interpreter(nn.Module):
    """Apply program tokens to slots via tiny MLP ops; returns reconstructed token features.
    """
    def __init__(self, slot_dim: int, out_dim: int):
        super().__init__()
        self.bind = nn.Linear(slot_dim, slot_dim)
        self.move = nn.Linear(slot_dim, slot_dim)
        self.recolor = nn.Linear(slot_dim, slot_dim)
        self.to_pixels = nn.Sequential(nn.Linear(slot_dim, out_dim), nn.Sigmoid())

    def forward(self, slots: torch.Tensor, prog_logits: torch.Tensor, tokens_n: int) -> torch.Tensor:
        # slots: [B,S,D], prog_logits: [B,L,V]
        # We apply a simple recurrent edit sequence over slots, then broadcast to N tokens.
        z = slots
        for i in range(prog_logits.shape[1]):
            if i % 3 == 0:
                z = z + torch.tanh(self.bind(z))
            elif i % 3 == 1:
                z = z + torch.tanh(self.move(z))
            else:
                z = z + torch.tanh(self.recolor(z))
        # Collapse slots and map to pixel/token space, then broadcast to tokens_n
        z_sum = z.sum(dim=1)  # [B,D]
        tokens = z_sum.unsqueeze(1).expand(-1, tokens_n, -1)  # [B,N,D]
        return self.to_pixels(tokens)  # [B,N,out_dim]
