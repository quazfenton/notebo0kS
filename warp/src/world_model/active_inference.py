"""Active inference planning stubs (EIG-driven action selection hooks).
Currently unused in toy training; provided for future control tasks.
"""
from __future__ import annotations

import torch


def expected_information_gain_proxy(belief: torch.Tensor) -> torch.Tensor:
    # Placeholder: entropy of belief as curiosity bonus
    probs = torch.softmax(belief, dim=-1)
    logp = torch.log_softmax(belief, dim=-1)
    return -(probs * logp).sum(dim=-1).mean()
