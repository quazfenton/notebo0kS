"""Checkpoint utilities (save/load state dicts, configs, and training position)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: str | Path, state: Dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    return torch.load(str(path), map_location="cpu")
