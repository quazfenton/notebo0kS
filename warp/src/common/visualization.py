"""TensorBoard helpers and simple plotting wrappers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, log_dir: str | Path):
        self.writer = SummaryWriter(str(log_dir))

    def log_scalars(self, scalars: Dict[str, float], step: int, prefix: str = ""):
        for k, v in scalars.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.writer.add_scalar(tag, v, step)

    def close(self):
        self.writer.flush()
        self.writer.close()
