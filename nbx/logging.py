# NBX logging helper
from __future__ import annotations
from pathlib import Path
from typing import Dict

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # type: ignore

class TBLogger:
    def __init__(self, log_dir: str | Path):
        if SummaryWriter is None:
            raise RuntimeError("tensorboard is required for TBLogger")
        self.writer = SummaryWriter(str(log_dir))

    def scalars(self, scalars: Dict[str, float], step: int, prefix: str = ""):
        for k, v in scalars.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.writer.add_scalar(tag, v, step)

    def close(self):
        self.writer.flush()
        self.writer.close()
