"""Schedulers and curriculum utilities: novelty thresholds, annealing, curricula.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def cosine_anneal(step: int, max_steps: int, start: float, end: float) -> float:
    if max_steps <= 0:
        return end
    cos_inner = math.pi * min(step, max_steps) / max_steps
    return end + (start - end) * (1 + math.cos(cos_inner)) / 2


@dataclass
class EntropyNoveltySchedule:
    q: float = 0.85
    scale: float = 0.05

    def trigger(self, entropy: float, hist_quantile: float) -> bool:
        return entropy > max(self.q, hist_quantile)

    def magnitude(self, curvature: float | None = None) -> float:
        if curvature is None or curvature <= 0:
            return self.scale
        return min(self.scale * (1 + 0.1 * math.log1p(curvature)), 10 * self.scale)
