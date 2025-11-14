"""Slot-program curriculum helpers: bucket building and level switching.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

from src.data.clevr_synth import generate_dataset


@dataclass
class Bucket:
    path: Path
    img_size: int
    min_objs: int
    max_objs: int


def build_buckets(base_dir: Path, levels: int, per_level_n: int, base_min_objs: int, base_max_objs: int, base_img_size: int, obj_step: int, img_step: int, seed: int = 0) -> List[Bucket]:
    buckets: List[Bucket] = []
    for i in range(levels):
        min_o = base_min_objs + i * obj_step
        max_o = base_max_objs + i * obj_step
        img_s = base_img_size + i * img_step
        lvl_dir = base_dir.parent / f"{base_dir.name}_lvl{i+1}"
        lvl_dir.mkdir(parents=True, exist_ok=True)
        # If directory is empty, generate
        if not any(lvl_dir.glob("*.png")):
            generate_dataset(lvl_dir, per_level_n, img_s, min_o, max_o, seed=seed + i)
        buckets.append(Bucket(lvl_dir, img_s, min_o, max_o))
    return buckets