"""Minimal CLEVR-like renderer (circles/squares/triangles) using OpenCV.
Generates images and slot-level masks for supervision-free training (slots via reconstruction only).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2


SHAPES = ("circle", "square", "triangle")


@dataclass
class Obj:
    shape: str
    color: Tuple[int, int, int]  # BGR
    pos: Tuple[int, int]
    size: int


def rand_color(rng: np.random.Generator) -> Tuple[int, int, int]:
    hsv = np.array([rng.uniform(0, 180), rng.uniform(180, 255), rng.uniform(180, 255)], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv[None, None, :], cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def render_obj(img: np.ndarray, obj: Obj):
    x, y = obj.pos
    s = obj.size
    if obj.shape == "circle":
        cv2.circle(img, (x, y), s, obj.color, -1)
    elif obj.shape == "square":
        cv2.rectangle(img, (x - s, y - s), (x + s, y + s), obj.color, -1)
    elif obj.shape == "triangle":
        pts = np.array([[x, y - s], [x - s, y + s], [x + s, y + s]], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, obj.color)


def generate_scene(img_size: int, min_objs: int, max_objs: int, rng: np.random.Generator) -> np.ndarray:
    n = rng.integers(min_objs, max_objs + 1)
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    for _ in range(n):
        shape = rng.choice(SHAPES)
        color = rand_color(rng)
        size = int(rng.integers(img_size // 12, img_size // 6))
        x = int(rng.integers(size, img_size - size))
        y = int(rng.integers(size, img_size - size))
        render_obj(img, Obj(shape, color, (x, y), size))
    return img


def generate_dataset(out_dir: Path, n: int, img_size: int, min_objs: int, max_objs: int, seed: int = 0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        img = generate_scene(img_size, min_objs, max_objs, rng)
        cv2.imwrite(str(out_dir / f"img_{i:06d}.png"), img)
