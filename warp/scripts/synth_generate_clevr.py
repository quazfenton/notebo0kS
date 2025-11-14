#!/usr/bin/env python3
"""Generate tiny CLEVR-like images.
Usage:
  python scripts/synth_generate_clevr.py --n 512 --out data/clevr_tiny --img 64 --min 2 --max 5 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.clevr_synth import generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", type=str, default="data/clevr_tiny")
    ap.add_argument("--img", type=int, default=64)
    ap.add_argument("--min", dest="min_objs", type=int, default=2)
    ap.add_argument("--max", dest="max_objs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    generate_dataset(Path(args.out), args.n, args.img, args.min_objs, args.max_objs, args.seed)
    print(f"Wrote {args.n} images to {args.out}")


if __name__ == "__main__":
    main()
