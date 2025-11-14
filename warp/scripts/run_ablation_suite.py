#!/usr/bin/env python3
"""Run a small ablation suite for both models.
Usage:
  python scripts/run_ablation_suite.py --suite tiny
"""
from __future__ import annotations

import argparse
import subprocess


def run(cmd: list[str]):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def suite_tiny():
    # World-model: symplectic on/off
    run(["python", "scripts/train_worldmodel.py", "tiny=true", "logging.every=50", "optim.steps=300", "model.symplectic=true", "workdir=outputs/abl_world_symp_on"]) 
    run(["python", "scripts/train_worldmodel.py", "tiny=true", "logging.every=50", "optim.steps=300", "model.symplectic=false", "workdir=outputs/abl_world_symp_off"]) 
    # Slot-program: comm on/off
    run(["python", "scripts/train_slotprog.py", "tiny=true", "logging.every=50", "optim.steps=300", "comm.enable=true", "workdir=outputs/abl_slot_comm_on"]) 
    run(["python", "scripts/train_slotprog.py", "tiny=true", "logging.every=50", "optim.steps=300", "comm.enable=false", "workdir=outputs/abl_slot_comm_off"]) 


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["tiny"], default="tiny")
    args = ap.parse_args()
    if args.suite == "tiny":
        suite_tiny()


if __name__ == "__main__":
    main()
