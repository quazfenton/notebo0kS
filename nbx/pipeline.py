# NBX: lightweight data pipeline helpers for ad-hoc training scripts
from __future__ import annotations

__all__ = [
    "DataConfig",
    "DataSplit",
    "DataPipeline",
    "load_from_cli",
    "summarize_dataset",
]

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import os
import json

import torch
from torch.utils.data import Dataset, DataLoader

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from datasets import load_dataset as hf_load_dataset  # optional
except Exception:
    hf_load_dataset = None

from PIL import Image
import glob
import random


@dataclass
class DataConfig:
    kind: str  # csv|parquet|jsonl|images|text|hf
    path: str
    input_cols: List[str] = field(default_factory=list)
    target_col: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 0
    val_split: float = 0.1
    shuffle: bool = True
    img_size: int = 128
    hf_name: Optional[str] = None


@dataclass
class DataSplit:
    train: DataLoader
    val: Optional[DataLoader]
    meta: Dict[str, Any]


class _TabularDS(Dataset):
    def __init__(self, df, input_cols, target_col=None):
        self.df = df
        self.input_cols = input_cols or [c for c in df.columns if c != target_col]
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.input_cols].values, dtype=torch.float32)
        if self.target_col is None:
            return x
        y = row[self.target_col]
        if isinstance(y, (float, int)):
            y = torch.tensor(y, dtype=torch.float32)
        return x, y


class _TextFolderDS(Dataset):
    def __init__(self, folder: Path):
        self.files = sorted(Path(folder).glob("**/*.txt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        text = p.read_text(encoding="utf-8", errors="ignore")
        return text


class _ImageFolderDS(Dataset):
    def __init__(self, folder: Path, img_size: int):
        self.root = Path(folder)
        self.img_size = img_size
        self.samples = []
        # class is subfolder name; fallback to single-class if no subfolders
        subdirs = [d for d in self.root.iterdir() if d.is_dir()]
        if not subdirs:
            imgs = list(self.root.glob("*.png")) + list(self.root.glob("*.jpg")) + list(self.root.glob("*.jpeg"))
            self.samples = [(p, 0) for p in imgs]
        else:
            class_to_idx = {d.name: i for i, d in enumerate(sorted(subdirs))}
            for d, idx in class_to_idx.items():
                imgs = list((self.root / d).glob("*.png")) + list((self.root / d).glob("*.jpg")) + list((self.root / d).glob("*.jpeg"))
                self.samples.extend([(p, idx) for p in imgs])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB").resize((self.img_size, self.img_size))
        x = torch.from_numpy(torch.ByteTensor(bytearray(img.tobytes())).numpy())  # raw bytes to numpy
        x = x.view(self.img_size, self.img_size, 3).permute(2, 0, 1).float() / 255.0
        return x, torch.tensor(y, dtype=torch.long)


class DataPipeline:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    def build(self) -> DataSplit:
        kind = self.cfg.kind
        meta: Dict[str, Any] = {"kind": kind}
        if kind in ("csv", "parquet", "jsonl"):
            assert pd is not None, "pandas is required for tabular datasets"
            if kind == "csv":
                df = pd.read_csv(self.cfg.path)
            elif kind == "parquet":
                df = pd.read_parquet(self.cfg.path)
            else:
                df = pd.read_json(self.cfg.path, lines=True)
            n = len(df)
            val_n = int(self.cfg.val_split * n)
            if self.cfg.shuffle:
                df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            df_train = df.iloc[:-val_n] if val_n > 0 else df
            df_val = df.iloc[-val_n:] if val_n > 0 else None
            ds_train = _TabularDS(df_train, self.cfg.input_cols, self.cfg.target_col)
            dl_train = DataLoader(ds_train, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)
            if df_val is not None:
                ds_val = _TabularDS(df_val, self.cfg.input_cols, self.cfg.target_col)
                dl_val = DataLoader(ds_val, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)
            else:
                dl_val = None
            meta.update({"n_train": len(df_train), "n_val": len(df_val) if df_val is not None else 0, "cols": list(df.columns)})
            return DataSplit(dl_train, dl_val, meta)

        if kind == "text":
            ds = _TextFolderDS(Path(self.cfg.path))
            n = len(ds)
            idx = list(range(n))
            random.Random(42).shuffle(idx)
            val_n = int(self.cfg.val_split * n)
            idx_train = idx[:-val_n] if val_n > 0 else idx
            idx_val = idx[-val_n:] if val_n > 0 else []
            train_subset = torch.utils.data.Subset(ds, idx_train)
            val_subset = torch.utils.data.Subset(ds, idx_val) if idx_val else None
            dl_train = DataLoader(train_subset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)
            dl_val = DataLoader(val_subset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers) if val_subset else None
            meta.update({"n_train": len(idx_train), "n_val": len(idx_val)})
            return DataSplit(dl_train, dl_val, meta)

        if kind == "images":
            ds = _ImageFolderDS(Path(self.cfg.path), img_size=self.cfg.img_size)
            n = len(ds)
            idx = list(range(n))
            random.Random(42).shuffle(idx)
            val_n = int(self.cfg.val_split * n)
            idx_train = idx[:-val_n] if val_n > 0 else idx
            idx_val = idx[-val_n:] if val_n > 0 else []
            train_subset = torch.utils.data.Subset(ds, idx_train)
            val_subset = torch.utils.data.Subset(ds, idx_val) if idx_val else None
            dl_train = DataLoader(train_subset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)
            dl_val = DataLoader(val_subset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers) if val_subset else None
            meta.update({"n_train": len(idx_train), "n_val": len(idx_val)})
            return DataSplit(dl_train, dl_val, meta)

        if kind == "hf":
            assert hf_load_dataset is not None, "pip install datasets to use hf datasets"
            name = self.cfg.hf_name or self.cfg.path
            ds = hf_load_dataset(name)
            # Try to find train/validation splits
            train_split = ds.get("train") or ds[list(ds.keys())[0]]
            val_split = ds.get("validation") if "validation" in ds.keys() else None
            meta.update({"hf_name": name, "splits": list(ds.keys())})
            # Wrap huggingface dataset by returning torch tensors lazily could be added; for now, just return meta
            return DataSplit(train=train_split, val=val_split, meta=meta)  # type: ignore

        raise ValueError(f"Unsupported kind: {kind}")


def summarize_dataset(split: DataSplit) -> Dict[str, Any]:
    meta = split.meta.copy()
    meta["example_batch"] = None
    try:
        batch = next(iter(split.train))
        if isinstance(batch, (list, tuple)):
            meta["example_batch"] = str([b.shape if hasattr(b, 'shape') else type(b) for b in batch])
        else:
            meta["example_batch"] = str(batch)
    except Exception:
        pass
    return meta


def load_from_cli(argv: Optional[List[str]] = None) -> Tuple[Optional[DataSplit], DataConfig]:
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--nbx.use", action="store_true", help="Enable NBX pipeline")
    ap.add_argument("--nbx.kind", type=str, default=None, help="csv|parquet|jsonl|images|text|hf")
    ap.add_argument("--nbx.path", type=str, default=None)
    ap.add_argument("--nbx.input-cols", type=str, default="")
    ap.add_argument("--nbx.target-col", type=str, default=None)
    ap.add_argument("--nbx.batch-size", type=int, default=32)
    ap.add_argument("--nbx.val-split", type=float, default=0.1)
    ap.add_argument("--nbx.img-size", type=int, default=128)
    ap.add_argument("--nbx.hf-name", type=str, default=None)
    args, _ = ap.parse_known_args(argv)

    cfg = DataConfig(
        kind=args.nbx_kind or "csv",
        path=args.nbx_path or "",
        input_cols=[c for c in args.nbx_input_cols.split(",") if c],
        target_col=args.nbx_target_col,
        batch_size=args.nbx_batch_size,
        val_split=args.nbx_val_split,
        img_size=args.nbx_img_size,
        hf_name=args.nbx_hf_name,
    )

    if not args.nbx_use or not cfg.path:
        return None, cfg

    dp = DataPipeline(cfg)
    split = dp.build()
    meta = summarize_dataset(split)
    print("[NBX] Loaded dataset:", json.dumps(meta, indent=2, default=str))
    # Expose to executed code via environment in case script wants to read
    os.environ["NBX_DATA_KIND"] = cfg.kind
    os.environ["NBX_DATA_PATH"] = cfg.path
    os.environ["NBX_DATA_META"] = json.dumps(meta, default=str)
    return split, cfg
