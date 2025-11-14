from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import json

try:
    import pandas as pd
except Exception:
    pd = None


@dataclass
class PairSchema:
    task_col: str = "task"
    command_col: str = "command"


def validate_pairs_df(df, schema: PairSchema):
    cols = set(df.columns)
    if schema.task_col not in cols or schema.command_col not in cols:
        raise ValueError(f"Missing required columns: {schema.task_col}, {schema.command_col}")
    # Optional: ensure string-like
    return df


def load_pairs(path: str, schema: PairSchema, fmt: Optional[str] = None) -> List[Tuple[str, str]]:
    """Load (task, command) pairs from CSV/JSONL/Parquet.
    fmt can be 'csv'|'jsonl'|'parquet' or inferred from extension.
    """
    ext = fmt or Path(path).suffix.lstrip('.').lower()
    if ext == 'csv':
        assert pd is not None, "pandas required for CSV"
        df = pd.read_csv(path)
        df = validate_pairs_df(df, schema)
        pairs = [(str(t), str(c)) for t, c in zip(df[schema.task_col], df[schema.command_col])]
        return [(t, c) for t, c in pairs if t and c]
    if ext in ('parquet', 'pq'):
        assert pd is not None, "pandas required for Parquet"
        df = pd.read_parquet(path)
        df = validate_pairs_df(df, schema)
        pairs = [(str(t), str(c)) for t, c in zip(df[schema.task_col], df[schema.command_col])]
        return [(t, c) for t, c in pairs if t and c]
    if ext in ('jsonl', 'json'):
        pairs: List[Tuple[str, str]] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = obj.get(schema.task_col)
                c = obj.get(schema.command_col)
                if t and c:
                    pairs.append((str(t), str(c)))
        return pairs
    raise ValueError(f"Unsupported format: {ext}")
