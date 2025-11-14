#!/usr/bin/env python3
"""
Terminal Machine Learning Runner (sanitized)
-------------------------------------------
A clean, runnable training script extracted from the tErMiNAlmachineL prototype.

Features:
- NBX optional data pipeline flags to discover datasets
- CSV ingestion for (task, command) supervised pairs
- Synthetic dataset fallback for quick demo
- Seq2Seq training using a small T5-family model by default (flan-t5-small)
- Saves model + tokenizer to outputs/terminal_ml/

Usage examples:
  # Synthetic quick demo
  python terminal_machineL_runner.py --epochs 1 --batch-size 8

  # CSV training via NBX
  python terminal_machineL_runner.py --nbx.use --nbx.kind csv --nbx.path data/commands.csv \
    --task-col task --command-col command --epochs 2 --batch-size 8

Notes:
- You can switch to CodeT5 or other seq2seq models with --model-name.
- For very large models (e.g., CodeLlama), use Accelerate or DeepSpeed.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import json
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import pandas as pd
except Exception:
    pd = None

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)

# Optional Hydra
try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except Exception:
    hydra = None  # type: ignore

# Optional NBX loader (no-op if flags not provided)
try:
    from nbx.pipeline import load_from_cli
except Exception:
    load_from_cli = None  # type: ignore

# Optional Accelerate for distributed training and optimization
try:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    ACCELERATE_AVAILABLE = True
except ImportError:
    Accelerator = None
    ACCELERATE_AVAILABLE = False

# Optional DeepSpeed
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False


@dataclass
class RunnerConfig:
    model_name: str = "google/flan-t5-small"
    epochs: int = 1
    batch_size: int = 8
    lr: float = 5e-5
    warmup_steps: int = 50
    max_source_len: int = 256
    max_target_len: int = 128
    save_dir: str = "outputs/terminal_ml"
    task_col: str = "task"
    command_col: str = "command"
    seed: int = 42
    # logging
    wandb_enable: bool = False
    wandb_project: str = "terminal-ml"
    wandb_run_name: Optional[str] = None
    # offline RL fine-tune
    rl_enable: bool = False
    rl_epochs: int = 1
    rl_weight: float = 0.2
    # Accelerate and DeepSpeed options
    use_accelerate: bool = False
    use_deepspeed: bool = False
    deepspeed_config_path: Optional[str] = None


class CommandDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.pairs[idx]


def build_synthetic_data(n: int = 200) -> List[Tuple[str, str]]:
    base = [
        ("List all files in current directory", "ls -la"),
        ("Change to home directory", "cd ~"),
        ("Create a new file named test.txt", "touch test.txt"),
        ("Search for error in syslog", "grep 'error' /var/log/syslog | sort | uniq"),
        ("Install lodash with npm", "npm install lodash"),
        ("Run Python script with args", "python script.py --input data.csv"),
        ("Show current working directory", "pwd"),
        ("Display first 10 lines of file.txt", "head -n 10 file.txt"),
    ]
    # repeat to reach n
    out: List[Tuple[str, str]] = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out


def _validate_columns(df, task_col: str, command_col: str):
    if task_col not in df.columns or command_col not in df.columns:
        raise ValueError(f"Dataset must contain columns '{task_col}' and '{command_col}' (found: {list(df.columns)})")


def load_pairs_from_csv(path: str, task_col: str, command_col: str) -> List[Tuple[str, str]]:
    assert pd is not None, "pandas is required for CSV ingestion"
    df = pd.read_csv(path)
    _validate_columns(df, task_col, command_col)
    pairs = [(str(t), str(c)) for t, c in zip(df[task_col], df[command_col])]
    pairs = [(t, c) for t, c in pairs if (t and c and t != 'nan' and c != 'nan')]
    return pairs


def load_pairs_from_parquet(path: str, task_col: str, command_col: str) -> List[Tuple[str, str]]:
    assert pd is not None, "pandas is required for Parquet ingestion"
    df = pd.read_parquet(path)
    _validate_columns(df, task_col, command_col)
    return [(str(t), str(c)) for t, c in zip(df[task_col], df[command_col]) if t and c]


def load_pairs_from_jsonl(path: str, task_col: str, command_col: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get(task_col)
            c = obj.get(command_col)
            if t and c:
                pairs.append((str(t), str(c)))
    return pairs


def collate_fn(tokenizer: AutoTokenizer, max_source_len: int, max_target_len: int):
    def _collate(batch: List[Tuple[str, str]]):
        tasks = [f"Task: {t}\nCommand:" for t, _ in batch]
        commands = [c for _, c in batch]
        enc = tokenizer(
            tasks,
            padding=True,
            truncation=True,
            max_length=max_source_len,
            return_tensors="pt",
        )
        with tokenizer.as_target_tokenizer():
            dec = tokenizer(
                commands,
                padding=True,
                truncation=True,
                max_length=max_target_len,
                return_tensors="pt",
            )
        labels = dec["input_ids"]
        # Replace pad token id with -100 to ignore in loss
        labels[labels == tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc

    return _collate


def compute_reward(pred_command: str, true_command: str, task: str) -> float:
    # Safe, heuristic reward
    task_completion = 1.0 if pred_command.strip() == true_command.strip() else 0.5
    efficiency = 1.0 / max(len(pred_command.split()), 1)
    safety = 1.0 if 'rm -rf' not in pred_command else -1.0
    generalization = 0.8
    alphas = [0.4, 0.2, 0.2, 0.2]
    return sum(a * s for a, s in zip(alphas, [task_completion, efficiency, safety, generalization]))


# Optional W&B
try:
    import wandb
except Exception:
    wandb = None  # type: ignore


def train(cfg: RunnerConfig, pairs: List[Tuple[str, str]]):
    os.makedirs(cfg.save_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    model.train()

    # Initialize Accelerate or DeepSpeed based on configuration
    if cfg.use_accelerate and ACCELERATE_AVAILABLE:
        # Initialize the accelerator
        accelerator = Accelerator()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        total_steps = len(pairs) // cfg.batch_size * cfg.epochs  # Calculate steps differently
        sched = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
        
        # Prepare model, optimizer, scheduler, and dataloader with accelerator
        model, optimizer, sched = accelerator.prepare(model, optimizer, sched)
        
        # Create DataLoader after accelerator preparation
        dataset = CommandDataset(pairs)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn(tokenizer, cfg.max_source_len, cfg.max_target_len),
        )
        loader = accelerator.prepare(loader)
        
        device = accelerator.device
    elif cfg.use_deepspeed and DEEPSPEED_AVAILABLE:
        # Initialize DeepSpeed
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config_params={
                "train_batch_size": cfg.batch_size,
                "gradient_clipping": 1.0,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": cfg.lr,
                        "weight_decay": 0.01
                    }
                },
                "scheduler": {
                    "type": "WarmupLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": cfg.lr,
                        "warmup_num_steps": cfg.warmup_steps,
                    }
                }
            } if not cfg.deepspeed_config_path else None,
            config=cfg.deepspeed_config_path  # Use provided config file if available
        )
        # Note: In DeepSpeed, the optimizer and scheduler are handled by deepspeed
        
        # Create DataLoader for DeepSpeed
        dataset = CommandDataset(pairs)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn(tokenizer, cfg.max_source_len, cfg.max_target_len),
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # DeepSpeed manages this internally
    else:
        # Standard PyTorch training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Create DataLoader
        dataset = CommandDataset(pairs)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn(tokenizer, cfg.max_source_len, cfg.max_target_len),
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        total_steps = len(loader) * cfg.epochs
        sched = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)

    # W&B init
    if cfg.wandb_enable and wandb is not None:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config={k: getattr(cfg, k) for k in vars(cfg)})

    step = 0
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for batch in loader:
            if cfg.use_accelerate and ACCELERATE_AVAILABLE:
                # Use accelerator's prepared batch
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                sched.step()
                optimizer.zero_grad()
            elif cfg.use_deepspeed and DEEPSPEED_AVAILABLE:
                # Use DeepSpeed's training step
                outputs = model(**batch)
                loss = outputs.loss
                model.backward(loss)
                model.step()
            else:
                # Standard PyTorch training
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += float(loss.detach().cpu().item())
            step += 1
        avg_loss = epoch_loss/len(loader)
        print(f"Epoch {epoch+1}/{cfg.epochs} loss={avg_loss:.4f}")
        if cfg.wandb_enable and wandb is not None:
            wandb.log({"epoch": epoch+1, "train_loss": avg_loss})

    # Save model - handle distributed training cases
    if cfg.use_accelerate and ACCELERATE_AVAILABLE:
        # Save only on main process
        if accelerator.is_main_process:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(cfg.save_dir)
            tokenizer.save_pretrained(cfg.save_dir)
    elif cfg.use_deepspeed and DEEPSPEED_AVAILABLE:
        # For DeepSpeed, save the model using DeepSpeed's save function
        model.save_checkpoint(cfg.save_dir)
    else:
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)
    
    print("Saved model to", cfg.save_dir)

    return model, tokenizer


def offline_rl_finetune(cfg: RunnerConfig, model, tokenizer, pairs: List[Tuple[str, str]]):
    device = next(model.parameters()).device
    model.train()
    dataset = CommandDataset(pairs)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer, cfg.max_source_len, cfg.max_target_len),
    )
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.rl_epochs):
        total_loss = 0.0
        total_reward = 0.0
        n_examples = 0
        for batch in loader:
            tasks = []
            trues = []
            # Reconstruct raw texts from batch for reward calc
            # We approximate by decoding labels (with pads removed)
            input_ids = batch['input_ids']
            labels = batch['labels']
            for i in range(input_ids.size(0)):
                # task text is embedded in input_ids; decode prompt
                t = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                tasks.append(t)
                true_cmd_ids = labels[i]
                true_cmd_ids = true_cmd_ids[true_cmd_ids != -100]
                trues.append(tokenizer.decode(true_cmd_ids, skip_special_tokens=True))

            # Generate predictions
            with torch.no_grad():
                gen = model.generate(input_ids=input_ids.to(device), attention_mask=batch['attention_mask'].to(device), max_new_tokens=min(64, cfg.max_target_len))
            preds = [tokenizer.decode(g, skip_special_tokens=True) for g in gen]

            # Compute rewards and normalize
            rewards = [compute_reward(p, y, t) for p, y, t in zip(preds, trues, tasks)]
            rmin, rmax = min(rewards), max(rewards)
            denom = (rmax - rmin) if (rmax - rmin) > 1e-8 else 1.0
            rnorm = [(r - rmin) / denom for r in rewards]

            # Teacher-forced pass for loss with per-example weights
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits  # [B,T,V]
            B, T, V = logits.shape
            labels = batch['labels']  # [B,T]
            ce = torch.nn.functional.cross_entropy(
                logits.view(-1, V), labels.view(-1), ignore_index=-100, reduction='none'
            ).view(B, T)
            valid = (labels != -100).float()
            ce_per_ex = (ce * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-8)
            weights = torch.tensor([1.0 - cfg.rl_weight * rn for rn in rnorm], device=device).float()
            loss = (ce_per_ex * weights).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += float(loss.detach().cpu().item())
            total_reward += float(sum(rewards))
            n_examples += len(rewards)
        avg_loss = total_loss / max(1, len(loader))
        avg_reward = total_reward / max(1, n_examples)
        print(f"[RL] epoch {epoch+1}/{cfg.rl_epochs} loss={avg_loss:.4f} avg_reward={avg_reward:.4f}")
        if cfg.wandb_enable and wandb is not None:
            wandb.log({"rl_epoch": epoch+1, "rl_loss": avg_loss, "avg_reward": avg_reward})


def _run_with_cfg(cfg: RunnerConfig, nbx_path: Optional[str]):
    if nbx_path and os.path.exists(nbx_path):
        ext = os.path.splitext(nbx_path)[1].lower()
        if ext == '.csv':
            pairs = load_pairs_from_csv(nbx_path, cfg.task_col, cfg.command_col)
        elif ext in ('.parquet', '.pq'):
            pairs = load_pairs_from_parquet(nbx_path, cfg.task_col, cfg.command_col)
        elif ext in ('.jsonl', '.json'):
            pairs = load_pairs_from_jsonl(nbx_path, cfg.task_col, cfg.command_col)
        else:
            print(f"Unsupported data format: {ext}; using synthetic dataset.")
            pairs = build_synthetic_data(200)
    else:
        pairs = build_synthetic_data(200)

    model, tokenizer = train(cfg, pairs)
    if cfg.rl_enable:
        offline_rl_finetune(cfg, model, tokenizer, pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="google/flan-t5-small")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--max-source-len", type=int, default=256)
    ap.add_argument("--max-target-len", type=int, default=128)
    ap.add_argument("--save-dir", type=str, default="outputs/terminal_ml")
    ap.add_argument("--task-col", type=str, default="task")
    ap.add_argument("--command-col", type=str, default="command")
    # W&B and RL flags
    ap.add_argument("--wandb-enable", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="terminal-ml")
    ap.add_argument("--wandb-run-name", type=str, default=None)
    ap.add_argument("--rl-enable", action="store_true")
    ap.add_argument("--rl-epochs", type=int, default=1)
    ap.add_argument("--rl-weight", type=float, default=0.2)
    # Accelerate and DeepSpeed flags
    ap.add_argument("--use-accelerate", action="store_true", help="Use HuggingFace Accelerate for training")
    ap.add_argument("--use-deepspeed", action="store_true", help="Use DeepSpeed for training")
    ap.add_argument("--deepspeed-config", type=str, help="Path to DeepSpeed configuration file")
    # NBX passthrough flags handled inside load_from_cli
    args, _ = ap.parse_known_args()

    cfg = RunnerConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        save_dir=args.save_dir,
        task_col=args.task_col,
        command_col=args.command_col,
        wandb_enable=bool(args.wandb_enable),
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        rl_enable=bool(args.rl_enable),
        rl_epochs=args.rl_epochs,
        rl_weight=args.rl_weight,
        use_accelerate=bool(args.use_accelerate),
        use_deepspeed=bool(args.use_deepspeed),
        deepspeed_config_path=args.deepspeed_config,
    )

    pairs: List[Tuple[str, str]]

    # Optional NBX discovery
    nbx_path = None
    if load_from_cli is not None:
        try:
            split, nbx_cfg = load_from_cli()
            if nbx_cfg and nbx_cfg.path:
                nbx_path = nbx_cfg.path
        except Exception:
            pass

    _run_with_cfg(cfg, nbx_path)


# Optional Hydra entry (use: python terminal_machineL_runner.py --hydra)
if hydra is not None:
    @hydra.main(config_path="configs", config_name="terminal_ml", version_base=None)
    def hydra_main(hcfg: DictConfig):
        hcfg = OmegaConf.to_container(hcfg, resolve=True)
        rcfg = RunnerConfig(**hcfg)  # type: ignore
        nbx_path = os.environ.get("NBX_DATA_PATH")
        _run_with_cfg(rcfg, nbx_path)

if __name__ == "__main__":
    import sys
    if hydra is not None and "--hydra" in sys.argv:
        hydra_main()  # type: ignore
    else:
        main()
