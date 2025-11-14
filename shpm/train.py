from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except Exception:
    hydra = None  # type: ignore

from .models import TinySHPM, SHPMConfig, create_shpm_model
from .data import PairSchema, load_pairs
import os

# Optional Hugging Face tokenizer and Weights & Biases
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None  # type: ignore

try:
    import wandb
except Exception:
    wandb = None  # type: ignore


@dataclass
class TrainConfig:
    # data
    data_path: Optional[str] = None
    data_format: Optional[str] = None  # csv|jsonl|parquet
    task_col: str = "task"
    command_col: str = "command"

    # tokenizer/model
    use_hf_tokenizer: bool = False
    hf_tokenizer_name: str = "google/flan-t5-small"
    max_len: int = 64

    # model
    use_hf_model: bool = False  # Use HuggingFace model (SciBERT/T5) instead of TinySHPM
    hf_model_name: str = "microsoft/scibert-scivocab-uncased"  # Default SciBERT
    freeze_hf_backbone: bool = False  # Whether to freeze the HuggingFace model backbone
    vocab_size: int = 50000
    d_model: int = 512
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1

    # training
    epochs: int = 1
    batch_size: int = 8
    lr: float = 1e-3
    save_dir: str = "outputs/shpm"

    # logging
    wandb_enable: bool = False
    wandb_project: str = "shpm"
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None  # For grouped runs
    wandb_artifact_name: Optional[str] = None  # For artifact saving


def _tokenize_dummy(texts: List[str], vocab_size: int, max_len: int = 128) -> torch.Tensor:
    # Simple whitespace hashing tokenizer for a runnable stub (replace with HF tokenizer in practice)
    X = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        toks = [abs(hash(tok)) % vocab_size for tok in t.split()[:max_len]]
        if toks:
            X[i, :len(toks)] = torch.tensor(toks)
    return X


def _build_tiny_pairs(cfg: TrainConfig) -> List[Tuple[str, str]]:
    pairs = [
        ("Propose hypothesis about enzyme inhibition", "Enzyme X likely inhibited by Y at 10uM"),
        ("Design experiment to test catalyst activity", "Run reaction A at 60C with catalyst C1"),
        ("Predict outcome of adding salt", "Ionic strength increases reaction rate by ~15%"),
    ]
    # Duplicate for small batches
    return pairs * 100


def train_entry(cfg: TrainConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Data selection: prefer explicit cfg.data_path; fallback to NBX env; else synthetic
    pairs: List[Tuple[str, str]]
    data_src = "synthetic"
    nbx_env_path = os.environ.get("NBX_DATA_PATH")
    if cfg.data_path:
        pairs = load_pairs(cfg.data_path, PairSchema(cfg.task_col, cfg.command_col), cfg.data_format)
        data_src = cfg.data_path
    elif nbx_env_path:
        pairs = load_pairs(nbx_env_path, PairSchema(cfg.task_col, cfg.command_col), None)
        data_src = nbx_env_path
    else:
        pairs = _build_tiny_pairs(cfg)
        data_src = "synthetic"

    print(f"[SHPM] Using data: {data_src} (pairs={len(pairs)})")

    inputs = [f"Q: {t}" for t, _ in pairs]
    targets = [c for _, c in pairs]

    # Tokenization
    if cfg.use_hf_tokenizer and AutoTokenizer is not None:
        tok = AutoTokenizer.from_pretrained(cfg.hf_tokenizer_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if hasattr(tok, "eos_token") and tok.eos_token else tok.unk_token
        X = tok(inputs, padding=True, truncation=True, max_length=cfg.max_len, return_tensors="pt")
        Y = tok(targets, padding=True, truncation=True, max_length=cfg.max_len, return_tensors="pt")
        # Use attention masks for HuggingFace models
        X_input_ids = X["input_ids"]
        X_attention_mask = X["attention_mask"]
        Y_input_ids = Y["input_ids"]
        Y_attention_mask = Y["attention_mask"]
        vocab_size = tok.vocab_size if hasattr(tok, "vocab_size") and tok.vocab_size is not None else int(max(X_input_ids.max().item(), Y_input_ids.max().item()) + 10)
        ignore_index = tok.pad_token_id if tok.pad_token_id is not None else 0
    else:
        tok = None
        X_input_ids = _tokenize_dummy(inputs, cfg.vocab_size, max_len=cfg.max_len)
        Y_input_ids = _tokenize_dummy(targets, cfg.vocab_size, max_len=cfg.max_len)
        X_attention_mask = torch.ones_like(X_input_ids)
        Y_attention_mask = torch.ones_like(Y_input_ids)
        vocab_size = cfg.vocab_size
        ignore_index = 0

    # Model - use the factory function to create the appropriate model
    model_config = SHPMConfig(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        use_hf_model=cfg.use_hf_model,
        hf_model_name=cfg.hf_model_name,
        hf_model_type="bert" if "bert" in cfg.hf_model_name.lower() else "t5",  # Simplified model type detection
        freeze_hf_backbone=cfg.freeze_hf_backbone
    )
    model = create_shpm_model(model_config)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # W&B init (optional) - with tags and config
    if cfg.wandb_enable and wandb is not None:
        # Determine tags based on model and data used
        tags = cfg.wandb_tags or []
        if cfg.use_hf_model:
            tags.append("hf_model")
            tags.append(cfg.hf_model_name.replace("/", "_"))
        else:
            tags.append("tiny_shpm")
        if "synthetic" in data_src:
            tags.append("synthetic_data")
        else:
            tags.append("real_data")
            
        wandb.init(
            project=cfg.wandb_project, 
            name=cfg.wandb_run_name, 
            config={k: getattr(cfg, k) for k in vars(cfg)},
            tags=tags
        )

    # Tiny loop
    steps = max(1, len(pairs) // cfg.batch_size)
    for epoch in range(cfg.epochs):
        loss_epoch = 0.0
        for i in range(steps):
            bX_input_ids = X_input_ids[i*cfg.batch_size:(i+1)*cfg.batch_size].to(device)
            bX_attention_mask = X_attention_mask[i*cfg.batch_size:(i+1)*cfg.batch_size].to(device)
            bY = Y_input_ids[i*cfg.batch_size:(i+1)*cfg.batch_size].to(device)
            
            out = model(input_ids=bX_input_ids, attention_mask=bX_attention_mask)
            logits = out["hypothesis_logits"]
            tgt = bY
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=ignore_index)
            loss = ce
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_epoch += float(loss.detach().cpu().item())
        avg_loss = loss_epoch/steps
        print(f"[SHPM] epoch {epoch+1}/{cfg.epochs} loss={avg_loss:.4f}")
        if cfg.wandb_enable and wandb is not None:
            wandb.log({"loss": avg_loss, "pairs": len(pairs), "epoch": epoch+1, "data_source": data_src})

    # Save model with Weights & Biases artifact support
    if cfg.use_hf_model:
        # For HuggingFace models, save the full model
        model_save_path = os.path.join(cfg.save_dir, "hf_shpm")
        os.makedirs(model_save_path, exist_ok=True)
        model.hf_model.save_pretrained(model_save_path)
        if tok:
            tok.save_pretrained(model_save_path)
        
        # Log as artifact if W&B is enabled
        if cfg.wandb_enable and wandb is not None and cfg.wandb_artifact_name:
            artifact = wandb.Artifact(cfg.wandb_artifact_name or f"shpm-model-{wandb.run.id}", type="model")
            artifact.add_dir(model_save_path)
            wandb.log_artifact(artifact)
    else:
        # For TinySHPM, save the state dict
        torch.save(model.state_dict(), os.path.join(cfg.save_dir, "tiny_shpm.pt"))
        
        # Log as artifact if W&B is enabled
        if cfg.wandb_enable and wandb is not None and cfg.wandb_artifact_name:
            artifact = wandb.Artifact(cfg.wandb_artifact_name or f"shpm-model-{wandb.run.id}", type="model")
            artifact.add_file(os.path.join(cfg.save_dir, "tiny_shpm.pt"))
            wandb.log_artifact(artifact)

    print("[SHPM] Saved to", cfg.save_dir)


if hydra is not None:
    @hydra.main(config_path="../configs", config_name="shpm", version_base=None)
    def hydra_main(hcfg: DictConfig):
        # Map DictConfig to TrainConfig
        tcfg = TrainConfig(**{k: v for k, v in hcfg.items()})
        train_entry(tcfg)
else:
    def hydra_main():  # type: ignore
        raise RuntimeError("hydra not installed")
