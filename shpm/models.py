from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    AutoModel, AutoTokenizer, AutoConfig = None, None, None


@dataclass
class SHPMConfig:
    vocab_size: int = 50000
    d_model: int = 512
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    # HuggingFace model parameters
    use_hf_model: bool = False
    hf_model_name: str = "microsoft/scibert-scivocab-uncased"  # Default SciBERT
    hf_model_type: str = "bert"  # bert, t5, etc.
    freeze_hf_backbone: bool = False  # Whether to freeze the backbone model


class HFBackboneSHPM(nn.Module):
    """SHPM model using HuggingFace models (SciBERT/T5) as backbone.
    This uses pre-trained models for better performance while maintaining
    the hypothesis and outcome prediction structure.
    """
    
    def __init__(self, cfg: SHPMConfig):
        super().__init__()
        self.cfg = cfg
        
        if not HF_AVAILABLE:
            raise ImportError("transformers library is required for HuggingFace models")
        
        # Load pre-trained model
        self.hf_model = AutoModel.from_pretrained(cfg.hf_model_name)
        self.config = AutoConfig.from_pretrained(cfg.hf_model_name)
        
        # Determine embedding dimension from the model
        if hasattr(self.config, 'hidden_size'):
            self.d_model = self.config.hidden_size
        elif hasattr(self.config, 'd_model'):
            self.d_model = self.config.d_model
        else:
            self.d_model = cfg.d_model  # fallback
            
        # Freeze backbone if specified
        if cfg.freeze_hf_backbone:
            for param in self.hf_model.parameters():
                param.requires_grad = False
        
        # Additional heads for hypothesis and outcome prediction
        self.hypothesis_head = nn.Linear(self.d_model, self.config.vocab_size if hasattr(self.config, 'vocab_size') else cfg.vocab_size)
        self.outcome_head = nn.Linear(self.d_model, 256)  # generic outcome vector
        
        # Additional linear layer if needed to match expected dimensions
        if self.d_model != cfg.d_model:
            self.projection = nn.Linear(self.d_model, cfg.d_model)
        else:
            self.projection = nn.Identity()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Forward pass through the pre-trained model
        outputs = self.hf_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the sequence output (last hidden states)
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        else:
            # For models that don't have last_hidden_state, use the hidden state directly
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Apply projection if needed
        hidden_states = self.projection(hidden_states)
        
        # Hypothesis logits for next-token prediction
        hyp_logits = self.hypothesis_head(hidden_states)
        
        # Outcome prediction from the last token representation
        outcome = self.outcome_head(hidden_states[:, -1, :])  # [batch_size, 256]
        
        return {"hypothesis_logits": hyp_logits, "outcome": outcome}


class TinyTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.self_attn(x, x, x, attn_mask=attn_mask)[0])
        x = self.norm2(x + self.ff(x))
        return x


class TinySHPM(nn.Module):
    """Minimal text transformer with heads for hypothesis and outcome.
    This avoids heavy external dependencies while providing a usable baseline.
    """

    def __init__(self, cfg: SHPMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(1024, cfg.d_model)
        self.layers = nn.ModuleList([TinyTransformerLayer(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])
        self.hypothesis_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.outcome_head = nn.Linear(cfg.d_model, 256)  # generic outcome vector

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos(pos)
        for layer in self.layers:
            x = layer(x, None)
        # logits for next-token prediction
        hyp_logits = self.hypothesis_head(x)
        outcome = self.outcome_head(x[:, -1])
        return {"hypothesis_logits": hyp_logits, "outcome": outcome}


def create_shpm_model(cfg: SHPMConfig) -> nn.Module:
    """Factory function to create appropriate SHPM model based on config"""
    if cfg.use_hf_model and HF_AVAILABLE:
        return HFBackboneSHPM(cfg)
    else:
        return TinySHPM(cfg)
