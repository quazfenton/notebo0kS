# SHPM (Scientific-Hypothesis Predictor Macro): Extended Notes

Purpose
- Provide a neurosymbolic baseline for hypothesis generation, experimental design, and outcome modeling in a compact, runnable form.
- Strive for parsimony: small transformer, clear data schema, and optional Hydra.

Architecture overview
- TinySHPM
  - Embedding + positional encoding
  - N transformer layers (MultiheadAttention + FFN with LayerNorm residuals)
  - Two heads: hypothesis_logits (autoregressive next-token) and outcome vector head (placeholder for regression tasks)
- Tokenizer
  - A hashing-based stub in this repo; replace with a Hugging Face tokenizer for real workloads.

Data and schema
- Expected schema for supervised pairs is (task, command) by default.
- shpm.data.load_pairs supports csv/jsonl/parquet and validates required columns via PairSchema.
- SHPM trainer prefers cfg.data_path (Hydra), then NBX_DATA_PATH (if set), else synthetic pairs.

Training loop (high-level)
- Tokenize input/output into integer IDs (stub)
- Forward pass through TinySHPM
- Cross-entropy over hypothesis_logits against target IDs (pad/zero ignored)
- Simple AdamW optimization

Information-theoretic rationale (intuitive)
- Hypothesis generation benefits from strong priors (token embeddings and transformer layers capture latent ‘grammar’ of domain text).
- Outcome head represents downstream predictive goals: outcome distributions link hypotheses and empirical measurements—ideally calibrated uncertainty.
- Minimal MDL bias: simpler models/outputs with adequate accuracy are preferred (implicit via small model and regularization).

Future upgrades
- Replace hashing tokenizer with pretrained SciBERT tokenizer and tie embeddings.
- Add uncertainty-aware outcome head (Gaussian NLL + calibration metrics).
- Add KG cross-attention or adapters to form a true neuro-symbolic hybrid.
- Integrate acquisition (EIG proxy) for active data selection.

Run examples
- Hydra:
  python -c "from shpm.train import hydra_main; hydra_main()" epochs=1 data_path=data/pairs.csv data_format=csv

- Programmatic:
  from shpm import train_entry, TrainConfig
  cfg = TrainConfig(epochs=1, data_path="data/pairs.parquet", data_format="parquet")
  train_entry(cfg)

Limitations
- Current tokenizer is a stub; use real tokenizers for non-trivial corpora.
- Outcome head is a placeholder. Plug real supervision when available.
