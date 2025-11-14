# Meta-Dynamics Extended: Notes

Goal
- Meta-learn to predict learning dynamics (Δθ) and downstream loss effects from per-layer summaries across time.

Key components
- Curriculum synthetic task generator with difficulty controls (noise, covariate shift, imbalance)
- Snapshot logger: captures per-layer stats (mean, std, grad_mean, numel) and loss
- Meta-model: Layer encoder (MLP) -> GRU over time -> decoder to multipliers and predicted loss
- Meta-objectives:
  - L_dyn: MSE between predicted Δθ proxy and observed Δθ (mean per-parameter update)
  - L_down: loss after applying predicted update to a fresh copy (proxy for task performance)

Why it can work
- Layer-level summaries capture low-frequency structure in dynamics sufficient for short-horizon predictions.
- Multipliers modulate current gradient directions, approximating a learned optimizer step.

How to extend
- Replace scalar-per-layer with low-rank update (UΛV^T) predicted per layer.
- Add Hessian-aware features (K-FAC approximations, curvature surrogates).
- Use implicit differentiation for longer unrolls.
- Add entropy-based curriculum: sample tasks that maximize information gain on meta-parameters.

Practical tips
- Normalize per-layer stats (mean/std) across tasks to stabilize meta-training.
- Batch multiple trajectories and time indices to improve signal.
- Log and visualize layer multipliers over epochs (TensorBoard).
