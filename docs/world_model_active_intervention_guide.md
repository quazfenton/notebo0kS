# World-Model + Active Intervention: Extended Notes

Core idea
- Learn a latent transition model (e.g., RSSM-like) of a controllable environment; choose interventions (actions) that maximize expected information gain about a target parameter (theta).

Acquisition approximations
- Ensemble disagreement: predictive entropy of ensemble mean minus mean entropies approximates mutual information.
- Bayesian last-layer: analytic predictive variance with ridge posterior on top of learned features.

Planning (CEM)
- Optimize action sequences to maximize acquisition with population-based sampling and elite refitting.

Replay + training
- Collect short episodes under random policy → train world-model → refit predictors → propose next experiment → repeat.

Why it’s principled
- Interventions that reduce posterior uncertainty fastest lead to sample-efficient discovery.
- Separation between world-model learning and acquisition reduces bias from myopic rewards.

Extensions
- Use a proper variational world model (Dreamer/TD-VAE) for image observations.
- Replace random rollouts with stochastic latents for predictive ensembles.
- Add constraints (safety, cost) to the planner objective.

Diagnostics
- Plot EIG over rounds, predictive calibration, and replay diversity.
