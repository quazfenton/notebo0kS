# Warp: Advanced Reward Architectures (World-Model + Slot-Program)

This repository scaffolds two research tracks:
- A) Latent World-Model with Differentiable Physics + Active Inference
- B) Compositional Slot-Latent + Symbolic Program Head with communication games

It includes:
- Synthetic data generators (ODE systems, CLEVR-like shapes)
- Novelty / curriculum scheduling and information-gain bonuses
- Training scripts and minimal notebooks
- Evaluation utilities and ablation runners

Quickstart (after installing deps):
- Synthetic CLEVR: python scripts/synth_generate_clevr.py --n 256
- Train world model (smoke): python scripts/train_worldmodel.py steps=200 tiny=true
- Train slot-program (smoke): python scripts/train_slotprog.py steps=200 tiny=true

See notebooks/ for interactive demos.

---

Background: terms and rationale (introductory overview)

- Symplectic/Hamiltonian dynamics: Many physical systems conserve a Hamiltonian H(q,p). A symplectic integrator (e.g., leapfrog) approximately preserves phase-space volume and energy over long horizons, reducing drift. In our world-model, we learn H(z) and use a symplectic step in latent space (z=(q,p)) to bias learning towards stable rollouts. Minimizing a physics residual (difference between learned step and symplectic flow) acts as an inductive prior.

- Predictive coding and residuals: If sender and receiver share a world-model, the sender can transmit only the residual between predicted and actual observations. This yields bandwidth savings that approach channel capacity for structured streams. Our world-model sets up this shared latent dynamics.

- Expected Information Gain (EIG): EIG measures the expected reduction in uncertainty about parameters or latent state after observing outcomes from an action a:
  EIG(a) = H(θ|D) − E_{x'|a}[ H(θ|D ∪ {(a,x')}) ].
  Exact EIG is hard; we include practical surrogates:
  - Variance proxy: latent trajectory variance encourages broader coverage.
  - Ensemble disagreement (weight noise): clone the dynamics with small weight perturbations; variance of next-step predictions ≈ information gain.
  - MC Dropout disagreement: activate Dropout in the dynamics and compute variance across multiple stochastic passes.
  Configure in configs/worldmodel.yaml under eig.use_dropout (with eig.mc_samples, eig.dropout_p) or eig.use_ensemble (with eig.ensemble_size, eig.weight_noise_scale).

- Posterior entropy quantiles (novelty triggers): We track a running entropy-like signal (variance for the world-model; token posterior entropy for the slot-program) and compute its 85th-quantile. If the current entropy exceeds the quantile, we trigger novelty: Gaussian perturbations of inputs and (for slot-program) harder channels and curricula. This injects structured “surprise” and reduces stagnation, akin to domain randomization and curiosity-driven learning.

- AWGN channel and SNR (dB): The communication game uses an additive white Gaussian noise channel. Signal-to-noise ratio (SNR) in decibels controls difficulty (lower SNR = harder). We automatically step SNR down by comm.snr_step_db upon novelty triggers and clamp within [comm.snr_min_db, comm.snr_max_db], exposing robust coding.

- Emergent codebooks: Sender and receiver are trained end-to-end. Under noise and length budgets, they often discover compressive, error-tolerant encodings tailored to the data/program distribution.

- MDL / program length penalty: Shorter programs are favored via a length/entropy penalty, reflecting Occam’s razor: prefer simpler generative grammars that still reconstruct the input.

- Slot Attention (compositionality): Attends from a fixed number of “slot” vectors to input tokens (image patches), competing to explain parts/objects. This disentangling enhances compositional generalization when combined with a program head and interpreter.

- Curriculum learning (adaptive difficulty): Upon novelty triggers, we can increase scene complexity (object count or resolution). Optionally, pre-generate difficulty buckets (curriculum.bucketed) and advance levels on triggers to avoid runtime regeneration. This keeps training in the “zone of proximal difficulty,” stabilizing progress.

Key config knobs

- World-model: configs/worldmodel.yaml
  - model.symplectic (true/false)
  - loss.info_gain (weight)
  - eig.use_ensemble, eig.ensemble_size, eig.weight_noise_scale
  - novelty.entropy_quantile, novelty.perturb_scale

- Slot-program: configs/slotprog.yaml
  - model.slots, model.program_vocab, model.program_len_max
  - comm.channel_snr_db, comm.snr_step_db, comm.snr_min_db, comm.snr_max_db, comm.code_len
  - novelty.posterior_entropy_q, novelty.mutate_prob
  - curriculum.enable, curriculum.inc_min_objs, curriculum.inc_max_objs, curriculum.inc_img_size, curriculum.max_objs_cap, curriculum.img_size_cap, curriculum.regen_n
  - curriculum.bucketed.enable, curriculum.bucketed.levels, curriculum.bucketed.per_level_n,
    curriculum.bucketed.base_min_objs/base_max_objs/base_img_size, curriculum.bucketed.obj_step/img_step

Philosophical note

These choices favor universal structure over narrow benchmarks: physics priors harness conservation laws; MDL penalization encodes parsimony; EIG-driven novelty aligns with the thermodynamic view of learning as free-energy reduction; and compositionality mirrors renormalization-like reuse of primitives across scales. The goal is not raw scale, but architectures that improve their improvement processes.
