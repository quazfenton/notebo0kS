# Offline RL Fine-Tune for Terminal Runner

Motivation
- Supervised training ensures the model learns to imitate ground-truth commands. Reward shaping can bias learning toward safer, shorter, or more generalizable commands without executing them.

Method
- For each (task, command) pair:
  1. Generate a predicted command from the current model via greedy decoding.
  2. Compute a safe heuristic reward `R` = f(task, predicted_command, true_command) combining:
     - Task completion (exact match proxy)
     - Efficiency (shorter sequences preferred)
     - Safety (penalize risky patterns like `rm -rf`)
     - Generalization (simple prior)
  3. Normalize rewards per batch to [0, 1].
  4. Compute teacher-forced cross-entropy, aggregate per-example loss, and weight by `(1 - rl_weight * R_norm)`. Examples with higher reward get lower effective loss weight, nudging the model toward higher-reward behavior.

Notes
- This is offline and safe: no command execution.
- For real-world evaluation, replace reward with environment outcomes in a sandbox.

Enabling
- CLI (argparse):
  python terminal_machineL_runner.py --rl-enable --rl-epochs 1 --rl-weight 0.2

- Hydra:
  python terminal_machineL_runner.py --hydra rl_enable=true rl_epochs=1 rl_weight=0.2

Logging
- If W&B is enabled, RL metrics (rl_loss, avg_reward) are logged per RL epoch.
