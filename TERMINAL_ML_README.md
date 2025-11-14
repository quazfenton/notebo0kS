# Terminal ML Runner

A clean training script for natural-language-to-command generation extracted from the prototype (tErMiNAlmachineL.py).

Quickstart

- Synthetic demo (no data files):

  python terminal_machineL_runner.py --epochs 1 --batch-size 8

- CSV via NBX flags (expects columns `task`, `command` by default):

  python terminal_machineL_runner.py --nbx.use --nbx.kind csv --nbx.path data/commands.csv \
    --task-col task --command-col command --epochs 2 --batch-size 8

Key options

- --model-name (default: google/flan-t5-small). Use another seq2seq model if you prefer CodeT5.
- --save-dir (default: outputs/terminal_ml)
- --max-source-len, --max-target-len
- --task-col, --command-col (CSV columns)

Notes

- The runner uses Seq2Seq training with label masking for pads and saves model+tokenizer.
- For large models or longer runs, consider `accelerate` or DeepSpeed.
- If you want me to build an RL phase (execute commands in a sandbox, shaped rewards), say the word and I’ll scaffold it with safety guards.
