#!/usr/bin/env python3
"""Train slot-program model on synthetic CLEVR with optional comm channel.
Hydra config: configs/slotprog.yaml
Run:
  python scripts/train_slotprog.py steps=2000 tiny=true
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

from src.common.losses import ReconLoss, LossWeights, aggregate_losses
from src.common.visualization import TBLogger
from src.slot_program.slot_attention import SlotAttention
from src.slot_program.grammar import ProgramHead, Interpreter
from src.data.comm_channel import NoisyChannel, Sender, Receiver
from src.novelty.thresholds import EntropyHistory, posterior_entropy, gaussian_perturb
from src.slot_program.curriculum import build_buckets
from pathlib import Path


def load_images(folder: str, img_size: int, limit: int | None = None) -> torch.Tensor:
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    if limit:
        files = files[:limit]
    tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    imgs = []
    for f in files:
        img = Image.open(f).convert("RGB")
        imgs.append(tfm(img))
    if not imgs:
        raise RuntimeError(f"No images found in {folder}. Run synth_generate_clevr.py first.")
    return torch.stack(imgs, 0)


@hydra.main(config_path="../configs", config_name="slotprog", version_base=None)
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tiny = bool(cfg.get("tiny", False))
    steps = 300 if tiny else int(cfg.optim.steps)

    data_dir = cfg.data.clevr.out_dir
    img_size = int(cfg.data.clevr.img_size)
    def _reload_images(current_img_size: int):
        _imgs = load_images(data_dir, current_img_size, limit=512 if tiny else None).to(device)
        _B, _C, _H, _W = _imgs.shape
        _tokens = _imgs.view(_B, _C, _H * _W).permute(0, 2, 1)
        return _imgs, _tokens, _B, _C, _H, _W

    imgs, tokens, B, C, H, W = _reload_images(img_size)
    N, D = tokens.shape[1], tokens.shape[2]

    # Optional pre-generated difficulty buckets
    use_buckets = bool(getattr(cfg.curriculum, 'bucketed', {}).get('enable', False))
    if use_buckets:
        buckets = build_buckets(
            Path(data_dir),
            levels=int(cfg.curriculum.bucketed.levels),
            per_level_n=int(cfg.curriculum.bucketed.per_level_n),
            base_min_objs=int(cfg.curriculum.bucketed.base_min_objs),
            base_max_objs=int(cfg.curriculum.bucketed.base_max_objs),
            base_img_size=int(cfg.curriculum.bucketed.base_img_size),
            obj_step=int(cfg.curriculum.bucketed.obj_step),
            img_step=int(cfg.curriculum.bucketed.img_step),
            seed=0
        )
        current_level = 0
        # load level 0 explicitly
        lvl = buckets[current_level]
        imgs, tokens, B, C, H, W = _reload_images(lvl.img_size)
        N, D = tokens.shape[1], tokens.shape[2]

    num_slots = int(cfg.model.slots)
    slot_dim = int(cfg.model.slot_dim)
    slotter = SlotAttention(num_slots, dim=D, iters=int(cfg.model.iters)).to(device)

    program_head = ProgramHead(vocab=int(cfg.model.program_vocab), slot_dim=slot_dim, max_len=int(cfg.model.program_len_max)).to(device)
    interpreter = Interpreter(slot_dim=slot_dim, out_dim=D).to(device)

    # Simple linear bottlenecks outside loop
    # Map token channel D -> slot_dim and back
    proj = nn.Linear(D, slot_dim).to(device)
    inv_proj = nn.Linear(slot_dim, D).to(device)

    # Sender/Receiver for communication of program tokens
    comm_en = bool(cfg.comm.enable)
    if comm_en:
        channel = NoisyChannel(code_len=int(cfg.comm.code_len), snr_db=float(cfg.comm.channel_snr_db)).to(device)
        sender = Sender(vocab=int(cfg.model.program_vocab), code_len=int(cfg.comm.code_len)).to(device)
        receiver = Receiver(vocab=int(cfg.model.program_vocab), code_len=int(cfg.comm.code_len)).to(device)

    params = list(slotter.parameters()) + list(program_head.parameters()) + list(interpreter.parameters()) + list(proj.parameters()) + list(inv_proj.parameters())
    if comm_en:
        params += list(channel.parameters()) + list(sender.parameters()) + list(receiver.parameters())
    opt = optim.Adam(params, lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.wd))

    recon = ReconLoss()
    w = LossWeights(**{k: float(v) for k, v in cfg.loss.items()})

    logdir = cfg.workdir
    os.makedirs(logdir, exist_ok=True)
    tb = TBLogger(logdir)

    # Novelty tracking
    entropy_hist = EntropyHistory(values=[])
    apply_perturb_next = False
    snr_step_db = float(cfg.comm.snr_step_db) if comm_en else 0.0
    snr_min_db = float(cfg.comm.snr_min_db) if comm_en else 0.0
    snr_max_db = float(cfg.comm.snr_max_db) if comm_en else 40.0

    for step in range(1, steps + 1):
        idx = torch.randint(0, B, (min(16, B),), device=device)
        x = tokens[idx]

        # Apply novelty perturbation to token inputs if scheduled from previous step
        if bool(cfg.novelty.enable) and apply_perturb_next:
            x = gaussian_perturb(x, float(cfg.novelty.perturb_scale))
            apply_perturb_next = False

        slots, attn = slotter(x)
        # Project tokens to slot_dim via a linear bottleneck for reconstruction path
        x_proj = proj(x)

        prog_logits = program_head(B=x.shape[0])
        if comm_en:
            # Discretize greedily (index) → code → channel → decode logits (teacher-forcing omitted)
            toks = prog_logits.argmax(dim=-1)  # [B,L]
            code = sender(toks)
            noisy = channel(code)
            dec_logits = receiver(noisy)  # [B,V]
            # Encourage consistency: next token distribution aligns with decoded distribution (proxy)
            comm_loss = nn.CrossEntropyLoss()(dec_logits, toks[:, 0])
        else:
            comm_loss = torch.tensor(0.0, device=device)

        # Novelty entropy from program logits
        ent = posterior_entropy(prog_logits).detach().cpu().item()
        entropy_hist.values.append(ent)
        qval = entropy_hist.quantile(float(cfg.novelty.posterior_entropy_q))
        trigger = bool(cfg.novelty.enable) and (ent > qval) and (step > max(5, int(0.05 * steps)))
        if trigger:
            apply_perturb_next = True
            if comm_en:
                # Make channel harder by lowering SNR
                channel.snr_db = max(snr_min_db, min(snr_max_db, float(channel.snr_db) - snr_step_db))
            # Curriculum: increase scene difficulty
            if bool(cfg.curriculum.enable):
                if use_buckets:
                    # Move to next pre-generated level if available
                    if current_level + 1 < len(buckets):
                        current_level += 1
                        lvl = buckets[current_level]
                        imgs, tokens, B, C, H, W = _reload_images(lvl.img_size)
                        N, D = tokens.shape[1], tokens.shape[2]
                else:
                    new_min = min(int(cfg.data.clevr.min_objs) + int(cfg.curriculum.inc_min_objs), int(cfg.curriculum.max_objs_cap))
                    new_max = min(int(cfg.data.clevr.max_objs) + int(cfg.curriculum.inc_max_objs), int(cfg.curriculum.max_objs_cap))
                    new_img = min(int(cfg.data.clevr.img_size) + int(cfg.curriculum.inc_img_size), int(cfg.curriculum.img_size_cap))
                    # Only regenerate if something changes
                    if (new_min != int(cfg.data.clevr.min_objs)) or (new_max != int(cfg.data.clevr.max_objs)) or (new_img != int(cfg.data.clevr.img_size)):
                        from src.data.clevr_synth import generate_dataset
                        generate_dataset(Path(data_dir), int(cfg.curriculum.regen_n), new_img, new_min, new_max, seed=step)
                        # Update cfg in-memory for continued runs
                        cfg.data.clevr.min_objs = new_min
                        cfg.data.clevr.max_objs = new_max
                        cfg.data.clevr.img_size = new_img
                        # Reload tokens with possibly new spatial size
                        imgs, tokens, B, C, H, W = _reload_images(new_img)
                        N, D = tokens.shape[1], tokens.shape[2]
        tb.log_scalars({
            "entropy": ent,
            "qval": qval,
            "snr_db": float(channel.snr_db) if comm_en else -1.0,
            "perturb_next": float(trigger)
        }, step, prefix="novelty")

        x_hat_tokens = interpreter(slots, prog_logits, tokens_n=x.shape[1])  # [B,N,D]
        # Map back to pixel domain via inverse projection
        x_hat_tokens = inv_proj(x_hat_tokens)
        x_hat = x_hat_tokens.permute(0, 2, 1).view(-1, C, H, W)
        x_true = x.permute(0, 2, 1).view(-1, C, H, W)

        parts = {"recon": recon(x_hat, x_true) + 0.01 * comm_loss}
        total, scalars = aggregate_losses(parts, w)

        opt.zero_grad()
        total.backward()
        opt.step()

        if step % int(cfg.logging.every) == 0 or step == 1:
            tb.log_scalars(scalars, step, prefix="train")
            print({k: round(v, 6) for k, v in scalars.items()})

        if step % int(cfg.logging.ckpt_every) == 0:
            torch.save({
                "slotter": slotter.state_dict(),
                "program_head": program_head.state_dict(),
                "interpreter": interpreter.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            }, os.path.join(logdir, f"ckpt_{step:06d}.pt"))

    tb.close()
    print("Training done")


if __name__ == "__main__":
    main()
}