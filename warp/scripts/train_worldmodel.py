#!/usr/bin/env python3
"""Train world-model with active inference and physics priors.
Hydra config: configs/worldmodel.yaml
Run:
  python scripts/train_worldmodel.py steps=1000 tiny=true
"""
from __future__ import annotations

import os
from dataclasses import asdict

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim

from src.common.losses import ReconLoss, PhysicsConsistency, LossWeights, aggregate_losses
from src.common.visualization import TBLogger
from src.world_model.encdec import MLPEnc, MLPDec
from src.world_model.dynamics import WorldModel
from src.data.ode_systems import sample_lorenz_batch
from src.novelty.thresholds import EntropyHistory, gaussian_perturb
from src.world_model.eig import eig_ensemble_disagreement, eig_dropout_disagreement


def eig_proxy(z: torch.Tensor) -> torch.Tensor:
    # Simple proxy: batch variance encourages spread/coverage (placeholder for EIG)
    return z.var(dim=0).mean()


@hydra.main(config_path="../configs", config_name="worldmodel", version_base=None)
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tiny = bool(cfg.get("tiny", False))
    steps = 300 if tiny else int(cfg.optim.steps)

    x_dim = int(cfg.model.x_dim)
    z_dim = int(cfg.model.z_dim)

    enc = MLPEnc(x_dim, z_dim, hidden=int(cfg.model.encoder_hidden)).to(device)
    dec = MLPDec(z_dim, x_dim, hidden=int(cfg.model.decoder_hidden)).to(device)
    dyn = WorldModel(z_dim, hidden=int(cfg.model.hamiltonian_hidden), symplectic=bool(cfg.model.symplectic), dropout_p=float(cfg.model.dropout_p)).to(device)

    params = list(enc.parameters()) + list(dec.parameters()) + list(dyn.parameters())
    opt = optim.Adam(params, lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.wd))

    recon_loss = ReconLoss()
    phys_loss = PhysicsConsistency()
    w = LossWeights(**{k: float(v) for k, v in cfg.loss.items()})

    logdir = cfg.workdir
    os.makedirs(logdir, exist_ok=True)
    tb = TBLogger(logdir)

    batch = 8 if tiny else int(cfg.env.batch_size)
    horizon = 40 if tiny else int(cfg.env.horizon)
    dt = float(cfg.env.dt)

    # Novelty tracking
    entropy_hist = EntropyHistory(values=[])
    apply_perturb_next = False

    for step in range(1, steps + 1):
        t, xs = sample_lorenz_batch(batch, horizon, dt, device)
        # Apply novelty perturbation to inputs if scheduled from previous step
        if bool(cfg.novelty.enable) and apply_perturb_next:
            xs = gaussian_perturb(xs, float(cfg.novelty.perturb_scale))
            apply_perturb_next = False  # reset after applying
        # Encode initial state
        z0 = enc(xs[:, 0, :])

        zs = [z0]
        residuals = []
        for ti in range(1, horizon):
            z_next, res = dyn.step(zs[-1], dt)
            zs.append(z_next)
            residuals.append(res)
        z_seq = torch.stack(zs, dim=1)  # [B, T, Z]
        x_hat = dec(z_seq.reshape(-1, z_dim)).reshape(batch, horizon, x_dim)

        # Novelty entropy proxy: variance of latent trajectory
        entropy_val = z_seq.var(dim=(0, 1)).mean().detach().cpu().item()
        entropy_hist.values.append(entropy_val)
        qval = entropy_hist.quantile(float(cfg.novelty.entropy_quantile))
        trigger = bool(cfg.novelty.enable) and (entropy_val > qval) and (step > max(5, int(0.05 * steps)))
        if trigger:
            apply_perturb_next = True
        tb.log_scalars({"entropy": entropy_val, "qval": qval, "perturb_next": float(trigger)}, step, prefix="novelty")

        parts = {}
        parts["recon"] = recon_loss(x_hat, xs)
        if residuals:
            parts["phys"] = phys_loss(torch.stack(residuals, dim=1))
        # EIG term (negative in aggregate losses since higher is better)
        if bool(cfg.eig.use_dropout):
            parts["eig"] = -eig_dropout_disagreement(dyn, z_seq, dt, T=int(cfg.eig.mc_samples), dropout_p=float(cfg.eig.dropout_p))
        elif bool(cfg.eig.use_ensemble):
            parts["eig"] = -eig_ensemble_disagreement(dyn, z_seq, dt, K=int(cfg.eig.ensemble_size), weight_noise_scale=float(cfg.eig.weight_noise_scale))
        else:
            parts["eig"] = -eig_proxy(z_seq)

        total, scalars = aggregate_losses(parts, w)
        opt.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(params, float(cfg.optim.grad_clip))
        opt.step()

        if step % int(cfg.logging.every) == 0 or step == 1:
            tb.log_scalars(scalars, step, prefix="train")
            print({k: round(v, 6) for k, v in scalars.items()})

        if step % int(cfg.logging.ckpt_every) == 0:
            torch.save({
                "enc": enc.state_dict(),
                "dec": dec.state_dict(),
                "dyn": dyn.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            }, os.path.join(logdir, f"ckpt_{step:06d}.pt"))

    tb.close()
    print("Training done")


if __name__ == "__main__":
    main()
