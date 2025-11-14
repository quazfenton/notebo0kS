# Creating a runnable "notebook-style" Python script that the user can open as a Jupyter notebook
# or run as a script. This writes a file to /mnt/data and prints its path.
notebook_text = r'''# %% [markdown]
# Meta-Learning by Predicting Learning Dynamics — Demo Notebook
#
# This "notebook-style" Python script provides a compact, runnable prototype of the
# Meta-Learning-by-Predicting-Learning-Dynamics idea you described.
#
# Goals:
# - Generate many small "base" training trajectories (synthetic / small MLP tasks)
# - Build a lightweight meta-model (layer-summary encoder + GRU temporal module + decoder)
#   that predicts parameter updates (Δθ) and short-horizon loss trajectories.
# - Train the meta-model with two objectives:
#     1) MSE between predicted Δθ and true Δθ (dynamics fidelity)
#     2) Downstream few-shot loss after applying predicted Δθ (task performance)
#
# Notes:
# - This is a **prototype**: simplified per-layer summaries (means/stds/grad-mean) are used
#   instead of a full high-dimensional GNN over every parameter for efficiency and clarity.
# - The code is written for clarity and small-scale experimentation. To scale to large
#   models / tasks, move to JAX/TPU, batched graph encoders, Hessian-mode primitives, and
#   distributed data collection of trajectories (W&B / blobstore).
#
# How to run:
# - Open with Jupyter or `python meta_dynamics_demo.py`
# - Requires PyTorch (CPU is fine for the demo): `pip install torch`
#
# %%
import os
import math
import copy
import random
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# %% [markdown]
# ## Synthetic trajectory generator (small MLP tasks)
#
# We create small random classification/regression tasks by sampling a tiny MLP and small dataset,
# then training that MLP for a few steps and logging snapshots (θ_t, grads, loss). The snapshots
# are intentionally shallow (few steps) so the meta-model learns short-horizon dynamics initially.

# %%
def make_synthetic_task(num_features=16, num_classes=3, n_train=128, n_val=128, hidden=64, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    X_train = torch.randn(n_train, num_features)
    y_train = torch.randint(0, num_classes, (n_train,))
    X_val = torch.randn(n_val, num_features)
    y_val = torch.randint(0, num_classes, (n_val,))
    model = nn.Sequential(
        nn.Linear(num_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, num_classes)
    )
    return dict(model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

def train_and_log_trajectory(task, inner_steps=6, inner_lr=0.05):
    model = copy.deepcopy(task["model"]).to(device)
    X = task["X_train"].to(device)
    y = task["y_train"].to(device)
    opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
    criterion = nn.CrossEntropyLoss()
    snapshots = []
    # initial snapshot (t=0)
    def snapshot():
        flat_params = []
        grads = []
        layer_summaries = []
        for p in model.parameters():
            flat_params.append(p.detach().cpu().flatten())
            g = p.grad.detach().cpu().flatten() if p.grad is not None else torch.zeros_like(p.detach().cpu().flatten())
            grads.append(g)
            layer_summaries.append({
                "mean": float(p.detach().cpu().mean()),
                "std": float(p.detach().cpu().std()),
                "grad_mean": float(g.mean()) if g.numel()>0 else 0.0,
                "numel": p.numel()
            })
        with torch.no_grad():
            logits = model(X)
            loss = float(criterion(logits, y).detach().cpu())
        snapshots.append({
            "params": [p.detach().cpu().clone() for p in model.parameters()],
            "grads": [ (p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p.detach().cpu().clone())) for p in model.parameters() ],
            "layer_summary": layer_summaries,
            "loss": loss
        })
    # zero grads first
    opt.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    snapshot()
    for t in range(inner_steps):
        opt.step()
        opt.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        snapshot()
    # also capture validation loss sequence
    model_cpu = model.to("cpu")
    with torch.no_grad():
        val_logits = model_cpu(task["X_val"])
        val_loss = float(F.cross_entropy(val_logits, task["y_val"]))
    return snapshots, val_loss

# %% [markdown]
# ## Utilities to convert snapshots into meta-inputs
#
# We'll represent each layer by a small vector summarizing its current param/grad statistics. The meta-model
# consumes a temporal sequence of these per-layer summaries and outputs a predicted layerwise delta multiplier
# (a scalar per layer) that gets multiplied by the layer's current gradient to produce a per-parameter update.
# (This is a design choice: the meta-output predicts a *direction-scaled* update, which saves memory.)

# %%
def summarize_snapshot(snapshot):
    # snapshot["layer_summary"] is already a list of dicts from our synthetic logger
    vecs = []
    for layer in snapshot["layer_summary"]:
        vecs.append([layer["mean"], layer["std"], layer["grad_mean"], float(layer["numel"])**0.5])
    return torch.tensor(vecs, dtype=torch.float32)  # shape: (n_layers, summary_dim)

# %% [markdown]
# ## Meta-model (Layer-level encoder + GRU + decoder)
#
# - Per-layer MLP encoder maps layer summary -> embedding
# - A GRU (shared across layers) integrates time
# - Decoder outputs a scalar multiplier for that layer's gradient and a predicted next-step loss

# %%
class LayerEncoder(nn.Module):
    def __init__(self, in_dim=4, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    def forward(self, x):
        # x: (..., in_dim)
        return self.net(x)

class MetaDynamicsModel(nn.Module):
    def __init__(self, n_layers, in_dim=4, emb_dim=64, rnn_hidden=128):
        super().__init__()
        self.n_layers = n_layers
        self.encoder = LayerEncoder(in_dim=in_dim, emb_dim=emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=rnn_hidden, batch_first=True)
        # we'll read out per-layer scalar multipliers from the RNN outputs via a small head
        self.decoder_head = nn.Sequential(
            nn.Linear(rnn_hidden, rnn_hidden//2),
            nn.ReLU(),
            nn.Linear(rnn_hidden//2, 2)  # outputs: [delta_multiplier, predicted_next_loss_scalar]
        )
    def forward(self, layer_summary_seq):
        # layer_summary_seq: shape (batch, time, n_layers, in_dim)
        B, T, L, D = layer_summary_seq.shape
        x = layer_summary_seq.view(B*T*L, D).to(device)  # encode every (time,layer)
        emb = self.encoder(x)  # (B*T*L, emb_dim)
        emb = emb.view(B, T*L, -1)  # treat sequence length = T*L for GRU
        rnn_out, _ = self.rnn(emb)  # (B, T*L, rnn_hidden)
        decoded = self.decoder_head(rnn_out)  # (B, T*L, 2)
        decoded = decoded.view(B, T, L, -1)  # (B, T, L, 2)
        # For each batch, take last time-step predictions (predict next delt for each layer)
        preds = decoded[:, -1, :, :]  # (B, L, 2)
        return preds[...,0], preds[...,1]  # multipliers, predicted_loss

# %% [markdown]
# ## Create a tiny dataset of synthetic trajectories
#
# We'll create N tasks, each with a short inner trajectory. For a real project you would
# gather thousands of longer trajectories from many architectures and store them (W&B, S3).

# %%
def build_trajectory_dataset(n_tasks=32, inner_steps=6):
    tasks = []
    trajectories = []
    for i in range(n_tasks):
        tsk = make_synthetic_task(num_features=12, num_classes=4, n_train=256, n_val=128, hidden=32, seed=i*13+7)
        snaps, val_loss = train_and_log_trajectory(tsk, inner_steps=inner_steps, inner_lr=0.08)
        # store per-task: list of snapshots (each snapshot has layer_summary), final val_loss
        trajectories.append({"snapshots": snaps, "val_loss": val_loss, "task_meta": tsk})
    return trajectories

print("Building small demo dataset (this will be quick)...")
demo_trajectories = build_trajectory_dataset(n_tasks=24, inner_steps=6)
print("Built", len(demo_trajectories), "trajectories. Example snapshot count:", len(demo_trajectories[0]["snapshots"]))

# %% [markdown]
# ## Training the meta-model (small demo loop)
#
# Losses:
#  - L_dyn = MSE between predicted per-layer multiplier * current_grad and true Δθ_mean
#  - L_downstream = validation loss after applying predicted Δθ to a fresh copy of the base model (few-shot)
#
# For stability in the small demo we treat layerwise Δθ target as the mean-per-parameter update (scalar).
# In production you would predict full-shape updates or low-rank modal updates (spectral lives).

# %%
def extract_layerwise_true_delta(snap_t, snap_t1):
    # compute mean-per-parameter delta per layer: mean(theta_{t+1} - theta_t)
    deltas = []
    for p_t, p_t1 in zip(snap_t["params"], snap_t1["params"]):
        d = (p_t1 - p_t).float().mean().item()
        deltas.append(d)
    return torch.tensor(deltas, dtype=torch.float32)  # shape: (L,)

def build_meta_batch(trajectories, batch_size=4):
    # sample trajectories and for each choose a time index t < T-1 to predict t->t+1
    batch = random.sample(trajectories, batch_size)
    batch_inputs = []
    batch_targets = []
    batch_val_losses = []
    batch_tasks = []
    for item in batch:
        T = len(item["snapshots"])
        if T < 2:
            continue
        t = random.randint(0, T-2)
        snap_t = item["snapshots"][t]
        snap_t1 = item["snapshots"][t+1]
        # build temporal sequence of per-layer summaries up to time t inclusive
        seq = []
        for k in range(0, t+1):
            seq.append(summarize_snapshot(item["snapshots"][k]))  # (L, D)
        # pad sequence to fixed length (we'll use max_len = 6 for demo)
        max_len = 6
        L = seq[0].shape[0]
        D = seq[0].shape[1]
        seq_tensors = torch.zeros(max_len, L, D)
        for idx, s in enumerate(seq):
            seq_tensors[idx] = s
        # target: true mean per-layer delta from t -> t+1, and next-step validation loss (not used heavily here)
        true_delta = extract_layerwise_true_delta(snap_t, snap_t1)
        batch_inputs.append(seq_tensors)
        batch_targets.append(true_delta)
        batch_val_losses.append(item["val_loss"])
        batch_tasks.append(item["task_meta"])
    # stack to shape (B, max_len, L, D)
    batch_inputs = torch.stack(batch_inputs, dim=0)
    batch_targets = torch.stack(batch_targets, dim=0)
    return batch_inputs, batch_targets, batch_tasks, batch_val_losses

# Instantiate meta-model.
n_layers = len(demo_trajectories[0]["snapshots"][0]["layer_summary"])
meta = MetaDynamicsModel(n_layers=n_layers, in_dim=4, emb_dim=64, rnn_hidden=128).to(device)
meta_opt = torch.optim.Adam(meta.parameters(), lr=1e-3)

# Demo training loop: a handful of meta-steps to show functionality
print("Starting a short meta-training demo (few steps)...")
for step in range(40):
    meta.train()
    batch_inputs, batch_targets, batch_tasks, batch_val_losses = build_meta_batch(demo_trajectories, batch_size=6)
    batch_inputs = batch_inputs.to(device)  # (B, T, L, D)
    batch_targets = batch_targets.to(device)  # (B, L)
    multipliers, pred_loss = meta(batch_inputs)  # multipliers: (B, L)
    # predicted Δθ (per-layer) := multiplier * grad_mean (approx)
    # we don't have exact grads in the batch_inputs (only grad_mean), but this demo uses grad_mean proxy
    grad_mean_vec = batch_inputs[:, -1, :, 2]  # last timestep grad_mean (B, L)
    predicted_delta = multipliers * grad_mean_vec  # (B, L)
    # dynamics loss (MSE)
    L_dyn = F.mse_loss(predicted_delta, batch_targets.to(device))
    # downstream few-shot loss: for each example apply predicted per-layer scalar update to a fresh task model
    L_down = 0.0
    for b, task in enumerate(batch_tasks):
        # load fresh base model
        base = copy.deepcopy(task["model"]).to(device)
        # apply predicted scalar * gradient_mean as a very rough "simulated" update
        # note: in production you'd apply a full-tensor update; this is a simplified demo
        with torch.no_grad():
            multipliers_b = multipliers[b].cpu().detach()
            # iterate layers and modify parameters in-place: p <- p + multiplier * sign_of_grad_mean * small_scale
            for p, m in zip(base.parameters(), multipliers_b):
                # small synthetic update magnitude
                p.add_( -0.1 * float(m.item()) * torch.sign(p) )  # crude; only for demo
        # compute validation loss after applying update
        base.eval()
        with torch.no_grad():
            val_logits = base(task["X_val"].to(device))
            val_loss = F.cross_entropy(val_logits, task["y_val"].to(device))
            L_down = L_down + val_loss
    L_down = L_down / len(batch_tasks)
    total_loss = L_dyn + 0.1 * L_down
    meta_opt.zero_grad()
    total_loss.backward()
    meta_opt.step()
    if step % 10 == 0:
        print(f"[meta step {step}] L_dyn={L_dyn.item():.4e}, L_down={L_down.item():.4e}, total={total_loss.item():.4e}")

print("Demo meta-training complete. This is a toy prototype — see notes in the header for scaling suggestions.")

# %% [markdown]
# ## Next steps to make this production-ready / research-grade
#
# 1. **Real logged trajectories**: collect many trajectories from multiple architectures (MLP/CNN/Transformer), store snapshots (params, grads, optimizer state, loss, val metrics) to W&B/MinIO.
# 2. **Graph-level model**: replace layer-summary design with a Graph Neural Network built from actual parameter graph (treat parameter tensors as nodes; edges for layer connectivity). Use sparse message passing to scale.
# 3. **Predict full-shape low-rank updates**: predict low-rank factors (U, V) or spectral-mode coefficients rather than a single scalar per layer.
# 4. **Unrolling and implicit differentiation**: for meta-gradient through many inner steps, use implicit differentiation for efficiency or truncated unrolls + and stability methods (Neumann series, checkpointing).
# 5. **Uncertainty**: predict distributions over Δθ (e.g., Gaussian with predicted variance) and train with NLL; use ensemble & MC sampling for robust deployment.
# 6. **Self-synth loop**: have the meta-model generate synthetic task modifications to focus its learning on high-entropy dynamics (active curriculum).
# 7. **Scaling**: move to JAX + pmap/TPU, use mixed precision, larger batch sizes of trajectories, and more sophisticated primitives (Hessian-vector products, K-FAC signals).
#
# %%
# Save this file for download.
out_path = "/mnt/data/meta_dynamics_demo.py"
with open(out_path, "w") as f:
    f.write(notebook_text)
print("Wrote demo script to:", out_path)
'''

# --- NBX pipeline (optional) ---
if __name__ == "__main__":
    try:
        from nbx.pipeline import load_from_cli
        split, cfg = load_from_cli()
        if split is not None:
            print("[NBX] Loaded dataset for metaLearn:", split.meta)
    except Exception:
        pass

