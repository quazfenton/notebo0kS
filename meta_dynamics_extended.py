a# Re-writing the enhanced demo script to /mnt/data/meta_dynamics_extended.py
script = r'''# Meta-Learning-by-Predicting-Learning-Dynamics (Extended Demo)
# - Adds: unrolled-backprop meta-gradient demo (truncated unroll)
# - Adds: finite-difference approximate meta-gradient as a cheap alternative
# - Adds: synthetic task generator with curriculum & domain shift options (harder tasks over time)
# - Still compact and runnable for experimentation
#
# Usage: python meta_dynamics_extended.py
# Dependencies: torch

import os, copy, math, random, time
from typing import List, Dict, Any
import torch, torch.nn as nn, torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------
# Synthetic curriculum task generator
# ---------------------------
def make_curriculum_task(difficulty=0.0, seed=None, num_features=16, base_hidden=32):
    # difficulty in [0,1]. 0 = trivial linear separable; 1 = hard (label noise, covariate shift, class overlap)
    if seed is not None:
        torch.manual_seed(seed); random.seed(seed)
    num_classes = 2 + int(difficulty * 4)  # between 2 and 6
    n_train = 256
    n_val = 256
    X_base = torch.randn(n_train, num_features)
    X_val_base = torch.randn(n_val, num_features)
    weight = torch.randn(num_features, num_classes)
    bias = torch.randn(num_classes) * (1.0 - 0.7 * difficulty)
    hidden = base_hidden + int(difficulty * 64)
    label_noise = difficulty * 0.4  # up to 40% noisy labels
    cov_shift = difficulty * 0.8    # how different val inputs are
    imbalance = 0.0 if difficulty < 0.3 else (0.1 + difficulty * 0.3)
    def sample_X(base, shift_factor):
        X = base + shift_factor * torch.randn_like(base) * 0.5
        if difficulty > 0.3:
            X = torch.cat([X, (X**2)*0.1], dim=1)[:, :num_features]
        return X
    X_train = sample_X(X_base, shift_factor=0.0)
    X_val = sample_X(X_val_base, shift_factor=cov_shift)
    logits_train = X_train @ weight + bias
    logits_val = X_val @ weight + bias
    temp = 1.0 + difficulty * 2.0
    probs_train = F.softmax(logits_train / temp, dim=1)
    probs_val = F.softmax(logits_val / temp, dim=1)
    y_train = torch.multinomial(probs_train, 1).squeeze(1)
    y_val = torch.multinomial(probs_val, 1).squeeze(1)
    if label_noise > 0:
        noisy_idx = torch.randperm(n_train)[:int(label_noise * n_train)]
        y_train[noisy_idx] = torch.randint(0, num_classes, (len(noisy_idx),))
    if imbalance > 0:
        for cls in range(num_classes):
            if random.random() < imbalance:
                idxs = (y_train == cls).nonzero(as_tuple=True)[0]
                if len(idxs) > 0:
                    sel = idxs[torch.randperm(len(idxs))[:len(idxs)//2]]
                    y_train[sel] = torch.randint(0, num_classes, (len(sel),))
    model = nn.Sequential(
        nn.Linear(num_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, num_classes)
    )
    return dict(model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, difficulty=difficulty)

# ---------------------------
# Trajectory logger (supports domain-shift mid-run)
# ---------------------------
def train_and_log_trajectory_with_shift(task, inner_steps=8, inner_lr=0.05, shift_at=None):
    model = copy.deepcopy(task["model"]).to(device)
    X = task["X_train"].to(device); y = task["y_train"].to(device)
    X_val = task["X_val"].to(device); y_val = task["y_val"].to(device)
    opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
    criterion = nn.CrossEntropyLoss()
    snapshots = []
    def snapshot():
        layer_summaries = []
        for p in model.parameters():
            g = p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(p.detach().cpu())
            layer_summaries.append({
                "mean": float(p.detach().cpu().mean()),
                "std": float(p.detach().cpu().std()),
                "grad_mean": float(g.mean()) if g.numel()>0 else 0.0,
                "numel": p.numel()
            })
        with torch.no_grad():
            loss = float(criterion(model(X), y).detach().cpu())
        snapshots.append({"params":[p.detach().cpu().clone() for p in model.parameters()],
                          "layer_summary": layer_summaries, "loss": loss})
    opt.zero_grad(); out = model(X); loss = criterion(out, y); loss.backward(); snapshot()
    for t in range(inner_steps):
        opt.step()
        if shift_at is not None and t == shift_at:
            with torch.no_grad():
                X.add_(0.5 * torch.randn_like(X))
        opt.zero_grad()
        out = model(X); loss = criterion(out, y); loss.backward(); snapshot()
    with torch.no_grad():
        val_loss = float(F.cross_entropy(model(X_val), y_val).detach().cpu())
    return snapshots, val_loss

# ---------------------------
# Meta-model (layer-summary encoder + GRU + decoder)
# ---------------------------
class LayerEncoder(nn.Module):
    def __init__(self, in_dim=4, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    def forward(self, x): return self.net(x)

class MetaDynamicsModel(nn.Module):
    def __init__(self, n_layers, in_dim=4, emb_dim=64, rnn_hidden=128):
        super().__init__()
        self.n_layers = n_layers
        self.encoder = LayerEncoder(in_dim=in_dim, emb_dim=emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=rnn_hidden, batch_first=True)
        self.decoder_head = nn.Sequential(
            nn.Linear(rnn_hidden, rnn_hidden//2),
            nn.ReLU(),
            nn.Linear(rnn_hidden//2, 2)
        )
    def forward(self, layer_summary_seq):
        B, T, L, D = layer_summary_seq.shape
        x = layer_summary_seq.view(B*T*L, D).to(device)
        emb = self.encoder(x)
        emb = emb.view(B, T*L, -1)
        rnn_out, _ = self.rnn(emb)
        decoded = self.decoder_head(rnn_out)
        decoded = decoded.view(B, T, L, -1)
        preds = decoded[:, -1, :, :]
        return preds[...,0], preds[...,1]

# ---------------------------
# Helpers: summarize snapshots, extract true deltas, batch builder
# ---------------------------
def summarize_snapshot(snapshot):
    vecs = []
    for layer in snapshot["layer_summary"]:
        vecs.append([layer["mean"], layer["std"], layer["grad_mean"], float(layer["numel"])**0.5])
    return torch.tensor(vecs, dtype=torch.float32)

def extract_layerwise_true_delta(snap_t, snap_t1):
    deltas = []
    for p_t, p_t1 in zip(snap_t["params"], snap_t1["params"]):
        d = (p_t1 - p_t).float().mean().item()
        deltas.append(d)
    return torch.tensor(deltas, dtype=torch.float32)

def build_meta_batch(trajectories, batch_size=4, max_len=8):
    batch = random.sample(trajectories, batch_size)
    batch_inputs, batch_targets, batch_tasks = [], [], []
    for item in batch:
        T = len(item["snapshots"])
        if T < 2: continue
        t = random.randint(0, T-2)
        seq = []
        for k in range(0, t+1):
            seq.append(summarize_snapshot(item["snapshots"][k]))
        L = seq[0].shape[0]; D = seq[0].shape[1]
        seq_tensors = torch.zeros(max_len, L, D)
        for idx, s in enumerate(seq): seq_tensors[idx] = s
        true_delta = extract_layerwise_true_delta(item["snapshots"][t], item["snapshots"][t+1])
        batch_inputs.append(seq_tensors); batch_targets.append(true_delta); batch_tasks.append(item["task_meta"])
    batch_inputs = torch.stack(batch_inputs, dim=0); batch_targets = torch.stack(batch_targets, dim=0)
    return batch_inputs, batch_targets, batch_tasks

# ---------------------------
# Meta-training primitives: unrolled-backprop and finite-difference meta-gradient
# ---------------------------
def meta_update_unrolled(meta, meta_opt, batch_inputs, batch_targets, batch_tasks, inner_step_scale=0.1):
    meta.train()
    multipliers, pred_loss = meta(batch_inputs.to(device))
    grad_mean_vec = batch_inputs[:, -1, :, 2].to(device)
    predicted_delta = multipliers * grad_mean_vec
    L_dyn = F.mse_loss(predicted_delta, batch_targets.to(device))
    L_down = 0.0
    for b, task in enumerate(batch_tasks):
        base = copy.deepcopy(task["model"]).to(device)
        base.train()
        multipliers_b = multipliers[b]
        for p_idx, p in enumerate(base.parameters()):
            gm = float(grad_mean_vec[b, p_idx].item())
            delta_tensor = torch.sign(p) * (0.05 * multipliers_b[p_idx] * gm)
            p.data = p.data - delta_tensor.to(p.device)
        base.eval()
        Xv = task['X_val'].to(device); yv = task['y_val'].to(device)
        val_logits = base(Xv)
        L_down = L_down + F.cross_entropy(val_logits, yv)
    L_down = L_down / len(batch_tasks)
    total = L_dyn + 0.1 * L_down
    meta_opt.zero_grad(); total.backward(); meta_opt.step()
    return L_dyn.item(), L_down.item(), total.item()

def meta_update_finite_diff(meta, meta_opt, batch_inputs, batch_targets, batch_tasks, eps=1e-3):
    meta.eval()
    with torch.no_grad():
        multipliers, _ = meta(batch_inputs.to(device))
        grad_mean_vec = batch_inputs[:, -1, :, 2].to(device)
        predicted_delta = multipliers * grad_mean_vec
        L_dyn = F.mse_loss(predicted_delta, batch_targets.to(device)).item()
    # finite-diff on a couple keys
    meta.train()
    baseline_state = {k:v.clone() for k,v in meta.state_dict().items()}
    approx_grads = {k: torch.zeros_like(v) for k,v in meta.state_dict().items()}
    perturb_keys = [k for k in baseline_state.keys() if "encoder.net.0.weight" in k or "decoder_head.0.weight" in k]
    if len(perturb_keys) == 0:
        perturb_keys = [list(baseline_state.keys())[0]]
    for k in perturb_keys:
        orig = baseline_state[k]
        eps_tensor = torch.randn_like(orig) * eps
        splus = {kk:vv.clone() for kk,vv in baseline_state.items()}; splus[k] = (orig + eps_tensor)
        meta.load_state_dict(splus)
        with torch.no_grad():
            multipliers_p, _ = meta(batch_inputs.to(device))
            predicted_p = multipliers_p * batch_inputs[:, -1, :, 2].to(device)
            Lp = F.mse_loss(predicted_p, batch_targets.to(device)).item()
        sminus = {kk:vv.clone() for kk,vv in baseline_state.items()}; sminus[k] = (orig - eps_tensor)
        meta.load_state_dict(sminus)
        with torch.no_grad():
            multipliers_m, _ = meta(batch_inputs.to(device))
            predicted_m = multipliers_m * batch_inputs[:, -1, :, 2].to(device)
            Lm = F.mse_loss(predicted_m, batch_targets.to(device)).item()
        est_scalar = (Lp - Lm) / (2*eps)
        approx_grads[k] = est_scalar * eps_tensor
    meta.load_state_dict(baseline_state)
    lr = 1e-3
    new_state = {}
    for k,v in meta.state_dict().items():
        g = approx_grads.get(k, torch.zeros_like(v))
        new_state[k] = v - lr * g
    meta.load_state_dict(new_state)
    return L_dyn, None, None

# ---------------------------
# Build curriculum trajectories
# ---------------------------
def build_curriculum_trajectories(n_tasks=24):
    trajectories = []
    for i in range(n_tasks):
        difficulty = min(1.0, (i / max(1, n_tasks-1)))
        task = make_curriculum_task(difficulty=difficulty, seed=100+i)
        shift_at = None
        if difficulty > 0.6:
            shift_at = random.randint(2,5)
        snaps, val_loss = train_and_log_trajectory_with_shift(task, inner_steps=8, inner_lr=0.08, shift_at=shift_at)
        trajectories.append({"snapshots": snaps, "val_loss": val_loss, "task_meta": task})
    return trajectories

# ---------------------------
# Demo run
# ---------------------------
def run_demo():
    print("Building curriculum trajectories...")
    trajs = build_curriculum_trajectories(n_tasks=20)
    print("Built", len(trajs), "trajectories. Example snapshot count:", len(trajs[0]['snapshots']))
    n_layers = len(trajs[0]["snapshots"][0]["layer_summary"])
    meta = MetaDynamicsModel(n_layers=n_layers, in_dim=4, emb_dim=64, rnn_hidden=128).to(device)
    meta_opt = torch.optim.Adam(meta.parameters(), lr=1e-3)
    for step in range(40):
        batch_inputs, batch_targets, batch_tasks = build_meta_batch(trajs, batch_size=6, max_len=8)
        if step % 7 == 0:
            info = meta_update_finite_diff(meta, meta_opt, batch_inputs, batch_targets, batch_tasks)
            print(f"[step {step}] finite-diff approx dyn_loss={info[0]:.4e}")
        else:
            Ld, Ldwn, tot = meta_update_unrolled(meta, meta_opt, batch_inputs, batch_targets, batch_tasks)
            if step % 5 == 0:
                print(f"[step {step}] unrolled L_dyn={Ld:.4e}, L_down={Ldwn:.4e}, total={tot:.4e}")
    print("Demo complete. Meta-model param count:", sum(p.numel() for p in meta.parameters()))

if __name__ == "__main__":
    # Optional NBX pipeline (no-op for this demo unless --nbx.use is supplied)
    try:
        from nbx.pipeline import load_from_cli
        split, cfg = load_from_cli()
        if split is not None:
            print("[NBX] Data pipeline active.")
    except Exception as e:
        pass
    run_demo()
'''
out_path = "/mnt/data/meta_dynamics_extended.py"
with open(out_path, "w") as f:
    f.write(script)
out_path


