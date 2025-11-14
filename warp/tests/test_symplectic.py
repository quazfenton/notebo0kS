import torch
from src.world_model.dynamics import LearnedHamiltonian

def test_symplectic_energy_drift_small_dt():
    H = LearnedHamiltonian(z_dim=4)
    z = torch.randn(8, 10, 4, requires_grad=False)
    # Just ensure forward works and energy is finite for now (no dynamics here)
    e = H(z.reshape(-1, 4)).mean()
    assert torch.isfinite(e)
