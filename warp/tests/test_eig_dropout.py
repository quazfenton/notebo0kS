import torch
from src.world_model.dynamics import WorldModel
from src.world_model.eig import eig_dropout_disagreement

def test_eig_dropout_positive_variance():
    z_dim = 4
    dyn = WorldModel(z_dim=z_dim, hidden=16, symplectic=False, dropout_p=0.2)
    z_seq = torch.randn(8, 5, z_dim)
    dt = 0.05
    v = eig_dropout_disagreement(dyn, z_seq, dt, T=6, dropout_p=0.2)
    assert torch.isfinite(v)
    assert v >= 0