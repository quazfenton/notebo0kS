import torch

def test_eig_proxy_sanity():
    z = torch.randn(8, 10, 16)
    var = z.var(dim=0).mean()
    assert torch.isfinite(var)
