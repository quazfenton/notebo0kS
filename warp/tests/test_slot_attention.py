import torch
from src.slot_program.slot_attention import SlotAttention

def test_slot_attention_runs():
    B, N, D = 2, 64, 32
    sa = SlotAttention(num_slots=4, dim=D, iters=2)
    x = torch.randn(B, N, D)
    slots, attn = sa(x)
    assert slots.shape == (B, 4, D)
    assert attn.shape == (B, 4, N)
