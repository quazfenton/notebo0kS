import torch
from src.slot_program.grammar import Interpreter

def test_interpreter_shapes():
    B, S, D, N = 4, 5, 32, 16
    interp = Interpreter(slot_dim=D, out_dim=D)
    slots = torch.randn(B, S, D)
    prog_logits = torch.randn(B, 8, 64)
    out = interp(slots, prog_logits, tokens_n=N)
    assert out.shape == (B, N, D)
