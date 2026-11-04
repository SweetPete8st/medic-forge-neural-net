"""Export a compiled ReLUNet to torch and run differentiable rollouts of the brain."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments/exp3_brain"))


class TorchReLUNet(torch.nn.Module):
    def __init__(self, net, dtype=torch.float64):
        super().__init__()
        self.Ws = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(W, dtype=dtype), requires_grad=False) for W, _ in net.layers])
        self.bs = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(b, dtype=dtype), requires_grad=False) for _, b in net.layers])

    def forward(self, x):
        h = x
        n = len(self.Ws)
        for i in range(n):
            h = self.Ws[i] @ h + self.bs[i]
            if i < n - 1: h = torch.relu(h)
        return h


def export(brain):
    return TorchReLUNet(brain.net)
