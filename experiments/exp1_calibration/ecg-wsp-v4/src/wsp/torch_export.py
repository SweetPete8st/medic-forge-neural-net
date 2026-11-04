"""Build the same network in PyTorch from the hand-set weights, so the artifact
is a standard nn.Module with a saveable state_dict. The wiring is copied from
the numpy Net op list; the numbers are copied from its layers. Nothing is
trained here either."""
import numpy as np
import torch
import torch.nn as nn

from .layers import (Conv1d, ReLU, MaxPool1d, CausalMaxPool1d, Concat, GlobalSum,
                     GlobalMax, ChannelMax, Ramp)


class SumPool(nn.Module):
    """Sum over time, keepdim. Parameter-free reduction."""
    def forward(self, x):
        return x.sum(dim=-1, keepdim=True)


class ChanMax(nn.Module):
    """Max over channels, keepdim (maxout). Parameter-free."""
    def forward(self, x):
        return x.max(dim=1, keepdim=True).values


class RampMod(nn.Module):
    """Fixed positional ramp t/fs: a constant, like a positional encoding."""
    def __init__(self, fs):
        super().__init__()
        self.fs = fs

    def forward(self, x):
        r = torch.arange(x.shape[-1], dtype=x.dtype, device=x.device) / self.fs
        return r[None, None, :].expand(x.shape[0], 1, -1)


class TorchNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.inputs = net.inputs
        self.output = net.output
        self.outputs = getattr(net, "outputs", [net.output])
        self.wiring = []          # (module key or None, op kind, ins, out)
        self.mods = nn.ModuleDict()
        for idx, (layer, ins, out) in enumerate(net.ops):
            key = f"op{idx}"
            if isinstance(layer, Conv1d):
                o, i, k = layer.w.shape
                m = nn.Conv1d(i, o, k, padding=k // 2)
                with torch.no_grad():
                    m.weight.copy_(torch.tensor(layer.w, dtype=torch.float32))
                    m.bias.copy_(torch.tensor(layer.b, dtype=torch.float32))
                self.mods[key] = m
                self.wiring.append((key, "conv", ins, out))
            elif isinstance(layer, ReLU):
                self.mods[key] = nn.ReLU()
                self.wiring.append((key, "relu", ins, out))
            elif isinstance(layer, MaxPool1d):
                self.mods[key] = nn.MaxPool1d(layer.k, stride=1, padding=layer.k // 2)
                self.wiring.append((key, "maxpool", ins, out))
            elif isinstance(layer, CausalMaxPool1d):
                self.mods[key] = nn.Sequential(
                    nn.ConstantPad1d((layer.k, -1), layer.pad_value),
                    nn.MaxPool1d(layer.k, stride=1))
                self.wiring.append((key, "causalmax", ins, out))
            elif isinstance(layer, Concat):
                self.wiring.append((None, "concat", ins, out))
            elif isinstance(layer, GlobalSum):
                self.mods[key] = SumPool()
                self.wiring.append((key, "gsum", ins, out))
            elif isinstance(layer, GlobalMax):
                self.mods[key] = nn.AdaptiveMaxPool1d(1)
                self.wiring.append((key, "gmax", ins, out))
            elif isinstance(layer, ChannelMax):
                self.mods[key] = ChanMax()
                self.wiring.append((key, "cmax", ins, out))
            elif isinstance(layer, Ramp):
                self.mods[key] = RampMod(layer.fs)
                self.wiring.append((key, "ramp", ins, out))
            else:
                raise TypeError(type(layer))

    def forward(self, x, return_all=False):
        t = {self.inputs[0]: x}
        for key, kind, ins, out in self.wiring:
            if kind == "concat":
                t[out] = torch.cat([t[i] for i in ins], dim=1)
            else:
                t[out] = self.mods[key](t[ins[0]])
        if return_all:
            return t
        if len(self.outputs) == 1:
            return t[self.output]
        return {k: t[k] for k in self.outputs}


def export(net, path_pt):
    tn = TorchNet(net)
    torch.save({"state_dict": tn.state_dict(),
                "wiring": tn.wiring,
                "outputs": tn.outputs,
                "design": getattr(net, "design", None)}, path_pt)
    return tn


def check_equivalence(net, x, keys=None):
    """Run numpy and torch versions on the same strip; return max abs diff per
    output and the torch parameter count."""
    tn = TorchNet(net).eval()
    keys = keys or getattr(net, "outputs", [net.output])
    with torch.no_grad():
        tt = tn(torch.tensor(x[None, None, :], dtype=torch.float32), return_all=True)
    tnp = net.forward(x[None, :], return_all=True)
    diffs = {k: float(np.max(np.abs(tt[k][0].numpy() - tnp[k]))) for k in keys}
    return diffs, sum(p.numel() for p in tn.parameters())
