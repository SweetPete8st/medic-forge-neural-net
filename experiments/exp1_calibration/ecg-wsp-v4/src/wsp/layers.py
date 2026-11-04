"""Minimal numpy implementations of standard neural-network layers.

Everything here has an exact PyTorch equivalent (noted on each class), so a
network expressed with these layers is a real neural network, not a Python
program wearing a costume. No layer here learns anything; the weights are
handed in from outside.

Tensor convention: (channels, time), float64.
"""
import numpy as np
from scipy.signal import correlate
from scipy.ndimage import maximum_filter1d


class Conv1d:
    """torch.nn.Conv1d(in_ch, out_ch, k, padding=k//2, bias=True).

    Cross-correlation with zero padding, odd kernel, output length == input.
    weight: (out_ch, in_ch, k); bias: (out_ch,)
    """

    def __init__(self, weight, bias=None, name=""):
        self.w = np.asarray(weight, dtype=np.float64)
        assert self.w.ndim == 3 and self.w.shape[2] % 2 == 1, "odd kernels only"
        self.b = (np.zeros(self.w.shape[0]) if bias is None
                  else np.asarray(bias, dtype=np.float64))
        self.name = name

    def forward(self, x):
        out_ch, in_ch, k = self.w.shape
        assert x.shape[0] == in_ch, (self.name, x.shape, self.w.shape)
        pad = k // 2
        xp = np.pad(x, ((0, 0), (pad, pad)))  # zero padding, like torch
        y = np.empty((out_ch, x.shape[1]))
        for o in range(out_ch):
            acc = np.full(x.shape[1], self.b[o])
            for i in range(in_ch):
                wk = self.w[o, i]
                if not np.any(wk):
                    continue
                # correlate(a, v, 'valid')[t] = sum_j a[t + j] v[j]
                acc += correlate(xp[i], wk, mode="valid", method="auto")
            y[o] = acc
        return y

    def n_params(self):
        return self.w.size + self.b.size


class ReLU:
    """torch.nn.ReLU"""
    name = "relu"

    def forward(self, x):
        return np.maximum(x, 0.0)

    def n_params(self):
        return 0


class MaxPool1d:
    """torch.nn.MaxPool1d(k, stride=1, padding=k//2); pads with -inf, odd k."""

    def __init__(self, k, name=""):
        assert k % 2 == 1
        self.k = k
        self.name = name

    def forward(self, x):
        return maximum_filter1d(x, size=self.k, axis=1, mode="constant",
                                cval=-np.inf)

    def n_params(self):
        return 0


class CausalMaxPool1d:
    """y[t] = max(x[t-k], ..., x[t-1]): the maximum over the k samples strictly
    before t. torch: nn.Sequential(nn.ConstantPad1d((k, -1), -inf),
    nn.MaxPool1d(k, stride=1)). Pads with `pad_value` (default 0, correct for
    non-negative indicator signals) where nothing precedes."""

    def __init__(self, k, name="", pad_value=0.0):
        self.k = k
        self.name = name
        self.pad_value = pad_value

    def forward(self, x):
        c, t = x.shape
        z = np.concatenate([np.full((c, self.k), self.pad_value), x], axis=1)
        win = np.lib.stride_tricks.sliding_window_view(z, self.k, axis=1)  # (c, t+1, k)
        return win.max(axis=2)[:, :t]

    def n_params(self):
        return 0


class GlobalSum:
    """x.sum(dim=-1, keepdim=True): (C, T) -> (C, 1). A parameter-free reduction
    (sum pooling over the whole window)."""
    name = "gsum"

    def forward(self, x):
        return x.sum(axis=1, keepdims=True)

    def n_params(self):
        return 0


class GlobalMax:
    """torch.nn.AdaptiveMaxPool1d(1): (C, T) -> (C, 1)."""
    name = "gmax"

    def forward(self, x):
        return x.max(axis=1, keepdims=True)

    def n_params(self):
        return 0


class ChannelMax:
    """x.max(dim=channel, keepdim=True): (C, T) -> (1, T). Max over channels,
    parameter-free (maxout)."""
    name = "cmax"

    def forward(self, x):
        return x.max(axis=0, keepdims=True)

    def n_params(self):
        return 0


class Ramp:
    """A fixed positional signal r[t] = t / fs seconds, shape (1, T). Ignores
    its input's values and uses only its length. In torch this is a registered
    buffer (a constant, not a parameter), the same device as a positional
    encoding."""

    def __init__(self, fs, name="ramp"):
        self.fs = fs
        self.name = name

    def forward(self, x):
        return (np.arange(x.shape[1]) / self.fs)[None, :]

    def n_params(self):
        return 0


class Concat:
    """torch.cat(dim=0) over the channel axis."""
    name = "concat"

    def forward(self, *xs):
        return np.concatenate(xs, axis=0)

    def n_params(self):
        return 0


class Net:
    """A fixed directed acyclic graph of layers. No data-dependent control flow.

    ops: list of (layer, input_names, output_name). Input tensors are looked up
    by name in a dict; outputs are stored by name. The forward pass is a fixed
    sequence of standard layers, which is what makes this a network rather than
    a program: swapping the numbers changes behavior, the wiring never does.
    """

    def __init__(self, ops, inputs=("x",), output="beat"):
        self.ops = ops
        self.inputs = inputs
        self.output = output

    def forward(self, x, return_all=False):
        t = {self.inputs[0]: x}
        for layer, ins, out in self.ops:
            t[out] = layer.forward(*[t[i] for i in ins])
        return t if return_all else t[self.output]

    def n_params(self):
        return sum(layer.n_params() for layer, _, _ in self.ops)

    def describe(self):
        lines = []
        for layer, ins, out in self.ops:
            nm = type(layer).__name__
            extra = ""
            if isinstance(layer, Conv1d):
                extra = f" w{tuple(layer.w.shape)}"
            elif isinstance(layer, MaxPool1d):
                extra = f" k={layer.k}"
            lines.append(f"{out:>12} = {nm}{extra}({', '.join(ins)})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Composite constructions. These are not new layer types; they are small
# subgraphs built only from Conv1d (kernel size 1) and ReLU, written once so
# the design notes can refer to them by name.
# ---------------------------------------------------------------------------

def clip01(K, bias=0.0, name=""):
    """clip(K*x + bias, 0, 1) as two layers:
        h = ReLU([K*x + bias, K*x + bias - 1])     (1 -> 2 channels)
        y = h[0] - h[1]                            (2 -> 1 channels)
    Equals 0 below the threshold, 1 above threshold + 1/K, linear in between.
    """
    up = Conv1d(np.array([[[K]], [[K]]]), bias=np.array([bias, bias - 1.0]),
                name=f"{name}.up")
    down = Conv1d(np.array([[[1.0], [-1.0]]]), name=f"{name}.down")
    return up, down


def and_gate(n, name=""):
    """ReLU(sum of n indicators in [0,1] minus (n-1)).
    Equals 1 only when every input is 1; 0 if any input is 0."""
    return Conv1d(np.ones((1, n, 1)), bias=np.array([-(n - 1.0)]),
                  name=f"{name}.and")
