import numpy as np
import io, hashlib

# Minimal ReLUNet implementation
class ReLUNet:
    def __init__(self):
        self.layers = []  # list of (W, b)

    def add(self, W, b):
        W = np.asarray(W, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        assert W.ndim == 2 and b.shape == (W.shape[0],)
        if self.layers:
            assert W.shape[1] == self.layers[-1][0].shape[0]
        self.layers.append((W, b))

    def forward(self, x):
        h = np.asarray(x, dtype=np.float64)
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h

    @property
    def n_params(self):
        return sum(W.size + b.size for W, b in self.layers)

    def to_bytes(self):
        buf = io.BytesIO()
        np.savez(buf, **{f"W{i}": W for i, (W, _) in enumerate(self.layers)},
                 **{f"b{i}": b for i, (_, b) in enumerate(self.layers)})
        return buf.getvalue()

    def sha256(self):
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def save(self, path):
        with open(path, "wb") as f:
            f.write(self.to_bytes())

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls()
        n = len([k for k in d.files if k.startswith("W")])
        for i in range(n):
            net.add(d[f"W{i}"], d[f"b{i}"])
        return net

# Create an empty network (no layers)
net = ReLUNet()

# Save the network weights to weights.npz
net.save('weights.npz')

print('DONE')
