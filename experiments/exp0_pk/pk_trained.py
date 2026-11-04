"""D compile target: a conventionally trained next-step MLP (numpy, Adam, manual backprop).
Trained only on DEV transitions. Inputs: normalised [A_d, A_c, C_e, tau, u_d, u_c]; outputs:
normalised deltas of [A_d, A_c, C_e] plus E and dHR of the new state. tau is handled by the same
rule as the spec (reset on dose) because learning an exact reset from data is not the point here."""
from __future__ import annotations
import numpy as np

SCALE_IN = np.array([5.0, 5.0, 5.0, 5400.0, 5.0, 1.0])
SCALE_OUT = np.array([0.05, 0.05, 0.05, 1.0, 40.0])   # deltas are small per 1 s tick


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (b, a)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(w) for w in self.W + self.b]; self.v = [np.zeros_like(w) for w in self.W + self.b]; self.t = 0

    def forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = acts[-1] @ W.T + b
            acts.append(np.maximum(Z, 0) if i < len(self.W) - 1 else Z)
        return acts

    def step(self, X, Y, lr=1e-3):
        acts = self.forward(X)
        n = X.shape[0]
        G = (acts[-1] - Y) / n
        grads_W, grads_b = [], []
        for i in range(len(self.W) - 1, -1, -1):
            grads_W.insert(0, G.T @ acts[i]); grads_b.insert(0, G.sum(0))
            if i > 0:
                G = (G @ self.W[i]) * (acts[i] > 0)
        self.t += 1
        params = self.W + self.b; grads = grads_W + grads_b
        for j, (p, g) in enumerate(zip(params, grads)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * g; self.v[j] = 0.999 * self.v[j] + 0.001 * g * g
            mh = self.m[j] / (1 - 0.9 ** self.t); vh = self.v[j] / (1 - 0.999 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return float(np.mean((acts[-1] - Y) ** 2))

    @property
    def n_params(self):
        return int(sum(w.size for w in self.W) + sum(b.size for b in self.b))


def make_dataset(traces):
    """traces: list of (code_rollout [T,6], doses_by_tick dict). Returns X, Y arrays."""
    Xs, Ys = [], []
    for arr, by_t in traces:
        T = arr.shape[0]
        prev = np.zeros(4)
        for t in range(T):
            u = by_t.get(t, [0.0, 0.0])
            x = np.array([prev[0], prev[1], prev[2], prev[3], u[0], u[1]]) / SCALE_IN
            new = arr[t]
            y = np.array([new[0] - prev[0], new[1] - prev[1], new[2] - prev[2], new[4], new[5]]) / SCALE_OUT
            Xs.append(x); Ys.append(y)
            prev = new[:4]
    return np.array(Xs), np.array(Ys)


def train(traces, hidden=64, steps=4000, batch=1024, seed=0, lr=1e-3):
    X, Y = make_dataset(traces)
    rng = np.random.default_rng(seed)
    net = MLP([6, hidden, hidden, 5], seed=seed)
    losses = []
    for s in range(steps):
        idx = rng.integers(0, X.shape[0], batch)
        losses.append(net.step(X[idx], Y[idx], lr=lr * (0.1 if s > steps * 0.75 else 1.0)))
    return net, losses


def train_dagger(v, schedules, step_fn, hidden=64, steps=4000, batch=1024, seed=0, lr=1e-3, rounds=3, extra_steps=2000):
    """Config change 1 (v2): after initial training, roll the model out on the DEV schedules, label the
    states it actually visits with the code one-step function, aggregate, and continue training.
    `schedules`: list of (doses_by_tick, n_ticks). `step_fn(s, u) -> (s_next, E, dHR)` is the code rule."""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    def label(states_inputs):
        Xs, Ys = [], []
        for s, u in states_inputs:
            s2, E, dHR = step_fn(s, u)
            Xs.append(_np.array([s[0], s[1], s[2], s[3], u[0], u[1]]) / SCALE_IN)
            Ys.append(_np.array([s2[0] - s[0], s2[1] - s[1], s2[2] - s[2], E, dHR]) / SCALE_OUT)
        return _np.array(Xs), _np.array(Ys)
    # initial data: exact trajectories
    pairs = []
    for by_t, n in schedules:
        s = _np.zeros(4)
        for t in range(n):
            u = by_t.get(t, [0.0, 0.0]); pairs.append((s.copy(), u)); s = _np.array(step_fn(s, u)[0])
    X, Y = label(pairs)
    net = MLP([6, hidden, hidden, 5], seed=seed); losses = []
    for st in range(steps):
        idx = rng.integers(0, X.shape[0], batch); losses.append(net.step(X[idx], Y[idx], lr=lr))
    for r in range(rounds):
        visited = []
        for by_t, n in schedules:
            arr = rollout(net, [(t, "depot" if u[0] > 0 else "central", u[0] if u[0] > 0 else u[1]) for t, u in by_t.items()], n)
            prev = _np.zeros(4)
            for t in range(n):
                u = by_t.get(t, [0.0, 0.0]); visited.append((prev.copy(), u)); prev = arr[t, :4]
        Xv, Yv = label(visited); X = _np.vstack([X, Xv]); Y = _np.vstack([Y, Yv])
        for st in range(extra_steps):
            idx = rng.integers(0, X.shape[0], batch); losses.append(net.step(X[idx], Y[idx], lr=lr * (0.3 if r == rounds - 1 else 1.0)))
    return net, losses


def rollout(net: MLP, doses, n_ticks: int, dt: float = 1.0) -> np.ndarray:
    by_t = {}
    for t, route, amt in doses:
        d = by_t.setdefault(int(t), [0.0, 0.0]); d[0 if route == "depot" else 1] += float(amt)
    s = np.zeros(4); out = np.zeros((n_ticks, 6))
    for t in range(n_ticks):
        u = by_t.get(t, [0.0, 0.0])
        x = np.array([s[0], s[1], s[2], s[3], u[0], u[1]]) / SCALE_IN
        y = net.forward(x[None, :])[-1][0] * SCALE_OUT
        s = np.array([max(s[0] + y[0], 0.0), max(s[1] + y[1], 0.0), max(s[2] + y[2], 0.0), 0.0 if (u[0] + u[1]) > 0 else s[3] + dt])
        out[t] = (s[0], s[1], s[2], s[3], y[3], y[4])
    return out
