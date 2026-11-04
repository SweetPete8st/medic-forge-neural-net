"""Reusable weight-space constructions, re-implemented in numpy.

PROVENANCE: the brief lists constructions that exist in `src/wsp/layers.py` of the ECG archive
(ecg-wsp-v4.zip): thresholds and logic gates (steep ReLU clips, AND/OR units), exact integer
division (one-hot gating), time-since-event (gated positional ramp), reciprocals and other
nonlinear functions (ReLU interpolants with chosen knots), delayed effects (lag-box and shift
convolutions), periodicity (comb banks), deduplication (causal max-pool), absolute floors.
That archive was not available, so this file is a fresh implementation of the subset needed for
the patient-dynamics experiments, written from the descriptions. When the archive is available,
prefer its tested versions and keep this file only for numpy-only environments.

Every helper returns explicit weight rows so a reader can audit what each hidden unit computes.
All networks here are plain ReLU MLP cells applied recurrently; no training anywhere.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


@dataclass
class ReLUNet:
    """Feed-forward ReLU network: ReLU on every layer except the last (linear)."""
    layers: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)   # (W [out,in], b [out])

    def add(self, W: np.ndarray, b: np.ndarray) -> None:
        W = np.asarray(W, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
        assert W.ndim == 2 and b.shape == (W.shape[0],)
        if self.layers:
            assert W.shape[1] == self.layers[-1][0].shape[0], "layer width mismatch"
        self.layers.append((W, b))

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=np.float64)
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1:
                h = relu(h)
        return h

    @property
    def n_params(self) -> int:
        return int(sum(W.size + b.size for W, b in self.layers))

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez(buf, **{f"W{i}": W for i, (W, _) in enumerate(self.layers)}, **{f"b{i}": b for i, (_, b) in enumerate(self.layers)})
        return buf.getvalue()

    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(self.to_bytes())

    @classmethod
    def load(cls, path: str) -> "ReLUNet":
        d = np.load(path)
        n = len([k for k in d.files if k.startswith("W")])
        net = cls()
        for i in range(n):
            net.add(d[f"W{i}"], d[f"b{i}"])
        return net


# ----------------------------------------------------------------------------- constructions
# Each construction is expressed as rows of a hidden ReLU layer plus how to read them out.

def clip01_rows(in_dim: int, idx: int, scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """min(1, max(0, scale * x[idx])) = ReLU(scale x) - ReLU(scale x - 1).
    Returns (W_rows [2,in], b_rows [2], readout [2]) so that readout @ relu(W x + b) is the clip.
    With a large `scale` this is a step/threshold unit (steep ReLU clip)."""
    W = np.zeros((2, in_dim)); W[0, idx] = scale; W[1, idx] = scale
    b = np.array([0.0, -1.0])
    return W, b, np.array([1.0, -1.0])


def threshold_rows(in_dim: int, idx: int, thresh: float, sharpness: float = 1e3):
    """Indicator(x[idx] > thresh), as a steep clip around the threshold."""
    W = np.zeros((2, in_dim)); W[0, idx] = sharpness; W[1, idx] = sharpness
    b = np.array([-sharpness * thresh, -sharpness * thresh - 1.0])
    return W, b, np.array([1.0, -1.0])


def and_gate_rows(in_dim: int, idxs: Sequence[int]):
    """AND of binary inputs: ReLU(sum - (n-1)). Single unit."""
    W = np.zeros((1, in_dim))
    for i in idxs:
        W[0, i] = 1.0
    return W, np.array([-(len(idxs) - 1.0)]), np.array([1.0])


def or_gate_rows(in_dim: int, idxs: Sequence[int]):
    """OR of binary inputs: min(1, sum) = ReLU(sum) - ReLU(sum - 1)."""
    W = np.zeros((2, in_dim))
    for i in idxs:
        W[0, i] = 1.0; W[1, i] = 1.0
    return W, np.array([0.0, -1.0]), np.array([1.0, -1.0])


def gated_ramp_rows(in_dim: int, ramp_idx: int, gate_idx: int, dt: float, big: float):
    """Time-since-event: tau' = ReLU(tau + dt - big * gate). gate in {0,1}; big >= tau_max + dt.
    When gate=1 the ramp resets to 0; otherwise it advances by dt."""
    W = np.zeros((1, in_dim)); W[0, ramp_idx] = 1.0; W[0, gate_idx] = -big
    return W, np.array([dt]), np.array([1.0])


def interpolant_rows(in_dim: int, idx: int, f: Callable[[float], float], knots: Sequence[float]):
    """Piecewise-linear interpolant of f on `knots` (sorted, k0 is the left edge, f evaluated there).
    g(x) = f(k0) + sum_j w_j ReLU(x - k_j), with w_j = slope_j - slope_{j-1}.
    Exact at the knots; between knots the error is the chord error of f.
    Returns (W_rows [K-1, in], b_rows, readout, bias)."""
    ks = list(knots)
    assert all(ks[i] < ks[i + 1] for i in range(len(ks) - 1))
    slopes = [(f(ks[i + 1]) - f(ks[i])) / (ks[i + 1] - ks[i]) for i in range(len(ks) - 1)]
    w, prev = [], 0.0
    for s in slopes:
        w.append(s - prev); prev = s
    K = len(ks) - 1
    W = np.zeros((K, in_dim)); b = np.zeros(K)
    for j in range(K):
        W[j, idx] = 1.0; b[j] = -ks[j]
    return W, b, np.array(w), float(f(ks[0]))


def shift_register_matrix(n: int) -> np.ndarray:
    """Delayed effects: a state vector of the last n inputs; s' = S s + e_0 * u, S shifts by one."""
    S = np.zeros((n, n))
    for i in range(1, n):
        S[i, i - 1] = 1.0
    return S


def identity_rows(in_dim: int, idxs: Sequence[int]):
    """Pass non-negative quantities through a ReLU layer unchanged (absolute floor at 0)."""
    W = np.zeros((len(idxs), in_dim))
    for r, i in enumerate(idxs):
        W[r, i] = 1.0
    return W, np.zeros(len(idxs))


def stack_rows(parts: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    W = np.vstack([p[0] for p in parts]); b = np.concatenate([p[1] for p in parts])
    return W, b
