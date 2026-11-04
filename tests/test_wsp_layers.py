import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mfnn.wsp.layers import (ReLUNet, and_gate_rows, clip01_rows, gated_ramp_rows, interpolant_rows,
                             or_gate_rows, relu, threshold_rows)


def test_clip_and_threshold():
    W, b, r = clip01_rows(1, 0, 1000.0)
    for x, want in [(-1, 0), (0, 0), (0.0005, 0.5), (0.5, 1), (5, 1)]:
        assert abs(r @ relu(W @ np.array([x]) + b) - want) < 1e-9
    W, b, r = threshold_rows(1, 0, 2.0, 1e4)
    assert r @ relu(W @ np.array([2.1]) + b) == 1.0 and r @ relu(W @ np.array([1.9]) + b) == 0.0


def test_gates():
    W, b, r = and_gate_rows(2, [0, 1])
    assert r @ relu(W @ np.array([1, 1]) + b) == 1 and r @ relu(W @ np.array([1, 0]) + b) == 0
    W, b, r = or_gate_rows(2, [0, 1])
    assert r @ relu(W @ np.array([0, 1]) + b) == 1 and r @ relu(W @ np.array([0, 0]) + b) == 0


def test_gated_ramp():
    W, b, r = gated_ramp_rows(2, 0, 1, dt=1.0, big=1e6)
    tau = 0.0
    seq = [0, 0, 1, 0, 0, 0, 1, 0]
    got = []
    for g in seq:
        tau = float(r @ relu(W @ np.array([tau, g]) + b)); got.append(tau)
    assert got == [1, 2, 0, 1, 2, 3, 0, 1]


def test_interpolant_exact_at_knots():
    f = lambda c: c / (1.0 + c)
    knots = [0, 0.25, 0.5, 1, 2, 4, 8]
    W, b, r, bias = interpolant_rows(1, 0, f, knots)
    for k in knots:
        assert abs(bias + r @ relu(W @ np.array([k]) + b) - f(k)) < 1e-12
    assert abs(bias + r @ relu(W @ np.array([0.75]) + b) - f(0.75)) < 0.02
