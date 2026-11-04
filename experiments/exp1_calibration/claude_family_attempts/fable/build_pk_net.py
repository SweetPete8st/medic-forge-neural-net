#!/usr/bin/env python3
"""Build ReLU-network weights for the PK/PD cell by construction from pk_spec_dev.yaml.
No training, no fitting: every weight is a closed-form function of the spec parameters.
Usage (from the task directory):  python3 attempts/fable/build_pk_net.py
"""
import math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TASK_DIR)
import yaml
from wsp_layers import ReLUNet, interpolant_rows

LN2 = math.log(2.0)

# Interpolant knots in units of the half-saturation constant (z = C / EC50).
# For f(z) = Emax * z/(1+z), |f''| = 2 Emax/(1+z)^3 (decreasing), so the chord error on
# [z_l, z_l + h] is <= Emax * h^2 / (4 (1+z_l)^3).  Each interval below has h <= 0.1*(1+z_l)^1.5,
# giving a bound <= 0.0025 * Emax everywhere (4x margin on the 0.01 criterion with Emax = 1).
# Last knot z = 50 (C = 40..100) is far beyond the largest reachable C (total dosed amount <= 10.3).
Z_KNOTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
           1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0,
           10.0, 12.0, 14.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0]
K = len(Z_KNOTS) - 1          # ReLU units per interpolant (one per left knot)

GATE_SCALE = 1.0e6            # gate = clip01(GATE_SCALE*(u_d+u_c)) == 1 for any dose >= 1e-6
BIG = 2.0e4                   # >= tau_max + dt; rollouts are <= 90 min = 5400 s

IN = 6                        # [A_d, A_c, C_e, tau, u_d, u_c]


def rate_consts(v, dt):
    k = lambda half_min: LN2 / (half_min * 60.0)
    a_d = math.exp(-k(v["ka_half_min"]) * dt)          # depot fraction retained per tick
    a_c = math.exp(-k(v["ke_half_min"]) * dt)          # central fraction retained per tick
    al = 1.0 - math.exp(-k(v["keo_half_min"]) * dt)    # effect-site mixing fraction per tick
    return a_d, a_c, al


def build(v, dt):
    a_d, a_c, al = rate_consts(v, dt)
    fE = lambda c: v["Emax"] * c / (v["EC50"] + c)
    fH = lambda c: v["HRmax"] * c / (v["EC50_hr"] + c)
    knots_E = [z * v["EC50"] for z in Z_KNOTS]
    knots_H = [z * v["EC50_hr"] for z in Z_KNOTS]

    net = ReLUNet()

    # ---------------- hidden layer 1 (6 -> 6): kinetics + tau pass-through + dose gate halves
    W1 = np.zeros((6, IN)); b1 = np.zeros(6)
    # h1[0] = A_d' = a_d*(A_d + u_d)
    W1[0, 0] = a_d; W1[0, 4] = a_d
    # h1[1] = A_c' = a_c*(A_c + u_c) + a_c*(1-a_d)*(A_d + u_d)
    W1[1, 0] = a_c * (1 - a_d); W1[1, 1] = a_c; W1[1, 4] = a_c * (1 - a_d); W1[1, 5] = a_c
    # h1[2] = C_e' = (1-al)*C_e + al*A_c'
    W1[2, 0] = al * a_c * (1 - a_d); W1[2, 1] = al * a_c; W1[2, 2] = 1 - al
    W1[2, 4] = al * a_c * (1 - a_d); W1[2, 5] = al * a_c
    # h1[3] = tau (non-negative pass-through)
    W1[3, 3] = 1.0
    # h1[4] = ReLU(S*(u_d+u_c)),  h1[5] = ReLU(S*(u_d+u_c) - 1);  gate = h1[4] - h1[5] in [0,1]
    W1[4, 4] = GATE_SCALE; W1[4, 5] = GATE_SCALE; b1[4] = 0.0
    W1[5, 4] = GATE_SCALE; W1[5, 5] = GATE_SCALE; b1[5] = -1.0
    net.add(W1, b1)

    # ---------------- hidden layer 2 (6 -> 4 + 2K): state pass-through, gated ramp, interpolant knots
    H2 = 4 + 2 * K
    W2 = np.zeros((H2, 6)); b2 = np.zeros(H2)
    W2[0, 0] = 1.0; W2[1, 1] = 1.0; W2[2, 2] = 1.0          # A_d', A_c', C_e' (identity, all >= 0)
    W2[3, 3] = 1.0; W2[3, 4] = -BIG; W2[3, 5] = +BIG; b2[3] = dt   # tau' = ReLU(tau + dt - BIG*gate)
    WE, bE, wE, cE = interpolant_rows(6, 2, fE, knots_E)     # ReLU(C_e' - kE_j), j = 0..K-1
    WH, bH, wH, cH = interpolant_rows(6, 2, fH, knots_H)     # ReLU(C_e' - kH_j), j = 0..K-1
    W2[4:4 + K] = WE; b2[4:4 + K] = bE
    W2[4 + K:4 + 2 * K] = WH; b2[4 + K:4 + 2 * K] = bH
    net.add(W2, b2)

    # ---------------- output layer (4 + 2K -> 6), linear
    W3 = np.zeros((6, H2)); b3 = np.zeros(6)
    for i in range(4):
        W3[i, i] = 1.0                                       # A_d', A_c', C_e', tau'
    W3[4, 4:4 + K] = wE; b3[4] = cE                          # E   = fE(0) + sum_j wE_j ReLU(C_e' - kE_j)
    W3[5, 4 + K:4 + 2 * K] = wH; b3[5] = cH                  # dHR = fH(0) + sum_j wH_j ReLU(C_e' - kH_j)
    net.add(W3, b3)
    return net


# ---------------------------------------------------------------- self-checks (against the spec text)
def spec_reference(v, dt, doses, n):
    """Straight transcription of the TASK.md spec (dose, kinetics, tau, readouts)."""
    a_d, a_c, al = rate_consts(v, dt)
    by_t = {}
    for t, route, amt in doses:
        by_t.setdefault(int(t), []).append((route, float(amt)))
    A_d = A_c = C_e = tau = 0.0; out = np.zeros((n, 6))
    for t in range(n):
        dosed = False
        for route, amt in by_t.get(t, []):
            if route == "depot": A_d += amt
            else: A_c += amt
            dosed = True
        moved = A_d * (1 - a_d); A_d -= moved; A_c += moved
        A_c *= a_c; C_e += (A_c - C_e) * al
        tau = 0.0 if dosed else tau + dt
        out[t] = (A_d, A_c, C_e, tau, v["Emax"] * C_e / (v["EC50"] + C_e), v["HRmax"] * C_e / (v["EC50_hr"] + C_e))
    return out


def chord_error(f, knots, grid=2000):
    worst = 0.0
    for a, b in zip(knots[:-1], knots[1:]):
        xs = np.linspace(a, b, grid); fa, fb = f(a), f(b)
        chord = fa + (fb - fa) * (xs - a) / (b - a)
        worst = max(worst, float(np.max(np.abs(chord - np.array([f(x) for x in xs])))))
    return worst


if __name__ == "__main__":
    spec = yaml.safe_load(open(os.path.join(TASK_DIR, "pk_spec_dev.yaml")))
    dt = float(spec["dt_s"])
    # synthetic check schedule (not one of the scoring schedules): mixed routes, 90 min
    check_doses = [(0, "depot", 5.0), (100, "central", 0.3), (1500, "depot", 2.5), (3000, "depot", 2.5), (3001, "central", 0.3)]
    n = 90 * 60
    for name, v in spec["variants"].items():
        net = build(v, dt)
        path = os.path.join(HERE, f"weights_{name}.npz")
        net.save(path)
        ref = spec_reference(v, dt, check_doses, n)
        by_t = {}
        for t, route, amt in check_doses:
            d = by_t.setdefault(int(t), [0.0, 0.0]); d[0 if route == "depot" else 1] += float(amt)
        s = np.zeros(4); worst = np.zeros(6)
        for t in range(n):
            u = by_t.get(t, [0.0, 0.0])
            y = net.forward(np.array([s[0], s[1], s[2], s[3], u[0], u[1]])); s = y[:4]
            worst = np.maximum(worst, np.abs(y - ref[t]))
        fE = lambda c: v["Emax"] * c / (v["EC50"] + c)
        fH = lambda c: v["HRmax"] * c / (v["EC50_hr"] + c)
        print(f"[{name}] layers={[W.shape for W, _ in net.layers]} n_params={net.n_params}")
        print(f"   a_d={rate_consts(v, dt)[0]:.12f} a_c={rate_consts(v, dt)[1]:.12f} al={rate_consts(v, dt)[2]:.12f}")
        print(f"   chord-error bound check: E={chord_error(fE, [z * v['EC50'] for z in Z_KNOTS]):.5f}  dHR={chord_error(fH, [z * v['EC50_hr'] for z in Z_KNOTS]):.5f}")
        print(f"   synthetic rollout max|net-spec| per output [A_d A_c C_e tau E dHR]: {worst}")
        print(f"   max C_e reached in synthetic rollout: {ref[:, 2].max():.4f}")
        print(f"   saved {path}  sha256={net.sha256()[:16]}")
