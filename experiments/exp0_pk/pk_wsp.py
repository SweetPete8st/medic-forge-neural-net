"""A2 compile target: the specification compiled into hand-set ReLU weights.

Construction (written from pk_spec semantics; no training, no fitting, no search):
  Cell input x = [A_d, A_c, C_e, tau, u_d, u_c, 1(bias handled by b)]  (6 inputs)
  Layer 1 (ReLU), 8 units:
    0: A_d'  = a_d * A_d + a_d * u_d                          (linear, >= 0 so ReLU is identity)
    1: A_c'  = a_c*(1-a_d)*A_d + a_c*A_c + a_c*(1-a_d)*u_d + a_c*u_c
    2: C_e'  = (1-alpha)*C_e + alpha*A_c'   (expanded in terms of inputs)
    3: tau + dt                                                (ramp before gating)
    4,5: steep clip of (u_d + u_c) -> dose indicator g in {0,1}
    6,7: unused spare (zero rows) kept so widths are explicit
  Layer 2 (ReLU), 4 + 1 + 2K units:
    0..2: identity on A_d', A_c', C_e'
    3:    tau' = ReLU(tau + dt - BIG * g)                      (gated ramp)
    4..:  interpolant knots on C_e' for E (K-1 units) and for dHR (K-1 units)
  Layer 3 (linear) readout: [A_d', A_c', C_e', tau', E, dHR]
Parameter count is reported by the experiment runner.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mfnn.wsp.layers import ReLUNet, clip01_rows, gated_ramp_rows, interpolant_rows, relu  # noqa: E402
from pk_code import rates  # noqa: E402

BIG = 1.0e7       # >= tau_max + dt for a 180 min rollout (10,800 s)
CLIP_SCALE = 1e6  # any dose >= 1e-6 units registers as a dose event
KNOTS_V1 = [0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 13.0, 17.0, 22.0, 30.0, 40.0]
# v2 (DC 2, DATA-INFORMED): knots are placed as multiples of EC50 so the chord error of c/(EC50+c) is
# bounded independently of the variant; absolute far knots cover concentrations beyond 100*EC50.
KNOTS_REL = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 13.0, 17.0, 22.0, 30.0, 40.0, 60.0, 100.0]
KNOTS_ABS_FAR = [40.0, 60.0, 100.0]
VERSION = 2


def knots_for(ec50: float):
    if VERSION == 1:
        return KNOTS_V1
    return sorted(set([k * ec50 for k in KNOTS_REL] + KNOTS_ABS_FAR))


def compile_pk(v: dict, dt: float = 1.0) -> ReLUNet:
    r = rates(v, dt)
    a_d, a_c, al = r["a_d"], r["a_c"], r["alpha"]
    net = ReLUNet()
    # ---- layer 1: inputs [A_d, A_c, C_e, tau, u_d, u_c]
    W1 = np.zeros((8, 6)); b1 = np.zeros(8)
    W1[0] = [a_d, 0, 0, 0, a_d, 0]
    W1[1] = [a_c * (1 - a_d), a_c, 0, 0, a_c * (1 - a_d), a_c]
    W1[2] = [al * a_c * (1 - a_d), al * a_c, (1 - al), 0, al * a_c * (1 - a_d), al * a_c]
    W1[3] = [0, 0, 0, 1, 0, 0]; b1[3] = dt
    # dose gate: sum of doses through a steep clip (two units)
    Wc = np.zeros((2, 6)); Wc[0, 4] = CLIP_SCALE; Wc[0, 5] = CLIP_SCALE; Wc[1] = Wc[0]
    W1[4:6] = Wc; b1[4] = 0.0; b1[5] = -1.0
    net.add(W1, b1)
    # ---- layer 2: inputs are the 8 hidden units of layer 1
    fE = lambda c: v["Emax"] * c / (v["EC50"] + c)
    fH = lambda c: v["HRmax"] * c / (v["EC50_hr"] + c)
    WE, bE, rE, biasE = interpolant_rows(8, 2, fE, knots_for(v["EC50"]))
    WH, bH, rH, biasH = interpolant_rows(8, 2, fH, knots_for(v["EC50_hr"]))
    KE, KH = WE.shape[0], WH.shape[0]
    K = max(KE, KH)
    # pad the shorter interpolant with zero rows so both blocks are K wide
    pad = lambda W, b, n: (np.vstack([W, np.zeros((n - W.shape[0], W.shape[1]))]), np.concatenate([b, np.zeros(n - b.shape[0])]))
    WE, bE = pad(WE, bE, K); WH, bH = pad(WH, bH, K)
    rE = np.concatenate([rE, np.zeros(K - rE.shape[0])]); rH = np.concatenate([rH, np.zeros(K - rH.shape[0])])
    W2 = np.zeros((4 + 2 * K, 8)); b2 = np.zeros(4 + 2 * K)
    W2[0, 0] = 1; W2[1, 1] = 1; W2[2, 2] = 1                    # identities
    W2[3, 3] = 1; W2[3, 4] = -BIG; W2[3, 5] = +BIG              # tau' = ReLU((tau+dt) - BIG*(clip0 - clip1))
    W2[4:4 + K] = WE; b2[4:4 + K] = bE
    W2[4 + K:4 + 2 * K] = WH; b2[4 + K:4 + 2 * K] = bH
    net.add(W2, b2)
    # ---- layer 3: linear readout [A_d', A_c', C_e', tau', E, dHR]
    W3 = np.zeros((6, 4 + 2 * K)); b3 = np.zeros(6)
    W3[0, 0] = 1; W3[1, 1] = 1; W3[2, 2] = 1; W3[3, 3] = 1
    W3[4, 4:4 + K] = rE; b3[4] = biasE
    W3[5, 4 + K:4 + 2 * K] = rH; b3[5] = biasH
    net.add(W3, b3)
    return net


def rollout(net: ReLUNet, doses, n_ticks: int, dt: float = 1.0) -> np.ndarray:
    by_t = {}
    for t, route, amt in doses:
        d = by_t.setdefault(int(t), [0.0, 0.0])
        d[0 if route == "depot" else 1] += float(amt)
    s = np.zeros(4)
    out = np.zeros((n_ticks, 6))
    for t in range(n_ticks):
        u = by_t.get(t, [0.0, 0.0])
        y = net.forward(np.array([s[0], s[1], s[2], s[3], u[0], u[1]]))
        s = y[:4]
        out[t] = y
    return out
