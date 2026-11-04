"""Build the three ReLU networks by construction from pk_spec_dev.yaml (no training, no fitting).

Run from the task directory:  cd <task dir> && python3 attempts/haiku/build_weights.py

Network input  x = [A_d, A_c, C_e, tau, u_d, u_c]
Hidden layer 1 H1 = [C', A_d', A_c', G1, G2, tau]           (6 units)
Hidden layer 2 H2 = [C', A_d', A_c', T, I_0 .. I_{K-1}]      (4 + K units)
Output layer      y = [A_d', A_c', C_e', tau', E, dHR]       (linear)
"""
import math
import os
import sys

import numpy as np
import yaml

TASK = os.getcwd()
sys.path.insert(0, TASK)
from wsp_layers import ReLUNet, interpolant_rows  # noqa: E402

OUT = os.path.join(TASK, "attempts", "haiku")
LN2 = math.log(2.0)
DT = 1.0
S = 1000.0    # dose gate steepness: gate is exactly 1 whenever total dose >= 1e-3, exactly 0 when no dose
BIG = 1.0e4   # tau reset weight; BIG > tau_max + dt (90 min -> tau <= 5400 s)
CMAX = 12.0   # interpolant range for C_e'; total dose in dev schedules is <= 10.3
TOL = 2e-3    # target bound on the chord error of the Emax and dHR interpolants

spec = yaml.safe_load(open(os.path.join(TASK, "pk_spec_dev.yaml")))


def rate(half_min):
    return LN2 / (half_min * 60.0)


def choose_h(v):
    """Chord error of a piecewise-linear interpolant with step h is <= h^2/8 * max|f''|.
    f = Emax*C/(EC50+C): max|f''| = 2*Emax/EC50^2 at C=0  -> err_E  <= Emax*h^2/(4*EC50^2).
    f = HRmax*C/(EC50_hr+C): err_HR <= HRmax*h^2/(4*EC50_hr^2).
    Pick the largest step from a fixed list whose bound is <= TOL."""
    for h in [0.05, 0.02, 0.01, 0.005]:
        e = max(v["Emax"] * h * h / (4 * v["EC50"] ** 2),
                v["HRmax"] * h * h / (4 * v["EC50_hr"] ** 2))
        if e <= TOL:
            return h, e
    raise ValueError("no step satisfies TOL")


def build(name, v):
    a_d = math.exp(-rate(v["ka_half_min"]) * DT)   # depot retention per tick
    a_c = math.exp(-rate(v["ke_half_min"]) * DT)   # central retention per tick
    al = 1.0 - math.exp(-rate(v["keo_half_min"]) * DT)  # effect-compartment step

    h, bound = choose_h(v)
    N = int(round(CMAX / h))
    knots = [j * h for j in range(N + 1)]

    f_E = lambda c: v["Emax"] * c / (v["EC50"] + c)
    f_HR = lambda c: v["HRmax"] * c / (v["EC50_hr"] + c)

    # ---- layer 1: pre-activations are non-negative combinations of non-negative quantities,
    # so the ReLUs on u0,u1,u2,u5 are identities; G1/G2 are the dose gate.
    W1 = np.zeros((6, 6)); b1 = np.zeros(6)
    # u0 = C' = (1-al) C_e + al*A_c'   (A_c' from the doses-added, moved, decayed chain)
    W1[0] = [al * a_c * (1 - a_d), al * a_c, 1 - al, 0.0, al * a_c * (1 - a_d), al * a_c]
    # u1 = A_d' = a_d * (A_d + u_d)
    W1[1] = [a_d, 0.0, 0.0, 0.0, a_d, 0.0]
    # u2 = A_c' = a_c*(A_c + u_c) + a_c*(1-a_d)*(A_d + u_d)
    W1[2] = [a_c * (1 - a_d), a_c, 0.0, 0.0, a_c * (1 - a_d), a_c]
    # u3 = G1 = ReLU(S*(u_d+u_c)),  u4 = G2 = ReLU(S*(u_d+u_c) - 1);  G1 - G2 = dose indicator
    W1[3] = [0.0, 0.0, 0.0, 0.0, S, S]; b1[3] = 0.0
    W1[4] = [0.0, 0.0, 0.0, 0.0, S, S]; b1[4] = -1.0
    # u5 = tau pass-through
    W1[5] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    # ---- layer 2
    # interpolant units I_j = ReLU(C' - k_j) (C' >= 0, so this is exact); shared by E and dHR
    Wi, bi, wE, fE0 = interpolant_rows(6, 0, f_E, knots)
    _, _, wHR, fHR0 = interpolant_rows(6, 0, f_HR, knots)
    assert abs(fE0) < 1e-15 and abs(fHR0) < 1e-15  # f(0) = 0 for both readouts
    K = Wi.shape[0]
    W2 = np.zeros((4 + K, 6)); b2 = np.zeros(4 + K)
    W2[0] = [1, 0, 0, 0, 0, 0]                  # P0 = C'
    W2[1] = [0, 1, 0, 0, 0, 0]                  # P1 = A_d'
    W2[2] = [0, 0, 1, 0, 0, 0]                  # P2 = A_c'
    W2[3] = [0, 0, 0, -BIG, BIG, 1.0]           # T = ReLU(tau + dt - BIG*(G1-G2)) = tau'
    b2[3] = DT
    W2[4:] = Wi; b2[4:] = bi                    # I_j

    # ---- output (linear)
    W3 = np.zeros((6, 4 + K)); b3 = np.zeros(6)
    W3[0, 1] = 1.0                 # A_d'
    W3[1, 2] = 1.0                 # A_c'
    W3[2, 0] = 1.0                 # C_e'
    W3[3, 3] = 1.0                 # tau'
    W3[4, 4:] = wE                 # E = Emax*C'/(EC50+C') via interpolant (Emax inside f_E)
    W3[5, 4:] = wHR                # dHR = HRmax*C'/(EC50_hr+C') via interpolant

    net = ReLUNet()
    net.add(W1, b1); net.add(W2, b2); net.add(W3, b3)
    path = os.path.join(OUT, f"weights_{name}.npz")
    net.save(path)
    print(f"{name}: knot_step={h} intervals={K} chord_err_bound={bound:.2e} "
          f"shapes={[W.shape for W, _ in net.layers]} n_params={net.n_params} saved={path}")
    return net


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name in ["typical", "fast_absorber", "slow_eliminator"]:
        build(name, spec["variants"][name])
