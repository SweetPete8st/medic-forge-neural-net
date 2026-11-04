"""Construct PK ReLU networks by hand from pk_spec_dev.yaml. No fitting of any kind."""
import math, sys, os
import numpy as np, yaml
TASK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK)
from wsp_layers import ReLUNet, interpolant_rows

LN2 = math.log(2.0)
C_MAX = 25.0          # interpolant domain upper edge; C_e <= total dose given (10.3 on dev schedules)
TOL_E, TOL_HR = 1e-5, 1e-3   # target chord-error bounds
S_GATE = 1e6          # dose-gate steepness: gate = clip01(S_GATE*(u_d+u_c))
BIG = 1e6             # reset magnitude for tau (>= tau_max + dt = 5401)

def knots_for(A, K, tol, cmax=C_MAX):
    """Knots so the chord error of f(C)=A C/(K+C) is <= tol on each segment.
    Chord error <= h^2/8 * max|f''| on segment; |f''(C)| = 2AK/(K+C)^3 is decreasing, so use the left end."""
    ks = [0.0]
    while ks[-1] < cmax:
        c = ks[-1]
        h = math.sqrt(8.0 * tol * (K + c) ** 3 / (2.0 * A * K))
        ks.append(min(c + h, cmax) if cmax - (c + h) > 1e-9 else cmax)
    return ks

def build(v, dt=1.0):
    k = lambda h: LN2 / (h * 60.0)
    a_d = math.exp(-k(v["ka_half_min"]) * dt)
    a_c = math.exp(-k(v["ke_half_min"]) * dt)
    al = 1.0 - math.exp(-k(v["keo_half_min"]) * dt)
    # input x = [Ad, Ac, Ce, tau, ud, uc]
    # Layer 1 (6 units): new state (linear, non-negative => ReLU is identity), tau passthrough, gate pair
    rAd = np.array([a_d, 0, 0, 0, a_d, 0])
    rAc = np.array([a_c * (1 - a_d), a_c, 0, 0, a_c * (1 - a_d), a_c])
    rCe = (1 - al) * np.array([0, 0, 1, 0, 0, 0]) + al * rAc
    W1 = np.vstack([rAd, rAc, rCe,
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, S_GATE, S_GATE],
                    [0, 0, 0, 0, S_GATE, S_GATE]])
    b1 = np.array([0, 0, 0, 0, 0, -1.0])
    # Layer 2: Ad', Ac', Ce' identity; tau' = ReLU(tau + dt - BIG*(g0-g1)); interpolant units on Ce'
    fE = lambda c: v["Emax"] * c / (v["EC50"] + c)
    fH = lambda c: v["HRmax"] * c / (v["EC50_hr"] + c)
    kE = knots_for(v["Emax"], v["EC50"], TOL_E)
    kH = knots_for(v["HRmax"], v["EC50_hr"], TOL_HR)
    WE, bE, rdE, cE = interpolant_rows(6, 2, fE, kE)   # idx 2 = Ce' within layer-1 output
    WH, bH, rdH, cH = interpolant_rows(6, 2, fH, kH)
    Wpass = np.zeros((4, 6)); Wpass[0, 0] = Wpass[1, 1] = Wpass[2, 2] = 1.0
    Wpass[3, 3] = 1.0; Wpass[3, 4] = -BIG; Wpass[3, 5] = BIG
    bpass = np.array([0, 0, 0, dt])
    W2 = np.vstack([Wpass, WE, WH]); b2 = np.concatenate([bpass, bE, bH])
    nE, nH = len(bE), len(bH)
    # Layer 3 (linear output)
    W3 = np.zeros((6, 4 + nE + nH)); b3 = np.zeros(6)
    for i in range(4):
        W3[i, i] = 1.0
    W3[4, 4:4 + nE] = rdE; b3[4] = cE
    W3[5, 4 + nE:] = rdH; b3[5] = cH
    net = ReLUNet(); net.add(W1, b1); net.add(W2, b2); net.add(W3, b3)
    return net, dict(a_d=a_d, a_c=a_c, al=al, nE=nE, nH=nH, kE0=kE[:4], kH0=kH[:4])

if __name__ == "__main__":
    spec = yaml.safe_load(open(os.path.join(TASK, "pk_spec_dev.yaml")))
    out = os.path.dirname(os.path.abspath(__file__))
    for name, v in spec["variants"].items():
        net, info = build(v, spec["dt_s"])
        net.save(os.path.join(out, f"weights_{name}.npz"))
        print(name, "params", net.n_params, "shapes", [W.shape for W, _ in net.layers], info)
