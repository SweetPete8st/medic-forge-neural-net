"""Construct the three ReLU-network weight sets from the specification (no training, no fitting).

Input x = [A_d, A_c, C_e, tau, u_d, u_c]  (index 0..5)
Layer 1 (6 ReLU units, all arguments provably >= 0 so ReLU is identity on them):
  H1[0] = u1 : ReLU(S*(u_d+u_c))             dose-gate, upper ramp
  H1[1] = u2 : ReLU(S*(u_d+u_c) - 1)         dose-gate, lower ramp
  H1[2] = T  : ReLU(tau + dt)                 tau advanced by one tick
  H1[3] = A_d': exp(-ka dt) (A_d + u_d)       depot after absorption
  H1[4] = A_c': exp(-ke dt) (A_c + u_c + a (A_d + u_d)),  a = 1 - exp(-ka dt)
  H1[5] = C_e': (1-q) C_e + q A_c',           q = 1 - exp(-keo dt)
Layer 2 (ReLU units):
  H2[0] = tau' : ReLU(T - BIG*(u1 - u2))      reset to 0 on dose ticks
  H2[1..3]     : pass-through of A_d', A_c', C_e'
  H2[4..]      : ReLU(C_e' - k_j) hinges of the E interpolant (knots k_j)
  H2[..]       : ReLU(C_e' - k_j) hinges of the dHR interpolant
Layer 3 (linear output): y = [A_d', A_c', C_e', tau', E, dHR].
"""
import math
import sys

sys.dont_write_bytecode = True
import numpy as np
import yaml

TASK = "/tmp/claude-0/-home-user-book-bot/2e9296c2-9c9d-5584-a4a2-006603fef937/scratchpad/builder_tasks/pk"
ATT = TASK + "/attempts/haiku2"
sys.path.insert(0, TASK)
from wsp_layers import ReLUNet, interpolant_rows  # noqa: E402

SPEC = yaml.safe_load(open(TASK + "/pk_spec_dev.yaml"))
DT = float(SPEC["dt_s"])
S = 100.0        # dose gate slope: exact 0/1 indicator for any dose amount >= 0.01
BIG = 1e4        # > max tau + dt (rollouts <= 90 min = 5400 ticks)
CMAX = 20.0      # knot range for readouts; total dose in dev schedules is 10.3
EPS_E = 0.001    # max chord error allowed for E (pass criterion is 0.01)
EPS_H = 0.002    # max chord error for dHR (not in pass criterion, kept small anyway)


def knots_for(EC, Hmax, eps, cmax):
    """Greedy knots so every chord interval [c, c+h] has error bound h^2/8 * |f''|_max <= eps.
    f(C) = Hmax*C/(EC+C), |f''(C)| = 2*Hmax*EC/(EC+C)^3 is decreasing, so its max on [c, c+h]
    is at c. Spacing therefore grows like (EC+C)^1.5: fine near EC50, coarse far away."""
    ks = [0.0]
    while ks[-1] < cmax:
        c = ks[-1]
        f2 = 2.0 * Hmax * EC / (EC + c) ** 3
        ks.append(c + math.sqrt(8.0 * eps / f2))
    return ks


def build(v):
    ka = math.log(2) / (v["ka_half_min"] * 60.0)
    ke = math.log(2) / (v["ke_half_min"] * 60.0)
    keo = math.log(2) / (v["keo_half_min"] * 60.0)
    ea = math.exp(-ka * DT)          # 1 - a
    a = 1.0 - ea
    p = math.exp(-ke * DT)
    q = 1.0 - math.exp(-keo * DT)

    W1 = np.zeros((6, 6)); b1 = np.zeros(6)
    W1[0, 4] = S; W1[0, 5] = S                       # u1
    W1[1, 4] = S; W1[1, 5] = S; b1[1] = -1.0         # u2
    W1[2, 3] = 1.0; b1[2] = DT                       # T = tau + dt
    W1[3, 0] = ea; W1[3, 4] = ea                     # A_d'
    W1[4, 0] = p * a; W1[4, 1] = p; W1[4, 4] = p * a; W1[4, 5] = p   # A_c'
    W1[5, 0] = q * p * a; W1[5, 1] = q * p; W1[5, 2] = 1.0 - q       # C_e'
    W1[5, 4] = q * p * a; W1[5, 5] = q * p

    KE = knots_for(v["EC50"], v["Emax"], EPS_E, CMAX)
    fE = lambda c: v["Emax"] * c / (v["EC50"] + c)
    WE, bE, wE, f0E = interpolant_rows(6, 5, fE, KE)
    KH = knots_for(v["EC50_hr"], v["HRmax"], EPS_H, CMAX)
    fH = lambda c: v["HRmax"] * c / (v["EC50_hr"] + c)
    WH, bH, wH, f0H = interpolant_rows(6, 5, fH, KH)
    assert f0E == 0.0 and f0H == 0.0
    nE, nH = WE.shape[0], WH.shape[0]
    n2 = 4 + nE + nH
    W2 = np.zeros((n2, 6)); b2 = np.zeros(n2)
    W2[0, 0] = -BIG; W2[0, 1] = BIG; W2[0, 2] = 1.0  # tau'
    W2[1, 3] = 1.0                                    # A_d' pass
    W2[2, 4] = 1.0                                    # A_c' pass
    W2[3, 5] = 1.0                                    # C_e' pass
    W2[4:4 + nE] = WE; b2[4:4 + nE] = bE
    W2[4 + nE:] = WH; b2[4 + nE:] = bH
    W3 = np.zeros((6, n2)); b3 = np.zeros(6)
    W3[0, 1] = 1.0; W3[1, 2] = 1.0; W3[2, 3] = 1.0; W3[3, 0] = 1.0
    W3[4, 4:4 + nE] = wE
    W3[5, 4 + nE:] = wH
    net = ReLUNet()
    net.add(W1, b1); net.add(W2, b2); net.add(W3, b3)
    info = dict(ka=ka, ke=ke, keo=keo, nE=nE, nH=nH, n_params=net.n_params,
                chordE=max(fE(KE[i]) for i in range(len(KE))),
                ea=ea, p=p, q=q, a=a)
    return net, info, fE, KE, fH, KH


def ref_sim(v, sched):
    ka = math.log(2) / (v["ka_half_min"] * 60.0)
    ke = math.log(2) / (v["ke_half_min"] * 60.0)
    keo = math.log(2) / (v["keo_half_min"] * 60.0)
    N = int(sched["minutes"] * 60)
    doses = {}
    for t, comp, amt in sched["doses"]:
        d = doses.setdefault(int(t), [0.0, 0.0])
        d[0 if comp == "depot" else 1] += amt
    Ad = Ac = C = tau = 0.0
    out = []
    for t in range(N):
        dd, dc = doses.get(t, [0.0, 0.0])
        Ad += dd; Ac += dc
        moved = Ad * (1 - math.exp(-ka * DT)); Ad -= moved; Ac += moved
        Ac *= math.exp(-ke * DT)
        C += (Ac - C) * (1 - math.exp(-keo * DT))
        tau = 0.0 if (dd + dc) > 0 else tau + DT
        E = v["Emax"] * C / (v["EC50"] + C)
        dH = v["HRmax"] * C / (v["EC50_hr"] + C)
        out.append((Ad, Ac, C, tau, E, dH, dd, dc))
    return out


def net_sim(net, sched):
    N = int(sched["minutes"] * 60)
    doses = {}
    for t, comp, amt in sched["doses"]:
        d = doses.setdefault(int(t), [0.0, 0.0])
        d[0 if comp == "depot" else 1] += amt
    st = np.zeros(4)
    out = []
    for t in range(N):
        dd, dc = doses.get(t, [0.0, 0.0])
        y = net.forward(np.array([st[0], st[1], st[2], st[3], dd, dc]))
        st = y[:4]
        out.append((st[0], st[1], st[2], st[3], y[4], y[5], dd, dc))
    return out


if __name__ == "__main__":
    for name, v in SPEC["variants"].items():
        net, info, fE, KE, fH, KH = build(v)
        # shape check
        assert [W.shape for W, _ in net.layers] == [(6, 6), (net.layers[1][0].shape[0], 6), (6, net.layers[1][0].shape[0])]
        # chord error of the knots (grid check of the interpolant against f)
        grid = np.linspace(0, CMAX, 20001)
        errE = max(abs(float(np.interp(c, KE, [fE(k) for k in KE])) - fE(c)) for c in grid[::50])
        print(f"[{name}] ka={info['ka']:.6g} ke={info['ke']:.6g} keo={info['keo']:.6g} "
              f"knotsE={info['nE']} knotsH={info['nH']} n_params={info['n_params']}")
        print(f"[{name}] E chord grid max err (approx, 400 pts) = {errE:.2e}  knot range [0,{KE[-1]:.2f}]")
        worst = dict(A=0.0, C=0.0, tau=0.0, E=0.0, dHR=0.0)
        for sname, sched in SPEC["schedules"].items():
            r = ref_sim(v, sched)
            n = net_sim(net, sched)
            r = np.array(r); n = np.array(n)
            worst["A"] = max(worst["A"], np.abs(r[:, 0:2] - n[:, 0:2]).max())
            worst["C"] = max(worst["C"], np.abs(r[:, 2] - n[:, 2]).max())
            worst["tau"] = max(worst["tau"], np.abs(r[:, 3] - n[:, 3]).max())
            worst["E"] = max(worst["E"], np.abs(r[:, 4] - n[:, 4]).max())
            worst["dHR"] = max(worst["dHR"], np.abs(r[:, 5] - n[:, 5]).max())
        print(f"[{name}] self-check vs my reference simulator over all dev schedules: "
              + " ".join(f"{k}={val:.2e}" for k, val in worst.items()))
        net.save(f"{ATT}/weights_{name}.npz")
        print(f"[{name}] saved {ATT}/weights_{name}.npz")
