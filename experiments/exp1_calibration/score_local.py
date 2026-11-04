"""Scorer for the PK compile task. Self-contained. Prints ONLY the aggregate numbers on the dev cases.
Usage: python score.py weights.npz [variant]   (variant default: typical; others: fast_absorber, slow_eliminator)
Rollout convention the network must follow (one ReLUNet per variant, built from that variant's parameters):
  y = net.forward([A_d, A_c, C_e, tau, u_d, u_c]) -> [A_d', A_c', C_e', tau', E, dHR]; next state = y[:4];
  initial state zeros; u_d / u_c are the amounts dosed to depot / central at that tick (0 otherwise)."""
import math, sys, numpy as np, yaml
from wsp_layers import ReLUNet
LN2 = math.log(2.0)

def reference(v, doses, n, dt=1.0):
    k = lambda h: LN2 / (h * 60.0)
    a_d, a_c, al = math.exp(-k(v["ka_half_min"]) * dt), math.exp(-k(v["ke_half_min"]) * dt), 1 - math.exp(-k(v["keo_half_min"]) * dt)
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
        moved = A_d * (1 - a_d); A_d -= moved; A_c = (A_c + moved) * a_c; C_e += (A_c - C_e) * al
        tau = 0.0 if dosed else tau + dt
        out[t] = (A_d, A_c, C_e, tau, v["Emax"] * C_e / (v["EC50"] + C_e), v["HRmax"] * C_e / (v["EC50_hr"] + C_e))
    return out

def score(weights_path, variant="typical"):
    spec = yaml.safe_load(open("pk_spec_dev.yaml")); v = spec["variants"][variant]
    net = ReLUNet.load(weights_path); worst_E = worst_C = 0.0
    for sn, sch in spec["schedules"].items():
        doses = [(d[0], d[1], d[2]) for d in sch["doses"]]; n = int(sch["minutes"] * 60)
        ref = reference(v, doses, n)
        by_t = {}
        for t, route, amt in doses:
            d = by_t.setdefault(int(t), [0.0, 0.0]); d[0 if route == "depot" else 1] += float(amt)
        s = np.zeros(4)
        for t in range(n):
            u = by_t.get(t, [0.0, 0.0]); y = net.forward(np.array([s[0], s[1], s[2], s[3], u[0], u[1]])); s = y[:4]
            worst_E = max(worst_E, abs(y[4] - ref[t, 4])); worst_C = max(worst_C, abs(y[2] - ref[t, 2]))
    return {"variant": variant, "max_dE_dev": float(worst_E), "max_dC_dev": float(worst_C), "pass": worst_E <= 0.01}

if __name__ == "__main__":
    print(score(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "typical"))
