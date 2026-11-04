"""CODE compile target: the specification executed as ordinary Python. Ground truth by definition."""
from __future__ import annotations
import math
import numpy as np

LN2 = math.log(2.0)


def rates(v: dict, dt: float) -> dict:
    k = lambda half_min: LN2 / (half_min * 60.0)
    return {"a_d": math.exp(-k(v["ka_half_min"]) * dt), "a_c": math.exp(-k(v["ke_half_min"]) * dt),
            "alpha": 1.0 - math.exp(-k(v["keo_half_min"]) * dt)}


def step(v: dict, s, u, dt: float = 1.0):
    """One tick from arbitrary state s=[A_d,A_c,C_e,tau] with inputs u=[u_d,u_c]. Returns (s', E, dHR)."""
    r = rates(v, dt)
    A_d, A_c, C_e, tau = float(s[0]) + float(u[0]), float(s[1]) + float(u[1]), float(s[2]), float(s[3])
    moved = A_d * (1.0 - r["a_d"]); A_d -= moved; A_c = (A_c + moved) * r["a_c"]; C_e += (A_c - C_e) * r["alpha"]
    tau = 0.0 if (u[0] + u[1]) > 0 else tau + dt
    return (A_d, A_c, C_e, tau), v["Emax"] * C_e / (v["EC50"] + C_e), v["HRmax"] * C_e / (v["EC50_hr"] + C_e)


def rollout(v: dict, doses, n_ticks: int, dt: float = 1.0) -> np.ndarray:
    """Returns array [n_ticks, 6]: A_d, A_c, C_e, tau, E, dHR after each tick."""
    r = rates(v, dt)
    by_t = {}
    for t, route, amt in doses:
        by_t.setdefault(int(t), []).append((route, float(amt)))
    A_d = A_c = C_e = tau = 0.0
    out = np.zeros((n_ticks, 6))
    for t in range(n_ticks):
        dosed = False
        for route, amt in by_t.get(t, []):
            if route == "depot": A_d += amt
            else: A_c += amt
            dosed = True
        moved = A_d * (1.0 - r["a_d"])
        A_d -= moved
        A_c = (A_c + moved) * r["a_c"]
        C_e += (A_c - C_e) * r["alpha"]
        tau = 0.0 if dosed else tau + dt
        E = v["Emax"] * C_e / (v["EC50"] + C_e)
        dHR = v["HRmax"] * C_e / (v["EC50_hr"] + C_e)
        out[t] = (A_d, A_c, C_e, tau, E, dHR)
    return out
