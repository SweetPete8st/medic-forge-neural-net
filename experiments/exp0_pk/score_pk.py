"""Scoring function for the PK calibration task: returns only the aggregate numbers on DEV cases."""
import json, sys
from pathlib import Path
import numpy as np, yaml
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import pk_code  # noqa: E402
from mfnn.wsp.layers import ReLUNet  # noqa: E402


def score(weights_path: str, variant: str = "typical") -> dict:
    """Rollout convention: the net maps [A_d, A_c, C_e, tau, u_d, u_c] -> [A_d', A_c', C_e', tau', E, dHR]."""
    spec = yaml.safe_load((HERE / "pk_spec_dev.yaml").read_text())
    net = ReLUNet.load(weights_path); v = spec["variants"][variant]
    worst = 0.0
    for sn, sch in spec["schedules"].items():
        doses = [(d[0], d[1], d[2]) for d in sch["doses"]]; n = int(sch["minutes"] * 60)
        ref = pk_code.rollout(v, doses, n)
        by_t = {}
        for t, route, amt in doses:
            d = by_t.setdefault(int(t), [0.0, 0.0]); d[0 if route == "depot" else 1] += float(amt)
        s = np.zeros(4); dE = 0.0
        for t in range(n):
            u = by_t.get(t, [0.0, 0.0]); y = net.forward(np.array([s[0], s[1], s[2], s[3], u[0], u[1]])); s = y[:4]
            dE = max(dE, abs(y[4] - ref[t, 4]))
        worst = max(worst, dE)
    return {"max_dE_dev": worst, "pass": worst <= 0.01}
