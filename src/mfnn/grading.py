"""Learner performance grading. Reads the action log and the trace. Never influences dynamics.
Protocol grading is NOT a substitute for modelling consequences: the engine models consequences;
this module only scores what the learner did against a rubric."""
from __future__ import annotations

from typing import Any, Dict, List


def grade(spec: Dict[str, Any], history: List[Dict[str, Any]], trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    acts = history
    def first_tick(kind, **match):
        for a in acts:
            if a["kind"] == kind and all(a["params"].get(k) == v for k, v in match.items()):
                return a["tick"]
        return None
    def count(kind, **match):
        return sum(1 for a in acts if a["kind"] == kind and all(a["params"].get(k) == v for k, v in match.items()))

    results = []
    for item in spec.get("grading_rubric", []):
        rid, w = item["id"], item.get("weight", 1)
        ok = None
        if rid == "early_bronchodilator":
            t0 = first_tick("assess_general"); t0 = 0 if t0 is None else t0
            tb = first_tick("give_drug", drug="albuterol")
            ok = tb is not None and (tb - t0) <= 300
        elif rid == "oxygen_titrated":
            to2 = first_tick("oxygen"); tspo2 = first_tick("attach_device", device="spo2")
            ok = to2 is None or (tspo2 is not None and tspo2 <= to2)
        elif rid == "no_sedation":
            ok = count("give_drug", drug="midazolam_iv") == 0
        elif rid == "safe_ventilation":
            ok = all(a["params"].get("rate_per_min", 0) <= 10 for a in acts if a["kind"] == "bvm" and a["params"].get("on", True))
        elif rid == "cpap_contra_respected":
            ok = True
            loc_by_tick = {r["tick"]: r["LOC"] for r in trace}
            for a in acts:
                if a["kind"] == "cpap" and a["params"].get("on", True):
                    if loc_by_tick.get(a["tick"], 3.0) <= 1.5:
                        ok = False
        elif rid == "reassess":
            ok = count("auscultate") >= 2
        results.append({"id": rid, "weight": w, "passed": ok})
    total = sum(r["weight"] for r in results)
    got = sum(r["weight"] for r in results if r["passed"])
    return {"items": results, "score": got, "max": total}
