"""The confidence gate: completeness + schema validity + engine runs + checks. Deterministic, no LLM."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import jsonschema

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/exp2_asthma"))
from mfnn.actions import ActionKind, LearnerAction
from mfnn.engine import PatientEngine
from mfnn.spec import SCHEMA_PATH


def _acts(items):
    out = []
    for i, a in enumerate(items):
        a = dict(a); t = int(round(float(a.pop("t_min")) * 60)); kind = ActionKind(a.pop("kind"))
        out.append(LearnerAction(tick=t, kind=kind, params=a, seq=i))
    return out


def _invariants(trace):
    bad = []
    for r in trace:
        if not (0 <= r["O"] <= 1 and 0 <= r["F"] <= 1 and 0 <= r["SpO2"] <= 100 and -1e-9 <= r["LOC"] <= 3 + 1e-9): bad.append(f"range at {r['t_min']:.1f} min")
        if any(r[k] != r[k] or r[k] < 0 for k in ("HR", "SBP", "RR", "PaCO2", "PaO2")): bad.append(f"NaN/negative at {r['t_min']:.1f} min")
    return bad[:5]


def run_gate(spec: Dict[str, Any], preset: Dict[str, Any], overlay: Dict[str, Any], completeness: Dict[str, Any]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"completeness": completeness, "schema_valid": False, "trajectories": {}, "objectives": {}, "pass": False}
    try:
        jsonschema.validate(spec, json.loads(SCHEMA_PATH.read_text())); report["schema_valid"] = True
    except jsonschema.ValidationError as e:
        report["schema_error"] = str(e.message)[:300]; return report
    minutes = int(overlay.get("duration_min", 60))
    events_seen: Dict[str, List[str]] = {}
    for name, traj in spec["trajectories"].items():
        eng = PatientEngine(spec)
        trace, events = eng.run(_acts(traj["actions"]), minutes * 60, record_every=30)
        names = [e.name for e in events if e.name != "drug_given"]
        events_seen[name] = names
        checks = []
        for ex in traj.get("expect", []):
            row = min(trace, key=lambda r: abs(r["t_min"] - ex["t_min"])); v = float(row[ex["var"]])
            checks.append({"t_min": ex["t_min"], "var": ex["var"], "value": round(v, 2), "min": ex["min"], "max": ex["max"], "pass": ex["min"] <= v <= ex["max"]})
        report["trajectories"][name] = {"invariant_violations": _invariants(trace), "events": [(round(e.tick / 60, 1), e.name) for e in events if e.name != "drug_given"],
                                        "checkpoints": checks, "end": {k: round(float(trace[-1][k]), 1) for k in ("SpO2", "HR", "RR", "PaCO2", "LOC")},
                                        "arrest_min": next((round(e.tick / 60, 1) for e in events if e.name == "cardiac_arrest"), None)}
    # teaching objectives must be reachable on some path
    for obj in overlay.get("teaching_objectives", []):
        for kw, ev in preset["objective_events"].items():
            if kw in obj.lower():
                reachable = any(ev in names for names in events_seen.values())
                report["objectives"][obj] = {"requires_event": ev, "reachable": reachable}
    # sanity: timely treatment must end better than untreated
    u, t = report["trajectories"].get("untreated", {}).get("end", {}), report["trajectories"].get("timely_treatment", {}).get("end", {})
    report["treatment_helps"] = bool(u and t and t.get("SpO2", 0) >= u.get("SpO2", 0) and t.get("LOC", 0) >= u.get("LOC", 0))
    ok = (completeness["complete"] and report["schema_valid"] and report["treatment_helps"]
          and all(not v["invariant_violations"] for v in report["trajectories"].values())
          and all(c["pass"] for v in report["trajectories"].values() for c in v["checkpoints"])
          and all(o["reachable"] for o in report["objectives"].values()))
    report["pass"] = ok
    return report
