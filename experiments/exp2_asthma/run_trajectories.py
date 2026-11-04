#!/usr/bin/env python3
"""Run the four specified asthma trajectories through the code-compiled engine, check the
specification's expected envelopes, enumerate interaction cases, and run the pairwise causal
consistency sweep. Writes results to experiments/exp2_asthma/results/."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mfnn.actions import ActionKind, LearnerAction  # noqa: E402
from mfnn.engine import PatientEngine  # noqa: E402
from mfnn.ledger import log_forward_passes  # noqa: E402
from mfnn.spec import load_spec  # noqa: E402

SPEC = ROOT / "spec" / "examples" / "asthma_adult_acute.yaml"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
SESSION_MIN = 60


def actions_from_spec(items):
    out = []
    for i, a in enumerate(items):
        a = dict(a)
        t = int(round(a.pop("t_min") * 60))
        kind = ActionKind(a.pop("kind"))
        out.append(LearnerAction(tick=t, kind=kind, params=a, seq=i))
    return out


def run_trajectory(spec, name):
    traj = spec["trajectories"][name]
    eng = PatientEngine(spec)
    acts = actions_from_spec(traj["actions"])
    trace, events = eng.run(acts, SESSION_MIN * 60, record_every=30)
    checks = []
    for ex in traj["expect"]:
        row = min(trace, key=lambda r: abs(r["t_min"] - ex["t_min"]))
        v = row[ex["var"]]
        ok = ex["min"] <= v <= ex["max"]
        checks.append({"t_min": ex["t_min"], "var": ex["var"], "value": round(float(v), 2), "min": ex["min"], "max": ex["max"], "pass": ok})
    return trace, events, checks


INTERVENTIONS = {
    "albuterol": [("give_drug", {"drug": "albuterol", "dose": 2.5})],
    "ipratropium": [("give_drug", {"drug": "ipratropium", "dose": 0.5})],
    "epinephrine_im": [("give_drug", {"drug": "epinephrine_im", "dose": 0.3})],
    "magnesium_iv": [("give_drug", {"drug": "magnesium_iv", "dose": 2.0})],
    "methylprednisolone_iv": [("give_drug", {"drug": "methylprednisolone_iv", "dose": 125})],
    "ketamine_iv": [("give_drug", {"drug": "ketamine_iv", "dose": 70})],
    "midazolam_iv": [("give_drug", {"drug": "midazolam_iv", "dose": 2.0})],
    "nitroglycerin_sl": [("give_drug", {"drug": "nitroglycerin_sl", "dose": 0.4})],
    "oxygen_nc_4": [("oxygen", {"device": "nc", "flow_lpm": 4})],
    "oxygen_nrb_15": [("oxygen", {"device": "nrb", "flow_lpm": 15})],
    "cpap_7.5": [("cpap", {"on": True, "pressure_cmh2o": 7.5, "fio2": 1.0})],
    "bvm_safe_8": [("bvm", {"on": True, "rate_per_min": 8})],
    "bvm_fast_20": [("bvm", {"on": True, "rate_per_min": 20})],
    "intubate_vent_8": [("intubate", {"success": True, "vent_rate": 8})],
    "position_supine": [("position", {"position": "supine"})],
}
STAGES = ["S0_moderate", "S1_severe", "S2_life_threatening", "S3_near_arrest", "S4_arrest"]
BRONCHODILATORS = {"albuterol", "ipratropium", "epinephrine_im", "magnesium_iv", "ketamine_iv"}
OXYGEN = {"oxygen_nc_4", "oxygen_nrb_15", "cpap_7.5", "bvm_safe_8", "intubate_vent_8"}


def invariants(trace):
    bad = []
    for r in trace:
        for k in ("O", "F"):
            if not (0.0 <= r[k] <= 1.0): bad.append(f"{k} out of [0,1] at {r['t_min']:.1f}")
        if not (0 <= r["SpO2"] <= 100): bad.append(f"SpO2 out of range at {r['t_min']:.1f}")
        for k in ("HR", "SBP", "RR", "PaCO2", "PaO2"):
            if r[k] != r[k] or r[k] < 0: bad.append(f"{k} NaN/negative at {r['t_min']:.1f}")
        if r["LOC"] < -1e-9 or r["LOC"] > 3 + 1e-9: bad.append(f"LOC out of range at {r['t_min']:.1f}")
    return bad


def causal_checks(name_a, trace_a, trace_ctrl):
    """Directional expectations relative to the untreated control, from the same start time."""
    bad = []
    for ra, rc in zip(trace_a, trace_ctrl):
        if rc["arrest"] or ra["arrest"]:
            break
        if name_a in BRONCHODILATORS and ra["O"] > rc["O"] + 1e-6:
            bad.append(f"{name_a} raised O at {ra['t_min']:.1f}")
        if name_a in OXYGEN and ra["PaO2"] < rc["PaO2"] - 1e-6:
            bad.append(f"{name_a} lowered PaO2 at {ra['t_min']:.1f}")
    return bad


def stage_entry_ticks(spec):
    eng = PatientEngine(spec)
    trace, _ = eng.run([], SESSION_MIN * 60, record_every=1)
    first = {}
    for r in trace:
        first.setdefault(r["stage"], r["tick"])
    return first


def sweep(spec):
    """Every intervention at every reachable stage, and every ordered pair at S1 (5 and 10 min)."""
    stage_ticks = stage_entry_ticks(spec)
    n_runs = 1
    report = {"single_by_stage": [], "ordered_pairs": [], "counts": {}}
    # control traces per stage start (untreated) for causal comparisons
    ctrl_eng = PatientEngine(spec)
    ctrl_trace, _ = ctrl_eng.run([], SESSION_MIN * 60, record_every=10)
    n_runs += 1
    for stage in STAGES:
        if stage not in stage_ticks:
            continue
        t_apply = stage_ticks[stage] + 5
        for name, steps in INTERVENTIONS.items():
            eng = PatientEngine(spec)
            acts = [LearnerAction(tick=t_apply, kind=ActionKind(k), params=p, seq=i) for i, (k, p) in enumerate(steps)]
            trace, _ = eng.run(acts, min(SESSION_MIN * 60, t_apply + 20 * 60), record_every=10)
            n_runs += 1
            inv = invariants(trace)
            caus = causal_checks(name, trace, ctrl_trace[: len(trace)])
            report["single_by_stage"].append({"stage": stage, "intervention": name, "t_apply_s": t_apply,
                                              "invariant_violations": inv, "causal_violations": caus,
                                              "end": {k: round(float(trace[-1][k]), 2) for k in ("O", "F", "PaCO2", "SpO2", "HR", "SBP", "LOC")},
                                              "arrest": bool(trace[-1]["arrest"])})
    names = list(INTERVENTIONS)
    for a, b in itertools.permutations(names, 2):
        eng = PatientEngine(spec)
        acts = [LearnerAction(tick=5 * 60, kind=ActionKind(k), params=p, seq=i) for i, (k, p) in enumerate(INTERVENTIONS[a])]
        acts += [LearnerAction(tick=10 * 60, kind=ActionKind(k), params=p, seq=10 + i) for i, (k, p) in enumerate(INTERVENTIONS[b])]
        trace, _ = eng.run(acts, 35 * 60, record_every=10)
        n_runs += 1
        inv = invariants(trace)
        report["ordered_pairs"].append({"first": a, "second": b, "invariant_violations": inv,
                                        "end": {k: round(float(trace[-1][k]), 2) for k in ("O", "F", "PaCO2", "SpO2", "HR", "SBP", "LOC")},
                                        "arrest": bool(trace[-1]["arrest"])})
    n_int = len(names)
    n_drugs = sum(1 for n in names if INTERVENTIONS[n][0][0] == "give_drug")
    report["counts"] = {
        "interventions": n_int,
        "stages": len(STAGES),
        "intervention_x_stage": n_int * len(STAGES),
        "unordered_pairs": n_int * (n_int - 1) // 2,
        "ordered_pairs": n_int * (n_int - 1),
        "repeat_dosing_cases": sum(spec["drugs"][INTERVENTIONS[n][0][1]["drug"]].get("max_repeats", 0) for n in names if INTERVENTIONS[n][0][0] == "give_drug"),
        "explicit_interaction_rules_in_spec": len(spec.get("interaction_rules", [])),
        "rollouts_executed": n_runs,
    }
    return report


def main():
    spec = load_spec(SPEC)
    summary = {"spec_hash": spec["_hash"], "trajectories": {}}
    total_rollouts = 0
    for name in spec["trajectories"]:
        trace, events, checks = run_trajectory(spec, name)
        total_rollouts += 1
        (OUT / f"{name}_trace.json").write_text(json.dumps(trace))
        summary["trajectories"][name] = {
            "events": [(round(e.tick / 60, 1), e.name) for e in events if e.name not in ("drug_given",)],
            "checks": checks,
            "all_pass": all(c["pass"] for c in checks),
        }
        print(f"== {name}: {'PASS' if summary['trajectories'][name]['all_pass'] else 'FAIL'}")
        for c in checks:
            print(f"   t={c['t_min']:>4} {c['var']:<7} {c['value']:>7} in [{c['min']}, {c['max']}] {'ok' if c['pass'] else 'FAIL'}")
        print("   events:", summary["trajectories"][name]["events"])
    rep = sweep(spec)
    summary["interaction_sweep"] = rep["counts"]
    n_inv = sum(1 for r in rep["single_by_stage"] + rep["ordered_pairs"] if r["invariant_violations"])
    n_caus = sum(1 for r in rep["single_by_stage"] if r["causal_violations"])
    summary["interaction_sweep"]["cases_with_invariant_violations"] = n_inv
    summary["interaction_sweep"]["cases_with_causal_violations"] = n_caus
    (OUT / "interaction_sweep.json").write_text(json.dumps(rep, indent=1))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("== interaction counts:", json.dumps(rep["counts"]))
    print(f"== invariant violations in {n_inv} cases; causal violations in {n_caus} cases")
    for r in rep["single_by_stage"]:
        if r["causal_violations"] or r["invariant_violations"]:
            print("   ", r["stage"], r["intervention"], r["causal_violations"][:2], r["invariant_violations"][:2])
    log_forward_passes("exp2_asthma", total_rollouts + rep["counts"]["rollouts_executed"],
                       "four trajectories + interaction sweep (code control)", "dev")


if __name__ == "__main__":
    main()
