#!/usr/bin/env python3
"""exp3_brain runner: dev trajectories (code vs brain) -> freeze -> held-out variants x sequences.
Usage: run_exp3.py dev | heldout | all"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "experiments/exp2_asthma"))
from mfnn.actions import ActionKind, LearnerAction
from mfnn.engine import PatientEngine
from mfnn.ledger import freeze, is_frozen, log_forward_passes
from mfnn.spec import load_spec
from brain_runner import BrainEngine
from run_trajectories import actions_from_spec

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("phase", nargs="?", default="all"); _ap.add_argument("--exp", default="exp3_brain_v2"); _ap.add_argument("--heldout", default="v2")
ARGS = _ap.parse_args()
EXP = ARGS.exp; RES = HERE / "results"; RES.mkdir(exist_ok=True)
RANGES = {"SpO2": 100, "HR": 200, "RR": 60, "SBP": 200, "PaCO2": 150, "EtCO2": 150, "LOC": 3, "O": 1, "F": 1}
HELDOUT_VARIANTS = {
    "V1_poorly_responsive": {"beta_agonist_responsiveness": 0.3, "progression_rate": 0.02, "fatigue_rate": 0.012},
    "V2_brittle_fast": {"beta_agonist_responsiveness": 0.9, "progression_rate": 0.05, "fatigue_rate": 0.03},
    "V3_slow_tired": {"beta_agonist_responsiveness": 0.6, "progression_rate": 0.01, "fatigue_rate": 0.025},
}
HELDOUT_VARIANTS_V2 = {
    "V4_very_poor_responder": {"beta_agonist_responsiveness": 0.2, "progression_rate": 0.03, "fatigue_rate": 0.02},
    "V5_good_responder_slow": {"beta_agonist_responsiveness": 1.0, "progression_rate": 0.015, "fatigue_rate": 0.008},
    "V6_fast_progressor": {"beta_agonist_responsiveness": 0.5, "progression_rate": 0.06, "fatigue_rate": 0.015},
}
def A(t_min, kind, **p): return {"t_min": t_min, "kind": kind, **p}
HELDOUT_SEQUENCES_V2 = {
    "S7_magnesium_then_albuterol": (60, [A(4, "give_drug", drug="magnesium_iv", dose=2.0), A(6, "give_drug", drug="albuterol", dose=2.5)]),
    "S8_ketamine_intubate_vent14": (60, [A(10, "give_drug", drug="ketamine_iv", dose=70), A(12, "intubate", success=True, vent_rate=14)]),
    "S9_ntg_then_late_albuterol": (60, [A(3, "give_drug", drug="nitroglycerin_sl", dose=0.4), A(20, "give_drug", drug="albuterol", dose=2.5)]),
    "S10_early_cpap_plus_albuterol": (60, [A(2, "cpap", on=True, pressure_cmh2o=7.5, fio2=1.0), A(3, "give_drug", drug="albuterol", dose=2.5)]),
    "S11_supine_nothing_75": (75, [A(1, "position", position="supine")]),
    "S12_full_bundle_90": (90, [A(1, "oxygen", device="nrb", flow_lpm=15), A(2, "give_drug", drug="albuterol", dose=2.5), A(3, "give_drug", drug="ipratropium", dose=0.5), A(15, "give_drug", drug="methylprednisolone_iv", dose=125), A(22, "give_drug", drug="albuterol", dose=2.5), A(23, "give_drug", drug="ipratropium", dose=0.5), A(42, "give_drug", drug="albuterol", dose=2.5)]),
}
HELDOUT_SEQUENCES = {
    "S1_ipra_first": (60, [A(2, "oxygen", device="nrb", flow_lpm=15), A(2, "give_drug", drug="ipratropium", dose=0.5), A(4, "give_drug", drug="albuterol", dose=2.5)]),
    "S2_epi_first": (60, [A(3, "give_drug", drug="epinephrine_im", dose=0.3), A(8, "give_drug", drug="albuterol", dose=2.5)]),
    "S3_midaz_then_cpap": (60, [A(6, "give_drug", drug="midazolam_iv", dose=2.0), A(15, "cpap", on=True, pressure_cmh2o=7.5, fio2=1.0)]),
    "S4_bvm8_early": (60, [A(5, "bvm", on=True, rate_per_min=8), A(6, "give_drug", drug="albuterol", dose=2.5)]),
    "S5_two_albuterol_3min": (90, [A(2, "oxygen", device="nrb", flow_lpm=15), A(2, "give_drug", drug="albuterol", dose=2.5), A(5, "give_drug", drug="albuterol", dose=2.5), A(5, "give_drug", drug="ipratropium", dose=0.5)]),
    "S6_nothing_90": (90, []),
}


def rollout(engine, acts, minutes):
    trace, events = engine.run(acts, minutes * 60, record_every=1)
    return trace, [(e.tick, e.name) for e in events if e.name != "drug_given"]


def compare(trace_c, trace_b, ev_c, ev_b):
    errs = {}
    n = min(len(trace_c), len(trace_b))
    mask = np.ones(n, dtype=bool)                      # outside +-30 s of any code event (discontinuities)
    for t, _ in ev_c:
        mask[max(0, t - 30):min(n, t + 31)] = False
    for k, rng in RANGES.items():
        a = np.array([r[k] for r in trace_c])[:n]; b = np.array([r[k] for r in trace_b])[:n]
        d = np.abs(a - b)
        # shift-tolerant: for each tick the smallest difference to the other trace within +-30 s
        W = 30; best = d.copy()
        for sh in range(-W, W + 1):
            if sh == 0: continue
            lo, hi = max(0, -sh), min(n, n - sh)
            diff = np.abs(a[lo:hi] - b[lo + sh:hi + sh]); best[lo:hi] = np.minimum(best[lo:hi], diff)
        errs[k] = {"max_abs": float(d.max()), "max_pct_range": float(100 * d.max() / rng),
                   "max_pct_range_shift30s": float(100 * best.max() / rng),
                   "max_pct_range_away_from_events": float(100 * (d[mask].max() if mask.any() else 0.0) / rng),
                   "t_of_max_s": int(d.argmax()),
                   "at_10min": float(abs(a[min(599, n - 1)] - b[min(599, n - 1)])), "at_end": float(abs(a[-1] - b[-1]))}
    # events: match by name within 30 s
    unmatched_code, unmatched_brain, timing = [], [], []
    bl = list(ev_b)
    for t, name in ev_c:
        cand = [(abs(t2 - t), j) for j, (t2, n2) in enumerate(bl) if n2 == name]
        if cand and min(cand)[0] <= 30:
            d, j = min(cand); timing.append((name, d)); bl.pop(j)
        else: unmatched_code.append((t, name))
    unmatched_brain = bl
    stage_c = [r["stage"] for r in trace_c]; stage_b = [r["stage"] for r in trace_b]
    stage_mismatch_s = sum(1 for x, y in zip(stage_c, stage_b) if x != y)
    worst_pct = max(v["max_pct_range"] for v in errs.values())
    worst_away = max(v["max_pct_range_away_from_events"] for v in errs.values())
    worst_shift = max(v["max_pct_range_shift30s"] for v in errs.values())
    worst_var = max(errs, key=lambda k: errs[k]["max_pct_range"])
    return {"errors": errs, "worst_pct_of_range": worst_pct, "worst_pct_away_from_events": worst_away, "worst_pct_shift30s": worst_shift, "worst_var": worst_var,
            "t_of_worst_s": errs[worst_var]["t_of_max_s"], "events_matched": timing, "events_missing_in_brain": unmatched_code,
            "events_extra_in_brain": unmatched_brain, "max_event_delay_s": max((d for _, d in timing), default=0),
            "stage_mismatch_seconds": stage_mismatch_s,
            "pass_2pct_strict": worst_pct <= 2.0 and not unmatched_code and not unmatched_brain,
            "pass_2pct": worst_shift <= 2.0 and not unmatched_code and not unmatched_brain and max((d for _, d in timing), default=0) <= 30}


def run_cases(spec, cases, split):
    out = {}; n_fp = 0
    for name, (variant, overrides, minutes, action_items) in cases.items():
        acts = actions_from_spec(action_items)
        code = PatientEngine(spec); code.p.update(overrides or {})
        brain = BrainEngine(spec, port_overrides=overrides)
        t0 = time.perf_counter(); tc, ec = rollout(code, acts, minutes); t_code = time.perf_counter() - t0
        try:
            t0 = time.perf_counter(); tb, eb = rollout(brain, acts, minutes); t_brain = time.perf_counter() - t0
            brain2 = BrainEngine(spec, net=brain.net, manifest=brain.manifest, port_overrides=overrides); tb2, eb2 = rollout(brain2, acts, minutes)
        except Exception as e:                      # a crash is a FAIL for that case, recorded, and the run continues
            n_fp += 1
            out[name] = {"variant": variant, "minutes": minutes, "crash": f"{e.__class__.__name__}: {e}", "pass_2pct": False, "pass_2pct_strict": False}
            print(f"{name:<28} CRASH {e}"); continue
        replay = brain.state.digest() == brain2.state.digest() and eb == eb2
        n_fp += 2
        res = compare(tc, tb, ec, eb); res.update({"variant": variant, "minutes": minutes, "replay_bit_exact": replay,
                                                   "ms_per_tick_code": 1000 * t_code / (minutes * 60), "ms_per_tick_brain": 1000 * t_brain / (minutes * 60)})
        out[name] = res
        print(f"{name:<28} worst {res['worst_pct_of_range']:.3f}% ({res['worst_var']} @ {res['t_of_worst_s']}s) | away-from-events {res['worst_pct_away_from_events']:.3f}% | shift-30s {res['worst_pct_shift30s']:.3f}% | max ev delay {res['max_event_delay_s']}s | missing {len(res['events_missing_in_brain'])} extra {len(res['events_extra_in_brain'])} | stage mismatch {res['stage_mismatch_seconds']}s | replay {replay} | {res['ms_per_tick_brain']:.2f} ms/tick")
    log_forward_passes(EXP, n_fp, f"{split}: brain rollouts (+replay) for {len(cases)} cases", split)
    return out


def main():
    phase = ARGS.phase
    spec = load_spec(ROOT / "spec/examples/asthma_adult_acute.yaml")
    HV, HS = (HELDOUT_VARIANTS_V2, HELDOUT_SEQUENCES_V2) if ARGS.heldout == "v2" else (HELDOUT_VARIANTS, HELDOUT_SEQUENCES)
    if phase in ("dev", "all"):
        cases = {name: ("spec_default", None, 60, spec["trajectories"][name]["actions"]) for name in spec["trajectories"]}
        dev = run_cases(spec, cases, "dev")
        b = BrainEngine(spec); h = b.net.sha256()[:16]
        dev["_brain"] = {"sha256": h, "widths": b.manifest["widths"], "dense_params": b.manifest["dense_params"], "nonzero_params": b.manifest["nonzero_params"], "n_layers": b.manifest["n_layers"]}
        (RES / f"dev_results_{EXP}.json").write_text(json.dumps(dev, indent=1, default=str))
        print("brain", dev["_brain"])
    if phase == "freeze":
        b = BrainEngine(spec); freeze(EXP, b.net.sha256()[:16], "compiled asthma brain frozen (graph + compiler)"); print("frozen", b.net.sha256()[:16])
    if phase in ("heldout",):
        assert is_frozen(EXP), "freeze first (RULES R4)"
        b = BrainEngine(spec); assert b.net.sha256()[:16] == is_frozen(EXP), "brain changed since freeze"
        cases = {f"{v}/{s}": (v, ov, mins, items) for v, ov in HV.items() for s, (mins, items) in HS.items()}
        ho = run_cases(spec, cases, "heldout")
        (RES / f"heldout_results_{EXP}.json").write_text(json.dumps(ho, indent=1, default=str))
        n_pass = sum(1 for r in ho.values() if r["pass_2pct"]); print(f"HELD-OUT: {n_pass}/{len(ho)} cases pass the 2%-of-range + event criteria")


if __name__ == "__main__":
    main()
