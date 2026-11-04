#!/usr/bin/env python3
"""Runs exp0_pk end to end under RULES.md: dev evaluation -> freeze -> held-out evaluation.
Every rollout of a non-code target is a logged forward pass."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))
from mfnn.ledger import freeze, is_frozen, log_forward_passes  # noqa: E402
import pk_code, pk_wsp, pk_trained  # noqa: E402

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("phase", nargs="?", default="all")
_ap.add_argument("--exp", default="exp0_pk_v2"); _ap.add_argument("--heldout", default="heldout_v2")
_ap.add_argument("--burned", default="heldout", help="earlier held-out block, reported but no longer a clean claim")
_ap.add_argument("--splits", default="splits_v2.json"); _ap.add_argument("--dagger", type=int, default=3)
ARGS = _ap.parse_args()
EXP = ARGS.exp
spec = yaml.safe_load((HERE / "pk_spec.yaml").read_text())
splits = json.loads((HERE / ARGS.splits).read_text())
DT = float(spec["dt_s"])
RES = HERE / "results"; RES.mkdir(exist_ok=True)


def cases(split):
    block = ARGS.heldout if split == "heldout" else split
    for vn in splits[block]["variants"]:
        for sn in splits[block]["schedules"]:
            sch = spec["schedules"][block][sn]
            yield vn, sn, spec["variants"][block][vn], [(d[0], d[1], d[2]) for d in sch["doses"]], int(sch["minutes"] * 60 / DT)


def causal_checks(arr, doses, v=None, dt=DT):
    bad = []
    for t, _, _ in doses:
        t = int(t)
        if t + 60 < arr.shape[0] and not arr[t + 60, 2] > arr[t, 2]:
            bad.append(f"no onset after dose at {t}s")
    last = max([int(d[0]) for d in doses], default=-1)
    # v2 fix (DC 1): washout window starts after absorption and effect-site link are >98% complete
    settle_min = 6.0 * (v["ka_half_min"] + v["keo_half_min"]) if v else 15.0
    start = last + int(settle_min * 60)
    if start + 30 * 60 < arr.shape[0]:
        seg = arr[start:start + 30 * 60, 2]
        if np.any(np.diff(seg) > 1e-9):
            bad.append("C_e increased during washout")
    return bad


def violations(arr, doses):
    bad = []
    if np.any(~np.isfinite(arr)): bad.append("non-finite")
    if np.any(arr[:, :3] < -1e-12): bad.append("negative amount")
    for t, _, _ in doses:
        if arr[int(t), 3] != 0.0: bad.append(f"tau not reset at {t}")
    return bad


def evaluate(split, target, make_rollout, count_fp=True):
    rows = []
    for vn, sn, v, doses, n in cases(split):
        ref = pk_code.rollout(v, doses, n, DT)
        t0 = time.perf_counter()
        got = make_rollout(vn, v, doses, n)
        wall = time.perf_counter() - t0
        got2 = make_rollout(vn, v, doses, n)
        if count_fp and target != "code":
            log_forward_passes(EXP, 2, f"{target} {split} {vn}/{sn} (+replay)", split)
        dE = np.abs(got[:, 4] - ref[:, 4]); dC = np.abs(got[:, 2] - ref[:, 2]) / np.maximum(np.abs(ref[:, 2]), 1e-12)
        rows.append({"variant": vn, "schedule": sn, "ticks": n,
                     "max_dE": float(dE.max()), "rmse_E": float(np.sqrt((dE ** 2).mean())),
                     "max_relC": float(dC[ref[:, 2] > 1e-9].max()) if np.any(ref[:, 2] > 1e-9) else 0.0,
                     "max_dHR": float(np.abs(got[:, 5] - ref[:, 5]).max()),
                     "violations": violations(got, doses), "causal": causal_checks(got, doses, v),
                     "replay_bit_exact": bool(np.array_equal(got, got2)), "wall_s": wall})
    return rows


def summarize(rows):
    return {"n_cases": len(rows), "max_dE": max(r["max_dE"] for r in rows), "max_relC": max(r["max_relC"] for r in rows),
            "max_dHR": max(r["max_dHR"] for r in rows), "cases_with_violations": sum(1 for r in rows if r["violations"]),
            "cases_with_causal_failures": sum(1 for r in rows if r["causal"]), "all_replay_exact": all(r["replay_bit_exact"] for r in rows),
            "mean_wall_s_per_rollout": float(np.mean([r["wall_s"] for r in rows]))}


def main():
    phase = ARGS.phase
    out = {}
    nets = {}
    def wsp_roll(vn, v, doses, n):
        if vn not in nets: nets[vn] = pk_wsp.compile_pk(v, DT)
        return pk_wsp.rollout(nets[vn], doses, n, DT)
    # ---------------- DEV ----------------
    if phase in ("dev", "all"):
        out["code_dev"] = summarize(evaluate("dev", "code", lambda vn, v, d, n: pk_code.rollout(v, d, n, DT)))
        rows = evaluate("dev", "wsp", wsp_roll); out["wsp_dev"] = summarize(rows)
        out["wsp_params"] = pk_wsp.compile_pk(spec["variants"]["dev"]["typical"], DT).n_params
        out["wsp_checkpoint_bytes"] = len(pk_wsp.compile_pk(spec["variants"]["dev"]["typical"], DT).to_bytes())
        print("DEV wsp:", json.dumps(out["wsp_dev"]))
        # trained: one model per variant family? No: one model conditioned only on state+inputs cannot know ka/ke.
        # The spec says parameters vary per patient, so D must take parameters as inputs or be per-variant.
        # Smallest credible version: per-variant model trained on that variant's dev schedules.
        traces_by_variant = {}
        for vn, sn, v, doses, n in cases("dev"):
            by_t = {}
            for t, route, amt in doses:
                d = by_t.setdefault(int(t), [0.0, 0.0]); d[0 if route == "depot" else 1] += float(amt)
            traces_by_variant.setdefault(vn, []).append((pk_code.rollout(v, doses, n, DT), by_t))
        trained = {}
        t0 = time.perf_counter()
        for vn, traces in traces_by_variant.items():
            v = spec["variants"]["dev"][vn]
            scheds = [(by_t, arr.shape[0]) for arr, by_t in traces]
            if ARGS.dagger > 0:
                trained[vn], losses = pk_trained.train_dagger(v, scheds, lambda s_, u_: pk_code.step(v, s_, u_, DT), hidden=64, steps=4000, batch=1024, seed=0, rounds=ARGS.dagger)
            else:
                trained[vn], losses = pk_trained.train(traces, hidden=64, steps=4000, batch=1024, seed=0)
            print(f"trained {vn}: final loss {losses[-1]:.3e}, params {trained[vn].n_params}, dagger rounds {ARGS.dagger}")
        out["trained_train_wall_s"] = time.perf_counter() - t0
        out["trained_params"] = trained[vn].n_params
        rows = evaluate("dev", "trained", lambda vn, v, d, n: pk_trained.rollout(trained[vn], d, n, DT)); out["trained_dev"] = summarize(rows)
        print("DEV trained:", json.dumps(out["trained_dev"]))
        # freeze WSP (hash of the compile function's output on every dev variant) and trained weights
        h = __import__("hashlib").sha256(b"".join(pk_wsp.compile_pk(spec["variants"]["dev"][vn], DT).to_bytes() for vn in splits["dev"]["variants"])).hexdigest()
        np.savez(RES / f"trained_weights_{EXP}.npz", **{f"{vn}_W{i}": W for vn, m in trained.items() for i, W in enumerate(m.W)}, **{f"{vn}_b{i}": b for vn, m in trained.items() for i, b in enumerate(m.b)})
        if not is_frozen(EXP):
            freeze(EXP, h[:16], "wsp compile_pk frozen on dev variants; trained weights saved to results/trained_weights.npz")
        (RES / f"dev_results_{EXP}.json").write_text(json.dumps(out, indent=1))
        # keep trained nets for the held-out phase in the same process
        main.trained = trained
    # ---------------- HELD-OUT ----------------
    if phase in ("heldout", "all"):
        assert is_frozen(EXP), "freeze before held-out (RULES R4)"
        trained = getattr(main, "trained", None)
        ho = {}
        ho["code_heldout"] = summarize(evaluate("heldout", "code", lambda vn, v, d, n: pk_code.rollout(v, d, n, DT)))
        rows = evaluate("heldout", "wsp", wsp_roll); ho["wsp_heldout"] = summarize(rows); ho["wsp_heldout_rows"] = rows
        print("HELD-OUT wsp:", json.dumps(ho["wsp_heldout"]))
        if trained is not None:
            # held-out variants have NEW parameters; a per-variant trained model cannot exist for them without
            # training on their data. The honest D evaluation on unseen variants is therefore: apply the
            # 'typical' model to the extreme variants (parameter generalisation = fail by construction) AND
            # train fresh per-variant models on the held-out variants' DEV-style schedules only, then test on
            # the held-out schedules (schedule generalisation). Both are reported.
            rows_t = evaluate("heldout", "trained_typical_model_on_unseen_variants", lambda vn, v, d, n: pk_trained.rollout(trained["typical"], d, n, DT))
            ho["trained_param_generalisation"] = summarize(rows_t)
            fresh = {}
            for vn in splits[ARGS.heldout]["variants"]:
                v = spec["variants"][ARGS.heldout][vn]
                traces = []
                for sn in splits["dev"]["schedules"]:
                    sch = spec["schedules"]["dev"][sn]; doses = [(d[0], d[1], d[2]) for d in sch["doses"]]; n = int(sch["minutes"] * 60 / DT)
                    by_t = {}
                    for t, route, amt in doses:
                        d = by_t.setdefault(int(t), [0.0, 0.0]); d[0 if route == "depot" else 1] += float(amt)
                    traces.append((pk_code.rollout(v, doses, n, DT), by_t))
                scheds = [(by_t, arr.shape[0]) for arr, by_t in traces]
                fresh[vn], _ = pk_trained.train_dagger(v, scheds, lambda s_, u_, v=v: pk_code.step(v, s_, u_, DT), hidden=64, steps=4000, batch=1024, seed=0, rounds=ARGS.dagger) if ARGS.dagger > 0 else pk_trained.train(traces, hidden=64, steps=4000, batch=1024, seed=0)
            rows_s = evaluate("heldout", "trained_fresh_variant_models_on_unseen_schedules", lambda vn, v, d, n: pk_trained.rollout(fresh[vn], d, n, DT))
            ho["trained_schedule_generalisation"] = summarize(rows_s); ho["trained_schedule_rows"] = rows_s
            print("HELD-OUT trained (param gen):", json.dumps(ho["trained_param_generalisation"]))
            print("HELD-OUT trained (schedule gen):", json.dumps(ho["trained_schedule_generalisation"]))
        if ARGS.burned and ARGS.burned != ARGS.heldout and ARGS.burned in splits:
            saved = ARGS.heldout; ARGS.heldout = ARGS.burned
            rows_b = evaluate("heldout", f"wsp_on_BURNED_{ARGS.burned}", wsp_roll); ho[f"wsp_on_burned_{ARGS.burned}"] = summarize(rows_b)
            ARGS.heldout = saved
            print(f"BURNED {ARGS.burned} (informational only) wsp:", json.dumps(ho[f"wsp_on_burned_{ARGS.burned}"]))
        (RES / f"heldout_results_{EXP}.json").write_text(json.dumps(ho, indent=1))


if __name__ == "__main__":
    main()
