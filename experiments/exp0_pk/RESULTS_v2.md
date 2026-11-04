# exp0_pk v2 results (frozen sha256 prefix ad34aa59bc425591; held-out block `heldout_v2`; 2026-11-04)

Same specification as v1. Three logged changes (LEDGER.md, exp0_pk_v2 DC 1-3). Held-out block
`heldout_v2` (3 new variants: EC50 0.1 ultra-potent, EC50 4.0 weak/slow, 20-min effect link; 4 new
schedules: two doses 2 min apart, twenty 0.25-unit doses q3min, one 25-unit dose with 120 min
rollout, five IV boluses q10min) was written into `pk_spec.yaml` and `splits_v2.json` before the run.

| Target | Params | Dev max dE | heldout_v2 max dE | burned v1 held-out (info only) | C_effect rel. error | Violations | Causal failures | Replay bit-exact | Criterion |
|---|---|---|---|---|---|---|---|---|---|
| CODE (control) | n/a | 0 (ref) | 0 | 0 | 0 | 0 | 0 | yes | reference |
| A2 hand-set weights (v2 knots) | 1,010 (EC50-relative knots) | 0.0030 | **0.0030** | 0.0030 | 1.5e-14 | 0 | 0 | yes | **PASS** (<= 0.01) |
| D trained MLP + DAgger x3 | 4,933 | 0.59 | 94 (unseen variants, typical model) / 1.5e17 (fresh per-variant models, unseen schedules) | n/a | diverged | 0 | 4 / 8 | yes | **FAIL** (<= 0.05) |

Wall time per 60-min rollout, this container's CPU, pure Python/numpy: CODE ~0.03 s, A2 ~0.05 s,
D ~0.10 s. Checkpoint: A2 weights 812 to 1,010 float64 values (about 8 KB as .npz).
Forward passes logged: exp0_pk 204, exp0_pk_v2 204 (every rollout, replay included). Design changes:
v1 0, v2 3 (all DATA-INFORMED). Builder runs: 0 (the compile was written by the authoring frontier
model in-session; no local builder has been tested, see experiments/exp1_calibration).

## What this demonstrates (and what it does not)
- **Demonstrated:** an explicit, fully specified linear-kinetics-plus-Emax model with dosing events can
  be compiled by hand into a ~1K-weight ReLU cell that is exact on the hidden state and within 0.003 of
  the code control on the effect readout, across unseen parameter extremes, unseen dosing orders,
  repeated and simultaneous doses, and a 3-hour rollout. The constructions used were only: linear
  recurrence rows, a steep ReLU clip (dose gate), a gated ramp (time since dose), and ReLU interpolants.
- **Demonstrated:** the only failure of the weight target (v1) came from a constant chosen by reasoning
  (knot placement) breaking on a parameter extreme. The fix was a spec-derived rule. This is the ECG
  experiment's "reasoned threshold" lesson reproduced on the first try.
- **Demonstrated:** a conventionally trained next-step network of 5x the parameter count, trained on
  exact trajectories from the same code, did not reach 0.05 effect error on 1-second rollouts in two
  configurations (single-step; DAgger x3). Per-variant models cannot generalise to unseen parameters at
  all. This does not prove training cannot work (parameter-conditioned inputs, multi-step losses,
  structure-preserving parametrisations are untried); it shows the cheap version is not a free lunch.
- **Not demonstrated:** that weights are *better* than code on anything. On every instructor-visible
  criterion in this experiment (accuracy, repeatability, speed) the code control was equal or better and
  was far easier to write and read. The weight target's hypothetical advantages (uniform runtime,
  differentiability for later calibration, parameter blending) were not tested here.
- **Not demonstrated:** that any local builder can produce `pk_wsp.py`. That is `exp1_calibration`.
