# exp0_pk v1 results (frozen weights sha256 prefix e3138aef1cb52bcd; 2026-11-04)

Read with `PROTOCOL.md`. Numbers from `results/dev_results.json` and `results/heldout_results.json`.

| Target | Params | Dev max dE | Held-out max dE | C_effect rel. error | Violations | Replay bit-exact | Predefined criterion |
|---|---|---|---|---|---|---|---|
| CODE (control) | n/a | 0 (reference) | 0 | 0 | 0 | yes | reference |
| A2 hand-set weights | 812 | 0.0029 | **0.0120** | 1.2e-14 | 0 | yes | **FAIL** (limit 0.01) |
| D trained MLP, per-variant | 4,933 | **5,397** (diverged) | 9e8 (unseen variants) / 933 (unseen schedules) | diverged | 0 | yes | **FAIL** (limit 0.05) |

**What failed and why (demonstrated):**
- A2 failed on exactly one held-out variant, `extreme_fast` (EC50 = 0.2), on every schedule with a
  dose: max dE 0.0114 to 0.0120. The Emax readout is a ReLU interpolant whose knots were chosen by
  reasoning for EC50 near 1 (`KNOTS = [0, 0.05, 0.1, 0.2, ...]`). For EC50 = 0.2 the first chord
  [0, 0.05] has error 0.011. This is the brief's lesson in miniature: a constant set by reasoning
  alone (knot placement) was the weakest point, and it broke on a parameter extreme. The fix is
  derivable from the specification (place knots relative to EC50), but because it was motivated by a
  held-out failure it is a DATA-INFORMED design change and the v1 held-out set is now burned.
- The linear recurrence (depot, central, effect site, time-since-dose) was exact to 1e-14 on
  every case, including the 180 min rollout and the 10-dose overdose. Weight compilation of the
  linear part is not where the risk is.
- D (conventionally trained next-step MLP, 4,933 params, Adam, 4,000 steps on single-step
  transitions) reached a training loss of 5e-4 and still diverged in autoregressive rollout on the
  DEV schedules it was trained on (compounding error; the model drifts from the zero state even with
  no doses). A per-variant model cannot generalise to unseen parameter sets at all by construction.
- Harness defect found after seeing results: the "no increase during washout" causal check started
  15 min after the last dose, which is before absorption completes for slow variants, so it flagged
  CODE itself. It is corrected in v2 (window starts after 6 absorption + 6 link half-times).

**Decision per PROTOCOL.md:** A2 did not pass v1. The weight track is not killed (the failure is a
knot-placement rule, not a capability limit) but the claim "as exact as code" is unproven until v2
passes on a NEW held-out set. D in this configuration is not a viable target; one configuration change
(DAgger-style data aggregation) is allowed in v2 under the revision limit.
