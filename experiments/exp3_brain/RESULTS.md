# exp3_brain results (2026-11-04)

Two frozen versions. v1 (2d18e68cc896fe47) FAILED its held-out set by producing a NaN on the long
untreated case (unbounded post-arrest PaCO2 in the reference engine pushed an input past its declared
range and a gate leaked); that set is burned. v2 (9c9992111e4fdb6f) bounds the post-arrest placeholder
in engine and graph, adds a loud range guard in the shell, and was evaluated on a new held-out set of
3 unseen parameter variants x 6 unseen action sequences. A harness bug (the code engine settled with
default parameters for non-default variants) was found from the first v2 run and fixed as a logged,
data-informed harness change; the buggy run's log is kept. Numbers below are from the corrected rerun.

## Held-out v2 (18 cases, brain vs code engine, same inputs)

| Case (variant / sequence) | Strict max % | Shift-tolerant % | Max event delay | Events missing/extra | Replay | Result |
|---|---|---|---|---|---|---|
| V4_very_poor_responder/S7_magnesium_then_albuterol | 59.3 | 0.07 | 5 | 0/0 | yes | PASS |
| V4_very_poor_responder/S8_ketamine_intubate_vent14 | 0.9 | 0.15 | 3 | 0/0 | yes | PASS |
| V4_very_poor_responder/S9_ntg_then_late_albuterol | 62.2 | 0.13 | 4 | 0/0 | yes | PASS |
| V4_very_poor_responder/S10_early_cpap_plus_albuterol | 12.0 | 11.96 | 5 | 1/1 | yes | FAIL |
| V4_very_poor_responder/S11_supine_nothing_75 | 62.5 | 0.18 | 4 | 0/0 | yes | PASS |
| V4_very_poor_responder/S12_full_bundle_90 | 12.8 | 12.79 | 5 | 0/1 | yes | FAIL |
| V5_good_responder_slow/S7_magnesium_then_albuterol | 0.1 | 0.04 | 4 | 0/0 | yes | PASS |
| V5_good_responder_slow/S8_ketamine_intubate_vent14 | 0.8 | 0.26 | 1 | 0/0 | yes | PASS |
| V5_good_responder_slow/S9_ntg_then_late_albuterol | 0.1 | 0.07 | 4 | 0/0 | yes | PASS |
| V5_good_responder_slow/S10_early_cpap_plus_albuterol | 0.1 | 0.04 | 7 | 0/0 | yes | PASS |
| V5_good_responder_slow/S11_supine_nothing_75 | 60.5 | 0.04 | 6 | 0/0 | yes | PASS |
| V5_good_responder_slow/S12_full_bundle_90 | 0.1 | 0.03 | 3 | 0/0 | yes | PASS |
| V6_fast_progressor/S7_magnesium_then_albuterol | 61.2 | 0.43 | 3 | 0/0 | yes | PASS |
| V6_fast_progressor/S8_ketamine_intubate_vent14 | 2.9 | 1.19 | 2 | 0/0 | yes | PASS |
| V6_fast_progressor/S9_ntg_then_late_albuterol | 60.5 | 0.43 | 3 | 0/0 | yes | PASS |
| V6_fast_progressor/S10_early_cpap_plus_albuterol | 11.5 | 11.50 | 6 | 0/1 | yes | FAIL |
| V6_fast_progressor/S11_supine_nothing_75 | 60.8 | 0.48 | 3 | 0/0 | yes | PASS |
| V6_fast_progressor/S12_full_bundle_90 | 33.3 | 33.33 | 3 | 0/1 | yes | FAIL |

**Summary: 14 of 18 PASS; 4 FAIL; 0 crashes; replay bit-exact on all 18; max event delay 7 s on the
passing cases; mean cost 4.2 ms per tick (brain) vs 0.17 ms (code) on this CPU.** Under the
predefined all-cases rule, exp3_brain_v2 is a **FAIL**.

## What failed, and why it matters
All four failures share one mechanism: late in a long session (36 to 90 min), in a sensitive regime
(a very poor responder re-deteriorating after bronchodilators fade; early CPAP with a wavering
contraindication gate), a sub-percent numerical difference between brain and code moves a steep
threshold crossing by more than 30 s, which then produces a spurious or shifted `silent_chest` event
and, downstream, a second collapse minutes apart. The two patients follow the same story; they do not
follow it on the same clock once the dynamics become sensitive. Three readings:
- For instructors this is tolerable in most cases and not in all: a scripted event that fires 13
  minutes apart in two implementations would be noticed.
- The remedy is not more knots (the residual bias of 0.1% in fatigue did not move when the gas-exchange
  interpolants were tightened 3x) but specification design: events on asymptotic or wavering signals
  need hysteresis and a confirmation window. That change would benefit the code engine too.
- The reference engine, not the brain, remains the source of truth. The brain is a faithful,
  reproducible export of the program, within a tolerance that must be stated.

## Demonstrated
- The entire asthma engine (352-node program) compiles deterministically into a 26-layer ReLU network
  (517K nonzero weights) that reproduces the code engine within a 30 s time shift on 14 of 18 unseen
  variant-by-sequence cases and on all four development paths except one asymptotic event.
- Compiled brains are valid only inside declared input ranges; the shell now enforces that loudly.
- Replay is bit-exact; the same program always yields the same weights.
