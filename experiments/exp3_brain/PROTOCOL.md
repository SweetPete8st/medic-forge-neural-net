# exp3_brain: the whole asthma patient as a recurrent ReLU network, built by a deterministic compiler

Written BEFORE the compiler was applied to the asthma engine (RULES.md R3, E1, E3).

## Hypothesis under test (the user's "patient brain" idea, strongest workable form)
A frontier model hand-writing weights does not scale to a 15-state-variable physiology with ~20
products and ~25 nonlinear functions (exp0 took one correction for 4 state variables). The scalable
form is: the builder writes the physiology as an explicit expression program (the DSL in
`src/mfnn/wsp/compiler.py`), and a deterministic compiler emits the ReLU weights with per-node error
bounds. Then "designing a patient brain" means writing or editing that program, and every brain is
reproducible by construction (same program, same weights, byte for byte). The test is whether such a
compiled brain reproduces the code engine within instructor-visible tolerance over a full session,
and what it costs (width, parameters, step time, error growth).

## Targets
- CODE: `src/mfnn/engine.py` (reference by definition).
- BRAIN: `experiments/exp3_brain/asthma_graph.py` (the engine's per-tick step written in the DSL)
  compiled by `mfnn.wsp.compiler` into one ReLUNet applied recurrently. Device logic (FiO2 table,
  assisted-rate selection, monitors) and action parsing stay in software and enter the net as inputs.
  A parameter port exposes three patient-specific knobs as inputs (beta_agonist_responsiveness,
  progression_rate, fatigue_rate) so one compiled brain serves parameter variants and can be
  gradient-calibrated (exp4).

## Development set (coherence checks already run on CODE in exp2)
The four scripted trajectories and the 287-case interaction sweep, from
`spec/examples/asthma_adult_acute.yaml`.

## Held-out set (defined now; CODE is run on it only after the BRAIN is frozen)
Patient variants (parameter port values, everything else as in the spec):
- V1 poorly_responsive: beta_agonist_responsiveness 0.3, progression_rate 0.02, fatigue_rate 0.012
- V2 brittle_fast: responsiveness 0.9, progression_rate 0.05, fatigue_rate 0.03
- V3 slow_tired: responsiveness 0.6, progression_rate 0.01, fatigue_rate 0.025
Action sequences (60 min unless stated):
- S1 ipratropium at 2 min, albuterol at 4 min, NRB at 2 min
- S2 epinephrine IM at 3 min first, albuterol at 8 min
- S3 midazolam at 6 min, then CPAP at 15 min (LOC may be P: contraindication path)
- S4 BVM at 8/min from 5 min, albuterol at 6 min
- S5 albuterol at 2 and 5 min (3 min apart), ipratropium at 5 min, NRB at 2 min, 90 min
- S6 nothing for 90 min (long untreated rollout; arrest path)
Total: 18 held-out rollouts; plus replay of every rollout.

## Metrics (BRAIN vs CODE, same inputs)
- max absolute error per observable over the rollout, as % of the observable's clinical range:
  SpO2 (range 100), HR (200), RR (60), SBP (200), PaCO2 (150), EtCO2 (150), LOC (3), O (1), F (1)
- event timing: every event the code emits must be emitted by the brain within 30 s, and no extra events
- drift: error at 60 min vs at 10 min on the untreated path
- state violations, replay bit-exactness
- cost: hidden units per layer, dense parameter count, nonzero parameter count, ms per tick (numpy)
- ledger: forward passes (rollouts), design changes to the compiler or the graph after first rollout

## Pass / fail (predefined)
PASS: every observable within 2% of range on every held-out rollout; all events within 30 s; no extra
events; replay bit-exact; no violations; at most 3 logged design changes after the first rollout.
FAIL otherwise. A FAIL on accuracy with error concentrated in identifiable nodes is reported with the
node names (the compiler's manifest makes this possible; a hand-written net would not).

## What result would change the recommendation
- PASS: "patient brain as a compiled network" is a viable runtime format, reproducible by construction;
  the remaining question is whether it buys anything over code (exp4 calibration test decides).
- FAIL by accuracy: products/interpolants are the bottleneck; report the width needed to pass and
  whether it fits a 1 s tick on a laptop CPU.
- FAIL by event timing only: the brain is usable for vitals but thresholds need exact gates.
