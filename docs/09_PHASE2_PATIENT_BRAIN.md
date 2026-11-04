# Phase 2: the patient as a custom neural network, pushed as far as the evidence allows

Date: 2026-11-04 (same day, second session segment). Labels as in the main package: **demonstrated**,
**plausible extension**, **speculation**. Numbers come from code in this repository run in the authoring
container (4 CPU cores, no GPU) unless stated.

## 0. The idea, restated in its strongest workable form

The user's framing: stop filling in boxes and flowcharts; design the patient's "brain" as a custom
neural network, written by a frontier model, reliably and reproducibly. Phase 1 showed hand-written
weights work for a 4-variable kinetics model but need corrections at extremes, and that the fair
comparison against code gave weights no accuracy advantage.

The strongest version of the idea that survives those results is not "the model emits numbers". It is:

> The builder writes the physiology as an explicit program. A deterministic compiler turns the program
> into ReLU weights with per-node error bounds and a manifest naming every hidden unit. The network IS
> the patient brain: one tensor artefact per patient, reproducible byte for byte from the program, with a
> parameter port so instructor corrections can be applied by gradient descent.

That is the lineage of RASP and Tracr (compile explicit programs into transformer weights) applied to
continuous-state physiology instead of sequence processing. This session built it and tested it.

## 1. What was built (all demonstrated, in `src/mfnn/wsp/compiler.py` and `experiments/exp3_brain/`)

- **Compiler.** A scalar computation graph (affine, relu, max, min, clip, gate, step, piecewise-linear
  `fn`, product `mul` via squaring interpolants, positive products via logs) is laid out into a layered
  ReLU network. Values that can be negative are carried as canonical pairs (relu(v), relu(-v)).
  Interval arithmetic sizes every knot set and gate automatically. Four unit tests include the Phase 1
  drug-kinetics cell compiled through the compiler and matched to the code control (state exact to
  1e-9, effect within 0.01).
- **The asthma brain.** The whole code engine step (13 physiology state variables, 24 drug compartments,
  8 dose inputs, 6 device inputs, 3 parameter-port inputs; 49 outputs including observables) written as
  a 352-node program and compiled: 26 ReLU layers, widths up to 1,859, 9.2M dense / 517K nonzero
  parameters, 3 to 4 ms per 1-second tick in pure numpy on this container when the CPU is not shared
  (35 ms when two other jobs share the four cores), 0.2 s to compile.
- **The shell.** The same software shell as the code engine (actions, devices, FiO2 table, monitors,
  observation gating, events, stages, replay) with the physiology step replaced by the network.
- **Torch export.** The compiled brain exported to torch reproduces the numpy rollout to 7e-10, and the
  rollout is differentiable with respect to the parameter port (used in exp4).

## 2. Does the compiled brain reproduce the patient? (exp3_brain)

Per-tick check on 400 random states (physiology, drugs, devices, arrest): every state variable within
0.01 mmHg/bpm/breaths of the code step (O 3e-6, F 2e-4, LOC 8e-5); no arrest-flag, speech or
silent-chest mismatches after two logged fixes (a speech-level arithmetic error of mine, and the
arrest reporting convention of the shell).

Development trajectories (the four scripted paths), brain vs code. v1 numbers were taken after two
logged fixes and one knot tightening; v2 (frozen 9c9992111e4fdb6f) adds a bound on the post-arrest
PaCO2 placeholder in both engine and graph and a loud range guard in the shell (see the v1 held-out
failure below). Strict = max over every tick; shift-tolerant = per tick, the smallest difference to
the other trace within +-30 s.

| Path | Strict max (% of range) | Away from events | Shift-tolerant (30 s) | Max event delay | Events missing/extra |
|---|---|---|---|---|---|
| untreated | 61.2 (EtCO2 at the arrest instant) | 3.5 | **0.02** | 5 s | 0 / 0 |
| timely treatment | 0.08 | 0.08 | **0.04** | 4 s | 0 / 0 |
| delayed treatment | 2.15 (LOC) | 2.15 | **0.20** | 3 s | 0 / 0 |
| inappropriate | 4.5 (LOC) | 1.1 | **3.6** | 4 s | 1 / 1 (loc_drop 58 s apart) |

Reading: the brain follows the same trajectory as the code within a 30-second time shift on three
paths. The untreated patient arrests 5 s later than the code's (fatigue runs 0.1% low through the
compile chain, a systematic but clinically immaterial bias; tightening the interpolants did not move
it). The one real miss is an event on a signal that approaches its threshold asymptotically (LOC
relaxing toward 2.0 under sedation), where a 1e-3 difference becomes a 58 s timing difference. That is
a specification design issue (events on asymptotic approaches need hysteresis), not a compile error,
and it would bite any two implementations.

**v1 held-out: FAIL, and a finding.** The first frozen brain (2d18e68cc896fe47) produced a NaN on
the long untreated held-out case at 80 minutes. Cause: after arrest the code engine raised PaCO2 by
0.05 mmHg/s without bound (251 mmHg), the brain's PaCO2 input left its declared range [10, 200], and
the arrest-blend gate, exact only inside its range, leaked; the recurrence then grew geometrically.
Two lessons went into the ledger: the reference engine had a physiologically meaningless unbounded
tail, and a compiled brain is valid only inside its declared ranges, so the shell now refuses loudly
when any input leaves them. The v1 held-out set is burned; v2 uses three new variants and six new
sequences defined before the run.

**Held-out v2 (frozen 9c9992111e4fdb6f; 3 unseen variants x 6 unseen sequences; harness initial-condition
bug fixed after the first run, logged): 14 of 18 PASS on the shift-tolerant 2% + 30 s event rule, 4 FAIL,
0 crashes, replay bit-exact on all 18, max event delay 7 s on passing cases. FAIL under the all-cases
rule.** All four failures are late-session second collapses or ventilation dips in sensitive regimes
where a sub-percent difference shifts a steep threshold crossing by more than 30 s and spawns a
spurious `silent_chest` event. Full table and reading: `experiments/exp3_brain/RESULTS.md`.

## 3. Can frontier models reliably and reproducibly write these networks? (exp1, Claude family)

Six isolated attempts at the Phase 1 drug-kinetics compile, one per fresh context, across four
Claude tiers (Fable, Opus, Sonnet x2, Haiku x2). Full table and caveats:
`experiments/exp1_calibration/RESULTS_claude_family.md`; each attempt's build script and construction
notes are preserved under `claude_family_attempts/`.

- **Reliability on the development criterion: 6 of 6, first build, 4 Python runs each.** Even Haiku
  produced a correct, documented construction (two hidden layers, exact linear kinetics, interpolant
  readouts, gated time-since-dose).
- **Robustness to held-out extremes: 2 of 6 fully (Opus; Haiku's second attempt).** Both derived knot
  placement from an explicit chord-error bound out to the largest possible concentration. Fable
  (my own sibling) failed 4 of 108 cases by extrapolating beyond its last knot; both Sonnet attempts
  and the first Haiku placed knots only as far as the development schedules needed and failed on the
  overdose and 25-unit cases; the first Haiku's step chooser refused to build for the most potent
  variant. The failure mode is identical to my own Phase 1 v1 failure, and it did not track model
  tier.
- **What this says.** At this scale, "writing the network" is a solved task for the whole family; the
  variance is in whether the builder reasons about ranges it was not shown. That is exactly what a
  deterministic compiler removes: in Section 1 the knot ranges come from interval arithmetic, not from
  the builder's judgement. Caveats: same model family as the task author; hints in the brief; builders
  could read the scorer; isolation self-reported; nothing here is about open-weight local builders.

## 4. Does a brain buy anything the code does not? (exp4_calibrate)

Task (PROTOCOL.md): a hidden instructor patient = the code engine with port values
responsiveness 0.45, progression 0.035, fatigue 0.018; two corrected observable traces
(SpO2, HR, RR, PaCO2 every 30 s over the timely and delayed scripts, 20-minute horizon after a logged
budget reduction); recover the three values from the spec defaults (0.8, 0.02, 0.012).

| Method | Rollouts | Wall | Final loss | Relative error (responsiveness / progression / fatigue) | Recovered values |
|---|---|---|---|---|---|
| CODE, Nelder-Mead (black box) | 160 | 51 s | 3.53e-04 | 66.1% / 16.3% / 23.1% | 0.748 / 0.0407 / 0.0139 |
| CODE, finite-difference Adam (same schedule as brain) | 122 | 26 s | 1.45e-05 | 5.8% / 5.4% / 11.0% | 0.424 / 0.0331 / 0.0200 |
| BRAIN, autograd Adam (torch export) | 30 | 3636 s | 2.35e-03 | 5.3% / 1.5% / 16.6% | 0.426 / 0.0345 / 0.0210 |

Brain autograd best intermediate step: step 10, loss 2.42e-05. Brain wall time was measured
while two other jobs shared the four CPU cores; uncontended, one Adam step took about 15 s (steps 11-14),
so an uncontended run would be about 4 minutes, still 10x the finite-difference code run.

## Verdict (demonstrated)
- Gradient descent recovers instructor corrections far better than Nelder-Mead at this budget (both
  gradient methods within 1-17% on every parameter; Nelder-Mead 16-66%).
- **Differentiability of the brain is not what delivered that.** Finite differences on the code engine
  match the brain's accuracy with 4x the rollouts (122 vs 30) but 100x less wall time, because the code engine
  steps in 0.3 ms and the brain in 3-35 ms. With three parameters, finite differences are cheap.
- The brain's advantage would appear only when many parameters are calibrated at once (dozens of spec
  constants, or hidden weights), where finite differences scale linearly in cost and autograd does
  not. That is a plausible extension, untested here, and it is also achievable by writing the code
  engine in an autograd framework without any weight compilation.

Decision per PROTOCOL.md: BRAIN-GRAD did not beat CODE on rollouts or wall time at equal accuracy.
Differentiability is not an instructor-visible advantage at this scale.

## 5. What this means for Medic Forge (assessment)

- **Demonstrated:** a reviewable physiology program can be compiled deterministically into a patient
  network that reproduces the reference engine within a 30 s time shift over full sessions in 14 of 18
  unseen cases, with bit-exact replay and a manifest that names every hidden unit; the 4 misses are
  sensitive late-session threshold crossings, a specification-design problem (hysteresis) rather than a
  compile error, but a real limit on treating the brain as the reference. "Designing the brain" becomes writing
  or editing the program and the specification; the weights are a build artefact, like a binary.
- **Demonstrated:** the frontier-model-writes-weights route works reliably on small, fully specified
  pieces and fails on unseen ranges unless the builder reasons about bounds. It does not scale to the
  whole engine by hand: the asthma program has 351 nodes and the compiler emits 517K nonzero weights;
  nobody should hand-write that.
- **Still true from Phase 1:** nothing here makes the network more accurate, faster or easier to
  review than the code it was compiled from. The code engine runs the same patient at 0.3 ms per
  tick; the brain needs 3 to 4 ms in numpy. The brain's distinct claims were the uniform tensor
  runtime and differentiability. **exp4 tested differentiability and it did not win:** gradient descent
  beats black-box fitting, but finite-difference gradients on the code engine match the brain's
  accuracy at 100x less wall time with three parameters. The honest recommendation is unchanged: code
  first; the compiled brain is a reproducible export format and a research instrument, not yet a
  better patient.

## 6. Hardware: 40-series at home now, 5090 at work as the end goal

- **The brain needs no GPU.** 3 to 4 ms per tick on a laptop CPU leaves 99% of a 1-second tick idle;
  a 4090 or 5090 changes nothing for running patients. GPUs matter for two things only: the dialogue
  model and the local builder.
- **40-series interim.** A 4090 (24 GB) runs Qwen3.6-35B-A3B at Q4 (about 19 GB) or gpt-oss-20b
  (11.3 GiB) for the local-builder calibration (exp1 Task C first). A 4080 (16 GB) or 4070 (12 GB) runs
  gpt-oss-20b or a 9B-class model only. Fine-tuning is out on all of them.
- **5090 end goal.** 32 GB fits the 27B dense builders (Qwen3.8-27B) with long context, or the 35B-A3B
  builder plus a 4B dialogue model concurrently (budget table in `docs/builders_and_5090_budget.md`).
- **What to run on the home machine first:** connect it through Remote Control (steps in the session
  summary), then `experiments/exp1_calibration/harness.py --task pk` against a local llama.cpp server
  with gpt-oss-20b or Qwen3.6-35B-A3B. That measures the central unknown: whether any open-weight model
  can do what all six Claude tiers did.
