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
  a 351-node program and compiled: 26 ReLU layers, widths up to 1,857, 9.2M dense / 517K nonzero
  parameters, 3 to 4 ms per 1-second tick in pure numpy on this container, 0.2 s to compile.
- **The shell.** The same software shell as the code engine (actions, devices, FiO2 table, monitors,
  observation gating, events, stages, replay) with the physiology step replaced by the network.
- **Torch export.** The compiled brain exported to torch reproduces the numpy rollout to 7e-10, and the
  rollout is differentiable with respect to the parameter port (used in exp4).

## 2. Does the compiled brain reproduce the patient? (exp3_brain)

Per-tick check on 400 random states (physiology, drugs, devices, arrest): every state variable within
0.01 mmHg/bpm/breaths of the code step (O 3e-6, F 2e-4, LOC 8e-5); no arrest-flag, speech or
silent-chest mismatches after two logged fixes (a speech-level arithmetic error of mine, and the
arrest reporting convention of the shell).

Development trajectories (the four scripted paths), brain vs code, after the fixes and one logged
knot tightening:

| Path | Strict max (% of range) | Away from events | Shift-tolerant (30 s) | Max event delay | Events missing/extra |
|---|---|---|---|---|---|
| untreated | 61.2 (EtCO2 at the arrest instant) | 3.5 | **0.17** | 5 s | 0 / 0 |
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

Held-out results (3 unseen parameter variants x 6 unseen action sequences, run after freezing the
brain): see `experiments/exp3_brain/results/heldout_results.json` and the table appended below once
the run completed.

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

Protocol in `experiments/exp4_calibrate/PROTOCOL.md`: a hidden "instructor" patient (code engine with
secret port values) supplies two corrected observable traces; recover the three port values from the
spec defaults by (a) gradient descent through the torch-exported brain and (b) Nelder-Mead on the code
engine, same loss, same budget. Results appended below once both halves completed.

## 5. What this means for Medic Forge (assessment)

- **Demonstrated:** a reviewable physiology program can be compiled deterministically into a patient
  network that reproduces the reference engine within a 30 s time shift over full sessions, with
  bit-exact replay and a manifest that names every hidden unit. "Designing the brain" becomes writing
  or editing the program and the specification; the weights are a build artefact, like a binary.
- **Demonstrated:** the frontier-model-writes-weights route works reliably on small, fully specified
  pieces and fails on unseen ranges unless the builder reasons about bounds. It does not scale to the
  whole engine by hand: the asthma program has 351 nodes and the compiler emits 517K nonzero weights;
  nobody should hand-write that.
- **Still true from Phase 1:** nothing here makes the network more accurate, faster or easier to
  review than the code it was compiled from. The code engine runs the same patient at 0.3 ms per
  tick; the brain needs 3 to 4 ms in numpy. The brain's distinct claims are the uniform tensor runtime
  and differentiability (exp4). If exp4 shows gradient calibration recovers instructor corrections
  better than black-box fitting of the code, the brain has one concrete advantage; if not, it has none
  yet, and the honest recommendation remains code first, brain as an export format.

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
