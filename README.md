# Medic Forge Neural Net

Research and prototype repository for **custom patient models for Medic Forge**, an EMS /
paramedic education platform. The question under test:

> Can a locally running "builder" (one RTX 5090) turn an instructor's plain-language scenario
> description into a small, clinically coherent, repeatable patient model, and does compiling
> that model into neural-network weights beat compiling it into ordinary code on any criterion
> an instructor would notice?

This repository treats that as an **engineering hypothesis**, not a conclusion. Every claim in
`docs/` is labelled **demonstrated**, **plausible extension**, or **speculation**.

## Layout

| Path | What it is |
|---|---|
| `docs/00_RESEARCH_PACKAGE.md` | The full research package (Sections 1 to 8 of the brief). Start here. |
| `docs/10_PARENT_AGENT.md` | The conversational scenario builder (parent agent) and the continuous-improvement pathway. |
| `docs/09_PHASE2_PATIENT_BRAIN.md` | Phase 2: the patient as a compiled neural network (deterministic compiler, whole-engine brain, Claude-family builder test, differentiable calibration). |
| `docs/instructor_explainer.md` | Plain-language explanation for paramedic instructors. |
| `docs/evidence_table.md` | Literature evidence table with categories and verification levels. |
| `docs/builders_and_5090_budget.md` | Local builder candidates, VRAM budget, profiling procedure. |
| `docs/clinical_sources.md` | Sourced clinical parameters for the respiratory scenario. |
| `RULES.md` | Accounting rules for every experiment (forward passes, design changes, held-out data). |
| `LEDGER.md` | The running ledger. Every forward pass and design change is logged here. |
| `spec/schema/patient_spec.schema.json` | JSON Schema for the human-readable patient specification (Layer 1). |
| `spec/examples/asthma_adult_acute.yaml` | Worked specification: adult acute asthma exacerbation. |
| `src/mfnn/` | Runtime: patient state interface, code-compiled engine (the control), observation gating, dialogue fact sheet, ledger tooling. |
| `src/mfnn/parent/`, `src/mfnn/learning/` | Parent agent (interview, gate, deliverable, CLI) and the learning pathway (store, importer, grading, dataset, eval, promotion). |
| `src/mfnn/wsp/` | Weight-space constructions (numpy re-implementation), the hand-set PK network, and `compiler.py`: a deterministic scalar-program-to-ReLU-weights compiler with error bounds. |
| `experiments/exp3_brain/` | The whole asthma engine as a compiled recurrent ReLU network, checked tick-by-tick and over sessions against the code engine. |
| `experiments/exp4_calibrate/` | Differentiable calibration of the compiled brain (torch) vs black-box fitting of the code engine. |
| `experiments/exp0_pk/` | First experiment: one-compartment drug kinetics with dosing, code vs hand-set weights. |
| `experiments/exp1_calibration/` | Calibration test for local builder candidates (needs the ECG archive and a GPU). |
| `experiments/exp2_asthma/` | Respiratory scenario trajectories and interaction enumeration. |
| `scripts/profile_local.py` | Local profiling procedure for the 5090 and academy laptops. |
| `tests/` | Unit tests. |

## Three-layer pipeline (the reframed hypothesis)

1. **Specification.** The builder writes an explicit, human-readable physiology specification
   (state variables, sourced parameters, response rules with onset and duration, interaction
   rules, trajectories, observation gating, dialogue facts). Instructors and clinical reviewers
   edit and sign this document. It is the unit of version control.
2. **Compilation.** The specification is compiled into an executable patient. The default target
   is ordinary code. Weights, adapters, or trained models must beat the code target on a
   measurable criterion to be adopted.
3. **Runtime.** The compiled patient runs locally with instructor controls, event logs and replay.

## Status (updated 2026-11-05; see docs/HANDOFF.md)

- Repository bootstrapped from the retired `book_bot` project. Old branches are preserved under
  `archive/*` tags.
- The ECG weight-space archive (`ecg-wsp-v4.zip`) referenced by the brief was **not available**
  in the authoring session. `RULES.md` is a reconstruction from the brief's description and must be
  replaced verbatim when the archive is brought in. `src/mfnn/wsp/layers.py` is a fresh numpy
  re-implementation of the constructions the brief lists, not the original `src/wsp/layers.py`.
- No GPU was available in the authoring session. Every calibration number is therefore a
  **procedure**, not a result, until it is run on the 5090.
- exp0_pk ran to completion in-session: hand-set weights matched the code control on a new held-out
  set after one logged correction; the trained network did not. See `experiments/exp0_pk/RESULTS_v2.md`.
- exp2_asthma: four trajectories pass their envelopes; 287-case interaction sweep clean.
- Clinical drug constants in the asthma spec are unverified training knowledge pending reviewer sign-off.
- Phase 3 (2026-11-05): conversational scenario builder (Claude Opus 5.5 default) and the
  continuous-improvement pathway are built and tested on a mock backend; no live-model run yet.
  See `docs/10_PARENT_AGENT.md`.
- Phase 2 (2026-11-04): a deterministic compiler turns the engine's physiology program (351 nodes) into a
  26-layer ReLU brain (517K nonzero weights) that reproduces the code engine within a 30 s time shift on
  the development paths; six isolated Claude-family builder attempts all passed the PK compile on the
  first build, two of six were robust on held-out extremes. See `docs/09_PHASE2_PATIENT_BRAIN.md`.

## Quick start

```bash
pip install -e .[dev]
python -m pytest -q
python experiments/exp0_pk/run_experiment.py      # code vs hand-set weights, with ledger
python experiments/exp2_asthma/run_trajectories.py  # four asthma trajectories + interaction count
```
