# HANDOFF (written 2026-11-05 at the end of the second working day)

## Where things stand

| Track | State | Evidence |
|---|---|---|
| Specification + code engine (asthma) | working; 4 trajectories pass; 287-case sweep clean; clinically UNREVIEWED | `experiments/exp2_asthma/`, `spec/examples/asthma_adult_acute.yaml` |
| Hand-set weights (exp0) | PASS on v2 held-out after one logged correction; no advantage over code | `experiments/exp0_pk/RESULTS_v2.md` |
| Claude-family builders (6 attempts) | 6/6 pass dev; 2/6 robust on held-out extremes | `experiments/exp1_calibration/RESULTS_claude_family.md` |
| Local open-weight builder (gpt-oss-20b, home PC) | attempt 1 FAIL (no network; refusals; harness v1 faults). Harness v2 pushed, rerun pending | `experiments/exp1_calibration/RESULTS_local_builders.md` |
| Compiled whole-engine brain (exp3) | v2 held-out 14/18 pass; 4 late-session timing failures; FAIL under all-cases rule; export format only | `experiments/exp3_brain/RESULTS.md` |
| Differentiable calibration (exp4) | brain autograd matched finite-difference code at 100x wall time; no advantage | `experiments/exp4_calibrate/RESULTS.md` |
| Parent agent + learning pathway | built, 21 tests green on a mock backend; NOT yet run against a live model | `docs/10_PARENT_AGENT.md` |
| Clinical sourcing | GINA 2025 severity criteria verified; drug constants unverified (`verified: false`) | `docs/clinical_sources.md` |

## Next steps, in order (work computer with the 5090)
1. Verify hardware: `nvidia-smi` (VRAM, driver), RAM; record in `scripts/profile_local.py` output (`python scripts/profile_local.py`).
2. Live run of the parent: `python -m mfnn.parent --model claude-opus-5-5`; one real interview; `build`; read the preview; store a correction; report what broke.
3. Grade the 20 bradycardia transcripts: `python -m mfnn.learning import "data/samples/*" --kind patient_dialogue --names ...`, then `grade --grader David`, then `dataset --no-gate`. Use them to (a) evaluate the patient voice (consistency/leakage), (b) build the student-question test set.
4. Local builder calibration on the 5090: harness v2, three attempts each: gpt-oss:20b, qwen3.6:35b-a3b (24 GB+), qwen3.8:27b if available; then the small ladder. Score every produced network on held-out (the session does this; see `claude_family_attempts/heldout_results.json` for the format).
5. Clinical review of the asthma spec's `verified: false` entries (drugs first).
6. Only then: second preset (COPD on the same engine) and the voice model wiring (fact sheet + checker exist).

## Open questions for the user at restart
- Work computer OS and where the Medic Forge source lives (integration assumptions in `docs/10_PARENT_AGENT.md` §6 are unverified).
- Whether the class datasets contain instructor-builder interviews as well as student-patient dialogues.
- Whether to branch the builder test to a second model family (needs an API key via the secure form).

## Restart prompt (paste into a fresh Claude Code session opened in the repo folder)
See the bottom of this file; it is also in the final message of the 2026-11-05 session.

---
You are continuing the Medic Forge Neural Net project with David (paramedic educator, owner of Medic Forge). Read CLAUDE.md and docs/HANDOFF.md in this repository before doing anything else, then docs/10_PARENT_AGENT.md and docs/09_PHASE2_PATIENT_BRAIN.md.

Hardware: you are now on the work computer with one NVIDIA RTX 5090 (32 GB VRAM) and 64 GB RAM; verify with nvidia-smi and report. This machine also holds the Medic Forge source and the live-class datasets; ask me for their paths before touching them, and never commit anything from data/ to the repository.

State: the specification-first architecture with a code engine is settled and tested; the compiled-brain and weight-writing experiments are finished and documented with honest results (brain as export only; no per-patient networks); the conversational scenario builder (Claude Opus 5.5 by default) and the learning pathway are built and tested on a mock backend but have never run against a live model.

What we are testing next, in order: (1) a live interview with the builder and one build with preview; (2) importing and grading 20 bradycardia student-patient transcripts that I will provide, then building the first training set and a student-question test set for the patient voice; (3) the local open-weight builder calibration on the 5090 with harness v2 (gpt-oss:20b first, then larger Qwen models), scoring every produced network on the held-out sets; (4) clinical review of the asthma specification's unverified constants.

Rules: follow RULES.md (log every forward pass and design change in LEDGER.md, define held-out sets before building, never look at held-out numbers before freezing); label every claim demonstrated, plausible extension, or speculation; report failures as failures; no em dashes; tell me the objective truth and disagree when I am wrong. Start by running python -m pytest -q and nvidia-smi, then tell me what you found and ask me for the dataset paths.
