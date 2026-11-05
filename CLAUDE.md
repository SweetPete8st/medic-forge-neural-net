# CLAUDE.md: project memory for Medic Forge Neural Net

Read `docs/HANDOFF.md` first in any new session. Then `docs/00_RESEARCH_PACKAGE.md` (phase 1),
`docs/09_PHASE2_PATIENT_BRAIN.md` (phase 2), `docs/10_PARENT_AGENT.md` (phase 3).

## What this project is
Research and prototype code for custom, repeatable teaching patients for Medic Forge, an EMS /
paramedic education platform owned by David (paramedic educator). Three layers: a human-readable
patient specification (YAML, schema-validated, clinically signed), a compile target (the code engine
is the reference; a deterministic compiler can export a ReLU "brain"), and a runtime with instructor
controls, replay and logging. A conversational builder (the "parent", Claude Opus 5.5 by default)
interviews the instructor and produces the spec plus a runnable patient; a learning pathway stores,
grades and turns interviews into training data behind a human approval gate.

## Decisions already made (do not relitigate without new evidence)
- Code engine is the reference and the shipped runtime. The compiled brain is an export format and a
  research instrument; it did not beat code on any instructor-visible criterion (exp0, exp3, exp4).
- No trained neural network per patient; no model that edits weights. Patients are presets + overlays.
- Parent builder = Claude Opus 5.5 via the Anthropic SDK (`src/mfnn/parent/llm.py`). Open-weight
  models are tested only through the calibration harness and the separate local backend. Claude-only
  for the builder for now; other families later, with the user's go-ahead.
- Every experiment follows `RULES.md` (original ECG rules verbatim + extensions): every forward pass
  logged in `LEDGER.md`, every design change logged with DATA-INFORMED yes/no, held-out sets defined
  before building, no held-out number seen before freezing. Failures are reported as failures.
- Every claim in docs is labelled demonstrated / plausible extension / speculation.
- No em dashes in prose for this user. Objective truth over comfort; disagree when warranted.
- Student transcripts are educational records: `data/` is git-ignored, de-identify before use, never
  push them to the public repo.

## Hardware
- Work computer (target): one NVIDIA RTX 5090, 32 GB VRAM, 64 GB RAM pending `nvidia-smi` / verification.
  Also holds the Medic Forge source and the class datasets.
- Home PC: NVIDIA 40-series, Ollama with gpt-oss:20b, qwen3.5:9b, gemma4:12b, qwen3:4b, llama3.1:8b, qwen2.5:7b.
- The engine needs no GPU (0.17 ms per simulated second). GPUs are for the builder model and the voice model.

## How to work here
- `pip install -e ".[dev]" anthropic`; `python -m pytest -q` (21 tests must pass).
- Scenario builder: `python -m mfnn.parent --model claude-opus-5-5`.
- Learning pathway: `python -m mfnn.learning import|grade|dataset|summary`.
- Local builder calibration: `python experiments/exp1_calibration/harness.py --model <ollama tag> --task pk --max-tokens 8192`;
  runs land in `experiments/exp1_calibration/runs/`; score held-out with the session (see RESULTS_local_builders.md).
- Commit messages: plain; no model names in commits or code.
