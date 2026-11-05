# The conversational scenario builder ("parent") and the continuous-improvement pathway

Date: 2026-11-05. Status labels as elsewhere. Everything here runs on a laptop CPU; the only GPU use is
an optional local model as the builder's language model.

## 1. What it is, in one picture

```
instructor  <-- "Let's get started" -->  PARENT (LLM agent)  -->  slot ledger  -->  preset + overlay  -->  full spec (YAML)
                                                                      |                                        |
                                                                 completeness gate                  engine runs 4 paths, checks
                                                                      |                                        |
                                                            <------ preview + corrections <------  deliverable: spec sheet + runnable patient
                                                                      |
                                                       store (interview, ledger, gate, corrections)
                                                                      |
                                                   grading (auto + human; LLM pre-grade advisory)
                                                                      |
                                                   dataset (SFT JSONL)  -->  candidate parent  -->  eval on fixed briefs  -->  human approval  -->  current parent
```

The LLM only talks and fills slots. Everything that decides what the patient does is deterministic:
the preset tables (`src/mfnn/parent/presets/*.yaml`), the composer, the schema, the gate, the engine.

## 2. Components (demonstrated with a mock backend; not yet run against a live model)

| Piece | File | What it does |
|---|---|---|
| Backends | `src/mfnn/parent/llm.py` | Claude via the Anthropic SDK (default `claude-opus-5-5`, adaptive thinking, explicit effort, server-side refusal fallback); a separate OpenAI-compatible backend for open-weight models on Ollama/llama.cpp; a scripted mock for tests. |
| Slot ledger | `slots.py` | 21 slots (14 required) with types, allowed values, plain-language prompts; per-slot confidence; completeness = required slots at or above the threshold, no unacknowledged unsupported requests. The threshold is the instructor's confidence setting. |
| Preset | `presets/asthma_adult_acute.yaml` | How slot choices become spec overlays (severity, pace, responsiveness tables), the traps the engine can model, the four canonical scripts, the objective-to-event map. All reviewable. |
| Composer | `compose.py` | Preset + ledger -> overlay (small, per patient) -> full spec. No LLM. |
| Gate | `gate.py` | Schema validity, completeness, four engine rollouts, invariants, instructor checkpoints, objective reachability, "treatment helps" sanity. |
| Deliverable | `deliverable.py` | Scenario package: `spec.yaml`, `overlay.yaml`, `parameters.json`, `scripted_events.yaml`, `dialogue_facts.yaml`, `gate_report.json`, `interview.json`, `README.md` with the preview table; optional `brain.npz` from the compiler. |
| Interview | `interview.py` | The loop: system prompt with the slot schema and what is already filled; model returns JSON (say, slots, unsupported, done); the ledger and gate decide; every turn stored. |
| CLI | `python -m mfnn.parent` | Terminal front end; writes to `scenarios/` and appends to the store. |
| Store | `src/mfnn/learning/store.py` | Append-only JSONL: interviews, corrections, grades, outcomes. |
| Importer | `importer.py` | Legacy transcripts (JSON/JSONL, plain text, CSV) -> store, with role normalisation, kind inference, de-identification. |
| Grading | `grading.py` | Automatic metrics per record; a human rubric per kind; an advisory LLM pre-grade that records the grader family. |
| Dataset | `dataset.py` | Graded records -> chat-format SFT examples; only human-graded records at or above the bar; live interviews must also have passed the gate. |
| Evaluation | `evaluate.py` + `data/eval_briefs/briefs_v1.json` | Fixed briefs; a simulated instructor (a backend playing the persona, family recorded); metrics: completion, turns, enum slot accuracy, gate pass, parse errors. |
| Promotion | `promote.py` | Registry of parent versions with prompt hash and eval; a candidate is eligible only if not worse on the gated metrics; approval requires a named human. |

## 3. How to run it (MacBook or PC)

```bash
pip install -e .[dev] anthropic
export ANTHROPIC_API_KEY=...          # or `ant auth login`
python -m mfnn.parent                 # Claude Opus 5.5, threshold 0.8, writes scenarios/<id>/
python -m mfnn.parent --backend local --model qwen3.5:9b   # open-weight builder via Ollama (later track)
```
Type what you want; answer its questions; type `build`; read the preview; adjust; `build` again. Every
build is stored under `data/interviews/`. To import your existing transcripts (any of the three formats
in `docs/examples/`):

```bash
python -c "from pathlib import Path; from mfnn.learning.store import InterviewStore; from mfnn.learning.importer import import_legacy; print(import_legacy(Path('path/to/transcripts').glob('*'), InterviewStore(Path('data/interviews')), names=['Student Name 1']))"
```
Grade a record (human): `store.add_grade(record_id, 'David', {...rubric scores...})`. Build training data:
`build_sft(store, Path('data/sft/builder_v1.jsonl'))`. Evaluate a candidate: `run_eval(parent_backend,
instructor_backend, briefs, Path('scenarios/eval'))`. Register and approve: `Registry(...).register(...)`,
`.approve(vid, 'David')`.

## 4. The improvement loop, explicitly

1. Every live interview is stored with its ledger history, gate report, backend, model and token usage.
2. Every correction after a preview is stored as (what you said, spec before, spec after).
3. Automatic metrics run on everything. You grade a sample by rubric (1 to 5 per item). An LLM
   pre-grade can triage, but it is advisory and carries its model family, so same-family bias stays visible.
4. The dataset builder emits SFT examples only from human-graded, gate-passing records.
5. A candidate parent (new prompt, new model, or a LoRA on an open model) is evaluated on the fixed
   briefs by a simulated instructor; the registry compares it with the current version.
6. Promotion requires a named human. The current parent's prompt hash and eval are in the registry.
7. Nothing in the loop changes the engine or the preset tables. Those change only through reviewed
   edits to YAML, which is the clinical sign-off path.

## 5. Your 400 transcripts: what they are good for, and the two things to decide

Student-patient dialogues train and evaluate the voice (consistency, realism, no leakage), and they are
the best source of the questions students actually ask, which becomes a test set the fact sheet must
answer. Instructor-builder interviews train the parent. The importer infers which is which from the
speaker roles; you can override per file. Decide: (a) who grades, with the rubric in `grading.py`,
and how many (a few dozen well-graded records are worth more than 400 ungraded); (b) de-identification:
the importer scrubs emails, phones, ID-like numbers and names you list, but student transcripts are
educational records, so review a sample before anything leaves your machine, and keep `data/` out of
public repositories (it is git-ignored).

## 6. Integration assumptions (not verified against the Medic Forge source)

- Medic Forge can host a chat box that calls `Interview.step()` and shows `say`; a "build" button calls
  `Interview.build()` and renders `README.md`'s preview table; a "start" button loads
  `scenarios/<id>/spec.yaml` into the engine and `dialogue_facts.yaml` into the voice model.
- The monitor reads the engine's observations through `observe()` gating; the voice model reads
  `fact_sheet()` each turn and its replies pass `check_utterance()` before display.
- Instructor corrections after preview are posted back as free text to `Interview.step()`; numeric
  corrections ("sats 88 by 10 minutes") become checkpoints the gate enforces.
- Review and sign-off happen on `spec.yaml` (`review.status: APPROVED`); the engine refuses classroom
  mode otherwise.

## 7. What is not done
- No live-model run yet; the pipeline is demonstrated end to end on scripted replies only.
- Only one preset exists (adult acute asthma). Chest pain, COPD, anaphylaxis need presets, and the
  first two need engine modules (cardiac, CO2 retention) that do not exist yet.
- The voice model is designed but not wired (fact sheet + checker exist; the chat loop does not).
- Fine-tuning scripts are not included; the SFT JSONL is in the standard chat format that Unsloth and
  Hugging Face trainers accept.
