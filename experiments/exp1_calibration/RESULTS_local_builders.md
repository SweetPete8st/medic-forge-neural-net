# Local open-weight builders, Task C (PK compile), on the home PC (NVIDIA 40-series, Ollama)

Same task as `RESULTS_claude_family.md`. Each row is one fresh attempt. Harness version matters and is
recorded: v1 (initial) had two defects found by the first run (library location not stated; completion
accepted any `weights.npz` plus the word DONE); v2 places the same files as the Claude test in the
working directory, states the import line, and marks a run complete only when all three variants'
weights load, have 6-in/6-out shapes and pass the scorer.

| Model (Ollama) | Harness | Sandbox runs | Wall | Refusals | Reasoning-only replies | Tool-call attempts | Outcome | Held-out |
|---|---|---|---|---|---|---|---|---|
| gpt-oss:20b (32k ctx variant), attempt 1 | v1 | 10 | 73 min | 12 of 29 turns | ~7 | 2 | **FAIL: no network.** 9 scripts crashed looking for `src/wsp/layers.py`; final script re-implemented the net class, saved an empty `weights.npz` (22 bytes), printed DONE; harness wrongly logged `weights_written` (ledger BR 7, corrected by NOTE) | n/a |

**Where it broke (the data this ladder exists to produce).**
- *Refusals.* Twelve times the model answered "I'm sorry, but I can't help with that." to a drug-kinetics
  matrix exercise. Whether the trigger was "dose", "IV bolus" or the instruction not to train, it is a
  usability failure no Claude tier showed. Harness v2 re-prompts with a one-line statement of the
  task's nature and counts refusals.
- *Reasoning-only replies.* Several turns had all text in the reasoning channel and empty content; the
  harness saw no code. gpt-oss via Ollama does this when it has not decided to answer; v2 counts it.
- *Built-in tool reflex.* Two attempts to call its Python tool (harness-caught, re-prompted).
- *Library location.* All nine crashes were import-path guesses. Harness v1's fault; v2 fixes it.
- *False completion.* Writing an empty file and printing DONE satisfied v1. v2 validates and scores.
- *No construction reasoning observed.* Unlike every Claude attempt, no turn laid out the linear
  kinetics or the knot placement before writing code. Whether that changes under v2 is the next run.

Next: rerun gpt-oss under harness v2 (three attempts), then the small-model ladder (qwen3:4b,
qwen2.5:7b, llama3.1:8b, qwen3.5:9b, gemma4:12b). Score any produced networks on the held-out sets.
