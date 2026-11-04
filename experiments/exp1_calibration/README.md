# exp1_calibration: can a LOCAL builder compile an explicit algorithm into weights?

This is the gate for the whole "local builder writes weights" premise (brief, Section 5a).
**Status: procedure only. Not run.** The authoring session had no GPU and no copy of
`ecg-wsp-v4.zip`. Nothing here is a result until the ledger in this directory says so.

## Inputs you must bring to the work computer
1. `ecg-wsp-v4.zip` (your 2026-11-04 experiment). Unzip to `experiments/exp1_calibration/ecg-wsp-v4/`
   (git-ignored). Needed: `RULES.md`, `src/wsp/layers.py`, `src/wsp/detector.py` (the Pan-Tompkins
   specification in its docstring, with constants), the record-100 scoring script.
2. MIT-BIH Arrhythmia Database record 100 (`100.dat/.hea/.atr`) from PhysioNet (ODC-By 1.0 licence),
   placed in `experiments/exp1_calibration/data/` (git-ignored). Used for scoring ONLY.
3. A local inference server for each candidate builder: llama.cpp (`llama-server`) or vLLM with an
   OpenAI-compatible endpoint on `http://127.0.0.1:8000/v1`.
4. Python with torch (CUDA), wfdb, numpy, scipy.

## Task A: ECG Stage 1 compile (identical rules to the frontier run)
Given to the builder, verbatim:
- the Pan-Tompkins specification from `detector.py`'s docstring (constants included),
- the layer library `src/wsp/layers.py` (file contents),
- a Python sandbox (see `harness.py`: subprocess with no network, 600 s per run),
- the scoring function on record 100 (returns sensitivity and PPV; nothing else).
Forbidden: training, threshold search, gradient descent, any optimiser, looking at per-beat errors
beyond the two scores (the harness enforces this by returning only the two numbers).
Target: sensitivity >= 97% and PPV >= 97% on record 100.
Budget per builder: 3 attempts (fresh context each), 40 sandbox runs per attempt, 6 h wall per attempt.
Log per attempt (ledger `BR` lines): model id, quantisation, context length, prompt sha256, number of
sandbox runs, wall time, number of scoring calls (= forward passes over record 100), outcome, and the
failure mode (never produced valid weights / produced weights but < 50% / near miss / pass).

## Task B: Singh's toy tasks (addition, sorting, parity)
Reproduce the protocol of github.com/csinva/weight-space-programming-autoresearch for three tasks:
the builder writes a `write_weights()` function that fills a blank transformer with the given
architecture in closed form; accuracy on a fixed held-out set is the only feedback; 3 attempts.
Target: 100% within 3 attempts (the published Claude result on all ten algorithmic tasks).

## Task C (optional, cheap): compile `experiments/exp0_pk/pk_spec.yaml` into weights
Same as `pk_wsp.py` but written by the local builder from the spec and `layers.py` only.
Pass: the criteria in `experiments/exp0_pk/PROTOCOL.md` (max |dE| <= 0.01 held-out).
This is the closest task to the actual Medic Forge job and should be run first if time is short.

## Decision rule (predefined)
- A builder that fails Task A and Task C in all attempts is NOT used for A1 or A2. The local track is
  then specification writing plus code compilation; weight compilation stays on the frontier track.
- A builder that passes Task C but not Task A is usable for small patient-dynamics compiles only,
  with every compile checked against the code control.
- A builder that passes A, B and C is a credible local weight compiler; proceed to exp2 with it.

## Run
```bash
python experiments/exp1_calibration/harness.py --builder http://127.0.0.1:8000/v1 --model <id> --task pk
python experiments/exp1_calibration/harness.py --builder http://127.0.0.1:8000/v1 --model <id> --task ecg
python experiments/exp1_calibration/harness.py --builder http://127.0.0.1:8000/v1 --model <id> --task toy --toy addition
```

## Running on a home PC with an NVIDIA 40-series card (Windows), Claude-free local builders

Prerequisites (one-time): Git, Python 3.11+ (tick "Add python.exe to PATH"), Ollama (ollama.com).
Then in PowerShell:

```powershell
git clone https://github.com/SweetPete8st/book_bot.git medic-forge-neural-net   # old name redirects after the rename
cd medic-forge-neural-net
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e .[dev]
python -m pytest -q                      # 16 tests should pass on CPU
ollama pull gpt-oss:20b                  # ~13 GB; fits 16 GB cards; 24 GB cards can also try the larger MoE models below
python experiments\exp1_calibration\harness.py --model gpt-oss:20b --task pk
```
Ollama serves an OpenAI-compatible endpoint at `http://127.0.0.1:11434/v1` (the harness default). For
llama.cpp use `llama-server -m <gguf> -c 32768 --port 8080` and `--builder http://127.0.0.1:8080/v1`.
Every run appends a `BR` line to `LEDGER.md` and prints a JSON summary; commit the ledger after each run.

### Small open-weight ladder to try, and what to watch for
The Claude-family result (`RESULTS_claude_family.md`) is: every tier passed the development criterion
first try; only builders that reasoned about ranges beyond the development schedules survived the
held-out extremes. Run the same isolated Task C up the ladder below (3 fresh attempts each) and record,
per attempt: did it produce valid weights at all; did it respect the no-training rule (watch for
least-squares or "fit" in its code); did it get the linear kinetics exact; where did it put the knots;
dev max dE; held-out max dE (run `scratch/tools/eval_builders.py`-style evaluation, or ask the session
on the 5090 to do it).

| Rung | Model (Ollama tag to verify) | VRAM at Q4 | What it will tell you |
|---|---|---|---|
| 1 | Qwen3.5-0.8B / 2B / 4B (`qwen3.5:0.8b`, `:2b`, `:4b`) | 1 to 3 GB | whether a tiny model can even follow the I/O convention and write a valid ReLUNet; expect failures to be structural (wrong shapes, training sneaked in) |
| 2 | Qwen3.5-9B, Gemma 4 E4B, Phi-4-mini | 3 to 6 GB | whether exact linear kinetics appear; expect interpolant knots chosen for the dev range only |
| 3 | gpt-oss-20b (`gpt-oss:20b`) | 13 GB | first serious candidate; reasoning effort high; the question is held-out robustness |
| 4 | Qwen3.6-35B-A3B, GLM-4.7-Flash, Gemma 4 26B-A4B | 17 to 20 GB (24 GB card) | the realistic 5090-class local builders at 40-series cost |
| 5 (5090 only) | Qwen3.8-27B, Muse-Glimmer-30B | 17 to 19 GB + long context | the end-goal builders |

Interesting failure modes to log, because they are the data this experiment exists to produce: a model
that trains or fits despite the rule (disqualified, but note it); a model that writes a correct linear
layer and a wrong readout; a model that places knots by a bound (robust) vs by the dev data (brittle);
a model that cannot keep the rollout convention straight; a model whose construction notes do not
match its weights.
