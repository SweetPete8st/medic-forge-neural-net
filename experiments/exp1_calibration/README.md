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
