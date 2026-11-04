# Medic Forge custom patient models: research package

Date: 2026-11-04. Author: the authoring session (Claude Fable 5.1) with three background research
agents. Status of every claim is labelled **demonstrated** (run in this repository or verified from a
primary source), **plausible extension** (reasoned from demonstrated results, not yet run), or
**speculation** (no direct evidence). Verification levels for literature are those of
`docs/appendix/A_evidence_table_raw.md`: primary (github.com), secondary coverage, search snippet
only, unverified. Where this package says "we measured", the number came from code in this repository
run in the authoring container (4 CPU cores, no GPU).

**Two gaps you should know about before reading.** (1) The ECG archive `ecg-wsp-v4.zip` was not
attached to the session, so nothing in it could be re-run or reused; the layer library here is a
fresh re-implementation and `RULES.md` is a reconstruction. (2) The session's network policy blocked
every primary-source host except github.com, and the web-search budget ran out before the clinical
inventory was finished. One research agent (clinical pharmacology sourcing) was stopped by the
model's safeguards before producing anything. Drug onset/duration values in the asthma
specification are therefore from training knowledge and are flagged `verified: false` for your
clinical reviewer.

---

## 1. What we know, updated (brief Section 1)

### 1a. The ECG experiment
The brief describes it; the archive was not available. Everything below about it is taken from the
brief and is **not independently checked here**. In particular, whether `results/heldout_summary.txt`
exists is unknown, so nothing beyond record 100 is treated as proven. The lessons listed in the brief
were used as priors, and the first one (the PK experiment, Section 7) reproduced two of them on day
one: the only failure of the weight target was a constant chosen by reasoning (interpolant knots),
and the fair comparison against the same rules as code showed no accuracy advantage for weights.

### 1b. Literature status as of 2026-11-04 (verified where possible)
Full table with every field the brief asked for: `docs/appendix/A_evidence_table_raw.md` (Section A),
per-source notes and quotes (Section B), bearing on the simulator (Section C), contradictions
(Section D). Condensed version: `docs/evidence_table.md`. Headlines:

- **MIT CSAIL "Sparks of Weight-Space Programming"** (Takehi, Teoh, Zhang, Khattab; blog
  2026-11-02; logs repo read directly, primary). Builder gpt-6-astra via Codex CLI with all built-in
  tools off. Categories are mixed and must be kept apart: experiments 1, 2, 4, 5 are *direct
  numerical weight writing in context* on 17K to 172K-weight nets; experiments 3, 6, 7, 8 are
  *tensor-editing code in a sandbox*. Exp 8 (Qwen3.5-0.8B): 131,072 weights edited in MLP
  down-projections of layers 15 and 19; MATH-500 7.9 to 48.1; the winning run took 158 turns,
  80 Python runs and 20.3 h; the six other completed Qwen runs topped out at 14.1 (three submitted no
  edit); the task prompt explicitly named repetition/token-limit failure as in scope (primary quote in
  the appendix). Gemma 3 270M, which does not loop, did not improve (6.8 to at most 7.7). The authors'
  own framing in the logs: the Qwen gain "came from removing its looping". Not peer reviewed, no
  arXiv, no benchmark or harness released as of 2026-11-04. The committed harness config names
  `gpt-6-luna` at medium effort while run records say `gpt-6-astra`; unexplained.
- **Independent re-runs of the released Qwen edit** confirm the artifact (Khaliq 51.0%; openwsp
  50.6%; KidIkaros bit-exact rebuild) but **a fresh Astra run on the same task by openwsp reached
  only 12.9%** and their car-repair replication failed 0/1,024. The headline is a one-run result.
- **Cross-model replication (Alan Turing Institute, primary):** writing a working MinAtar policy
  from the game source is frontier-generation-gated. Astra 457/25.5/66.2 (Breakout/Seaquest/
  Asterix); Claude Opus 5.5 112/0/45.1; GPT-5.2 and Opus 4.5 near zero; **DeepSeek-V4-Pro, the only
  open-weight model tested, near zero on writing and at chance on reading.** No 30B-class open model
  was tested by anyone for writing. Willison ran Qwen3.8-27B locally on the *reading* task only:
  3 of 4, at 45 to 60+ minutes per answer. HarleyCoops drove qwen3.6-35b-a3b as an editor under a
  verifier: one accepted two-parameter sign fix, then nothing.
- **Compilation of explicit algorithms into weights by LLM-written code** is the one category with
  reproducible positive results: Chandan Singh's repo (an off-the-shelf Claude coding agent, version
  unstated; all ten algorithmic toy tasks at 100% within three attempts; the three language tasks are
  marked invalid or inflated by the author himself) and ctrimm's benefits-formula nets (three
  different LLMs compiled statutory piecewise-linear rules into ~1.3K-weight nets exact to 1e-12,
  and a 2-parameter surgical update worked). Both are **stateless** feedforward targets. **No source
  anywhere compiled a recurrent or stateful dynamics model**, which is what a patient is.
- **Critiques** (Honnibal, Goedecke, Nanda, Keruac, HN threads 50341174/50351111/50344861, Latent
  Space): consensus that the real capability shown is "LLM writes analysis and construction code",
  not "LLM emits weight numbers", and that the Qwen fix is steering/abliteration-style obstacle
  removal. Keruac's checkable numbers match the primary logs.
- **Lineage:** RASP and Tracr (deterministic compilation of explicit programs into exact transformer
  weights; Tracr archived 2025-04); Arditi et al. (rank-one direction edits remove behaviour);
  HyperNetworks, Text2Weight (~17K-param classifier adapters from 12,000 generated tasks),
  NeuroGen (no code), Text-to-LoRA (479 tasks, ~5 H100-days, matches multi-task LoRA baseline),
  Drag-and-Drop LLMs. Every trained generator needs hundreds to tens of thousands of already-trained
  target networks and none is conditioned on an algorithm.

**Which evidence bears on a stateful patient simulator (demonstrated vs analogy):** direct evidence
exists only for compiling *stateless* explicit rules into small nets via code with a feedback loop
(csinva, ctrimm; and our own PK experiment adds a *stateful* linear-recurrence case, Section 7).
Everything about editing pretrained LMs (Exp 8, Arditi) is analogy. The open-weight writer evidence
is weak-to-negative, which is exactly the risk the brief named.

---

## 2. The central hypothesis, reframed (brief Section 2)

Adopted as the architecture: **specification, compilation, runtime**. The specification is a
human-readable YAML document validated against `spec/schema/patient_spec.schema.json`; it is what
instructors edit and clinical reviewers sign. Compilation targets are compared against the same
specification. The code target is the default hypothesis. The runtime is deterministic and replayable.

The two questions the brief separates were answered as far as this session could:
- *Can a local builder write a correct specification?* **Not tested** (no GPU, no local model here).
  It is ordinary structured text generation plus medical knowledge, so it is the most plausible
  place for a local model to add value; it is also where review burden lands. Measurement plan in
  Section 6.
- *Does compiling to weights beat compiling to code on anything instructors notice?* **Demonstrated:
  no, on the first bounded task.** Section 7 has the numbers. The weight target was at best equal on
  accuracy, repeatability and speed, was harder to write, and failed once on a reasoned constant.

---

## 3. What a custom patient model must do (brief Section 3)

Implemented in `src/mfnn/` and documented in `state.py`. Summary:

**Interface.** Discrete time, one tick = 1 s. `step(state, actions_at_tick, spec) -> (next_state,
events)`; `observe(state, assessment_action | attached devices, spec) -> Observation`. Nothing
happens between ticks; actions are stamped with the tick at which they apply and ordered by
(seq, kind) so replay is deterministic; pause freezes the tick counter; fast-forward applies N ticks
with no actions; the engine is seedless and deterministic (identical spec + action log gives an
identical state digest, tested in `tests/test_engine.py`).

**Separation kept explicit.** Hidden physiology (`state.phys`), drug compartments and devices,
append-only intervention history (part of state, for replay), observation gating (`observe.py`:
SpO2 only with a pulse oximeter attached, breath sounds only on auscultation, BP only on a
measurement action or cycling NIBP), patient history facts (in the spec, not state), dialogue
(`dialogue.py`: a fact sheet with a speech mode derived from state, a shared language model speaks
from it, and a contradiction checker scores the utterance; the dialogue layer never writes state),
and learner grading (`grading.py`: reads the action log; never influences dynamics).

**Persistent memory, delays, repeats, simultaneity, deterioration, recovery.** Per-drug depot,
central and effect-site compartments give onset, peak, duration and re-dosing by construction;
device state persists; fatigue accumulates and recovers; obstruction progresses logistically without
treatment and falls under bronchodilation with a patient-specific responsiveness; simultaneous drugs
combine through shared targets (bronchodilation, heart rate, blood pressure, respiratory drive,
consciousness) rather than through enumerated rules.

**Two designs.**
1. *Compact patient-dynamics network plus shared local language model for conversation* (the brief's
   first design). The network would represent only the hidden-state recurrence and the derived
   observables (the part the PK experiment compiled). Everything else stays in software: action
   parsing, device logic, observation gating, event emission, the fact sheet, grading, logging,
   replay. The language model is shared across patients and prompted from the fact sheet; it must be
   constrained to the speech mode and checked. A small patient network cannot hold medical
   conversation and should not try.
2. *Integrated code engine* (what is implemented): the same recurrence as ordinary code
   (`engine.py`, about 250 lines a reviewer can read top to bottom), spec-driven. The language model
   is the same shared one. This is approach C executed in code: a fixed, validated-once dynamics
   engine whose scenario-specific content is a parameter vector plus a short list of typed rules.

**Worked scenario: adult acute asthma exacerbation** (`spec/examples/asthma_adult_acute.yaml`).
38 sourced parameters (every one carries a citation field and a grade; most are
`modeling_assumption` or `training_knowledge_unverified` with `verified: false`), 8 drugs
(albuterol, ipratropium, IM epinephrine, IV magnesium, methylprednisolone, ketamine; midazolam and
nitroglycerin as inappropriate), oxygen devices, CPAP, BVM, intubation, positioning, 5 stages,
9 events, 5 explicit interaction rules, four trajectories with expected envelopes, observation
gating, dialogue facts and forbidden statements, a grading rubric, and a review block that is
`UNREVIEWED`. **Demonstrated** (`experiments/exp2_asthma/run_trajectories.py`): all four
trajectories meet their envelopes; the untreated patient reaches hypercapnia at 11 min, silent chest
at 28 min, arrest at 30 min; timely treatment recovers; delayed treatment recovers more slowly with
persistent fatigue; the inappropriate path (sedation, nitroglycerin, bag-valve-mask at 20/min)
produces hypoventilation, hypotension from breath stacking, and a barotrauma-risk event. These are
coherence results, not clinical validation: the time constants were chosen to a teaching timeline and
are the first things a reviewer should change.

**Interaction cases, counted** (input to Section 4):

| Category | Count |
|---|---|
| Interventions modelled | 15 |
| Deterioration stages | 5 |
| Intervention x stage | 75 |
| Unordered pairs of interventions | 105 |
| Ordered pairs (order matters because effects are state-dependent) | 210 |
| Repeat-dosing cases (max repeats per drug) | 12 |
| Explicit interaction rules a reviewer must read | 5 |
| Rollouts executed in the sweep (every intervention at every reachable stage, every ordered pair) | 287 |
| Cases with state-constraint violations / causal violations after fixes | 0 / 0 |

The point of the count: with a mechanistic shared-state engine, **300+ cases are handled by
composition and tested automatically**; only 5 needed explicit rules. If the same behaviour were
compiled rule-by-rule into weights (A1/A2), each of those cases is a construction to write and audit.
The first sweep found a real defect (auto-PEEP grew without bound and drove blood pressure negative in
31 cases); it was fixed in one line of the engine and one parameter in the spec. That kind of fix is
what a clinical reviewer can read; its weight-space equivalent is not.

---

## 4. Candidate approaches compared against the same specification (brief Section 4)

| Approach | Smallest credible implementation | Data needs | Local compute | Authoring burden | Validation burden | Likely failure modes | What a per-scenario model buys over a shared parameterised model | Status |
|---|---|---|---|---|---|---|---|---|
| **A1** local model writes weights from the description, no spec | A recurrent ReLU cell of ~1-10K weights written in one shot | none | one builder pass | lowest on paper | impossible to audit except by black-box rollouts | interaction count (300+ cases) with no spec to check against; reasoned constants wrong at extremes (seen in exp0 v1); no evidence any open model can write even stateless nets (ATI) | nothing: the spec is the only reviewable artefact and A1 has none | **speculation**; expected to fail; keep only as a calibration-test negative control |
| **A2** local model compiles the reviewed spec into weights | `experiments/exp0_pk/pk_wsp.py` is the frontier-written instance (1,010 weights, 3 layers) | none | one builder pass + sandbox checks | spec + compile + audit of constructions | must be checked against the code control on held-out variants and sequences | reasoned constants (knots, thresholds); construction errors at parameter extremes; local builders unproven | same behaviour as code with a uniform tensor runtime; differentiability | **demonstrated for the frontier builder on PK** (passes v2); **not tested for any local builder** (exp1) |
| **B** trained scenario-conditioned hypernetwork | Text2Weight-style DiT emitting a ~1K-weight cell | thousands of (description, validated patient) pairs | days of 5090 time for the generator, after producing the dataset | the dataset IS the authoring burden | every generated patient still needs the full test suite | no dataset exists; generators match multi-task baselines at best (Text-to-LoRA); cannot extrapolate to new physiology | nothing until a large validated library exists, at which point a parameterised engine already covers the space | **speculation**; not recommended (Section 5d) |
| **C** shared patient model + generated parameters/adapters | **implemented in code**: engine + YAML parameter vector | none | negligible | spec only | validate the engine once per family, then validate each spec's parameters and trajectories | engine coverage limits (a new mechanism needs engine work); parameter extremes | n/a (this is the shared model) | **demonstrated as code** (exp2); weight-adapter variant untested and offers nothing the YAML does not |
| **D** distillation / training of small patient nets from validated trajectories | `pk_trained.py`: 4,933-weight next-step MLP | code-generated trajectories (free) | minutes on CPU | spec + training config | held-out variants and sequences | compounding rollout error (diverged in two configurations); per-variant models cannot generalise to unseen parameters; needs parameter-conditioning and multi-step losses to be credible | nothing visible to an instructor; the training data came from code that already runs in real time | **demonstrated negative** in the cheap form (exp0); fuller forms untried |
| **E** hybrid: explicit equations + neural components only where equations are unavailable | the code engine plus a learned term for, e.g., individual bronchodilator response curves fitted to data | only for the learned term, and only once data exist | small | spec + a data-fitting step | the learned term must be bounded and reviewed | overfitting a learned term to tiny data; hiding clinical disagreement in a black box | lets one scenario carry a calibrated response curve without changing the shared engine | **plausible extension**; the right home for the differentiability argument |
| Control 1: shared scenario-conditioned model, no new weights | = C in code | | | | | | | **this is the recommendation** |
| Control 2: LLM-generated executable scenario code from the spec | a builder writes a Python module equivalent to `engine.py` from the YAML | none | one builder pass | spec + code review | same test suite as everything else | the generated code diverges from the spec (caught by the harness) | n/a | **plausible**; it is the local-track version of compilation and the natural companion of the calibration test |

Where a method compiles rules into weights, say so plainly: A2 compiles the same rules the code
executes. Comparison on the criteria the brief names, from exp0 (demonstrated) and reasoning
(labelled): *debuggability*: the v1 weight failure was found by comparing against code and locating
the chord error by hand; the code engine's defect in the asthma sweep was found by reading one line.
*Auditability by a clinical reviewer*: a reviewer can read `engine.py` and the YAML; a reviewer
cannot read a weight matrix, so A2 audit reduces to "does it match the code", which concedes that code
is the reference. *Behaviour at parameter extremes*: code is exact by construction; the weight target
failed at EC50 = 0.2 in v1 and passed at EC50 = 0.1 and 4.0 only after knots were made relative to
the spec parameter (demonstrated). The hypothetical advantages of weights (uniform runtime,
differentiability for calibration, parameter blending) are untested and are not unique: the code
engine is already a parameter vector, blending is parameter interpolation, and an autograd version of
`engine.py` would be differentiable without any hand-set weights (plausible extension, approach E).

---

## 5. The 5090 plan (brief Section 5)

### 5a. Calibration test first (procedure, not run)
`experiments/exp1_calibration/README.md` and `harness.py`. Three tasks under the ECG rules: (A) the
ECG Stage 1 compile from the Pan-Tompkins specification and the original `layers.py` (needs the
archive and MIT-BIH record 100), (B) Singh's toy tasks (addition, sorting, parity), (C) the PK compile
from `pk_spec.yaml`, scored by `score_pk.py`, which is the closest proxy for the real job and the
cheapest; run it first. Budgets, logging fields and the predefined decision rule are in the README:
a builder that fails A and C in all attempts is not used for A1/A2, and the local track becomes
specification writing plus code compilation. **Expected outcome (speculation, grounded in the ATI
cross-model table and the HarleyCoops/Willison results): the 27B-class local builders will fail Task
A, may pass Task C with the sandbox feedback loop, and will be slow (tens of minutes per attempt).**

### 5b. Builder candidates (from `docs/appendix/B_builders_and_hardware_raw.md`; vendor benchmarks are
not independent and independent SWE-bench reproductions run 15 to 20 points below vendor numbers)

| Candidate | Params | Licence | Fits 32 GB at 32k / 64k ctx (Q4 + f16 KV) | Measured on a 5090 | Why |
|---|---|---|---|---|---|
| **Qwen3.8-27B** (primary pick) | 27B dense | Apache-2.0 (secondary) | Q4_K_M ~17 GB + 2 / 4 GiB KV: yes / yes | ~77 t/s decode for the 3.6-27B sibling (snippet) | strongest terminal-style agentic numbers in the size class (TB 2.1 73.0 vendor; 30/30 independent local tool tasks) |
| **Qwen3.6-35B-A3B** (fast alternate) | 35B / 3B active | Apache-2.0 | ~19 GB + 0.6 / 1.25 GiB: yes / yes | 194 t/s decode, 7,026 t/s prompt for the 3.5-35B-A3B sibling (primary, llama.cpp) | 2.5 to 3x faster loops; independent SWE-bench 57.0 vs vendor 73.4 |
| **gpt-oss-20b** (cheap third) | 21B / 3.6B | Apache-2.0 | 11.3 GiB + 0.75 GiB: yes / yes | 282 t/s decode, 6,291 t/s prompt at 32k (primary) | leaves room to run the patient dialogue model concurrently |
| gpt-oss-120b (offload tier) | 117B / 5.1B | Apache-2.0 | 59 GiB MXFP4: CPU offload with 64 GB RAM | ~30 t/s (primary, ggerganov) | stronger, slower; sequential use only |
| Gemma 4 31B, GLM-4.7-Flash, Devstral Small 2, Muse-Glimmer-30B | 24-31B | Apache-2.0 / MIT | yes / yes | not measured | reserves |

Runtime: llama.cpp (GGUF, GBNF/JSON-schema structured output) or vLLM (sm_120 supported with CUDA
12.8+; structured outputs via xgrammar). Fine-tuning: QLoRA fits 14B dense and 30B-A3B MoE
(Unsloth minimums 8.5 GB / 17.5 GB); Unsloth advises against QLoRA for Qwen3.5-series; 16-bit LoRA on
27B needs ~60-76 GB and does not fit; full fine-tuning is infeasible. **Inference is feasible;
fine-tuning a builder is not the plan.** Run generation and simulation sequentially: the patient
engine needs no GPU, and the dialogue model is 2-9B.

### 5c. Estimates (all estimates, assumptions shown; replace with `scripts/profile_local.py` output)

| Quantity | Estimate | Assumption |
|---|---|---|
| Builder setup | 2-4 h | llama.cpp build for sm_120, one GGUF download (17-19 GB), harness smoke test |
| Spec generation per patient (local builder) | 3-10 min of generation + instructor review | 4-8K output tokens at 40-80 t/s with reasoning; review is the dominant cost |
| Code compile per patient | 0 min (spec-driven engine) or 5-20 min if a builder writes a code module and the harness checks it | Control 2 |
| Weight compile per patient (A2, local) | unknown; frontier took ~2 h for the ECG net and under an hour for the PK cell in this session | exp1 measures it |
| Validation per patient | 2-5 min machine time | 287-rollout sweep took ~80 s on 4 CPU cores in pure Python (measured); engine step cost dominates |
| Target parameter count | 1-10K weights for a dynamics cell; 0 for code | exp0 cell is 1,010 |
| Checkpoint size | ~10 KB (weights) / ~20 KB YAML (spec) | measured |
| Runtime latency | engine: ~0.3 ms per tick in Python (measured ~0.03 s per 60-min PK rollout; ~0.1-0.3 s per simulated minute for the asthma engine) | pure Python; a 1 s tick leaves >99% idle |
| Academy Lenovo CPU | same engine cost; dialogue model 8-20 t/s decode for a 3-4B Q4 model (inferred from published CPU numbers, not measured) | memory bandwidth bound |
| Classroom concurrency | engine: hundreds of patients per CPU core at 1 s ticks (plausible from measured step cost); dialogue: 1 reply at a time per laptop CPU, or 4-8 concurrent on the 5090 server | profile to confirm |

### 5d. Hypernetwork (B) on this machine
Trainable in principle (a Text2Weight-scale DiT fits a 24 GB card). Not worth it: the only published
generators need 479 to 12,000 trained targets and reach the multi-task baseline on unseen descriptions;
building a thousand validated patient variants costs far more reviewer time than any authoring it
would save, and once that library exists a parameterised engine already interpolates it. **Recommendation:
do not build B.**

---

## 6. Clinical ground truth and real authoring savings (brief Section 6)

**Usable references and data** (`docs/appendix/C_sim_data_standards_raw.md`; verification levels
inside): Pulse Physiology Engine (Kitware; Apache-2.0; `AsthmaAttack` action with severity 0-1, COPD
condition, Python API, JSON scenarios) is the best open reference simulator for generating candidate
trajectories and for cross-checking our engine's gas-exchange behaviour; BioGears (Apache-2.0; asthma
action not confirmed). Datasets: MIMIC-IV-ED has triage and repeated ED vitals at about hourly
charting; HiRID at 2 min; NEMSIS is event-level with timestamped vital sets; MIT-BIH for ECG.
**No dataset with minute-level prehospital vitals during treatment of respiratory distress was
found** (not proven absent; the search was cut off). Guidelines: GINA 2025, BTS/SIGN (acute severe:
PEF 33-50%, RR >= 25, HR >= 110, cannot complete sentences; life-threatening: PEF < 33%, SpO2 < 92%,
silent chest, cyanosis, poor effort, arrhythmia, hypotension, exhaustion, altered consciousness;
near-fatal: raised PaCO2; secondary coverage), NASEMSO National Model EMS Clinical Guidelines v3.0
(March 2022; respiratory distress section not retrieved in full), GOLD 2026 (Rome classification).
The exact GINA table and all drug kinetics remain for the reviewer to confirm; the YAML marks each.

**LLM-generated trajectories are candidate data.** Checking: (1) every candidate is replayed
through the code engine and must stay inside the spec's envelopes; (2) causal checks (dose then
onset; no improvement while obstruction rises; oxygen never lowers PaO2) are run automatically;
(3) a second generator from a different model family produces independent candidates, and
disagreement beyond the envelope is flagged for a human; (4) agreement between generator and evaluator
from the same family is not evidence, so the evaluator for any LLM-generated trajectory is the
engine plus a human, never the generating family. Uncertainty is represented as the parameter
`range` fields in the spec and as trajectory envelopes, not as point values.

**Units and semantics** are fixed in the schema (minutes in the file, seconds in the engine; mmHg,
bpm, breaths/min, mL, %). State constraints are checked every rollout (`invariants()`), causal
relationships and timing are checked (`causal_checks()`), unsupported drugs emit an
`unsupported_drug` event and change nothing, and arrest freezes the physiology pending a resuscitation
module. Instructor override, pause/reset, event logs, versioned definitions (spec hash in every
result file), replay (bit-exact test) and model freezing (`FREEZE_MODEL`, ledger `freeze()`) are in
the interface; the UI is not.

**Measuring instructor effort end to end** (plan, not yet run): time five instructors on three
scenarios each under two conditions, (i) today's method and (ii) description, generated
specification, review and correction, validation report, later update. Record minutes per phase,
number of corrections to the generated spec, number of trajectories rejected, and minutes to the
accepted scenario. The hypothesis to beat: authoring time falls but review time rises; the net is
positive only if the generated spec needs fewer than a handful of corrections. **Assume, until
measured, that this project moves complexity into specification review and the test harness**,
exactly as the ECG experiment moved it into design reasoning and a harness.

---

## 7. The first experiment (brief Section 7): done, and what comes next

**exp0_pk, run in this session under RULES.md.** One specification (one-compartment kinetics with
first-order absorption, effect-site link, Emax readouts, dosing events, time-since-dose); three
targets: code, hand-set weights (A2, written by the frontier model in-session), trained MLP (D).
Splits written before building: 24 dev cases; held-out = whole new variants x whole new schedules
(unseen orders, repeats, simultaneous routes, 180-min rollout, overdose, nothing). Results tables:
`experiments/exp0_pk/RESULTS_v1.md` and `RESULTS_v2.md`. Ledger: 204 forward passes per version,
3 logged design changes (all DATA-INFORMED), 0 builder runs.

| | v1 (frozen e3138aef) | v2 (frozen ad34aa59, new held-out block) |
|---|---|---|
| A2 weights, held-out max effect error | 0.0120 (FAIL, limit 0.01; EC50 0.2 variant, first interpolant chord) | **0.0030 (PASS)** on EC50 0.1 and 4.0, 25-unit dose, 20 micro-doses, 5 IV boluses |
| A2 hidden-state error | 1e-14 | 1e-14 |
| A2 replay / violations / causal | bit-exact / 0 / harness bug | bit-exact / 0 / 0 |
| D trained MLP | diverged on dev | 0.59 dev, diverged on unseen schedules (FAIL) |
| Code control | reference | reference |

What it demonstrates and does not is spelled out in `RESULTS_v2.md`. The short form: weights can be
as exact as code on a stateful linear-plus-Emax model; they were not better on anything; the one
failure was a reasoned constant at a parameter extreme; cheap conventional training is not a free
lunch.

**exp2_asthma, next.** Development set: the four scripted trajectories and the 287-case sweep (done
as coherence checks). Held-out set to define before any compile target other than code is built:
three new patient variants (poorly responsive phenotype, COPD-overlap with CO2 retention on high-flow
oxygen, small-adult weight extremes) and six new action sequences (ipratropium before albuterol;
epinephrine first; CPAP started at LOC = P; BVM at 8/min from minute 5; two albuterol doses 3 minutes
apart; 90-minute session with no second dose). Metrics: the brief's list, each already implemented
except dialogue contradictions (checker exists; needs the language model) and instructor minutes.
Pass/fail: code control must satisfy all envelopes and invariants on held-out; any weight target must
match the code within 2% of range on every observable, reproduce every event within 30 s, replay
bit-exactly, and need at most 3 design changes; otherwise the weight track for respiratory scenarios
ends. Independent review required before any classroom use: one paramedic educator and one physician
sign the YAML `review` block; the engine refuses APPROVED status without named reviewers. A finite
suite passing is not clinical validity.

---

## 8. Deliverables, roadmap, decision table, direct answers (brief Section 8)

**Delivered in this repository:** evidence table (appendix A, condensed `docs/evidence_table.md`);
calibration procedure (exp1); recommended architecture (Sections 2-3, code in `src/mfnn/`);
instructor explanation (`docs/instructor_explainer.md`); 5090 budget (Section 5, appendix B);
specification schema and example with an update interface (the YAML is the update interface: edit a
parameter's `value` inside its `range`, bump nothing else, the spec hash changes and the result files
record it; `InstructorControl.OVERRIDE_STATE` is the live override); prototype repository with core
code (not pseudocode: it runs); data-generation and validation plan (Section 6); roadmap and decision
table (below).

**Roadmap.**
1. *Calibration (first week on the work computer):* bring `ecg-wsp-v4.zip` and record 100; run exp1
   Task C then A then B for Qwen3.8-27B and Qwen3.6-35B-A3B via llama.cpp; log every attempt; apply
   the decision rule.
2. *First experiment, local-builder version (week 2):* repeat exp0 with the local builder writing
   `pk_wsp.py` (Task C) and, separately, writing a code module from `pk_spec.yaml` (Control 2);
   compare against the frontier-written versions already in the repo.
3. *First usable scenario (weeks 3-6):* clinical reviewer edits `asthma_adult_acute.yaml` (drug
   constants first, then time constants); define the exp2 held-out set; wire the dialogue model
   (3-4B Q4 on laptops) to the fact sheet and measure contradictions; pilot with two instructors and
   time them.
4. *Broader coverage (after a signed scenario):* COPD, anaphylaxis, opioid overdose as parameter
   sets of the same respiratory/cardiovascular engine; add engine mechanisms only when a scenario
   needs one; keep every addition behind the same sweep.

**Decision table.**

| Claim | Status | What result would change it |
|---|---|---|
| A specification-first pipeline with a code engine gives repeatable, coherent patients | demonstrated (exp2 coherence; replay test) | clinical review rejecting the engine's mechanisms rather than its constants |
| Hand-set weights can match code on a stateful kinetics model | demonstrated (exp0 v2) | failure on the respiratory held-out set |
| Weights beat code on an instructor-visible criterion | not demonstrated; no evidence | a measured win on calibration-from-data or parameter blending that the parameterised code engine cannot match |
| A local 27B-class builder can compile specs into weights | needs testing (exp1) | passing Task C within budget keeps A2 alive locally; failing A and C ends the local weight track |
| A local builder can write a usable specification | needs testing | fewer than ~5 reviewer corrections per generated spec in the pilot |
| Trained small patient nets (D) are a cheap compile target | demonstrated negative in cheap form | parameter-conditioned, multi-step-trained model passing exp0 held-out at 0.05 |
| Hypernetworks (B) are worth building | not recommended | a validated library of >1,000 patients appearing for free |
| **Kill criterion for the weight track:** | | exp1 fails for every local builder AND the respiratory exp2 shows no criterion where weights beat the code control. Then weights stay a frontier-only research curiosity and Medic Forge ships code. |

**Direct answers.**

*Can a locally running builder on the 5090 produce custom patient models that materially reduce
scenario-authoring work?* Plausibly yes, but not by producing networks. The reduction comes from a
local model drafting the specification and the shared engine executing it; what remains for the
instructor is review, which is the right place for their time. Whether the drafted specifications are
good enough to save net time is unmeasured and is the pilot in step 3. Whether any local model can
compile weights at all is unmeasured and, on the cross-model evidence, doubtful; that is step 1.

*Does compiling to weights beat compiling to code on any criterion an instructor would notice?* No,
not on anything measured: equal accuracy at best, equal repeatability, no speed benefit, harder to
write, harder to audit, and the single failure seen came from a constant set by reasoning. The
remaining arguments for weights (differentiable calibration, blending) are untested and available to a
parameterised code engine anyway. Recommend code; keep A2 as a research track that must earn its place
on the respiratory held-out set.

*What exact experiment should you run first?* `experiments/exp1_calibration` Task C: give
Qwen3.8-27B (Q4_K_M, llama.cpp, 32k context) `pk_spec.yaml`, `src/mfnn/wsp/layers.py` and the
sandbox, and ask for `weights.npz` that passes `score_pk.py` (max effect error <= 0.01 on the dev
cases) within 40 sandbox runs and 6 hours, three fresh attempts. Log everything. Then Task A with the
ECG archive. The outcome decides whether the local track includes weights or is specification plus
code from here on.
