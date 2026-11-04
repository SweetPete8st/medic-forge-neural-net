# Evidence table (condensed). Full version with all fields and quotes: `appendix/A_evidence_table_raw.md`

Categories: DW = direct numerical weight writing in context; TE = tensor-editing code in sandbox;
CA = compilation of explicit algorithm into weights; HN = trained hypernetwork; AG = adapter
generation; OT = ordinary training. Verification: P = primary (github.com), S = secondary coverage,
N = search snippet only, U = unverified.

| Source (date) | Cat. | Builder | Target | Feedback / optimisation | Result | Replication | Ver. |
|---|---|---|---|---|---|---|---|
| MIT Exp 1-2 reading (2026-11-02) | DW | gpt-6-astra, no tools | 17K-24K-weight MLPs | none | 4/4 games, 3/3 shapes identified | reproduced by openwsp, ATI (Astra, Opus 5.5), Willison (Qwen3.8-27B 3/4, 45-60 min each) | P |
| MIT Exp 4 car repair | DW | gpt-6-astra | 171,664-weight policy | none, one shot | 0/256 to 1,024/1,024 after 6,073 weights edited | **failed** (openwsp 0/1,024) | P |
| MIT Exp 5 MinAtar from source | DW | gpt-6-astra | 20,486-weight policy, zero-init | none | 517/53/53 vs PPO-500M 25/89/55 | ATI Astra 457/25/66; Opus 5.5 112/0/45; GPT-5.2, Opus 4.5, DeepSeek-V4-Pro near zero | P |
| MIT Exp 6 Go | TE | gpt-6-astra + Python + game passes | 1.48M weights | per-pass summaries; 50 passes | 84% vs small PPO | none | P |
| MIT Exp 7 sycophancy | TE | gpt-6-astra | Gemma 3 270M pairs | none | weights-only at chance; 3/3 with attention tool | none | P |
| MIT Exp 8 Qwen edit | TE | gpt-6-astra, 158 turns, 80 Python runs, 20.3 h | Qwen3.5-0.8B, 131,072 weights | paired practice results, 50 passes | MATH-500 7.9 to 48.1 (token-limit hits 93 to 33%); other 6 runs <= 14.1 | artifact confirmed (Khaliq 51.0, openwsp 50.6, KidIkaros bit-exact); **fresh Astra run 12.9%** (openwsp) | P |
| MIT Gemma follow-up | TE | gpt-6-astra | Gemma 3 270M | same | 6.8 to <= 7.7 (no gain) | none | P |
| Singh / csinva toy tasks | CA | "Claude" coding agent (version unstated) | blank transformers 174 to 22M params | accuracy on 200 held-out, <= 3 attempts | 10/10 algorithmic tasks at 100%; language tasks invalid/inflated per author | none | P |
| ctrimm benefits formulas | CA | unnamed LLM; GPT-5.6 Sol, Gemini 3.1 Pro | ~1.3K-param piecewise-linear nets | none, SHA-frozen before scoring | exact to 1e-12; 2-param surgical repair exact | self-replicated with 2 writers | P |
| **This repo, exp0_pk** (2026-11-04) | CA | Claude Fable 5.1 in-session | 1,010-weight recurrent ReLU cell | code control; ledger; frozen before held-out | v1 FAIL (0.012 > 0.01 on EC50 0.2); v2 PASS (0.003) on new held-out | n/a | demonstrated |
| HarleyCoops | DW | qwen3.6-35b-a3b (open) | 17K-weight policy | verifier accept/reject | one accepted 2-parameter fix; later versions submitted nothing | n/a | P |
| ATI cross-model | DW | 8 models | Exp 1, 2, 5 nets | none | writing is Sep-2026-frontier-gated; only open model tested (DeepSeek-V4-Pro) near zero | this is the replication | P |
| RASP / Tracr (2021/2023) | CA | human / compiler | transformers | none | exact compiled programs; Tracr archived 2025-04 | by construction | N/P |
| Arditi 2024 | TE | code | 13 LMs 1.8-72B | activations | rank-one edit removes refusal | widely | N |
| HyperNetworks 2016 | HN | trained generator | LSTM/CNN weights | gradient | weight sharing, not text-conditioned | long-established | N |
| Text2Weight 2025 | HN | DiT on CLIP text | ~17K-param classifier adapters | 12,000 generated tasks | contested by a reviewer | code released | N/P |
| NeuroGen 2025 | HN | LLM generator | small CNN/MLP | checkpoint dataset (size unreported) | numbers unverified; no code | none | N |
| Text-to-LoRA 2025 | AG | hypernet | LoRA for 7-8B LMs | 479 tasks, ~5 H100-days | matches multi-task-LoRA baseline zero-shot | code+weights | P |
| Survey 2603.10090 (2026) | HN | n/a | n/a | n/a | no "spec to stateful net via LLM" line of work | n/a | N |
| Critiques: Honnibal, Goedecke, Nanda, Keruac, HN x3, Latent Space (2026-11-03) | TE | n/a | n/a | n/a | capability shown is "writes construction/analysis code", Qwen fix is obstacle removal; Keruac's numbers match logs | n/a | N |
| MIT benchmark/harness; Anthropic/OpenAI statements; peer review | | | | | **none exist as of 2026-11-04** | | N (absence) |

**Bearing on a stateful patient simulator.** Direct evidence: compile-by-code of stateless explicit
rules (csinva, ctrimm) and now one stateful linear-plus-Emax cell (this repo, frontier builder).
Analogy only: every pretrained-LM edit. Negative-leaning on the local-builder axis: no open-weight
model has written a non-trivial net in any published test; a 27B model reads a 17K net slowly.


## Verification update (2026-11-04, after the session's network allowlist was approved)

The following were re-read from their primary pages (downloaded into the session, not committed):

| Source | Now verified | What the primary text says |
|---|---|---|
| MIT blog (rikiyat.github.io, 2026-11-02) | primary | "the edit changes only 131,072 of the model's 0.8 billion weights (0.016%)". "Fixing loops, however, is not the same as making a model better at math: it lets the model finish answers it could already work out." Gemma 3 270M: "Astra could not improve it." "We are currently building a benchmark and a harness for LLMs' capability on Weight-Space Programming." Experiments 1, 2, 4, 5 read or write weights as text with no tools; 3, 6, 7, 8 use a Python sandbox; 6 and 8 add budgeted forward passes. |
| Honnibal, "The 'Astra can read and write weights to improve Qwen' post is misleading" | primary, dated 2026-11-03 | Only the raw-floats-in-context experiments are "just by looking at the raw weights"; "Every experiment involving a real language model uses Python, and the Qwen result ... also gets to run the model on up to 50 batches of practice problems"; concedes the work is what "a human ML engineer could do". HN thread 50351111: 1,015 points, 206 comments; top comment: "the key result has nothing to do with making Qwen better at math, it just reduces reasoning loops." |
| Goedecke, "LLMs are not superhuman at reading and writing neural network weights" | primary (HN 50344861, 146 points) | as summarised in the appendix |
| Hugging Face card RikiyaT/Qwen3.5-0.8B-weight-space-programming | primary | Apache-2.0. "Independent re-run of this checkpoint: full MATH-500, 500 problems x 4 samples = 2,000 generations ... The edited model scored 51.0%, with 31.6% of answers hitting the token limit." "This checkpoint and the re-run above were built by akhaliq". |
| Singh, "Weight-Space Programming from scratch on simple tasks" (csinva.io) | primary, dated November 2, 2026 | "using Claude. Claude, running as an autonomous coding agent ... Across 13 tasks Claude wrote 959 weight-space programs. All ten algorithmic tasks reached 100% ... within three attempts; five on the first attempt." No Claude version is named. Language tasks: the evaluation examples were sampled from a pool the agent could read, so sentiment and inference results are built from statistics that include the evaluation examples (author's own caveat). The run folders named "may15" are therefore from an earlier date than the report. |
| arXiv dates | primary | RASP 2106.06981 v1 13 Jun 2021; Tracr 2301.05062 v1 12 Jan 2023, v5 3 Nov 2023; Arditi 2406.11717 v1 17 Jun 2024, v3 30 Oct 2024; NeuroGen 2505.12470 v1 18 May 2025, v2 23 May 2025; Text2Weight 2508.13633 19 Aug 2025; survey 2603.10090 10 Mar 2026. |
| Latent Space AINews recap | primary page | The page carries the date 2026-11-02, not 2026-11-03 as the brief states; it files the item under "Weight-Space Editing and Model Steering". |
| Pulse Physiology Engine front page | primary | "Distributed under the Apache License, Version 2.0." Version, drug list and asthma/COPD details are on documentation pages not read in this session. |
| NASEMSO site | not retrievable | nasemso.org returned 403 to the session's fetches. |
| Keruac X thread | not retrievable | x.com requires a browser; not read. |
