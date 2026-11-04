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
