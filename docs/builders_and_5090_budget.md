# Local builders and the RTX 5090 budget (condensed; full research notes in `appendix/B_builders_and_hardware_raw.md`)

## Hardware (secondary sources, agreeing)
RTX 5090: 32 GB GDDR7, 1,792 GB/s, GB202, compute capability 12.0 (sm_120), CUDA 12.8+, FP4 tensor
cores, 575 W. vLLM: CUDA 12.9 wheels, CC >= 7.5. llama.cpp: FlashAttention kernels for q4_0/q8_0/f16 KV.

## VRAM budget for the builder (KV = layers_with_cache x 2 x kv_heads x head_dim x 2 bytes x tokens)

| Model | Q4 weights | KV per token | KV at 32k / 64k | Buffers | Total at 64k | Headroom for a 3-4B dialogue model (~2.5 GB) |
|---|---|---|---|---|---|---|
| Qwen3.8-27B (16 of 64 layers cache KV, 4 x 256) | ~17 GB | 64 KiB | 2 / 4 GiB | ~2 GiB | ~23 GiB | yes |
| Qwen3.6-35B-A3B (10 of 40, 2 x 256) | ~19 GB | 20 KiB | 0.6 / 1.25 GiB | ~2 GiB | ~22 GiB | yes |
| gpt-oss-20b (12 global of 24, 8 x 64) | 11.3 GiB | 24 KiB | 0.75 / 1.5 GiB | 2.7 GiB (measured) | ~15.5 GiB | yes, with room for two |
| Qwen3-32B dense (64 layers, 8 x 128) | 18.4 GiB | 256 KiB | 8 / 16 GiB | ~2 GiB | 36 GiB (no) / 27 GiB with q8_0 KV | no |
| gpt-oss-120b | 59 GiB MXFP4 | 36 KiB | 1.1 / 2.25 GiB | | CPU offload (`--n-cpu-moe`), ~30 t/s | sequential only |

Verification: Qwen configs from HF config snippets (secondary) and llama.cpp discussions (primary);
gpt-oss from the model source (primary); 5090 throughput rows primary where marked in the appendix.

## Published 5090 throughput (tokens/s)
gpt-oss-20b MXFP4: 282 decode, 6,291 prompt at 32k (primary). Qwen3.5-35B-A3B Q4: 194 decode,
7,026 prompt (primary). Qwen3 32B Q4: 44-61 decode (secondary). Qwen3.6-27B Q4: ~77 decode (snippet).

## Fine-tuning feasibility
QLoRA minimums (Unsloth, secondary): 7B 5 GB, 14B 8.5 GB, 32B 26 GB; Qwen3-30B-A3B 17.5 GB;
gpt-oss-20b 14 GB. Unsloth advises against QLoRA for Qwen3.5-series (16-bit LoRA on 27B needs
60-76 GB: does not fit). Full fine-tuning ~16 bytes/param: infeasible for any candidate.
Conclusion: inference yes, builder fine-tuning no.

## Dialogue models for academy laptops (CPU only)
Qwen3.5 2B/4B (Apache-2.0), Qwen3-4B-2507 (12.5 t/s at 1k context on a Core Ultra 9 185H, 4.6 t/s at
8k; third-party primary), Gemma 4 E2B/E4B (Apache-2.0), Phi-4-mini (MIT), SmolLM3-3B (Apache-2.0).
Structured output: llama.cpp GBNF/JSON-schema; vLLM xgrammar/guidance. Expect 8-20 t/s for a 3-4B Q4
model on a 2024-2026 business laptop at short context (inference, not measurement). A 250-token
patient reply at 10 t/s is 25 s, too slow for conversation; keep replies to 20-40 tokens or use the
5090 as a shared dialogue server (4-8 concurrent students) and measure with `scripts/profile_local.py --llm`.

## Reality check
No model that fits a 5090 appears on MLE-bench; no open-weight RE-Bench results found. Closest proxies:
Terminal-Bench 2.x (Qwen3.8-27B 73.0 vendor; Qwen3.6-27B 59.3 vendor) and small independent tool-loop
tests (Qwen3.8-27B 30/30). Independent SWE-bench reproductions run 15-20 points below vendor claims.
Treat all vendor numbers as upper bounds and run `experiments/exp1_calibration`.

## Profiling procedure (so numbers are measured, not invented)
```
python scripts/profile_local.py                       # engine, concurrency, PK targets (CPU; torch/GPU if present)
python scripts/profile_local.py --llm http://127.0.0.1:8080/v1 --model <id>   # dialogue latency at concurrency 1/4/8
```
Run once on the 5090 box and once on an academy Lenovo; commit the two `profile_<host>.json` files.
