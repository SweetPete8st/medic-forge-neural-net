# Builder calibration, Task C (PK compile): Claude family, 2026-11-04

Six independent, isolated attempts (fresh context each; dev-only spec, layer library, scorer returning aggregates only; TASK.md named the useful constructions and warned about knot placement relative to EC50; 25-run budget). Every attempt used 4 Python runs (1 build + 3 scorer) and passed the development criterion (max |dE| <= 0.01) on its first build. Held-out evaluation was run afterwards by the session with the attempts' own build scripts on held-out parameter variants (names relabelled) and all schedule blocks.

| Attempt | Dev pass | Unseen schedules (dev variants) fails/30 | Unseen variants fails/108 | Worst held-out dE | Params (typical) | Build on extremes |
|---|---|---|---|---|---|---|
| Claude Fable 5.1 (subagent) | yes | 0 | 4/108 | 0.0755 | 906 | ok |
| Claude Opus (subagent) | yes | 0 | 0/108 | 0.0000 | 5040 | ok |
| Claude Sonnet (subagent) | yes | 6 | 14/108 | 0.3431 | 490 | ok |
| Claude Sonnet (subagent, 2nd) | yes | 5 | 5/108 | 0.0901 | 646 | ok |
| Claude Haiku (subagent) | yes | 6 | 4/54 | 0.0632 | 7900 | FAILED (no step satisfies TOL for EC50 0.1) |
| Claude Haiku (subagent, 2nd) | yes | 0 | 0/108 | 0.0030 | 1556 | ok |

**Reading.** Reliability on the development criterion: 6/6 first-try. Robustness to held-out extremes: Opus and the second Haiku attempt were exact everywhere (knots placed from an explicit chord-error bound out to the maximum possible concentration); Fable failed 4 cases by linear extrapolation beyond its last knot (50 x EC50) on the 25-unit dose for the most potent variants; Sonnet (both), and the first Haiku, placed knots only up to the development schedules' maximum concentration (6 to 12) and failed on the overdose and 25-unit schedules; the first Haiku's step chooser refused to build for EC50 = 0.1. The failure mode is the same reasoned-constant failure seen in exp0 v1, and it did not track model tier.
**Caveats.** All six are Claude models (same family as the author of the task); the task brief contained hints; the scorer exposed only aggregate numbers but the builders could read score.py and the reference rollout inside it; isolation from the repository was instructed and self-reported, not enforced; the task is small (4 state variables, linear kinetics). None of this says anything about open-weight local builders, which is exp1 Tasks A to C on the 5090.

Raw results: claude_family_attempts/heldout_results.json; per-attempt build scripts and construction notes were retained in the session scratchpad.