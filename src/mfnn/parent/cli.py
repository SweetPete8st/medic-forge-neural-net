"""Terminal front end: `python -m mfnn.parent --backend anthropic` (or --backend local --model qwen3.5:9b)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .interview import OPENING, Interview
from .llm import make_backend

ROOT = Path(__file__).resolve().parents[3]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Medic Forge scenario builder (parent agent)")
    ap.add_argument("--backend", default="anthropic", choices=["anthropic", "local"]); ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None); ap.add_argument("--effort", default="medium")
    ap.add_argument("--preset", default="asthma_adult_acute"); ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--out", default=str(ROOT / "scenarios")); ap.add_argument("--compile-brain", action="store_true")
    ap.add_argument("--store", default=str(ROOT / "data/interviews"), help="append the interview record here for learning")
    a = ap.parse_args(argv)
    backend = make_backend(a.backend, a.model, a.base_url, a.effort)
    iv = Interview(backend, a.preset, a.threshold)
    print(f"\n{OPENING}\n(type 'build' when ready, 'quit' to stop)\n")
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if not text: continue
        if text.lower() in ("quit", "exit"): break
        if text.lower() == "build":
            res = iv.build(Path(a.out), compile_brain=a.compile_brain)
            g = res["gate"]; print(f"\nbuilt {res['path']}  gate: {'PASS' if g['pass'] else 'NOT PASSED'}  completeness {g['completeness']['score']:.0%}")
            for name, t in g["trajectories"].items():
                print(f"  {name:<26} end SpO2 {t['end']['SpO2']:>5} HR {t['end']['HR']:>5} RR {t['end']['RR']:>4} PaCO2 {t['end']['PaCO2']:>5} LOC {t['end']['LOC']:>3}  arrest {t['arrest_min']}")
            if not g["pass"]:
                print("  not passed:", {k: v for k, v in g.items() if k in ("schema_error", "objectives", "treatment_helps")}, g["completeness"])
            from mfnn.learning.store import InterviewStore
            InterviewStore(Path(a.store)).add_interview(res["record"], gate=g)
            print("  (say more to adjust, then 'build' again; 'quit' to stop)\n")
            continue
        r = iv.step(text)
        print(f"\nbuilder> {r['say']}\n   [completeness {r['completeness']['score']:.0%}; missing {r['completeness']['missing'][:3]}]\n")


if __name__ == "__main__":
    main()
