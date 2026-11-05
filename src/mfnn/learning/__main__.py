"""One-line commands for the learning pathway:
  python -m mfnn.learning import <files...> [--kind patient_dialogue|builder_interview] [--names "Name One" "Name Two"]
  python -m mfnn.learning grade --grader David        # walks through ungraded records, asks rubric scores
  python -m mfnn.learning dataset [--min-grade 4]     # writes data/sft/*.jsonl from graded records
  python -m mfnn.learning summary
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from .dataset import build_sft
from .grading import RUBRIC, auto_grade, human_grade_prompt
from .importer import import_legacy
from .store import InterviewStore

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = Path.cwd() / "data" / "interviews"


def cmd_import(a):
    files = [Path(p) for pat in a.files for p in sorted(glob.glob(pat))]
    if not files: sys.exit("no files matched")
    store = InterviewStore(Path(a.store))
    print(import_legacy(files, store, names=a.names or [], kind_override=a.kind))
    print("store now:", store.summary())


def cmd_grade(a):
    store = InterviewStore(Path(a.store)); n = 0
    for rec in store.iter("interviews"):
        if store.grades_for(rec["id"]): continue
        kind = rec.get("kind", "unknown"); items = RUBRIC.get(kind)
        if not items:
            print(f"skip {rec['id']}: kind {kind} has no rubric (set --kind on import)"); continue
        r = rec.get("record", {}); shown = {**rec, "transcript": r.get("transcript", []), "id": rec["id"]}
        print("\n" + "=" * 78); print(human_grade_prompt(shown)); print("auto:", auto_grade({**rec, **r}))
        scores = {}
        for k in items:
            while True:
                v = input(f"  {k} (1-5, s=skip record, q=quit): ").strip().lower()
                if v in ("q", "s"): break
                if v.isdigit() and 1 <= int(v) <= 5: scores[k] = int(v); break
            if v == "q": print("stopped."); return
            if v == "s": scores = None; break
        if scores is None: continue
        comment = input("  comment (optional): ").strip()
        store.add_grade(rec["id"], a.grader, scores, comment); n += 1
        print(f"  saved ({n} graded this session)")
    print("no more ungraded records.")


def cmd_dataset(a):
    store = InterviewStore(Path(a.store)); out = Path(a.out)
    print(build_sft(store, out, min_grade=a.min_grade, require_gate=not a.no_gate)); print("wrote", out)


def cmd_summary(a):
    store = InterviewStore(Path(a.store)); s = store.summary()
    graded = sum(1 for r in store.iter("interviews") if store.grades_for(r["id"]))
    kinds = {}
    for r in store.iter("interviews"): kinds[r.get("kind", "unknown")] = kinds.get(r.get("kind", "unknown"), 0) + 1
    print({**s, "graded_interviews": graded, "by_kind": kinds})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mfnn.learning"); ap.add_argument("--store", default=str(DEFAULT_STORE))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import"); p.add_argument("files", nargs="+"); p.add_argument("--kind", default=None, choices=[None, "patient_dialogue", "builder_interview"]); p.add_argument("--names", nargs="*"); p.set_defaults(f=cmd_import)
    p = sub.add_parser("grade"); p.add_argument("--grader", required=True); p.set_defaults(f=cmd_grade)
    p = sub.add_parser("dataset"); p.add_argument("--out", default=str(Path.cwd() / "data/sft/builder_and_voice_v1.jsonl")); p.add_argument("--min-grade", type=float, default=4.0); p.add_argument("--no-gate", action="store_true"); p.set_defaults(f=cmd_dataset)
    p = sub.add_parser("summary"); p.set_defaults(f=cmd_summary)
    a = ap.parse_args(argv); a.f(a)


if __name__ == "__main__":
    main()
