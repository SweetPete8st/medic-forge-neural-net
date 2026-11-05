"""Turn graded records into training examples (chat JSONL, Unsloth/HF compatible).

Builder-interview examples: (system + conversation prefix) -> the builder's JSON turn. Only interviews
with a human grade >= min_grade on every rubric item, and (for live records) a passing gate, are used.
Patient-dialogue examples: (fact sheet + speech mode + student line) -> patient reply, only from
dialogues graded consistent; legacy dialogues without a fact sheet are emitted as 'voice-style'
examples with an explicit flag so they can be excluded from fact-grounded training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from mfnn.parent.interview import SYSTEM


def _human_ok(store, rec_id: str, min_grade: float) -> bool:
    grades = [g for g in store.grades_for(rec_id) if g.get("method", "human") == "human"]
    if not grades: return False
    latest = grades[-1]["scores"]
    return bool(latest) and min(latest.values()) >= min_grade


def build_sft(store, out_path: Path, min_grade: float = 4.0, require_gate: bool = True) -> Dict[str, int]:
    counts = {"builder_examples": 0, "voice_examples": 0, "skipped_ungraded": 0, "skipped_low_grade": 0, "skipped_gate": 0}
    examples: List[Dict[str, Any]] = []
    for rec in store.iter("interviews"):
        if not store.grades_for(rec["id"]):
            counts["skipped_ungraded"] += 1; continue
        if not _human_ok(store, rec["id"], min_grade):
            counts["skipped_low_grade"] += 1; continue
        kind = rec.get("kind"); r = rec.get("record") or {}
        turns = r.get("transcript") or []
        if kind == "builder_interview":
            if require_gate and rec.get("source") == "live" and not rec.get("gate_pass"):
                counts["skipped_gate"] += 1; continue
            msgs = [{"role": "system", "content": SYSTEM}]
            for t in turns:
                if t.get("role") == "instructor":
                    msgs.append({"role": "user", "content": t["text"]})
                elif t.get("role") == "builder" and msgs[-1]["role"] == "user":
                    target = json.dumps({"say": t["text"], "slots": t.get("slots", {}), "done": t.get("done", False)}, ensure_ascii=False)
                    examples.append({"messages": msgs + [{"role": "assistant", "content": target}], "kind": kind, "record_id": rec["id"]})
                    msgs = msgs + [{"role": "assistant", "content": target}]; counts["builder_examples"] += 1
        elif kind == "patient_dialogue":
            sheet = r.get("fact_sheet"); sys_prompt = ("You are the simulated patient. Speak only from this fact sheet and speech mode:\n" + json.dumps(sheet, ensure_ascii=False)) if sheet else "You are a simulated patient (voice-style example, no fact sheet)."
            msgs = [{"role": "system", "content": sys_prompt}]
            for t in turns:
                if t.get("role") == "student":
                    msgs.append({"role": "user", "content": t["text"]})
                elif t.get("role") == "patient" and msgs[-1]["role"] == "user":
                    examples.append({"messages": msgs + [{"role": "assistant", "content": t["text"]}], "kind": kind, "record_id": rec["id"], "fact_grounded": bool(sheet)})
                    msgs = msgs + [{"role": "assistant", "content": t["text"]}]; counts["voice_examples"] += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ex in examples: f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    counts["written"] = len(examples)
    return counts
