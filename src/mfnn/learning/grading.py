"""Grading: automatic metrics for every stored record, a human rubric, and an optional LLM pre-grade.
A record only counts as training data when a HUMAN grade exists at or above the threshold; LLM
pre-grades are advisory and are labelled with the grader's model family."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from mfnn.dialogue import check_utterance

RUBRIC = {
    "builder_interview": {
        "efficiency": "Did the builder reach a complete patient in few, well-chosen questions? (1-5)",
        "fidelity": "Did the built patient match what the instructor asked for? (1-5)",
        "honesty": "Did the builder say clearly when something could not be modelled, instead of pretending? (1-5)",
        "clinical_sense": "Were its normalisations and suggestions clinically sensible? (1-5)",
    },
    "patient_dialogue": {
        "consistency": "Did the patient stay consistent with its own history and vitals? (1-5)",
        "realism": "Did it sound like a real patient at that severity (speech limits, affect)? (1-5)",
        "no_leakage": "Did it avoid giving away diagnoses, numbers or instructor information? (1-5)",
    },
}
REQUIRED_SLOT_WORDS = ["age", "allerg", "medic", "history", "complain", "how long", "objective", "learn", "sick", "worse", "respond"]


def auto_grade(record: Dict[str, Any]) -> Dict[str, Any]:
    kind = record.get("kind", "unknown"); turns = record.get("transcript") or []
    builder_turns = [t for t in turns if t.get("role") == "builder"]; other = [t for t in turns if t.get("role") not in ("builder", "patient")]
    out: Dict[str, Any] = {"kind": kind, "n_turns": len(turns), "n_builder_turns": len(builder_turns)}
    if kind == "builder_interview":
        text = " ".join(t.get("text", "").lower() for t in builder_turns)
        out["slot_coverage"] = sum(1 for w in REQUIRED_SLOT_WORDS if w in text) / len(REQUIRED_SLOT_WORDS)
        out["questions_per_turn"] = (sum(t.get("text", "").count("?") for t in builder_turns) / max(1, len(builder_turns)))
        out["parse_errors"] = sum(1 for t in builder_turns if t.get("parse_error"))
        out["gate_pass"] = bool(record.get("gate_pass"))
        out["unsupported_declared"] = len(record.get("record", {}).get("unsupported", []) if isinstance(record.get("record"), dict) else [])
    elif kind == "patient_dialogue":
        sheet = record.get("fact_sheet") or {"speech_mode": "phrases", "facts": {}, "trend": "unchanged", "forbidden": []}
        viol = [check_utterance(t.get("text", ""), sheet) for t in turns if t.get("role") == "patient"]
        out["patient_turns"] = len(viol); out["turns_with_violations"] = sum(1 for v in viol if v)
        out["violation_rate"] = out["turns_with_violations"] / max(1, out["patient_turns"])
        out["numbers_stated"] = sum(1 for t in turns if t.get("role") == "patient" and re.search(r"\b\d{2,3}\s*(%|bpm|over)", t.get("text", ""), re.I))
    return out


def human_grade_prompt(record: Dict[str, Any]) -> str:
    kind = record.get("kind", "unknown"); items = RUBRIC.get(kind, {})
    lines = [f"Record {record.get('id')} ({kind}, {len(record.get('transcript') or [])} turns)", ""]
    for t in (record.get("transcript") or [])[:40]:
        lines.append(f"  {t.get('role', '?'):<10} {t.get('text', '')[:160]}")
    lines += ["", "Rubric (enter 1-5 each):"] + [f"  {k}: {v}" for k, v in items.items()]
    return "\n".join(lines)


def llm_pregrade(record: Dict[str, Any], backend, grader_family: str) -> Dict[str, Any]:
    """Advisory grade from a model. Flagged with the grader family so same-family bias is visible."""
    kind = record.get("kind", "unknown"); items = RUBRIC.get(kind, {})
    convo = "\n".join(f"{t.get('role')}: {t.get('text')}" for t in (record.get("transcript") or [])[:60])
    prompt = (f"Grade this {kind} transcript on each rubric item from 1 to 5. Reply ONLY with JSON {{item: score}} plus a 'comment'.\nRubric: {json.dumps(items)}\n\nTranscript:\n{convo}")
    raw = backend.complete("You are a strict grader for EMS education simulations.", [{"role": "user", "content": prompt}], max_tokens=600)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        js = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        js = {}
    scores = {k: float(js[k]) for k in items if k in js and str(js[k]).replace(".", "", 1).isdigit()}
    return {"method": f"llm:{grader_family}", "scores": scores, "comment": str(js.get("comment", ""))[:500], "advisory": True}
