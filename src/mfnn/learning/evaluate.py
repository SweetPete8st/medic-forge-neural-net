"""Evaluate a parent (backend + prompt version) on fixed briefs with a simulated instructor.

A brief is a hidden target: the instructor persona knows the answers and reveals them when asked.
The simulated instructor is itself a backend (ideally a different model family; the family is recorded
so same-family evaluations are visibly flagged). Metrics: turns to completion, completeness at the
end, slot accuracy against the brief, gate pass, unsupported requests handled, parse errors."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from mfnn.parent.interview import Interview
from mfnn.parent.slots import REQUIRED

INSTRUCTOR_SYSTEM = """You are a paramedic instructor being interviewed by a scenario-builder assistant. You have a target patient in mind (given below as JSON). Answer the builder's questions briefly and naturally, in an instructor's voice, revealing only what is asked. If asked something not in your target, improvise plausibly and consistently. Never output JSON; just talk. Start by describing the patient in two sentences."""


def simulate(parent_backend, instructor_backend, brief: Dict[str, Any], threshold: float = 0.8, max_turns: int = 14, out_root: Path | None = None) -> Dict[str, Any]:
    iv = Interview(parent_backend, brief.get("preset", "asthma_adult_acute"), threshold, scenario_id=f"eval_{brief['id']}_{int(time.time())}")
    imsgs: List[Dict[str, str]] = [{"role": "user", "content": "Builder: " + "Let's get started. Tell me about the patient you want."}]
    sys_i = INSTRUCTOR_SYSTEM + "\nTARGET:\n" + json.dumps(brief["target"], ensure_ascii=False)
    done = False; turns = 0; parse_errors = 0
    while turns < max_turns and not done:
        itext = instructor_backend.complete(sys_i, imsgs, max_tokens=400)
        imsgs.append({"role": "assistant", "content": itext})
        r = iv.step(itext); turns += 1
        parse_errors += int(iv.transcript[-1].get("parse_error", False))
        imsgs.append({"role": "user", "content": "Builder: " + r["say"]})
        done = r["done"]
    comp = iv.ledger.completeness(threshold); got = iv.ledger.values(); tgt = brief["target"]
    enum_keys = [k for k in tgt if k in ("severity_on_arrival", "deterioration_pace", "treatment_responsiveness", "sex")]
    slot_acc = sum(1 for k in enum_keys if str(got.get(k)) == str(tgt[k])) / max(1, len(enum_keys))
    res = {"brief": brief["id"], "turns": turns, "completed": done, "completeness": comp["score"], "slot_accuracy_enum": slot_acc,
           "parse_errors": parse_errors, "unsupported": iv.ledger.unsupported, "parent": f"{parent_backend.name}:{parent_backend.model}",
           "instructor": f"{instructor_backend.name}:{instructor_backend.model}", "same_family_warning": parent_backend.name == instructor_backend.name}
    if out_root is not None:
        try:
            b = iv.build(out_root); res["gate_pass"] = b["gate"]["pass"]; res["path"] = b["path"]
        except Exception as e:
            res["gate_pass"] = False; res["build_error"] = f"{e.__class__.__name__}: {e}"
    return res


def run_eval(parent_backend, instructor_backend, briefs: List[Dict[str, Any]], out_root: Path, threshold: float = 0.8) -> Dict[str, Any]:
    rows = [simulate(parent_backend, instructor_backend, b, threshold, out_root=out_root) for b in briefs]
    n = len(rows)
    summary = {"n": n, "completed_rate": sum(r["completed"] for r in rows) / n, "gate_pass_rate": sum(bool(r.get("gate_pass")) for r in rows) / n,
               "mean_turns": sum(r["turns"] for r in rows) / n, "mean_slot_accuracy_enum": sum(r["slot_accuracy_enum"] for r in rows) / n,
               "parse_errors": sum(r["parse_errors"] for r in rows), "same_family_warning": any(r["same_family_warning"] for r in rows),
               "parent": rows[0]["parent"] if rows else None, "rows": rows}
    return summary
