"""Import legacy Medic Forge transcripts (instructor<->builder interviews, student<->patient dialogues).

Accepted inputs (liberal on purpose; adapt `parse_any` once a real sample is available):
- JSON / JSONL: {"id", "kind", "turns": [{"role", "text", "t"}], ...} or a list of such; role names are
  normalised: instructor|user|teacher -> instructor; builder|assistant|model|ai -> builder;
  student|learner|medic -> student; patient|sim|npc -> patient.
- plain text: lines like "Student: ..." / "Patient: ..." / "[12:01] Instructor: ...".
- CSV with columns speaker,text[,time].
Every record is de-identified with `scrub()` before storage (emails, phones, SSN-like, dates of birth,
and an optional list of names). De-identification is best-effort; review before training on it.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROLE_MAP = {"instructor": "instructor", "user": "instructor", "teacher": "instructor", "proctor": "instructor",
            "builder": "builder", "assistant": "builder", "model": "builder", "ai": "builder", "claude": "builder",
            "student": "student", "learner": "student", "medic": "student", "paramedic": "student", "emt": "student",
            "patient": "patient", "sim": "patient", "npc": "patient", "simulated patient": "patient", "system": "system"}
_LINE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?([A-Za-z ]{2,24}):\s*(.+)$")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"); _PHONE = re.compile(r"\b(?:\+?1[ -.]?)?\(?\d{3}\)?[ -.]?\d{3}[ -.]?\d{4}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b"); _DOB = re.compile(r"\b(?:DOB|date of birth)\s*[:=]?\s*[\d/.-]+\b", re.I)


def scrub(text: str, names: Iterable[str] = ()) -> str:
    text = _EMAIL.sub("[email]", text); text = _PHONE.sub("[phone]", text); text = _SSN.sub("[id]", text); text = _DOB.sub("[dob]", text)
    for n in names:
        if n and len(n) > 1: text = re.sub(rf"\b{re.escape(n)}\b", "[name]", text, flags=re.I)
    return text


def _norm_role(r: str) -> str:
    return ROLE_MAP.get(str(r).strip().lower(), str(r).strip().lower() or "unknown")


def infer_kind(turns: List[Dict[str, Any]]) -> str:
    roles = {t["role"] for t in turns}
    if roles & {"student", "patient"}: return "patient_dialogue"
    if roles & {"instructor", "builder"}: return "builder_interview"
    return "unknown"


def parse_text(text: str) -> List[Dict[str, Any]]:
    turns = []
    for line in text.splitlines():
        m = _LINE.match(line)
        if m: turns.append({"role": _norm_role(m.group(1)), "text": m.group(2).strip()})
        elif turns and line.strip(): turns[-1]["text"] += " " + line.strip()
    return turns


def parse_any(path: Path) -> List[Dict[str, Any]]:
    """Return a list of records {id, kind, turns, source_file, metadata}."""
    p = Path(path); recs: List[Dict[str, Any]] = []
    if p.suffix.lower() in (".json", ".jsonl"):
        raw = p.read_text()
        items = [json.loads(l) for l in raw.splitlines() if l.strip()] if p.suffix.lower() == ".jsonl" else json.loads(raw)
        if isinstance(items, dict): items = items.get("transcripts") or items.get("records") or [items]
        for i, it in enumerate(items):
            turns = [{"role": _norm_role(t.get("role") or t.get("speaker") or "unknown"), "text": str(t.get("text") or t.get("content") or ""), "t": t.get("t") or t.get("time")}
                     for t in (it.get("turns") or it.get("messages") or [])]
            recs.append({"id": str(it.get("id") or f"{p.stem}-{i}"), "kind": it.get("kind") or infer_kind(turns), "turns": turns, "source_file": p.name,
                         "metadata": {k: v for k, v in it.items() if k not in ("turns", "messages")}})
    elif p.suffix.lower() == ".csv":
        with p.open(newline="") as f:
            rows = list(csv.DictReader(f))
        turns = [{"role": _norm_role(r.get("speaker") or r.get("role") or ""), "text": r.get("text") or "", "t": r.get("time")} for r in rows]
        recs.append({"id": p.stem, "kind": infer_kind(turns), "turns": turns, "source_file": p.name, "metadata": {}})
    else:
        turns = parse_text(p.read_text(errors="replace"))
        recs.append({"id": p.stem, "kind": infer_kind(turns), "turns": turns, "source_file": p.name, "metadata": {}})
    return recs


def import_legacy(paths: Iterable[Path], store, names: Iterable[str] = (), kind_override: Optional[str] = None) -> Dict[str, int]:
    counts = {"files": 0, "records": 0, "builder_interview": 0, "patient_dialogue": 0, "unknown": 0}
    for p in paths:
        counts["files"] += 1
        for rec in parse_any(Path(p)):
            for t in rec["turns"]: t["text"] = scrub(t["text"], names)
            if kind_override: rec["kind"] = kind_override
            rec["pii_scrubbed"] = True
            store.add_interview({"kind": rec["kind"], "legacy": rec, "transcript": [{"role": t["role"], "text": t["text"]} for t in rec["turns"]]}, gate=None, source="legacy")
            counts["records"] += 1; counts[rec["kind"]] = counts.get(rec["kind"], 0) + 1
    return counts
