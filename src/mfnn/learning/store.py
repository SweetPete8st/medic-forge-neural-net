"""Append-only JSONL store for interviews, corrections, grades and outcomes. No PII policy is enforced
here beyond what the importer scrubs; keep the store out of public repos (see docs/10_PARENT_AGENT.md)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class InterviewStore:
    def __init__(self, root: Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.files = {k: self.root / f"{k}.jsonl" for k in ("interviews", "corrections", "grades", "outcomes")}

    def _append(self, kind: str, rec: Dict[str, Any]) -> str:
        rec = dict(rec); rec.setdefault("stored_at", time.time())
        rec.setdefault("id", hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()[:12])
        with self.files[kind].open("a") as f: f.write(json.dumps(rec, default=str) + "\n")
        return rec["id"]

    def add_interview(self, record: Dict[str, Any], gate: Optional[Dict[str, Any]] = None, source: str = "live") -> str:
        return self._append("interviews", {"kind": record.get("kind", "builder_interview"), "source": source, "record": record,
                                           "gate_pass": bool(gate and gate.get("pass")), "gate": gate})

    def add_correction(self, scenario_id: str, text: str, before: Dict[str, Any], after: Dict[str, Any], author: str = "instructor") -> str:
        """A correction is what the instructor said after seeing a preview, and what changed because of it."""
        return self._append("corrections", {"scenario_id": scenario_id, "text": text, "before": before, "after": after, "author": author})

    def add_grade(self, record_id: str, grader: str, scores: Dict[str, float], comment: str = "", method: str = "human") -> str:
        return self._append("grades", {"record_id": record_id, "grader": grader, "method": method, "scores": scores, "comment": comment})

    def add_outcome(self, scenario_id: str, session_log: Dict[str, Any]) -> str:
        return self._append("outcomes", {"scenario_id": scenario_id, "session": session_log})

    def iter(self, kind: str) -> Iterator[Dict[str, Any]]:
        p = self.files[kind]
        if not p.exists(): return iter(())
        return (json.loads(l) for l in p.read_text().splitlines() if l.strip())

    def grades_for(self, record_id: str) -> List[Dict[str, Any]]:
        return [g for g in self.iter("grades") if g["record_id"] == record_id]

    def summary(self) -> Dict[str, int]:
        return {k: sum(1 for _ in self.iter(k)) for k in self.files}
