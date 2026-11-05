"""Promotion gate ("continuous approval"): a candidate parent version becomes current only when its
evaluation beats the current one by a margin AND a named human approves. Registry is a JSON file."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from mfnn.parent.interview import SYSTEM

MARGINS = {"completed_rate": 0.0, "gate_pass_rate": 0.0, "mean_slot_accuracy_enum": 0.0}   # candidate must be >= current on each
TIEBREAK = "mean_turns"                                                                     # and not worse by more than 1 turn


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM.encode()).hexdigest()[:12]


class Registry:
    def __init__(self, path: Path):
        self.path = Path(path); self.data = json.loads(self.path.read_text()) if self.path.exists() else {"current": None, "versions": []}

    def save(self): self.path.parent.mkdir(parents=True, exist_ok=True); self.path.write_text(json.dumps(self.data, indent=1))

    def register(self, backend_kind: str, model: str, eval_summary: Dict[str, Any], note: str = "", adapter: Optional[str] = None) -> str:
        vid = f"v{len(self.data['versions']) + 1}"
        self.data["versions"].append({"id": vid, "backend": backend_kind, "model": model, "adapter": adapter, "prompt_hash": prompt_hash(),
                                      "eval": {k: v for k, v in eval_summary.items() if k != "rows"}, "note": note, "registered_at": time.time(), "approved_by": None, "approved_at": None})
        self.save(); return vid

    def compare(self, vid: str) -> Dict[str, Any]:
        cand = next(v for v in self.data["versions"] if v["id"] == vid); cur = self.current()
        if cur is None: return {"eligible": True, "reason": "no current version", "deltas": {}}
        deltas = {k: cand["eval"].get(k, 0) - cur["eval"].get(k, 0) for k in MARGINS}
        ok = all(deltas[k] >= -1e-9 - MARGINS[k] for k in MARGINS) and (cand["eval"].get(TIEBREAK, 99) - cur["eval"].get(TIEBREAK, 99)) <= 1.0
        warn = cand["eval"].get("same_family_warning") or cur["eval"].get("same_family_warning")
        return {"eligible": ok, "deltas": deltas, "same_family_warning": bool(warn), "reason": "" if ok else "candidate is worse than current on a gated metric"}

    def approve(self, vid: str, approver: str) -> Dict[str, Any]:
        if not approver or approver.lower() in ("ai", "claude", "model", "auto"): raise ValueError("approval requires a named human")
        cmp = self.compare(vid)
        if not cmp["eligible"]: raise ValueError(f"not eligible: {cmp['reason']}")
        v = next(v for v in self.data["versions"] if v["id"] == vid); v["approved_by"] = approver; v["approved_at"] = time.time()
        self.data["current"] = vid; self.save(); return v

    def current(self) -> Optional[Dict[str, Any]]:
        return next((v for v in self.data["versions"] if v["id"] == self.data["current"]), None) if self.data["current"] else None
