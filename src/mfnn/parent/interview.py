"""The interview loop. The LLM drives the conversation; the ledger, composer and gate are deterministic.

Each model turn returns JSON: {"say": str, "slots": {name: {"value", "confidence"}}, "unsupported": [str], "done": bool}.
`done` is advisory; the gate decides. Stored for learning: every turn, every ledger snapshot, every gate report.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .compose import build_overlay, compose, load_preset
from .gate import run_gate
from .llm import Backend
from .slots import REQUIRED, SLOT_DEFS, SlotLedger

OPENING = "Let's get started. Tell me about the patient you want, in your own words, and I'll ask about anything I still need."

SYSTEM = """You are the scenario builder for Medic Forge, an EMS education simulator. You interview an instructor to fill a slot ledger that configures a shared, physician-reviewed physiology engine. You do not invent physiology: you only set the knobs the preset offers, and you tell the instructor plainly when they ask for something the engine cannot model (put it in "unsupported").

Rules:
- Ask about at most two things per turn, in plain language a paramedic instructor uses. Never ask what is already filled.
- Accept the instructor's own words for facts (chief complaint, history, meds, allergies, persona); normalise choices to the allowed values for enum slots.
- Give each slot a confidence 0-1 reflecting how directly the instructor stated it (1.0 explicit, 0.6 inferred, 0.3 guessed). Do not fill a slot you are guessing.
- When every required slot is filled, summarise the patient in three sentences and set "done": true.
- Reply ONLY with a JSON object: {"say": "...", "slots": {"slot_name": {"value": ..., "confidence": 0.0}}, "unsupported": [], "done": false}.
"""


def _slot_doc() -> str:
    rows = []
    for k, d in SLOT_DEFS.items():
        allowed = f" allowed: {d['values']}" if d.get("values") else (f" range: {d['range']}" if d.get("range") else "")
        rows.append(f"- {k} ({d['type']}{', required' if d['required'] else ''}){allowed}: {d['ask']}")
    return "\n".join(rows)


def _parse(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"say": text.strip() or "Could you say more?", "slots": {}, "unsupported": [], "done": False, "_parse_error": True}
    try:
        js = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"say": text.strip()[:500], "slots": {}, "unsupported": [], "done": False, "_parse_error": True}
    js.setdefault("say", ""); js.setdefault("slots", {}); js.setdefault("unsupported", []); js.setdefault("done", False)
    return js


class Interview:
    def __init__(self, backend: Backend, preset_name: str = "asthma_adult_acute", threshold: float = 0.8, scenario_id: Optional[str] = None):
        self.backend, self.threshold = backend, threshold
        self.preset = load_preset(preset_name); self.preset_name = preset_name
        self.ledger = SlotLedger(); self.turn = 0
        self.messages: List[Dict[str, str]] = []
        self.transcript: List[Dict[str, Any]] = [{"role": "builder", "text": OPENING, "t": time.time()}]
        self.scenario_id = scenario_id or f"{preset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.ledger.update({"family": {"value": preset_name, "confidence": 1.0}}, 0)
        self.last_gate: Optional[Dict[str, Any]] = None

    def _system(self) -> str:
        filled = json.dumps(self.ledger.values(), ensure_ascii=False)
        return SYSTEM + "\nSLOTS:\n" + _slot_doc() + f"\n\nPreset knobs you may set: {list(self.preset['knobs'])}. Supported traps: {list(self.preset['traps'])}." + f"\nAlready filled: {filled}"

    def step(self, instructor_text: str) -> Dict[str, Any]:
        """One instructor turn -> one builder turn. Returns the builder's reply and the completeness."""
        self.turn += 1
        self.transcript.append({"role": "instructor", "text": instructor_text, "t": time.time()})
        self.messages.append({"role": "user", "content": instructor_text})
        raw = self.backend.complete(self._system(), self.messages, max_tokens=2048)
        js = _parse(raw)
        self.messages.append({"role": "assistant", "content": raw})
        self.ledger.update(js.get("slots", {}), self.turn)
        for u in js.get("unsupported", []) or []:
            if u and u not in self.ledger.unsupported: self.ledger.unsupported.append(str(u))
        comp = self.ledger.completeness(self.threshold)
        say = js.get("say") or ""
        if comp["complete"] and js.get("done"):
            say += "\n\nI have what I need. Say 'build' to generate the patient and preview, or keep adjusting."
        elif js.get("done") and not comp["complete"]:
            gaps = comp["missing"] + comp["low_confidence"] + comp["invalid"]
            say += f"\n\n(Still need: {', '.join(gaps[:4])})"
        self.transcript.append({"role": "builder", "text": say, "t": time.time(), "slots": js.get("slots", {}), "done": js.get("done", False), "parse_error": js.get("_parse_error", False)})
        return {"say": say, "completeness": comp, "done": bool(js.get("done")) and comp["complete"]}

    def acknowledge_unsupported(self) -> None:
        self.ledger.unsupported.clear()

    def build(self, out_root: Path, compile_brain: bool = False) -> Dict[str, Any]:
        from .deliverable import write_deliverable
        slots = self.ledger.values(); overlay = build_overlay(self.preset, slots); overlay["severity_label"] = str(slots.get("severity_on_arrival", ""))
        spec = compose(self.preset, overlay, self.scenario_id)
        comp = self.ledger.completeness(self.threshold)
        report = run_gate(spec, self.preset, overlay, comp); self.last_gate = report
        record = {"scenario_id": self.scenario_id, "preset": self.preset_name, "threshold": self.threshold, "backend": self.backend.name, "model": self.backend.model,
                  "transcript": self.transcript, "ledger_history": self.ledger.history, "final_slots": self.ledger.slots, "unsupported": self.ledger.unsupported,
                  "calls": [c.__dict__ for c in self.backend.calls], "built_at": time.time()}
        path = write_deliverable(out_root, self.scenario_id, spec, overlay, self.preset_name, report, record, compile_brain=compile_brain)
        return {"path": str(path), "gate": report, "spec": spec, "overlay": overlay, "record": record}
