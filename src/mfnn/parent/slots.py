"""The slot ledger: what the parent must know before it may build, and how sure it is.

Completeness = every required slot filled with confidence >= threshold, no blocking unsupported
requests. The threshold is the instructor-set 'confidence level'. Slots are plain data so the ledger
can be stored, diffed, graded and used as training targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SLOT_DEFS: Dict[str, Dict[str, Any]] = {
    # physiology-facing (map to preset knobs)
    "family": {"required": True, "type": "enum", "values": ["asthma_adult_acute"], "ask": "Which kind of patient is this? (currently supported: adult acute asthma)"},
    "severity_on_arrival": {"required": True, "type": "enum", "values": ["moderate", "severe", "life_threatening"], "ask": "How sick is the patient when the crew arrives?"},
    "deterioration_pace": {"required": True, "type": "enum", "values": ["slow", "typical", "fast"], "ask": "If nobody treats them, how fast should they get worse?"},
    "treatment_responsiveness": {"required": True, "type": "enum", "values": ["poor", "typical", "good"], "ask": "How well should bronchodilators work on this patient?"},
    "traps": {"required": False, "type": "list", "values": ["sedation", "nitroglycerin_misdiagnosis", "over_ventilation", "cpap_without_airway_protection", "supine_positioning"], "ask": "Any wrong moves you want the scenario to punish?"},
    "duration_min": {"required": True, "type": "number", "range": [15, 120], "ask": "How long should the scenario run?"},
    "checkpoints": {"required": False, "type": "list", "ask": "Any specific numbers you expect at specific times (e.g. sats 88 by 10 minutes untreated)?"},
    # patient-facing
    "age": {"required": True, "type": "number", "range": [18, 100], "ask": "How old is the patient?"},
    "sex": {"required": True, "type": "enum", "values": ["female", "male"], "ask": "Sex?"},
    "weight_kg": {"required": True, "type": "number", "range": [35, 250], "ask": "Approximate weight in kg?"},
    "chief_complaint": {"required": True, "type": "text", "ask": "What does the patient say is wrong, in their words?"},
    "history_of_present_illness": {"required": True, "type": "text", "ask": "How did this start and what happened before the crew arrived?"},
    "past_medical_history": {"required": True, "type": "list", "ask": "Past medical history?"},
    "medications": {"required": True, "type": "list", "ask": "Home medications?"},
    "allergies": {"required": True, "type": "list", "ask": "Allergies?"},
    "last_oral_intake": {"required": False, "type": "text", "ask": "Last oral intake?"},
    "persona": {"required": True, "type": "text", "ask": "How does the patient come across: anxious, calm, irritable, minimising?"},
    "scene": {"required": False, "type": "text", "ask": "Where is the call and who is there?"},
    # teaching-facing
    "teaching_objectives": {"required": True, "type": "list", "ask": "What should students learn from this run?"},
    "audience": {"required": False, "type": "text", "ask": "Who is this for (year, level)?"},
}
REQUIRED = [k for k, v in SLOT_DEFS.items() if v["required"]]


@dataclass
class SlotLedger:
    slots: Dict[str, Dict[str, Any]] = field(default_factory=dict)      # name -> {"value":..., "confidence":..., "source_turn": n}
    unsupported: List[str] = field(default_factory=list)                 # requests the engine cannot model (blocking until acknowledged)
    history: List[Dict[str, Any]] = field(default_factory=list)          # per-turn snapshots for learning

    def update(self, updates: Dict[str, Dict[str, Any]], turn: int) -> None:
        for name, u in (updates or {}).items():
            if name not in SLOT_DEFS or not isinstance(u, dict) or "value" not in u:
                continue
            conf = float(u.get("confidence", 0.5)); conf = max(0.0, min(1.0, conf))
            self.slots[name] = {"value": u["value"], "confidence": conf, "source_turn": turn}
        self.history.append({"turn": turn, "slots": {k: dict(v) for k, v in self.slots.items()}})

    def completeness(self, threshold: float) -> Dict[str, Any]:
        missing = [k for k in REQUIRED if k not in self.slots]
        low = [k for k in REQUIRED if k in self.slots and self.slots[k]["confidence"] < threshold]
        invalid = []
        for k, s in self.slots.items():
            d = SLOT_DEFS[k]; v = s["value"]
            if d["type"] == "enum" and v not in d["values"]: invalid.append(k)
            if d["type"] == "number":
                try:
                    x = float(v); lo, hi = d.get("range", [-1e9, 1e9])
                    if not (lo <= x <= hi): invalid.append(k)
                except (TypeError, ValueError): invalid.append(k)
            if d["type"] == "list" and d.get("values") and isinstance(v, list) and any(x not in d["values"] for x in v): invalid.append(k)
        filled = sum(1 for k in REQUIRED if k in self.slots and self.slots[k]["confidence"] >= threshold and k not in invalid)
        return {"score": filled / len(REQUIRED), "missing": missing, "low_confidence": low, "invalid": invalid,
                "unsupported": list(self.unsupported), "complete": not missing and not low and not invalid and not self.unsupported}

    def values(self) -> Dict[str, Any]:
        return {k: v["value"] for k, v in self.slots.items()}
