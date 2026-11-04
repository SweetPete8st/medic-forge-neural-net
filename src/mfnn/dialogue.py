"""Dialogue layer: the patient speaks only from a fact sheet derived from the specification and
the current state. The language model (shared, local) receives the fact sheet and a speech mode;
it never writes state. `check_utterance` is the prototype contradiction detector used by the
"dialogue contradictions" metric in the experiments."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from .engine import PatientEngine

_NUMERIC_VITAL = re.compile(r"\b(\d{2,3})\s*(%|percent|bpm|over\s*\d{2,3}|mmhg|breaths)", re.I)
_MED_WORDS = ["albuterol", "salbutamol", "fluticasone", "prednisone", "inhaler", "insulin", "metformin",
              "aspirin", "warfarin", "lisinopril", "metoprolol", "ventolin", "symbicort", "advair"]
_ALLERGY_WORDS = ["penicillin", "sulfa", "latex", "codeine", "morphine", "peanut", "shellfish", "nsaid", "ibuprofen"]


def fact_sheet(engine: PatientEngine) -> Dict[str, Any]:
    snap = engine.snapshot()
    d = engine.spec["dialogue"]
    o, f = snap["O"], snap["F"]
    symptoms: List[str] = []
    if o > 0.4:
        symptoms.append("chest feels tight, hard to breathe out")
    if o > 0.6:
        symptoms.append("can only manage a few words at a time")
    if f > 0.5:
        symptoms.append("getting tired, exhausted from breathing")
    if snap["SpO2"] < 90:
        symptoms.append("feels dizzy / foggy")
    trend = "worsening" if engine.state.phys["O"] > engine.spec["scenario"]["initial"]["O"] + 0.02 else (
        "improving" if engine.state.phys["O"] < engine.spec["scenario"]["initial"]["O"] - 0.05 else "unchanged")
    return {
        "persona": d["persona"],
        "facts": d["facts"],
        "speech_mode": snap["speech"],           # sentences | phrases | words | none
        "loc": snap["LOC_avpu"],
        "symptoms_now": symptoms,
        "trend": trend,
        "forbidden": d["forbidden"],
    }


def check_utterance(text: str, sheet: Dict[str, Any]) -> List[str]:
    """Return a list of contradiction/violation labels for a candidate patient utterance."""
    violations: List[str] = []
    mode = sheet["speech_mode"]
    words = text.strip().split()
    if mode == "none" and words:
        violations.append("spoke_while_unable")
        return violations
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    longest = max((len(s.split()) for s in sentences), default=0)
    if mode == "words" and longest > 3:
        violations.append("too_fluent_for_words_mode")
    if mode == "phrases" and longest > 7:
        violations.append("too_fluent_for_phrases_mode")
    if _NUMERIC_VITAL.search(text):
        violations.append("stated_vital_sign_number")
    facts_blob = " ".join(str(v) for v in sheet["facts"].values()).lower()
    low = text.lower()
    for w in _MED_WORDS:
        if w in low and w not in facts_blob:
            violations.append(f"medication_not_in_facts:{w}")
    for w in _ALLERGY_WORDS:
        if w in low and w not in facts_blob:
            violations.append(f"allergy_not_in_facts:{w}")
    if sheet["trend"] == "worsening" and re.search(r"\b(better|improving|easier)\b", low):
        violations.append("claims_improvement_while_worsening")
    return violations
