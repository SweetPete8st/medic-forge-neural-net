"""Observation gating: what a learner can see depends on attached devices and assessment actions.

Hidden physiology is never returned directly. Each observable in `observation_gating` lists the
gates that reveal it: `device:<name>` (continuously, while attached) or `action:<kind>` (at the
tick the action is performed)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .actions import ActionKind
from .engine import PatientEngine
from .state import Observation


def observe(engine: PatientEngine, action_kind: Optional[ActionKind] = None) -> Observation:
    st = engine.state
    spec = engine.spec
    gates_open = {f"device:{d}" for d in st.devices.monitors}
    if action_kind is not None:
        gates_open.add(f"action:{action_kind.value}")
    snap = engine.snapshot()
    vals: Dict[str, Any] = {}
    for obs_name, gates in spec["observation_gating"].items():
        if any(g in gates_open for g in gates):
            vals[obs_name] = _render(obs_name, snap, st)
    return Observation(tick=st.tick, values=vals, source=(action_kind.value if action_kind else "monitors"))


def _render(name: str, snap: Dict[str, Any], st) -> Any:
    """Round and format the way a monitor or clinician would report it."""
    if name == "SpO2":
        return int(round(snap["SpO2"])) if snap["SpO2"] > 50 else "--"   # pulse ox fails at low perfusion/arrest
    if name == "EtCO2":
        return int(round(snap["EtCO2"]))
    if name == "EtCO2_waveform":
        return snap["EtCO2_waveform"]
    if name == "HR":
        return int(round(snap["HR"]))
    if name == "ECG_rhythm":
        hr = snap["HR"]
        if snap["arrest"]:
            return "asystole_or_PEA"
        return "sinus_tachycardia" if hr > 100 else ("sinus_bradycardia" if hr < 60 else "sinus_rhythm")
    if name in ("SBP", "DBP"):
        return int(round(snap[name] / 2.0) * 2)   # manual cuff resolution
    if name == "pulsus_paradoxus":
        return int(round(snap["pulsus_paradoxus"]))
    if name == "RR":
        return int(round(snap["RR"]))
    if name == "work_of_breathing":
        w = snap["work_of_breathing"]
        return "severe" if w > 0.6 else ("moderate" if w > 0.35 else "mild")
    if name == "speech":
        return snap["speech"]
    if name == "position":
        return snap["position_observed"]
    if name == "skin":
        return snap["skin"]
    if name == "breath_sounds":
        if snap["silent_chest"]:
            return "silent_chest"
        w = snap["wheeze"]
        return "loud_expiratory_wheeze" if w > 0.7 else ("expiratory_wheeze" if w > 0.35 else "mild_wheeze")
    if name == "LOC":
        return snap["LOC_avpu"]
    if name == "airway":
        return "patent" if snap["LOC"] > 1.5 else "at_risk"
    return snap.get(name)
