"""Compose a full specification from a preset plus the slot ledger. Pure data transformation, no LLM."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[3]
PRESET_DIR = Path(__file__).resolve().parent / "presets"


def load_preset(name: str) -> Dict[str, Any]:
    return yaml.safe_load((PRESET_DIR / f"{name}.yaml").read_text())


def _set_path(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split("."); cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _as_list(v) -> List[str]:
    if v is None: return []
    if isinstance(v, list): return [str(x) for x in v]
    return [s.strip() for s in str(v).split(",") if s.strip()]


def build_overlay(preset: Dict[str, Any], slots: Dict[str, Any]) -> Dict[str, Any]:
    """The overlay is the per-patient part: a small, reviewable dict of dotted paths -> values plus narrative blocks."""
    ov: Dict[str, Any] = {"paths": {}, "patient": {}, "dialogue": {}, "traps": [], "duration_min": int(float(slots.get("duration_min", 60)))}
    for knob, table in preset["knobs"].items():
        choice = slots.get(knob)
        if choice in table:
            ov["paths"].update(table[choice])
    ov["patient"] = {"age": int(float(slots.get("age", 40))), "sex": str(slots.get("sex", "female")), "weight_kg": float(slots.get("weight_kg", 70))}
    ov["dialogue"] = {
        "persona": str(slots.get("persona", "")),
        "facts": {
            "chief_complaint": str(slots.get("chief_complaint", "")),
            "history": _as_list(slots.get("past_medical_history")),
            "history_of_present_illness": str(slots.get("history_of_present_illness", "")),
            "medications": _as_list(slots.get("medications")),
            "allergies": _as_list(slots.get("allergies")) or ["no known allergies"],
            "last_oral_intake": str(slots.get("last_oral_intake", "unknown")),
            "scene": str(slots.get("scene", "")),
        },
    }
    ov["traps"] = [t for t in _as_list(slots.get("traps")) if t in preset["traps"]]
    ov["pace_label"] = str(slots.get("deterioration_pace", "typical"))
    ov["teaching_objectives"] = _as_list(slots.get("teaching_objectives"))
    ov["checkpoints"] = slots.get("checkpoints") or []
    return ov


def compose(preset: Dict[str, Any], overlay: Dict[str, Any], spec_id: str) -> Dict[str, Any]:
    base = yaml.safe_load((ROOT / preset["base_spec"]).read_text())
    spec = copy.deepcopy(base)
    for path, value in overlay["paths"].items():
        _set_path(spec, path, value)
    spec["patient"].update(overlay["patient"])
    spec["dialogue"]["persona"] = overlay["dialogue"]["persona"] or spec["dialogue"]["persona"]
    spec["dialogue"]["facts"] = {**spec["dialogue"]["facts"], **{k: v for k, v in overlay["dialogue"]["facts"].items() if v}}
    spec["id"] = spec_id
    spec["title"] = f"{preset['display_name']}: {overlay['patient']['age']}-year-old {overlay['patient']['sex']}, {overlay.get('severity_label', '')}".strip(": ")
    spec["teaching_objective"] = "; ".join(overlay.get("teaching_objectives", [])) or spec["teaching_objective"]
    # trajectories: the preset's canonical scripts plus one inappropriate path built from the chosen traps
    traj = {"untreated": {"description": "No treatment.", "actions": [], "expect": []},
            "timely_treatment": {"description": "Canonical timely bundle.", "actions": preset["canonical_scripts"]["timely_treatment"], "expect": []},
            "delayed_treatment": {"description": "Canonical delayed bundle.", "actions": preset["canonical_scripts"]["delayed_treatment"], "expect": []}}
    trap_actions: List[Dict[str, Any]] = []
    scale = preset.get("trap_time_scale", {}).get(overlay.get("pace_label", "typical"), 1.0)
    for t in overlay["traps"]:
        for a in preset["traps"][t]["script"]:
            a = dict(a); a["t_min"] = round(float(a["t_min"]) * scale, 1); trap_actions.append(a)
    traj["inappropriate_intervention"] = {"description": "Traps chosen by the instructor: " + (", ".join(overlay["traps"]) or "none"),
                                          "actions": sorted(trap_actions, key=lambda a: a["t_min"]) + [{"t_min": 25.0, "kind": "give_drug", "drug": "albuterol", "dose": 2.5}], "expect": []}
    for cp in overlay.get("checkpoints", []):
        if isinstance(cp, dict) and {"path", "t_min", "var", "min", "max"} <= set(cp) and cp["path"] in traj:
            traj[cp["path"]]["expect"].append({"t_min": float(cp["t_min"]), "var": cp["var"], "min": float(cp["min"]), "max": float(cp["max"]), "source": "instructor checkpoint"})
    spec["trajectories"] = traj
    spec["review"] = {"status": "DRAFT", "reviewers": [], "notes": "Composed by the parent agent from an instructor interview. Requires review before classroom use."}
    return spec
