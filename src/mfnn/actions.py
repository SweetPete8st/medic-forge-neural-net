"""Structured learner actions and instructor controls.

Design rules (from the brief):
- Actions are structured, not free text. The dialogue layer translates speech into these.
- Every action is stamped with the simulation tick at which it takes effect.
- Observations become available only through assessment actions or attached devices.
- Instructor controls are separate from learner actions and are logged separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ActionKind(str, Enum):
    # Assessment actions (reveal observations; do not change physiology)
    ASSESS_GENERAL = "assess_general"          # general impression: position, speech, work of breathing, skin
    ASSESS_AIRWAY = "assess_airway"
    ASSESS_AUSCULTATE = "auscultate"
    ASSESS_PULSE = "palpate_pulse"
    ASSESS_LOC = "assess_loc"                  # AVPU / GCS
    ASSESS_HISTORY = "take_history"            # routed to dialogue layer; gated by speech ability
    MEASURE_BP = "measure_bp"                  # manual or NIBP single reading
    # Device attach / detach (make continuous observables available)
    ATTACH_DEVICE = "attach_device"            # params: device in {spo2, etco2, ecg, nibp_cycle}
    DETACH_DEVICE = "detach_device"
    # Treatments (change physiology)
    GIVE_DRUG = "give_drug"                    # params: drug, dose, route
    OXYGEN = "oxygen"                          # params: device in {none, nc, nrb}, flow_lpm
    CPAP = "cpap"                              # params: on bool, pressure_cmh2o, fio2
    BVM = "bvm"                                # params: on bool, rate_per_min
    INTUBATE = "intubate"                      # params: success bool (instructor-controlled), vent_rate
    POSITION = "position"                      # params: position in {sitting, supine, tripod}
    # Other
    NO_OP = "no_op"


@dataclass(frozen=True)
class LearnerAction:
    """A single structured action. `tick` is the simulation tick at which it is applied."""
    tick: int
    kind: ActionKind
    params: Dict[str, Any] = field(default_factory=dict)
    actor: str = "learner"
    seq: int = 0  # submission order within a tick; used for deterministic replay

    def key(self) -> tuple:
        return (self.tick, self.seq, self.kind.value)


class InstructorControl(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    FAST_FORWARD = "fast_forward"     # params: ticks
    RESET = "reset"
    OVERRIDE_STATE = "override_state"  # params: {var: value}; logged, breaks "pure" replay unless replayed too
    SCRIPTED_EVENT = "scripted_event"  # params: event name from spec (e.g. "progress_to_severe")
    FREEZE_MODEL = "freeze_model"      # params: spec hash; prevents edits during a session


@dataclass(frozen=True)
class ControlAction:
    tick: int
    kind: InstructorControl
    params: Dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    actor: str = "instructor"
