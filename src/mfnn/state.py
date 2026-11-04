"""Patient state interface: a discrete-time state-space model.

    next_state, events = step(state, actions_at_tick, spec)
    observation       = observe(state, assessment_action | devices, spec)

Time semantics
--------------
- One tick = DT_SECONDS of simulated time (1 s). All rates in the specification are per second
  internally; the specification file uses per-minute units and is converted on load.
- Nothing happens "between" ticks. Actions submitted during wall-clock time are stamped with the
  tick at which they are applied; several actions in the same tick are applied in (seq, kind)
  order so replay is deterministic.
- Real-time mode advances one tick per wall-clock second. Pause stops the tick counter; the
  physiology does not advance while paused. Fast-forward applies N ticks with no actions.
- The engine is deterministic: identical (spec, initial state, action log) gives identical
  (state log, event log). Randomness, if ever introduced, must be seeded from the spec.

What lives where (kept separate on purpose)
-------------------------------------------
- Hidden physiology:            PatientState.phys  (never shown to the learner directly)
- Drug / device state:          PatientState.drugs, PatientState.devices
- Intervention history:         PatientState.history (append-only, part of state for replay)
- What a monitor or exam shows: observe.py  (gated by attached devices / assessment actions)
- Patient history facts:        the specification's `dialogue.facts` (not state)
- Spoken dialogue:              dialogue.py (reads state + facts; never writes state)
- Learner performance grading:  grading.py  (reads the action log; never influences dynamics)
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

DT_SECONDS = 1.0


@dataclass
class DrugState:
    """Per-drug compartments. Amounts in the dose unit of the specification (mg or mcg)."""
    depot: float = 0.0        # absorption compartment (inhaled / IM / gut), 0 for IV bolus
    central: float = 0.0      # central compartment amount
    effect: float = 0.0       # effect-site concentration proxy (first-order link from central)
    doses_given: int = 0
    last_dose_tick: Optional[int] = None


@dataclass
class DeviceState:
    oxygen_device: str = "none"     # none | nc | nrb
    oxygen_flow_lpm: float = 0.0
    cpap_on: bool = False
    cpap_pressure: float = 0.0
    cpap_fio2: float = 0.21
    bvm_on: bool = False
    bvm_rate: float = 0.0
    intubated: bool = False
    vent_rate: float = 0.0
    monitors: List[str] = field(default_factory=list)   # spo2, etco2, ecg, nibp_cycle
    position: str = "sitting"


@dataclass
class PatientState:
    tick: int = 0
    phys: Dict[str, float] = field(default_factory=dict)
    drugs: Dict[str, DrugState] = field(default_factory=dict)
    devices: DeviceState = field(default_factory=DeviceState)
    flags: Dict[str, bool] = field(default_factory=dict)
    timers: Dict[str, int] = field(default_factory=dict)      # name -> tick when event started
    history: List[Dict[str, Any]] = field(default_factory=list)  # applied actions, append-only
    stage: str = "S0"

    @property
    def elapsed_seconds(self) -> float:
        return self.tick * DT_SECONDS

    def clone(self) -> "PatientState":
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def digest(self) -> str:
        """Stable hash of the full state, for replay checks."""
        s = json.dumps(self.to_dict(), sort_keys=True, default=float)
        return hashlib.sha256(s.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Event:
    tick: int
    name: str
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """What an assessment action or attached monitor reveals at one tick. Missing keys mean
    'not available' (device not attached, exam not performed)."""
    tick: int
    values: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
