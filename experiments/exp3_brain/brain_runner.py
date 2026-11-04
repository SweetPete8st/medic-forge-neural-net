"""BrainEngine: the PatientEngine software shell (actions, devices, events, logging, replay) with the
physiology step replaced by the compiled network. Everything the brief said must stay explicit in
software does: action parsing, FiO2 table, assisted-rate selection, monitors, event/stage rules."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from mfnn.engine import PatientEngine  # noqa: E402
from mfnn.state import PatientState  # noqa: E402
from mfnn.wsp.compiler import compile_graph  # noqa: E402
from asthma_graph import build_asthma_graph, STATE, PORT  # noqa: E402


class BrainEngine(PatientEngine):
    def __init__(self, spec, net=None, manifest=None, port_overrides=None):
        self.graph = build_asthma_graph(spec)
        if net is None:
            net, manifest = compile_graph(self.graph)
        self.net, self.manifest = net, manifest
        self.in_idx = {n.name: i for i, n in enumerate(self.graph.inputs)}
        self.out_idx = {name: i for i, (name, _) in enumerate(self.graph.outputs)}
        self.port = {k: float(spec["scenario"]["params"][k]["value"]) for k in PORT}
        if port_overrides: self.port.update(port_overrides)
        self._x = np.zeros(len(self.graph.inputs))
        super().__init__(spec)

    def _inputs(self, st: PatientState) -> np.ndarray:
        x = self._x; x[:] = 0.0; ii = self.in_idx; ph = st.phys; dev = st.devices
        for n in STATE:
            if n == "arrest": x[ii[n]] = 1.0 if st.flags.get("arrest") else 0.0
            else: x[ii[n]] = ph[n]
        for name, ds in st.drugs.items():
            x[ii[f"{name}.depot"]] = ds.depot; x[ii[f"{name}.central"]] = ds.central; x[ii[f"{name}.effect"]] = ds.effect
            x[ii[f"{name}.dose"]] = 0.0          # doses are already in the compartments (applied by the shell)
        x[ii["fio2"]] = self._fio2(st)
        assisted = dev.vent_rate if dev.intubated else (dev.bvm_rate if dev.bvm_on else 0.0)
        x[ii["assisted_rate"]] = assisted
        x[ii["cpap_on"]] = 1.0 if dev.cpap_on else 0.0; x[ii["intubated"]] = 1.0 if dev.intubated else 0.0
        x[ii["supine"]] = 1.0 if dev.position == "supine" else 0.0
        x[ii["vt_assist"]] = float(self.spec["devices"]["ventilator" if dev.intubated else "bvm"]["tidal_volume_ml"])
        for k, v in self.port.items(): x[ii[k]] = v
        return x

    def _physiology_step(self, st: PatientState) -> None:
        x = self._inputs(st)
        for n in self.graph.inputs:              # a compiled brain is valid only inside its declared ranges: fail loudly, never silently
            v = x[n.params["index"]]
            if not (n.lo - 1e-9 <= v <= n.hi + 1e-9) or not np.isfinite(v):
                raise RuntimeError(f"brain input '{n.name}' = {v:.6g} outside declared range [{n.lo}, {n.hi}] at tick {st.tick}")
        y = self.net.forward(x); oi = self.out_idx; ph = st.phys; p = self.p; dev = st.devices
        if not np.all(np.isfinite(y)):
            bad = [name for name, i in oi.items() if not np.isfinite(y[i])][:5]
            raise RuntimeError(f"brain produced non-finite outputs {bad} at tick {st.tick}")
        for n in STATE:
            if n == "arrest":
                if y[oi[n]] > 0.5: st.flags["arrest"] = True
            else:
                ph[n] = float(y[oi[n]])
        for name, ds in st.drugs.items():
            ds.depot = float(y[oi[f"{name}.depot"]]); ds.central = float(y[oi[f"{name}.central"]]); ds.effect = float(y[oi[f"{name}.effect"]])
        ph["DBP"] = ph["SBP"] * (self.base["DBP"] / self.base["SBP"])
        arrest = bool(st.flags.get("arrest"))
        O = ph["O"]; spo2 = float(y[oi["SpO2"]]); loc_round = int(round(y[oi["LOC_round"]])); level = int(round(y[oi["speech_level"]]))
        silent = 1.0 if y[oi["silent_chest"]] > 0.5 else 0.0
        assisted = dev.vent_rate if dev.intubated else (dev.bvm_rate if dev.bvm_on else 0.0)
        st.derived = {
            "SpO2": spo2, "EtCO2": float(y[oi["EtCO2"]]), "EtCO2_waveform": "shark_fin" if O > 0.3 else "normal",
            "wheeze": float(y[oi["wheeze"]]), "silent_chest": 0.0 if arrest else silent, "pulsus_paradoxus": float(y[oi["pulsus_paradoxus"]]),
            "LOC_avpu": ["U", "P", "V", "A"][max(0, min(3, loc_round))],
            "speech": ["none", "words", "phrases", "sentences"][max(0, min(3, level))],
            "work_of_breathing": (1.0 / 12.0) if arrest else float(y[oi["work_of_breathing"]]), "accessory_muscles": 1.0 if y[oi["accessory_muscles"]] > 0.5 else 0.0,
            "v_alv": 4.6 if arrest else float(y[oi["v_alv"]]), "vent_frac": 1.0 if arrest else float(y[oi["vent_frac"]]), "wob": 1.0 if arrest else float(y[oi["wob"]]),
            "safe_rate": 10.0 * (1.0 - O) + 4.0, "assisted_rate": assisted, "resp_drive_factor": 1.0,
            "cpap_effective": 1.0 if (dev.cpap_on and ph["LOC"] > 1.0 and ph["SBP"] >= 90.0) else 0.0,
            "arrest": 1.0 if arrest else 0.0,
            "skin": "cyanotic" if spo2 < 85 else ("pale_diaphoretic" if O > 0.5 else "normal"),
            "position_observed": "tripod" if (O > 0.5 and ph["LOC"] > 2 and dev.position != "supine") else dev.position,
        }
