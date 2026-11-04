"""Code-compiled patient engine (the control in every comparison).

This executes the specification directly as ordinary code. It is deliberately small, explicit
and auditable: a clinical reviewer should be able to read `_physiology_step` top to bottom.

Everything scenario-specific comes from the specification (parameters, drugs, devices, events,
stages, trajectories). The equations below are the shared "fixed and validated once" dynamics
of approach C executed in code; the specification is the parameter vector plus typed rules.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

from .actions import ActionKind, LearnerAction
from .spec import eval_condition, param
from .state import DT_SECONDS, DeviceState, DrugState, Event, PatientState

LN2 = math.log(2.0)


def _k_per_s(half_time_min: float) -> float:
    """First-order rate constant per second from a half time in minutes. 0 -> instantaneous."""
    if half_time_min <= 0:
        return float("inf")
    return LN2 / (half_time_min * 60.0)


def severinghaus_spo2(pao2: float) -> float:
    """Severinghaus (1979) O2 dissociation curve. PaO2 mmHg -> SpO2 %."""
    p = max(pao2, 1.0)
    s = 1.0 / (23400.0 / (p ** 3 + 150.0 * p) + 1.0)
    return 100.0 * s


def emax(conc: float, e_max: float, ec50: float, hill: float = 1.0) -> float:
    c = max(conc, 0.0) ** hill
    return e_max * c / (ec50 ** hill + c)


class PatientEngine:
    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.p = {k: float(v["value"]) for k, v in spec["scenario"]["params"].items()}
        self.base = spec["patient"]["baseline"]
        self.state = self.reset()
        self._fired_once: set = set()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> PatientState:
        b = self.base
        init = self.spec["scenario"]["initial"]
        phys = {
            "O": init["O"], "F": init["F"], "I": init.get("I", 1.0),
            "PaCO2": init["PaCO2"], "PaO2": init["PaO2"], "autoPEEP": init.get("autoPEEP", 0.0),
            "HR": b["HR"] + 0.0, "SBP": b["SBP"] + 0.0, "DBP": b["DBP"] + 0.0,
            "RR": b["RR"] + 0.0, "Vt": b["Vt"] + 0.0, "LOC": 3.0, "arrest_timer": 0.0,
        }
        # Start vitals at their quasi-steady values for the initial hidden state rather than
        # healthy baseline, so t=0 observations are coherent with O and F.
        st = PatientState(phys=phys, drugs={d: DrugState() for d in self.spec["drugs"]},
                          devices=DeviceState(), stage="S0_moderate")
        self._derive(st)
        for _ in range(120):   # 2 simulated minutes of settling with no actions, not logged
            self._physiology_step(st)
        st.tick = 0
        st.history = []
        self._fired_once = set()
        self.state = st
        return st

    # ------------------------------------------------------------------ public API
    def step(self, actions: Iterable[LearnerAction] = ()) -> Tuple[PatientState, List[Event]]:
        st = self.state
        events: List[Event] = []
        for a in sorted(actions, key=lambda a: a.key()):
            events.extend(self._apply_action(st, a))
        self._physiology_step(st)
        st.tick += 1
        events.extend(self._emit_events(st))
        return st, events

    def run(self, action_log: List[LearnerAction], n_ticks: int, record_every: int = 1):
        """Deterministic rollout. Returns (trace, events). trace rows are dicts of observables+hidden."""
        by_tick: Dict[int, List[LearnerAction]] = {}
        for a in action_log:
            by_tick.setdefault(a.tick, []).append(a)
        trace, all_events = [], []
        for t in range(n_ticks):
            _, ev = self.step(by_tick.get(t, []))
            all_events.extend(ev)
            if t % record_every == 0:
                trace.append(self.snapshot())
        return trace, all_events

    def snapshot(self) -> Dict[str, float]:
        st = self.state
        row = {"tick": st.tick, "t_min": st.tick * DT_SECONDS / 60.0, "stage": st.stage}
        row.update(st.phys)
        row.update(st.derived)
        for d, ds in st.drugs.items():
            row[f"{d}.effect"] = ds.effect
        row["cpap_on"] = float(st.devices.cpap_on)
        row["bvm_on"] = float(st.devices.bvm_on)
        row["intubated"] = float(st.devices.intubated)
        row["fio2"] = self._fio2(st)
        return row

    # ------------------------------------------------------------------ actions
    def _apply_action(self, st: PatientState, a: LearnerAction) -> List[Event]:
        ev: List[Event] = []
        k, p = a.kind, a.params
        st.history.append({"tick": a.tick, "kind": k.value, "params": dict(p), "actor": a.actor, "seq": a.seq})
        if k == ActionKind.GIVE_DRUG:
            drug = p["drug"]
            if drug not in self.spec["drugs"]:
                ev.append(Event(st.tick, "unsupported_drug", {"drug": drug}))
                return ev
            dose = float(p.get("dose", self.spec["drugs"][drug]["typical_dose"]))
            ds = st.drugs[drug]
            route = self.spec["drugs"][drug]["route"]
            if route in ("iv_bolus", "sl") and self.spec["drugs"][drug]["kinetics"]["absorption_half_time_min"] <= 0:
                ds.central += dose
            else:
                ds.depot += dose
            ds.doses_given += 1
            ds.last_dose_tick = st.tick
            ev.append(Event(st.tick, "drug_given", {"drug": drug, "dose": dose}))
        elif k == ActionKind.OXYGEN:
            st.devices.oxygen_device = p.get("device", "none")
            st.devices.oxygen_flow_lpm = float(p.get("flow_lpm", 0.0))
        elif k == ActionKind.CPAP:
            st.devices.cpap_on = bool(p.get("on", True))
            st.devices.cpap_pressure = float(p.get("pressure_cmh2o", 5.0))
            st.devices.cpap_fio2 = float(p.get("fio2", 1.0))
        elif k == ActionKind.BVM:
            st.devices.bvm_on = bool(p.get("on", True))
            st.devices.bvm_rate = float(p.get("rate_per_min", 10.0))
        elif k == ActionKind.INTUBATE:
            if bool(p.get("success", True)):
                st.devices.intubated = True
                st.devices.vent_rate = float(p.get("vent_rate", 8.0))
                st.devices.bvm_on = False
                st.devices.cpap_on = False
                ev.append(Event(st.tick, "intubated", {}))
            else:
                ev.append(Event(st.tick, "intubation_failed", {}))
        elif k == ActionKind.POSITION:
            st.devices.position = p.get("position", "sitting")
        elif k == ActionKind.ATTACH_DEVICE:
            dev = p["device"]
            if dev not in st.devices.monitors:
                st.devices.monitors.append(dev)
        elif k == ActionKind.DETACH_DEVICE:
            dev = p["device"]
            if dev in st.devices.monitors:
                st.devices.monitors.remove(dev)
        # assessment actions change nothing here; observe.py reads them from history
        return ev

    # ------------------------------------------------------------------ physiology
    def _fio2(self, st: PatientState) -> float:
        d = st.devices
        if d.intubated:
            return float(self.spec["devices"]["ventilator"]["fio2"])
        if d.bvm_on:
            return float(self.spec["devices"]["bvm"]["fio2"])
        if d.cpap_on:
            return d.cpap_fio2
        tbl = self.spec["devices"]["oxygen"]["fio2_by_device"][d.oxygen_device]
        flow = min(d.oxygen_flow_lpm, tbl["max_lpm"])
        return min(1.0, tbl["base"] + tbl["per_lpm"] * flow)

    def _drug_effects(self, st: PatientState, dt: float) -> Dict[str, float]:
        """Advance every drug's compartments one tick and sum effects by target."""
        totals = {"bronchodilation": 0.0, "heart_rate": 0.0, "sbp": 0.0, "resp_drive": 0.0,
                  "loc": 0.0, "inflammation": 0.0, "wob": 0.0}
        for name, dspec in self.spec["drugs"].items():
            ds = st.drugs[name]
            kin = dspec["kinetics"]
            ka = _k_per_s(kin["absorption_half_time_min"])
            ke = _k_per_s(kin["elimination_half_life_min"])
            keo = _k_per_s(kin.get("effect_link_half_time_min", 0.0))
            # depot -> central (exact first-order over dt)
            if ds.depot > 0:
                moved = ds.depot if ka == float("inf") else ds.depot * (1 - math.exp(-ka * dt))
                ds.depot -= moved
                ds.central += moved
            # elimination (exact)
            ds.central *= math.exp(-ke * dt)
            # effect site (first-order link)
            if keo == float("inf"):
                ds.effect = ds.central
            else:
                ds.effect += (ds.central - ds.effect) * (1 - math.exp(-keo * dt))
            for ename, eff in dspec["effects"].items():
                totals[eff["target"]] += emax(ds.effect, eff["emax"], eff["ec50"], eff.get("hill", 1.0))
        return totals

    def _physiology_step(self, st: PatientState) -> None:
        p, b, ph = self.p, self.base, st.phys
        dt = DT_SECONDS
        dmin = dt / 60.0
        dev = st.devices
        eff = self._drug_effects(st, dt)

        if st.flags.get("arrest"):
            ph["HR"] = 0.0; ph["SBP"] = 0.0; ph["DBP"] = 0.0; ph["RR"] = 0.0
            ph["PaO2"] = max(ph["PaO2"] - 0.5 * dt, 10.0)
            ph["PaCO2"] = ph["PaCO2"] + 0.05 * dt
            ph["LOC"] = 0.0
            self._derive(st)
            return

        # --- obstruction dynamics -------------------------------------------------
        resp = p["beta_agonist_responsiveness"]
        E_dil = resp * eff["bronchodilation"]
        O = ph["O"]
        dO = p["progression_rate"] * ph["I"] * (1.0 - O) * dmin - p["dilation_rate"] * E_dil * (O - p["O_floor"]) * dmin
        ph["O"] = min(max(O + dO, 0.0), 0.98)
        # inflammation driver (steroids act here, slowly)
        I_target = 1.0 - eff["inflammation"]
        ph["I"] += (I_target - ph["I"]) * (1 - math.exp(-dmin / 30.0))
        O = ph["O"]

        # --- ventilation: attempted vs achievable ---------------------------------
        spo2_prev = severinghaus_spo2(ph["PaO2"])
        drive_factor = max(0.05, 1.0 + eff["resp_drive"])  # sedatives make this < 1
        co2_term = min(p["rr_co2_gain"] * max(ph["PaCO2"] - 40.0, 0.0), p["rr_co2_cap"])
        if ph["PaCO2"] > p["loc_co2_1"]:                           # CO2 narcosis blunts the drive
            co2_term *= max(0.0, 1.0 - (ph["PaCO2"] - p["loc_co2_1"]) / 30.0)
        rr_target = (b["RR"] + co2_term
                     + p["rr_hypox_gain"] * max(92.0 - spo2_prev, 0.0)
                     + p["rr_dyspnea_gain"] * O)
        F = ph["F"]
        rr_target *= drive_factor * (1.0 - p["rr_fatigue_loss"] * F ** 2)   # fatigue -> bradypnoea late
        if ph["LOC"] <= 1.0:
            rr_target *= 0.7
        if dev.position == "supine":
            rr_target *= 1.05
        vt_target = b["Vt"] * (1.0 - p["vt_obstruction_loss"] * O) * (1.0 - 0.4 * F) * (0.6 + 0.4 * drive_factor)
        ph["RR"] += (rr_target - ph["RR"]) * (1 - math.exp(-dt / 20.0))
        ph["Vt"] += (vt_target - ph["Vt"]) * (1 - math.exp(-dt / 20.0))
        Vd = 150.0
        v_alv_attempt = ph["RR"] * max(ph["Vt"] - Vd, 0.0) / 1000.0   # L/min
        v_alv_base = b["RR"] * (b["Vt"] - Vd) / 1000.0

        cap = p["V_alv_max"] * (1.0 - O) ** 1.5 * (1.0 - 0.8 * F)
        safe_rate = 10.0 * (1.0 - O) + 4.0
        assisted_rate = dev.vent_rate if dev.intubated else (dev.bvm_rate if dev.bvm_on else 0.0)
        # auto-PEEP from over-ventilation (interaction rule assisted_ventilation_rate_too_fast)
        excess = max(assisted_rate - safe_rate, 0.0) if assisted_rate > 0 else 0.0
        ph["autoPEEP"] = min(p["autopeep_max"], max(0.0, ph["autoPEEP"] + p["autopeep_rate"] * excess * dmin - p["autopeep_decay"] * dmin))
        ap_penalty = max(0.0, 1.0 - 0.04 * max(ph["autoPEEP"] - 5.0, 0.0))

        cpap_ok = dev.cpap_on and ph["LOC"] > 1.0 and ph["SBP"] >= 90.0
        if assisted_rate > 0:
            vt_assist = float(self.spec["devices"]["ventilator" if dev.intubated else "bvm"]["tidal_volume_ml"])
            v_assist = assisted_rate * max(vt_assist * (1.0 - 0.5 * O) - Vd, 0.0) / 1000.0
            v_alv = max(v_alv_attempt, v_assist) * ap_penalty
            v_alv = min(v_alv, cap * 2.0)       # positive pressure can exceed spontaneous capacity, not unboundedly
        else:
            v_alv = min(v_alv_attempt, cap * (1.0 + (p["cpap_vent_gain"] if cpap_ok else 0.0)))
        v_alv = max(v_alv, 0.05)

        # work of breathing and fatigue
        r_mult = 1.0 / (1.0 - O) ** 2
        wob = r_mult * (v_alv_attempt / max(v_alv_base, 0.1))
        if cpap_ok:
            wob *= (1.0 - p["cpap_wob_reduction"])
        if dev.intubated:
            wob *= 0.3
        if dev.position == "supine":
            wob *= 1.2
        excess_wob = wob - p["WOB_sustainable"]
        if excess_wob > 0:
            ph["F"] = min(1.0, F + p["fatigue_rate"] * excess_wob * (1.0 - F) * dmin)
        else:
            ph["F"] = max(0.0, F - p["recovery_rate"] * dmin)

        # --- gas exchange ---------------------------------------------------------
        paco2_ss = 0.863 * p["VCO2"] / v_alv
        paco2_ss = min(paco2_ss, 130.0)
        ph["PaCO2"] += (paco2_ss - ph["PaCO2"]) * (1 - math.exp(-dmin / p["tau_CO2_min"]))
        fio2 = self._fio2(st)
        pao2_alv = fio2 * (760.0 - 47.0) - ph["PaCO2"] / 0.8
        admix = min(0.6, p["shunt_base"] + p["shunt_per_O2"] * O ** 2)
        pao2_target = max(pao2_alv * (1.0 - admix), 15.0)
        ph["PaO2"] += (pao2_target - ph["PaO2"]) * (1 - math.exp(-dmin / p["tau_O2_min"]))
        spo2 = severinghaus_spo2(ph["PaO2"])

        # --- cardiovascular -------------------------------------------------------
        hr_target = (b["HR"] + p["hr_hypox_gain"] * max(92.0 - spo2, 0.0) + p["hr_dyspnea_gain"] * O
                     + p["hr_co2_gain"] * max(ph["PaCO2"] - 45.0, 0.0) + eff["heart_rate"])
        sbp_target = b["SBP"] + 0.3 * (hr_target - b["HR"]) + eff["sbp"] - p["sbp_per_autopeep"] * ph["autoPEEP"]
        if dev.cpap_on and ph["SBP"] < 90.0:
            sbp_target -= 8.0                     # interaction rule cpap_with_hypotension
        if ph["PaO2"] < p["prearrest_PaO2"]:
            w = min(1.0, (p["prearrest_PaO2"] - ph["PaO2"]) / 10.0)
            hr_target = (1 - w) * hr_target + w * 40.0
            sbp_target = (1 - w) * sbp_target + w * 60.0
        ph["HR"] += (hr_target - ph["HR"]) * (1 - math.exp(-dt / 15.0))
        ph["SBP"] = max(0.0, ph["SBP"] + (sbp_target - ph["SBP"]) * (1 - math.exp(-dt / 20.0)))
        ph["DBP"] = ph["SBP"] * (b["DBP"] / b["SBP"])

        # --- consciousness ---------------------------------------------------------
        loc_target = 3.0
        if ph["PaCO2"] > p["loc_co2_1"]: loc_target -= 1.0
        if ph["PaCO2"] > p["loc_co2_2"]: loc_target -= 1.0
        if ph["PaO2"] < p["loc_o2_1"]: loc_target -= 1.0
        if ph["PaO2"] < p["loc_o2_2"]: loc_target -= 1.0
        loc_target += eff["loc"]
        loc_target = min(3.0, max(0.0, loc_target))
        ph["LOC"] += (loc_target - ph["LOC"]) * (1 - math.exp(-dt / 45.0))

        # --- arrest ---------------------------------------------------------------
        if ph["PaO2"] < p["arrest_PaO2"] or ph["HR"] < 30.0:
            ph["arrest_timer"] += dt
        else:
            ph["arrest_timer"] = 0.0
        if ph["arrest_timer"] >= p["arrest_hold_s"]:
            st.flags["arrest"] = True

        self._derive(st, v_alv=v_alv, v_alv_base=v_alv_base, wob=wob, fio2=fio2, safe_rate=safe_rate,
                     assisted_rate=assisted_rate, drive_factor=drive_factor, cpap_ok=cpap_ok)

    def _derive(self, st: PatientState, **kw) -> None:
        """Derived observables (what monitors and exam would show), computed from hidden state."""
        p, ph = self.p, st.phys
        O = ph["O"]
        spo2 = severinghaus_spo2(ph["PaO2"])
        v_alv = kw.get("v_alv", 4.6)
        v_alv_base = kw.get("v_alv_base", 4.6)
        vent_frac = v_alv / max(v_alv_base, 0.1)
        silent = 1.0 if (vent_frac < p["silent_chest_vent_fraction"] and not st.flags.get("arrest")) else 0.0
        wheeze = 0.0 if silent else min(1.0, O * 1.3)
        loc_round = int(round(ph["LOC"]))
        if st.flags.get("arrest"):
            speech = "none"
        elif loc_round <= 1:
            speech = "none"
        elif O >= 0.78 or ph["F"] >= 0.5:
            speech = "words"
        elif O >= 0.6 or ph["RR"] >= 28:
            speech = "phrases"
        else:
            speech = "sentences"
        st.derived = {
            "SpO2": spo2,
            "EtCO2": max(ph["PaCO2"] - (3.0 + p["etco2_gradient_gain"] * O), 0.0) if not st.flags.get("arrest") else 0.0,
            "EtCO2_waveform": "shark_fin" if O > 0.3 else "normal",
            "wheeze": wheeze,
            "silent_chest": silent,
            "pulsus_paradoxus": 5.0 + p["pulsus_paradoxus_gain"] * O,
            "LOC_avpu": ["U", "P", "V", "A"][max(0, min(3, loc_round))],
            "speech": speech,
            "work_of_breathing": min(1.0, kw.get("wob", 1.0) / 12.0),
            "accessory_muscles": 1.0 if O > 0.45 else 0.0,
            "v_alv": v_alv,
            "vent_frac": vent_frac,
            "wob": kw.get("wob", 1.0),
            "safe_rate": kw.get("safe_rate", 10.0),
            "assisted_rate": kw.get("assisted_rate", 0.0),
            "resp_drive_factor": kw.get("drive_factor", 1.0),
            "cpap_effective": 1.0 if kw.get("cpap_ok", False) else 0.0,
            "arrest": 1.0 if st.flags.get("arrest") else 0.0,
            "skin": "cyanotic" if spo2 < 85 else ("pale_diaphoretic" if O > 0.5 else "normal"),
            "position_observed": "tripod" if (O > 0.5 and ph["LOC"] > 2 and st.devices.position != "supine") else st.devices.position,
        }

    # ------------------------------------------------------------------ events & stages
    def _env(self, st: PatientState) -> Dict[str, float]:
        env = dict(st.phys)
        env.update({k: v for k, v in st.derived.items() if isinstance(v, (int, float))})
        env["cpap_on"] = float(st.devices.cpap_on)
        env["bvm_on"] = float(st.devices.bvm_on)
        env["intubated"] = float(st.devices.intubated)
        for d, ds in st.drugs.items():
            env[f"{d}.effect"] = ds.effect
        return env

    def _emit_events(self, st: PatientState) -> List[Event]:
        env = self._env(st)
        out: List[Event] = []
        # stage: highest matching stage wins
        for stg in self.spec.get("stages", []):
            if eval_condition(stg["when"], env):
                st.stage = stg["name"]
        for e in self.spec.get("events", []):
            if eval_condition(e["when"], env):
                if e.get("once", True):
                    if e["name"] in self._fired_once:
                        continue
                    self._fired_once.add(e["name"])
                out.append(Event(st.tick, e["name"], {}))
        if st.devices.cpap_on and st.phys["LOC"] <= 1.0 and st.tick % 60 == 0:
            out.append(Event(st.tick, "cpap_aspiration_risk", {}))
        return out


# PatientState gets a `derived` attribute at runtime; declare it for clarity.
PatientState.derived = {}  # type: ignore[attr-defined]
