"""The asthma engine's per-tick step as an explicit program for the deterministic compiler.

Mirrors `mfnn.engine.PatientEngine._physiology_step` and `_drug_effects` line by line. Device logic
(FiO2 table, which assisted rate applies, monitors) stays in software and enters as inputs. Three
patient-specific knobs enter through a parameter port so one compiled brain serves variants and can
be gradient-calibrated. Every node is named; the compiler manifest maps hidden units to these names.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from mfnn.engine import severinghaus_spo2, _k_per_s  # noqa: E402
from mfnn.wsp.compiler import Graph  # noqa: E402

DT = 1.0; DMIN = DT / 60.0
STATE = ["O", "F", "I", "PaCO2", "PaO2", "autoPEEP", "HR", "SBP", "RR", "Vt", "LOC", "arrest_timer", "arrest"]
DEVICE_INPUTS = ["fio2", "assisted_rate", "cpap_on", "intubated", "supine", "vt_assist"]
PORT = ["beta_agonist_responsiveness", "progression_rate", "fatigue_rate"]
STATE_RANGE = {"O": (0, 0.98), "F": (0, 1), "I": (0.2, 1.0), "PaCO2": (10, 200), "PaO2": (10, 713), "autoPEEP": (0, 25),
               "HR": (0, 220), "SBP": (0, 230), "RR": (0, 80), "Vt": (0, 700), "LOC": (0, 3), "arrest_timer": (0, 400), "arrest": (0, 1)}
PORT_RANGE = {"beta_agonist_responsiveness": (0.05, 1.0), "progression_rate": (0.0, 0.08), "fatigue_rate": (0.0, 0.06)}
EMAX_KNOTS_REL = [0, .02, .05, .1, .15, .2, .3, .4, .5, .65, .8, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6.5, 8, 10, 13, 17, 22, 30, 40, 60, 100]


def drug_amount_cap(dspec):
    return float(dspec["typical_dose"]) * (int(dspec.get("max_repeats", 0)) + 1) * 2.0 + 1e-3


def build_asthma_graph(spec):
    p = {k: float(v["value"]) for k, v in spec["scenario"]["params"].items()}
    b = spec["patient"]["baseline"]
    g = Graph()
    S = {n: g.input(n, *STATE_RANGE[n]) for n in STATE}
    D = {}
    for name, ds in spec["drugs"].items():
        cap = drug_amount_cap(ds)
        D[name] = {"depot": g.input(f"{name}.depot", 0, cap), "central": g.input(f"{name}.central", 0, cap),
                   "effect": g.input(f"{name}.effect", 0, cap), "dose": g.input(f"{name}.dose", 0, cap)}
    fio2 = g.input("fio2", 0.21, 1.0); assisted_rate = g.input("assisted_rate", 0, 30); cpap_on = g.input("cpap_on", 0, 1)
    intubated = g.input("intubated", 0, 1); supine = g.input("supine", 0, 1); vt_assist = g.input("vt_assist", 0, 600)
    P = {n: g.input(n, *PORT_RANGE[n]) for n in PORT}
    one = g.const(1.0, "one")
    notflag = g.affine([(S["arrest"], -1.0)], 1.0, "not_arrest")

    # ------------------------------------------------ drugs (linear kinetics, exact) + Emax effects
    eff = {t: [] for t in ("bronchodilation", "heart_rate", "sbp", "resp_drive", "loc", "inflammation", "wob")}
    drug_next = {}
    for name, ds in spec["drugs"].items():
        kin = ds["kinetics"]; ka = _k_per_s(kin["absorption_half_time_min"]); ke = _k_per_s(kin["elimination_half_life_min"])
        keo = _k_per_s(kin.get("effect_link_half_time_min", 0.0))
        d = D[name]; route = ds["route"]
        to_central = route in ("iv_bolus", "sl") and kin["absorption_half_time_min"] <= 0
        a_d = 0.0 if ka == float("inf") else math.exp(-ka * DT); a_c = math.exp(-ke * DT)
        if to_central:
            depot1 = g.affine([(d["depot"], a_d)], 0.0, f"{name}.depot'")                       # depot stays (unused route)
            central1 = g.affine([(d["central"], a_c), (d["dose"], a_c), (d["depot"], a_c * (1 - a_d))], 0.0, f"{name}.central'")
        else:
            depot1 = g.affine([(d["depot"], a_d), (d["dose"], a_d)], 0.0, f"{name}.depot'")
            central1 = g.affine([(d["depot"], a_c * (1 - a_d)), (d["dose"], a_c * (1 - a_d)), (d["central"], a_c)], 0.0, f"{name}.central'")
        if keo == float("inf"):
            effect1 = central1
        else:
            al = 1 - math.exp(-keo * DT)
            effect1 = g.affine([(d["effect"], 1 - al), (central1, al)], 0.0, f"{name}.effect'")
        drug_next[name] = (depot1, central1, effect1)
        for ename, e in ds["effects"].items():
            emax, ec50, hill = float(e["emax"]), float(e["ec50"]), float(e.get("hill", 1.0))
            f = lambda c, emax=emax, ec50=ec50, hill=hill: emax * (max(c, 0.0) ** hill) / (ec50 ** hill + max(c, 0.0) ** hill)
            knots = sorted(set([k * ec50 for k in EMAX_KNOTS_REL] + [drug_amount_cap(ds)]))
            knots = [k for k in knots if k <= drug_amount_cap(ds)] + [drug_amount_cap(ds)]
            eff[e["target"]].append((g.fn(effect1, f, knots=sorted(set(knots)), name=f"{name}.{ename}"), 1.0))
    def effsum(t, name):
        return g.affine(eff[t], 0.0, name) if eff[t] else g.const(0.0, name)
    E_bronch = effsum("bronchodilation", "eff_bronch"); E_hr = effsum("heart_rate", "eff_hr"); E_sbp = effsum("sbp", "eff_sbp")
    E_drive = effsum("resp_drive", "eff_drive"); E_loc = effsum("loc", "eff_loc"); E_infl = effsum("inflammation", "eff_infl")

    # ------------------------------------------------ obstruction and inflammation
    O, F, I = S["O"], S["F"], S["I"]
    E_dil = g.mul(P["beta_agonist_responsiveness"], E_bronch, eps_rel=2e-4, name="E_dil")
    oneminusO = g.affine([(O, -1.0)], 1.0, "1-O")
    prog = g.mul_pos([(P["progression_rate"], 1e-6), (I, 1e-3), (oneminusO, 1e-3)], scale=DMIN, name="prog_term")
    O_minus_floor = g.affine([(O, 1.0)], -p["O_floor"], "O-floor")
    dil = g.mul(E_dil, O_minus_floor, eps_rel=2e-4, name="dil_term")
    O_new = g.clip(g.affine([(O, 1.0), (prog, 1.0), (dil, -p["dilation_rate"] * DMIN)], 0.0, "O_pre"), 0.0, 0.98, "O'")
    kI = 1 - math.exp(-DMIN / 30.0)
    I_new = g.affine([(I, 1 - kI), (E_infl, -kI)], kI, "I'")

    # ------------------------------------------------ ventilation
    sev_knots = list(__import__("numpy").exp(__import__("numpy").linspace(math.log(10), math.log(713), 480)))
    spo2_prev = g.fn(S["PaO2"], severinghaus_spo2, knots=sev_knots, name="spo2_prev")
    drive = g.max(g.const(0.05), g.affine([(E_drive, 1.0)], 1.0, "drive_raw"), name="drive_factor")
    co2_term = g.min(g.scale(g.relu(g.affine([(S["PaCO2"], 1.0)], -40.0, "PaCO2-40")), p["rr_co2_gain"]), g.const(p["rr_co2_cap"]), name="co2_term")
    blunt = g.relu(g.affine([(g.relu(g.affine([(S["PaCO2"], 1.0)], -p["loc_co2_1"], "PaCO2-c1")), -1.0 / 30.0)], 1.0, "blunt_raw"), name="blunt")
    co2b = g.mul(co2_term, blunt, eps_rel=1e-3, name="co2_blunted")
    hypox = g.relu(g.affine([(spo2_prev, -1.0)], 92.0, "92-spo2"), name="hypox_term")
    rr_base = g.affine([(co2b, 1.0), (hypox, p["rr_hypox_gain"]), (O, p["rr_dyspnea_gain"])], b["RR"], "rr_base")
    F2 = g.fn(F, lambda f: f * f, n_knots=128, name="F^2")
    fat_factor = g.affine([(F2, -p["rr_fatigue_loss"])], 1.0, "1-lossF2")
    factor = g.mul(drive, fat_factor, eps_rel=2e-4, name="rr_factor")
    rr_t = g.mul(rr_base, factor, eps_rel=2e-4, name="rr_target0")
    rr_t = g.affine([(rr_t, 1.0), (g.gate(rr_t, supine, "rr_supine_gate"), 0.05)], 0.0, "rr_target1")
    loc_le1 = g.affine([(g.step(S["LOC"], 1.0, 1e4, "LOC>1"), -1.0)], 1.0, "LOC<=1")
    rr_t = g.affine([(rr_t, 1.0), (g.gate(rr_t, loc_le1, "rr_loc_gate"), -0.3)], 0.0, "rr_target")
    vt_a = g.affine([(O, -p["vt_obstruction_loss"])], 1.0, "1-aO"); vt_b = g.affine([(F, -0.4)], 1.0, "1-0.4F")
    vt_c = g.affine([(drive, 0.4)], 0.6, "0.6+0.4drive")
    vt_t = g.scale(g.mul(g.mul(vt_a, vt_b, eps_rel=1e-4, name="vt_ab"), vt_c, eps_rel=1e-4, name="vt_abc"), b["Vt"], name="vt_target")
    k20 = 1 - math.exp(-DT / 20.0)
    RR_new = g.affine([(S["RR"], 1 - k20), (rr_t, k20)], 0.0, "RR'")
    Vt_new = g.affine([(S["Vt"], 1 - k20), (vt_t, k20)], 0.0, "Vt'")
    Vd = 150.0
    vt_eff = g.relu(g.affine([(Vt_new, 1.0)], -Vd, "Vt-Vd"), name="vt_eff")
    v_attempt = g.scale(g.mul(RR_new, vt_eff, eps_rel=1e-5, name="RRxVt"), 1e-3, name="v_alv_attempt")
    v_alv_base = b["RR"] * (b["Vt"] - Vd) / 1000.0
    cap_f = g.fn(oneminusO, lambda x: max(x, 0.0) ** 1.5, knots=[1 - k for k in sorted(set([0.98 - 0.98 * (1 - math.exp(-i / 12.0)) for i in range(80)] + [0.0]))][::-1], name="(1-O)^1.5")
    cap_g = g.affine([(F, -0.8)], 1.0, "1-0.8F")
    cap = g.scale(g.mul(cap_f, cap_g, eps_rel=2e-5, name="cap_fg"), p["V_alv_max"], name="cap")
    safe_rate = g.affine([(O, -10.0)], 14.0, "safe_rate")
    is_assisted = g.step(assisted_rate, 1e-6, 1e6, "is_assisted")
    excess = g.gate(g.relu(g.sub(assisted_rate, safe_rate, "rate-safe")), is_assisted, "excess_rate")
    autoPEEP_new = g.clip(g.affine([(S["autoPEEP"], 1.0), (excess, p["autopeep_rate"] * DMIN)], -p["autopeep_decay"] * DMIN, "ap_pre"), 0.0, p["autopeep_max"], "autoPEEP'")
    ap_pen = g.relu(g.affine([(g.relu(g.affine([(autoPEEP_new, 1.0)], -5.0, "ap-5")), -0.04)], 1.0, "ap_pen_raw"), name="ap_penalty")
    loc_gt1 = g.step(S["LOC"], 1.0, 1e4, "LOC>1b"); sbp_ge90 = g.step(S["SBP"], 89.999, 1e4, "SBP>=90")
    cpap_ok = g.relu(g.affine([(cpap_on, 1.0), (loc_gt1, 1.0), (sbp_ge90, 1.0)], -2.0, "cpap_and"), name="cpap_ok")
    # assisted branch
    vt_assist_eff = g.relu(g.affine([(vt_assist, 1.0), (g.mul(vt_assist, O, eps_rel=1e-4, name="vtassist_x_O"), -0.5)], -Vd, "vt_assist_eff_raw"), name="vt_assist_eff")
    v_assist = g.scale(g.mul(assisted_rate, vt_assist_eff, eps_rel=1e-4, name="rate_x_vt"), 1e-3, name="v_assist")
    v_alv_a = g.mul(g.max(v_attempt, v_assist, name="max_att_assist"), ap_pen, eps_rel=1e-4, name="v_alv_a_raw")
    v_alv_a = g.min(v_alv_a, g.scale(cap, 2.0), name="v_alv_a")
    # spontaneous branch
    cap_cpap = g.affine([(cap, 1.0), (g.gate(cap, cpap_ok, "cap_cpap_gate"), p["cpap_vent_gain"])], 0.0, "cap_cpap")
    v_alv_s = g.min(v_attempt, cap_cpap, name="v_alv_s")
    not_assisted = g.affine([(is_assisted, -1.0)], 1.0, "not_assisted")
    v_alv = g.affine([(g.gate(v_alv_a, is_assisted, "va_gate"), 1.0), (g.gate(v_alv_s, not_assisted, "vs_gate"), 1.0)], 0.0, "v_alv_sel")
    v_alv = g.max(v_alv, g.const(0.05), name="v_alv")
    # work of breathing via logs: wob = (1-O)^-2 * v_attempt / base
    log_1mO = g.log(oneminusO, 0.02, n_knots=256, name="log(1-O)")
    log_vatt = g.log(v_attempt, 1e-3, n_knots=256, name="log(v_att)")
    wob = g.exp(g.affine([(log_1mO, -2.0), (log_vatt, 1.0)], -math.log(v_alv_base), "log_wob"), n_knots=640, name="wob_raw")
    wob = g.affine([(wob, 1.0), (g.gate(wob, cpap_ok, "wob_cpap_gate"), -p["cpap_wob_reduction"])], 0.0, "wob_cpap")
    wob = g.affine([(wob, 1.0), (g.gate(wob, intubated, "wob_intub_gate"), -0.7)], 0.0, "wob_intub")
    wob = g.affine([(wob, 1.0), (g.gate(wob, supine, "wob_supine_gate"), 0.2)], 0.0, "wob")
    excess_wob = g.affine([(wob, 1.0)], -p["WOB_sustainable"], "excess_wob")
    oneminusF = g.affine([(F, -1.0)], 1.0, "1-F")
    F_inc = g.mul_pos([(P["fatigue_rate"], 1e-6), (g.relu(excess_wob, "excess_wob+"), 1e-6), (oneminusF, 1e-4)], scale=DMIN, name="F_inc")
    ex_pos = g.step(excess_wob, 0.0, 1e4, "excess>0")
    F_new = g.clip(g.affine([(F, 1.0), (F_inc, 1.0), (ex_pos, p["recovery_rate"] * DMIN)], -p["recovery_rate"] * DMIN, "F_pre"), 0.0, 1.0, "F'")

    # ------------------------------------------------ gas exchange
    recip = g.fn(v_alv, lambda v: 1.0 / v, n_knots=480, spacing="log", name="1/v_alv")
    paco2_ss = g.min(g.scale(recip, 0.863 * p["VCO2"]), g.const(130.0), name="paco2_ss")
    kco2 = 1 - math.exp(-DMIN / p["tau_CO2_min"])
    PaCO2_new = g.affine([(S["PaCO2"], 1 - kco2), (paco2_ss, kco2)], 0.0, "PaCO2'")
    pao2_alv = g.affine([(fio2, 713.0), (PaCO2_new, -1.0 / 0.8)], 0.0, "pao2_alv")
    O2n = g.fn(O, lambda o: o * o, n_knots=128, name="O^2")
    admix = g.min(g.affine([(O2n, p["shunt_per_O2"])], p["shunt_base"], "admix_raw"), g.const(0.6), name="admix")
    one_minus_admix = g.affine([(admix, -1.0)], 1.0, "1-admix")
    pao2_t = g.max(g.mul(pao2_alv, one_minus_admix, eps_rel=1e-5, name="pao2_x"), g.const(15.0), name="pao2_target")
    ko2 = 1 - math.exp(-DMIN / p["tau_O2_min"])
    PaO2_new = g.affine([(S["PaO2"], 1 - ko2), (pao2_t, ko2)], 0.0, "PaO2'")
    spo2 = g.fn(PaO2_new, severinghaus_spo2, knots=sev_knots, name="SpO2")

    # ------------------------------------------------ cardiovascular
    hr_t = g.affine([(g.relu(g.affine([(spo2, -1.0)], 92.0, "92-spo2b")), p["hr_hypox_gain"]), (O, p["hr_dyspnea_gain"]),
                     (g.relu(g.affine([(PaCO2_new, 1.0)], -45.0, "PaCO2-45")), p["hr_co2_gain"]), (E_hr, 1.0)], b["HR"], "hr_target")
    sbp_lt90 = g.affine([(sbp_ge90, -1.0)], 1.0, "SBP<90")
    cpap_hypo = g.relu(g.affine([(cpap_on, 1.0), (sbp_lt90, 1.0)], -1.0, "cpap_and_hypo"), name="cpap_hypo")
    sbp_t = g.affine([(hr_t, 0.3), (E_sbp, 1.0), (autoPEEP_new, -p["sbp_per_autopeep"]), (cpap_hypo, -8.0)], b["SBP"] - 0.3 * b["HR"], "sbp_target")
    w = g.clip(g.affine([(PaO2_new, -0.1)], p["prearrest_PaO2"] / 10.0, "w_raw"), 0.0, 1.0, "prearrest_w")
    hr_t2 = g.affine([(hr_t, 1.0), (g.mul(w, g.affine([(hr_t, -1.0)], 40.0, "40-hr"), eps_rel=1e-4, name="w_x_hr"), 1.0)], 0.0, "hr_target2")
    sbp_t2 = g.affine([(sbp_t, 1.0), (g.mul(w, g.affine([(sbp_t, -1.0)], 60.0, "60-sbp"), eps_rel=1e-4, name="w_x_sbp"), 1.0)], 0.0, "sbp_target2")
    k15 = 1 - math.exp(-DT / 15.0)
    HR_new = g.affine([(S["HR"], 1 - k15), (hr_t2, k15)], 0.0, "HR'")
    SBP_new = g.relu(g.affine([(S["SBP"], 1 - k20), (sbp_t2, k20)], 0.0, "SBP_pre"), name="SBP'")

    # ------------------------------------------------ consciousness
    loc_t = g.affine([(g.step(PaCO2_new, p["loc_co2_1"], 1e4, "PaCO2>c1"), -1.0), (g.step(PaCO2_new, p["loc_co2_2"], 1e4, "PaCO2>c2"), -1.0),
                      (g.step(PaO2_new, p["loc_o2_1"], 1e4, "PaO2>o1"), 1.0), (g.step(PaO2_new, p["loc_o2_2"], 1e4, "PaO2>o2"), 1.0), (E_loc, 1.0)], 1.0, "loc_target_raw")
    loc_t = g.clip(loc_t, 0.0, 3.0, "loc_target")
    k45 = 1 - math.exp(-DT / 45.0)
    LOC_new = g.affine([(S["LOC"], 1 - k45), (loc_t, k45)], 0.0, "LOC'")

    # ------------------------------------------------ arrest timer and flag
    cond = g.min(g.affine([(g.step(PaO2_new, p["arrest_PaO2"], 1e4, "PaO2>arrest"), -1.0), (g.affine([(g.step(HR_new, 30.0, 1e4, "HR>30"), -1.0)], 1.0, "HR<30"), 1.0)], 1.0, "cond_raw"), one, name="arrest_cond")
    cond = g.relu(cond, "arrest_cond+")
    timer_new = g.gate(g.affine([(S["arrest_timer"], 1.0)], DT, "timer+dt"), cond, "arrest_timer'")
    flag_new = g.max(S["arrest"], g.step(timer_new, p["arrest_hold_s"] - 0.5, 1e4, "timer>=hold"), name="arrest'")

    # ------------------------------------------------ arrest branch blending (engine returns early in arrest)
    def blend(normal, arrest_val, name):
        return g.affine([(g.gate(normal, notflag, f"{name}_n"), 1.0), (g.gate(arrest_val, S["arrest"], f"{name}_a"), 1.0)], 0.0, name)
    zero = g.const(0.0)
    out = {}
    out["O"] = blend(O_new, O, "O_out"); out["F"] = blend(F_new, F, "F_out"); out["I"] = blend(I_new, I, "I_out")
    out["PaCO2"] = blend(PaCO2_new, g.affine([(S["PaCO2"], 1.0)], 0.05 * DT, "PaCO2_arrest"), "PaCO2_out")
    out["PaO2"] = blend(PaO2_new, g.max(g.affine([(S["PaO2"], 1.0)], -0.5 * DT, "PaO2_arrest_raw"), g.const(10.0), name="PaO2_arrest"), "PaO2_out")
    out["autoPEEP"] = blend(autoPEEP_new, S["autoPEEP"], "autoPEEP_out")
    out["HR"] = g.gate(HR_new, notflag, "HR_out"); out["SBP"] = g.gate(SBP_new, notflag, "SBP_out")
    out["RR"] = blend(RR_new, zero, "RR_out"); out["Vt"] = blend(Vt_new, S["Vt"], "Vt_out")
    out["LOC"] = g.gate(LOC_new, notflag, "LOC_out")
    out["arrest_timer"] = blend(timer_new, S["arrest_timer"], "timer_out"); out["arrest"] = flag_new
    for n in STATE: g.output(out[n], n)
    for name, (d1, c1, e1) in drug_next.items():
        g.output(d1, f"{name}.depot"); g.output(c1, f"{name}.central"); g.output(e1, f"{name}.effect")
    # ------------------------------------------------ observables (from the post-update state)
    spo2_out = blend(spo2, g.fn(out["PaO2"], severinghaus_spo2, knots=sev_knots, name="spo2_arrest"), "SpO2_obs")
    g.output(spo2_out, "SpO2")
    etco2 = g.relu(g.affine([(PaCO2_new, 1.0), (O_new, -p["etco2_gradient_gain"])], -3.0, "etco2_raw"), name="EtCO2_raw")
    g.output(g.gate(etco2, notflag, "EtCO2"), "EtCO2")
    vent_frac = g.scale(v_alv, 1.0 / v_alv_base, name="vent_frac")
    silent = g.gate(g.affine([(g.step(vent_frac, p["silent_chest_vent_fraction"], 1e4, "vf>thr"), -1.0)], 1.0, "vf<thr"), notflag, "silent_chest")
    g.output(silent, "silent_chest")
    wheeze = g.gate(g.min(g.scale(O_new, 1.3), one, name="wheeze_raw"), g.affine([(silent, -1.0)], 1.0, "not_silent"), "wheeze")
    g.output(wheeze, "wheeze")
    g.output(g.affine([(O_new, p["pulsus_paradoxus_gain"])], 5.0, "pulsus"), "pulsus_paradoxus")
    loc_round = g.affine([(g.step(out["LOC"], 0.5, 1e4, "LOC>0.5"), 1.0), (g.step(out["LOC"], 1.5, 1e4, "LOC>1.5"), 1.0), (g.step(out["LOC"], 2.5, 1e4, "LOC>2.5"), 1.0)], 0.0, "LOC_round")
    g.output(loc_round, "LOC_round")
    # speech level: 3 sentences, 2 phrases, 1 words, 0 none  (engine: none if arrest or LOC_round<=1; words if O>=0.78 or F>=0.5; phrases if O>=0.6 or RR>=28)
    sev_words = g.min(g.affine([(g.step(O_new, 0.78 - 1e-9, 1e4, "O>=.78"), 1.0), (g.step(F_new, 0.5 - 1e-9, 1e4, "F>=.5"), 1.0)], 0.0, "words_raw"), one, name="words_flag")
    sev_phr = g.min(g.affine([(g.step(O_new, 0.6 - 1e-9, 1e4, "O>=.6"), 1.0), (g.step(RR_new, 28.0 - 1e-9, 1e4, "RR>=28"), 1.0)], 0.0, "phr_raw"), one, name="phrases_flag")
    can_speak = g.gate(g.step(loc_round, 1.5, 1e4, "LOCr>1"), notflag, "can_speak")
    level = g.affine([(sev_words, -2.0), (sev_phr, -1.0), (g.gate(sev_words, sev_phr, "w_and_p"), 1.0)], 3.0, "level_raw")  # words->1, phrases->2, else 3
    g.output(g.gate(level, can_speak, "speech_level"), "speech_level")
    g.output(g.step(O_new, 0.45, 1e4, "accessory"), "accessory_muscles")
    g.output(g.min(g.scale(wob, 1.0 / 12.0), one, name="wob_index"), "work_of_breathing")
    g.output(vent_frac, "vent_frac"); g.output(v_alv, "v_alv"); g.output(wob, "wob")
    return g


if __name__ == "__main__":
    import json, time
    from mfnn.spec import load_spec
    from mfnn.wsp.compiler import compile_graph
    spec = load_spec(ROOT / "spec/examples/asthma_adult_acute.yaml")
    t0 = time.perf_counter(); g = build_asthma_graph(spec); net, man = compile_graph(g); dt = time.perf_counter() - t0
    print("nodes", len(g.nodes), "inputs", len(g.inputs), "outputs", len(g.outputs))
    print("widths", man["widths"], "layers", man["n_layers"], "dense", man["dense_params"], "nonzero", man["nonzero_params"], f"compile {dt:.1f}s")
