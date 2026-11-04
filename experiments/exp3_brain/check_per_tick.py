"""Per-tick equivalence: random states through the code engine step and the brain step."""
import copy, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))
from mfnn.engine import PatientEngine
from mfnn.spec import load_spec
from mfnn.state import DeviceState, DrugState, PatientState
from brain_runner import BrainEngine
from asthma_graph import STATE_RANGE, STATE

spec = load_spec(ROOT / "spec/examples/asthma_adult_acute.yaml")
code = PatientEngine(spec); brain = BrainEngine(spec)
rng = np.random.default_rng(0)
worst = {k: 0.0 for k in STATE if k != "arrest"}; worst.update({"SpO2": 0.0, "EtCO2": 0.0, "wob": 0.0, "v_alv": 0.0}); nflag = 0; nspeech = 0; nsilent = 0
N = 400; t0 = time.perf_counter()
for i in range(N):
    st = PatientState(phys={}, drugs={d: DrugState() for d in spec["drugs"]}, devices=DeviceState())
    for k, (lo, hi) in STATE_RANGE.items():
        if k == "arrest": continue
        st.phys[k] = float(rng.uniform(lo, hi))
    st.phys["O"] = float(rng.uniform(0.1, 0.9)); st.phys["PaO2"] = float(rng.uniform(20, 300)); st.phys["PaCO2"] = float(rng.uniform(20, 110))
    st.phys["HR"] = float(rng.uniform(30, 180)); st.phys["SBP"] = float(rng.uniform(50, 180)); st.phys["RR"] = float(rng.uniform(5, 50)); st.phys["Vt"] = float(rng.uniform(150, 600))
    st.flags["arrest"] = bool(rng.random() < 0.1)
    for d, ds in st.drugs.items():
        if rng.random() < 0.4:
            cap = spec["drugs"][d]["typical_dose"]; ds.depot = float(rng.uniform(0, cap)); ds.central = float(rng.uniform(0, cap)); ds.effect = float(rng.uniform(0, cap))
    dev = st.devices; r = rng.random()
    if r < 0.3: dev.oxygen_device = "nrb"; dev.oxygen_flow_lpm = 15
    elif r < 0.45: dev.cpap_on = True; dev.cpap_fio2 = 1.0
    elif r < 0.6: dev.bvm_on = True; dev.bvm_rate = float(rng.choice([8, 12, 20]))
    elif r < 0.7: dev.intubated = True; dev.vent_rate = float(rng.choice([6, 8, 14]))
    if rng.random() < 0.2: dev.position = "supine"
    a = copy.deepcopy(st); b = copy.deepcopy(st)
    code._derive(a); brain._derive(b)
    code._physiology_step(a); brain._physiology_step(b)
    for k in worst:
        va = a.phys.get(k, a.derived.get(k)); vb = b.phys.get(k, b.derived.get(k))
        worst[k] = max(worst[k], abs(va - vb))
    nflag += int(bool(a.flags.get("arrest")) != bool(b.flags.get("arrest")))
    nspeech += int(a.derived["speech"] != b.derived["speech"]); nsilent += int(a.derived["silent_chest"] != b.derived["silent_chest"])
dt = (time.perf_counter() - t0) / N
print("per-tick max abs error over", N, "random states:")
for k, v in worst.items(): print(f"  {k:<9} {v:.3e}")
print("arrest flag mismatches", nflag, "| speech mismatches", nspeech, "| silent-chest mismatches", nsilent)
print(f"mean wall per (code+brain) tick pair {dt*1000:.2f} ms; brain widths {brain.manifest['widths'][:6]}... nonzero {brain.manifest['nonzero_params']}")
