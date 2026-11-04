import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfnn.actions import ActionKind, LearnerAction
from mfnn.dialogue import check_utterance, fact_sheet
from mfnn.engine import PatientEngine, severinghaus_spo2
from mfnn.observe import observe
from mfnn.spec import eval_condition, load_spec

SPEC = ROOT / "spec" / "examples" / "asthma_adult_acute.yaml"


def test_condition_grammar():
    env = {"O": 0.7, "F": 0.2, "silent_chest": 1}
    assert eval_condition("O >= 0.65 or F > 0.5", env)
    assert not eval_condition("O < 0.5 and F < 0.5", env)
    assert eval_condition("(O < 0.5 or silent_chest == 1) and F < 0.5", env)


def test_severinghaus_anchor_points():
    assert 96 < severinghaus_spo2(90) < 98
    assert 88 < severinghaus_spo2(60) < 92
    assert 70 < severinghaus_spo2(40) < 78


def test_replay_is_deterministic():
    spec = load_spec(SPEC)
    acts = [LearnerAction(120, ActionKind.GIVE_DRUG, {"drug": "albuterol", "dose": 2.5}),
            LearnerAction(120, ActionKind.OXYGEN, {"device": "nrb", "flow_lpm": 15}, seq=1)]
    a = PatientEngine(spec); ta, ea = a.run(acts, 1800)
    b = PatientEngine(spec); tb, eb = b.run(acts, 1800)
    assert a.state.digest() == b.state.digest()
    assert [(e.tick, e.name) for e in ea] == [(e.tick, e.name) for e in eb]


def test_bronchodilator_lowers_obstruction_vs_control():
    spec = load_spec(SPEC)
    c = PatientEngine(spec); tc, _ = c.run([], 1800)
    t = PatientEngine(spec); tt, _ = t.run([LearnerAction(60, ActionKind.GIVE_DRUG, {"drug": "albuterol", "dose": 2.5})], 1800)
    assert tt[-1]["O"] < tc[-1]["O"]
    assert all(rt["O"] <= rc["O"] + 1e-9 for rt, rc in zip(tt, tc))


def test_observation_gating():
    spec = load_spec(SPEC)
    e = PatientEngine(spec)
    o = observe(e)
    assert "SpO2" not in o.values
    e.step([LearnerAction(0, ActionKind.ATTACH_DEVICE, {"device": "spo2"})])
    o = observe(e)
    assert "SpO2" in o.values and "breath_sounds" not in o.values
    o = observe(e, ActionKind.ASSESS_AUSCULTATE)
    assert "breath_sounds" in o.values


def test_dialogue_checker():
    spec = load_spec(SPEC)
    e = PatientEngine(spec)
    sheet = fact_sheet(e)
    assert sheet["speech_mode"] in ("phrases", "words")
    v = check_utterance("I feel much better now, my sats are 98 percent and I take metformin.", sheet)
    assert "stated_vital_sign_number" in v and "medication_not_in_facts:metformin" in v
    assert not check_utterance("Can't... breathe.", sheet)
