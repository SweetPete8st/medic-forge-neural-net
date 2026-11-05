import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from mfnn.parent.interview import Interview, OPENING
from mfnn.parent.llm import MockBackend
from mfnn.learning.store import InterviewStore
from mfnn.learning.importer import import_legacy, parse_text, scrub
from mfnn.learning.grading import auto_grade, human_grade_prompt
from mfnn.learning.dataset import build_sft
from mfnn.learning.promote import Registry
from mfnn.learning.evaluate import simulate

FULL_SLOTS = {
    "severity_on_arrival": {"value": "severe", "confidence": 1.0}, "deterioration_pace": {"value": "typical", "confidence": 0.9},
    "treatment_responsiveness": {"value": "poor", "confidence": 1.0}, "traps": {"value": ["sedation", "over_ventilation"], "confidence": 1.0},
    "duration_min": {"value": 45, "confidence": 1.0}, "age": {"value": 52, "confidence": 1.0}, "sex": {"value": "male", "confidence": 1.0},
    "weight_kg": {"value": 90, "confidence": 0.9}, "chief_complaint": {"value": "Can't breathe, chest feels tight", "confidence": 1.0},
    "history_of_present_illness": {"value": "Started 2 hours ago after mowing, inhaler not helping", "confidence": 1.0},
    "past_medical_history": {"value": ["asthma", "hypertension"], "confidence": 1.0}, "medications": {"value": ["albuterol inhaler", "lisinopril"], "confidence": 1.0},
    "allergies": {"value": ["sulfa"], "confidence": 1.0}, "persona": {"value": "anxious, minimises symptoms", "confidence": 0.9},
    "teaching_objectives": {"value": ["recognise silent chest", "avoid sedation"], "confidence": 1.0},
}


def _turn(say, slots, done=False):
    return json.dumps({"say": say, "slots": slots, "unsupported": [], "done": done})


def test_interview_to_deliverable(tmp_path):
    script = [_turn("Got it. How old is the patient and how sick on arrival?", {k: FULL_SLOTS[k] for k in ("chief_complaint", "history_of_present_illness", "persona")}),
              _turn("Thanks. I have everything: a 52-year-old man, severe on arrival, poor responder. Ready.", {k: v for k, v in FULL_SLOTS.items() if k not in ("chief_complaint", "history_of_present_illness", "persona")}, done=True)]
    iv = Interview(MockBackend(script), "asthma_adult_acute", threshold=0.8, scenario_id="test_case")
    r1 = iv.step("52 year old guy, severe asthma attack, poor responder, make it punish sedation and bagging too fast")
    assert not r1["done"] and r1["completeness"]["score"] < 1.0
    r2 = iv.step("he's 90 kg, 45 minutes, objectives: silent chest and avoid sedation; allergic to sulfa")
    assert r2["done"] and r2["completeness"]["complete"]
    res = iv.build(tmp_path / "scenarios")
    g = res["gate"]
    assert g["schema_valid"] and g["treatment_helps"]
    assert set(g["trajectories"]) == {"untreated", "timely_treatment", "delayed_treatment", "inappropriate_intervention"}
    assert all(not t["invariant_violations"] for t in g["trajectories"].values())
    assert g["objectives"]["recognise silent chest"]["reachable"] and g["objectives"]["avoid sedation"]["reachable"]
    assert g["pass"], g
    d = Path(res["path"])
    for f in ("spec.yaml", "overlay.yaml", "parameters.json", "dialogue_facts.yaml", "scripted_events.yaml", "gate_report.json", "interview.json", "README.md"):
        assert (d / f).exists(), f
    spec = res["spec"]
    assert spec["patient"]["age"] == 52 and spec["scenario"]["params"]["beta_agonist_responsiveness"]["value"] == 0.30
    assert spec["scenario"]["initial"]["O"] == 0.62 and spec["review"]["status"] == "DRAFT"
    assert "sulfa" in spec["dialogue"]["facts"]["allergies"]
    assert any(a.get("drug") == "midazolam_iv" for a in spec["trajectories"]["inappropriate_intervention"]["actions"])


def test_gate_refuses_incomplete(tmp_path):
    iv = Interview(MockBackend([_turn("ok", {"age": {"value": 40, "confidence": 1.0}}, done=True)]), scenario_id="incomplete")
    r = iv.step("40 year old")
    assert not r["done"]
    res = iv.build(tmp_path / "s")
    assert not res["gate"]["pass"] and res["gate"]["completeness"]["missing"]


def test_importer_and_scrub(tmp_path):
    txt = "Student: Hi, what's your name?\nPatient: Call me Dave, I can't breathe, my number is 555-123-4567\nStudent: any allergies?\nPatient: penicillin, email me at dave@example.com\n"
    (tmp_path / "t1.txt").write_text(txt)
    (tmp_path / "t2.jsonl").write_text(json.dumps({"id": "iv7", "turns": [{"role": "user", "text": "I want a COPD patient"}, {"role": "assistant", "text": "How old?"}]}) + "\n")
    store = InterviewStore(tmp_path / "store")
    counts = import_legacy([tmp_path / "t1.txt", tmp_path / "t2.jsonl"], store, names=["Dave"])
    assert counts["records"] == 2 and counts["patient_dialogue"] == 1 and counts["builder_interview"] == 1
    recs = list(store.iter("interviews"))
    blob = json.dumps(recs)
    assert "555-123-4567" not in blob and "dave@example.com" not in blob and "[name]" in blob
    assert scrub("DOB: 01/02/1980") == "[dob]"
    turns = parse_text("[12:01] Instructor: build me a patient\nBuilder: sure")
    assert turns[0]["role"] == "instructor" and turns[1]["role"] == "builder"


def test_grading_dataset_and_promotion(tmp_path):
    store = InterviewStore(tmp_path / "store")
    rec = {"kind": "builder_interview", "transcript": [{"role": "builder", "text": OPENING}, {"role": "instructor", "text": "asthma, 30 year old"},
                                                       {"role": "builder", "text": "How sick on arrival? Any allergies?", "slots": {"age": {"value": 30, "confidence": 1.0}}, "done": False}], "unsupported": []}
    rid = store.add_interview(rec, gate={"pass": True})
    ag = auto_grade({**rec, "gate_pass": True, "record": rec})
    assert ag["kind"] == "builder_interview" and 0 <= ag["slot_coverage"] <= 1
    assert "Rubric" in human_grade_prompt({**rec, "id": rid})
    out = tmp_path / "sft.jsonl"
    c0 = build_sft(store, out); assert c0["written"] == 0 and c0["skipped_ungraded"] == 1
    store.add_grade(rid, "david", {"efficiency": 5, "fidelity": 4, "honesty": 5, "clinical_sense": 4})
    c1 = build_sft(store, out); assert c1["builder_examples"] == 1
    ex = json.loads(out.read_text().splitlines()[0]); assert ex["messages"][-1]["role"] == "assistant" and "slots" in ex["messages"][-1]["content"]
    reg = Registry(tmp_path / "registry.json")
    v1 = reg.register("anthropic", "claude-opus-5-5", {"completed_rate": 0.8, "gate_pass_rate": 0.7, "mean_slot_accuracy_enum": 0.9, "mean_turns": 6})
    assert reg.compare(v1)["eligible"]
    reg.approve(v1, "David"); assert reg.current()["id"] == v1
    v2 = reg.register("local", "qwen3.5:9b", {"completed_rate": 0.5, "gate_pass_rate": 0.7, "mean_slot_accuracy_enum": 0.9, "mean_turns": 6})
    assert not reg.compare(v2)["eligible"]
    import pytest
    with pytest.raises(ValueError): reg.approve(v2, "David")
    with pytest.raises(ValueError): reg.approve(v1, "claude")


def test_simulated_instructor_eval(tmp_path):
    brief = {"id": "b1", "preset": "asthma_adult_acute", "target": {"severity_on_arrival": "moderate", "deterioration_pace": "slow", "treatment_responsiveness": "good", "sex": "female", "age": 28}}
    parent = MockBackend([_turn("Tell me more", {}), _turn("All set.", {**{k: {"value": v, "confidence": 1.0} for k, v in brief["target"].items()},
                                                                      **{k: FULL_SLOTS[k] for k in FULL_SLOTS if k not in brief["target"]}}, done=True)])
    instructor = MockBackend(["A 28 year old woman with moderate asthma.", "Slow deterioration, responds well, 90 kg, 45 minutes, objectives silent chest."])
    res = simulate(parent, instructor, brief, out_root=tmp_path / "eval")
    assert res["completed"] and res["slot_accuracy_enum"] == 1.0 and res["same_family_warning"]
    assert res.get("gate_pass") is True, res
