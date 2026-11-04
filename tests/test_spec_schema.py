import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import jsonschema, yaml
from mfnn.spec import load_spec, SCHEMA_PATH


def test_example_validates():
    spec = load_spec(ROOT / "spec/examples/asthma_adult_acute.yaml")
    assert spec["review"]["status"] == "UNREVIEWED"


def test_every_param_has_source_and_grade():
    spec = load_spec(ROOT / "spec/examples/asthma_adult_acute.yaml")
    for name, p in spec["scenario"]["params"].items():
        assert "source" in p and "grade" in p["source"], name
    for name, d in spec["drugs"].items():
        assert "source" in d, name
