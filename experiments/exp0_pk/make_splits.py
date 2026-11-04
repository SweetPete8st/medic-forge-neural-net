"""Writes the fixed dev / held-out split files with the spec hash, BEFORE any build. Idempotent and
refuses to overwrite a predefined split that no longer matches the spec (RULES R3).
splits.json  = v1 (dev + heldout). Its held-out block is BURNED after the v1 run (see RESULTS_v1.md).
splits_v2.json = v2 (dev + heldout + heldout_v2)."""
import hashlib, json, sys
from pathlib import Path
import yaml
HERE = Path(__file__).resolve().parent
spec = yaml.safe_load((HERE / "pk_spec.yaml").read_text())


def block(name):
    return {"variants": sorted(spec["variants"][name]), "schedules": sorted(spec["schedules"][name])}


def write(path, splits):
    if path.exists():
        old = json.loads(path.read_text())
        if old != splits:
            sys.exit(f"{path.name} differs from the spec: refusing to overwrite a predefined split (RULES R3)")
        print(f"{path.name} unchanged")
    else:
        path.write_text(json.dumps(splits, indent=1)); print(f"{path.name} written")


v2 = {"seed": 20261104, "dev": block("dev"), "heldout": block("heldout"), "heldout_v2": block("heldout_v2"),
      "spec_sha256": hashlib.sha256((HERE / "pk_spec.yaml").read_bytes()).hexdigest()}
write(HERE / "splits_v2.json", v2)
