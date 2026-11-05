"""Write the scenario package: spec sheet + runnable patient + preview + interview record."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import yaml


def write_deliverable(out_root: Path, spec_id: str, spec: Dict[str, Any], overlay: Dict[str, Any], preset_name: str,
                      gate_report: Dict[str, Any], interview: Dict[str, Any], compile_brain: bool = False) -> Path:
    d = out_root / spec_id; d.mkdir(parents=True, exist_ok=True)
    (d / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    (d / "overlay.yaml").write_text(yaml.safe_dump(overlay, sort_keys=False, allow_unicode=True))
    (d / "preset.txt").write_text(preset_name + "\n")
    params = {k: v["value"] for k, v in spec["scenario"]["params"].items()}
    (d / "parameters.json").write_text(json.dumps({"initial": spec["scenario"]["initial"], "params": params}, indent=1))
    (d / "dialogue_facts.yaml").write_text(yaml.safe_dump({"persona": spec["dialogue"]["persona"], "facts": spec["dialogue"]["facts"], "forbidden": spec["dialogue"]["forbidden"]}, sort_keys=False, allow_unicode=True))
    (d / "scripted_events.yaml").write_text(yaml.safe_dump({k: v["actions"] for k, v in spec["trajectories"].items()}, sort_keys=False))
    (d / "gate_report.json").write_text(json.dumps(gate_report, indent=1, default=str))
    (d / "interview.json").write_text(json.dumps(interview, indent=1, default=str))
    lines = [f"# {spec['title']}", "", f"Preset: {preset_name}. Composed {time.strftime('%Y-%m-%d %H:%M')}. Review status: {spec['review']['status']}.", "",
             "## Preview (what happens on each path)", "", "| Path | Events (min) | End SpO2 | End HR | End RR | End PaCO2 | End LOC | Arrest |", "|---|---|---|---|---|---|---|---|"]
    for name, t in gate_report.get("trajectories", {}).items():
        ev = ", ".join(f"{n}@{m}" for m, n in t["events"][:6]); e = t["end"]
        lines.append(f"| {name} | {ev} | {e['SpO2']} | {e['HR']} | {e['RR']} | {e['PaCO2']} | {e['LOC']} | {t['arrest_min'] if t['arrest_min'] is not None else 'no'} |")
    lines += ["", f"Gate: {'PASS' if gate_report.get('pass') else 'NOT PASSED'}; completeness {gate_report['completeness']['score']:.0%}.", "",
              "## Files", "", "- `spec.yaml`: the full specification (edit this; it is the source of truth)", "- `overlay.yaml`: only what the interview changed from the preset",
              "- `parameters.json`, `scripted_events.yaml`, `dialogue_facts.yaml`: the runnable patient", "- `gate_report.json`, `interview.json`: how it was built and checked",
              "", "Not for classroom use until a reviewer signs `spec.yaml` (review.status: APPROVED)."]
    (d / "README.md").write_text("\n".join(lines) + "\n")
    if compile_brain:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments/exp3_brain"))
        from asthma_graph import build_asthma_graph
        from mfnn.wsp.compiler import compile_graph
        net, man = compile_graph(build_asthma_graph(spec)); net.save(str(d / "brain.npz"))
        (d / "brain_manifest.json").write_text(json.dumps({k: v for k, v in man.items() if k != "units"}, indent=1))
    return d
