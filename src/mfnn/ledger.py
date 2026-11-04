"""Ledger tooling (RULES.md R1, R2, E4). Appends machine-generated lines to LEDGER.md and keeps
per-experiment counters in .ledger_counters.json (committed)."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "LEDGER.md"
COUNTERS = ROOT / ".ledger_counters.json"


def _today() -> str:
    return _dt.date.today().isoformat()


def _load():
    return json.loads(COUNTERS.read_text()) if COUNTERS.exists() else {}


def _save(c):
    COUNTERS.write_text(json.dumps(c, indent=2, sort_keys=True))


def _append(line: str) -> None:
    with LEDGER.open("a") as f:
        f.write(line.rstrip() + "\n")


def log_forward_passes(exp: str, n: int, what: str, split: str = "dev") -> int:
    assert split in ("dev", "heldout")
    c = _load()
    e = c.setdefault(exp, {"fp": 0, "dc": 0, "br": 0, "frozen": None})
    if split == "heldout" and not e.get("frozen"):
        raise RuntimeError(f"{exp}: held-out evaluation attempted before freeze (RULES R4)")
    e["fp"] += n
    _save(c)
    _append(f"[auto] FP {e['fp']} | {_today()} | {exp} | +{n}: {what} | {split}")
    return e["fp"]


def log_design_change(exp: str, change: str, reason: str, data_informed: bool) -> int:
    c = _load()
    e = c.setdefault(exp, {"fp": 0, "dc": 0, "br": 0, "frozen": None})
    e["dc"] += 1
    _save(c)
    _append(f"[auto] DC {e['dc']} | {_today()} | {exp} | {change} | reason: {reason} | DATA-INFORMED: {'yes' if data_informed else 'no'}")
    return e["dc"]


def log_builder_run(exp: str, builder: str, task: str, sandbox_runs: int, wall_s: float, attempts: int, outcome: str) -> int:
    c = _load()
    e = c.setdefault(exp, {"fp": 0, "dc": 0, "br": 0, "frozen": None})
    e["br"] += 1
    _save(c)
    _append(f"[auto] BR {e['br']} | {_today()} | {builder} | {task} | sandbox runs: {sandbox_runs} | wall: {wall_s:.0f}s | attempts: {attempts} | {outcome}")
    return e["br"]


def freeze(exp: str, weights_hash: str, note: str = "") -> None:
    c = _load()
    e = c.setdefault(exp, {"fp": 0, "dc": 0, "br": 0, "frozen": None})
    if e.get("frozen"):
        raise RuntimeError(f"{exp}: already frozen at {e['frozen']}")
    e["frozen"] = weights_hash
    _save(c)
    _append(f"[auto] FREEZE | {_today()} | {exp} | weights sha256:{weights_hash} | {note}")


def is_frozen(exp: str) -> Optional[str]:
    return _load().get(exp, {}).get("frozen")
