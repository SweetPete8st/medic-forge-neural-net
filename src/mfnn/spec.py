"""Specification loading, validation and a tiny safe condition grammar (no eval)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict

import jsonschema
import yaml

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "spec" / "schema" / "patient_spec.schema.json"

_CMP = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*(<=|>=|==|!=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")


def load_spec(path: str | Path) -> Dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text())
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(spec, schema)
    spec["_hash"] = spec_hash(spec)
    return spec


def spec_hash(spec: Dict[str, Any]) -> str:
    d = {k: v for k, v in spec.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def param(spec: Dict[str, Any], name: str) -> float:
    return float(spec["scenario"]["params"][name]["value"])


def eval_condition(expr: str, env: Dict[str, float]) -> bool:
    """Evaluate 'a < 3 and (b >= 1 or c == 0)' over env, with a restricted grammar.
    Supported: comparisons of a variable against a number, 'and', 'or', parentheses.
    Unknown variables evaluate as 0. No function calls, no arithmetic."""
    tokens = _tokenize(expr)
    pos = 0

    def parse_or():
        nonlocal pos
        left = parse_and()
        while pos < len(tokens) and tokens[pos] == "or":
            pos += 1
            right = parse_and()
            left = left or right
        return left

    def parse_and():
        nonlocal pos
        left = parse_atom()
        while pos < len(tokens) and tokens[pos] == "and":
            pos += 1
            right = parse_atom()
            left = left and right
        return left

    def parse_atom():
        nonlocal pos
        if tokens[pos] == "(":
            pos += 1
            v = parse_or()
            assert tokens[pos] == ")", f"expected ')' in {expr!r}"
            pos += 1
            return v
        tok = tokens[pos]
        pos += 1
        m = _CMP.match(tok)
        if not m:
            raise ValueError(f"bad comparison {tok!r} in {expr!r}")
        var, op, num = m.group(1), m.group(2), float(m.group(3))
        val = float(env.get(var, 0.0))
        return {"<": val < num, "<=": val <= num, ">": val > num, ">=": val >= num,
                "==": val == num, "!=": val != num}[op]

    result = parse_or()
    if pos != len(tokens):
        raise ValueError(f"trailing tokens in {expr!r}")
    return bool(result)


def _tokenize(expr: str):
    out, buf = [], ""
    i = 0
    while i < len(expr):
        c = expr[i]
        if c in "()":
            if buf.strip():
                out.append(buf.strip())
            buf = ""
            out.append(c)
            i += 1
            continue
        if expr.startswith(" and ", i) or expr.startswith(" or ", i):
            if buf.strip():
                out.append(buf.strip())
            buf = ""
            word = "and" if expr.startswith(" and ", i) else "or"
            out.append(word)
            i += len(word) + 2
            continue
        buf += c
        i += 1
    if buf.strip():
        out.append(buf.strip())
    return out
