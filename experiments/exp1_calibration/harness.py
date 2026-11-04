#!/usr/bin/env python3
"""Calibration harness for local builders (procedure; needs a running OpenAI-compatible server).

Agent loop: the builder receives the task prompt and may reply with a ```python block; the block runs
in a subprocess sandbox (no network, time limit) with the task's files on the path; stdout (truncated)
is returned. The ONLY scoring feedback is the task's score() output. Every sandbox run and every
score() call is logged to LEDGER.md as a builder run / forward passes.

Nothing in this file has been executed against a real builder in the authoring session."""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys, tempfile, textwrap, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mfnn.ledger import log_builder_run  # noqa: E402

HERE = Path(__file__).resolve().parent
TASKS = {
    "pk": {"spec": ROOT / "experiments/exp0_pk/pk_spec_dev.yaml", "library": ROOT / "src/mfnn/wsp/layers.py",
           "score_module": "experiments/exp0_pk/score_pk.py", "target": "max |dE| <= 0.01 on the dev cases"},
    "ecg": {"spec": HERE / "ecg-wsp-v4/src/wsp/detector.py", "library": HERE / "ecg-wsp-v4/src/wsp/layers.py",
            "score_module": "experiments/exp1_calibration/score_ecg.py", "target": "sensitivity >= 0.97 and PPV >= 0.97 on MIT-BIH record 100"},
    "toy": {"spec": None, "library": None, "score_module": "experiments/exp1_calibration/score_toy.py", "target": "100% on the held-out set"},
}

SYSTEM = textwrap.dedent("""
You are compiling an explicit, fully specified algorithm into neural-network weights by reasoning.
Rules: no training, no gradient descent, no parameter search, no threshold tuning. Every weight you
write must follow from the specification. You may run Python in a sandbox to check shapes and to
call score(); score() returns only the aggregate numbers. Reply with one ```python block per turn.
When finished, write the weights to weights.npz and print DONE.
""").strip()


def chat(endpoint, model, messages, max_tokens=8000, temperature=0.0):
    import urllib.request
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def run_sandbox(code: str, workdir: Path, timeout_s: int = 600) -> str:
    script = workdir / "step.py"; script.write_text(code)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(workdir), "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
           "PYTHONPATH": os.pathsep.join(str(p) for p in (ROOT / "src", ROOT / "experiments/exp0_pk", ROOT / "experiments/exp1_calibration"))}
    try:
        p = subprocess.run([sys.executable, "-I", str(script)], cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout_s)
        out = (p.stdout + "\n" + p.stderr)
    except subprocess.TimeoutExpired:
        out = f"TIMEOUT after {timeout_s}s"
    return out[-6000:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builder", default="http://127.0.0.1:11434/v1", help="OpenAI-compatible base URL (Ollama default shown; llama-server: http://127.0.0.1:8080/v1)")
    ap.add_argument("--model", required=True, help="model id as the server names it, e.g. gpt-oss:20b")
    ap.add_argument("--task", choices=TASKS, required=True); ap.add_argument("--toy", default="addition")
    ap.add_argument("--max-runs", type=int, default=40); ap.add_argument("--wall-h", type=float, default=6.0)
    a = ap.parse_args()
    task = TASKS[a.task]
    parts = [SYSTEM, f"TASK: compile the following specification into weights. Target: {task['target']}."]
    if task["spec"]: parts.append("SPECIFICATION:\n" + Path(task["spec"]).read_text())
    if task["library"]: parts.append("LAYER LIBRARY (importable as given):\n" + Path(task["library"]).read_text())
    parts.append(f"SCORING: `from {Path(task['score_module']).stem} import score; print(score('weights.npz'))`")
    prompt = "\n\n".join(parts)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    workdir = Path(tempfile.mkdtemp(prefix=f"calib_{a.task}_"))
    t0 = time.time(); runs = 0; outcome = "no_valid_weights"
    while runs < a.max_runs and (time.time() - t0) < a.wall_h * 3600:
        reply = chat(a.builder, a.model, messages)
        messages.append({"role": "assistant", "content": reply})
        m = re.search(r"```python\n(.*?)```", reply, re.S)
        if not m:
            messages.append({"role": "user", "content": "Reply with exactly one ```python block."}); continue
        runs += 1
        out = run_sandbox(m.group(1), workdir)
        messages.append({"role": "user", "content": f"SANDBOX OUTPUT (run {runs}/{a.max_runs}):\n{out}"})
        if "DONE" in out and (workdir / "weights.npz").exists():
            outcome = "weights_written"; break
    wall = time.time() - t0
    log_builder_run("exp1_calibration", f"{a.model}", f"{a.task}", runs, wall, 1, outcome)
    print(json.dumps({"model": a.model, "task": a.task, "sandbox_runs": runs, "wall_s": wall, "outcome": outcome,
                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16], "workdir": str(workdir)}, indent=1))


if __name__ == "__main__":
    main()
