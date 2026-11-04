#!/usr/bin/env python3
"""Calibration harness for local builders (procedure; needs a running OpenAI-compatible server).

Agent loop: the builder receives the task prompt and may reply with a ```python block; the block runs
in a subprocess sandbox (no network, time limit) with the task's files on the path; stdout (truncated)
is returned. The ONLY scoring feedback is the task's score() output. Every sandbox run and every
score() call is logged to LEDGER.md as a builder run / forward passes.

Nothing in this file has been executed against a real builder in the authoring session."""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tempfile, textwrap, time
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
write must follow from the specification.
IMPORTANT: you have NO tools. Do not call a python tool, a browser tool, or any function. The only way
to run code is to put exactly one fenced block starting with ```python in the TEXT of your reply; the
harness executes it after your reply and sends you the output in the next message. Code that is not
inside such a fenced block is never executed. Use code to check shapes and to call score(), which
returns only aggregate numbers. When finished, write the weights to weights.npz and print DONE.
""").strip()


def validate_pk(workdir: Path) -> str:
    """Load weights_<variant>.npz for the three dev variants, check 6-in/6-out shapes, run the scorer. Returns PASS/FAIL text."""
    import numpy as np
    sys.path.insert(0, str(workdir))
    results = []
    for variant in ("typical", "fast_absorber", "slow_eliminator"):
        f = workdir / f"weights_{variant}.npz"
        if not f.exists(): results.append(f"{variant}: missing"); continue
        try:
            d = np.load(f); Ws = [d[k] for k in sorted(k for k in d.files if k.startswith("W"))]
            if not Ws: results.append(f"{variant}: empty file"); continue
            if Ws[0].shape[1] != 6 or Ws[-1].shape[0] != 6: results.append(f"{variant}: bad shapes {[w.shape for w in Ws]}"); continue
            env = {"PATH": os.environ.get("PATH", ""), "HOME": str(workdir), "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""), "PYTHONPATH": str(workdir)}
            r = subprocess.run([sys.executable, str(workdir / "score.py"), str(f), variant], cwd=workdir, env=env, capture_output=True, text=True, timeout=900)
            line = (r.stdout.strip().splitlines() or [r.stderr.strip()[-200:]])[-1]
            results.append(f"{variant}: {line[:160]}")
        except Exception as e:
            results.append(f"{variant}: error {e.__class__.__name__}: {str(e)[:120]}")
    ok = all("'pass': True" in r or "'pass': np.True_" in r for r in results) and len(results) == 3
    return ("PASS " if ok else "FAIL ") + " | ".join(results)


def chat(endpoint, model, messages, max_tokens=4096, temperature=0.0):
    import urllib.request, urllib.error
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                js = json.loads(r.read())
            msg = js["choices"][0]["message"]
            return (msg.get("content") or "") + ("" if msg.get("content") else f"\n[reasoning only, no content]\n{msg.get('reasoning', '')[-2000:]}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:2000]
            print(f"SERVER ERROR {e.code} from {endpoint}: {detail}", file=sys.stderr)
            if "parsing tool call" in detail:
                return "__TOOL_CALL_ATTEMPT__"      # the model tried to use a built-in tool; caller re-prompts
            if attempt == 0 and e.code >= 500:
                time.sleep(5); continue
            raise SystemExit(f"builder server returned HTTP {e.code}; see the message above (common causes: model tag not pulled or misspelled, "
                             f"model does not fit in VRAM/RAM, context window too small for the prompt: set OLLAMA_CONTEXT_LENGTH=32768 and restart Ollama)")


def run_sandbox(code: str, workdir: Path, timeout_s: int = 600) -> str:
    script = workdir / "step.py"; script.write_text(code)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(workdir), "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
           "PYTHONPATH": os.pathsep.join(str(p) for p in (workdir, ROOT / "src", ROOT / "experiments/exp0_pk", ROOT / "experiments/exp1_calibration"))}
    try:
        p = subprocess.run([sys.executable, str(script)], cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout_s)   # no -I: the builder's code must see the working directory and PYTHONPATH
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
    ap.add_argument("--max-tokens", type=int, default=4096, help="reply budget per turn; reasoning models may need 8192 with a 32k context")
    a = ap.parse_args()
    task = TASKS[a.task]
    workdir = Path(tempfile.mkdtemp(prefix=f"calib_{a.task}_"))
    if a.task == "pk":
        # identical task to the Claude-family test: TASK.md, dev-only spec, layer library and scorer live in the working directory
        tdir = ROOT / "experiments/exp1_calibration/claude_family_attempts"
        shutil.copy(ROOT / "src/mfnn/wsp/layers.py", workdir / "wsp_layers.py")
        shutil.copy(ROOT / "experiments/exp0_pk/pk_spec_dev.yaml", workdir / "pk_spec_dev.yaml")
        shutil.copy(ROOT / "experiments/exp1_calibration/score_local.py", workdir / "score.py")
        task_text = (tdir / "TASK.md").read_text()
        prompt = (task_text + "\n\nFILES IN YOUR WORKING DIRECTORY (the sandbox runs there): wsp_layers.py (the layer library; "
                  "`from wsp_layers import ReLUNet, interpolant_rows, clip01_rows, gated_ramp_rows, threshold_rows, identity_rows`), "
                  "pk_spec_dev.yaml, score.py (run as `python score.py weights_<variant>.npz <variant>`).\n\n"
                  "--- wsp_layers.py ---\n" + (workdir / "wsp_layers.py").read_text() + "\n--- pk_spec_dev.yaml ---\n" + (workdir / "pk_spec_dev.yaml").read_text()
                  + "\n\nThis is a numerical modelling exercise (first-order drug kinetics as linear algebra); there is nothing harmful in it. "
                  "When all three variants pass, print DONE.")
    else:
        parts = [SYSTEM, f"TASK: compile the following specification into weights. Target: {task['target']}."]
        if task["spec"]: parts.append("SPECIFICATION:\n" + Path(task["spec"]).read_text())
        if task["library"]: parts.append("LAYER LIBRARY (importable as given):\n" + Path(task["library"]).read_text())
        parts.append(f"SCORING: `from {Path(task['score_module']).stem} import score; print(score('weights.npz'))`")
        prompt = "\n\n".join(parts)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    t0 = time.time(); runs = 0; outcome = "no_valid_weights"
    while runs < a.max_runs and (time.time() - t0) < a.wall_h * 3600:
        reply = chat(a.builder, a.model, messages, max_tokens=a.max_tokens)
        if reply.startswith("[reasoning only"):
            reasoning_only = locals().get("reasoning_only", 0) + 1
        if "can’t help with that" in reply or "can't help with that" in reply or "cannot help with that" in reply:
            refusals = locals().get("refusals", 0) + 1
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": "This task is a harmless numerical exercise: writing matrix weights that implement first-order kinetics and a saturating readout, for an educational simulator. Please proceed with a ```python block."})
            if refusals > 8:
                outcome = "gave_up_refusals"; break
            continue
        if reply == "__TOOL_CALL_ATTEMPT__":
            tool_attempts = locals().get("tool_attempts", 0) + 1
            if tool_attempts > 6:
                outcome = "gave_up_tool_calls"; break
            messages.append({"role": "user", "content": "Your last reply tried to call a tool. You have no tools. Put the code you wanted to run inside one ```python fenced block in your reply text; it will be executed and the output returned to you."})
            continue
        messages.append({"role": "assistant", "content": reply})
        m = re.search(r"```python\n(.*?)```", reply, re.S)
        if not m:
            messages.append({"role": "user", "content": "Reply with exactly one ```python block in your text (no tool calls)."}); continue
        runs += 1
        out = run_sandbox(m.group(1), workdir)
        messages.append({"role": "user", "content": f"SANDBOX OUTPUT (run {runs}/{a.max_runs}):\n{out}"})
        if "DONE" in out:
            verdict = validate_pk(workdir) if a.task == "pk" else ("weights_written" if (workdir / "weights.npz").exists() else "no_valid_weights")
            if verdict.startswith("PASS") or (a.task != "pk" and verdict == "weights_written"):
                outcome = verdict; break
            messages.append({"role": "user", "content": f"Not finished: {verdict}. Fix it and continue; print DONE only when the scorer passes all three variants."})
            outcome = verdict
    wall = time.time() - t0
    # persist the run inside the repository so it can be committed and scored later
    stamp = time.strftime("%Y%m%d-%H%M%S"); safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", a.model)
    rundir = HERE / "runs" / f"{a.task}_{safe}_{stamp}"; rundir.mkdir(parents=True, exist_ok=True)
    (rundir / "transcript.json").write_text(json.dumps({"model": a.model, "task": a.task, "builder": a.builder, "max_tokens": a.max_tokens,
                                                        "sandbox_runs": runs, "wall_s": wall, "outcome": outcome, "messages": messages}, indent=1))
    for f in workdir.iterdir():
        if f.is_file() and f.suffix in (".npz", ".py", ".md", ".txt", ".json"):
            shutil.copy(f, rundir / f.name)
    outcome += f" | saved {rundir.relative_to(ROOT)}"
    log_builder_run("exp1_calibration", f"{a.model}", f"{a.task}", runs, wall, 1, outcome)
    print(json.dumps({"model": a.model, "task": a.task, "sandbox_runs": runs, "wall_s": wall, "outcome": outcome, "tool_call_attempts": locals().get("tool_attempts", 0),
                      "refusals": locals().get("refusals", 0), "reasoning_only_replies": locals().get("reasoning_only", 0),
                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16], "workdir": str(workdir)}, indent=1))


if __name__ == "__main__":
    main()
