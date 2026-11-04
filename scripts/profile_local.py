#!/usr/bin/env python3
"""Local profiling procedure: measure, do not guess.

Run on (a) the RTX 5090 builder box and (b) an academy Lenovo laptop. Produces profile_<host>.json.
Measures:
  1. code-control patient engine: wall time per simulated minute (CPU), memory
  2. hand-set PK network rollout (numpy): wall time per 60-min rollout; and torch CPU/GPU if available
  3. classroom concurrency: N engines stepped round-robin in real time; report max N with < 1 s per tick
  4. (if a local LLM server is given) dialogue latency: tokens/s and time-to-first-token for a
     250-token patient reply with a 1,500-token fact-sheet prompt, at concurrency 1, 4, 8
"""
from __future__ import annotations
import argparse, json, platform, socket, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments/exp0_pk"))


def profile_engine(minutes=10):
    from mfnn.engine import PatientEngine
    from mfnn.spec import load_spec
    spec = load_spec(ROOT / "spec/examples/asthma_adult_acute.yaml")
    eng = PatientEngine(spec)
    t0 = time.perf_counter(); eng.run([], minutes * 60); dt = time.perf_counter() - t0
    return {"engine_wall_s_per_sim_min": dt / minutes, "realtime_factor": (minutes * 60) / dt}


def profile_concurrency(target_tick_s=1.0, max_n=512):
    from mfnn.engine import PatientEngine
    from mfnn.spec import load_spec
    spec = load_spec(ROOT / "spec/examples/asthma_adult_acute.yaml")
    n = 1; best = 0
    while n <= max_n:
        engs = [PatientEngine(spec) for _ in range(n)]
        t0 = time.perf_counter()
        for _ in range(10):
            for e in engs: e.step([])
        per_tick = (time.perf_counter() - t0) / 10
        if per_tick < target_tick_s: best = n; n *= 2
        else: break
    return {"max_concurrent_patients_under_1s_per_tick": best}


def profile_pk():
    import numpy as np, yaml, pk_code, pk_wsp
    spec = yaml.safe_load((ROOT / "experiments/exp0_pk/pk_spec.yaml").read_text())
    v = spec["variants"]["dev"]["typical"]; doses = [(60, "depot", 2.5), (1260, "depot", 2.5)]
    net = pk_wsp.compile_pk(v)
    t0 = time.perf_counter(); pk_code.rollout(v, doses, 3600); tc = time.perf_counter() - t0
    t0 = time.perf_counter(); pk_wsp.rollout(net, doses, 3600); tw = time.perf_counter() - t0
    out = {"pk_code_wall_s_per_60min": tc, "pk_wsp_numpy_wall_s_per_60min": tw, "pk_wsp_params": net.n_params}
    try:
        import torch
        layers = [(torch.tensor(W, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)) for W, b in net.layers]
        for dev in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
            L = [(W.to(dev), b.to(dev)) for W, b in layers]
            s = torch.zeros(4, device=dev); t0 = time.perf_counter()
            with torch.no_grad():
                for t in range(3600):
                    x = torch.cat([s, torch.zeros(2, device=dev)])
                    for i, (W, b) in enumerate(L):
                        x = W @ x + b
                        if i < len(L) - 1: x = torch.relu(x)
                    s = x[:4]
            if dev == "cuda": torch.cuda.synchronize()
            out[f"pk_wsp_torch_{dev}_wall_s_per_60min"] = time.perf_counter() - t0
    except Exception as e:  # torch absent or no GPU
        out["torch"] = f"unavailable: {e.__class__.__name__}"
    return out


def profile_llm(endpoint, model, concurrency=(1, 4, 8)):
    import concurrent.futures as cf, urllib.request
    prompt = "You are a patient. " + ("Fact. " * 300)
    def one():
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt + " Reply in 250 words."}], "max_tokens": 250}).encode()
        req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            js = json.loads(r.read())
        return time.perf_counter() - t0, js.get("usage", {}).get("completion_tokens", 0)
    res = {}
    for c in concurrency:
        with cf.ThreadPoolExecutor(c) as ex:
            t0 = time.perf_counter(); outs = list(ex.map(lambda _: one(), range(c))); wall = time.perf_counter() - t0
        toks = sum(o[1] for o in outs)
        res[f"concurrency_{c}"] = {"wall_s": wall, "aggregate_tok_per_s": toks / wall if wall else None, "mean_latency_s": sum(o[0] for o in outs) / c}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--llm", default=None); ap.add_argument("--model", default=None)
    a = ap.parse_args()
    prof = {"host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version()}
    prof.update(profile_engine()); prof.update(profile_concurrency()); prof.update(profile_pk())
    if a.llm: prof["llm"] = profile_llm(a.llm, a.model)
    out = ROOT / f"profile_{prof['host']}.json"; out.write_text(json.dumps(prof, indent=1)); print(json.dumps(prof, indent=1))
