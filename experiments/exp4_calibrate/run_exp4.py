#!/usr/bin/env python3
"""exp4: differentiable calibration of the compiled brain (torch) vs black-box fitting of the code engine.
Usage: run_exp4.py check | brain | code | all"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments/exp3_brain")); sys.path.insert(0, str(ROOT / "experiments/exp2_asthma")); sys.path.insert(0, str(HERE))
from mfnn.engine import PatientEngine
from mfnn.ledger import log_forward_passes
from mfnn.spec import load_spec
from brain_runner import BrainEngine
from asthma_graph import STATE, PORT
from run_trajectories import actions_from_spec
from torch_export import TorchReLUNet

EXP = "exp4_calibrate"; RES = HERE / "results"; RES.mkdir(exist_ok=True)
TRUE_PORT = {"beta_agonist_responsiveness": 0.45, "progression_rate": 0.035, "fatigue_rate": 0.018}
START_PORT = {"beta_agonist_responsiveness": 0.8, "progression_rate": 0.02, "fatigue_rate": 0.012}
OBS = ["SpO2", "HR", "RR", "PaCO2"]; OBS_SCALE = {"SpO2": 100.0, "HR": 200.0, "RR": 60.0, "PaCO2": 150.0}
import argparse as _ap_mod
_ap = _ap_mod.ArgumentParser(); _ap.add_argument("phase", nargs="?", default="all"); _ap.add_argument("--minutes", type=int, default=30)
_ap.add_argument("--steps", type=int, default=40); _ap.add_argument("--scripts", default="timely_treatment,delayed_treatment"); _ap.add_argument("--maxfev", type=int, default=80)
ARGS = _ap.parse_args()
MINUTES = ARGS.minutes; SAMPLE_EVERY = 30
spec = load_spec(ROOT / "spec/examples/asthma_adult_acute.yaml")
SCRIPTS = {k: actions_from_spec(spec["trajectories"][k]["actions"]) for k in ARGS.scripts.split(",")}


def code_observables(port):
    out = {}
    for name, acts in SCRIPTS.items():
        e = PatientEngine(spec); e.p.update(port); tr, _ = e.run(acts, MINUTES * 60, record_every=1)
        out[name] = np.array([[r[k] for k in OBS] for r in tr[SAMPLE_EVERY - 1::SAMPLE_EVERY]])
    return out


def loss_np(obs, target):
    return float(np.mean([np.mean(((obs[n] - target[n]) / np.array([OBS_SCALE[k] for k in OBS])) ** 2) for n in target]))


class BrainRollout:
    """Differentiable rollout: the shell's device/dose logic is precomputed per tick from the action log."""
    def __init__(self, brain: BrainEngine, acts, minutes):
        self.brain = brain; self.tnet = TorchReLUNet(brain.net); self.n = minutes * 60
        ii = brain.in_idx; self.ii = ii; self.oi = brain.out_idx
        # precompute device inputs and dose inputs per tick by replaying the shell with a dummy physiology
        shell = BrainEngine(spec, net=brain.net, manifest=brain.manifest)
        by_tick = {}
        for a in acts: by_tick.setdefault(a.tick, []).append(a)
        self.dev = np.zeros((self.n, len(ii))); self.dose = np.zeros((self.n, len(ii)))
        for t in range(self.n):
            before = {d: (ds.depot, ds.central) for d, ds in shell.state.drugs.items()}
            for a in sorted(by_tick.get(t, []), key=lambda a: a.key()): shell._apply_action(shell.state, a)
            for d, ds in shell.state.drugs.items():
                dd = (ds.depot - before[d][0]) + (ds.central - before[d][1])
                if dd > 0: self.dose[t, ii[f"{d}.dose"]] = dd; ds.depot, ds.central = before[d]   # undo, the net adds it
            x = shell._inputs(shell.state)
            for k in ("fio2", "assisted_rate", "cpap_on", "intubated", "supine", "vt_assist"): self.dev[t, ii[k]] = x[ii[k]]
            shell.state.tick += 1
        self.x0 = torch.tensor(BrainEngine(spec, net=brain.net, manifest=brain.manifest)._inputs(brain.state), dtype=torch.float64)
        self.state_idx = [ii[n] for n in STATE] + [ii[f"{d}.{c}"] for d in spec["drugs"] for c in ("depot", "central", "effect")]
        self.state_out = [self.oi[n] for n in STATE] + [self.oi[f"{d}.{c}"] for d in spec["drugs"] for c in ("depot", "central", "effect")]
        self.port_idx = [ii[k] for k in PORT]; self.obs_out = [self.oi[k] if k in self.oi else None for k in OBS]
        self.dev_t = torch.tensor(self.dev, dtype=torch.float64); self.dose_t = torch.tensor(self.dose, dtype=torch.float64)

    def run(self, port_vec: torch.Tensor):
        x = self.x0.clone(); rows = []
        state = x[self.state_idx]
        for t in range(self.n):
            xt = torch.zeros_like(x); xt = xt + self.dev_t[t] + self.dose_t[t]
            xt = xt.index_put((torch.tensor(self.state_idx),), state)
            xt = xt.index_put((torch.tensor(self.port_idx),), port_vec)
            y = self.tnet(xt)
            state = y[self.state_out]
            if (t + 1) % SAMPLE_EVERY == 0:
                rows.append(torch.stack([y[self.oi["SpO2"]], y[self.oi["HR"]], y[self.oi["RR"]], y[self.oi["PaCO2"]]]))
        return torch.stack(rows)


def main():
    phase = ARGS.phase
    target = code_observables(TRUE_PORT)
    out = json.loads((RES / "results.json").read_text()) if (RES / "results.json").exists() else {}
    out.update({"true": TRUE_PORT, "start": START_PORT, "minutes": MINUTES})
    if phase in ("check", "all"):
        brain = BrainEngine(spec, port_overrides=START_PORT); ro = BrainRollout(brain, SCRIPTS["timely_treatment"], MINUTES)
        with torch.no_grad(): obs_t = ro.run(torch.tensor([START_PORT[k] for k in PORT], dtype=torch.float64)).numpy()
        tr, _ = brain.run(SCRIPTS["timely_treatment"], MINUTES * 60, record_every=1)
        obs_np = np.array([[r[k] for k in OBS] for r in tr[SAMPLE_EVERY - 1::SAMPLE_EVERY]])
        out["torch_vs_numpy_max_abs"] = float(np.abs(obs_t - obs_np).max()); print("torch vs numpy brain rollout max abs diff:", out["torch_vs_numpy_max_abs"])
    if phase in ("brain", "all"):
        brain = BrainEngine(spec, port_overrides=START_PORT)
        ros = {n: BrainRollout(brain, SCRIPTS[n], MINUTES) for n in SCRIPTS}
        logp = torch.tensor([math.log(START_PORT[k]) for k in PORT], dtype=torch.float64, requires_grad=True)
        optim = torch.optim.Adam([logp], lr=0.05); t0 = time.perf_counter(); hist = []; n_roll = 0
        tgt = {n: torch.tensor(target[n], dtype=torch.float64) for n in SCRIPTS}
        scale = torch.tensor([OBS_SCALE[k] for k in OBS], dtype=torch.float64)
        for step in range(ARGS.steps):
            optim.zero_grad(); loss = 0.0
            for n, ro in ros.items():
                obs = ro.run(torch.exp(logp)); loss = loss + torch.mean(((obs - tgt[n]) / scale) ** 2); n_roll += 1
            loss = loss / len(ros); loss.backward(); optim.step()
            with torch.no_grad(): logp.clamp_(min=math.log(1e-3), max=math.log(1.0))
            hist.append(float(loss)); msg = f"brain-grad step {step:2d} loss {float(loss):.3e} port {[round(float(v),4) for v in torch.exp(logp)]} ({time.perf_counter()-t0:.0f}s)"
            print(msg, flush=True); (RES / "brain_grad_progress.log").open("a").write(msg + "\n")
            if float(loss) < 1e-6: break
        rec = {k: float(v) for k, v in zip(PORT, torch.exp(logp).detach())}
        out["brain_grad"] = {"recovered": rec, "final_loss": hist[-1], "steps": len(hist), "rollouts": n_roll, "wall_s": time.perf_counter() - t0,
                             "rel_err": {k: abs(rec[k] - TRUE_PORT[k]) / TRUE_PORT[k] for k in PORT}, "loss_history": hist}
        log_forward_passes(EXP, n_roll, "brain-grad differentiable rollouts (torch)", "dev")
    if phase in ("code_fd",):
        # fair comparison: the same Adam schedule as brain-grad, gradients by forward finite differences on the code engine
        t0 = time.perf_counter(); n_roll = [0]
        def f(logv):
            port = {k: float(math.exp(v)) for k, v in zip(PORT, logv)}; n_roll[0] += 2 * len(SCRIPTS)
            return loss_np(code_observables(port), target)
        logv = np.array([math.log(START_PORT[k]) for k in PORT]); m = np.zeros(3); v2 = np.zeros(3); hist = []; eps = 0.02
        for step in range(ARGS.steps):
            base = f(logv); g = np.zeros(3)
            for i in range(3):
                d = np.zeros(3); d[i] = eps; g[i] = (f(logv + d) - base) / eps
            m = 0.9 * m + 0.1 * g; v2 = 0.999 * v2 + 0.001 * g * g
            mh = m / (1 - 0.9 ** (step + 1)); vh = v2 / (1 - 0.999 ** (step + 1))
            logv = np.clip(logv - 0.05 * mh / (np.sqrt(vh) + 1e-8), math.log(1e-3), math.log(1.0))
            hist.append(float(base)); msg = f"code-fd step {step:2d} loss {base:.3e} port {[round(float(math.exp(x)),4) for x in logv]} ({time.perf_counter()-t0:.0f}s)"
            print(msg, flush=True); (RES / "code_fd_progress.log").open("a").write(msg + "\n")
        rec = {k: float(math.exp(x)) for k, x in zip(PORT, logv)}
        out["code_fd"] = {"recovered": rec, "final_loss": f(logv), "steps": ARGS.steps, "rollouts": n_roll[0], "wall_s": time.perf_counter() - t0,
                          "rel_err": {k: abs(rec[k] - TRUE_PORT[k]) / TRUE_PORT[k] for k in PORT}, "loss_history": hist}
        log_forward_passes(EXP, n_roll[0], "code finite-difference Adam rollouts", "dev")
        print("code-fd:", json.dumps({k: v for k, v in out["code_fd"].items() if k != "loss_history"}, indent=1))
    if phase in ("code", "all"):
        from scipy.optimize import minimize
        t0 = time.perf_counter(); evals = [0]
        def f(logv):
            evals[0] += 1; port = {k: float(math.exp(v)) for k, v in zip(PORT, logv)}
            return loss_np(code_observables(port), target)
        r = minimize(f, [math.log(START_PORT[k]) for k in PORT], method="Nelder-Mead", options={"maxfev": ARGS.maxfev, "xatol": 1e-3, "fatol": 1e-8})
        rec = {k: float(math.exp(v)) for k, v in zip(PORT, r.x)}
        out["code_search"] = {"recovered": rec, "final_loss": float(r.fun), "evaluations": evals[0], "rollouts": 2 * evals[0], "wall_s": time.perf_counter() - t0,
                              "rel_err": {k: abs(rec[k] - TRUE_PORT[k]) / TRUE_PORT[k] for k in PORT}}
        print("code-search:", json.dumps(out["code_search"], indent=1))
    (RES / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
