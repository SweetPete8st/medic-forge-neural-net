# Task: compile a drug-kinetics specification into neural-network weights by reasoning (no training)

You are a builder in a calibration experiment. Work ONLY inside your attempt directory (given in your
instructions). Do NOT read or search any other directory on this machine (in particular nothing under
/home/user), and do not look for existing solutions. Use only the files in this task directory.

## Specification (the source of truth)
Discrete time, dt = 1 s. State s = [A_depot, A_central, C_effect, tau_since_dose]. Inputs per tick
u = [dose_to_depot, dose_to_central] (amounts >= 0; both zero on ticks with no dose).
Per tick, in this order:
1. doses are added: A_depot += dose_to_depot; A_central += dose_to_central;
2. kinetics advance: moved = A_depot * (1 - exp(-ka dt)); A_depot -= moved; A_central += moved;
   A_central *= exp(-ke dt); C_effect += (A_central - C_effect) * (1 - exp(-keo dt));
3. tau_since_dose = 0 if any dose was given this tick, else tau_since_dose + dt;
4. readouts from the NEW state: E = Emax * C/(EC50 + C); dHR = HRmax * C/(EC50_hr + C).
Rate constants from half-times in minutes: k = ln(2) / (half_min * 60).
Parameters per patient variant are in `pk_spec_dev.yaml` (three variants: typical, fast_absorber,
slow_eliminator). Dosing schedules there are for scoring. Rollouts are 60 to 90 minutes.

## Target
A feed-forward ReLU network (class `ReLUNet` in `wsp_layers.py`: `net.add(W, b)` per layer, ReLU on
every layer except the last; `net.forward(x)`; `net.save(path)`), applied recurrently:
  y = net.forward([A_d, A_c, C_e, tau, u_d, u_c]) -> [A_d', A_c', C_e', tau', E, dHR]; next state = y[:4].
Initial state is zeros. One network per variant; build it from that variant's parameters.
Deliverable: `weights_typical.npz`, `weights_fast_absorber.npz`, `weights_slow_eliminator.npz`,
plus `construction_notes.md` (what every hidden unit computes) and `run_log.md` (every Python run
you executed, numbered, with one line on what it was for).

## Rules (strict)
- No training, no gradient descent, no least squares, no parameter search, no tuning of any value
  against the scorer's output. Every weight must follow from the specification by construction.
- You may run Python to check shapes, to compute constants from the formulas, and to run
  `python score.py weights_<variant>.npz <variant>` (returns only aggregate numbers). At most 25
  Python runs in total. Count them in `run_log.md`.
- Pass criterion: max |E_net - E_spec| <= 0.01 on the dev schedules for all three variants, with the
  state exact (max |C_net - C_spec| < 1e-9 would be ideal).
- Constructions you may find useful (all in `wsp_layers.py`): `clip01_rows`, `threshold_rows`,
  `gated_ramp_rows`, `interpolant_rows` (piecewise-linear interpolant of a scalar function),
  `identity_rows`. Hints: the kinetics are linear, so one ReLU layer can carry them exactly when all
  quantities are non-negative; the Emax readouts are nonlinear and need an interpolant; think about
  where the interpolant's knots should be relative to EC50 so the chord error stays under 0.01 for
  every variant; the time-since-dose needs a gate derived from the dose inputs.
Report at the end: number of Python runs, final scorer output for each variant, and any difficulty.
