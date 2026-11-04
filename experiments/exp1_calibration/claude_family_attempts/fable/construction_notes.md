# Construction notes: PK/PD cell as a 3-layer ReLU network (no training)

Built by `build_pk_net.py` (in this directory) from `pk_spec_dev.yaml`. One network per variant,
identical architecture; only the constants differ. Architecture: 6 -> 6 (ReLU) -> 66 (ReLU) -> 6 (linear),
906 parameters per network (42 + 462 + 402).

Input x = [A_d, A_c, C_e, tau, u_d, u_c]; output y = [A_d', A_c', C_e', tau', E, dHR].
Per-tick constants (dt = 1 s, k = ln2/(half_min*60)):
  a_d = exp(-ka dt)  depot fraction retained;  a_c = exp(-ke dt)  central fraction retained;
  al  = 1 - exp(-keo dt)  effect-site mixing fraction.
| variant | a_d | a_c | al |
|---|---|---|---|
| typical | 0.995389679103 | 0.999922986612 | 0.011485979647 |
| fast_absorber | 0.988514020353 | 0.999903734192 | 0.022840031566 |
| slow_eliminator | 0.997116053335 | 0.999961492565 | 0.005759576182 |

## Why the kinetics fit in one ReLU layer
Unrolling the spec order (dose, absorb, eliminate, mix) gives linear maps with non-negative coefficients:
  A_d' = a_d (A_d + u_d)
  A_c' = a_c (A_c + u_c) + a_c (1 - a_d)(A_d + u_d)
  C_e' = (1 - al) C_e + al A_c'
All inputs are non-negative and all coefficients are non-negative, so each pre-activation is >= 0 and
ReLU is the identity on it: the state update is exact (floating-point only, ~1e-14 observed).

## Hidden layer 1 (6 units, ReLU)
| unit | computes | weights |
|---|---|---|
| h1[0] | A_d' | a_d on A_d and on u_d |
| h1[1] | A_c' | a_c(1-a_d) on A_d and u_d; a_c on A_c and u_c |
| h1[2] | C_e' | al·a_c(1-a_d) on A_d and u_d; al·a_c on A_c and u_c; (1-al) on C_e |
| h1[3] | tau (pass-through) | 1 on tau |
| h1[4] | ReLU(S·(u_d + u_c)), S = 1e6 | first half of a steep clip (gate) |
| h1[5] | ReLU(S·(u_d + u_c) - 1) | second half; gate = h1[4] - h1[5] = clip01(S·(u_d+u_c)) |
The gate is exactly 0 on ticks with no dose and exactly 1 for any total dose >= 1e-6 (the two halves differ
by exactly 1 in floating point, Sterbenz). Spec says tau resets if "any dose was given"; the smallest dose in
the dev schedules is 0.3, so the gate is saturated on every dosed tick.

## Hidden layer 2 (66 units, ReLU)
| unit | computes |
|---|---|
| h2[0..2] | A_d', A_c', C_e' passed through (identity rows; all >= 0) |
| h2[3] | tau' = ReLU(tau + dt - BIG·(h1[4] - h1[5])), BIG = 2e4 >= tau_max + dt (90 min = 5400 s). gate=1 -> 0; gate=0 -> tau+dt |
| h2[4+j], j=0..30 | ReLU(C_e' - kE_j), kE_j = EC50 · z_j (interpolant knots for E) |
| h2[35+j], j=0..30 | ReLU(C_e' - kH_j), kH_j = EC50_hr · z_j (interpolant knots for dHR) |
Knot list in units of the half-saturation constant (z = C/EC50), 32 knots -> 31 units per readout:
  z = 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 25, 30, 40, 50
Knot placement reasoning: f(z) = Emax z/(1+z) has |f''| = 2Emax/(1+z)^3, decreasing in z, so the chord error on
[z_l, z_l+h] is <= Emax h^2 / (4 (1+z_l)^3). Every interval above satisfies h <= 0.1 (1+z_l)^1.5, so the bound is
<= 0.0025·Emax on every interval; the first interval [0, 0.1] is the binding one (actual error 0.00217 with
Emax = 1, the observed max dE on all variants). The last knot z = 50 (C = 40..100 depending on EC50) is far
beyond the largest reachable C (the total amount dosed in any schedule is <= 10.3 and C_e <= A_c); beyond the
last knot the readout extrapolates with the last chord slope, which stays under 0.01 error out to z ~ 100.
Using the same z-list scaled by each readout's own EC50 makes the bound variant-independent.

## Output layer (6 units, linear)
| output | computes |
|---|---|
| y[0..3] | h2[0..3] (A_d', A_c', C_e', tau') |
| y[4] = E | fE(0) + sum_j wE_j · h2[4+j], wE_j = chord slope on [kE_j, kE_{j+1}] minus the previous chord slope (fE(0) = 0) |
| y[5] = dHR | fH(0) + sum_j wH_j · h2[35+j], same construction with HRmax, EC50_hr (fH(0) = 0) |
This is `interpolant_rows` from wsp_layers.py: exact at the knots, chord error between them.

## Checks performed (not tuning)
`build_pk_net.py` prints, per variant: the analytic chord-error maximum per interval on a fine grid
(E: 0.00217; dHR: HRmax x that), and a rollout on a synthetic mixed-route schedule against a transcription of
the spec text: state error ~1e-14, tau exact, E error 0.00217. Scorer results are in run_log.md.
Nothing was adjusted after seeing scorer output; the scorer was run once per variant.

## Known limits
- tau reset relies on a positive total dose amount (a zero-amount "dose" would not reset tau).
- dHR absolute error is HRmax x 0.00217 (~0.04-0.065 bpm); the same knot list could be refined if a tighter
  heart-rate tolerance were specified.
