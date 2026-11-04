# exp4_calibrate results (2026-11-04)

Task (PROTOCOL.md): a hidden instructor patient = the code engine with port values
responsiveness 0.45, progression 0.035, fatigue 0.018; two corrected observable traces
(SpO2, HR, RR, PaCO2 every 30 s over the timely and delayed scripts, 20-minute horizon after a logged
budget reduction); recover the three values from the spec defaults (0.8, 0.02, 0.012).

| Method | Rollouts | Wall | Final loss | Relative error (responsiveness / progression / fatigue) | Recovered values |
|---|---|---|---|---|---|
| CODE, Nelder-Mead (black box) | 160 | 51 s | 3.53e-04 | 66.1% / 16.3% / 23.1% | 0.748 / 0.0407 / 0.0139 |
| CODE, finite-difference Adam (same schedule as brain) | 244 | 26 s | 1.45e-05 | 5.8% / 5.4% / 11.0% | 0.424 / 0.0331 / 0.0200 |
| BRAIN, autograd Adam (torch export) | 30 | 3636 s | 2.35e-03 | 5.3% / 1.5% / 16.6% | 0.426 / 0.0345 / 0.0210 |

Brain autograd best intermediate step: step 10, loss 2.42e-05. Brain wall time was measured
while two other jobs shared the four CPU cores; uncontended, one Adam step took about 15 s (steps 11-14),
so an uncontended run would be about 4 minutes, still 10x the finite-difference code run.

## Verdict (demonstrated)
- Gradient descent recovers instructor corrections far better than Nelder-Mead at this budget (both
  gradient methods within 1-17% on every parameter; Nelder-Mead 16-66%).
- **Differentiability of the brain is not what delivered that.** Finite differences on the code engine
  match the brain's accuracy with 4x the rollouts but 100x less wall time, because the code engine
  steps in 0.3 ms and the brain in 3-35 ms. With three parameters, finite differences are cheap.
- The brain's advantage would appear only when many parameters are calibrated at once (dozens of spec
  constants, or hidden weights), where finite differences scale linearly in cost and autograd does
  not. That is a plausible extension, untested here, and it is also achievable by writing the code
  engine in an autograd framework without any weight compilation.

Decision per PROTOCOL.md: BRAIN-GRAD did not beat CODE on rollouts or wall time at equal accuracy.
Differentiability is not an instructor-visible advantage at this scale.
