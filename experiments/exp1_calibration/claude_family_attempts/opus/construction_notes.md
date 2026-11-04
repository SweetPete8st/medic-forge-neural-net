# Construction notes

Builder script: `build.py` (this directory). Input x = [A_d, A_c, C_e, tau, u_d, u_c].
Per variant: a_d = exp(-ka), a_c = exp(-ke), al = 1 - exp(-keo), with k = ln2/(half_min*60), dt = 1.

Three layers: 6 -> 6 (ReLU) -> 4+nE+nH (ReLU) -> 6 (linear).

## Layer 1 (6 ReLU units)
Each tick's new state is linear in (state, dose), and with non-negative state and doses every
coefficient is non-negative, so ReLU is the identity on these units (exact).
- h0 = A_d' = a_d (A_d + u_d)
- h1 = A_c' = a_c [ (1-a_d)(A_d + u_d) + A_c + u_c ]   (dose, then transfer, then elimination)
- h2 = C_e' = (1-al) C_e + al * A_c'                    (A_c' row folded in)
- h3 = tau (passthrough, tau >= 0)
- h4 = ReLU(S (u_d+u_c)),  h5 = ReLU(S (u_d+u_c) - 1), S = 1e6; gate = h4 - h5 = clip01(S * total dose)
  (equals 1 for any dose >= 1e-6, 0 when no dose).

## Layer 2 (ReLU)
- A_d', A_c', C_e' identity copies of h0..h2.
- tau' = ReLU(h3 + 1 - BIG (h4 - h5)), BIG = 1e6 >= tau_max + dt (gated ramp: reset to 0 on dose ticks, else +1).
- E interpolant units ReLU(C_e' - k_j), knots k_j from 0 to 25 (C_e is bounded by total dose, 10.3 on dev).
  Knot spacing h at left knot c: h = sqrt(8 tol (EC50+c)^3 / (2 Emax EC50)), i.e. the chord-error bound
  h^2/8 * max|f''| with |f''| = 2 Emax EC50/(EC50+C)^3 (decreasing, so max at left end) is <= tol = 1e-5.
  Dense near 0 (spacing ~0.005-0.008) where curvature is largest relative to EC50, widening as (EC50+C)^1.5.
- dHR interpolant units ReLU(C_e' - k_j), same rule with HRmax, EC50_hr, tol = 1e-3.
  Unit counts: typical nE=257, nH=123; fast_absorber 264/142; slow_eliminator 251/105.

## Layer 3 (linear)
- outputs 0..3 copy A_d', A_c', C_e', tau'.
- E = f(0) + sum_j w_j ReLU(C_e' - k_j), w_j = slope_j - slope_{j-1} (from `interpolant_rows`); same for dHR.

## Parameter counts
typical 5040, fast_absorber 5378, slow_eliminator 4728.
Scores: max|dE| ~9.9e-6 (bound 1e-5), max|dC| <= 5.5e-14 for all three variants.
