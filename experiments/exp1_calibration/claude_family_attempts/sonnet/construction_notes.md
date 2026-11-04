# Construction notes (build.py generates all three nets; 3 weight layers, widths 6 / 4+K / 6)
Constants: ea=exp(-ka), ek=exp(-ke), eo=exp(-keo), k=ln2/(half_min*60).
Layer 1 (6 ReLU units, input [Ad,Ac,Ce,tau,ud,uc]); all linear combos have nonnegative coefficients so ReLU is identity:
- h0 = ea*(Ad+ud)  (new depot)
- h1 = ek*[(Ac+uc) + (1-ea)(Ad+ud)]  (new central)
- h2 = eo*Ce + (1-eo)*h1  (new effect-site C)
- h3 = tau (pass-through)
- h4 = ReLU(1000*(ud+uc)); h5 = ReLU(1000*(ud+uc)-1); gate = h4-h5 = clip01(1000*dose) (doses >= 1e-3 give exactly 1, zero gives 0).
Layer 2 (4+K units):
- units 0-2 pass h0,h1,h2 (identity, nonnegative).
- unit 3 = ReLU(tau + 1 - 1e4*(h4-h5)): time since dose (reset to 0 on dose, else +1; 1e4 > max tau+1).
- units 4..4+K-1: ReLU(C - knot_j), j=0..K-1, interpolant units of the new C.
Output layer (linear): y0..y2 = units 0..2, y3 = unit 3; E = f_E(0) + sum w_j unit_j and dHR likewise, with w_j slope changes of the chord interpolants of Emax*C/(EC50+C) and HRmax*C/(EC50_hr+C) (interpolant_rows).
Knots: greedy from 0 to 6 with step h = sqrt(8*tol/|f''(x)|), f''=2*EC50/(EC50+x)^3, tol=0.002 (normalised, chord error h^2/8 f''), for both EC50 and EC50_hr, union of the two sets. Derived from formulas only. K=30/33/28.
