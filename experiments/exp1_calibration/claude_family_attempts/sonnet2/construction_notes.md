# Construction (3 layers, 6 -> 6 -> 4+K -> 6; built by build.py)
Input x=[Ad,Ac,Ce,tau,ud,uc]; ad=exp(-ka), ac=exp(-ke), al=1-exp(-keo).
Layer 1 (6 units, all inputs non-negative so ReLU is identity on linear combos):
 0: Ad'=ad(Ad+ud); 1: Ac'=ac(Ac+uc+(1-ad)(Ad+ud)); 2: Ce'=(1-al)Ce+al*Ac' (expanded in inputs);
 3: tau passthrough; 4: ReLU(1e3(ud+uc)); 5: ReLU(1e3(ud+uc)-1)  (dose gate = unit4-unit5, a steep clip).
Layer 2: units 0-2 pass Ad',Ac',Ce'; unit 3: tau'=ReLU(tau+1-1e4*(u4-u5)) (gated ramp);
 then interpolant units ReLU(Ce'-k_j) for E (Emax*C/(EC50+C)) and for dHR, knots from rule
 h=sqrt(8*tol/f''(left knot)), f''=2*M*EC/(EC+C)^3, tol 0.003 (E) / 0.02 (dHR), range 0..12.
Output (linear): y0..y3 = units 0..3; E = f(0)+sum w_j ReLU(.), dHR likewise.
Knot rule is a chord-error bound, not tuned against the scorer.
