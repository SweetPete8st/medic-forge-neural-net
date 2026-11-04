# Run log (haiku2)

Total Python runs: 4 of 25.

1. `python3 attempts/haiku2/build_haiku2.py` (run from the task dir). Constructs the three weight sets from the
   spec and the parameters in pk_spec_dev.yaml, saves weights_typical.npz, weights_fast_absorber.npz and
   weights_slow_eliminator.npz, and runs a self-check against my own reference simulator of the spec
   (not the scorer) over the dev schedules. Self-check max errors: state 6e-14, E about 9.1e-4, dHR 2e-3.
2. `python3 score.py attempts/haiku2/weights_typical.npz typical`. Scorer run for typical.
3. `python3 score.py attempts/haiku2/weights_fast_absorber.npz fast_absorber`. Scorer run for fast_absorber.
4. `python3 score.py attempts/haiku2/weights_slow_eliminator.npz slow_eliminator`. Scorer run for slow_eliminator.

Final scorer output (all pass):
- typical: max_dE_dev 9.106e-4, max_dC_dev 3.0e-14, pass True
- fast_absorber: max_dE_dev 9.102e-4, max_dC_dev 3.5e-14, pass True
- slow_eliminator: max_dE_dev 9.119e-4, max_dC_dev 5.5e-14, pass True
