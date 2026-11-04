# Run log (4 Python runs total)

1. `python3 attempts/opus/build.py` - compute rate constants and knots from the formulas, construct and save the three networks, print shapes and parameter counts.
2. `python3 score.py attempts/opus/weights_typical.npz typical` - score: max_dE 9.90e-06, max_dC 3.0e-14, pass.
3. `python3 score.py attempts/opus/weights_fast_absorber.npz fast_absorber` - score: max_dE 9.90e-06, max_dC 3.5e-14, pass.
4. `python3 score.py attempts/opus/weights_slow_eliminator.npz slow_eliminator` - score: max_dE 9.90e-06, max_dC 5.5e-14, pass.

No value was changed after seeing scorer output; the first construction was final.
