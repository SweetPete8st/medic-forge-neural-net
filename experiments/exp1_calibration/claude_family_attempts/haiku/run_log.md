# Run log (haiku attempt)

Total Python runs: 4 (limit 25).

1. `cd <task dir> && python3 attempts/haiku/build_weights.py`: builds all three networks from pk_spec_dev.yaml by construction, prints the shapes and parameter counts, and saves weights_typical.npz, weights_fast_absorber.npz, weights_slow_eliminator.npz.
2. `cd <task dir> && python3 score.py attempts/haiku/weights_typical.npz typical`: scorer check for the typical variant.
3. `cd <task dir> && python3 score.py attempts/haiku/weights_fast_absorber.npz fast_absorber`: scorer check for the fast_absorber variant.
4. `cd <task dir> && python3 score.py attempts/haiku/weights_slow_eliminator.npz slow_eliminator`: scorer check for the slow_eliminator variant.
