# Python run log (attempt: fable)

Total Python runs: 4 (limit 25).

1. `cd <task dir> && python3 attempts/fable/build_pk_net.py` -- build the three weight files from the spec, print
   layer shapes / parameter count (906), the per-tick constants, the analytic chord-error check of the knot list,
   and a rollout of each network against a transcription of the spec on a synthetic schedule (state err ~1e-14,
   tau exact, E err 0.00217). Wrote weights_typical.npz, weights_fast_absorber.npz, weights_slow_eliminator.npz.
2. `python3 score.py attempts/fable/weights_typical.npz typical` ->
   {'variant': 'typical', 'max_dE_dev': 0.0021651904165988917, 'max_dC_dev': 3.019806626980426e-14, 'pass': True}
3. `python3 score.py attempts/fable/weights_fast_absorber.npz fast_absorber` ->
   {'variant': 'fast_absorber', 'max_dE_dev': 0.0021656616621457056, 'max_dC_dev': 3.4638958368304884e-14, 'pass': True}
4. `python3 score.py attempts/fable/weights_slow_eliminator.npz slow_eliminator` ->
   {'variant': 'slow_eliminator', 'max_dE_dev': 0.002165725624332669, 'max_dC_dev': 5.5067062021407764e-14, 'pass': True}

No weights were changed after any scorer run.
