# exp4_calibrate: does a compiled patient brain buy anything code does not? Differentiable calibration.

Written before running. The claim under test: because the compiled brain is a ReLU network with a
parameter port, instructor corrections can be turned into gradient updates of the port values (and,
optionally, of selected hidden weights) without touching the program. The code engine can be fitted
too, by black-box search over the same three parameters. The test compares the two on the same task.

## Task
A "ground-truth instructor" patient is the code engine with hidden port values
(responsiveness 0.45, progression 0.035, fatigue 0.018). The instructor supplies two corrected
trajectories (observables only: SpO2, HR, RR, PaCO2 every 30 s) for the timely and delayed scripts.
Starting from the spec defaults (0.8, 0.02, 0.012), recover the port values.

## Methods
- BRAIN-GRAD: torch export of the compiled net; port values are leaf tensors; loss = mean squared
  error on the four observables over both trajectories (truncated BPTT over 60-min rollouts, 1 s
  ticks, gradient through the recurrence); Adam, 60 steps max, learning rate 0.05 on log-parameters.
- CODE-SEARCH: Nelder-Mead on the code engine with the same loss (scipy), 60 function evaluations max.
- Both report: recovered parameters, final loss, wall time, number of rollouts.

## Pass / fail
BRAIN-GRAD "buys something" only if it recovers the hidden parameters within 15% relative error with
fewer rollouts or less wall time than CODE-SEARCH, or reaches a lower loss within the same budget.
If CODE-SEARCH does as well, differentiability is not an instructor-visible advantage at this scale.
