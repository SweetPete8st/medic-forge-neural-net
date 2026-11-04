# Weight-Space Programming: ECG rhythm network

Experiment: can a frontier LLM (Claude Fable 5.1) write the weights of a neural
network that reads a raw single-lead ECG and reports rate, regularity, QRS width,
P-wave presence and a rhythm label, with every weight set by reasoning and none
by optimization?

Run by David Gold (MCFRS paramedic instructor) and Claude. Started 2026-11-04.

## What counts as "writing weights"

1. Every parameter is set by a stated rule: a filter design formula, a logic
   gate construction, a physiological constant (refractory period, PR interval,
   QRS duration bands), or an explicit argument written in the design notes.
2. No gradient descent. No loss function. No optimizer of any kind.
3. No parameter search against data. No sweeping a threshold to find the value
   that scores best. If a constant is chosen, the reason is written down before
   the number is tested.
4. Code is allowed as the pen that writes the weights (a sinc formula evaluated
   in numpy is still a hand-designed filter). Code is not allowed as a substitute
   for the network: the final artifact must run as plain Conv1d / ReLU / pooling /
   linear layers with no Python logic inside the forward pass.

## Forward passes

A forward pass is one run of the network over one record or one batch of
windows. Every forward pass is logged in LEDGER.md with its purpose. The count is
the honest measure of how much trial and error went into the result, the same
accounting the MIT Weight-Space Programming post used.

## Design changes

Every change to the network after a forward pass is logged with:
- the version number,
- what changed,
- why, as a reasoning argument,
- whether the change was motivated by a specific failing record. If it was, the
  change is flagged DATA-INFORMED. Those changes are legitimate (the MIT work
  allowed feedback) but they are the moment the experiment starts leaning on
  data, so they are counted.

## Development set vs held-out set

- Development: MIT-BIH record 100 (the only reference-annotated record available
  in Claude's sandbox) plus synthetic strips. Record 100 is a clean record;
  results on it are an upper bound.
- Held-out: the remaining 47 MIT-BIH arrhythmia records and the MIT-BIH atrial
  fibrillation database, run on David's machine with the frozen network. No
  changes to the network after the held-out numbers are seen, unless the ledger
  says so explicitly and the held-out set is then considered burned.

## Success criteria, stated before building

Beat detection: published Pan-Tompkins implementations score about 99.3%
sensitivity and 99.5% PPV on MIT-BIH. A hand-compiled version with fixed
thresholds instead of adaptive ones should land lower. Above 97% on the held-out
set would mean the compile was faithful. Below 90% means the fixed-threshold
design is the limit.

Rate: within 5 bpm of annotation-derived rate on 10-second windows for regular
rhythms.

Regularity: agreement with an RR-variability rule on annotations above 90%.

QRS width: narrow/wide agreement with beat-type annotations (N vs V) above 85%
is the target. The MWI pulse-width method is crude and this may not be reached.

P-wave presence: unknown. This is the expected break point. Any result above
chance on sinus-vs-AFib discrimination from P detection alone is a finding.

Rhythm labels: reported as a confusion matrix against MIT-BIH rhythm
annotations, with no target number. The goal is to see which boundaries the
compiled rules can and cannot draw from raw signal.

## Data

PhysioNet MIT-BIH data only. No MCFRS patient data enters this experiment.
