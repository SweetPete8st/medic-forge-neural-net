# ECG rhythm network written in weight space

Stages 1-4 are built and frozen (v4) pending the held-out run. See RULES.md
for the protocol and LEDGER.md for every forward pass and design change.

Network v4: 204,922 parameters, every one set by a stated rule, in standard
layers (Conv1d, ReLU, MaxPool1d, a causal max-pool, global sum/max pooling,
a max over channels, one fixed positional ramp). Loads as a PyTorch nn.Module
from ecg_rhythm_net_v4.pt. Outputs per 10 s single-lead strip: beat train,
beat count, 6-second-method rate, mean-RR rate, regularity score and flag,
mean and max QRS steep-slope width and wide flags, P-wave fraction and flag,
and a 15-way rhythm label vector.

## Layout

    RULES.md                    the protocol: no gradients, no fitting, counted forward passes
    LEDGER.md                   every forward pass and every design change, with reasons
    src/wsp/layers.py           numpy Conv1d / ReLU / MaxPool1d / CausalMaxPool1d / Concat (torch-equivalent)
    src/wsp/filters.py          windowed-sinc bandpass design (a formula, not a fit)
    src/wsp/detector.py         Stage 1 network: every constant and its reason
    src/wsp/torch_export.py     builds the identical nn.Module and saves a state_dict
    src/wsp/heads.py            Stage 2: rate, regularity, QRS width heads (the three constructions explained in the docstring)
    src/wsp/pwave.py            Stage 3: P-wave head
    src/wsp/labels.py           Stage 4: rhythm label head
    src/wsp/synth.py            synthetic strips for implementation sanity checks
    src/eval_beats.py           beat detection vs PhysioNet annotations
    src/eval_stage2.py          rate / regularity / width vs annotation-derived references
    src/eval_stage3.py          window-level outputs, P head and labels vs rhythm annotations
    src/run_heldout.py          THE HELD-OUT RUN (see below)
    download_mitdb.py           fetches the held-out data (MIT-BIH arrhythmia + AF databases)
    beat_detector_v1_1.pt       Stage 1 alone as a PyTorch checkpoint (458 parameters)
    ecg_rhythm_net_v4.pt        the full network as a PyTorch checkpoint (204,922 parameters)
    stage1_record100_stages.png every Stage 1 layer on 10 s of record 100
    stage3_pwave_record100.png  the P-wave head's internals on 10 s of record 100

## Run the held-out evaluation (David's machine)

    pip install wfdb torch numpy scipy
    python download_mitdb.py          # ~110 MB from PhysioNet
    python src/run_heldout.py         # 47 MIT-BIH records + 23 AF-database records (every 20th window)

Expect roughly an hour. Results land in results/: heldout_summary.txt is the
one to send back, plus the three CSVs. Record 100 is the development record;
everything else is held out. Nothing in the network changes during the run;
every record is logged in LEDGER.md as a forward pass.

What the summary shows: beat detection Se/PPV over all held-out beats and the
worst records; steep-slope width by beat type (N vs V is the QRS-width test);
and a table of annotated rhythm (rows) against the network's label (columns)
with an agreement percentage per rhythm.

## Stage 1 results so far

| Record | Source | Reference | Beats | Sensitivity | PPV | Notes |
|--------|--------|-----------|-------|-------------|-----|-------|
| 100 (MLII) | MIT-BIH, 360 Hz | human (.atr) | 2273 | 100.00% | 100.00% | clean dev record, offset +0.3 +/- 1.5 ms |
| 12726 ch1 | MIMIC-style ICU, 250 Hz resampled | wqrs detector output | 3649 | 99.21% | 98.88% | 6 of 29 misses are a 15x amplitude dropout; the rest are disagreements with another detector, not necessarily our errors |
| 03700181 MCL1 | 125 Hz resampled | gqrs output | n/a | n/a | n/a | reference file on a different time base; qualitatively every QRS found |

## Stages 2-4 results on the development record (100, all sinus rhythm)

- Beat count exact in 99.4% of 10 s windows (max error 1 beat, at window edges).
- Mean-RR rate: median error 0.06 bpm, 95th percentile 0.41 bpm.
- Regularity: 100% agreement with the annotation RR rule (177 regular, 3 irregular windows; the 3 contain APCs).
- Steep-slope width: N beats median 61 ms (p10-p90 56-64), the record's one PVC 75 ms. The 85 ms wide threshold is likely too high; held-out N-vs-V distributions will decide, and any change will be flagged in the ledger.
- P waves: present in 97% of windows (median fraction 1.00). On an unannotated ICU record with small P waves the head reads present in only 22% of windows: the 0.05 mV absolute floor is the predicted weak point.
- Labels: 171/180 windows "sinus rhythm", 6 "junctional" (P head missed the P), 3 "irregular with P" (APC windows).
- Synthetic strips of 11 rhythms all labelled correctly; a flat line read as VT until an absolute QRS floor was added (v1.2).

Forward passes used through Stage 4: 28 (see ledger). Design changes after
seeing data: 1 on real data (double-fire suppression, logic only), 3 on
synthetic strips (width gate constant, P head redesign, QRS floor).

## Not attempted (would need new constructions)

AV blocks (P:QRS ratio), atrial flutter (F-wave comb), paced rhythm,
ventricular fibrillation, and P assessment above ~86 bpm.
