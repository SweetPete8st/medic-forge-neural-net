# Ledger

## Forward passes

| # | Date | Network version | Data | Purpose |
|---|------|-----------------|------|---------|
| 1 | 2026-11-04 | v1 | synthetic x6 | implementation sanity check on synthetic strips |
| 2 | 2026-11-04 | v1 | 100 ch0 (MLII) | first run on real data: Se 100.00% PPV 100.00%, 2273/2273, offset +0.3 +/- 1.5 ms (save step crashed after scoring) |
| 3 | 2026-11-04 | v1 | 100 ch0 | repeat of #2 to save intermediate tensors for the stage figure |
| 4 | 2026-11-04 | v1 | 12726 ch1 | ICU ECG at 250 Hz (resampled), weak reference = wqrs detector output |
| 5 | 2026-11-04 | v1 | 03700181 ch0 | MCL1 at 125 Hz (resampled), weak reference = gqrs detector output |
| 6 | 2026-11-04 | v1 | 12726 ch1 segment 995-1015 s | diagnose the 6-beat miss cluster around 1005-1010 s |
| 7 | 2026-11-04 | v1.1 | 100 ch0 | regression check after dedup gate |
| 8 | 2026-11-04 | v1.1 | 12726 ch1 | regression check after dedup gate |
| 9 | 2026-11-04 | v1.1 | 100 ch0 | regression check after fixing dedup edge padding (bug fix, not design) |
| 10 | 2026-11-04 | v1.1 | 12726 ch1 | same |
| 11 | 2026-11-04 | v1.1 | 100 ch0 | regression check after dedup edge-padding bug fix (passes 9-10 ran with the bug: my edit failed) |
| 12 | 2026-11-04 | v1.1 | 12726 ch1 | same |
| 13 | 2026-11-04 | v1.1 | 100 ch0, first 60 s | torch float32 equivalence check |
| 14 | 2026-11-04 | v2 | synthetic x7 | stage 2 heads implementation sanity check |
| 15 | 2026-11-04 | v2 | synthetic x3 | recheck width head after gate-constant fix |
| 16 | 2026-11-04 | v2 | 100 ch0, 180 windows + 1 full run | stage 2 first run on real data |
| 17 | 2026-11-04 | v2 | 100 ch0, 5 windows | torch float32 equivalence of stage 2 |
| 18 | 2026-11-04 | v3 | synthetic x8 | stage 3 P-wave head implementation check |
| 19 | 2026-11-04 | v3.1 | synthetic x10 | stage 3.1 hump-count P head check |
| 20 | 2026-11-04 | v3.1 | 100 ch0, 180 windows | stage 3 first run on real data |
| 21 | 2026-11-04 | v3.1 | 12726 ch1, 330 windows | stage 3 on the ICU record (rhythm unknown, no annotations) |
| 22 | 2026-11-04 | v3.1 | 100 ch0, 3 windows | torch equivalence of stage 3 |
| 23 | 2026-11-04 | v3.1 | 12726 ch1 window 40 | look at why P reads absent on the ICU record |
| 24 | 2026-11-04 | v4 | synthetic x12 | stage 4 label head implementation check |
| 25 | 2026-11-04 | v1.2 | 100 ch0 | regression check after absolute floor |
| 26 | 2026-11-04 | v4/v1.2 | synthetic x3 | flat-line and tiny-QRS check after floor |
| 27 | 2026-11-04 | v4 | 100 ch0, 180 windows | stage 4 labels on record 100 |
| 28 | 2026-11-04 | v4 | 100 ch0, window 20 (+1 for torch check) | stage 4 torch equivalence and P-head figure |

## Design changes

| Version | Date | Change | Reasoning | Data-informed? |
|---------|------|--------|-----------|----------------|
| v1 | 2026-11-04 | Initial beat detector design | See the docstring of src/wsp/detector.py: every constant with its reason | No |
| v1.1 | 2026-11-04 | Double-fire suppression: causal max-pool over the 50 ms before each sample, final beat = ReLU(beat0 - prev) | Local-max gate can fire on two samples that tie within 1e-6 at a rounded hump top; one such double seen on 03700181 (resampled 125 Hz, smoother humps). Pure logic, no threshold; no physiological beat follows another within 50 ms. | DATA-INFORMED (one double on 03700181 triggered it; mechanism anticipated) |

Notes:
- Forward pass #5 (03700181, gqrsh reference) is not a valid score: the annotation samples appear to be on a different time base (reference spacing 1.95 s where the signal shows QRS every 0.49 s). The detector found every visible QRS in the stretches inspected. Record kept for qualitative checks only.
- Forward pass #6 diagnosed the 12726 miss cluster at 1004-1010 s: QRS amplitude dropped about 15x for 6 s (signal dropout). The floor term 0.05*F, with F remembering the previous large beats for 5 s, held the threshold above the small beats. Known limitation of fixed-fraction thresholding; NOT changed.
| v2 | 2026-11-04 | Stage 2 heads added: count rate, mean-RR rate (one-hot gated division + ReLU reciprocal), comb-bank regularity, wide-band steep-slope QRS width | See docstring of src/wsp/heads.py | No |
| v2 (bug) | 2026-11-04 | Width-mean gate constant raised from 1e3 to 1e5 | width_sum (up to 60 beats x 240 ms = 14,400) exceeded the gate constant, so every one-hot slot leaked. Found on synthetic strips (forward pass #14). Implementation error in a construction whose logic was already stated, not a design change. | No (synthetic) |
| v3 | 2026-11-04 | Stage 3 P-wave head: matched filter, PR-window peak, baseline-contrast test, assessability gate at RR >= 600 ms | See earlier docstring (replaced) | No |
| v3.1 | 2026-11-04 | Contrast-against-baseline test replaced by "exactly one hump in diastole [R-400, R-131] ms and a hump in the PR window"; assessability raised to RR >= 700 ms | On synthetic strips at 95 bpm (forward pass #18) the previous T wave fell in the baseline window and every P was rejected. Hump counting needs no baseline estimate; the T-peak collision now sets the assessability limit instead. | Synthetic-informed (no real data involved) |
- Forward passes #21 and #23 (12726, ICU record, no annotations): the P head reads "present" in only 22% of windows at 67 bpm regular. The strip at window 40 shows real but small P waves whose matched-filter response sits at the 0.05 mV floor, so beats fall on either side of the threshold. This is the predicted weak point: an absolute amplitude floor chosen by reasoning, with no data behind it. NOT changed before the held-out run; a relative (P-to-R) threshold is the candidate v3.2 and will be flagged DATA-INFORMED if adopted.
| v1.2 | 2026-11-04 | Absolute floor added to the detector threshold: c = m - 0.3L - 0.05F - 0.15 (0.15 MWI units ~ 0.05 mV QRS) | A pure-noise strip (synthetic, forward pass #24) produced 18 "beats" and the label head called it VT; a relative-only threshold has no notion of "too small to be a QRS". 0.05 mV is below any QRS visible at standard gain. | Synthetic-informed |
| v4 | 2026-11-04 | Stage 4 label head: ten soft threshold indicators, four derived gates, fifteen label units (one ReLU per label), see src/wsp/labels.py | The standard rate / regular? / wide? / P? algorithm compiled as two ReLU layers. | No |

Status at hand-off for the held-out run: network frozen at v4 (detector v1.2, heads v2, P head v3.1). 28 forward passes used in development, all on record 100, two unannotated sample records, and synthetic strips. Held-out records untouched.
