# LEDGER.md

Running ledger for every experiment in this repository. Governed by `RULES.md`.
Each experiment has its own section. Forward-pass totals are cumulative within an experiment.

Format for forward passes: `FP <n_cumulative> | <date> | <experiment> | <what was evaluated> | <dev|heldout>`
Format for design changes: `DC <k> | <date> | <experiment> | <change> | reason: <...> | DATA-INFORMED: yes|no`
Format for builder runs: `BR <k> | <date> | <builder model> | <task> | sandbox runs: <n> | wall: <t> | attempts: <n> | outcome`

Entries are appended by `src/mfnn/ledger.py` (machine-appended lines are prefixed `[auto]`) or by hand.

---

## exp0_pk: one-compartment drug kinetics with dosing (code control vs hand-set weights)

Status: see `experiments/exp0_pk/PROTOCOL.md` for predefined criteria. Entries follow.


## exp2_asthma: specification drafting and code-control engine (development only, no held-out data yet)

DC 1 | 2026-11-04 | exp2_asthma | progression_rate 0.012 -> 0.03 /min | reason: the spec's own description claimed 0.012 reaches O=0.80 by 30 min; arithmetic was wrong (0.68 at 20 min). Rate set so the logistic matches the stated expectation. | DATA-INFORMED: no (shaped to the spec's written expectation, not to patient data)
DC 2 | 2026-11-04 | exp2_asthma | RR model: cap CO2 drive (rr_co2_cap), blunt above loc_co2_1, fatigue loss F^2 (rr_fatigue_loss), x0.7 when LOC <= P | reason: untreated dev rollout showed RR 44/min at F=0.9 and PaCO2 78, contradicting the clinical expectation of late bradypnoea. | DATA-INFORMED: no (motivated by a dev rollout of the model itself, not by patient data)
DC 3 | 2026-11-04 | exp2_asthma | progression_rate 0.03 -> 0.02 /min; fatigue_rate 0.03 -> 0.012 /min; untreated PaCO2 envelope at 30 min widened to [44,115] | reason: 12-rollout dev sweep of the untreated trajectory (scratchpad/tune/sweep.py) over progression {0.02,0.03} x fatigue {0.012,0.02,0.03} x V_alv_max {60,75}; chosen pair gives life-threatening features at ~20 min, silent chest ~28 min, arrest ~30 min, a teaching timeline. V_alv_max had no effect in this regime (tidal-volume collapse is the binding constraint). | DATA-INFORMED: no (no patient data; shaped to a teaching timeline, needs reviewer sign-off)
FP 12 | 2026-11-04 | exp2_asthma | 12 untreated dev rollouts in the parameter sweep | dev
[auto] FP 291 | 2026-11-04 | exp2_asthma | +291: four trajectories + interaction sweep (code control) | dev
DC 4 | 2026-11-04 | exp2_asthma | autoPEEP saturates at autopeep_max (25 cmH2O); SBP floored at 0; barotrauma_risk event at autoPEEP >= 20 | reason: interaction sweep found 31 cases (all bvm at 20/min) with negative SBP because auto-PEEP accumulated without bound. | DATA-INFORMED: no (model self-consistency; no patient data)
DC 5 | 2026-11-04 | exp2_asthma | inappropriate_intervention script: BVM moved from 28 min to 17 min, albuterol from 25 to 20 min; envelopes: timely PaCO2 at 45 min [30,44], inappropriate LOC at 15 min [1,2.6], autoPEEP/SBP checks at 25 min | reason: dev rollout showed arrest at 25 min before BVM was applied, so the breath-stacking teaching point never occurred; two envelopes were author guesses that the model missed by small margins (PaCO2 32.7 vs 33; LOC 2.43 vs 2). | DATA-INFORMED: no (envelopes are author expectations pending clinical review)
[auto] FP 582 | 2026-11-04 | exp2_asthma | +291: four trajectories + interaction sweep (code control) | dev
[auto] FP 2 | 2026-11-04 | exp0_pk | +2: wsp dev fast_absorber/dose_then_wait (+replay) | dev
[auto] FP 4 | 2026-11-04 | exp0_pk | +2: wsp dev fast_absorber/inhaled_then_iv (+replay) | dev
[auto] FP 6 | 2026-11-04 | exp0_pk | +2: wsp dev fast_absorber/irregular_four (+replay) | dev
[auto] FP 8 | 2026-11-04 | exp0_pk | +2: wsp dev fast_absorber/iv_bolus (+replay) | dev
[auto] FP 10 | 2026-11-04 | exp0_pk | +2: wsp dev fast_absorber/single_2p5 (+replay) | dev
[auto] FP 12 | 2026-11-04 | exp0_pk | +2: wsp dev fast_absorber/three_q20 (+replay) | dev
[auto] FP 14 | 2026-11-04 | exp0_pk | +2: wsp dev fast_absorber/two_q20 (+replay) | dev
[auto] FP 16 | 2026-11-04 | exp0_pk | +2: wsp dev fast_absorber/two_q5 (+replay) | dev
[auto] FP 18 | 2026-11-04 | exp0_pk | +2: wsp dev slow_eliminator/dose_then_wait (+replay) | dev
[auto] FP 20 | 2026-11-04 | exp0_pk | +2: wsp dev slow_eliminator/inhaled_then_iv (+replay) | dev
[auto] FP 22 | 2026-11-04 | exp0_pk | +2: wsp dev slow_eliminator/irregular_four (+replay) | dev
[auto] FP 24 | 2026-11-04 | exp0_pk | +2: wsp dev slow_eliminator/iv_bolus (+replay) | dev
[auto] FP 26 | 2026-11-04 | exp0_pk | +2: wsp dev slow_eliminator/single_2p5 (+replay) | dev
[auto] FP 28 | 2026-11-04 | exp0_pk | +2: wsp dev slow_eliminator/three_q20 (+replay) | dev
[auto] FP 30 | 2026-11-04 | exp0_pk | +2: wsp dev slow_eliminator/two_q20 (+replay) | dev
[auto] FP 32 | 2026-11-04 | exp0_pk | +2: wsp dev slow_eliminator/two_q5 (+replay) | dev
[auto] FP 34 | 2026-11-04 | exp0_pk | +2: wsp dev typical/dose_then_wait (+replay) | dev
[auto] FP 36 | 2026-11-04 | exp0_pk | +2: wsp dev typical/inhaled_then_iv (+replay) | dev
[auto] FP 38 | 2026-11-04 | exp0_pk | +2: wsp dev typical/irregular_four (+replay) | dev
[auto] FP 40 | 2026-11-04 | exp0_pk | +2: wsp dev typical/iv_bolus (+replay) | dev
[auto] FP 42 | 2026-11-04 | exp0_pk | +2: wsp dev typical/single_2p5 (+replay) | dev
[auto] FP 44 | 2026-11-04 | exp0_pk | +2: wsp dev typical/three_q20 (+replay) | dev
[auto] FP 46 | 2026-11-04 | exp0_pk | +2: wsp dev typical/two_q20 (+replay) | dev
[auto] FP 48 | 2026-11-04 | exp0_pk | +2: wsp dev typical/two_q5 (+replay) | dev
[auto] FP 50 | 2026-11-04 | exp0_pk | +2: trained dev fast_absorber/dose_then_wait (+replay) | dev
[auto] FP 52 | 2026-11-04 | exp0_pk | +2: trained dev fast_absorber/inhaled_then_iv (+replay) | dev
[auto] FP 54 | 2026-11-04 | exp0_pk | +2: trained dev fast_absorber/irregular_four (+replay) | dev
[auto] FP 56 | 2026-11-04 | exp0_pk | +2: trained dev fast_absorber/iv_bolus (+replay) | dev
[auto] FP 58 | 2026-11-04 | exp0_pk | +2: trained dev fast_absorber/single_2p5 (+replay) | dev
[auto] FP 60 | 2026-11-04 | exp0_pk | +2: trained dev fast_absorber/three_q20 (+replay) | dev
[auto] FP 62 | 2026-11-04 | exp0_pk | +2: trained dev fast_absorber/two_q20 (+replay) | dev
[auto] FP 64 | 2026-11-04 | exp0_pk | +2: trained dev fast_absorber/two_q5 (+replay) | dev
[auto] FP 66 | 2026-11-04 | exp0_pk | +2: trained dev slow_eliminator/dose_then_wait (+replay) | dev
[auto] FP 68 | 2026-11-04 | exp0_pk | +2: trained dev slow_eliminator/inhaled_then_iv (+replay) | dev
[auto] FP 70 | 2026-11-04 | exp0_pk | +2: trained dev slow_eliminator/irregular_four (+replay) | dev
[auto] FP 72 | 2026-11-04 | exp0_pk | +2: trained dev slow_eliminator/iv_bolus (+replay) | dev
[auto] FP 74 | 2026-11-04 | exp0_pk | +2: trained dev slow_eliminator/single_2p5 (+replay) | dev
[auto] FP 76 | 2026-11-04 | exp0_pk | +2: trained dev slow_eliminator/three_q20 (+replay) | dev
[auto] FP 78 | 2026-11-04 | exp0_pk | +2: trained dev slow_eliminator/two_q20 (+replay) | dev
[auto] FP 80 | 2026-11-04 | exp0_pk | +2: trained dev slow_eliminator/two_q5 (+replay) | dev
[auto] FP 82 | 2026-11-04 | exp0_pk | +2: trained dev typical/dose_then_wait (+replay) | dev
[auto] FP 84 | 2026-11-04 | exp0_pk | +2: trained dev typical/inhaled_then_iv (+replay) | dev
[auto] FP 86 | 2026-11-04 | exp0_pk | +2: trained dev typical/irregular_four (+replay) | dev
[auto] FP 88 | 2026-11-04 | exp0_pk | +2: trained dev typical/iv_bolus (+replay) | dev
[auto] FP 90 | 2026-11-04 | exp0_pk | +2: trained dev typical/single_2p5 (+replay) | dev
[auto] FP 92 | 2026-11-04 | exp0_pk | +2: trained dev typical/three_q20 (+replay) | dev
[auto] FP 94 | 2026-11-04 | exp0_pk | +2: trained dev typical/two_q20 (+replay) | dev
[auto] FP 96 | 2026-11-04 | exp0_pk | +2: trained dev typical/two_q5 (+replay) | dev
[auto] FREEZE | 2026-11-04 | exp0_pk | weights sha256:e3138aef1cb52bcd | wsp compile_pk frozen on dev variants; trained weights saved to results/trained_weights.npz
[auto] FP 98 | 2026-11-04 | exp0_pk | +2: wsp heldout extreme_fast/alternating_q15 (+replay) | heldout
[auto] FP 100 | 2026-11-04 | exp0_pk | +2: wsp heldout extreme_fast/both_routes_same_tick (+replay) | heldout
[auto] FP 102 | 2026-11-04 | exp0_pk | +2: wsp heldout extreme_fast/long_rollout (+replay) | heldout
[auto] FP 104 | 2026-11-04 | exp0_pk | +2: wsp heldout extreme_fast/nothing (+replay) | heldout
[auto] FP 106 | 2026-11-04 | exp0_pk | +2: wsp heldout extreme_fast/overdose_ten_q6 (+replay) | heldout
[auto] FP 108 | 2026-11-04 | exp0_pk | +2: wsp heldout extreme_fast/three_fast_then_nothing (+replay) | heldout
[auto] FP 110 | 2026-11-04 | exp0_pk | +2: wsp heldout extreme_slow/alternating_q15 (+replay) | heldout
[auto] FP 112 | 2026-11-04 | exp0_pk | +2: wsp heldout extreme_slow/both_routes_same_tick (+replay) | heldout
[auto] FP 114 | 2026-11-04 | exp0_pk | +2: wsp heldout extreme_slow/long_rollout (+replay) | heldout
[auto] FP 116 | 2026-11-04 | exp0_pk | +2: wsp heldout extreme_slow/nothing (+replay) | heldout
[auto] FP 118 | 2026-11-04 | exp0_pk | +2: wsp heldout extreme_slow/overdose_ten_q6 (+replay) | heldout
[auto] FP 120 | 2026-11-04 | exp0_pk | +2: wsp heldout extreme_slow/three_fast_then_nothing (+replay) | heldout
[auto] FP 122 | 2026-11-04 | exp0_pk | +2: wsp heldout slow_link/alternating_q15 (+replay) | heldout
[auto] FP 124 | 2026-11-04 | exp0_pk | +2: wsp heldout slow_link/both_routes_same_tick (+replay) | heldout
[auto] FP 126 | 2026-11-04 | exp0_pk | +2: wsp heldout slow_link/long_rollout (+replay) | heldout
[auto] FP 128 | 2026-11-04 | exp0_pk | +2: wsp heldout slow_link/nothing (+replay) | heldout
[auto] FP 130 | 2026-11-04 | exp0_pk | +2: wsp heldout slow_link/overdose_ten_q6 (+replay) | heldout
[auto] FP 132 | 2026-11-04 | exp0_pk | +2: wsp heldout slow_link/three_fast_then_nothing (+replay) | heldout
[auto] FP 134 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout extreme_fast/alternating_q15 (+replay) | heldout
[auto] FP 136 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout extreme_fast/both_routes_same_tick (+replay) | heldout
[auto] FP 138 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout extreme_fast/long_rollout (+replay) | heldout
[auto] FP 140 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout extreme_fast/nothing (+replay) | heldout
[auto] FP 142 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout extreme_fast/overdose_ten_q6 (+replay) | heldout
[auto] FP 144 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout extreme_fast/three_fast_then_nothing (+replay) | heldout
[auto] FP 146 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout extreme_slow/alternating_q15 (+replay) | heldout
[auto] FP 148 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout extreme_slow/both_routes_same_tick (+replay) | heldout
[auto] FP 150 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout extreme_slow/long_rollout (+replay) | heldout
[auto] FP 152 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout extreme_slow/nothing (+replay) | heldout
[auto] FP 154 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout extreme_slow/overdose_ten_q6 (+replay) | heldout
[auto] FP 156 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout extreme_slow/three_fast_then_nothing (+replay) | heldout
[auto] FP 158 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout slow_link/alternating_q15 (+replay) | heldout
[auto] FP 160 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout slow_link/both_routes_same_tick (+replay) | heldout
[auto] FP 162 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout slow_link/long_rollout (+replay) | heldout
[auto] FP 164 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout slow_link/nothing (+replay) | heldout
[auto] FP 166 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout slow_link/overdose_ten_q6 (+replay) | heldout
[auto] FP 168 | 2026-11-04 | exp0_pk | +2: trained_typical_model_on_unseen_variants heldout slow_link/three_fast_then_nothing (+replay) | heldout
[auto] FP 170 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout extreme_fast/alternating_q15 (+replay) | heldout
[auto] FP 172 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout extreme_fast/both_routes_same_tick (+replay) | heldout
[auto] FP 174 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout extreme_fast/long_rollout (+replay) | heldout
[auto] FP 176 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout extreme_fast/nothing (+replay) | heldout
[auto] FP 178 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout extreme_fast/overdose_ten_q6 (+replay) | heldout
[auto] FP 180 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout extreme_fast/three_fast_then_nothing (+replay) | heldout
[auto] FP 182 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout extreme_slow/alternating_q15 (+replay) | heldout
[auto] FP 184 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout extreme_slow/both_routes_same_tick (+replay) | heldout
[auto] FP 186 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout extreme_slow/long_rollout (+replay) | heldout
[auto] FP 188 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout extreme_slow/nothing (+replay) | heldout
[auto] FP 190 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout extreme_slow/overdose_ten_q6 (+replay) | heldout
[auto] FP 192 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout extreme_slow/three_fast_then_nothing (+replay) | heldout
[auto] FP 194 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout slow_link/alternating_q15 (+replay) | heldout
[auto] FP 196 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout slow_link/both_routes_same_tick (+replay) | heldout
[auto] FP 198 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout slow_link/long_rollout (+replay) | heldout
[auto] FP 200 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout slow_link/nothing (+replay) | heldout
[auto] FP 202 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout slow_link/overdose_ten_q6 (+replay) | heldout
[auto] FP 204 | 2026-11-04 | exp0_pk | +2: trained_fresh_variant_models_on_unseen_schedules heldout slow_link/three_fast_then_nothing (+replay) | heldout

### exp0_pk v1 outcome (frozen e3138aef1cb52bcd): A2 FAIL on held-out (max dE 0.0120 > 0.01, variant extreme_fast); D FAIL (diverged). See experiments/exp0_pk/RESULTS_v1.md. The v1 held-out block is BURNED.

## exp0_pk_v2: same specification, revised compile and training configuration, NEW held-out block heldout_v2

DC 1 | 2026-11-04 | exp0_pk_v2 | evaluation harness: washout causal check starts after 6 absorption + 6 link half-times instead of a fixed 15 min | reason: the v1 check flagged the CODE reference itself on slow variants (7 dev, 6 held-out cases); metric defect, not a model defect. | DATA-INFORMED: yes (motivated by observed failing cases, including held-out)
DC 2 | 2026-11-04 | exp0_pk_v2 | pk_wsp: interpolant knots placed as multiples of EC50 (KNOTS_REL) plus absolute far knots, replacing fixed absolute knots | reason: v1 held-out failure on extreme_fast (EC50 0.2): first chord [0,0.05] error 0.011. Rule is derivable from the spec but was found by a held-out failure. | DATA-INFORMED: yes (v1 held-out burned; v2 uses heldout_v2, defined in pk_spec.yaml and splits_v2.json before this run)
DC 3 | 2026-11-04 | exp0_pk_v2 | pk_trained: training configuration change 1 of 3: DAgger-style aggregation (3 rounds) of states visited by the model's own rollouts, labelled by the code rule | reason: v1 single-step training diverged in rollout on dev schedules. | DATA-INFORMED: yes
[auto] FP 2 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev fast_absorber/dose_then_wait (+replay) | dev
[auto] FP 4 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev fast_absorber/inhaled_then_iv (+replay) | dev
[auto] FP 6 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev fast_absorber/irregular_four (+replay) | dev
[auto] FP 8 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev fast_absorber/iv_bolus (+replay) | dev
[auto] FP 10 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev fast_absorber/single_2p5 (+replay) | dev
[auto] FP 12 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev fast_absorber/three_q20 (+replay) | dev
[auto] FP 14 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev fast_absorber/two_q20 (+replay) | dev
[auto] FP 16 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev fast_absorber/two_q5 (+replay) | dev
[auto] FP 18 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev slow_eliminator/dose_then_wait (+replay) | dev
[auto] FP 20 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev slow_eliminator/inhaled_then_iv (+replay) | dev
[auto] FP 22 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev slow_eliminator/irregular_four (+replay) | dev
[auto] FP 24 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev slow_eliminator/iv_bolus (+replay) | dev
[auto] FP 26 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev slow_eliminator/single_2p5 (+replay) | dev
[auto] FP 28 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev slow_eliminator/three_q20 (+replay) | dev
[auto] FP 30 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev slow_eliminator/two_q20 (+replay) | dev
[auto] FP 32 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev slow_eliminator/two_q5 (+replay) | dev
[auto] FP 34 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev typical/dose_then_wait (+replay) | dev
[auto] FP 36 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev typical/inhaled_then_iv (+replay) | dev
[auto] FP 38 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev typical/irregular_four (+replay) | dev
[auto] FP 40 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev typical/iv_bolus (+replay) | dev
[auto] FP 42 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev typical/single_2p5 (+replay) | dev
[auto] FP 44 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev typical/three_q20 (+replay) | dev
[auto] FP 46 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev typical/two_q20 (+replay) | dev
[auto] FP 48 | 2026-11-04 | exp0_pk_v2 | +2: wsp dev typical/two_q5 (+replay) | dev
[auto] FP 50 | 2026-11-04 | exp0_pk_v2 | +2: trained dev fast_absorber/dose_then_wait (+replay) | dev
[auto] FP 52 | 2026-11-04 | exp0_pk_v2 | +2: trained dev fast_absorber/inhaled_then_iv (+replay) | dev
[auto] FP 54 | 2026-11-04 | exp0_pk_v2 | +2: trained dev fast_absorber/irregular_four (+replay) | dev
[auto] FP 56 | 2026-11-04 | exp0_pk_v2 | +2: trained dev fast_absorber/iv_bolus (+replay) | dev
[auto] FP 58 | 2026-11-04 | exp0_pk_v2 | +2: trained dev fast_absorber/single_2p5 (+replay) | dev
[auto] FP 60 | 2026-11-04 | exp0_pk_v2 | +2: trained dev fast_absorber/three_q20 (+replay) | dev
[auto] FP 62 | 2026-11-04 | exp0_pk_v2 | +2: trained dev fast_absorber/two_q20 (+replay) | dev
[auto] FP 64 | 2026-11-04 | exp0_pk_v2 | +2: trained dev fast_absorber/two_q5 (+replay) | dev
[auto] FP 66 | 2026-11-04 | exp0_pk_v2 | +2: trained dev slow_eliminator/dose_then_wait (+replay) | dev
[auto] FP 68 | 2026-11-04 | exp0_pk_v2 | +2: trained dev slow_eliminator/inhaled_then_iv (+replay) | dev
[auto] FP 70 | 2026-11-04 | exp0_pk_v2 | +2: trained dev slow_eliminator/irregular_four (+replay) | dev
[auto] FP 72 | 2026-11-04 | exp0_pk_v2 | +2: trained dev slow_eliminator/iv_bolus (+replay) | dev
[auto] FP 74 | 2026-11-04 | exp0_pk_v2 | +2: trained dev slow_eliminator/single_2p5 (+replay) | dev
[auto] FP 76 | 2026-11-04 | exp0_pk_v2 | +2: trained dev slow_eliminator/three_q20 (+replay) | dev
[auto] FP 78 | 2026-11-04 | exp0_pk_v2 | +2: trained dev slow_eliminator/two_q20 (+replay) | dev
[auto] FP 80 | 2026-11-04 | exp0_pk_v2 | +2: trained dev slow_eliminator/two_q5 (+replay) | dev
[auto] FP 82 | 2026-11-04 | exp0_pk_v2 | +2: trained dev typical/dose_then_wait (+replay) | dev
[auto] FP 84 | 2026-11-04 | exp0_pk_v2 | +2: trained dev typical/inhaled_then_iv (+replay) | dev
[auto] FP 86 | 2026-11-04 | exp0_pk_v2 | +2: trained dev typical/irregular_four (+replay) | dev
[auto] FP 88 | 2026-11-04 | exp0_pk_v2 | +2: trained dev typical/iv_bolus (+replay) | dev
[auto] FP 90 | 2026-11-04 | exp0_pk_v2 | +2: trained dev typical/single_2p5 (+replay) | dev
[auto] FP 92 | 2026-11-04 | exp0_pk_v2 | +2: trained dev typical/three_q20 (+replay) | dev
[auto] FP 94 | 2026-11-04 | exp0_pk_v2 | +2: trained dev typical/two_q20 (+replay) | dev
[auto] FP 96 | 2026-11-04 | exp0_pk_v2 | +2: trained dev typical/two_q5 (+replay) | dev
[auto] FREEZE | 2026-11-04 | exp0_pk_v2 | weights sha256:ad34aa59bc425591 | wsp compile_pk frozen on dev variants; trained weights saved to results/trained_weights.npz
[auto] FP 98 | 2026-11-04 | exp0_pk_v2 | +2: wsp heldout slow_link_20/iv_q10_x5 (+replay) | heldout
[auto] FP 100 | 2026-11-04 | exp0_pk_v2 | +2: wsp heldout slow_link_20/one_huge_25mg (+replay) | heldout
[auto] FP 102 | 2026-11-04 | exp0_pk_v2 | +2: wsp heldout slow_link_20/twenty_tiny_q3 (+replay) | heldout
[auto] FP 104 | 2026-11-04 | exp0_pk_v2 | +2: wsp heldout slow_link_20/two_q2_then_wait (+replay) | heldout
[auto] FP 106 | 2026-11-04 | exp0_pk_v2 | +2: wsp heldout ultra_potent/iv_q10_x5 (+replay) | heldout
[auto] FP 108 | 2026-11-04 | exp0_pk_v2 | +2: wsp heldout ultra_potent/one_huge_25mg (+replay) | heldout
[auto] FP 110 | 2026-11-04 | exp0_pk_v2 | +2: wsp heldout ultra_potent/twenty_tiny_q3 (+replay) | heldout
[auto] FP 112 | 2026-11-04 | exp0_pk_v2 | +2: wsp heldout ultra_potent/two_q2_then_wait (+replay) | heldout
[auto] FP 114 | 2026-11-04 | exp0_pk_v2 | +2: wsp heldout weak_slow/iv_q10_x5 (+replay) | heldout
[auto] FP 116 | 2026-11-04 | exp0_pk_v2 | +2: wsp heldout weak_slow/one_huge_25mg (+replay) | heldout
[auto] FP 118 | 2026-11-04 | exp0_pk_v2 | +2: wsp heldout weak_slow/twenty_tiny_q3 (+replay) | heldout
[auto] FP 120 | 2026-11-04 | exp0_pk_v2 | +2: wsp heldout weak_slow/two_q2_then_wait (+replay) | heldout
[auto] FP 122 | 2026-11-04 | exp0_pk_v2 | +2: trained_typical_model_on_unseen_variants heldout slow_link_20/iv_q10_x5 (+replay) | heldout
[auto] FP 124 | 2026-11-04 | exp0_pk_v2 | +2: trained_typical_model_on_unseen_variants heldout slow_link_20/one_huge_25mg (+replay) | heldout
[auto] FP 126 | 2026-11-04 | exp0_pk_v2 | +2: trained_typical_model_on_unseen_variants heldout slow_link_20/twenty_tiny_q3 (+replay) | heldout
[auto] FP 128 | 2026-11-04 | exp0_pk_v2 | +2: trained_typical_model_on_unseen_variants heldout slow_link_20/two_q2_then_wait (+replay) | heldout
[auto] FP 130 | 2026-11-04 | exp0_pk_v2 | +2: trained_typical_model_on_unseen_variants heldout ultra_potent/iv_q10_x5 (+replay) | heldout
[auto] FP 132 | 2026-11-04 | exp0_pk_v2 | +2: trained_typical_model_on_unseen_variants heldout ultra_potent/one_huge_25mg (+replay) | heldout
[auto] FP 134 | 2026-11-04 | exp0_pk_v2 | +2: trained_typical_model_on_unseen_variants heldout ultra_potent/twenty_tiny_q3 (+replay) | heldout
[auto] FP 136 | 2026-11-04 | exp0_pk_v2 | +2: trained_typical_model_on_unseen_variants heldout ultra_potent/two_q2_then_wait (+replay) | heldout
[auto] FP 138 | 2026-11-04 | exp0_pk_v2 | +2: trained_typical_model_on_unseen_variants heldout weak_slow/iv_q10_x5 (+replay) | heldout
[auto] FP 140 | 2026-11-04 | exp0_pk_v2 | +2: trained_typical_model_on_unseen_variants heldout weak_slow/one_huge_25mg (+replay) | heldout
[auto] FP 142 | 2026-11-04 | exp0_pk_v2 | +2: trained_typical_model_on_unseen_variants heldout weak_slow/twenty_tiny_q3 (+replay) | heldout
[auto] FP 144 | 2026-11-04 | exp0_pk_v2 | +2: trained_typical_model_on_unseen_variants heldout weak_slow/two_q2_then_wait (+replay) | heldout
[auto] FP 146 | 2026-11-04 | exp0_pk_v2 | +2: trained_fresh_variant_models_on_unseen_schedules heldout slow_link_20/iv_q10_x5 (+replay) | heldout
[auto] FP 148 | 2026-11-04 | exp0_pk_v2 | +2: trained_fresh_variant_models_on_unseen_schedules heldout slow_link_20/one_huge_25mg (+replay) | heldout
[auto] FP 150 | 2026-11-04 | exp0_pk_v2 | +2: trained_fresh_variant_models_on_unseen_schedules heldout slow_link_20/twenty_tiny_q3 (+replay) | heldout
[auto] FP 152 | 2026-11-04 | exp0_pk_v2 | +2: trained_fresh_variant_models_on_unseen_schedules heldout slow_link_20/two_q2_then_wait (+replay) | heldout
[auto] FP 154 | 2026-11-04 | exp0_pk_v2 | +2: trained_fresh_variant_models_on_unseen_schedules heldout ultra_potent/iv_q10_x5 (+replay) | heldout
[auto] FP 156 | 2026-11-04 | exp0_pk_v2 | +2: trained_fresh_variant_models_on_unseen_schedules heldout ultra_potent/one_huge_25mg (+replay) | heldout
[auto] FP 158 | 2026-11-04 | exp0_pk_v2 | +2: trained_fresh_variant_models_on_unseen_schedules heldout ultra_potent/twenty_tiny_q3 (+replay) | heldout
[auto] FP 160 | 2026-11-04 | exp0_pk_v2 | +2: trained_fresh_variant_models_on_unseen_schedules heldout ultra_potent/two_q2_then_wait (+replay) | heldout
[auto] FP 162 | 2026-11-04 | exp0_pk_v2 | +2: trained_fresh_variant_models_on_unseen_schedules heldout weak_slow/iv_q10_x5 (+replay) | heldout
[auto] FP 164 | 2026-11-04 | exp0_pk_v2 | +2: trained_fresh_variant_models_on_unseen_schedules heldout weak_slow/one_huge_25mg (+replay) | heldout
[auto] FP 166 | 2026-11-04 | exp0_pk_v2 | +2: trained_fresh_variant_models_on_unseen_schedules heldout weak_slow/twenty_tiny_q3 (+replay) | heldout
[auto] FP 168 | 2026-11-04 | exp0_pk_v2 | +2: trained_fresh_variant_models_on_unseen_schedules heldout weak_slow/two_q2_then_wait (+replay) | heldout
[auto] FP 170 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout extreme_fast/alternating_q15 (+replay) | heldout
[auto] FP 172 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout extreme_fast/both_routes_same_tick (+replay) | heldout
[auto] FP 174 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout extreme_fast/long_rollout (+replay) | heldout
[auto] FP 176 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout extreme_fast/nothing (+replay) | heldout
[auto] FP 178 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout extreme_fast/overdose_ten_q6 (+replay) | heldout
[auto] FP 180 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout extreme_fast/three_fast_then_nothing (+replay) | heldout
[auto] FP 182 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout extreme_slow/alternating_q15 (+replay) | heldout
[auto] FP 184 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout extreme_slow/both_routes_same_tick (+replay) | heldout
[auto] FP 186 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout extreme_slow/long_rollout (+replay) | heldout
[auto] FP 188 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout extreme_slow/nothing (+replay) | heldout
[auto] FP 190 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout extreme_slow/overdose_ten_q6 (+replay) | heldout
[auto] FP 192 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout extreme_slow/three_fast_then_nothing (+replay) | heldout
[auto] FP 194 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout slow_link/alternating_q15 (+replay) | heldout
[auto] FP 196 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout slow_link/both_routes_same_tick (+replay) | heldout
[auto] FP 198 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout slow_link/long_rollout (+replay) | heldout
[auto] FP 200 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout slow_link/nothing (+replay) | heldout
[auto] FP 202 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout slow_link/overdose_ten_q6 (+replay) | heldout
[auto] FP 204 | 2026-11-04 | exp0_pk_v2 | +2: wsp_on_BURNED_heldout heldout slow_link/three_fast_then_nothing (+replay) | heldout
