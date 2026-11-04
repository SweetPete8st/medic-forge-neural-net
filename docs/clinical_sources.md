# Clinical sourcing status for the respiratory scenario

Full inventory with verification labels: `appendix/C_sim_data_standards_raw.md`. The pharmacology
sourcing agent was stopped by the model's safeguards before producing output, and the session's web
search budget ran out before the inventory's last three sections (scenario libraries, simulation
standards, reference equations) were verified. Those sections are labelled unverified in the appendix.

## What is sourced (secondary coverage unless noted)
- BTS/SIGN acute asthma features: acute severe PEF 33-50%, RR >= 25, HR >= 110, cannot complete
  sentences; life-threatening PEF < 33%, SpO2 < 92%, silent chest, cyanosis, poor effort, arrhythmia,
  hypotension, exhaustion, altered consciousness; near-fatal raised PaCO2.
- **GINA 2025 Summary Guide, primary care exacerbation figure (primary, read from the PDF on
  2026-11-04):** SEVERE: talks in words, sits hunched forwards, agitated; respiratory rate > 30/min;
  accessory muscles in use; pulse rate > 120 bpm; O2 saturation (on air) < 90%; PEF <= 50% predicted or
  best. LIFE-THREATENING: drowsy, confused or quiet chest. Discharge criteria include oxygen saturation
  > 94% on room air. (The PDF is copyrighted; cite, do not redistribute.)
- NASEMSO National Model EMS Clinical Guidelines v3.0 (March 2022): respiratory distress section
  lists rate, breath sounds, accessory muscle use, retractions, SpO2, waveform capnography, SpO2 target
  94-98% (secondary); the NASEMSO site refused the session's fetches (403), so the full text was not
  retrieved. Download it on the work computer.
- GOLD 2026: Rome mild/moderate/severe classification; numeric cut-offs not retrieved.
- Pulse Physiology Engine (primary, github mirror): Apache-2.0, AsthmaAttack action, COPD condition,
  Python API, JSON scenarios. BioGears: Apache-2.0; asthma action unconfirmed.
- Datasets: MIMIC-IV-ED (hourly-ish ED vitals), HiRID (2 min), NEMSIS (timestamped vital sets,
  request-based access), MIT-BIH (ODC-By 1.0). No minute-level prehospital treatment-response dataset found.

## What the clinical reviewer must supply before classroom use
Every entry in `spec/examples/asthma_adult_acute.yaml` with `verified: false`, in this order:
1. Drug kinetics and effect sizes: albuterol, ipratropium, IM epinephrine, IV magnesium,
   methylprednisolone, ketamine; and the inappropriate drugs midazolam and nitroglycerin. Sources to
   use: DailyMed labels, GINA 2025 acute chapter, NASEMSO v3.0, a current emergency medicine text.
2. Severity thresholds used by the stages and events (align to the GINA table once retrieved).
3. Device FiO2 values and CPAP contraindications.
4. Time-course constants (progression, fatigue, auto-PEEP): these have no literature values; the
   reviewer sets them to a defensible teaching timeline and the ledger records the choice.
5. The four trajectory envelopes.

The engine refuses `review.status: APPROVED` without named reviewers and dates in the review block.
