# Simulation data, severity tables, and standards inventory (EMS respiratory-distress teaching sims)

Compiled 2026-11-04. Scope: software, data, guideline severity tables, education standards. No drug dosing or intervention-harm content.

## Read this first: limits of this research
- WebFetch worked only for github.com. All other hosts were unreachable. The WebSearch budget ran out partway through, so **no searches could be done for sections 4, 5 and 6, nor for several gap-filling lookups in sections 1 and 3** (VitalDB license, eICU resolution/license, MIMIC-IV-ED license, prehospital datasets).
- Verification labels used: `primary (github.com)`, `secondary coverage`, `search snippet only`, `unverified`.
- **`unverified (background knowledge)`** means the item was NOT confirmed by any tool call this session. It comes from the assistant's prior knowledge and must be checked before use.
- Search tools returned summaries, not page text. Every numeric threshold below is quoted as the summary gave it. Where a summary was unsure, this file says so.
- Not obtained: the GINA exacerbation severity table, the GOLD Rome table, and the NASEMSO respiratory-distress guideline text. See section 1.

## (1) Guideline severity classification, adult acute asthma and COPD exacerbation

### 1a. Asthma: GINA

| Item | Finding | Verification | Source |
|---|---|---|---|
| Most recent GINA report confirmed | Documents exist for 2024 (May strategy report; Dec summary guide) and 2025 (Summary Guide and full "Updated November 2025" report). A secondary site refers to "GINA 2026" with different oxygen thresholds; not confirmed against a GINA document. | search snippet only | https://ginasthma.org/wp-content/uploads/2025/11/GINA-2025-Update-25_11_08-WMS.pdf ; https://ginasthma.org/wp-content/uploads/2025/11/GINA-Summary-Guide-2025-WEB_FINAL-WMS.pdf ; https://ginasthma.org/wp-content/uploads/2024/05/GINA-2024-Strategy-Report-24_05_22_WMS.pdf |
| GINA definitions | Moderate exacerbation: troublesome for the patient and requires a change in treatment. Severe: urgent treatment needed to prevent hospitalization or death. A separate life-threatening category exists in the summary guide's exacerbation figure (Figure 9 in the 2024 summary guide per search summary). | search snippet only | GINA 2024 Summary Guide: https://ginasthma.org/wp-content/uploads/2024/12/GINA-Summary-Guide-2024-WEB-WMS.pdf |
| Exact GINA table (talks in phrases/words, RR, pulse, SpO2, PEF, mental status) | **NOT CONFIRMED from GINA.** Secondary pages paraphrase it: mild-to-moderate = "talks in phrases", SpO2 90-95%; severe = "talks in words", RR >30/min, pulse >120 bpm, SpO2 <90% on air (AMBOSS); life-threatening = drowsy/confused, silent chest, bradycardia, SpO2 <90% (attendme.ai). These two pages disagree on some details and were not checked against GINA. The user should read the GINA Summary Guide figure directly. | secondary coverage, unconfirmed | https://www.amboss.com/us/knowledge/acute-asthma-exacerbation ; https://attendme.ai/pathways/asthma-exacerbation-gina-2025 |
| GINA 2026 note | One secondary page states revised oxygen thresholds (SpO2 <92%, target 92-95%) for "GINA 2026". Unverified; treat as a flag to check. | search snippet only | https://updatesinmedicine.com/articles/gina-2026-acute-asthma-management |

### 1b. Asthma: BTS/SIGN (SIGN 158, 2019 with later amendments)

Thresholds as the search summary stated them. The summary said primary text should be checked for exact wording and that a May 2025 amendment exists (not confirmed).

| Category | Features listed | Verification | Source |
|---|---|---|---|
| Moderate | Increasing symptoms; PEF >50-75% of best or predicted; no features of acute severe asthma | secondary coverage | https://revisemrcem.com/asthma-severity-scoring ; https://www.anaestheticinterview.com/knowledge/acute-asthma-severity-and-escalation |
| Acute severe | Any of: PEF 33-50% of best or predicted; cannot complete sentences in one breath; respiration >=25/min; pulse >=110/min | secondary coverage | same as above; also https://www.rightdecisions.scot.nhs.uk/asthma-pathway-bts-nice-sign-sign-244/managing-acute-asthma/management-of-acute-asthma-in-adults/asthma-management-algorithms-for-adults/management-of-acute-asthma-in-adults-in-hospital/ |
| Life-threatening | Any of: PEF <33% of best or predicted; SpO2 <92%; silent chest; cyanosis; poor respiratory effort; arrhythmia; hypotension; exhaustion; altered consciousness. Blood gas: PaO2 <8 kPa (60 mmHg) irrespective of oxygen | secondary coverage | same as above |
| Near-fatal | Raised PaCO2 (summary stated "a raised PaCO2 is near-fatal asthma"). Whether the guideline also lists "requiring mechanical ventilation with raised inflation pressures" was not in the snippet. | secondary coverage (partial) | same as above |
| Oxygen target (context only) | SpO2 94-98% | secondary coverage | same as above |

Note: sources said clinical signs correlate poorly with airway obstruction, and absent wheeze can mean severe obstruction. This matters for how a simulation presents "silent chest".

### 1c. NASEMSO National Model EMS Clinical Guidelines

| Item | Finding | Verification | Source |
|---|---|---|---|
| Current version | **Version 3.0, March 2022** (an August 2021 draft of v3.0 also circulates; v2.2 Jan 2019 and v2.1 June 2018 are older). No newer version appeared in results. NASEMSO announced v3 on 2022-03-23. No later version confirmed; the user should check the NASEMSO site. | search snippet only | https://ambulance.org/2022/03/23/v3clinical/ ; https://ems.utah.gov/wp-content/uploads/sites/34/2024/05/National-Model-EMS-Clinical-Guidelines_2022.pdf ; https://hhs.iowa.gov/media/198/download ; https://nasemso.org/content.aspx?page_id=22&club_id=157064&module_id=701974 |
| Section structure | A standalone "Respiratory Distress (Includes Bronchospasm, Pulmonary Edema)" guideline at about p. 190 of the v3.0 PDF, separate from a general "Respiratory" guideline (about p. 178). | search snippet only | Utah/Iowa PDFs above |
| Assessment findings and monitoring listed | From snippets: assess rate, breath sounds, accessory muscle use, retractions, patient positioning, oxygen saturation; waveform capnography recommended for respiratory complaints; supplemental oxygen to target SpO2 94-98%, guided by ventilation assessment such as EtCO2. Conditions flagged to watch for: CHF, COPD, asthma. **The full text of the respiratory-distress section was NOT retrieved**, so the complete assessment/monitoring list is unknown. The albuterol-for-wheezing wording seen was from the 2014 edition only. | search snippet only | Utah/Iowa PDFs above; 2014 edition https://www.bcm.edu/sites/default/files/2015/3f/national-model-ems-clinical-guidelines-23oct2014.pdf |
| Severity thresholds | The snippets showed no numeric asthma severity table in NASEMSO. Unknown whether one exists. | unknown | n/a |

### 1d. COPD exacerbation: GOLD

| Item | Finding | Verification | Source |
|---|---|---|---|
| Most recent report | **GOLD 2026** (PDF v1.3, 8 Dec 2025; key-changes summary dated 10 Nov 2025). GOLD 2025 also exists. | search snippet only | https://goldcopd.org/wp-content/uploads/2026/01/GOLD-REPORT-2026-v1.3-8Dec2025_WMV2.pdf ; https://goldcopd.org/wp-content/uploads/2025/11/KEY-CHANGES-GOLD-2026-10Nov2025.pdf |
| Classification basis | GOLD 2026 adopts the "Rome proposal": mild / moderate / severe by clinical features at presentation, not by treatment or utilization. Rome criteria were developed by modified Delphi and retrospectively validated in 347 hospitalized patients (3-year follow-up). | search snippet only | https://pmc.ncbi.nlm.nih.gov/articles/PMC13109179/ ; https://www.healio.com/news/pulmonology/20251117/2026-gold-report-addresses-disease-activity-technology-exacerbations-in-copd ; https://www.tandfonline.com/doi/full/10.1080/15412555.2026.2622744 |
| Parameters used | Dyspnoea VAS (0-10), respiratory rate, heart rate, SaO2 (when labs unavailable); CRP where available; blood gas items. **Numeric cut-offs per category were NOT retrieved.** The cutoffs in the query (RR <24, HR <95, etc.) were not confirmed and must not be used. | search snippet only | https://reference.medscape.com/cc2/p10/gold-diagnosis-and-management-copd-guideline-2026a1000k8y |
| Partial table excerpt | One secondary table listed PaO2 <=60 mmHg and/or PaCO2 >45 mmHg and pH <7.35 with altered mental status as severe-range features. Mild = worsening symptoms without vital-sign abnormality. Secondhand; not verified. | secondary coverage | https://www.guidelinecentral.com/guideline/2231686/ |

## (2) Open-source physiology engines

| Engine | License | Language / platforms / bindings | Asthma, COPD | Respiratory model, drugs | Version / date | Validation | Verification | URL |
|---|---|---|---|---|---|---|---|---|
| **Pulse Physiology Engine (Kitware)** | Apache 2.0 (Kitware FAQ: aims to keep permissive Apache 2.0). Uses Google protobuf (BSD-3). A GitHub mirror shows LICENSE and NOTICE files. | C++17 core. Linux, macOS, Windows, Docker, Android, WebAssembly, iOS, others per a mirror README. Python API built by default (CMake option); also C#, Java bindings per Kitware homepage. | **Asthma: yes.** `AsthmaAttack` patient action with Severity 0-1 (0 removes it); engine scales the normal IE ratio by a severity-based multiplier; bronchodilator drugs act against it (max effect 1.0 reverses the constriction). **COPD: yes**, a condition with severity 0-1; separate bronchitis and emphysema severities; modeled by raising small-airway/alveolar-membrane flow resistance in the circuit model. | Lumped circuit respiratory model (see Respiratory Methodology page). Drug library: full list in Table 5 of the Drugs Methodology validation section (not retrieved). Albuterol uses a coefficient-override method; docs note plasma albuterol falls faster than literature, and the inhaler model treats albuterol as a gas. | Kitware docs say **4.4.0** is current (4.3.0 dated Sept 2024 on the version page). 4.4.0 release date unknown. | Per-system/action/condition V&V in docs; a "Combined Effects Validation" section includes an albuterol stabilization case. Details not read. | search snippet only (docs); primary (github.com) for mirror README | https://pulse.kitware.com/ ; https://pulse.kitware.com/_respiratory_methodology.html ; https://pulse.kitware.com/group___asthma_attack_data.html ; https://pulse.kitware.com/_scenario_file.html ; https://pulse.kitware.com/_drugs_methodology.html ; https://pulse.kitware.com/_inhaler_methodology.html ; https://pulse.kitware.com/version.html ; https://pulse.kitware.com/_f_a_q.html ; mirror https://github.com/Naijaoracle/pulse-physiology-engine |
| **BioGears (ARA)** | Apache 2.0, stated on the GitHub README. A search summary said source headers cite Apache 2.0. **No NOTICE file appears in the repo file listing.** The "BioGears NOTICE license terms" phrase in the task was not found; unresolved. | C++; Windows, macOS, Linux; SWIG templates for other-language bindings (README). | **Asthma attack action: not confirmed.** The BioGears GitHub README never mentions asthma; the releases page does not mention it; the docs hits that mentioned asthma/COPD were Pulse pages. A BioGears-doc snippet mentioned COPD validation scenarios (COPDSevereEmphysema, COPDSevereBronchitisLeft), unverified. | Lumped circuit respiratory model (per search summary). Pulse is a fork/successor of the same lineage (HumanSim), per search summary. | Releases 7.2.1 to **8.2.0 (listed "Latest"; release notes said Jan 2025; the release listing shows 12 Dec with no year)**. | JOSS paper exists (see URL); scientific validation not described on the repo. | primary (github.com) for license, language, releases | https://github.com/BioGearsEngine/core ; https://github.com/BioGearsEngine/core/releases ; https://www.biogearsengine.com/ ; https://www.theoj.org/joss-papers/joss.02645/10.21105.joss.02645.pdf |
| **HumMod** | **Unknown.** Search returned no usable result. | n/a | n/a | n/a | n/a | n/a | unverified (background knowledge: HumMod is a large Guyton-lineage model distributed free as an executable with XML model files by University of Mississippi Medical Center; license terms unconfirmed) | none confirmed |
| **Physiome / CellML / JSim lung models** | Per-model; **not verified** | CellML (PMR), JSim web models | Resistance-compliance and multi-level airway models only; no asthma/COPD scenario confirmed | Ben-Tal 2006 (lung as single container, conducting-airway resistance parameter); JSim Lung_RC_P (airflow from resistance and compliance); BronchBronchiolAlv; Airway_bronchiole_alveolus | n/a | Published papers only | search snippet only | https://models.cellml.org/e/414/view ; https://www.imagwiki.nibib.nih.gov/physiome/jsim/models/webmodel/NSR/lungrcp ; https://www.imagwiki.nibib.nih.gov/physiome/jsim/models/webmodel/NSR/bronchbronchiolalv ; https://www.imagwiki.nibib.nih.gov/physiome/jsim/models/webmodel/NSR/airwaybronchiolealveolus |
| Other | A 2021 arXiv paper describes model-based synthetic patient-ventilator waveform generation for ML/education (not an engine; relevant as a data-generation precedent). Other open engines (for example Physiolibrary/Modelica, Python respiratory models) were not researched. | | | | | | search snippet only | https://arxiv.org/pdf/2103.15684 |

Gitlab/GitHub note: the official Pulse source is on gitlab.kitware.com (not reachable here). GitHub third-party mirrors exist; their currency is unknown. A GitHub search also showed Pulse-based projects including a scenario builder (https://github.com/dr-evan-ross/Pulse_Scenario_Builder) and a Pulse 4.3.1 reliability/sensitivity study (https://github.com/sadc-lab/pulse_reliability_atlas); both unreviewed. Pulse scenarios are JSON files (see Scenario Files page), which is directly relevant to a scenario format.

## (3) Clinical time-series datasets

| Dataset | License / access | Time resolution and content | Verification | Source |
|---|---|---|---|---|
| MIMIC-IV (core) | PhysioNet Credentialed Health Data License 1.5.0 (one paper cites this for MIMIC-IV v2.2). Credentialed access plus training required (the training requirement is background knowledge). | Hospital and ICU data, ED or ICU admissions 2008-2019 (GitHub page). | search snippet only; primary (github.com) for date range | https://github.com/MIT-LCP/mimic-code ; https://physionet.org/content/mimic-iv-fhir/1.0/ |
| **MIMIC-IV-ED v2.2** | **License text not retrieved.** Presumed same credentialed PhysioNet license; unverified. | `triage` table: vitals at triage (temp, HR, RR, SpO2, SBP, DBP, pain), one row per stay, no timestamp (per one secondary source). `vitalsign` table: repeated aperiodic vitals during the ED stay, similar variables plus rhythm; one source says routine every 1-4 hours. ED data 2011-2019. So **triage and repeated ED vitals both exist, at charting (about hourly) resolution, not minute-level.** | search snippet only | https://physionet.org/content/mimic-iv-ed/2.2/ ; https://physionet.org/content/mimic-iv-ed-demo/2.2/ed/ ; https://github.com/MIT-LCP/mimic-code |
| eICU-CRD | **Not confirmed.** | Over 200,000 ICU admissions, 335 hospitals, 2014-2015 (secondary). Resolution of `vitalPeriodic` not confirmed (background knowledge: roughly 5-minute medians; unverified). | search snippet only / unverified | https://arxiv.org/pdf/2105.05728 (mentions eICU) |
| HiRID v1.1.1 | PhysioNet Contributor Review Health Data License 1.5.0 | One entry every 2 minutes; 33,905 patients (ICU, Bern). | search snippet only | https://physionet.org/content/hirid/1.1.1/ |
| VitalDB | **CC BY 4.0 not confirmed** (search failed; GitHub repo page not found). | Surgical patients; one paper reports 6,388 cases; waveforms at 62.5-500 Hz; numerics at 1-7 s intervals. Operating room, not emergency. | search snippet only (case count and sampling); license unverified | https://www.bjanaesthesia.org.uk/article/S0007-0912(21)00175-6/fulltext ; https://www.anesth-pain-med.org/journal/view.php?number=1212 |
| MIT-BIH Arrhythmia Database v1.0.0 | **Open Data Commons Attribution License v1.0**; open access; DOI 10.13026/C2F305 | 48 half-hour two-channel ambulatory ECG excerpts from 47 subjects (records 1975-1979); 360 samples/s/channel, 11-bit, 10 mV range. ECG only. | search snippet only | https://physionet.org/content/mitdb/1.0.0/ ; https://physionet.org/content/mitdb |
| NEMSIS public-release research dataset | **Access by request form to NEMSIS TAC**; price not confirmed. Annual releases (2023, 2024 v3.5.0, 2025 released May 2026). | v3.5.0 has 43 files (6 reference tables, 1 computed table); ASCII about 209 GB uncompressed; also SAS/STATA. Vital signs are a one-to-many table (144+ million rows in the 2023 release). `eVitals.01` is the date/time vital signs were taken, and the manual says each set should have its own time, so **repeated vitals with timestamps exist, at the resolution crews chart (typically minutes apart, unknown)**. Not designed for treatment-response time series; the 2024 manual is the place to confirm field lists. | search snippet only | https://nemsis.org/2025-nemsis-public-release-research-dataset-now-available/ ; https://nemsis.org/wp-content/uploads/2025/09/2024-NEMSIS-RDS-350-User-Manual-v3.pdf ; https://nemsis.org/wp-content/uploads/2024/04/Research-dataset-FAQ-2023.pdf ; https://nemsis.org/wp-content/uploads/2021/05/2020-NEMSIS-RDS-340-User-Manual_v3-FINAL.pdf |
| **Minute-level prehospital vitals during respiratory-distress treatment** | **None found.** The search for this was not executed (budget exhausted), so "none found" means not searched, not proven absent. NEMSIS is the closest known source and is event-level, not monitor-level. | | unverified | n/a |

## (4) Scenario libraries and virtual-patient products

**Not researched with tools; the search budget ran out before this section.** Everything below is background knowledge, `unverified`, and URLs are stated from memory and not checked. Treat as a to-do list.

| Item | What is believed | Verification |
|---|---|---|
| MedEdPORTAL (AAMC) simulation cases | Peer-reviewed, open access, CC BY-NC (believed) cases with learning objectives, case stems, facilitator guides, debriefing, checklists. Typically scripted vitals per phase ("state") rather than continuous timelines. URL: https://www.mededportal.org | unverified (background knowledge) |
| Laerdal SimStore / SimCenter | Commercial scenario marketplace for Laerdal manikins (SimMan etc.); scenarios define states with vitals and transitions in Laerdal software. | unverified (background knowledge) |
| iSimulate | Simulated monitor/defibrillator (for example ALSi, REALITi360) driven by instructor-set vitals. | unverified (background knowledge) |
| Body Interact | Commercial screen-based virtual patient; branching cases with simulated physiology. | unverified (background knowledge) |
| Prodigy EMS and others (EMS1, JEMS, NAEMT, state scenario banks) | Not confirmed to exist in the form described in the task. | unverified |

Known general pattern (background knowledge, unverified): instructor scenarios are commonly authored as a stem, initial vitals, 3-5 "states" with vitals and expected learner actions, trigger rules to move between states, and a debrief guide. Confirm against actual products before relying.

## (5) Standards for validating and signing off educational simulations

**Not researched with tools.** All entries are `unverified (background knowledge)` and need checking.

| Standard | Believed content | URL (unchecked; the network blocked these hosts) |
|---|---|---|
| INACSL Healthcare Simulation Standards of Best Practice (HSSOBP) | Published in Clinical Simulation in Nursing; a 2021 edition is believed to be the latest the assistant knew of, with standards such as Professional Integrity, Participant Objectives, Facilitation, Simulation Design, Debriefing Process, Operations, Outcomes and Objectives, Professional Development, Evaluation of Learning and Performance, Simulation-Enhanced Interprofessional Education, and Simulation Glossary. A newer edition may exist; not checked. The Simulation Design standard is the most relevant to scenario authoring and validation. | https://www.inacsl.org/healthcare-simulation-standards |
| SSH accreditation | Accreditation of simulation programs (Core and add-on areas such as Teaching/Education, Assessment). Standards version not checked. | https://www.ssih.org/accreditation |
| CoAEMSP | Paramedic program standards (CAAHEP Standards and Guidelines) address clinical and field experience; how simulation is credited is not confirmed. | https://coaemsp.org |
| NAEMSE | Instructor course content on scenario design not confirmed. | https://naemse.org |

## (6) Reference equations for respiratory mechanics in airway obstruction

**Not searched; textbook citations from background knowledge, `unverified` as to URLs and details.** The Pulse and Physiome pages in section 2 are the only verified model sources in this area.

| Topic | Reference | Note |
|---|---|---|
| Airway resistance (Poiseuille, R ~ 1/r^4 as a first approximation; turbulent flow regimes) and work of breathing | West JB, Respiratory Physiology: The Essentials (Lippincott); Nunn's Applied Respiratory Physiology (Elsevier) | Textbook; edition not checked |
| Single-compartment lung (resistance-compliance) equation of motion | Same textbooks; verified analogues: JSim Lung_RC_P https://www.imagwiki.nibib.nih.gov/physiome/jsim/models/webmodel/NSR/lungrcp | |
| Dynamic hyperinflation / auto-PEEP | Pepe PE, Marini JJ, Am Rev Respir Dis 1982 ("Occult positive end-expiratory pressure in mechanically ventilated patients with airflow obstruction") | citation from memory; unverified |
| Alveolar ventilation / alveolar gas equation | West's Respiratory Physiology | unverified |
| Oxygen dissociation curve equation | Severinghaus JW, "Simple, accurate equations for human blood O2 dissociation computations", J Appl Physiol 1979;46(3):599-602 | citation from memory; unverified |
| Respiratory muscle fatigue | Roussos C, Macklem PT, "The respiratory muscles", N Engl J Med 1982 | citation from memory; unverified |
| Circulatory/ventilation overall | Guyton and Hall Textbook of Medical Physiology | unverified |

## Suggested follow-up (needs another session with search budget)
1. Read the GINA 2025 exacerbation figure and the GOLD 2026 Rome table directly; capture exact thresholds.
2. Read NASEMSO v3.0 pp. 178-190 for the assessment and monitoring lists.
3. Confirm licenses for MIMIC-IV-ED, eICU-CRD, VitalDB, and the HumMod terms.
4. Complete sections 4-6 with real searches; verify the INACSL latest edition.
5. Inspect Pulse's Table 5 drug list and 4.4.0 release date; BioGears asthma action existence.
