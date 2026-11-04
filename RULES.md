# RULES.md: accounting discipline for every experiment in this repository

> **Provenance.** The brief says to carry `RULES.md` from `ecg-wsp-v4.zip` over verbatim and
> extend it. That archive was not available when this file was written. The rules below are a
> reconstruction from the brief's description of the ECG experiment (every forward pass counted,
> every design change logged with its reason and a DATA-INFORMED flag, development data separated
> from held-out data before building, no held-out number seen before the model is frozen).
> **When the archive is available, replace the "Core rules" section with the original text
> verbatim and keep the "Extensions" section.**

## Core rules (reconstructed)

R1. **Count every forward pass.** A forward pass is one evaluation of the model under construction
    on one input unit (for the ECG network: one record window; for a patient model: one full
    rollout of one trajectory). Batched evaluation counts each unit. Log the running total in
    `LEDGER.md`.

R2. **Log every design change.** Any change to architecture, weights, thresholds, constants or
    rules after the first forward pass is a design change. Each entry records: date, what changed,
    why, and the flag `DATA-INFORMED: yes|no`. The flag is `yes` whenever a specific observed
    failing case motivated the change, even if the fix was derived by reasoning.

R3. **Separate development data from held-out data before building.** The split is written down
    (file list, hash) before the first forward pass. Held-out data is never inspected, plotted,
    or summarised during development.

R4. **Freeze before testing.** The model is frozen (weights hashed, hash recorded in the ledger)
    before any held-out number is computed. A held-out number seen before freezing voids the
    held-out claim; the data then becomes development data and new held-out data is needed.

R5. **No training, no fitting, no parameter search unless the experiment says so.** When the
    method is weight-space programming, every weight is set by explicit reasoning from the
    specification. If a value is tuned against data, it is a design change under R2 and the
    experiment is relabelled as data-informed.

R6. **Report failures as failures.** A result that misses its predefined criterion is reported
    with the number, not softened.

## Extensions for patient models

E1. **Specification first.** No compile target (code, weights, adapter, trained model) is built
    until the specification it implements is written down and version-stamped. The compile is
    checked against the specification, not the other way round.

E2. **Held-out units are whole variants and whole action sequences.** Never hold out neighbouring
    timesteps of a trajectory the model has seen. Hold out entire patient variants (parameter
    sets) and entire intervention sequences (orders, omissions, repeats, extremes).

E3. **Predefine pass/fail.** Each experiment file states the criteria, the revision limit (number
    of design changes allowed), and what result would end the track, before the first forward
    pass.

E4. **Log builder runs the same way.** When an LLM builder (local or frontier) produces a
    specification or a compile, log: builder model and version, prompt hash, tools allowed,
    number of sandbox executions, wall time, and the number of attempts before acceptance.

E5. **Clinical sign-off is separate from passing tests.** A finite test suite passing is not
    clinical validity. The ledger records who reviewed a specification and when; a specification
    without a named reviewer is marked `UNREVIEWED` and must not be used in a class session.

E6. **Replay is a test.** Every accepted patient model must reproduce an identical event log from
    an identical action log. A replay mismatch is a failure under R6.
