# Predeclared Protocol

Complete before inspecting validation/locked-test outcomes.

## Target and horizon

- target version: `hko_daily_absolute_max_first_published`, pending G1
- rules/target adapter version: initial fail-closed G1 adapter to be implemented
- horizon: not applicable; G1 is target/rules validation only
- exact cutoff: not applicable
- prediction unit: no forecasts or model predictions

## Sample

- development: all archived/source-discoverable target and rules evidence
  before implementation acceptance
- validation: real resolved HKG Tmax events whose rules, resolved winner, and
  authoritative HKO target value are recoverable
- locked test: not opened; G1 is not a forecast-performance experiment
- live shadow: current/unresolved event rules and target publication archive
  only, no prediction
- inclusion:
  - all discoverable resolved Polymarket Hong Kong highest-temperature events;
  - current Hong Kong highest-temperature event if discoverable;
  - HKO Daily Extract pages/payloads available for event dates;
  - HKO `CLMMAXT station=HKO` history;
  - a stratified sample of non-market dates across seasons and boundary values
    when Daily Extract values can be obtained;
  - every raw payload must be archived before parsing.
- exclusion:
  - any event without archived rules text and outcome mapping;
  - any date whose HKO Daily Extract field cannot be found or date semantics are
    ambiguous;
  - any inferred first-publication claim not backed by an archived payload;
  - any modelling, baseline fitting, or locked-test forecast scoring.
- expected row count: unknown until discovery; every verifiable resolved event
  must be included, and unverifiable events must be listed with reason.
- user-directed deferral:
  - Polymarket backtesting is out of scope for this experiment checkpoint.
  - Do not fetch or process Polymarket price history, books, trades, liquidity,
    execution, or market replay artifacts.
  - The active scope is the HKO target-station Tmax data system: Daily Extract,
    CLMMAXT, station metadata, parser safety, and parity artifacts.

## Baseline

- champion/baseline version: none
- frozen prediction artifact: none
- reason: target truth precedes baselines

## Candidate

- feature/formula/model: HKO Daily Extract parser, CLMMAXT parser, fail-closed
  target adapter primitives, parity table
- transformations:
  - preserve raw payloads and original rules text;
  - normalize rules text only for hashing/change detection;
  - parse decimal temperatures with `Decimal`;
  - map labels to explicit `[lower, upper)` intervals only where rules/outcomes
    justify it.
- allowed hyperparameters: none
- selection procedure: include all verifiable events; no cherry-picking
- seeds: none used
- compute budget: local parsing/tests plus external HTTP archival

## Metrics

- primary:
  - Daily Extract vs CLMMAXT exact-match rate on verifiable dates.
- guardrails:
  - missing source/field/date/precision halts;
  - all mismatches quarantined with taxonomy;
  - no future training label is called canonical without first-publication
    evidence.
- calibration: not applicable
- subgroup:
  - event dates;
  - boundary values near integer buckets;
  - seasonal sample dates;
  - missing/completeness-code dates.
- operational:
  - current event rules archived;
  - source contracts drafted for target/rules sources;
  - rules-change monitor or testable hash registry created.

## Uncertainty

- method: deterministic equality and interval-containment checks
- block length: not applicable
- repetitions: not applicable
- confidence level: not applicable

## Multiplicity

- experiment family: G1 target truth and rules parity
- number of variants: one adapter/parser family
- correction/confirmation approach: all parser changes require tests and
  fixture evidence; unverifiable cases remain quarantined

## Acceptance

- Exact HKO source, field, precision, timezone/date, station identity, revision
  language, and fallback language are extracted or explicitly fail closed.
- `CLMMAXT station=HKO` parity is quantified against Daily Extract values on
  all verifiable dates.
- Every mismatch is resolved, quarantined, or blocks G1.
- `data/gold/target_parity/` contains machine-readable parity artifacts.
- `reports/target_parity.md` documents evidence, gaps, mismatches, and gate
  status.
- Fail-closed adapter tests cover missing source, missing field, ambiguous date,
  unsupported precision, station mismatch, missing value, and source failure.
- `validate all`, `pytest`, `ruff`, and `mypy` pass before any commit.

## Rejection

Reject or block G1 if target semantics cannot be proven, if CLMMAXT parity
cannot be quantified, if the HKO target-station field/date/station semantics are
ambiguous, or if fail-closed tests cannot be made to pass without weakening
safety.

## Locked-test decision

Not authorized. G1 does not inspect forecast holdout outcomes or perform model
evaluation. No `TEST_ACCESS_LOG.md` entry is needed unless later work attempts
to open forecast-performance data.
