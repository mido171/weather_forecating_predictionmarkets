# Predeclared Protocol

## Target and horizon

- target version: `hko_daily_absolute_max_first_published_pending_g1`
- rules/target adapter version: not used; target publication evidence only
- horizon: not applicable
- exact cutoff: not applicable
- prediction unit: HKO Daily Extract row for local date `2026-06-18`

## Sample

- development: not applicable
- validation: not applicable
- locked test: not opened
- live shadow: current-month Daily Extract publication watch
- inclusion: HKO Daily Extract catalog and June 2026 monthly Daily Extract page
- exclusion: any forecast features, CLMMAXT-as-canonical training labels, market
  data, market backtesting, predictive modelling, or machine learning
- expected row count: whatever HKO has published for June 2026 at retrieval time

## Baseline

- champion/baseline version: none
- frozen prediction artifact: none
- reason: G1 target-publication evidence is not a predictive experiment

## Candidate

- feature/formula/model: none
- transformations: parse monthly Daily Extract rows into the publication ledger
- allowed hyperparameters: six iterations, 30 second interval, three fetch
  attempts, two second retry sleep
- selection procedure: predeclared bounded live poll
- seeds: none
- compute budget: network fetch plus local validation

## Metrics

- primary: provider-first-publication candidate count for watched date
  `2026-06-18`
- guardrails: immutable raw snapshot count, metadata sidecar count, watched date
  missing/present state, revision count
- calibration: not applicable
- subgroup: not applicable
- operational: poll completes without unhandled transient fetch failure

## Uncertainty

- method: not applicable
- block length: not applicable
- repetitions: not applicable
- confidence level: not applicable

## Multiplicity

- experiment family: G1 Daily Extract publication watch
- number of variants: one bounded continuation poll
- correction/confirmation approach: no model selection; if row appears, later
  snapshots must be reviewed for revision before G1 parity can pass

## Acceptance

Accept as G1 infrastructure if the bounded poll completes, writes immutable raw
snapshots and metadata sidecars, updates the publication ledger/report, and all
repository gates pass. This can extend active absence evidence or capture a
candidate first publication. It is not milestone-eligible by itself unless full
G1 parity criteria are later satisfied.

## Rejection

Reject or block if fetching fails after retry exhaustion, raw snapshot metadata
is missing, overwrite/no-provenance behavior is detected, the watched-date
candidate logic violates absent-before-present requirements, or tests/validation
fail.

## Locked-test decision

This experiment is not authorized to open any locked predictive test. It uses
target-publication source evidence only.
