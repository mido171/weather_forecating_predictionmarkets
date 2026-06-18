# Predeclared Protocol

Completed before the EXP-0026 HKO poll.

## Target and horizon

- target version: HKO Daily Extract HKO station daily maximum temperature
  candidate for `2026-06-18`
- rules/target adapter version: G1 Daily Extract publication ledger, fail-closed
  until provider-first evidence is proven
- horizon: not applicable
- exact cutoff: not applicable
- prediction unit: not applicable

## Sample

- development: none
- validation: live HKO Daily Extract June 2026 monthly payload and catalog
- locked test: not opened
- live shadow: not applicable
- inclusion: catalog snapshot and monthly payload snapshots fetched during this
  bounded EXP-0026 poll
- exclusion: predictive features, market prices, model outputs, locked test data
- expected row count: unknown before poll; prior EXP-0025 row count was 17

## Baseline

- champion/baseline version: none
- frozen prediction artifact: none
- reason: G1 truth/timing evidence must pass before any model baseline

## Candidate

- feature/formula/model: none
- transformations: parse Daily Extract rows, update first-seen ledger, classify
  watched-date absent/present status
- allowed hyperparameters: six iterations, 30 second interval, three fetch
  attempts, two second retry sleep
- selection procedure: none
- seeds: not applicable
- compute budget: bounded live poll only

## Metrics

- primary: provider first-publication candidate count for watched date
- guardrails: immutable raw snapshots, metadata sidecars, content hash matches,
  HTTP status/final URL/request/response metadata, no duplicated raw paths
- calibration: not applicable
- subgroup: not applicable
- operational: bounded poll completes without unhandled source failure

## Uncertainty

- method: direct provider archive and ledger inspection
- block length: not applicable
- repetitions: six poll iterations
- confidence level: not statistical

## Multiplicity

- experiment family: G1 Daily Extract active publication watch
- number of variants: one continuation window
- correction/confirmation approach: any candidate first publication must be
  reviewed against prior absence snapshots and later revisions before G1 can pass

## Acceptance

Accept this checkpoint if all are true:

- the pre-poll validation gate passes or only emits existing G1/G2 gate warnings;
- the poll completes with immutable raw snapshots and metadata sidecars;
- content hashes match sidecars;
- sidecars include source ID, retrieval timestamp, storage schema version, HTTP
  status, requested URL, final URL, request headers, and response headers;
- `reports/daily_extract_publication.md` and `results/metrics.json` are
  generated;
- final pytest, validation, Ruff, and MyPy gates pass.

G1 itself passes only if the watched row appears with defensible
absent-before-present provider-first evidence and no later disqualifying
revision or ambiguity.

## Rejection

Reject or block if raw archive immutability, sidecar metadata, content hashes,
date parsing, HTTP retrieval, or watched-date classification cannot be verified.

## Locked-test decision

This experiment is not authorized to open the locked test. No
`TEST_ACCESS_LOG` entry is needed.
