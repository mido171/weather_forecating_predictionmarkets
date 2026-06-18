# Protocol

## Predeclared Sample

- target source: HKO Daily Extract monthly backing payload
- target month: 2026-06
- watched local date: 2026-06-18
- inherited active polling start: `2026-06-18T17:48:59.956593Z`
- scope: G1 target-publication evidence only

## Inputs

- `hko_daily_extract_catalog`
- `hko_daily_extract_202606`
- prior active absent snapshots from EXP-0005 and EXP-0006
- new EXP-0007 raw snapshots

## Method

1. Run a bounded six-iteration continuation poll with 30-second spacing.
2. Use bounded fetch retries (`--fetch-attempts 3 --retry-sleep-seconds 2`).
3. Rebuild the publication ledger using absent-before-present candidate gating.
4. Record per-iteration raw snapshot metadata in `results/metrics.json`.

## Metrics

- polling iterations completed
- poll snapshot count
- watched date present/missing
- provider-first candidate count
- revision count
- row count
- evidence class counts

## Acceptance Criteria

- Poll exits normally and archives immutable raw snapshots.
- Metrics include every poll iteration snapshot.
- No provider-first candidate is emitted without active absent-before-present
  evidence.
- No G2, modelling, ML, or market backtesting is performed.

## Failure Criteria

- Polling fails after configured retries.
- Any raw snapshot metadata is missing.
- Timestamps are missing or naive.
- The experiment claims G1 passed without first-present evidence.

## Locked-Test Decision

No locked test access is authorized.
