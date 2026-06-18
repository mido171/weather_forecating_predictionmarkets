# Protocol

## Predeclared Sample

- target source: HKO Daily Extract monthly backing payload
- target month: 2026-06
- watched local date: 2026-06-18
- horizon/model: not applicable; G1 target-publication evidence only
- exclusions: any claim based only on a latest payload, any watched date lacking
  an active absent-before-present snapshot, and any revised value not resolved
  by explicit review.

## Inputs

- `hko_daily_extract_catalog`
- `hko_daily_extract_202606`
- raw snapshot sidecars under `data/raw/`
- prior EXP-0003/EXP-0004 publication-ledger mechanics

## Method

1. Modify the ledger so provider-first candidate status requires:
   - explicit watched date;
   - timezone-aware active polling start;
   - first archived presence at or after active start;
   - at least one monthly snapshot at or after active start and before first
     presence where the watched date is absent;
   - no revision observed.
2. Add regression tests for absent-before-present, present-without-active-
   absence, and revision override behavior.
3. Run bounded active polling:
   `--active-polling-start-at now --watch-candidate-date 2026-06-18`.
4. Save metrics, raw hashes, report, and conclusion.

## Metrics

- polling iterations completed
- watched dates present/missing
- row count
- evidence class counts
- provider-first candidate count
- revision count
- active absence metadata for any candidate

## Acceptance Criteria

- The stricter candidate rule is covered by tests.
- The bounded poll exits normally and archives immutable raw snapshots.
- No row is labelled provider-first candidate without active absent-before-
  present evidence.
- G1 remains blocked unless candidate evidence survives cadence/revision review.

## Failure Criteria

- Polling fails or overwrites raw data.
- A candidate can be produced from a first-present snapshot with no active
  absence evidence.
- Timestamps are naive or unavailable.
- The report implies predictive modelling or market backtesting was run.

## Locked-Test Decision

No locked test access is authorized. This is target-publication infrastructure
only.
