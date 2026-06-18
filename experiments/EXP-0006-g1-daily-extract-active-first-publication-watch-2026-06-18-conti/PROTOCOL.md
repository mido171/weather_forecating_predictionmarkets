# Protocol

## Predeclared Sample

- target source: HKO Daily Extract monthly backing payload
- target month: 2026-06
- watched local date: 2026-06-18
- inherited active polling start: `2026-06-18T17:48:59.956593Z`
- horizon/model: not applicable; G1 target-publication evidence only

## Inputs

- `hko_daily_extract_catalog`
- `hko_daily_extract_202606`
- EXP-0005 active absent snapshots
- new EXP-0006 raw snapshots

## Method

1. Extend polling metrics with a `poll_snapshots` array containing every
   iteration's catalog and monthly raw snapshot hash, path, and retrieval time.
2. Run a bounded continuation poll for `2026-06-18` with the inherited active
   polling start.
3. Rebuild the publication ledger with absent-before-present candidate gating.
4. Document whether the watched row is still missing or has candidate evidence.

## Metrics

- polling iterations completed
- per-iteration catalog/monthly snapshot hashes
- watched date present/missing
- provider-first candidate count
- revision count
- row count
- evidence class counts

## Acceptance Criteria

- The poll exits normally and archives immutable raw snapshots.
- Metrics include every poll iteration snapshot.
- Candidate status requires active absent-before-present evidence.
- No predictive modelling, ML, G2 horizon selection, or market backtesting is
  performed.

## Failure Criteria

- Any poll iteration silently fails or overwrites raw data.
- Candidate status appears without a prior active absence.
- Timestamps are missing or naive.
- The report implies settlement parity is proven without actual candidate
  evidence.

## Locked-Test Decision

No locked test access is authorized.
