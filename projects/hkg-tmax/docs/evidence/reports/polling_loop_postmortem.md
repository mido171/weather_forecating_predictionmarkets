# Daily Extract Polling Loop Postmortem

## Summary

The active Daily Extract watch repeated the same provider request family from
EXP-0005 through the interrupted EXP-0032 scaffold. The work preserved useful
evidence that the `2026-06-18` row was absent for the observed windows, but it
became operational monitoring disguised as research.

The acquisition reset closes this family. Future Daily Extract retrieval is a
lightweight collector task, not an experiment loop.

## What Was Repeatedly Fetched

- HKO Daily Extract catalog: `hko_daily_extract_catalog`
- HKO June 2026 Daily Extract monthly payload: `hko_daily_extract_202606`
- Watched date: `2026-06-18`

## Evidence From Existing Metrics

- experiment folders in the repeated active-watch family: 28
- poll iterations recorded across metrics: 166
- poll snapshot pairs recorded in metrics: 162
- implied raw catalog/monthly fetches recorded in poll snapshots: 324
- first recorded monthly retrieval in this family: `2026-06-18T17:50:43.847758Z`
- last recorded monthly retrieval before reset: `2026-06-18T21:49:47.131681Z`
- final row count: 17
- watched date present: false

## Unique Payload Hashes

| Source | Unique payload hashes |
|---|---|
| `hko_daily_extract_catalog` | `f80772b68545c56e6842c34998696fd11b7b9a80c0088bb1f6e4da65102616eb` |
| `hko_daily_extract_202606` | `c50910ab74e2ba8bff1f661fb1ae663d15b128dae0dfb4ed97c0e40c97bcbefc` |

## Why It Blocked Progress

The loop created repeated experiment folders, reports, tests, and commits for an
unchanged provider payload. That consumed engineering time that should have gone
to data acquisition: source adapters, durable collectors, provenance, coverage,
quality control, and historical backfills.

## New Policy

- No more rapid/bounded Daily Extract polling windows.
- No EXP-0033 or later experiment whose sole purpose is checking the same row.
- Daily Extract collection is scheduled at most once per local day at 09:00
  `Asia/Hong_Kong`.
- One retry six hours later is allowed only after a failed request.
- Successful unchanged payloads add one retrieval-ledger row and reuse the
  existing content-addressed raw object.
- Routine retrievals do not create experiment folders, full test runs, or
  commits.
- The collector must never block unrelated acquisition work.
