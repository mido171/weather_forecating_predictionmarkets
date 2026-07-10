# 0214 Tactical H24N GribStream backfill

Status: `historical_backfill_audited_with_warnings`.

## Purpose

Acquire and audit a broad tactical H24N NWP archive for HKG Tmax research.
This is an acquisition record, not a model-promotion experiment.

## Rate-limit stop and resume history

The first full run was manually stopped on 2026-06-25 after GribStream returned
HTTP 429:

- 1,079 chunks planned; 217 completed.
- 788,147 rows returned/upserted.
- 769,286 estimated credits consumed.
- `Retry-After` was 51,407 seconds.
- No earlier HTTP error was recorded in that run.

The later audited store contained 1,965,090 rows: 1,964,157 from the full
backfill and 933 older smoke rows.

## Final deep sanity audit

- 1,163 raw objects checked.
- Missing files: 0.
- Size mismatches: 0.
- Reported hash mismatches: 0.
- H24N lead/target-date/stencil structural mismatches: 0.

The live table was not pure full-run data. Downstream work must scope rows
through `source_response_object_id` and the full-run object URI until the 933
older smoke rows are removed or isolated.

## Mandatory warnings

- `ifsenfo` has recent groups missing member 0.
- `fourcastnetgfs` ends before the requested tail.
- `nbmoc` returned no usable rows.
- `aigfspres` is upper-air support, not a surface Tmax source.
- `aigefssfc` has poor usable 2 m temperature coverage.
- Raw rows are not feature-safe until the H24N cutoff filter is applied.
- The API-event log referenced by the original stop note is not present in the
  canonical campaign folder.

## As-of contract

Only forecast rows available by the governing H24N cutoff may be used.
`valid_at` and target date are not substitutes for real publication
availability. Source-specific availability and member/lead filters remain
mandatory.

## Reproduce the offline audit

From `projects/hkg-tmax`:

```powershell
.\.venv\Scripts\python.exe scripts\audit_tactical_gribstream_deep_sanity.py --help
```

Review the command and database scope before execution. Any GribStream request
requires a separate explicit `--execute` decision and a bounded credit budget.

## Evidence map

- `full_tactical_backfill_ok_tmax/deep_sanity_audit_20260625.json`: complete
  machine-readable audit.
- `full_tactical_backfill_ok_tmax/progress.json`: progress state.
- `full_tactical_backfill_ok_tmax/batch_results.csv`: chunk ledger.
- `full_tactical_backfill_ok_tmax/batch_summary.json`: aggregate counts.

The two retired Markdown reports and their hashes are in
[`DOCUMENT_PROVENANCE.csv`](../../DOCUMENT_PROVENANCE.csv).
