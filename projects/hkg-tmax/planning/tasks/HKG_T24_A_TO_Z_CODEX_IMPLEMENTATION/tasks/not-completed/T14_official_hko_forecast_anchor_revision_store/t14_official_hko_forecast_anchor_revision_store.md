# T14 — Canonical Official HKO Anchor and Revision Store

## Assignment

**Phase:** B Acquisition  
**Required dependencies:** T00, T01, T04  
**Bookkeeping folder suffix:** `official_hko_forecast_anchor_revision_store`

## Mission

Canonicalize the 2000–2026 HKO official local forecast archive into all-vintage and one-row-at-cutoff views, preserving text, issue history and parser era.

## Why this task exists

This near-continuous archive is the central anchor and longest supervised residual-learning source.

## Non-negotiable controls for this task

- Target date T is forecast at 15:00 HKT on T−1 under cutoff contract `hkg_t24_1500hkt_v1`, unless T01 formally versions an existing different contract.
- No value enters strict scoring unless availability before cutoff is proven.
- GribStream `asOf` alone is not proof of historical API availability.
- Store UTC as timezone-aware canonical time; derive HKT explicitly.
- Preserve raw data and lineage; clean into normalized tables and quarantine invalid rows.
- Keep 2024+ outcomes sealed unless this task is T36 and the frozen protocol authorizes access.
- Never use target T, same-row residuals, realized error flags, post-cutoff revisions, full-history preprocessing, or in-sample expert predictions.
- Candidate and baseline are compared on identical rows.


## Required inputs and prerequisites

1. public.hko_historical_forecasts_2000_2026
2. verified T00 facts
3. cutoff function
4. target labels under sealed permissions

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Preserve all raw rows and clean usable_local_minmax rows.
2. Verify issue timezone and convert to UTC/HKT correctly.
3. Define eligible rows for target T as issued/available before cutoff and meteorologically targeting T.
4. Create latest-at-cutoff anchor view and all-pre-cutoff-vintages view.
5. Derive first/latest max/min, revision count/path, issue intervals, revision momentum, issue age, source/product/parser era and missingness.
6. Populate stale_hours from verified issue/cutoff semantics.
7. Preserve full forecast/weather/wind/RH/PSR text.
8. Reconcile the single missing target date without fabrication.
9. Compute official raw baseline only on eligible target dates.

## Database/code objects that must exist or be updated

1. operational_anchor official_vintage/target_snapshot views
2. research official residual table behind sealed role

## Required task-folder artifacts

In addition to the global folder contract, create:

1. canonicalization SQL/code
2. coverage_calendar.csv
3. revision_feature_profile.csv
4. baseline_scoreboard_pre2024.csv
5. timezone_audit.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Reproduce 115,795 clean rows and 9,667 dates
2. latest-at-cutoff uniqueness
3. no post-cutoff issue selected
4. text preservation
5. baseline reproducibility

## Acceptance criteria

1. Near-continuous anchor established with exact selection
2. Discrepancies resolved or blocked

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Ambiguous issue timezone: quarantine affected era until proven

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T14",
  "status": "passed|rejected|blocked|partial",
  "git_commit": "...",
  "database_migration_version": "...",
  "input_manifest_sha256": "...",
  "output_manifest_sha256": "...",
  "created_tables_or_views": [],
  "created_files": [],
  "open_blockers": [],
  "downstream_ready": true
}
```

Every path in the handoff must be repository-relative and every listed artifact must exist.
