# T17 — End-to-End Data Quality, Completeness, and Idempotency Gate

## Assignment

**Phase:** B Acquisition  
**Required dependencies:** T07, T08, T09, T10, T11, T12, T13, T14, T15, T16  
**Bookkeeping folder suffix:** `end_to_end_data_quality_reconciliation`

## Mission

Run full integrity checks across raw objects, database rows, coverage, units, timestamps, stations, members, variables, runs and eligibility before feature engineering.

## Why this task exists

Feature and model work must not proceed on incomplete or silently corrupted acquisition.

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

1. All acquisition manifests
2. normalized tables
3. quality rules

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Reconcile request/response/value counts and hashes.
2. Detect missing cycles, variable gaps, member gaps, duplicate keys and impossible leads.
3. Validate physical ranges and units by variable/model.
4. Audit UTC/HKT semantics and target-date mapping.
5. Validate station/location coverage and date-effective metadata.
6. Compare rerun idempotency.
7. Produce model/source readiness statuses.
8. Block failed source families from downstream strict tasks.

## Database/code objects that must exist or be updated

1. governance quality status tables
2. quarantine issues

## Required task-folder artifacts

In addition to the global folder contract, create:

1. data_quality_scorecard.csv
2. coverage_heatmaps data
3. missing_cycles.csv
4. unit_range_violations.csv
5. readiness_matrix.csv

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Full rerun
2. checksum verification
3. physical-range rules
4. coverage thresholds

## Acceptance criteria

1. Core GFS/GEFS/official data pass
2. Any gaps explicitly represented, never silently imputed
3. Downstream readiness matrix approved

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Critical failure blocks dependent feature tasks but not unrelated sources

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T17",
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
