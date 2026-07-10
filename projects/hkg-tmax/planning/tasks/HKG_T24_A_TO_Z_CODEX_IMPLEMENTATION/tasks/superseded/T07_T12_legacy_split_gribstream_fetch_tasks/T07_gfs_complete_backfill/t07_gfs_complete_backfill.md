# T07 — GFS Complete HKG-Relevant Backfill

## Assignment

**Phase:** B Acquisition  
**Required dependencies:** T06  
**Bookkeeping folder suffix:** `gfs_complete_backfill`

## Mission

Backfill all useful GFS cycles and variables from the verified GribStream start date through current availability at all target/station/reference locations and approved grids.

## Why this task exists

GFS is the longest convenient deterministic global NWP anchor in GribStream and defines the core modern MOS frame.

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

1. T03 GFS selector map
2. T05 location/domain plan
3. T06 client
4. P0/P1 semantic variables

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Refresh GFS catalog introduction dates per selector.
2. Backfill from 2021-03-22 or later selector introduction date.
3. Collect all 00/06/12/18 cycles and 0–84h leads.
4. Collect point trajectories at HKO, all stations and reference points.
5. Collect local patch and compressed synoptic context fields.
6. Separate surface and pressure-level variable groups.
7. Persist raw responses, normalized values and coverage manifests.
8. Do not mark historical values strict until T16 approves availability.

## Database/code objects that must exist or be updated

1. nwp_core.model_run/point_value
2. raw object manifests
3. catalog selector versions

## Required task-folder artifacts

In addition to the global folder contract, create:

1. gfs_backfill_manifest.csv
2. coverage_by_run_variable_location.csv
3. missing_run_report.csv
4. volume_and_quota_report.csv
5. quality_summary.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Known-date spot checks
2. Kelvin/Celsius conversion tests
3. run/valid/lead consistency
4. duplicate check
5. expected cycle coverage

## Acceptance criteria

1. All planned chunks terminal success/rejected state
2. Coverage gaps enumerated
3. No dates before selector introduction requested

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Quota exhaustion: checkpoint and resume; do not silently downsample variables

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T07",
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
