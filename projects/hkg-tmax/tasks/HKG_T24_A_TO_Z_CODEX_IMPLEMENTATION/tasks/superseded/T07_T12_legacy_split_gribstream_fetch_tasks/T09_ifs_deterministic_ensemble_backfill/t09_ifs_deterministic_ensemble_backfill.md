# T09 — IFS Deterministic and Ensemble Backfill

## Assignment

**Phase:** B Acquisition  
**Required dependencies:** T06  
**Bookkeeping folder suffix:** `ifs_deterministic_ensemble_backfill`

## Mission

Backfill IFS operational and IFS ensemble data, preserving ECMWF attribution, as independent short-history challengers.

## Why this task exists

Independent model physics and assimilation can add diversity beyond GFS/GEFS.

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

1. T03 IFS selectors/licence
2. T05 locations
3. T06 client

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Backfill ifsoper from 2024-02-28 and ifsenfo from 2024-03-01 subject to selector dates.
2. Acquire all relevant cycles and 0–84h leads.
3. Acquire deterministic points/patches and ensemble members at selected locations.
4. Record model/parameter changes and ECMWF attribution in metadata.
5. Do not allow IFS to influence pre-2024 core development.
6. Persist raw and normalized manifests.

## Database/code objects that must exist or be updated

1. nwp_core values
2. catalog.source_license attribution

## Required task-folder artifacts

In addition to the global folder contract, create:

1. ifsoper_manifest.csv
2. ifsenfo_manifest.csv
3. member_coverage.csv
4. attribution_notice.md
5. quality_summary.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Selector/unit checks
2. cycle coverage
3. member checks
4. license metadata presence

## Acceptance criteria

1. Complete short-history backfill
2. Every IFS-derived artifact includes attribution metadata

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Missing selector/member: record and adjust only through approved semantic mapping

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T09",
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
