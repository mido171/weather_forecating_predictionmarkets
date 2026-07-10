# T08 — GEFS Atmospheric Members and Mean Complete Backfill

## Assignment

**Phase:** B Acquisition  
**Required dependencies:** T06  
**Bookkeeping folder suffix:** `gefs_members_mean_complete_backfill`

## Mission

Backfill GEFS mean and all atmospheric members with enough variables and locations to build calibrated Tmax distributions and uncertainty features.

## Why this task exists

GEFS supplies the central probabilistic expert and model-confidence information.

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

1. T03 GEFS selector map
2. T05 locations
3. T06 client
4. ensemble member metadata

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Backfill GEFS from 2020-10-01 or selector introduction.
2. Acquire gefsatmosmean over broader spatial domains.
3. Acquire all available gefsatmos members at HKO/station/reference points for P0/P1 subset.
4. Collect all cycles and 0–84h leads.
5. Persist member number explicitly; member 0 must not be mistaken for full ensemble.
6. Calculate acquisition-only coverage summaries, not target-informed features.
7. Document member count/model changes by era.

## Database/code objects that must exist or be updated

1. nwp_core point/member tables
2. ensemble coverage registry

## Required task-folder artifacts

In addition to the global folder contract, create:

1. gefs_mean_manifest.csv
2. gefs_member_manifest.csv
3. member_completeness.csv
4. model_change_log.csv
5. quality_summary.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. 31-member expectation where applicable
2. member uniqueness
3. mean-versus-members sanity
4. run/lead coverage

## Acceptance criteria

1. Member coverage sufficient for target-day distributions or exact gaps documented
2. No ensemble collapsed prematurely

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Quota constraint: prioritize all members at selected points before broad member grids

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T08",
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
