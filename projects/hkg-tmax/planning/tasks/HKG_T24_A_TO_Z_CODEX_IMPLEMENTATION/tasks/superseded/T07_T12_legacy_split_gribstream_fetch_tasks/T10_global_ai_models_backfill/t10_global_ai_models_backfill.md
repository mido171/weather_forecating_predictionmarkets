# T10 — Global AI Weather Model Backfill

## Assignment

**Phase:** B Acquisition  
**Required dependencies:** T06  
**Bookkeeping folder suffix:** `global_ai_models_backfill`

## Mission

Backfill available GraphCast, FourCastNet, AIFS, AIFS ensemble, AIGFS and AIGEFS surface/pressure sources as model-diversity challengers.

## Why this task exists

AI models may provide independent error structure but their histories are short and must be strongly controlled.

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

1. T03 AI model catalog
2. T05 locations
3. T06 client

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Backfill each model only over its documented archive/window.
2. Keep surface and pressure products linked but separate.
3. Acquire all relevant cycles/leads and ensemble members where applicable.
4. Record historical-archive end dates for GraphCast/FourCastNet.
5. Record exact start dates and selector changes.
6. Classify outputs as challenger/shadow; no unrestricted router weight.

## Database/code objects that must exist or be updated

1. nwp_core values and model registry

## Required task-folder artifacts

In addition to the global folder contract, create:

1. ai_model_manifests/
2. cross_model_coverage.csv
3. short_history_limits.md
4. quality_summary.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Date-window enforcement
2. surface/pressure join tests
3. ensemble member tests

## Acceptance criteria

1. Every AI model has complete or explicitly partial coverage manifest
2. No model represented outside its real archive

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Tiny sample models remain storage/shadow only

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T10",
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
