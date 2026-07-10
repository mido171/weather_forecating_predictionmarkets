# T24 — Diagnostic Physics Teacher-to-Safe-Student Features

## Assignment

**Phase:** C Feature Platform  
**Required dependencies:** T15, T18, T21, T22  
**Bookkeeping folder suffix:** `diagnostic_physics_teacher_safe_student`

## Mission

Use timestamp-blocked IGRA, finalized climate, marine and TC sources to discover mechanisms and train deployable proxy students from safe inputs.

## Why this task exists

Blocked physics data remains valuable as a teacher but cannot be a live predictor.

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

1. Clean diagnostic sources
2. safe station/NWP/official features
3. target snapshots

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Define teacher states such as ridge height, low-level heat/moisture, marine temperature suppression, cloud/radiation and TC geometry.
2. Train cross-fitted student models using only safe inputs to predict each teacher state.
3. Evaluate student fidelity and incremental residual value separately.
4. Never include teacher values in production scoring.
5. Record exact teacher availability status and student feature lineage.
6. Reject students that only reproduce target outcomes indirectly or fail folds.

## Database/code objects that must exist or be updated

1. research teacher/student artifacts
2. feature_store safe_student outputs if promoted

## Required task-folder artifacts

In addition to the global folder contract, create:

1. teacher_state_definitions.csv
2. student_oof_predictions.parquet
3. fidelity_metrics.csv
4. incremental_value.csv
5. promotion_decisions.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Teacher exclusion from live schema
2. cross-fitting
3. ablation against safe inputs

## Acceptance criteria

1. Only causal safe-student outputs promoted
2. Diagnostic teacher remains blocked

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Poor fidelity or no incremental lift: preserve negative result and reject

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T24",
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
