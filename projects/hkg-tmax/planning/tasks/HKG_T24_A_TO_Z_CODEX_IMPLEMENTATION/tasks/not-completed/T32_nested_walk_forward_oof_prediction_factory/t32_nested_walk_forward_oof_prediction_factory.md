# T32 — Nested Walk-Forward OOF Prediction Factory

## Assignment

**Phase:** D Modelling  
**Required dependencies:** T28, T29, T31  
**Bookkeeping folder suffix:** `nested_walk_forward_oof_prediction_factory`

## Mission

Create the authoritative out-of-fold prediction factory for every expert, specialist and baseline across long and modern frames.

## Why this task exists

The router and distribution layer are invalid if trained on in-sample expert predictions.

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

1. All expert training pipelines
2. frame registry
3. feature manifests

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Define expanding outer folds and inner tuning folds per frame.
2. Recompute all fold-fitted transformations inside folds.
3. Train/predict every expert for every eligible test date.
4. Store model/config/data hashes and support counts.
5. Verify one prediction per expert/date/frame.
6. Create common-row meta-training tables with availability masks.
7. Prevent any final/sealed outcomes from hyperparameter selection.

## Database/code objects that must exist or be updated

1. research.expert_oof_prediction
2. research.fold_registry

## Required task-folder artifacts

In addition to the global folder contract, create:

1. fold_definitions.csv
2. oof_prediction_matrix.parquet
3. fold_model_manifest.csv
4. missing_expert_matrix.csv
5. OOF_integrity_report.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. No train/test date overlap
2. preprocessing fit dates
3. prediction uniqueness
4. recompute sample folds

## Acceptance criteria

1. Router-ready OOF table exists
2. Every prediction traceable to fold artifact
3. No in-sample fallback

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Expert missing on date: availability mask; never substitute its fitted full-sample prediction

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T32",
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
