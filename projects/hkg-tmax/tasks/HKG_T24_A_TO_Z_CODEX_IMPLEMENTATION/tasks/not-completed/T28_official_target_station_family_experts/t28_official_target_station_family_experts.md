# T28 — Official, Target-Memory, and Station Family Experts

## Assignment

**Phase:** D Modelling  
**Required dependencies:** T27  
**Bookkeeping folder suffix:** `official_target_station_family_experts`

## Mission

Train the long-history official residual expert, target-memory expert and station microclimate expert with genuine temporal OOF predictions.

## Why this task exists

These experts exploit the current corpus before NWP routing and provide fallback when NWP is absent.

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

1. Official/target/station feature families
2. canonical frames
3. OOF factory conventions

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Train regularized linear/GAM baselines first.
2. Train constrained boosting candidates with nested temporal tuning.
3. Official expert predicts actual minus official anchor with source-era structure.
4. Target-memory expert predicts Tmax or official residual using long history.
5. Station expert predicts small residual correction and expected error; separate proxy/live status.
6. Generate OOF predictions and feature ablations.
7. Cap corrections and compare with causal residual-memory baseline.

## Database/code objects that must exist or be updated

1. research model registry/OOF predictions

## Required task-folder artifacts

In addition to the global folder contract, create:

1. expert_scoreboards.csv
2. oof_predictions.parquet
3. feature_ablation.csv
4. model_cards/
5. correction_cap_study.csv

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Nested fold isolation
2. feature whitelist
3. identical rows
4. negative-control shuffled features

## Acceptance criteria

1. At least one OOF artifact per expert
2. No in-sample prediction used downstream
3. Weak experts retained only as diagnostics/fallback

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Station proxy status prevents production promotion until availability proof

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T28",
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
