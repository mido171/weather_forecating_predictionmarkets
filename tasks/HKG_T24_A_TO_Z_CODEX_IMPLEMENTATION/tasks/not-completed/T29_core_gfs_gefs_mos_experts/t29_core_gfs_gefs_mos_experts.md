# T29 — Core GFS and GEFS Local MOS Experts

## Assignment

**Phase:** D Modelling  
**Required dependencies:** T22, T23, T27  
**Bookkeeping folder suffix:** `core_gfs_gefs_mos_experts`

## Mission

Train deterministic GFS MOS and probabilistic GEFS MOS on the strict modern core frame, including expected-error outputs.

## Why this task exists

This is the first major forward-looking atmospheric model layer capable of step-change improvement.

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

1. GFS/GEFS features
2. core frame 2021-03-22–2023-12-31
3. official and target labels behind research role

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Construct direct model Tmax baselines.
2. Train GFS residual-to-HKO MOS using trajectory, spatial and vertical features.
3. Train GEFS conditional median/quantiles and residual calibration.
4. Train expected absolute-error models for both experts.
5. Use nested expanding folds and model-version features.
6. Compare direct, simple bias, linear MOS, GAM and boosting.
7. Ablate radiation/cloud/moisture/spatial/vertical groups.

## Database/code objects that must exist or be updated

1. research GFS/GEFS model artifacts and OOF predictions

## Required task-folder artifacts

In addition to the global folder contract, create:

1. gfs_mos_scoreboard.csv
2. gefs_mos_scoreboard.csv
3. oof_predictions.parquet
4. quantile_metrics.csv
5. feature_group_ablation.csv

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. OOF-only downstream outputs
2. quantile coverage
3. model version/era checks
4. strict eligibility

## Acceptance criteria

1. MOS beats direct model or is rejected
2. GEFS distribution calibrated enough for router/distribution layer
3. All corrections bounded and documented

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Historical availability grade below B: results labeled diagnostic-proxy, not strict champion

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T29",
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
