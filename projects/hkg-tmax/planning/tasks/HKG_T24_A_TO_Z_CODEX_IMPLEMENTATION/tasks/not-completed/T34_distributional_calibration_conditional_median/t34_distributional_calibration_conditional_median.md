# T34 — Distributional Calibration and Conditional-Median Point Forecast

## Assignment

**Phase:** D Modelling  
**Required dependencies:** T23, T33  
**Bookkeeping folder suffix:** `distributional_calibration_conditional_median`

## Mission

Calibrate the final residual distribution, quantiles and threshold probabilities; use the calibrated conditional median as the MAE-oriented point forecast.

## Why this task exists

Trading and robust point forecasting require calibrated uncertainty, not just a single mean.

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

1. Router OOF forecast
2. ensemble distributions
3. actual labels in development frame

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Fit EMOS/GAMLSS/quantile and conformal candidates using OOF residuals.
2. Calibrate P10/P25/P50/P75/P90 and threshold probabilities.
3. Condition scale on ensemble spread, expert disagreement, regime and recent error.
4. Enforce quantile monotonicity.
5. Compare final P50 to router point and official baseline on MAE.
6. Report CRPS, pinball, Brier and calibration.

## Database/code objects that must exist or be updated

1. research distribution artifacts

## Required task-folder artifacts

In addition to the global folder contract, create:

1. quantile_oof_predictions.parquet
2. calibration_curves.csv
3. probability_metrics.csv
4. point_median_scoreboard.csv
5. distribution_model_card.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Quantile ordering
2. coverage by season
3. OOF calibration
4. no sealed tuning

## Acceptance criteria

1. P50 point is no worse than router beyond tolerance and probabilities calibrated
2. Otherwise keep router point and distribution separately

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Small-sample regime scale models shrink to global

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T34",
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
