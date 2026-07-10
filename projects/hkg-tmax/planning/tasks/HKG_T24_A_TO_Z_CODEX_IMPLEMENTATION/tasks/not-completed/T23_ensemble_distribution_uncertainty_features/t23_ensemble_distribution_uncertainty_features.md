# T23 — Ensemble Distribution and Uncertainty Feature Store

## Assignment

**Phase:** C Feature Platform  
**Required dependencies:** T08, T09, T10, T16, T18  
**Bookkeeping folder suffix:** `ensemble_distribution_uncertainty_features`

## Mission

Convert GEFS/IFS/AI ensemble members into calibrated-ready distributions, clusters, probabilities and disagreement features.

## Why this task exists

Ensembles provide uncertainty, tail scenarios and router trust information unavailable from deterministic forecasts.

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

1. Eligible ensemble member values
2. target snapshots
3. threshold definitions

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Compute member target-day Tmax from each trajectory.
2. Compute mean/median/quantiles/IQR/std/skew/kurtosis and threshold probabilities.
3. Compute spread in cloud/rain/radiation/moisture trajectories.
4. Detect member clusters/multimodality using fold-independent unsupervised algorithms fitted within training periods.
5. Create control-minus-mean, deterministic-minus-ensemble and cross-ensemble disagreement.
6. Retain member completeness and model-change flags.

## Database/code objects that must exist or be updated

1. feature_store ensemble features/member derived tables

## Required task-folder artifacts

In addition to the global folder contract, create:

1. ensemble_feature_definitions.csv
2. member_completeness.csv
3. distribution_examples.parquet
4. cluster_contract.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Quantile ordering
2. member count
3. missing-member sensitivity
4. no target-informed clustering

## Acceptance criteria

1. GEFS distribution features complete on core frame
2. Short ensembles marked challenger

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Insufficient members: set distribution unavailable; do not fake with interpolation

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T23",
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
