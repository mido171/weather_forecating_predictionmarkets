# T22 — Deterministic NWP Trajectory, Vertical, and Spatial Feature Store

## Assignment

**Phase:** C Feature Platform  
**Required dependencies:** T16, T18  
**Bookkeeping folder suffix:** `deterministic_nwp_trajectory_spatial_features`

## Mission

Convert eligible deterministic GFS/IFS/AI/CWA runs into physically meaningful target-day features without target outcomes.

## Why this task exists

Raw grid values are not yet useful; trajectories, gradients, radiation and vertical structure form the MOS inputs.

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

1. Eligible deterministic run values
2. semantic variable map
3. locations/domains
4. target snapshots

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Build target-day local-hour trajectories.
2. Derive direct Tmax, peak hour, heating/cooling slopes, plateau and diurnal range.
3. Integrate shortwave/cloud/rain over the heating window.
4. Derive moisture, T−Td, wind/onshore, pressure, ridge/subsidence and vertical-stability indices.
5. Derive HKO/inland/coastal/marine and synoptic gradients.
6. Derive run-age and run-to-run revision features from earlier eligible cycles.
7. Handle model-specific missing variables through availability masks, not cross-model imputation.
8. Record model version and selector era.

## Database/code objects that must exist or be updated

1. feature_store deterministic_nwp features

## Required task-folder artifacts

In addition to the global folder contract, create:

1. feature_definitions_nwp_deterministic.csv
2. trajectory_examples.parquet
3. spatial_summary_manifest.csv
4. model_feature_coverage.csv

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Local-day hour mapping
2. unit conversions
3. run selection
4. spatial aggregation
5. no future cycle

## Acceptance criteria

1. GFS P0 feature coverage sufficient for core frame
2. Each model family separately versioned

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Variable absent: feature unavailable for that model; do not substitute a different semantic silently

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T22",
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
