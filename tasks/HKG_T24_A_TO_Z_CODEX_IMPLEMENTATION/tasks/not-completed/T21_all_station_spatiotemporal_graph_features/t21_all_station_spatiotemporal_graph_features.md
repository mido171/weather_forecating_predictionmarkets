# T21 — All-Station Spatiotemporal and Graph Feature Store

## Assignment

**Phase:** C Feature Platform  
**Required dependencies:** T05, T15, T18  
**Bookkeeping folder suffix:** `all_station_spatiotemporal_graph_features`

## Mission

Use every valid station as a spatial sensor array, generating causal anomalies, gradients, group modes, propagation and disagreement features.

## Why this task exists

Local HKO microclimate and regional advection are major potential residual-correction sources.

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

1. Clean station observations/summaries
2. station registry/groups
3. target snapshots

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. For each station create latest-before-cutoff levels, changes, slopes and prior-only anomalies.
2. Generate T−Td, pressure tendency, repaired u/v wind and circular shifts.
3. Generate pairwise/group gradients: coastal-inland, north-south, east-west, urban-airport-marine, upwind-downwind.
4. Generate station ranks, rank reversals, disagreement and missingness.
5. Create graph adjacency from distance/elevation/coast role; fit graph/PCA modes inside each fold only.
6. Separate historical proxy features from exact-vintage live features.
7. Produce station contribution coverage and dropout profiles.

## Database/code objects that must exist or be updated

1. feature_store station features
2. station graph version registry

## Required task-folder artifacts

In addition to the global folder contract, create:

1. station_feature_definitions.csv
2. pair_group_registry.csv
3. graph_edges.csv
4. coverage_by_station_year.csv
5. proxy_vs_live_status.csv

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Wind parser variance
2. as-of cutoff
3. fold-local PCA
4. station identity mapping
5. missing station robustness

## Acceptance criteria

1. All 36 ISD stations explicitly included or rejected with reason
2. No post-cutoff observation in summaries

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Historical availability unproven: features remain research-proxy while live-safe pipeline still built

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T21",
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
