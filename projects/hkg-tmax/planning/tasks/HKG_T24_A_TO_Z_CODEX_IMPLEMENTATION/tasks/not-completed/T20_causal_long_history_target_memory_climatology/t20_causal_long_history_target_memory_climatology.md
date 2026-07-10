# T20 — Causal Long-History Target Memory and Climatology Feature Store

## Assignment

**Phase:** C Feature Platform  
**Required dependencies:** T01, T15, T18  
**Bookkeeping folder suffix:** `causal_long_history_target_memory_climatology`

## Mission

Exploit the 1884+ HKO target record through strictly available lags, causal climatology, trends, volatility and spell-state features.

## Why this task exists

Long history supplies stable local priors and rare-regime context that modern NWP history lacks.

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

1. Canonical target labels
2. label availability contract
3. target snapshots

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Determine earliest label date available at each cutoff; do not assume T−1.
2. Create explicit lag and lagged rolling windows with availability-safe offsets.
3. Create causal day-of-year, harmonic, decayed and trend-adjusted climatologies fitted on prior years only.
4. Create 7/30/60-day slopes, curvature, range, volatility, MAD/IQR, spell duration, breakout, reversal and year-over-year analog features.
5. Version feature formulas and effective history counts.
6. Create long-history target-memory expert-ready matrices independent of modern NWP.

## Database/code objects that must exist or be updated

1. feature_store target_memory definitions/values

## Required task-folder artifacts

In addition to the global folder contract, create:

1. feature_definitions_target_memory.csv
2. availability_lag_audit.csv
3. history_count_profile.csv
4. causal_climatology_validation.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Artificial future-value injection tests
2. fold-local climatology
3. lag boundary tests
4. no T/T−1 if unavailable

## Acceptance criteria

1. Every feature uses labels provably available
2. Recomputed values match reference tests

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Publication timing unresolved: enforce conservative approved lag and flag

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T20",
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
