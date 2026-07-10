# T25 — Causal Online Residual and Source Performance State Engine

## Assignment

**Phase:** C Feature Platform  
**Required dependencies:** T14, T18  
**Bookkeeping folder suffix:** `causal_online_residual_state_engine`

## Mission

Implement multi-timescale, source/model/regime residual states that update only after settlement.

## Why this task exists

Existing evidence shows causal residual memory is one of the most reliable incremental signals.

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

1. Official and expert predictions
2. settled labels through allowed dates
3. target snapshots

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Implement EWMA half-lives 5/10/20/40 plus robust median and shrinkage.
2. Maintain signed bias, absolute error, volatility, over/under streaks and support counts.
3. Key states by source era, model, season and carefully selected regimes with hierarchical fallback.
4. Enforce score-then-update order.
5. Cold-start and missing-settlement handling.
6. Replay states from scratch deterministically.

## Database/code objects that must exist or be updated

1. feature_store online_state tables
2. live state store

## Required task-folder artifacts

In addition to the global folder contract, create:

1. state_definition.csv
2. state_replay_tests.csv
3. cold_start_policy.md
4. sample_state_history.parquet

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. No current target in state
2. deterministic replay
3. gap handling
4. era reset/change-point tests

## Acceptance criteria

1. State at T depends only on settled dates before T
2. No low-support unshrunk context state

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Settlement unavailable: freeze state; never infer residual

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T25",
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
