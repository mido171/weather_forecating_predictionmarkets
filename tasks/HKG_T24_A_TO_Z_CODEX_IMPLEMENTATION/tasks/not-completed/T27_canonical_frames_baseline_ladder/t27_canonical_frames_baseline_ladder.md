# T27 — Canonical Evaluation Frames and Baseline Ladder

## Assignment

**Phase:** D Modelling  
**Required dependencies:** T18, T26  
**Bookkeeping folder suffix:** `canonical_frames_baseline_ladder`

## Mission

Define immutable target-date frames and score all simple baselines before complex modelling.

## Why this task exists

Model gains are meaningless without identical-row comparisons and frame transparency.

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

1. Target snapshots
2. labels under research role
3. split policy
4. official anchor
5. feature matrices

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Create OFFICIAL_ANCHOR_DEV 2000–2023 and CORE_NWP_DEV 2021-03-22–2023-12-31.
2. Create long-history target and diagnostic frames.
3. Score causal climatology, target-memory, official raw, official residual memory, direct GFS, direct GEFS median, simple MOS baselines and static blends.
4. Report identical-row MAE/RMSE/bias/tails and yearly/season slices.
5. Record missing-source masks and frame membership.
6. Register previous trustworthy champions only on comparable frames.

## Database/code objects that must exist or be updated

1. research.frame_registry
2. research.baseline_scores
3. research.predictions

## Required task-folder artifacts

In addition to the global folder contract, create:

1. frame_membership.csv
2. baseline_scoreboard.csv
3. slice_metrics.csv
4. yearly_metrics.csv
5. comparability_report.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Row identity checks
2. score recomputation
3. sealed-period exclusion
4. baseline prediction lineage

## Acceptance criteria

1. Official raw baseline reproduced
2. All later tasks name a frame and baseline
3. No cross-frame leaderboard mixing

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Core source coverage insufficient: adjust frame transparently, never impute future data

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T27",
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
