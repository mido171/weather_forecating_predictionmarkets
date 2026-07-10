# T36 — One-Time 2024/2025/2026 Sealed Validation and Confirmation

## Assignment

**Phase:** E Validation  
**Required dependencies:** T35  
**Bookkeeping folder suffix:** `one_time_sealed_validation_confirmation`

## Mission

Execute the predeclared one-time temporal confirmation sequence without turning holdouts into tuning data.

## Why this task exists

This is the only trustworthy way to know whether development gains survive unseen years and model eras.

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

1. Frozen candidate manifest
2. sealed permissions
3. validation protocol
4. contamination register

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Audit whether 2024/2025/2026 outcomes have previously influenced choices.
2. Open 2024 exactly once and score frozen candidate/baselines.
3. Apply predeclared pass/fail criteria.
4. If passed, refit using unchanged architecture/hyperparameters through 2024 and open 2025 once.
5. If passed, optionally replay 2026 YTD as a second untouched holdout; never call it prospective unless live predictions existed.
6. Do not tune from failed slices; return findings to a new development cycle with remaining holdout policy revised.

## Database/code objects that must exist or be updated

1. governance confirmation runs
2. research locked scores

## Required task-folder artifacts

In addition to the global folder contract, create:

1. contamination_audit.md
2. 2024_validation_scoreboard.csv
3. 2025_final_test_scoreboard.csv
4. 2026_holdout_scoreboard.csv if eligible
5. confirmation_conclusion.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Frozen hash verification
2. one-time access logs
3. baseline identical rows
4. no post-open mutation

## Acceptance criteria

1. Truthful confirmed/rejected status
2. Holdout use fully logged

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Contaminated holdout: label exploratory and do not claim confirmation

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T36",
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
