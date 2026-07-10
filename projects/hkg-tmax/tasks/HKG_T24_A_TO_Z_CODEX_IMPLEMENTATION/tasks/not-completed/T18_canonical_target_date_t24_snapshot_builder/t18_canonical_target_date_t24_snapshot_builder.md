# T18 — Canonical Target-Date T−24 Snapshot Builder

## Assignment

**Phase:** C Feature Platform  
**Required dependencies:** T01, T14, T16, T17  
**Bookkeeping folder suffix:** `canonical_target_date_t24_snapshot_builder`

## Mission

Build deterministic, replayable source snapshots for each target date containing only information eligible at the cutoff.

## Why this task exists

This is the single as-of join layer from which every strict feature and prediction must be derived.

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

1. Eligible official/NWP/live/source views
2. target-date calendar
3. cutoff contract
4. location registry

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. For each target date calculate cutoff UTC.
2. Select latest eligible official vintage and retain revision history IDs.
3. Select latest eligible run per model and all required target-day valid hours; retain earlier eligible runs for aging/revision.
4. Select latest eligible station/live observations.
5. Attach causally available target-memory source dates.
6. Store source availability/missingness masks.
7. Hash selected source IDs/objects and builder version.
8. Create development snapshots without joining labels; labels join only in research role.

## Database/code objects that must exist or be updated

1. feature_store.target_snapshot_manifest
2. snapshot source-link tables

## Required task-folder artifacts

In addition to the global folder contract, create:

1. snapshot builder code/SQL
2. snapshot_coverage.csv
3. source_selection_examples.md
4. snapshot_hash_reproducibility.csv

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. No selected available_at > cutoff
2. same inputs same hash
3. target label inaccessible
4. year boundary/leap tests

## Acceptance criteria

1. One reproducible snapshot per target/date/version
2. No forward-looking row can enter feature generation

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Missing source yields explicit availability mask, not future fallback

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T18",
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
