# T30 — IFS, AI, CWA WRF, and ARWF Short-History Challenger Experts

## Assignment

**Phase:** D Modelling  
**Required dependencies:** T09, T10, T11, T13, T27  
**Bookkeeping folder suffix:** `short_history_challenger_experts`

## Mission

Build shadow/challenger experts for sources that begin after the pre-2024 development window, without contaminating the core model selection.

## Why this task exists

New models can add diversity, but short history requires caps and staged evidence.

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

1. IFS/AI/CWA/ARWF feature-ready data
2. core frozen expert predictions
3. sealed protocol

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Create direct and lightly calibrated challenger forecasts.
2. Use predeclared 2024/2025/2026 sequencing only after core freeze.
3. Start with intercept/seasonal bias and strong shrinkage; compare more flexible MOS only when sample supports.
4. Train expected-error adapters.
5. Keep CWA/ARWF prospective shadow until minimum seasonal coverage.
6. Calculate incremental benefit versus core forecast, not standalone vanity scores.
7. Define initial rho caps and promotion gates.

## Database/code objects that must exist or be updated

1. research challenger registry
2. shadow live outputs

## Required task-folder artifacts

In addition to the global folder contract, create:

1. challenger_scoreboards.csv
2. incremental_lift.csv
3. weight_cap_recommendations.csv
4. history_sufficiency.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. No pre-existence backfill
2. sealed sequencing
3. cap enforcement
4. sample-size gates

## Acceptance criteria

1. Each challenger has explicit shadow/capped/promoted/rejected status
2. Core router remains reproducible without challengers

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Insufficient history: collect and shadow only; no flexible model

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T30",
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
