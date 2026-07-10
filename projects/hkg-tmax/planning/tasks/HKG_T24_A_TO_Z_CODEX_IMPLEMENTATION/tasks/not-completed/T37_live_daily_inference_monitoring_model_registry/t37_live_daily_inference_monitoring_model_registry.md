# T37 — Live Daily Inference, Monitoring, and Model Registry

## Assignment

**Phase:** F Production  
**Required dependencies:** T35, T36  
**Bookkeeping folder suffix:** `live_daily_inference_monitoring_model_registry`

## Mission

Deploy the exact-vintage daily pipeline that freezes the T−24 snapshot, issues the forecast, records every decision, and updates states only after settlement.

## Why this task exists

Historical skill has no value unless live operation reproduces the same information contract.

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

1. Confirmed/frozen system
2. collectors
3. snapshot builder
4. model registry
5. scheduler

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Schedule source polling before cutoff with freshness checks.
2. At cutoff freeze snapshot and source manifest.
3. Run all available experts, router, specialists and distribution.
4. Persist expert forecasts, predicted errors, weights, corrections, abstention and final P50.
5. Emit warnings/no-forecast if critical sources fail rather than using future data.
6. After target settles, score every expert and update online states.
7. Monitor drift, calibration, source latency, missing cycles and feature ranges.
8. Support rollback to previous model version.

## Database/code objects that must exist or be updated

1. live issued forecasts/expert outputs/state updates/monitoring

## Required task-folder artifacts

In addition to the global folder contract, create:

1. production service/CLI
2. scheduler definitions
3. dashboard queries
4. alert rules
5. rollback runbook
6. live model card

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Dry-run historical replay
2. cutoff race conditions
3. missing-source fallback
4. state update order
5. rollback

## Acceptance criteria

1. One command/service produces auditable forecast
2. No post-cutoff source can enter
3. Every issued forecast immutable

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Critical data unavailable: issue degraded/no-forecast status according to policy

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T37",
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
