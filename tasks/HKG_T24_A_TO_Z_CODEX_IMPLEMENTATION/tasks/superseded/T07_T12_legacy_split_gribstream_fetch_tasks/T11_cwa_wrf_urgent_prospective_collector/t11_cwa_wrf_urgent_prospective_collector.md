# T11 — CWA WRF 15 km Urgent Prospective Collector

## Assignment

**Phase:** B Acquisition  
**Required dependencies:** T06  
**Bookkeeping folder suffix:** `cwa_wrf_urgent_prospective_collector`

## Mission

Deploy an always-on collector for the rolling three-day CWA WRF archive and preserve every future issue with exact first-seen timestamps.

## Why this task exists

CWA WRF is HKG-relevant regional guidance, but historical runs disappear quickly.

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

1. T03 CWA selectors
2. T05 local/synoptic domains
3. T06 client
4. scheduler/secret infrastructure

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Poll on a cadence that captures every 6-hourly run with retries and overlap.
2. Use /runs over the trailing window and deduplicate by run.
3. Collect HKO/station points, local patch and high-value surface/pressure variables.
4. Record first successful observation time per run/variable group.
5. Alert on missed cycles and coverage drops.
6. Backfill currently retained runs immediately.

## Database/code objects that must exist or be updated

1. live acquisition scheduler
2. nwp_core runs/values
3. availability events

## Required task-folder artifacts

In addition to the global folder contract, create:

1. collector service
2. scheduler config
3. first_seen_log.csv
4. missed_cycle_alerts
5. initial retained-window backfill

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Simulated missed poll recovery
2. duplicate polling
3. first-seen immutability
4. alert tests

## Acceptance criteria

1. Collector continuously archives new cycles
2. No retained run silently lost after deployment

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. API outage: preserve retry/alert evidence and recover within retention window

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T11",
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
