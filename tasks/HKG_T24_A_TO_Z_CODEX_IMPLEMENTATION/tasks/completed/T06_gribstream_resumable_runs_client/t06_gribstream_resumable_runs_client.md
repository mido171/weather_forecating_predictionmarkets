# T06 — Resumable GribStream Runs Client and Raw Landing Zone

## Assignment

**Phase:** B Acquisition  
**Required dependencies:** T03, T04, T05  
**Bookkeeping folder suffix:** `gribstream_resumable_runs_client`

## Mission

Implement the reusable authenticated client, request planner, selector resolver, chunker, retry/quota logic, raw landing, normalization and manifests used by every model acquisition task.

## Why this task exists

A single audited client prevents endpoint drift, duplicated downloads and inconsistent time handling.

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

1. GribStream API key via GRIBSTREAM_API_KEY
2. T03 catalog/selector snapshots
3. T04 schema
4. T05 locations
5. acquisition_policy.yaml

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Use official Python client or direct HTTP while preserving exact payloads.
2. Use /runs for backfill; support timeseries only for controlled operational convenience.
3. Canonicalize request JSON and derive request SHA-256.
4. Support coordinate and grid requests, ensemble members, expressions, filters, min/max lead.
5. Stream NDJSON/gzip to immutable object files while computing checksum and row count.
6. Normalize forecasted_at to run_time_utc and forecasted_time to valid_time_utc; compute lead_minutes.
7. Capture retrieved_at and prospective first_seen.
8. Implement resumable state machine and bounded retries.
9. Prevent tokens in logs.
10. Expose dry-run row/quota estimates.

## Database/code objects that must exist or be updated

1. raw_audit.acquisition_request
2. raw_audit.response_object
3. nwp_core.model_run
4. nwp_core.point_value

## Required task-folder artifacts

In addition to the global folder contract, create:

1. client package
2. CLI commands
3. request planner
4. normalizer
5. resume ledger
6. sample manifests
7. operator runbook

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Mock API tests
2. real tiny smoke query
3. out-of-order response handling
4. gzip/NDJSON
5. idempotent rerun
6. 429 Retry-After
7. empty response classification
8. secret scan

## Acceptance criteria

1. Same request never creates duplicate values
2. Interrupted download resumes
3. All values preserve exact selector/run/valid/member lineage

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Catalog selector change: stop affected semantic variable and require new selector snapshot

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T06",
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
