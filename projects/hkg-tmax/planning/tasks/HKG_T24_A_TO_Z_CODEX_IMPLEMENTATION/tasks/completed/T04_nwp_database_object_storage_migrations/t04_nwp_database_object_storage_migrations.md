# T04 — NWP Database, Object Storage, and Lineage Migrations

## Assignment

**Phase:** A Foundation  
**Required dependencies:** T00, T01, T02, T03  
**Bookkeeping folder suffix:** `nwp_database_object_storage_migrations`

## Mission

Implement the production storage architecture for raw requests/responses, model runs, values, availability contracts, features, research predictions and live forecasts.

## Why this task exists

Exact lineage, idempotency, partitioning and access control must exist before millions of forecast values arrive.

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

1. REFERENCE_POSTGRES_SCHEMA.sql
2. existing migration framework
3. T02 registry
4. T03 sizing

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Adapt schemas to repository conventions.
2. Create source/model/location/selector/run/request/response/value/feature/prediction/live/quarantine objects.
3. Partition high-volume values by month/model.
4. Create deterministic natural keys and conflict-safe upserts.
5. Integrate object storage or repository data lake for raw NDJSON/Parquet.
6. Implement retention and checksum verification.
7. Create development/live/diagnostic roles and sealed label permissions.
8. Add backup and rollback documentation.

## Database/code objects that must exist or be updated

1. catalog.*, governance.*, raw_audit.*, nwp_core.*, feature_store.*, research.*, live.*, quarantine.*

## Required task-folder artifacts

In addition to the global folder contract, create:

1. versioned migrations
2. schema diagram
3. index/partition plan
4. rollback plan
5. migration test log

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Apply/rollback on isolated DB
2. duplicate insert idempotency
3. permission tests
4. partition routing tests
5. foreign-key lineage tests

## Acceptance criteria

1. Migrations execute cleanly twice
2. No raw data overwritten
3. Live role isolated
4. Expected query plans use indexes

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Production DB unavailable: execute against isolated PostgreSQL and leave exact production command

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T04",
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
