# T00 — Repository, Database, and Contract Preflight

## Assignment

**Phase:** A Foundation  
**Required dependencies:** None; this is a starting task.  
**Bookkeeping folder suffix:** `repository_database_preflight`

## Mission

Inspect the real repository and database, verify the corrected official-forecast facts, identify existing conventions, and produce the immutable starting-state manifest.

## Why this task exists

Every later task depends on correct table names, time semantics, migration tooling, credentials, and existing code. This task prevents parallel architectures and false assumptions.

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

1. Repository root and all AGENTS.md files
2. Configured Postgres connection
3. CURRENT_HKO_FORECAST_DB_FACTS.json
4. Audit bundle and existing experiment evidence

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Inspect repository structure, git status, branches, installed skills, migration stack, database libraries, environment-variable conventions, schedulers, object storage, tests and CI.
2. Enumerate database schemas, tables, views, functions, indexes, row counts, min/max timestamps, permissions and owners.
3. Run direct SQL against public.hko_historical_forecasts_2000_2026 to reproduce the 115,795-row clean subset, 9,667 distinct target dates, one missing date, product counts and numerical summaries.
4. Determine exact timezone semantics of issue columns; do not infer from names.
5. Locate all current dataset tables, loaders, acquisition scripts, model code and experiment registries.
6. Hash the repository inputs and record the starting git commit.
7. Create a discrepancy register; user-supplied facts are canonical only after verification.

## Database/code objects that must exist or be updated

1. catalog.preflight_snapshot or repository equivalent
2. No destructive schema changes

## Required task-folder artifacts

In addition to the global folder contract, create:

1. repo_inventory.md
2. database_inventory.csv
3. official_forecast_verification.sql
4. official_forecast_verification_results.csv
5. starting_state.json
6. discrepancies.csv

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. SQL result checks
2. timezone conversion tests
3. read-only safety check

## Acceptance criteria

1. All repository conventions documented
2. Official forecast counts exactly reproduced or discrepancy clearly proven
3. Every existing data/model table mapped to an owner and purpose
4. No uncommitted user work overwritten

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Database unavailable: complete repository inspection and provide exact SQL command/blocker
2. Count mismatch: stop canonicalization and open blocker; do not silently continue

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T00",
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
