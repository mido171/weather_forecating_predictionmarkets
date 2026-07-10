# T13 — HKO ARWF Exact-Vintage Prospective Collector

## Assignment

**Phase:** B Acquisition  
**Required dependencies:** T04, T05  
**Bookkeeping folder suffix:** `hko_arwf_exact_vintage_collector`

## Mission

Implement prospective collection of every ARWF station forecast issue, raw payload and hourly station trajectory with exact first-seen time.

## Why this task exists

ARWF is already locally corrected multi-model guidance and may become a high-value local anchor, but current historical depth is insufficient.

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

1. Existing ARWF acquisition code/table
2. HKO official endpoint/page/API
3. T05 station registry
4. scheduler

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Discover and document the actual machine-readable source used by current code.
2. Poll frequently enough to detect issues without assuming a fixed 0400/1600 or noon/midnight schedule; store observed issue/first-seen times.
3. Preserve raw payload and response headers.
4. Normalize station, issue, valid time, lead, temperature, RH, wind and precipitation/weather fields.
5. Map every station to the registry.
6. Store all issues, not just latest.
7. Alert on parser/schema changes and missing station/lead cells.

## Database/code objects that must exist or be updated

1. raw ARWF objects
2. live_nwp_anchor.arwf_run/hourly_value or repository equivalent

## Required task-folder artifacts

In addition to the global folder contract, create:

1. collector
2. schema mapping
3. initial current backfill
4. first_seen manifest
5. station mapping
6. quality dashboard

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Schema drift fixture
2. issue deduplication
3. valid/lead tests
4. station completeness
5. first-seen immutability

## Acceptance criteria

1. Every future issue archived with raw lineage
2. No model-time numeric placeholder remains
3. ARWF remains challenger until evidence threshold

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. No machine-readable historical archive: document; do not invent past issues

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T13",
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
