# T15 — Clean and Normalize Every Existing Dataset and Station

## Assignment

**Phase:** B Acquisition  
**Required dependencies:** T02, T04, T05  
**Bookkeeping folder suffix:** `existing_dataset_clean_normalize_all_stations`

## Mission

Implement all audit-driven repairs and create clean diagnostic/operational/live views for every current data family.

## Why this task exists

New NWP cannot compensate for corrupted winds, sentinels, timezones or unlabeled source eligibility.

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

1. Audit quality issues
2. current datasets and DB tables
3. station registry

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Use canonical target labels; reconcile payload duplicates.
2. Daily climate: preserve Trace flag, null missing values, quarantine invalid dates, remain diagnostic.
3. IGRA: map documented sentinels, correct proven scaling, deduplicate POR/YTD, regenerate sounding features, remain diagnostic until latency proof.
4. ISD: repair wind parser from raw fields, use date-effective station metadata, rebuild pre-cutoff summaries from observations only, separate proxy and live-safe status.
5. TC best track: diagnostic labels only.
6. Radar/satellite/lightning/nowcast: normalize issue/frame/first-seen and preserve imagery/object manifests.
7. Marine/tide: normalize station/time/unit and availability.
8. NCEP inventory: repair cycle/valid metadata from filenames/GRIB.
9. Static geospatial: derive reproducible station features.
10. Experiment outputs: register as research artifacts only.

## Database/code objects that must exist or be updated

1. normalized/diagnostic/live/object/quarantine tables

## Required task-folder artifacts

In addition to the global folder contract, create:

1. repair reports per dataset
2. before_after_ranges.csv
3. quarantine_counts.csv
4. clean_view_manifest.csv
5. station_coverage.csv

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Sentinel absence in clean views
2. ISD direction variance
3. pre-cutoff summary proof
4. timezone tests
5. duplicate tests

## Acceptance criteria

1. All 13 families receive explicit completed disposition
2. Critical defects blocked from modelling

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Unrepairable field: set NULL/quarantine and record; never substitute fabricated values

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T15",
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
