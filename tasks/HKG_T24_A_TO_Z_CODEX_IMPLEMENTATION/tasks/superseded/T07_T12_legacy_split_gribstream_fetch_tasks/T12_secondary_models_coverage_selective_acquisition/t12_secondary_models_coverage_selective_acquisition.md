# T12 — Secondary Model Coverage Tests and Selective Acquisition

## Assignment

**Phase:** B Acquisition  
**Required dependencies:** T03, T06  
**Bookkeeping folder suffix:** `secondary_models_coverage_selective_acquisition`

## Mission

Probe and selectively acquire only secondary sources with a plausible HKG mechanism while formally excluding irrelevant domains.

## Why this task exists

This captures unconventional edges without wasting quota on US-only products.

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

1. Final model disposition matrix
2. T05 coordinates
3. T06 client

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Run HKO/marine coverage probes for nbmoc and nbmparoc.
2. Probe GDAS/CDAS analysis usefulness and exact variables.
3. Acquire small targeted GEFS chemistry aerosol subset for haze/radiation hypotheses.
4. Acquire targeted wave/sea-state variables only at marine points.
5. Evaluate UVI short-window collection.
6. Evaluate CFS fields only as slow background features.
7. Write explicit no-query exclusions for every US-only product.
8. Promote a source to backfill only when coverage, variable relevance, retention and quota criteria pass.

## Database/code objects that must exist or be updated

1. catalog model dispositions
2. targeted nwp values where approved

## Required task-folder artifacts

In addition to the global folder contract, create:

1. coverage_and_value_tests.csv
2. secondary_acquisition_manifests
3. excluded_models.md
4. promotion_decisions.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. HKG coordinate return tests
2. information redundancy screening without sealed outcomes
3. quota checks

## Acceptance criteria

1. Every listed secondary model has explicit acquire/probe/exclude result
2. No geographic assumption left implicit

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. No coverage or no plausible mechanism: create rejection folder; do not force inclusion

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T12",
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
