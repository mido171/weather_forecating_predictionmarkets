# T03 — GribStream Catalog, Coverage, Licence, and Quota Audit

## Assignment

**Phase:** A Foundation  
**Required dependencies:** T00, T01  
**Bookkeeping folder suffix:** `gribstream_catalog_coverage_licence_quota_audit`

## Mission

Snapshot the live GribStream catalog/OpenAPI/client metadata, verify every listed model, test geographic coverage, resolve selectors, estimate quota/storage, and document contractual constraints before bulk acquisition.

## Why this task exists

The API catalog changes and many listed models do not cover Hong Kong. Acquiring blindly wastes quota and can create false coverage assumptions.

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

1. GRIBSTREAM_MODEL_DISPOSITION_MATRIX.csv
2. SEMANTIC_VARIABLE_REQUIREMENTS.csv
3. GribStream public catalog/docs/terms
4. API key if available
5. HKO/reference coordinates from T05 when ready

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Fetch and hash current docs, OpenAPI YAML, model catalog and selector metadata.
2. Enumerate all current models, including user-listed and newly discovered entries.
3. For each model, record domain, resolution, cycles, lead range, ensemble members, archive/retention and selector introduction dates.
4. Run minimal coordinate probes at HKO, inland, coastal and marine points for coverage-probe models.
5. Resolve exact selector tuples for every P0/P1 semantic variable.
6. Estimate request rows, bytes, quota and storage by acquisition tier.
7. Record GribStream ToS constraints and request written clarification for long-term internal commercial retention if needed.
8. Record upstream attribution/licensing requirements, especially ECMWF CC BY 4.0.
9. Produce final acquire/probe/exclude decisions; no bulk download yet.

## Database/code objects that must exist or be updated

1. catalog.catalog_snapshot
2. catalog.model_registry
3. catalog.selector_snapshot
4. catalog.source_license

## Required task-folder artifacts

In addition to the global folder contract, create:

1. catalog_snapshot.json
2. selector_map.csv
3. coverage_probe_results.csv
4. quota_storage_estimate.csv
5. licence_register.md
6. final_model_disposition.csv

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Catalog hash repeatability
2. selector query smoke tests
3. geographic coverage assertions
4. no secret in logs

## Acceptance criteria

1. Every user-listed model has an explicit disposition
2. Every core variable has exact selectors or a blocker
3. Bulk acquisition plan fits quota/storage or is staged

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. No API key: complete public catalog work and emit authenticated probe command
2. Unclear retention rights: do not block code, but block bulk commercial ingestion pending written confirmation

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T03",
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
