# T02 — Full Current Data and Experiment Census Reconciliation

## Assignment

**Phase:** A Foundation  
**Required dependencies:** T00, T01  
**Bookkeeping folder suffix:** `full_current_data_census_reconciliation`

## Mission

Reconcile every existing dataset, database table, file, attribute, station, experiment output and quality issue into a machine-readable source registry.

## Why this task exists

The system must use all valuable data while keeping diagnostic, operational, live-only, research and quarantine roles distinct.

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

1. 13-dataset audit evidence
2. 52 table decisions
3. 1,869 attribute decisions
4. 36-station dossier
5. quality issues
6. current DB inventory
7. experiments evidence registry

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Read every supplied evidence file completely.
2. Reconcile audit records against actual files and database objects.
3. Update the official forecast disposition using the corrected near-continuous archive.
4. Assign each source/table/attribute to label, operational, diagnostic, live, object, research or quarantine layer.
5. Record date coverage, cadence, stations, variables, unit, timestamp fields, availability proof and blocker.
6. Preserve full free text from source tables, not profile examples.
7. Register all experiment outputs as evidence, never as canonical live feature sources.

## Database/code objects that must exist or be updated

1. catalog.dataset_registry
2. catalog.source_registry
3. governance.attribute_contract
4. governance.quality_issue

## Required task-folder artifacts

In addition to the global folder contract, create:

1. source_eligibility_matrix.csv
2. table_reconciliation.csv
3. attribute_reconciliation.csv
4. station_reconciliation.csv
5. experiment_evidence_linkage.csv
6. updated_quality_blockers.csv

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Expected count reconciliation
2. unmapped-object zero check
3. duplicate physical representation check

## Acceptance criteria

1. Every actual source has exactly one disposition
2. Every attribute has a contract
3. No source silently omitted

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Unprofiled/new source: register as UNREVIEWED and block production promotion until audited

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T02",
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
