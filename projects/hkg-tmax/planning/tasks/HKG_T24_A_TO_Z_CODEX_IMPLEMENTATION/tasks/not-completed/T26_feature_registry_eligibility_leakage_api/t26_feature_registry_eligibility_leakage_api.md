# T26 — Feature Registry, Eligibility API, and Automated Leakage Gate

## Assignment

**Phase:** C Feature Platform  
**Required dependencies:** T19, T20, T21, T22, T23, T24, T25  
**Bookkeeping folder suffix:** `feature_registry_eligibility_leakage_api`

## Mission

Register every feature and enforce production/research/diagnostic eligibility automatically before any model fit.

## Why this task exists

A blacklist in prose is insufficient; the pipeline must mechanically reject leakage and blocked sources.

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

1. All feature definitions
2. availability contracts
3. hard denylist/evidence

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Create feature registry with source, formula, unit, availability, fit scope, role and status.
2. Implement feature materialization API requiring target date/cutoff/frame.
3. Implement denylist patterns for target/residual/error/outcome-derived fields.
4. Reject full-history normalization/PCA/text models outside folds.
5. Generate feature manifests and hashes for every matrix.
6. Implement model-training whitelist checks.

## Database/code objects that must exist or be updated

1. feature_store.feature_definition
2. governance.feature_eligibility_audit

## Required task-folder artifacts

In addition to the global folder contract, create:

1. feature_registry.csv
2. hard_denylist.csv
3. eligibility_validator
4. matrix_manifest_schema.json
5. audit reports

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Inject forbidden fields and prove rejection
2. blocked source test
3. fold-fit metadata test
4. hash reproducibility

## Acceptance criteria

1. No training job can run without a passing feature manifest
2. Every feature status explicit

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Unknown feature defaults to blocked, not allowed

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T26",
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
