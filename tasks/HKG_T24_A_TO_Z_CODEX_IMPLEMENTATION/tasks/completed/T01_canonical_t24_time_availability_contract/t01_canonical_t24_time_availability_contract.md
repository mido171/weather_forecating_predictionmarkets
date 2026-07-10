# T01 — Canonical T−24 Time and Availability Contract

## Assignment

**Phase:** A Foundation  
**Required dependencies:** T00  
**Bookkeeping folder suffix:** `canonical_t24_time_availability_contract`

## Mission

Implement the single operational cutoff function, timestamp normalization policy, availability-proof grades, sealed-period controls, and eligibility predicate used by every subsequent task.

## Why this task exists

Without one enforced point-in-time contract, all model gains are untrustworthy.

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

1. T00 inventories
2. operational_contract.yaml
3. T24_POINT_IN_TIME_CONSTITUTION.md

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Implement governance.hkg_t24_cutoff_utc(target_date) or repository equivalent.
2. Store canonical UTC timestamptz and derive HKT explicitly.
3. Implement availability grades A–E and an eligibility function that requires available_at <= cutoff.
4. Create sealed-period metadata and development/live roles.
5. Implement a conservative target-label availability contract; forbid casual T−1 daily Tmax use.
6. Add database constraints/views and application helpers.
7. Document conflict resolution if an existing cutoff differs.

## Database/code objects that must exist or be updated

1. governance.operational_contract
2. governance.availability_grade
3. governance.sealed_period
4. cutoff/eligibility functions

## Required task-folder artifacts

In addition to the global folder contract, create:

1. migration files
2. time_contract.md
3. availability_grade_contract.md
4. role_permissions.sql
5. test evidence

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Leap day and year-boundary tests
2. UTC/HKT conversion tests
3. no-DST tests
4. sealed-role access tests
5. eligible/ineligible row tests

## Acceptance criteria

1. One canonical cutoff version used everywhere
2. Live role cannot read labels/residuals/sealed outcomes
3. Rows without proof cannot appear in strict views

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Existing incompatible contract: preserve it, document, and require owner decision; never silently switch

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T01",
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
