# T16 — Historical Availability Proof and Eligibility Ledger

## Assignment

**Phase:** B Acquisition  
**Required dependencies:** T01, T03, T07, T08, T09, T10, T11, T13, T14, T15  
**Bookkeeping folder suffix:** `historical_availability_proof_ledger`

## Mission

Build authoritative source/model release-latency contracts and row/run-level eligibility at the 15:00 HKT cutoff.

## Why this task exists

GribStream run time and archive presence do not prove historical availability; this task is the gate between diagnostic and strict deployable evidence.

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

1. Provider dissemination schedules
2. GribStream ToS/docs
3. prospective first-seen logs
4. official HKO timestamps
5. all source metadata

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Snapshot official NCEP/ECMWF/HKO release schedule evidence with hashes.
2. For each source/model/cycle/field group, define release rule and conservative buffer.
3. Use prospective first-seen distributions to audit/update buffers without backdating proof.
4. Grade each run A/B/C/D/E.
5. Implement row/run eligibility views at target cutoff.
6. For historical GribStream runs, permit strict use only when B proof is approved; otherwise diagnostic.
7. Record schedule/model-version eras and exceptions.
8. Create an audit explaining which cycles are normally eligible at 07:00 UTC without relying on assumption.

## Database/code objects that must exist or be updated

1. governance.availability_contract
2. nwp_core run availability grade
3. eligible model-run views

## Required task-folder artifacts

In addition to the global folder contract, create:

1. provider_evidence_archive/
2. availability_contracts.csv
3. eligible_cycle_matrix.csv
4. strict_vs_diagnostic_counts.csv
5. latency_buffer_rationale.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Boundary-time tests
2. late-run exclusion
3. schedule-era tests
4. prospective first-seen comparison

## Acceptance criteria

1. Every source/run has explicit grade
2. Strict views contain only A/B rows
3. No asOf-only record promoted

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Cannot prove historical release: retain C diagnostic and continue prospective collection

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T16",
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
