# T35 — Full System Integration, Ablation, and Championship Tournament

## Assignment

**Phase:** E Validation  
**Required dependencies:** T33, T34  
**Bookkeeping folder suffix:** `full_system_integration_ablation_tournament`

## Mission

Assemble the complete pre-2024 system, run all ablations and select one frozen core candidate under predeclared rules.

## Why this task exists

This converts independent components into the actual end-to-end point forecast.

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

1. All OOF experts/router/specialists/distribution
2. baseline ladder
3. promotion rules

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Implement deterministic inference graph from snapshot to final P50.
2. Compare official raw, residual memory, individual MOS, static blend, dynamic router, router+specialists and distributional P50.
3. Run component/family/source ablations.
4. Run negative controls and shuffled-date tests.
5. Report full/slice/tail metrics and complexity cost.
6. Select the simplest candidate meeting promotion gates.
7. Freeze code/config/features/artifact hashes before opening confirmation.

## Database/code objects that must exist or be updated

1. research system registry
2. frozen candidate manifest

## Required task-folder artifacts

In addition to the global folder contract, create:

1. system_scoreboard.csv
2. ablation_matrix.csv
3. slice_metrics.csv
4. tail_metrics.csv
5. negative_controls.csv
6. FROZEN_CANDIDATE_MANIFEST.json
7. model_card.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. End-to-end deterministic replay
2. all source lineage
3. identical rows
4. denylist
5. sealing check

## Acceptance criteria

1. Exactly one frozen core candidate or explicit no-promotion result
2. No sealed labels accessed

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. No candidate beats baselines robustly: freeze best baseline and return to research loop

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T35",
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
