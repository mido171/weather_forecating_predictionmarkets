# T39 — Final Audit, Handoff, and Controlled Research Continuation Loop

## Assignment

**Phase:** F Production  
**Required dependencies:** T36, T37  
**Bookkeeping folder suffix:** `final_system_audit_handoff_research_loop`

## Mission

Produce the complete implementation and results dossier, archive all artifacts, and define the next research loop if 0.45°C has not been achieved.

## Why this task exists

The project needs a trustworthy research machine, not a one-off opaque score.

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

1. All task folders and manifests
2. confirmation results
3. live pipeline status

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Validate every task acceptance criterion and dependency.
2. Generate final data/model/feature/source cards.
3. Summarize proven, proxy, diagnostic, rejected and pending findings.
4. Report current best MAE by frame without cross-frame confusion.
5. Archive hashes, migrations, configs, models and commands.
6. If confirmed MAE > 0.45, invoke Research Director to propose a new pre-registered lane using accumulated evidence, while preserving remaining holdouts.
7. Never claim inevitable convergence to 0.45.

## Database/code objects that must exist or be updated

1. research master registry and archive

## Required task-folder artifacts

In addition to the global folder contract, create:

1. FINAL_IMPLEMENTATION_REPORT.md
2. FINAL_SCOREBOARD.csv
3. PROVENANCE_INDEX.csv
4. OPEN_BLOCKERS.csv
5. NEXT_RESEARCH_BRIEF.md
6. HANDOFF_RUNBOOK.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Bundle hash verification
2. all task acceptance aggregation
3. reproduction from clean environment

## Acceptance criteria

1. Complete truthful A-to-Z handoff
2. Best score and uncertainty stated on exact frame
3. Next loop has no holdout leakage

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Any incomplete task remains explicit blocker; final report cannot mark system complete

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T39",
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
