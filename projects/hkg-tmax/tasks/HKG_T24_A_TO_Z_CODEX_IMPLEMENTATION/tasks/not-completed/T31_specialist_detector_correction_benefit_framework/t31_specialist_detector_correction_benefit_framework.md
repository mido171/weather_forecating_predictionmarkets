# T31 — Specialist Detector, Correction, and Benefit-Gate Framework

## Assignment

**Phase:** D Modelling  
**Required dependencies:** T27, T28, T29  
**Bookkeeping folder suffix:** `specialist_detector_correction_benefit_framework`

## Mission

Implement and evaluate the first complete set of regime specialists with learned activation, bounded correction and abstention.

## Why this task exists

Specialists can reduce known failure tails only when their benefit is proven rather than manually assumed.

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

1. SPECIALIST_TRAINING_SPECIFICATION.md
2. OOF anchor/expert predictions
3. official/NWP/station/target features

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Implement reusable three-model specialist framework.
2. Pre-register marine suppression, weak-wind heat, MAM transition, cloud/rain suppression, cool-surge rebound, dry subsidence, TC-peripheral proxy, hot/cold asymmetry and high-error tail hypotheses.
3. Use official-only long history where appropriate and modern NWP frame where required.
4. Generate cross-fitted regime probabilities, corrections and expected benefit.
5. Tune thresholds/caps inside inner folds.
6. Apply sample support and no-harm gates.
7. Preserve negative specialists and failure reasons.

## Database/code objects that must exist or be updated

1. research specialist registry/OOF outputs

## Required task-folder artifacts

In addition to the global folder contract, create:

1. specialist_definitions.csv
2. specialist_oof_predictions.parquet
3. activation_metrics.csv
4. benefit_metrics.csv
5. tail_metrics.csv
6. promotion_decisions.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Detector/correction/gate cross-fitting
2. minimum sample tests
3. outside-slice harm
4. sign stability

## Acceptance criteria

1. Only specialists meeting all evidence gates promoted
2. No hard-coded unconditional correction

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Sparse/unstable specialist rejected or shrunk to zero

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T31",
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
