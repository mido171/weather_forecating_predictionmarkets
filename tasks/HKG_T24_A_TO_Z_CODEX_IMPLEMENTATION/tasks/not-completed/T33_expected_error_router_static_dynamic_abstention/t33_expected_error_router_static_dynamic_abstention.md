# T33 — Expected-Error Router with Static Priors, Dynamic Weights, and Abstention

## Assignment

**Phase:** D Modelling  
**Required dependencies:** T32  
**Bookkeeping folder suffix:** `expected_error_router_static_dynamic_abstention`

## Mission

Train the concrete facts-on-the-ground → predicted expert error → weight distribution router and prove it beats static/simple blends.

## Why this task exists

This task implements the strategy requested by the owner: router decisions must arise from training, not narrative judgment.

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

1. ROUTER_TRAINING_SPECIFICATION.md
2. OOF meta-table
3. context features
4. expert availability masks

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Fit non-negative static MAE-optimal blend on inner OOF data.
2. Fit one regularized expected-absolute-error model per expert using context only.
3. Calibrate predicted errors.
4. Convert predicted errors to softmax dynamic weights.
5. Tune temperature and static/dynamic shrinkage lambda inside temporal folds.
6. Apply per-expert weight caps, minimum history and missing-source masks.
7. Train expected-benefit abstention versus stable blend.
8. Add recent causal performance states.
9. Evaluate weight turnover, context calibration and expert win rates.
10. Create capped challenger adapter interface.

## Database/code objects that must exist or be updated

1. research router artifacts/OOF decisions

## Required task-folder artifacts

In addition to the global folder contract, create:

1. router_oof_predictions.parquet
2. router_weights.parquet
3. predicted_vs_realized_error.csv
4. static_vs_dynamic_scoreboard.csv
5. abstention_metrics.csv
6. router_model_card.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. OOF-only meta-training
2. weights nonnegative/sum one
3. missing expert zero weight
4. no outcome context
5. fold-tuned parameters

## Acceptance criteria

1. Dynamic router improves or is rejected in favor of static blend
2. Abstention prevents low-confidence damage
3. All decisions explainable

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. No stable dynamic lift: deploy static blend; preserve router as negative result

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T33",
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
