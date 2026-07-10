# T38 — Probability and Market-Threshold Interface with No-Trade Gate

## Assignment

**Phase:** F Production  
**Required dependencies:** T34, T37  
**Bookkeeping folder suffix:** `probability_market_interface_no_trade_gate`

## Mission

Expose calibrated Tmax probabilities and a conservative decision interface without altering the forecasting model or leaking market outcomes.

## Why this task exists

Point MAE is central, but trading requires calibrated threshold probabilities and abstention.

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

1. Live distribution forecast
2. market contract definitions supplied externally
3. fee/slippage policy

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Map continuous calibrated distribution to exact settlement buckets/thresholds.
2. Record market snapshot time separately from weather cutoff.
3. Calculate implied probability, model probability, uncertainty and margin of safety.
4. Implement no-trade when calibration, liquidity, spread, source freshness or edge thresholds fail.
5. Do not use future market prices or settlement to train weather features.
6. Backtest only after forecasting confirmation and with exact historical price vintages.

## Database/code objects that must exist or be updated

1. live probability outputs and decision logs

## Required task-folder artifacts

In addition to the global folder contract, create:

1. probability_api_schema.json
2. threshold_mapping_tests.csv
3. no_trade_rules.md
4. market_interface_runbook.md

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Bucket probabilities sum to one
2. settlement rule mapping
3. no future price
4. calibration drift

## Acceptance criteria

1. Interface accurately maps forecast distribution
2. No automatic trade execution unless separately authorized

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Missing exact market history: do not fabricate trading backtest

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T38",
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
