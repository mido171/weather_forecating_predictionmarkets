# 0010 Wide-window lean DB backfill test

Status: `incomplete_superseded`.

## Purpose

Exercise day-sharded DB-backed GFS, GEFS control, and Himawari ingestion with
immediate raw cleanup before the optimized pipeline was finalized.

## Consolidated run ledger

Twelve shards from 2026-06-10 through 2026-06-21 completed with source
failures. Aggregate retained metrics:

| Metric | Value |
|---|---:|
| Completed days | 12 |
| Fetch/normalize successes | 4,772 |
| Fetch failures | 28 |
| Station features upserted | 212,793 |
| Area features upserted | 372,599 |
| Raw bytes deleted | 21,503,381,461 |
| Sum of shard runtimes | 35,288.1 s |
| Maximum sampled staging | 23,291,429 bytes |

Shards 2026-06-22 through 2026-06-25 retain stale `RUNNING` statuses with no
final metrics. No evidence proves completion through Jul 8.

## Decision

Do not resume or cite 0010 as complete. Experiment 0012 corrected the
orchestration, completed the validation window, recovered transient GFS
misses, and is the accepted evidence.

## Evidence map

Each shard retains `STATUS.yaml`, `RUN_CONFIG.yaml`, `DATA_MANIFEST.yaml`, and
`results/metrics.json`. The 129 former Markdown reports were redundant
renderings and are recoverable through the provenance ledger.
