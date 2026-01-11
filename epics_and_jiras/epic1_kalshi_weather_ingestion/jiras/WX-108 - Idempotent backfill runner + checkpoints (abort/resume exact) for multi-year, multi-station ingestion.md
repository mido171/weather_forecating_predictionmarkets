# WX-108 — Idempotent backfill runner + checkpoints (abort/resume exact) for multi-year, multi-station ingestion

## Objective
Implement a robust backfill orchestration mechanism that:
- can be terminated at any time
- can be restarted and continue exactly from last checkpoint
- guarantees no duplicates and no gaps

## Jobs to support (Epic #1)
- `kalshi_series_sync`
- `cli_ingest_year`
- `mos_ingest_window`
- `mos_asof_materialize_range`

## Checkpointing strategy
Use `ingest_checkpoint` table keyed by:
- job_name
- station_id
- model (nullable)

For each job:
- update checkpoint after each successful unit of work (year, day window, or target day)
- store:
  - last completed date
  - last completed runtime window (if relevant)
  - status
  - updated_at_utc

## Restart semantics (must be exact)
- If job restarts and checkpoint exists:
  - resume from the next unprocessed unit
- If partial unit was in-flight:
  - safe to repeat it (idempotent upserts ensure no duplicates)

## Acceptance Criteria
- [ ] Demonstrate abort/resume:
  - run a backfill for 365 days
  - kill at ~30%
  - restart and complete
  - row counts match a clean one-shot run
- [ ] Each job has clear “unit of work” granularity documented.
- [ ] Checkpoints are updated at least once per minute during backfill.
- [ ] Failures mark status FAILED and include error details.

