# Backfill tool rules

The root `AGENTS.md` applies. Backfills are external-cost and data-mutation work,
never startup checks.

- Default to a plan/dry run that performs zero provider calls and writes no DB
  rows.
- Require `--execute`, explicit provider/station/model/date scope, and hard
  request, byte, row, retry, runtime, and worker budgets.
- Default workers to one; maximum two without explicit user approval.
- Persist raw payloads before normalization, use idempotent resume keys, and
  record a run ledger/manifests under the configured external data/run root.
- Stop on authentication errors, repeated rate limits, quota/budget exhaustion,
  or free-space thresholds.
- Never auto-install dependencies, schedules, services, or database schemas.
- Add focused mocked tests for planning, budget rejection, retry/stop behavior,
  and resume/idempotency.
