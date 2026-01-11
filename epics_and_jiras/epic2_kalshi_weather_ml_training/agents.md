# agents.md (Epic #2) — ML Training Handoff Rules

## Scope
This module implements ML training and evaluation for Epic #2.

## Non-negotiables
1) Read and follow `docs/data-contract.md` and `docs/time-semantics.md` (Epic #1 doc).
2) No leakage:
   - any record where chosen_runtime_utc > asof_utc must fail the run.
3) Deterministic:
   - set and store global random seed.
4) Reports are required:
   - metrics.json (machine)
   - report.md (human)
5) Use time-based splits (no random shuffle) — TimeSeriesSplit or explicit date cutoffs.

## Persistence safety
- joblib is pickle-based; only load trusted artifacts.
- store hashes and versions.

