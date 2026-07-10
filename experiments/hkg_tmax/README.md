# HKG Tmax Experiment Workspace

This folder is the organized experiment namespace for HKG Tmax research runs.

Canonical implementation code stays outside this tree:

- Source modules: `code/src/hkg_tmax/`
- CLI runners: `scripts/`
- Configs: `configs/hkg_tmax/`
- Tests: `code/tests/`

Each numbered experiment folder contains or points to:

- `README.md`: purpose, status, command, and artifact map.
- `inputs/`: external memos/specs or immutable input references.
- `results/`: generated scoreboards, audits, predictions, and model cards.
- `run_logs/`: console logs or command transcripts when retained.

Sequential experiments:

| Experiment | Purpose | Status |
|---|---|---|
| `0001_residual_ml_strategy_20260705` | Prior broad residual-ML ladder A0-A8 and A7 baseline | completed |
| `0002_selective_no_harm_router_20260705` | GPT-Pro next round: pruned features, selective router, tail overlay, anchor provenance | completed, no promote |
| `0003_official_residual_memory_20260706` | GPT-Pro official residual-memory D0-D5 point-forecast benchmark | completed, no promote |
| `0004_station_hour_residual_information_atlas_20260708` | Postgres Info.gov hourly station/HKO residual-signal atlas at T-1 23:59 HKT | completed, information gain positive, no promote |
| `0005_public_gfs_gefs_himawari_fetch_smoke_20260708` | Public latest-issued GFS, GEFS, and Himawari-9 fetch smoke without GribStream | completed, fetch smoke pass |

The older flat folder `experiments/hkg_tmax_residual_ml_strategy/` is preserved for compatibility with the original run. New HKG Tmax runs should be registered here first.
