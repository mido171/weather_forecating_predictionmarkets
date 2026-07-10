# 0009 Lean DB-backed public-weather smoke

Status: `complete` for the recorded one-day run on 2026-07-07.

## Hypothesis and protocol

Stream GFS, GEFS control, Himawari B13/S0510, and radar metadata/features into
Postgres with conservative availability clocks, then delete transient raw
payloads only after successful normalization and DB commit.

## Recorded result

| Metric | Value |
|---|---:|
| Runtime | 49.41 s |
| Fetch/normalize successes | 4/4 |
| Station features upserted | 224 |
| Area features upserted | 412 |
| Raw bytes deleted | 24,963,547 |
| Source issues touched | 4 |

The directory name describes a broader intended window, but the retained root
metrics prove only 2026-07-07. Do not infer a Jun-25-to-Jul-7 completion.

## Decision

Acquisition/persistence smoke passed. This was not a model-promotion
experiment and was superseded operationally by experiment 0012.

## Safe command shape

```powershell
$env:HKG_TMAX_DATABASE_URL = '<local PostgreSQL URL>'
.\.venv\Scripts\python.exe scripts\backfill_public_weather_to_postgres.py --start-date 2026-07-07 --end-date 2026-07-07 --sources gfs,gefs_control,himawari_b13_s0510,radar --max-static-tasks 100 --max-radar-frames 24 --execute
```

Review provider/database budgets before `--execute`.

## Evidence map

`STATUS.yaml`, `RUN_CONFIG.yaml`, `DATA_MANIFEST.yaml`, and
`results/metrics.json` are the compact source of truth.
