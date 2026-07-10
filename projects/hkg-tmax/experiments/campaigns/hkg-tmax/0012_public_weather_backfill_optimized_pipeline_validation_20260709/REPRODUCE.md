# Reproduce

Dry-run inventory:

```powershell
$env:HKG_TMAX_DATABASE_URL = '<local postgres url>'
.\.venv\Scripts\python.exe .\scripts\backfill_public_weather_to_postgres.py `
  --experiment-id 0012_public_weather_backfill_optimized_pipeline_validation_20260709 `
  --experiment-dir .\experiments\hkg_tmax\0012_public_weather_backfill_optimized_pipeline_validation_20260709 `
  --start-date 2026-06-21 --end-date 2026-06-21 `
  --sources gfs,gefs_control,himawari_b13_s0510 `
  --execution-mode optimized --dry-run
```

Corrected full robustness run:

```powershell
$env:HKG_TMAX_DATABASE_URL = '<local postgres url>'
.\.venv\Scripts\python.exe .\scripts\run_public_weather_backfill_day_shards.py `
  --experiment-id 0012_public_weather_backfill_optimized_pipeline_validation_20260709 `
  --experiment-dir .\experiments\hkg_tmax\0012_public_weather_backfill_optimized_pipeline_validation_20260709\full_robustness_jun10_jul8_rerun4 `
  --start-date 2026-06-10 --end-date 2026-07-08 `
  --sources gfs,gefs_control,himawari_b13_s0510 `
  --execution-mode optimized `
  --model-fetch-workers 8 --model-range-workers 4 --model-normalize-workers 2 `
  --himawari-workers 8 --model-range-coalesce-gap-bytes 0 `
  --max-workers 2 --max-staging-gb 1 --stop-free-gb 50 `
  --progress-every 50 --monitor-interval-seconds 60 --max-attempts 5
```

Targeted GFS retry used after the full run exposed six transient disconnects:

```powershell
$env:HKG_TMAX_DATABASE_URL = '<local postgres url>'
.\.venv\Scripts\python.exe .\scripts\backfill_public_weather_to_postgres.py `
  --experiment-id 0012_public_weather_backfill_optimized_pipeline_validation_20260709 `
  --experiment-dir .\experiments\hkg_tmax\0012_public_weather_backfill_optimized_pipeline_validation_20260709\targeted_gfs_retry_after_full_rerun4 `
  --start-date 2026-06-27 --end-date 2026-07-07 `
  --sources gfs --execution-mode optimized `
  --model-fetch-workers 4 --model-range-workers 4 --model-normalize-workers 2 `
  --model-range-coalesce-gap-bytes 0 `
  --max-staging-gb 1 --stop-free-gb 50 `
  --progress-every 5 --max-attempts 7 --skip-existing-complete
```
