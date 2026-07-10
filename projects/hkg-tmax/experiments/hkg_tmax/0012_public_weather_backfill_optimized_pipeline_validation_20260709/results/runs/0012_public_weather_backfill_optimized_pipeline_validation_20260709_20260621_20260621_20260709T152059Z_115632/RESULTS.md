# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260621_20260621_20260709T152059Z_115632`
Execution mode: `optimized`
Elapsed seconds: `1.537017399969045`
Date range: `2026-06-21` to `2026-06-21`

## Counts

- Source issues touched: `2`
- Fetch ok: `None`
- Fetch failed: `2`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `None`
- Max staging bytes observed: `None`
- Final staging bytes: `0`
- Max raw object bytes observed: `None`
- Minimum free disk bytes observed: `None`

## By Source

- `gefs_control`: {'skipped_existing': 68}
- `gfs`: {'skipped_existing': 68}
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.5442538000643253, 'normalize_seconds_total': 0.0002989000058732927, 'db_write_seconds_total': 0.043903900019358844, 'total_seconds_total': 1.5884566000895575}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.021951950009679422, 'p50_seconds': 0.004070600029081106, 'p90_seconds': 0.004070600029081106, 'max_seconds': 0.03983329999027774}
- `fetch`: {'count': 2, 'mean_seconds': 0.7721269000321627, 'p50_seconds': 0.7666661000112072, 'p90_seconds': 0.7666661000112072, 'max_seconds': 0.7775877000531182}
- `normalize`: {'count': 2, 'mean_seconds': 0.00014945000293664634, 'p50_seconds': 0.00014479999663308263, 'p90_seconds': 0.00014479999663308263, 'max_seconds': 0.00015410000924021006}
- `total`: {'count': 2, 'mean_seconds': 0.7942283000447787, 'p50_seconds': 0.7708815000369214, 'p90_seconds': 0.7708815000369214, 'max_seconds': 0.8175751000526361}

## Resource Telemetry

- `{'cpu_sampler_available': False, 'cpu_mean_percent': None, 'cpu_max_percent': None, 'staging_max_bytes': 0, 'staging_end_bytes': 0}`

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
