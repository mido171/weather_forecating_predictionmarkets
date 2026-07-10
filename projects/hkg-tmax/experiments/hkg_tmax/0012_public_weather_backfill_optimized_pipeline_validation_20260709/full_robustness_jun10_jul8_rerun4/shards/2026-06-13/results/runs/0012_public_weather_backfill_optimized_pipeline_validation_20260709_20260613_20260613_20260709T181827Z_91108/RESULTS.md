# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260613_20260613_20260709T181827Z_91108`
Execution mode: `optimized`
Elapsed seconds: `1.378725000016857`
Date range: `2026-06-13` to `2026-06-13`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.3787627999554388, 'normalize_seconds_total': 0.0002456999500282109, 'db_write_seconds_total': 0.016908999939914793, 'total_seconds_total': 1.3959174998453818}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.008454499969957396, 'p50_seconds': 0.0046582999639213085, 'p90_seconds': 0.0046582999639213085, 'max_seconds': 0.012250699975993484}
- `fetch`: {'count': 2, 'mean_seconds': 0.6893813999777194, 'p50_seconds': 0.6645513999974355, 'p90_seconds': 0.6645513999974355, 'max_seconds': 0.7142113999580033}
- `normalize`: {'count': 2, 'mean_seconds': 0.00012284997501410544, 'p50_seconds': 0.00012079998850822449, 'p90_seconds': 0.00012079998850822449, 'max_seconds': 0.0001248999615199864}
- `total`: {'count': 2, 'mean_seconds': 0.6979587499226909, 'p50_seconds': 0.6693345999228768, 'p90_seconds': 0.6693345999228768, 'max_seconds': 0.726582899922505}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
