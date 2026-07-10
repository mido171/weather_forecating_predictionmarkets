# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260610_20260610_20260709T181403Z_125068`
Execution mode: `optimized`
Elapsed seconds: `1.4751729000126943`
Date range: `2026-06-10` to `2026-06-10`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.2716990999761038, 'normalize_seconds_total': 0.00022950000129640102, 'db_write_seconds_total': 0.019387399952393025, 'total_seconds_total': 1.2913159999297932}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.009693699976196513, 'p50_seconds': 0.006528099998831749, 'p90_seconds': 0.006528099998831749, 'max_seconds': 0.012859299953561276}
- `fetch`: {'count': 2, 'mean_seconds': 0.6358495499880519, 'p50_seconds': 0.6028445999836549, 'p90_seconds': 0.6028445999836549, 'max_seconds': 0.6688544999924488}
- `normalize`: {'count': 2, 'mean_seconds': 0.00011475000064820051, 'p50_seconds': 0.0001145999995060265, 'p90_seconds': 0.0001145999995060265, 'max_seconds': 0.00011490000179037452}
- `total`: {'count': 2, 'mean_seconds': 0.6456579999648966, 'p50_seconds': 0.6158187999390066, 'p90_seconds': 0.6158187999390066, 'max_seconds': 0.6754971999907866}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
