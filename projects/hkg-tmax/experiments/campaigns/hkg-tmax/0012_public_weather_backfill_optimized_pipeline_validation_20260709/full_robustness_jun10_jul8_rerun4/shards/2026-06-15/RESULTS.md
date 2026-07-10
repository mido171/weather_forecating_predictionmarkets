# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260615_20260615_20260709T182028Z_126160`
Execution mode: `optimized`
Elapsed seconds: `2.0923543000244536`
Date range: `2026-06-15` to `2026-06-15`

## Counts

- Source issues touched: `3`
- Fetch ok: `None`
- Fetch failed: `3`
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
- `himawari9_b13_s0510`: {'skipped_existing': 141, 'source_issues_touched': 3, 'fetch_failed': 3, 'fetch_seconds_total': 2.4382318999851122, 'normalize_seconds_total': 0.00046419998398050666, 'db_write_seconds_total': 0.024744800000917166, 'total_seconds_total': 2.46344089997001}

## Phase Runtime

- `db_write`: {'count': 3, 'mean_seconds': 0.00824826666697239, 'p50_seconds': 0.0068026999942958355, 'p90_seconds': 0.0068026999942958355, 'max_seconds': 0.013271500007249415}
- `fetch`: {'count': 3, 'mean_seconds': 0.8127439666617041, 'p50_seconds': 0.6815594999934547, 'p90_seconds': 0.6815594999934547, 'max_seconds': 1.1295877000084147}
- `normalize`: {'count': 3, 'mean_seconds': 0.00015473332799350223, 'p50_seconds': 0.00012199999764561653, 'p90_seconds': 0.00012199999764561653, 'max_seconds': 0.00022329995408654213}
- `total`: {'count': 3, 'mean_seconds': 0.82114696665667, 'p50_seconds': 0.6864533999469131, 'p90_seconds': 0.6864533999469131, 'max_seconds': 1.1429781000479124}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
