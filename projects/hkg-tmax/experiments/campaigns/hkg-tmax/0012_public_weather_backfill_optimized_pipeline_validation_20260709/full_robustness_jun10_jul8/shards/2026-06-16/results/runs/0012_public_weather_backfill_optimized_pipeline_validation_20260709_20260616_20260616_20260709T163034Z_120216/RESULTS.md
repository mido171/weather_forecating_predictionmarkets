# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260616_20260616_20260709T163034Z_120216`
Execution mode: `optimized`
Elapsed seconds: `1.2864704999956302`
Date range: `2026-06-16` to `2026-06-16`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.218558699998539, 'normalize_seconds_total': 0.00027949997456744313, 'db_write_seconds_total': 0.010822299926076084, 'total_seconds_total': 1.2296604998991825}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.005411149963038042, 'p50_seconds': 0.004260399960912764, 'p90_seconds': 0.004260399960912764, 'max_seconds': 0.00656189996516332}
- `fetch`: {'count': 2, 'mean_seconds': 0.6092793499992695, 'p50_seconds': 0.5148905999958515, 'p90_seconds': 0.5148905999958515, 'max_seconds': 0.7036681000026874}
- `normalize`: {'count': 2, 'mean_seconds': 0.00013974998728372157, 'p50_seconds': 0.0001259999698959291, 'p90_seconds': 0.0001259999698959291, 'max_seconds': 0.00015350000467151403}
- `total`: {'count': 2, 'mean_seconds': 0.6148302499495912, 'p50_seconds': 0.5192769999266602, 'p90_seconds': 0.5192769999266602, 'max_seconds': 0.7103834999725223}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
