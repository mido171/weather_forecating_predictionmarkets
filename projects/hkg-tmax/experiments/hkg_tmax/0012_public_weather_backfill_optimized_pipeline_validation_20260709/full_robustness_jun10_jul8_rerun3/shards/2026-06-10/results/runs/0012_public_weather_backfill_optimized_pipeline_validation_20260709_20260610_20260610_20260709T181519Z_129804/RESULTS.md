# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260610_20260610_20260709T181519Z_129804`
Execution mode: `optimized`
Elapsed seconds: `1.4908710999879986`
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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.330431399983354, 'normalize_seconds_total': 0.0002443999983370304, 'db_write_seconds_total': 0.07564400002593175, 'total_seconds_total': 1.4063198000076227}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.03782200001296587, 'p50_seconds': 0.02234389999648556, 'p90_seconds': 0.02234389999648556, 'max_seconds': 0.053300100029446185}
- `fetch`: {'count': 2, 'mean_seconds': 0.665215699991677, 'p50_seconds': 0.655468099983409, 'p90_seconds': 0.655468099983409, 'max_seconds': 0.6749632999999449}
- `normalize`: {'count': 2, 'mean_seconds': 0.0001221999991685152, 'p50_seconds': 0.00012149999383836985, 'p90_seconds': 0.00012149999383836985, 'max_seconds': 0.00012290000449866056}
- `total`: {'count': 2, 'mean_seconds': 0.7031599000038113, 'p50_seconds': 0.677933499973733, 'p90_seconds': 0.677933499973733, 'max_seconds': 0.7283863000338897}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
