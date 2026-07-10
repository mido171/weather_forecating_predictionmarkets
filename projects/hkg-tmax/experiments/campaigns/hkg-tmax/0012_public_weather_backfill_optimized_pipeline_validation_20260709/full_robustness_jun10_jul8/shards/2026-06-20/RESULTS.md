# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260620_20260620_20260709T163436Z_119028`
Execution mode: `optimized`
Elapsed seconds: `1.4156850000144914`
Date range: `2026-06-20` to `2026-06-20`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.4478315000305884, 'normalize_seconds_total': 0.000235399988014251, 'db_write_seconds_total': 0.01605009997729212, 'total_seconds_total': 1.4641169999958947}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.00802504998864606, 'p50_seconds': 0.0037311000050976872, 'p90_seconds': 0.0037311000050976872, 'max_seconds': 0.012318999972194433}
- `fetch`: {'count': 2, 'mean_seconds': 0.7239157500152942, 'p50_seconds': 0.7040968000073917, 'p90_seconds': 0.7040968000073917, 'max_seconds': 0.7437347000231966}
- `normalize`: {'count': 2, 'mean_seconds': 0.0001176999940071255, 'p50_seconds': 0.00011660001473501325, 'p90_seconds': 0.00011660001473501325, 'max_seconds': 0.00011879997327923775}
- `total`: {'count': 2, 'mean_seconds': 0.7320584999979474, 'p50_seconds': 0.7165345999528654, 'p90_seconds': 0.7165345999528654, 'max_seconds': 0.7475824000430293}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
