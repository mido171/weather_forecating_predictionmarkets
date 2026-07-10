# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260617_20260617_20260709T163034Z_25432`
Execution mode: `optimized`
Elapsed seconds: `1.3057448000181466`
Date range: `2026-06-17` to `2026-06-17`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.2750082000275142, 'normalize_seconds_total': 0.00024090002989396453, 'db_write_seconds_total': 0.014708199945744127, 'total_seconds_total': 1.2899573000031523}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.0073540999728720635, 'p50_seconds': 0.006906899972818792, 'p90_seconds': 0.006906899972818792, 'max_seconds': 0.007801299972925335}
- `fetch`: {'count': 2, 'mean_seconds': 0.6375041000137571, 'p50_seconds': 0.604826100054197, 'p90_seconds': 0.604826100054197, 'max_seconds': 0.6701820999733172}
- `normalize`: {'count': 2, 'mean_seconds': 0.00012045001494698226, 'p50_seconds': 0.00011640001321211457, 'p90_seconds': 0.00011640001321211457, 'max_seconds': 0.00012450001668184996}
- `total`: {'count': 2, 'mean_seconds': 0.6449786500015762, 'p50_seconds': 0.6118575000436977, 'p90_seconds': 0.6118575000436977, 'max_seconds': 0.6780997999594547}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
