# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260621_20260621_20260709T163437Z_117868`
Execution mode: `optimized`
Elapsed seconds: `1.3658241999801248`
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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.5038202999858186, 'normalize_seconds_total': 0.0002471000188961625, 'db_write_seconds_total': 0.015158900001551956, 'total_seconds_total': 1.5192263000062667}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.007579450000775978, 'p50_seconds': 0.0030578000005334616, 'p90_seconds': 0.0030578000005334616, 'max_seconds': 0.012101100001018494}
- `fetch`: {'count': 2, 'mean_seconds': 0.7519101499929093, 'p50_seconds': 0.6906628999859095, 'p90_seconds': 0.6906628999859095, 'max_seconds': 0.8131573999999091}
- `normalize`: {'count': 2, 'mean_seconds': 0.00012355000944808125, 'p50_seconds': 0.0001233000075444579, 'p90_seconds': 0.0001233000075444579, 'max_seconds': 0.0001238000113517046}
- `total`: {'count': 2, 'mean_seconds': 0.7596131500031333, 'p50_seconds': 0.7028877999982797, 'p90_seconds': 0.7028877999982797, 'max_seconds': 0.816338500007987}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
