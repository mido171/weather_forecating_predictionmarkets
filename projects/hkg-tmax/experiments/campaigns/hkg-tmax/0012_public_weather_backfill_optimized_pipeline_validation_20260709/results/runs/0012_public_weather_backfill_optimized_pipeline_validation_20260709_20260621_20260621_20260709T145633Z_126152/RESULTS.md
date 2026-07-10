# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260621_20260621_20260709T145633Z_126152`
Execution mode: `optimized`
Elapsed seconds: `1.6477907000225969`
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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.3800728999776766, 'normalize_seconds_total': 0.00023310002870857716, 'db_write_seconds_total': 0.1028025999548845, 'total_seconds_total': 1.4831085999612696}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.05140129997744225, 'p50_seconds': 0.0038257999694906175, 'p90_seconds': 0.0038257999694906175, 'max_seconds': 0.09897679998539388}
- `fetch`: {'count': 2, 'mean_seconds': 0.6900364499888383, 'p50_seconds': 0.6402507000020705, 'p90_seconds': 0.6402507000020705, 'max_seconds': 0.739822199975606}
- `normalize`: {'count': 2, 'mean_seconds': 0.00011655001435428858, 'p50_seconds': 0.0001078000059351325, 'p90_seconds': 0.0001078000059351325, 'max_seconds': 0.00012530002277344465}
- `total`: {'count': 2, 'mean_seconds': 0.7415542999806348, 'p50_seconds': 0.7393528000102378, 'p90_seconds': 0.7393528000102378, 'max_seconds': 0.7437557999510318}

## Resource Telemetry

- `{'cpu_sampler_available': False, 'cpu_mean_percent': None, 'cpu_max_percent': None, 'staging_max_bytes': 0, 'staging_end_bytes': 0}`

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
