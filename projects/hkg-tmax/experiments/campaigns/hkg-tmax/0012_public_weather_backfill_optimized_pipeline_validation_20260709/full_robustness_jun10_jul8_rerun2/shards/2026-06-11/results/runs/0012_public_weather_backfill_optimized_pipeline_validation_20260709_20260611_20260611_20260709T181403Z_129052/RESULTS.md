# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260611_20260611_20260709T181403Z_129052`
Execution mode: `optimized`
Elapsed seconds: `1.4737138000200503`
Date range: `2026-06-11` to `2026-06-11`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.5528453000006266, 'normalize_seconds_total': 0.0002470999606885016, 'db_write_seconds_total': 0.054013999935705215, 'total_seconds_total': 1.6071063998970203}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.027006999967852607, 'p50_seconds': 0.004428699961863458, 'p90_seconds': 0.004428699961863458, 'max_seconds': 0.04958529997384176}
- `fetch`: {'count': 2, 'mean_seconds': 0.7764226500003133, 'p50_seconds': 0.7365789000177756, 'p90_seconds': 0.7365789000177756, 'max_seconds': 0.816266399982851}
- `normalize`: {'count': 2, 'mean_seconds': 0.0001235499803442508, 'p50_seconds': 0.00012279994552955031, 'p90_seconds': 0.00012279994552955031, 'max_seconds': 0.00012430001515895128}
- `total`: {'count': 2, 'mean_seconds': 0.8035531999485102, 'p50_seconds': 0.741131899994798, 'p90_seconds': 0.741131899994798, 'max_seconds': 0.8659744999022223}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
