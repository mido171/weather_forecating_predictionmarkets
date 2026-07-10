# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260627_20260627_20260709T164939Z_109044`
Execution mode: `optimized`
Elapsed seconds: `652.5786147000035`
Date range: `2026-06-27` to `2026-06-27`

## Counts

- Source issues touched: `280`
- Fetch ok: `278`
- Fetch failed: `2`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `1843628211`
- Max staging bytes observed: `130160427`
- Final staging bytes: `0`
- Max raw object bytes observed: `12391452`
- Minimum free disk bytes observed: `237794455552`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 9629296, 'max_staging_bytes_after_fetch': 110244969, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1215.829670200008, 'normalize_seconds_total': 428.1829088003142, 'db_write_seconds_total': 13.482477099983953, 'total_seconds_total': 1657.495056100306, 'raw_bytes_deleted': 577557045, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12391452, 'max_staging_bytes_after_fetch': 130160427, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1785.919440699974, 'normalize_seconds_total': 499.5822548000142, 'db_write_seconds_total': 11.937115499633364, 'total_seconds_total': 2297.4388109996216, 'raw_bytes_deleted': 808959585, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3316709, 'max_staging_bytes_after_fetch': 23081229, 'source_issues_touched': 144, 'fetch_ok': 142, 'task_errors': 142, 'fetch_seconds_total': 785.6473878998659, 'normalize_seconds_total': 376.1374527999433, 'db_write_seconds_total': 27.77287959982641, 'total_seconds_total': 1189.5577202996355, 'raw_bytes_deleted': 457111581, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.1899731149980133, 'p50_seconds': 0.05996659997617826, 'p90_seconds': 0.3998175999731757, 'max_seconds': 2.387297999986913}
- `fetch`: {'count': 280, 'mean_seconds': 13.526416067142314, 'p50_seconds': 10.98503569996683, 'p90_seconds': 28.46486200002255, 'max_seconds': 46.25934619997861}
- `normalize`: {'count': 280, 'mean_seconds': 4.656795058572399, 'p50_seconds': 4.010394200042356, 'p90_seconds': 7.525253000028897, 'max_seconds': 11.49750589998439}
- `total`: {'count': 280, 'mean_seconds': 18.373184240712725, 'p50_seconds': 15.233735399902798, 'p90_seconds': 35.793947500002105, 'max_seconds': 52.729955999937374}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
