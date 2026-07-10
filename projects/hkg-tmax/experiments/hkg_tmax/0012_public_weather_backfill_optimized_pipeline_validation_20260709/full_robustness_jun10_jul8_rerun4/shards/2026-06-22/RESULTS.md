# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260622_20260622_20260709T182835Z_116228`
Execution mode: `optimized`
Elapsed seconds: `252.73197209998034`
Date range: `2026-06-22` to `2026-06-22`

## Counts

- Source issues touched: `166`
- Fetch ok: `163`
- Fetch failed: `3`
- Normalize ok: `163`
- Normalize failed: `None`
- Station features upserted: `13766`
- Area features upserted: `5802`
- Raw bytes deleted: `631608470`
- Max staging bytes observed: `121471396`
- Final staging bytes: `0`
- Max raw object bytes observed: `8509302`
- Minimum free disk bytes observed: `237012000768`

## By Source

- `gefs_control`: {'skipped_existing': 46, 'max_raw_object_bytes': 8509302, 'max_staging_bytes_after_fetch': 121471396, 'source_issues_touched': 22, 'fetch_ok': 22, 'normalize_ok': 22, 'station_features_upserted': 371, 'area_features_upserted': 4251, 'fetch_seconds_total': 413.155023600033, 'normalize_seconds_total': 145.40069829998538, 'db_write_seconds_total': 8.499066799937282, 'total_seconds_total': 567.0547886999557, 'raw_bytes_deleted': 178685088, 'raw_files_deleted': 22}
- `gfs`: {'skipped_existing': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3243971, 'max_staging_bytes_after_fetch': 22657858, 'source_issues_touched': 144, 'fetch_ok': 141, 'normalize_ok': 141, 'station_features_upserted': 13395, 'area_features_upserted': 1551, 'fetch_seconds_total': 712.9717447998701, 'normalize_seconds_total': 404.22577159985667, 'db_write_seconds_total': 32.73659300006693, 'total_seconds_total': 1149.9341093997937, 'raw_bytes_deleted': 452923382, 'raw_files_deleted': 141, 'fetch_failed': 3}

## Phase Runtime

- `db_write`: {'count': 166, 'mean_seconds': 0.2484075891566519, 'p50_seconds': 0.2225406999932602, 'p90_seconds': 0.3586440000217408, 'max_seconds': 3.9750311000389047}
- `fetch`: {'count': 166, 'mean_seconds': 6.783896195180139, 'p50_seconds': 4.746820200001821, 'p90_seconds': 14.244664400001056, 'max_seconds': 29.265839600004256}
- `normalize`: {'count': 166, 'mean_seconds': 3.31100283072194, 'p50_seconds': 2.748424799996428, 'p90_seconds': 5.782718499947805, 'max_seconds': 15.633878099964932}
- `total`: {'count': 166, 'mean_seconds': 10.343306615058731, 'p50_seconds': 7.971755699953064, 'p90_seconds': 20.464618199970573, 'max_seconds': 36.968147200066596}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
