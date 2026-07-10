# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260627_20260627_20260709T184043Z_128980`
Execution mode: `optimized`
Elapsed seconds: `684.5764389000251`
Date range: `2026-06-27` to `2026-06-27`

## Counts

- Source issues touched: `280`
- Fetch ok: `277`
- Fetch failed: `3`
- Normalize ok: `277`
- Normalize failed: `None`
- Station features upserted: `15820`
- Area features upserted: `28342`
- Raw bytes deleted: `1831378269`
- Max staging bytes observed: `145435025`
- Final staging bytes: `0`
- Max raw object bytes observed: `12391452`
- Minimum free disk bytes observed: `236961964032`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 9629296, 'max_staging_bytes_after_fetch': 100984021, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1249.6824896995095, 'normalize_seconds_total': 448.456183700182, 'db_write_seconds_total': 36.74893699988024, 'total_seconds_total': 1734.8876103995717, 'raw_bytes_deleted': 577557045, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12391452, 'max_staging_bytes_after_fetch': 145435025, 'source_issues_touched': 68, 'fetch_ok': 67, 'normalize_ok': 67, 'station_features_upserted': 1186, 'area_features_upserted': 13676, 'fetch_seconds_total': 1655.1786424003658, 'normalize_seconds_total': 526.0737070997711, 'db_write_seconds_total': 65.6833095997572, 'total_seconds_total': 2246.935659099894, 'raw_bytes_deleted': 796709643, 'raw_files_deleted': 67, 'fetch_failed': 1}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3316709, 'max_staging_bytes_after_fetch': 22874037, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 725.975971600099, 'normalize_seconds_total': 430.810925299942, 'db_write_seconds_total': 34.183629400038626, 'total_seconds_total': 1190.9705263000797, 'raw_bytes_deleted': 457111581, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.48791384285598594, 'p50_seconds': 0.2605222000274807, 'p90_seconds': 0.7955827999976464, 'max_seconds': 6.982378899992909}
- `fetch`: {'count': 280, 'mean_seconds': 12.96727537035705, 'p50_seconds': 9.315716199984308, 'p90_seconds': 28.24481449997984, 'max_seconds': 37.769018699997105}
- `normalize`: {'count': 279, 'mean_seconds': 5.037063856988872, 'p50_seconds': 4.810154500009958, 'p90_seconds': 7.823600000003353, 'max_seconds': 14.956617799995001}
- `total`: {'count': 280, 'mean_seconds': 18.474263556426948, 'p50_seconds': 14.249117900035344, 'p90_seconds': 36.57611580006778, 'max_seconds': 48.10967699997127}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
