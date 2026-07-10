# Results

State: `complete`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260616_20260617_20260709T161207Z_48204`
Execution mode: `optimized`
Elapsed seconds: `574.8442150999908`
Date range: `2026-06-16` to `2026-06-17`

## Counts

- Source issues touched: `136`
- Fetch ok: `136`
- Fetch failed: `None`
- Normalize ok: `136`
- Normalize failed: `None`
- Station features upserted: `2408`
- Area features upserted: `27768`
- Raw bytes deleted: `1616760626`
- Max staging bytes observed: `130481978`
- Final staging bytes: `0`
- Max raw object bytes observed: `12446099`
- Minimum free disk bytes observed: `238126866432`

## By Source

- `gfs`: {'max_raw_object_bytes': 12446099, 'max_staging_bytes_after_fetch': 130481978, 'source_issues_touched': 136, 'fetch_ok': 136, 'normalize_ok': 136, 'station_features_upserted': 2408, 'area_features_upserted': 27768, 'fetch_seconds_total': 1603.0542344999267, 'normalize_seconds_total': 1033.6847804003628, 'db_write_seconds_total': 74.95387430035044, 'total_seconds_total': 2711.69288920064, 'raw_bytes_deleted': 1616760626, 'raw_files_deleted': 136}

## Phase Runtime

- `db_write`: {'count': 136, 'mean_seconds': 0.5511314286790473, 'p50_seconds': 0.22987219999777153, 'p90_seconds': 1.6490171000477858, 'max_seconds': 5.017481199989561}
- `fetch`: {'count': 136, 'mean_seconds': 11.78716348897005, 'p50_seconds': 11.084306100034155, 'p90_seconds': 15.898740399978124, 'max_seconds': 25.929440999985673}
- `normalize`: {'count': 136, 'mean_seconds': 7.600623385296785, 'p50_seconds': 7.589688800042495, 'p90_seconds': 9.265966100036167, 'max_seconds': 11.849505999998655}
- `total`: {'count': 136, 'mean_seconds': 19.938918302945883, 'p50_seconds': 19.287343599949963, 'p90_seconds': 24.00392119999742, 'max_seconds': 34.90086900006281}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
