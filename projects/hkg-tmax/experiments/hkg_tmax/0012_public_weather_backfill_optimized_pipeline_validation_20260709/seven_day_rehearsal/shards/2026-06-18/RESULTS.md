# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260618_20260618_20260709T153442Z_115056`
Execution mode: `optimized`
Elapsed seconds: `698.7725570000475`
Date range: `2026-06-18` to `2026-06-18`

## Counts

- Source issues touched: `280`
- Fetch ok: `277`
- Fetch failed: `3`
- Normalize ok: `277`
- Normalize failed: `None`
- Station features upserted: `15743`
- Area features upserted: `28539`
- Raw bytes deleted: `1810691021`
- Max staging bytes observed: `167752665`
- Final staging bytes: `0`
- Max raw object bytes observed: `12190564`
- Minimum free disk bytes observed: `238015729664`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8811665, 'max_staging_bytes_after_fetch': 99092988, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1069.561166600266, 'normalize_seconds_total': 461.51475050020963, 'db_write_seconds_total': 24.585207499971148, 'total_seconds_total': 1555.6611246004468, 'raw_bytes_deleted': 561676719, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12190564, 'max_staging_bytes_after_fetch': 167752665, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 1894.3065598001122, 'normalize_seconds_total': 525.1435047999839, 'db_write_seconds_total': 55.22862500004703, 'total_seconds_total': 2474.678689600143, 'raw_bytes_deleted': 799623148, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3207019, 'max_staging_bytes_after_fetch': 19223576, 'source_issues_touched': 144, 'fetch_ok': 141, 'normalize_ok': 141, 'station_features_upserted': 13395, 'area_features_upserted': 1551, 'fetch_seconds_total': 650.4784016000922, 'normalize_seconds_total': 460.7281409003772, 'db_write_seconds_total': 39.820374000410084, 'total_seconds_total': 1151.0269165008795, 'raw_bytes_deleted': 449391154, 'raw_files_deleted': 141, 'fetch_failed': 3}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.4272650232158152, 'p50_seconds': 0.25283690000651404, 'p90_seconds': 0.7239526000339538, 'max_seconds': 11.922550799965393}
- `fetch`: {'count': 280, 'mean_seconds': 12.908379028573108, 'p50_seconds': 8.121791700017639, 'p90_seconds': 29.468612299999222, 'max_seconds': 65.52486130001489}
- `normalize`: {'count': 280, 'mean_seconds': 5.169237129287753, 'p50_seconds': 4.729251199983992, 'p90_seconds': 8.235011000011582, 'max_seconds': 14.154579200025182}
- `total`: {'count': 280, 'mean_seconds': 18.504881181076676, 'p50_seconds': 14.905300199985504, 'p90_seconds': 37.72013239999069, 'max_seconds': 74.84805790003156}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
