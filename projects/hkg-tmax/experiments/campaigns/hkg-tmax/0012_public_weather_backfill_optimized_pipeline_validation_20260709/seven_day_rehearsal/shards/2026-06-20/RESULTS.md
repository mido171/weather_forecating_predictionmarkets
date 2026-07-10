# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260620_20260620_20260709T154718Z_123644`
Execution mode: `optimized`
Elapsed seconds: `675.992928599997`
Date range: `2026-06-20` to `2026-06-20`

## Counts

- Source issues touched: `280`
- Fetch ok: `278`
- Fetch failed: `2`
- Normalize ok: `278`
- Normalize failed: `None`
- Station features upserted: `15838`
- Area features upserted: `28550`
- Raw bytes deleted: `1821807900`
- Max staging bytes observed: `164244354`
- Final staging bytes: `0`
- Max raw object bytes observed: `12401525`
- Minimum free disk bytes observed: `238031310848`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 9258668, 'max_staging_bytes_after_fetch': 108033924, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1158.2211828000145, 'normalize_seconds_total': 466.37973980011884, 'db_write_seconds_total': 41.49361490027513, 'total_seconds_total': 1666.0945375004085, 'raw_bytes_deleted': 562565320, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12401525, 'max_staging_bytes_after_fetch': 164244354, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 1606.9763064997387, 'normalize_seconds_total': 527.8557966998196, 'db_write_seconds_total': 57.642308600014076, 'total_seconds_total': 2192.4744117995724, 'raw_bytes_deleted': 803487679, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3246186, 'max_staging_bytes_after_fetch': 25947045, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 630.6495152006974, 'normalize_seconds_total': 441.6020467997878, 'db_write_seconds_total': 41.36304240016034, 'total_seconds_total': 1113.6146044006455, 'raw_bytes_deleted': 455754901, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.501782021073034, 'p50_seconds': 0.26409059995785356, 'p90_seconds': 0.789944399963133, 'max_seconds': 9.127202299947385}
- `fetch`: {'count': 280, 'mean_seconds': 12.128025016073037, 'p50_seconds': 8.692160600039642, 'p90_seconds': 25.646074199990835, 'max_seconds': 41.20463559997734}
- `normalize`: {'count': 280, 'mean_seconds': 5.127991368927594, 'p50_seconds': 4.591824900009669, 'p90_seconds': 7.923926200019196, 'max_seconds': 15.762601100024767}
- `total`: {'count': 280, 'mean_seconds': 17.757798406073665, 'p50_seconds': 14.292866699921433, 'p90_seconds': 34.13617259997409, 'max_seconds': 49.91540649998933}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
