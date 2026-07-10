# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260705_20260705_20260709T193343Z_10152`
Execution mode: `optimized`
Elapsed seconds: `651.8927140999585`
Date range: `2026-07-05` to `2026-07-05`

## Counts

- Source issues touched: `280`
- Fetch ok: `278`
- Fetch failed: `2`
- Normalize ok: `278`
- Normalize failed: `None`
- Station features upserted: `15838`
- Area features upserted: `28550`
- Raw bytes deleted: `1703835465`
- Max staging bytes observed: `143396258`
- Final staging bytes: `0`
- Max raw object bytes observed: `11343997`
- Minimum free disk bytes observed: `231354880000`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8528393, 'max_staging_bytes_after_fetch': 102504441, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1025.4997764999862, 'normalize_seconds_total': 446.0503675999935, 'db_write_seconds_total': 55.79822779970709, 'total_seconds_total': 1527.3483718996868, 'raw_bytes_deleted': 518959489, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 11343997, 'max_staging_bytes_after_fetch': 143396258, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 1491.1622875002795, 'normalize_seconds_total': 503.5112299998291, 'db_write_seconds_total': 71.08740450016921, 'total_seconds_total': 2065.760922000278, 'raw_bytes_deleted': 738236500, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3230592, 'max_staging_bytes_after_fetch': 25016123, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 566.6220226999139, 'normalize_seconds_total': 415.81705369980773, 'db_write_seconds_total': 36.165454300295096, 'total_seconds_total': 1018.6045307000168, 'raw_bytes_deleted': 446639476, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.5823253092863264, 'p50_seconds': 0.26421470002969727, 'p90_seconds': 1.2899555999902077, 'max_seconds': 7.261003300023731}
- `fetch`: {'count': 280, 'mean_seconds': 11.01172888107207, 'p50_seconds': 7.637746900029015, 'p90_seconds': 25.187983700016048, 'max_seconds': 41.34147790004499}
- `normalize`: {'count': 280, 'mean_seconds': 4.8763523260701085, 'p50_seconds': 4.818530099990312, 'p90_seconds': 7.6604208999779075, 'max_seconds': 13.588516499963589}
- `total`: {'count': 280, 'mean_seconds': 16.470406516428504, 'p50_seconds': 12.945918300072663, 'p90_seconds': 33.481487199955154, 'max_seconds': 48.447340700018685}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
