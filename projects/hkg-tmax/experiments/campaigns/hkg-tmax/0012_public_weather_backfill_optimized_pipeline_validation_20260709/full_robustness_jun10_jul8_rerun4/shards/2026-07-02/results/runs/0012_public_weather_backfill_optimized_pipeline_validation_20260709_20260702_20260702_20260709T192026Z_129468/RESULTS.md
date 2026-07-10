# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260702_20260702_20260709T192026Z_129468`
Execution mode: `optimized`
Elapsed seconds: `662.1696186999907`
Date range: `2026-07-02` to `2026-07-02`

## Counts

- Source issues touched: `280`
- Fetch ok: `276`
- Fetch failed: `4`
- Normalize ok: `276`
- Normalize failed: `None`
- Station features upserted: `15648`
- Area features upserted: `28528`
- Raw bytes deleted: `1798056953`
- Max staging bytes observed: `144445021`
- Final staging bytes: `0`
- Max raw object bytes observed: `12209414`
- Minimum free disk bytes observed: `233415593984`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8929176, 'max_staging_bytes_after_fetch': 101160306, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1153.700375000015, 'normalize_seconds_total': 428.2919381001266, 'db_write_seconds_total': 52.845414600218646, 'total_seconds_total': 1634.8377277003601, 'raw_bytes_deleted': 559614185, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12209414, 'max_staging_bytes_after_fetch': 144445021, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 1772.2565909003606, 'normalize_seconds_total': 504.780978900264, 'db_write_seconds_total': 56.23335190030048, 'total_seconds_total': 2333.270921700925, 'raw_bytes_deleted': 795549489, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3206180, 'max_staging_bytes_after_fetch': 25218969, 'source_issues_touched': 144, 'fetch_ok': 140, 'normalize_ok': 140, 'station_features_upserted': 13300, 'area_features_upserted': 1540, 'fetch_seconds_total': 724.2343262001523, 'normalize_seconds_total': 419.90490279963706, 'db_write_seconds_total': 34.516329900186975, 'total_seconds_total': 1178.6555588999763, 'raw_bytes_deleted': 442893279, 'raw_files_deleted': 140, 'fetch_failed': 4}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.5128396300025218, 'p50_seconds': 0.26533000002382323, 'p90_seconds': 0.79900199995609, 'max_seconds': 7.862658100028057}
- `fetch`: {'count': 280, 'mean_seconds': 13.0363974717876, 'p50_seconds': 10.566056600015145, 'p90_seconds': 29.488033500034362, 'max_seconds': 52.50111429998651}
- `normalize`: {'count': 280, 'mean_seconds': 4.8320636421429555, 'p50_seconds': 4.361803199979477, 'p90_seconds': 7.671600400004536, 'max_seconds': 12.684826500015333}
- `total`: {'count': 280, 'mean_seconds': 18.381300743933078, 'p50_seconds': 16.39996000000974, 'p90_seconds': 37.517710499989334, 'max_seconds': 58.8171342999558}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
