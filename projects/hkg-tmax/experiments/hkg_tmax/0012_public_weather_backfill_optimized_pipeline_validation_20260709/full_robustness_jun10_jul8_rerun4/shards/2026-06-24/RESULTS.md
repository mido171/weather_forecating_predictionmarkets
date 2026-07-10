# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260624_20260624_20260709T183440Z_127008`
Execution mode: `optimized`
Elapsed seconds: `276.20331820001593`
Date range: `2026-06-24` to `2026-06-24`

## Counts

- Source issues touched: `172`
- Fetch ok: `170`
- Fetch failed: `2`
- Normalize ok: `170`
- Normalize failed: `None`
- Station features upserted: `13963`
- Area features upserted: `6983`
- Raw bytes deleted: `691597332`
- Max staging bytes observed: `121981818`
- Final staging bytes: `0`
- Max raw object bytes observed: `8565584`
- Minimum free disk bytes observed: `236986290176`

## By Source

- `gefs_control`: {'skipped_existing': 40, 'max_raw_object_bytes': 8565584, 'max_staging_bytes_after_fetch': 121981818, 'source_issues_touched': 28, 'fetch_ok': 28, 'normalize_ok': 28, 'station_features_upserted': 473, 'area_features_upserted': 5421, 'fetch_seconds_total': 502.10015860013664, 'normalize_seconds_total': 189.15726990014082, 'db_write_seconds_total': 12.010988799796905, 'total_seconds_total': 703.2684173000744, 'raw_bytes_deleted': 230921841, 'raw_files_deleted': 28}
- `gfs`: {'skipped_existing': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3265018, 'max_staging_bytes_after_fetch': 22823031, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 683.5003005999606, 'normalize_seconds_total': 448.3326282001217, 'db_write_seconds_total': 35.03537210030481, 'total_seconds_total': 1166.868300900387, 'raw_bytes_deleted': 460675491, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 172, 'mean_seconds': 0.27352535407035883, 'p50_seconds': 0.2373645000043325, 'p90_seconds': 0.47830770001746714, 'max_seconds': 1.705452099966351}
- `fetch`: {'count': 172, 'mean_seconds': 6.893025925581961, 'p50_seconds': 4.342825300002005, 'p90_seconds': 15.722416699980386, 'max_seconds': 29.470904000045266}
- `normalize`: {'count': 172, 'mean_seconds': 3.706336616861991, 'p50_seconds': 3.110679799981881, 'p90_seconds': 6.30853019998176, 'max_seconds': 11.800212600035593}
- `total`: {'count': 172, 'mean_seconds': 10.87288789651431, 'p50_seconds': 8.025556200009305, 'p90_seconds': 22.706787399947643, 'max_seconds': 36.39084660005756}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
