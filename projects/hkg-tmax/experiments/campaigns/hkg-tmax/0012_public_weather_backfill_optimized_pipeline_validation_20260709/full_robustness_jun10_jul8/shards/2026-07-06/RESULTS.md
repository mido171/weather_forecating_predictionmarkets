# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260706_20260706_20260709T175016Z_105236`
Execution mode: `optimized`
Elapsed seconds: `628.5565598999965`
Date range: `2026-07-06` to `2026-07-06`

## Counts

- Source issues touched: `280`
- Fetch ok: `278`
- Fetch failed: `2`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `1718751695`
- Max staging bytes observed: `143666233`
- Final staging bytes: `0`
- Max raw object bytes observed: `11685822`
- Minimum free disk bytes observed: `237145886720`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8681431, 'max_staging_bytes_after_fetch': 104755005, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1189.440887899953, 'normalize_seconds_total': 407.81596250028815, 'db_write_seconds_total': 19.00928890047362, 'total_seconds_total': 1616.2661393007147, 'raw_bytes_deleted': 534447058, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 11685822, 'max_staging_bytes_after_fetch': 143666233, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1540.8182327001705, 'normalize_seconds_total': 493.19296380027663, 'db_write_seconds_total': 42.385005399934016, 'total_seconds_total': 2076.396201900381, 'raw_bytes_deleted': 741300283, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3173102, 'max_staging_bytes_after_fetch': 21858560, 'source_issues_touched': 144, 'fetch_ok': 142, 'task_errors': 142, 'fetch_seconds_total': 624.1761280002538, 'normalize_seconds_total': 394.15954989980673, 'db_write_seconds_total': 26.699920999642927, 'total_seconds_total': 1045.0355988997035, 'raw_bytes_deleted': 443004354, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.3146221975001806, 'p50_seconds': 0.08232119999593124, 'p90_seconds': 0.4932805999997072, 'max_seconds': 10.754350799950771}
- `fetch`: {'count': 280, 'mean_seconds': 11.98012588785849, 'p50_seconds': 8.805906199966557, 'p90_seconds': 27.555345900007524, 'max_seconds': 44.339137899980415}
- `normalize`: {'count': 280, 'mean_seconds': 4.625601700715612, 'p50_seconds': 4.364200399955735, 'p90_seconds': 7.343263800023124, 'max_seconds': 11.844613100052811}
- `total`: {'count': 280, 'mean_seconds': 16.920349786074283, 'p50_seconds': 12.958890399953816, 'p90_seconds': 34.97682880004868, 'max_seconds': 51.20636909996392}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
