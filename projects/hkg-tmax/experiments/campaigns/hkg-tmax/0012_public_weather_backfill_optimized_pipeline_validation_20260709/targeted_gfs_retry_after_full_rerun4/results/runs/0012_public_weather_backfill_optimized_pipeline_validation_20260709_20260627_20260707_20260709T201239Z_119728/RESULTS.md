# Results

State: `complete`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260627_20260707_20260709T201239Z_119728`
Execution mode: `optimized`
Elapsed seconds: `125.86055340000894`
Date range: `2026-06-27` to `2026-07-07`

## Counts

- Source issues touched: `6`
- Fetch ok: `6`
- Fetch failed: `None`
- Normalize ok: `6`
- Normalize failed: `None`
- Station features upserted: `103`
- Area features upserted: `1183`
- Raw bytes deleted: `68556803`
- Max staging bytes observed: `23659243`
- Final staging bytes: `0`
- Max raw object bytes observed: `12316106`
- Minimum free disk bytes observed: `232462237696`

## By Source

- `gfs`: {'skipped_existing': 742, 'max_raw_object_bytes': 12316106, 'max_staging_bytes_after_fetch': 23659243, 'source_issues_touched': 6, 'fetch_ok': 6, 'normalize_ok': 6, 'station_features_upserted': 103, 'area_features_upserted': 1183, 'fetch_seconds_total': 45.71464749996085, 'normalize_seconds_total': 55.763059800025076, 'db_write_seconds_total': 1.614508799975738, 'total_seconds_total': 103.09221609996166, 'raw_bytes_deleted': 68556803, 'raw_files_deleted': 6}

## Phase Runtime

- `db_write`: {'count': 6, 'mean_seconds': 0.2690847999959563, 'p50_seconds': 0.13591259997338057, 'p90_seconds': 0.20894640003098175, 'max_seconds': 0.8800498999771662}
- `fetch`: {'count': 6, 'mean_seconds': 7.619107916660141, 'p50_seconds': 6.526112399995327, 'p90_seconds': 8.389691699994728, 'max_seconds': 11.684577799984254}
- `normalize`: {'count': 6, 'mean_seconds': 9.293843300004179, 'p50_seconds': 9.430910400056746, 'p90_seconds': 10.446398800006136, 'max_seconds': 11.769275599974208}
- `total`: {'count': 6, 'mean_seconds': 17.182036016660277, 'p50_seconds': 16.131177100061905, 'p90_seconds': 18.609229199937545, 'max_seconds': 22.23954129999038}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
