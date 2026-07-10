# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260708_20260708_20260709T195700Z_127596`
Execution mode: `optimized`
Elapsed seconds: `565.7007927000523`
Date range: `2026-07-08` to `2026-07-08`

## Counts

- Source issues touched: `279`
- Fetch ok: `277`
- Fetch failed: `2`
- Normalize ok: `277`
- Normalize failed: `None`
- Station features upserted: `15743`
- Area features upserted: `28539`
- Raw bytes deleted: `1730497854`
- Max staging bytes observed: `155824765`
- Final staging bytes: `0`
- Max raw object bytes observed: `12117980`
- Minimum free disk bytes observed: `232379125760`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8670278, 'max_staging_bytes_after_fetch': 97072259, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 802.7857584002195, 'normalize_seconds_total': 384.9939117998583, 'db_write_seconds_total': 30.152525000157766, 'total_seconds_total': 1217.9321952002356, 'raw_bytes_deleted': 523588212, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12117980, 'max_staging_bytes_after_fetch': 155824765, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 1127.881035599683, 'normalize_seconds_total': 447.82593040016945, 'db_write_seconds_total': 55.12402070022654, 'total_seconds_total': 1630.830986700079, 'raw_bytes_deleted': 753037405, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'skipped_existing': 1, 'max_raw_object_bytes': 3245434, 'max_staging_bytes_after_fetch': 25710344, 'source_issues_touched': 143, 'fetch_ok': 141, 'normalize_ok': 141, 'station_features_upserted': 13395, 'area_features_upserted': 1551, 'fetch_seconds_total': 443.7937190005323, 'normalize_seconds_total': 398.5028903999482, 'db_write_seconds_total': 33.44419529999141, 'total_seconds_total': 875.7408047004719, 'raw_bytes_deleted': 453872237, 'raw_files_deleted': 141, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 279, 'mean_seconds': 0.4255223691769739, 'p50_seconds': 0.23452529998030514, 'p90_seconds': 0.7056744000292383, 'max_seconds': 7.5036709000123665}
- `fetch`: {'count': 279, 'mean_seconds': 8.510611157707652, 'p50_seconds': 6.283184500003699, 'p90_seconds': 17.963995500002056, 'max_seconds': 35.60050619998947}
- `normalize`: {'count': 279, 'mean_seconds': 4.4133431275984805, 'p50_seconds': 4.117712699982803, 'p90_seconds': 6.832175499992445, 'max_seconds': 11.057038499973714}
- `total`: {'count': 279, 'mean_seconds': 13.349476654483105, 'p50_seconds': 11.242420999973547, 'p90_seconds': 24.66765769995982, 'max_seconds': 42.16461239999626}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
