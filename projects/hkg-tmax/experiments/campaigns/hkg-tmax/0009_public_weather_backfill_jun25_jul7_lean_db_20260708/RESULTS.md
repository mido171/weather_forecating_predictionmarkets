# Results

State: `complete`
Run id: `0009_public_weather_backfill_jun25_jul7_lean_db_20260708_20260708T143234Z`
Elapsed seconds: `49.41119430004619`
Date range: `2026-07-07` to `2026-07-07`

## Counts

- Source issues touched: `4`
- Fetch ok: `4`
- Fetch failed: `None`
- Normalize ok: `4`
- Normalize failed: `None`
- Station features upserted: `224`
- Area features upserted: `412`
- Raw bytes deleted: `24963547`
- Max staging bytes observed: `0`

## By Source

- `envf_hkust_hko_radar`: {'manifest_frames': 1, 'skipped_existing': 1}
- `gefs_control`: {'source_issues_touched': 1, 'fetch_ok': 1, 'normalize_ok': 1, 'station_features_upserted': 16, 'area_features_upserted': 182, 'raw_bytes_deleted': 8124301}
- `gfs`: {'source_issues_touched': 1, 'fetch_ok': 1, 'normalize_ok': 1, 'station_features_upserted': 18, 'area_features_upserted': 208, 'raw_bytes_deleted': 10566094}
- `himawari9_b13_s0510`: {'skipped_existing': 1, 'source_issues_touched': 2, 'fetch_ok': 2, 'normalize_ok': 2, 'station_features_upserted': 190, 'area_features_upserted': 22, 'raw_bytes_deleted': 6273152}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
