# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260618_20260618_20260709T182432Z_124744`
Execution mode: `optimized`
Elapsed seconds: `1.389362899994012`
Date range: `2026-06-18` to `2026-06-18`

## Counts

- Source issues touched: `3`
- Fetch ok: `None`
- Fetch failed: `3`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `None`
- Max staging bytes observed: `None`
- Final staging bytes: `0`
- Max raw object bytes observed: `None`
- Minimum free disk bytes observed: `None`

## By Source

- `gefs_control`: {'skipped_existing': 68}
- `gfs`: {'skipped_existing': 68}
- `himawari9_b13_s0510`: {'skipped_existing': 141, 'source_issues_touched': 3, 'fetch_failed': 3, 'fetch_seconds_total': 2.247942799993325, 'normalize_seconds_total': 0.0004159000236541033, 'db_write_seconds_total': 0.02496820001397282, 'total_seconds_total': 2.273326900030952}

## Phase Runtime

- `db_write`: {'count': 3, 'mean_seconds': 0.00832273333799094, 'p50_seconds': 0.00616200000513345, 'p90_seconds': 0.00616200000513345, 'max_seconds': 0.013711499981582165}
- `fetch`: {'count': 3, 'mean_seconds': 0.7493142666644417, 'p50_seconds': 0.80689429998165, 'p90_seconds': 0.80689429998165, 'max_seconds': 0.888896900054533}
- `normalize`: {'count': 3, 'mean_seconds': 0.00013863334121803442, 'p50_seconds': 0.00012390001211315393, 'p90_seconds': 0.00012390001211315393, 'max_seconds': 0.00017449998995289207}
- `total`: {'count': 3, 'mean_seconds': 0.7577756333436506, 'p50_seconds': 0.8132307999767363, 'p90_seconds': 0.8132307999767363, 'max_seconds': 0.9027323000482284}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
