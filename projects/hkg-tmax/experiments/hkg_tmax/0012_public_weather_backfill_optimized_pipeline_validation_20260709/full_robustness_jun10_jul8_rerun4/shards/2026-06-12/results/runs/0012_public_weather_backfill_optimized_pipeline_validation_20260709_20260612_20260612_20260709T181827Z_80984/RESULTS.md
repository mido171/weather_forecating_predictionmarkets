# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260612_20260612_20260709T181827Z_80984`
Execution mode: `optimized`
Elapsed seconds: `1.4080661999760196`
Date range: `2026-06-12` to `2026-06-12`

## Counts

- Source issues touched: `2`
- Fetch ok: `None`
- Fetch failed: `2`
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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.3907724000164308, 'normalize_seconds_total': 0.00023790000705048442, 'db_write_seconds_total': 0.017242600035388023, 'total_seconds_total': 1.4082529000588693}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.008621300017694011, 'p50_seconds': 0.005616899987217039, 'p90_seconds': 0.005616899987217039, 'max_seconds': 0.011625700048170984}
- `fetch`: {'count': 2, 'mean_seconds': 0.6953862000082154, 'p50_seconds': 0.6469594000373036, 'p90_seconds': 0.6469594000373036, 'max_seconds': 0.7438129999791272}
- `normalize`: {'count': 2, 'mean_seconds': 0.00011895000352524221, 'p50_seconds': 0.00011630001245066524, 'p90_seconds': 0.00011630001245066524, 'max_seconds': 0.00012159999459981918}
- `total`: {'count': 2, 'mean_seconds': 0.7041264500294346, 'p50_seconds': 0.6587014000979252, 'p90_seconds': 0.6587014000979252, 'max_seconds': 0.7495514999609441}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
