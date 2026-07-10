# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260611_20260611_20260709T181628Z_119804`
Execution mode: `optimized`
Elapsed seconds: `1.2889299999806099`
Date range: `2026-06-11` to `2026-06-11`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.4483667999738827, 'normalize_seconds_total': 0.00023990002227947116, 'db_write_seconds_total': 0.015347399981692433, 'total_seconds_total': 1.4639540999778546}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.007673699990846217, 'p50_seconds': 0.004975099989678711, 'p90_seconds': 0.004975099989678711, 'max_seconds': 0.010372299992013723}
- `fetch`: {'count': 2, 'mean_seconds': 0.7241833999869414, 'p50_seconds': 0.6896920999861322, 'p90_seconds': 0.6896920999861322, 'max_seconds': 0.7586746999877505}
- `normalize`: {'count': 2, 'mean_seconds': 0.00011995001113973558, 'p50_seconds': 0.00011909997556358576, 'p90_seconds': 0.00011909997556358576, 'max_seconds': 0.0001208000467158854}
- `total`: {'count': 2, 'mean_seconds': 0.7319770499889273, 'p50_seconds': 0.6947880000225268, 'p90_seconds': 0.6947880000225268, 'max_seconds': 0.7691660999553278}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
