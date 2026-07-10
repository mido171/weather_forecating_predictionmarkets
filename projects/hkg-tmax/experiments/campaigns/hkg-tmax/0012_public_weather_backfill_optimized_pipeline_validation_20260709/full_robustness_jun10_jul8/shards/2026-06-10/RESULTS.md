# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260610_20260610_20260709T162423Z_113132`
Execution mode: `optimized`
Elapsed seconds: `1.4685341999866068`
Date range: `2026-06-10` to `2026-06-10`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.1430643000057898, 'normalize_seconds_total': 0.0002595999976620078, 'db_write_seconds_total': 0.16617439995752648, 'total_seconds_total': 1.3094982999609783}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.08308719997876324, 'p50_seconds': 0.004874999984167516, 'p90_seconds': 0.004874999984167516, 'max_seconds': 0.16129939997335896}
- `fetch`: {'count': 2, 'mean_seconds': 0.5715321500028949, 'p50_seconds': 0.4763798000058159, 'p90_seconds': 0.4763798000058159, 'max_seconds': 0.6666844999999739}
- `normalize`: {'count': 2, 'mean_seconds': 0.0001297999988310039, 'p50_seconds': 0.00012860004790127277, 'p90_seconds': 0.00012860004790127277, 'max_seconds': 0.00013099994976073503}
- `total`: {'count': 2, 'mean_seconds': 0.6547491499804892, 'p50_seconds': 0.6378101999289356, 'p90_seconds': 0.6378101999289356, 'max_seconds': 0.6716881000320427}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
