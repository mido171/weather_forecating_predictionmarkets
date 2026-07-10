# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260611_20260611_20260709T162423Z_126436`
Execution mode: `optimized`
Elapsed seconds: `1.2688454000162892`
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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.367143000010401, 'normalize_seconds_total': 0.0002466000150889158, 'db_write_seconds_total': 0.016919300018344074, 'total_seconds_total': 1.384308900043834}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.008459650009172037, 'p50_seconds': 0.00328619999345392, 'p90_seconds': 0.00328619999345392, 'max_seconds': 0.013633100024890155}
- `fetch`: {'count': 2, 'mean_seconds': 0.6835715000052005, 'p50_seconds': 0.6479862999985926, 'p90_seconds': 0.6479862999985926, 'max_seconds': 0.7191567000118084}
- `normalize`: {'count': 2, 'mean_seconds': 0.0001233000075444579, 'p50_seconds': 0.00012139999307692051, 'p90_seconds': 0.00012139999307692051, 'max_seconds': 0.00012520002201199532}
- `total`: {'count': 2, 'mean_seconds': 0.692154450021917, 'p50_seconds': 0.6617408000165597, 'p90_seconds': 0.6617408000165597, 'max_seconds': 0.7225681000272743}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
