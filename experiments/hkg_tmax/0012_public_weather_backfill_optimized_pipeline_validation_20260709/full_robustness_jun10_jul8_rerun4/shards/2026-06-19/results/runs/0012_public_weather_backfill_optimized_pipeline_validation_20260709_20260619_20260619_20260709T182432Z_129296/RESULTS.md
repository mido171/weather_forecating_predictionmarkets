# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260619_20260619_20260709T182432Z_129296`
Execution mode: `optimized`
Elapsed seconds: `1.3925092999706976`
Date range: `2026-06-19` to `2026-06-19`

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
- `himawari9_b13_s0510`: {'skipped_existing': 141, 'source_issues_touched': 3, 'fetch_failed': 3, 'fetch_seconds_total': 1.9710925000254065, 'normalize_seconds_total': 0.0003869999782182276, 'db_write_seconds_total': 0.024436600040644407, 'total_seconds_total': 1.9959161000442691}

## Phase Runtime

- `db_write`: {'count': 3, 'mean_seconds': 0.00814553334688147, 'p50_seconds': 0.006196600035764277, 'p90_seconds': 0.006196600035764277, 'max_seconds': 0.013377099996432662}
- `fetch`: {'count': 3, 'mean_seconds': 0.6570308333418021, 'p50_seconds': 0.6414247000357136, 'p90_seconds': 0.6414247000357136, 'max_seconds': 0.8156567999976687}
- `normalize`: {'count': 3, 'mean_seconds': 0.0001289999927394092, 'p50_seconds': 0.00013240001862868667, 'p90_seconds': 0.00013240001862868667, 'max_seconds': 0.0001346999779343605}
- `total`: {'count': 3, 'mean_seconds': 0.665305366681423, 'p50_seconds': 0.6477560000494123, 'p90_seconds': 0.6477560000494123, 'max_seconds': 0.8291537999757566}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
