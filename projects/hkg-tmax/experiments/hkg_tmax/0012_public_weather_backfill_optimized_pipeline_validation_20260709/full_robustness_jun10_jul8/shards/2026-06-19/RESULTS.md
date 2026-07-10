# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260619_20260619_20260709T163234Z_106748`
Execution mode: `optimized`
Elapsed seconds: `2.075786800007336`
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
- `himawari9_b13_s0510`: {'skipped_existing': 141, 'source_issues_touched': 3, 'fetch_failed': 3, 'fetch_seconds_total': 2.7609427999705076, 'normalize_seconds_total': 0.0003662000526674092, 'db_write_seconds_total': 0.022888200008310378, 'total_seconds_total': 2.7841972000314854}

## Phase Runtime

- `db_write`: {'count': 3, 'mean_seconds': 0.007629400002770126, 'p50_seconds': 0.008158400014508516, 'p90_seconds': 0.008158400014508516, 'max_seconds': 0.010376500023994595}
- `fetch`: {'count': 3, 'mean_seconds': 0.9203142666568359, 'p50_seconds': 1.0401905999751762, 'p90_seconds': 1.0401905999751762, 'max_seconds': 1.0728894000058062}
- `normalize`: {'count': 3, 'mean_seconds': 0.00012206668422246973, 'p50_seconds': 0.00012450001668184996, 'p90_seconds': 0.00012450001668184996, 'max_seconds': 0.00012600002810359}
- `total`: {'count': 3, 'mean_seconds': 0.9280657333438285, 'p50_seconds': 1.0506916000158526, 'p90_seconds': 1.0506916000158526, 'max_seconds': 1.077368700003717}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
