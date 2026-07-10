# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260614_20260614_20260709T182028Z_114936`
Execution mode: `optimized`
Elapsed seconds: `1.8260544000077061`
Date range: `2026-06-14` to `2026-06-14`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 2.1238457000581548, 'normalize_seconds_total': 0.0002579999854788184, 'db_write_seconds_total': 0.07262500002980232, 'total_seconds_total': 2.196728700073436}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.03631250001490116, 'p50_seconds': 0.004183300014119595, 'p90_seconds': 0.004183300014119595, 'max_seconds': 0.06844170001568273}
- `fetch`: {'count': 2, 'mean_seconds': 1.0619228500290774, 'p50_seconds': 1.0277125000138767, 'p90_seconds': 1.0277125000138767, 'max_seconds': 1.096133200044278}
- `normalize`: {'count': 2, 'mean_seconds': 0.0001289999927394092, 'p50_seconds': 0.0001275999820791185, 'p90_seconds': 0.0001275999820791185, 'max_seconds': 0.00013040000339969993}
- `total`: {'count': 2, 'mean_seconds': 1.098364350036718, 'p50_seconds': 1.0962818000116386, 'p90_seconds': 1.0962818000116386, 'max_seconds': 1.1004469000617974}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
