# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260611_20260611_20260709T181523Z_114336`
Execution mode: `optimized`
Elapsed seconds: `1.5010805999627337`
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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.452940400049556, 'normalize_seconds_total': 0.0002373000024817884, 'db_write_seconds_total': 0.01772010000422597, 'total_seconds_total': 1.4708978000562638}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.008860050002112985, 'p50_seconds': 0.006571199977770448, 'p90_seconds': 0.006571199977770448, 'max_seconds': 0.011148900026455522}
- `fetch`: {'count': 2, 'mean_seconds': 0.726470200024778, 'p50_seconds': 0.6984229999943636, 'p90_seconds': 0.6984229999943636, 'max_seconds': 0.7545174000551924}
- `normalize`: {'count': 2, 'mean_seconds': 0.0001186500012408942, 'p50_seconds': 0.00011660001473501325, 'p90_seconds': 0.00011660001473501325, 'max_seconds': 0.00012069998774677515}
- `total`: {'count': 2, 'mean_seconds': 0.7354489000281319, 'p50_seconds': 0.7096926000085659, 'p90_seconds': 0.7096926000085659, 'max_seconds': 0.7612052000476979}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
