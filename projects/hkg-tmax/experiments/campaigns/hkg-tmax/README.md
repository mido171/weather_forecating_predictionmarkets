# HKG Tmax campaign

This campaign contains the numbered HKG Tmax modeling and public-weather
engineering experiments. Read this index, then the selected experiment README;
do not browse shard/run folders for conclusions.

| ID | Experiment | Status | Headline |
|---:|---|---|---|
| 0001 | [Broad residual ML](0001_residual_ml_strategy_20260705/README.md) | No promote | A7 gained 0.032193 C MAE, below the 0.035 C gate |
| 0002 | [Selective no-harm router](0002_selective_no_harm_router_20260705/README.md) | No promote | C2/C3 did not beat A7 |
| 0003 | [Official residual memory](0003_official_residual_memory_20260706/README.md) | No promote | D5's tiny gain failed development/presealed gates |
| 0004 | [Station-hour information atlas](0004_station_hour_residual_information_atlas_20260708/README.md) | Information positive; no promote | 82/100 signal, but only 0.002969 C gain vs bias-only |
| 0005 | [Public fetch smoke](0005_public_gfs_gefs_himawari_fetch_smoke_20260708/README.md) | Pass | GFS, GEFS control, and Himawari payloads fetched |
| 0006 | [Daily coverage benchmark](0006_public_daily_coverage_benchmark_20260707/README.md) | Complete | 8/8 model and 141/144 Himawari items |
| 0007 | [Seven-day rehearsal](0007_public_7day_gfs_gefs_himawari_backfill_rehearsal_20260708/README.md) | Complete with provider gaps | 1,671/1,960 requests succeeded |
| 0008 | [Two-day structured delivery](0008_last2_gfs_gefs_radar_structured_delivery_20260708/README.md) | Complete | Compact model/radar tables, no raw retention |
| 0009 | [Lean DB smoke](0009_public_weather_backfill_jun25_jul7_lean_db_20260708/README.md) | Complete one-day run | DB persistence and immediate raw cleanup passed |
| 0010 | [Wide-window lean DB test](0010_public_weather_backfill_jun10_jul8_lean_db_test_20260709/README.md) | Incomplete; superseded | 12 completed days, four stale-running shards |
| 0011 | [Speed optimization](0011_public_weather_speed_optimization_20260709/README.md) | Strong operational result | Safe gap=0 coalescing and bounded Himawari concurrency won |
| 0012 | [Optimized pipeline validation](0012_public_weather_backfill_optimized_pipeline_validation_20260709/README.md) | Accepted with notes | Correctness passed; 89/100; 121.4 GB projected |

## Campaign decision map

- Point-forecast experiments 0001-0004 found real but non-promotable
  calibration/signal improvements.
- Experiments 0005-0012 established and hardened public-weather acquisition.
- Experiment 0012 supersedes 0009/0010 as the accepted three-source backfill
  evidence.

Retired campaign indexes and per-run prose are recoverable through
[`DOCUMENT_PROVENANCE.csv`](../DOCUMENT_PROVENANCE.csv).
