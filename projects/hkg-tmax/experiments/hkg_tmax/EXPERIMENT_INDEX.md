# HKG Tmax Experiment Index

| ID | Folder | Primary Question | Main Artifacts |
|---|---|---|---|
| 0001 | `0001_residual_ml_strategy_20260705` | Can broad residual ML improve the official HKG Tmax anchor enough to promote? | Legacy result folder: `experiments/hkg_tmax_residual_ml_strategy/results/` |
| 0002 | `0002_selective_no_harm_router_20260705` | Can a selective no-harm router improve the current A7 baseline by correcting only when evidence says correction beats abstention? | `results/scoreboard.csv`, `results/no_harm_audit.json`, `results/next_round_model_card.md`, `results/anchor_provenance_audit.csv` |
| 0003 | `0003_official_residual_memory_20260706` | Can lag-safe same-cutoff official residual memory improve A7 enough to promote? | `results/scoreboard.csv`, `results/model_card.md`, `results/leakage_audit.json`, `results/residual_memory_publication_safety_audit.json`, `results/row_identity_gate.json` |
| 0004 | `0004_station_hour_residual_information_atlas_20260708` | Do Postgres Info.gov hourly HKO/station snapshots explain official Tmax residuals at T-1 23:59 HKT? | `RESULTS.md`, `results/metrics.json`, `artifacts/station_hour_feature_correlations.csv`, `artifacts/top_feature_spearman_and_spreads.csv` |
| 0005 | `0005_public_gfs_gefs_himawari_fetch_smoke_20260708` | Can public latest-issued GFS, GEFS, and Himawari-9 payloads be fetched directly without GribStream? | `RESULTS.md`, `artifacts/fetch_summary.json`, `raw/gfs/`, `raw/gefs/`, `raw/himawari/` |
| 0006 | `0006_public_daily_coverage_benchmark_20260707` | How long/large is one complete practical daily public GFS/GEFS/Himawari coverage set? | `README.md`, `normalized/daily_coverage_benchmark_summary.json`, `normalized/model_cycle_station_features.csv`, `normalized/himawari_b13_s0510_scan_features.csv` |
| 0007 | `0007_public_7day_gfs_gefs_himawari_backfill_rehearsal_20260708` | Can a 7-day public backfill rehearsal fetch, timestamp, normalize, sanity-check, and size GFS/GEFS/Himawari issues? | `README.md`, `normalized/sanity_report.json`, `normalized/fetch_manifest.csv`, `normalized/backfill_size_estimates.json` |
| 0008 | `0008_last2_gfs_gefs_radar_structured_delivery_20260708` | Can last-two-complete-day GFS/GEFS and radar records be delivered as structured normalized files with Postgres glue metadata and no retained raw payloads? | `README.md`, `metadata/summary.json`, `metadata/postgres_glue_schema.sql`, `normalized/source_issue_glue_last2.csv`, `normalized/attribute_catalog_last2.csv` |
| 0009 | `0009_public_weather_backfill_jun25_jul7_lean_db_20260708` | Can public GFS/GEFS control, Himawari B13/S0510, and radar proxy issues be streamed into Postgres with leakage clocks and immediate raw deletion? | `SMOKE_REPORT.md`, `RESULTS.md`, `results/metrics.json`, `metadata/expected_inventory.json`, `logs/live_summary.json` |
| 0010 | `0010_public_weather_backfill_jun10_jul8_lean_db_test_20260709` | How does the lean DB backfill behave across a wider Jun 10-Jul 8 rehearsal window? | `RESULTS.md`, `results/metrics.json`, `logs/live_summary.json` |
| 0011 | `0011_public_weather_speed_optimization_20260709` | Which low-risk fetch/decode settings reduce public weather acquisition time before touching the DB path? | `RESULTS.md`, `artifacts/trial_summary.csv`, `r/*/summary.json` |
| 0012 | `0012_public_weather_backfill_optimized_pipeline_validation_20260709` | Does the optimized full DB pipeline pass dry-run, one-day, idempotency, seven-day, and robustness validation gates? | `PROTOCOL.md`, `RESULTS.md`, `documentation/README.md`, `documentation/PUBLIC_WEATHER_BACKFILL_IMPLEMENTATION_AND_VALIDATION.md`, `documentation/POSTGRES_STORAGE_CAPACITY_ESTIMATE_2017_TO_2026.md` |

## Conventions

- Code is never authored inside experiment result folders.
- Experiment folders may contain copied input memos and generated outputs.
- If an old run already exists in a legacy flat location, the numbered folder documents the legacy path rather than moving or deleting prior artifacts.
- Sealed confirmation rows are report-only and must not affect feature selection, threshold selection, model selection, or hyperparameter choices.
