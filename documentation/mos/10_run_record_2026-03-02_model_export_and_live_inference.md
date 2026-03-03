# 10 - Model Lineage, Retrain, Export, and Live Inference (KNYC + KMIA blend_12)

Companion record:

- `documentation/mos/11_run_record_2026-03-02_cojoined_blend12_live_script_replay.md`
  - documents the replay backtest where daily forecasts are sourced by invoking the live inference script per target date.

This record closes the loop for the co-joined backtest:

- `all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c_with_balance.csv`

It documents:

1. which model artifacts were used by that backtest,
2. full retrain outputs for KNYC and KMIA,
3. exported live inference bundles (joblib),
4. live script defaults and leakage-proof outputs.

## 1) Exact Model Artifacts Used by the Backtest

The co-joined backtest script is:

- `backtesting/mos_blend12_knyc_kmia_cojoined_audit.py`

The model inputs are quantile prediction parquet artifacts (not serialized booster binaries):

- KNYC:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_12\dev_predictions.parquet`
  - `D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_12\test_predictions.parquet`
- KMIA:
  - `D:\Ahmed\data\kalshi\Experiments\MOS_KMIA\03_blends\blend_12\dev_predictions.parquet`
  - `D:\Ahmed\data\kalshi\Experiments\MOS_KMIA\03_blends\blend_12\test_predictions.parquet`

Model family/version semantics for both stations:

- family: `blend_12` (GFS 12Z + NAM 12Z blend),
- training split design: `dev=2022-01-01..2023-12-31`, `test=2024-01-01..2025-12-31`,
- generation pipeline: `ml/run_knyc_mos_first_plan.py`.

## 2) Retrain (Completed)

Executed commands:

```powershell
python ml/run_knyc_mos_first_plan.py --mos-csv "D:\Ahmed\data\kalshi\training_data\04_mos\archive_merged\KNYC_mos_archive_2000_2025.csv.gz" --truth-csv "D:\Ahmed\data\kalshi\training_data\02_truth\KNYC_settled_tmax.csv" --out-root "D:\Ahmed\data\kalshi\Experiments\MOS_RETRAIN_20260302\KNYC" --dev-start 2022-01-01 --dev-end 2023-12-31 --test-start 2024-01-01 --test-end 2025-12-31 --seed 42

python ml/run_knyc_mos_first_plan.py --mos-csv "D:\Ahmed\data\kalshi\training_data\04_mos\archive_merged\KMIA_mos_archive_2000_2025.csv.gz" --truth-csv "D:\Ahmed\data\kalshi\training_data\02_truth\KMIA_settled_tmax.csv" --out-root "D:\Ahmed\data\kalshi\Experiments\MOS_RETRAIN_20260302\KMIA" --dev-start 2022-01-01 --dev-end 2023-12-31 --test-start 2024-01-01 --test-end 2025-12-31 --seed 42
```

Completed retrain blend outputs:

- `D:\Ahmed\data\kalshi\Experiments\MOS_RETRAIN_20260302\KNYC\03_blends\blend_12\*`
- `D:\Ahmed\data\kalshi\Experiments\MOS_RETRAIN_20260302\KMIA\03_blends\blend_12\*`

Retrain summaries:

- `D:\Ahmed\data\kalshi\Experiments\MOS_RETRAIN_20260302\KNYC\09_reports\summary.json`
- `D:\Ahmed\data\kalshi\Experiments\MOS_RETRAIN_20260302\KMIA\09_reports\summary.json`

## 3) Exported Live Bundles (Completed)

Versioned live model bundles exported from training data:

- KNYC:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_12\live_model_bundle_v2_20260302`
- KMIA:
  - `D:\Ahmed\data\kalshi\Experiments\MOS_KMIA\03_blends\blend_12\live_model_bundle_v2_20260302`

Each bundle contains:

- `manifest.json`
- `blend_weights.json`
- `gfs_12\point_model.joblib`
- `gfs_12\q_0.05 ... q_0.95.joblib`
- `gfs_12\feature_medians.json`
- `nam_12\point_model.joblib`
- `nam_12\q_0.05 ... q_0.95.joblib`
- `nam_12\feature_medians.json`

## 4) Lineage Manifest (Hashes + Provenance)

Comprehensive lineage/export manifest:

- `D:\Ahmed\data\kalshi\Experiments\MOS_RETRAIN_20260302\model_lineage_and_exports_for_cojoined_ev0p30_win75_risk6.json`

This includes:

- backtest output paths,
- prediction input paths,
- retrain output roots,
- exported live bundle paths,
- SHA256 hashes and timestamps for all critical artifacts.

## 5) Live Script Implementation Status

Live script:

- `tools/live/mos_quantile_live_inference.py`

Default data root:

- `D:\Ahmed\data\live\mos_quantile_live_inference`

Default bundle inputs now point to the new exports:

- `...MOS\03_blends\blend_12\live_model_bundle_v2_20260302`
- `...MOS_KMIA\03_blends\blend_12\live_model_bundle_v2_20260302`

Leakage-proof outputs per run include:

- station inference blocks (`inference_kmia`, `inference_knyc`) with only:
  - point prediction,
  - quantiles (`q_0.05 ... q_0.95`),
  - no bucket probability table,
- per-feature and per-raw-field evidence tables,
- guardrail counters enforcing:
  - `runtime_utc <= quote_asof_utc`,
  - `runtime_utc == T-1 12:00:00Z`,
  - no non-leakage-free feature rows.

## 6) Required MOS Claim Anchors For This Backtest

For run:

- `cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c`

exact execution anchors are:

- entry gate rule:
  - `entry_timestamp >= max(T-1 12:00:00Z, market_open_utc + 30m)`
- stake sizing rule:
  - `stake = min(balance_before * 0.06, 500)`
- summary JSON:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\summary_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c.json`
- sanity JSON:
  - `D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\sanity_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p30_win75_risk6_cap500_minprice10c.json`
