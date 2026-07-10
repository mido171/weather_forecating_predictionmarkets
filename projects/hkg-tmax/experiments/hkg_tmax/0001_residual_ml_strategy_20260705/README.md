# 0001 Residual ML Strategy

Status: completed.

This is the prior broad residual-ML experiment that built leakage-safe HKG Tmax features and scored A0-A8:

- A0 raw official anchor
- A1 grouped residual
- A2 forecast revision residual LGBM
- A3 HKO hourly-state residual LGBM
- A4 station-network residual LGBM
- A5 text/warning residual LGBM
- A6 target-memory residual LGBM
- A7 final residual ensemble
- A8 direct absolute LGBM diagnostic

Canonical artifacts remain in the original compatibility folder:

`experiments/hkg_tmax_residual_ml_strategy/results/`

Key files there:

- `summary.json`
- `final_model_card.md`
- `scoreboard.csv`
- `scoreboard_by_split.csv`
- `scoreboard_by_month.csv`
- `scoreboard_by_regime.csv`
- `prediction_rows.csv`
- `prediction_rows.parquet`
- `feature_matrix_trainval.parquet`
- `feature_matrix_presealed_holdout.parquet`
- `feature_matrix_sealed_confirmation.parquet`
- `feature_lineage.json`
- `leakage_audit.json`
- `artifact_manifest.csv`

Outcome: no promote. The best primary A7 result improved MAE by about `0.0322 C` versus raw official at T-1 23:59 HKT, which was useful calibration but not a trading-grade edge.
