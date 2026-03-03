# KNYC MOS-First Runtime Matrix Run Record (2026-03-01)

## Summary

This run implements and executes a MOS-first, leakage-safe chronological benchmark suite for KNYC day-T Tmax:

- Runtime slices: `GFS 00/06/12/18`, `NAM 00/12`
- Blend slices: `GFS+NAM @00Z`, `GFS+NAM @12Z` (prediction-stage blend, dev-tuned)
- Baselines: raw MOS (`mos_tmax_raw`) per slice
- Residual models: LightGBM point + quantile residual models
- KNN ablation on top-2 residual slices: `K0/K1/K2/K3`
- Calibration diagnostics: pinball, interval coverage, PIT KS, threshold Brier/logloss/ECE

## Code + Command

- Runner: `ml/run_knyc_mos_first_plan.py`
- Executed command:

```powershell
python ml/run_knyc_mos_first_plan.py `
  --mos-csv D:\Ahmed\data\kalshi\training_data\KNYC_mos_archive_2000_2025.csv.gz `
  --truth-csv D:\Ahmed\data\kalshi\training_data\KNYC_settled_tmax.csv `
  --out-root D:\Ahmed\data\kalshi\Experiments\MOS
```

## Data Inputs

- MOS archive: `D:\Ahmed\data\kalshi\training_data\KNYC_mos_archive_2000_2025.csv.gz`
- Truth: `D:\Ahmed\data\kalshi\training_data\KNYC_settled_tmax.csv`

## Date Protocol

- Dev OOF window: `2022-01-01 .. 2023-12-31` (monthly expanding chronological OOF)
- Test window: `2024-01-01 .. 2025-12-31` (frozen design)
- Train starts:
  - Common slices (`GFS00/NAM00/GFS12/NAM12`): `2009-01-01`
  - GFS-only slices (`GFS06/GFS18`): `2004-01-01`

## Key Output Root

- `D:\Ahmed\data\kalshi\Experiments\MOS`

Subfolders:

- `00_data` (slice row datasets)
- `01_phaseA_raw` (raw baseline metrics)
- `02_phaseB_residual` (residual model predictions + metrics)
- `03_blends` (blend outputs + weights)
- `04_knn_ablation` (KNN K0/K1/K2/K3 outputs)
- `09_reports` (comparison + summary + executive markdown)

## Headline Result

As recorded in `09_reports/summary.json` and `09_reports/final_comparison.csv`, the best test MAE row in this run is:

- family: `blend_ml`
- slice: `blend_00`
- test MAE: `1.7072`
- test avg pinball: `0.4972`

## Notes

- Runtime-day alignment is strict (`runtime_local_date == target_date_local - 1 day`).
- KNN candidate policy enforces `candidate_date < query_date`.
- Blend weights are tuned on dev only and then frozen for test scoring.

