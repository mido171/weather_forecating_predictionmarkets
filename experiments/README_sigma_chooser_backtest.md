# Sigma Chooser Backtest

This experiment trains a mean model on 2021-2024, builds a global sigma and a LOW/MID/HIGH sigma chooser from out-of-fold residuals, then evaluates both on 2025.

## Setup

Make sure the ML package is installed (once):

```
python -m pip install -e ml
```

## Run

From repo root:

```
python experiments\sigma_chooser_backtest.py --config experiments\configs\sigma_chooser_backtest.json --run-id sigma_chooser_run
```

## Outputs

Results are written under:

```
experiments/results/sigma_chooser_backtest/<run_id>/
```

Key files:
- `config_resolved.json`
- `data_snapshot_meta.json`
- `mean_model_artifacts/mean_model.joblib`
- `calibration_residuals.csv`
- `calibration_global.json`
- `calibration_sigma_chooser.json`
- `predictions_2025.csv`
- `scores_summary.json`
- `scores_breakdown_by_bin.csv`
- `scores_breakdown_by_month.csv`
- `plots/coverage_comparison.png`
- `plots/sharpness_comparison.png`
- `plots/reliability_thresholds.png`
- `plots/crps_timeseries.png`
- `plots/pit_histogram.png`
