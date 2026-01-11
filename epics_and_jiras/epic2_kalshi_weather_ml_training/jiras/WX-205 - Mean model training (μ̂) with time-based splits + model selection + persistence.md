# WX-205 — Mean model training (μ̂) with time-based splits + model selection + persistence

## Objective
Train a model to predict CLI Tmax (settlement) from MOS as-of features.

## Requirements
- Use time-based splitting (no shuffle).
  - Either explicit date cutoffs or TimeSeriesSplit.
  - Reference: scikit-learn TimeSeriesSplit.
- Candidate estimators (minimum):
  1) Ridge regression (baseline strong)
  2) Gradient boosting regressor (optional but recommended)

- Support per-station training:
  - either train separate models per station OR include station_id as feature.
  - Start with separate models (simpler).

- Persist the trained mean model:
  - `artifacts/<run_id>/<station>/model_mean.joblib`
  - include metadata.json with versions and feature list

## Acceptance Criteria
- [ ] Training produces a mean model for each station in config.
- [ ] Report includes MAE/RMSE on validation and test.
- [ ] Time-based split is proven (no leakage).
- [ ] Artifacts saved with joblib and metadata includes sklearn version.

