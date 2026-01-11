# WX-204 — Baseline forecasters + baseline reports (climatology, simple MOS mean) for benchmarking

## Objective
Implement baselines so we can judge whether ML models add real value.

## Baselines (minimum)
1) Climatology baseline:
   - predict mean CLI Tmax by day-of-year (computed from training history per station)
2) Simple MOS mean baseline:
   - average of available MOS model tmax features (GFS/MEX/NAM/NBS/NBE)

## Evaluation
Compute:
- MAE, RMSE on test
- (Optional) bin event Brier score using a fixed sigma assumption

## Acceptance Criteria
- [ ] Baseline predictions produced for test dates.
- [ ] Baseline metrics written to:
  - `artifacts/<run_id>/baseline_metrics.json`
  - and included in report.md.

