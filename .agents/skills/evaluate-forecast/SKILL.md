---
name: evaluate-forecast
description: Evaluate continuous and Polymarket-bucket Tmax forecasts with rolling-origin, calibration, uncertainty, subgroup, and robustness diagnostics. Use for all model comparisons.
---

1. Confirm target parity and as-of audit status.
2. Load row-level predictions from immutable experiment output.
3. Evaluate continuous distribution with CRPS, MAE, RMSE, bias, interval coverage, and sharpness.
4. Evaluate contract probabilities with multiclass log loss, Brier score, calibration, and boundary-day diagnostics.
5. Use paired moving-block bootstrap uncertainty.
6. Report year, month, regime, temperature level, spread, and quality slices.
7. Compare against the frozen champion on exactly the same rows.
8. Report missing-row differences and common-sample metrics.
9. Run sensitivity to window, cutoff latency, calibration period, and major preprocessing choices.
10. Never select a winner on the locked test after repeated tuning.
