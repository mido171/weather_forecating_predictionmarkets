# WX-208 — Probability calibration for bin events (isotonic regression per bin) + reliability outputs

## Objective
Calibrate predicted probabilities for each bin-event so that:
- predicted probability ~ observed frequency

## Method (minimum)
For each bin-event:
- y_true_bin(T) = 1 if CLI Tmax in bin else 0
- y_prob_bin(T) = model predicted P(bin)

Fit IsotonicRegression on (y_prob_bin, y_true_bin) using training/validation only.

Reference:
- scikit-learn IsotonicRegression and isotonic regression guide.

## Requirements
- Calibration must be trained on a validation split only (no test contamination).
- Calibration artifacts persisted:
  - `artifacts/<run_id>/<station>/calibrators/<bin_id>.joblib`

## Acceptance Criteria
- [ ] Calibration improves or maintains Brier score on validation.
- [ ] Calibration curves are produced (reliability diagrams) pre- and post-calibration.
- [ ] Calibrator does not produce probabilities outside [0,1].

