# References (primary)

## Model persistence / security
- scikit-learn: Model persistence guide (stable)
  https://scikit-learn.org/stable/model_persistence.html
- joblib persistence warning (pickle-based; arbitrary code execution risk)
  https://joblib.readthedocs.io/en/latest/persistence.html

## Time-series safe splits
- TimeSeriesSplit (stable)
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

## Probabilistic scoring metrics
- brier_score_loss (strictly proper scoring rule; lower is better)
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
- log_loss (cross-entropy / negative log-likelihood for discrete class probs)
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html

## Regression accuracy metrics
- mean_absolute_error
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
- root_mean_squared_error
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html

## Calibration diagnostics and methods
- calibration_curve (reliability diagram)
  https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html
- CalibrationDisplay (plotting reliability diagram)
  https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html
- Isotonic regression user guide
  https://scikit-learn.org/stable/modules/isotonic.html
- IsotonicRegression estimator
  https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html

