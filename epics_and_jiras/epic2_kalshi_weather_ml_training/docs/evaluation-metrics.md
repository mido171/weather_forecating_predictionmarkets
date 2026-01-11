# Evaluation Metrics & What They Mean (for trading relevance)

## 1) Point forecast accuracy (μ̂ only)
Compute on test set:
- MAE (mean_absolute_error): lower is better
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
- RMSE (root_mean_squared_error): penalizes big misses
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html

## 2) Probabilistic forecast quality (μ̂ + σ̂)
Kalshi trades on probabilities. You need to score the distribution, not just the mean.

### 2.1 Log loss / negative log-likelihood on discrete integer temperatures
Treat each integer temperature as a class and compute log_loss on the probability assigned to the realized class.
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html

### 2.2 Brier score loss for each bin-event (e.g., “81–82°F”)
For each bin event, treat as binary classification:
- y_true = 1 if Tmax in bin else 0
- y_prob = predicted P(bin)
Then compute brier_score_loss.
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html

Brier is a strictly proper scoring rule and measures mean squared error of probabilities.

## 3) Calibration diagnostics (are probabilities honest?)
Calibration curves aka reliability diagrams can be produced from (y_true, y_prob).
https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html

You can also plot with CalibrationDisplay:
https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html

## 4) Time-series split (avoid leakage)
Never shuffle. Use time-ordered splits or TimeSeriesSplit.
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

