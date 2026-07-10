# Evaluation and Baseline Protocol

## Baseline hierarchy

The specification names a primary baseline and required secondary baselines. Primary comparison always uses the candidate's exact scored rows. Common baselines are raw official forecast, source-aware residual memory, target-memory model, current canonical champion, and a simple model using the same feature family.

## Metrics

At minimum save `n`, MAE, RMSE, bias, median absolute error, P90/P95/max absolute error, severe-error counts and rates, hot-underforecast and cold-overforecast diagnostics, and coverage. For every metric report candidate, baseline, and candidate-minus-baseline delta. Negative MAE delta means improvement.

## Temporal integrity

Use expanding or rolling walk-forward. Record each prediction's training end date. No random K-fold. Model and feature tuning must be nested. For adaptive online states, prediction precedes update.

## Stability

Report temporal folds, calendar years, MAM/JJA/SON/DJF, source/source-era, late window, and predeclared meteorological regimes. Promotion requires declared counts and stability. A globally positive result with severe seasonal damage may fail the no-harm gate.

## Multiplicity

When many features, interactions, thresholds, or models are screened, save the full candidate table and attempt count. Declare the selection rule. Use permutation/null, FDR, nested selection, or a conservative holdout as specified.

## Comparability

Never compare an old published MAE with a new candidate unless both are replayed on the same frame. Label non-comparable scores as context only.
