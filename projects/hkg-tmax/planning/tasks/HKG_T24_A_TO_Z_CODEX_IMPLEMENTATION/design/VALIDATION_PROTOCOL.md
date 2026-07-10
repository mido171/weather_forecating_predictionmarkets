# Validation, Sealing, and Promotion Protocol

## Development frames

1. `LONG_TARGET_FRAME`: long HKO history for causal climatology and target-memory research.
2. `OFFICIAL_ANCHOR_DEV`: 2000-01-02 through 2023-12-31, clean official forecasts selected at cutoff.
3. `CORE_NWP_DEV`: 2021-03-22 through 2023-12-31, official + GFS + GEFS common strict frame.
4. `DIAGNOSTIC_PHYSICS`: timestamp-blocked sources for mechanism discovery only.
5. `LIVE_PROSPECTIVE`: exact first-seen sources collected after implementation.

## Nested walk-forward

All feature fitting, normalization, selector choice, hyperparameter tuning, regime thresholds, router parameters, and specialist caps occur inside training folds. Use expanding windows with adequate warm-up. Preserve every OOF prediction.

## Sealed sequence

- Develop and select architecture on pre-2024 only.
- Freeze code, features, hyperparameters, model list, and acceptance criteria.
- Open 2024 once as locked validation.
- If the predeclared gate passes, refit under the same rules through 2024 and open 2025 once as final historical test.
- Only after that may 2026 YTD be used as an additional untouched temporal replay, and it must not be called prospective unless timestamped predictions were actually issued live.
- Any prior access to a period must be declared; contaminated periods cannot be called confirmation.

## Metrics

Point metrics: MAE, RMSE, bias, median AE, P75/P90/P95 AE, hot-underforecast and cold-overforecast errors. Probability metrics: CRPS, pinball loss, Brier score, log loss and calibration. Report yearly, monthly, seasonal, source, regime, and error-tail slices.

## Baselines

- causal climatology;
- long-history target memory;
- official raw at cutoff;
- official plus causal residual memory;
- direct GFS Tmax;
- GFS MOS;
- GEFS median and calibrated median;
- static blend;
- previous trustworthy champion.

## Promotion

No candidate is promoted unless it improves the relevant baseline on identical rows, survives multiple folds, does not materially worsen tails, passes leakage audit, and has complete artifacts. A 0.45°C target is aspirational, not a license to tune on sealed outcomes.
