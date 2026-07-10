# Router Training Specification

## Goal

Map facts known at T−24 to a conservative weight distribution over expert forecasts. The router is trained only on genuine out-of-fold expert predictions from dates where the competing experts all exist.

## Router table

For each target date store:

```text
target_date, cutoff_utc
expert_*_oof_point_forecast
expert_*_oof_absolute_error       # training-only
expert_*_predicted_error
expert_availability_mask
model disagreement features
ensemble spread/skew features
station disagreement/regime features
official revision and text features
target-memory regime
recent causal source performance
actual_tmax                        # training-only and sealed as required
```

## Core router date range

The corrected official archive is near-continuous. The first core modern common frame begins when GFS is available: 2021-03-22. Use only pre-2024 dates for router development unless the sealed protocol explicitly opens later periods.

## Expected-error routing

For expert e and date t:

```text
loss_e_t = abs(actual_tmax_t - oof_forecast_e_t)
predicted_loss_e_t = h_e(facts_t)
```

Train one strongly regularized expected-loss model per expert. Convert losses to dynamic weights:

```text
dynamic_weight_e = exp(-predicted_loss_e / temperature) / sum_j(...)
```

Learn stable static non-negative weights from OOF predictions by minimizing MAE. Final weights are shrunk:

```text
final_weight_e = (1-lambda) * static_weight_e + lambda * dynamic_weight_e
```

Tune `temperature` and `lambda` inside inner temporal folds. Enforce non-negative weights, sum to one, per-expert caps, minimum history, and availability masks.

## Abstention

Official raw or a stable core blend must be an explicit expert. Train expected benefit of dynamic routing versus the stable baseline. When expected benefit is non-positive, support is weak, inputs are missing, or disagreement is extreme without directional evidence, abstain to the stable blend.

## New-model adapters

IFS/AI/ARWF/CWA experts are introduced as capped adapters:

```text
forecast_new = (1-rho) * core_forecast + rho * challenger_forecast
```

`rho` starts at zero/shadow and increases only after locked out-of-sample evidence. Never retrain the 2021–2023 core router as if a 2025 model existed historically.

## Router diagnostics

Report weight distributions, weight turnover, expert win rates, predicted-vs-realized error calibration, regime slices, missing-source behavior, and counterfactual performance of static-only versus dynamic routing.
