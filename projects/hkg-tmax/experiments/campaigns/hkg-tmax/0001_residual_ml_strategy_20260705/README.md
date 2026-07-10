# 0001 Broad residual-ML strategy

Status: `complete_no_promote_cosmetic`.

## Question and method

Test leakage-safe residual correction around the latest eligible Info.gov
local forecast maximum. The A0-A8 ladder added grouped calibration, forecast
revision, HKO hourly state, station network, text/warnings, lag-safe target
memory, a constrained ensemble, and a direct-absolute diagnostic.

Primary target: HKO Daily Extract absolute daily maximum. Primary historical
cutoff: T-1 23:59 HKT. Sealed 2024+ rows were confirmation-only and target
history used a lag-2 floor.

## Result

| Model | Rows | MAE | RMSE | p90 AE | Bias |
|---|---:|---:|---:|---:|---:|
| A0 raw official | 5,629 | 0.930858 | 1.195757 | 2.000000 | -0.122935 |
| A7 final residual ensemble | 5,629 | 0.898665 | 1.154088 | 1.904225 | -0.000486 |

MAE improved by 0.032193 C. Leakage passed with zero violations, but the gain
missed the predeclared 0.035 C meaningful-edge gate.

## Decision

No promotion. A7 is a historical research reference, not a production/trading
release. The signal mostly removed bias and supported narrower no-harm
follow-ups rather than another broad model sweep.

## Reproduce and evidence

```powershell
.\.venv\Scripts\python.exe scripts\run_hkg_tmax_residual_ml_strategy.py --config config\experiments\hkg_tmax\residual_ml_strategy.yaml --output-dir experiments\campaigns\residual-modeling\strategy\results
```

The detailed machine evidence and frozen source specification are indexed by
[the residual-strategy dossier](../../residual-modeling/strategy/README.md).
