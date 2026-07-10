# 0003 Official residual memory

Status: `complete_no_promote`.

## Hypothesis and as-of contract

Test whether lag-safe memory of prior official-forecast residuals improves A7.
For target date T, the newest eligible residual source date was T-2; lag-1
residuals were disabled. Sealed rows were report-only.

## Result

| Model | MAE | RMSE | p90 AE |
|---|---:|---:|---:|
| D4 constrained memory stack | 0.898479 | 1.153563 | 1.904831 |
| D5 conservative A7-memory blend | 0.898594 | 1.154004 | 1.905844 |
| A7 reference | 0.898665 | 1.154088 | 1.904225 |
| A0 raw official | 0.930858 | 1.195757 | 2.000000 |

D5's overall improvement over A7 was only 0.0000715 C. Development,
presealed gain, and presealed p90 gates failed. Leakage, publication safety,
row identity, and slice no-harm checks passed.

## Decision

No promotion. The memory signal was real but too small and unstable to change
deployment; raw official remained the deployment baseline under the historical
contract and A7 remained research-only.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\run_hkg_tmax_residual_ml_official_memory.py --config config\experiments\hkg_tmax\residual_ml_official_memory.yaml --output-dir experiments\campaigns\hkg-tmax\0003_official_residual_memory_20260706\results --no-compat-copy
```

## Evidence map

- `results/summary.json`, `scoreboard.csv`, and split scoreboards.
- `results/leakage_audit.json`.
- `results/residual_memory_publication_safety_audit.json`.
- `results/row_identity_gate.json`.
- `inputs/gpt_pro_point_forecast_ml_strategy_deep_analysis_next_round_spec_20260706.txt`.
