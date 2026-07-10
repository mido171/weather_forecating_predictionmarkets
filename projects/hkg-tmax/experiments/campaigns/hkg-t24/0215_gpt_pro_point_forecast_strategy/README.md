# 0215 GPT-Pro HKO lead-1 point forecast strategy

Status: `historical_candidate_no_promotion`.

## Question and hypothesis

Test whether a leakage-safe residual system around the latest eligible HKO
lead-1 local forecast can materially improve HKG daily Tmax MAE and RMSE using
revision state, prior-year residual structure, T-2 target/climate state,
regimes, analogs, and a constrained stack.

## Historical as-of contract

- Target window: 2000-01-02 through 2023-12-31.
- Confirmation rows from 2024 onward were excluded.
- Cutoffs tested: 17:00, 18:00, 21:00, and 23:59 HKT on T-1.
- Forecast rows had to be eligible local lead-1 min/max forecasts by cutoff.
- Target and climate history used T-2 or older information.
- Residual statistics and analogs were prior-year/fold safe.

This late-cutoff contract is historical and does not replace the current H24N
15:00 HKT contract.

## Method

The runner built one row per target date and cutoff, evaluated expanding
yearly walk-forward folds for 2011-2023, compared official and climate
baselines, fitted residual candidates, and selected under predeclared
improvement gates. Lead-0 and external-source checks were diagnostic-only.

## Results

Selected historical candidate: `B3_grouped_residual_shrinkage` at 23:59 HKT.

| Metric | Candidate | Raw official | Delta |
|---|---:|---:|---:|
| Rows | 4,747 | 4,747 | — |
| MAE (C) | 0.92161 | 0.92749 | -0.00588 |
| RMSE (C) | 1.18268 | 1.19152 | -0.00884 |
| Bias (C) | 0.01270 | — | — |

The 0.035 C MAE and RMSE improvement gates failed. Bias and sample-size gates
passed. Leakage checks passed under this experiment's historical contract.

Weak slices remained material: cases at or above 34 C had roughly 1.1361 C
MAE and -1.0073 C bias; spring-transition MAE was roughly 1.1484 C.

## Decision and limitations

No promotion. The result is a small historical calibration gain, not a current
deployable H24N champion. The original GPT-Pro attachment was referenced by a
user-local path and is not tracked, so its source text cannot be reproduced
from Git.

## Reproduce

Set `HKG_TMAX_DATABASE_URL`, then run from `projects/hkg-tmax`:

```powershell
$env:HKG_TMAX_DATABASE_URL = '<local PostgreSQL URL>'
.\.venv\Scripts\python.exe scripts\run_hkg_t24_0215_gpt_pro_point_forecast_strategy.py
```

The command writes to this canonical experiment directory. It does not make
the historical cutoff current.

## Evidence map

- `results/selected_model_metadata.json`: selected model and gates.
- `results/summary.json`: headline counts and metrics.
- `results/model_scoreboard.csv` and `results/cutoff_scoreboard.csv`: full
  comparisons.
- `results/leakage_row_audit.csv` and `results/leakage_feature_audit.csv`:
  leakage evidence.
- `results/*_diagnostics.csv`: slice tables.

All ten retired Markdown records, including the exact duplicate final reports,
are indexed in [`DOCUMENT_PROVENANCE.csv`](../../DOCUMENT_PROVENANCE.csv).
