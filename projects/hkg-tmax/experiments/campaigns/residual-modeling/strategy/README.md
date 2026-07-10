# Broad HKG Tmax residual-ML strategy

Status: `complete_no_promote_cosmetic`.

## Question and contract

Test leakage-safe residual correction around the latest eligible Info.gov
local forecast maximum. The primary target was HKO Daily Extract absolute daily
maximum temperature and the primary cutoff was T-1 23:59 HKT. Target-history
predictors used a lag-2 floor; sealed 2024+ rows were confirmation-only; no
post-cutoff forecasts/hourly observations or raw Daily Extract payload rows
were predictors.

This is historical late-cutoff research and does not replace the current H24N
contract.

## Method

The A0-A8 ladder compared raw official, grouped residual calibration, revision,
hourly-state, station-network, text/warning, target-memory, final constrained
residual ensemble, and direct absolute LGBM. The run used 323 features and
5,629 primary scored rows. T-1 15:00 had no strict eligible anchor folds.

## Primary result

| Model | MAE | RMSE | p90 AE | Bias |
|---|---:|---:|---:|---:|
| A0 raw official | 0.930858 | 1.195757 | 2.000000 | -0.122935 |
| A7 final residual ensemble | 0.898665 | 1.154088 | 1.904225 | -0.000486 |

MAE improved by 0.032193 C, RMSE by 0.041669 C, and p90 absolute error by
0.095775 C. Leakage violations were zero. The gain missed the predeclared
0.035 C meaningful-edge threshold, so the outcome was
`no_promote_cosmetic`.

Most practical gain appeared by the hourly-state stage. Station gradients,
text/warnings, target memory, and the final ensemble added little afterward;
direct absolute prediction underperformed residual modeling.

## Decision

This was a clean calibration improvement, not a trading-grade breakthrough.
It motivated selective no-harm routing and tail-specialist follow-ups rather
than another broad generic model sweep.

## Reproduce

From `projects/hkg-tmax` with `HKG_TMAX_DATABASE_URL` set:

```powershell
.\.venv\Scripts\python.exe scripts\run_hkg_tmax_residual_ml_strategy.py --config config\experiments\hkg_tmax\residual_ml_strategy.yaml --output-dir experiments\campaigns\residual-modeling\strategy\results
.\.venv\Scripts\python.exe -m pytest tests\test_hkg_tmax_residual_ml_strategy.py
```

## Evidence map

- `results/summary.json`, `scoreboard.csv`, and `scoreboard_by_split.csv`.
- `results/leakage_audit.json` and `row_count_audit.json`.
- `results/ablation_scoreboard.csv` and feature-importance artifacts.
- `inputs/hkg_tmax_ml_strategy_codex_implementation_20260705.txt`: exact frozen
  source specification, SHA-256
  `cad74f64eb9250d073a00c08bc47245bb50f1274f70aa56e8924b75e21a62e63`.

The old handoff prose, duplicated result cards, live-trading context copies,
and noisy Git-status snapshots were retired. Their exact hashes and recovery
commit are in [`DOCUMENT_PROVENANCE.csv`](../../DOCUMENT_PROVENANCE.csv).
