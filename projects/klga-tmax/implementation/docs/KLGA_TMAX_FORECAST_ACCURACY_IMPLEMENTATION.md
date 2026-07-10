# KLGA Tmax Forecast Accuracy Implementation

This implementation adds the local, leakage-safe KLGA Tmax forecasting and Wunderground-settlement accuracy path under `bootstrap/klga_tmax/implementation`.

## Scope

- Polymarket backtesting and trade simulation are intentionally deferred.
- The implemented evaluation objective is forecast accuracy for KLGA daily Tmax on target day `T`.
- The source of truth is current settled Wunderground KLGA Tmax in `silver.target_daily_actuals`.
- Feature materialization enforces source availability by cutoff and does not feed the target-day settled WU label into features.

## Implemented Components

- Forecast/evaluation Alembic migration `0009_forecast_eval.py`.
- Expanded `db inspect-contract` table, index, and view checks.
- Canonical acquisition-normalization map in `config/acquisition_table_map.yaml`.
- Acquisition normalization into canonical WU actual, station observation, and MOS guidance tables.
- Leakage-aware materialization context and strategy feature builder.
- Regime, risk, staleness, disagreement, and availability features.
- Full-grid PMF utilities over `TEMP_GRID_F = 50..115`.
- Nine deterministic expert PMF forecasters and a static regularized combiner.
- Calibration against settled WU labels.
- Forecast-accuracy evaluation and report generation.
- CLI commands for materialization, prediction, calibration, evaluation, report generation, and settlement updates.

## Verified Local Run

Environment:

```powershell
$env:KLGA_DB_URL='postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research'
```

Verified command sequence:

```powershell
python -m compileall -q src tests
python -m pytest -q
python -m klga_tmax.cli --help
python -m klga_tmax.cli db migrate
python -m klga_tmax.cli db inspect-contract
python -m klga_tmax.cli features materialize --start-date 2026-06-20 --end-date 2026-06-27 --cutoff-id T_1245UTC --feature-version supplemental_doc_1_v1 --replace
python -m klga_tmax.cli train experts --start-date 2026-06-20 --end-date 2026-06-27 --cutoff-id T_1245UTC --fold-policy annual_walk_forward
python -m klga_tmax.cli predict oof --start-date 2026-06-20 --end-date 2026-06-27 --cutoff-id T_1245UTC
python -m klga_tmax.cli train combiner --start-date 2026-06-20 --end-date 2026-06-27 --cutoff-id T_1245UTC
python -m klga_tmax.cli calibrate --start-date 2026-06-20 --end-date 2026-06-27 --cutoff-id T_1245UTC --prediction-kind oof
python -m klga_tmax.cli evaluate accuracy --start-date 2026-06-20 --end-date 2026-06-27 --cutoff-id T_1245UTC --prediction-kind oof
python -m klga_tmax.cli report generate --run-id forecast_eval_oof_20260702T064929Z
python -m klga_tmax.cli evaluate day --target-date 2026-06-27 --cutoff-id T_1245UTC --prediction-kind oof
python -m klga_tmax.cli forecast run --target-date 2026-06-27 --cutoff-id T_1245UTC
python -m klga_tmax.cli settlement update --start-date 2026-06-20 --end-date 2026-06-27
```

Live OOF forecast-accuracy result for `2026-06-20..2026-06-27`, cutoff `T_1245UTC`:

- Row count: 8
- MAE: 1.7934024485395277 F
- RMSE: 2.3709179423541316 F
- Bias: -0.46587591411087814 F
- Exact-degree hit rate: 0.25
- Within 1 F hit rate: 0.5
- Within 2 F hit rate: 0.625
- Prediction interval coverage: 1.0
- Mean log score: 2.918826397334783
- Mean discrete CRPS: 0.0314168860097144

Report artifacts:

```text
artifacts/klga_tmax/reports/forecast_accuracy/forecast_eval_oof_20260702T064929Z/summary.json
artifacts/klga_tmax/reports/forecast_accuracy/forecast_eval_oof_20260702T064929Z/daily_scores.csv
artifacts/klga_tmax/reports/forecast_accuracy/forecast_eval_oof_20260702T064929Z/report.md
```
