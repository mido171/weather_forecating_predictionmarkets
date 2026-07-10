# KNYC Gribstream V1

This package implements a leakage-safe, non-ML KNYC Tmax backtester and daily predictor under:

`C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets\backtesting\gribstream\V1`

The SQLite database lives at:

`C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets\backtesting\gribstream\V1\sqlite\knyc_gribstream_v1.sqlite3`

## Leakage Safety

V1 uses the Gribstream `timeseries` endpoint with an explicit `asOf` timestamp. For each settlement date `D`, every request uses:

- `asOf = D 13:00:00Z`
- `fromTime = America/New_York midnight at start of D, converted to UTC`
- `untilTime = America/New_York midnight at start of D+1, converted to UTC`

That makes each request a reconstruction of what was knowable at exactly `13:00:00Z` on the same day being predicted. Rolling bias corrections and rolling weights are trained only on prior dates `P < D`, so the calibration layer is also leakage-safe.

## Evaluation Window

- Warmup range: `2022-01-01` through `2022-12-31`
- Scored backtest range: `2023-01-01` through `2024-12-31`

The 2022 warmup data is used only for historical error formation, rolling bias, and rolling weights. It is intentionally excluded from the scored summary metrics.

## Model Roles

The model catalog is curated in code. Historical scoring only includes models marked `backtest` or `backtest_partial`. Models marked `live_only` remain in the catalog and are supported by `predict-date`, but they are excluded from the 2023-2024 historical score window because they do not have archive coverage there.

## Environment

Set the Gribstream bearer token in:

`GRIBSTREAM_API_TOKEN`

No token is hardcoded. The client reads that environment variable directly.

## CLI

Run from the repo root.

Initialize the database and seed the model catalog:

```powershell
python -m backtesting.gribstream.V1.cli init-db
```

Fetch NWS settlement truth for `2022-01-01` through `2024-12-31`:

```powershell
python -m backtesting.gribstream.V1.cli fetch-truth
```

Fetch historical Gribstream forecasts with multithreading:

```powershell
python -m backtesting.gribstream.V1.cli fetch-forecasts --threads 2
```

Derive one daily Tmax per model per date and compute raw model errors:

```powershell
python -m backtesting.gribstream.V1.cli derive-daily
```

Run the rolling-weight backtest and populate prediction tables:

```powershell
python -m backtesting.gribstream.V1.cli backtest
```

Export the required CSV files into the SQLite directory:

```powershell
python -m backtesting.gribstream.V1.cli export
```

Run the full historical pipeline end to end:

```powershell
python -m backtesting.gribstream.V1.cli run-all --forecast-threads 2
```

Generate a leakage-safe prediction for a single target date:

```powershell
python -m backtesting.gribstream.V1.cli predict-date --date 2025-03-01 --threads 2
```

`predict-date` always stores the fetches, derived daily model rows, model weights, and prediction components for the requested date. It stores a row in `daily_predictions` only when truth exists for that date, because `daily_predictions.actual_tmax_f` is intentionally non-null.

## Outputs

The `export` step writes:

- `nws_daily_settlements.csv`
- `gribstream_raw_forecasts.csv`
- `daily_model_tmax.csv`
- `model_daily_errors.csv`
- `daily_model_weights.csv`
- `daily_prediction_components.csv`
- `daily_predictions.csv`
- `metrics_summary.csv`
- `coverage_summary.csv`

## Notes

- SQLite uses `WAL` journal mode and `NORMAL` synchronous mode.
- Forecast fetching is multithreaded, but SQLite writes are serialized on the main thread.
- Reruns are safe because truth rows are upserted, request rows are upserted, and raw forecast rows use a natural-key uniqueness constraint.
