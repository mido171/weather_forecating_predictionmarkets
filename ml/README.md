# Weather ML (Epic 2)

This package hosts the Python ML training pipeline for Kalshi weather markets.
It trains mean and uncertainty models from the CSV snapshot produced by the
ingestion service.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Optional gradient-boosting dependencies:

```powershell
python -m pip install -e ".[gbdt]"
```

## Run unit tests

```powershell
pytest
```

## Train models

```powershell
python -m weather_ml.train --config configs/train_mean_sigma.yaml
```

## Run inference

```powershell
python -m weather_ml.predict --run-dir artifacts/runs/<run_id> --csv <input.csv> --output predictions.parquet
```

## Input dataset

CSV location (default config):
`ingestion-service/src/main/resources/trainingdata_output/gribstream_training_data.csv`

Expected columns:
- station_id, target_date_local, asof_utc
- gfs_tmax_f, nam_tmax_f, gefsatmosmean_tmax_f, rap_tmax_f, hrrr_tmax_f, nbm_tmax_f
- gefsatmos_tmp_spread_f, actual_tmax_f

## Outputs

Artifacts are written under `artifacts/runs/<run_id>/` including:
- resolved config, dataset metadata + hash, feature list
- trained mean/sigma models (joblib)
- metrics.json and report.md
- plots and test-set predictions
