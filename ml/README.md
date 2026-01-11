# Weather ML (Epic 2)

This package hosts the Python ML training pipeline for Kalshi weather markets.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Run unit tests

```powershell
pytest
```

## Run training CLI (stub)

```powershell
python -m weather_ml.train --help
python -m weather_ml.train --config configs/train_kalshi_default.yaml --stage dataset
```