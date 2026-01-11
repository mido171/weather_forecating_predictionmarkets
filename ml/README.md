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

## Dataset extraction (WX-202)

The dataset extraction layer uses SQLAlchemy. Configure the database URL via
`WEATHER_ML_DB_URL` or pass an engine to `weather_ml.dataset.build_dataset`.

```python
from datetime import date

from weather_ml import dataset

df = dataset.build_dataset(
    ["KMIA"],
    date(2024, 1, 1),
    date(2024, 1, 31),
    1,
)
```
