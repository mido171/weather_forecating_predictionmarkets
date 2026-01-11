# Epic #2 Architecture (Python training inside the same repo)

## Folder layout (recommended)
repo-root/
  ml/
    pyproject.toml OR requirements.txt
    src/weather_ml/
      __init__.py
      config.py
      db.py
      dataset.py
      features.py
      models_mean.py
      models_sigma.py
      distribution.py
      calibration.py
      metrics.py
      report.py
      train.py
    tests/
  docs/ (from Epic #1 + Epic #2 docs)
  models/ (Java module from Epic #1)
  ingestion-service/ (Java module from Epic #1)

## Main pipeline stages
1) Extract dataset from MySQL
2) Validate dataset:
   - no leakage (chosen_runtime_utc <= asof_utc)
   - no missing keys
3) Split time-wise:
   - train / val / test based on target_date_local
4) Train μ model
5) Compute residuals and train σ model
6) Produce discrete probabilities:
   - integer temperatures
   - and/or Kalshi bins
7) Calibrate probabilities per bin (isotonic)
8) Compute metrics and write report
9) Persist artifacts + metadata; optionally write to DB registry tables

## Station strategy
Start with per-station models for simplicity:
- Train one μ/σ model pair per station and as-of policy.
Optionally later:
- multi-station model with station_id categorical feature.

