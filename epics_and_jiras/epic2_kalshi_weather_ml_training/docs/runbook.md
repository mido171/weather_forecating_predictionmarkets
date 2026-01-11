# Runbook (Epic #2)

## 1) Inputs required
- MySQL connection string
- station list (e.g., KMIA,KNYC,KPHL,KMDW,KLAX)
- date range (local dates)
- asof_policy_id (from DB; default created in Epic #1)

## 2) Typical commands (examples)
- Build dataset snapshot:
  python -m weather_ml.train --config configs/train_kalshi_default.yaml --stage dataset

- Train models + report:
  python -m weather_ml.train --config configs/train_kalshi_default.yaml --stage train

- Evaluate only:
  python -m weather_ml.train --config configs/train_kalshi_default.yaml --stage eval

## 3) Output artifacts
- artifacts/<run_id>/{models,report,metrics,metadata}
- datasets/<dataset_id>/{data.parquet,metadata.json} when running the dataset stage

## 4) Safety checks
- Pipeline should abort if any row violates:
  chosen_runtime_utc <= asof_utc

- Pipeline should log:
  station, date range, #rows, missing fraction, leakage check results.
