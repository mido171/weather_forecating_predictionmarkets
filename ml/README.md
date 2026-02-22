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

## Time feature sweep trees (research runs)

Run the time-feature sweep runner across tree models and write results under `artifacts/time_feature_sweep_trees/...`:

```powershell
python ..\\scripts\\run_time_feature_sweep_trees.py --config ml\\configs\\train_mean_sigma_gribstream_cli.yaml --models xgb `
  --experiment-ids EX201
```

### RS-MoE mean model (EX201)

EX201 runs the **RS‑MoE** mean model (3-class bust gate + 3 expert regressors, OOF weights + temperature scaling) and writes:
- `gate_model.cbm`, `expert_*_model.joblib`, `gate_calibration.json`
- `oof_gate_logits_train.parquet`, `oof_gate_probs_train.parquet`
- `predictions_test.parquet` with additional columns: `p_cool/p_normal/p_warm`, `mu_cool/mu_normal/mu_warm`, `gate_temperature`, `model_type`

Switch expert objective variant in the YAML config:

```yaml
rs_moe:
  experts:
    objective_variant: quantile_median  # or: absoluteerror
```

Inspect outputs in the experiment run folder, e.g.:
- `artifacts/time_feature_sweep_trees/<timestamp>/xgb/EX201/metrics.json`
- `artifacts/time_feature_sweep_trees/<timestamp>/xgb/EX201/report.md`
- `artifacts/time_feature_sweep_trees/<timestamp>/xgb/EX201/oof_gate_probs_train.parquet`

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

## MOS dataset builder (KMIA Tmax next-day)

Generate the MOS-based feature dataset directly from `mos_daily_value` and
`station_daily_truth`:

```powershell
python -m weather_ml.mos_dataset_builder --config configs/mos_kmia_tmax_v1.json --db-config <path-to-db.json>
```

The default config enforces the **T-1 12Z as-of rule** (for target date D, as-of
timestamp is D-1 at 12:00Z) and uses only GFS/NAM MOS runs.

DB config JSON:

```json
{
  "host": "localhost",
  "port": 3306,
  "database": "weather_predictionmarkets",
  "user": "user",
  "password_env": "WEATHER_DB_PASSWORD"
}
```

Truth ingestion (IEM daily API):

```powershell
python -m weather_ml.mos_truth_ingest --station-id KMIA --station-zoneid America/New_York `
  --network FL_ASOS --source-station MIA --start-date 2007-01-01 --end-date 2025-12-31 `
  --db-config <path-to-db.json>
```
