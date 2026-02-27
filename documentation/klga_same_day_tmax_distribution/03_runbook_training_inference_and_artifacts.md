# 03 - Runbook: Training, Inference, and Artifacts

This runbook is for running, monitoring, validating, and recovering KLGA pipeline jobs.

It covers:

- the primary LGBM pipeline (`ml/run_klga_daily_tmax_dist.py`)
- the portable CSV exporter (`ml/run_klga_data_exporter.py`)
- the experimental TabM-from-exports trainer (`ml/run_training_tabm_klga_from_exports.py`)

## 1) Prerequisites

From repo root:

- Python environment with required dependencies installed.
- MySQL reachable with expected KLGA tables populated.
- Existing feature store available unless you intentionally rebuild.

Default DB connection path:

- from env vars in `db.py`
- fallback defaults in code

## 2) Main command matrix

## 2.1 Full pipeline with analog enabled

```powershell
python ml/run_klga_daily_tmax_dist.py --output-root artifacts/same_day_res_poly
```

## 2.2 LGBM-only mode (skip analog blend)

```powershell
python ml/run_klga_daily_tmax_dist.py `
  --output-root artifacts/same_day_res_poly `
  --skip-analog-blend
```

Use this when you want:

- faster completion
- clean peak+delta export without analog dependency

## 2.3 Force feature-store rebuild

```powershell
python ml/run_klga_daily_tmax_dist.py `
  --output-root artifacts/same_day_res_poly `
  --force-rebuild-dataset
```

## 2.4 High-granularity logging mode

```powershell
python ml/run_klga_daily_tmax_dist.py `
  --output-root artifacts/same_day_res_poly `
  --log-level INFO `
  --log-every-rows 500 `
  --log-every-seconds 5 `
  --peak-train-log-period 10 `
  --delta-train-log-period 5 `
  --train-log-every-seconds 5 `
  --train-heartbeat-seconds 5
```

## 2.5 Export raw inputs to a portable CSV bundle (DB -> files)

Use this when you want to train/evaluate on another machine without DB access.

```powershell
python ml/run_klga_data_exporter.py --log-level INFO
```

Outputs a folder containing:

- `daily_max_truth_klga.csv`
- `observations_30m_required_columns.csv`
- `station_universe.csv`
- `export_manifest.json`

Important:

- these exports can be large (hundreds of MB)
- do not commit them to Git

Details:

- `07_exporter_and_remote_training_tabm.md`

## 2.6 Train TabM (tabular neural net) from exported CSVs

Use this when you want to compare a tabular NN against the LGBM baseline using the same features/splits.

```powershell
python ml/run_training_tabm_klga_from_exports.py `
  --data-dir "D:\\path\\to\\export_bundle_folder" `
  --log-level INFO
```

Behavior:

- checks and installs missing Python dependencies via pip (including torch/tabm)
- writes a full run folder under:
  - `<data-dir>/training_results_tabm/<RUN_ID>/`

Details + known Windows issues:

- `07_exporter_and_remote_training_tabm.md`
- (troubleshooting is merged into `07_exporter_and_remote_training_tabm.md`)

## 3) What gets reused vs recomputed

If feature store exists and `--force-rebuild-dataset` is not set:

- stage 1 reuses existing feature store
- model training still reruns
- evaluation and artifact writing rerun

So reruns are much faster than full rebuild.

## 4) Stage flow by mode

## 4.1 Analog enabled

1. build feature store
2. load feature store
3. prepare splits and features
4. train peak
5. predict peak
6. train delta
7. predict delta
8. build analog library
9. analog K selection
10. blend posteriors
11. evaluate metrics
12. write artifacts

## 4.2 Analog disabled

1. build feature store
2. load feature store
3. prepare splits and features
4. train peak
5. predict peak
6. train delta
7. predict delta
8. evaluate metrics
9. write artifacts

## 5) Monitoring commands

Tail active run log:

```powershell
Get-Content artifacts/same_day_res_poly/<RUN_ID>/run.log -Wait
```

Check python processes:

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,StartTime,CPU
```

Check latest run dirs:

```powershell
Get-ChildItem artifacts/same_day_res_poly -Directory | Sort-Object Name
```

### 5.1 Monitoring a TabM-from-exports run

Tail the TabM run log:

```powershell
Get-Content "<data-dir>\\training_results_tabm\\<RUN_ID>\\run.log" -Wait
```

TabM emits stage-level progress, for example:

- `STAGE_START [3/10 20.0%] build_feature_rows`

## 6) Expected slow stages

Primary runtime sinks:

1. `train_delta_model`
2. `analog_k_selection` (if enabled)

Peak training is usually much shorter than delta training.

## 7) Success criteria

A run is complete when all are true:

1. `run.log` ends with `PIPELINE_DONE`.
2. `metrics.json` exists.
3. model files exist.
4. prediction parquet files exist.
5. report CSV files exist.

## 8) Artifact checklist

Top-level run files:

- `run.log`
- `config.json`
- `feature_list.json`
- `imputer_values.json`
- `analog_standardizer.json`
- `train_date_range.txt`
- `val_date_range.txt`
- `test_date_range.txt`
- `metrics.json`
- `metrics.md`

Model exports:

- `models/peak_model.txt`
- `models/peak_isotonic.pkl`
- `models/delta_model.txt`
- `models/delta_temperature_T.json`

Predictions:

- `predictions/predictions_val.parquet`
- `predictions/predictions_test.parquet`
- `predictions/distribution_eval_val.parquet`
- `predictions/distribution_eval_test.parquet`

Reports:

- `reports/cutoff_metrics_val.csv`
- `reports/cutoff_metrics_test.csv`
- `reports/bucket_calibration_val.csv`
- `reports/bucket_calibration_test.csv`

### 8.1 TabM artifact checklist (from exports trainer)

TabM run folders contain a similar artifact set, but with different model filenames:

- `models/tabm_peak_model.pt`
- `models/tabm_delta_model.pt`
- `models/peak_isotonic.pkl`
- `models/delta_temperature_T.json`

TabM predictions are written as both CSV and parquet for portability.

## 9) Fast validation script pattern

```powershell
Get-ChildItem artifacts/same_day_res_poly/<RUN_ID> -Recurse -File
```

Confirm key metrics quickly:

```powershell
python - <<'PY'
import json
from pathlib import Path
p = Path('artifacts/same_day_res_poly/<RUN_ID>/metrics.json')
m = json.loads(p.read_text())
print(m['peak']['test']['logloss_cal'])
print(m['delta']['val']['multi_logloss_temp'])
print(m['combined_blended']['test']['nll'])
PY
```

## 10) Failure recovery procedure

If run stops mid-way:

1. preserve run folder for forensic analysis
2. inspect last log line
3. identify last completed stage
4. rerun with detailed logging
5. avoid force rebuild unless feature store itself is suspect

## 11) Known failure class and fix

Known issue previously observed:

- metric stage crash when `tmax_sofar` is NaN in calibration helper

Status:

- fixed by skipping invalid rows in bucket-calibration loop

## 12) Export robustness

Current implementation checkpoint-saves models immediately after training stages:

- peak model and isotonic after stage 4
- delta model and temperature scaler after stage 6

This means model artifacts can still exist even if a later stage fails.

## 13) Standalone reuse notes

You can reuse exported models, but inference requires:

1. exact feature engineering contract
2. exact feature order from `feature_list.json`
3. same imputer medians from `imputer_values.json`
4. same delta temperature scaler and peak isotonic calibrator

Without this alignment, outputs are invalid.

## 14) Current recommended exported run for no-analog use

Run id:

- `20260226T081223Z`

Path:

- `artifacts/same_day_res_poly/20260226T081223Z/`

Mode:

- `--skip-analog-blend`

## 15) Operational hygiene

Before sharing results:

1. quote run id
2. quote mode (analog enabled or skipped)
3. quote split used
4. quote exact metric name, not just a number

## 16) When something breaks

Start with:

- `07_exporter_and_remote_training_tabm.md` (see section "Troubleshooting and Known Issues")

That checklist covers:

- "looks stuck" vs actually dead
- missing final artifacts
- Windows venv activation policy errors
- PyTorch DLL initialization failures (WinError 1114)

## 17) Exhaustive Inventory: Training, Evaluation, Calibration Files

Chosen folder scope for this inventory:

- `ml/`
- `ml/src/weather_ml/klga_daily_tmax_dist/`
- `ml/src/weather_ml/training/`
- `ml/src/weather_ml/data_exporter/`

This section is intentionally exhaustive for the KLGA same-day Tmax workflow currently in use.

Excluded from this inventory:

- unrelated legacy/sweep families (`mos_*`, `tfs2`, `exp30`, KMIA/KPHL/KNYC experiment runners)
- Java ingestion implementation details beyond DB table availability

### 17.1 Top-level entry scripts (actual run commands)

| File | Primary role | Training/Eval/Calibration relevance | Writes artifacts |
|---|---|---|---|
| `ml/run_klga_daily_tmax_dist.py` | Main CLI entrypoint for KLGA LGBM peak+delta pipeline | Triggers full train/eval/calibration flow, optional analog blend, logging controls | Yes (`artifacts/same_day_res_poly/<RUN_ID>/...`) |
| `ml/run_klga_data_exporter.py` | Exports raw DB inputs to portable CSV bundle | Enables reproducible off-DB training/evaluation on another machine | Yes (`exports/klga_same_day_tmax_dist_<TS>/...`) |
| `ml/run_training_tabm_klga_from_exports.py` | CLI entrypoint for TabM training from exported CSVs | Trains TabM peak+delta, calibrates, evaluates, writes full run artifacts | Yes (`<data-dir>/training_results_tabm/<RUN_ID>/...`) |

### 17.2 LGBM pipeline module inventory (core path)

#### 17.2.1 Orchestration and stage flow

| File | Key functions/classes | What it does in the training lifecycle |
|---|---|---|
| `ml/src/weather_ml/klga_daily_tmax_dist/pipeline.py` | `run_training_pipeline`, `PipelineRunResult` | Master orchestrator. Runs dataset build/reuse, split guards, feature preparation, peak train, delta train, optional analog kNN, posterior blend, evaluation, calibration reports, and artifact persistence. |
| `ml/src/weather_ml/klga_daily_tmax_dist/logging_utils.py` | `format_duration`, `ProgressTracker` | Shared progress/heartbeat formatting utilities used by long-running dataset and analog loops. |

`pipeline.py` stage sequence (analog on):

1. `build_feature_store`
2. `load_feature_store`
3. `prepare_splits_and_features`
4. `train_peak_model`
5. `predict_peak_probabilities`
6. `train_delta_model`
7. `predict_delta_conditionals`
8. `build_analog_library`
9. `analog_k_selection`
10. `blend_posteriors`
11. `evaluate_metrics`
12. `write_artifacts`

`pipeline.py` stage sequence (analog off):

1. `build_feature_store`
2. `load_feature_store`
3. `prepare_splits_and_features`
4. `train_peak_model`
5. `predict_peak_probabilities`
6. `train_delta_model`
7. `predict_delta_conditionals`
8. `evaluate_metrics`
9. `write_artifacts`

#### 17.2.2 Data access, leakage guards, and time grid

| File | Key functions/classes | Training/Eval/Calibration impact |
|---|---|---|
| `ml/src/weather_ml/klga_daily_tmax_dist/config.py` | `PipelineConfig`, `SplitConfig`, constants | Canonical station set, split boundaries, cutoff grid, allowed columns, banned columns, analog settings, output root. |
| `ml/src/weather_ml/klga_daily_tmax_dist/db.py` | `fetch_daily_max`, `fetch_observations`, `ensure_required_indexes` | Pulls truth and observations from MySQL, enforces allowed observation column set, prevents banned columns in feature inputs. |
| `ml/src/weather_ml/klga_daily_tmax_dist/timegrid.py` | `make_cutoffs_for_date`, `make_calendar_grid`, `CutoffPoint` | DST-safe local-time cutoff generation, expected 30-min bin counts, local->UTC conversion for strict as-of logic. |

Important leakage controls in this layer:

- observations are fetched with explicit column allowlist and `valid_time_utc <= cutoff` contract
- banned columns (`max_temp`, `min_temp`, `precip_total`) are rejected
- split masks are chronological by `target_date_local` and validated for overlap/out-of-range rows

#### 17.2.3 Feature engineering and label construction

| File | Key functions/classes | Training/Eval/Calibration impact |
|---|---|---|
| `ml/src/weather_ml/klga_daily_tmax_dist/features.py` | `prepare_station_series`, `build_feature_rows`, `build_daily_prior_frame`, `model_feature_columns` | Builds KLGA+neighbor snapshots, slopes, volatility, so-far extrema, composites, priors, labels (`peak`, `delta`), as-of guard timestamps. |
| `ml/src/weather_ml/klga_daily_tmax_dist/make_dataset.py` | `build_feature_store`, `DatasetBuildResult` | End-to-end dataset materialization into parquet + integrity JSON, including missing-rate profiling and index creation. |

What is produced for downstream training:

- row grain: `(target_date_local, cutoff_minutes)` (one row per cutoff)
- label columns: `peak`, `delta`, `tmax_truth`
- numeric feature matrix and imputation medians
- audit fields: `max_valid_time_used_utc`, integrity payload, as-of violation counts

#### 17.2.4 Peak model training and calibration

| File | Key functions/classes | Calibration details |
|---|---|---|
| `ml/src/weather_ml/klga_daily_tmax_dist/train_peak.py` | `train_peak_model`, `predict_peak_probability`, `PeakTrainResult` | Trains binary LightGBM for peak/no-peak; calibrates probabilities with isotonic regression on validation; reports logloss, brier, reliability bins. |

Peak training details:

- model type: `LGBMClassifier(objective='binary')`
- inputs: all rows with valid `peak`
- calibration: `IsotonicRegression` fit on validation raw probabilities
- outputs used later:
  - `p_peak_raw`
  - `p_peak_cal` (calibrated)

#### 17.2.5 Delta model training and calibration

| File | Key functions/classes | Calibration details |
|---|---|---|
| `ml/src/weather_ml/klga_daily_tmax_dist/train_delta.py` | `train_delta_model`, `predict_delta_conditional`, `DeltaTrainResult` | Trains multiclass LightGBM for `delta` conditional on non-peak rows (`peak==0` and `delta>=1`), applies multiclass temperature scaling on validation logits. |

Delta training details:

- model type: `LGBMClassifier(objective='multiclass', num_class=K)`
- class target: `delta_class = clip(delta,1..K)-1`
- training row filter: only rows with truth non-peak regime
- calibration: optimize scalar `T` so `softmax(logits/T)` minimizes validation multiclass logloss
- outputs used later:
  - raw conditional class probs
  - temperature-scaled conditional probs

#### 17.2.6 Posterior construction, bucket parsing, and evaluation utilities

| File | Key functions/classes | Training/Eval/Calibration impact |
|---|---|---|
| `ml/src/weather_ml/klga_daily_tmax_dist/infer.py` | `build_delta_pmf`, `delta_pmf_to_tmax_pmf`, `parse_bucket_label`, `compute_bucket_probabilities` | Converts peak+delta outputs to full integer Tmax PMF and bucket probabilities; defines range parsing contract used for market labels. |
| `ml/src/weather_ml/klga_daily_tmax_dist/pipeline.py` (helpers) | `_evaluate_distribution_rows`, `_cutoff_metrics`, `_temperature_bucket_calibration` | Computes combined NLL/top1, per-cutoff metrics CSVs, and temperature-bucket calibration tables for val/test. |

#### 17.2.7 Analog kNN posterior module (optional branch)

| File | Key functions/classes | Training/Eval/Calibration impact |
|---|---|---|
| `ml/src/weather_ml/klga_daily_tmax_dist/analog_knn.py` | `fit_analog_standardizer`, `build_analog_library`, `predict_knn_posterior`, `calibrate_blend_bounds`, `blend_posteriors` | Retrieval-based posterior estimation from historical analog rows; selects K on validation and blends analog posterior with LGBM posterior by distance-quality score. |

Analog branch summary:

- candidate restrictions:
  - same `cutoff_minutes`
  - seasonal window (`DOY +- window`)
  - strict causal dates (`candidate_date < query_date`)
- outputs:
  - `p_peak_knn`
  - `p_delta_cond_knn`
  - `q_score`
- blend:
  - computes `w_lgbm` from `q_score` and quantile bounds (`q_low`, `q_high`)
  - mixed posterior is what `combined_blended` metrics evaluate

### 17.3 LGBM run artifacts and where each is created

| Artifact | Created by | Purpose |
|---|---|---|
| `feature_store/klga_feature_store.parquet` | `make_dataset.build_feature_store` | Canonical engineered dataset used by training/eval. |
| `feature_store/klga_feature_store_integrity.json` | `make_dataset.build_feature_store` | Dataset audit: row/date counts, missingness, guard metadata. |
| `<run>/models/peak_model.txt` | `pipeline.run_training_pipeline` | Serialized LightGBM peak model. |
| `<run>/models/peak_isotonic.pkl` | `pipeline.run_training_pipeline` | Peak calibration object. |
| `<run>/models/delta_model.txt` | `pipeline.run_training_pipeline` | Serialized LightGBM delta model. |
| `<run>/models/delta_temperature_T.json` | `pipeline.run_training_pipeline` | Delta temperature scaling parameter. |
| `<run>/predictions/predictions_val.parquet` | `pipeline.run_training_pipeline` | Full posterior components for validation rows. |
| `<run>/predictions/predictions_test.parquet` | `pipeline.run_training_pipeline` | Full posterior components for test rows. |
| `<run>/predictions/distribution_eval_*.parquet` | `pipeline.run_training_pipeline` | Row-level NLL/top1/true-prob evaluation details. |
| `<run>/reports/cutoff_metrics_*.csv` | `pipeline.run_training_pipeline` | Per-cutoff performance breakdown. |
| `<run>/reports/bucket_calibration_*.csv` | `pipeline.run_training_pipeline` | Empirical vs predicted calibration by integer temperature bucket. |
| `<run>/metrics.json` | `pipeline.run_training_pipeline` | Full machine-readable metrics payload (peak/delta/combined/analog). |
| `<run>/metrics.md` | `pipeline.run_training_pipeline` | Human summary metrics. |
| `<run>/run.log` | `pipeline logger` | Stage and heartbeat progress logs. |
| `<run>/feature_list.json` | `pipeline.run_training_pipeline` | Exact feature order used by models. |
| `<run>/imputer_values.json` | `pipeline.run_training_pipeline` | Median fills used pre-training/inference. |
| `<run>/analog_standardizer.json` | `pipeline.run_training_pipeline` | Analog standardization and blend metadata. |

### 17.4 Export pipeline files (portable training input bundle)

| File | Key functions/classes | Role in train/eval/calibration process |
|---|---|---|
| `ml/src/weather_ml/data_exporter/klga_training_data_exporter.py` | `ExportConfig`, `export_klga_training_eval_csvs` | Exports raw DB inputs needed to rebuild features and train/evaluate without DB access. |
| `ml/run_klga_data_exporter.py` | CLI wrapper | Operational entrypoint for exporter; date range and output controls. |

Export bundle contract:

- required files:
  - `daily_max_truth_klga.csv`
  - `observations_30m_required_columns.csv`
  - `station_universe.csv`
- metadata:
  - `export_manifest.json`

### 17.5 TabM-from-exports training/eval/calibration inventory

| File | Key functions/classes | Training/Eval/Calibration impact |
|---|---|---|
| `ml/run_training_tabm_klga_from_exports.py` | `ensure_dependencies`, `main` | Self-installs missing deps, parses CLI args, launches TabM training from exported CSVs. |
| `ml/src/weather_ml/training/tabm_klga_from_exports.py` | `TabMTrainingConfig`, `run_tabm_training_from_exports`, `_fit_peak`, `_fit_delta` | Rebuilds same engineered feature rows from CSV exports; trains TabM peak/delta models, applies isotonic + temperature scaling, computes combined NLL/top1 and calibration reports. |

TabM training stages:

1. validate input files
2. load raw CSVs
3. build feature rows
4. prepare training matrices
5. train peak model
6. train delta model
7. predict probabilities
8. evaluate metrics
9. write prediction/report files
10. write models and metadata

TabM calibration and evaluation behavior:

- peak calibration: isotonic regression on validation raw probabilities
- delta calibration: multiclass temperature scaling on validation logits
- combined evaluation: same helper logic as LGBM pipeline for PMF/NLL/top1
- calibration CSVs: same output shape as LGBM run (`bucket_calibration_val/test.csv`)

### 17.6 Exact call chain (LGBM canonical run)

1. `ml/run_klga_daily_tmax_dist.py:main`
2. `klga_daily_tmax_dist.pipeline.run_training_pipeline`
3. `make_dataset.build_feature_store` (unless feature store reuse)
4. `train_peak.train_peak_model` + isotonic calibration
5. `train_delta.train_delta_model` + temperature scaling
6. `analog_knn.*` (if analog enabled)
7. `pipeline._evaluate_distribution_rows` and calibration report writers
8. artifact persistence (`models`, `predictions`, `reports`, `metrics`)

### 17.7 Exact call chain (TabM-from-exports run)

1. `ml/run_training_tabm_klga_from_exports.py:main`
2. dependency check/install (`ensure_dependencies`)
3. `training.tabm_klga_from_exports.run_tabm_training_from_exports`
4. CSV load -> feature rebuild using `klga_daily_tmax_dist.features` + `timegrid`
5. `_fit_peak` (TabM binary head) + isotonic
6. `_fit_delta` (TabM multiclass head) + temperature scaling
7. combined evaluation and calibration CSV generation
8. artifact persistence (`models`, `predictions`, `reports`, `metrics`)

### 17.8 Calibration-specific files and responsibilities

| Calibration step | File(s) | Output artifact(s) |
|---|---|---|
| Peak isotonic fit and apply (LGBM) | `train_peak.py`, `pipeline.py` | `models/peak_isotonic.pkl`, calibrated peak metrics |
| Delta temperature scaling (LGBM) | `train_delta.py`, `pipeline.py` | `models/delta_temperature_T.json`, temp-scaled delta metrics |
| Blended PMF bucket calibration (LGBM) | `pipeline.py::_temperature_bucket_calibration` | `reports/bucket_calibration_val.csv`, `reports/bucket_calibration_test.csv` |
| Peak isotonic fit/apply (TabM) | `training/tabm_klga_from_exports.py` | `models/peak_isotonic.pkl`, peak calibrated metrics |
| Delta temperature scaling (TabM) | `training/tabm_klga_from_exports.py` | `models/delta_temperature_T.json`, delta temp-scaled metrics |
| PMF bucket calibration (TabM) | `training/tabm_klga_from_exports.py` + shared helpers | `reports/bucket_calibration_val.csv`, `reports/bucket_calibration_test.csv` |

### 17.9 Files that define the feature contract for all training variants

| File | Why it is critical |
|---|---|
| `klga_daily_tmax_dist/config.py` | Defines allowed columns, banned columns, station universe, split windows, and analog feature set. |
| `klga_daily_tmax_dist/features.py` | Defines actual engineered features and labels; this is the canonical transformation contract. |
| `<RUN_ID>/feature_list.json` | Locks exact feature ordering used in a specific trained run. |
| `<RUN_ID>/imputer_values.json` | Locks exact missing-value treatment used in that run. |

### 17.10 Practical interpretation: what is "actual training/evaluation/calibration" here

For this KLGA system, the core "actual ML process" is exactly:

1. feature-store construction from WU observations + daily truth
2. peak model fit + peak calibration
3. delta model fit + delta calibration
4. posterior composition into Tmax PMF
5. PMF-level evaluation (NLL/top1) and bucket calibration exports
6. artifact export for deterministic reuse

The files listed in sections 17.1-17.9 are the complete script/module set implementing those steps.
