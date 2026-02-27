# 07 - Exporter and Remote Training (TabM From Exports)

This document covers two operational additions we implemented after the initial KLGA same-day Tmax distribution system:

1. A **data exporter** that materializes the exact raw inputs needed for training/evaluation into portable CSVs.
2. A **Tabular Neural Network (TabM)** training runner that trains the same peak+delta decomposition from those exported files, and writes a full evaluation artifact set.

This exists because:

- you may want to train on a different/more powerful machine,
- you may want to compare "GBDT vs Tabular NN" on the same feature contract,
- the database should remain the canonical source, but portable exports are useful for iteration.

Important:

- These exports can be large. Do not commit them to Git.
- The TabM run is an experimental alternative model family; LightGBM remains the current reference baseline for this system.

## 1) The Exported Data Contract (What Gets Written)

The exporter writes exactly the three files the feature builder and trainers need:

- `station_universe.csv`
- `daily_max_truth_klga.csv`
- `observations_30m_required_columns.csv`

It also writes:

- `export_manifest.json` (provenance + row counts)

### 1.1 station_universe.csv

Purpose:

- defines the station set used by the pipeline (target + neighbors)

Schema:

- `request_location_id`
- `role` in `{target, neighbor}`

Rules:

- target is `KLGA:9:US`
- neighbors are the 7 airports (no `KNYC:9:US`)

### 1.2 daily_max_truth_klga.csv

Purpose:

- provides the daily truth `Tmax_final(D)` used to build labels and daily priors

Source table:

- `wunderground_ml.wunderground_station_daily_max_temperature`

Required columns (must exist in the CSV):

- `request_location_id`
- `target_date_local` (DATE in America/New_York local calendar)
- `max_temp_f` (daily max truth; later rounded to integer F in label construction)
- `station_zoneid` (expected `America/New_York`)

Leakage safety:

- on a given day `D`, the pipeline uses daily max rows only for dates `< D` when building priors
- the daily max for `D` is label only (never a feature)

### 1.3 observations_30m_required_columns.csv

Purpose:

- provides instantaneous observations needed to compute same-day "as-of" features

Source table:

- `wunderground_ml.wunderground_station_observation_30m`

Allowed (instantaneous) column set is enforced by the pipeline config:

- `request_location_id`
- `valid_time_utc`
- `temp`, `dew_pt`, `rh`, `pressure`, `vis`, `wspd`, `wdir`, `gust`, `precip_hrly`

Hard-banned columns (never export/use as features):

- `max_temp`
- `min_temp`
- `precip_total`

Reason:

- they behave like summary/day-total fields in many APIs and can leak post-cutoff information

### 1.4 export_manifest.json

Purpose:

- allows you to verify provenance and confirm row counts after moving data

Contains:

- export time
- exported date range
- station universe list
- file paths and row counts

## 2) Running the Exporter (DB -> CSV bundle)

Runner:

- `ml/run_klga_data_exporter.py`

Core module:

- `ml/src/weather_ml/data_exporter/klga_training_data_exporter.py`

Example (default 1973-01-01..2025-12-31):

```powershell
python ml/run_klga_data_exporter.py --log-level INFO
```

Example (explicit output directory):

```powershell
python ml/run_klga_data_exporter.py `
  --start-date 1973-01-01 `
  --end-date 2025-12-31 `
  --output-dir "C:\path\to\exports\klga_same_day_tmax_dist_export_1973_2025" `
  --log-level INFO
```

Notes:

- observation export is chunked; logs show per-chunk row counts
- MySQL connection comes from `--mysql-url` or `MYSQL_*` env vars

## 3) Moving the Export Bundle to Another Machine

The export bundle is self-contained. You can copy the entire folder:

- the three CSVs
- the manifest

Example copy target:

- `D:\Ahmed\data\early_peak_data\`

Do not rename the CSV filenames unless you also update the training script expectations.

## 4) Training TabM From Exports

TabM runner:

- `ml/run_training_tabm_klga_from_exports.py`

Training module:

- `ml/src/weather_ml/training/tabm_klga_from_exports.py`

### 4.1 What TabM Trains

It trains the same two-head system as the LightGBM baseline:

1. **Peak model** (binary)
   - predicts `P(delta=0)` = probability the day has already peaked by the cutoff
   - calibrated with isotonic regression

2. **Delta model** (multiclass, conditional)
   - predicts `P(delta=k | delta>0)` for `k=1..60` (tail bin at 60)
   - calibrated with temperature scaling

Then it composes the final distribution the same way:

- `P(delta=0) = p_peak`
- `P(delta=k>0) = (1 - p_peak) * P(delta=k | delta>0)`

### 4.2 Feature Contract (Critical)

TabM uses the **same feature builder** as the LightGBM pipeline:

- calendar/cutoff identity features
- KLGA snapshot + so-far extrema + trajectories
- neighbor gradients + composites
- daily priors from `<D` only
- train-only climatology lookup (`climo_rem_delta_mean/std`)

So this is a true apples-to-apples "model family swap" experiment:

- same inputs
- same labels
- same splits
- different learner

### 4.3 Self-Sufficient Dependency Install

The runner checks imports and installs missing packages via pip at runtime:

- numpy, pandas, scikit-learn, joblib
- torch, tabm, rtdl_num_embeddings
- pyarrow (parquet outputs)

This is convenient for fresh machines, but it requires:

- working internet access
- permission to install packages into the active Python environment

If you prefer strict reproducibility, use a dedicated venv and pin versions.

### 4.4 Output Directory Rules

Default output location:

- `<data-dir>/training_results_tabm/<RUN_ID>/`

This is deliberate:

- you can keep data + model artifacts in one portable folder tree
- you can run multiple experiments and keep them separated by run id

### 4.5 What Artifacts Are Produced

Run folder structure:

- `run.log` (full logs)
- `config.json`
- `metrics.json`
- `metrics.md`
- `feature_list.json`
- `imputer_values.json`
- `train_date_range.txt`, `val_date_range.txt`, `test_date_range.txt`
- `models/`
  - `tabm_peak_model.pt`
  - `tabm_delta_model.pt`
  - `peak_isotonic.pkl`
  - `delta_temperature_T.json`
- `predictions/`
  - `predictions_val.csv` + `.parquet`
  - `predictions_test.csv` + `.parquet`
  - `distribution_eval_val.csv` + `.parquet`
  - `distribution_eval_test.csv` + `.parquet`
- `reports/`
  - `cutoff_metrics_val.csv`
  - `cutoff_metrics_test.csv`
  - `bucket_calibration_val.csv`
  - `bucket_calibration_test.csv`

### 4.6 Command Examples

Run (most common):

```powershell
python ml/run_training_tabm_klga_from_exports.py `
  --data-dir "D:\Ahmed\data\early_peak_data" `
  --log-level INFO
```

Higher-verbosity progress logs:

```powershell
python ml/run_training_tabm_klga_from_exports.py `
  --data-dir "D:\Ahmed\data\early_peak_data" `
  --log-level INFO `
  --log-every-batches 10 `
  --log-every-rows 500 `
  --log-every-seconds 5
```

CPU-only (avoid any accidental GPU issues):

```powershell
python ml/run_training_tabm_klga_from_exports.py `
  --data-dir "D:\Ahmed\data\early_peak_data" `
  --device cpu `
  --log-level INFO
```

## 5) Troubleshooting and Known Issues (Merged)

This section is a practical checklist for diagnosing and fixing common issues when running:

- the KLGA same-day Tmax distribution pipeline,
- the exporter workflow,
- the TabM-from-exports training workflow.

It was previously a separate document and is merged here to keep the documentation set compact without omitting anything.

Goal:

- make failures actionable
- avoid wasting time on "is it stuck?" uncertainty
- preserve leakage safety and artifact integrity

### 5.1 "It looks stuck" (Long Runtime Stages)

#### 5.1.1 Delta training is slow (expected)

Most total runtime is usually spent in delta training:

- the delta model is multiclass and is trained on a very large row set
- the peak model is a much smaller binary problem and usually trains quickly

What to do:

- tail the `run.log` and confirm periodic progress lines still appear
- check CPU usage for the python process

#### 5.1.2 Analog kNN can be very slow (when enabled)

If analog is enabled, the kNN search stage is an additional heavy runtime sink.

Mitigation:

- run with `--skip-analog-blend` when you want faster "baseline export" runs

### 5.2 Confirming A Run Is Alive

#### 5.2.1 Tail the run log

```powershell
Get-Content artifacts/same_day_res_poly/<RUN_ID>/run.log -Wait
```

If logs stop updating for a long time and CPU is near zero, the process may be stalled or dead.

#### 5.2.2 Check python processes

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,StartTime
```

### 5.3 Missing Final Results / Artifacts

#### 5.3.1 "It trained something but no metrics.json appeared"

Causes:

- run interrupted mid-stage (killed process)
- crash occurred after training but before artifact writing

What to check:

- does `run.log` end with `PIPELINE_DONE`?
- do the model files exist in `models/`?
- does `metrics.json` exist?

Current robustness expectation:

- peak/delta models are checkpoint-saved immediately after their training stages in the main LGBM pipeline, so long runs should not end with "nothing".

### 5.4 Windows: PowerShell ExecutionPolicy Blocks venv Activate.ps1

Symptom:

- `Activate.ps1 cannot be loaded because running scripts is disabled on this system`

Fix options:

1. Use CMD activation instead of PowerShell:

```bat
.\.venv\Scripts\activate.bat
```

2. Avoid activation and call venv Python directly:

```powershell
.\.venv\Scripts\python.exe ml\run_training_tabm_klga_from_exports.py --data-dir "D:\path\to\data"
```

3. If you control the machine policy, set an execution policy that allows local scripts (organizational constraints may block this).

### 5.5 Windows: PyTorch WinError 1114 (c10.dll)

Symptom:

- `OSError: [WinError 1114] ... Error loading ... torch\\lib\\c10.dll`

Meaning:

- torch installed, but native dependencies are missing or incompatible (Visual C++ runtime is the most common)

Checklist:

1. Install Microsoft Visual C++ Redistributable 2015-2022 (x64).
2. Reboot (sometimes required after runtime installs).
3. Use a clean venv.
4. Prefer CPU torch build:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
python -c "import torch; print(torch.__version__)"
```

If `import torch` works but your training still fails:

- set `--device cpu`
- ensure no conflicting torch installs exist in the global site-packages

### 5.6 GitHub Push Rejected (Large Files)

Symptom:

- push rejected with `GH001: Large files detected` and mentions `> 100.00 MB`

Cause:

- exported datasets (especially observations) are large and exceed GitHub limits

Correct behavior:

- never commit export bundles
- keep them on local disk / storage and move by file transfer

Repo guardrails:

- ensure `exports/**` is in `.gitignore`

### 5.7 Data Issues: "No rows after filtering" / "Empty split"

Typical causes:

- export date range does not cover configured split dates
- time columns parsed incorrectly
- station ids in exports are missing or incomplete

Checks:

- verify `daily_max_truth_klga.csv` includes `target_date_local` rows across train/val/test ranges
- verify `observations_30m_required_columns.csv` includes all station ids and valid UTC timestamps

### 5.8 Leakage Guard Failures (Hard Fail by Design)

If you see assertion/guard failures, treat it as a correctness issue, not something to suppress.

Common guard failures:

- observation timestamps after cutoff were used
- daily max for date `D` was used as a feature
- a banned observation column is present/used
- station universe accidentally includes `KNYC:9:US`

Fix:

- identify the exact guard, then fix the upstream feature selection/windowing logic

## 6) What We Learned From the First TabM Run

The initial TabM experiment (trained from exports) underperformed the LightGBM baseline on:

- peak calibrated logloss and Brier
- delta multiclass logloss
- combined distribution NLL

Interpretation:

- Tabular NNs are not automatically better than boosted trees on engineered tabular feature sets.
- GBDTs often win on robustness, missingness handling, and quick high-quality calibration.

Concrete results and comparisons live in:

- `04_run_history_and_current_status_2026-02-26.md`
