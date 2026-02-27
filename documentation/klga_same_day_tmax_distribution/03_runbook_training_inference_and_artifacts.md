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
