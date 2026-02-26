# 04 - Run History and Current Status (as of 2026-02-26)

This document is the operational truth source for what has actually run, what completed, what failed, and which run should be treated as the current reference export.

## 1) Why this file exists

This system had multiple runs with different modes and different completion states.

Without a run-history file, it is easy to accidentally compare:

- a smoke run against a full run,
- a partial run against a completed run,
- an analog-enabled partial run against a no-analog completed run.

This file prevents that confusion.

## 2) Run catalog at a glance

| Run ID | Path Root | Mode | Status | Key Outcome |
|---|---|---|---|---|
| `20260225T130502Z` | `artifacts/same_day_res_poly_smoke3/` | smoke | completed | pipeline mechanics validated; quality intentionally weak |
| `20260225T144103Z` | `artifacts/same_day_res_poly_smoke4/` | smoke | completed | same as smoke3; sanity check repeat |
| `20260225T155726Z` | `artifacts/same_day_res_poly/` | full + analog | interrupted | reached stage 10/12; did not write final artifacts |
| `20260226T081223Z` | `artifacts/same_day_res_poly/` | full no-analog (`--skip-analog-blend`) | completed | full export package present (peak + delta models and reports) |

## 3) Smoke runs (for mechanics, not performance)

### 3.1 Smoke run `20260225T130502Z`

Dataset size:

- rows_total: `12,325`
- split_rows: train `10,614`, val `899`, test `812`

Main metrics:

- combined val NLL: `2.3674`
- combined test NLL: `2.6021`
- peak val calibrated logloss: `0.4850`
- delta val temp-scaled multi-logloss: `3.1573`

Interpretation:

- this run proves pipeline flow and output writing,
- this run does not represent final quality expectations.

### 3.2 Smoke run `20260225T144103Z`

Metrics are effectively identical to smoke3.

Interpretation:

- confirms reproducibility of smoke behavior,
- does not upgrade the quality conclusion.

## 4) Interrupted analog-enabled full run `20260225T155726Z`

### 4.1 What completed

Stage progression from run log:

1. build feature store
2. load feature store
3. prepare splits and features
4. train peak model
5. predict peak
6. train delta model
7. predict delta
8. build analog library
9. analog K selection
10. blend posteriors
11. evaluate metrics started
12. write artifacts not reached

### 4.2 What was learned before interruption

Strong intermediate signals were already visible:

- peak val calibrated logloss: `0.2189975`
- peak val calibrated Brier: `0.0677481`
- delta val temp-scaled multi-logloss: `2.4075509`

Analog K-search on validation:

- `K=50` => val NLL `3.7724`
- `K=100` => val NLL `2.9412`
- `K=200` => val NLL `2.6625` (best among tested K)

Blend thresholds:

- `q_low=2.8472`
- `q_high=4.1291`

### 4.3 Why this run is not reference-quality

Final completion markers missing:

- no `PIPELINE_DONE`
- no `metrics.json`
- no final model export directory for this run id

So this run is analytically useful but operationally incomplete.

## 5) Completed no-analog export run `20260226T081223Z`

This is the current reference run for exported LGBM peak+delta artifacts.

Path:

- `artifacts/same_day_res_poly/20260226T081223Z/`

Mode:

- `--skip-analog-blend` (analog disabled by design)

Completion evidence:

- `PIPELINE_DONE` appears in `run.log`
- `metrics.json` and `metrics.md` present
- model exports present
- prediction and report files present

### 5.1 Main metrics (from `metrics.json`)

Combined distribution:

- val NLL: `2.2939528`
- val top1: `0.2329`
- test NLL: `2.3387056`
- test top1: `0.2299`

Peak model:

- val calibrated logloss: `0.2189975`
- val calibrated Brier: `0.0677481`
- test calibrated logloss: `0.2098557`
- test calibrated Brier: `0.0652467`

Delta model:

- val temp-scaled multi-logloss: `2.4075509`
- temperature scaler `T`: `1.0636161`

### 5.2 Stage durations (from run log)

- stage 4 train peak: `02:55`
- stage 6 train delta: `56:09`
- stage 7 predict delta: `04:15`
- stage 8 evaluate metrics: `00:50`
- total pipeline elapsed: `01:04:34`

Interpretation:

- delta training is the dominant runtime cost,
- long runtime at delta stage is expected, not necessarily a stuck process.

## 6) Time-of-day behavior (from cutoff reports, completed no-analog run)

Test split examples:

- cutoff `240` (04:00 NY) => NLL `2.8052`, top1 `0.1040`
- cutoff `1080` (18:00 NY) => NLL `1.5636`, top1 `0.4815`

Interpretation:

- performance improves strongly later in day,
- this is expected due to reduced uncertainty as more of the day is observed.

## 7) Artifact status matrix

| Artifact | Interrupted Full Run `20260225T155726Z` | Completed No-Analog Run `20260226T081223Z` |
|---|---|---|
| `run.log` | yes | yes |
| `metrics.json` | no | yes |
| `metrics.md` | no | yes |
| `models/peak_model.txt` | no | yes |
| `models/peak_isotonic.pkl` | no | yes |
| `models/delta_model.txt` | no | yes |
| `models/delta_temperature_T.json` | no | yes |
| `predictions/*.parquet` | no | yes |
| `reports/cutoff_metrics_*.csv` | yes | yes |
| `reports/bucket_calibration_*.csv` | no | yes |

## 8) Current operational conclusion

Use `20260226T081223Z` as the active reference export for peak+delta standalone use.

Reason:

- full completion,
- full artifact package,
- strong peak metrics,
- reasonable combined NLL,
- reproducible run path.

Treat `20260225T155726Z` as a diagnostic run only.

## 9) What changed to improve reliability

Two robustness changes were implemented in pipeline code before the successful no-analog export run:

1. model checkpoint save immediately after training stages,
2. defensive handling for invalid calibration rows during evaluation.

Practical effect:

- reduced risk of ending with no model files after long training,
- improved end-of-run stability for artifact writing.

## 10) Next run recommendation

If you want a new production-style export quickly:

```powershell
python ml/run_klga_daily_tmax_dist.py `
  --output-root artifacts/same_day_res_poly `
  --skip-analog-blend
```

If you want analog experiments, run full mode separately and treat it as a heavier, longer analysis run.
