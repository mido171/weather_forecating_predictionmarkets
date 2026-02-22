# WFPM-202 — Retrain & evaluate gribstream KMIA Tmax model with external-context features (ablation + calibration + report)

**Type:** Story  
**Priority:** P0  
**Epic:** WFPM-EPIC-EXTDATA  

## Goal

Use the new external-context features (WFPM‑201 output) to retrain the existing tree-based gribstream Tmax model and produce a rigorous, optimism‑free evaluation that answers:

1) **Does MAE improve materially on the held‑out test window (2024–2025)?**
2) **Do Kalshi-style event probabilities become better calibrated (ECE/MCE, Brier, LogLoss, reliability curves)?**
3) **Which subset of new sources contributes real incremental skill (ablation)?**

This ticket ends with a single “best” model artifact ready for live trading.

## Inputs

- Existing gribstream training dataset builder (current pipeline)
- New merged feature dataset with external features:
  - output of WFPM‑201 (merged CSV/DB view)
- Ground-truth label series for KMIA daily Tmax (same as current pipeline)
- Existing evaluation harness (pinball, CRPS‑approx, Brier, LogLoss, reliability bins)

## Train/val/test split (do not change)

Use the exact time split already used for trustworthy reporting:

- **Train:** 2021‑01‑01 → 2023‑12‑31 (or earliest gribstream start through 2023‑12‑31)
- **Validation:** 2024‑01‑01 → 2024‑12‑31 (for calibration + early stopping)
- **Test:** 2025‑01‑01 → 2025‑12‑31 (final report)

If your project currently uses 2024–2025 as test, keep that: the important requirement is **no random split** and **no leakage**.

## Model family (keep tree models)

Use the same algorithm family currently producing your best gribstream results (XGBoost or LightGBM). Do not switch algorithms in this ticket; we are isolating the feature gain.

### Outputs required

- Point forecast model (MAE/RMSE optimized)
- Quantile models (at least q05/q10/q25/q50/q75/q90/q95) OR a single model that can output quantiles if your stack supports it
- A probability mass function (PMF) over integer Fahrenheit buckets if your Kalshi evaluator uses it

## Step 1 — Build ablation datasets

Create these feature sets (all share identical rows and splits):

A) **Baseline**: existing gribstream feature columns only  
B) Baseline + **ASOS ring** features  
C) Baseline + ASOS + **NDBC marine**  
D) Baseline + ASOS + NDBC + **FAWN**  
E) Baseline + ASOS + NDBC + FAWN + **IGRA**  
F) Baseline + ASOS + NDBC + FAWN + IGRA + **OISST** (full)

Also create a “minimal best-of” set:
G) Baseline + {top 2 sources by validation MAE gain}

Implementation requirement:
- Each dataset must have a **frozen column list** written to disk:
  - `feature_columns_A.json`, `feature_columns_B.json`, etc.

## Step 2 — Training protocol (avoid optimistic bias)

For each dataset A..G:

1) Train point model on Train, tune hyperparameters ONLY with Validation.
2) Train quantile models using the same Train/Validation split.
3) Freeze models and evaluate on Test exactly once.

**No peeking** at Test during hyperparameter selection.

### Recommended hyperparameters (starting point)

(These are safe defaults; adjust only based on validation.)

For LightGBM:
- objective: regression (point) / quantile (quantiles)
- num_leaves: 64–256
- min_data_in_leaf: 20–100
- feature_fraction: 0.7–0.9
- bagging_fraction: 0.7–0.9
- bagging_freq: 1
- learning_rate: 0.02–0.05
- n_estimators: 2000–8000 with early stopping
- lambda_l1/l2: small nonzero (e.g., 0.1)

For XGBoost:
- tree_method: hist
- max_depth: 4–8
- min_child_weight: 1–10
- subsample: 0.7–0.9
- colsample_bytree: 0.7–0.9
- eta: 0.02–0.05
- n_estimators: 3000–10000 with early stopping
- reg_lambda: 1–10

## Step 3 — Scoring metrics (must report all)

### Point forecast

- MAE (primary)
- RMSE
- Bias (mean error)
- Error by month (seasonality sanity check)

### Probabilistic

- Pinball loss for each quantile
- Coverage for p50/p80/p90 intervals + average width
- CRPS-approx (from quantiles or PMF)
- PIT histogram + chi-square (uniformity check)

### Kalshi event scoring (these are the trading-relevant metrics)

Compute for each market definition you trade (examples):

- `lt_70`, `lt_75`, `ge_85`, `ge_90`
- `range_80_84`, `range_85_89` (buckets)

For each event:
- Brier score
- Log loss
- Reliability table (10 bins):
  - avg_pred vs empirical rate
  - ECE and MCE

## Step 4 — Post‑hoc calibration (make “65% ≈ 65%”)

After training the probabilistic model for a dataset, calibrate event probabilities using **validation** only:

1) For each event (e.g., `ge_90`), compute raw model probability `p_raw` on Validation.
2) Fit an **isotonic regression** mapping `p_raw → p_cal` (monotone calibration).
3) Save calibrator object per event to disk.
4) On Test and in live inference, report both:
   - `p_raw` (for diagnostics)
   - `p_cal` (for trading decisions)

Acceptance requirement:
- Calibration must not use Test.
- ECE on Validation must improve or remain similar; if it gets worse, drop calibration for that event.

## Step 5 — Produce a single comparison report artifact

Write a markdown report:

`artifacts/external_context_eval/report.md`

It must include:

- Table: A..G datasets with Test MAE/RMSE/Bias
- Table: event Brier/LogLoss + ECE/MCE (raw vs calibrated)
- Reliability bin tables (top 3 events you trade most)
- A short “conclusion” that names the best dataset and why (MAE + calibration)

Also export:
- `predictions_test.csv` with:
  - date, y_true
  - point_pred
  - quantiles
  - p_raw_{event}, p_cal_{event}

## Step 6 — Select the final model (decision rule)

Choose the final model as:

1) Lowest Test MAE **subject to**:
2) No catastrophic miscalibration in key markets:
   - ECE <= 0.05 on major events OR improves vs baseline
   - LogLoss does not materially worsen on `ge_90`/hot tail (risk control)
3) Bias magnitude <= 0.3°F (or document correction)

If the best MAE model fails calibration badly, select the next best that meets calibration constraints (trading requires honest probabilities).

## Acceptance criteria

- All datasets A..G trained and evaluated with identical splits.
- Report exists with clear winner and ablation insights.
- Calibrators trained on Validation and applied to Test are saved and versioned.
- Best model artifact is stored under:
  - `artifacts/gribstream_kmia_tmax_external_context/<run_id>/`
  - includes config, feature columns, model files, calibrators, and predictions.

## Definition of done

- A single run folder contains everything needed for live inference:
  - `model_point.bin`
  - `model_q05..q95.bin` (or equivalent)
  - `event_calibrators.pkl`
  - `feature_columns.json`
  - `run_metadata.json`
- The report demonstrates whether the epic’s features improved skill.
