# E37 Leakage Audit Report

## Executive Summary
- Station: KMIA
- Date range: 2002-01-22 to 2025-12-31
- Official feature store: artifacts\MOS\experiments\20260218T230007Z\feature_store.csv
- Verdict (rule-based): **LEAKAGE**

## Data & Split Integrity (A)
| Split | Count | Min Date | Max Date | Missing Targets |
|---|---:|---|---|---:|
| train | 6552 | 2002-01-23 | 2019-12-31 | 0 |
| validation | 1096 | 2020-01-01 | 2022-12-31 | 0 |
| test | 1096 | 2023-01-01 | 2025-12-31 | 0 |
- Duplicate target_date_local rows: 0

### Target Sanity (tmax_f)
| Split | Min | Median | Max |
|---|---:|---:|---:|
| train | 48.00 | 85.00 | 98.00 |
| validation | 50.00 | 86.00 | 98.00 |
| test | 58.00 | 86.00 | 98.00 |

## As-of Cutoff Provenance Audit (B)
- Total MOS rows audited: 131831
- Violations (asof_utc > cutoff_utc): 0
- Worst violation hours: 0.0

### Top 20 Worst Days (asof_utc - cutoff_utc)
| target_date_local | cutoff_utc | asof_utc | violation_hours | feature | model | variable |
|---|---|---|---:|---|---|---|
| 2002-01-22 | 2002-01-21 12:00:00+00:00 | 2002-01-21 12:00:00+00:00 | 0.000 | latest_b0 | gfs | cig |
| 2018-02-02 | 2018-02-01 12:00:00+00:00 | 2018-02-01 12:00:00+00:00 | 0.000 | latest_b0 | nam | wsp |
| 2018-02-03 | 2018-02-02 12:00:00+00:00 | 2018-02-02 12:00:00+00:00 | 0.000 | latest_b0 | nam | tmp |
| 2018-02-03 | 2018-02-02 12:00:00+00:00 | 2018-02-02 12:00:00+00:00 | 0.000 | latest_b0 | nam | q12 |
| 2018-02-03 | 2018-02-02 12:00:00+00:00 | 2018-02-02 12:00:00+00:00 | 0.000 | latest_b0 | nam | p12 |
| 2018-02-03 | 2018-02-02 12:00:00+00:00 | 2018-02-02 12:00:00+00:00 | 0.000 | latest_b0 | nam | dpt |
| 2018-02-03 | 2018-02-02 12:00:00+00:00 | 2018-02-02 12:00:00+00:00 | 0.000 | latest_b0 | nam | cig |
| 2018-02-03 | 2018-02-02 12:00:00+00:00 | 2018-02-02 12:00:00+00:00 | 0.000 | latest_b0 | gfs | wsp |
| 2018-02-03 | 2018-02-02 12:00:00+00:00 | 2018-02-02 12:00:00+00:00 | 0.000 | latest_b0 | gfs | wdr |
| 2018-02-03 | 2018-02-02 12:00:00+00:00 | 2018-02-02 12:00:00+00:00 | 0.000 | latest_b0 | gfs | tmp |
| 2018-02-03 | 2018-02-02 12:00:00+00:00 | 2018-02-02 12:00:00+00:00 | 0.000 | latest_b0 | gfs | q12 |
| 2018-02-03 | 2018-02-02 12:00:00+00:00 | 2018-02-02 12:00:00+00:00 | 0.000 | latest_b0 | gfs | p12 |
| 2018-02-03 | 2018-02-02 12:00:00+00:00 | 2018-02-02 12:00:00+00:00 | 0.000 | latest_b0 | gfs | dpt |
| 2018-02-03 | 2018-02-02 12:00:00+00:00 | 2018-02-02 12:00:00+00:00 | 0.000 | latest_b0 | gfs | cig |
| 2018-02-02 | 2018-02-01 12:00:00+00:00 | 2018-02-01 12:00:00+00:00 | 0.000 | latest_b0 | nam | wdr |
| 2018-02-03 | 2018-02-02 12:00:00+00:00 | 2018-02-02 12:00:00+00:00 | 0.000 | latest_b0 | nam | wsp |
| 2018-02-02 | 2018-02-01 12:00:00+00:00 | 2018-02-01 12:00:00+00:00 | 0.000 | latest_b0 | nam | tmp |
| 2018-02-02 | 2018-02-01 12:00:00+00:00 | 2018-02-01 12:00:00+00:00 | 0.000 | latest_b0 | nam | q12 |
| 2018-02-02 | 2018-02-01 12:00:00+00:00 | 2018-02-01 12:00:00+00:00 | 0.000 | latest_b0 | nam | p12 |
| 2018-02-02 | 2018-02-01 12:00:00+00:00 | 2018-02-01 12:00:00+00:00 | 0.000 | latest_b0 | nam | dpt |

### Forbidden Feature Scan (inputs only)
- Offending feature inputs found: None

## Truth Leakage Audit (C)
- Policy: truth allowed up to T-1 (obs_cutoff_lag_days=0)
- Violations in 100-sample test check: 0

### Sample (10 of 100)
| target_date_local | max_truth_date_used | allowed_max_truth_date | ok |
|---|---|---|---|
| 2023-02-07 | None | 2023-02-06 | True |
| 2025-02-27 | None | 2025-02-26 | True |
| 2025-03-10 | None | 2025-03-09 | True |
| 2024-09-23 | None | 2024-09-22 | True |
| 2024-08-30 | None | 2024-08-29 | True |
| 2024-10-21 | None | 2024-10-20 | True |
| 2023-07-13 | None | 2023-07-12 | True |
| 2023-01-05 | None | 2023-01-04 | True |
| 2025-01-11 | None | 2025-01-10 | True |
| 2023-01-28 | None | 2023-01-27 | True |

## Feature Rebuild & Parity (D)
### Parity vs official feature_store.csv (all dates)
| feature | max_abs_diff | mean_abs_diff | pct_diff_gt_1e-6 |
|---|---:|---:|---:|
| feat_u | 3.55271e-15 | 1.10448e-16 | 0.0000 |
| feat_v | 3.55271e-15 | 5.92741e-17 | 0.0000 |
| feat_wsp_mean | 0 | 0 | 0.0000 |
| cal_d_doy_sin | 1.11022e-16 | 2.48384e-17 | 0.0000 |
| cal_d_doy_cos | 1.11022e-16 | 2.56846e-17 | 0.0000 |
| feat_dd_models | 0 | 0 | 0.0000 |
| feat_tmp_range_mean_models | 0 | 0 | 0.0000 |
| feat_p12_max | 0 | 0 | 0.0000 |
| feat_q12_max | 0 | 0 | 0.0000 |
| feat_cig_min | 0 | 0 | 0.0000 |
| feat_le_median_biascorr | 0.911722 | 0.00205251 | 0.0274 |
| feat_onshore | 0 | 0 | 0.0000 |

### Parity vs official feature_store.csv (test only)
| feature | max_abs_diff | mean_abs_diff | pct_diff_gt_1e-6 |
|---|---:|---:|---:|
| feat_u | 1.77636e-15 | 1.01114e-16 | 0.0000 |
| feat_v | 1.77636e-15 | 7.0079e-17 | 0.0000 |
| feat_wsp_mean | 0 | 0 | 0.0000 |
| cal_d_doy_sin | 1.11022e-16 | 2.48456e-17 | 0.0000 |
| cal_d_doy_cos | 1.11022e-16 | 2.57915e-17 | 0.0000 |
| feat_dd_models | 0 | 0 | 0.0000 |
| feat_tmp_range_mean_models | 0 | 0 | 0.0000 |
| feat_p12_max | 0 | 0 | 0.0000 |
| feat_q12_max | 0 | 0 | 0.0000 |
| feat_cig_min | 0 | 0 | 0.0000 |
| feat_le_median_biascorr | 1.42109e-14 | 1.42627e-15 | 0.0000 |
| feat_onshore | 0 | 0 | 0.0000 |

## Training Reproduction & Alignment (E)
### E1 Official pipeline reproduction (feature_store.csv)
| Split | MAE | RMSE | Bias | P50 | P90 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.9045 | 1.4313 | 0.0702 | 0.5589 | 2.2245 | 2.9993 |
| validation | 0.8399 | 1.3730 | 0.0648 | 0.4790 | 2.2264 | 3.0034 |
| test | 0.8211 | 1.3284 | 0.0896 | 0.4783 | 2.2096 | 2.7616 |

### E2 Leak-safe rebuild + faithful MoE
| Split | MAE | RMSE | Bias | P50 | P90 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.9066 | 1.4364 | 0.0704 | 0.5543 | 2.2547 | 2.9990 |
| validation | 0.8440 | 1.3794 | 0.0654 | 0.4901 | 2.1975 | 3.0440 |
| test | 0.8219 | 1.3309 | 0.0927 | 0.4803 | 2.1472 | 2.7519 |

### Alignment audit (sample 10 of 50)
| target_date_local | y_true | y_pred |
|---|---:|---:|
| 2024-03-07 | 86.00 | 85.10 |
| 2024-06-29 | 92.00 | 91.27 |
| 2024-04-25 | 83.00 | 81.83 |
| 2023-04-04 | 86.00 | 87.18 |
| 2025-05-03 | 83.00 | 83.95 |
| 2025-11-26 | 85.00 | 85.08 |
| 2024-11-21 | 78.00 | 78.00 |
| 2025-02-01 | 84.00 | 83.97 |
| 2023-10-28 | 86.00 | 87.18 |
| 2024-05-02 | 85.00 | 86.00 |

### Shifted-target MAE check
- MAE (correct): 0.8219
- MAE (shift +1): 2.1438
- MAE (shift -1): 2.1333

## Negative Controls (F)
| Control | Test MAE | Test Bias | PASS/FAIL |
|---|---:|---:|---|
| label_shuffle | 1.3165 | 0.1870 | PASS |
| feature_shift_plus7 | 2.4565 | 0.1437 | PASS |
| doy_only | 1.3057 | 0.0568 | PASS |
| constant_mean | 5.1744 | -1.0940 | PASS |
| leaky_cutoff_plus24h | 0.8422 | 0.0985 | PASS |

## Forward Simulation (G)
- Batch test MAE: 0.8219
- Forward test MAE: 0.8219
- Absolute MAE diff: 0.0000

### Per-day prediction diffs (top 20)
| target_date_local | y_true | y_pred_batch | y_pred_forward | abs_diff |
|---|---:|---:|---:|---:|
| (none) | - | - | - | - |

## Code & DB Analysis (additional)
- E37 is implemented in `ml/run_mos_45_suite.py` using `run_moe_gate` with gate features `[feat_u, feat_v, feat_wsp_mean, cal_d_doy_sin, cal_d_doy_cos]`, expert features `[feat_dd_models, feat_tmp_range_mean_models, feat_p12_max, feat_q12_max, feat_cig_min, feat_u, feat_v, cal_d_doy_sin, cal_d_doy_cos]`, gate target `feat_onshore`, and base series `feat_le_median_biascorr`.
- `feat_le_median_biascorr` is computed from bias-corrected lagged-ensemble members using EWMA errors shifted by 1 day; with `obs_cutoff_lag_days=0`, this uses truth through T-1 only.
- MOS selection uses `select_latest_mos` with `asof_utc <= cutoff` and `runtime_utc <= asof_utc`; retrieved_at is only used for tie-break ordering and is not a feature.
- The audit re-queries `mos_daily_value` and `station_daily_truth` directly and recomputes E37 features from raw MOS rows with strict cutoff logic.

## Rule Application & Failures
Failures detected: SplitLeakage_early_stopping_uses_val

FINAL VERDICT: LEAKAGE