# Training Report

## Dataset Summary
```json
{
  "date_coverage": {
    "max": "2025-12-31",
    "min": "2021-02-23"
  },
  "missing_by_column": {
    "actual_tmax_f": 0,
    "asof_utc": 0,
    "gefsatmos_tmp_spread_f": 0,
    "gefsatmosmean_tmax_f": 0,
    "gfs_tmax_f": 0,
    "hrrr_tmax_f": 0,
    "nam_tmax_f": 0,
    "nbm_tmax_f": 0,
    "rap_tmax_f": 0,
    "station_id": 0,
    "target_date_local": 0
  },
  "row_count": 1771,
  "split_counts": {
    "test": 363,
    "train": 1224,
    "validation": 184
  },
  "station_counts": {
    "KNYC": 1771
  }
}
```

## Model Summary
```json
{
  "best_params": {
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "lambda_l1": 0.1,
    "lambda_l2": 0.0,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "n_estimators": 300,
    "num_leaves": 31
  },
  "candidates": [
    {
      "best_params": {
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "lambda_l1": 0.1,
        "lambda_l2": 0.0,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "n_estimators": 300,
        "num_leaves": 31
      },
      "cv_score": -2.395654948836431,
      "model": "lgbm",
      "val_mae": 1.7931864916895268
    }
  ],
  "selected_model": "lgbm",
  "sigma_model": "lgbm"
}
```

## Metrics Summary
```json
{
  "baseline": {
    "climatology_test": {
      "bias": 1.878099173553719,
      "mae": 7.247245179063361,
      "p50_abs_error": 6.0,
      "p90_abs_error": 15.466666666666669,
      "p95_abs_error": 18.224999999999994,
      "rmse": 9.302089454941393
    },
    "climatology_val": {
      "bias": -0.3800174055129297,
      "mae": 6.145344913327649,
      "p50_abs_error": 4.833333333333336,
      "p90_abs_error": 12.233333333333341,
      "p95_abs_error": 15.616666666666662,
      "rmse": 7.826476359449888
    },
    "ensemble_mean_test": {
      "bias": -1.006985970393393,
      "mae": 2.026175480053542,
      "p50_abs_error": 1.7249564865275957,
      "p90_abs_error": 3.9618212977255185,
      "p95_abs_error": 5.0827168228641275,
      "rmse": 2.5600414415064736
    },
    "ensemble_mean_val": {
      "bias": -1.0084753178243502,
      "mae": 1.8594145773146986,
      "p50_abs_error": 1.6392003879341708,
      "p90_abs_error": 3.8295593884539487,
      "p95_abs_error": 4.937016528714049,
      "rmse": 2.3912225695300124
    }
  },
  "per_station_test": {
    "KNYC": {
      "bias": 0.486378302696682,
      "mae": 1.9250826387660744,
      "p50_abs_error": 1.5465052854041232,
      "p90_abs_error": 4.058776567937561,
      "p95_abs_error": 5.2251107650230635,
      "rmse": 2.513279617626549
    }
  },
  "probabilistic_test": {
    "brier_scores": {
      "ge_90": 0.026618199648782998,
      "lt_52": 0.04280116878213198
    },
    "log_loss": 3.530218480349292
  },
  "probabilistic_validation": {
    "brier_scores": {
      "ge_90": 0.02824459382040443,
      "lt_52": 0.02715698373499319
    },
    "log_loss": 3.569712275337055
  },
  "test": {
    "bias": 0.486378302696682,
    "mae": 1.9250826387660744,
    "p50_abs_error": 1.5465052854041232,
    "p90_abs_error": 4.058776567937561,
    "p95_abs_error": 5.2251107650230635,
    "rmse": 2.513279617626549
  },
  "train": {
    "bias": 7.391236156828317e-06,
    "mae": 0.3376094024739352,
    "p50_abs_error": 0.2556889791709622,
    "p90_abs_error": 0.6961985502456082,
    "p95_abs_error": 0.9071125574686391,
    "rmse": 0.49432704383385456
  },
  "validation": {
    "bias": -0.4066735941889763,
    "mae": 1.7931864916895268,
    "p50_abs_error": 1.3572778130573937,
    "p90_abs_error": 3.9554880359107436,
    "p95_abs_error": 5.014035991903153,
    "rmse": 2.377351141705024
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 436.0
    },
    {
      "feature": "ens_range",
      "importance": 392.0
    },
    {
      "feature": "ens_std",
      "importance": 379.0
    },
    {
      "feature": "ens_iqr",
      "importance": 376.0
    },
    {
      "feature": "cos_doy",
      "importance": 376.0
    },
    {
      "feature": "nbm_tmax_f_minus_ens_mean_abs",
      "importance": 348.0
    },
    {
      "feature": "rap_tmax_f_minus_ens_mean_abs",
      "importance": 337.0
    },
    {
      "feature": "hrrr_tmax_f_minus_ens_mean_abs",
      "importance": 337.0
    },
    {
      "feature": "nam_tmax_f_minus_ens_mean_abs",
      "importance": 332.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_ens_mean_abs",
      "importance": 331.0
    },
    {
      "feature": "rap_tmax_f_minus_nbm_tmax_f",
      "importance": 315.0
    },
    {
      "feature": "sin_doy",
      "importance": 302.0
    },
    {
      "feature": "rap_tmax_f_minus_ens_mean",
      "importance": 293.0
    },
    {
      "feature": "day_of_year",
      "importance": 285.0
    },
    {
      "feature": "hrrr_tmax_f_minus_gfs_tmax_f",
      "importance": 281.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 279.0
    },
    {
      "feature": "nam_tmax_f_minus_nbm_tmax_f",
      "importance": 273.0
    },
    {
      "feature": "nbm_tmax_f_minus_ens_mean",
      "importance": 273.0
    },
    {
      "feature": "gfs_tmax_f_minus_ens_mean_abs",
      "importance": 258.0
    },
    {
      "feature": "nbm_tmax_f_minus_hrrr_tmax_f",
      "importance": 251.0
    },
    {
      "feature": "hrrr_tmax_f_minus_ens_mean",
      "importance": 238.0
    },
    {
      "feature": "nam_tmax_f_minus_ens_mean",
      "importance": 227.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_gfs_tmax_f",
      "importance": 216.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_ens_mean",
      "importance": 204.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 192.0
    },
    {
      "feature": "gfs_tmax_f_minus_ens_mean",
      "importance": 187.0
    },
    {
      "feature": "ens_median",
      "importance": 183.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 182.0
    },
    {
      "feature": "ens_max",
      "importance": 177.0
    },
    {
      "feature": "ens_mean",
      "importance": 144.0
    },
    {
      "feature": "ens_min",
      "importance": 142.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 122.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 94.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 93.0
    },
    {
      "feature": "gfs_tmax_f_minus_nam_tmax_f",
      "importance": 74.0
    },
    {
      "feature": "month",
      "importance": 44.0
    },
    {
      "feature": "is_weekend",
      "importance": 27.0
    },
    {
      "feature": "gfs_tmax_f_missing",
      "importance": 0.0
    },
    {
      "feature": "nam_tmax_f_missing",
      "importance": 0.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_missing",
      "importance": 0.0
    },
    {
      "feature": "rap_tmax_f_missing",
      "importance": 0.0
    },
    {
      "feature": "hrrr_tmax_f_missing",
      "importance": 0.0
    },
    {
      "feature": "nbm_tmax_f_missing",
      "importance": 0.0
    },
    {
      "feature": "gefsatmos_tmp_spread_f_missing",
      "importance": 0.0
    },
    {
      "feature": "station_KNYC",
      "importance": 0.0
    }
  ],
  "type": "tree"
}
```

## Global Normal Calibration (2025)

Calibration window: 2025-01-01 to 2025-12-31
Rows used: 363

```json
{
  "bias_mean_error_f": -0.486378302696682,
  "calibration_window": {
    "end": "2025-12-31",
    "start": "2025-01-01"
  },
  "created_utc": "2026-01-14T12:50:09Z",
  "dataset_ref": {
    "dataset_hash": "f6b85f651688c71dae39804536b17739674ba99558b3d58188f79fcd51a1bb29",
    "dataset_id": "37e63ed3c15b93d8df31499e4c2818ee1f509b767b9c51bf0ab3019e021370bd",
    "rows_used": 363
  },
  "ddof": 1,
  "error_definition": "e = actual_tmax_f - mu_hat",
  "mae_f": 1.9250826387660744,
  "method": "global_normal_residual",
  "model_ref": {
    "mean_model_artifact": "mean_model.joblib",
    "model_hash": "109b40f7968405b6626c656c5fcf28c2c8aba48e2427d0cbe6e85ff74018a5b8",
    "run_dir": "C:\\Users\\ahmad\\Desktop\\generalFiles\\git\\weather-forecasting-predictionmarkets\\weather_forecating_predictionmarkets\\artifacts\\runs\\my_run_2025_2"
  },
  "n": 363,
  "residual_quantiles_f": {
    "p01": -6.551104143156936,
    "p05": -4.811580703614967,
    "p10": -3.590889560942162,
    "p50": -0.36662027706272227,
    "p90": 2.4507139578604207,
    "p95": 2.9561268460080177,
    "p99": 5.291303364306234
  },
  "rmse_f": 2.513279617626549,
  "sigma_std_error_f": 2.4691711549401285,
  "station_scope": {
    "mode": "ALL",
    "stations": []
  }
}
```

## Baseline Median Calibration (2021-2025)

Calibration window: 2021-02-23 to 2025-12-31
Rows used: 1771

```json
{
  "bias_mean_error_f": 1.2650826472566772,
  "calibration_window": {
    "end": "2025-12-31",
    "start": "2021-02-23"
  },
  "created_utc": "2026-01-14T12:50:09Z",
  "dataset_ref": {
    "dataset_hash": "f6b85f651688c71dae39804536b17739674ba99558b3d58188f79fcd51a1bb29",
    "dataset_id": "37e63ed3c15b93d8df31499e4c2818ee1f509b767b9c51bf0ab3019e021370bd",
    "rows_used": 1771
  },
  "ddof": 1,
  "error_definition": "e = actual_tmax_f - median_forecast_f",
  "forecast_columns": [
    "gfs_tmax_f",
    "nam_tmax_f",
    "gefsatmosmean_tmax_f",
    "rap_tmax_f",
    "hrrr_tmax_f",
    "nbm_tmax_f"
  ],
  "mae_f": 2.154745109210591,
  "method": "baseline_median_residual",
  "n": 1771,
  "residual_quantiles_f": {
    "p01": -4.4575614013672435,
    "p05": -2.6714450073242517,
    "p10": -1.7534118652344262,
    "p50": 1.0990087890624665,
    "p90": 4.424562200730833,
    "p95": 5.436510162353475,
    "p99": 8.177164459228495
  },
  "rmse_f": 2.8144026779797136,
  "sigma_std_error_f": 2.5147562889656676,
  "station_scope": {
    "mode": "ALL",
    "stations": []
  }
}
```

## Config Snapshot
```json
{
  "artifacts": {
    "overwrite": false,
    "root_dir": "C:\\Users\\ahmad\\Desktop\\generalFiles\\git\\weather-forecasting-predictionmarkets\\weather_forecating_predictionmarkets\\artifacts",
    "run_id": null
  },
  "calibration": {
    "bins_to_calibrate": [
      {
        "lt": 52,
        "name": "lt_52",
        "type": "threshold"
      },
      {
        "ge": 90,
        "name": "ge_90",
        "type": "threshold"
      }
    ],
    "enabled": true,
    "method": "isotonic"
  },
  "data": {
    "csv_path": "C:\\Users\\ahmad\\Desktop\\generalFiles\\git\\weather-forecasting-predictionmarkets\\weather_forecating_predictionmarkets\\ingestion-service\\src\\main\\resources\\trainingdata_output\\gribstream_training_data.csv",
    "dataset_schema_version": 1
  },
  "distribution": {
    "support_max_f": 130,
    "support_min_f": -30
  },
  "features": {
    "base_features": [
      "gfs_tmax_f",
      "nam_tmax_f",
      "gefsatmosmean_tmax_f",
      "rap_tmax_f",
      "hrrr_tmax_f",
      "nbm_tmax_f",
      "gefsatmos_tmp_spread_f"
    ],
    "calendar": true,
    "climatology": {
      "enabled": false,
      "label_lag_days": 2,
      "rolling_windows_days": [
        7,
        30
      ]
    },
    "ensemble_stats": true,
    "model_vs_ens_deltas": true,
    "pairwise_deltas": true,
    "pairwise_pairs": [],
    "station_onehot": true
  },
  "models": {
    "mean": {
      "candidates": [
        "lgbm"
      ],
      "param_grid": {
        "lgbm": {
          "bagging_fraction": [
            0.8
          ],
          "feature_fraction": [
            0.8
          ],
          "lambda_l1": [
            0.0,
            0.1
          ],
          "lambda_l2": [
            0.0,
            0.1
          ],
          "learning_rate": [
            0.05,
            0.1
          ],
          "min_data_in_leaf": [
            20,
            50
          ],
          "n_estimators": [
            300
          ],
          "num_leaves": [
            31,
            63
          ]
        }
      },
      "primary": "lgbm"
    },
    "sigma": {
      "eps": 1e-06,
      "method": "two_stage",
      "param_grid": {
        "lgbm": {
          "learning_rate": [
            0.05,
            0.1
          ],
          "min_data_in_leaf": [
            20,
            50
          ],
          "n_estimators": [
            300
          ],
          "num_leaves": [
            31,
            63
          ]
        }
      },
      "primary": "lgbm",
      "sigma_floor": 0.25
    }
  },
  "postprocess": {
    "baseline_median_calibration": {
      "cal_end": "2025-12-31",
      "cal_start": "2021-02-23",
      "ddof": 1,
      "enabled": true,
      "forecast_columns": [],
      "station_scope": null
    },
    "global_normal_calibration": {
      "cal_end": "2025-12-31",
      "cal_start": "2025-01-01",
      "ddof": 1,
      "enabled": true,
      "station_scope": null
    }
  },
  "seeds": {
    "force_single_thread": true,
    "global_seed": 1337
  },
  "split": {
    "cv": {
      "enabled": true,
      "gap_days": 2,
      "n_splits": 5
    },
    "gap_dates": [],
    "test_end": "2025-12-31",
    "test_start": "2025-01-01",
    "train_end": "2024-12-31",
    "train_start": "2021-02-23",
    "validation": {
      "enabled": true,
      "val_end": "2024-12-31",
      "val_start": "2024-07-01"
    }
  },
  "validation": {
    "forecast_max_f": 140.0,
    "forecast_min_f": -80.0,
    "require_asof_not_after_target": true,
    "spread_min_f": 0.0,
    "strict_schema": true
  }
}
```