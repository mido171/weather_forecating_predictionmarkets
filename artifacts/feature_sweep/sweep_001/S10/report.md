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
  "allow_tuning": false,
  "model": "lgbm",
  "params": {
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "n_estimators": 300,
    "num_leaves": 31
  }
}
```

## Metrics Summary
```json
{
  "test": {
    "bias": 0.27571286564486086,
    "corr": 0.9915100476118964,
    "mae": 1.85817982790556,
    "maxAE": 9.888158590562888,
    "medianAE": 1.4314280623427749,
    "n": 363,
    "rmse": 2.4238590425695685
  },
  "train": {
    "bias": -2.0163449704383085e-12,
    "corr": 0.9989791623987561,
    "mae": 0.5602237767715336,
    "maxAE": 6.779867891264285,
    "medianAE": 0.4458125287354342,
    "n": 1224,
    "rmse": 0.7484718794221134
  },
  "validation": {
    "bias": -0.3561979504028229,
    "corr": 0.9891749072125152,
    "mae": 1.9205627707567388,
    "maxAE": 8.563915153574499,
    "medianAE": 1.5407785848886775,
    "n": 184,
    "rmse": 2.4688641160846507
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 944.0
    },
    {
      "feature": "ens_wmean_minus_stack",
      "importance": 782.0
    },
    {
      "feature": "ens_wmean_minus_median",
      "importance": 696.0
    },
    {
      "feature": "stack_minus_median",
      "importance": 683.0
    },
    {
      "feature": "day_of_year",
      "importance": 640.0
    },
    {
      "feature": "cos_doy",
      "importance": 604.0
    },
    {
      "feature": "sin_doy",
      "importance": 590.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 545.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 542.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 540.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 503.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 438.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 396.0
    },
    {
      "feature": "stack_ridge_pred",
      "importance": 370.0
    },
    {
      "feature": "ens_median_6",
      "importance": 366.0
    },
    {
      "feature": "ens_wmean_6",
      "importance": 265.0
    },
    {
      "feature": "is_weekend",
      "importance": 49.0
    },
    {
      "feature": "month",
      "importance": 47.0
    },
    {
      "feature": "w_nbm",
      "importance": 0.0
    },
    {
      "feature": "w_gfs",
      "importance": 0.0
    },
    {
      "feature": "w_gefsatmosmean",
      "importance": 0.0
    },
    {
      "feature": "w_nam",
      "importance": 0.0
    },
    {
      "feature": "w_hrrr",
      "importance": 0.0
    },
    {
      "feature": "w_rap",
      "importance": 0.0
    }
  ],
  "type": "tree"
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
  "strategy": {
    "calendar": true,
    "description": "reliability_weighted_and_stacking_meta",
    "feature_columns": [
      "nbm_tmax_f",
      "gfs_tmax_f",
      "gefsatmosmean_tmax_f",
      "nam_tmax_f",
      "hrrr_tmax_f",
      "rap_tmax_f",
      "gefsatmos_tmp_spread_f",
      "ens_median_6",
      "w_nbm",
      "w_gfs",
      "w_gefsatmosmean",
      "w_nam",
      "w_hrrr",
      "w_rap",
      "ens_wmean_6",
      "ens_wmean_minus_median",
      "stack_ridge_pred",
      "ens_wmean_minus_stack",
      "stack_minus_median",
      "month",
      "day_of_year",
      "sin_doy",
      "cos_doy",
      "is_weekend"
    ],
    "include_spread": true,
    "model_cols": [
      "nbm_tmax_f",
      "gfs_tmax_f",
      "gefsatmosmean_tmax_f",
      "nam_tmax_f",
      "hrrr_tmax_f",
      "rap_tmax_f"
    ],
    "strategy_id": "S10"
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