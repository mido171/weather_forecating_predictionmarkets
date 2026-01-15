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
    "bias": 0.3545179111946714,
    "corr": 0.9914234216236929,
    "mae": 1.8374441406752196,
    "maxAE": 10.921719463515402,
    "medianAE": 1.3982459855929363,
    "n": 363,
    "rmse": 2.4448087456663163
  },
  "train": {
    "bias": 8.893978944754612e-11,
    "corr": 0.998571837343503,
    "mae": 0.6602977308371885,
    "maxAE": 6.669339528483018,
    "medianAE": 0.5216033278991503,
    "n": 1224,
    "rmse": 0.8849126940126014
  },
  "validation": {
    "bias": -0.44650850521487195,
    "corr": 0.9896467973182573,
    "mae": 1.8339419771433099,
    "maxAE": 8.96077063054598,
    "medianAE": 1.422463374747391,
    "n": 184,
    "rmse": 2.42998399884983
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 1037.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 784.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 682.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 681.0
    },
    {
      "feature": "day_of_year",
      "importance": 656.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 651.0
    },
    {
      "feature": "ens_min_6",
      "importance": 629.0
    },
    {
      "feature": "sin_doy",
      "importance": 626.0
    },
    {
      "feature": "cos_doy",
      "importance": 577.0
    },
    {
      "feature": "ens_max_6",
      "importance": 511.0
    },
    {
      "feature": "ens_median_6",
      "importance": 411.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 408.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 406.0
    },
    {
      "feature": "ens_mean_6",
      "importance": 341.0
    },
    {
      "feature": "ens_winsor_mean_6_1",
      "importance": 265.0
    },
    {
      "feature": "ens_trimmed_mean_6_1",
      "importance": 216.0
    },
    {
      "feature": "is_weekend",
      "importance": 62.0
    },
    {
      "feature": "month",
      "importance": 57.0
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
    "description": "add_robust_blends",
    "feature_columns": [
      "nbm_tmax_f",
      "gfs_tmax_f",
      "gefsatmosmean_tmax_f",
      "nam_tmax_f",
      "hrrr_tmax_f",
      "rap_tmax_f",
      "gefsatmos_tmp_spread_f",
      "ens_mean_6",
      "ens_median_6",
      "ens_min_6",
      "ens_max_6",
      "ens_trimmed_mean_6_1",
      "ens_winsor_mean_6_1",
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
    "strategy_id": "S04"
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