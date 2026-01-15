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
    "bias": 0.394746297300892,
    "corr": 0.9914088588095153,
    "mae": 1.8888495566676802,
    "maxAE": 8.259472280102088,
    "medianAE": 1.5376025046669781,
    "n": 363,
    "rmse": 2.4526480762787153
  },
  "train": {
    "bias": -9.336217202388195e-11,
    "corr": 0.998945662965711,
    "mae": 0.5654046709674955,
    "maxAE": 6.031895736537162,
    "medianAE": 0.46151000332276126,
    "n": 1224,
    "rmse": 0.7608643982377872
  },
  "validation": {
    "bias": -0.1542156160782601,
    "corr": 0.9888574956928762,
    "mae": 1.8778786516906167,
    "maxAE": 8.930470111658025,
    "medianAE": 1.4362598539819693,
    "n": 184,
    "rmse": 2.4745581395788654
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 837.0
    },
    {
      "feature": "ens_range_3",
      "importance": 668.0
    },
    {
      "feature": "ens_std_3",
      "importance": 648.0
    },
    {
      "feature": "cos_doy",
      "importance": 618.0
    },
    {
      "feature": "day_of_year",
      "importance": 610.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 585.0
    },
    {
      "feature": "sin_doy",
      "importance": 584.0
    },
    {
      "feature": "gfs_tmax_f_minus_ens_median_abs",
      "importance": 560.0
    },
    {
      "feature": "gfs_tmax_f_minus_ens_median",
      "importance": 540.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_ens_median",
      "importance": 537.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_ens_median_abs",
      "importance": 478.0
    },
    {
      "feature": "nbm_tmax_f_minus_ens_median",
      "importance": 371.0
    },
    {
      "feature": "ens_median_3",
      "importance": 331.0
    },
    {
      "feature": "ens_min_3",
      "importance": 303.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 274.0
    },
    {
      "feature": "nbm_tmax_f_minus_ens_median_abs",
      "importance": 261.0
    },
    {
      "feature": "ens_mean_3",
      "importance": 256.0
    },
    {
      "feature": "ens_max_3",
      "importance": 250.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 189.0
    },
    {
      "feature": "is_weekend",
      "importance": 55.0
    },
    {
      "feature": "month",
      "importance": 45.0
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
    "description": "top3_only_plus_robust (ens_closest3_mean omitted)",
    "feature_columns": [
      "nbm_tmax_f",
      "gfs_tmax_f",
      "gefsatmosmean_tmax_f",
      "gefsatmos_tmp_spread_f",
      "ens_mean_3",
      "ens_median_3",
      "ens_min_3",
      "ens_max_3",
      "ens_std_3",
      "ens_range_3",
      "nbm_tmax_f_minus_ens_median",
      "nbm_tmax_f_minus_ens_median_abs",
      "gfs_tmax_f_minus_ens_median",
      "gfs_tmax_f_minus_ens_median_abs",
      "gefsatmosmean_tmax_f_minus_ens_median",
      "gefsatmosmean_tmax_f_minus_ens_median_abs",
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
      "gefsatmosmean_tmax_f"
    ],
    "strategy_id": "S08"
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