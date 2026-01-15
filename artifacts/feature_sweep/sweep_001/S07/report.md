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
    "bias": 0.42910652582856823,
    "corr": 0.9916523640077456,
    "mae": 1.872457679331737,
    "maxAE": 10.848910351534812,
    "medianAE": 1.5235665583826403,
    "n": 363,
    "rmse": 2.4248384956646265
  },
  "train": {
    "bias": 9.037357354843296e-11,
    "corr": 0.9993879322710026,
    "mae": 0.41009601910099064,
    "maxAE": 7.219652365291417,
    "medianAE": 0.3141988415008896,
    "n": 1224,
    "rmse": 0.580239316352962
  },
  "validation": {
    "bias": -0.3697793366579188,
    "corr": 0.9898099348182251,
    "mae": 1.8517082377266942,
    "maxAE": 7.787370003716703,
    "medianAE": 1.3648172177862392,
    "n": 184,
    "rmse": 2.4194047207142684
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "ens_std_5",
      "importance": 672.0
    },
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 610.0
    },
    {
      "feature": "ens_iqr_5",
      "importance": 548.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_ens_median",
      "importance": 467.0
    },
    {
      "feature": "cos_doy",
      "importance": 451.0
    },
    {
      "feature": "sin_doy",
      "importance": 434.0
    },
    {
      "feature": "gfs_tmax_f_minus_ens_median_abs",
      "importance": 410.0
    },
    {
      "feature": "day_of_year",
      "importance": 403.0
    },
    {
      "feature": "nam_tmax_f_minus_ens_median_abs",
      "importance": 389.0
    },
    {
      "feature": "hrrr_tmax_f_minus_ens_median_abs",
      "importance": 386.0
    },
    {
      "feature": "nam_tmax_f_minus_ens_median",
      "importance": 366.0
    },
    {
      "feature": "hrrr_tmax_f_minus_ens_median",
      "importance": 339.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 334.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 334.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_ens_median_abs",
      "importance": 333.0
    },
    {
      "feature": "nbm_tmax_f_minus_ens_median",
      "importance": 318.0
    },
    {
      "feature": "nbm_tmax_f_minus_ens_median_abs",
      "importance": 284.0
    },
    {
      "feature": "gfs_tmax_f_minus_ens_median",
      "importance": 276.0
    },
    {
      "feature": "ens_max_5",
      "importance": 272.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 211.0
    },
    {
      "feature": "ens_median_5",
      "importance": 200.0
    },
    {
      "feature": "ens_mean_5",
      "importance": 168.0
    },
    {
      "feature": "ens_min_5",
      "importance": 165.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 160.0
    },
    {
      "feature": "ens_trimmed_mean_5_1",
      "importance": 156.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 148.0
    },
    {
      "feature": "ens_winsor_mean_5_1",
      "importance": 69.0
    },
    {
      "feature": "month",
      "importance": 51.0
    },
    {
      "feature": "is_weekend",
      "importance": 46.0
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
    "description": "drop_worst_rap_plus_derived",
    "feature_columns": [
      "nbm_tmax_f",
      "gfs_tmax_f",
      "gefsatmosmean_tmax_f",
      "nam_tmax_f",
      "hrrr_tmax_f",
      "gefsatmos_tmp_spread_f",
      "ens_mean_5",
      "ens_median_5",
      "ens_min_5",
      "ens_max_5",
      "ens_trimmed_mean_5_1",
      "ens_winsor_mean_5_1",
      "ens_std_5",
      "ens_iqr_5",
      "nbm_tmax_f_minus_ens_median",
      "nbm_tmax_f_minus_ens_median_abs",
      "gfs_tmax_f_minus_ens_median",
      "gfs_tmax_f_minus_ens_median_abs",
      "gefsatmosmean_tmax_f_minus_ens_median",
      "gefsatmosmean_tmax_f_minus_ens_median_abs",
      "nam_tmax_f_minus_ens_median",
      "nam_tmax_f_minus_ens_median_abs",
      "hrrr_tmax_f_minus_ens_median",
      "hrrr_tmax_f_minus_ens_median_abs",
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
      "hrrr_tmax_f"
    ],
    "strategy_id": "S07"
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