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
    "bias": 0.4329932696142063,
    "corr": 0.9914872906706045,
    "mae": 1.8787520415560641,
    "maxAE": 9.82349589639994,
    "medianAE": 1.4520667995452925,
    "n": 363,
    "rmse": 2.4504380295934767
  },
  "train": {
    "bias": -5.175795014074508e-11,
    "corr": 0.9995913115593564,
    "mae": 0.32652091059955246,
    "maxAE": 6.494724946535687,
    "medianAE": 0.255158491239186,
    "n": 1224,
    "rmse": 0.47431800675479735
  },
  "validation": {
    "bias": -0.352660258202876,
    "corr": 0.9896994815088618,
    "mae": 1.8200836963639961,
    "maxAE": 7.981292726818978,
    "medianAE": 1.3341346271735262,
    "n": 184,
    "rmse": 2.4272428128560155
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 466.0
    },
    {
      "feature": "cos_doy",
      "importance": 348.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 336.0
    },
    {
      "feature": "rap_tmax_f_minus_ens_median_abs",
      "importance": 336.0
    },
    {
      "feature": "sin_doy",
      "importance": 314.0
    },
    {
      "feature": "rap_tmax_f_minus_ens_mean_abs",
      "importance": 306.0
    },
    {
      "feature": "nbm_tmax_f_minus_ens_mean_abs",
      "importance": 305.0
    },
    {
      "feature": "nbm_tmax_f_minus_ens_median_abs",
      "importance": 302.0
    },
    {
      "feature": "day_of_year",
      "importance": 294.0
    },
    {
      "feature": "nam_tmax_f_minus_ens_mean_abs",
      "importance": 293.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_ens_mean_abs",
      "importance": 278.0
    },
    {
      "feature": "nam_tmax_f_minus_ens_mean",
      "importance": 273.0
    },
    {
      "feature": "rap_tmax_f_minus_ens_median",
      "importance": 273.0
    },
    {
      "feature": "hrrr_tmax_f_minus_ens_mean_abs",
      "importance": 270.0
    },
    {
      "feature": "nbm_tmax_f_minus_ens_median",
      "importance": 265.0
    },
    {
      "feature": "rap_tmax_f_minus_ens_mean",
      "importance": 259.0
    },
    {
      "feature": "gfs_tmax_f_minus_ens_mean_abs",
      "importance": 257.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_ens_median_abs",
      "importance": 234.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 230.0
    },
    {
      "feature": "hrrr_tmax_f_minus_ens_mean",
      "importance": 230.0
    },
    {
      "feature": "hrrr_tmax_f_minus_ens_median_abs",
      "importance": 230.0
    },
    {
      "feature": "nbm_tmax_f_minus_ens_mean",
      "importance": 224.0
    },
    {
      "feature": "gfs_tmax_f_minus_ens_median_abs",
      "importance": 216.0
    },
    {
      "feature": "ens_median_6",
      "importance": 212.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_ens_median",
      "importance": 212.0
    },
    {
      "feature": "ens_mean_6",
      "importance": 210.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 205.0
    },
    {
      "feature": "hrrr_tmax_f_minus_ens_median",
      "importance": 203.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_ens_mean",
      "importance": 188.0
    },
    {
      "feature": "nam_tmax_f_minus_ens_median_abs",
      "importance": 184.0
    },
    {
      "feature": "gfs_tmax_f_minus_ens_mean",
      "importance": 177.0
    },
    {
      "feature": "gfs_tmax_f_minus_ens_median",
      "importance": 177.0
    },
    {
      "feature": "nam_tmax_f_minus_ens_median",
      "importance": 174.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 169.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 156.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 117.0
    },
    {
      "feature": "month",
      "importance": 47.0
    },
    {
      "feature": "is_weekend",
      "importance": 30.0
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
    "description": "add_model_vs_consensus_deltas",
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
      "nbm_tmax_f_minus_ens_mean",
      "nbm_tmax_f_minus_ens_mean_abs",
      "nbm_tmax_f_minus_ens_median",
      "nbm_tmax_f_minus_ens_median_abs",
      "gfs_tmax_f_minus_ens_mean",
      "gfs_tmax_f_minus_ens_mean_abs",
      "gfs_tmax_f_minus_ens_median",
      "gfs_tmax_f_minus_ens_median_abs",
      "gefsatmosmean_tmax_f_minus_ens_mean",
      "gefsatmosmean_tmax_f_minus_ens_mean_abs",
      "gefsatmosmean_tmax_f_minus_ens_median",
      "gefsatmosmean_tmax_f_minus_ens_median_abs",
      "nam_tmax_f_minus_ens_mean",
      "nam_tmax_f_minus_ens_mean_abs",
      "nam_tmax_f_minus_ens_median",
      "nam_tmax_f_minus_ens_median_abs",
      "hrrr_tmax_f_minus_ens_mean",
      "hrrr_tmax_f_minus_ens_mean_abs",
      "hrrr_tmax_f_minus_ens_median",
      "hrrr_tmax_f_minus_ens_median_abs",
      "rap_tmax_f_minus_ens_mean",
      "rap_tmax_f_minus_ens_mean_abs",
      "rap_tmax_f_minus_ens_median",
      "rap_tmax_f_minus_ens_median_abs",
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
    "strategy_id": "S06"
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