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
    "bias": 0.3392103276650728,
    "corr": 0.9917706137896442,
    "mae": 1.8233745428060484,
    "maxAE": 9.315850611441917,
    "medianAE": 1.4450810317990346,
    "n": 363,
    "rmse": 2.393857680967293
  },
  "train": {
    "bias": 2.299526233916733e-10,
    "corr": 0.9984278332481824,
    "mae": 0.7054075988897202,
    "maxAE": 6.967873671449375,
    "medianAE": 0.5626503520827111,
    "n": 1224,
    "rmse": 0.928239448349184
  },
  "validation": {
    "bias": -0.34791659544399606,
    "corr": 0.9892267412669103,
    "mae": 1.8849966628132042,
    "maxAE": 7.856461689708539,
    "medianAE": 1.558761547881133,
    "n": 184,
    "rmse": 2.4714029954850343
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 1396.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 1056.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 953.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 939.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 896.0
    },
    {
      "feature": "cos_doy",
      "importance": 842.0
    },
    {
      "feature": "sin_doy",
      "importance": 760.0
    },
    {
      "feature": "day_of_year",
      "importance": 751.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 678.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 592.0
    },
    {
      "feature": "is_weekend",
      "importance": 75.0
    },
    {
      "feature": "month",
      "importance": 62.0
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
    "description": "raw_all_6_plus_calendar",
    "feature_columns": [
      "nbm_tmax_f",
      "gfs_tmax_f",
      "gefsatmosmean_tmax_f",
      "nam_tmax_f",
      "hrrr_tmax_f",
      "rap_tmax_f",
      "gefsatmos_tmp_spread_f",
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
    "strategy_id": "S02"
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