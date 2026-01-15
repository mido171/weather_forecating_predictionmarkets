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
    "cos_doy": 0,
    "day_of_year": 0,
    "ens_iqr": 0,
    "ens_mad": 0,
    "ens_max": 0,
    "ens_mean": 0,
    "ens_median": 0,
    "ens_min": 0,
    "ens_outlier_gap": 0,
    "ens_range": 0,
    "ens_std": 0,
    "gefs_spread": 0,
    "gefsatmos_tmp_spread_f": 0,
    "gefsatmosmean_tmax_f": 0,
    "gfs_tmax_f": 0,
    "hrrr_tmax_f": 0,
    "is_weekend": 0,
    "month": 0,
    "nam_tmax_f": 0,
    "nbm_tmax_f": 0,
    "rap_tmax_f": 0,
    "resid_ens_mean": 0,
    "resid_ens_median": 0,
    "sin_doy": 0,
    "station_id": 0,
    "target_date_local": 0
  },
  "row_count": 1770,
  "split_counts": {
    "test": 332,
    "train": 1224,
    "validation": 214
  },
  "station_counts": {
    "KNYC": 1770
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
    "bias": 0.45142724344699703,
    "corr": 0.9906781439871861,
    "mae": 1.8702110390515108,
    "maxAE": 9.6414064066296,
    "medianAE": 1.5996161655278556,
    "n": 332,
    "rmse": 2.3992135016823357
  },
  "train": {
    "bias": -1.108250571922897e-10,
    "corr": 0.9989372951640165,
    "mae": 0.5706328291766388,
    "maxAE": 6.931812368359257,
    "medianAE": 0.45959941769967116,
    "n": 1224,
    "rmse": 0.7637011951739624
  },
  "validation": {
    "bias": -0.3380249499898285,
    "corr": 0.9920778038197386,
    "mae": 1.9161046993258672,
    "maxAE": 7.738919199127547,
    "medianAE": 1.5064313192541547,
    "n": 214,
    "rmse": 2.4911529023516863
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 1235.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 805.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 712.0
    },
    {
      "feature": "days_since_warm_bust_l2",
      "importance": 706.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 700.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 678.0
    },
    {
      "feature": "days_since_cold_bust_l2",
      "importance": 657.0
    },
    {
      "feature": "sin_doy",
      "importance": 643.0
    },
    {
      "feature": "cos_doy",
      "importance": 636.0
    },
    {
      "feature": "day_of_year",
      "importance": 588.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 516.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 500.0
    },
    {
      "feature": "bust_balance_60_l2",
      "importance": 216.0
    },
    {
      "feature": "warm_bust_count_60_l2",
      "importance": 153.0
    },
    {
      "feature": "cold_bust_count_60_l2",
      "importance": 144.0
    },
    {
      "feature": "is_weekend",
      "importance": 72.0
    },
    {
      "feature": "month",
      "importance": 39.0
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
  "experiment": {
    "derived_features": {
      "formulas": [
        {
          "formula": "days_since(cold_bust)",
          "name": "days_since_cold_bust_l2",
          "params": {
            "cap": 365,
            "lag": 2
          }
        },
        {
          "formula": "days_since(warm_bust)",
          "name": "days_since_warm_bust_l2",
          "params": {
            "cap": 365,
            "lag": 2
          }
        },
        {
          "formula": "roll_sum(cold_bust)",
          "name": "cold_bust_count_60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_sum(warm_bust)",
          "name": "warm_bust_count_60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "cold_bust_count_60_l - warm_bust_count_60_l",
          "name": "bust_balance_60_l2"
        }
      ],
      "imputation": {
        "fill_values": {
          "bust_balance_60_l2": 1.0,
          "cold_bust_count_60_l2": 2.0,
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "days_since_cold_bust_l2": 15.0,
          "days_since_warm_bust_l2": 22.0,
          "gefsatmos_tmp_spread_f": 1.985091821481542,
          "gefsatmosmean_tmax_f": 63.38288233996978,
          "gfs_tmax_f": 64.5,
          "hrrr_tmax_f": 63.429676513671915,
          "is_weekend": 0.0,
          "month": 6.0,
          "nam_tmax_f": 63.0,
          "nbm_tmax_f": 63.175991210937525,
          "rap_tmax_f": 62.58964813232426,
          "sin_doy": 0.15951094710994368,
          "warm_bust_count_60_l2": 2.0
        },
        "method": "train_median"
      },
      "train_fitted": [
        {
          "description": "q05/q95_train(resid_ens_mean) per station",
          "fit_on": "train",
          "name": "thr_bust",
          "thr_cold": {
            "KNYC": -2.6133007195585836
          },
          "thr_warm": {
            "KNYC": 5.680269374888397
          }
        }
      ]
    },
    "description": "Cold/warm bust memory",
    "experiment_id": "E49",
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
      "is_weekend",
      "days_since_cold_bust_l2",
      "days_since_warm_bust_l2",
      "cold_bust_count_60_l2",
      "warm_bust_count_60_l2",
      "bust_balance_60_l2"
    ]
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
        "ridge",
        "random_forest",
        "gbr",
        "lgbm",
        "xgb",
        "catboost"
      ],
      "param_grid": {
        "catboost": {
          "depth": [
            6,
            8
          ],
          "iterations": [
            500
          ],
          "l2_leaf_reg": [
            3,
            10
          ],
          "learning_rate": [
            0.05,
            0.1
          ]
        },
        "gbr": {
          "learning_rate": [
            0.05,
            0.1
          ],
          "max_depth": [
            3,
            5
          ],
          "n_estimators": [
            200
          ]
        },
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
        },
        "random_forest": {
          "max_depth": [
            8,
            16,
            null
          ],
          "min_samples_leaf": [
            1,
            3
          ],
          "n_estimators": [
            200
          ]
        },
        "ridge": {
          "alpha": [
            0.1,
            1.0,
            10.0
          ]
        },
        "xgb": {
          "colsample_bytree": [
            0.8
          ],
          "learning_rate": [
            0.05,
            0.1
          ],
          "max_depth": [
            4,
            6
          ],
          "min_child_weight": [
            1,
            5
          ],
          "n_estimators": [
            300
          ],
          "reg_lambda": [
            1.0
          ],
          "subsample": [
            0.8
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
      "cal_end": null,
      "cal_start": null,
      "ddof": 1,
      "enabled": false,
      "forecast_columns": [],
      "station_scope": null
    },
    "global_normal_calibration": {
      "cal_end": null,
      "cal_start": null,
      "ddof": 1,
      "enabled": false,
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
    "gap_dates": [
      "2025-01-31"
    ],
    "test_end": "2025-12-31",
    "test_start": "2025-02-01",
    "train_end": "2025-01-30",
    "train_start": "2021-02-23",
    "validation": {
      "enabled": true,
      "val_end": "2025-01-30",
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