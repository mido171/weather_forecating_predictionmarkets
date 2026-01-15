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
    "bias": 0.21051550033869332,
    "corr": 0.9910782840384839,
    "mae": 1.801313019238597,
    "maxAE": 8.35163666106586,
    "medianAE": 1.4560181176219125,
    "n": 332,
    "rmse": 2.3165521424693187
  },
  "train": {
    "bias": -7.47370792704019e-12,
    "corr": 0.9992494788305222,
    "mae": 0.47616463630142447,
    "maxAE": 6.920834544458902,
    "medianAE": 0.37982531945671383,
    "n": 1224,
    "rmse": 0.6427235224007577
  },
  "validation": {
    "bias": -0.23511086377133475,
    "corr": 0.9921670926717677,
    "mae": 1.9337185814791944,
    "maxAE": 7.139326847266119,
    "medianAE": 1.552220333265545,
    "n": 214,
    "rmse": 2.457677889329216
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 644.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 596.0
    },
    {
      "feature": "gfs_tmax_f_roll_std_15_l1",
      "importance": 477.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_roll_std_15_l1",
      "importance": 472.0
    },
    {
      "feature": "nam_tmax_f_roll_std_15_l1",
      "importance": 461.0
    },
    {
      "feature": "hrrr_tmax_f_roll_std_15_l1",
      "importance": 414.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 407.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 399.0
    },
    {
      "feature": "sin_doy",
      "importance": 398.0
    },
    {
      "feature": "nbm_tmax_f_roll_std_15_l1",
      "importance": 385.0
    },
    {
      "feature": "rap_tmax_f_roll_std_15_l1",
      "importance": 370.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 345.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 332.0
    },
    {
      "feature": "gfs_tmax_f_roll_std_60_l1",
      "importance": 327.0
    },
    {
      "feature": "model_vol_mean_15",
      "importance": 327.0
    },
    {
      "feature": "day_of_year",
      "importance": 326.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 313.0
    },
    {
      "feature": "model_vol_max_15",
      "importance": 309.0
    },
    {
      "feature": "cos_doy",
      "importance": 305.0
    },
    {
      "feature": "hrrr_tmax_f_roll_std_60_l1",
      "importance": 299.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_roll_std_60_l1",
      "importance": 283.0
    },
    {
      "feature": "nam_tmax_f_roll_std_60_l1",
      "importance": 267.0
    },
    {
      "feature": "rap_tmax_f_roll_std_60_l1",
      "importance": 229.0
    },
    {
      "feature": "nbm_tmax_f_roll_std_60_l1",
      "importance": 227.0
    },
    {
      "feature": "is_weekend",
      "importance": 66.0
    },
    {
      "feature": "month",
      "importance": 22.0
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
          "formula": "roll_std(nbm_tmax_f)",
          "name": "nbm_tmax_f_roll_std_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "roll_std(nbm_tmax_f)",
          "name": "nbm_tmax_f_roll_std_60_l1",
          "params": {
            "lag": 1,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_std(gfs_tmax_f)",
          "name": "gfs_tmax_f_roll_std_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "roll_std(gfs_tmax_f)",
          "name": "gfs_tmax_f_roll_std_60_l1",
          "params": {
            "lag": 1,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_std(gefsatmosmean_tmax_f)",
          "name": "gefsatmosmean_tmax_f_roll_std_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "roll_std(gefsatmosmean_tmax_f)",
          "name": "gefsatmosmean_tmax_f_roll_std_60_l1",
          "params": {
            "lag": 1,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_std(nam_tmax_f)",
          "name": "nam_tmax_f_roll_std_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "roll_std(nam_tmax_f)",
          "name": "nam_tmax_f_roll_std_60_l1",
          "params": {
            "lag": 1,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_std(hrrr_tmax_f)",
          "name": "hrrr_tmax_f_roll_std_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "roll_std(hrrr_tmax_f)",
          "name": "hrrr_tmax_f_roll_std_60_l1",
          "params": {
            "lag": 1,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_std(rap_tmax_f)",
          "name": "rap_tmax_f_roll_std_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "roll_std(rap_tmax_f)",
          "name": "rap_tmax_f_roll_std_60_l1",
          "params": {
            "lag": 1,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "mean({model}_roll_std_15_l1)",
          "name": "model_vol_mean_15"
        },
        {
          "formula": "max({model}_roll_std_15_l1)",
          "name": "model_vol_max_15"
        }
      ],
      "imputation": {
        "fill_values": {
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "gefsatmos_tmp_spread_f": 1.985091821481542,
          "gefsatmosmean_tmax_f": 63.38288233996978,
          "gefsatmosmean_tmax_f_roll_std_15_l1": 6.230013002169559,
          "gefsatmosmean_tmax_f_roll_std_60_l1": 8.740882577880416,
          "gfs_tmax_f": 64.5,
          "gfs_tmax_f_roll_std_15_l1": 6.280835047101035,
          "gfs_tmax_f_roll_std_60_l1": 8.61132161521893,
          "hrrr_tmax_f": 63.429676513671915,
          "hrrr_tmax_f_roll_std_15_l1": 7.303277344202559,
          "hrrr_tmax_f_roll_std_60_l1": 9.778430731045592,
          "is_weekend": 0.0,
          "model_vol_max_15": 7.664874869464849,
          "model_vol_mean_15": 6.50695914811043,
          "month": 6.0,
          "nam_tmax_f": 63.0,
          "nam_tmax_f_roll_std_15_l1": 5.852255025960838,
          "nam_tmax_f_roll_std_60_l1": 8.059637983868946,
          "nbm_tmax_f": 63.175991210937525,
          "nbm_tmax_f_roll_std_15_l1": 6.245688450144451,
          "nbm_tmax_f_roll_std_60_l1": 8.541881601503905,
          "rap_tmax_f": 62.58964813232426,
          "rap_tmax_f_roll_std_15_l1": 7.391928536033707,
          "rap_tmax_f_roll_std_60_l1": 9.72495568781575,
          "sin_doy": 0.15951094710994368
        },
        "method": "train_median"
      },
      "train_fitted": []
    },
    "description": "Per-model forecast volatility (15/60)",
    "experiment_id": "E09",
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
      "nbm_tmax_f_roll_std_15_l1",
      "nbm_tmax_f_roll_std_60_l1",
      "gfs_tmax_f_roll_std_15_l1",
      "gfs_tmax_f_roll_std_60_l1",
      "gefsatmosmean_tmax_f_roll_std_15_l1",
      "gefsatmosmean_tmax_f_roll_std_60_l1",
      "nam_tmax_f_roll_std_15_l1",
      "nam_tmax_f_roll_std_60_l1",
      "hrrr_tmax_f_roll_std_15_l1",
      "hrrr_tmax_f_roll_std_60_l1",
      "rap_tmax_f_roll_std_15_l1",
      "rap_tmax_f_roll_std_60_l1",
      "model_vol_mean_15",
      "model_vol_max_15"
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