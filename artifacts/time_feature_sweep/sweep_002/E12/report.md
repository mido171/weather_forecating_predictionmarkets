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
    "bias": 0.3892272408273945,
    "corr": 0.9904680957712694,
    "mae": 1.8614234266976697,
    "maxAE": 10.124405521371067,
    "medianAE": 1.4832311894279187,
    "n": 332,
    "rmse": 2.415782956188052
  },
  "train": {
    "bias": -8.748646542142589e-11,
    "corr": 0.9995222033013859,
    "mae": 0.3634315137022848,
    "maxAE": 6.298363873993075,
    "medianAE": 0.28093608072457243,
    "n": 1224,
    "rmse": 0.5128244031861919
  },
  "validation": {
    "bias": -0.24698125730209536,
    "corr": 0.9919974513897124,
    "mae": 1.8839629962244542,
    "maxAE": 7.435791342630381,
    "medianAE": 1.490644189907151,
    "n": 214,
    "rmse": 2.5284146035862047
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 509.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 473.0
    },
    {
      "feature": "delta_std_across_models_last",
      "importance": 461.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 403.0
    },
    {
      "feature": "delta_gefsatmosmean_tmax_f_last_l1",
      "importance": 384.0
    },
    {
      "feature": "delta_hrrr_tmax_f_last_l1",
      "importance": 374.0
    },
    {
      "feature": "delta_rap_tmax_f_last_l1",
      "importance": 363.0
    },
    {
      "feature": "delta_hrrr_tmax_f_absmean_15_l1",
      "importance": 359.0
    },
    {
      "feature": "delta_nbm_tmax_f_absmean_15_l1",
      "importance": 348.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 323.0
    },
    {
      "feature": "delta_nam_tmax_f_absmean_15_l1",
      "importance": 322.0
    },
    {
      "feature": "delta_rap_tmax_f_absmean_15_l1",
      "importance": 306.0
    },
    {
      "feature": "sin_doy",
      "importance": 305.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 296.0
    },
    {
      "feature": "delta_hrrr_tmax_f_std_15_l1",
      "importance": 290.0
    },
    {
      "feature": "day_of_year",
      "importance": 283.0
    },
    {
      "feature": "delta_gfs_tmax_f_std_15_l1",
      "importance": 277.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 268.0
    },
    {
      "feature": "delta_gefsatmosmean_tmax_f_absmean_15_l1",
      "importance": 263.0
    },
    {
      "feature": "delta_nbm_tmax_f_std_15_l1",
      "importance": 262.0
    },
    {
      "feature": "delta_nbm_tmax_f_last_l1",
      "importance": 261.0
    },
    {
      "feature": "delta_rap_tmax_f_std_15_l1",
      "importance": 255.0
    },
    {
      "feature": "cos_doy",
      "importance": 254.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 236.0
    },
    {
      "feature": "delta_gefsatmosmean_tmax_f_std_15_l1",
      "importance": 236.0
    },
    {
      "feature": "delta_gfs_tmax_f_last_l1",
      "importance": 217.0
    },
    {
      "feature": "delta_nam_tmax_f_std_15_l1",
      "importance": 216.0
    },
    {
      "feature": "delta_nam_tmax_f_last_l1",
      "importance": 204.0
    },
    {
      "feature": "delta_gfs_tmax_f_absmean_15_l1",
      "importance": 194.0
    },
    {
      "feature": "is_weekend",
      "importance": 36.0
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
          "formula": "diff(nbm_tmax_f) shifted by 1",
          "name": "delta_nbm_tmax_f_last_l1"
        },
        {
          "formula": "roll_mean(|diff(nbm_tmax_f)|)",
          "name": "delta_nbm_tmax_f_absmean_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "roll_std(diff(nbm_tmax_f))",
          "name": "delta_nbm_tmax_f_std_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "diff(gfs_tmax_f) shifted by 1",
          "name": "delta_gfs_tmax_f_last_l1"
        },
        {
          "formula": "roll_mean(|diff(gfs_tmax_f)|)",
          "name": "delta_gfs_tmax_f_absmean_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "roll_std(diff(gfs_tmax_f))",
          "name": "delta_gfs_tmax_f_std_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "diff(gefsatmosmean_tmax_f) shifted by 1",
          "name": "delta_gefsatmosmean_tmax_f_last_l1"
        },
        {
          "formula": "roll_mean(|diff(gefsatmosmean_tmax_f)|)",
          "name": "delta_gefsatmosmean_tmax_f_absmean_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "roll_std(diff(gefsatmosmean_tmax_f))",
          "name": "delta_gefsatmosmean_tmax_f_std_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "diff(nam_tmax_f) shifted by 1",
          "name": "delta_nam_tmax_f_last_l1"
        },
        {
          "formula": "roll_mean(|diff(nam_tmax_f)|)",
          "name": "delta_nam_tmax_f_absmean_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "roll_std(diff(nam_tmax_f))",
          "name": "delta_nam_tmax_f_std_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "diff(hrrr_tmax_f) shifted by 1",
          "name": "delta_hrrr_tmax_f_last_l1"
        },
        {
          "formula": "roll_mean(|diff(hrrr_tmax_f)|)",
          "name": "delta_hrrr_tmax_f_absmean_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "roll_std(diff(hrrr_tmax_f))",
          "name": "delta_hrrr_tmax_f_std_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "diff(rap_tmax_f) shifted by 1",
          "name": "delta_rap_tmax_f_last_l1"
        },
        {
          "formula": "roll_mean(|diff(rap_tmax_f)|)",
          "name": "delta_rap_tmax_f_absmean_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "roll_std(diff(rap_tmax_f))",
          "name": "delta_rap_tmax_f_std_15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "std(delta_model_last_l1)",
          "name": "delta_std_across_models_last"
        }
      ],
      "imputation": {
        "fill_values": {
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "delta_gefsatmosmean_tmax_f_absmean_15_l1": 4.879949470766118,
          "delta_gefsatmosmean_tmax_f_last_l1": 0.6705764081402101,
          "delta_gefsatmosmean_tmax_f_std_15_l1": 6.209033400198127,
          "delta_gfs_tmax_f_absmean_15_l1": 4.866666666666666,
          "delta_gfs_tmax_f_last_l1": 1.0,
          "delta_gfs_tmax_f_std_15_l1": 6.162611268589574,
          "delta_hrrr_tmax_f_absmean_15_l1": 6.089896545410156,
          "delta_hrrr_tmax_f_last_l1": 0.6456390380859389,
          "delta_hrrr_tmax_f_std_15_l1": 7.67515074130737,
          "delta_nam_tmax_f_absmean_15_l1": 5.066666666666666,
          "delta_nam_tmax_f_last_l1": 0.0,
          "delta_nam_tmax_f_std_15_l1": 6.508114270530554,
          "delta_nbm_tmax_f_absmean_15_l1": 4.952998535156242,
          "delta_nbm_tmax_f_last_l1": 0.4320263671875324,
          "delta_nbm_tmax_f_std_15_l1": 6.153872993672696,
          "delta_rap_tmax_f_absmean_15_l1": 6.4464413452148435,
          "delta_rap_tmax_f_last_l1": 0.8090881347656236,
          "delta_rap_tmax_f_std_15_l1": 7.929999906647558,
          "delta_std_across_models_last": 1.9856576024429562,
          "gefsatmos_tmp_spread_f": 1.985091821481542,
          "gefsatmosmean_tmax_f": 63.38288233996978,
          "gfs_tmax_f": 64.5,
          "hrrr_tmax_f": 63.429676513671915,
          "is_weekend": 0.0,
          "month": 6.0,
          "nam_tmax_f": 63.0,
          "nbm_tmax_f": 63.175991210937525,
          "rap_tmax_f": 62.58964813232426,
          "sin_doy": 0.15951094710994368
        },
        "method": "train_median"
      },
      "train_fitted": []
    },
    "description": "Rolling mean/std of model day-to-day changes",
    "experiment_id": "E12",
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
      "delta_nbm_tmax_f_last_l1",
      "delta_nbm_tmax_f_absmean_15_l1",
      "delta_nbm_tmax_f_std_15_l1",
      "delta_gfs_tmax_f_last_l1",
      "delta_gfs_tmax_f_absmean_15_l1",
      "delta_gfs_tmax_f_std_15_l1",
      "delta_gefsatmosmean_tmax_f_last_l1",
      "delta_gefsatmosmean_tmax_f_absmean_15_l1",
      "delta_gefsatmosmean_tmax_f_std_15_l1",
      "delta_nam_tmax_f_last_l1",
      "delta_nam_tmax_f_absmean_15_l1",
      "delta_nam_tmax_f_std_15_l1",
      "delta_hrrr_tmax_f_last_l1",
      "delta_hrrr_tmax_f_absmean_15_l1",
      "delta_hrrr_tmax_f_std_15_l1",
      "delta_rap_tmax_f_last_l1",
      "delta_rap_tmax_f_absmean_15_l1",
      "delta_rap_tmax_f_std_15_l1",
      "delta_std_across_models_last"
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