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
    "bias": 0.48669815297475244,
    "corr": 0.9906618306190842,
    "mae": 1.8296088781701825,
    "maxAE": 10.384073732726556,
    "medianAE": 1.3882159101996088,
    "n": 332,
    "rmse": 2.4086295684570094
  },
  "train": {
    "bias": -6.763993287041986e-11,
    "corr": 0.9992904606773143,
    "mae": 0.4595716596839164,
    "maxAE": 6.53809070651819,
    "medianAE": 0.35926663554026206,
    "n": 1224,
    "rmse": 0.624324017115631
  },
  "validation": {
    "bias": -0.16590378611534526,
    "corr": 0.9919046845554436,
    "mae": 1.9991073063753708,
    "maxAE": 7.695191804942368,
    "medianAE": 1.4799184996433468,
    "n": 214,
    "rmse": 2.566485997407296
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 812.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 628.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 465.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 455.0
    },
    {
      "feature": "drift_nam_tmax_f_roll_std_30_l1",
      "importance": 454.0
    },
    {
      "feature": "drift_nam_tmax_f_roll_mean_30_l1",
      "importance": 443.0
    },
    {
      "feature": "drift_rap_tmax_f_roll_mean_30_l1",
      "importance": 420.0
    },
    {
      "feature": "drift_nbm_tmax_f_roll_std_30_l1",
      "importance": 412.0
    },
    {
      "feature": "drift_gefsatmosmean_tmax_f_roll_mean_30_l1",
      "importance": 406.0
    },
    {
      "feature": "drift_gfs_tmax_f_roll_std_30_l1",
      "importance": 399.0
    },
    {
      "feature": "drift_gefsatmosmean_tmax_f_roll_std_30_l1",
      "importance": 384.0
    },
    {
      "feature": "drift_abs_max_rm30",
      "importance": 364.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 338.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 338.0
    },
    {
      "feature": "drift_hrrr_tmax_f_roll_std_30_l1",
      "importance": 334.0
    },
    {
      "feature": "sin_doy",
      "importance": 326.0
    },
    {
      "feature": "drift_gfs_tmax_f_roll_mean_30_l1",
      "importance": 324.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 322.0
    },
    {
      "feature": "drift_rap_tmax_f_roll_std_30_l1",
      "importance": 281.0
    },
    {
      "feature": "drift_nbm_tmax_f_roll_mean_30_l1",
      "importance": 279.0
    },
    {
      "feature": "day_of_year",
      "importance": 276.0
    },
    {
      "feature": "drift_hrrr_tmax_f_roll_mean_30_l1",
      "importance": 259.0
    },
    {
      "feature": "cos_doy",
      "importance": 201.0
    },
    {
      "feature": "is_weekend",
      "importance": 56.0
    },
    {
      "feature": "month",
      "importance": 24.0
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
          "formula": "roll_mean(nbm_tmax_f - ens_mean)",
          "name": "drift_nbm_tmax_f_roll_mean_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(nbm_tmax_f - ens_mean)",
          "name": "drift_nbm_tmax_f_roll_std_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_mean(gfs_tmax_f - ens_mean)",
          "name": "drift_gfs_tmax_f_roll_mean_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(gfs_tmax_f - ens_mean)",
          "name": "drift_gfs_tmax_f_roll_std_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_mean(gefsatmosmean_tmax_f - ens_mean)",
          "name": "drift_gefsatmosmean_tmax_f_roll_mean_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(gefsatmosmean_tmax_f - ens_mean)",
          "name": "drift_gefsatmosmean_tmax_f_roll_std_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_mean(nam_tmax_f - ens_mean)",
          "name": "drift_nam_tmax_f_roll_mean_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(nam_tmax_f - ens_mean)",
          "name": "drift_nam_tmax_f_roll_std_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_mean(hrrr_tmax_f - ens_mean)",
          "name": "drift_hrrr_tmax_f_roll_mean_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(hrrr_tmax_f - ens_mean)",
          "name": "drift_hrrr_tmax_f_roll_std_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_mean(rap_tmax_f - ens_mean)",
          "name": "drift_rap_tmax_f_roll_mean_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(rap_tmax_f - ens_mean)",
          "name": "drift_rap_tmax_f_roll_std_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "max(|drift_*_roll_mean_30_l1|)",
          "name": "drift_abs_max_rm30"
        }
      ],
      "imputation": {
        "fill_values": {
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "drift_abs_max_rm30": 1.4910936685322513,
          "drift_gefsatmosmean_tmax_f_roll_mean_30_l1": 0.26206700905421854,
          "drift_gefsatmosmean_tmax_f_roll_std_30_l1": 1.5806536096177695,
          "drift_gfs_tmax_f_roll_mean_30_l1": 0.43253269215789253,
          "drift_gfs_tmax_f_roll_std_30_l1": 1.918482060000408,
          "drift_hrrr_tmax_f_roll_mean_30_l1": -0.06341540450775299,
          "drift_hrrr_tmax_f_roll_std_30_l1": 1.9337237441635184,
          "drift_nam_tmax_f_roll_mean_30_l1": 0.08687819758586694,
          "drift_nam_tmax_f_roll_std_30_l1": 2.1435083620699404,
          "drift_nbm_tmax_f_roll_mean_30_l1": -0.12215695982352864,
          "drift_nbm_tmax_f_roll_std_30_l1": 0.9839151596045611,
          "drift_rap_tmax_f_roll_mean_30_l1": -1.04372807306542,
          "drift_rap_tmax_f_roll_std_30_l1": 2.195176237364876,
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
    "description": "Rolling mean/std of model drift vs ensemble",
    "experiment_id": "E10",
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
      "drift_nbm_tmax_f_roll_mean_30_l1",
      "drift_nbm_tmax_f_roll_std_30_l1",
      "drift_gfs_tmax_f_roll_mean_30_l1",
      "drift_gfs_tmax_f_roll_std_30_l1",
      "drift_gefsatmosmean_tmax_f_roll_mean_30_l1",
      "drift_gefsatmosmean_tmax_f_roll_std_30_l1",
      "drift_nam_tmax_f_roll_mean_30_l1",
      "drift_nam_tmax_f_roll_std_30_l1",
      "drift_hrrr_tmax_f_roll_mean_30_l1",
      "drift_hrrr_tmax_f_roll_std_30_l1",
      "drift_rap_tmax_f_roll_mean_30_l1",
      "drift_rap_tmax_f_roll_std_30_l1",
      "drift_abs_max_rm30"
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