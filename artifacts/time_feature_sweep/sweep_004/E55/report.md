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
    "bias": 0.2923103802555853,
    "corr": 0.9910203510991926,
    "mae": 1.7749391911782322,
    "maxAE": 8.939964202108605,
    "medianAE": 1.4217607980756952,
    "n": 332,
    "rmse": 2.3308541580390423
  },
  "train": {
    "bias": -1.3217027182224162e-10,
    "corr": 0.999281282784894,
    "mae": 0.46255079192107174,
    "maxAE": 6.496342362336119,
    "medianAE": 0.36869968805749664,
    "n": 1224,
    "rmse": 0.6283491437727025
  },
  "validation": {
    "bias": -0.05662422620973506,
    "corr": 0.992610667366396,
    "mae": 1.818848160486744,
    "maxAE": 6.929169863178998,
    "medianAE": 1.3617476961752253,
    "n": 214,
    "rmse": 2.3892472910968445
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 727.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 582.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 432.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 386.0
    },
    {
      "feature": "abs_err_cv_gefsatmosmean_tmax_f_rm60_l2",
      "importance": 381.0
    },
    {
      "feature": "min_cv_model_rm60_l2",
      "importance": 379.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 374.0
    },
    {
      "feature": "abs_err_cv_nbm_tmax_f_rm60_l2",
      "importance": 357.0
    },
    {
      "feature": "sin_doy",
      "importance": 356.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 345.0
    },
    {
      "feature": "abs_err_cv_rap_tmax_f_rm60_l2",
      "importance": 311.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 306.0
    },
    {
      "feature": "abs_err_cv_nam_tmax_f_rm60_l2",
      "importance": 298.0
    },
    {
      "feature": "abs_err_cv_gfs_tmax_f_rm60_l2",
      "importance": 276.0
    },
    {
      "feature": "abs_err_cv_hrrr_tmax_f_rm60_l2",
      "importance": 265.0
    },
    {
      "feature": "mean_cv_models_rm60_l2",
      "importance": 261.0
    },
    {
      "feature": "abs_err_std_gefsatmosmean_tmax_f_rm60_l2",
      "importance": 257.0
    },
    {
      "feature": "cos_doy",
      "importance": 252.0
    },
    {
      "feature": "abs_err_mean_nam_tmax_f_rm60_l2",
      "importance": 252.0
    },
    {
      "feature": "abs_err_mean_gfs_tmax_f_rm60_l2",
      "importance": 249.0
    },
    {
      "feature": "abs_err_mean_gefsatmosmean_tmax_f_rm60_l2",
      "importance": 229.0
    },
    {
      "feature": "abs_err_std_gfs_tmax_f_rm60_l2",
      "importance": 217.0
    },
    {
      "feature": "day_of_year",
      "importance": 215.0
    },
    {
      "feature": "abs_err_std_nam_tmax_f_rm60_l2",
      "importance": 212.0
    },
    {
      "feature": "abs_err_mean_rap_tmax_f_rm60_l2",
      "importance": 192.0
    },
    {
      "feature": "abs_err_mean_hrrr_tmax_f_rm60_l2",
      "importance": 184.0
    },
    {
      "feature": "abs_err_std_hrrr_tmax_f_rm60_l2",
      "importance": 171.0
    },
    {
      "feature": "abs_err_mean_nbm_tmax_f_rm60_l2",
      "importance": 158.0
    },
    {
      "feature": "abs_err_std_rap_tmax_f_rm60_l2",
      "importance": 158.0
    },
    {
      "feature": "abs_err_std_nbm_tmax_f_rm60_l2",
      "importance": 156.0
    },
    {
      "feature": "is_weekend",
      "importance": 47.0
    },
    {
      "feature": "month",
      "importance": 15.0
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
          "formula": "roll_mean(|resid|)",
          "name": "abs_err_mean_nbm_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_std(|resid|)",
          "name": "abs_err_std_nbm_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "abs_err_std / (abs_err_mean+eps)",
          "name": "abs_err_cv_nbm_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "abs_err_mean_gfs_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_std(|resid|)",
          "name": "abs_err_std_gfs_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "abs_err_std / (abs_err_mean+eps)",
          "name": "abs_err_cv_gfs_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "abs_err_mean_gefsatmosmean_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_std(|resid|)",
          "name": "abs_err_std_gefsatmosmean_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "abs_err_std / (abs_err_mean+eps)",
          "name": "abs_err_cv_gefsatmosmean_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "abs_err_mean_nam_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_std(|resid|)",
          "name": "abs_err_std_nam_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "abs_err_std / (abs_err_mean+eps)",
          "name": "abs_err_cv_nam_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "abs_err_mean_hrrr_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_std(|resid|)",
          "name": "abs_err_std_hrrr_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "abs_err_std / (abs_err_mean+eps)",
          "name": "abs_err_cv_hrrr_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "abs_err_mean_rap_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_std(|resid|)",
          "name": "abs_err_std_rap_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "abs_err_std / (abs_err_mean+eps)",
          "name": "abs_err_cv_rap_tmax_f_rm60_l2"
        },
        {
          "formula": "min(abs_err_cv)",
          "name": "min_cv_model_rm60_l2"
        },
        {
          "formula": "mean(abs_err_cv)",
          "name": "mean_cv_models_rm60_l2"
        }
      ],
      "imputation": {
        "fill_values": {
          "abs_err_cv_gefsatmosmean_tmax_f_rm60_l2": 0.709097973642532,
          "abs_err_cv_gfs_tmax_f_rm60_l2": 0.802971485492165,
          "abs_err_cv_hrrr_tmax_f_rm60_l2": 0.7036595653015434,
          "abs_err_cv_nam_tmax_f_rm60_l2": 0.7949260028059453,
          "abs_err_cv_nbm_tmax_f_rm60_l2": 0.7317045826333981,
          "abs_err_cv_rap_tmax_f_rm60_l2": 0.6741881106195895,
          "abs_err_mean_gefsatmosmean_tmax_f_rm60_l2": 2.6858807899384423,
          "abs_err_mean_gfs_tmax_f_rm60_l2": 2.45,
          "abs_err_mean_hrrr_tmax_f_rm60_l2": 2.7992452697753674,
          "abs_err_mean_nam_tmax_f_rm60_l2": 2.7333333333333334,
          "abs_err_mean_nbm_tmax_f_rm60_l2": 2.2609038085937034,
          "abs_err_mean_rap_tmax_f_rm60_l2": 3.2189987284342156,
          "abs_err_std_gefsatmosmean_tmax_f_rm60_l2": 2.0249114250493627,
          "abs_err_std_gfs_tmax_f_rm60_l2": 2.1558061137310096,
          "abs_err_std_hrrr_tmax_f_rm60_l2": 2.0659928867810713,
          "abs_err_std_nam_tmax_f_rm60_l2": 2.281020726682583,
          "abs_err_std_nbm_tmax_f_rm60_l2": 1.7443807904807263,
          "abs_err_std_rap_tmax_f_rm60_l2": 2.3133170835962353,
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "gefsatmos_tmp_spread_f": 1.985091821481542,
          "gefsatmosmean_tmax_f": 63.38288233996978,
          "gfs_tmax_f": 64.5,
          "hrrr_tmax_f": 63.429676513671915,
          "is_weekend": 0.0,
          "mean_cv_models_rm60_l2": 0.743881658650754,
          "min_cv_model_rm60_l2": 0.6388090377912863,
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
    "description": "Error volatility state (std/CV)",
    "experiment_id": "E55",
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
      "abs_err_mean_nbm_tmax_f_rm60_l2",
      "abs_err_std_nbm_tmax_f_rm60_l2",
      "abs_err_cv_nbm_tmax_f_rm60_l2",
      "abs_err_mean_gfs_tmax_f_rm60_l2",
      "abs_err_std_gfs_tmax_f_rm60_l2",
      "abs_err_cv_gfs_tmax_f_rm60_l2",
      "abs_err_mean_gefsatmosmean_tmax_f_rm60_l2",
      "abs_err_std_gefsatmosmean_tmax_f_rm60_l2",
      "abs_err_cv_gefsatmosmean_tmax_f_rm60_l2",
      "abs_err_mean_nam_tmax_f_rm60_l2",
      "abs_err_std_nam_tmax_f_rm60_l2",
      "abs_err_cv_nam_tmax_f_rm60_l2",
      "abs_err_mean_hrrr_tmax_f_rm60_l2",
      "abs_err_std_hrrr_tmax_f_rm60_l2",
      "abs_err_cv_hrrr_tmax_f_rm60_l2",
      "abs_err_mean_rap_tmax_f_rm60_l2",
      "abs_err_std_rap_tmax_f_rm60_l2",
      "abs_err_cv_rap_tmax_f_rm60_l2",
      "min_cv_model_rm60_l2",
      "mean_cv_models_rm60_l2"
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