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
    "bias": 0.5032393348713968,
    "corr": 0.9904568236295399,
    "mae": 1.8877889344255057,
    "maxAE": 9.875728969127465,
    "medianAE": 1.5149253629295494,
    "n": 332,
    "rmse": 2.439341318433908
  },
  "train": {
    "bias": 4.9218743822769643e-11,
    "corr": 0.9991627809183541,
    "mae": 0.4975535583250534,
    "maxAE": 6.818214084853551,
    "medianAE": 0.3927925154051515,
    "n": 1224,
    "rmse": 0.677954499257096
  },
  "validation": {
    "bias": -0.07909080287927452,
    "corr": 0.9923723600475419,
    "mae": 1.8822427080192587,
    "maxAE": 7.375439524830355,
    "medianAE": 1.3682981491908386,
    "n": 214,
    "rmse": 2.460413933224693
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 746.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 617.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 457.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 445.0
    },
    {
      "feature": "rel_abs_bias_rap_tmax_f_vs_ens_rm60_l2",
      "importance": 444.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 422.0
    },
    {
      "feature": "sin_doy",
      "importance": 390.0
    },
    {
      "feature": "rel_abs_bias_hrrr_tmax_f_vs_ens_rm60_l2",
      "importance": 389.0
    },
    {
      "feature": "rel_abs_bias_nam_tmax_f_vs_ens_rm60_l2",
      "importance": 376.0
    },
    {
      "feature": "best_bias_model_forecast_today",
      "importance": 374.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 320.0
    },
    {
      "feature": "rel_abs_bias_gfs_tmax_f_vs_ens_rm60_l2",
      "importance": 319.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 305.0
    },
    {
      "feature": "rel_abs_bias_gefsatmosmean_tmax_f_vs_ens_rm60_l2",
      "importance": 272.0
    },
    {
      "feature": "day_of_year",
      "importance": 269.0
    },
    {
      "feature": "rel_abs_bias_nbm_tmax_f_vs_ens_rm60_l2",
      "importance": 263.0
    },
    {
      "feature": "bias_nam_tmax_f_rm60_l2",
      "importance": 258.0
    },
    {
      "feature": "bias_gfs_tmax_f_rm60_l2",
      "importance": 252.0
    },
    {
      "feature": "cos_doy",
      "importance": 250.0
    },
    {
      "feature": "bias_nbm_tmax_f_rm60_l2",
      "importance": 250.0
    },
    {
      "feature": "abs_bias_hrrr_tmax_f_rm60_l2",
      "importance": 208.0
    },
    {
      "feature": "bias_rap_tmax_f_rm60_l2",
      "importance": 208.0
    },
    {
      "feature": "abs_bias_gefsatmosmean_tmax_f_rm60_l2",
      "importance": 206.0
    },
    {
      "feature": "bias_hrrr_tmax_f_rm60_l2",
      "importance": 175.0
    },
    {
      "feature": "abs_bias_gfs_tmax_f_rm60_l2",
      "importance": 168.0
    },
    {
      "feature": "bias_gefsatmosmean_tmax_f_rm60_l2",
      "importance": 152.0
    },
    {
      "feature": "bias_ensmean_rm60_l2",
      "importance": 112.0
    },
    {
      "feature": "abs_bias_nam_tmax_f_rm60_l2",
      "importance": 98.0
    },
    {
      "feature": "abs_bias_rap_tmax_f_rm60_l2",
      "importance": 76.0
    },
    {
      "feature": "abs_bias_ensmean_rm60_l2",
      "importance": 68.0
    },
    {
      "feature": "is_weekend",
      "importance": 49.0
    },
    {
      "feature": "abs_bias_nbm_tmax_f_rm60_l2",
      "importance": 23.0
    },
    {
      "feature": "best_bias_is_gfs_tmax_f",
      "importance": 12.0
    },
    {
      "feature": "month",
      "importance": 11.0
    },
    {
      "feature": "best_bias_is_nam_tmax_f",
      "importance": 6.0
    },
    {
      "feature": "best_bias_is_gefsatmosmean_tmax_f",
      "importance": 4.0
    },
    {
      "feature": "best_bias_is_hrrr_tmax_f",
      "importance": 4.0
    },
    {
      "feature": "best_bias_is_nbm_tmax_f",
      "importance": 2.0
    },
    {
      "feature": "best_bias_is_rap_tmax_f",
      "importance": 0.0
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
          "formula": "roll_mean(resid_nbm_tmax_f)",
          "name": "bias_nbm_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "|bias_nbm_tmax_f_rm60_l2|",
          "name": "abs_bias_nbm_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(resid_gfs_tmax_f)",
          "name": "bias_gfs_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "|bias_gfs_tmax_f_rm60_l2|",
          "name": "abs_bias_gfs_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(resid_gefsatmosmean_tmax_f)",
          "name": "bias_gefsatmosmean_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "|bias_gefsatmosmean_tmax_f_rm60_l2|",
          "name": "abs_bias_gefsatmosmean_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(resid_nam_tmax_f)",
          "name": "bias_nam_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "|bias_nam_tmax_f_rm60_l2|",
          "name": "abs_bias_nam_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(resid_hrrr_tmax_f)",
          "name": "bias_hrrr_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "|bias_hrrr_tmax_f_rm60_l2|",
          "name": "abs_bias_hrrr_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(resid_rap_tmax_f)",
          "name": "bias_rap_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "|bias_rap_tmax_f_rm60_l2|",
          "name": "abs_bias_rap_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(resid_ens_mean)",
          "name": "bias_ensmean_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "|bias_ensmean_rm60_l2|",
          "name": "abs_bias_ensmean_rm60_l2"
        },
        {
          "formula": "abs_bias_nbm_tmax_f_rm60_l - abs_bias_ensmean_rm60_l",
          "name": "rel_abs_bias_nbm_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "abs_bias_gfs_tmax_f_rm60_l - abs_bias_ensmean_rm60_l",
          "name": "rel_abs_bias_gfs_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "abs_bias_gefsatmosmean_tmax_f_rm60_l - abs_bias_ensmean_rm60_l",
          "name": "rel_abs_bias_gefsatmosmean_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "abs_bias_nam_tmax_f_rm60_l - abs_bias_ensmean_rm60_l",
          "name": "rel_abs_bias_nam_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "abs_bias_hrrr_tmax_f_rm60_l - abs_bias_ensmean_rm60_l",
          "name": "rel_abs_bias_hrrr_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "abs_bias_rap_tmax_f_rm60_l - abs_bias_ensmean_rm60_l",
          "name": "rel_abs_bias_rap_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "1[best_bias_model_id == nbm_tmax_f]",
          "name": "best_bias_is_nbm_tmax_f"
        },
        {
          "formula": "1[best_bias_model_id == gfs_tmax_f]",
          "name": "best_bias_is_gfs_tmax_f"
        },
        {
          "formula": "1[best_bias_model_id == gefsatmosmean_tmax_f]",
          "name": "best_bias_is_gefsatmosmean_tmax_f"
        },
        {
          "formula": "1[best_bias_model_id == nam_tmax_f]",
          "name": "best_bias_is_nam_tmax_f"
        },
        {
          "formula": "1[best_bias_model_id == hrrr_tmax_f]",
          "name": "best_bias_is_hrrr_tmax_f"
        },
        {
          "formula": "1[best_bias_model_id == rap_tmax_f]",
          "name": "best_bias_is_rap_tmax_f"
        },
        {
          "formula": "forecast(best_bias_model_id)",
          "name": "best_bias_model_forecast_today"
        }
      ],
      "imputation": {
        "fill_values": {
          "abs_bias_ensmean_rm60_l2": 1.3529924368581134,
          "abs_bias_gefsatmosmean_tmax_f_rm60_l2": 1.9356973748319442,
          "abs_bias_gfs_tmax_f_rm60_l2": 0.75,
          "abs_bias_hrrr_tmax_f_rm60_l2": 1.7686642456054271,
          "abs_bias_nam_tmax_f_rm60_l2": 1.5,
          "abs_bias_nbm_tmax_f_rm60_l2": 1.4753722330728531,
          "abs_bias_rap_tmax_f_rm60_l2": 2.465282745361286,
          "best_bias_is_gefsatmosmean_tmax_f": 0.0,
          "best_bias_is_gfs_tmax_f": 0.0,
          "best_bias_is_hrrr_tmax_f": 0.0,
          "best_bias_is_nam_tmax_f": 0.0,
          "best_bias_is_nbm_tmax_f": 0.0,
          "best_bias_is_rap_tmax_f": 0.0,
          "best_bias_model_forecast_today": 65.0,
          "bias_ensmean_rm60_l2": 1.3529924368581134,
          "bias_gefsatmosmean_tmax_f_rm60_l2": 1.28967427807874,
          "bias_gfs_tmax_f_rm60_l2": 0.6833333333333333,
          "bias_hrrr_tmax_f_rm60_l2": 1.6668624369302976,
          "bias_nam_tmax_f_rm60_l2": 1.5,
          "bias_nbm_tmax_f_rm60_l2": 1.4753722330728531,
          "bias_rap_tmax_f_rm60_l2": 2.465282745361286,
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "gefsatmos_tmp_spread_f": 1.985091821481542,
          "gefsatmosmean_tmax_f": 63.38288233996978,
          "gfs_tmax_f": 64.5,
          "hrrr_tmax_f": 63.429676513671915,
          "is_weekend": 0.0,
          "month": 6.0,
          "nam_tmax_f": 63.0,
          "nbm_tmax_f": 63.175991210937525,
          "rap_tmax_f": 62.58964813232426,
          "rel_abs_bias_gefsatmosmean_tmax_f_vs_ens_rm60_l2": 0.18574389637498867,
          "rel_abs_bias_gfs_tmax_f_vs_ens_rm60_l2": -0.27117201880448527,
          "rel_abs_bias_hrrr_tmax_f_vs_ens_rm60_l2": 0.32838849800137915,
          "rel_abs_bias_nam_tmax_f_vs_ens_rm60_l2": -0.08788380771209825,
          "rel_abs_bias_nbm_tmax_f_vs_ens_rm60_l2": 0.04843254474376879,
          "rel_abs_bias_rap_tmax_f_vs_ens_rm60_l2": 0.9862939057271447,
          "sin_doy": 0.15951094710994368
        },
        "method": "train_median"
      },
      "train_fitted": []
    },
    "description": "Signed bias state + least-biased model identity",
    "experiment_id": "E51",
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
      "bias_nbm_tmax_f_rm60_l2",
      "abs_bias_nbm_tmax_f_rm60_l2",
      "bias_gfs_tmax_f_rm60_l2",
      "abs_bias_gfs_tmax_f_rm60_l2",
      "bias_gefsatmosmean_tmax_f_rm60_l2",
      "abs_bias_gefsatmosmean_tmax_f_rm60_l2",
      "bias_nam_tmax_f_rm60_l2",
      "abs_bias_nam_tmax_f_rm60_l2",
      "bias_hrrr_tmax_f_rm60_l2",
      "abs_bias_hrrr_tmax_f_rm60_l2",
      "bias_rap_tmax_f_rm60_l2",
      "abs_bias_rap_tmax_f_rm60_l2",
      "bias_ensmean_rm60_l2",
      "abs_bias_ensmean_rm60_l2",
      "rel_abs_bias_nbm_tmax_f_vs_ens_rm60_l2",
      "rel_abs_bias_gfs_tmax_f_vs_ens_rm60_l2",
      "rel_abs_bias_gefsatmosmean_tmax_f_vs_ens_rm60_l2",
      "rel_abs_bias_nam_tmax_f_vs_ens_rm60_l2",
      "rel_abs_bias_hrrr_tmax_f_vs_ens_rm60_l2",
      "rel_abs_bias_rap_tmax_f_vs_ens_rm60_l2",
      "best_bias_is_nbm_tmax_f",
      "best_bias_is_gfs_tmax_f",
      "best_bias_is_gefsatmosmean_tmax_f",
      "best_bias_is_nam_tmax_f",
      "best_bias_is_hrrr_tmax_f",
      "best_bias_is_rap_tmax_f",
      "best_bias_model_forecast_today"
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