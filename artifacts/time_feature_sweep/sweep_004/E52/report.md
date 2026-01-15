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
    "bias": 0.2164541468597777,
    "corr": 0.9920363526716168,
    "mae": 1.7314827472950134,
    "maxAE": 8.058547582444824,
    "medianAE": 1.4411043265779178,
    "n": 332,
    "rmse": 2.1888037583308355
  },
  "train": {
    "bias": 9.807857358512094e-11,
    "corr": 0.9991155954096639,
    "mae": 0.516032540034272,
    "maxAE": 6.765604109938611,
    "medianAE": 0.3999323857046235,
    "n": 1224,
    "rmse": 0.6967505291325359
  },
  "validation": {
    "bias": -0.2592480579615333,
    "corr": 0.9919529481498915,
    "mae": 1.9274007818992713,
    "maxAE": 7.67672087058348,
    "medianAE": 1.3908475297097311,
    "n": 214,
    "rmse": 2.5411051653480867
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 816.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 520.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 458.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 419.0
    },
    {
      "feature": "sin_doy",
      "importance": 416.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 414.0
    },
    {
      "feature": "rel_rmse_nbm_tmax_f_vs_ens_rm60_l2",
      "importance": 409.0
    },
    {
      "feature": "rel_rmse_gfs_tmax_f_vs_ens_rm60_l2",
      "importance": 398.0
    },
    {
      "feature": "rel_rmse_rap_tmax_f_vs_ens_rm60_l2",
      "importance": 388.0
    },
    {
      "feature": "rel_rmse_gefsatmosmean_tmax_f_vs_ens_rm60_l2",
      "importance": 386.0
    },
    {
      "feature": "rmse_gfs_tmax_f_rm60_l2",
      "importance": 359.0
    },
    {
      "feature": "rel_rmse_nam_tmax_f_vs_ens_rm60_l2",
      "importance": 356.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 348.0
    },
    {
      "feature": "rmse_nam_tmax_f_rm60_l2",
      "importance": 348.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 345.0
    },
    {
      "feature": "rel_rmse_hrrr_tmax_f_vs_ens_rm60_l2",
      "importance": 333.0
    },
    {
      "feature": "best_rmse_model_forecast_today",
      "importance": 325.0
    },
    {
      "feature": "rmse_gefsatmosmean_tmax_f_rm60_l2",
      "importance": 315.0
    },
    {
      "feature": "cos_doy",
      "importance": 298.0
    },
    {
      "feature": "day_of_year",
      "importance": 296.0
    },
    {
      "feature": "rmse_nbm_tmax_f_rm60_l2",
      "importance": 275.0
    },
    {
      "feature": "rmse_hrrr_tmax_f_rm60_l2",
      "importance": 275.0
    },
    {
      "feature": "rmse_rap_tmax_f_rm60_l2",
      "importance": 244.0
    },
    {
      "feature": "rmse_ensmean_rm60_l2",
      "importance": 168.0
    },
    {
      "feature": "is_weekend",
      "importance": 54.0
    },
    {
      "feature": "month",
      "importance": 25.0
    },
    {
      "feature": "best_rmse_is_nbm_tmax_f",
      "importance": 12.0
    },
    {
      "feature": "best_rmse_is_gfs_tmax_f",
      "importance": 0.0
    },
    {
      "feature": "best_rmse_is_gefsatmosmean_tmax_f",
      "importance": 0.0
    },
    {
      "feature": "best_rmse_is_nam_tmax_f",
      "importance": 0.0
    },
    {
      "feature": "best_rmse_is_hrrr_tmax_f",
      "importance": 0.0
    },
    {
      "feature": "best_rmse_is_rap_tmax_f",
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
          "formula": "rmse(resid_nbm_tmax_f)",
          "name": "rmse_nbm_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "rmse(resid_gfs_tmax_f)",
          "name": "rmse_gfs_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "rmse(resid_gefsatmosmean_tmax_f)",
          "name": "rmse_gefsatmosmean_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "rmse(resid_nam_tmax_f)",
          "name": "rmse_nam_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "rmse(resid_hrrr_tmax_f)",
          "name": "rmse_hrrr_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "rmse(resid_rap_tmax_f)",
          "name": "rmse_rap_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "rmse(resid_ens_mean)",
          "name": "rmse_ensmean_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "rmse_nbm_tmax_f_rm60_l - rmse_ensmean_rm60_l",
          "name": "rel_rmse_nbm_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "rmse_gfs_tmax_f_rm60_l - rmse_ensmean_rm60_l",
          "name": "rel_rmse_gfs_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "rmse_gefsatmosmean_tmax_f_rm60_l - rmse_ensmean_rm60_l",
          "name": "rel_rmse_gefsatmosmean_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "rmse_nam_tmax_f_rm60_l - rmse_ensmean_rm60_l",
          "name": "rel_rmse_nam_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "rmse_hrrr_tmax_f_rm60_l - rmse_ensmean_rm60_l",
          "name": "rel_rmse_hrrr_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "rmse_rap_tmax_f_rm60_l - rmse_ensmean_rm60_l",
          "name": "rel_rmse_rap_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "1[best_rmse_model_id == nbm_tmax_f]",
          "name": "best_rmse_is_nbm_tmax_f"
        },
        {
          "formula": "1[best_rmse_model_id == gfs_tmax_f]",
          "name": "best_rmse_is_gfs_tmax_f"
        },
        {
          "formula": "1[best_rmse_model_id == gefsatmosmean_tmax_f]",
          "name": "best_rmse_is_gefsatmosmean_tmax_f"
        },
        {
          "formula": "1[best_rmse_model_id == nam_tmax_f]",
          "name": "best_rmse_is_nam_tmax_f"
        },
        {
          "formula": "1[best_rmse_model_id == hrrr_tmax_f]",
          "name": "best_rmse_is_hrrr_tmax_f"
        },
        {
          "formula": "1[best_rmse_model_id == rap_tmax_f]",
          "name": "best_rmse_is_rap_tmax_f"
        },
        {
          "formula": "forecast(best_rmse_model_id)",
          "name": "best_rmse_model_forecast_today"
        }
      ],
      "imputation": {
        "fill_values": {
          "best_rmse_is_gefsatmosmean_tmax_f": 0.0,
          "best_rmse_is_gfs_tmax_f": 0.0,
          "best_rmse_is_hrrr_tmax_f": 0.0,
          "best_rmse_is_nam_tmax_f": 0.0,
          "best_rmse_is_nbm_tmax_f": 1.0,
          "best_rmse_is_rap_tmax_f": 0.0,
          "best_rmse_model_forecast_today": 64.81400878906254,
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
          "rel_rmse_gefsatmosmean_tmax_f_vs_ens_rm60_l2": 0.5918105615575158,
          "rel_rmse_gfs_tmax_f_vs_ens_rm60_l2": 0.6308745385901351,
          "rel_rmse_hrrr_tmax_f_vs_ens_rm60_l2": 0.7331007696556653,
          "rel_rmse_nam_tmax_f_vs_ens_rm60_l2": 0.794706514187816,
          "rel_rmse_nbm_tmax_f_vs_ens_rm60_l2": 0.05701316392065259,
          "rel_rmse_rap_tmax_f_vs_ens_rm60_l2": 1.3434134775491202,
          "rmse_ensmean_rm60_l2": 2.750153993450365,
          "rmse_gefsatmosmean_tmax_f_rm60_l2": 3.4368891108416544,
          "rmse_gfs_tmax_f_rm60_l2": 3.2812599206199238,
          "rmse_hrrr_tmax_f_rm60_l2": 3.484239426776739,
          "rmse_nam_tmax_f_rm60_l2": 3.5683796509527026,
          "rmse_nbm_tmax_f_rm60_l2": 2.885511629386308,
          "rmse_rap_tmax_f_rm60_l2": 4.057840666999412,
          "sin_doy": 0.15951094710994368
        },
        "method": "train_median"
      },
      "train_fitted": []
    },
    "description": "RMSE skill state + best-RMSE selector",
    "experiment_id": "E52",
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
      "rmse_nbm_tmax_f_rm60_l2",
      "rmse_gfs_tmax_f_rm60_l2",
      "rmse_gefsatmosmean_tmax_f_rm60_l2",
      "rmse_nam_tmax_f_rm60_l2",
      "rmse_hrrr_tmax_f_rm60_l2",
      "rmse_rap_tmax_f_rm60_l2",
      "rmse_ensmean_rm60_l2",
      "rel_rmse_nbm_tmax_f_vs_ens_rm60_l2",
      "rel_rmse_gfs_tmax_f_vs_ens_rm60_l2",
      "rel_rmse_gefsatmosmean_tmax_f_vs_ens_rm60_l2",
      "rel_rmse_nam_tmax_f_vs_ens_rm60_l2",
      "rel_rmse_hrrr_tmax_f_vs_ens_rm60_l2",
      "rel_rmse_rap_tmax_f_vs_ens_rm60_l2",
      "best_rmse_is_nbm_tmax_f",
      "best_rmse_is_gfs_tmax_f",
      "best_rmse_is_gefsatmosmean_tmax_f",
      "best_rmse_is_nam_tmax_f",
      "best_rmse_is_hrrr_tmax_f",
      "best_rmse_is_rap_tmax_f",
      "best_rmse_model_forecast_today"
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