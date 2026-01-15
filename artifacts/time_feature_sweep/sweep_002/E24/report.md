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
    "bias": 0.058531518612253815,
    "corr": 0.9919454851592953,
    "mae": 1.6960850219758976,
    "maxAE": 8.73546112992819,
    "medianAE": 1.4045074045214285,
    "n": 332,
    "rmse": 2.1912052911368995
  },
  "train": {
    "bias": -1.4473140188135932e-11,
    "corr": 0.9992039997639169,
    "mae": 0.4917144127457485,
    "maxAE": 6.400630394556924,
    "medianAE": 0.39260380450236454,
    "n": 1224,
    "rmse": 0.6612278082848727
  },
  "validation": {
    "bias": -0.4501285600071329,
    "corr": 0.9918672086314198,
    "mae": 1.941382508238583,
    "maxAE": 8.068296965186121,
    "medianAE": 1.4468898096152216,
    "n": 214,
    "rmse": 2.546859840081554
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 859.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 535.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 462.0
    },
    {
      "feature": "sin_doy",
      "importance": 441.0
    },
    {
      "feature": "rel_mae_gfs_tmax_f_vs_ens_rm60_l2",
      "importance": 430.0
    },
    {
      "feature": "rel_mae_rap_tmax_f_vs_ens_rm60_l2",
      "importance": 417.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 413.0
    },
    {
      "feature": "rel_mae_nam_tmax_f_vs_ens_rm60_l2",
      "importance": 395.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 393.0
    },
    {
      "feature": "rel_mae_nbm_tmax_f_vs_ens_rm60_l2",
      "importance": 393.0
    },
    {
      "feature": "rel_mae_hrrr_tmax_f_vs_ens_rm60_l2",
      "importance": 393.0
    },
    {
      "feature": "rel_mae_gefsatmosmean_tmax_f_vs_ens_rm60_l2",
      "importance": 376.0
    },
    {
      "feature": "cos_doy",
      "importance": 358.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 338.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 337.0
    },
    {
      "feature": "mae_gefsatmosmean_tmax_f_rm60_l2",
      "importance": 307.0
    },
    {
      "feature": "day_of_year",
      "importance": 288.0
    },
    {
      "feature": "best_model_forecast_today",
      "importance": 279.0
    },
    {
      "feature": "mae_gfs_tmax_f_rm60_l2",
      "importance": 277.0
    },
    {
      "feature": "mae_nam_tmax_f_rm60_l2",
      "importance": 274.0
    },
    {
      "feature": "mae_hrrr_tmax_f_rm60_l2",
      "importance": 246.0
    },
    {
      "feature": "mae_ensmean_rm60_l2",
      "importance": 237.0
    },
    {
      "feature": "mae_nbm_tmax_f_rm60_l2",
      "importance": 213.0
    },
    {
      "feature": "mae_rap_tmax_f_rm60_l2",
      "importance": 208.0
    },
    {
      "feature": "is_weekend",
      "importance": 53.0
    },
    {
      "feature": "best_is_gefsatmosmean_tmax_f",
      "importance": 33.0
    },
    {
      "feature": "month",
      "importance": 22.0
    },
    {
      "feature": "best_is_nbm_tmax_f",
      "importance": 19.0
    },
    {
      "feature": "best_is_gfs_tmax_f",
      "importance": 3.0
    },
    {
      "feature": "best_is_hrrr_tmax_f",
      "importance": 1.0
    },
    {
      "feature": "best_is_nam_tmax_f",
      "importance": 0.0
    },
    {
      "feature": "best_is_rap_tmax_f",
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
          "formula": "roll_mean(|resid_nbm_tmax_f|)",
          "name": "mae_nbm_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_mean(|resid_gfs_tmax_f|)",
          "name": "mae_gfs_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_mean(|resid_gefsatmosmean_tmax_f|)",
          "name": "mae_gefsatmosmean_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_mean(|resid_nam_tmax_f|)",
          "name": "mae_nam_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_mean(|resid_hrrr_tmax_f|)",
          "name": "mae_hrrr_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_mean(|resid_rap_tmax_f|)",
          "name": "mae_rap_tmax_f_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_mean(|resid_ens_mean|)",
          "name": "mae_ensmean_rm60_l2",
          "params": {
            "lag": 2,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "mae_nbm_tmax_f_rm60_l2 - mae_ensmean_rm60_l2",
          "name": "rel_mae_nbm_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "mae_gfs_tmax_f_rm60_l2 - mae_ensmean_rm60_l2",
          "name": "rel_mae_gfs_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "mae_gefsatmosmean_tmax_f_rm60_l2 - mae_ensmean_rm60_l2",
          "name": "rel_mae_gefsatmosmean_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "mae_nam_tmax_f_rm60_l2 - mae_ensmean_rm60_l2",
          "name": "rel_mae_nam_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "mae_hrrr_tmax_f_rm60_l2 - mae_ensmean_rm60_l2",
          "name": "rel_mae_hrrr_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "mae_rap_tmax_f_rm60_l2 - mae_ensmean_rm60_l2",
          "name": "rel_mae_rap_tmax_f_vs_ens_rm60_l2"
        },
        {
          "formula": "1[best_model_id == nbm_tmax_f]",
          "name": "best_is_nbm_tmax_f"
        },
        {
          "formula": "1[best_model_id == gfs_tmax_f]",
          "name": "best_is_gfs_tmax_f"
        },
        {
          "formula": "1[best_model_id == gefsatmosmean_tmax_f]",
          "name": "best_is_gefsatmosmean_tmax_f"
        },
        {
          "formula": "1[best_model_id == nam_tmax_f]",
          "name": "best_is_nam_tmax_f"
        },
        {
          "formula": "1[best_model_id == hrrr_tmax_f]",
          "name": "best_is_hrrr_tmax_f"
        },
        {
          "formula": "1[best_model_id == rap_tmax_f]",
          "name": "best_is_rap_tmax_f"
        },
        {
          "formula": "forecast(best_model_id)",
          "name": "best_model_forecast_today"
        }
      ],
      "imputation": {
        "fill_values": {
          "best_is_gefsatmosmean_tmax_f": 0.0,
          "best_is_gfs_tmax_f": 0.0,
          "best_is_hrrr_tmax_f": 0.0,
          "best_is_nam_tmax_f": 0.0,
          "best_is_nbm_tmax_f": 1.0,
          "best_is_rap_tmax_f": 0.0,
          "best_model_forecast_today": 64.29198242187508,
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "gefsatmos_tmp_spread_f": 1.985091821481542,
          "gefsatmosmean_tmax_f": 63.38288233996978,
          "gfs_tmax_f": 64.5,
          "hrrr_tmax_f": 63.429676513671915,
          "is_weekend": 0.0,
          "mae_ensmean_rm60_l2": 2.1654362193992727,
          "mae_gefsatmosmean_tmax_f_rm60_l2": 2.6858807899384423,
          "mae_gfs_tmax_f_rm60_l2": 2.45,
          "mae_hrrr_tmax_f_rm60_l2": 2.7992452697753674,
          "mae_nam_tmax_f_rm60_l2": 2.7333333333333334,
          "mae_nbm_tmax_f_rm60_l2": 2.2609038085937034,
          "mae_rap_tmax_f_rm60_l2": 3.2189987284342156,
          "month": 6.0,
          "nam_tmax_f": 63.0,
          "nbm_tmax_f": 63.175991210937525,
          "rap_tmax_f": 62.58964813232426,
          "rel_mae_gefsatmosmean_tmax_f_vs_ens_rm60_l2": 0.45091277052359313,
          "rel_mae_gfs_tmax_f_vs_ens_rm60_l2": 0.42968457733483323,
          "rel_mae_hrrr_tmax_f_vs_ens_rm60_l2": 0.6203255588326306,
          "rel_mae_nam_tmax_f_vs_ens_rm60_l2": 0.5650733354575519,
          "rel_mae_nbm_tmax_f_vs_ens_rm60_l2": 0.04196524869308105,
          "rel_mae_rap_tmax_f_vs_ens_rm60_l2": 1.1114887968986333,
          "sin_doy": 0.15951094710994368
        },
        "method": "train_median"
      },
      "train_fitted": []
    },
    "description": "Relative skill vs ensemble + best model flag",
    "experiment_id": "E24",
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
      "mae_nbm_tmax_f_rm60_l2",
      "mae_gfs_tmax_f_rm60_l2",
      "mae_gefsatmosmean_tmax_f_rm60_l2",
      "mae_nam_tmax_f_rm60_l2",
      "mae_hrrr_tmax_f_rm60_l2",
      "mae_rap_tmax_f_rm60_l2",
      "mae_ensmean_rm60_l2",
      "rel_mae_nbm_tmax_f_vs_ens_rm60_l2",
      "rel_mae_gfs_tmax_f_vs_ens_rm60_l2",
      "rel_mae_gefsatmosmean_tmax_f_vs_ens_rm60_l2",
      "rel_mae_nam_tmax_f_vs_ens_rm60_l2",
      "rel_mae_hrrr_tmax_f_vs_ens_rm60_l2",
      "rel_mae_rap_tmax_f_vs_ens_rm60_l2",
      "best_is_nbm_tmax_f",
      "best_is_gfs_tmax_f",
      "best_is_gefsatmosmean_tmax_f",
      "best_is_nam_tmax_f",
      "best_is_hrrr_tmax_f",
      "best_is_rap_tmax_f",
      "best_model_forecast_today"
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