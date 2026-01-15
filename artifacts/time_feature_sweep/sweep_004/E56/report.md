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
    "bias": 0.06203329855082504,
    "corr": 0.9914920668732471,
    "mae": 1.7422419563142069,
    "maxAE": 8.483730463810659,
    "medianAE": 1.3997750683003432,
    "n": 332,
    "rmse": 2.2528530374334417
  },
  "train": {
    "bias": -2.2811257603245775e-11,
    "corr": 0.9993660466205453,
    "mae": 0.43102823027514575,
    "maxAE": 6.4164151240737475,
    "medianAE": 0.335492402302318,
    "n": 1224,
    "rmse": 0.5903077373944843
  },
  "validation": {
    "bias": -0.5314003920769055,
    "corr": 0.9924953214176057,
    "mae": 1.878624909397818,
    "maxAE": 7.563705286184742,
    "medianAE": 1.3773810139078506,
    "n": 214,
    "rmse": 2.438032944086262
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 639.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 458.0
    },
    {
      "feature": "rel_ewm_mae_nam_tmax_f_vs_ens_hl14_l2",
      "importance": 416.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 381.0
    },
    {
      "feature": "rel_ewm_mae_gfs_tmax_f_vs_ens_hl14_l2",
      "importance": 372.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 370.0
    },
    {
      "feature": "sin_doy",
      "importance": 342.0
    },
    {
      "feature": "ewm_mae_gfs_tmax_f_hl14_l2",
      "importance": 329.0
    },
    {
      "feature": "ewm_mae_rap_tmax_f_hl14_l2",
      "importance": 329.0
    },
    {
      "feature": "rel_ewm_mae_rap_tmax_f_vs_ens_hl14_l2",
      "importance": 322.0
    },
    {
      "feature": "rel_ewm_mae_nbm_tmax_f_vs_ens_hl14_l2",
      "importance": 315.0
    },
    {
      "feature": "rel_ewm_mae_gefsatmosmean_tmax_f_vs_ens_hl14_l2",
      "importance": 307.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 290.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 285.0
    },
    {
      "feature": "ewm_mae_nbm_tmax_f_hl14_l2",
      "importance": 284.0
    },
    {
      "feature": "ewm_mae_nam_tmax_f_hl30_l2",
      "importance": 278.0
    },
    {
      "feature": "best_ewm_model_forecast_today",
      "importance": 271.0
    },
    {
      "feature": "ewm_mae_nam_tmax_f_hl14_l2",
      "importance": 261.0
    },
    {
      "feature": "rel_ewm_mae_hrrr_tmax_f_vs_ens_hl14_l2",
      "importance": 258.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 257.0
    },
    {
      "feature": "ewm_mae_gfs_tmax_f_hl30_l2",
      "importance": 255.0
    },
    {
      "feature": "ewm_mae_ensmean_hl14_l2",
      "importance": 251.0
    },
    {
      "feature": "ewm_mae_hrrr_tmax_f_hl30_l2",
      "importance": 242.0
    },
    {
      "feature": "cos_doy",
      "importance": 234.0
    },
    {
      "feature": "ewm_mae_gefsatmosmean_tmax_f_hl14_l2",
      "importance": 225.0
    },
    {
      "feature": "ewm_mae_hrrr_tmax_f_hl14_l2",
      "importance": 207.0
    },
    {
      "feature": "ewm_mae_nbm_tmax_f_hl30_l2",
      "importance": 203.0
    },
    {
      "feature": "ewm_mae_rap_tmax_f_hl30_l2",
      "importance": 193.0
    },
    {
      "feature": "day_of_year",
      "importance": 178.0
    },
    {
      "feature": "ewm_mae_gefsatmosmean_tmax_f_hl30_l2",
      "importance": 173.0
    },
    {
      "feature": "is_weekend",
      "importance": 43.0
    },
    {
      "feature": "month",
      "importance": 10.0
    },
    {
      "feature": "best_ewm_is_nbm_tmax_f",
      "importance": 9.0
    },
    {
      "feature": "best_ewm_is_nam_tmax_f",
      "importance": 9.0
    },
    {
      "feature": "best_ewm_is_hrrr_tmax_f",
      "importance": 2.0
    },
    {
      "feature": "best_ewm_is_gfs_tmax_f",
      "importance": 1.0
    },
    {
      "feature": "best_ewm_is_gefsatmosmean_tmax_f",
      "importance": 1.0
    },
    {
      "feature": "best_ewm_is_rap_tmax_f",
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
          "formula": "ewm_mean(|resid|)",
          "name": "ewm_mae_nbm_tmax_f_hl14_l2",
          "params": {
            "halflife": 14,
            "lag": 2,
            "min_periods": 10
          }
        },
        {
          "formula": "ewm_mean(|resid|)",
          "name": "ewm_mae_nbm_tmax_f_hl30_l2",
          "params": {
            "halflife": 30,
            "lag": 2,
            "min_periods": 10
          }
        },
        {
          "formula": "ewm_mean(|resid|)",
          "name": "ewm_mae_gfs_tmax_f_hl14_l2",
          "params": {
            "halflife": 14,
            "lag": 2,
            "min_periods": 10
          }
        },
        {
          "formula": "ewm_mean(|resid|)",
          "name": "ewm_mae_gfs_tmax_f_hl30_l2",
          "params": {
            "halflife": 30,
            "lag": 2,
            "min_periods": 10
          }
        },
        {
          "formula": "ewm_mean(|resid|)",
          "name": "ewm_mae_gefsatmosmean_tmax_f_hl14_l2",
          "params": {
            "halflife": 14,
            "lag": 2,
            "min_periods": 10
          }
        },
        {
          "formula": "ewm_mean(|resid|)",
          "name": "ewm_mae_gefsatmosmean_tmax_f_hl30_l2",
          "params": {
            "halflife": 30,
            "lag": 2,
            "min_periods": 10
          }
        },
        {
          "formula": "ewm_mean(|resid|)",
          "name": "ewm_mae_nam_tmax_f_hl14_l2",
          "params": {
            "halflife": 14,
            "lag": 2,
            "min_periods": 10
          }
        },
        {
          "formula": "ewm_mean(|resid|)",
          "name": "ewm_mae_nam_tmax_f_hl30_l2",
          "params": {
            "halflife": 30,
            "lag": 2,
            "min_periods": 10
          }
        },
        {
          "formula": "ewm_mean(|resid|)",
          "name": "ewm_mae_hrrr_tmax_f_hl14_l2",
          "params": {
            "halflife": 14,
            "lag": 2,
            "min_periods": 10
          }
        },
        {
          "formula": "ewm_mean(|resid|)",
          "name": "ewm_mae_hrrr_tmax_f_hl30_l2",
          "params": {
            "halflife": 30,
            "lag": 2,
            "min_periods": 10
          }
        },
        {
          "formula": "ewm_mean(|resid|)",
          "name": "ewm_mae_rap_tmax_f_hl14_l2",
          "params": {
            "halflife": 14,
            "lag": 2,
            "min_periods": 10
          }
        },
        {
          "formula": "ewm_mean(|resid|)",
          "name": "ewm_mae_rap_tmax_f_hl30_l2",
          "params": {
            "halflife": 30,
            "lag": 2,
            "min_periods": 10
          }
        },
        {
          "formula": "ewm_mean(|resid_ens_mean|)",
          "name": "ewm_mae_ensmean_hl14_l2",
          "params": {
            "halflife": 14,
            "lag": 2,
            "min_periods": 10
          }
        },
        {
          "formula": "ewm_mae_model - ewm_mae_ensmean",
          "name": "rel_ewm_mae_nbm_tmax_f_vs_ens_hl14_l2"
        },
        {
          "formula": "ewm_mae_model - ewm_mae_ensmean",
          "name": "rel_ewm_mae_gfs_tmax_f_vs_ens_hl14_l2"
        },
        {
          "formula": "ewm_mae_model - ewm_mae_ensmean",
          "name": "rel_ewm_mae_gefsatmosmean_tmax_f_vs_ens_hl14_l2"
        },
        {
          "formula": "ewm_mae_model - ewm_mae_ensmean",
          "name": "rel_ewm_mae_nam_tmax_f_vs_ens_hl14_l2"
        },
        {
          "formula": "ewm_mae_model - ewm_mae_ensmean",
          "name": "rel_ewm_mae_hrrr_tmax_f_vs_ens_hl14_l2"
        },
        {
          "formula": "ewm_mae_model - ewm_mae_ensmean",
          "name": "rel_ewm_mae_rap_tmax_f_vs_ens_hl14_l2"
        },
        {
          "formula": "1[best_ewm_model_id == nbm_tmax_f]",
          "name": "best_ewm_is_nbm_tmax_f"
        },
        {
          "formula": "1[best_ewm_model_id == gfs_tmax_f]",
          "name": "best_ewm_is_gfs_tmax_f"
        },
        {
          "formula": "1[best_ewm_model_id == gefsatmosmean_tmax_f]",
          "name": "best_ewm_is_gefsatmosmean_tmax_f"
        },
        {
          "formula": "1[best_ewm_model_id == nam_tmax_f]",
          "name": "best_ewm_is_nam_tmax_f"
        },
        {
          "formula": "1[best_ewm_model_id == hrrr_tmax_f]",
          "name": "best_ewm_is_hrrr_tmax_f"
        },
        {
          "formula": "1[best_ewm_model_id == rap_tmax_f]",
          "name": "best_ewm_is_rap_tmax_f"
        },
        {
          "formula": "forecast(best_ewm_model_id)",
          "name": "best_ewm_model_forecast_today"
        }
      ],
      "imputation": {
        "fill_values": {
          "best_ewm_is_gefsatmosmean_tmax_f": 0.0,
          "best_ewm_is_gfs_tmax_f": 0.0,
          "best_ewm_is_hrrr_tmax_f": 0.0,
          "best_ewm_is_nam_tmax_f": 0.0,
          "best_ewm_is_nbm_tmax_f": 1.0,
          "best_ewm_is_rap_tmax_f": 0.0,
          "best_ewm_model_forecast_today": 64.0,
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "ewm_mae_ensmean_hl14_l2": 2.133131215672137,
          "ewm_mae_gefsatmosmean_tmax_f_hl14_l2": 2.6834674010470576,
          "ewm_mae_gefsatmosmean_tmax_f_hl30_l2": 2.727334159880652,
          "ewm_mae_gfs_tmax_f_hl14_l2": 2.46611648011901,
          "ewm_mae_gfs_tmax_f_hl30_l2": 2.51068303907906,
          "ewm_mae_hrrr_tmax_f_hl14_l2": 2.7460840600560124,
          "ewm_mae_hrrr_tmax_f_hl30_l2": 2.748288690683307,
          "ewm_mae_nam_tmax_f_hl14_l2": 2.745951237806988,
          "ewm_mae_nam_tmax_f_hl30_l2": 2.7513154420701156,
          "ewm_mae_nbm_tmax_f_hl14_l2": 2.2183441518170586,
          "ewm_mae_nbm_tmax_f_hl30_l2": 2.3037461154591266,
          "ewm_mae_rap_tmax_f_hl14_l2": 3.254565032816094,
          "ewm_mae_rap_tmax_f_hl30_l2": 3.232383589532951,
          "gefsatmos_tmp_spread_f": 1.985091821481542,
          "gefsatmosmean_tmax_f": 63.38288233996978,
          "gfs_tmax_f": 64.5,
          "hrrr_tmax_f": 63.429676513671915,
          "is_weekend": 0.0,
          "month": 6.0,
          "nam_tmax_f": 63.0,
          "nbm_tmax_f": 63.175991210937525,
          "rap_tmax_f": 62.58964813232426,
          "rel_ewm_mae_gefsatmosmean_tmax_f_vs_ens_hl14_l2": 0.45824015575290966,
          "rel_ewm_mae_gfs_tmax_f_vs_ens_hl14_l2": 0.41865345179135915,
          "rel_ewm_mae_hrrr_tmax_f_vs_ens_hl14_l2": 0.6341888379658798,
          "rel_ewm_mae_nam_tmax_f_vs_ens_hl14_l2": 0.5708563784598026,
          "rel_ewm_mae_nbm_tmax_f_vs_ens_hl14_l2": 0.05684001522789561,
          "rel_ewm_mae_rap_tmax_f_vs_ens_hl14_l2": 1.1051286988062623,
          "sin_doy": 0.15951094710994368
        },
        "method": "train_median"
      },
      "train_fitted": []
    },
    "description": "EWMA(|resid|) skill state + best EWMA model",
    "experiment_id": "E56",
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
      "ewm_mae_nbm_tmax_f_hl14_l2",
      "ewm_mae_nbm_tmax_f_hl30_l2",
      "ewm_mae_gfs_tmax_f_hl14_l2",
      "ewm_mae_gfs_tmax_f_hl30_l2",
      "ewm_mae_gefsatmosmean_tmax_f_hl14_l2",
      "ewm_mae_gefsatmosmean_tmax_f_hl30_l2",
      "ewm_mae_nam_tmax_f_hl14_l2",
      "ewm_mae_nam_tmax_f_hl30_l2",
      "ewm_mae_hrrr_tmax_f_hl14_l2",
      "ewm_mae_hrrr_tmax_f_hl30_l2",
      "ewm_mae_rap_tmax_f_hl14_l2",
      "ewm_mae_rap_tmax_f_hl30_l2",
      "ewm_mae_ensmean_hl14_l2",
      "rel_ewm_mae_nbm_tmax_f_vs_ens_hl14_l2",
      "rel_ewm_mae_gfs_tmax_f_vs_ens_hl14_l2",
      "rel_ewm_mae_gefsatmosmean_tmax_f_vs_ens_hl14_l2",
      "rel_ewm_mae_nam_tmax_f_vs_ens_hl14_l2",
      "rel_ewm_mae_hrrr_tmax_f_vs_ens_hl14_l2",
      "rel_ewm_mae_rap_tmax_f_vs_ens_hl14_l2",
      "best_ewm_is_nbm_tmax_f",
      "best_ewm_is_gfs_tmax_f",
      "best_ewm_is_gefsatmosmean_tmax_f",
      "best_ewm_is_nam_tmax_f",
      "best_ewm_is_hrrr_tmax_f",
      "best_ewm_is_rap_tmax_f",
      "best_ewm_model_forecast_today"
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