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
    "bias": 0.34435088197581953,
    "corr": 0.9905165206286207,
    "mae": 1.8214425932813387,
    "maxAE": 8.242766920852546,
    "medianAE": 1.4332546401415982,
    "n": 332,
    "rmse": 2.4020011269703687
  },
  "train": {
    "bias": -4.3392217923396586e-11,
    "corr": 0.9990821949217094,
    "mae": 0.5127413341576919,
    "maxAE": 7.025492663837294,
    "medianAE": 0.40423500707673554,
    "n": 1224,
    "rmse": 0.7101372015957558
  },
  "validation": {
    "bias": -0.09272192648380931,
    "corr": 0.9920031948773498,
    "mae": 1.8955140014837801,
    "maxAE": 8.124967944468708,
    "medianAE": 1.4831728987321107,
    "n": 214,
    "rmse": 2.4995958146275337
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "ens_std_shock_hl30",
      "importance": 955.0
    },
    {
      "feature": "ens_std_ewm_hl7_l1",
      "importance": 785.0
    },
    {
      "feature": "gefs_spread_shock_hl14",
      "importance": 772.0
    },
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 692.0
    },
    {
      "feature": "gefs_spread_ewm_hl14_l1",
      "importance": 685.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 669.0
    },
    {
      "feature": "ens_std_ewm_hl30_l1",
      "importance": 595.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 534.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 528.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 507.0
    },
    {
      "feature": "sin_doy",
      "importance": 503.0
    },
    {
      "feature": "cos_doy",
      "importance": 476.0
    },
    {
      "feature": "day_of_year",
      "importance": 454.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 388.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 357.0
    },
    {
      "feature": "is_weekend",
      "importance": 60.0
    },
    {
      "feature": "month",
      "importance": 40.0
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
          "formula": "ewm_mean(ens_std)",
          "name": "ens_std_ewm_hl7_l1",
          "params": {
            "halflife": 7,
            "lag": 1,
            "min_periods": 10
          }
        },
        {
          "formula": "ewm_mean(ens_std)",
          "name": "ens_std_ewm_hl30_l1",
          "params": {
            "halflife": 30,
            "lag": 1,
            "min_periods": 10
          }
        },
        {
          "formula": "ens_std - ens_std_ewm_hl30_l1",
          "name": "ens_std_shock_hl30"
        },
        {
          "formula": "ewm_mean(gefs_spread)",
          "name": "gefs_spread_ewm_hl14_l1",
          "params": {
            "halflife": 14,
            "lag": 1,
            "min_periods": 10
          }
        },
        {
          "formula": "gefs_spread - gefs_spread_ewm_hl14_l1",
          "name": "gefs_spread_shock_hl14"
        }
      ],
      "imputation": {
        "fill_values": {
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "ens_std_ewm_hl30_l1": 1.9892920616511307,
          "ens_std_ewm_hl7_l1": 1.9831301029894108,
          "ens_std_shock_hl30": -0.17154117975468086,
          "gefs_spread_ewm_hl14_l1": 2.2540014343037,
          "gefs_spread_shock_hl14": -0.1487607473705197,
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
    "description": "EWMA spread baseline + shock features",
    "experiment_id": "E06",
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
      "ens_std_ewm_hl7_l1",
      "ens_std_ewm_hl30_l1",
      "ens_std_shock_hl30",
      "gefs_spread_ewm_hl14_l1",
      "gefs_spread_shock_hl14"
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