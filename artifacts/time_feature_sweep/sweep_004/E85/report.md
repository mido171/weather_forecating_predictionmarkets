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
    "bias": 0.4142946307215978,
    "corr": 0.9910547276458156,
    "mae": 1.77273643297744,
    "maxAE": 8.819139520035314,
    "medianAE": 1.400430815303153,
    "n": 332,
    "rmse": 2.345893920490374
  },
  "train": {
    "bias": -1.8405916732647151e-10,
    "corr": 0.998790441087636,
    "mae": 0.6039327439290788,
    "maxAE": 6.758893092910917,
    "medianAE": 0.4924605410881,
    "n": 1224,
    "rmse": 0.8149047360757887
  },
  "validation": {
    "bias": -0.2611047314805766,
    "corr": 0.9924851228126453,
    "mae": 1.8354086287268967,
    "maxAE": 8.372586201572929,
    "medianAE": 1.4652178453103843,
    "n": 214,
    "rmse": 2.451458432201155
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "knn_reg_dist_mean_k20_l2",
      "importance": 1102.0
    },
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 1075.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 831.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 813.0
    },
    {
      "feature": "knn_reg_resid_mean_k20_l2",
      "importance": 761.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 740.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 687.0
    },
    {
      "feature": "sin_doy",
      "importance": 666.0
    },
    {
      "feature": "cos_doy",
      "importance": 621.0
    },
    {
      "feature": "day_of_year",
      "importance": 567.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 518.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 468.0
    },
    {
      "feature": "is_weekend",
      "importance": 63.0
    },
    {
      "feature": "month",
      "importance": 52.0
    },
    {
      "feature": "knn_reg_neighbor_count_l2",
      "importance": 36.0
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
          "formula": "knn_regime_resid_mean",
          "name": "knn_reg_resid_mean_k20_l2"
        },
        {
          "formula": "knn_regime_dist_mean",
          "name": "knn_reg_dist_mean_k20_l2"
        },
        {
          "formula": "knn_regime_neighbor_count",
          "name": "knn_reg_neighbor_count_l2"
        }
      ],
      "imputation": {
        "fill_values": {
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "gefsatmos_tmp_spread_f": 1.985091821481542,
          "gefsatmosmean_tmax_f": 63.38288233996978,
          "gfs_tmax_f": 64.5,
          "hrrr_tmax_f": 63.429676513671915,
          "is_weekend": 0.0,
          "knn_reg_dist_mean_k20_l2": 1.0626670742952755,
          "knn_reg_neighbor_count_l2": 20.0,
          "knn_reg_resid_mean_k20_l2": 1.34473165616268,
          "month": 6.0,
          "nam_tmax_f": 63.0,
          "nbm_tmax_f": 63.175991210937525,
          "rap_tmax_f": 62.58964813232426,
          "sin_doy": 0.15951094710994368
        },
        "method": "train_median"
      },
      "train_fitted": [
        {
          "default": 2.465098209625438,
          "features": [
            "nbm_tmax_f",
            "gfs_tmax_f",
            "gefsatmosmean_tmax_f",
            "nam_tmax_f",
            "hrrr_tmax_f",
            "rap_tmax_f",
            "ens_std",
            "sin_doy",
            "cos_doy"
          ],
          "fit_on": "train",
          "name": "knn_regime_restricted",
          "scaler": {
            "cols": [
              "nbm_tmax_f",
              "gfs_tmax_f",
              "gefsatmosmean_tmax_f",
              "nam_tmax_f",
              "hrrr_tmax_f",
              "rap_tmax_f",
              "ens_std",
              "sin_doy",
              "cos_doy"
            ],
            "mean": [
              63.22669649011956,
              63.84395424836601,
              63.93225178197049,
              63.30555555555556,
              63.549676836799215,
              62.38739490284644,
              1.9724350522718126,
              0.076249519026095,
              -0.03830601337952942
            ],
            "scale": [
              16.744913671392055,
              16.43538534849312,
              17.715798139763308,
              16.16064478057846,
              18.04587973742932,
              17.794629792899077,
              0.8943820168637893,
              0.7113279903785942,
              0.6976612002191372
            ]
          },
          "thresholds": {
            "KNYC": 2.465098209625438
          }
        }
      ]
    },
    "description": "Regime-restricted analogs",
    "experiment_id": "E85",
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
      "knn_reg_resid_mean_k20_l2",
      "knn_reg_dist_mean_k20_l2",
      "knn_reg_neighbor_count_l2"
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