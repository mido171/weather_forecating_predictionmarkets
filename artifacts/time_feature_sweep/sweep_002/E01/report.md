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
    "bias": 0.42925691243461944,
    "corr": 0.9907492954992049,
    "mae": 1.8164321618553774,
    "maxAE": 11.05362968897414,
    "medianAE": 1.4497621567180659,
    "n": 332,
    "rmse": 2.3872124678073043
  },
  "train": {
    "bias": 9.402361242524314e-11,
    "corr": 0.9989590496682825,
    "mae": 0.5696695850224759,
    "maxAE": 6.835924477981557,
    "medianAE": 0.462486131868296,
    "n": 1224,
    "rmse": 0.7558201820399634
  },
  "validation": {
    "bias": -0.16360407646049982,
    "corr": 0.9923297063858685,
    "mae": 1.906389892997307,
    "maxAE": 7.328695328693058,
    "medianAE": 1.489601705732433,
    "n": 214,
    "rmse": 2.4634991276458047
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 918.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 719.0
    },
    {
      "feature": "ens_median_dev_from_rm30",
      "importance": 632.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 612.0
    },
    {
      "feature": "ens_mean_roll_mean_7_l1",
      "importance": 608.0
    },
    {
      "feature": "ens_mean_dev_from_rm30",
      "importance": 581.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 538.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 527.0
    },
    {
      "feature": "ens_mean_roll_mean_30_l1",
      "importance": 452.0
    },
    {
      "feature": "ens_mean_roll_mean_60_l1",
      "importance": 441.0
    },
    {
      "feature": "day_of_year",
      "importance": 418.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 409.0
    },
    {
      "feature": "sin_doy",
      "importance": 391.0
    },
    {
      "feature": "cos_doy",
      "importance": 390.0
    },
    {
      "feature": "ens_median_roll_mean_7_l1",
      "importance": 375.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 366.0
    },
    {
      "feature": "ens_median_roll_mean_60_l1",
      "importance": 292.0
    },
    {
      "feature": "ens_median_roll_mean_30_l1",
      "importance": 250.0
    },
    {
      "feature": "is_weekend",
      "importance": 55.0
    },
    {
      "feature": "month",
      "importance": 26.0
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
          "formula": "roll_mean(ens_mean)",
          "name": "ens_mean_roll_mean_7_l1",
          "params": {
            "lag": 1,
            "min_periods": 5,
            "window": 7
          }
        },
        {
          "formula": "roll_mean(ens_mean)",
          "name": "ens_mean_roll_mean_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_mean(ens_mean)",
          "name": "ens_mean_roll_mean_60_l1",
          "params": {
            "lag": 1,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "roll_mean(ens_median)",
          "name": "ens_median_roll_mean_7_l1",
          "params": {
            "lag": 1,
            "min_periods": 5,
            "window": 7
          }
        },
        {
          "formula": "roll_mean(ens_median)",
          "name": "ens_median_roll_mean_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_mean(ens_median)",
          "name": "ens_median_roll_mean_60_l1",
          "params": {
            "lag": 1,
            "min_periods": 42,
            "window": 60
          }
        },
        {
          "formula": "ens_mean - ens_mean_roll_mean_30_l1",
          "name": "ens_mean_dev_from_rm30"
        },
        {
          "formula": "ens_median - ens_median_roll_mean_30_l1",
          "name": "ens_median_dev_from_rm30"
        }
      ],
      "imputation": {
        "fill_values": {
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "ens_mean_dev_from_rm30": 0.2685007778810373,
          "ens_mean_roll_mean_30_l1": 63.335662870608125,
          "ens_mean_roll_mean_60_l1": 62.47873450517008,
          "ens_mean_roll_mean_7_l1": 62.49603949227754,
          "ens_median_dev_from_rm30": 0.23432988917854658,
          "ens_median_roll_mean_30_l1": 63.312475535796835,
          "ens_median_roll_mean_60_l1": 62.572613461763154,
          "ens_median_roll_mean_7_l1": 62.58779383083645,
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
    "description": "Rolling mean/median level (7/30/60) with deviation",
    "experiment_id": "E01",
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
      "ens_mean_roll_mean_7_l1",
      "ens_mean_roll_mean_30_l1",
      "ens_mean_roll_mean_60_l1",
      "ens_median_roll_mean_7_l1",
      "ens_median_roll_mean_30_l1",
      "ens_median_roll_mean_60_l1",
      "ens_mean_dev_from_rm30",
      "ens_median_dev_from_rm30"
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