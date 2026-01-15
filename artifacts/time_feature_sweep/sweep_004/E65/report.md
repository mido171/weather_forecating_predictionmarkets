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
    "bias": 0.4393466733405175,
    "corr": 0.9907194457087117,
    "mae": 1.8034995058479917,
    "maxAE": 9.887108438916904,
    "medianAE": 1.3497336256193346,
    "n": 332,
    "rmse": 2.399030808119709
  },
  "train": {
    "bias": -4.703769690381021e-11,
    "corr": 0.9990681070268486,
    "mae": 0.539194955301658,
    "maxAE": 6.482816984139301,
    "medianAE": 0.43907142196456306,
    "n": 1224,
    "rmse": 0.7151705852344395
  },
  "validation": {
    "bias": -0.6622454299472187,
    "corr": 0.992090098346356,
    "mae": 1.9757532770190886,
    "maxAE": 7.989028199239058,
    "medianAE": 1.547780961553162,
    "n": 214,
    "rmse": 2.5594037166761114
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 912.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 695.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 579.0
    },
    {
      "feature": "mae_reg_selected_rm60_l2",
      "importance": 528.0
    },
    {
      "feature": "bias_reg_selected_rm60_l2",
      "importance": 518.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 505.0
    },
    {
      "feature": "sin_doy",
      "importance": 498.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 496.0
    },
    {
      "feature": "bias_reg0_rm60_l2",
      "importance": 475.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 431.0
    },
    {
      "feature": "cos_doy",
      "importance": 409.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 406.0
    },
    {
      "feature": "mae_reg0_rm60_l2",
      "importance": 368.0
    },
    {
      "feature": "mae_reg2_rm60_l2",
      "importance": 365.0
    },
    {
      "feature": "day_of_year",
      "importance": 362.0
    },
    {
      "feature": "bias_reg2_rm60_l2",
      "importance": 309.0
    },
    {
      "feature": "bias_reg1_rm60_l2",
      "importance": 308.0
    },
    {
      "feature": "mae_reg1_rm60_l2",
      "importance": 269.0
    },
    {
      "feature": "bias_reg3_rm60_l2",
      "importance": 263.0
    },
    {
      "feature": "mae_reg3_rm60_l2",
      "importance": 217.0
    },
    {
      "feature": "is_weekend",
      "importance": 64.0
    },
    {
      "feature": "month",
      "importance": 23.0
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
          "formula": "conditional_bias_regime",
          "name": "bias_reg0_rm60_l2"
        },
        {
          "formula": "conditional_mae_regime",
          "name": "mae_reg0_rm60_l2"
        },
        {
          "formula": "conditional_bias_regime",
          "name": "bias_reg1_rm60_l2"
        },
        {
          "formula": "conditional_mae_regime",
          "name": "mae_reg1_rm60_l2"
        },
        {
          "formula": "conditional_bias_regime",
          "name": "bias_reg2_rm60_l2"
        },
        {
          "formula": "conditional_mae_regime",
          "name": "mae_reg2_rm60_l2"
        },
        {
          "formula": "conditional_bias_regime",
          "name": "bias_reg3_rm60_l2"
        },
        {
          "formula": "conditional_mae_regime",
          "name": "mae_reg3_rm60_l2"
        },
        {
          "formula": "bias_selected_by_regime",
          "name": "bias_reg_selected_rm60_l2"
        },
        {
          "formula": "mae_selected_by_regime",
          "name": "mae_reg_selected_rm60_l2"
        }
      ],
      "imputation": {
        "fill_values": {
          "bias_reg0_rm60_l2": 1.656911019273954,
          "bias_reg1_rm60_l2": 2.5280102531051574,
          "bias_reg2_rm60_l2": 0.7187714003696067,
          "bias_reg3_rm60_l2": 1.0894673296528155,
          "bias_reg_selected_rm60_l2": 1.292785770504742,
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "gefsatmos_tmp_spread_f": 1.985091821481542,
          "gefsatmosmean_tmax_f": 63.38288233996978,
          "gfs_tmax_f": 64.5,
          "hrrr_tmax_f": 63.429676513671915,
          "is_weekend": 0.0,
          "mae_reg0_rm60_l2": 2.2712111842699487,
          "mae_reg1_rm60_l2": 3.237240132902151,
          "mae_reg2_rm60_l2": 1.7739841980636668,
          "mae_reg3_rm60_l2": 2.3381835291708626,
          "mae_reg_selected_rm60_l2": 2.0567283768981404,
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
          "fit_on": "train",
          "name": "regime_thresholds",
          "thr_spread": {
            "KNYC": 2.465098209625438
          },
          "thr_temp": {
            "KNYC": 63.284222166984335
          }
        }
      ]
    },
    "description": "4-regime residual bias/MAE for ensemble mean",
    "experiment_id": "E65",
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
      "bias_reg0_rm60_l2",
      "mae_reg0_rm60_l2",
      "bias_reg1_rm60_l2",
      "mae_reg1_rm60_l2",
      "bias_reg2_rm60_l2",
      "mae_reg2_rm60_l2",
      "bias_reg3_rm60_l2",
      "mae_reg3_rm60_l2",
      "bias_reg_selected_rm60_l2",
      "mae_reg_selected_rm60_l2"
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