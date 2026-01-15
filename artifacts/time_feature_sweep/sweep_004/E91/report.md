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
    "bias": 0.4467746319075636,
    "corr": 0.991443157069136,
    "mae": 1.7315964370065513,
    "maxAE": 8.681823751221508,
    "medianAE": 1.4039498319834607,
    "n": 332,
    "rmse": 2.3012147833404417
  },
  "train": {
    "bias": 6.350805720092234e-11,
    "corr": 0.9996529824574588,
    "mae": 0.2978548917404873,
    "maxAE": 6.32597694828344,
    "medianAE": 0.22391881229080468,
    "n": 1224,
    "rmse": 0.4370398097583814
  },
  "validation": {
    "bias": -0.21793036511014693,
    "corr": 0.9921591049947884,
    "mae": 1.8925672825804685,
    "maxAE": 7.64810127986577,
    "medianAE": 1.4404007764823703,
    "n": 214,
    "rmse": 2.509085595550309
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 475.0
    },
    {
      "feature": "delta_nam_minus_nbm_dev_rm30",
      "importance": 440.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 431.0
    },
    {
      "feature": "delta_rap_minus_nbm_dev_rm30",
      "importance": 422.0
    },
    {
      "feature": "delta_gefsatmosmean_minus_nbm_dev_rm30",
      "importance": 420.0
    },
    {
      "feature": "delta_gfs_minus_nbm_dev_rm30",
      "importance": 400.0
    },
    {
      "feature": "delta_nam_minus_nbm_rm30_l1",
      "importance": 360.0
    },
    {
      "feature": "delta_hrrr_minus_nbm_slope15_l1",
      "importance": 352.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 336.0
    },
    {
      "feature": "delta_gfs_minus_nbm_slope15_l1",
      "importance": 328.0
    },
    {
      "feature": "delta_gefsatmosmean_minus_nbm_slope15_l1",
      "importance": 323.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 311.0
    },
    {
      "feature": "delta_hrrr_minus_nbm_dev_rm30",
      "importance": 307.0
    },
    {
      "feature": "delta_nam_minus_nbm_slope15_l1",
      "importance": 305.0
    },
    {
      "feature": "delta_rap_minus_nbm_slope15_l1",
      "importance": 301.0
    },
    {
      "feature": "delta_gefsatmosmean_minus_nbm_rs30_l1",
      "importance": 283.0
    },
    {
      "feature": "delta_nam_minus_nbm_rs30_l1",
      "importance": 262.0
    },
    {
      "feature": "delta_gefsatmosmean_minus_nbm_rm30_l1",
      "importance": 260.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 257.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 247.0
    },
    {
      "feature": "delta_gfs_minus_nbm_rs30_l1",
      "importance": 242.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 236.0
    },
    {
      "feature": "delta_gfs_minus_nbm_rm30_l1",
      "importance": 236.0
    },
    {
      "feature": "delta_hrrr_minus_nbm_rm30_l1",
      "importance": 233.0
    },
    {
      "feature": "delta_rap_minus_nbm_rm30_l1",
      "importance": 230.0
    },
    {
      "feature": "delta_hrrr_minus_nbm_rs30_l1",
      "importance": 208.0
    },
    {
      "feature": "day_of_year",
      "importance": 196.0
    },
    {
      "feature": "sin_doy",
      "importance": 194.0
    },
    {
      "feature": "cos_doy",
      "importance": 184.0
    },
    {
      "feature": "delta_rap_minus_nbm_rs30_l1",
      "importance": 164.0
    },
    {
      "feature": "is_weekend",
      "importance": 36.0
    },
    {
      "feature": "month",
      "importance": 21.0
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
          "formula": "roll_mean(delta)",
          "name": "delta_rap_minus_nbm_rm30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(delta)",
          "name": "delta_rap_minus_nbm_rs30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_slope(delta)",
          "name": "delta_rap_minus_nbm_slope15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "delta - delta_rm30",
          "name": "delta_rap_minus_nbm_dev_rm30"
        },
        {
          "formula": "roll_mean(delta)",
          "name": "delta_hrrr_minus_nbm_rm30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(delta)",
          "name": "delta_hrrr_minus_nbm_rs30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_slope(delta)",
          "name": "delta_hrrr_minus_nbm_slope15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "delta - delta_rm30",
          "name": "delta_hrrr_minus_nbm_dev_rm30"
        },
        {
          "formula": "roll_mean(delta)",
          "name": "delta_nam_minus_nbm_rm30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(delta)",
          "name": "delta_nam_minus_nbm_rs30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_slope(delta)",
          "name": "delta_nam_minus_nbm_slope15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "delta - delta_rm30",
          "name": "delta_nam_minus_nbm_dev_rm30"
        },
        {
          "formula": "roll_mean(delta)",
          "name": "delta_gfs_minus_nbm_rm30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(delta)",
          "name": "delta_gfs_minus_nbm_rs30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_slope(delta)",
          "name": "delta_gfs_minus_nbm_slope15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "delta - delta_rm30",
          "name": "delta_gfs_minus_nbm_dev_rm30"
        },
        {
          "formula": "roll_mean(delta)",
          "name": "delta_gefsatmosmean_minus_nbm_rm30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(delta)",
          "name": "delta_gefsatmosmean_minus_nbm_rs30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_slope(delta)",
          "name": "delta_gefsatmosmean_minus_nbm_slope15_l1",
          "params": {
            "lag": 1,
            "min_periods": 11,
            "window": 15
          }
        },
        {
          "formula": "delta - delta_rm30",
          "name": "delta_gefsatmosmean_minus_nbm_dev_rm30"
        }
      ],
      "imputation": {
        "fill_values": {
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "delta_gefsatmosmean_minus_nbm_dev_rm30": -0.03785982232874385,
          "delta_gefsatmosmean_minus_nbm_rm30_l1": 0.1503625819052272,
          "delta_gefsatmosmean_minus_nbm_rs30_l1": 1.5307563684675065,
          "delta_gefsatmosmean_minus_nbm_slope15_l1": -0.006067650804649333,
          "delta_gfs_minus_nbm_dev_rm30": 0.18224820963543303,
          "delta_gfs_minus_nbm_rm30_l1": 0.7028149414061995,
          "delta_gfs_minus_nbm_rs30_l1": 2.2057307034952234,
          "delta_gfs_minus_nbm_slope15_l1": 0.0017273472377221464,
          "delta_hrrr_minus_nbm_dev_rm30": 0.005471374511736821,
          "delta_hrrr_minus_nbm_rm30_l1": 0.08309271240233258,
          "delta_hrrr_minus_nbm_rs30_l1": 2.3208142549583286,
          "delta_hrrr_minus_nbm_slope15_l1": 0.006885838099889,
          "delta_nam_minus_nbm_dev_rm30": 0.0813025716145972,
          "delta_nam_minus_nbm_rm30_l1": 0.14768277994785492,
          "delta_nam_minus_nbm_rs30_l1": 2.429883692704318,
          "delta_nam_minus_nbm_slope15_l1": -0.0050066092354892535,
          "delta_rap_minus_nbm_dev_rm30": -0.02022491455085995,
          "delta_rap_minus_nbm_rm30_l1": -0.9612511596679906,
          "delta_rap_minus_nbm_rs30_l1": 2.5209564955944375,
          "delta_rap_minus_nbm_slope15_l1": 0.00048186819893964686,
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
    "description": "Pairwise delta time structure",
    "experiment_id": "E91",
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
      "delta_rap_minus_nbm_rm30_l1",
      "delta_rap_minus_nbm_rs30_l1",
      "delta_rap_minus_nbm_slope15_l1",
      "delta_rap_minus_nbm_dev_rm30",
      "delta_hrrr_minus_nbm_rm30_l1",
      "delta_hrrr_minus_nbm_rs30_l1",
      "delta_hrrr_minus_nbm_slope15_l1",
      "delta_hrrr_minus_nbm_dev_rm30",
      "delta_nam_minus_nbm_rm30_l1",
      "delta_nam_minus_nbm_rs30_l1",
      "delta_nam_minus_nbm_slope15_l1",
      "delta_nam_minus_nbm_dev_rm30",
      "delta_gfs_minus_nbm_rm30_l1",
      "delta_gfs_minus_nbm_rs30_l1",
      "delta_gfs_minus_nbm_slope15_l1",
      "delta_gfs_minus_nbm_dev_rm30",
      "delta_gefsatmosmean_minus_nbm_rm30_l1",
      "delta_gefsatmosmean_minus_nbm_rs30_l1",
      "delta_gefsatmosmean_minus_nbm_slope15_l1",
      "delta_gefsatmosmean_minus_nbm_dev_rm30"
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