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
    "bias": 0.3072370072223671,
    "corr": 0.9904774566484126,
    "mae": 1.844656213311922,
    "maxAE": 10.446924040526206,
    "medianAE": 1.5342839892614357,
    "n": 332,
    "rmse": 2.401520016531885
  },
  "train": {
    "bias": -3.303693992302512e-11,
    "corr": 0.9992326520239483,
    "mae": 0.47696120534344344,
    "maxAE": 6.505613083594621,
    "medianAE": 0.3756318957171807,
    "n": 1224,
    "rmse": 0.6491315006492241
  },
  "validation": {
    "bias": -0.1695596604057565,
    "corr": 0.9922391165980917,
    "mae": 1.9126634635181383,
    "maxAE": 7.2551332532889745,
    "medianAE": 1.440183394374813,
    "n": 214,
    "rmse": 2.470339331414503
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 793.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 584.0
    },
    {
      "feature": "rank_hrrr_tmax_f_roll_std_30_l1",
      "importance": 528.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 491.0
    },
    {
      "feature": "rank_gfs_tmax_f_roll_std_30_l1",
      "importance": 447.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 431.0
    },
    {
      "feature": "rank_gefsatmosmean_tmax_f_roll_std_30_l1",
      "importance": 420.0
    },
    {
      "feature": "rank_nam_tmax_f_roll_std_30_l1",
      "importance": 409.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 378.0
    },
    {
      "feature": "sin_doy",
      "importance": 374.0
    },
    {
      "feature": "rank_nbm_tmax_f_roll_std_30_l1",
      "importance": 373.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 359.0
    },
    {
      "feature": "top_identity_entropy_30_l1",
      "importance": 355.0
    },
    {
      "feature": "rank_rap_tmax_f_roll_std_30_l1",
      "importance": 346.0
    },
    {
      "feature": "rank_rap_tmax_f_roll_mean_30_l1",
      "importance": 333.0
    },
    {
      "feature": "rank_gefsatmosmean_tmax_f_roll_mean_30_l1",
      "importance": 326.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 322.0
    },
    {
      "feature": "rank_nam_tmax_f_roll_mean_30_l1",
      "importance": 320.0
    },
    {
      "feature": "day_of_year",
      "importance": 294.0
    },
    {
      "feature": "cos_doy",
      "importance": 285.0
    },
    {
      "feature": "rank_gfs_tmax_f_roll_mean_30_l1",
      "importance": 274.0
    },
    {
      "feature": "rank_hrrr_tmax_f_roll_mean_30_l1",
      "importance": 266.0
    },
    {
      "feature": "rank_nbm_tmax_f_roll_mean_30_l1",
      "importance": 223.0
    },
    {
      "feature": "is_weekend",
      "importance": 55.0
    },
    {
      "feature": "month",
      "importance": 14.0
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
          "formula": "roll_mean(rank_nbm_tmax_f)",
          "name": "rank_nbm_tmax_f_roll_mean_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(rank_nbm_tmax_f)",
          "name": "rank_nbm_tmax_f_roll_std_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_mean(rank_gfs_tmax_f)",
          "name": "rank_gfs_tmax_f_roll_mean_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(rank_gfs_tmax_f)",
          "name": "rank_gfs_tmax_f_roll_std_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_mean(rank_gefsatmosmean_tmax_f)",
          "name": "rank_gefsatmosmean_tmax_f_roll_mean_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(rank_gefsatmosmean_tmax_f)",
          "name": "rank_gefsatmosmean_tmax_f_roll_std_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_mean(rank_nam_tmax_f)",
          "name": "rank_nam_tmax_f_roll_mean_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(rank_nam_tmax_f)",
          "name": "rank_nam_tmax_f_roll_std_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_mean(rank_hrrr_tmax_f)",
          "name": "rank_hrrr_tmax_f_roll_mean_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(rank_hrrr_tmax_f)",
          "name": "rank_hrrr_tmax_f_roll_std_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_mean(rank_rap_tmax_f)",
          "name": "rank_rap_tmax_f_roll_mean_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(rank_rap_tmax_f)",
          "name": "rank_rap_tmax_f_roll_std_30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "entropy(top_model_freq_30_l1)",
          "name": "top_identity_entropy_30_l1",
          "params": {
            "eps": 1e-09,
            "lag": 1,
            "window": 30
          }
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
          "month": 6.0,
          "nam_tmax_f": 63.0,
          "nbm_tmax_f": 63.175991210937525,
          "rank_gefsatmosmean_tmax_f_roll_mean_30_l1": 3.6,
          "rank_gefsatmosmean_tmax_f_roll_std_30_l1": 1.4282856857085735,
          "rank_gfs_tmax_f_roll_mean_30_l1": 3.8333333333333335,
          "rank_gfs_tmax_f_roll_std_30_l1": 1.5807874268505866,
          "rank_hrrr_tmax_f_roll_mean_30_l1": 3.533333333333333,
          "rank_hrrr_tmax_f_roll_std_30_l1": 1.5860503004493747,
          "rank_nam_tmax_f_roll_mean_30_l1": 3.7666666666666666,
          "rank_nam_tmax_f_roll_std_30_l1": 1.7307673314329568,
          "rank_nbm_tmax_f_roll_mean_30_l1": 3.3666666666666667,
          "rank_nbm_tmax_f_roll_std_30_l1": 1.0749676997731406,
          "rank_rap_tmax_f_roll_mean_30_l1": 2.566666666666667,
          "rank_rap_tmax_f_roll_std_30_l1": 1.6428295373802138,
          "rap_tmax_f": 62.58964813232426,
          "sin_doy": 0.15951094710994368,
          "top_identity_entropy_30_l1": 1.3825947200942477
        },
        "method": "train_median"
      },
      "train_fitted": []
    },
    "description": "Rank stability stats + top identity entropy",
    "experiment_id": "E11",
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
      "rank_nbm_tmax_f_roll_mean_30_l1",
      "rank_nbm_tmax_f_roll_std_30_l1",
      "rank_gfs_tmax_f_roll_mean_30_l1",
      "rank_gfs_tmax_f_roll_std_30_l1",
      "rank_gefsatmosmean_tmax_f_roll_mean_30_l1",
      "rank_gefsatmosmean_tmax_f_roll_std_30_l1",
      "rank_nam_tmax_f_roll_mean_30_l1",
      "rank_nam_tmax_f_roll_std_30_l1",
      "rank_hrrr_tmax_f_roll_mean_30_l1",
      "rank_hrrr_tmax_f_roll_std_30_l1",
      "rank_rap_tmax_f_roll_mean_30_l1",
      "rank_rap_tmax_f_roll_std_30_l1",
      "top_identity_entropy_30_l1"
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