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
    "bias": 0.5414798138416399,
    "corr": 0.9915085484366912,
    "mae": 1.783223811888062,
    "maxAE": 9.373010583900886,
    "medianAE": 1.3967298681116453,
    "n": 332,
    "rmse": 2.313673067869926
  },
  "train": {
    "bias": -6.152367794596168e-11,
    "corr": 0.9991947399408376,
    "mae": 0.48611105753605277,
    "maxAE": 6.575271262940994,
    "medianAE": 0.37494244527385234,
    "n": 1224,
    "rmse": 0.6651314073778077
  },
  "validation": {
    "bias": -0.3691530688299308,
    "corr": 0.9922918380538909,
    "mae": 1.930169341325457,
    "maxAE": 7.6842355754747445,
    "medianAE": 1.4742392744472603,
    "n": 214,
    "rmse": 2.549496707393441
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 813.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 525.0
    },
    {
      "feature": "w_max_rm60_l2",
      "importance": 445.0
    },
    {
      "feature": "w_entropy_rm60_l2",
      "importance": 376.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 372.0
    },
    {
      "feature": "sin_doy",
      "importance": 372.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 371.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 367.0
    },
    {
      "feature": "bias_nam_tmax_f_rm60_l2",
      "importance": 342.0
    },
    {
      "feature": "fcst_wmean_invmae_rm60_l2",
      "importance": 333.0
    },
    {
      "feature": "fcst_wmean_invmae_biascorr_rm60_l2",
      "importance": 311.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 308.0
    },
    {
      "feature": "mae_gfs_tmax_f_rm60_l2",
      "importance": 308.0
    },
    {
      "feature": "mae_rap_tmax_f_rm60_l2",
      "importance": 305.0
    },
    {
      "feature": "mae_gefsatmosmean_tmax_f_rm60_l2",
      "importance": 297.0
    },
    {
      "feature": "bias_nbm_tmax_f_rm60_l2",
      "importance": 295.0
    },
    {
      "feature": "mae_nam_tmax_f_rm60_l2",
      "importance": 291.0
    },
    {
      "feature": "bias_hrrr_tmax_f_rm60_l2",
      "importance": 290.0
    },
    {
      "feature": "cos_doy",
      "importance": 288.0
    },
    {
      "feature": "bias_gfs_tmax_f_rm60_l2",
      "importance": 283.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 273.0
    },
    {
      "feature": "day_of_year",
      "importance": 257.0
    },
    {
      "feature": "mae_hrrr_tmax_f_rm60_l2",
      "importance": 254.0
    },
    {
      "feature": "bias_rap_tmax_f_rm60_l2",
      "importance": 225.0
    },
    {
      "feature": "bias_gefsatmosmean_tmax_f_rm60_l2",
      "importance": 223.0
    },
    {
      "feature": "bias_wmean_invmae_rm60_l2",
      "importance": 214.0
    },
    {
      "feature": "mae_nbm_tmax_f_rm60_l2",
      "importance": 210.0
    },
    {
      "feature": "is_weekend",
      "importance": 41.0
    },
    {
      "feature": "month",
      "importance": 11.0
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
          "formula": "roll_mean(|resid|)",
          "name": "mae_nbm_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(resid)",
          "name": "bias_nbm_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "mae_gfs_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(resid)",
          "name": "bias_gfs_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "mae_gefsatmosmean_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(resid)",
          "name": "bias_gefsatmosmean_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "mae_nam_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(resid)",
          "name": "bias_nam_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "mae_hrrr_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(resid)",
          "name": "bias_hrrr_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "mae_rap_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(resid)",
          "name": "bias_rap_tmax_f_rm60_l2"
        },
        {
          "formula": "sum(w * model)",
          "name": "fcst_wmean_invmae_rm60_l2"
        },
        {
          "formula": "sum(w * bias)",
          "name": "bias_wmean_invmae_rm60_l2"
        },
        {
          "formula": "fcst_wmean + bias_wmean",
          "name": "fcst_wmean_invmae_biascorr_rm60_l2"
        },
        {
          "formula": "entropy(weights)",
          "name": "w_entropy_rm60_l2"
        },
        {
          "formula": "max(weights)",
          "name": "w_max_rm60_l2"
        }
      ],
      "imputation": {
        "fill_values": {
          "bias_gefsatmosmean_tmax_f_rm60_l2": 1.28967427807874,
          "bias_gfs_tmax_f_rm60_l2": 0.6833333333333333,
          "bias_hrrr_tmax_f_rm60_l2": 1.6668624369302976,
          "bias_nam_tmax_f_rm60_l2": 1.5,
          "bias_nbm_tmax_f_rm60_l2": 1.4753722330728531,
          "bias_rap_tmax_f_rm60_l2": 2.465282745361286,
          "bias_wmean_invmae_rm60_l2": 1.31987226559834,
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "fcst_wmean_invmae_biascorr_rm60_l2": 65.79760831705416,
          "fcst_wmean_invmae_rm60_l2": 63.423139209299705,
          "gefsatmos_tmp_spread_f": 1.985091821481542,
          "gefsatmosmean_tmax_f": 63.38288233996978,
          "gfs_tmax_f": 64.5,
          "hrrr_tmax_f": 63.429676513671915,
          "is_weekend": 0.0,
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
          "sin_doy": 0.15951094710994368,
          "w_entropy_rm60_l2": 1.7806623484481987,
          "w_max_rm60_l2": 0.20319216189939418
        },
        "method": "train_median"
      },
      "train_fitted": []
    },
    "description": "Inverse-MAE weights + bias correction",
    "experiment_id": "E94",
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
      "bias_nbm_tmax_f_rm60_l2",
      "mae_gfs_tmax_f_rm60_l2",
      "bias_gfs_tmax_f_rm60_l2",
      "mae_gefsatmosmean_tmax_f_rm60_l2",
      "bias_gefsatmosmean_tmax_f_rm60_l2",
      "mae_nam_tmax_f_rm60_l2",
      "bias_nam_tmax_f_rm60_l2",
      "mae_hrrr_tmax_f_rm60_l2",
      "bias_hrrr_tmax_f_rm60_l2",
      "mae_rap_tmax_f_rm60_l2",
      "bias_rap_tmax_f_rm60_l2",
      "fcst_wmean_invmae_rm60_l2",
      "bias_wmean_invmae_rm60_l2",
      "fcst_wmean_invmae_biascorr_rm60_l2",
      "w_entropy_rm60_l2",
      "w_max_rm60_l2"
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