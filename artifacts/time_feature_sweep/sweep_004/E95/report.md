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
    "bias": 0.6172566883417434,
    "corr": 0.990907769325547,
    "mae": 1.852395091197645,
    "maxAE": 9.632911916542653,
    "medianAE": 1.5187725613340604,
    "n": 332,
    "rmse": 2.410405321773071
  },
  "train": {
    "bias": 2.809872305782567e-11,
    "corr": 0.9991175089905854,
    "mae": 0.5209615052680752,
    "maxAE": 6.520829333675415,
    "medianAE": 0.4235850551926603,
    "n": 1224,
    "rmse": 0.6959326247770244
  },
  "validation": {
    "bias": -0.41627046551898156,
    "corr": 0.9915180944219716,
    "mae": 2.030828193473124,
    "maxAE": 7.728100415145917,
    "medianAE": 1.5191020083132116,
    "n": 214,
    "rmse": 2.6765168060934084
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 870.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 670.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 534.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 508.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 488.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 408.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 398.0
    },
    {
      "feature": "resid_gefsatmosmean_tmax_f_iqr_rm60_l2",
      "importance": 318.0
    },
    {
      "feature": "sin_doy",
      "importance": 316.0
    },
    {
      "feature": "day_of_year",
      "importance": 307.0
    },
    {
      "feature": "cos_doy",
      "importance": 292.0
    },
    {
      "feature": "resid_rap_tmax_f_asym_rm60_l2",
      "importance": 246.0
    },
    {
      "feature": "resid_hrrr_tmax_f_iqr_rm60_l2",
      "importance": 232.0
    },
    {
      "feature": "resid_gefsatmosmean_tmax_f_q50_rm60_l2",
      "importance": 230.0
    },
    {
      "feature": "resid_rap_tmax_f_iqr_rm60_l2",
      "importance": 225.0
    },
    {
      "feature": "resid_nbm_tmax_f_q50_rm60_l2",
      "importance": 213.0
    },
    {
      "feature": "resid_gefsatmosmean_tmax_f_q10_rm60_l2",
      "importance": 211.0
    },
    {
      "feature": "resid_nbm_tmax_f_iqr_rm60_l2",
      "importance": 199.0
    },
    {
      "feature": "resid_hrrr_tmax_f_asym_rm60_l2",
      "importance": 192.0
    },
    {
      "feature": "resid_nbm_tmax_f_q10_rm60_l2",
      "importance": 185.0
    },
    {
      "feature": "resid_rap_tmax_f_q50_rm60_l2",
      "importance": 161.0
    },
    {
      "feature": "resid_nbm_tmax_f_asym_rm60_l2",
      "importance": 159.0
    },
    {
      "feature": "resid_hrrr_tmax_f_q10_rm60_l2",
      "importance": 159.0
    },
    {
      "feature": "resid_hrrr_tmax_f_q50_rm60_l2",
      "importance": 143.0
    },
    {
      "feature": "resid_rap_tmax_f_q90_rm60_l2",
      "importance": 142.0
    },
    {
      "feature": "resid_gefsatmosmean_tmax_f_asym_rm60_l2",
      "importance": 134.0
    },
    {
      "feature": "resid_gfs_tmax_f_asym_rm60_l2",
      "importance": 126.0
    },
    {
      "feature": "resid_rap_tmax_f_q10_rm60_l2",
      "importance": 122.0
    },
    {
      "feature": "resid_hrrr_tmax_f_q90_rm60_l2",
      "importance": 111.0
    },
    {
      "feature": "resid_nam_tmax_f_iqr_rm60_l2",
      "importance": 90.0
    },
    {
      "feature": "resid_nam_tmax_f_asym_rm60_l2",
      "importance": 84.0
    },
    {
      "feature": "resid_gfs_tmax_f_q10_rm60_l2",
      "importance": 81.0
    },
    {
      "feature": "resid_gefsatmosmean_tmax_f_q90_rm60_l2",
      "importance": 80.0
    },
    {
      "feature": "resid_gfs_tmax_f_iqr_rm60_l2",
      "importance": 65.0
    },
    {
      "feature": "is_weekend",
      "importance": 61.0
    },
    {
      "feature": "resid_nbm_tmax_f_q90_rm60_l2",
      "importance": 54.0
    },
    {
      "feature": "resid_nam_tmax_f_q90_rm60_l2",
      "importance": 40.0
    },
    {
      "feature": "resid_gfs_tmax_f_q50_rm60_l2",
      "importance": 38.0
    },
    {
      "feature": "resid_nam_tmax_f_q10_rm60_l2",
      "importance": 35.0
    },
    {
      "feature": "resid_nam_tmax_f_q50_rm60_l2",
      "importance": 34.0
    },
    {
      "feature": "resid_gfs_tmax_f_q90_rm60_l2",
      "importance": 32.0
    },
    {
      "feature": "month",
      "importance": 7.0
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
          "formula": "rolling_quantile(resid, q=0.10)",
          "name": "resid_nbm_tmax_f_q10_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.50)",
          "name": "resid_nbm_tmax_f_q50_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.90)",
          "name": "resid_nbm_tmax_f_q90_rm60_l2"
        },
        {
          "formula": "resid_q90 - resid_q10",
          "name": "resid_nbm_tmax_f_iqr_rm60_l2"
        },
        {
          "formula": "|q10| - |q90|",
          "name": "resid_nbm_tmax_f_asym_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.10)",
          "name": "resid_gfs_tmax_f_q10_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.50)",
          "name": "resid_gfs_tmax_f_q50_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.90)",
          "name": "resid_gfs_tmax_f_q90_rm60_l2"
        },
        {
          "formula": "resid_q90 - resid_q10",
          "name": "resid_gfs_tmax_f_iqr_rm60_l2"
        },
        {
          "formula": "|q10| - |q90|",
          "name": "resid_gfs_tmax_f_asym_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.10)",
          "name": "resid_gefsatmosmean_tmax_f_q10_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.50)",
          "name": "resid_gefsatmosmean_tmax_f_q50_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.90)",
          "name": "resid_gefsatmosmean_tmax_f_q90_rm60_l2"
        },
        {
          "formula": "resid_q90 - resid_q10",
          "name": "resid_gefsatmosmean_tmax_f_iqr_rm60_l2"
        },
        {
          "formula": "|q10| - |q90|",
          "name": "resid_gefsatmosmean_tmax_f_asym_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.10)",
          "name": "resid_nam_tmax_f_q10_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.50)",
          "name": "resid_nam_tmax_f_q50_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.90)",
          "name": "resid_nam_tmax_f_q90_rm60_l2"
        },
        {
          "formula": "resid_q90 - resid_q10",
          "name": "resid_nam_tmax_f_iqr_rm60_l2"
        },
        {
          "formula": "|q10| - |q90|",
          "name": "resid_nam_tmax_f_asym_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.10)",
          "name": "resid_hrrr_tmax_f_q10_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.50)",
          "name": "resid_hrrr_tmax_f_q50_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.90)",
          "name": "resid_hrrr_tmax_f_q90_rm60_l2"
        },
        {
          "formula": "resid_q90 - resid_q10",
          "name": "resid_hrrr_tmax_f_iqr_rm60_l2"
        },
        {
          "formula": "|q10| - |q90|",
          "name": "resid_hrrr_tmax_f_asym_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.10)",
          "name": "resid_rap_tmax_f_q10_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.50)",
          "name": "resid_rap_tmax_f_q50_rm60_l2"
        },
        {
          "formula": "rolling_quantile(resid, q=0.90)",
          "name": "resid_rap_tmax_f_q90_rm60_l2"
        },
        {
          "formula": "resid_q90 - resid_q10",
          "name": "resid_rap_tmax_f_iqr_rm60_l2"
        },
        {
          "formula": "|q10| - |q90|",
          "name": "resid_rap_tmax_f_asym_rm60_l2"
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
          "rap_tmax_f": 62.58964813232426,
          "resid_gefsatmosmean_tmax_f_asym_rm60_l2": -2.9389040417086636,
          "resid_gefsatmosmean_tmax_f_iqr_rm60_l2": 6.608345514112665,
          "resid_gefsatmosmean_tmax_f_q10_rm60_l2": -1.8361726436490877,
          "resid_gefsatmosmean_tmax_f_q50_rm60_l2": 1.3306907762097424,
          "resid_gefsatmosmean_tmax_f_q90_rm60_l2": 4.685800604838702,
          "resid_gfs_tmax_f_asym_rm60_l2": -1.9000000000000004,
          "resid_gfs_tmax_f_iqr_rm60_l2": 7.100000000000001,
          "resid_gfs_tmax_f_q10_rm60_l2": -3.0,
          "resid_gfs_tmax_f_q50_rm60_l2": 1.0,
          "resid_gfs_tmax_f_q90_rm60_l2": 4.100000000000001,
          "resid_hrrr_tmax_f_asym_rm60_l2": -3.411673278808511,
          "resid_hrrr_tmax_f_iqr_rm60_l2": 7.389031372070312,
          "resid_hrrr_tmax_f_q10_rm60_l2": -2.0921029663086324,
          "resid_hrrr_tmax_f_q50_rm60_l2": 1.463450927734332,
          "resid_hrrr_tmax_f_q90_rm60_l2": 4.959398193359336,
          "resid_nam_tmax_f_asym_rm60_l2": -3.1000000000000014,
          "resid_nam_tmax_f_iqr_rm60_l2": 8.0,
          "resid_nam_tmax_f_q10_rm60_l2": -2.0,
          "resid_nam_tmax_f_q50_rm60_l2": 1.0,
          "resid_nam_tmax_f_q90_rm60_l2": 5.100000000000001,
          "resid_nbm_tmax_f_asym_rm60_l2": -2.9784228515622884,
          "resid_nbm_tmax_f_iqr_rm60_l2": 5.397246582031231,
          "resid_nbm_tmax_f_q10_rm60_l2": -1.1112070312500426,
          "resid_nbm_tmax_f_q50_rm60_l2": 1.4070043945311674,
          "resid_nbm_tmax_f_q90_rm60_l2": 4.109603515624951,
          "resid_rap_tmax_f_asym_rm60_l2": -4.920300903320223,
          "resid_rap_tmax_f_iqr_rm60_l2": 7.7687533569336065,
          "resid_rap_tmax_f_q10_rm60_l2": -1.3362762451172228,
          "resid_rap_tmax_f_q50_rm60_l2": 2.491603393554648,
          "resid_rap_tmax_f_q90_rm60_l2": 6.213323974609337,
          "sin_doy": 0.15951094710994368
        },
        "method": "train_median"
      },
      "train_fitted": []
    },
    "description": "Residual quantiles per model",
    "experiment_id": "E95",
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
      "resid_nbm_tmax_f_q10_rm60_l2",
      "resid_nbm_tmax_f_q50_rm60_l2",
      "resid_nbm_tmax_f_q90_rm60_l2",
      "resid_nbm_tmax_f_iqr_rm60_l2",
      "resid_nbm_tmax_f_asym_rm60_l2",
      "resid_gfs_tmax_f_q10_rm60_l2",
      "resid_gfs_tmax_f_q50_rm60_l2",
      "resid_gfs_tmax_f_q90_rm60_l2",
      "resid_gfs_tmax_f_iqr_rm60_l2",
      "resid_gfs_tmax_f_asym_rm60_l2",
      "resid_gefsatmosmean_tmax_f_q10_rm60_l2",
      "resid_gefsatmosmean_tmax_f_q50_rm60_l2",
      "resid_gefsatmosmean_tmax_f_q90_rm60_l2",
      "resid_gefsatmosmean_tmax_f_iqr_rm60_l2",
      "resid_gefsatmosmean_tmax_f_asym_rm60_l2",
      "resid_nam_tmax_f_q10_rm60_l2",
      "resid_nam_tmax_f_q50_rm60_l2",
      "resid_nam_tmax_f_q90_rm60_l2",
      "resid_nam_tmax_f_iqr_rm60_l2",
      "resid_nam_tmax_f_asym_rm60_l2",
      "resid_hrrr_tmax_f_q10_rm60_l2",
      "resid_hrrr_tmax_f_q50_rm60_l2",
      "resid_hrrr_tmax_f_q90_rm60_l2",
      "resid_hrrr_tmax_f_iqr_rm60_l2",
      "resid_hrrr_tmax_f_asym_rm60_l2",
      "resid_rap_tmax_f_q10_rm60_l2",
      "resid_rap_tmax_f_q50_rm60_l2",
      "resid_rap_tmax_f_q90_rm60_l2",
      "resid_rap_tmax_f_iqr_rm60_l2",
      "resid_rap_tmax_f_asym_rm60_l2"
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