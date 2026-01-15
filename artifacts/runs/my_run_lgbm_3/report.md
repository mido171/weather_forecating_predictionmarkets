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
    "gefsatmos_tmp_spread_f": 0,
    "gefsatmosmean_tmax_f": 0,
    "gfs_tmax_f": 0,
    "hrrr_tmax_f": 0,
    "nam_tmax_f": 0,
    "nbm_tmax_f": 0,
    "rap_tmax_f": 0,
    "station_id": 0,
    "target_date_local": 0
  },
  "row_count": 1771,
  "split_counts": {
    "test": 332,
    "train": 1224,
    "validation": 214
  },
  "station_counts": {
    "KNYC": 1771
  }
}
```

## Model Summary
```json
{
  "best_params": {
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "lambda_l1": 0.1,
    "lambda_l2": 0.0,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "n_estimators": 300,
    "num_leaves": 31
  },
  "candidates": [
    {
      "best_params": {
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "lambda_l1": 0.1,
        "lambda_l2": 0.0,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "n_estimators": 300,
        "num_leaves": 31
      },
      "cv_score": -2.395654948836431,
      "model": "lgbm",
      "val_mae": 1.8473060844293916
    }
  ],
  "selected_model": "lgbm",
  "sigma_model": "lgbm"
}
```

## Metrics Summary
```json
{
  "baseline": {
    "climatology_test": {
      "bias": 1.4871987951807228,
      "mae": 7.180973895582328,
      "p50_abs_error": 5.875,
      "p90_abs_error": 15.5,
      "p95_abs_error": 18.474999999999994,
      "rmse": 9.283245774427915
    },
    "climatology_val": {
      "bias": 0.5875862806181661,
      "mae": 6.400670392767698,
      "p50_abs_error": 5.333333333333332,
      "p90_abs_error": 13.233333333333341,
      "p95_abs_error": 15.899999999999995,
      "rmse": 8.091352143340583
    },
    "ensemble_mean_test": {
      "bias": -0.9757020454431103,
      "mae": 2.040630921090158,
      "p50_abs_error": 1.7530219857820768,
      "p90_abs_error": 3.974477927849831,
      "p95_abs_error": 5.016255478594331,
      "rmse": 2.573524609161314
    },
    "ensemble_mean_val": {
      "bias": -1.0437450881480794,
      "mae": 1.8520732612230244,
      "p50_abs_error": 1.5978093524235177,
      "p90_abs_error": 3.7974776308777014,
      "p95_abs_error": 4.961002867274086,
      "rmse": 2.38555211604075
    }
  },
  "per_station_test": {
    "KNYC": {
      "bias": 0.47060300422225654,
      "mae": 1.8646580851368875,
      "p50_abs_error": 1.5043623264070227,
      "p90_abs_error": 3.9083932233684875,
      "p95_abs_error": 4.807935060814377,
      "rmse": 2.4521609354524543
    }
  },
  "probabilistic_test": {
    "brier_scores": {
      "ge_90": 0.027114544847333246,
      "lt_52": 0.03709330570886903
    },
    "log_loss": 3.4252950957812978
  },
  "probabilistic_validation": {
    "brier_scores": {
      "ge_90": 0.024285071322216894,
      "lt_52": 0.02888935873764984
    },
    "log_loss": 3.480630146810686
  },
  "test": {
    "bias": 0.47060300422225654,
    "mae": 1.8646580851368875,
    "p50_abs_error": 1.5043623264070227,
    "p90_abs_error": 3.9083932233684875,
    "p95_abs_error": 4.807935060814377,
    "rmse": 2.4521609354524543
  },
  "train": {
    "bias": 7.391236156828317e-06,
    "mae": 0.3376094024739352,
    "p50_abs_error": 0.2556889791709622,
    "p90_abs_error": 0.6961985502456082,
    "p95_abs_error": 0.9071125574686391,
    "rmse": 0.49432704383385456
  },
  "validation": {
    "bias": -0.2841026181737234,
    "mae": 1.8473060844293916,
    "p50_abs_error": 1.37248149825243,
    "p90_abs_error": 4.501273408918873,
    "p95_abs_error": 5.2617855096585435,
    "rmse": 2.461910952562048
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 450.0
    },
    {
      "feature": "ens_range",
      "importance": 407.0
    },
    {
      "feature": "hrrr_tmax_f_minus_ens_mean_abs",
      "importance": 392.0
    },
    {
      "feature": "nbm_tmax_f_minus_ens_mean_abs",
      "importance": 383.0
    },
    {
      "feature": "ens_std",
      "importance": 360.0
    },
    {
      "feature": "ens_iqr",
      "importance": 346.0
    },
    {
      "feature": "rap_tmax_f_minus_ens_mean_abs",
      "importance": 338.0
    },
    {
      "feature": "rap_tmax_f_minus_nbm_tmax_f",
      "importance": 329.0
    },
    {
      "feature": "cos_doy",
      "importance": 327.0
    },
    {
      "feature": "nam_tmax_f_minus_nbm_tmax_f",
      "importance": 322.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_ens_mean_abs",
      "importance": 314.0
    },
    {
      "feature": "nam_tmax_f_minus_ens_mean_abs",
      "importance": 313.0
    },
    {
      "feature": "day_of_year",
      "importance": 296.0
    },
    {
      "feature": "rap_tmax_f_minus_ens_mean",
      "importance": 288.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 270.0
    },
    {
      "feature": "sin_doy",
      "importance": 270.0
    },
    {
      "feature": "hrrr_tmax_f_minus_gfs_tmax_f",
      "importance": 260.0
    },
    {
      "feature": "nbm_tmax_f_minus_ens_mean",
      "importance": 256.0
    },
    {
      "feature": "nbm_tmax_f_minus_hrrr_tmax_f",
      "importance": 255.0
    },
    {
      "feature": "nam_tmax_f_minus_ens_mean",
      "importance": 232.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_gfs_tmax_f",
      "importance": 231.0
    },
    {
      "feature": "gfs_tmax_f_minus_ens_mean_abs",
      "importance": 230.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_ens_mean",
      "importance": 210.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 200.0
    },
    {
      "feature": "ens_max",
      "importance": 193.0
    },
    {
      "feature": "hrrr_tmax_f_minus_ens_mean",
      "importance": 189.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 185.0
    },
    {
      "feature": "ens_median",
      "importance": 182.0
    },
    {
      "feature": "gfs_tmax_f_minus_ens_mean",
      "importance": 171.0
    },
    {
      "feature": "ens_min",
      "importance": 155.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 153.0
    },
    {
      "feature": "ens_mean",
      "importance": 151.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 105.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 101.0
    },
    {
      "feature": "gfs_tmax_f_minus_nam_tmax_f",
      "importance": 74.0
    },
    {
      "feature": "month",
      "importance": 40.0
    },
    {
      "feature": "is_weekend",
      "importance": 22.0
    },
    {
      "feature": "gfs_tmax_f_missing",
      "importance": 0.0
    },
    {
      "feature": "nam_tmax_f_missing",
      "importance": 0.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_missing",
      "importance": 0.0
    },
    {
      "feature": "rap_tmax_f_missing",
      "importance": 0.0
    },
    {
      "feature": "hrrr_tmax_f_missing",
      "importance": 0.0
    },
    {
      "feature": "nbm_tmax_f_missing",
      "importance": 0.0
    },
    {
      "feature": "gefsatmos_tmp_spread_f_missing",
      "importance": 0.0
    },
    {
      "feature": "station_KNYC",
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
        "lgbm"
      ],
      "param_grid": {
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