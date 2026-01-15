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
    "bias": 0.30135495056161654,
    "corr": 0.9907567244915868,
    "mae": 1.800633902763751,
    "maxAE": 9.211425508649988,
    "medianAE": 1.349085774441587,
    "n": 332,
    "rmse": 2.3670480839498573
  },
  "train": {
    "bias": 2.2833386597598042e-11,
    "corr": 0.9992251518778247,
    "mae": 0.48306230686991514,
    "maxAE": 6.1609366741171065,
    "medianAE": 0.38601311965796725,
    "n": 1224,
    "rmse": 0.652593611531172
  },
  "validation": {
    "bias": -0.27418685586809255,
    "corr": 0.9923073703874904,
    "mae": 1.8548124454243442,
    "maxAE": 7.906418265477598,
    "medianAE": 1.4418293225576058,
    "n": 214,
    "rmse": 2.4588256583507166
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "pc2_score",
      "importance": 852.0
    },
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 723.0
    },
    {
      "feature": "pc1_score",
      "importance": 713.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 694.0
    },
    {
      "feature": "pc1_rs30_l1",
      "importance": 658.0
    },
    {
      "feature": "pc2_rm30_l1",
      "importance": 654.0
    },
    {
      "feature": "pc1_rm30_l1",
      "importance": 565.0
    },
    {
      "feature": "pc2_rs30_l1",
      "importance": 538.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 513.0
    },
    {
      "feature": "sin_doy",
      "importance": 485.0
    },
    {
      "feature": "day_of_year",
      "importance": 460.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 459.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 441.0
    },
    {
      "feature": "cos_doy",
      "importance": 425.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 374.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 352.0
    },
    {
      "feature": "is_weekend",
      "importance": 51.0
    },
    {
      "feature": "month",
      "importance": 43.0
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
          "formula": "PCA1(disagreement)",
          "name": "pc1_score"
        },
        {
          "formula": "PCA2(disagreement)",
          "name": "pc2_score"
        },
        {
          "formula": "roll_mean(pc1)",
          "name": "pc1_rm30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(pc1)",
          "name": "pc1_rs30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_mean(pc2)",
          "name": "pc2_rm30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
            "window": 30
          }
        },
        {
          "formula": "roll_std(pc2)",
          "name": "pc2_rs30_l1",
          "params": {
            "lag": 1,
            "min_periods": 21,
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
          "pc1_rm30_l1": -0.212238448535623,
          "pc1_rs30_l1": 3.1555885292009087,
          "pc1_score": -0.1970602024153067,
          "pc2_rm30_l1": -0.26731402438034607,
          "pc2_rs30_l1": 1.902390854609786,
          "pc2_score": 0.020120783715390778,
          "rap_tmax_f": 62.58964813232426,
          "sin_doy": 0.15951094710994368
        },
        "method": "train_median"
      },
      "train_fitted": [
        {
          "components": [
            [
              -0.03535918428502761,
              -0.4876205502516332,
              -0.05283921495132042,
              -0.4648576103690134,
              0.5031992107233756,
              0.5374773491336189
            ],
            [
              0.15376508797145158,
              -0.05561020909439385,
              0.7981133075932194,
              -0.4966055323487641,
              -0.12984283765323362,
              -0.2698198164682806
            ]
          ],
          "description": "PCA on disagreement vector",
          "explained_variance_ratio": [
            0.5176409409086095,
            0.19296513474491014
          ],
          "fit_on": "train",
          "name": "pca_disagreement"
        }
      ]
    },
    "description": "PCA disagreement regime + rolling stats",
    "experiment_id": "E34",
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
      "pc1_score",
      "pc2_score",
      "pc1_rm30_l1",
      "pc1_rs30_l1",
      "pc2_rm30_l1",
      "pc2_rs30_l1"
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