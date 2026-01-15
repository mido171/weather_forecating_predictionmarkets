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
    "bias": 0.30756877839966734,
    "corr": 0.9914476908295592,
    "mae": 1.7355704894090258,
    "maxAE": 8.865100497852211,
    "medianAE": 1.4299839592322563,
    "n": 332,
    "rmse": 2.279117896848574
  },
  "train": {
    "bias": 4.679808320092948e-11,
    "corr": 0.9992206907013779,
    "mae": 0.47433313467364313,
    "maxAE": 6.782615973291406,
    "medianAE": 0.3738754018854955,
    "n": 1224,
    "rmse": 0.6542745317589042
  },
  "validation": {
    "bias": -0.45068049774049324,
    "corr": 0.9919174140193174,
    "mae": 1.9634172590705041,
    "maxAE": 7.943050209181514,
    "medianAE": 1.403962034127595,
    "n": 214,
    "rmse": 2.5794897848635796
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 772.0
    },
    {
      "feature": "w_max_rm60_l2",
      "importance": 552.0
    },
    {
      "feature": "hybrid_minus_ensmean",
      "importance": 499.0
    },
    {
      "feature": "w_entropy_rm60_l2",
      "importance": 453.0
    },
    {
      "feature": "cos_doy",
      "importance": 427.0
    },
    {
      "feature": "mae_gefsatmosmean_tmax_f_rm60_l2",
      "importance": 388.0
    },
    {
      "feature": "mae_gfs_tmax_f_rm60_l2",
      "importance": 365.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 337.0
    },
    {
      "feature": "mae_nam_tmax_f_rm60_l2",
      "importance": 337.0
    },
    {
      "feature": "fcst_hybrid_seas_invmae_rm60_l2",
      "importance": 331.0
    },
    {
      "feature": "day_of_year",
      "importance": 330.0
    },
    {
      "feature": "mae_hrrr_tmax_f_rm60_l2",
      "importance": 328.0
    },
    {
      "feature": "nbm_tmax_f_seas_corr",
      "importance": 327.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_seas_corr",
      "importance": 326.0
    },
    {
      "feature": "rap_tmax_f_seas_corr",
      "importance": 322.0
    },
    {
      "feature": "gfs_tmax_f_seas_corr",
      "importance": 310.0
    },
    {
      "feature": "hrrr_tmax_f_seas_corr",
      "importance": 309.0
    },
    {
      "feature": "nam_tmax_f_seas_corr",
      "importance": 305.0
    },
    {
      "feature": "sin_doy",
      "importance": 290.0
    },
    {
      "feature": "mae_rap_tmax_f_rm60_l2",
      "importance": 290.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 283.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 252.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 251.0
    },
    {
      "feature": "mae_nbm_tmax_f_rm60_l2",
      "importance": 250.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 158.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 157.0
    },
    {
      "feature": "is_weekend",
      "importance": 36.0
    },
    {
      "feature": "month",
      "importance": 15.0
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
          "formula": "forecast + seasonal_bias",
          "name": "nbm_tmax_f_seas_corr"
        },
        {
          "formula": "forecast + seasonal_bias",
          "name": "gfs_tmax_f_seas_corr"
        },
        {
          "formula": "forecast + seasonal_bias",
          "name": "gefsatmosmean_tmax_f_seas_corr"
        },
        {
          "formula": "forecast + seasonal_bias",
          "name": "nam_tmax_f_seas_corr"
        },
        {
          "formula": "forecast + seasonal_bias",
          "name": "hrrr_tmax_f_seas_corr"
        },
        {
          "formula": "forecast + seasonal_bias",
          "name": "rap_tmax_f_seas_corr"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "mae_nbm_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "mae_gfs_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "mae_gefsatmosmean_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "mae_nam_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "mae_hrrr_tmax_f_rm60_l2"
        },
        {
          "formula": "roll_mean(|resid|)",
          "name": "mae_rap_tmax_f_rm60_l2"
        },
        {
          "formula": "sum(w * seasonal_corrected_forecast)",
          "name": "fcst_hybrid_seas_invmae_rm60_l2"
        },
        {
          "formula": "entropy(weights)",
          "name": "w_entropy_rm60_l2"
        },
        {
          "formula": "max(weights)",
          "name": "w_max_rm60_l2"
        },
        {
          "formula": "hybrid_forecast - ens_mean",
          "name": "hybrid_minus_ensmean"
        }
      ],
      "imputation": {
        "fill_values": {
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "fcst_hybrid_seas_invmae_rm60_l2": 65.01755705687174,
          "gefsatmos_tmp_spread_f": 1.985091821481542,
          "gefsatmosmean_tmax_f": 63.38288233996978,
          "gefsatmosmean_tmax_f_seas_corr": 64.77960502982195,
          "gfs_tmax_f": 64.5,
          "gfs_tmax_f_seas_corr": 65.46236559139786,
          "hrrr_tmax_f": 63.429676513671915,
          "hrrr_tmax_f_seas_corr": 64.94649271647137,
          "hybrid_minus_ensmean": 1.5381633248184414,
          "is_weekend": 0.0,
          "mae_gefsatmosmean_tmax_f_rm60_l2": 2.6858807899384423,
          "mae_gfs_tmax_f_rm60_l2": 2.45,
          "mae_hrrr_tmax_f_rm60_l2": 2.7992452697753674,
          "mae_nam_tmax_f_rm60_l2": 2.7333333333333334,
          "mae_nbm_tmax_f_rm60_l2": 2.2609038085937034,
          "mae_rap_tmax_f_rm60_l2": 3.2189987284342156,
          "month": 6.0,
          "nam_tmax_f": 63.0,
          "nam_tmax_f_seas_corr": 64.65833333333333,
          "nbm_tmax_f": 63.175991210937525,
          "nbm_tmax_f_seas_corr": 64.88439278157549,
          "rap_tmax_f": 62.58964813232426,
          "rap_tmax_f_seas_corr": 65.1962021891276,
          "sin_doy": 0.15951094710994368,
          "w_entropy_rm60_l2": 1.7806623484481987,
          "w_max_rm60_l2": 0.20319216189939418
        },
        "method": "train_median"
      },
      "train_fitted": [
        {
          "by_station": {
            "KNYC": {
              "1": 1.7691700793850136,
              "2": 2.518820237379745,
              "3": 2.54716993762594,
              "4": 2.6443927815754553,
              "5": 1.1122679876511368,
              "6": 0.25705908203118866,
              "7": 0.6584183362734544,
              "8": 1.1916886655745298,
              "9": 0.9880117187499312,
              "10": 0.8030642536121683,
              "11": 1.7591414388020186,
              "12": 1.7940301054266816
            }
          },
          "default": 1.51676756216803,
          "fit_on": "train",
          "name": "bias_nbm_tmax_f_month"
        },
        {
          "by_station": {
            "KNYC": {
              "1": 0.7311827956989247,
              "2": 1.901098901098901,
              "3": 2.588709677419355,
              "4": 1.6666666666666667,
              "5": 1.1370967741935485,
              "6": 0.35833333333333334,
              "7": 0.6559139784946236,
              "8": 1.021505376344086,
              "9": 0.5222222222222223,
              "10": -0.5376344086021505,
              "11": 0.28888888888888886,
              "12": -0.25806451612903225
            }
          },
          "default": 0.8995098039215687,
          "fit_on": "train",
          "name": "bias_gfs_tmax_f_month"
        },
        {
          "by_station": {
            "KNYC": {
              "1": 1.980709195743315,
              "2": 2.898551981788316,
              "3": 3.0226840327457927,
              "4": 3.0164573551956915,
              "5": 1.2665808184223368,
              "6": -0.8331334025979817,
              "7": -2.4674448235413475,
              "8": -1.6182967952288336,
              "9": -1.3764182428175902,
              "10": -0.703184343023817,
              "11": 1.4136988606770424,
              "12": 2.0763094226255827
            }
          },
          "default": 0.8112122703170966,
          "fit_on": "train",
          "name": "bias_gefsatmosmean_tmax_f_month"
        },
        {
          "by_station": {
            "KNYC": {
              "1": 1.10752688172043,
              "2": 2.197802197802198,
              "3": 2.4838709677419355,
              "4": 2.658333333333333,
              "5": 2.056451612903226,
              "6": 1.2083333333333333,
              "7": 1.3440860215053763,
              "8": 1.3870967741935485,
              "9": 0.18888888888888888,
              "10": -0.21505376344086022,
              "11": 1.1444444444444444,
              "12": 0.8172043010752689
            }
          },
          "default": 1.4379084967320261,
          "fit_on": "train",
          "name": "bias_nam_tmax_f_month"
        },
        {
          "by_station": {
            "KNYC": {
              "1": 1.86685826455389,
              "2": 2.28502960540432,
              "3": 3.032566424954281,
              "4": 3.2187180074055575,
              "5": 1.8317503012379885,
              "6": -0.7251734924316815,
              "7": -1.9772940424437575,
              "8": -1.397282202936029,
              "9": 0.18598876953120963,
              "10": 1.0714785602528147,
              "11": 1.6631111653645418,
              "12": 2.4194463848810908
            }
          },
          "default": 1.1937872154883677,
          "fit_on": "train",
          "name": "bias_hrrr_tmax_f_month"
        },
        {
          "by_station": {
            "KNYC": {
              "1": 2.9404265963646257,
              "2": 3.6958787721067585,
              "3": 4.052552189980743,
              "4": 3.985689366658488,
              "5": 2.35266129032254,
              "6": 0.4411758422851156,
              "7": 0.43504195018477126,
              "8": 0.7712896137852405,
              "9": 1.6998968505858965,
              "10": 1.849571369130084,
              "11": 2.6484673055012604,
              "12": 2.935719932638147
            }
          },
          "default": 2.356069149441148,
          "fit_on": "train",
          "name": "bias_rap_tmax_f_month"
        }
      ]
    },
    "description": "Seasonal-corrected inverse-MAE blend",
    "experiment_id": "E100",
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
      "nbm_tmax_f_seas_corr",
      "gfs_tmax_f_seas_corr",
      "gefsatmosmean_tmax_f_seas_corr",
      "nam_tmax_f_seas_corr",
      "hrrr_tmax_f_seas_corr",
      "rap_tmax_f_seas_corr",
      "mae_nbm_tmax_f_rm60_l2",
      "mae_gfs_tmax_f_rm60_l2",
      "mae_gefsatmosmean_tmax_f_rm60_l2",
      "mae_nam_tmax_f_rm60_l2",
      "mae_hrrr_tmax_f_rm60_l2",
      "mae_rap_tmax_f_rm60_l2",
      "fcst_hybrid_seas_invmae_rm60_l2",
      "w_entropy_rm60_l2",
      "w_max_rm60_l2",
      "hybrid_minus_ensmean"
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