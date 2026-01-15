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
    "bias": 0.36073078835979483,
    "corr": 0.9905704430875719,
    "mae": 1.8400594968441226,
    "maxAE": 8.579721081480074,
    "medianAE": 1.4361777633787334,
    "n": 332,
    "rmse": 2.3995457787228682
  },
  "train": {
    "bias": 9.64379716243736e-11,
    "corr": 0.9987028884349893,
    "mae": 0.6315067461504635,
    "maxAE": 6.946857078080427,
    "medianAE": 0.492408353178142,
    "n": 1224,
    "rmse": 0.8434315184131405
  },
  "validation": {
    "bias": -0.23544622869748724,
    "corr": 0.9915424638794678,
    "mae": 1.9211479428483178,
    "maxAE": 8.935695302817916,
    "medianAE": 1.5042679893960447,
    "n": 214,
    "rmse": 2.558332813927569
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 944.0
    },
    {
      "feature": "hrrr_tmax_f_seas_corr",
      "importance": 638.0
    },
    {
      "feature": "day_of_year",
      "importance": 575.0
    },
    {
      "feature": "gfs_tmax_f_seas_corr",
      "importance": 566.0
    },
    {
      "feature": "gefsatmosmean_tmax_f_seas_corr",
      "importance": 566.0
    },
    {
      "feature": "rap_tmax_f_seas_corr",
      "importance": 554.0
    },
    {
      "feature": "nbm_tmax_f_seas_corr",
      "importance": 551.0
    },
    {
      "feature": "cos_doy",
      "importance": 526.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 516.0
    },
    {
      "feature": "nam_tmax_f_seas_corr",
      "importance": 512.0
    },
    {
      "feature": "ensmean_seas_corr",
      "importance": 498.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 489.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 479.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 438.0
    },
    {
      "feature": "sin_doy",
      "importance": 421.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 271.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 244.0
    },
    {
      "feature": "bias_nbm_tmax_f_month",
      "importance": 49.0
    },
    {
      "feature": "is_weekend",
      "importance": 42.0
    },
    {
      "feature": "bias_gefsatmosmean_tmax_f_month",
      "importance": 38.0
    },
    {
      "feature": "bias_gfs_tmax_f_month",
      "importance": 31.0
    },
    {
      "feature": "month",
      "importance": 20.0
    },
    {
      "feature": "bias_hrrr_tmax_f_month",
      "importance": 18.0
    },
    {
      "feature": "bias_nam_tmax_f_month",
      "importance": 11.0
    },
    {
      "feature": "bias_rap_tmax_f_month",
      "importance": 3.0
    },
    {
      "feature": "bias_ensmean_month",
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
  "experiment": {
    "derived_features": {
      "formulas": [
        {
          "formula": "seasonal_bias_by_month",
          "name": "bias_nbm_tmax_f_month"
        },
        {
          "formula": "forecast + seasonal_bias",
          "name": "nbm_tmax_f_seas_corr"
        },
        {
          "formula": "seasonal_bias_by_month",
          "name": "bias_gfs_tmax_f_month"
        },
        {
          "formula": "forecast + seasonal_bias",
          "name": "gfs_tmax_f_seas_corr"
        },
        {
          "formula": "seasonal_bias_by_month",
          "name": "bias_gefsatmosmean_tmax_f_month"
        },
        {
          "formula": "forecast + seasonal_bias",
          "name": "gefsatmosmean_tmax_f_seas_corr"
        },
        {
          "formula": "seasonal_bias_by_month",
          "name": "bias_nam_tmax_f_month"
        },
        {
          "formula": "forecast + seasonal_bias",
          "name": "nam_tmax_f_seas_corr"
        },
        {
          "formula": "seasonal_bias_by_month",
          "name": "bias_hrrr_tmax_f_month"
        },
        {
          "formula": "forecast + seasonal_bias",
          "name": "hrrr_tmax_f_seas_corr"
        },
        {
          "formula": "seasonal_bias_by_month",
          "name": "bias_rap_tmax_f_month"
        },
        {
          "formula": "forecast + seasonal_bias",
          "name": "rap_tmax_f_seas_corr"
        },
        {
          "formula": "seasonal_bias_by_month",
          "name": "bias_ensmean_month"
        },
        {
          "formula": "ens_mean + seasonal_bias",
          "name": "ensmean_seas_corr"
        }
      ],
      "imputation": {
        "fill_values": {
          "bias_ensmean_month": 1.62613479745513,
          "bias_gefsatmosmean_tmax_f_month": 1.2665808184223368,
          "bias_gfs_tmax_f_month": 0.7311827956989247,
          "bias_hrrr_tmax_f_month": 1.8317503012379885,
          "bias_nam_tmax_f_month": 1.3440860215053763,
          "bias_nbm_tmax_f_month": 1.1916886655745298,
          "bias_rap_tmax_f_month": 2.35266129032254,
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "ensmean_seas_corr": 64.9926728058791,
          "gefsatmos_tmp_spread_f": 1.985091821481542,
          "gefsatmosmean_tmax_f": 63.38288233996978,
          "gefsatmosmean_tmax_f_seas_corr": 64.77960502982195,
          "gfs_tmax_f": 64.5,
          "gfs_tmax_f_seas_corr": 65.46236559139786,
          "hrrr_tmax_f": 63.429676513671915,
          "hrrr_tmax_f_seas_corr": 64.94649271647137,
          "is_weekend": 0.0,
          "month": 6.0,
          "nam_tmax_f": 63.0,
          "nam_tmax_f_seas_corr": 64.65833333333333,
          "nbm_tmax_f": 63.175991210937525,
          "nbm_tmax_f_seas_corr": 64.88439278157549,
          "rap_tmax_f": 62.58964813232426,
          "rap_tmax_f_seas_corr": 65.1962021891276,
          "sin_doy": 0.15951094710994368
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
        },
        {
          "by_station": {
            "KNYC": {
              "1": 1.7326456355776996,
              "2": 2.5828636159300395,
              "3": 2.9545922050780073,
              "4": 2.8650429184725312,
              "5": 1.62613479745513,
              "6": 0.11776578265888418,
              "7": -0.22521309658781344,
              "8": 0.22600023862209,
              "9": 0.3680983678600917,
              "10": 0.3780402779880391,
              "11": 1.4862920172796994,
              "12": 1.6307742717529563
            }
          },
          "default": 1.3692090830113726,
          "fit_on": "train",
          "name": "bias_ensmean_month"
        }
      ]
    },
    "description": "Seasonal bias lookup by month",
    "experiment_id": "E99",
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
      "bias_nbm_tmax_f_month",
      "nbm_tmax_f_seas_corr",
      "bias_gfs_tmax_f_month",
      "gfs_tmax_f_seas_corr",
      "bias_gefsatmosmean_tmax_f_month",
      "gefsatmosmean_tmax_f_seas_corr",
      "bias_nam_tmax_f_month",
      "nam_tmax_f_seas_corr",
      "bias_hrrr_tmax_f_month",
      "hrrr_tmax_f_seas_corr",
      "bias_rap_tmax_f_month",
      "rap_tmax_f_seas_corr",
      "bias_ensmean_month",
      "ensmean_seas_corr"
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