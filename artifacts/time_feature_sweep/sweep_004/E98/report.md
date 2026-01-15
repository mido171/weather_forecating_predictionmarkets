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
    "bias": 0.3959343223022739,
    "corr": 0.9905105375667702,
    "mae": 1.8476625814247694,
    "maxAE": 9.850617283170422,
    "medianAE": 1.4891775194085017,
    "n": 332,
    "rmse": 2.4096575522365753
  },
  "train": {
    "bias": -1.3231439763737304e-10,
    "corr": 0.9992551574565707,
    "mae": 0.46159834705172426,
    "maxAE": 7.067779030336169,
    "medianAE": 0.35383687915852136,
    "n": 1224,
    "rmse": 0.639973900438592
  },
  "validation": {
    "bias": -0.2369151557061511,
    "corr": 0.9920368111753557,
    "mae": 1.9141540883199415,
    "maxAE": 7.5907012884924825,
    "medianAE": 1.458266764363266,
    "n": 214,
    "rmse": 2.5065672415930518
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 809.0
    },
    {
      "feature": "p_best_gfs_tmax_f",
      "importance": 708.0
    },
    {
      "feature": "p_best_nam_tmax_f",
      "importance": 662.0
    },
    {
      "feature": "p_best_hrrr_tmax_f",
      "importance": 647.0
    },
    {
      "feature": "p_best_gefsatmosmean_tmax_f",
      "importance": 636.0
    },
    {
      "feature": "p_best_rap_tmax_f",
      "importance": 598.0
    },
    {
      "feature": "p_best_nbm_tmax_f",
      "importance": 595.0
    },
    {
      "feature": "bestprob_entropy",
      "importance": 563.0
    },
    {
      "feature": "nbm_tmax_f",
      "importance": 546.0
    },
    {
      "feature": "sin_doy",
      "importance": 499.0
    },
    {
      "feature": "cos_doy",
      "importance": 395.0
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 393.0
    },
    {
      "feature": "day_of_year",
      "importance": 363.0
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 362.0
    },
    {
      "feature": "fcst_expected_bestprob",
      "importance": 362.0
    },
    {
      "feature": "rap_tmax_f",
      "importance": 315.0
    },
    {
      "feature": "nam_tmax_f",
      "importance": 247.0
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 230.0
    },
    {
      "feature": "is_weekend",
      "importance": 41.0
    },
    {
      "feature": "month",
      "importance": 29.0
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
          "formula": "P(best_model == nbm_tmax_f)",
          "name": "p_best_nbm_tmax_f"
        },
        {
          "formula": "P(best_model == gfs_tmax_f)",
          "name": "p_best_gfs_tmax_f"
        },
        {
          "formula": "P(best_model == gefsatmosmean_tmax_f)",
          "name": "p_best_gefsatmosmean_tmax_f"
        },
        {
          "formula": "P(best_model == nam_tmax_f)",
          "name": "p_best_nam_tmax_f"
        },
        {
          "formula": "P(best_model == hrrr_tmax_f)",
          "name": "p_best_hrrr_tmax_f"
        },
        {
          "formula": "P(best_model == rap_tmax_f)",
          "name": "p_best_rap_tmax_f"
        },
        {
          "formula": "sum(p_best * model)",
          "name": "fcst_expected_bestprob"
        },
        {
          "formula": "entropy(p_best)",
          "name": "bestprob_entropy"
        }
      ],
      "imputation": {
        "fill_values": {
          "bestprob_entropy": 1.6899574493984209,
          "cos_doy": -0.06553708331203101,
          "day_of_year": 167.0,
          "fcst_expected_bestprob": 64.09475617617397,
          "gefsatmos_tmp_spread_f": 1.985091821481542,
          "gefsatmosmean_tmax_f": 63.38288233996978,
          "gfs_tmax_f": 64.5,
          "hrrr_tmax_f": 63.429676513671915,
          "is_weekend": 0.0,
          "month": 6.0,
          "nam_tmax_f": 63.0,
          "nbm_tmax_f": 63.175991210937525,
          "p_best_gefsatmosmean_tmax_f": 0.1483741709630612,
          "p_best_gfs_tmax_f": 0.25062541775434294,
          "p_best_hrrr_tmax_f": 0.13932802497332364,
          "p_best_nam_tmax_f": 0.15159218055540546,
          "p_best_nbm_tmax_f": 0.17536313925503344,
          "p_best_rap_tmax_f": 0.0854113589343507,
          "rap_tmax_f": 62.58964813232426,
          "sin_doy": 0.15951094710994368
        },
        "method": "train_median"
      },
      "train_fitted": [
        {
          "coef": [
            [
              1.1599247077819934,
              -0.1843013042669481,
              -0.3681057777398184,
              -0.4954652437803356,
              -1.0023957361102533,
              0.3274815958432149,
              0.7333029425595861,
              -0.6502782324545173,
              -0.4057070327593679,
              -0.48907744376069817,
              -0.0279745676937471
            ],
            [
              -0.7108747535694859,
              2.1269914051522902,
              -0.15977609105399948,
              -0.3137917079851882,
              -0.45203350156873573,
              -0.5256738760639479,
              0.16883360617078258,
              -0.25965406433636373,
              0.0700257305349739,
              0.0658270031170206,
              -0.030914785109990794
            ],
            [
              0.3748973251745396,
              -0.6885212974694013,
              1.5499610118681042,
              -0.22858564469818785,
              -1.3695446863364544,
              0.1729256207703692,
              0.26478496479186153,
              -0.07030587606272583,
              -0.16224048145456824,
              -0.1080854508988318,
              0.04429405445995809
            ],
            [
              -0.1880097276389995,
              -0.4695745131552942,
              0.051115172647982425,
              2.3884670078598296,
              -0.21218579038931634,
              -1.4878862191673627,
              -0.8557014500448671,
              0.7744880919050521,
              0.11484955964246821,
              0.19073575264108245,
              -0.0874248624947516
            ],
            [
              -0.7071592705149883,
              -0.8028122636003032,
              0.20185716463378311,
              -0.6252637525341174,
              2.5086842986255773,
              -0.25207220060019087,
              -0.26037875028850227,
              0.14135913164232486,
              0.23586952884237086,
              0.3519938600417825,
              0.05304684566780748
            ],
            [
              0.07122171876693731,
              0.018217973339663386,
              -1.2750514803560524,
              -0.7253606588619986,
              0.5274754157791886,
              1.7652250792179227,
              -0.05084131318885825,
              0.06439094930623722,
              0.14720269519412504,
              -0.011393721140356559,
              0.04897331517072242
            ]
          ],
          "description": "multinomial logistic regression",
          "features": [
            "nbm_tmax_f",
            "gfs_tmax_f",
            "gefsatmosmean_tmax_f",
            "nam_tmax_f",
            "hrrr_tmax_f",
            "rap_tmax_f",
            "ens_std",
            "ens_range",
            "sin_doy",
            "cos_doy",
            "month"
          ],
          "fit_on": "train",
          "intercept": [
            0.15420200985457364,
            0.4921769602908641,
            0.01787068852392743,
            -0.030246524992933163,
            -0.10908852777841538,
            -0.5249146058979969
          ],
          "name": "best_model_classifier",
          "scaler_mean": [
            63.22669649011956,
            63.84395424836601,
            63.93225178197049,
            63.30555555555556,
            63.549676836799215,
            62.38739490284644,
            1.9724350522718126,
            5.725624766265416,
            0.076249519026095,
            -0.03830601337952942,
            6.2973856209150325
          ],
          "scaler_scale": [
            16.744913671392055,
            16.43538534849312,
            17.715798139763308,
            16.16064478057846,
            18.04587973742932,
            17.794629792899077,
            0.8943820168637893,
            2.6020079527995814,
            0.7113279903785942,
            0.6976612002191372,
            3.3521224904776066
          ]
        }
      ]
    },
    "description": "Best-model probability classifier",
    "experiment_id": "E98",
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
      "p_best_nbm_tmax_f",
      "p_best_gfs_tmax_f",
      "p_best_gefsatmosmean_tmax_f",
      "p_best_nam_tmax_f",
      "p_best_hrrr_tmax_f",
      "p_best_rap_tmax_f",
      "fcst_expected_bestprob",
      "bestprob_entropy"
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