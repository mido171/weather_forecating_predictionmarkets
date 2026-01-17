# Training Report

## Dataset Summary
```json
{
  "date_coverage": {
    "max": "2025-12-31",
    "min": "2007-01-01"
  },
  "missing_by_column": {
    "actual_tmax_f": 0,
    "asof_utc": 0,
    "gfs_cig_max": 4,
    "gfs_cig_mean": 4,
    "gfs_cig_median": 4,
    "gfs_cig_min": 4,
    "gfs_dpt_max": 4,
    "gfs_dpt_mean": 4,
    "gfs_dpt_median": 4,
    "gfs_dpt_min": 4,
    "gfs_n_x_max": 4,
    "gfs_n_x_mean": 4,
    "gfs_n_x_median": 4,
    "gfs_n_x_min": 4,
    "gfs_p06_max": 4,
    "gfs_p06_mean": 4,
    "gfs_p06_median": 4,
    "gfs_p06_min": 4,
    "gfs_p12_max": 4,
    "gfs_p12_mean": 4,
    "gfs_p12_median": 4,
    "gfs_p12_min": 4,
    "gfs_pos_max": 2629,
    "gfs_pos_mean": 2629,
    "gfs_pos_median": 2629,
    "gfs_pos_min": 2629,
    "gfs_poz_max": 2629,
    "gfs_poz_mean": 2629,
    "gfs_poz_median": 2629,
    "gfs_poz_min": 2629,
    "gfs_q06_max": 4,
    "gfs_q06_mean": 4,
    "gfs_q06_median": 4,
    "gfs_q06_min": 4,
    "gfs_q12_max": 4,
    "gfs_q12_mean": 4,
    "gfs_q12_median": 4,
    "gfs_q12_min": 4,
    "gfs_snw_max": 6921,
    "gfs_snw_mean": 6921,
    "gfs_snw_median": 6921,
    "gfs_snw_min": 6921,
    "gfs_t06_1_max": 4,
    "gfs_t06_1_mean": 4,
    "gfs_t06_1_median": 4,
    "gfs_t06_1_min": 4,
    "gfs_t06_2_max": 4,
    "gfs_t06_2_mean": 4,
    "gfs_t06_2_median": 4,
    "gfs_t06_2_min": 4,
    "gfs_t06_max": 4,
    "gfs_t06_mean": 4,
    "gfs_t06_median": 4,
    "gfs_t06_min": 4,
    "gfs_tmp_max": 4,
    "gfs_tmp_mean": 4,
    "gfs_tmp_median": 4,
    "gfs_tmp_min": 4,
    "gfs_vis_max": 4,
    "gfs_vis_mean": 4,
    "gfs_vis_median": 4,
    "gfs_vis_min": 4,
    "gfs_wdr_max": 4,
    "gfs_wdr_mean": 4,
    "gfs_wdr_median": 4,
    "gfs_wdr_min": 4,
    "gfs_wsp_max": 4,
    "gfs_wsp_mean": 4,
    "gfs_wsp_median": 4,
    "gfs_wsp_min": 4,
    "nam_cig_max": 5,
    "nam_cig_mean": 5,
    "nam_cig_median": 5,
    "nam_cig_min": 5,
    "nam_dpt_max": 5,
    "nam_dpt_mean": 5,
    "nam_dpt_median": 5,
    "nam_dpt_min": 5,
    "nam_n_x_max": 5,
    "nam_n_x_mean": 5,
    "nam_n_x_median": 5,
    "nam_n_x_min": 5,
    "nam_p06_max": 5,
    "nam_p06_mean": 5,
    "nam_p06_median": 5,
    "nam_p06_min": 5,
    "nam_p12_max": 5,
    "nam_p12_mean": 5,
    "nam_p12_median": 5,
    "nam_p12_min": 5,
    "nam_pos_max": 4214,
    "nam_pos_mean": 4214,
    "nam_pos_median": 4214,
    "nam_pos_min": 4214,
    "nam_poz_max": 4214,
    "nam_poz_mean": 4214,
    "nam_poz_median": 4214,
    "nam_poz_min": 4214,
    "nam_q06_max": 5,
    "nam_q06_mean": 5,
    "nam_q06_median": 5,
    "nam_q06_min": 5,
    "nam_q12_max": 5,
    "nam_q12_mean": 5,
    "nam_q12_median": 5,
    "nam_q12_min": 5,
    "nam_snw_max": 6921,
    "nam_snw_mean": 6921,
    "nam_snw_median": 6921,
    "nam_snw_min": 6921,
    "nam_t06_1_max": 5,
    "nam_t06_1_mean": 5,
    "nam_t06_1_median": 5,
    "nam_t06_1_min": 5,
    "nam_t06_2_max": 5,
    "nam_t06_2_mean": 5,
    "nam_t06_2_median": 5,
    "nam_t06_2_min": 5,
    "nam_t06_max": 5,
    "nam_t06_mean": 5,
    "nam_t06_median": 5,
    "nam_t06_min": 5,
    "nam_tmp_max": 5,
    "nam_tmp_mean": 5,
    "nam_tmp_median": 5,
    "nam_tmp_min": 5,
    "nam_vis_max": 5,
    "nam_vis_mean": 5,
    "nam_vis_median": 5,
    "nam_vis_min": 5,
    "nam_wdr_max": 5,
    "nam_wdr_mean": 5,
    "nam_wdr_median": 5,
    "nam_wdr_min": 5,
    "nam_wsp_max": 5,
    "nam_wsp_mean": 5,
    "nam_wsp_median": 5,
    "nam_wsp_min": 5,
    "station_id": 0,
    "target_date_local": 0
  },
  "row_count": 6921,
  "split_counts": {
    "test": 725,
    "train": 5833,
    "validation": 363
  },
  "station_counts": {
    "KMIA": 6921
  }
}
```

## Model Summary
```json
{
  "best_params": {
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "lambda_l1": 0.0,
    "lambda_l2": 0.1,
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
        "lambda_l1": 0.0,
        "lambda_l2": 0.1,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "n_estimators": 300,
        "num_leaves": 31
      },
      "cv_score": -1.4263255751792225,
      "model": "lgbm",
      "val_mae": 1.410066498133179
    }
  ],
  "selected_model": "lgbm",
  "sigma_method": "none",
  "sigma_model": null
}
```

## Metrics Summary
```json
{
  "baseline": {
    "climatology_test": {
      "bias": -0.09384381338742387,
      "mae": 2.6899391480730226,
      "p50_abs_error": 2.117647058823536,
      "p90_abs_error": 5.764705882352942,
      "p95_abs_error": 7.17647058823529,
      "rmse": 3.6103322015400012
    },
    "climatology_val": {
      "bias": -1.9515495867768595,
      "mae": 3.4897727272727277,
      "p50_abs_error": 2.9375,
      "p90_abs_error": 6.875,
      "p95_abs_error": 8.856249999999996,
      "rmse": 4.340931098671449
    }
  },
  "per_station_test": {
    "KMIA": {
      "bias": 0.5086191897641623,
      "mae": 1.2198896630598364,
      "p50_abs_error": 0.899872485736168,
      "p90_abs_error": 2.562811105417623,
      "p95_abs_error": 3.5357380526993634,
      "rmse": 1.6842047440717687
    }
  },
  "probabilistic_test": null,
  "probabilistic_validation": {},
  "test": {
    "bias": 0.5086191897641623,
    "mae": 1.2198896630598364,
    "p50_abs_error": 0.899872485736168,
    "p90_abs_error": 2.562811105417623,
    "p95_abs_error": 3.5357380526993634,
    "rmse": 1.6842047440717687
  },
  "train": {
    "bias": 1.0643871705232417e-05,
    "mae": 0.7186710205516755,
    "p50_abs_error": 0.5853271437455447,
    "p90_abs_error": 1.5128730020564376,
    "p95_abs_error": 1.8414629558725824,
    "rmse": 0.9316091779971938
  },
  "validation": {
    "bias": -0.6216586515262549,
    "mae": 1.410066498133179,
    "p50_abs_error": 1.2532198078683052,
    "p90_abs_error": 2.73396594040116,
    "p95_abs_error": 3.2796327139241943,
    "rmse": 1.7283247281532004
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "nam_wdr_mean",
      "importance": 247.0
    },
    {
      "feature": "gfs_wdr_median",
      "importance": 232.0
    },
    {
      "feature": "nam_wdr_median",
      "importance": 232.0
    },
    {
      "feature": "day_of_year",
      "importance": 217.0
    },
    {
      "feature": "sin_doy",
      "importance": 214.0
    },
    {
      "feature": "gfs_wdr_mean",
      "importance": 211.0
    },
    {
      "feature": "cos_doy",
      "importance": 196.0
    },
    {
      "feature": "gfs_cig_mean",
      "importance": 177.0
    },
    {
      "feature": "gfs_n_x_max",
      "importance": 177.0
    },
    {
      "feature": "nam_wsp_mean",
      "importance": 173.0
    },
    {
      "feature": "nam_cig_mean",
      "importance": 168.0
    },
    {
      "feature": "nam_wdr_min",
      "importance": 157.0
    },
    {
      "feature": "gfs_wdr_max",
      "importance": 154.0
    },
    {
      "feature": "gfs_tmp_max",
      "importance": 151.0
    },
    {
      "feature": "nam_dpt_mean",
      "importance": 151.0
    },
    {
      "feature": "gfs_p12_max",
      "importance": 146.0
    },
    {
      "feature": "nam_tmp_max",
      "importance": 145.0
    },
    {
      "feature": "gfs_dpt_mean",
      "importance": 141.0
    },
    {
      "feature": "gfs_p06_median",
      "importance": 140.0
    },
    {
      "feature": "nam_n_x_max",
      "importance": 131.0
    },
    {
      "feature": "gfs_p12_min",
      "importance": 129.0
    },
    {
      "feature": "gfs_p06_max",
      "importance": 126.0
    },
    {
      "feature": "nam_wdr_max",
      "importance": 126.0
    },
    {
      "feature": "gfs_tmp_mean",
      "importance": 122.0
    },
    {
      "feature": "nam_t06_2_mean",
      "importance": 122.0
    },
    {
      "feature": "gfs_wdr_min",
      "importance": 119.0
    },
    {
      "feature": "gfs_wsp_mean",
      "importance": 116.0
    },
    {
      "feature": "nam_tmp_mean",
      "importance": 116.0
    },
    {
      "feature": "nam_p06_min",
      "importance": 115.0
    },
    {
      "feature": "gfs_t06_max",
      "importance": 114.0
    },
    {
      "feature": "gfs_t06_median",
      "importance": 113.0
    },
    {
      "feature": "nam_t06_max",
      "importance": 112.0
    },
    {
      "feature": "nam_p06_median",
      "importance": 111.0
    },
    {
      "feature": "nam_p12_max",
      "importance": 110.0
    },
    {
      "feature": "nam_p12_min",
      "importance": 106.0
    },
    {
      "feature": "gfs_p06_min",
      "importance": 105.0
    },
    {
      "feature": "nam_t06_2_max",
      "importance": 105.0
    },
    {
      "feature": "gfs_t06_mean",
      "importance": 103.0
    },
    {
      "feature": "gfs_p12_mean",
      "importance": 102.0
    },
    {
      "feature": "nam_t06_median",
      "importance": 102.0
    },
    {
      "feature": "nam_p12_mean",
      "importance": 101.0
    },
    {
      "feature": "nam_dpt_median",
      "importance": 92.0
    },
    {
      "feature": "gfs_dpt_median",
      "importance": 91.0
    },
    {
      "feature": "nam_p06_max",
      "importance": 91.0
    },
    {
      "feature": "gfs_dpt_max",
      "importance": 89.0
    },
    {
      "feature": "nam_p06_mean",
      "importance": 84.0
    },
    {
      "feature": "gfs_t06_2_max",
      "importance": 82.0
    },
    {
      "feature": "gfs_p06_mean",
      "importance": 81.0
    },
    {
      "feature": "gfs_dpt_min",
      "importance": 80.0
    },
    {
      "feature": "gfs_tmp_median",
      "importance": 80.0
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
    "csv_path": "C:\\Users\\ahmad\\Desktop\\generalFiles\\git\\weather-forecasting-predictionmarkets\\weather_forecating_predictionmarkets\\ingestion-service\\src\\main\\resources\\trainingdata_output\\KMIA_mos_training_data.csv",
    "dataset_schema_version": 1
  },
  "distribution": {
    "support_max_f": 130,
    "support_min_f": -30
  },
  "features": {
    "base_features": [
      "gfs_cig_min",
      "gfs_cig_max",
      "gfs_cig_mean",
      "gfs_cig_median",
      "gfs_dpt_min",
      "gfs_dpt_max",
      "gfs_dpt_mean",
      "gfs_dpt_median",
      "gfs_n_x_min",
      "gfs_n_x_max",
      "gfs_n_x_mean",
      "gfs_n_x_median",
      "gfs_p06_min",
      "gfs_p06_max",
      "gfs_p06_mean",
      "gfs_p06_median",
      "gfs_p12_min",
      "gfs_p12_max",
      "gfs_p12_mean",
      "gfs_p12_median",
      "gfs_pos_min",
      "gfs_pos_max",
      "gfs_pos_mean",
      "gfs_pos_median",
      "gfs_poz_min",
      "gfs_poz_max",
      "gfs_poz_mean",
      "gfs_poz_median",
      "gfs_q06_min",
      "gfs_q06_max",
      "gfs_q06_mean",
      "gfs_q06_median",
      "gfs_q12_min",
      "gfs_q12_max",
      "gfs_q12_mean",
      "gfs_q12_median",
      "gfs_t06_min",
      "gfs_t06_max",
      "gfs_t06_mean",
      "gfs_t06_median",
      "gfs_t06_1_min",
      "gfs_t06_1_max",
      "gfs_t06_1_mean",
      "gfs_t06_1_median",
      "gfs_t06_2_min",
      "gfs_t06_2_max",
      "gfs_t06_2_mean",
      "gfs_t06_2_median",
      "gfs_tmp_min",
      "gfs_tmp_max",
      "gfs_tmp_mean",
      "gfs_tmp_median",
      "gfs_vis_min",
      "gfs_vis_max",
      "gfs_vis_mean",
      "gfs_vis_median",
      "gfs_wdr_min",
      "gfs_wdr_max",
      "gfs_wdr_mean",
      "gfs_wdr_median",
      "gfs_wsp_min",
      "gfs_wsp_max",
      "gfs_wsp_mean",
      "gfs_wsp_median",
      "nam_cig_min",
      "nam_cig_max",
      "nam_cig_mean",
      "nam_cig_median",
      "nam_dpt_min",
      "nam_dpt_max",
      "nam_dpt_mean",
      "nam_dpt_median",
      "nam_n_x_min",
      "nam_n_x_max",
      "nam_n_x_mean",
      "nam_n_x_median",
      "nam_p06_min",
      "nam_p06_max",
      "nam_p06_mean",
      "nam_p06_median",
      "nam_p12_min",
      "nam_p12_max",
      "nam_p12_mean",
      "nam_p12_median",
      "nam_pos_min",
      "nam_pos_max",
      "nam_pos_mean",
      "nam_pos_median",
      "nam_poz_min",
      "nam_poz_max",
      "nam_poz_mean",
      "nam_poz_median",
      "nam_q06_min",
      "nam_q06_max",
      "nam_q06_mean",
      "nam_q06_median",
      "nam_q12_min",
      "nam_q12_max",
      "nam_q12_mean",
      "nam_q12_median",
      "nam_t06_min",
      "nam_t06_max",
      "nam_t06_mean",
      "nam_t06_median",
      "nam_t06_1_min",
      "nam_t06_1_max",
      "nam_t06_1_mean",
      "nam_t06_1_median",
      "nam_t06_2_min",
      "nam_t06_2_max",
      "nam_t06_2_mean",
      "nam_t06_2_median",
      "nam_tmp_min",
      "nam_tmp_max",
      "nam_tmp_mean",
      "nam_tmp_median",
      "nam_vis_min",
      "nam_vis_max",
      "nam_vis_mean",
      "nam_vis_median",
      "nam_wdr_min",
      "nam_wdr_max",
      "nam_wdr_mean",
      "nam_wdr_median",
      "nam_wsp_min",
      "nam_wsp_max",
      "nam_wsp_mean",
      "nam_wsp_median"
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
    "ensemble_stats": false,
    "model_vs_ens_deltas": false,
    "pairwise_deltas": false,
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
      "method": "none",
      "param_grid": {},
      "primary": "ridge",
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
      "n_splits": 3
    },
    "gap_dates": [],
    "test_end": "2025-12-31",
    "test_start": "2024-01-01",
    "train_end": "2022-12-31",
    "train_start": "2007-01-01",
    "validation": {
      "enabled": true,
      "val_end": "2023-12-31",
      "val_start": "2023-01-01"
    }
  },
  "validation": {
    "forecast_max_f": 140.0,
    "forecast_min_f": -80.0,
    "require_asof_not_after_target": true,
    "spread_min_f": 0.0,
    "strict_schema": false
  }
}
```