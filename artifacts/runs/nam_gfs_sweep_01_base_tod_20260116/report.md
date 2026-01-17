# Training Report

## Dataset Summary
```json
{
  "date_coverage": {
    "max": "2025-12-31",
    "min": "2012-01-11"
  },
  "missing_by_column": {
    "actual_tmax_f": 0,
    "asof_utc": 0,
    "gfs_cig_max": 3,
    "gfs_cig_mean": 3,
    "gfs_cig_median": 3,
    "gfs_cig_min": 3,
    "gfs_dpt_max": 3,
    "gfs_dpt_mean": 3,
    "gfs_dpt_median": 3,
    "gfs_dpt_min": 3,
    "gfs_n_x_max": 3,
    "gfs_n_x_mean": 3,
    "gfs_n_x_median": 3,
    "gfs_n_x_min": 3,
    "gfs_p06_max": 3,
    "gfs_p06_mean": 3,
    "gfs_p06_median": 3,
    "gfs_p06_min": 3,
    "gfs_p12_max": 3,
    "gfs_p12_mean": 3,
    "gfs_p12_median": 3,
    "gfs_p12_min": 3,
    "gfs_pos_max": 1290,
    "gfs_pos_mean": 1290,
    "gfs_pos_median": 1290,
    "gfs_pos_min": 1290,
    "gfs_poz_max": 1290,
    "gfs_poz_mean": 1290,
    "gfs_poz_median": 1290,
    "gfs_poz_min": 1290,
    "gfs_q06_max": 3,
    "gfs_q06_mean": 3,
    "gfs_q06_median": 3,
    "gfs_q06_min": 3,
    "gfs_q12_max": 3,
    "gfs_q12_mean": 3,
    "gfs_q12_median": 3,
    "gfs_q12_min": 3,
    "gfs_snw_max": 1290,
    "gfs_snw_mean": 1290,
    "gfs_snw_median": 1290,
    "gfs_snw_min": 1290,
    "gfs_t06_1_max": 3,
    "gfs_t06_1_mean": 3,
    "gfs_t06_1_median": 3,
    "gfs_t06_1_min": 3,
    "gfs_t06_2_max": 3,
    "gfs_t06_2_mean": 3,
    "gfs_t06_2_median": 3,
    "gfs_t06_2_min": 3,
    "gfs_t06_max": 3,
    "gfs_t06_mean": 3,
    "gfs_t06_median": 3,
    "gfs_t06_min": 3,
    "gfs_tmp_max": 3,
    "gfs_tmp_mean": 3,
    "gfs_tmp_median": 3,
    "gfs_tmp_min": 3,
    "gfs_vis_max": 3,
    "gfs_vis_mean": 3,
    "gfs_vis_median": 3,
    "gfs_vis_min": 3,
    "gfs_wdr_max": 3,
    "gfs_wdr_mean": 3,
    "gfs_wdr_median": 3,
    "gfs_wdr_min": 3,
    "gfs_wsp_max": 3,
    "gfs_wsp_mean": 3,
    "gfs_wsp_median": 3,
    "gfs_wsp_min": 3,
    "nam_cig_max": 4,
    "nam_cig_mean": 4,
    "nam_cig_median": 4,
    "nam_cig_min": 4,
    "nam_dpt_max": 3,
    "nam_dpt_mean": 3,
    "nam_dpt_median": 3,
    "nam_dpt_min": 3,
    "nam_n_x_max": 3,
    "nam_n_x_mean": 3,
    "nam_n_x_median": 3,
    "nam_n_x_min": 3,
    "nam_p06_max": 3,
    "nam_p06_mean": 3,
    "nam_p06_median": 3,
    "nam_p06_min": 3,
    "nam_p12_max": 3,
    "nam_p12_mean": 3,
    "nam_p12_median": 3,
    "nam_p12_min": 3,
    "nam_pos_max": 2383,
    "nam_pos_mean": 2383,
    "nam_pos_median": 2383,
    "nam_pos_min": 2383,
    "nam_poz_max": 2383,
    "nam_poz_mean": 2383,
    "nam_poz_median": 2383,
    "nam_poz_min": 2383,
    "nam_q06_max": 3,
    "nam_q06_mean": 3,
    "nam_q06_median": 3,
    "nam_q06_min": 3,
    "nam_q12_max": 3,
    "nam_q12_mean": 3,
    "nam_q12_median": 3,
    "nam_q12_min": 3,
    "nam_snw_max": 1288,
    "nam_snw_mean": 1288,
    "nam_snw_median": 1288,
    "nam_snw_min": 1288,
    "nam_t06_1_max": 3,
    "nam_t06_1_mean": 3,
    "nam_t06_1_median": 3,
    "nam_t06_1_min": 3,
    "nam_t06_2_max": 3,
    "nam_t06_2_mean": 3,
    "nam_t06_2_median": 3,
    "nam_t06_2_min": 3,
    "nam_t06_max": 3,
    "nam_t06_mean": 3,
    "nam_t06_median": 3,
    "nam_t06_min": 3,
    "nam_tmp_max": 3,
    "nam_tmp_mean": 3,
    "nam_tmp_median": 3,
    "nam_tmp_min": 3,
    "nam_vis_max": 4,
    "nam_vis_mean": 4,
    "nam_vis_median": 4,
    "nam_vis_min": 4,
    "nam_wdr_max": 3,
    "nam_wdr_mean": 3,
    "nam_wdr_median": 3,
    "nam_wdr_min": 3,
    "nam_wsp_max": 3,
    "nam_wsp_mean": 3,
    "nam_wsp_median": 3,
    "nam_wsp_min": 3,
    "station_id": 0,
    "target_date_local": 0
  },
  "row_count": 5099,
  "split_counts": {
    "test": 731,
    "train": 4003,
    "validation": 365
  },
  "station_counts": {
    "KMDW": 5099
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
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "n_estimators": 300,
        "num_leaves": 31
      },
      "cv_score": -2.5912783005508984,
      "model": "lgbm",
      "val_mae": 2.338631747885967
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
      "bias": -2.033453550553414,
      "mae": 8.355345520872197,
      "p50_abs_error": 7.166666666666664,
      "p90_abs_error": 17.333333333333336,
      "p95_abs_error": 21.041666666666668,
      "rmse": 10.605735274322937
    },
    "climatology_val": {
      "bias": -1.7677210460772106,
      "mae": 7.72129514321295,
      "p50_abs_error": 5.9090909090909065,
      "p90_abs_error": 16.436363636363645,
      "p95_abs_error": 20.254545454545458,
      "rmse": 9.90765856180408
    }
  },
  "per_station_test": {
    "KMDW": {
      "bias": -0.193121114047781,
      "mae": 2.3171084843416416,
      "p50_abs_error": 1.8344941859710175,
      "p90_abs_error": 4.933789064381941,
      "p95_abs_error": 5.920705143204994,
      "rmse": 3.1151274702508993
    }
  },
  "probabilistic_test": null,
  "probabilistic_validation": {},
  "test": {
    "bias": -0.193121114047781,
    "mae": 2.3171084843416416,
    "p50_abs_error": 1.8344941859710175,
    "p90_abs_error": 4.933789064381941,
    "p95_abs_error": 5.920705143204994,
    "rmse": 3.1151274702508993
  },
  "train": {
    "bias": -5.2877557330384785e-11,
    "mae": 1.10285670196967,
    "p50_abs_error": 0.8790530036769013,
    "p90_abs_error": 2.331490820994844,
    "p95_abs_error": 2.8799083389788156,
    "rmse": 1.4385187437436433
  },
  "validation": {
    "bias": 0.4619878043246326,
    "mae": 2.338631747885967,
    "p50_abs_error": 1.916953490371391,
    "p90_abs_error": 4.915425879528496,
    "p95_abs_error": 5.973234360290923,
    "rmse": 2.995987557891068
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "gfs_tmp_max",
      "importance": 289.0
    },
    {
      "feature": "sin_doy",
      "importance": 260.0
    },
    {
      "feature": "nam_tmp_max",
      "importance": 246.0
    },
    {
      "feature": "gfs_wdr_median",
      "importance": 197.0
    },
    {
      "feature": "gfs_wdr_mean",
      "importance": 189.0
    },
    {
      "feature": "cos_doy",
      "importance": 183.0
    },
    {
      "feature": "day_of_year",
      "importance": 179.0
    },
    {
      "feature": "nam_wdr_median",
      "importance": 162.0
    },
    {
      "feature": "nam_wdr_mean",
      "importance": 150.0
    },
    {
      "feature": "nam_wdr_max",
      "importance": 148.0
    },
    {
      "feature": "nam_wsp_mean",
      "importance": 147.0
    },
    {
      "feature": "gfs_wdr_min",
      "importance": 145.0
    },
    {
      "feature": "gfs_wsp_mean",
      "importance": 141.0
    },
    {
      "feature": "nam_t06_2_mean",
      "importance": 139.0
    },
    {
      "feature": "gfs_wdr_max",
      "importance": 132.0
    },
    {
      "feature": "nam_cig_mean",
      "importance": 131.0
    },
    {
      "feature": "gfs_n_x_max",
      "importance": 130.0
    },
    {
      "feature": "gfs_dpt_max",
      "importance": 128.0
    },
    {
      "feature": "nam_wdr_min",
      "importance": 123.0
    },
    {
      "feature": "gfs_p06_median",
      "importance": 120.0
    },
    {
      "feature": "nam_dpt_max",
      "importance": 120.0
    },
    {
      "feature": "nam_n_x_max",
      "importance": 120.0
    },
    {
      "feature": "nam_p12_min",
      "importance": 109.0
    },
    {
      "feature": "nam_wsp_min",
      "importance": 109.0
    },
    {
      "feature": "gfs_p06_max",
      "importance": 108.0
    },
    {
      "feature": "nam_wsp_median",
      "importance": 108.0
    },
    {
      "feature": "gfs_t06_2_max",
      "importance": 105.0
    },
    {
      "feature": "nam_p12_mean",
      "importance": 103.0
    },
    {
      "feature": "gfs_p12_max",
      "importance": 102.0
    },
    {
      "feature": "gfs_dpt_mean",
      "importance": 100.0
    },
    {
      "feature": "nam_p06_max",
      "importance": 100.0
    },
    {
      "feature": "nam_p12_max",
      "importance": 100.0
    },
    {
      "feature": "gfs_tmp_mean",
      "importance": 99.0
    },
    {
      "feature": "nam_p06_mean",
      "importance": 99.0
    },
    {
      "feature": "nam_t06_2_median",
      "importance": 97.0
    },
    {
      "feature": "gfs_t06_2_mean",
      "importance": 96.0
    },
    {
      "feature": "gfs_p12_mean",
      "importance": 92.0
    },
    {
      "feature": "gfs_p06_mean",
      "importance": 89.0
    },
    {
      "feature": "gfs_wsp_max",
      "importance": 89.0
    },
    {
      "feature": "nam_cig_min",
      "importance": 89.0
    },
    {
      "feature": "gfs_tmp_median",
      "importance": 88.0
    },
    {
      "feature": "gfs_p12_min",
      "importance": 86.0
    },
    {
      "feature": "gfs_t06_max",
      "importance": 86.0
    },
    {
      "feature": "gfs_wsp_min",
      "importance": 85.0
    },
    {
      "feature": "nam_t06_2_max",
      "importance": 85.0
    },
    {
      "feature": "nam_t06_max",
      "importance": 83.0
    },
    {
      "feature": "gfs_dpt_min",
      "importance": 79.0
    },
    {
      "feature": "gfs_cig_mean",
      "importance": 78.0
    },
    {
      "feature": "gfs_t06_min",
      "importance": 75.0
    },
    {
      "feature": "gfs_t06_mean",
      "importance": 75.0
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
    "csv_path": "C:\\Users\\ahmad\\Desktop\\generalFiles\\git\\weather-forecasting-predictionmarkets\\weather_forecating_predictionmarkets\\ml\\data\\mos\\mos_training_data.csv",
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
      "gfs_snw_min",
      "gfs_snw_max",
      "gfs_snw_mean",
      "gfs_snw_median",
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
      "nam_snw_min",
      "nam_snw_max",
      "nam_snw_mean",
      "nam_snw_median",
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
    "train_start": "2012-01-11",
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
    "strict_schema": true
  }
}
```