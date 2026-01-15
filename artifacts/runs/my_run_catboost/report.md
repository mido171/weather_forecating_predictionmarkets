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
    "depth": 6,
    "iterations": 500,
    "l2_leaf_reg": 3,
    "learning_rate": 0.05
  },
  "candidates": [
    {
      "best_params": {
        "depth": 6,
        "iterations": 500,
        "l2_leaf_reg": 3,
        "learning_rate": 0.05
      },
      "cv_score": -2.564161510830547,
      "model": "catboost",
      "val_mae": 1.7021685850925736
    }
  ],
  "selected_model": "catboost",
  "sigma_model": "catboost"
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
      "bias": 0.4728416025368997,
      "mae": 1.841200972329406,
      "p50_abs_error": 1.5553961954437057,
      "p90_abs_error": 3.825500791984842,
      "p95_abs_error": 4.662870264860277,
      "rmse": 2.404506755089886
    }
  },
  "probabilistic_test": {
    "brier_scores": {
      "ge_90": 0.03220831792788387,
      "lt_52": 0.038303085732363885
    },
    "log_loss": 2.849641535267277
  },
  "probabilistic_validation": {
    "brier_scores": {
      "ge_90": 0.027463932812445496,
      "lt_52": 0.029161302314039295
    },
    "log_loss": 2.609245633804944
  },
  "test": {
    "bias": 0.4728416025368997,
    "mae": 1.841200972329406,
    "p50_abs_error": 1.5553961954437057,
    "p90_abs_error": 3.825500791984842,
    "p95_abs_error": 4.662870264860277,
    "rmse": 2.404506755089886
  },
  "train": {
    "bias": 0.0001917834742898545,
    "mae": 0.9061612831028577,
    "p50_abs_error": 0.7416343412575195,
    "p90_abs_error": 1.888482302779589,
    "p95_abs_error": 2.2303948432997514,
    "rmse": 1.1499192986569957
  },
  "validation": {
    "bias": -0.13841587525944693,
    "mae": 1.7021685850925736,
    "p50_abs_error": 1.3638148776344075,
    "p90_abs_error": 3.5038800968326824,
    "p95_abs_error": 4.638663092137894,
    "rmse": 2.223260346572769
  }
}
```

## Feature Importance
```json
{
  "top_features": [
    {
      "feature": "nbm_tmax_f",
      "importance": 16.51012402176679
    },
    {
      "feature": "ens_median",
      "importance": 12.975254188743765
    },
    {
      "feature": "rap_tmax_f",
      "importance": 12.720157076931974
    },
    {
      "feature": "ens_mean",
      "importance": 10.482567840222224
    },
    {
      "feature": "ens_max",
      "importance": 8.815761708520125
    },
    {
      "feature": "ens_min",
      "importance": 8.387314272281202
    },
    {
      "feature": "gfs_tmax_f",
      "importance": 6.15292537351095
    },
    {
      "feature": "gefsatmosmean_tmax_f",
      "importance": 5.256952545465496
    },
    {
      "feature": "nam_tmax_f",
      "importance": 4.193910237489318
    },
    {
      "feature": "hrrr_tmax_f",
      "importance": 3.702772340222173
    },
    {
      "feature": "cos_doy",
      "importance": 0.931608591819087
    },
    {
      "feature": "hrrr_tmax_f_minus_ens_mean",
      "importance": 0.8000759361870228
    },
    {
      "feature": "sin_doy",
      "importance": 0.682992832566196
    },
    {
      "feature": "nbm_tmax_f_minus_ens_mean",
      "importance": 0.6127331581996222
    },
    {
      "feature": "gefsatmos_tmp_spread_f",
      "importance": 0.5949504025197938
    },
    {
      "feature": "gfs_tmax_f_minus_ens_mean",
      "importance": 0.5352785956886502
    },
    {
      "feature": "nam_tmax_f_minus_ens_mean",
      "importance": 0.51813506374531
    },
    {
      "feature": "rap_tmax_f_minus_nbm_tmax_f",
      "importance": 0.47433125280834226
    },
    {
      "feature": "day_of_year",
      "importance": 0.46576139746240897
    },
    {
      "feature": "ens_range",
      "importance": 0.4512985246189894
    },
    {
      "feature": "nbm_tmax_f_minus_ens_mean_abs",
      "importance": 0.44558482422447415
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_ens_mean_abs",
      "importance": 0.3884734120498153
    },
    {
      "feature": "hrrr_tmax_f_minus_ens_mean_abs",
      "importance": 0.37355087995114505
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_gfs_tmax_f",
      "importance": 0.35625890639127955
    },
    {
      "feature": "rap_tmax_f_minus_ens_mean_abs",
      "importance": 0.3411103768274238
    },
    {
      "feature": "rap_tmax_f_minus_ens_mean",
      "importance": 0.34018873758966633
    },
    {
      "feature": "gfs_tmax_f_minus_ens_mean_abs",
      "importance": 0.32668948801534625
    },
    {
      "feature": "ens_iqr",
      "importance": 0.32158997843632786
    },
    {
      "feature": "hrrr_tmax_f_minus_gfs_tmax_f",
      "importance": 0.2709921487769103
    },
    {
      "feature": "ens_std",
      "importance": 0.2700603328565811
    },
    {
      "feature": "nbm_tmax_f_minus_hrrr_tmax_f",
      "importance": 0.254083756257416
    },
    {
      "feature": "month",
      "importance": 0.23018966466595384
    },
    {
      "feature": "gefsatmosmean_tmax_f_minus_ens_mean",
      "importance": 0.2052714179753028
    },
    {
      "feature": "nam_tmax_f_minus_ens_mean_abs",
      "importance": 0.20463252534562087
    },
    {
      "feature": "nam_tmax_f_minus_nbm_tmax_f",
      "importance": 0.18688588127816513
    },
    {
      "feature": "gfs_tmax_f_minus_nam_tmax_f",
      "importance": 0.11480710141669817
    },
    {
      "feature": "is_weekend",
      "importance": 0.10472520717246205
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
        }
      },
      "primary": "catboost"
    },
    "sigma": {
      "eps": 1e-06,
      "method": "two_stage",
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
        }
      },
      "primary": "catboost",
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