"""Bias/forecast-error feature blocks."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .mos_config import MosDatasetConfig
from .mos_utils import rolling_linear_regression


def compute_bias_features(df: pd.DataFrame, cfg: MosDatasetConfig) -> pd.DataFrame:
    df = df.copy().sort_values("target_date_local")
    y = df["y_actual_tmax_f"].astype(float)
    forecasts = {
        "gfs": df.get("base_tmax_gfs"),
        "nam": df.get("base_tmax_nam"),
        "blend": df.get("base_tmax_blend"),
        # Treat KNN analog means as forecasts too; useful in regime shifts.
        "knn_v0": df.get("knn_v0_analog_mu"),
        "knn_views": df.get("knn_views_analog_mu_weighted"),
    }
    out = pd.DataFrame({"target_date_local": df["target_date_local"].values})

    rmse_store: dict[int, dict[str, pd.Series]] = defaultdict(dict)
    # If observed truth is only available through (asof_date_local - obs_cutoff_lag_days),
    # then for a target date D (built from asof_date_local=D-1) the newest known truth
    # is D-1-obs_cutoff_lag_days. That implies we must shift by (1 + obs_cutoff_lag_days)
    # when building leakage-safe error history features.
    shift_days = 1 + int(cfg.obs_cutoff_lag_days)

    for name, f_series in forecasts.items():
        if f_series is None:
            continue
        f = f_series.astype(float)
        err = y - f
        err_abs = err.abs()
        err_sq = err**2
        for window in cfg.bias_windows_days or []:
            mean = err.rolling(window=window, min_periods=2).mean().shift(shift_days)
            mae = err_abs.rolling(window=window, min_periods=2).mean().shift(shift_days)
            rmse = np.sqrt(err_sq.rolling(window=window, min_periods=2).mean()).shift(shift_days)
            p90 = err_abs.rolling(window=window, min_periods=2).quantile(0.90).shift(shift_days)
            p95 = err_abs.rolling(window=window, min_periods=2).quantile(0.95).shift(shift_days)

            out[f"bias_{name}_mean_{window}"] = mean
            out[f"bias_{name}_mae_{window}"] = mae
            out[f"bias_{name}_rmse_{window}"] = rmse
            out[f"bias_{name}_p90_abs_{window}"] = p90
            out[f"bias_{name}_p95_abs_{window}"] = p95

            rmse_store[window][name] = rmse

            a, b, r2 = rolling_linear_regression(f.to_numpy(), y.to_numpy(), window=window)
            a = pd.Series(a).shift(shift_days)
            b = pd.Series(b).shift(shift_days)
            r2 = pd.Series(r2).shift(shift_days)
            out[f"bias_{name}_calib_a_{window}"] = a
            out[f"bias_{name}_calib_b_{window}"] = b
            out[f"bias_{name}_calib_r2_{window}"] = r2
            out[f"bias_{name}_calibrated_pred_{window}"] = a + b * f

    model_order = ["gfs", "nam", "blend"]
    for window in cfg.bias_windows_days or []:
        rmse_cols = [rmse_store.get(window, {}).get(name) for name in model_order]
        if any(col is None for col in rmse_cols):
            continue
        rmse_stack = np.vstack([col.to_numpy() for col in rmse_cols])
        all_nan = np.all(np.isnan(rmse_stack), axis=0)
        rmse_filled = np.where(np.isnan(rmse_stack), np.inf, rmse_stack)
        best_idx = np.argmin(rmse_filled, axis=0).astype(float)
        best_idx[all_nan] = np.nan
        out[f"bias_best_model_rmse_{window}"] = best_idx
        for i, name in enumerate(model_order):
            out[f"bias_best_is_{name}_{window}"] = (best_idx == i).astype(float)
        ranks = np.argsort(rmse_filled, axis=0)
        rank_matrix = np.empty_like(ranks, dtype=float)
        for r in range(ranks.shape[0]):
            rank_matrix[ranks[r, :], np.arange(ranks.shape[1])] = r + 1
        rank_matrix[:, all_nan] = np.nan
        for i, name in enumerate(model_order):
            out[f"bias_rank_{name}_{window}"] = rank_matrix[i, :]
    return out
