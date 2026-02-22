"""Derived/second-order features from KNN outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_knn_consistency_zscores(
    df: pd.DataFrame,
    *,
    view_names: list[str],
    consistency_features: list[str],
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Add per-view z-scores comparing current row to its neighbor cloud."""

    df = df.copy()
    for view in view_names:
        z_cols: list[str] = []
        for feature in consistency_features:
            if feature not in df.columns:
                continue
            mean_col = f"knn_{view}_nbr_mean_{feature}"
            std_col = f"knn_{view}_nbr_std_{feature}"
            if mean_col not in df.columns or std_col not in df.columns:
                continue
            z_col = f"knn_{view}_nbr_z_{feature}"
            df[z_col] = (df[feature] - df[mean_col]) / (df[std_col] + eps)
            z_cols.append(z_col)
        if z_cols:
            df[f"knn_{view}_nbr_z_abs_mean"] = df[z_cols].abs().mean(axis=1)
            df[f"knn_{view}_nbr_z_abs_max"] = df[z_cols].abs().max(axis=1)
    return df


def add_knn_cross_view_features(
    df: pd.DataFrame,
    *,
    view_names: list[str],
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Add consensus/disagreement features across KNN views."""

    df = df.copy()
    view_names = [v for v in view_names if f"knn_{v}_analog_mu" in df.columns]
    if not view_names:
        return df

    mu_cols = [f"knn_{v}_analog_mu" for v in view_names]
    mu = df[mu_cols]
    df["knn_views_analog_mu_mean"] = mu.mean(axis=1)
    df["knn_views_analog_mu_median"] = mu.median(axis=1)
    df["knn_views_analog_mu_std"] = mu.std(axis=1)
    df["knn_views_analog_mu_range"] = mu.max(axis=1) - mu.min(axis=1)

    std_cols = [f"knn_{v}_analog_std" for v in view_names if f"knn_{v}_analog_std" in df.columns]
    if std_cols:
        analog_std_mean = df[std_cols].mean(axis=1)
        df["knn_views_analog_std_mean"] = analog_std_mean
        df["knn_views_disagreement"] = df["knn_views_analog_mu_std"] / (analog_std_mean + eps)

    dist_cols = [f"knn_{v}_dist_p50" for v in view_names if f"knn_{v}_dist_p50" in df.columns]
    if dist_cols:
        dist = df[dist_cols]
        df["knn_views_dist_p50_mean"] = dist.mean(axis=1)
        df["knn_views_dist_p50_std"] = dist.std(axis=1)
        df["knn_views_dist_p50_range"] = dist.max(axis=1) - dist.min(axis=1)

    rob_cols = [f"knn_{v}_robustness" for v in view_names if f"knn_{v}_robustness" in df.columns]
    if rob_cols:
        rob = df[rob_cols]
        df["knn_views_robustness_mean"] = rob.mean(axis=1)
        df["knn_views_robustness_std"] = rob.std(axis=1)
        df["knn_views_robustness_range"] = rob.max(axis=1) - rob.min(axis=1)

    # Weighted cross-view blend of analog_mu (downweight low-ESS / far views).
    ess_cols = [f"knn_{v}_ess_ratio" for v in view_names if f"knn_{v}_ess_ratio" in df.columns]
    dist_p50_cols = [f"knn_{v}_dist_p50" for v in view_names if f"knn_{v}_dist_p50" in df.columns]
    if len(ess_cols) == len(view_names) and len(dist_p50_cols) == len(view_names):
        mu_arr = mu.to_numpy(dtype=float)
        ess_arr = df[ess_cols].to_numpy(dtype=float)
        dist_arr = df[dist_p50_cols].to_numpy(dtype=float)
        w = ess_arr / (dist_arr + eps)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        mu_mask = np.isfinite(mu_arr)
        w = w * mu_mask
        w_sum = np.sum(w, axis=1)
        weighted_mu = np.full(len(df), np.nan, dtype=float)
        ok = w_sum > 0
        weighted_mu[ok] = np.sum(w[ok] * mu_arr[ok], axis=1) / w_sum[ok]
        df["knn_views_analog_mu_weighted"] = weighted_mu
        df["knn_views_analog_mu_weight_sum"] = w_sum

    return df

