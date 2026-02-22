"""Minimal feature set builder for KMIA next-day Tmax experiments.

This is intentionally small (20-30ish features) and is derived from the existing
KNN v0 neighbor slots + GFS/NAM base forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_minimal_feature_frame(df: pd.DataFrame, *, top_k: int = 30) -> pd.DataFrame:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    required_scalar = ["base_tmax_gfs", "base_tmax_nam", "base_tmax_blend"]
    missing_scalar = [c for c in required_scalar if c not in df.columns]
    if missing_scalar:
        raise ValueError(f"Missing required columns: {missing_scalar}")

    def slot_cols(suffix: str) -> list[str]:
        return [f"knn_v0_nn{i}_{suffix}" for i in range(1, top_k + 1)]

    required_slots = {
        "y": slot_cols("y_actual_tmax_nextday"),
        "dist": slot_cols("dist"),
        "age": slot_cols("age_days"),
        "resid": slot_cols("residual"),
        "w": slot_cols("weight_norm"),
        "same_month": slot_cols("same_month"),
        "same_doy45": slot_cols("same_doy_window_45"),
        "nbr_base": slot_cols("base_tmax_blend"),
    }
    missing_slots = [c for cols in required_slots.values() for c in cols if c not in df.columns]
    if missing_slots:
        raise ValueError(f"Missing required knn_v0 slot columns (top_k={top_k}): {missing_slots[:10]}")

    y = df[required_slots["y"]].to_numpy(dtype=float)
    dist = df[required_slots["dist"]].to_numpy(dtype=float)
    age = df[required_slots["age"]].to_numpy(dtype=float)
    resid = df[required_slots["resid"]].to_numpy(dtype=float)
    w = df[required_slots["w"]].to_numpy(dtype=float)
    same_month = df[required_slots["same_month"]].to_numpy(dtype=float)
    same_doy45 = df[required_slots["same_doy45"]].to_numpy(dtype=float)
    nbr_base = df[required_slots["nbr_base"]].to_numpy(dtype=float)

    base_now = pd.to_numeric(df["base_tmax_blend"], errors="coerce").to_numpy(dtype=float)

    out: dict[str, np.ndarray] = {}

    # (1-3) Top-30 neighbor max-temp (actual next-day) stats.
    out["knn30_neighbor_tmax_mean"] = np.nanmean(y, axis=1)
    out["knn30_neighbor_tmax_median"] = np.nanmedian(y, axis=1)
    out["knn30_neighbor_tmax_std"] = np.nanstd(y, axis=1)

    # (5) Main KNN-top30 metrics (kept compact).
    out["knn30_dist_mean"] = np.nanmean(dist, axis=1)
    out["knn30_dist_median"] = np.nanmedian(dist, axis=1)
    out["knn30_dist_std"] = np.nanstd(dist, axis=1)
    out["knn30_dist_sq_mean"] = np.nanmean(dist**2, axis=1)

    out["knn30_age_days_mean"] = np.nanmean(age, axis=1)
    out["knn30_age_days_median"] = np.nanmedian(age, axis=1)
    out["knn30_age_days_std"] = np.nanstd(age, axis=1)

    out["knn30_residual_mean"] = np.nanmean(resid, axis=1)
    out["knn30_residual_median"] = np.nanmedian(resid, axis=1)
    out["knn30_residual_std"] = np.nanstd(resid, axis=1)
    out["knn30_abs_residual_mean"] = np.nanmean(np.abs(resid), axis=1)

    out["knn30_neighbor_base_mean"] = np.nanmean(nbr_base, axis=1)
    base_diff = base_now[:, None] - nbr_base
    out["knn30_base_diff_mean"] = np.nanmean(base_diff, axis=1)

    # Sum of the existing v0 weights over the first 30 neighbors (proxy for concentration).
    out["knn30_weight_top30_sum"] = np.nansum(w, axis=1)
    out["knn30_frac_same_month"] = np.nanmean(same_month, axis=1)
    out["knn30_frac_same_doy45"] = np.nanmean(same_doy45, axis=1)

    # (4) Robustness score summary across views.
    rob_cols = [c for c in df.columns if c.startswith("knn_") and c.endswith("_robustness")]
    rob_vals = None
    if rob_cols:
        rob = df[sorted(rob_cols)].to_numpy(dtype=float)
        rob_vals = rob
    if rob_vals is None or rob_vals.size == 0:
        out["knn_robustness_mean"] = np.full(len(df), np.nan, dtype=float)
        out["knn_robustness_median"] = np.full(len(df), np.nan, dtype=float)
        out["knn_robustness_std"] = np.full(len(df), np.nan, dtype=float)
    else:
        out["knn_robustness_mean"] = np.nanmean(rob_vals, axis=1)
        out["knn_robustness_median"] = np.nanmedian(rob_vals, axis=1)
        out["knn_robustness_std"] = np.nanstd(rob_vals, axis=1)

    # (6) GFS/NAM summary stats.
    gfs = pd.to_numeric(df["base_tmax_gfs"], errors="coerce").to_numpy(dtype=float)
    nam = pd.to_numeric(df["base_tmax_nam"], errors="coerce").to_numpy(dtype=float)
    pair = np.vstack([gfs, nam])
    out["gfs_nam_mean"] = np.nanmean(pair, axis=0)
    out["gfs_nam_median"] = np.nanmedian(pair, axis=0)
    out["gfs_nam_std"] = np.nanstd(pair, axis=0)

    return pd.DataFrame(out)

