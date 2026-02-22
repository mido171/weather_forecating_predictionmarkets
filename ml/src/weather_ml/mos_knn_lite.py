"""Lightweight KNN feature block derived from v0 neighbor slots.

This is meant for "nearest-neighbor focused" experiments where we only keep the
closest K neighbors (e.g. K=30) and compute compact neighbor-set disagreement
and robustness features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .mos_utils import effective_sample_size, weighted_entropy


@dataclass(frozen=True)
class KnnLiteSpec:
    k: int = 30
    # Exclude neighbors whose asof_date_local is too recent to have a known label
    # under the chosen as-of policy. For "target_minus_one_day_12z" with
    # obs_cutoff_lag_days=1, we must exclude age_days <= 1.
    label_lag_days: int = 1
    prefix: str = "knn30_"


def _max_available_slots(df: pd.DataFrame) -> int:
    max_i = 0
    for col in df.columns:
        if not col.startswith("knn_v0_nn") or not col.endswith("_dist"):
            continue
        mid = col[len("knn_v0_nn") : -len("_dist")]
        if mid.isdigit():
            max_i = max(max_i, int(mid))
    return max_i


def compute_knn_lite_features_from_v0_slots(
    df: pd.DataFrame,
    *,
    spec: KnnLiteSpec | None = None,
) -> pd.DataFrame:
    """Compute a compact KNN feature set using existing `knn_v0_nn*` columns.

    Expected columns in `df`:
    - base_tmax_blend
    - knn_v0_nn{i}_dist
    - knn_v0_nn{i}_age_days
    - knn_v0_nn{i}_y_actual_tmax_nextday
    - knn_v0_nn{i}_residual

    The input `knn_v0_nn*` slots must be ordered by increasing distance (as
    produced by `mos_knn.compute_knn_features`).
    """

    spec = spec or KnnLiteSpec()
    k = int(spec.k)
    if k <= 0:
        raise ValueError("k must be > 0")
    label_lag_days = max(int(spec.label_lag_days), 0)
    prefix = str(spec.prefix)

    if "base_tmax_blend" not in df.columns:
        raise ValueError("Missing required column: base_tmax_blend")

    pool_k = _max_available_slots(df)
    if pool_k < k:
        raise ValueError(
            f"Not enough knn_v0_nn* slots in df for k={k}. Found pool_k={pool_k}."
        )

    n = len(df)
    base_now = df["base_tmax_blend"].to_numpy(dtype=float)
    # Guard against MOS missing-code contamination (e.g. 999 -> blend ~ 545.5).
    # If blend is out of plausible Fahrenheit bounds, fall back to GFS/NAM if available.
    base_gfs = df["base_tmax_gfs"].to_numpy(dtype=float) if "base_tmax_gfs" in df.columns else None
    base_nam = df["base_tmax_nam"].to_numpy(dtype=float) if "base_tmax_nam" in df.columns else None
    invalid = ~np.isfinite(base_now) | (base_now < -100) | (base_now > 200)
    if np.any(invalid):
        if base_gfs is not None:
            ok_gfs = np.isfinite(base_gfs) & (base_gfs >= -100) & (base_gfs <= 200)
            base_now = np.where(invalid & ok_gfs, base_gfs, base_now)
        if base_nam is not None:
            ok_nam = np.isfinite(base_nam) & (base_nam >= -100) & (base_nam <= 200)
            base_now = np.where(invalid & ok_nam, base_nam, base_now)

    # Extract the pool of neighbor slots as matrices (n, pool_k).
    dist_pool = np.full((n, pool_k), np.nan, dtype=float)
    age_pool = np.full((n, pool_k), np.nan, dtype=float)
    y_pool = np.full((n, pool_k), np.nan, dtype=float)
    resid_pool = np.full((n, pool_k), np.nan, dtype=float)

    for i in range(1, pool_k + 1):
        j = i - 1
        dist_pool[:, j] = df[f"knn_v0_nn{i}_dist"].to_numpy(dtype=float)
        age_pool[:, j] = df[f"knn_v0_nn{i}_age_days"].to_numpy(dtype=float)
        y_pool[:, j] = df[f"knn_v0_nn{i}_y_actual_tmax_nextday"].to_numpy(dtype=float)
        resid_pool[:, j] = df[f"knn_v0_nn{i}_residual"].to_numpy(dtype=float)

    # Select the top-k eligible neighbors, preserving distance ordering.
    dist = np.full((n, k), np.nan, dtype=float)
    age_days = np.full((n, k), np.nan, dtype=float)
    y = np.full((n, k), np.nan, dtype=float)
    resid = np.full((n, k), np.nan, dtype=float)

    for row in range(n):
        eligible = np.isfinite(dist_pool[row, :]) & (age_pool[row, :] > label_lag_days)
        idxs = np.flatnonzero(eligible)
        if idxs.size == 0:
            continue
        idxs = idxs[:k]
        kk = idxs.size
        dist[row, :kk] = dist_pool[row, idxs]
        age_days[row, :kk] = age_pool[row, idxs]
        y[row, :kk] = y_pool[row, idxs]
        resid[row, :kk] = resid_pool[row, idxs]

    k_used = np.sum(np.isfinite(dist), axis=1).astype(float)
    tau = np.nanmedian(dist, axis=1)
    tau = np.where(np.isfinite(tau), np.maximum(tau, 1e-6), np.nan)

    # Gaussian kernel weights (same functional form as mos_knn).
    w_raw = np.exp(-0.5 * (dist / tau[:, None]) ** 2)
    w_raw = np.where(np.isfinite(w_raw), w_raw, 0.0)
    w_raw_sum = np.sum(w_raw, axis=1)
    w_norm = w_raw / (w_raw_sum[:, None] + 1e-12)

    ess = np.full(n, np.nan, dtype=float)
    for i in range(n):
        ess[i] = effective_sample_size(w_raw[i, :])
    ess_ratio = np.where(k_used > 0, ess / k_used, np.nan)

    analog = base_now[:, None] + resid

    # Vectorized weighted mean/std with NaN-aware masking.
    def wmean(values: np.ndarray) -> np.ndarray:
        mask = np.isfinite(values) & np.isfinite(w_norm)
        w = np.where(mask, w_norm, 0.0)
        wsum = np.sum(w, axis=1)
        vsum = np.sum(np.where(mask, values * w_norm, 0.0), axis=1)
        out = np.full(values.shape[0], np.nan, dtype=float)
        ok = wsum > 0
        out[ok] = vsum[ok] / wsum[ok]
        return out

    def wstd(values: np.ndarray, mean: np.ndarray) -> np.ndarray:
        mask = np.isfinite(values) & np.isfinite(w_norm)
        w = np.where(mask, w_norm, 0.0)
        wsum = np.sum(w, axis=1)
        diff = values - mean[:, None]
        var = np.sum(np.where(mask, w_norm * diff**2, 0.0), axis=1) / np.where(wsum > 0, wsum, 1.0)
        return np.where(wsum > 0, np.sqrt(var), np.nan)

    analog_mu = wmean(analog)
    analog_std = wstd(analog, analog_mu)
    y_mu = wmean(y)
    y_std = wstd(y, y_mu)
    resid_mu = wmean(resid)
    resid_std = wstd(resid, resid_mu)

    dist_mean = np.nanmean(dist, axis=1)
    dist_std = np.nanstd(dist, axis=1)
    dist_p50 = np.nanmedian(dist, axis=1)
    dist_min = np.nanmin(dist, axis=1)
    dist_max = np.nanmax(dist, axis=1)

    # Robustness: high when distances are tight, analog spread is small, and ESS is high.
    numer = dist_p50 * (analog_std + 1e-6)
    denom = ess_ratio + 1e-6
    raw = np.full_like(numer, np.nan, dtype=float)
    ok = np.isfinite(numer) & np.isfinite(denom) & (denom != 0)
    raw[ok] = numer[ok] / denom[ok]
    robustness = 1.0 / (1.0 + raw)

    # Neighbor-level outputs (lightweight, k slots).
    out: dict[str, np.ndarray] = {
        f"{prefix}k_used": k_used,
        f"{prefix}tau": tau,
        f"{prefix}dist_mean": dist_mean,
        f"{prefix}dist_std": dist_std,
        f"{prefix}dist_p50": dist_p50,
        f"{prefix}dist_min": dist_min,
        f"{prefix}dist_max": dist_max,
        f"{prefix}ess": ess,
        f"{prefix}ess_ratio": ess_ratio,
        f"{prefix}max_weight_share": np.nanmax(w_norm, axis=1),
        f"{prefix}entropy_weight": np.array([weighted_entropy(w_norm[i, :]) for i in range(n)], dtype=float),
        f"{prefix}analog_mu": analog_mu,
        f"{prefix}analog_std": analog_std,
        f"{prefix}y_mu": y_mu,
        f"{prefix}y_std": y_std,
        f"{prefix}resid_mu": resid_mu,
        f"{prefix}resid_std": resid_std,
        f"{prefix}robustness_score": robustness,
    }

    # Nearest-neighbor emphasis / gaps.
    out[f"{prefix}nn1_dist"] = dist[:, 0]
    out[f"{prefix}nn1_age_days"] = age_days[:, 0]
    out[f"{prefix}nn1_residual"] = resid[:, 0]
    out[f"{prefix}nn1_analog"] = analog[:, 0]
    out[f"{prefix}nn1_y"] = y[:, 0]
    out[f"{prefix}nn2_dist"] = dist[:, 1] if k >= 2 else np.full(n, np.nan, dtype=float)
    out[f"{prefix}dist_gap_1_2"] = out[f"{prefix}nn2_dist"] - out[f"{prefix}nn1_dist"]
    out[f"{prefix}dist_gap_1_k"] = dist[:, k - 1] - dist[:, 0]

    # Store per-neighbor metrics (dist/age/weights/resid/y/analog) for the chosen K.
    for i in range(1, k + 1):
        j = i - 1
        out[f"{prefix}nn{i}_dist"] = dist[:, j]
        out[f"{prefix}nn{i}_age_days"] = age_days[:, j]
        out[f"{prefix}nn{i}_w_norm"] = w_norm[:, j]
        out[f"{prefix}nn{i}_residual"] = resid[:, j]
        out[f"{prefix}nn{i}_y"] = y[:, j]
        out[f"{prefix}nn{i}_analog"] = analog[:, j]

    return pd.DataFrame(out)
