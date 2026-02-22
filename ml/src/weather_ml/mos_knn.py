"""KNN feature generation for MOS dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .mos_config import KnnViewConfig
from .mos_utils import (
    date_to_int,
    doy_diff,
    effective_sample_size,
    gini_from_weights,
    safe_div,
    weighted_entropy,
    weighted_mean,
    weighted_quantile,
    weighted_std,
)


@dataclass(frozen=True)
class DistanceData:
    feature_names: list[str]
    scaled: np.ndarray
    missing: np.ndarray
    u: np.ndarray
    u_cos: np.ndarray
    rank: np.ndarray
    u_rank: np.ndarray
    weight_vector: np.ndarray


def prepare_distance_data(
    df: pd.DataFrame,
    feature_names: Iterable[str],
    calib_mask: np.ndarray,
    missing_penalty: float,
    weight_map: dict[str, float] | None = None,
) -> DistanceData:
    feature_names = list(feature_names)
    if not feature_names:
        raise ValueError("Distance feature list is empty.")
    X = df[feature_names].to_numpy(dtype=float)
    missing = np.isnan(X)
    calib_values = X[calib_mask]
    med = np.nanmedian(calib_values, axis=0)
    p25 = np.nanpercentile(calib_values, 25, axis=0)
    p75 = np.nanpercentile(calib_values, 75, axis=0)
    iqr = np.maximum(p75 - p25, 1e-3)
    scaled = (X - med) / iqr
    scaled = np.where(missing, 0.0, scaled)
    missing_flags = missing.astype(float)
    u = np.concatenate([scaled, missing_penalty * missing_flags], axis=1)

    weights = []
    weight_map = weight_map or {}
    for name in feature_names:
        weights.append(float(weight_map.get(name, 1.0)))
    w_scaled = np.asarray(weights, dtype=float)
    weight_vector = np.concatenate([w_scaled, w_scaled], axis=0)

    # cosine normalization
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_cos = u / np.maximum(norms, 1e-12)

    # rank/quantile transform
    rank = np.full_like(X, 0.5, dtype=float)
    for j in range(X.shape[1]):
        cal = calib_values[:, j]
        cal = cal[~np.isnan(cal)]
        if cal.size == 0:
            continue
        cal_sorted = np.sort(cal)
        vals = X[:, j]
        mask = ~np.isnan(vals)
        idx = np.searchsorted(cal_sorted, vals[mask], side="right")
        rank[mask, j] = idx / cal_sorted.size
    u_rank = np.concatenate([rank, missing_penalty * missing_flags], axis=1)

    return DistanceData(
        feature_names=feature_names,
        scaled=scaled,
        missing=missing,
        u=u,
        u_cos=u_cos,
        rank=rank,
        u_rank=u_rank,
        weight_vector=weight_vector,
    )


def compute_knn_features(
    df: pd.DataFrame,
    distance_data: DistanceData,
    views: list[KnnViewConfig],
    k: int,
    thresholds: list[int],
    tau_fixed: list[float],
    season_window: int,
    label_lag_days: int,
    consistency_features: list[str],
    base_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    n = len(df)
    if n == 0:
        return pd.DataFrame(), {}

    asof_dates = pd.to_datetime(df["asof_date_local"]).dt.date.to_numpy()
    asof_ordinal = np.array([d.toordinal() for d in asof_dates], dtype=int)
    asof_int = np.array([date_to_int(d) for d in asof_dates], dtype=int)
    doy = pd.to_datetime(df["asof_date_local"]).dt.dayofyear.to_numpy()
    month = pd.to_datetime(df["asof_date_local"]).dt.month.to_numpy()
    year = pd.to_datetime(df["asof_date_local"]).dt.year.to_numpy()

    base_vals = df[base_col].to_numpy(dtype=float)
    y_vals = df[target_col].to_numpy(dtype=float)

    # Pre-extract consistency feature matrix
    consistency_features = [f for f in consistency_features if f in df.columns]
    consistency_matrix = df[consistency_features].to_numpy(dtype=float)

    results: dict[str, np.ndarray] = {}
    failure_counts = {
        "knn_candidate_insufficient": 0,
        "knn_zero_candidates": 0,
    }

    thresholds = sorted(set(int(t) for t in thresholds))
    tau_fixed = [float(t) for t in tau_fixed]

    def ensure_column(name: str) -> np.ndarray:
        if name not in results:
            results[name] = np.full(n, np.nan, dtype=float)
        return results[name]

    def weighted_corr_and_slope(
        x: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> tuple[float, float]:
        """Return (corr(x,y), slope of y~x) under non-negative weights.

        Uses a NaN-safe mask and renormalizes weights on the valid subset.
        """

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
        if np.sum(mask) < 2:
            return np.nan, np.nan
        x = x[mask]
        y = y[mask]
        w = w[mask]
        w_sum = np.sum(w)
        if w_sum <= 0:
            return np.nan, np.nan
        w = w / w_sum
        mx = np.sum(w * x)
        my = np.sum(w * y)
        dx = x - mx
        dy = y - my
        cov = np.sum(w * dx * dy)
        varx = np.sum(w * dx * dx)
        vary = np.sum(w * dy * dy)
        corr = cov / np.sqrt(varx * vary) if varx > 0 and vary > 0 else np.nan
        slope = cov / varx if varx > 0 else np.nan
        return float(corr) if np.isfinite(corr) else np.nan, float(slope) if np.isfinite(slope) else np.nan

    # Distance-feature missingness for the current row (independent of view).
    xdist_missing_count = np.sum(distance_data.missing, axis=1).astype(float)
    ensure_column("knn_xdist_missing_count")[:] = xdist_missing_count

    for view in views:
        prefix = f"knn_{view.name}_"
        for col in [
            "dist_min",
            "dist_p10",
            "dist_p25",
            "dist_p50",
            "dist_p75",
            "dist_p90",
            "dist_max",
            "dist_mean",
            "dist_std",
            "dist_iqr",
            "dist_skew_proxy",
            "ratio_dist_p90_p10",
            "ratio_dist_max_min",
            "weight_raw_sum",
            "weight_norm_sum",
            "ess",
            "ess_ratio",
            "max_weight_share",
            "top5_weight_share",
            "entropy_weight",
            "gini_weight",
            "unique_years_topk",
            "year_hhi",
            "age_days_wmean",
            "frac_weight_age_le_365",
            "frac_weight_age_le_730",
            "frac_weight_age_le_1825",
            "k_used",
            "candidate_count",
            "flag_insufficient_candidates",
            "analog_mu",
            "analog_std",
            "analog_p05",
            "analog_p10",
            "analog_p25",
            "analog_p50",
            "analog_p75",
            "analog_p90",
            "analog_p95",
            "analog_iqr",
            "analog_tail_width",
            "analog_asym",
            "resid_mu",
            "resid_std",
            "resid_p10",
            "resid_p50",
            "resid_p90",
            "resid_iqr",
            # additional KNN diagnostics
            "frac_weight_same_month",
            "frac_weight_same_doy_45",
            "nbr_mean_xdist_missing_count",
            "xdist_missing_mismatch",
            "resid_dist_corr",
            "abs_resid_dist_corr",
            "resid_vs_dist_slope",
            "abs_resid_vs_dist_slope",
            "robustness",
            "density",
            "sharpness",
        ]:
            ensure_column(prefix + col)

        for thr in thresholds:
            ensure_column(f"{prefix}analog_p_ge_{thr}")
            ensure_column(f"{prefix}y_p_ge_{thr}")

        for tau in tau_fixed:
            tau_label = _format_tau_label(tau)
            ensure_column(f"{prefix}tau{tau_label}_analog_mu")
            ensure_column(f"{prefix}tau{tau_label}_analog_std")
            ensure_column(f"{prefix}tau{tau_label}_analog_p50")
            ensure_column(f"{prefix}tau{tau_label}_ess_ratio")

        for feature in consistency_features:
            ensure_column(f"{prefix}nbr_mean_{feature}")
            ensure_column(f"{prefix}nbr_std_{feature}")
            ensure_column(f"{prefix}nbr_mean_abs_diff_{feature}")
            ensure_column(f"{prefix}nbr_mean_diff_{feature}")
            ensure_column(f"{prefix}nbr_p50_abs_diff_{feature}")

    # v0 top-k sensitivity (nearest-neighbor focus) and stability deltas.
    topk_sensitivity = [1, 3, 5, 10, 30]
    for kk in topk_sensitivity:
        ensure_column(f"knn_v0_top{kk}_analog_mu")
        ensure_column(f"knn_v0_top{kk}_analog_std")
        ensure_column(f"knn_v0_top{kk}_dist_p50")
        ensure_column(f"knn_v0_top{kk}_ess_ratio")
    ensure_column("knn_v0_delta_top30_minus_full_analog_mu")
    ensure_column("knn_v0_delta_top10_minus_top30_analog_mu")
    ensure_column("knn_v0_delta_top5_minus_top30_analog_mu")

    # v0 neighbor slots
    max_k = max(k, 1)
    for i in range(1, max_k + 1):
        for col in [
            "asof_yyyymmdd",
            "dist",
            "dist_sq",
            "weight_raw",
            "weight_norm",
            "age_days",
            "same_month",
            "same_doy_window_45",
            "base_tmax_blend",
            "y_actual_tmax_nextday",
            "residual",
        ]:
            ensure_column(f"knn_v0_nn{i}_{col}")

    label_lag_days = max(int(label_lag_days), 0)

    for idx in range(n):
        # If the truth series is only known through (asof_date_local - label_lag_days),
        # we must exclude the most-recent label_lag_days rows from neighbor candidates.
        cand_end = idx - label_lag_days
        cand_idx = np.arange(0, cand_end, dtype=int) if cand_end > 0 else np.array([], dtype=int)
        if cand_idx.size == 0:
            failure_counts["knn_zero_candidates"] += 1
            for view in views:
                results[f"knn_{view.name}_candidate_count"][idx] = 0.0
                results[f"knn_{view.name}_k_used"][idx] = 0.0
                results[f"knn_{view.name}_flag_insufficient_candidates"][idx] = 1.0
            continue

        for view in views:
            if view.pool.lower() in ("season45", "s45", "season"):
                diff = np.array([doy_diff(doy[idx], doy[c]) for c in cand_idx], dtype=int)
                pool_mask = diff <= season_window
                pool_idx = cand_idx[pool_mask]
            else:
                pool_idx = cand_idx

            candidate_count = pool_idx.size
            if candidate_count == 0:
                failure_counts["knn_zero_candidates"] += 1
                failure_counts[f"{view.name}_zero_candidates"] = (
                    failure_counts.get(f"{view.name}_zero_candidates", 0) + 1
                )
                results[f"knn_{view.name}_candidate_count"][idx] = 0.0
                results[f"knn_{view.name}_k_used"][idx] = 0.0
                results[f"knn_{view.name}_flag_insufficient_candidates"][idx] = 1.0
                continue

            if candidate_count < k:
                failure_counts["knn_candidate_insufficient"] += 1
                failure_counts[f"{view.name}_candidate_lt_k"] = (
                    failure_counts.get(f"{view.name}_candidate_lt_k", 0) + 1
                )

            k_used = min(k, candidate_count)

            if view.distance.lower() in ("l2", "euclid", "euclidean"):
                diff = distance_data.u[pool_idx] - distance_data.u[idx]
                dists = np.sqrt(np.sum((diff**2) * distance_data.weight_vector, axis=1))
            elif view.distance.lower() in ("cos", "cosine"):
                vec = distance_data.u_cos[idx]
                dists = 1.0 - np.dot(distance_data.u_cos[pool_idx], vec)
            elif view.distance.lower() in ("rank", "quantile"):
                diff = distance_data.u_rank[pool_idx] - distance_data.u_rank[idx]
                dists = np.sqrt(np.sum((diff**2) * distance_data.weight_vector, axis=1))
            else:
                raise ValueError(f"Unknown distance: {view.distance}")

            age_days = asof_ordinal[idx] - asof_ordinal[pool_idx]
            order = np.lexsort((asof_int[pool_idx], age_days, dists))
            top = pool_idx[order[:k_used]]
            dist_top = dists[order[:k_used]]

            tau = max(float(np.median(dist_top)), 1e-6)
            w_raw = np.exp(-0.5 * (dist_top / tau) ** 2)
            w_norm = w_raw / (np.sum(w_raw) + 1e-12)

            prefix = f"knn_{view.name}_"
            results[f"{prefix}candidate_count"][idx] = float(candidate_count)
            results[f"{prefix}k_used"][idx] = float(k_used)
            results[f"{prefix}flag_insufficient_candidates"][idx] = float(candidate_count < k)

            # distance stats
            quant = np.quantile(dist_top, [0.1, 0.25, 0.5, 0.75, 0.9])
            results[f"{prefix}dist_min"][idx] = float(np.min(dist_top))
            results[f"{prefix}dist_p10"][idx] = float(quant[0])
            results[f"{prefix}dist_p25"][idx] = float(quant[1])
            results[f"{prefix}dist_p50"][idx] = float(quant[2])
            results[f"{prefix}dist_p75"][idx] = float(quant[3])
            results[f"{prefix}dist_p90"][idx] = float(quant[4])
            results[f"{prefix}dist_max"][idx] = float(np.max(dist_top))
            results[f"{prefix}dist_mean"][idx] = float(np.mean(dist_top))
            results[f"{prefix}dist_std"][idx] = float(np.std(dist_top))
            results[f"{prefix}dist_iqr"][idx] = float(quant[3] - quant[1])
            results[f"{prefix}dist_skew_proxy"][idx] = float(quant[4] + quant[0] - 2 * quant[2])
            results[f"{prefix}ratio_dist_p90_p10"][idx] = safe_div(float(quant[4]), float(quant[0]), np.nan)
            results[f"{prefix}ratio_dist_max_min"][idx] = safe_div(
                float(np.max(dist_top)), float(np.min(dist_top)), np.nan
            )

            # weights
            results[f"{prefix}weight_raw_sum"][idx] = float(np.sum(w_raw))
            results[f"{prefix}weight_norm_sum"][idx] = float(np.sum(w_norm))
            ess = effective_sample_size(w_raw)
            results[f"{prefix}ess"][idx] = ess
            results[f"{prefix}ess_ratio"][idx] = safe_div(ess, float(k_used), np.nan)
            results[f"{prefix}max_weight_share"][idx] = float(np.max(w_norm))
            top5 = np.sort(w_norm)[-5:] if w_norm.size >= 5 else w_norm
            results[f"{prefix}top5_weight_share"][idx] = float(np.sum(top5))
            results[f"{prefix}entropy_weight"][idx] = weighted_entropy(w_norm)
            results[f"{prefix}gini_weight"][idx] = gini_from_weights(w_norm)

            # seasonal alignment (weight shares)
            same_month = month[top] == month[idx]
            results[f"{prefix}frac_weight_same_month"][idx] = float(np.sum(w_norm[same_month]))
            doy_delta = np.abs(doy[top] - doy[idx])
            doy_delta = np.minimum(doy_delta, 365 - doy_delta)
            same_doy_45 = doy_delta <= 45
            results[f"{prefix}frac_weight_same_doy_45"][idx] = float(
                np.sum(w_norm[same_doy_45])
            )

            # x_dist missingness diagnostics
            results[f"{prefix}nbr_mean_xdist_missing_count"][idx] = weighted_mean(
                xdist_missing_count[top], w_norm
            )
            current_missing = distance_data.missing[idx]
            neigh_missing = distance_data.missing[top]
            mismatch_rate = np.mean(np.logical_xor(neigh_missing, current_missing), axis=1)
            results[f"{prefix}xdist_missing_mismatch"][idx] = weighted_mean(
                mismatch_rate, w_norm
            )

            # diversity
            years = year[top]
            results[f"{prefix}unique_years_topk"][idx] = float(len(set(years)))
            year_weights: dict[int, float] = {}
            for y, w in zip(years, w_norm, strict=False):
                year_weights[int(y)] = year_weights.get(int(y), 0.0) + float(w)
            results[f"{prefix}year_hhi"][idx] = float(
                sum(v * v for v in year_weights.values()) if year_weights else np.nan
            )
            age_top = asof_ordinal[idx] - asof_ordinal[top]
            results[f"{prefix}age_days_wmean"][idx] = float(np.sum(w_norm * age_top))
            for label, thresh in [
                ("frac_weight_age_le_365", 365),
                ("frac_weight_age_le_730", 730),
                ("frac_weight_age_le_1825", 1825),
            ]:
                results[f"{prefix}{label}"][idx] = float(np.sum(w_norm[age_top <= thresh]))

            # analog distribution
            base_now = base_vals[idx]
            resid = y_vals[top] - base_vals[top]
            analog = base_now + resid
            w_norm_safe = w_norm
            analog_mu = weighted_mean(analog, w_norm_safe)
            analog_std = weighted_std(analog, w_norm_safe)
            q_analog = weighted_quantile(analog, w_norm_safe, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
            results[f"{prefix}analog_mu"][idx] = analog_mu
            results[f"{prefix}analog_std"][idx] = analog_std
            results[f"{prefix}analog_p05"][idx] = q_analog[0]
            results[f"{prefix}analog_p10"][idx] = q_analog[1]
            results[f"{prefix}analog_p25"][idx] = q_analog[2]
            results[f"{prefix}analog_p50"][idx] = q_analog[3]
            results[f"{prefix}analog_p75"][idx] = q_analog[4]
            results[f"{prefix}analog_p90"][idx] = q_analog[5]
            results[f"{prefix}analog_p95"][idx] = q_analog[6]
            results[f"{prefix}analog_iqr"][idx] = q_analog[4] - q_analog[2]
            results[f"{prefix}analog_tail_width"][idx] = q_analog[5] - q_analog[1]
            results[f"{prefix}analog_asym"][idx] = q_analog[5] + q_analog[1] - 2 * q_analog[3]

            q_resid = weighted_quantile(resid, w_norm_safe, [0.1, 0.5, 0.9])
            results[f"{prefix}resid_mu"][idx] = weighted_mean(resid, w_norm_safe)
            results[f"{prefix}resid_std"][idx] = weighted_std(resid, w_norm_safe)
            results[f"{prefix}resid_p10"][idx] = q_resid[0]
            results[f"{prefix}resid_p50"][idx] = q_resid[1]
            results[f"{prefix}resid_p90"][idx] = q_resid[2]
            results[f"{prefix}resid_iqr"][idx] = q_resid[2] - q_resid[0]

            # residual-vs-distance relationship (nearest-neighbor reliability signal)
            corr, slope = weighted_corr_and_slope(dist_top, resid, w_norm_safe)
            results[f"{prefix}resid_dist_corr"][idx] = corr
            results[f"{prefix}resid_vs_dist_slope"][idx] = slope
            abs_resid = np.abs(resid)
            corr, slope = weighted_corr_and_slope(dist_top, abs_resid, w_norm_safe)
            results[f"{prefix}abs_resid_dist_corr"][idx] = corr
            results[f"{prefix}abs_resid_vs_dist_slope"][idx] = slope

            # robustness/density/sharpness (how much to trust this KNN set)
            ess_ratio = safe_div(ess, float(k_used), np.nan)
            dist_p50 = float(quant[2])
            results[f"{prefix}density"][idx] = safe_div(1.0, dist_p50 + 1e-6, np.nan)
            results[f"{prefix}sharpness"][idx] = safe_div(1.0, analog_std + 1e-6, np.nan)
            robustness_raw = safe_div(
                dist_p50 * (analog_std + 1e-6),
                ess_ratio + 1e-6,
                np.nan,
            )
            results[f"{prefix}robustness"][idx] = safe_div(1.0, 1.0 + robustness_raw, np.nan)

            # top-k sensitivity (v0 only): how much the analog forecast shifts when we
            # keep only the closest neighbors.
            if view.name == "v0":
                mu_by_k: dict[int, float] = {}
                for kk in topk_sensitivity:
                    if k_used < kk:
                        continue
                    analog_sub = analog[:kk]
                    dist_sub = dist_top[:kk]
                    w_raw_sub = w_raw[:kk]
                    w_norm_sub = w_raw_sub / (np.sum(w_raw_sub) + 1e-12)
                    mu = weighted_mean(analog_sub, w_norm_sub)
                    mu_by_k[kk] = mu
                    results[f"knn_v0_top{kk}_analog_mu"][idx] = mu
                    results[f"knn_v0_top{kk}_analog_std"][idx] = weighted_std(
                        analog_sub, w_norm_sub
                    )
                    results[f"knn_v0_top{kk}_dist_p50"][idx] = float(np.median(dist_sub))
                    ess_sub = effective_sample_size(w_raw_sub)
                    results[f"knn_v0_top{kk}_ess_ratio"][idx] = safe_div(
                        ess_sub, float(kk), np.nan
                    )

                mu_full = analog_mu
                mu_top30 = mu_by_k.get(30)
                if np.isfinite(mu_full) and mu_top30 is not None and np.isfinite(mu_top30):
                    results["knn_v0_delta_top30_minus_full_analog_mu"][idx] = float(
                        mu_top30 - mu_full
                    )
                mu_top10 = mu_by_k.get(10)
                if mu_top10 is not None and mu_top30 is not None and np.isfinite(mu_top10) and np.isfinite(mu_top30):
                    results["knn_v0_delta_top10_minus_top30_analog_mu"][idx] = float(
                        mu_top10 - mu_top30
                    )
                mu_top5 = mu_by_k.get(5)
                if mu_top5 is not None and mu_top30 is not None and np.isfinite(mu_top5) and np.isfinite(mu_top30):
                    results["knn_v0_delta_top5_minus_top30_analog_mu"][idx] = float(
                        mu_top5 - mu_top30
                    )

            # fixed-tau variants
            for tau in tau_fixed:
                tau_label = _format_tau_label(tau)
                tau_val = max(float(tau), 1e-6)
                w_raw_tau = np.exp(-0.5 * (dist_top / tau_val) ** 2)
                w_norm_tau = w_raw_tau / (np.sum(w_raw_tau) + 1e-12)
                results[f"{prefix}tau{tau_label}_analog_mu"][idx] = weighted_mean(analog, w_norm_tau)
                results[f"{prefix}tau{tau_label}_analog_std"][idx] = weighted_std(analog, w_norm_tau)
                results[f"{prefix}tau{tau_label}_analog_p50"][idx] = weighted_quantile(
                    analog, w_norm_tau, [0.5]
                )[0]
                ess_tau = effective_sample_size(w_raw_tau)
                results[f"{prefix}tau{tau_label}_ess_ratio"][idx] = safe_div(
                    ess_tau, float(k_used), np.nan
                )

            # threshold probabilities
            for thr in thresholds:
                analog_mask = ~np.isnan(analog)
                if np.any(analog_mask):
                    w_use = w_norm_safe[analog_mask]
                    w_sum = np.sum(w_use)
                    if w_sum > 0:
                        results[f"{prefix}analog_p_ge_{thr}"][idx] = float(
                            np.sum(w_use[analog[analog_mask] >= thr]) / w_sum
                        )
                y_mask = ~np.isnan(y_vals[top])
                if np.any(y_mask):
                    w_use = w_norm_safe[y_mask]
                    w_sum = np.sum(w_use)
                    if w_sum > 0:
                        results[f"{prefix}y_p_ge_{thr}"][idx] = float(
                            np.sum(w_use[y_vals[top][y_mask] >= thr]) / w_sum
                        )

            # consistency features
            for f_idx, feature in enumerate(consistency_features):
                vals = consistency_matrix[top, f_idx]
                mask = ~np.isnan(vals)
                if not np.any(mask):
                    continue
                w = w_norm_safe[mask]
                vals = vals[mask]
                w = w / (np.sum(w) + 1e-12)
                results[f"{prefix}nbr_mean_{feature}"][idx] = weighted_mean(vals, w)
                results[f"{prefix}nbr_std_{feature}"][idx] = weighted_std(vals, w)
                current_val = consistency_matrix[idx, f_idx]
                if not np.isnan(current_val):
                    diffs = current_val - vals
                    abs_diffs = np.abs(diffs)
                    results[f"{prefix}nbr_mean_abs_diff_{feature}"][idx] = weighted_mean(abs_diffs, w)
                    results[f"{prefix}nbr_mean_diff_{feature}"][idx] = weighted_mean(diffs, w)
                    results[f"{prefix}nbr_p50_abs_diff_{feature}"][idx] = weighted_quantile(
                        abs_diffs, w, [0.5]
                    )[0]

            # v0 neighbor slots
            if view.name == "v0":
                for slot_idx, neigh in enumerate(top, start=1):
                    results[f"knn_v0_nn{slot_idx}_asof_yyyymmdd"][idx] = float(
                        asof_int[neigh]
                    )
                    results[f"knn_v0_nn{slot_idx}_dist"][idx] = float(dist_top[slot_idx - 1])
                    results[f"knn_v0_nn{slot_idx}_dist_sq"][idx] = float(
                        dist_top[slot_idx - 1] ** 2
                    )
                    results[f"knn_v0_nn{slot_idx}_weight_raw"][idx] = float(
                        w_raw[slot_idx - 1]
                    )
                    results[f"knn_v0_nn{slot_idx}_weight_norm"][idx] = float(
                        w_norm[slot_idx - 1]
                    )
                    age = asof_ordinal[idx] - asof_ordinal[neigh]
                    results[f"knn_v0_nn{slot_idx}_age_days"][idx] = float(age)
                    results[f"knn_v0_nn{slot_idx}_same_month"][idx] = float(
                        month[idx] == month[neigh]
                    )
                    results[f"knn_v0_nn{slot_idx}_same_doy_window_45"][idx] = float(
                        doy_diff(doy[idx], doy[neigh]) <= 45
                    )
                    results[f"knn_v0_nn{slot_idx}_base_tmax_blend"][idx] = float(
                        base_vals[neigh]
                    )
                    results[f"knn_v0_nn{slot_idx}_y_actual_tmax_nextday"][idx] = float(
                        y_vals[neigh]
                    )
                    results[f"knn_v0_nn{slot_idx}_residual"][idx] = float(
                        y_vals[neigh] - base_vals[neigh]
                    )

    return pd.DataFrame(results), failure_counts


def _format_tau_label(value: float) -> str:
    text = f"{value:.2f}"
    text = text.rstrip("0").rstrip(".")
    return text.replace(".", "p")
