from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import logging

import numpy as np
import pandas as pd
from .logging_utils import ProgressTracker


@dataclass(frozen=True)
class AnalogStandardizer:
    feature_columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    weight: np.ndarray


@dataclass(frozen=True)
class AnalogLibrary:
    row_index: np.ndarray
    dates: np.ndarray
    doy: np.ndarray
    cutoff_minutes: np.ndarray
    features_std: np.ndarray
    peak: np.ndarray
    delta_class: np.ndarray
    index_by_cutoff: dict[int, np.ndarray]


@dataclass(frozen=True)
class AnalogPosterior:
    p_peak: np.ndarray
    p_delta_cond: np.ndarray
    q_score: np.ndarray
    candidate_count: np.ndarray
    non_peak_count: np.ndarray


def fit_analog_standardizer(
    *,
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    train_mask: np.ndarray,
    feature_weights: dict[str, float] | None = None,
) -> AnalogStandardizer:
    cols = tuple(feature_columns)
    if not cols:
        raise ValueError("Analog feature column list is empty.")
    mat = df.loc[:, cols].to_numpy(dtype=float)
    train_mat = mat[train_mask]
    mean = np.nanmean(train_mat, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.nanstd(train_mat, axis=0)
    std = np.where((std > 0) & np.isfinite(std), std, 1.0)
    w = np.ones(len(cols), dtype=float)
    if feature_weights:
        for i, c in enumerate(cols):
            if c in feature_weights:
                w[i] = float(feature_weights[c])
    return AnalogStandardizer(feature_columns=cols, mean=mean, std=std, weight=w)


def build_analog_library(
    *,
    df: pd.DataFrame,
    standardizer: AnalogStandardizer,
    delta_class_max: int,
) -> AnalogLibrary:
    cols = list(standardizer.feature_columns)
    mat = df.loc[:, cols].to_numpy(dtype=float)
    mat = np.where(np.isfinite(mat), mat, standardizer.mean)
    mat_std = (mat - standardizer.mean) / np.maximum(standardizer.std, 1e-6)

    dates = pd.to_datetime(df["target_date_local"]).dt.date.to_numpy()
    doy = pd.to_datetime(df["target_date_local"]).dt.dayofyear.to_numpy(dtype=int)
    cutoff_minutes = pd.to_numeric(df["cutoff_minutes"], errors="coerce").to_numpy(dtype=int)
    peak = pd.to_numeric(df["peak"], errors="coerce").fillna(0).astype(int).to_numpy()
    delta = pd.to_numeric(df["delta"], errors="coerce").fillna(0).astype(int).to_numpy()
    delta_class = np.clip(delta, 1, delta_class_max)
    delta_class = np.where(peak > 0, 0, delta_class)
    row_index = np.arange(len(df), dtype=int)

    index_by_cutoff: dict[int, np.ndarray] = {}
    for cm in np.unique(cutoff_minutes):
        idx = np.where(cutoff_minutes == cm)[0]
        idx = idx[np.argsort(dates[idx])]
        index_by_cutoff[int(cm)] = idx

    return AnalogLibrary(
        row_index=row_index,
        dates=dates,
        doy=doy,
        cutoff_minutes=cutoff_minutes,
        features_std=mat_std,
        peak=peak,
        delta_class=delta_class,
        index_by_cutoff=index_by_cutoff,
    )


def _doy_distance_wrap(a: int, b: np.ndarray) -> np.ndarray:
    raw = np.abs(b - a)
    return np.minimum(raw, 366 - raw)


def predict_knn_posterior(
    *,
    library: AnalogLibrary,
    standardizer: AnalogStandardizer,
    query_indices: np.ndarray,
    k: int,
    delta_class_max: int,
    season_window_doy: int,
    min_pool: int,
    min_non_peak: int,
    logger: logging.Logger | None = None,
    log_every_rows: int = 1000,
    log_every_seconds: float = 20.0,
    log_label: str = "ANALOG_KNN",
) -> AnalogPosterior:
    n_query = len(query_indices)
    p_peak = np.full(n_query, np.nan, dtype=float)
    p_delta = np.full((n_query, delta_class_max), np.nan, dtype=float)
    q_score = np.full(n_query, np.nan, dtype=float)
    candidate_count = np.zeros(n_query, dtype=float)
    non_peak_count = np.zeros(n_query, dtype=float)
    active_logger = logger or logging.getLogger(__name__)
    tracker = ProgressTracker(
        logger=active_logger,
        name=log_label,
        total=n_query,
        log_every_rows=log_every_rows,
        log_every_seconds=log_every_seconds,
    )

    for qi, row_idx in enumerate(query_indices):
        q_date = library.dates[row_idx]
        q_cutoff = int(library.cutoff_minutes[row_idx])
        q_doy = int(library.doy[row_idx])
        q_vec = library.features_std[row_idx]

        pool_idx = library.index_by_cutoff.get(q_cutoff)
        if pool_idx is None or len(pool_idx) == 0:
            continue

        causal_mask = library.dates[pool_idx] < q_date
        if not np.any(causal_mask):
            continue
        pool_idx = pool_idx[causal_mask]
        doy_dist = _doy_distance_wrap(q_doy, library.doy[pool_idx])
        pool_idx = pool_idx[doy_dist <= season_window_doy]
        if len(pool_idx) < min_pool:
            continue

        candidate_count[qi] = float(len(pool_idx))

        diff = library.features_std[pool_idx] - q_vec[None, :]
        weighted_sq = diff * diff * standardizer.weight[None, :]
        dists = np.sqrt(np.sum(weighted_sq, axis=1))
        order = np.argsort(dists)
        k_eff = min(int(k), len(order))
        sel = order[:k_eff]
        sel_idx = pool_idx[sel]
        sel_dist = dists[sel]

        tau = float(np.median(sel_dist)) if np.isfinite(np.median(sel_dist)) else 1.0
        tau = max(tau, 1e-6)
        w = np.exp(-sel_dist / tau)
        if not np.isfinite(w).all() or np.sum(w) <= 0:
            w = 1.0 / np.maximum(sel_dist, 1e-6)
        w = w / np.sum(w)

        q_score[qi] = float(np.median(sel_dist))
        peak_sel = library.peak[sel_idx].astype(float)
        p_peak[qi] = float(np.sum(w * peak_sel))

        non_peak_sel = library.peak[sel_idx] == 0
        non_peak_count[qi] = float(np.sum(non_peak_sel))
        if np.sum(non_peak_sel) >= min_non_peak:
            w_np = w[non_peak_sel]
            w_np = w_np / np.sum(w_np)
            delta_np = library.delta_class[sel_idx][non_peak_sel]
        else:
            valid_delta = library.delta_class[sel_idx] >= 1
            if not np.any(valid_delta):
                continue
            w_np = w[valid_delta]
            w_np = w_np / np.sum(w_np)
            delta_np = library.delta_class[sel_idx][valid_delta]

        hist = np.zeros(delta_class_max, dtype=float)
        for cls, ww in zip(delta_np, w_np):
            cls = int(cls)
            if cls < 1:
                continue
            hist[min(cls, delta_class_max) - 1] += float(ww)
        if np.sum(hist) <= 0:
            tracker.maybe_log(
                qi + 1,
                extra=f"k={k} q_score={q_score[qi]:.3f} candidates={candidate_count[qi]:.0f} non_peak={non_peak_count[qi]:.0f}",
            )
            continue
        p_delta[qi] = hist / np.sum(hist)
        tracker.maybe_log(
            qi + 1,
            extra=f"k={k} q_score={q_score[qi]:.3f} candidates={candidate_count[qi]:.0f} non_peak={non_peak_count[qi]:.0f}",
        )

    finite_cand = candidate_count[np.isfinite(candidate_count)]
    finite_non_peak = non_peak_count[np.isfinite(non_peak_count)]
    mean_candidates = float(np.mean(finite_cand)) if finite_cand.size else float("nan")
    mean_non_peak = float(np.mean(finite_non_peak)) if finite_non_peak.size else float("nan")
    tracker.done(extra=f"k={k} mean_candidates={mean_candidates:.1f} mean_non_peak={mean_non_peak:.1f}")
    return AnalogPosterior(
        p_peak=p_peak,
        p_delta_cond=p_delta,
        q_score=q_score,
        candidate_count=candidate_count,
        non_peak_count=non_peak_count,
    )


def calibrate_blend_bounds(q_score: np.ndarray) -> tuple[float, float]:
    finite = q_score[np.isfinite(q_score)]
    if finite.size == 0:
        return 0.0, 1.0
    q_low = float(np.quantile(finite, 0.25))
    q_high = float(np.quantile(finite, 0.75))
    if q_high <= q_low:
        q_high = q_low + 1e-6
    return q_low, q_high


def blend_posteriors(
    *,
    p_peak_lgbm: np.ndarray,
    p_delta_lgbm: np.ndarray,
    p_peak_knn: np.ndarray,
    p_delta_knn: np.ndarray,
    q_score: np.ndarray,
    q_low: float,
    q_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_peak_lgbm = np.asarray(p_peak_lgbm, dtype=float)
    p_delta_lgbm = np.asarray(p_delta_lgbm, dtype=float)
    p_peak_knn = np.asarray(p_peak_knn, dtype=float)
    p_delta_knn = np.asarray(p_delta_knn, dtype=float)
    q_score = np.asarray(q_score, dtype=float)

    w_lgbm = np.ones_like(p_peak_lgbm, dtype=float)
    finite_q = np.isfinite(q_score)
    w_lgbm[finite_q] = (q_score[finite_q] - q_low) / max((q_high - q_low), 1e-6)
    w_lgbm = np.clip(w_lgbm, 0.0, 1.0)

    missing_knn = ~np.isfinite(p_peak_knn)
    w_lgbm[missing_knn] = 1.0

    p_peak = w_lgbm * p_peak_lgbm + (1.0 - w_lgbm) * np.nan_to_num(p_peak_knn, nan=0.0)
    p_peak = np.clip(p_peak, 0.0, 1.0)

    p_delta = p_delta_lgbm.copy()
    for i in range(len(p_delta)):
        if w_lgbm[i] >= 1.0 or not np.isfinite(np.sum(p_delta_knn[i])):
            continue
        mix = w_lgbm[i] * p_delta_lgbm[i] + (1.0 - w_lgbm[i]) * p_delta_knn[i]
        s = np.sum(mix)
        if s <= 0:
            continue
        p_delta[i] = mix / s

    return p_peak, p_delta, w_lgbm
