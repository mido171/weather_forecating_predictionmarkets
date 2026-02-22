"""Analog/kNN feature helpers for exp30."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

from weather_ml.mos_utils import effective_sample_size

LOGGER = logging.getLogger(__name__)

MAX_ANALOG_SECONDS: float | None = None
LOG_EVERY_ROWS = 200


def configure_runtime(
    *,
    max_seconds: float | None = None,
    log_every_rows: int | None = None,
) -> None:
    global MAX_ANALOG_SECONDS, LOG_EVERY_ROWS
    if max_seconds is not None:
        MAX_ANALOG_SECONDS = float(max_seconds)
    if log_every_rows is not None:
        LOG_EVERY_ROWS = int(log_every_rows)


@dataclass
class KNNResult:
    predictions: dict[int, np.ndarray]
    diagnostics: dict[str, np.ndarray]


def standardize_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    values = df[feature_cols].to_numpy(dtype=float)
    train_vals = values[train_mask]
    mean = np.nanmean(train_vals, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.nanstd(train_vals, axis=0)
    std = np.where(std == 0, 1.0, std)
    values = np.where(np.isfinite(values), values, mean)
    values = (values - mean) / std
    return values, {"mean": mean.tolist(), "std": std.tolist(), "columns": feature_cols}


def compute_knn_analogs(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    target: np.ndarray,
    *,
    ks: Iterable[int],
    group_key: pd.Series,
    min_pool: int = 200,
    max_history_days: int | None = None,
) -> KNNResult:
    df = df.copy()
    dates = pd.to_datetime(df["target_date_local"]).dt.date
    ks = sorted(set(int(k) for k in ks))
    preds = {k: np.full(len(df), np.nan, dtype=float) for k in ks}
    dist_q50 = np.full(len(df), np.nan, dtype=float)
    dist_q25 = np.full(len(df), np.nan, dtype=float)
    dist_q75 = np.full(len(df), np.nan, dtype=float)
    eff_n = np.full(len(df), np.nan, dtype=float)
    truth_std = np.full(len(df), np.nan, dtype=float)

    groups = group_key.to_numpy()
    start_time = time.time()
    processed = 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        # sort idx by date
        idx = idx[np.argsort(dates[idx])]
        LOGGER.info("KNN_GROUP_START group=%s rows=%d", g, len(idx))
        for pos, row_idx in enumerate(idx):
            if (
                MAX_ANALOG_SECONDS is not None
                and time.time() - start_time > MAX_ANALOG_SECONDS
            ):
                LOGGER.warning(
                    "KNN analog budget exceeded after %d rows (%.1fs).",
                    processed,
                    time.time() - start_time,
                )
                diagnostics = {
                    "analog_dist_q25": dist_q25,
                    "analog_dist_q50": dist_q50,
                    "analog_dist_q75": dist_q75,
                    "analog_eff_n": eff_n,
                    "analog_truth_std": truth_std,
                }
                return KNNResult(predictions=preds, diagnostics=diagnostics)
            if pos == 0:
                continue
            past_idx = idx[:pos]
            if max_history_days is not None:
                cutoff = dates[row_idx] - pd.Timedelta(days=max_history_days)
                past_idx = past_idx[dates[past_idx] >= cutoff]
            if len(past_idx) < min_pool:
                continue
            current = feature_matrix[row_idx]
            past_matrix = feature_matrix[past_idx]
            # compute distances
            dists = np.linalg.norm(past_matrix - current, axis=1)
            order = np.argsort(dists)
            for k in ks:
                k_eff = min(k, len(order))
                sel = order[:k_eff]
                sel_dists = dists[sel]
                weights = 1.0 / (sel_dists + 1e-3)
                weights = weights / np.sum(weights)
                preds[k][row_idx] = float(np.sum(weights * target[past_idx][sel]))
                if k == ks[0]:
                    dist_q50[row_idx] = float(np.quantile(sel_dists, 0.5))
                    dist_q25[row_idx] = float(np.quantile(sel_dists, 0.25))
                    dist_q75[row_idx] = float(np.quantile(sel_dists, 0.75))
                    eff_n[row_idx] = effective_sample_size(weights)
                    truth_std[row_idx] = float(np.sqrt(np.sum(weights * (target[past_idx][sel] - preds[k][row_idx]) ** 2)))
            processed += 1
            if LOG_EVERY_ROWS and processed % LOG_EVERY_ROWS == 0:
                LOGGER.info(
                    "KNN_PROGRESS processed=%d elapsed=%.1fs",
                    processed,
                    time.time() - start_time,
                )
        LOGGER.info("KNN_GROUP_END group=%s processed=%d", g, processed)

    diagnostics = {
        "analog_dist_q25": dist_q25,
        "analog_dist_q50": dist_q50,
        "analog_dist_q75": dist_q75,
        "analog_eff_n": eff_n,
        "analog_truth_std": truth_std,
    }
    return KNNResult(predictions=preds, diagnostics=diagnostics)


def compute_local_ridge(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    target: np.ndarray,
    *,
    ks: Iterable[int],
    alphas: Iterable[float],
    group_key: pd.Series,
    min_pool: int = 150,
) -> dict[tuple[int, float], np.ndarray]:
    dates = pd.to_datetime(df["target_date_local"]).dt.date
    groups = group_key.to_numpy()
    results: dict[tuple[int, float], np.ndarray] = {}
    for k in ks:
        for alpha in alphas:
            results[(k, alpha)] = np.full(len(df), np.nan, dtype=float)
    start_time = time.time()
    processed = 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        idx = idx[np.argsort(dates[idx])]
        LOGGER.info("LOCAL_RIDGE_GROUP_START group=%s rows=%d", g, len(idx))
        for pos, row_idx in enumerate(idx):
            if (
                MAX_ANALOG_SECONDS is not None
                and time.time() - start_time > MAX_ANALOG_SECONDS
            ):
                LOGGER.warning(
                    "Local ridge budget exceeded after %d rows (%.1fs).",
                    processed,
                    time.time() - start_time,
                )
                return results
            if pos == 0:
                continue
            past_idx = idx[:pos]
            if len(past_idx) < min_pool:
                continue
            current = feature_matrix[row_idx]
            past_matrix = feature_matrix[past_idx]
            dists = np.linalg.norm(past_matrix - current, axis=1)
            order = np.argsort(dists)
            for k in ks:
                k_eff = min(k, len(order))
                sel = order[:k_eff]
                weights = 1.0 / (dists[sel] + 1e-3)
                weights = weights / np.sum(weights)
                X = past_matrix[sel]
                y = target[past_idx][sel]
                for alpha in alphas:
                    model = Ridge(alpha=alpha)
                    model.fit(X, y, sample_weight=weights)
                    results[(k, alpha)][row_idx] = float(model.predict(current.reshape(1, -1))[0])
            processed += 1
            if LOG_EVERY_ROWS and processed % LOG_EVERY_ROWS == 0:
                LOGGER.info(
                    "LOCAL_RIDGE_PROGRESS processed=%d elapsed=%.1fs",
                    processed,
                    time.time() - start_time,
                )
        LOGGER.info("LOCAL_RIDGE_GROUP_END group=%s processed=%d", g, processed)
    return results


def compute_prototypes(
    feature_matrix: np.ndarray,
    residuals: np.ndarray,
    *,
    train_mask: np.ndarray,
    k_prototypes: int,
) -> dict:
    LOGGER.info("PROTOTYPE_FIT_START k=%d", k_prototypes)
    train_x = feature_matrix[train_mask]
    train_res = residuals[train_mask]
    if train_x.shape[0] < k_prototypes:
        raise ValueError("Not enough data to fit prototypes.")
    km = KMeans(n_clusters=k_prototypes, random_state=42, n_init=10)
    km.fit(train_x)
    centers = km.cluster_centers_
    # find medoid (closest actual row) for each center
    medoid_idx = []
    for center in centers:
        dists = np.linalg.norm(train_x - center, axis=1)
        medoid_idx.append(int(np.argmin(dists)))
    medoid_vecs = train_x[medoid_idx]
    assignments = km.predict(train_x)
    proto_resid = {}
    for p in range(k_prototypes):
        vals = train_res[assignments == p]
        proto_resid[p] = float(np.nanmean(vals)) if vals.size else 0.0
    LOGGER.info("PROTOTYPE_FIT_END k=%d", k_prototypes)
    return {
        "centers": centers,
        "medoids": medoid_vecs,
        "proto_resid": proto_resid,
        "k": k_prototypes,
    }


def assign_prototypes(
    feature_matrix: np.ndarray,
    proto: dict,
) -> tuple[np.ndarray, np.ndarray]:
    centers = proto["medoids"]
    dists = np.linalg.norm(feature_matrix[:, None, :] - centers[None, :, :], axis=2)
    proto_id = np.argmin(dists, axis=1)
    proto_dist = np.min(dists, axis=1)
    return proto_id, proto_dist
