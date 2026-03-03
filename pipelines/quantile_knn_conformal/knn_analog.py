from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: list[float]) -> np.ndarray:
    if len(values) == 0:
        return np.full(len(qs), np.nan)
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    wsum = np.sum(w)
    if wsum <= 0:
        w = np.full_like(w, 1.0 / len(w), dtype=float)
    else:
        w = w / wsum
    cdf = np.cumsum(w)
    out = []
    for q in qs:
        idx = np.searchsorted(cdf, q, side="left")
        idx = min(idx, len(v) - 1)
        out.append(v[idx])
    return np.array(out, dtype=float)


def _entropy(weights: np.ndarray) -> float:
    w = weights[weights > 0]
    if len(w) == 0:
        return 0.0
    return float(-np.sum(w * np.log(w + 1e-12)))


@dataclass
class KNNAnalogModel:
    feature_cols: list[str]
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    y_train: np.ndarray
    train_dates: np.ndarray
    train_ids: np.ndarray
    nn: NearestNeighbors
    raw_x_train: np.ndarray
    k: int


def fit_knn_analog(train_df: pd.DataFrame, feature_cols: list[str], k_neighbors: int) -> KNNAnalogModel:
    x = train_df[feature_cols].to_numpy(dtype=float)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[~np.isfinite(std) | (std < 1e-8)] = 1.0
    x_imp = np.where(np.isfinite(x), x, mean)
    xz = (x_imp - mean) / std

    nn = NearestNeighbors(n_neighbors=min(len(train_df), max(k_neighbors * 4, k_neighbors)), metric="euclidean")
    nn.fit(xz)

    return KNNAnalogModel(
        feature_cols=feature_cols,
        scaler_mean=mean,
        scaler_std=std,
        y_train=train_df["y_tmax"].to_numpy(dtype=float),
        train_dates=pd.to_datetime(train_df["target_date_local"]).to_numpy(),
        train_ids=train_df.index.to_numpy(),
        nn=nn,
        raw_x_train=xz,
        k=k_neighbors,
    )


def predict_knn_analog(
    model: KNNAnalogModel,
    query_df: pd.DataFrame,
    quantiles: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if query_df.empty:
        cols = [f"q_{q:.3f}" for q in quantiles]
        return pd.DataFrame(columns=cols), pd.DataFrame()

    x = query_df[model.feature_cols].to_numpy(dtype=float)
    x_imp = np.where(np.isfinite(x), x, model.scaler_mean)
    xz = (x_imp - model.scaler_mean) / model.scaler_std

    n_req = min(model.raw_x_train.shape[0], max(model.k * 5, model.k))
    dists, idxs = model.nn.kneighbors(xz, n_neighbors=n_req, return_distance=True)
    query_dates = pd.to_datetime(query_df["target_date_local"]).to_numpy()

    q_rows: list[dict[str, Any]] = []
    t_rows: list[dict[str, Any]] = []

    for i in range(len(query_df)):
        cand_idx = idxs[i]
        cand_dist = dists[i]
        valid = model.train_dates[cand_idx] < query_dates[i]
        cand_idx = cand_idx[valid]
        cand_dist = cand_dist[valid]

        if len(cand_idx) == 0:
            q_vals = np.full(len(quantiles), np.nan)
            trust = {
                "knn_dist_min": np.nan,
                "knn_dist_p10": np.nan,
                "knn_dist_mean": np.nan,
                "knn_weighted_iqr": np.nan,
                "knn_weighted_mad": np.nan,
                "knn_effective_k": 0.0,
                "knn_support_span": np.nan,
                "knn_entropy": np.nan,
            }
        else:
            cand_idx = cand_idx[: model.k]
            cand_dist = cand_dist[: model.k]
            scale = np.nanmedian(cand_dist)
            if not np.isfinite(scale) or scale <= 1e-9:
                scale = 1.0
            w = np.exp(-cand_dist / scale)
            sw = np.sum(w)
            if sw <= 0:
                w = np.full(len(cand_idx), 1.0 / len(cand_idx), dtype=float)
            else:
                w = w / sw

            y = model.y_train[cand_idx]
            q_vals = _weighted_quantile(y, w, quantiles)
            q25, q75 = _weighted_quantile(y, w, [0.25, 0.75])
            med = _weighted_quantile(y, w, [0.5])[0]
            mad = _weighted_quantile(np.abs(y - med), w, [0.5])[0]
            trust = {
                "knn_dist_min": float(np.nanmin(cand_dist)),
                "knn_dist_p10": float(np.nanquantile(cand_dist, 0.1)),
                "knn_dist_mean": float(np.nanmean(cand_dist)),
                "knn_weighted_iqr": float(q75 - q25),
                "knn_weighted_mad": float(mad),
                "knn_effective_k": float(1.0 / np.sum(np.square(w))),
                "knn_support_span": float(np.nanmax(y) - np.nanmin(y)),
                "knn_entropy": _entropy(w),
            }

        q_rows.append({f"q_{q:.3f}": float(v) for q, v in zip(quantiles, q_vals, strict=False)})
        t_rows.append(trust)

    return pd.DataFrame(q_rows, index=query_df.index), pd.DataFrame(t_rows, index=query_df.index)


def neighbor_diagnostics_sample(
    model: KNNAnalogModel,
    query_df: pd.DataFrame,
    sample_n: int = 200,
) -> pd.DataFrame:
    if query_df.empty:
        return pd.DataFrame()
    sample = query_df.sample(n=min(sample_n, len(query_df)), random_state=7).copy()
    x = sample[model.feature_cols].to_numpy(dtype=float)
    x_imp = np.where(np.isfinite(x), x, model.scaler_mean)
    xz = (x_imp - model.scaler_mean) / model.scaler_std

    n_req = min(model.raw_x_train.shape[0], max(model.k, 32))
    dists, idxs = model.nn.kneighbors(xz, n_neighbors=n_req, return_distance=True)

    rows = []
    q_dates = pd.to_datetime(sample["target_date_local"]).to_numpy()
    for i, idx in enumerate(sample.index):
        cand_idx = idxs[i]
        cand_dist = dists[i]
        valid = model.train_dates[cand_idx] < q_dates[i]
        cand_idx = cand_idx[valid][: model.k]
        cand_dist = cand_dist[valid][: model.k]
        for rank, (ci, cd) in enumerate(zip(cand_idx, cand_dist, strict=False), start=1):
            rows.append(
                {
                    "query_index": int(idx),
                    "query_date": str(sample.loc[idx, "target_date_local"]),
                    "neighbor_rank": rank,
                    "neighbor_train_index": int(model.train_ids[ci]),
                    "neighbor_date": str(pd.Timestamp(model.train_dates[ci]).date()),
                    "distance": float(cd),
                    "neighbor_y_tmax": float(model.y_train[ci]),
                }
            )
    return pd.DataFrame(rows)
