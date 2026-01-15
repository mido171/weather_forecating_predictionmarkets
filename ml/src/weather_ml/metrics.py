"""Metrics helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    bias = float(np.mean(y_pred - y_true))
    abs_err = np.abs(y_pred - y_true)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": bias,
        "p50_abs_error": float(np.quantile(abs_err, 0.5)),
        "p90_abs_error": float(np.quantile(abs_err, 0.9)),
        "p95_abs_error": float(np.quantile(abs_err, 0.95)),
    }


def log_loss_from_pmf(y_true: np.ndarray, pmf: np.ndarray, *, support_min: int) -> float:
    y_indices = (y_true.astype(int) - support_min).clip(0, pmf.shape[1] - 1)
    probs = pmf[np.arange(len(y_true)), y_indices]
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.mean(np.log(probs)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 0.0, 1.0)
    return float(brier_score_loss(y_true, y_prob))


def per_station_metrics(
    df,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    stations = df["station_id"].to_numpy()
    for station in np.unique(stations):
        mask = stations == station
        metrics[str(station)] = regression_metrics(y_true[mask], y_pred[mask])
    return metrics


def event_indicator(y_true: np.ndarray, spec: dict) -> np.ndarray:
    if spec.get("type") == "threshold":
        if "lt" in spec:
            return y_true < float(spec["lt"])
        if "ge" in spec:
            return y_true >= float(spec["ge"])
    if spec.get("type") == "range":
        start = float(spec["start"])
        end = float(spec["end"])
        return (y_true >= start) & (y_true <= end)
    raise ValueError(f"Unsupported event spec: {spec}")
