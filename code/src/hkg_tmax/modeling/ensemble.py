"""Constrained residual ensemble utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fit_nonnegative_weights(
    frame: pd.DataFrame,
    residual_prediction_columns: list[str],
) -> dict[str, Any]:
    if not residual_prediction_columns:
        return {"weights": {}, "lambda": 0.0, "status": "no_candidates"}
    y = pd.to_numeric(frame["y_true_c"], errors="coerce").to_numpy(dtype=float)
    anchor = pd.to_numeric(frame["anchor_forecast_max_c"], errors="coerce").to_numpy(dtype=float)
    residual_matrix = frame[residual_prediction_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(anchor) & np.isfinite(residual_matrix).all(axis=1)
    if mask.sum() < 50:
        zero_weights = {col: (1.0 if col == "resid_M0_zero" else 0.0) for col in residual_prediction_columns}
        return {"weights": zero_weights, "lambda": 0.0, "status": "insufficient_rows"}
    y = y[mask]
    anchor = anchor[mask]
    residual_matrix = residual_matrix[mask]
    n = len(residual_prediction_columns)

    def objective(weights: np.ndarray) -> float:
        pred = anchor + residual_matrix @ weights
        return float(np.mean(np.abs(pred - y)))

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    initial = np.ones(n, dtype=float) / n
    result = minimize(objective, initial, bounds=bounds, constraints=constraints, method="SLSQP")
    weights = result.x if result.success else initial
    raw_official_mae = float(np.mean(np.abs(anchor - y)))
    best_lambda = 0.0
    best_mae = raw_official_mae
    blend = residual_matrix @ weights
    for shrinkage in np.linspace(0.0, 1.0, 21):
        pred = anchor + shrinkage * blend
        mae = float(np.mean(np.abs(pred - y)))
        err = pred - y
        raw_err = anchor - y
        rmse_delta = float(np.sqrt(np.mean(err * err)) - np.sqrt(np.mean(raw_err * raw_err)))
        p90_delta = float(np.quantile(np.abs(err), 0.90) - np.quantile(np.abs(raw_err), 0.90))
        bias_delta = float(abs(np.mean(err)) - abs(np.mean(raw_err)))
        if mae <= best_mae and rmse_delta <= 0.010 and p90_delta <= 0.020 and bias_delta <= 0.030:
            best_mae = mae
            best_lambda = float(shrinkage)
    return {
        "weights": {col: float(weight) for col, weight in zip(residual_prediction_columns, weights, strict=True)},
        "lambda": best_lambda,
        "status": "fit" if result.success else "fallback_initial",
        "validation_mae": best_mae,
        "validation_raw_official_mae": raw_official_mae,
    }


def apply_ensemble(frame: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    anchor = pd.to_numeric(frame["anchor_forecast_max_c"], errors="coerce").to_numpy(dtype=float)
    residual = np.zeros(len(frame), dtype=float)
    for column, weight in model.get("weights", {}).items():
        if column in frame:
            residual += float(weight) * pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return anchor + float(model.get("lambda", 0.0)) * residual

