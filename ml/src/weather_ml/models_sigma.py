"""Sigma model training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone

from weather_ml.models_mean import get_mean_model, tune_model_timecv


@dataclass(frozen=True)
class SigmaArtifacts:
    sigma_model: Any
    mu_oof: np.ndarray
    z_target: np.ndarray


def build_oof_predictions(
    mean_model_name: str,
    *,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    param_grid: dict[str, list[Any]],
    fixed_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base_model = get_mean_model(mean_model_name, seed=seed)
    if fixed_params:
        base_model.set_params(**fixed_params)
        tuned = base_model
    else:
        tuned = tune_model_timecv(base_model, X, y, splits, param_grid).estimator
    mu_oof = np.full_like(y, fill_value=np.nan, dtype=float)
    for train_idx, val_idx in splits:
        estimator = clone(tuned)
        estimator.fit(X[train_idx], y[train_idx])
        mu_oof[val_idx] = estimator.predict(X[val_idx])
    oof_mask = ~np.isnan(mu_oof)
    return mu_oof, oof_mask


def build_sigma_targets(y_true: np.ndarray, mu_oof: np.ndarray, *, eps: float) -> np.ndarray:
    residual = y_true - mu_oof
    v_target = np.square(residual)
    return np.log(v_target + eps)


def fit_sigma_model(
    sigma_model_name: str,
    *,
    X: np.ndarray,
    z_target: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    param_grid: dict[str, list[Any]],
) -> Any:
    base_model = get_mean_model(sigma_model_name, seed=seed)
    tuned = tune_model_timecv(base_model, X, z_target, splits, param_grid)
    return tuned.estimator


def predict_sigma(
    sigma_model: Any,
    *,
    X: np.ndarray,
    eps: float,
    sigma_floor: float,
) -> np.ndarray:
    z_hat = sigma_model.predict(X)
    v_hat = np.exp(z_hat) - eps
    v_hat = np.maximum(v_hat, sigma_floor * sigma_floor)
    return np.sqrt(v_hat)
