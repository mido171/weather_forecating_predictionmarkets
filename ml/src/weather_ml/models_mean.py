"""Mean model training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


@dataclass(frozen=True)
class ModelResult:
    estimator: Any
    best_params: dict[str, Any]
    best_score: float


def get_mean_model(name: str, *, seed: int) -> Any:
    normalized = name.lower()
    if normalized == "ridge":
        return Ridge(random_state=seed)
    if normalized == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=1)
    if normalized in {"gbr", "gradient_boosting"}:
        return GradientBoostingRegressor(random_state=seed)
    if normalized == "lgbm":
        return _get_lightgbm(seed)
    if normalized == "xgb":
        return _get_xgboost(seed)
    if normalized == "catboost":
        return _get_catboost(seed)
    raise ValueError(f"Unknown mean model: {name}")


def tune_model_timecv(
    base_estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    param_grid: dict[str, list[Any]],
) -> ModelResult:
    if not param_grid:
        estimator = clone(base_estimator)
        estimator.fit(X, y)
        score = -mean_absolute_error(y, estimator.predict(X))
        return ModelResult(estimator=estimator, best_params={}, best_score=score)
    if not splits:
        estimator = clone(base_estimator)
        grid = _expand_grid(param_grid)
        best_params = grid[0] if grid else {}
        if best_params:
            estimator.set_params(**best_params)
        estimator.fit(X, y)
        score = -mean_absolute_error(y, estimator.predict(X))
        return ModelResult(estimator=estimator, best_params=best_params, best_score=score)

    best_score = float("-inf")
    best_params: dict[str, Any] = {}
    best_estimator = None
    for params in _expand_grid(param_grid):
        estimator = clone(base_estimator)
        estimator.set_params(**params)
        scores = []
        for train_idx, val_idx in splits:
            estimator.fit(X[train_idx], y[train_idx])
            preds = estimator.predict(X[val_idx])
            scores.append(-mean_absolute_error(y[val_idx], preds))
        score = float(np.mean(scores)) if scores else float("-inf")
        if score > best_score:
            best_score = score
            best_params = params
            best_estimator = clone(estimator)
    if best_estimator is None:
        best_estimator = clone(base_estimator)
    best_estimator.set_params(**best_params)
    best_estimator.fit(X, y)
    return ModelResult(estimator=best_estimator, best_params=best_params, best_score=best_score)


def _expand_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    if not keys:
        return []
    grids: list[dict[str, Any]] = [{}]
    for key in keys:
        values = param_grid[key]
        next_grids = []
        for grid in grids:
            for value in values:
                new_grid = dict(grid)
                new_grid[key] = value
                next_grids.append(new_grid)
        grids = next_grids
    return grids


def _get_lightgbm(seed: int):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is not installed.") from exc
    return lgb.LGBMRegressor(
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbose=-1,
    )


def _get_xgboost(seed: int):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is not installed.") from exc
    return xgb.XGBRegressor(
        random_state=seed,
        nthread=1,
        verbosity=0,
    )


def _get_catboost(seed: int):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError("catboost is not installed.") from exc
    return CatBoostRegressor(
        random_seed=seed,
        verbose=False,
        thread_count=1,
    )
