"""Model training utilities for TFS2 sweep."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


@dataclass
class SearchResult:
    params: dict[str, Any]
    score: float
    estimator: Any


def _lgbm():
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbose=-1,
    )


def _xgb():
    import xgboost as xgb

    return xgb.XGBRegressor(nthread=1, verbosity=0)


def _catboost():
    from catboost import CatBoostRegressor

    return CatBoostRegressor(verbose=False, thread_count=1)


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    objective: str | None = None,
    early_stopping_rounds: int | None = None,
) -> Any:
    model = _lgbm()
    if objective:
        params = dict(params)
        params["objective"] = objective
    model.set_params(**params)
    if X_val is not None and y_val is not None and early_stopping_rounds:
        try:
            import lightgbm as lgb

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="mae",
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            return model
        except Exception:
            pass
    model.fit(X_train, y_train)
    return model


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> Any:
    model = _xgb()
    model.set_params(**params)
    if X_val is not None and y_val is not None:
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="mae",
                early_stopping_rounds=200,
                verbose=False,
            )
            return model
        except Exception:
            pass
    model.fit(X_train, y_train)
    return model


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> Any:
    model = _catboost()
    model.set_params(**params)
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    else:
        model.fit(X_train, y_train)
    return model


def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    seed: int = 0,
) -> Ridge:
    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(X_train, y_train)
    return model


def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_iter: int = 500,
    seed: int = 0,
    n_classes: int = 3,
) -> LogisticRegression:
    multi = "multinomial" if n_classes > 2 else "auto"
    model = LogisticRegression(max_iter=max_iter, random_state=seed, multi_class=multi)
    model.fit(X_train, y_train)
    return model


def standardize(X_train: np.ndarray, X_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_other), scaler


def search_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    candidates: list[dict[str, Any]],
    trainer: Callable[[np.ndarray, np.ndarray, dict[str, Any]], Any],
) -> SearchResult:
    best: SearchResult | None = None
    for params in candidates:
        model = trainer(X_train, y_train, params)
        preds = model.predict(X_val)
        mae = float(mean_absolute_error(y_val, preds))
        if best is None or mae < best.score:
            best = SearchResult(params=params, score=mae, estimator=model)
    if best is None:
        raise RuntimeError("No valid model candidates.")
    return best
