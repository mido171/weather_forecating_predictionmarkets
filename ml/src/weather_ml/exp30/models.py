"""Model training and hyperparameter search utilities for exp30."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

TRIALS_SCALE = 1.0
MAX_SEARCH_SECONDS: float | None = None
LOG_EVERY_TRIAL = True
LOG_EVERY_SEED = False


def configure_runtime(
    *,
    trials_scale: float | None = None,
    max_search_seconds: float | None = None,
    log_every_trial: bool | None = None,
    log_every_seed: bool | None = None,
) -> None:
    global TRIALS_SCALE, MAX_SEARCH_SECONDS, LOG_EVERY_TRIAL, LOG_EVERY_SEED
    if trials_scale is not None:
        TRIALS_SCALE = max(0.05, float(trials_scale))
    if max_search_seconds is not None:
        MAX_SEARCH_SECONDS = float(max_search_seconds)
    if log_every_trial is not None:
        LOG_EVERY_TRIAL = bool(log_every_trial)
    if log_every_seed is not None:
        LOG_EVERY_SEED = bool(log_every_seed)


@dataclass
class SearchResult:
    model_name: str
    params: dict[str, Any]
    median_val_mae: float
    seed_scores: list[float]
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


def _sample_int(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _sample_float(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def sample_lgbm_params(rng: np.random.Generator, objective: str) -> dict[str, Any]:
    return {
        "objective": objective,
        "num_leaves": _sample_int(rng, 31, 255),
        "max_depth": int(rng.choice([3, 4, 5, 6, 7, 8, 9, 10, -1])),
        "min_data_in_leaf": _sample_int(rng, 20, 400),
        "learning_rate": _sample_float(rng, 0.01, 0.12),
        "n_estimators": _sample_int(rng, 500, 5000),
        "feature_fraction": _sample_float(rng, 0.6, 1.0),
        "bagging_fraction": _sample_float(rng, 0.6, 1.0),
        "bagging_freq": _sample_int(rng, 1, 10),
        "reg_alpha": _sample_float(rng, 0.0, 10.0),
        "reg_lambda": _sample_float(rng, 0.0, 10.0),
    }


def sample_lgbm_l1_params(rng: np.random.Generator) -> dict[str, Any]:
    params = sample_lgbm_params(rng, "regression_l1")
    params["learning_rate"] = _sample_float(rng, 0.005, 0.08)
    params["n_estimators"] = _sample_int(rng, 800, 6000)
    return params


def sample_xgb_params(rng: np.random.Generator, objective: str) -> dict[str, Any]:
    return {
        "objective": objective,
        "max_depth": _sample_int(rng, 2, 8),
        "min_child_weight": _sample_int(rng, 1, 20),
        "subsample": _sample_float(rng, 0.6, 1.0),
        "colsample_bytree": _sample_float(rng, 0.6, 1.0),
        "eta": _sample_float(rng, 0.01, 0.15),
        "reg_lambda": _sample_float(rng, 0.0, 10.0),
        "reg_alpha": _sample_float(rng, 0.0, 10.0),
        "gamma": _sample_float(rng, 0.0, 5.0),
        "n_estimators": _sample_int(rng, 500, 6000),
    }


def sample_catboost_params(rng: np.random.Generator, loss: str) -> dict[str, Any]:
    return {
        "loss_function": loss,
        "depth": _sample_int(rng, 4, 10),
        "learning_rate": _sample_float(rng, 0.01, 0.2),
        "l2_leaf_reg": _sample_float(rng, 1.0, 20.0),
        "iterations": _sample_int(rng, 500, 8000),
        "bagging_temperature": _sample_float(rng, 0.0, 1.0),
        "random_strength": _sample_float(rng, 0.0, 2.0),
    }


def _fit_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> Any:
    model = _lgbm()
    model.set_params(random_state=seed, **params)
    try:
        import lightgbm as lgb
        callbacks = [lgb.early_stopping(200, verbose=False)]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=callbacks,
        )
    except Exception:
        # Fallback to fit without early stopping for compatibility.
        model.fit(X_train, y_train)
    return model


def _fit_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> Any:
    model = _xgb()
    model.set_params(random_state=seed, **params)
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            early_stopping_rounds=200,
            verbose=False,
        )
    except Exception:
        model.fit(X_train, y_train)
    return model


def _fit_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> Any:
    model = _catboost()
    model.set_params(random_seed=seed, **params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False,
    )
    return model


def _fit_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    seed: int,
) -> Ridge:
    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(X_train, y_train)
    return model


def train_with_search(
    *,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seeds: list[int],
    trials: int,
    param_sampler: Callable[[np.random.Generator], dict[str, Any]],
    rng: np.random.Generator,
) -> SearchResult:
    best = None
    effective_trials = max(1, int(round(trials * TRIALS_SCALE)))
    start_time = time.time()
    for trial_idx in range(effective_trials):
        if (
            MAX_SEARCH_SECONDS is not None
            and trial_idx > 0
            and time.time() - start_time > MAX_SEARCH_SECONDS
        ):
            LOGGER.warning(
                "Search budget exceeded for %s after %d/%d trials (%.1fs).",
                model_name,
                trial_idx,
                effective_trials,
                time.time() - start_time,
            )
            break
        params = param_sampler(rng)
        if LOG_EVERY_TRIAL:
            LOGGER.info(
                "SEARCH_TRIAL_START model=%s trial=%d/%d params=%s",
                model_name,
                trial_idx + 1,
                effective_trials,
                params,
            )
        scores = []
        last_model = None
        for seed in seeds:
            try:
                if model_name == "lgbm":
                    model = _fit_lgbm(X_train, y_train, X_val, y_val, params, seed)
                elif model_name == "xgb":
                    model = _fit_xgb(X_train, y_train, X_val, y_val, params, seed)
                elif model_name == "catboost":
                    model = _fit_catboost(X_train, y_train, X_val, y_val, params, seed)
                else:
                    raise ValueError(f"Unsupported model in search: {model_name}")
                preds = model.predict(X_val)
                score = float(mean_absolute_error(y_val, preds))
                scores.append(score)
                last_model = model
                if LOG_EVERY_SEED:
                    LOGGER.info(
                        "SEARCH_SEED_SCORE model=%s trial=%d seed=%d mae=%.4f",
                        model_name,
                        trial_idx + 1,
                        seed,
                        score,
                    )
            except Exception as exc:
                LOGGER.warning(
                    "Trial failed for %s trial=%d seed=%d: %s",
                    model_name,
                    trial_idx + 1,
                    seed,
                    exc,
                )
                scores.append(float("inf"))
        median = float(np.median(scores))
        if LOG_EVERY_TRIAL:
            LOGGER.info(
                "SEARCH_TRIAL_END model=%s trial=%d median_mae=%.4f",
                model_name,
                trial_idx + 1,
                median,
            )
        if last_model is None:
            continue
        if best is None or median < best.median_val_mae:
            best = SearchResult(
                model_name=model_name,
                params=params,
                median_val_mae=median,
                seed_scores=scores,
                estimator=last_model,
            )
    if best is None:
        raise RuntimeError("Model search failed to produce a candidate.")
    return best


def train_ridge_search(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seeds: list[int],
    alphas: list[float],
) -> SearchResult:
    best = None
    start_time = time.time()
    for idx, alpha in enumerate(alphas):
        if (
            MAX_SEARCH_SECONDS is not None
            and idx > 0
            and time.time() - start_time > MAX_SEARCH_SECONDS
        ):
            LOGGER.warning(
                "Ridge search budget exceeded after %d/%d alphas (%.1fs).",
                idx,
                len(alphas),
                time.time() - start_time,
            )
            break
        scores = []
        for seed in seeds:
            model = _fit_ridge(X_train, y_train, alpha=alpha, seed=seed)
            preds = model.predict(X_val)
            scores.append(float(mean_absolute_error(y_val, preds)))
        median = float(np.median(scores))
        if LOG_EVERY_TRIAL:
            LOGGER.info(
                "RIDGE_TRIAL alpha=%.4f median_mae=%.4f",
                alpha,
                median,
            )
        if best is None or median < best.median_val_mae:
            best = SearchResult(
                model_name="ridge",
                params={"alpha": alpha},
                median_val_mae=median,
                seed_scores=scores,
                estimator=model,
            )
    if best is None:
        raise RuntimeError("Ridge search failed.")
    return best


def refit_model(
    *,
    model_name: str,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Any:
    if model_name == "lgbm":
        model = _lgbm()
        model.set_params(random_state=seed, **params)
        model.fit(X_train, y_train)
        return model
    if model_name == "xgb":
        model = _xgb()
        model.set_params(random_state=seed, **params)
        model.fit(X_train, y_train)
        return model
    if model_name == "catboost":
        model = _catboost()
        model.set_params(random_seed=seed, **params)
        model.fit(X_train, y_train)
        return model
    if model_name == "ridge":
        alpha = float(params.get("alpha", 1.0))
        return _fit_ridge(X_train, y_train, alpha=alpha, seed=seed)
    raise ValueError(f"Unsupported model for refit: {model_name}")


def standardize_for_linear(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    # Impute NaNs with train means before scaling.
    train_means = np.nanmean(X_train, axis=0)
    train_means = np.where(np.isfinite(train_means), train_means, 0.0)
    X_train_filled = np.where(np.isfinite(X_train), X_train, train_means)
    X_val_filled = np.where(np.isfinite(X_val), X_val, train_means)
    X_test_filled = np.where(np.isfinite(X_test), X_test, train_means)
    scaler = StandardScaler()
    scaler.fit(X_train_filled)
    return (
        scaler.transform(X_train_filled),
        scaler.transform(X_val_filled),
        scaler.transform(X_test_filled),
        scaler,
    )
