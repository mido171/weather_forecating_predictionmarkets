"""Residual model wrappers used by the HKG Tmax residual-ML runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # CatBoost is a declared dependency, but keep import failure explicit in artifacts.
    from catboost import CatBoostRegressor, Pool
except Exception:  # pragma: no cover - environment fallback
    CatBoostRegressor = None
    Pool = None


@dataclass
class FittedModel:
    model_id: str
    estimator: Any
    feature_names: list[str]
    encoded_feature_names: list[str]
    medians: dict[str, float]
    categorical_columns: list[str]
    status: str = "fit"
    diagnostics: dict[str, Any] | None = None


def categorical_columns(frame: pd.DataFrame, feature_names: list[str]) -> list[str]:
    explicit = {
        "official_max_bin",
        "official_range_bin",
        "issue_hour_bucket",
        "season_bucket",
        "cutoff_profile",
    }
    return [col for col in feature_names if col in frame.columns and (col in explicit or frame[col].dtype == "object")]


def prepare_design(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], list[str]]:
    cat_cols = categorical_columns(train, feature_names)
    numeric_cols = [col for col in feature_names if col not in cat_cols]
    medians: dict[str, float] = {}
    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    if numeric_cols:
        train_num = train[numeric_cols].apply(pd.to_numeric, errors="coerce")
        valid_num = valid[numeric_cols].apply(pd.to_numeric, errors="coerce")
        for col in numeric_cols:
            median = float(train_num[col].median()) if train_num[col].notna().any() else 0.0
            medians[col] = median
            train_num[col] = train_num[col].fillna(median)
            valid_num[col] = valid_num[col].fillna(median)
        train_parts.append(train_num)
        valid_parts.append(valid_num)
    if cat_cols:
        train_cat = train[cat_cols].astype("object").where(train[cat_cols].notna(), "MISSING").astype(str)
        valid_cat = valid[cat_cols].astype("object").where(valid[cat_cols].notna(), "MISSING").astype(str)
        combined = pd.get_dummies(pd.concat([train_cat, valid_cat], axis=0), dummy_na=False)
        train_parts.append(combined.iloc[: len(train)].reset_index(drop=True))
        valid_parts.append(combined.iloc[len(train):].reset_index(drop=True))
    x_train = pd.concat([part.reset_index(drop=True) for part in train_parts], axis=1) if train_parts else pd.DataFrame(index=train.index)
    x_valid = pd.concat([part.reset_index(drop=True) for part in valid_parts], axis=1) if valid_parts else pd.DataFrame(index=valid.index)
    x_train.columns = [str(col) for col in x_train.columns]
    x_valid.columns = [str(col) for col in x_valid.columns]
    return x_train, x_valid.reindex(columns=x_train.columns, fill_value=0.0), medians, cat_cols


def apply_design(frame: pd.DataFrame, fitted: FittedModel) -> pd.DataFrame:
    cat_cols = fitted.categorical_columns
    numeric_cols = [col for col in fitted.feature_names if col not in cat_cols]
    parts: list[pd.DataFrame] = []
    if numeric_cols:
        num = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
        for col in numeric_cols:
            num[col] = num[col].fillna(fitted.medians.get(col, 0.0))
        parts.append(num)
    if cat_cols:
        cat = frame[cat_cols].astype("object").where(frame[cat_cols].notna(), "MISSING").astype(str)
        parts.append(pd.get_dummies(cat, dummy_na=False))
    design = pd.concat([part.reset_index(drop=True) for part in parts], axis=1) if parts else pd.DataFrame(index=frame.index)
    design.columns = [str(col) for col in design.columns]
    return design.reindex(columns=fitted.encoded_feature_names, fill_value=0.0)


def fit_lgbm_residual(train: pd.DataFrame, valid: pd.DataFrame, feature_names: list[str], seed: int) -> tuple[np.ndarray, FittedModel]:
    x_train, x_valid, medians, cat_cols = prepare_design(train, valid, feature_names)
    y_train = pd.to_numeric(train["residual_y_c"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=120,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=5,
        min_child_samples=80,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=20.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    fitted = FittedModel(
        model_id="M2_lgbm_residual",
        estimator=model,
        feature_names=feature_names,
        encoded_feature_names=list(x_train.columns),
        medians=medians,
        categorical_columns=cat_cols,
    )
    return np.asarray(pred, dtype=float), fitted


def fit_direct_lgbm(train: pd.DataFrame, valid: pd.DataFrame, feature_names: list[str], seed: int) -> tuple[np.ndarray, FittedModel]:
    x_train, x_valid, medians, cat_cols = prepare_design(train, valid, feature_names)
    y_train = pd.to_numeric(train["y_true_c"], errors="coerce").to_numpy(dtype=float)
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=120,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=5,
        min_child_samples=80,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=20.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    fitted = FittedModel(
        model_id="B4_direct_lgbm_absolute",
        estimator=model,
        feature_names=feature_names,
        encoded_feature_names=list(x_train.columns),
        medians=medians,
        categorical_columns=cat_cols,
    )
    return np.asarray(pred, dtype=float), fitted


def fit_huber_residual(train: pd.DataFrame, valid: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, FittedModel]:
    x_train, x_valid, medians, cat_cols = prepare_design(train, valid, feature_names)
    y_train = pd.to_numeric(train["residual_y_c"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    model = Pipeline(
        [
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("huber", HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=2000, tol=1e-4)),
        ]
    )
    status = "huber_converged"
    diagnostics: dict[str, Any] = {"fallback_used": False, "estimator": "HuberRegressor"}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_train, y_train)
        convergence_warnings = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
        if convergence_warnings:
            raise RuntimeError("; ".join(str(item.message) for item in convergence_warnings))
    except Exception as exc:
        status = "fallback_ridge_after_huber_warning_or_failure"
        diagnostics = {
            "fallback_used": True,
            "estimator": "Ridge",
            "huber_failure": str(exc),
        }
        model = Pipeline(
            [
                ("scale", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=5.0)),
            ]
        )
        model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    fitted = FittedModel(
        model_id="M4_huber_residual",
        estimator=model,
        feature_names=feature_names,
        encoded_feature_names=list(x_train.columns),
        medians=medians,
        categorical_columns=cat_cols,
        status=status,
        diagnostics=diagnostics,
    )
    return np.asarray(pred, dtype=float), fitted


def fit_robust_linear_residual(train: pd.DataFrame, valid: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, FittedModel]:
    pred, fitted = fit_huber_residual(train, valid, feature_names)
    fitted.model_id = "M4_robust_linear_residual"
    return pred, fitted


def fit_lgbm_classifier(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: list[str],
    label_column: str,
    seed: int,
    *,
    model_id: str = "lgbm_classifier",
) -> tuple[np.ndarray, FittedModel]:
    x_train, x_valid, medians, cat_cols = prepare_design(train, valid, feature_names)
    y_train = pd.to_numeric(train[label_column], errors="coerce").fillna(0).astype(int).to_numpy()
    unique = sorted(set(y_train.tolist()))
    if len(unique) < 2:
        probability = float(unique[0]) if unique else 0.0
        fitted = FittedModel(
            model_id=model_id,
            estimator=None,
            feature_names=feature_names,
            encoded_feature_names=list(x_train.columns),
            medians=medians,
            categorical_columns=cat_cols,
            status="constant_single_class",
            diagnostics={"constant_probability": probability, "label_column": label_column},
        )
        return np.full(len(valid), probability, dtype=float), fitted
    model = LGBMClassifier(
        objective="binary",
        n_estimators=140,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=15.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    pred = model.predict_proba(x_valid)[:, 1]
    fitted = FittedModel(
        model_id=model_id,
        estimator=model,
        feature_names=feature_names,
        encoded_feature_names=list(x_train.columns),
        medians=medians,
        categorical_columns=cat_cols,
        diagnostics={"label_column": label_column},
    )
    return np.asarray(pred, dtype=float), fitted


def fit_lgbm_benefit_regressor(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: list[str],
    seed: int,
    *,
    label_column: str = "benefit_c",
    model_id: str = "lgbm_benefit_regressor",
) -> tuple[np.ndarray, FittedModel]:
    x_train, x_valid, medians, cat_cols = prepare_design(train, valid, feature_names)
    y_train = pd.to_numeric(train[label_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=140,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=15.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    fitted = FittedModel(
        model_id=model_id,
        estimator=model,
        feature_names=feature_names,
        encoded_feature_names=list(x_train.columns),
        medians=medians,
        categorical_columns=cat_cols,
        diagnostics={"label_column": label_column},
    )
    return np.asarray(pred, dtype=float), fitted


def fit_catboost_residual(train: pd.DataFrame, valid: pd.DataFrame, feature_names: list[str], seed: int) -> tuple[np.ndarray, FittedModel]:
    if CatBoostRegressor is None or Pool is None:
        return np.zeros(len(valid), dtype=float), FittedModel(
            model_id="M3_catboost_residual",
            estimator=None,
            feature_names=feature_names,
            encoded_feature_names=feature_names,
            medians={},
            categorical_columns=[],
            status="skipped_missing_catboost",
        )
    cat_cols = categorical_columns(train, feature_names)
    x_train = train[feature_names].copy()
    x_valid = valid[feature_names].copy()
    for col in cat_cols:
        x_train[col] = x_train[col].astype("object").where(x_train[col].notna(), "MISSING").astype(str)
        x_valid[col] = x_valid[col].astype("object").where(x_valid[col].notna(), "MISSING").astype(str)
    for col in feature_names:
        if col not in cat_cols:
            x_train[col] = pd.to_numeric(x_train[col], errors="coerce")
            x_valid[col] = pd.to_numeric(x_valid[col], errors="coerce")
    cat_idx = [x_train.columns.get_loc(col) for col in cat_cols]
    y_train = pd.to_numeric(train["residual_y_c"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    model = CatBoostRegressor(
        loss_function="MAE",
        iterations=120,
        depth=4,
        learning_rate=0.04,
        l2_leaf_reg=10.0,
        random_strength=0.5,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Pool(x_train, y_train, cat_features=cat_idx))
    pred = model.predict(Pool(x_valid, cat_features=cat_idx))
    fitted = FittedModel(
        model_id="M3_catboost_residual",
        estimator=model,
        feature_names=feature_names,
        encoded_feature_names=feature_names,
        medians={},
        categorical_columns=cat_cols,
    )
    return np.asarray(pred, dtype=float), fitted


def predict_with_fitted(frame: pd.DataFrame, fitted: FittedModel) -> np.ndarray:
    if fitted.estimator is None:
        if fitted.diagnostics and "constant_probability" in fitted.diagnostics:
            return np.full(len(frame), float(fitted.diagnostics["constant_probability"]), dtype=float)
        return np.zeros(len(frame), dtype=float)
    if fitted.model_id == "M3_catboost_residual":
        x = frame[fitted.feature_names].copy()
        for col in fitted.categorical_columns:
            x[col] = x[col].astype("object").where(x[col].notna(), "MISSING").astype(str)
        cat_idx = [x.columns.get_loc(col) for col in fitted.categorical_columns]
        pred = fitted.estimator.predict(Pool(x, cat_features=cat_idx))
        return np.asarray(pred, dtype=float)
    x = apply_design(frame, fitted)
    return np.asarray(fitted.estimator.predict(x), dtype=float)


def predict_proba_with_fitted(frame: pd.DataFrame, fitted: FittedModel) -> np.ndarray:
    if fitted.estimator is None:
        probability = 0.0
        if fitted.diagnostics and "constant_probability" in fitted.diagnostics:
            probability = float(fitted.diagnostics["constant_probability"])
        return np.full(len(frame), probability, dtype=float)
    x = apply_design(frame, fitted)
    if hasattr(fitted.estimator, "predict_proba"):
        return np.asarray(fitted.estimator.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(fitted.estimator.predict(x), dtype=float)


def feature_importance_frame(fitted: FittedModel) -> pd.DataFrame:
    if fitted.estimator is None:
        return pd.DataFrame({"feature": [], "importance": [], "model_id": []})
    if hasattr(fitted.estimator, "feature_importances_"):
        values = np.asarray(fitted.estimator.feature_importances_, dtype=float)
        names = fitted.encoded_feature_names
    elif fitted.model_id == "M3_catboost_residual":
        values = np.asarray(fitted.estimator.get_feature_importance(), dtype=float)
        names = fitted.feature_names
    else:
        estimator = getattr(fitted.estimator, "named_steps", {}).get("huber")
        if estimator is None:
            estimator = getattr(fitted.estimator, "named_steps", {}).get("ridge")
        values = np.asarray(getattr(estimator, "coef_", []), dtype=float)
        names = fitted.encoded_feature_names
    return pd.DataFrame(
        {
            "model_id": fitted.model_id,
            "feature": names[: len(values)],
            "importance": values[: len(names)],
        }
    ).sort_values("importance", key=lambda s: s.abs(), ascending=False)
