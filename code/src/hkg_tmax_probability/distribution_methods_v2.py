"""Distribution-method challengers for HKG Tmax probability engine V2.

This module deliberately sits next to, rather than inside, the V1 model
module.  The existing B4 hierarchical residual PMF remains the reference
champion and is imported read-only for comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge

from hkg_tmax_probability.bucket_rules import (
    BUCKET_KEYS,
    PROBABILITY_COLUMNS,
    bucket_boundaries_for_cdf,
    normalize_probability_matrix,
)
from hkg_tmax_probability.models import (
    RESIDUAL_SUPPORT_TENTHS,
    MethodOutput,
    _continuous_bucket_probs,
    _feature_matrix,
    _standardize_train_validation,
    chronological_inner_split,
    hierarchical_month_forecast_pmf_predict,
    predict_base_methods,
    residual_pmf_to_bucket_probs,
)
from hkg_tmax_probability.scoring import ranked_probability_score

V2_PREDICTOR_COLUMNS: tuple[str, ...] = (
    "forecast_max_c",
    "forecast_min_c",
    "forecast_range_c",
    "forecast_midpoint_c",
    "issue_hour_hkt",
    "revision_count",
    "forecast_max_revision_c",
    "forecast_max_path_width_c",
    "forecast_max_std_path",
    "target_month",
    "target_dayofyear",
)


@dataclass(frozen=True)
class ContinuousFit:
    probabilities: np.ndarray
    details: dict[str, Any]
    params: pd.DataFrame


def distribution_v2_predictor_columns() -> list[str]:
    """Return the allowed predictor surface for leakage tests and audits."""
    return list(V2_PREDICTOR_COLUMNS)


def _model_cfg(config: dict[str, Any], name: str) -> dict[str, Any]:
    return config.get("models", {}).get(name, {})


def _rps_mean(probs: np.ndarray, frame: pd.DataFrame) -> float:
    return float(np.mean(ranked_probability_score(probs, frame["bucket_index"].to_numpy(dtype=int))))


def _frame_param_base(validation: pd.DataFrame, method: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": validation["target_date"].to_numpy(),
            "cutoff_profile": validation["cutoff_profile"].to_numpy(),
            "row_identity": validation["row_identity"].to_numpy(),
            "method": method,
        }
    )


def _fit_ridge_location_scale(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    mean_alpha: float,
    scale_alpha: float,
    sigma_floor: float,
    sigma_cap: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_x, val_x = _standardize_train_validation(_feature_matrix(train), _feature_matrix(validation))
    y = train["residual_c"].to_numpy(dtype=float)
    mean_model = Ridge(alpha=float(mean_alpha))
    mean_model.fit(train_x, y)
    fitted_residual = mean_model.predict(train_x)
    residual_error = y - fitted_residual
    scale_target = np.log(np.maximum(np.abs(residual_error), float(sigma_floor)))
    scale_model = Ridge(alpha=float(scale_alpha))
    scale_model.fit(train_x, scale_target)
    residual_mu = mean_model.predict(val_x)
    location = validation["forecast_max_c"].to_numpy(dtype=float) + residual_mu
    scale = np.clip(np.exp(scale_model.predict(val_x)), float(sigma_floor), float(sigma_cap))
    details = {
        "mean_alpha": float(mean_alpha),
        "scale_alpha": float(scale_alpha),
        "sigma_floor": float(sigma_floor),
        "sigma_cap": float(sigma_cap),
        "train_residual_error_std_c": float(np.std(residual_error)),
    }
    return location, scale, details


def _select_emos_grid(
    train: pd.DataFrame,
    config: dict[str, Any],
    method_name: str,
    family: str,
) -> dict[str, Any]:
    cfg = _model_cfg(config, method_name)
    mean_alphas = cfg.get("mean_ridge_alphas", [0.01, 0.1, 1.0, 10.0])
    scale_alphas = cfg.get("scale_ridge_alphas", [0.01, 0.1, 1.0, 10.0])
    sigma_floors = cfg.get("sigma_floor_grid", [0.25, 0.35, 0.50])
    df_grid = cfg.get("student_t_df", [3, 5, 8, 12]) if family == "student_t" else [None]
    sigma_cap = float(cfg.get("sigma_cap", 4.0))
    inner_train, inner_val = chronological_inner_split(train)
    default = {
        "mean_alpha": float(mean_alphas[0]),
        "scale_alpha": float(scale_alphas[0]),
        "sigma_floor": float(sigma_floors[0]),
        "sigma_cap": sigma_cap,
        "student_t_df": None if family != "student_t" else float(df_grid[0]),
        "selection": "default_insufficient_inner_rows",
        "tried": [],
    }
    if len(inner_train) < 100 or len(inner_val) < 30:
        return default
    best = default.copy()
    best_rps = np.inf
    tried: list[dict[str, Any]] = []
    for mean_alpha in mean_alphas:
        for scale_alpha in scale_alphas:
            for sigma_floor in sigma_floors:
                location, scale, _ = _fit_ridge_location_scale(
                    inner_train,
                    inner_val,
                    float(mean_alpha),
                    float(scale_alpha),
                    float(sigma_floor),
                    sigma_cap,
                )
                for df in df_grid:
                    probs = _continuous_bucket_probs(location, scale, family=family, df=None if df is None else float(df))
                    rps = _rps_mean(probs, inner_val)
                    record = {
                        "mean_alpha": float(mean_alpha),
                        "scale_alpha": float(scale_alpha),
                        "sigma_floor": float(sigma_floor),
                        "student_t_df": None if df is None else float(df),
                        "inner_rps": rps,
                    }
                    tried.append(record)
                    if rps < best_rps:
                        best_rps = rps
                        best = {
                            **record,
                            "sigma_cap": sigma_cap,
                            "selection": "inner_chronological_rps",
                        }
    best["best_inner_rps"] = best_rps
    best["tried"] = tried
    return best


def emos_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict[str, Any],
    method_name: str,
    family: str,
) -> ContinuousFit:
    """Fit a true location-and-scale EMOS-style ridge distribution."""
    selected = _select_emos_grid(train, config, method_name, family)
    location, scale, fit_details = _fit_ridge_location_scale(
        train,
        validation,
        float(selected["mean_alpha"]),
        float(selected["scale_alpha"]),
        float(selected["sigma_floor"]),
        float(selected["sigma_cap"]),
    )
    df = selected.get("student_t_df")
    probs = _continuous_bucket_probs(location, scale, family=family, df=None if df is None else float(df))
    params = _frame_param_base(validation, method_name)
    params["location_c"] = location
    params["scale_c"] = scale
    params["student_t_df"] = np.nan if df is None else float(df)
    details = {**fit_details, **selected, "family": family}
    return ContinuousFit(probs, details, params)


def two_piece_normal_cdf(x: np.ndarray, location: np.ndarray, scale_left: np.ndarray, scale_right: np.ndarray) -> np.ndarray:
    """CDF for a split normal with different cold/hot side scales."""
    x_arr = np.asarray(x, dtype=float)
    loc = np.asarray(location, dtype=float)
    left = np.maximum(np.asarray(scale_left, dtype=float), 0.20)
    right = np.maximum(np.asarray(scale_right, dtype=float), 0.20)
    left_weight = left / (left + right)
    out = np.empty(np.broadcast_shapes(x_arr.shape, loc.shape), dtype=float)
    x_b = np.broadcast_to(x_arr, out.shape)
    loc_b = np.broadcast_to(loc, out.shape)
    left_b = np.broadcast_to(left, out.shape)
    right_b = np.broadcast_to(right, out.shape)
    weight_b = np.broadcast_to(left_weight, out.shape)
    cold = x_b < loc_b
    out[cold] = 2.0 * weight_b[cold] * norm.cdf((x_b[cold] - loc_b[cold]) / left_b[cold])
    out[~cold] = weight_b[~cold] + 2.0 * (1.0 - weight_b[~cold]) * (
        norm.cdf((x_b[~cold] - loc_b[~cold]) / right_b[~cold]) - 0.5
    )
    return np.clip(out, 0.0, 1.0)


def two_piece_normal_bucket_probs(location: np.ndarray, scale_left: np.ndarray, scale_right: np.ndarray) -> np.ndarray:
    edges = bucket_boundaries_for_cdf()
    cdf = np.vstack([two_piece_normal_cdf(np.full(len(location), edge), location, scale_left, scale_right) for edge in edges]).T
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0), axis=1)
    probs = np.zeros((len(location), len(BUCKET_KEYS)), dtype=float)
    probs[:, 0] = cdf[:, 0]
    probs[:, 1:-1] = cdf[:, 1:] - cdf[:, :-1]
    probs[:, -1] = 1.0 - cdf[:, -1]
    return normalize_probability_matrix(probs)


def two_piece_emos_predict(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]) -> ContinuousFit:
    selected = _select_emos_grid(train, config, "E3_two_piece_normal_emos", family="normal")
    location, base_scale, fit_details = _fit_ridge_location_scale(
        train,
        validation,
        float(selected["mean_alpha"]),
        float(selected["scale_alpha"]),
        float(selected["sigma_floor"]),
        float(selected["sigma_cap"]),
    )
    train_x, _ = _standardize_train_validation(_feature_matrix(train), _feature_matrix(validation))
    mean_model = Ridge(alpha=float(selected["mean_alpha"]))
    mean_model.fit(train_x, train["residual_c"].to_numpy(dtype=float))
    residual_error = train["residual_c"].to_numpy(dtype=float) - mean_model.predict(train_x)
    sigma_floor = float(selected["sigma_floor"])
    neg = np.abs(residual_error[residual_error < 0.0])
    pos = np.abs(residual_error[residual_error >= 0.0])
    all_scale = max(float(np.std(residual_error)), sigma_floor)
    left_ratio = float(np.clip((np.mean(neg) if len(neg) else all_scale) / all_scale, 0.45, 2.50))
    right_ratio = float(np.clip((np.mean(pos) if len(pos) else all_scale) / all_scale, 0.45, 2.50))
    scale_left = np.clip(base_scale * left_ratio, sigma_floor, float(selected["sigma_cap"]))
    scale_right = np.clip(base_scale * right_ratio, sigma_floor, float(selected["sigma_cap"]))
    probs = two_piece_normal_bucket_probs(location, scale_left, scale_right)
    params = _frame_param_base(validation, "E3_two_piece_normal_emos")
    params["location_c"] = location
    params["scale_left_c"] = scale_left
    params["scale_right_c"] = scale_right
    params["student_t_df"] = np.nan
    details = {
        **fit_details,
        **selected,
        "family": "two_piece_normal",
        "left_scale_ratio": left_ratio,
        "right_scale_ratio": right_ratio,
    }
    return ContinuousFit(probs, details, params)


def gamlss_tree_location_scale_predict(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]) -> ContinuousFit:
    cfg = _model_cfg(config, "G1_gamlss_tree_location_scale")
    n_estimators = int(cfg.get("n_estimators", 60))
    max_depth = int(cfg.get("max_depth", 2))
    learning_rate = float(cfg.get("learning_rate", 0.04))
    sigma_floor = float(cfg.get("sigma_floor", 0.30))
    sigma_cap = float(cfg.get("sigma_cap", 4.0))
    random_state = int(cfg.get("random_state", 20260706))
    train_x = _feature_matrix(train)
    val_x = _feature_matrix(validation)
    y = train["residual_c"].to_numpy(dtype=float)
    mean_model = GradientBoostingRegressor(
        loss="squared_error",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    mean_model.fit(train_x, y)
    fitted_residual = mean_model.predict(train_x)
    scale_target = np.log(np.maximum(np.abs(y - fitted_residual), sigma_floor))
    scale_model = GradientBoostingRegressor(
        loss="squared_error",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state + 1,
    )
    scale_model.fit(train_x, scale_target)
    location = validation["forecast_max_c"].to_numpy(dtype=float) + mean_model.predict(val_x)
    scale = np.clip(np.exp(scale_model.predict(val_x)), sigma_floor, sigma_cap)
    probs = _continuous_bucket_probs(location, scale, family="normal")
    params = _frame_param_base(validation, "G1_gamlss_tree_location_scale")
    params["location_c"] = location
    params["scale_c"] = scale
    params["student_t_df"] = np.nan
    details = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "sigma_floor": sigma_floor,
        "sigma_cap": sigma_cap,
        "random_state": random_state,
    }
    return ContinuousFit(probs, details, params)


def _cdf_probs_from_cdf(cdf: np.ndarray) -> np.ndarray:
    projected = np.maximum.accumulate(np.clip(np.asarray(cdf, dtype=float), 0.0, 1.0), axis=1)
    probs = np.zeros((projected.shape[0], projected.shape[1] + 1), dtype=float)
    probs[:, 0] = projected[:, 0]
    probs[:, 1:-1] = projected[:, 1:] - projected[:, :-1]
    probs[:, -1] = 1.0 - projected[:, -1]
    return normalize_probability_matrix(probs)


def quantile_cdf_gb_predict(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]) -> ContinuousFit:
    cfg = _model_cfg(config, "Q1_quantile_cdf_gb")
    quantiles = [float(value) for value in cfg.get("quantiles", [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95])]
    n_estimators = int(cfg.get("n_estimators", 50))
    max_depth = int(cfg.get("max_depth", 2))
    learning_rate = float(cfg.get("learning_rate", 0.04))
    random_state = int(cfg.get("random_state", 20260706))
    train_x = _feature_matrix(train)
    val_x = _feature_matrix(validation)
    y = train["target_tmax_c"].to_numpy(dtype=float)
    quantile_predictions = np.zeros((len(validation), len(quantiles)), dtype=float)
    for idx, quantile in enumerate(quantiles):
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state + idx,
        )
        model.fit(train_x, y)
        quantile_predictions[:, idx] = model.predict(val_x)
    quantile_predictions = np.maximum.accumulate(quantile_predictions, axis=1)
    edges = bucket_boundaries_for_cdf()
    cdf = np.zeros((len(validation), len(edges)), dtype=float)
    levels = np.asarray(quantiles, dtype=float)
    for row_idx in range(len(validation)):
        values = quantile_predictions[row_idx]
        x = np.r_[values[0] - 4.0, values, values[-1] + 4.0]
        q = np.r_[0.0, levels, 1.0]
        cdf[row_idx] = np.interp(edges, x, q, left=0.0, right=1.0)
    probs = _cdf_probs_from_cdf(cdf)
    params = _frame_param_base(validation, "Q1_quantile_cdf_gb")
    params["q10_c"] = [float(np.interp(0.10, levels, row)) for row in quantile_predictions]
    params["median_c"] = [float(np.interp(0.50, levels, row)) for row in quantile_predictions]
    params["q90_c"] = [float(np.interp(0.90, levels, row)) for row in quantile_predictions]
    details = {
        "quantiles": quantiles,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "random_state": random_state,
        "cdf_projection": "monotone_after_quantile_interpolation",
    }
    return ContinuousFit(probs, details, params)


def threshold_cdf_gb_predict(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]) -> ContinuousFit:
    cfg = _model_cfg(config, "Q2_threshold_cdf_gb")
    n_estimators = int(cfg.get("n_estimators", 50))
    max_depth = int(cfg.get("max_depth", 2))
    learning_rate = float(cfg.get("learning_rate", 0.04))
    random_state = int(cfg.get("random_state", 20260706))
    train_x = _feature_matrix(train)
    val_x = _feature_matrix(validation)
    y = train["target_tmax_c"].to_numpy(dtype=float)
    edges = bucket_boundaries_for_cdf()
    cdf = np.zeros((len(validation), len(edges)), dtype=float)
    constants: list[float | None] = []
    for idx, edge in enumerate(edges):
        binary = (y <= edge).astype(int)
        if binary.min() == binary.max():
            cdf[:, idx] = float(binary[0])
            constants.append(float(binary[0]))
            continue
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state + idx,
        )
        model.fit(train_x, binary)
        positive_col = list(model.classes_).index(1)
        cdf[:, idx] = model.predict_proba(val_x)[:, positive_col]
        constants.append(None)
    probs = _cdf_probs_from_cdf(cdf)
    params = _frame_param_base(validation, "Q2_threshold_cdf_gb")
    params["cdf_monotone_min_gap"] = np.min(np.diff(np.maximum.accumulate(np.clip(cdf, 0.0, 1.0), axis=1), axis=1), axis=1)
    details = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "random_state": random_state,
        "constant_thresholds": constants,
        "cdf_projection": "monotone_cdf_projection",
    }
    return ContinuousFit(probs, details, params)


def _weighted_residual_pmf(residual_tenths: np.ndarray, weights: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    support_min = int(RESIDUAL_SUPPORT_TENTHS[0])
    support_max = int(RESIDUAL_SUPPORT_TENTHS[-1])
    residuals = np.asarray(residual_tenths, dtype=int)
    clipped = np.clip(residuals, support_min, support_max)
    indices = clipped - support_min
    counts = np.bincount(indices, weights=np.asarray(weights, dtype=float), minlength=len(RESIDUAL_SUPPORT_TENTHS)).astype(float)
    counts += alpha
    return counts / counts.sum()


def time_decay_hierarchical_pmf_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    half_life_years: float,
    month_alpha: float,
    cell_alpha: float,
) -> np.ndarray:
    dated_train = train.copy()
    dates = pd.to_datetime(dated_train["target_date"])
    max_date = dates.max()
    ages_years = (max_date - dates).dt.days.to_numpy(dtype=float) / 365.25
    dated_train["_td_weight"] = 0.5 ** (ages_years / max(float(half_life_years), 0.10))
    global_pmf = _weighted_residual_pmf(dated_train["residual_tenths"].to_numpy(dtype=int), dated_train["_td_weight"].to_numpy(dtype=float))

    month_pmfs: dict[int, np.ndarray] = {}
    for month, group in dated_train.groupby("target_month", sort=False):
        counts = _weighted_residual_pmf(group["residual_tenths"].to_numpy(dtype=int), group["_td_weight"].to_numpy(dtype=float), alpha=0.0)
        pmf = counts + float(month_alpha) * global_pmf
        month_pmfs[int(month)] = pmf / pmf.sum()

    cell_pmfs: dict[tuple[int, int], np.ndarray] = {}
    for key, group in dated_train.groupby(["target_month", "official_max_round"], sort=False):
        month = int(key[0])
        counts = _weighted_residual_pmf(group["residual_tenths"].to_numpy(dtype=int), group["_td_weight"].to_numpy(dtype=float), alpha=0.0)
        parent = month_pmfs.get(month, global_pmf)
        pmf = counts + float(cell_alpha) * parent
        cell_pmfs[(month, int(key[1]))] = pmf / pmf.sum()

    probs = np.zeros((len(validation), len(BUCKET_KEYS)), dtype=float)
    for row_pos, (_, row) in enumerate(validation.iterrows()):
        month = int(row["target_month"])
        official_max = int(row["official_max_round"])
        pmf = cell_pmfs.get((month, official_max), month_pmfs.get(month, global_pmf))
        probs[row_pos] = residual_pmf_to_bucket_probs(np.array([row["forecast_max_tenths"]]), pmf)[0]
    return normalize_probability_matrix(probs)


def select_time_decay_b4_params(train: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    cfg = _model_cfg(config, "T1_time_decay_b4")
    half_lives = cfg.get("half_life_years", [3.0, 6.0, 10.0, 16.0])
    month_alphas = cfg.get("month_alpha", [2.0, 5.0, 10.0, 20.0])
    cell_alphas = cfg.get("cell_alpha", [2.0, 5.0, 10.0, 20.0, 40.0])
    inner_train, inner_val = chronological_inner_split(train)
    default = {
        "half_life_years": float(half_lives[0]),
        "month_alpha": float(month_alphas[0]),
        "cell_alpha": float(cell_alphas[0]),
        "selection": "default_insufficient_inner_rows",
        "tried": [],
    }
    if len(inner_train) < 100 or len(inner_val) < 30:
        return default
    best = default.copy()
    best_rps = np.inf
    tried: list[dict[str, Any]] = []
    for half_life in half_lives:
        for month_alpha in month_alphas:
            for cell_alpha in cell_alphas:
                probs = time_decay_hierarchical_pmf_predict(inner_train, inner_val, float(half_life), float(month_alpha), float(cell_alpha))
                rps = _rps_mean(probs, inner_val)
                record = {
                    "half_life_years": float(half_life),
                    "month_alpha": float(month_alpha),
                    "cell_alpha": float(cell_alpha),
                    "inner_rps": rps,
                }
                tried.append(record)
                if rps < best_rps:
                    best_rps = rps
                    best = {**record, "selection": "inner_chronological_rps"}
    best["best_inner_rps"] = best_rps
    best["tried"] = tried
    return best


def time_decay_b4_predict(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    selected = select_time_decay_b4_params(train, config)
    probs = time_decay_hierarchical_pmf_predict(
        train,
        validation,
        float(selected["half_life_years"]),
        float(selected["month_alpha"]),
        float(selected["cell_alpha"]),
    )
    return probs, selected


def _empty_params() -> pd.DataFrame:
    return pd.DataFrame(columns=["target_date", "cutoff_profile", "row_identity", "method"])


def predict_distribution_challengers(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict[str, Any],
    method_subset: set[str] | None = None,
) -> tuple[dict[str, MethodOutput], dict[str, Any], pd.DataFrame]:
    outputs: dict[str, MethodOutput] = {}
    details: dict[str, Any] = {}
    params: list[pd.DataFrame] = []

    def enabled(method: str) -> bool:
        return method_subset is None or method in method_subset

    if enabled("E1_normal_emos"):
        fit = emos_predict(train, validation, config, "E1_normal_emos", family="normal")
        outputs["E1_normal_emos"] = MethodOutput("E1_normal_emos", "emos", fit.probabilities, fit.details)
        details["E1_normal_emos"] = fit.details
        params.append(fit.params)

    if enabled("E2_student_t_emos"):
        fit = emos_predict(train, validation, config, "E2_student_t_emos", family="student_t")
        outputs["E2_student_t_emos"] = MethodOutput("E2_student_t_emos", "emos", fit.probabilities, fit.details)
        details["E2_student_t_emos"] = fit.details
        params.append(fit.params)

    if enabled("E3_two_piece_normal_emos"):
        fit = two_piece_emos_predict(train, validation, config)
        outputs["E3_two_piece_normal_emos"] = MethodOutput("E3_two_piece_normal_emos", "emos", fit.probabilities, fit.details)
        details["E3_two_piece_normal_emos"] = fit.details
        params.append(fit.params)

    if enabled("G1_gamlss_tree_location_scale"):
        fit = gamlss_tree_location_scale_predict(train, validation, config)
        outputs["G1_gamlss_tree_location_scale"] = MethodOutput(
            "G1_gamlss_tree_location_scale", "gamlss_tree", fit.probabilities, fit.details
        )
        details["G1_gamlss_tree_location_scale"] = fit.details
        params.append(fit.params)

    if enabled("Q1_quantile_cdf_gb"):
        fit = quantile_cdf_gb_predict(train, validation, config)
        outputs["Q1_quantile_cdf_gb"] = MethodOutput("Q1_quantile_cdf_gb", "quantile_cdf", fit.probabilities, fit.details)
        details["Q1_quantile_cdf_gb"] = fit.details
        params.append(fit.params)

    if enabled("Q2_threshold_cdf_gb"):
        fit = threshold_cdf_gb_predict(train, validation, config)
        outputs["Q2_threshold_cdf_gb"] = MethodOutput("Q2_threshold_cdf_gb", "threshold_cdf", fit.probabilities, fit.details)
        details["Q2_threshold_cdf_gb"] = fit.details
        params.append(fit.params)

    if enabled("T1_time_decay_b4"):
        probs, td_details = time_decay_b4_predict(train, validation, config)
        outputs["T1_time_decay_b4"] = MethodOutput("T1_time_decay_b4", "time_decay_residual_pmf", probs, td_details)
        details["T1_time_decay_b4"] = td_details

    return outputs, details, pd.concat(params, ignore_index=True) if params else _empty_params()


def _select_hybrid_pool(
    train: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    cfg = _model_cfg(config, "H1_b4_challenger_linear_pool")
    candidates = [str(value) for value in cfg.get("pool_candidates", ["E1_normal_emos", "E2_student_t_emos", "Q2_threshold_cdf_gb", "T1_time_decay_b4"])]
    weight_grid = [float(value) for value in cfg.get("weight_grid", [0.0, 0.05, 0.10, 0.20, 0.35, 0.50])]
    inner_train, inner_val = chronological_inner_split(train)
    default = {
        "selected_challenger": None,
        "challenger_weight": 0.0,
        "b4_weight": 1.0,
        "selection": "default_insufficient_inner_rows",
        "tried": [],
    }
    if len(inner_train) < 100 or len(inner_val) < 30:
        return default
    inner_base = predict_base_methods(inner_train, inner_val, config)
    b4_probs = inner_base["B4_hierarchical_residual_pmf"].probabilities
    challengers, _, _ = predict_distribution_challengers(inner_train, inner_val, config, method_subset=set(candidates))
    best = default.copy()
    best_rps = _rps_mean(b4_probs, inner_val)
    tried: list[dict[str, Any]] = [
        {"selected_challenger": None, "challenger_weight": 0.0, "b4_weight": 1.0, "inner_rps": best_rps}
    ]
    for challenger_name, challenger_output in challengers.items():
        for weight in weight_grid:
            weight = float(weight)
            pooled = normalize_probability_matrix((1.0 - weight) * b4_probs + weight * challenger_output.probabilities)
            rps = _rps_mean(pooled, inner_val)
            record = {
                "selected_challenger": challenger_name,
                "challenger_weight": weight,
                "b4_weight": 1.0 - weight,
                "inner_rps": rps,
            }
            tried.append(record)
            if rps < best_rps:
                best_rps = rps
                best = {**record, "selection": "inner_chronological_rps"}
    best["best_inner_rps"] = best_rps
    best["tried"] = tried
    return best


def predict_distribution_methods_v2(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict[str, Any],
    base_outputs: dict[str, MethodOutput] | None = None,
) -> tuple[dict[str, MethodOutput], dict[str, Any], pd.DataFrame]:
    challenger_outputs, details, params = predict_distribution_challengers(train, validation, config)
    if base_outputs is None:
        base_outputs = predict_base_methods(train, validation, config)
    hybrid_selection = _select_hybrid_pool(train, config)
    b4_probs = base_outputs["B4_hierarchical_residual_pmf"].probabilities
    selected = hybrid_selection.get("selected_challenger")
    weight = float(hybrid_selection.get("challenger_weight", 0.0) or 0.0)
    if selected and selected in challenger_outputs:
        hybrid_probs = normalize_probability_matrix((1.0 - weight) * b4_probs + weight * challenger_outputs[selected].probabilities)
    else:
        hybrid_probs = b4_probs
    details["H1_b4_challenger_linear_pool"] = hybrid_selection
    challenger_outputs["H1_b4_challenger_linear_pool"] = MethodOutput(
        "H1_b4_challenger_linear_pool",
        "hybrid_pool",
        hybrid_probs,
        hybrid_selection,
    )
    return challenger_outputs, details, params


def method_details_frame(selection_logs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for log in selection_logs:
        for method, details in log.get("method_details", {}).items():
            rows.append(
                {
                    "validation_split": log["window"],
                    "method": method,
                    "details_json": json.dumps(details, default=str),
                }
            )
    return pd.DataFrame(rows)
