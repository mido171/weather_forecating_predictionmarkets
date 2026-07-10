"""Probability model families for HKG Tmax bucket calibration V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t
from sklearn.linear_model import LogisticRegression, Ridge

from hkg_tmax_probability.bucket_rules import (
    BUCKET_KEYS,
    bucket_boundaries_for_cdf,
    normalize_probability_matrix,
)
from hkg_tmax_probability.scoring import ranked_probability_score

RESIDUAL_SUPPORT_TENTHS = np.arange(-120, 121, dtype=int)
STACK_BASE_METHODS = (
    "B0_climatology",
    "B1_global_residual_pmf",
    "B2_month_residual_pmf",
    "B3_forecast_level_residual_pmf",
    "B4_hierarchical_residual_pmf",
    "P1_normal_mos",
    "C1_multinomial_ridge",
)


@dataclass
class MethodOutput:
    method: str
    family: str
    probabilities: np.ndarray
    details: dict[str, Any]


def _bucket_indices_from_tenths(values_tenths: np.ndarray) -> np.ndarray:
    values = np.asarray(values_tenths, dtype=int)
    out = np.empty_like(values, dtype=int)
    out[values <= 249] = 0
    out[values >= 340] = 10
    mid = (values > 249) & (values < 340)
    out[mid] = (values[mid] // 10) - 24
    return np.clip(out, 0, 10)


def _counts_to_distribution(indices: np.ndarray, width: int = len(BUCKET_KEYS), alpha: float = 1e-6) -> np.ndarray:
    counts = np.bincount(np.asarray(indices, dtype=int), minlength=width).astype(float)
    counts += alpha
    return counts / counts.sum()


def _residual_distribution(train: pd.DataFrame, alpha: float = 1e-6) -> np.ndarray:
    counts = pd.Series(train["residual_tenths"].astype(int)).value_counts().reindex(RESIDUAL_SUPPORT_TENTHS, fill_value=0).to_numpy(dtype=float)
    counts += alpha
    return counts / counts.sum()


def residual_pmf_to_bucket_probs(forecast_max_tenths: np.ndarray, residual_pmf: np.ndarray) -> np.ndarray:
    forecasts = np.asarray(forecast_max_tenths, dtype=int)
    probs = np.zeros((len(forecasts), len(BUCKET_KEYS)), dtype=float)
    for row_index, forecast in enumerate(forecasts):
        buckets = _bucket_indices_from_tenths(forecast + RESIDUAL_SUPPORT_TENTHS)
        probs[row_index] = np.bincount(buckets, weights=residual_pmf, minlength=len(BUCKET_KEYS))
    return normalize_probability_matrix(probs)


def _group_key(row: pd.Series, group_cols: list[str]) -> tuple[Any, ...]:
    return tuple(row[col] for col in group_cols)


def grouped_residual_pmf_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    group_cols: list[str] | None = None,
    shrink_alpha: float = 0.0,
) -> np.ndarray:
    global_pmf = _residual_distribution(train)
    if not group_cols:
        return residual_pmf_to_bucket_probs(validation["forecast_max_tenths"].to_numpy(), global_pmf)

    grouped: dict[tuple[Any, ...], np.ndarray] = {}
    for key, group in train.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        counts = (
            pd.Series(group["residual_tenths"].astype(int))
            .value_counts()
            .reindex(RESIDUAL_SUPPORT_TENTHS, fill_value=0)
            .to_numpy(dtype=float)
        )
        if shrink_alpha > 0:
            pmf = counts + shrink_alpha * global_pmf
        else:
            pmf = counts + 1e-6
        grouped[key] = pmf / pmf.sum()

    probs = np.zeros((len(validation), len(BUCKET_KEYS)), dtype=float)
    for pos, (_, row) in enumerate(validation.iterrows()):
        key = _group_key(row, group_cols)
        pmf = grouped.get(key, global_pmf)
        probs[pos] = residual_pmf_to_bucket_probs(np.array([row["forecast_max_tenths"]]), pmf)[0]
    return normalize_probability_matrix(probs)


def hierarchical_month_forecast_pmf_predict(train: pd.DataFrame, validation: pd.DataFrame, month_alpha: float, cell_alpha: float) -> np.ndarray:
    global_pmf = _residual_distribution(train)
    month_pmfs: dict[int, np.ndarray] = {}
    for month, group in train.groupby("target_month"):
        counts = (
            pd.Series(group["residual_tenths"].astype(int))
            .value_counts()
            .reindex(RESIDUAL_SUPPORT_TENTHS, fill_value=0)
            .to_numpy(dtype=float)
        )
        pmf = counts + month_alpha * global_pmf
        month_pmfs[int(month)] = pmf / pmf.sum()

    cell_pmfs: dict[tuple[int, int], np.ndarray] = {}
    for key, group in train.groupby(["target_month", "official_max_round"], sort=False):
        month = int(key[0])
        counts = (
            pd.Series(group["residual_tenths"].astype(int))
            .value_counts()
            .reindex(RESIDUAL_SUPPORT_TENTHS, fill_value=0)
            .to_numpy(dtype=float)
        )
        parent = month_pmfs.get(month, global_pmf)
        pmf = counts + cell_alpha * parent
        cell_pmfs[(month, int(key[1]))] = pmf / pmf.sum()

    probs = np.zeros((len(validation), len(BUCKET_KEYS)), dtype=float)
    for pos, (_, row) in enumerate(validation.iterrows()):
        month = int(row["target_month"])
        official_max = int(row["official_max_round"])
        pmf = cell_pmfs.get((month, official_max), month_pmfs.get(month, global_pmf))
        probs[pos] = residual_pmf_to_bucket_probs(np.array([row["forecast_max_tenths"]]), pmf)[0]
    return normalize_probability_matrix(probs)


def climatology_predict(train: pd.DataFrame, validation: pd.DataFrame, group_col: str | None = None) -> np.ndarray:
    if group_col is None:
        dist = _counts_to_distribution(train["bucket_index"].to_numpy())
        return np.tile(dist, (len(validation), 1))
    global_dist = _counts_to_distribution(train["bucket_index"].to_numpy())
    by_group = {
        key: _counts_to_distribution(group["bucket_index"].to_numpy(), alpha=0.05)
        for key, group in train.groupby(group_col, sort=False)
    }
    probs = np.vstack([by_group.get(row[group_col], global_dist) for _, row in validation.iterrows()])
    return normalize_probability_matrix(probs)


def kernel_analog_residual_pmf_predict(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    kernel_cfg = config.get("models", {}).get("B5_kernel_analog", {})
    bandwidths = kernel_cfg.get("default_bandwidths", {})
    h_max = float(bandwidths.get("forecast_max_c", 1.0))
    h_range = float(bandwidths.get("forecast_range_c", 1.5))
    h_revision = float(bandwidths.get("forecast_max_revision_c", 1.0))
    h_month = float(bandwidths.get("month_circular", 2.0))
    k = int(kernel_cfg.get("nearest_neighbors", 180))
    floor = float(kernel_cfg.get("weight_floor", 1e-8))

    train_max = train["forecast_max_c"].to_numpy(dtype=float)
    train_range = train["forecast_range_c"].fillna(0.0).to_numpy(dtype=float)
    train_revision = train["forecast_max_revision_c"].fillna(0.0).to_numpy(dtype=float)
    train_month = train["target_month"].to_numpy(dtype=float)
    train_residuals = train["residual_tenths"].to_numpy(dtype=int)
    probs = np.zeros((len(validation), len(BUCKET_KEYS)), dtype=float)
    for row_pos, (_, row) in enumerate(validation.iterrows()):
        month_delta = np.minimum(np.abs(train_month - row["target_month"]), 12 - np.abs(train_month - row["target_month"]))
        d2 = (
            ((train_max - row["forecast_max_c"]) / h_max) ** 2
            + ((train_range - row["forecast_range_c"]) / h_range) ** 2
            + ((train_revision - row.get("forecast_max_revision_c", 0.0)) / h_revision) ** 2
            + (month_delta / h_month) ** 2
        )
        if len(d2) > k:
            keep = np.argpartition(d2, k)[:k]
        else:
            keep = np.arange(len(d2))
        weights = np.exp(-0.5 * d2[keep]) + floor
        bucket_idx = _bucket_indices_from_tenths(int(row["forecast_max_tenths"]) + train_residuals[keep])
        probs[row_pos] = np.bincount(bucket_idx, weights=weights, minlength=len(BUCKET_KEYS))
    return normalize_probability_matrix(probs)


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    month_angle = 2.0 * np.pi * frame["target_month"].to_numpy(dtype=float) / 12.0
    day_angle = 2.0 * np.pi * frame["target_dayofyear"].to_numpy(dtype=float) / 366.0
    columns = [
        frame["forecast_max_c"].to_numpy(dtype=float),
        frame["forecast_min_c"].to_numpy(dtype=float),
        frame["forecast_range_c"].fillna(0.0).to_numpy(dtype=float),
        frame["forecast_midpoint_c"].fillna(0.0).to_numpy(dtype=float),
        frame["issue_hour_hkt"].fillna(0.0).to_numpy(dtype=float),
        frame["revision_count"].fillna(1.0).to_numpy(dtype=float),
        frame["forecast_max_revision_c"].fillna(0.0).to_numpy(dtype=float),
        frame["forecast_max_path_width_c"].fillna(0.0).to_numpy(dtype=float),
        frame["forecast_max_std_path"].fillna(0.0).to_numpy(dtype=float),
        np.sin(month_angle),
        np.cos(month_angle),
        np.sin(day_angle),
        np.cos(day_angle),
    ]
    return np.vstack(columns).T


def _standardize_train_validation(train_x: np.ndarray, validation_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (train_x - mu) / sigma, (validation_x - mu) / sigma


def _continuous_bucket_probs(location: np.ndarray, scale: np.ndarray, family: str, df: float | None = None) -> np.ndarray:
    edges = bucket_boundaries_for_cdf()
    loc = np.asarray(location, dtype=float)
    sc = np.maximum(np.asarray(scale, dtype=float), 0.20)
    cdfs = []
    for edge in edges:
        z = (edge - loc) / sc
        cdfs.append(student_t.cdf(z, df=df) if family == "student_t" else norm.cdf(z))
    cdf = np.vstack(cdfs).T
    probs = np.zeros((len(loc), len(BUCKET_KEYS)), dtype=float)
    probs[:, 0] = cdf[:, 0]
    probs[:, 1:-1] = cdf[:, 1:] - cdf[:, :-1]
    probs[:, -1] = 1.0 - cdf[:, -1]
    return normalize_probability_matrix(probs)


def mos_predict(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any], family: str) -> tuple[np.ndarray, dict[str, Any]]:
    ridge_alphas = config.get("models", {}).get("mos", {}).get("ridge_alphas", [0.1, 1.0, 10.0])
    train_x, val_x = _standardize_train_validation(_feature_matrix(train), _feature_matrix(validation))
    y = train["residual_c"].to_numpy(dtype=float)
    best_alpha = float(ridge_alphas[0])
    best_rps = np.inf
    inner_train, inner_val = chronological_inner_split(train)
    if len(inner_train) >= 100 and len(inner_val) >= 50:
        inner_train_x, inner_val_x = _standardize_train_validation(_feature_matrix(inner_train), _feature_matrix(inner_val))
        for alpha in ridge_alphas:
            model = Ridge(alpha=float(alpha))
            model.fit(inner_train_x, inner_train["residual_c"].to_numpy(dtype=float))
            residual_mu = model.predict(inner_val_x)
            fitted = model.predict(inner_train_x)
            scale = np.full(len(inner_val), max(float(np.std(inner_train["residual_c"].to_numpy(dtype=float) - fitted)), 0.35))
            probs = _continuous_bucket_probs(inner_val["forecast_max_c"].to_numpy(dtype=float) + residual_mu, scale, family="normal")
            rps = float(np.mean(ranked_probability_score(probs, inner_val["bucket_index"].to_numpy(dtype=int))))
            if rps < best_rps:
                best_rps = rps
                best_alpha = float(alpha)
    model = Ridge(alpha=best_alpha)
    model.fit(train_x, y)
    pred_residual = model.predict(val_x)
    fitted_residual = model.predict(train_x)
    scale_value = max(float(np.std(y - fitted_residual)), 0.35)
    df = None
    if family == "student_t":
        df_grid = config.get("models", {}).get("mos", {}).get("student_t_df", [3, 5, 8, 12])
        best_df = float(df_grid[0])
        best_nll = np.inf
        residual_std = max(float(np.std(y - fitted_residual)), 0.35)
        residuals = y - fitted_residual
        for candidate_df in df_grid:
            nll = -float(np.mean(student_t.logpdf(residuals / residual_std, df=float(candidate_df)) - np.log(residual_std)))
            if nll < best_nll:
                best_nll = nll
                best_df = float(candidate_df)
        df = best_df
    probs = _continuous_bucket_probs(
        validation["forecast_max_c"].to_numpy(dtype=float) + pred_residual,
        np.full(len(validation), scale_value),
        family=family,
        df=df,
    )
    return probs, {"ridge_alpha": best_alpha, "scale_c": scale_value, "student_t_df": df}


def multinomial_predict(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    c_values = config.get("models", {}).get("C1_multinomial_ridge", {}).get("c_values", [0.05, 0.1, 0.5, 1.0])
    train_x, val_x = _standardize_train_validation(_feature_matrix(train), _feature_matrix(validation))
    y = train["bucket_index"].to_numpy(dtype=int)
    best_c = float(c_values[0])
    best_rps = np.inf
    inner_train, inner_val = chronological_inner_split(train)
    if len(inner_train) >= 100 and len(inner_val) >= 50 and inner_train["bucket_index"].nunique() > 1:
        inner_train_x, inner_val_x = _standardize_train_validation(_feature_matrix(inner_train), _feature_matrix(inner_val))
        for c_value in c_values:
            model = LogisticRegression(C=float(c_value), penalty="l2", solver="lbfgs", max_iter=1000)
            model.fit(inner_train_x, inner_train["bucket_index"].to_numpy(dtype=int))
            probs = _align_classifier_probs(model, inner_val_x)
            rps = float(np.mean(ranked_probability_score(probs, inner_val["bucket_index"].to_numpy(dtype=int))))
            if rps < best_rps:
                best_rps = rps
                best_c = float(c_value)
    model = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs", max_iter=1000)
    model.fit(train_x, y)
    return _align_classifier_probs(model, val_x), {"C": best_c}


def _align_classifier_probs(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), len(BUCKET_KEYS)), dtype=float)
    for col, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, col]
    return normalize_probability_matrix(out)


def ordinal_cdf_predict(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    c_value = float(config.get("models", {}).get("C2_ordinal_cdf_logistic", {}).get("C", 0.1))
    train_x, val_x = _standardize_train_validation(_feature_matrix(train), _feature_matrix(validation))
    y = train["bucket_index"].to_numpy(dtype=int)
    cdfs = np.zeros((len(validation), len(BUCKET_KEYS) - 1), dtype=float)
    constants: list[int | None] = []
    for threshold in range(len(BUCKET_KEYS) - 1):
        binary = (y <= threshold).astype(int)
        if binary.min() == binary.max():
            cdfs[:, threshold] = float(binary[0])
            constants.append(int(binary[0]))
            continue
        model = LogisticRegression(C=c_value, penalty="l2", solver="lbfgs", max_iter=1000)
        model.fit(train_x, binary)
        positive_col = list(model.classes_).index(1)
        cdfs[:, threshold] = model.predict_proba(val_x)[:, positive_col]
        constants.append(None)
    cdfs = monotone_cdf_projection(cdfs)
    probs = cdf_to_bucket_probs(cdfs)
    return probs, {"C": c_value, "constant_thresholds": constants}


def monotone_cdf_projection(cdf: np.ndarray) -> np.ndarray:
    projected = np.asarray(cdf, dtype=float).copy()
    projected = np.clip(projected, 0.0, 1.0)
    projected = np.maximum.accumulate(projected, axis=1)
    return np.clip(projected, 0.0, 1.0)


def cdf_to_bucket_probs(cdf: np.ndarray) -> np.ndarray:
    projected = monotone_cdf_projection(cdf)
    probs = np.zeros((projected.shape[0], projected.shape[1] + 1), dtype=float)
    probs[:, 0] = projected[:, 0]
    probs[:, 1:-1] = projected[:, 1:] - projected[:, :-1]
    probs[:, -1] = 1.0 - projected[:, -1]
    return normalize_probability_matrix(probs)


def power_calibration(probs: np.ndarray, gamma: float) -> np.ndarray:
    matrix = normalize_probability_matrix(probs)
    return normalize_probability_matrix(matrix**float(gamma))


def chronological_inner_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = train.sort_values("target_date")
    if frame.empty:
        return frame, frame
    max_date = pd.to_datetime(frame["target_date"]).max()
    split_at = max_date - pd.DateOffset(years=2)
    inner_train = frame[pd.to_datetime(frame["target_date"]) <= split_at]
    inner_val = frame[pd.to_datetime(frame["target_date"]) > split_at]
    if len(inner_train) < 100 or len(inner_val) < 30:
        midpoint = int(len(frame) * 0.8)
        inner_train = frame.iloc[:midpoint]
        inner_val = frame.iloc[midpoint:]
    return inner_train.copy(), inner_val.copy()


def select_b4_alphas(train: pd.DataFrame, config: dict[str, Any]) -> tuple[float, float, dict[str, Any]]:
    candidates = config.get("models", {}).get("B4_hierarchical_residual_pmf", {}).get("alpha_grid", {})
    month_values = candidates.get("month_alpha", [10.0])
    cell_values = candidates.get("cell_alpha", [5.0, 10.0, 20.0])
    inner_train, inner_val = chronological_inner_split(train)
    if len(inner_train) < 100 or len(inner_val) < 30:
        return float(month_values[0]), float(cell_values[0]), {"selection": "default_insufficient_inner_rows"}
    best = (float(month_values[0]), float(cell_values[0]))
    best_rps = np.inf
    tried = []
    for month_alpha in month_values:
        for cell_alpha in cell_values:
            probs = hierarchical_month_forecast_pmf_predict(inner_train, inner_val, float(month_alpha), float(cell_alpha))
            rps = float(np.mean(ranked_probability_score(probs, inner_val["bucket_index"].to_numpy(dtype=int))))
            tried.append({"month_alpha": float(month_alpha), "cell_alpha": float(cell_alpha), "inner_rps": rps})
            if rps < best_rps:
                best_rps = rps
                best = (float(month_alpha), float(cell_alpha))
    return best[0], best[1], {"selection": "inner_chronological", "best_inner_rps": best_rps, "tried": tried}


def optimize_stack_weights(probability_blocks: list[np.ndarray], true_indices: np.ndarray, l2_to_b4: float = 0.02) -> np.ndarray:
    n = len(probability_blocks)
    if n == 0:
        raise ValueError("No probability blocks supplied for stack.")
    x0 = np.full(n, 1.0 / n)
    b4_index = min(4, n - 1)

    def objective(weights: np.ndarray) -> float:
        combined = normalize_probability_matrix(sum(weight * block for weight, block in zip(weights, probability_blocks)))
        rps = float(np.mean(ranked_probability_score(combined, true_indices)))
        anchor = np.zeros(n)
        anchor[b4_index] = 1.0
        return rps + l2_to_b4 * float(np.sum((weights - anchor) ** 2))

    constraints = [{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}]
    bounds = [(0.0, 0.65) for _ in range(n)]
    bounds[b4_index] = (0.15, 0.85)
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})
    if not result.success:
        weights = x0
    else:
        weights = np.asarray(result.x, dtype=float)
    weights = np.clip(weights, 0.0, 1.0)
    return weights / weights.sum()


def fit_stack_weights(train: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    inner_train, inner_val = chronological_inner_split(train)
    if len(inner_train) < 100 or len(inner_val) < 30:
        weights = {method: 1.0 / len(STACK_BASE_METHODS) for method in STACK_BASE_METHODS}
        return weights, {"selection": "equal_weights_insufficient_inner_rows"}
    inner_outputs = predict_base_methods(inner_train, inner_val, config)
    blocks = [inner_outputs[method].probabilities for method in STACK_BASE_METHODS if method in inner_outputs]
    names = [method for method in STACK_BASE_METHODS if method in inner_outputs]
    weights_array = optimize_stack_weights(blocks, inner_val["bucket_index"].to_numpy(dtype=int), l2_to_b4=float(config.get("stacking", {}).get("l2_to_b4", 0.02)))
    return {name: float(weight) for name, weight in zip(names, weights_array)}, {"selection": "inner_chronological_simplex", "base_methods": names}


def predict_base_methods(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]) -> dict[str, MethodOutput]:
    outputs: dict[str, MethodOutput] = {}
    outputs["B0_climatology"] = MethodOutput("B0_climatology", "baseline", climatology_predict(train, validation), {})
    outputs["B1_global_residual_pmf"] = MethodOutput(
        "B1_global_residual_pmf", "residual_pmf", grouped_residual_pmf_predict(train, validation), {}
    )
    b2_alpha = float(config.get("models", {}).get("B2_month_residual_pmf", {}).get("shrink_alpha", 10.0))
    outputs["B2_month_residual_pmf"] = MethodOutput(
        "B2_month_residual_pmf",
        "residual_pmf",
        grouped_residual_pmf_predict(train, validation, ["target_month"], shrink_alpha=b2_alpha),
        {"shrink_alpha": b2_alpha},
    )
    b3_alpha = float(config.get("models", {}).get("B3_forecast_level_residual_pmf", {}).get("shrink_alpha", 12.0))
    outputs["B3_forecast_level_residual_pmf"] = MethodOutput(
        "B3_forecast_level_residual_pmf",
        "residual_pmf",
        grouped_residual_pmf_predict(train, validation, ["official_max_round"], shrink_alpha=b3_alpha),
        {"shrink_alpha": b3_alpha},
    )
    month_alpha, cell_alpha, b4_selection = select_b4_alphas(train, config)
    outputs["B4_hierarchical_residual_pmf"] = MethodOutput(
        "B4_hierarchical_residual_pmf",
        "residual_pmf",
        hierarchical_month_forecast_pmf_predict(train, validation, month_alpha, cell_alpha),
        {"month_alpha": month_alpha, "cell_alpha": cell_alpha, **b4_selection},
    )
    outputs["B5_kernel_analog_pmf"] = MethodOutput(
        "B5_kernel_analog_pmf", "analog_pmf", kernel_analog_residual_pmf_predict(train, validation, config), {}
    )
    outputs["B6_month_climatology_diagnostic"] = MethodOutput(
        "B6_month_climatology_diagnostic", "diagnostic", climatology_predict(train, validation, group_col="target_month"), {}
    )
    p1_probs, p1_details = mos_predict(train, validation, config, family="normal")
    outputs["P1_normal_mos"] = MethodOutput("P1_normal_mos", "mos", p1_probs, p1_details)
    p2_probs, p2_details = mos_predict(train, validation, config, family="student_t")
    outputs["P2_student_t_mos"] = MethodOutput("P2_student_t_mos", "mos", p2_probs, p2_details)
    c1_probs, c1_details = multinomial_predict(train, validation, config)
    outputs["C1_multinomial_ridge"] = MethodOutput("C1_multinomial_ridge", "direct_classifier", c1_probs, c1_details)
    c2_probs, c2_details = ordinal_cdf_predict(train, validation, config)
    outputs["C2_ordinal_cdf_logistic"] = MethodOutput("C2_ordinal_cdf_logistic", "direct_classifier", c2_probs, c2_details)
    b4_probs = outputs["B4_hierarchical_residual_pmf"].probabilities
    outputs["K0_B4_identity"] = MethodOutput("K0_B4_identity", "calibration", b4_probs, {"base": "B4_hierarchical_residual_pmf"})
    gamma = float(config.get("calibration", {}).get("K1_power_gamma", 0.95))
    outputs["K1_B4_power_calibrated"] = MethodOutput(
        "K1_B4_power_calibrated", "calibration", power_calibration(b4_probs, gamma), {"base": "B4_hierarchical_residual_pmf", "gamma": gamma}
    )
    outputs["K2_B4_monotone_cdf_projected"] = MethodOutput(
        "K2_B4_monotone_cdf_projected",
        "calibration",
        cdf_to_bucket_probs(monotone_cdf_projection(np.cumsum(b4_probs, axis=1)[:, :-1])),
        {"base": "B4_hierarchical_residual_pmf", "projection": "monotone_cdf"},
    )
    return outputs


def predict_all_methods(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, MethodOutput], dict[str, Any]]:
    outputs = predict_base_methods(train, validation, config)
    stack_weights, stack_details = fit_stack_weights(train, config)
    stack_probs = np.zeros_like(next(iter(outputs.values())).probabilities)
    for method, weight in stack_weights.items():
        stack_probs += weight * outputs[method].probabilities
    outputs["S1_conservative_simplex_stack"] = MethodOutput(
        "S1_conservative_simplex_stack",
        "stack",
        normalize_probability_matrix(stack_probs),
        {"weights": stack_weights, **stack_details},
    )
    details = {method: output.details for method, output in outputs.items()}
    return outputs, details
