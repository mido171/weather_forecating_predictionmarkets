"""Derived feature library for feature strategy sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Tie-break priority for equal forecasts (higher priority first).
RAW_MODEL_ORDER = [
    "nbm_tmax_f",
    "hrrr_tmax_f",
    "rap_tmax_f",
    "gefsatmosmean_tmax_f",
    "gfs_tmax_f",
    "nam_tmax_f",
]


def compute_rowwise_features(
    df: pd.DataFrame,
    model_cols: list[str],
    *,
    prefix: str = "",
    include: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute row-wise derived features for the provided model columns."""
    if not model_cols:
        raise ValueError("model_cols is empty.")
    values = df[model_cols].to_numpy(dtype=float)
    n = values.shape[1]
    sorted_values = np.sort(values, axis=1)
    ens_mean = values.mean(axis=1)
    ens_median = np.median(values, axis=1)
    ens_min = sorted_values[:, 0]
    ens_max = sorted_values[:, -1]
    features: dict[str, np.ndarray] = {}

    def name_with_prefix(name: str) -> str:
        if not prefix:
            return name
        if f"_{n}_" in name:
            return name.replace(f"_{n}_", f"_{prefix}_{n}_", 1)
        if name.endswith(f"_{n}"):
            return name[:-len(f"_{n}")] + f"_{prefix}_{n}"
        return f"{name}_{prefix}"

    def add(name: str, values_arr: np.ndarray) -> None:
        features[name_with_prefix(name)] = values_arr

    add(f"ens_mean_{n}", ens_mean)
    add(f"ens_median_{n}", ens_median)
    add(f"ens_min_{n}", ens_min)
    add(f"ens_max_{n}", ens_max)
    add(f"ens_range_{n}", ens_max - ens_min)
    add(f"ens_std_{n}", values.std(axis=1, ddof=0))
    add(
        f"ens_p25_{n}",
        np.quantile(values, 0.25, axis=1, method="linear"),
    )
    add(
        f"ens_p75_{n}",
        np.quantile(values, 0.75, axis=1, method="linear"),
    )
    add(
        f"ens_iqr_{n}",
        np.quantile(values, 0.75, axis=1, method="linear")
        - np.quantile(values, 0.25, axis=1, method="linear"),
    )
    add(f"ens_mad_{n}", np.median(np.abs(values - ens_median[:, None]), axis=1))
    add(
        f"ens_outlier_gap_{n}",
        np.max(np.abs(values - ens_median[:, None]), axis=1),
    )
    if n >= 2:
        add(f"ens_second_min_{n}", sorted_values[:, 1])
        add(f"ens_second_max_{n}", sorted_values[:, -2])
    if n >= 5:
        trimmed = sorted_values[:, 1:-1].mean(axis=1)
        add(f"ens_trimmed_mean_{n}_1", trimmed)
        lo = sorted_values[:, 1][:, None]
        hi = sorted_values[:, -2][:, None]
        winsor = np.clip(values, lo, hi).mean(axis=1)
        add(f"ens_winsor_mean_{n}_1", winsor)
    if n >= 3:
        add("ens_closest3_mean", _closest3_mean(values, ens_median, model_cols))

    for idx, col in enumerate(model_cols):
        delta_mean = values[:, idx] - ens_mean
        delta_median = values[:, idx] - ens_median
        add(f"{col}_minus_ens_mean", delta_mean)
        add(f"{col}_minus_ens_mean_abs", np.abs(delta_mean))
        add(f"{col}_minus_ens_median", delta_median)
        add(f"{col}_minus_ens_median_abs", np.abs(delta_median))

    ranks = _rank_with_tie_break(values, model_cols)
    for idx, col in enumerate(model_cols):
        add(f"rank_{col}_in_ens", ranks[:, idx])

    if include is not None:
        include_list = [name_with_prefix(name) for name in include]
        missing = [name for name in include_list if name not in features]
        if missing:
            raise ValueError(f"Requested features not available: {sorted(missing)}")
        features = {name: features[name] for name in include_list}

    return pd.DataFrame(features, index=df.index)


def fit_bias_correction(
    train_df: pd.DataFrame, model_cols: list[str], y_col: str
) -> dict[str, float]:
    biases: dict[str, float] = {}
    y = train_df[y_col].to_numpy(dtype=float)
    for col in model_cols:
        m = train_df[col].to_numpy(dtype=float)
        biases[col] = float(np.mean(y - m))
    return biases


def apply_bias_correction(
    df: pd.DataFrame, model_cols: list[str], bias_dict: dict[str, float]
) -> pd.DataFrame:
    df = df.copy()
    for col in model_cols:
        bias = bias_dict.get(col, 0.0)
        df[f"{col}_bc"] = df[col].astype(float) + bias
    return df


@dataclass(frozen=True)
class ReliabilityWeights:
    mae_by_model: dict[str, float]
    weights: dict[str, float]
    weights_norm: dict[str, float]


def fit_reliability_weights(
    train_df: pd.DataFrame, model_cols: list[str], y_col: str
) -> ReliabilityWeights:
    y = train_df[y_col].to_numpy(dtype=float)
    mae_by_model: dict[str, float] = {}
    weights: dict[str, float] = {}
    for col in model_cols:
        m = train_df[col].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(y - m)))
        mae_by_model[col] = mae
        weights[col] = 1.0 / (mae + 1e-6)
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Reliability weights sum to zero.")
    weights_norm = {col: weight / total for col, weight in weights.items()}
    return ReliabilityWeights(
        mae_by_model=mae_by_model,
        weights=weights,
        weights_norm=weights_norm,
    )


def apply_reliability_features(
    df: pd.DataFrame,
    model_cols: list[str],
    weights: ReliabilityWeights,
) -> pd.DataFrame:
    df = df.copy()
    weighted_sum = None
    for col in model_cols:
        weight = weights.weights_norm[col]
        contrib = df[col].astype(float) * weight
        weighted_sum = contrib if weighted_sum is None else weighted_sum + contrib
        df[f"w_{_short_name(col)}"] = float(weight)
    df[f"ens_wmean_{len(model_cols)}"] = weighted_sum
    median = np.median(df[model_cols].to_numpy(dtype=float), axis=1)
    df["ens_wmean_minus_median"] = df[f"ens_wmean_{len(model_cols)}"] - median
    return df


def _short_name(column: str) -> str:
    if column.endswith("_tmax_f"):
        return column[: -len("_tmax_f")]
    return column


def fit_stack_ridge_oof(
    train_df: pd.DataFrame,
    model_cols: list[str],
    y_col: str,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> tuple[np.ndarray, Ridge]:
    x = train_df[model_cols].to_numpy(dtype=float)
    y = train_df[y_col].to_numpy(dtype=float)
    oof = np.full(len(train_df), np.nan, dtype=float)
    for train_idx, val_idx in cv_splits:
        ridge = Ridge(random_state=seed)
        ridge.fit(x[train_idx], y[train_idx])
        oof[val_idx] = ridge.predict(x[val_idx])
    ridge_full = Ridge(random_state=seed)
    ridge_full.fit(x, y)
    return oof, ridge_full


def apply_stack_feature(
    df: pd.DataFrame, model_cols: list[str], ridge_model: Ridge
) -> pd.DataFrame:
    df = df.copy()
    x = df[model_cols].to_numpy(dtype=float)
    df["stack_ridge_pred"] = ridge_model.predict(x)
    return df


def _closest3_mean(
    values: np.ndarray, median: np.ndarray, model_cols: list[str]
) -> np.ndarray:
    priority = {name: idx for idx, name in enumerate(RAW_MODEL_ORDER)}
    output = np.zeros(values.shape[0], dtype=float)
    for row_idx, row_vals in enumerate(values):
        deviations = np.abs(row_vals - median[row_idx])
        tuples = []
        for idx, dev in enumerate(deviations):
            col = model_cols[idx]
            tuples.append((dev, priority.get(col, idx), idx))
        tuples.sort()
        selected = [row_vals[item[2]] for item in tuples[:3]]
        output[row_idx] = float(np.mean(selected))
    return output


def _rank_with_tie_break(values: np.ndarray, model_cols: list[str]) -> np.ndarray:
    priority = {name: idx for idx, name in enumerate(RAW_MODEL_ORDER)}
    ranks = np.zeros_like(values, dtype=int)
    for row_idx, row_vals in enumerate(values):
        tuples = []
        for idx, val in enumerate(row_vals):
            col = model_cols[idx]
            tuples.append((val, priority.get(col, idx), idx))
        tuples.sort()
        for rank, item in enumerate(tuples, start=1):
            ranks[row_idx, item[2]] = rank
    return ranks
