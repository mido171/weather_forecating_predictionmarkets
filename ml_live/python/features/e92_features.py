from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd


ENSEMBLE_COMPONENTS = [
    "nbm_tmax_f",
    "hrrr_tmax_f",
    "rap_tmax_f",
    "gefsatmosmean_tmax_f",
    "gfs_n_x_max",
    "nam_n_x_max",
]


def build_feature_vector(
    feature_list: list[str],
    base_row: dict,
    history_features: pd.DataFrame,
    history_truth: pd.DataFrame,
    target_date: date,
    truth_lag_days: int = 2,
    window_days: int = 60,
    allow_partial_history: bool = False,
    min_bias_residuals: int | None = None,
) -> pd.DataFrame:
    features: dict[str, float] = {}
    features.update(_calendar_features(target_date))

    for key, value in base_row.items():
        if value is None:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            features[key] = float(value)

    if "ensmean" in feature_list or any(
        name.startswith("bias_ensmean") or name.startswith("ensmean_corr") for name in feature_list
    ):
        features["ensmean"] = _compute_ensmean(pd.Series(features))

    history_features = history_features.copy()
    history_truth = history_truth.copy()
    if "target_date_local" in history_features.columns:
        history_features["target_date_local"] = pd.to_datetime(
            history_features["target_date_local"]
        ).dt.date
    if "target_date_local" in history_truth.columns:
        history_truth["target_date_local"] = pd.to_datetime(
            history_truth["target_date_local"]
        ).dt.date
    history_truth = history_truth[["station_id", "target_date_local", "actual_tmax_f"]]

    history_features["ensmean"] = history_features.apply(_compute_ensmean, axis=1)
    history = history_features.merge(history_truth, on=["station_id", "target_date_local"], how="inner")

    history_end = target_date - timedelta(days=truth_lag_days)
    history_start = history_end - timedelta(days=window_days - 1)
    history = history[
        (history["target_date_local"] >= history_start)
        & (history["target_date_local"] <= history_end)
    ]

    if history.empty:
        raise ValueError("No history available to compute rolling bias features")

    sources = _bias_sources(feature_list)
    for source in sources:
        bias = _compute_bias(
            history,
            source,
            window_days,
            allow_partial=allow_partial_history,
            min_residuals=min_bias_residuals,
        )
        features[f"bias_{source}_rm60_l2"] = bias
        base_value = features.get(source)
        if base_value is None:
            raise ValueError(f"Missing base feature for bias correction: {source}")
        features[f"{source}_corr_rm60_l2"] = base_value + bias

    missing = []
    for name in feature_list:
        if name not in features:
            missing.append(name)
            continue
        value = features[name]
        if value is None:
            missing.append(name)
            continue
        if pd.isna(value):
            missing.append(name)
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    ordered = {name: features[name] for name in feature_list}
    return pd.DataFrame([ordered])


def _calendar_features(target_date: date) -> dict[str, float]:
    day_of_year = target_date.timetuple().tm_yday
    month = target_date.month
    sin_doy = np.sin(2.0 * np.pi * day_of_year / 365.0)
    cos_doy = np.cos(2.0 * np.pi * day_of_year / 365.0)
    is_weekend = 1.0 if target_date.weekday() >= 5 else 0.0
    return {
        "month": float(month),
        "day_of_year": float(day_of_year),
        "sin_doy": float(sin_doy),
        "cos_doy": float(cos_doy),
        "is_weekend": float(is_weekend),
    }


def _compute_ensmean(row: pd.Series) -> float:
    values = [row.get(col) for col in ENSEMBLE_COMPONENTS]
    values = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _bias_sources(feature_list: Iterable[str]) -> list[str]:
    sources: list[str] = []
    for name in feature_list:
        if name.startswith("bias_") and name.endswith("_rm60_l2"):
            source = name[len("bias_") : -len("_rm60_l2")]
            sources.append(source)
    return sources


def _compute_bias(
    history: pd.DataFrame,
    source: str,
    window_days: int,
    allow_partial: bool = False,
    min_residuals: int | None = None,
) -> float:
    if source not in history.columns:
        raise ValueError(f"Missing source column in history: {source}")
    residuals = history["actual_tmax_f"].to_numpy(dtype=float) - history[source].to_numpy(dtype=float)
    residuals = residuals[~np.isnan(residuals)]
    if allow_partial:
        required = min_residuals if min_residuals is not None else 1
        if len(residuals) < required:
            raise ValueError(
                f"Insufficient residuals for bias_{source}_rm60_l2: {len(residuals)} < {required}"
            )
    elif len(residuals) < window_days:
        raise ValueError(
            f"Insufficient residuals for bias_{source}_rm60_l2: {len(residuals)} < {window_days}"
        )
    return float(np.mean(residuals))
