"""Leakage-safe baseline and grouped-residual predictors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


GROUP_COLUMNS = [
    "month",
    "season_bucket",
    "official_max_bin",
    "official_range_bin",
    "issue_hour_bucket",
    "month_x_official_max_round",
    "month_x_official_range_bin",
    "warm_x_official_max_bin",
]


def add_group_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["month_x_official_max_round"] = out["month"].astype(str) + "|" + out["official_max_round_c"].round().astype("Int64").astype(str)
    out["month_x_official_range_bin"] = out["month"].astype(str) + "|" + out["official_range_bin"].astype(str)
    out["warm_x_official_max_bin"] = out["warm_season_flag"].astype(str) + "|" + out["official_max_bin"].astype(str)
    return out


def raw_official_prediction(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame["anchor_forecast_max_c"], errors="coerce").to_numpy(dtype=float)


def climatology_persistence_prediction(frame: pd.DataFrame) -> np.ndarray:
    clim = pd.to_numeric(frame["target_clim_doy_30yr_median_c"], errors="coerce")
    lag2 = pd.to_numeric(frame["target_lag2_tmax_c"], errors="coerce")
    combined = 0.60 * clim + 0.40 * lag2
    return combined.fillna(clim).fillna(lag2).to_numpy(dtype=float)


def grouped_residual_prediction(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    shrink: float = 50.0,
) -> np.ndarray:
    train = add_group_keys(train)
    valid = add_group_keys(valid)
    global_mean = float(pd.to_numeric(train["residual_y_c"], errors="coerce").mean())
    if not math.isfinite(global_mean):
        global_mean = 0.0
    estimates: list[np.ndarray] = []
    weights: list[float] = []
    for column in GROUP_COLUMNS:
        mapping = shrunk_group_map(train, column, global_mean, shrink)
        estimate = valid[column].map(mapping).fillna(global_mean).to_numpy(dtype=float)
        estimates.append(estimate)
        weights.append(group_weight(column))
    if not estimates:
        residual = np.zeros(len(valid), dtype=float)
    else:
        weight_arr = np.array(weights, dtype=float)
        weight_arr = weight_arr / weight_arr.sum()
        residual = np.vstack(estimates).T @ weight_arr
    residual = np.clip(residual, -2.0, 2.0)
    return raw_official_prediction(valid) + residual


def shrunk_group_map(train: pd.DataFrame, column: str, global_mean: float, shrink: float) -> dict[Any, float]:
    if column not in train:
        return {}
    grouped = train.dropna(subset=[column, "residual_y_c"]).groupby(column)["residual_y_c"].agg(["count", "mean"])
    if grouped.empty:
        return {}
    grouped["shrunk"] = (grouped["count"] * grouped["mean"] + shrink * global_mean) / (grouped["count"] + shrink)
    grouped.loc[grouped["count"] < 20, "shrunk"] = global_mean
    return grouped["shrunk"].to_dict()


def group_weight(column: str) -> float:
    if column.startswith("month_x"):
        return 1.5
    if column in {"official_max_bin", "official_range_bin"}:
        return 1.2
    return 1.0

