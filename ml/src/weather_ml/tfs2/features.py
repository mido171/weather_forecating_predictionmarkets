"""Feature helpers for TFS2 experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from weather_ml import time_feature_library as tfl
from weather_ml import hmm_utils

from .config import GUIDANCE_COLS, TRUTH_LAG_DAYS

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    frame: pd.DataFrame
    columns: list[str]
    meta: dict


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df["target_date_local"])
    df["month"] = dates.dt.month.astype(int)
    df["day_of_year"] = dates.dt.dayofyear.astype(int)
    radians = 2 * np.pi * df["day_of_year"] / 365.25
    df["sin_doy"] = np.sin(radians)
    df["cos_doy"] = np.cos(radians)
    df["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
    return df


def ensemble_stats(df: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    values = df[cols].to_numpy(dtype=float)
    stats = {
        "mean": np.nanmean(values, axis=1),
        "median": np.nanmedian(values, axis=1),
        "min": np.nanmin(values, axis=1),
        "max": np.nanmax(values, axis=1),
    }
    stats["range"] = stats["max"] - stats["min"]
    stats["iqr"] = (
        np.nanquantile(values, 0.75, axis=1) - np.nanquantile(values, 0.25, axis=1)
    )
    stats["std"] = np.nanstd(values, axis=1)
    return stats


def sb_index(df: pd.DataFrame) -> pd.Series:
    hires = df[["rap_tmax_f", "hrrr_tmax_f"]].mean(axis=1)
    syn_cols = []
    for col in ["gefsatmosmean_tmax_f", "gfs_n_x_max", "nam_n_x_max"]:
        if col in df.columns:
            syn_cols.append(col)
    if not syn_cols:
        syn_cols = [col for col in ["gefsatmosmean_tmax_f", "gfs_tmax_f", "nam_tmax_f"] if col in df.columns]
    syn = df[syn_cols].mean(axis=1)
    return syn - hires


def mos_value(df: pd.DataFrame, code: str, stat: str = "mean") -> pd.Series:
    gfs_col = f"mos_gfs_{code}_{stat}"
    nam_col = f"mos_nam_{code}_{stat}"
    gfs = df.get(gfs_col, pd.Series(np.nan, index=df.index)).astype(float)
    nam = df.get(nam_col, pd.Series(np.nan, index=df.index)).astype(float)
    blend = 0.5 * (gfs + nam)
    blend = blend.where(~gfs.isna(), nam)
    blend = blend.where(~nam.isna(), gfs)
    return blend


def _grouped_shift(series: pd.Series, group_key: pd.Series, lag: int) -> pd.Series:
    return series.groupby(group_key).shift(lag)


def rolling_mean(series: pd.Series, window: int, min_obs: int, group_key: pd.Series, lag: int) -> pd.Series:
    shifted = _grouped_shift(series, group_key, lag)
    return shifted.groupby(group_key).rolling(window, min_periods=min_obs).mean().reset_index(level=0, drop=True)


def rolling_median(series: pd.Series, window: int, min_obs: int, group_key: pd.Series, lag: int) -> pd.Series:
    shifted = _grouped_shift(series, group_key, lag)
    return shifted.groupby(group_key).rolling(window, min_periods=min_obs).median().reset_index(level=0, drop=True)


def rolling_mae(series: pd.Series, window: int, min_obs: int, group_key: pd.Series, lag: int) -> pd.Series:
    shifted = _grouped_shift(series.abs(), group_key, lag)
    return shifted.groupby(group_key).rolling(window, min_periods=min_obs).mean().reset_index(level=0, drop=True)


def rolling_quantile(series: pd.Series, window: int, q: float, min_obs: int, group_key: pd.Series, lag: int) -> pd.Series:
    shifted = _grouped_shift(series, group_key, lag)
    return shifted.groupby(group_key).rolling(window, min_periods=min_obs).quantile(q).reset_index(level=0, drop=True)


def rolling_corr(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int,
    min_obs: int,
    group_key: pd.Series,
    lag: int,
) -> pd.Series:
    a_shift = _grouped_shift(series_a, group_key, lag)
    b_shift = _grouped_shift(series_b, group_key, lag)
    return (
        a_shift.groupby(group_key)
        .rolling(window, min_periods=min_obs)
        .corr(b_shift)
        .reset_index(level=0, drop=True)
    )


def ewma(series: pd.Series, halflife: float, group_key: pd.Series, lag: int) -> pd.Series:
    shifted = _grouped_shift(series, group_key, lag)
    return shifted.groupby(group_key).apply(
        lambda s: s.ewm(halflife=halflife, min_periods=1, adjust=False).mean()
    ).reset_index(level=0, drop=True)


def disagreement_pca(df: pd.DataFrame, cols: list[str], train_mask: np.ndarray, n_components: int = 2) -> pd.DataFrame:
    values = df[cols].to_numpy(dtype=float)
    ens_mean = np.nanmean(values, axis=1, keepdims=True)
    diff = values - ens_mean
    train_vals = diff[train_mask]
    train_vals = np.where(np.isfinite(train_vals), train_vals, 0.0)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_vals)
    transformed = pca.transform(np.where(np.isfinite(diff), diff, 0.0))
    out = pd.DataFrame(index=df.index)
    for i in range(n_components):
        out[f"pc{i+1}"] = transformed[:, i]
    return out


def compute_tail_depths(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    stats = ensemble_stats(df, cols)
    cool_tail = stats["median"] - stats["min"]
    warm_tail = stats["max"] - stats["median"]
    tail_asym = warm_tail - cool_tail
    return pd.DataFrame(
        {
            "ens_min": stats["min"],
            "ens_med": stats["median"],
            "ens_max": stats["max"],
            "cool_tail": cool_tail,
            "warm_tail": warm_tail,
            "tail_asym": tail_asym,
        },
        index=df.index,
    )


def transition_bumps(dates: pd.Series, center: int, width: int) -> np.ndarray:
    doy = pd.to_datetime(dates).dt.dayofyear.to_numpy(dtype=float)
    return np.exp(-((doy - center) / width) ** 2)


def fit_hmm_probs(z: np.ndarray, train_mask: np.ndarray, n_states: int, seed: int = 0) -> np.ndarray:
    train_vals = z[train_mask]
    train_vals = train_vals[np.isfinite(train_vals)]
    if train_vals.size < n_states:
        return np.full((len(z), n_states), np.nan)
    obs = train_vals.reshape(-1, 1)
    params = hmm_utils.fit_gaussian_hmm(obs, n_states=n_states, seed=seed)
    probs = hmm_utils.forward_filter(z.reshape(-1, 1), params)
    return probs


def truth_lag_days() -> int:
    return TRUTH_LAG_DAYS
