"""Time-based feature helpers for regime experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from weather_ml import derived_features


@dataclass(frozen=True)
class EnsembleStats:
    mean: str = "ens_mean"
    median: str = "ens_median"
    min: str = "ens_min"
    max: str = "ens_max"
    range: str = "ens_range"
    std: str = "ens_std"
    iqr: str = "ens_iqr"
    mad: str = "ens_mad"
    outlier_gap: str = "ens_outlier_gap"


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.normalize()
    df = df.sort_values(["station_id", "target_date_local"])
    return df


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


def add_ensemble_stats(df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    values = df[model_cols].to_numpy(dtype=float)
    df["ens_mean"] = values.mean(axis=1)
    df["ens_median"] = np.median(values, axis=1)
    df["ens_min"] = np.min(values, axis=1)
    df["ens_max"] = np.max(values, axis=1)
    df["ens_range"] = df["ens_max"] - df["ens_min"]
    df["ens_std"] = np.std(values, axis=1, ddof=0)
    df["ens_iqr"] = np.quantile(values, 0.75, axis=1) - np.quantile(values, 0.25, axis=1)
    df["ens_mad"] = np.median(np.abs(values - df["ens_median"].to_numpy()[:, None]), axis=1)
    df["ens_outlier_gap"] = np.max(np.abs(values - df["ens_median"].to_numpy()[:, None]), axis=1)
    return df


def rolling_mean(
    series: pd.Series,
    *,
    window: int,
    min_periods: int,
    lag: int,
    group_key: pd.Series | None,
) -> pd.Series:
    if group_key is None:
        return series.shift(lag).rolling(window, min_periods=min_periods).mean()
    return series.groupby(group_key).transform(
        lambda s: s.shift(lag).rolling(window, min_periods=min_periods).mean()
    )


def rolling_std(
    series: pd.Series,
    *,
    window: int,
    min_periods: int,
    lag: int,
    group_key: pd.Series | None,
) -> pd.Series:
    if group_key is None:
        return series.shift(lag).rolling(window, min_periods=min_periods).std(ddof=0)
    return series.groupby(group_key).transform(
        lambda s: s.shift(lag).rolling(window, min_periods=min_periods).std(ddof=0)
    )


def rolling_quantile(
    series: pd.Series,
    *,
    window: int,
    min_periods: int,
    lag: int,
    q: float,
    group_key: pd.Series | None,
) -> pd.Series:
    if group_key is None:
        return series.shift(lag).rolling(window, min_periods=min_periods).quantile(q)
    return series.groupby(group_key).transform(
        lambda s: s.shift(lag).rolling(window, min_periods=min_periods).quantile(q)
    )


def rolling_sum(
    series: pd.Series,
    *,
    window: int,
    min_periods: int,
    lag: int,
    group_key: pd.Series | None,
) -> pd.Series:
    if group_key is None:
        return series.shift(lag).rolling(window, min_periods=min_periods).sum()
    return series.groupby(group_key).transform(
        lambda s: s.shift(lag).rolling(window, min_periods=min_periods).sum()
    )


def rolling_apply(
    series: pd.Series,
    *,
    window: int,
    min_periods: int,
    lag: int,
    func,
    group_key: pd.Series | None,
) -> pd.Series:
    if group_key is None:
        return series.shift(lag).rolling(window, min_periods=min_periods).apply(func, raw=True)
    return series.groupby(group_key).transform(
        lambda s: s.shift(lag).rolling(window, min_periods=min_periods).apply(func, raw=True)
    )


def rolling_slope(
    series: pd.Series,
    *,
    window: int,
    min_periods: int,
    lag: int,
    group_key: pd.Series | None,
) -> pd.Series:
    def _slope(values: np.ndarray) -> float:
        n = len(values)
        if n == 0:
            return np.nan
        x = np.arange(n, dtype=float)
        x_mean = (n - 1) / 2.0
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return 0.0
        y_mean = float(np.mean(values))
        return float(np.sum((x - x_mean) * (values - y_mean)) / denom)

    return rolling_apply(
        series,
        window=window,
        min_periods=min_periods,
        lag=lag,
        func=_slope,
        group_key=group_key,
    )


def ewm_mean(
    series: pd.Series,
    *,
    halflife: int,
    min_periods: int,
    lag: int,
    group_key: pd.Series | None,
) -> pd.Series:
    if group_key is None:
        return series.shift(lag).ewm(halflife=halflife, min_periods=min_periods, adjust=False).mean()
    return series.groupby(group_key).transform(
        lambda s: s.shift(lag).ewm(halflife=halflife, min_periods=min_periods, adjust=False).mean()
    )


def percent_rank(
    series: pd.Series,
    *,
    window: int,
    min_periods: int,
    lag: int,
    group_key: pd.Series | None,
) -> pd.Series:
    values = series.to_numpy(dtype=float)
    if group_key is None:
        stations = np.zeros(len(series), dtype=int)
    else:
        stations = group_key.to_numpy()
    output = np.full_like(values, np.nan, dtype=float)
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        for pos, row_idx in enumerate(idx):
            if pos < lag:
                continue
            start = max(0, pos - window)
            end = pos - lag + 1
            window_vals = values[idx[start:end]]
            if len(window_vals) < min_periods:
                continue
            output[row_idx] = float(np.mean(window_vals <= values[row_idx]))
    return pd.Series(output, index=series.index)


def ranks_with_tie_break(
    df: pd.DataFrame, model_cols: list[str]
) -> pd.DataFrame:
    values = df[model_cols].to_numpy(dtype=float)
    output = np.zeros_like(values, dtype=int)
    priority = {name: idx for idx, name in enumerate(derived_features.RAW_MODEL_ORDER)}
    for row_idx, row_vals in enumerate(values):
        tuples = []
        for idx, val in enumerate(row_vals):
            col = model_cols[idx]
            tuples.append((val, priority.get(col, idx), idx))
        tuples.sort()
        for rank, item in enumerate(tuples, start=1):
            output[row_idx, item[2]] = rank
    return pd.DataFrame(output, columns=[f"rank_{c}" for c in model_cols], index=df.index)


def argmax_with_tie_break(df: pd.DataFrame, model_cols: list[str]) -> pd.Series:
    values = df[model_cols].to_numpy(dtype=float)
    priority = {name: idx for idx, name in enumerate(derived_features.RAW_MODEL_ORDER)}
    output = []
    for row_vals in values:
        best = None
        for idx, val in enumerate(row_vals):
            col = model_cols[idx]
            key = (-val, priority.get(col, idx))
            if best is None or key < best[0]:
                best = (key, col)
        output.append(best[1])
    return pd.Series(output, index=df.index)


def argmin_with_tie_break(df: pd.DataFrame, model_cols: list[str]) -> pd.Series:
    values = df[model_cols].to_numpy(dtype=float)
    priority = {name: idx for idx, name in enumerate(derived_features.RAW_MODEL_ORDER)}
    output = []
    for row_vals in values:
        best = None
        for idx, val in enumerate(row_vals):
            col = model_cols[idx]
            key = (val, priority.get(col, idx))
            if best is None or key < best[0]:
                best = (key, col)
        output.append(best[1])
    return pd.Series(output, index=df.index)


def rank_data(df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    priority = {name: idx for idx, name in enumerate(derived_features.RAW_MODEL_ORDER)}
    values = df[model_cols].to_numpy(dtype=float)
    ranks = np.full_like(values, np.nan, dtype=float)
    priorities = [priority.get(col, idx) for idx, col in enumerate(model_cols)]
    for row_idx, row in enumerate(values):
        order = sorted(
            range(len(model_cols)),
            key=lambda i: (
                row[i] if np.isfinite(row[i]) else np.inf,
                priorities[i],
            ),
        )
        for rank_pos, col_idx in enumerate(order, start=1):
            ranks[row_idx, col_idx] = rank_pos
    return pd.DataFrame(ranks, columns=model_cols, index=df.index)


def rolling_event_count(
    indicator: pd.Series,
    *,
    window: int,
    min_periods: int,
    lag: int,
    group_key: pd.Series | None,
) -> pd.Series:
    return rolling_sum(
        indicator,
        window=window,
        min_periods=min_periods,
        lag=lag,
        group_key=group_key,
    )


def rolling_event_mean(
    indicator: pd.Series,
    *,
    window: int,
    min_periods: int,
    lag: int,
    group_key: pd.Series | None,
) -> pd.Series:
    if group_key is None:
        return indicator.shift(lag).rolling(window, min_periods=min_periods).mean()
    return indicator.groupby(group_key).transform(
        lambda s: s.shift(lag).rolling(window, min_periods=min_periods).mean()
    )


def rolling_corr(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    window: int,
    min_periods: int,
    lag: int,
    group_key: pd.Series | None,
) -> pd.Series:
    if group_key is None:
        a = series_a.shift(lag)
        b = series_b.shift(lag)
        return a.rolling(window, min_periods=min_periods).corr(b)
    output = pd.Series(np.nan, index=series_a.index, dtype=float)
    stations = group_key.to_numpy()
    for station in np.unique(stations):
        mask = stations == station
        a = series_a.loc[mask].shift(lag)
        b = series_b.loc[mask].shift(lag)
        output.loc[mask] = a.rolling(window, min_periods=min_periods).corr(b)
    return output


def rolling_conditional_mean(
    values: pd.Series,
    indicator: pd.Series,
    *,
    window: int,
    min_periods: int,
    lag: int,
    group_key: pd.Series | None,
) -> pd.Series:
    weighted = values * indicator.astype(float)
    numerator = rolling_sum(
        weighted, window=window, min_periods=min_periods, lag=lag, group_key=group_key
    )
    denom = rolling_sum(
        indicator.astype(float),
        window=window,
        min_periods=min_periods,
        lag=lag,
        group_key=group_key,
    )
    denom = denom.replace(0.0, np.nan)
    return numerator / denom


def days_since_event(
    indicator: pd.Series, *, lag: int, cap: int, group_key: pd.Series | None
) -> pd.Series:
    values = indicator.to_numpy(dtype=int)
    if group_key is None:
        stations = np.zeros(len(indicator), dtype=int)
    else:
        stations = group_key.to_numpy()
    output = np.full_like(values, np.nan, dtype=float)
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        last_event = None
        for pos, row_idx in enumerate(idx):
            if pos < lag:
                continue
            search_pos = pos - lag
            if values[idx[search_pos]] == 1:
                last_event = search_pos
            if last_event is None:
                output[row_idx] = float(cap)
            else:
                output[row_idx] = float(min(cap, search_pos - last_event))
    return pd.Series(output, index=indicator.index)


def streak_length(
    series: pd.Series, *, lag: int, cap: int, group_key: pd.Series | None
) -> pd.Series:
    values = series.to_numpy()
    if group_key is None:
        stations = np.zeros(len(series), dtype=int)
    else:
        stations = group_key.to_numpy()
    output = np.full_like(values, np.nan, dtype=float)
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        current_len = 0
        last_val = None
        for pos, row_idx in enumerate(idx):
            if pos < lag:
                continue
            val = values[idx[pos - lag]]
            if last_val is None or val != last_val:
                current_len = 1
                last_val = val
            else:
                current_len += 1
            output[row_idx] = float(min(cap, current_len))
    return pd.Series(output, index=series.index)


def switch_count(
    series: pd.Series,
    *,
    window: int,
    min_periods: int,
    lag: int,
    group_key: pd.Series | None,
) -> pd.Series:
    values = series.to_numpy()
    if group_key is None:
        stations = np.zeros(len(series), dtype=int)
    else:
        stations = group_key.to_numpy()
    output = np.full_like(values, np.nan, dtype=float)
    for station in np.unique(stations):
        idx = np.where(stations == station)[0]
        for pos, row_idx in enumerate(idx):
            if pos < lag:
                continue
            start = max(0, pos - window)
            end = pos - lag + 1
            window_vals = values[idx[start:end]]
            if len(window_vals) < min_periods:
                continue
            switches = np.sum(window_vals[1:] != window_vals[:-1])
            output[row_idx] = float(switches)
    return pd.Series(output, index=series.index)
