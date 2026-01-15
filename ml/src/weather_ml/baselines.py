"""Baseline predictors."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def predict_ensemble_mean(df: pd.DataFrame, model_cols: list[str]) -> np.ndarray:
    return df[model_cols].mean(axis=1).to_numpy()


def predict_climatology(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    *,
    label_col: str,
    station_col: str,
    date_col: str,
    label_lag_days: int,
) -> np.ndarray:
    train_df = train_df.copy()
    train_df["day_of_year"] = pd.to_datetime(train_df[date_col]).dt.dayofyear
    train_df["target_date_local"] = pd.to_datetime(train_df[date_col]).dt.date
    climo_series = _build_climo_series(train_df, station_col, label_col)
    station_mean = train_df.groupby(station_col)[label_col].mean()

    predict_df = predict_df.copy()
    predict_df["day_of_year"] = pd.to_datetime(predict_df[date_col]).dt.dayofyear
    predict_df["target_date_local"] = pd.to_datetime(predict_df[date_col]).dt.date

    preds: list[float] = []
    for _, row in predict_df.iterrows():
        key = (str(row[station_col]), int(row["day_of_year"]))
        target_date = row["target_date_local"]
        series = climo_series.get(key)
        value = None
        if series is not None:
            dates_ord, temps = series
            cutoff = (target_date - timedelta(days=label_lag_days)).toordinal()
            idx = np.searchsorted(dates_ord, cutoff, side="right")
            subset = temps[:idx]
            if subset.size:
                value = float(subset.mean())
        if value is None or np.isnan(value):
            station_value = station_mean.get(row[station_col])
            value = float(station_value) if station_value is not None else np.nan
        preds.append(value)
    return np.array(preds, dtype=float)


def _build_climo_series(
    df: pd.DataFrame,
    station_col: str,
    label_col: str,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    series: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for (station_id, doy), group in df.groupby([station_col, "day_of_year"]):
        group = group.sort_values("target_date_local")
        dates_ord = np.array([value.toordinal() for value in group["target_date_local"]])
        temps = group[label_col].to_numpy(dtype=float)
        series[(str(station_id), int(doy))] = (dates_ord, temps)
    return series
