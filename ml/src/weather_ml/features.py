"""Feature engineering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd


@dataclass
class FeatureState:
    station_categories: list[str]
    base_impute_means: dict[str, float]
    feature_means: dict[str, float]
    feature_columns: list[str]
    climatology_series: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]
    rolling_series: dict[str, tuple[np.ndarray, np.ndarray]]
    label_lag_days: int


def build_features(
    df: pd.DataFrame,
    *,
    config,
    fit_state: FeatureState | None = None,
    training: bool = False,
) -> tuple[pd.DataFrame, np.ndarray | None, FeatureState]:
    df = df.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.normalize()

    y = None
    if "actual_tmax_f" in df.columns:
        y = df["actual_tmax_f"].to_numpy()

    if training or fit_state is None:
        fit_state = _build_state(df, config)

    base_features = list(config.features.base_features)
    missing_indicators = _missing_indicators(df, base_features)
    df = _impute_base_features(df, base_features, fit_state.base_impute_means)

    features = pd.concat(
        [missing_indicators, df[base_features].astype(float)],
        axis=1,
    )

    model_columns = [col for col in base_features if col.endswith("_tmax_f")]
    if config.features.ensemble_stats:
        features = _add_ensemble_stats(features, df, model_columns)
    if config.features.pairwise_deltas:
        features = _add_pairwise_deltas(
            features, df, model_columns, config.features.pairwise_pairs
        )
    if config.features.model_vs_ens_deltas:
        features = _add_model_vs_ens_deltas(features, df, model_columns)
    if config.features.calendar:
        features = _add_calendar_features(features, df)
    if config.features.station_onehot:
        features = _add_station_onehot(features, df, fit_state.station_categories)
    if config.features.climatology.enabled:
        features = _add_climatology_features(features, df, fit_state, config.features)

    features = features.astype(float)

    if training or not fit_state.feature_columns:
        feature_means = features.mean(axis=0, skipna=True).to_dict()
        fit_state.feature_means = feature_means
        fit_state.feature_columns = list(features.columns)
        features = features.fillna(pd.Series(feature_means))
    else:
        features = _align_feature_columns(features, fit_state)
        features = features.fillna(pd.Series(fit_state.feature_means))

    return features, y, fit_state


def _build_state(df: pd.DataFrame, config) -> FeatureState:
    station_categories = sorted(df["station_id"].dropna().unique().tolist())
    base_features = list(config.features.base_features)
    base_means = {}
    for column in base_features:
        mean_value = float(df[column].mean())
        if np.isnan(mean_value):
            raise ValueError(f"Cannot impute base feature {column}; mean is NaN.")
        base_means[column] = mean_value

    climatology_series: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    rolling_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    label_lag_days = config.features.climatology.label_lag_days

    if config.features.climatology.enabled:
        climatology_series = _build_climatology_series(df)
        rolling_series = _build_rolling_series(df)

    return FeatureState(
        station_categories=station_categories,
        base_impute_means=base_means,
        feature_means={},
        feature_columns=[],
        climatology_series=climatology_series,
        rolling_series=rolling_series,
        label_lag_days=label_lag_days,
    )


def _missing_indicators(df: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
    if not base_features:
        return pd.DataFrame(index=df.index)
    missing = df[base_features].isna().astype(float)
    missing.columns = [f"{column}_missing" for column in base_features]
    return missing


def _impute_base_features(
    df: pd.DataFrame,
    base_features: list[str],
    base_impute_means: dict[str, float],
) -> pd.DataFrame:
    df = df.copy()
    for column in base_features:
        df[column] = df[column].fillna(base_impute_means[column])
    return df


def _align_feature_columns(
    features: pd.DataFrame, fit_state: FeatureState
) -> pd.DataFrame:
    features = features.copy()
    for column in fit_state.feature_columns:
        if column not in features.columns:
            features[column] = np.nan
    extra_columns = [col for col in features.columns if col not in fit_state.feature_columns]
    if extra_columns:
        features = features.drop(columns=extra_columns)
    return features[fit_state.feature_columns]


def _add_ensemble_stats(
    features: pd.DataFrame,
    df: pd.DataFrame,
    model_columns: list[str],
) -> pd.DataFrame:
    if not model_columns:
        return features
    values = df[model_columns]
    ens_df = pd.DataFrame(
        {
            "ens_mean": values.mean(axis=1),
            "ens_median": values.median(axis=1),
            "ens_min": values.min(axis=1),
            "ens_max": values.max(axis=1),
            "ens_std": values.std(axis=1, ddof=0),
            "ens_iqr": values.quantile(0.75, axis=1) - values.quantile(0.25, axis=1),
        },
        index=df.index,
    )
    ens_df["ens_range"] = ens_df["ens_max"] - ens_df["ens_min"]
    return pd.concat([features, ens_df], axis=1)


def _add_pairwise_deltas(
    features: pd.DataFrame,
    df: pd.DataFrame,
    model_columns: list[str],
    pairs: list[list[str]],
) -> pd.DataFrame:
    if not pairs:
        pairs = [
            ["gfs_tmax_f", "nam_tmax_f"],
            ["gefsatmosmean_tmax_f", "gfs_tmax_f"],
            ["rap_tmax_f", "nbm_tmax_f"],
            ["nbm_tmax_f", "hrrr_tmax_f"],
            ["hrrr_tmax_f", "gfs_tmax_f"],
            ["nam_tmax_f", "nbm_tmax_f"],
        ]
    deltas: dict[str, pd.Series] = {}
    for left, right in pairs:
        if left in df.columns and right in df.columns:
            deltas[f"{left}_minus_{right}"] = df[left] - df[right]
    if not deltas:
        return features
    delta_df = pd.DataFrame(deltas, index=df.index)
    return pd.concat([features, delta_df], axis=1)


def _add_model_vs_ens_deltas(
    features: pd.DataFrame,
    df: pd.DataFrame,
    model_columns: list[str],
) -> pd.DataFrame:
    if "ens_mean" not in features.columns:
        features = _add_ensemble_stats(features, df, model_columns)
    if not model_columns:
        return features
    if "ens_mean" not in features.columns:
        return features
    ens_mean = features["ens_mean"]
    deltas: dict[str, pd.Series] = {}
    for column in model_columns:
        delta = df[column] - ens_mean
        deltas[f"{column}_minus_ens_mean"] = delta
        deltas[f"{column}_minus_ens_mean_abs"] = delta.abs()
    if not deltas:
        return features
    delta_df = pd.DataFrame(deltas, index=df.index)
    return pd.concat([features, delta_df], axis=1)


def _add_calendar_features(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["target_date_local"])
    calendar_df = pd.DataFrame(
        {
            "month": dates.dt.month.astype(int),
            "day_of_year": dates.dt.dayofyear.astype(int),
            "is_weekend": dates.dt.dayofweek.isin([5, 6]).astype(int),
        },
        index=df.index,
    )
    radians = 2 * np.pi * calendar_df["day_of_year"] / 365.25
    calendar_df["sin_doy"] = np.sin(radians)
    calendar_df["cos_doy"] = np.cos(radians)
    if "asof_utc" in df.columns:
        asof = pd.to_datetime(df["asof_utc"], utc=True, errors="coerce")
        hour = (
            asof.dt.hour.astype(float)
            + asof.dt.minute.astype(float) / 60.0
            + asof.dt.second.astype(float) / 3600.0
        )
        hour_radians = 2 * np.pi * hour / 24.0
        calendar_df["asof_sin_hour"] = np.sin(hour_radians)
        calendar_df["asof_cos_hour"] = np.cos(hour_radians)
    return pd.concat([features, calendar_df], axis=1)


def _add_station_onehot(
    features: pd.DataFrame,
    df: pd.DataFrame,
    station_categories: list[str],
) -> pd.DataFrame:
    categories = pd.Categorical(df["station_id"], categories=station_categories)
    dummies = pd.get_dummies(
        pd.Series(categories, index=df.index),
        prefix="station",
        dtype=float,
    )
    if dummies.empty:
        return features
    return pd.concat([features, dummies], axis=1)


def _build_climatology_series(
    df: pd.DataFrame,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    df = df[["station_id", "target_date_local", "actual_tmax_f"]].dropna()
    df = df.copy()
    df["day_of_year"] = pd.to_datetime(df["target_date_local"]).dt.dayofyear
    series: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for (station_id, doy), group in df.groupby(["station_id", "day_of_year"]):
        group = group.sort_values("target_date_local")
        dates = pd.to_datetime(group["target_date_local"]).dt.date
        dates_ord = np.array([value.toordinal() for value in dates])
        values = group["actual_tmax_f"].to_numpy(dtype=float)
        series[(str(station_id), int(doy))] = (dates_ord, values)
    return series


def _build_rolling_series(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    df = df[["station_id", "target_date_local", "actual_tmax_f"]].dropna()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    rolling_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for station_id, group in df.groupby("station_id"):
        sorted_group = group.sort_values("target_date_local")
        dates_ord = np.array(
            [value.toordinal() for value in sorted_group["target_date_local"]]
        )
        values = sorted_group["actual_tmax_f"].to_numpy(dtype=float)
        rolling_series[str(station_id)] = (dates_ord, values)
    return rolling_series


def _add_climatology_features(
    features: pd.DataFrame,
    df: pd.DataFrame,
    fit_state: FeatureState,
    feature_config,
) -> pd.DataFrame:
    dates = pd.to_datetime(df["target_date_local"]).dt.date
    doy = pd.to_datetime(df["target_date_local"]).dt.dayofyear
    climo_mean = []
    climo_std = []
    for station_id, day, target_date in zip(df["station_id"], doy, dates):
        values = fit_state.climatology_series.get((str(station_id), int(day)))
        if values is None:
            climo_mean.append(np.nan)
            climo_std.append(np.nan)
            continue
        dates_ord, temps = values
        cutoff = (target_date - timedelta(days=fit_state.label_lag_days)).toordinal()
        idx = np.searchsorted(dates_ord, cutoff, side="right")
        subset = temps[:idx]
        if subset.size == 0:
            climo_mean.append(np.nan)
            climo_std.append(np.nan)
        else:
            climo_mean.append(float(subset.mean()))
            climo_std.append(float(subset.std(ddof=0)))
    climo_df = pd.DataFrame(
        {
            "climo_mean_doy": climo_mean,
            "climo_std_doy": climo_std,
        },
        index=df.index,
    )

    if feature_config.climatology.rolling_windows_days:
        rolling = _compute_rolling_features(
            df,
            fit_state,
            feature_config.climatology.rolling_windows_days,
            fit_state.label_lag_days,
        )
        rolling_df = pd.DataFrame(rolling, index=df.index)
        climo_df = pd.concat([climo_df, rolling_df], axis=1)
    return pd.concat([features, climo_df], axis=1)


def _compute_rolling_features(
    df: pd.DataFrame,
    fit_state: FeatureState,
    windows: list[int],
    label_lag_days: int,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for window in windows:
        result[f"rolling_mean_{window}d"] = []
        result[f"rolling_std_{window}d"] = []

    for _, row in df.iterrows():
        station_id = str(row["station_id"])
        target_date = pd.to_datetime(row["target_date_local"]).date()
        series = fit_state.rolling_series.get(station_id)
        for window in windows:
            mean_key = f"rolling_mean_{window}d"
            std_key = f"rolling_std_{window}d"
            if series is None:
                result[mean_key].append(np.nan)
                result[std_key].append(np.nan)
                continue
            dates_ord, values = series
            end_date = target_date - timedelta(days=label_lag_days)
            start_date = end_date - timedelta(days=window - 1)
            end_ord = end_date.toordinal()
            start_ord = start_date.toordinal()
            left = np.searchsorted(dates_ord, start_ord, side="left")
            right = np.searchsorted(dates_ord, end_ord, side="right")
            slice_values = values[left:right]
            if slice_values.size == 0:
                result[mean_key].append(np.nan)
                result[std_key].append(np.nan)
            else:
                result[mean_key].append(float(slice_values.mean()))
                result[std_key].append(float(slice_values.std(ddof=0)))
    return result
