from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd


BASE_MODELS = [
    "nbm",
    "hrrr",
    "rap",
    "gefsmean",
    "gfsMOS",
    "namMOS",
]

BASE_MODEL_COLS = {
    "nbm": "nbm_tmax_f",
    "hrrr": "hrrr_tmax_f",
    "rap": "rap_tmax_f",
    "gefsmean": "gefsatmosmean_tmax_f",
    "gfsMOS": "gfs_n_x_max",
    "namMOS": "nam_n_x_max",
}

SPREAD_COL = "gefsatmos_tmp_spread_f"
ENSEMBLE_MEAN_COL = "ens_raw_mean"

BIAS_SOURCES = BASE_MODELS + [ENSEMBLE_MEAN_COL]

WINDOWS_MEAN = [7, 14, 30, 60, 120]
WINDOWS_MED = [30, 60, 120]
EWMA_HALFLIVES = [7, 30, 60]
WINDOWS_SKILL = [7, 14, 30, 60, 120]
WINDOWS_TRUTH = [7, 14, 30]

ANALOG_K = [10, 25, 50]


@dataclass(frozen=True)
class AnalogScaling:
    means: dict[str, float]
    stds: dict[str, float]


def generate_feature_list() -> list[str]:
    features: list[str] = []

    # Family A1: raw forecasts + spread
    features.extend(
        [
            "nbm_tmax_f",
            "hrrr_tmax_f",
            "rap_tmax_f",
            "gefsatmosmean_tmax_f",
            "gfs_n_x_max",
            "nam_n_x_max",
            "gefsatmos_tmp_spread_f",
        ]
    )

    # Family A2: ensemble summary stats over 6 forecasts
    features.extend(
        [
            "ens_raw_mean",
            "ens_raw_median",
            "ens_raw_min",
            "ens_raw_max",
            "ens_raw_range",
            "ens_raw_std",
            "ens_raw_mad",
            "ens_raw_iqr",
            "ens_raw_p10",
            "ens_raw_p90",
            "ens_raw_skew",
            "ens_raw_kurt",
        ]
    )

    # Family A3: missingness flags
    for col in [
        "nbm_tmax_f",
        "hrrr_tmax_f",
        "rap_tmax_f",
        "gefsatmosmean_tmax_f",
        "gfs_n_x_max",
        "nam_n_x_max",
        "gefsatmos_tmp_spread_f",
    ]:
        features.append(f"isnan_{col}")

    # Family A4: calendar features
    features.extend(
        [
            "month",
            "day_of_year",
            "day_of_month",
            "week_of_year",
            "is_weekend",
            "sin_doy",
            "cos_doy",
            "sin2_doy",
            "cos2_doy",
            "sin3_doy",
            "cos3_doy",
            "sin_month",
            "cos_month",
        ]
    )
    for month in range(1, 13):
        features.append(f"month_oh_{month:02d}")

    # Family B: multi-scale bias corrections
    for source in BIAS_SOURCES:
        for window in WINDOWS_MEAN:
            features.append(f"bias_mean_{source}_w{window}")
            features.append(f"corr_mean_{source}_w{window}")
    for source in BIAS_SOURCES:
        for window in WINDOWS_MED:
            features.append(f"bias_med_{source}_w{window}")
            features.append(f"corr_med_{source}_w{window}")
    for source in BIAS_SOURCES:
        for halflife in EWMA_HALFLIVES:
            features.append(f"bias_ewm_{source}_h{halflife}")
            features.append(f"corr_ewm_{source}_h{halflife}")
    for source in BIAS_SOURCES:
        features.append(f"bias_kf_fast_{source}")
        features.append(f"corr_kf_fast_{source}")
        features.append(f"bias_kf_slow_{source}")
        features.append(f"corr_kf_slow_{source}")

    # Family C: residual distribution stats
    for source in BIAS_SOURCES:
        for window in WINDOWS_MEAN:
            features.extend(
                [
                    f"resid_std_{source}_w{window}",
                    f"resid_mad_{source}_w{window}",
                    f"resid_iqr_{source}_w{window}",
                    f"resid_q10_{source}_w{window}",
                    f"resid_q90_{source}_w{window}",
                    f"resid_tail2_{source}_w{window}",
                ]
            )

    # Family D: recent skill + dynamic weights
    for model in BASE_MODELS:
        for window in WINDOWS_SKILL:
            features.extend(
                [
                    f"mae_raw_{model}_w{window}",
                    f"rmse_raw_{model}_w{window}",
                    f"bias_raw_{model}_w{window}",
                ]
            )
    for model in BASE_MODELS:
        for window in WINDOWS_SKILL:
            features.append(f"w_mae_{model}_w{window}")
    for window in WINDOWS_SKILL:
        features.append(f"ens_wt_mae_w{window}")
    for model in BASE_MODELS:
        features.append(f"best_mae_w30_{model}")
    for model in BASE_MODELS:
        features.append(f"best_mae_w60_{model}")

    # Family E: disagreement/regime signals
    pairs = _model_pairs(BASE_MODELS)
    for left, right in pairs:
        features.append(f"pair_diff_{left}_{right}")
    for left, right in pairs:
        features.append(f"pair_absdiff_{left}_{right}")
    for left, right in pairs:
        features.append(f"pair_sqdiff_{left}_{right}")
    for model in BASE_MODELS:
        features.append(f"dev_{model}_ensmean")
        features.append(f"absdev_{model}_ensmean")
    features.extend(
        [
            "cons_std_6",
            "cons_mad_6",
            "cons_iqr_6",
            "cons_range_6",
            "cons_std_minus_gefs",
            "cons_std_over_gefs",
            "cons_std_times_gefs",
        ]
    )
    for model in BASE_MODELS:
        features.append(f"is_max_{model}")
    for model in BASE_MODELS:
        features.append(f"is_min_{model}")

    # Family F: persistence + climatology
    for lag in range(2, 11):
        features.append(f"actual_lag{lag}")
    for window in WINDOWS_TRUTH:
        features.extend(
            [
                f"actual_mean_w{window}",
                f"actual_std_w{window}",
                f"actual_min_w{window}",
                f"actual_max_w{window}",
            ]
        )
    features.extend(["clim_mean_doy", "clim_std_doy", "clim_p10_doy", "clim_p90_doy"])
    for model in BASE_MODELS:
        features.append(f"anom_{model}")
    features.append("anom_ensmean")
    features.append("anom_actual_lag2")
    features.extend(
        [
            "delta_lag2_lag3",
            "delta_lag2_lag7",
            "trend_short",
            "abs_trend_short",
        ]
    )

    # Family G: analog ensemble features
    for k in ANALOG_K:
        features.extend(
            [
                f"anen_mean_k{k}",
                f"anen_median_k{k}",
                f"anen_p10_k{k}",
                f"anen_p90_k{k}",
                f"anen_iqr_k{k}",
                f"anen_std_k{k}",
                f"anen_min_k{k}",
                f"anen_max_k{k}",
            ]
        )
    for k in ANALOG_K:
        features.extend(
            [
                f"anen_dist_mean_k{k}",
                f"anen_dist_min_k{k}",
                f"anen_dist_std_k{k}",
            ]
        )
    for model in BASE_MODELS:
        features.append(f"anen_resid_mean_{model}_k25")
    features.append("anen_resid_mean_ens_k25")
    features.extend(["anen_daydiff_mean_k25", "anen_daydiff_min_k25"])
    features.extend(
        [
            "anen_wmean_k25",
            "anen_wmedian_k25",
            "anen_wstd_k25",
            "anen_wiqr_k25",
        ]
    )

    if len(features) != 739:
        raise ValueError(f"E450 feature list expected 739 features, got {len(features)}")
    return features


def build_e450_features(
    df: pd.DataFrame,
    *,
    train_start: date,
    train_end: date,
    truth_lag_days: int = 2,
) -> tuple[pd.DataFrame, AnalogScaling]:
    df = df.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df = df.sort_values("target_date_local").reset_index(drop=True)

    dates = pd.to_datetime(df["target_date_local"])
    df["month"] = dates.dt.month.astype(int)
    df["day_of_year"] = dates.dt.dayofyear.astype(int)
    df["day_of_month"] = dates.dt.day.astype(int)
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)

    doy_radians = 2.0 * np.pi * df["day_of_year"].to_numpy(dtype=float) / 365.0
    df["sin_doy"] = np.sin(doy_radians)
    df["cos_doy"] = np.cos(doy_radians)
    df["sin2_doy"] = np.sin(2.0 * doy_radians)
    df["cos2_doy"] = np.cos(2.0 * doy_radians)
    df["sin3_doy"] = np.sin(3.0 * doy_radians)
    df["cos3_doy"] = np.cos(3.0 * doy_radians)

    month_radians = 2.0 * np.pi * df["month"].to_numpy(dtype=float) / 12.0
    df["sin_month"] = np.sin(month_radians)
    df["cos_month"] = np.cos(month_radians)
    for month in range(1, 13):
        df[f"month_oh_{month:02d}"] = (df["month"] == month).astype(int)

    for col in [
        "nbm_tmax_f",
        "hrrr_tmax_f",
        "rap_tmax_f",
        "gefsatmosmean_tmax_f",
        "gfs_n_x_max",
        "nam_n_x_max",
        "gefsatmos_tmp_spread_f",
    ]:
        df[f"isnan_{col}"] = df[col].isna().astype(int)

    forecast_matrix = df[[BASE_MODEL_COLS[m] for m in BASE_MODELS]].to_numpy(dtype=float)
    stats = _ensemble_stats(forecast_matrix)
    for key, values in stats.items():
        df[key] = values

    # Base residuals
    actual = df["actual_tmax_f"].to_numpy(dtype=float)
    residuals: dict[str, pd.Series] = {}
    for model, col in BASE_MODEL_COLS.items():
        residuals[model] = pd.Series(actual - df[col].to_numpy(dtype=float), index=dates)
    residuals[ENSEMBLE_MEAN_COL] = pd.Series(
        actual - df[ENSEMBLE_MEAN_COL].to_numpy(dtype=float), index=dates
    )

    # Rolling bias/skill/residual features (time-safe via shift)
    for source in BIAS_SOURCES:
        series = residuals[source]
        for window in WINDOWS_MEAN:
            df[f"bias_mean_{source}_w{window}"] = (
                series.rolling(window=window, min_periods=window)
                .mean()
                .shift(truth_lag_days)
                .to_numpy(dtype=float)
            )
            df[f"corr_mean_{source}_w{window}"] = (
                df[_source_col(source)] + df[f"bias_mean_{source}_w{window}"]
            )
        for window in WINDOWS_MED:
            df[f"bias_med_{source}_w{window}"] = (
                series.rolling(window=window, min_periods=window)
                .median()
                .shift(truth_lag_days)
                .to_numpy(dtype=float)
            )
            df[f"corr_med_{source}_w{window}"] = (
                df[_source_col(source)] + df[f"bias_med_{source}_w{window}"]
            )

        for halflife in EWMA_HALFLIVES:
            df[f"bias_ewm_{source}_h{halflife}"] = (
                series.ewm(halflife=halflife, adjust=False, ignore_na=True)
                .mean()
                .shift(truth_lag_days)
                .to_numpy(dtype=float)
            )
            df[f"corr_ewm_{source}_h{halflife}"] = (
                df[_source_col(source)] + df[f"bias_ewm_{source}_h{halflife}"]
            )

    # Kalman bias states (computed per source, then shifted)
    kalman_params = _kalman_params(residuals, train_start, train_end)
    for source in BIAS_SOURCES:
        series = residuals[source]
        params = kalman_params.get(source)
        if params is None:
            df[f"bias_kf_fast_{source}"] = np.nan
            df[f"corr_kf_fast_{source}"] = np.nan
            df[f"bias_kf_slow_{source}"] = np.nan
            df[f"corr_kf_slow_{source}"] = np.nan
            continue
        bias_fast = _kalman_filter(series, params["q_fast"], params["r"]).shift(truth_lag_days)
        bias_slow = _kalman_filter(series, params["q_slow"], params["r"]).shift(truth_lag_days)
        df[f"bias_kf_fast_{source}"] = bias_fast.to_numpy(dtype=float)
        df[f"corr_kf_fast_{source}"] = df[_source_col(source)] + df[f"bias_kf_fast_{source}"]
        df[f"bias_kf_slow_{source}"] = bias_slow.to_numpy(dtype=float)
        df[f"corr_kf_slow_{source}"] = df[_source_col(source)] + df[f"bias_kf_slow_{source}"]

    # Residual distribution features
    for source in BIAS_SOURCES:
        series = residuals[source]
        for window in WINDOWS_MEAN:
            df[f"resid_std_{source}_w{window}"] = (
                series.rolling(window=window, min_periods=window)
                .std(ddof=0)
                .shift(truth_lag_days)
                .to_numpy(dtype=float)
            )
            df[f"resid_mad_{source}_w{window}"] = (
                series.rolling(window=window, min_periods=window)
                .apply(_mad, raw=True)
                .shift(truth_lag_days)
                .to_numpy(dtype=float)
            )
            df[f"resid_iqr_{source}_w{window}"] = (
                series.rolling(window=window, min_periods=window)
                .apply(_iqr, raw=True)
                .shift(truth_lag_days)
                .to_numpy(dtype=float)
            )
            df[f"resid_q10_{source}_w{window}"] = (
                series.rolling(window=window, min_periods=window)
                .quantile(0.1)
                .shift(truth_lag_days)
                .to_numpy(dtype=float)
            )
            df[f"resid_q90_{source}_w{window}"] = (
                series.rolling(window=window, min_periods=window)
                .quantile(0.9)
                .shift(truth_lag_days)
                .to_numpy(dtype=float)
            )
            df[f"resid_tail2_{source}_w{window}"] = (
                series.rolling(window=window, min_periods=window)
                .apply(_tail2, raw=True)
                .shift(truth_lag_days)
                .to_numpy(dtype=float)
            )

    # Skill metrics (MAE/RMSE/Bias) for base models
    for model in BASE_MODELS:
        series = residuals[model]
        for window in WINDOWS_SKILL:
            df[f"mae_raw_{model}_w{window}"] = (
                series.abs()
                .rolling(window=window, min_periods=window)
                .mean()
                .shift(truth_lag_days)
                .to_numpy(dtype=float)
            )
            df[f"rmse_raw_{model}_w{window}"] = (
                (series**2)
                .rolling(window=window, min_periods=window)
                .mean()
                .pow(0.5)
                .shift(truth_lag_days)
                .to_numpy(dtype=float)
            )
            df[f"bias_raw_{model}_w{window}"] = (
                series.rolling(window=window, min_periods=window)
                .mean()
                .shift(truth_lag_days)
                .to_numpy(dtype=float)
            )

    # Dynamic weights + weighted ensemble
    for window in WINDOWS_SKILL:
        weights = []
        for model in BASE_MODELS:
            col = f"mae_raw_{model}_w{window}"
            weight = 1.0 / (df[col].to_numpy(dtype=float) + 0.1)
            weights.append(weight)
        weight_matrix = np.vstack(weights).T
        weight_sum = np.nansum(weight_matrix, axis=1)
        for idx, model in enumerate(BASE_MODELS):
            w = weight_matrix[:, idx]
            normalized = np.where(weight_sum > 0.0, w / weight_sum, np.nan)
            df[f"w_mae_{model}_w{window}"] = normalized
        forecast_vals = df[[BASE_MODEL_COLS[m] for m in BASE_MODELS]].to_numpy(dtype=float)
        ens_wt = np.nansum(
            forecast_vals * np.where(weight_sum[:, None] > 0.0, weight_matrix / weight_sum[:, None], np.nan),
            axis=1,
        )
        df[f"ens_wt_mae_w{window}"] = ens_wt

    for window in [30, 60]:
        mae_cols = [f"mae_raw_{model}_w{window}" for model in BASE_MODELS]
        mae_vals = df[mae_cols].to_numpy(dtype=float)
        best_idx = np.full(mae_vals.shape[0], -1, dtype=int)
        valid = np.isfinite(mae_vals).any(axis=1)
        if np.any(valid):
            best_idx[valid] = np.nanargmin(mae_vals[valid], axis=1)
        for idx, model in enumerate(BASE_MODELS):
            df[f"best_mae_w{window}_{model}"] = (best_idx == idx).astype(int)

    # Disagreement features
    for left, right in _model_pairs(BASE_MODELS):
        left_vals = df[BASE_MODEL_COLS[left]].to_numpy(dtype=float)
        right_vals = df[BASE_MODEL_COLS[right]].to_numpy(dtype=float)
        diff = left_vals - right_vals
        df[f"pair_diff_{left}_{right}"] = diff
        df[f"pair_absdiff_{left}_{right}"] = np.abs(diff)
        df[f"pair_sqdiff_{left}_{right}"] = diff**2

    for model in BASE_MODELS:
        vals = df[BASE_MODEL_COLS[model]].to_numpy(dtype=float)
        ens_mean = df[ENSEMBLE_MEAN_COL].to_numpy(dtype=float)
        diff = vals - ens_mean
        df[f"dev_{model}_ensmean"] = diff
        df[f"absdev_{model}_ensmean"] = np.abs(diff)

    df["cons_std_6"] = df["ens_raw_std"]
    df["cons_mad_6"] = df["ens_raw_mad"]
    df["cons_iqr_6"] = df["ens_raw_iqr"]
    df["cons_range_6"] = df["ens_raw_range"]
    df["cons_std_minus_gefs"] = df["cons_std_6"] - df[SPREAD_COL]
    df["cons_std_over_gefs"] = df["cons_std_6"] / (df[SPREAD_COL] + 0.1)
    df["cons_std_times_gefs"] = df["cons_std_6"] * df[SPREAD_COL]

    forecasts = df[[BASE_MODEL_COLS[m] for m in BASE_MODELS]].to_numpy(dtype=float)
    max_vals = np.nanmax(forecasts, axis=1)
    min_vals = np.nanmin(forecasts, axis=1)
    for idx, model in enumerate(BASE_MODELS):
        model_vals = forecasts[:, idx]
        df[f"is_max_{model}"] = (model_vals == max_vals).astype(int)
        df[f"is_min_{model}"] = (model_vals == min_vals).astype(int)

    # Persistence and climatology
    actual_series = pd.Series(actual, index=dates)
    for lag in range(2, 11):
        lag_dates = dates - pd.Timedelta(days=lag)
        df[f"actual_lag{lag}"] = actual_series.reindex(lag_dates).to_numpy(dtype=float)

    for window in WINDOWS_TRUTH:
        df[f"actual_mean_w{window}"] = (
            actual_series.rolling(window=window, min_periods=window)
            .mean()
            .shift(truth_lag_days)
            .to_numpy(dtype=float)
        )
        df[f"actual_std_w{window}"] = (
            actual_series.rolling(window=window, min_periods=window)
            .std(ddof=0)
            .shift(truth_lag_days)
            .to_numpy(dtype=float)
        )
        df[f"actual_min_w{window}"] = (
            actual_series.rolling(window=window, min_periods=window)
            .min()
            .shift(truth_lag_days)
            .to_numpy(dtype=float)
        )
        df[f"actual_max_w{window}"] = (
            actual_series.rolling(window=window, min_periods=window)
            .max()
            .shift(truth_lag_days)
            .to_numpy(dtype=float)
        )

    clim_mean, clim_std, clim_p10, clim_p90 = _climatology_by_doy(
        dates, actual, truth_lag_days
    )
    df["clim_mean_doy"] = clim_mean
    df["clim_std_doy"] = clim_std
    df["clim_p10_doy"] = clim_p10
    df["clim_p90_doy"] = clim_p90

    for model in BASE_MODELS:
        df[f"anom_{model}"] = df[BASE_MODEL_COLS[model]] - df["clim_mean_doy"]
    df["anom_ensmean"] = df[ENSEMBLE_MEAN_COL] - df["clim_mean_doy"]
    df["anom_actual_lag2"] = df["actual_lag2"] - df["clim_mean_doy"]

    df["delta_lag2_lag3"] = df["actual_lag2"] - df["actual_lag3"]
    df["delta_lag2_lag7"] = df["actual_lag2"] - df["actual_lag7"]
    df["trend_short"] = df[
        [f"actual_lag{lag}" for lag in range(2, 6)]
    ].mean(axis=1) - df[[f"actual_lag{lag}" for lag in range(6, 10)]].mean(axis=1)
    df["abs_trend_short"] = df["trend_short"].abs()

    # Analog features
    z_cols = _analog_components()
    z_matrix = df[z_cols].to_numpy(dtype=float)
    train_mask = (df["target_date_local"] >= train_start) & (df["target_date_local"] <= train_end)
    z_train = z_matrix[train_mask]
    z_means = np.nanmean(z_train, axis=0)
    z_stds = np.nanstd(z_train, axis=0)
    z_stds = np.where((z_stds == 0.0) | np.isnan(z_stds), 1.0, z_stds)
    z_scaled = (z_matrix - z_means) / z_stds
    scaling = AnalogScaling(
        means={col: float(val) for col, val in zip(z_cols, z_means)},
        stds={col: float(val) for col, val in zip(z_cols, z_stds)},
    )

    _add_analog_features(
        df=df,
        dates=dates,
        z_scaled=z_scaled,
        z_cols=z_cols,
        truth_lag_days=truth_lag_days,
    )

    return df, scaling


def _source_col(source: str) -> str:
    if source == ENSEMBLE_MEAN_COL:
        return ENSEMBLE_MEAN_COL
    return BASE_MODEL_COLS[source]


def _model_pairs(models: Iterable[str]) -> list[tuple[str, str]]:
    items = list(models)
    pairs: list[tuple[str, str]] = []
    for i, left in enumerate(items):
        for right in items[i + 1 :]:
            pairs.append((left, right))
    return pairs


def _ensemble_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    n_rows = values.shape[0]
    stats = {
        "ens_raw_mean": np.full(n_rows, np.nan),
        "ens_raw_median": np.full(n_rows, np.nan),
        "ens_raw_min": np.full(n_rows, np.nan),
        "ens_raw_max": np.full(n_rows, np.nan),
        "ens_raw_range": np.full(n_rows, np.nan),
        "ens_raw_std": np.full(n_rows, np.nan),
        "ens_raw_mad": np.full(n_rows, np.nan),
        "ens_raw_iqr": np.full(n_rows, np.nan),
        "ens_raw_p10": np.full(n_rows, np.nan),
        "ens_raw_p90": np.full(n_rows, np.nan),
        "ens_raw_skew": np.full(n_rows, np.nan),
        "ens_raw_kurt": np.full(n_rows, np.nan),
    }
    for idx in range(n_rows):
        row = values[idx, :]
        row = row[~np.isnan(row)]
        if row.size == 0:
            continue
        mean = float(np.mean(row))
        median = float(np.median(row))
        min_val = float(np.min(row))
        max_val = float(np.max(row))
        stats["ens_raw_mean"][idx] = mean
        stats["ens_raw_median"][idx] = median
        stats["ens_raw_min"][idx] = min_val
        stats["ens_raw_max"][idx] = max_val
        stats["ens_raw_range"][idx] = max_val - min_val
        if row.size >= 2:
            stats["ens_raw_std"][idx] = float(np.std(row, ddof=1))
        stats["ens_raw_mad"][idx] = float(np.median(np.abs(row - median)))
        stats["ens_raw_iqr"][idx] = float(np.quantile(row, 0.75) - np.quantile(row, 0.25))
        stats["ens_raw_p10"][idx] = float(np.quantile(row, 0.10))
        stats["ens_raw_p90"][idx] = float(np.quantile(row, 0.90))
        if row.size >= 3 and stats["ens_raw_std"][idx] and stats["ens_raw_std"][idx] > 0:
            std = stats["ens_raw_std"][idx]
            stats["ens_raw_skew"][idx] = float(np.mean(((row - mean) / std) ** 3))
        if row.size >= 4 and stats["ens_raw_std"][idx] and stats["ens_raw_std"][idx] > 0:
            std = stats["ens_raw_std"][idx]
            stats["ens_raw_kurt"][idx] = float(np.mean(((row - mean) / std) ** 4) - 3.0)
    return stats


def _mad(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan")
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _iqr(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))


def _tail2(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(np.abs(values) > 2.0))


def _kalman_params(
    residuals: dict[str, pd.Series],
    train_start: date,
    train_end: date,
) -> dict[str, dict[str, float]]:
    params: dict[str, dict[str, float]] = {}
    for source, series in residuals.items():
        series = series.dropna()
        if series.empty:
            continue
        mask = (series.index.date >= train_start) & (series.index.date <= train_end)
        train_vals = series.loc[mask].to_numpy(dtype=float)
        if train_vals.size < 20:
            continue
        variance = float(np.var(train_vals, ddof=0))
        variance = max(variance, 0.1)
        params[source] = {
            "r": variance,
            "q_fast": variance * 0.5,
            "q_slow": variance * 0.05,
        }
    return params


def _kalman_filter(series: pd.Series, q: float, r: float) -> pd.Series:
    values = series.to_numpy(dtype=float)
    bias = np.full_like(values, np.nan, dtype=float)
    b = 0.0
    p = 1.0
    for idx, value in enumerate(values):
        p = p + q
        if not np.isnan(value):
            k = p / (p + r)
            b = b + k * (value - b)
            p = (1.0 - k) * p
        bias[idx] = b
    return pd.Series(bias, index=series.index)


def _climatology_by_doy(
    dates: pd.Series,
    actual: np.ndarray,
    truth_lag_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(dates)
    clim_mean = np.full(n_rows, np.nan)
    clim_std = np.full(n_rows, np.nan)
    clim_p10 = np.full(n_rows, np.nan)
    clim_p90 = np.full(n_rows, np.nan)

    history_lists: dict[int, list[float]] = {doy: [] for doy in range(1, 367)}
    idx = 0
    date_values = dates.dt.date.to_numpy()

    for i in range(n_rows):
        history_end = date_values[i] - timedelta(days=truth_lag_days)
        while idx < n_rows and date_values[idx] <= history_end:
            val = actual[idx]
            if not np.isnan(val):
                doy = dates.iloc[idx].dayofyear
                history_lists[doy].append(float(val))
            idx += 1
        doy = dates.iloc[i].dayofyear
        values = history_lists.get(doy, [])
        if not values:
            continue
        arr = np.array(values, dtype=float)
        clim_mean[i] = float(np.mean(arr))
        clim_std[i] = float(np.std(arr, ddof=0))
        clim_p10[i] = float(np.quantile(arr, 0.10))
        clim_p90[i] = float(np.quantile(arr, 0.90))

    return clim_mean, clim_std, clim_p10, clim_p90


def _analog_components() -> list[str]:
    return [
        "nbm_tmax_f",
        "hrrr_tmax_f",
        "rap_tmax_f",
        "gefsatmosmean_tmax_f",
        "gfs_n_x_max",
        "nam_n_x_max",
        "gefsatmos_tmp_spread_f",
        "sin_doy",
        "cos_doy",
        "anom_ensmean",
    ]


def _add_analog_features(
    *,
    df: pd.DataFrame,
    dates: pd.Series,
    z_scaled: np.ndarray,
    z_cols: list[str],
    truth_lag_days: int,
) -> None:
    n_rows = len(df)
    z_valid = np.isfinite(z_scaled).all(axis=1)
    date_values = dates.dt.date.to_numpy()
    actual = df["actual_tmax_f"].to_numpy(dtype=float)

    for i in range(n_rows):
        current_date = date_values[i]
        history_end = current_date - timedelta(days=truth_lag_days)
        candidate_mask = (date_values <= history_end) & z_valid
        if not z_valid[i]:
            _fill_analog_nan(df, i)
            continue
        candidates = np.where(candidate_mask)[0]
        if candidates.size == 0:
            _fill_analog_nan(df, i)
            continue

        z_current = z_scaled[i]
        z_past = z_scaled[candidates]
        distances = np.linalg.norm(z_past - z_current, axis=1)
        order = np.argsort(distances)
        candidates = candidates[order]
        distances = distances[order]

        for k in ANALOG_K:
            if candidates.size < k:
                _set_analog_stats(df, i, k, None, None)
                continue
            idx = candidates[:k]
            dist = distances[:k]
            ys = actual[idx]
            ys = ys[~np.isnan(ys)]
            if ys.size == 0:
                _set_analog_stats(df, i, k, None, None)
                continue
            _set_analog_stats(df, i, k, ys, dist)

        # Analog residual corrections (K=25)
        if candidates.size >= 25:
            idx = candidates[:25]
            for model in BASE_MODELS:
                col = BASE_MODEL_COLS[model]
                resid = actual[idx] - df[col].to_numpy(dtype=float)[idx]
                resid = resid[~np.isnan(resid)]
                df.at[i, f"anen_resid_mean_{model}_k25"] = (
                    float(np.mean(resid)) if resid.size else np.nan
                )
            ens_resid = actual[idx] - df[ENSEMBLE_MEAN_COL].to_numpy(dtype=float)[idx]
            ens_resid = ens_resid[~np.isnan(ens_resid)]
            df.at[i, "anen_resid_mean_ens_k25"] = (
                float(np.mean(ens_resid)) if ens_resid.size else np.nan
            )
            day_diffs = np.array(
                [abs((current_date - date_values[j]).days) for j in idx], dtype=float
            )
            df.at[i, "anen_daydiff_mean_k25"] = float(np.mean(day_diffs))
            df.at[i, "anen_daydiff_min_k25"] = float(np.min(day_diffs))

            dist = distances[:25]
            ys = actual[idx]
            weights = 1.0 / (dist + 1e-6)
            weights = np.where(np.isnan(ys), 0.0, weights)
            ys = np.where(np.isnan(ys), np.nan, ys)
            if np.sum(weights) > 0:
                wmean = np.sum(weights * np.nan_to_num(ys)) / np.sum(weights)
                df.at[i, "anen_wmean_k25"] = float(wmean)
                df.at[i, "anen_wmedian_k25"] = float(_weighted_quantile(ys, weights, 0.5))
                df.at[i, "anen_wstd_k25"] = float(
                    np.sqrt(np.sum(weights * (np.nan_to_num(ys) - wmean) ** 2) / np.sum(weights))
                )
                q25 = _weighted_quantile(ys, weights, 0.25)
                q75 = _weighted_quantile(ys, weights, 0.75)
                df.at[i, "anen_wiqr_k25"] = float(q75 - q25)
            else:
                df.at[i, "anen_wmean_k25"] = np.nan
                df.at[i, "anen_wmedian_k25"] = np.nan
                df.at[i, "anen_wstd_k25"] = np.nan
                df.at[i, "anen_wiqr_k25"] = np.nan
        else:
            for model in BASE_MODELS:
                df.at[i, f"anen_resid_mean_{model}_k25"] = np.nan
            df.at[i, "anen_resid_mean_ens_k25"] = np.nan
            df.at[i, "anen_daydiff_mean_k25"] = np.nan
            df.at[i, "anen_daydiff_min_k25"] = np.nan
            df.at[i, "anen_wmean_k25"] = np.nan
            df.at[i, "anen_wmedian_k25"] = np.nan
            df.at[i, "anen_wstd_k25"] = np.nan
            df.at[i, "anen_wiqr_k25"] = np.nan


def _set_analog_stats(
    df: pd.DataFrame,
    row_idx: int,
    k: int,
    ys: np.ndarray | None,
    dist: np.ndarray | None,
) -> None:
    if ys is None or dist is None or ys.size == 0:
        for name in [
            "mean",
            "median",
            "p10",
            "p90",
            "iqr",
            "std",
            "min",
            "max",
        ]:
            df.at[row_idx, f"anen_{name}_k{k}"] = np.nan
        df.at[row_idx, f"anen_dist_mean_k{k}"] = np.nan
        df.at[row_idx, f"anen_dist_min_k{k}"] = np.nan
        df.at[row_idx, f"anen_dist_std_k{k}"] = np.nan
        return
    ys = ys.astype(float)
    df.at[row_idx, f"anen_mean_k{k}"] = float(np.mean(ys))
    df.at[row_idx, f"anen_median_k{k}"] = float(np.median(ys))
    df.at[row_idx, f"anen_p10_k{k}"] = float(np.quantile(ys, 0.10))
    df.at[row_idx, f"anen_p90_k{k}"] = float(np.quantile(ys, 0.90))
    df.at[row_idx, f"anen_iqr_k{k}"] = float(np.quantile(ys, 0.75) - np.quantile(ys, 0.25))
    df.at[row_idx, f"anen_std_k{k}"] = float(np.std(ys, ddof=0))
    df.at[row_idx, f"anen_min_k{k}"] = float(np.min(ys))
    df.at[row_idx, f"anen_max_k{k}"] = float(np.max(ys))
    df.at[row_idx, f"anen_dist_mean_k{k}"] = float(np.mean(dist))
    df.at[row_idx, f"anen_dist_min_k{k}"] = float(np.min(dist))
    df.at[row_idx, f"anen_dist_std_k{k}"] = float(np.std(dist, ddof=0))


def _fill_analog_nan(df: pd.DataFrame, row_idx: int) -> None:
    for k in ANALOG_K:
        _set_analog_stats(df, row_idx, k, None, None)
    for model in BASE_MODELS:
        df.at[row_idx, f"anen_resid_mean_{model}_k25"] = np.nan
    df.at[row_idx, "anen_resid_mean_ens_k25"] = np.nan
    df.at[row_idx, "anen_daydiff_mean_k25"] = np.nan
    df.at[row_idx, "anen_daydiff_min_k25"] = np.nan
    df.at[row_idx, "anen_wmean_k25"] = np.nan
    df.at[row_idx, "anen_wmedian_k25"] = np.nan
    df.at[row_idx, "anen_wstd_k25"] = np.nan
    df.at[row_idx, "anen_wiqr_k25"] = np.nan


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = values.astype(float)
    weights = weights.astype(float)
    mask = np.isfinite(values) & np.isfinite(weights)
    if not np.any(mask):
        return float("nan")
    values = values[mask]
    weights = weights[mask]
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumulative = np.cumsum(weights)
    if cumulative[-1] == 0:
        return float("nan")
    cutoff = quantile * cumulative[-1]
    return float(values[np.searchsorted(cumulative, cutoff)])
