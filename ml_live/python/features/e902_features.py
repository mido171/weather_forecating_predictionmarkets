from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ml_live.features.e902_config import (
    BASE_MODELS,
    BASE_MODEL_COLS,
    BIAS_SOURCES,
    ENSEMBLE_MEAN_COL,
    HUBER_K,
    SPREAD_COL,
    TRUTH_LAG_DAYS,
    W_AR,
    W_BIAS,
    W_BMA,
    W_COND,
    W_RESID,
    W_RIDGE,
    W_SKILL,
    AnalogScaling,
    E902Metadata,
    RegimeThresholds,
)
from ml_live.features.e902_experts import (
    fit_ar1,
    fit_linear_calibration,
    fit_ridge,
    mixture_mean,
    mixture_quantile,
    mixture_variance,
    weights_eff_n,
    weights_entropy,
)

ANALOG_K = [25]


def generate_feature_list() -> list[str]:
    features: list[str] = []

    # Family A1: raw inputs
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

    # Family A2: ensemble summary stats
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

    # Family A4: calendar / seasonality
    features.extend(
        [
            "month",
            "day_of_year",
            "is_weekend",
            "sin_doy",
            "cos_doy",
            "sin2_doy",
            "cos2_doy",
            "sin3_doy",
            "cos3_doy",
        ]
    )
    for month in range(1, 13):
        features.append(f"month_oh_{month:02d}")
    features.extend(["day_of_month", "week_of_year", "sin_month", "cos_month"])

    # Family A5: climatology + anomaly
    features.extend(
        [
            "clim_mean_doy",
            "clim_std_doy",
            "clim_p10_doy",
            "clim_p90_doy",
            "ens_anom_clim",
        ]
    )

    # Family A6: regime one-hots
    features.extend(
        [
            "spread_bin_low",
            "spread_bin_mid",
            "spread_bin_high",
            "disagree_bin_low",
            "disagree_bin_mid",
            "disagree_bin_high",
        ]
    )

    # Family A7: deviations vs ensemble mean
    for model in BASE_MODELS:
        features.append(f"dev_{model}")
        features.append(f"absdev_{model}")

    # Family A8: ranks
    for model in BASE_MODELS:
        features.append(f"rank_{model}")

    # Family A9: max/min indicators
    for model in BASE_MODELS:
        features.append(f"is_max_{model}")
    for model in BASE_MODELS:
        features.append(f"is_min_{model}")

    # Family A10: selected pairwise abs differences
    features.extend(
        [
            "absdiff_nbm_hrrr",
            "absdiff_nbm_rap",
            "absdiff_hrrr_rap",
            "absdiff_gefsmean_nbm",
            "absdiff_gfsMOS_namMOS",
            "absdiff_gefsmean_ensmean",
        ]
    )

    # Family B: robust bias-correction cube
    for source in BIAS_SOURCES:
        for window in W_BIAS:
            for estimator in ["mean", "median", "huber"]:
                features.append(f"bias_{estimator}_{source}_w{window}")
                features.append(f"corr_{estimator}_{source}_w{window}")

    # Family C: Kalman bias states
    for source in BIAS_SOURCES:
        for variant in ["fast", "slow"]:
            features.append(f"bias_kf_{variant}_{source}")
            features.append(f"corr_kf_{variant}_{source}")
            features.append(f"kf_var_{variant}_{source}")

    # Family D: forecast evolution features
    for series in BASE_MODELS + ["ensmean", "spread"]:
        for lag in [1, 2, 3]:
            features.append(f"{series}_lag{lag}")
        for lag in [1, 2, 3]:
            features.append(f"{series}_chg{lag}")

    for series in BASE_MODELS + ["ensmean"]:
        features.append(f"{series}_accel")
        features.append(f"jump_vs_actual_lag2_{series}")
        features.append(f"jump_vs_clim_{series}")

    for model in BASE_MODELS:
        features.append(f"dev_{model}_lag1")
        features.append(f"dev_{model}_lag2")
        features.append(f"dev_{model}_lag3")

    features.extend(
        [
            "max_forecast_minus_actual_lag2",
            "min_forecast_minus_actual_lag2",
        ]
    )

    features.extend(
        [
            "ens_std_lag1",
            "ens_std_lag2",
            "ens_std_lag3",
            "ens_std_chg1",
            "ens_std_chg2",
            "ens_std_chg3",
            "ens_std_accel",
        ]
    )

    # Family E: residual distribution & tail risk
    for source in BIAS_SOURCES:
        for window in W_RESID:
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

    # Family F: regime-conditional skill
    for regime in ["spread", "disagree", "anom"]:
        for window in W_COND:
            for model in BASE_MODELS:
                features.append(f"cond_bias_{model}_{regime}_w{window}")
                features.append(f"cond_mae_{model}_{regime}_w{window}")
                features.append(f"cond_count_{model}_{regime}_w{window}")
        for window in W_COND:
            features.append(f"cond_entropy_{regime}_w{window}")
            features.append(f"cond_effN_{regime}_w{window}")

    # Family G: expert predictors
    for window in W_SKILL:
        features.append(f"ens_wt_mae_w{window}")
        features.append(f"ens_wt_rmse_w{window}")

    for window in W_BMA:
        features.extend(
            [
                f"bma_mean_w{window}",
                f"bma_median_w{window}",
                f"bma_p10_w{window}",
                f"bma_p90_w{window}",
                f"bma_effN_w{window}",
            ]
        )

    for window in W_RIDGE:
        features.append(f"ridge_pred_w{window}")
        features.append(f"ridge_intercept_w{window}")

    for window in W_AR:
        features.append(f"ar1_phi_ens_w{window}")
        features.append(f"ar1_predresid2_ens_w{window}")
        features.append(f"ar1_mu_ens_w{window}")

    features.extend(
        [
            "anen_mean_k25",
            "anen_median_k25",
            "anen_p10_k25",
            "anen_p90_k25",
            "anen_std_k25",
            "anen_iqr_k25",
        ]
    )

    for model in BASE_MODELS:
        features.append(f"bma_w_{model}_w180")

    features.extend(
        [
            "bma_entropy_w180",
            "bma_var_w180",
            "ridge_l1norm_w365",
            "ridge_l2norm_w365",
        ]
    )

    if len(features) != 902:
        raise ValueError(f"E902 expected 902 features, got {len(features)}")
    return features


def build_e902_features(
    df: pd.DataFrame,
    *,
    train_start: date,
    train_end: date,
    truth_lag_days: int = TRUTH_LAG_DAYS,
) -> tuple[pd.DataFrame, E902Metadata]:
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

    # Missingness flags
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

    # Ensemble stats
    forecast_matrix = df[[BASE_MODEL_COLS[m] for m in BASE_MODELS]].to_numpy(dtype=float)
    stats = _ensemble_stats(forecast_matrix)
    for key, values in stats.items():
        df[key] = values

    # Climatology
    actual = df["actual_tmax_f"].to_numpy(dtype=float)
    clim_mean, clim_std, clim_p10, clim_p90 = _climatology_by_doy(
        dates, actual, truth_lag_days
    )
    df["clim_mean_doy"] = clim_mean
    df["clim_std_doy"] = clim_std
    df["clim_p10_doy"] = clim_p10
    df["clim_p90_doy"] = clim_p90
    df["ens_anom_clim"] = df["ens_raw_mean"] - df["clim_mean_doy"]

    # Regime thresholds based on training range
    train_mask = (df["target_date_local"] >= train_start) & (df["target_date_local"] <= train_end)
    thresholds = _compute_thresholds(
        df.loc[train_mask, SPREAD_COL].to_numpy(dtype=float),
        df.loc[train_mask, "ens_raw_std"].to_numpy(dtype=float),
        df.loc[train_mask, "ens_anom_clim"].to_numpy(dtype=float),
    )

    spread_bin = _assign_bin(df[SPREAD_COL].to_numpy(dtype=float), thresholds.spread_low, thresholds.spread_high)
    disagree_bin = _assign_bin(df["ens_raw_std"].to_numpy(dtype=float), thresholds.disagree_low, thresholds.disagree_high)
    anom_bin = _assign_bin(df["ens_anom_clim"].to_numpy(dtype=float), thresholds.anom_low, thresholds.anom_high)

    _add_bin_features(df, "spread_bin", spread_bin)
    _add_bin_features(df, "disagree_bin", disagree_bin)

    # Deviations and abs deviations
    ens_mean = df["ens_raw_mean"].to_numpy(dtype=float)
    for model in BASE_MODELS:
        vals = df[BASE_MODEL_COLS[model]].to_numpy(dtype=float)
        diff = vals - ens_mean
        df[f"dev_{model}"] = diff
        df[f"absdev_{model}"] = np.abs(diff)

    # Ranks
    ranks = pd.DataFrame(forecast_matrix).rank(axis=1, method="average", na_option="keep")
    for idx, model in enumerate(BASE_MODELS):
        df[f"rank_{model}"] = ranks.iloc[:, idx].to_numpy(dtype=float)

    # Max/min indicators
    max_vals = np.nanmax(forecast_matrix, axis=1)
    min_vals = np.nanmin(forecast_matrix, axis=1)
    for idx, model in enumerate(BASE_MODELS):
        model_vals = forecast_matrix[:, idx]
        df[f"is_max_{model}"] = (model_vals == max_vals).astype(int)
        df[f"is_min_{model}"] = (model_vals == min_vals).astype(int)

    # Selected pairwise abs differences
    df["absdiff_nbm_hrrr"] = np.abs(df[BASE_MODEL_COLS["nbm"]] - df[BASE_MODEL_COLS["hrrr"]])
    df["absdiff_nbm_rap"] = np.abs(df[BASE_MODEL_COLS["nbm"]] - df[BASE_MODEL_COLS["rap"]])
    df["absdiff_hrrr_rap"] = np.abs(df[BASE_MODEL_COLS["hrrr"]] - df[BASE_MODEL_COLS["rap"]])
    df["absdiff_gefsmean_nbm"] = np.abs(df[BASE_MODEL_COLS["gefsmean"]] - df[BASE_MODEL_COLS["nbm"]])
    df["absdiff_gfsMOS_namMOS"] = np.abs(df[BASE_MODEL_COLS["gfsMOS"]] - df[BASE_MODEL_COLS["namMOS"]])
    df["absdiff_gefsmean_ensmean"] = np.abs(df[BASE_MODEL_COLS["gefsmean"]] - df["ens_raw_mean"])

    # Residuals
    residuals: dict[str, pd.Series] = {}
    for model, col in BASE_MODEL_COLS.items():
        residuals[model] = pd.Series(actual - df[col].to_numpy(dtype=float), index=dates)
    residuals["ensmean"] = pd.Series(actual - df["ens_raw_mean"].to_numpy(dtype=float), index=dates)

    # Family B: bias cube
    for source in BIAS_SOURCES:
        series = residuals[source]
        for window in W_BIAS:
            bias_mean = (
                series.rolling(window=window, min_periods=window).mean().shift(truth_lag_days)
            )
            bias_median = (
                series.rolling(window=window, min_periods=window).median().shift(truth_lag_days)
            )
            bias_huber = (
                series.rolling(window=window, min_periods=window)
                .apply(lambda x: _huber_mean(x, HUBER_K), raw=True)
                .shift(truth_lag_days)
            )
            df[f"bias_mean_{source}_w{window}"] = bias_mean.to_numpy(dtype=float)
            df[f"bias_median_{source}_w{window}"] = bias_median.to_numpy(dtype=float)
            df[f"bias_huber_{source}_w{window}"] = bias_huber.to_numpy(dtype=float)

            base_col = _source_col(source, df)
            df[f"corr_mean_{source}_w{window}"] = df[base_col] + df[f"bias_mean_{source}_w{window}"]
            df[f"corr_median_{source}_w{window}"] = df[base_col] + df[f"bias_median_{source}_w{window}"]
            df[f"corr_huber_{source}_w{window}"] = df[base_col] + df[f"bias_huber_{source}_w{window}"]

    # Family C: Kalman bias states
    kalman_params = _kalman_params(residuals, train_start, train_end)
    for source in BIAS_SOURCES:
        series = residuals[source]
        params = kalman_params.get(source)
        if params is None:
            df[f"bias_kf_fast_{source}"] = np.nan
            df[f"corr_kf_fast_{source}"] = np.nan
            df[f"kf_var_fast_{source}"] = np.nan
            df[f"bias_kf_slow_{source}"] = np.nan
            df[f"corr_kf_slow_{source}"] = np.nan
            df[f"kf_var_slow_{source}"] = np.nan
            continue

        bias_fast, var_fast = _kalman_filter(series, params["q_fast"], params["r"])
        bias_slow, var_slow = _kalman_filter(series, params["q_slow"], params["r"])

        df[f"bias_kf_fast_{source}"] = bias_fast.shift(truth_lag_days).to_numpy(dtype=float)
        df[f"kf_var_fast_{source}"] = var_fast.shift(truth_lag_days).to_numpy(dtype=float)
        df[f"bias_kf_slow_{source}"] = bias_slow.shift(truth_lag_days).to_numpy(dtype=float)
        df[f"kf_var_slow_{source}"] = var_slow.shift(truth_lag_days).to_numpy(dtype=float)

        base_col = _source_col(source, df)
        df[f"corr_kf_fast_{source}"] = df[base_col] + df[f"bias_kf_fast_{source}"]
        df[f"corr_kf_slow_{source}"] = df[base_col] + df[f"bias_kf_slow_{source}"]

    # Family D: forecast evolution
    series_map = {
        "nbm": df[BASE_MODEL_COLS["nbm"]].to_numpy(dtype=float),
        "hrrr": df[BASE_MODEL_COLS["hrrr"]].to_numpy(dtype=float),
        "rap": df[BASE_MODEL_COLS["rap"]].to_numpy(dtype=float),
        "gefsmean": df[BASE_MODEL_COLS["gefsmean"]].to_numpy(dtype=float),
        "gfsMOS": df[BASE_MODEL_COLS["gfsMOS"]].to_numpy(dtype=float),
        "namMOS": df[BASE_MODEL_COLS["namMOS"]].to_numpy(dtype=float),
        "ensmean": df["ens_raw_mean"].to_numpy(dtype=float),
        "spread": df[SPREAD_COL].to_numpy(dtype=float),
    }

    actual_lag2 = _lag_by_days(pd.Series(actual, index=dates), 2).to_numpy(dtype=float)

    for name, values in series_map.items():
        lag1 = _shift(values, 1)
        lag2 = _shift(values, 2)
        lag3 = _shift(values, 3)
        df[f"{name}_lag1"] = lag1
        df[f"{name}_lag2"] = lag2
        df[f"{name}_lag3"] = lag3
        df[f"{name}_chg1"] = values - lag1
        df[f"{name}_chg2"] = lag1 - lag2
        df[f"{name}_chg3"] = lag2 - lag3

    for name in BASE_MODELS + ["ensmean"]:
        values = series_map[name]
        lag1 = df[f"{name}_lag1"].to_numpy(dtype=float)
        lag2 = df[f"{name}_lag2"].to_numpy(dtype=float)
        df[f"{name}_accel"] = (values - lag1) - (lag1 - lag2)
        df[f"jump_vs_actual_lag2_{name}"] = values - actual_lag2
        df[f"jump_vs_clim_{name}"] = values - df["clim_mean_doy"].to_numpy(dtype=float)

    for model in BASE_MODELS:
        df[f"dev_{model}_lag1"] = df[f"{model}_lag1"] - df["ensmean_lag1"]
        df[f"dev_{model}_lag2"] = df[f"{model}_lag2"] - df["ensmean_lag2"]
        df[f"dev_{model}_lag3"] = df[f"{model}_lag3"] - df["ensmean_lag3"]

    df["max_forecast_minus_actual_lag2"] = df["ens_raw_max"] - actual_lag2
    df["min_forecast_minus_actual_lag2"] = df["ens_raw_min"] - actual_lag2

    ens_std = df["ens_raw_std"].to_numpy(dtype=float)
    ens_std_lag1 = _shift(ens_std, 1)
    ens_std_lag2 = _shift(ens_std, 2)
    ens_std_lag3 = _shift(ens_std, 3)
    df["ens_std_lag1"] = ens_std_lag1
    df["ens_std_lag2"] = ens_std_lag2
    df["ens_std_lag3"] = ens_std_lag3
    df["ens_std_chg1"] = ens_std - ens_std_lag1
    df["ens_std_chg2"] = ens_std_lag1 - ens_std_lag2
    df["ens_std_chg3"] = ens_std_lag2 - ens_std_lag3
    df["ens_std_accel"] = (ens_std - ens_std_lag1) - (ens_std_lag1 - ens_std_lag2)

    # Family E: residual distribution stats
    for source in BIAS_SOURCES:
        series = residuals[source].shift(truth_lag_days)
        for window in W_RESID:
            roll = series.rolling(window=window, min_periods=window)
            df[f"resid_std_{source}_w{window}"] = roll.std(ddof=0).to_numpy(dtype=float)
            df[f"resid_mad_{source}_w{window}"] = roll.apply(_mad, raw=True).to_numpy(dtype=float)
            df[f"resid_iqr_{source}_w{window}"] = roll.apply(_iqr, raw=True).to_numpy(dtype=float)
            df[f"resid_q10_{source}_w{window}"] = roll.quantile(0.10).to_numpy(dtype=float)
            df[f"resid_q90_{source}_w{window}"] = roll.quantile(0.90).to_numpy(dtype=float)
            df[f"resid_tail2_{source}_w{window}"] = roll.apply(_tail2, raw=True).to_numpy(dtype=float)

    # Family F + G: row-wise computations
    metadata = E902Metadata(
        thresholds=thresholds,
        analog_scaling=_analog_scaling(df, train_start, train_end),
    )
    _add_rowwise_features(
        df=df,
        dates=dates,
        residuals=residuals,
        actual=actual,
        spread_bin=spread_bin,
        disagree_bin=disagree_bin,
        anom_bin=anom_bin,
        metadata=metadata,
        truth_lag_days=truth_lag_days,
    )

    return df, metadata


def _ensemble_stats(matrix: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.nanmean(matrix, axis=1)
    median = np.nanmedian(matrix, axis=1)
    min_v = np.nanmin(matrix, axis=1)
    max_v = np.nanmax(matrix, axis=1)
    range_v = max_v - min_v
    std = np.nanstd(matrix, axis=1, ddof=1)
    mad = np.nanmedian(np.abs(matrix - median[:, None]), axis=1)
    q75 = np.nanquantile(matrix, 0.75, axis=1)
    q25 = np.nanquantile(matrix, 0.25, axis=1)
    iqr = q75 - q25
    p10 = np.nanquantile(matrix, 0.10, axis=1)
    p90 = np.nanquantile(matrix, 0.90, axis=1)
    return {
        "ens_raw_mean": mean,
        "ens_raw_median": median,
        "ens_raw_min": min_v,
        "ens_raw_max": max_v,
        "ens_raw_range": range_v,
        "ens_raw_std": std,
        "ens_raw_mad": mad,
        "ens_raw_iqr": iqr,
        "ens_raw_p10": p10,
        "ens_raw_p90": p90,
    }


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


def _compute_thresholds(
    spread: np.ndarray,
    disagree: np.ndarray,
    anom: np.ndarray,
) -> RegimeThresholds:
    spread_low, spread_high = np.nanquantile(spread, [1 / 3, 2 / 3])
    disagree_low, disagree_high = np.nanquantile(disagree, [1 / 3, 2 / 3])
    anom_low, anom_high = np.nanquantile(anom, [1 / 3, 2 / 3])
    return RegimeThresholds(
        spread_low=float(spread_low),
        spread_high=float(spread_high),
        disagree_low=float(disagree_low),
        disagree_high=float(disagree_high),
        anom_low=float(anom_low),
        anom_high=float(anom_high),
    )


def _assign_bin(values: np.ndarray, low: float, high: float) -> np.ndarray:
    bins = np.full(values.shape[0], -1, dtype=int)
    bins[values <= low] = 0
    bins[(values > low) & (values <= high)] = 1
    bins[values > high] = 2
    return bins


def _add_bin_features(df: pd.DataFrame, prefix: str, bins: np.ndarray) -> None:
    df[f"{prefix}_low"] = (bins == 0).astype(int)
    df[f"{prefix}_mid"] = (bins == 1).astype(int)
    df[f"{prefix}_high"] = (bins == 2).astype(int)


def _huber_mean(values: np.ndarray, k: float) -> float:
    values = values.astype(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad <= 0.0:
        return float(np.mean(values))
    lo = med - k * mad
    hi = med + k * mad
    capped = np.clip(values, lo, hi)
    return float(np.mean(capped))


def _source_col(source: str, df: pd.DataFrame) -> str:
    if source == "ensmean":
        return "ens_raw_mean"
    return BASE_MODEL_COLS[source]


def _shift(values: np.ndarray, periods: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    if periods <= 0:
        return values.copy()
    out[periods:] = values[:-periods]
    return out


def _lag_by_days(series: pd.Series, days: int) -> pd.Series:
    lag_dates = series.index - pd.Timedelta(days=days)
    return series.reindex(lag_dates)


def _mad(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def _iqr(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))


def _tail2(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
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


def _kalman_filter(series: pd.Series, q: float, r: float) -> tuple[pd.Series, pd.Series]:
    values = series.to_numpy(dtype=float)
    bias = np.full_like(values, np.nan, dtype=float)
    var = np.full_like(values, np.nan, dtype=float)
    b = 0.0
    p = 1.0
    for idx, value in enumerate(values):
        p = p + q
        if not np.isnan(value):
            k = p / (p + r)
            b = b + k * (value - b)
            p = (1.0 - k) * p
        bias[idx] = b
        var[idx] = p
    return pd.Series(bias, index=series.index), pd.Series(var, index=series.index)


def _analog_scaling(df: pd.DataFrame, train_start: date, train_end: date) -> AnalogScaling:
    z_cols = _analog_components()
    train_mask = (df["target_date_local"] >= train_start) & (df["target_date_local"] <= train_end)
    z_matrix = df.loc[train_mask, z_cols].to_numpy(dtype=float)
    means = np.nanmean(z_matrix, axis=0)
    stds = np.nanstd(z_matrix, axis=0, ddof=0)
    stds = np.where(stds <= 1e-6, 1.0, stds)
    return AnalogScaling(
        means={col: float(val) for col, val in zip(z_cols, means)},
        stds={col: float(val) for col, val in zip(z_cols, stds)},
    )


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
    ]


def _add_rowwise_features(
    *,
    df: pd.DataFrame,
    dates: pd.Series,
    residuals: dict[str, pd.Series],
    actual: np.ndarray,
    spread_bin: np.ndarray,
    disagree_bin: np.ndarray,
    anom_bin: np.ndarray,
    metadata: E902Metadata,
    truth_lag_days: int,
) -> None:
    date_values = dates.dt.date.to_numpy()
    doy_values = dates.dt.dayofyear.to_numpy()
    n_rows = len(df)

    z_cols = _analog_components()
    z_matrix = df[z_cols].to_numpy(dtype=float)
    means = np.array([metadata.analog_scaling.means[c] for c in z_cols], dtype=float)
    stds = np.array([metadata.analog_scaling.stds[c] for c in z_cols], dtype=float)
    z_scaled = (z_matrix - means) / stds
    z_valid = np.isfinite(z_scaled).all(axis=1)

    forecast_matrix = df[[BASE_MODEL_COLS[m] for m in BASE_MODELS]].to_numpy(dtype=float)
    ens_mean = df["ens_raw_mean"].to_numpy(dtype=float)

    resid_by_model = {m: residuals[m].to_numpy(dtype=float) for m in BASE_MODELS}

    for i in range(n_rows):
        current_date = date_values[i]
        history_end = current_date - timedelta(days=truth_lag_days)
        eligible = np.where(date_values <= history_end)[0]
        if eligible.size == 0:
            _fill_rowwise_nan(df, i)
            continue

        # Family F: regime-conditional skill
        for regime_name, regime_bins in [
            ("spread", spread_bin),
            ("disagree", disagree_bin),
            ("anom", anom_bin),
        ]:
            current_label = regime_bins[i]
            for window in W_COND:
                if eligible.size < window:
                    _fill_conditional_nan(df, i, regime_name, window)
                    continue
                window_idx = eligible[-window:]
                regime_idx = window_idx[regime_bins[window_idx] == current_label]
                use_idx = regime_idx
                cond_count = int(regime_idx.size)
                if cond_count < 15:
                    use_idx = window_idx
                for model in BASE_MODELS:
                    resid = resid_by_model[model][use_idx]
                    resid = resid[np.isfinite(resid)]
                    if resid.size == 0:
                        bias_val = np.nan
                        mae_val = np.nan
                    else:
                        bias_val = float(np.mean(resid))
                        mae_val = float(np.mean(np.abs(resid)))
                    df.at[i, f"cond_bias_{model}_{regime_name}_w{window}"] = bias_val
                    df.at[i, f"cond_mae_{model}_{regime_name}_w{window}"] = mae_val
                    df.at[i, f"cond_count_{model}_{regime_name}_w{window}"] = cond_count

                mae_vals = np.array(
                    [
                        df.at[i, f"cond_mae_{model}_{regime_name}_w{window}"]
                        for model in BASE_MODELS
                    ],
                    dtype=float,
                )
                weights = 1.0 / (mae_vals + 0.1)
                df.at[i, f"cond_entropy_{regime_name}_w{window}"] = weights_entropy(weights)
                df.at[i, f"cond_effN_{regime_name}_w{window}"] = weights_eff_n(weights)

        # Family G1: skill-weighted ensembles
        for window in W_SKILL:
            if eligible.size < window:
                df.at[i, f"ens_wt_mae_w{window}"] = np.nan
                df.at[i, f"ens_wt_rmse_w{window}"] = np.nan
                continue
            idx = eligible[-window:]
            mae_vals = []
            rmse_vals = []
            for model in BASE_MODELS:
                resid = resid_by_model[model][idx]
                resid = resid[np.isfinite(resid)]
                if resid.size == 0:
                    mae_vals.append(np.nan)
                    rmse_vals.append(np.nan)
                else:
                    mae_vals.append(float(np.mean(np.abs(resid))))
                    rmse_vals.append(float(np.sqrt(np.mean(resid**2))))
            mae_vals = np.array(mae_vals, dtype=float)
            rmse_vals = np.array(rmse_vals, dtype=float)
            w_mae = _normalize_weights(1.0 / (mae_vals + 0.1))
            w_rmse = _normalize_weights(1.0 / (rmse_vals + 0.1))
            if w_mae is None:
                df.at[i, f"ens_wt_mae_w{window}"] = np.nan
            else:
                df.at[i, f"ens_wt_mae_w{window}"] = float(np.sum(w_mae * forecast_matrix[i]))
            if w_rmse is None:
                df.at[i, f"ens_wt_rmse_w{window}"] = np.nan
            else:
                df.at[i, f"ens_wt_rmse_w{window}"] = float(np.sum(w_rmse * forecast_matrix[i]))

        # Family G2: BMA features
        bma_weights_w180 = None
        bma_mus_w180 = None
        bma_sigmas_w180 = None
        for window in W_BMA:
            if eligible.size < window:
                _fill_bma_nan(df, i, window)
                continue
            idx = eligible[-window:]
            mus = []
            sigmas = []
            for model in BASE_MODELS:
                x = forecast_matrix[idx, BASE_MODELS.index(model)]
                y = actual[idx]
                a, b, sigma = fit_linear_calibration(x, y)
                if np.isnan(a) or np.isnan(b) or np.isnan(sigma):
                    mus.append(np.nan)
                    sigmas.append(np.nan)
                else:
                    mus.append(float(a + b * forecast_matrix[i, BASE_MODELS.index(model)]))
                    sigmas.append(float(max(sigma, 0.5)))
            mus = np.array(mus, dtype=float)
            sigmas = np.array(sigmas, dtype=float)
            weights = 1.0 / (sigmas + 1e-6)
            df.at[i, f"bma_mean_w{window}"] = mixture_mean(weights, mus)
            df.at[i, f"bma_median_w{window}"] = mixture_quantile(0.5, weights, mus, sigmas)
            df.at[i, f"bma_p10_w{window}"] = mixture_quantile(0.10, weights, mus, sigmas)
            df.at[i, f"bma_p90_w{window}"] = mixture_quantile(0.90, weights, mus, sigmas)
            df.at[i, f"bma_effN_w{window}"] = weights_eff_n(weights)
            if window == 180:
                bma_weights_w180 = weights
                bma_mus_w180 = mus
                bma_sigmas_w180 = sigmas

        # Family G3: ridge superensemble
        ridge_coef_w365 = None
        for window in W_RIDGE:
            if eligible.size < window:
                df.at[i, f"ridge_pred_w{window}"] = np.nan
                df.at[i, f"ridge_intercept_w{window}"] = np.nan
                continue
            idx = eligible[-window:]
            X = forecast_matrix[idx]
            y = actual[idx]
            coef, intercept = fit_ridge(X, y, alpha=1.0)
            if np.isnan(intercept) or np.isnan(coef).all():
                df.at[i, f"ridge_pred_w{window}"] = np.nan
                df.at[i, f"ridge_intercept_w{window}"] = np.nan
            else:
                df.at[i, f"ridge_pred_w{window}"] = float(intercept + np.dot(coef, forecast_matrix[i]))
                df.at[i, f"ridge_intercept_w{window}"] = float(intercept)
            if window == 365:
                ridge_coef_w365 = coef

        # Family G4: AR(1) error correction
        for window in W_AR:
            if eligible.size < window:
                _fill_ar_nan(df, i, window)
                continue
            idx = eligible[-window:]
            resid = actual[idx] - ens_mean[idx]
            phi = fit_ar1(resid)
            if np.isnan(phi):
                _fill_ar_nan(df, i, window)
                continue
            last_resid = resid[-1] if resid.size else np.nan
            pred_resid2 = (phi**2) * last_resid if np.isfinite(last_resid) else np.nan
            df.at[i, f"ar1_phi_ens_w{window}"] = float(phi)
            df.at[i, f"ar1_predresid2_ens_w{window}"] = float(pred_resid2)
            df.at[i, f"ar1_mu_ens_w{window}"] = float(ens_mean[i] + pred_resid2)

        # Family G5: seasonally constrained analogs
        if not z_valid[i]:
            _fill_analog_nan(df, i)
        else:
            doy_i = doy_values[i]
            doy_diff = np.abs(doy_values[eligible] - doy_i)
            doy_diff = np.minimum(doy_diff, 365 - doy_diff)
            valid_idx = eligible[(doy_diff <= 30) & z_valid[eligible]]
            if valid_idx.size < 25:
                _fill_analog_nan(df, i)
            else:
                z_current = z_scaled[i]
                z_past = z_scaled[valid_idx]
                distances = np.linalg.norm(z_past - z_current, axis=1)
                order = np.argsort(distances)
                idx = valid_idx[order][:25]
                ys = actual[idx]
                ys = ys[np.isfinite(ys)]
                if ys.size == 0:
                    _fill_analog_nan(df, i)
                else:
                    df.at[i, "anen_mean_k25"] = float(np.mean(ys))
                    df.at[i, "anen_median_k25"] = float(np.median(ys))
                    df.at[i, "anen_p10_k25"] = float(np.quantile(ys, 0.10))
                    df.at[i, "anen_p90_k25"] = float(np.quantile(ys, 0.90))
                    df.at[i, "anen_std_k25"] = float(np.std(ys, ddof=0))
                    df.at[i, "anen_iqr_k25"] = float(np.quantile(ys, 0.75) - np.quantile(ys, 0.25))

        # Family G6/G7 diagnostics
        if bma_weights_w180 is None or bma_mus_w180 is None or bma_sigmas_w180 is None:
            for model in BASE_MODELS:
                df.at[i, f"bma_w_{model}_w180"] = np.nan
            df.at[i, "bma_entropy_w180"] = np.nan
            df.at[i, "bma_var_w180"] = np.nan
        else:
            weights = _normalize_weights(bma_weights_w180)
            if weights is None:
                for model in BASE_MODELS:
                    df.at[i, f"bma_w_{model}_w180"] = np.nan
                df.at[i, "bma_entropy_w180"] = np.nan
                df.at[i, "bma_var_w180"] = np.nan
            else:
                for idx, model in enumerate(BASE_MODELS):
                    df.at[i, f"bma_w_{model}_w180"] = float(weights[idx])
                df.at[i, "bma_entropy_w180"] = weights_entropy(weights)
                df.at[i, "bma_var_w180"] = mixture_variance(weights, bma_mus_w180, bma_sigmas_w180)

        if ridge_coef_w365 is None or np.isnan(ridge_coef_w365).all():
            df.at[i, "ridge_l1norm_w365"] = np.nan
            df.at[i, "ridge_l2norm_w365"] = np.nan
        else:
            df.at[i, "ridge_l1norm_w365"] = float(np.sum(np.abs(ridge_coef_w365)))
            df.at[i, "ridge_l2norm_w365"] = float(np.sqrt(np.sum(ridge_coef_w365**2)))


def _normalize_weights(weights: np.ndarray) -> np.ndarray | None:
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    s = float(np.sum(weights))
    if s <= 0.0:
        return None
    return weights / s


def _fill_rowwise_nan(df: pd.DataFrame, row_idx: int) -> None:
    for regime in ["spread", "disagree", "anom"]:
        for window in W_COND:
            _fill_conditional_nan(df, row_idx, regime, window)
    for window in W_SKILL:
        df.at[row_idx, f"ens_wt_mae_w{window}"] = np.nan
        df.at[row_idx, f"ens_wt_rmse_w{window}"] = np.nan
    for window in W_BMA:
        _fill_bma_nan(df, row_idx, window)
    for window in W_RIDGE:
        df.at[row_idx, f"ridge_pred_w{window}"] = np.nan
        df.at[row_idx, f"ridge_intercept_w{window}"] = np.nan
    for window in W_AR:
        _fill_ar_nan(df, row_idx, window)
    _fill_analog_nan(df, row_idx)
    for model in BASE_MODELS:
        df.at[row_idx, f"bma_w_{model}_w180"] = np.nan
    df.at[row_idx, "bma_entropy_w180"] = np.nan
    df.at[row_idx, "bma_var_w180"] = np.nan
    df.at[row_idx, "ridge_l1norm_w365"] = np.nan
    df.at[row_idx, "ridge_l2norm_w365"] = np.nan


def _fill_conditional_nan(df: pd.DataFrame, row_idx: int, regime: str, window: int) -> None:
    for model in BASE_MODELS:
        df.at[row_idx, f"cond_bias_{model}_{regime}_w{window}"] = np.nan
        df.at[row_idx, f"cond_mae_{model}_{regime}_w{window}"] = np.nan
        df.at[row_idx, f"cond_count_{model}_{regime}_w{window}"] = np.nan
    df.at[row_idx, f"cond_entropy_{regime}_w{window}"] = np.nan
    df.at[row_idx, f"cond_effN_{regime}_w{window}"] = np.nan


def _fill_bma_nan(df: pd.DataFrame, row_idx: int, window: int) -> None:
    df.at[row_idx, f"bma_mean_w{window}"] = np.nan
    df.at[row_idx, f"bma_median_w{window}"] = np.nan
    df.at[row_idx, f"bma_p10_w{window}"] = np.nan
    df.at[row_idx, f"bma_p90_w{window}"] = np.nan
    df.at[row_idx, f"bma_effN_w{window}"] = np.nan


def _fill_ar_nan(df: pd.DataFrame, row_idx: int, window: int) -> None:
    df.at[row_idx, f"ar1_phi_ens_w{window}"] = np.nan
    df.at[row_idx, f"ar1_predresid2_ens_w{window}"] = np.nan
    df.at[row_idx, f"ar1_mu_ens_w{window}"] = np.nan


def _fill_analog_nan(df: pd.DataFrame, row_idx: int) -> None:
    df.at[row_idx, "anen_mean_k25"] = np.nan
    df.at[row_idx, "anen_median_k25"] = np.nan
    df.at[row_idx, "anen_p10_k25"] = np.nan
    df.at[row_idx, "anen_p90_k25"] = np.nan
    df.at[row_idx, "anen_std_k25"] = np.nan
    df.at[row_idx, "anen_iqr_k25"] = np.nan
