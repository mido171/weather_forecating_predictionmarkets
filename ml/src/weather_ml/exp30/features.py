"""Feature engineering helpers for exp30 sweeps."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from weather_ml import derived_features
from weather_ml import time_feature_library as tfl
from weather_ml.mos_utils import effective_sample_size, weighted_entropy

from .config import GUIDANCE_COLS, MOS_VALUE_COLUMNS

LOGGER = logging.getLogger(__name__)


@contextmanager
def _feature_step(label: str):
    start = time.time()
    LOGGER.info("FEATURE_START %s", label)
    try:
        yield
    finally:
        LOGGER.info("FEATURE_END %s (%.1fs)", label, time.time() - start)


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
    asof = pd.to_datetime(df["asof_utc"], utc=True, errors="coerce")
    df["asof_hour"] = asof.dt.hour.fillna(0).astype(int)
    asof_radians = 2 * np.pi * df["asof_hour"] / 24.0
    df["asof_sin_hour"] = np.sin(asof_radians)
    df["asof_cos_hour"] = np.cos(asof_radians)
    return df


def add_station_onehot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    stations = sorted(df["station_id"].dropna().unique().tolist())
    if len(stations) <= 1:
        return df
    for station in stations:
        df[f"station_is_{station}"] = (
            df["station_id"].astype(str).str.upper() == str(station).upper()
        ).astype(int)
    return df


def build_guidance_base(df: pd.DataFrame, guidance_cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    return df[list(guidance_cols)].astype(float)


def build_ensemble_features(df: pd.DataFrame, guidance_cols: list[str]) -> pd.DataFrame:
    values = df[guidance_cols].to_numpy(dtype=float)
    n = len(guidance_cols)
    features = derived_features.compute_rowwise_features(df, guidance_cols)
    features = features.rename(columns={
        f"ens_mean_{n}": "ens_mean_guidance",
        f"ens_median_{n}": "ens_median_guidance",
        f"ens_min_{n}": "ens_min_guidance",
        f"ens_max_{n}": "ens_max_guidance",
        f"ens_range_{n}": "ens_range_guidance",
        f"ens_std_{n}": "ens_std_guidance",
        f"ens_iqr_{n}": "ens_iqr_guidance",
        f"ens_mad_{n}": "ens_mad_guidance",
        f"ens_outlier_gap_{n}": "ens_outlier_gap_guidance",
    })
    features["guid_spread"] = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
    features["guid_iqr"] = np.nanquantile(values, 0.75, axis=1) - np.nanquantile(values, 0.25, axis=1)
    return features


def rolling_count(series: pd.Series, window: int, lag: int, group_key: pd.Series) -> pd.Series:
    valid = series.notna().astype(float)
    return tfl.rolling_sum(
        valid,
        window=window,
        min_periods=1,
        lag=lag,
        group_key=group_key,
    )


def _apply_with_min_obs(
    series: pd.Series,
    values: pd.Series,
    min_obs: int,
) -> tuple[pd.Series, pd.Series]:
    mask = series >= min_obs
    flagged = (~mask).astype(int)
    out = values.where(mask)
    return out, flagged


def rolling_mean_feature(
    series: pd.Series,
    *,
    window: int,
    min_obs: int,
    lag: int,
    group_key: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    count = rolling_count(series, window, lag, group_key)
    mean = tfl.rolling_mean(
        series, window=window, min_periods=1, lag=lag, group_key=group_key
    )
    return _apply_with_min_obs(count, mean, min_obs)


def rolling_mae_feature(
    series: pd.Series,
    *,
    window: int,
    min_obs: int,
    lag: int,
    group_key: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    count = rolling_count(series, window, lag, group_key)
    mae = tfl.rolling_mean(
        series.abs(), window=window, min_periods=1, lag=lag, group_key=group_key
    )
    return _apply_with_min_obs(count, mae, min_obs)


def rolling_corr_feature(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    window: int,
    min_obs: int,
    lag: int,
    group_key: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    count = rolling_count(series_a, window, lag, group_key)
    corr = tfl.rolling_corr(
        series_a,
        series_b,
        window=window,
        min_periods=1,
        lag=lag,
        group_key=group_key,
    )
    return _apply_with_min_obs(count, corr, min_obs)


def rolling_trimmed_mean(
    series: pd.Series,
    *,
    window: int,
    min_obs: int,
    lag: int,
    trim: float,
    group_key: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    count = rolling_count(series, window, lag, group_key)

    def _trim(values: np.ndarray) -> float:
        vals = values[np.isfinite(values)]
        if vals.size == 0:
            return np.nan
        vals = np.sort(vals)
        k = int(np.floor(trim * len(vals)))
        if len(vals) - 2 * k <= 0:
            return np.nan
        return float(np.mean(vals[k:len(vals) - k]))

    trimmed = tfl.rolling_apply(
        series,
        window=window,
        min_periods=1,
        lag=lag,
        func=_trim,
        group_key=group_key,
    )
    return _apply_with_min_obs(count, trimmed, min_obs)


def rolling_winsor_mean(
    series: pd.Series,
    *,
    window: int,
    min_obs: int,
    lag: int,
    p_low: float,
    p_high: float,
    group_key: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    count = rolling_count(series, window, lag, group_key)

    def _winsor(values: np.ndarray) -> float:
        vals = values[np.isfinite(values)]
        if vals.size == 0:
            return np.nan
        lo = np.quantile(vals, p_low)
        hi = np.quantile(vals, p_high)
        clipped = np.clip(vals, lo, hi)
        return float(np.mean(clipped))

    winsor = tfl.rolling_apply(
        series,
        window=window,
        min_periods=1,
        lag=lag,
        func=_winsor,
        group_key=group_key,
    )
    return _apply_with_min_obs(count, winsor, min_obs)


def rolling_spearman(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    window: int,
    min_obs: int,
    lag: int,
    group_key: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    count = rolling_count(series_a, window, lag, group_key)

    def _spearman(values: np.ndarray) -> float:
        if values.ndim != 2 or values.shape[1] != 2:
            return np.nan
        a = values[:, 0]
        b = values[:, 1]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            return np.nan
        a = pd.Series(a[mask]).rank().to_numpy()
        b = pd.Series(b[mask]).rank().to_numpy()
        if a.size < 2:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    output = pd.Series(np.nan, index=series_a.index, dtype=float)
    groups = group_key.to_numpy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for pos, row_idx in enumerate(idx):
            if pos < lag:
                continue
            start = max(0, pos - window)
            end = pos - lag + 1
            window_idx = idx[start:end]
            if len(window_idx) < min_obs:
                continue
            a_vals = series_a.iloc[window_idx].to_numpy(dtype=float)
            b_vals = series_b.iloc[window_idx].to_numpy(dtype=float)
            stacked = np.vstack([a_vals, b_vals]).T
            output.iloc[row_idx] = _spearman(stacked)
    return _apply_with_min_obs(count, output, min_obs)


def compute_bias_features(
    df: pd.DataFrame,
    guidance_cols: list[str],
    *,
    windows: Iterable[int],
    group_key: pd.Series,
    suffix: str,
) -> FeatureSet:
    with _feature_step("compute_bias_features"):
        features = pd.DataFrame(index=df.index)
        meta = {"type": "bias_features", "windows": list(windows)}
        for col in guidance_cols:
            err = df[col] - df["actual_tmax_f"]
            for window in windows:
                min_obs = max(5, int(np.ceil(0.5 * window)))
                bias, flag = rolling_mean_feature(
                    err, window=window, min_obs=min_obs, lag=1, group_key=group_key
                )
                name = f"bias_{col}_rm{window}{suffix}"
                features[name] = bias.fillna(0.0)
                features[f"{name}_insufficient"] = flag
        return FeatureSet(frame=features, columns=list(features.columns), meta=meta)


def compute_corr_features(
    df: pd.DataFrame,
    guidance_cols: list[str],
    *,
    windows: Iterable[int],
    group_key: pd.Series,
    suffix: str,
) -> FeatureSet:
    with _feature_step("compute_corr_features"):
        features = pd.DataFrame(index=df.index)
        meta = {"type": "corr_features", "windows": list(windows)}
        for col in guidance_cols:
            for window in windows:
                min_obs = max(10, int(np.ceil(0.5 * window)))
                corr, flag = rolling_corr_feature(
                    df[col],
                    df["actual_tmax_f"],
                    window=window,
                    min_obs=min_obs,
                    lag=1,
                    group_key=group_key,
                )
                name = f"corr_{col}_rm{window}{suffix}"
                features[name] = corr.fillna(0.0)
                features[f"{name}_insufficient"] = flag
        return FeatureSet(frame=features, columns=list(features.columns), meta=meta)


def compute_trimmed_winsor_bias(
    df: pd.DataFrame,
    guidance_cols: list[str],
    *,
    windows: Iterable[int],
    group_key: pd.Series,
    trim: float = 0.1,
) -> FeatureSet:
    with _feature_step("compute_trimmed_winsor_bias"):
        features = pd.DataFrame(index=df.index)
        meta = {"type": "trimmed_bias", "windows": list(windows)}
        for col in guidance_cols:
            err = df[col] - df["actual_tmax_f"]
            for window in windows:
                min_obs = max(5, int(np.ceil(0.5 * window)))
                trim_mean, flag_trim = rolling_trimmed_mean(
                    err,
                    window=window,
                    min_obs=min_obs,
                    lag=1,
                    trim=trim,
                    group_key=group_key,
                )
                winsor_mean, flag_win = rolling_winsor_mean(
                    err,
                    window=window,
                    min_obs=min_obs,
                    lag=1,
                    p_low=0.05,
                    p_high=0.95,
                    group_key=group_key,
                )
                trim_name = f"bias_{col}_trim_rm{window}"
                win_name = f"bias_{col}_wins_rm{window}"
                features[trim_name] = trim_mean.fillna(0.0).clip(-8.0, 8.0)
                features[f"{trim_name}_insufficient"] = flag_trim
                features[win_name] = winsor_mean.fillna(0.0).clip(-8.0, 8.0)
                features[f"{win_name}_insufficient"] = flag_win
        return FeatureSet(frame=features, columns=list(features.columns), meta=meta)


def compute_drift_features(df: pd.DataFrame, guidance_cols: list[str]) -> FeatureSet:
    with _feature_step("compute_drift_features"):
        features = pd.DataFrame(index=df.index)
        meta = {"type": "bias_drift"}
        drift_vals = []
        for col in guidance_cols:
            b7 = df.get(f"bias_{col}_trim_rm7", pd.Series(0.0, index=df.index))
            b60 = df.get(f"bias_{col}_trim_rm60", pd.Series(0.0, index=df.index))
            b14 = df.get(f"bias_{col}_trim_rm14", pd.Series(0.0, index=df.index))
            b120 = df.get(f"bias_{col}_trim_rm120", pd.Series(0.0, index=df.index))
            drift_7v60 = (b7 - b60).clip(-8.0, 8.0)
            drift_14v120 = (b14 - b120).clip(-8.0, 8.0)
            features[f"drift_{col}_7v60"] = drift_7v60
            features[f"drift_{col}_14v120"] = drift_14v120
            drift_vals.append(drift_7v60.abs())
        if drift_vals:
            drift_stack = np.vstack([s.to_numpy(dtype=float) for s in drift_vals])
            features["drift_mean_abs_7v60"] = np.nanmean(drift_stack, axis=0)
            features["drift_max_abs_7v60"] = np.nanmax(drift_stack, axis=0)
        return FeatureSet(frame=features, columns=list(features.columns), meta=meta)


def compute_ewma_features(
    df: pd.DataFrame,
    guidance_cols: list[str],
    *,
    halflives: Iterable[int],
    group_key: pd.Series,
) -> FeatureSet:
    with _feature_step("compute_ewma_features"):
        features = pd.DataFrame(index=df.index)
        meta = {"type": "ewma_bias", "halflives": list(halflives)}

        def _group_ewm(series: pd.Series, halflife: int) -> pd.Series:
            return series.groupby(group_key).apply(
                lambda s: s.shift(1).ewm(
                    halflife=halflife, min_periods=1, adjust=False, ignore_na=True
                ).mean()
            ).reset_index(level=0, drop=True)

        def _group_ewm_corr(a: pd.Series, b: pd.Series, halflife: int) -> pd.Series:
            output = pd.Series(np.nan, index=a.index, dtype=float)
            groups = group_key.to_numpy()
            for g in np.unique(groups):
                idx = np.where(groups == g)[0]
                a_vals = a.iloc[idx].shift(1)
                b_vals = b.iloc[idx].shift(1)
                mean_a = a_vals.ewm(
                    halflife=halflife, min_periods=1, adjust=False, ignore_na=True
                ).mean()
                mean_b = b_vals.ewm(
                    halflife=halflife, min_periods=1, adjust=False, ignore_na=True
                ).mean()
                cov = (a_vals * b_vals).ewm(
                    halflife=halflife, min_periods=1, adjust=False, ignore_na=True
                ).mean() - mean_a * mean_b
                var_a = (a_vals * a_vals).ewm(
                    halflife=halflife, min_periods=1, adjust=False, ignore_na=True
                ).mean() - mean_a * mean_a
                var_b = (b_vals * b_vals).ewm(
                    halflife=halflife, min_periods=1, adjust=False, ignore_na=True
                ).mean() - mean_b * mean_b
                denom = np.sqrt(var_a * var_b)
                corr = cov / denom
                output.iloc[idx] = corr.to_numpy(dtype=float)
            return output

        for col in guidance_cols:
            err = df[col] - df["actual_tmax_f"]
            abs_err = err.abs()
            for hl in halflives:
                ewm_bias = _group_ewm(err, hl)
                ewm_mae = _group_ewm(abs_err, hl)
                count = rolling_count(err, window=60, lag=1, group_key=group_key)
                shrink = (count / 30.0).clip(0.0, 1.0)
                corr_raw = _group_ewm_corr(df[col], df["actual_tmax_f"], hl)
                corr = (corr_raw * shrink).clip(-1.0, 1.0)
                bias_name = f"ewma_bias_{col}_hl{hl}"
                mae_name = f"ewma_mae_{col}_hl{hl}"
                corr_name = f"ewma_corr_{col}_hl{hl}"
                features[bias_name] = ewm_bias.fillna(0.0).clip(-8.0, 8.0)
                features[mae_name] = ewm_mae.fillna(0.0)
                features[corr_name] = corr.fillna(0.0)
                features[f"{bias_name}_insufficient"] = (count < 10).astype(int)
                features[f"n_eff_{col}_hl{hl}"] = count.clip(upper=180).astype(float)
                if hl == 21:
                    rm60 = df.get(
                        f"bias_{col}_rm60_l2", pd.Series(0.0, index=df.index)
                    )
                    features[f"ewma_bias_minus_rm60_{col}"] = (
                        ewm_bias.fillna(0.0) - rm60
                    ).clip(-8.0, 8.0)
        return FeatureSet(frame=features, columns=list(features.columns), meta=meta)


def compute_gated_bias_features(
    df: pd.DataFrame,
    guidance_cols: list[str],
    *,
    threshold: float,
    slope: float,
) -> FeatureSet:
    with _feature_step("compute_gated_bias_features"):
        features = pd.DataFrame(index=df.index)
        meta = {"type": "gated_bias", "threshold": threshold, "slope": slope}
        global_ws = []
        for col in guidance_cols:
            b7 = df.get(f"bias_{col}_rm7_l2", pd.Series(0.0, index=df.index))
            b60 = df.get(f"bias_{col}_rm60_l2", pd.Series(0.0, index=df.index))
            u = (b7 - b60).abs()
            w = 1.0 / (1.0 + np.exp(-(u - threshold) / slope))
            gated = w * b7 + (1.0 - w) * b60
            features[f"bias_{col}_gated"] = gated
            features[f"w_gate_{col}"] = w
            global_ws.append(w)
        if global_ws:
            stack = np.vstack([s.to_numpy(dtype=float) for s in global_ws])
            features["w_global"] = np.nanmean(stack, axis=0)
            features["mean_abs_bias_gated"] = np.nanmean(
                np.abs(
                    np.vstack(
                        [
                            features[f"bias_{c}_gated"].to_numpy(dtype=float)
                            for c in guidance_cols
                        ]
                    )
                ),
                axis=0,
            )
        return FeatureSet(frame=features, columns=list(features.columns), meta=meta)


def compute_spearman_features(
    df: pd.DataFrame,
    guidance_cols: list[str],
    *,
    windows: Iterable[int],
    group_key: pd.Series,
) -> FeatureSet:
    with _feature_step("compute_spearman_features"):
        features = pd.DataFrame(index=df.index)
        meta = {"type": "spearman_corr", "windows": list(windows)}
        for col in guidance_cols:
            for window in windows:
                min_obs = 15 if window == 30 else 25 if window == 60 else 40
                corr, flag = rolling_spearman(
                    df[col],
                    df["actual_tmax_f"],
                    window=window,
                    min_obs=min_obs,
                    lag=1,
                    group_key=group_key,
                )
                name = f"spearman_{col}_rm{window}"
                corr = corr.fillna(0.0)
                features[name] = corr.clip(-1.0, 1.0)
                features[f"{name}_insufficient"] = flag
                if window == 60:
                    features[f"abs_{name}"] = corr.abs()
                    features[f"inv_abs_{name}"] = 1.0 - corr.abs()
        return FeatureSet(frame=features, columns=list(features.columns), meta=meta)


def compute_kalman_bias(
    df: pd.DataFrame,
    guidance_cols: list[str],
    *,
    q: float,
    r: float,
    p0: float,
    group_key: pd.Series,
) -> FeatureSet:
    with _feature_step("compute_kalman_bias"):
        features = pd.DataFrame(index=df.index)
        meta = {"type": "kalman_bias", "q": q, "r": r, "p0": p0}
        dates = pd.to_datetime(df["target_date_local"]).dt.date
        groups = group_key.to_numpy()
        for col in guidance_cols:
            err_all = (df[col] - df["actual_tmax_f"]).to_numpy(dtype=float)
            b = np.zeros(len(df), dtype=float)
            p = np.full(len(df), p0, dtype=float)
            k_gain = np.zeros(len(df), dtype=float)
            for g in np.unique(groups):
                idx = np.where(groups == g)[0]
                idx = idx[np.argsort(dates[idx])]
                last_b = 0.0
                last_p = p0
                for pos, row_idx in enumerate(idx):
                    if pos == 0:
                        b[row_idx] = 0.0
                        p[row_idx] = p0
                        k_gain[row_idx] = 0.0
                        continue
                    e = err_all[row_idx]
                    p_pred = last_p + q
                    b_pred = last_b
                    if np.isfinite(e):
                        k = p_pred / (p_pred + r)
                        b_upd = b_pred + k * (e - b_pred)
                        p_upd = (1.0 - k) * p_pred
                    else:
                        k = 0.0
                        b_upd = b_pred
                        p_upd = p_pred
                    b[row_idx] = b_upd
                    p[row_idx] = p_upd
                    k_gain[row_idx] = k
                    last_b = b_upd
                    last_p = p_upd
            bias = pd.Series(b, index=df.index).shift(1).fillna(0.0)
            gain = pd.Series(k_gain, index=df.index).shift(1).fillna(0.0)
            unc = pd.Series(np.sqrt(p), index=df.index).shift(1).fillna(0.0)
            features[f"kalman_bias_{col}"] = bias
            features[f"kalman_gain_{col}"] = gain
            features[f"kalman_unc_{col}"] = unc
            rm60 = df.get(f"bias_{col}_rm60_l2", pd.Series(0.0, index=df.index))
            features[f"kalman_bias_minus_rm60_{col}"] = (bias - rm60).clip(-8.0, 8.0)
        return FeatureSet(frame=features, columns=list(features.columns), meta=meta)


def compute_rolling_mae_weights(
    df: pd.DataFrame,
    guidance_cols: list[str],
    *,
    windows: Iterable[int],
    group_key: pd.Series,
    train_mask: np.ndarray,
) -> FeatureSet:
    with _feature_step("compute_rolling_mae_weights"):
        features = pd.DataFrame(index=df.index)
        meta = {"type": "rolling_mae_weights", "windows": list(windows)}
        global_mae = {}
        for col in guidance_cols:
            err = (df[col] - df["actual_tmax_f"]).abs()
            global_mae[col] = (
                float(np.nanmedian(err[train_mask])) if train_mask.any() else 1.0
            )

        for window in windows:
            mae_cols = []
            weight_cols = []
            for col in guidance_cols:
                err = df[col] - df["actual_tmax_f"]
                min_obs = 20 if window == 30 else 40
                mae, flag = rolling_mae_feature(
                    err, window=window, min_obs=min_obs, lag=1, group_key=group_key
                )
                mae = mae.fillna(global_mae[col])
                features[f"mae_{col}_rm{window}"] = mae
                features[f"mae_{col}_rm{window}_insufficient"] = flag
                weight = 1.0 / (mae + 0.1)
                mae_cols.append(mae)
                weight_cols.append(weight)
            weights = np.vstack([w.to_numpy(dtype=float) for w in weight_cols]).T
            weights = np.clip(weights, 0.05, 0.60)
            denom = np.sum(weights, axis=1, keepdims=True)
            weights = np.where(denom == 0, 1.0 / len(guidance_cols), weights / denom)
            guidance_vals = df[guidance_cols].to_numpy(dtype=float)
            ens_wmean = np.nansum(weights * guidance_vals, axis=1)
            spread = np.sqrt(
                np.nansum(weights * (guidance_vals - ens_wmean[:, None]) ** 2, axis=1)
            )
            features[f"ens_wmean_rm{window}"] = ens_wmean
            features[f"ens_wspread_rm{window}"] = spread
            features[f"w_entropy_rm{window}"] = [weighted_entropy(w) for w in weights]
            features[f"ens_wmean_minus_simple_mean_rm{window}"] = (
                ens_wmean - np.nanmean(guidance_vals, axis=1)
            )
            mae_stack = np.vstack([m.to_numpy(dtype=float) for m in mae_cols]).T
            all_nan = np.all(~np.isfinite(mae_stack), axis=1)
            mae_stack = np.where(np.isfinite(mae_stack), mae_stack, np.inf)
            best_idx = np.argmin(mae_stack, axis=1)
            for idx, col in enumerate(guidance_cols):
                features[f"best_model_is_{col}_rm{window}"] = (
                    (best_idx == idx) & ~all_nan
                ).astype(int)
        return FeatureSet(frame=features, columns=list(features.columns), meta=meta)


def build_mos_value(df: pd.DataFrame, code: str) -> pd.Series:
    cols = MOS_VALUE_COLUMNS.get(code)
    if not cols:
        return pd.Series(np.nan, index=df.index)
    gfs_col, nam_col = cols
    gfs = df.get(gfs_col, pd.Series(np.nan, index=df.index)).astype(float)
    nam = df.get(nam_col, pd.Series(np.nan, index=df.index)).astype(float)
    return gfs.combine(nam, lambda a, b: np.nanmean([a, b]), fill_value=np.nan)


def mos_trailing_stats(
    series: pd.Series,
    *,
    window: int,
    group_key: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    count = rolling_count(series, window, lag=1, group_key=group_key)
    median = tfl.rolling_quantile(
        series,
        window=window,
        min_periods=1,
        lag=1,
        q=0.5,
        group_key=group_key,
    )
    q75 = tfl.rolling_quantile(
        series,
        window=window,
        min_periods=1,
        lag=1,
        q=0.75,
        group_key=group_key,
    )
    q25 = tfl.rolling_quantile(
        series,
        window=window,
        min_periods=1,
        lag=1,
        q=0.25,
        group_key=group_key,
    )
    iqr = q75 - q25
    return median, iqr, count


def compute_mos_surface_anomalies(
    df: pd.DataFrame,
    *,
    codes: list[str],
    group_key: pd.Series,
) -> FeatureSet:
    with _feature_step("compute_mos_surface_anomalies"):
        features = pd.DataFrame(index=df.index)
        meta = {"type": "mos_surface_anom", "codes": codes}
        for code in codes:
            values = build_mos_value(df, code)
            features[f"mos_{code}"] = values
            for window in (7, 30):
                med, iqr, count = mos_trailing_stats(
                    values, window=window, group_key=group_key
                )
                features[f"mos_{code}_med_rm{window}"] = med
                features[f"mos_{code}_iqr_rm{window}"] = iqr
                features[f"mos_{code}_anom_rm{window}"] = values - med
                features[f"mos_{code}_count_rm{window}"] = count
                features[f"mos_{code}_missing_frac_rm{window}"] = 1.0 - (count / window)
            missing = values.isna().astype(int)
            features[f"mos_{code}_missing"] = missing
            if code == "wdr":
                radians = np.deg2rad(values.fillna(0.0))
                features["mos_wdr_sin"] = np.sin(radians)
                features["mos_wdr_cos"] = np.cos(radians)
        return FeatureSet(frame=features, columns=list(features.columns), meta=meta)


def compute_mos_dewpoint_wind(df: pd.DataFrame, group_key: pd.Series) -> FeatureSet:
    with _feature_step("compute_mos_dewpoint_wind"):
        features = pd.DataFrame(index=df.index)
        meta = {"type": "mos_dewpoint_wind"}
        tmp = build_mos_value(df, "tmp")
        dpt = build_mos_value(df, "dpt")
        wsp = build_mos_value(df, "wsp")
        wdr = build_mos_value(df, "wdr")
        dd = tmp - dpt
        radians = np.deg2rad(wdr.fillna(0.0))
        u = -wsp.fillna(0.0) * np.sin(radians)
        v = -wsp.fillna(0.0) * np.cos(radians)
        features["mos_dd"] = dd
        features["mos_u"] = u
        features["mos_v"] = v
        for window in (7, 30):
            med, _, _ = mos_trailing_stats(dd, window=window, group_key=group_key)
            features[f"mos_dd_anom_rm{window}"] = dd - med
            med_u, _, _ = mos_trailing_stats(u, window=window, group_key=group_key)
            med_v, _, _ = mos_trailing_stats(v, window=window, group_key=group_key)
            features[f"mos_u_anom_rm{window}"] = u - med_u
            features[f"mos_v_anom_rm{window}"] = v - med_v
        features["mos_dd_missing"] = (tmp.isna() | dpt.isna()).astype(int)
        features["mos_wind_missing"] = (wsp.isna() | wdr.isna()).astype(int)
        return FeatureSet(frame=features, columns=list(features.columns), meta=meta)


def compute_mos_cloud_precip(df: pd.DataFrame) -> FeatureSet:
    with _feature_step("compute_mos_cloud_precip"):
        features = pd.DataFrame(index=df.index)
        meta = {"type": "mos_cloud_precip"}
        precip_codes = ["p06", "p12", "q06", "q12", "pos", "poz"]
        precip_vals = []
        for code in precip_codes:
            values = build_mos_value(df, code)
            features[f"mos_{code}"] = values.fillna(0.0)
            features[f"mos_{code}_missing"] = values.isna().astype(int)
            precip_vals.append(values)
        precip_stack = (
            np.vstack([s.to_numpy(dtype=float) for s in precip_vals])
            if precip_vals
            else np.empty((0, len(df)))
        )
        precip_max = (
            np.nanmax(precip_stack, axis=0) if precip_stack.size else np.zeros(len(df))
        )
        precip_mean = (
            np.nanmean(precip_stack, axis=0) if precip_stack.size else np.zeros(len(df))
        )
        missing_all = (
            np.all(np.isnan(precip_stack), axis=0)
            if precip_stack.size
            else np.ones(len(df), dtype=bool)
        )
        features["mos_precip_proxy_max"] = np.where(missing_all, 0.0, precip_max)
        features["mos_precip_proxy_mean"] = np.where(missing_all, 0.0, precip_mean)
        features["mos_precip_proxy_all_missing"] = missing_all.astype(int)

        cig = build_mos_value(df, "cig")
        vis = build_mos_value(df, "vis")
        features["mos_log_cig"] = np.log1p(cig.clip(lower=0.0).fillna(0.0))
        features["mos_log_vis"] = np.log1p(vis.clip(lower=0.0).fillna(0.0))
        features["mos_cig_missing"] = cig.isna().astype(int)
        features["mos_vis_missing"] = vis.isna().astype(int)
    return FeatureSet(frame=features, columns=list(features.columns), meta=meta)


def compute_mos_missingness(
    df: pd.DataFrame, codes: list[str], group_key: pd.Series, train_mask: np.ndarray
) -> FeatureSet:
    with _feature_step("compute_mos_missingness"):
        features = pd.DataFrame(index=df.index)
        meta = {"type": "mos_missingness", "codes": codes}
        missing_matrix = []
        for code in codes:
            values = build_mos_value(df, code)
            missing = values.isna().astype(int)
            features[f"mos_{code}_missing"] = missing
            missing_matrix.append(missing.to_numpy(dtype=float))
            # days since last available
            dsl = tfl.days_since_event(missing == 0, lag=1, cap=60, group_key=group_key)
            features[f"mos_{code}_dsl"] = dsl
            features[f"mos_{code}_dsl_capped"] = (dsl >= 60).astype(int)
        if missing_matrix:
            miss = np.vstack(missing_matrix).T
            miss_count = miss.sum(axis=1)
            features["mos_missing_count"] = miss_count
            features["mos_missing_frac"] = miss_count / float(len(codes))
            # PCA on train only
            pca = PCA(n_components=3, random_state=42)
            pca.fit(miss[train_mask])
            comps = pca.transform(miss)
            features["mos_miss_pca1"] = comps[:, 0]
            features["mos_miss_pca2"] = comps[:, 1]
            features["mos_miss_pca3"] = comps[:, 2]
        return FeatureSet(frame=features, columns=list(features.columns), meta=meta)


def compute_walkforward_climatology(
    df: pd.DataFrame,
    *,
    group_key: pd.Series,
    window_doy: int = 3,
) -> FeatureSet:
    with _feature_step("compute_walkforward_climatology"):
        features = pd.DataFrame(index=df.index)
        meta = {"type": "walkforward_climatology", "window_doy": window_doy}
        dates = pd.to_datetime(df["target_date_local"])
        doy = dates.dt.dayofyear.to_numpy()
        y = df["actual_tmax_f"].to_numpy(dtype=float)
        output = np.full(len(df), np.nan, dtype=float)
        iqr_out = np.full(len(df), np.nan, dtype=float)
        stations = group_key.to_numpy()
        for station in np.unique(stations):
            idx = np.where(stations == station)[0]
            for pos, row_idx in enumerate(idx):
                past_idx = idx[:pos]
                if past_idx.size == 0:
                    continue
                past_doy = doy[past_idx]
                target = doy[row_idx]
                mask = np.array(
                    [
                        min(abs(int(d) - int(target)), 365 - abs(int(d) - int(target)))
                        <= window_doy
                        for d in past_doy
                    ]
                )
                selected = past_idx[mask]
                if selected.size == 0:
                    selected = past_idx
                vals = y[selected]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                output[row_idx] = float(np.median(vals))
                iqr_out[row_idx] = float(
                    np.quantile(vals, 0.75) - np.quantile(vals, 0.25)
                )
        fallback = np.nanmedian(y)
        output = np.where(np.isnan(output), fallback, output)
        iqr_out = np.where(np.isnan(iqr_out), 0.0, iqr_out)
        features["clim_med"] = output
        features["clim_iqr"] = iqr_out
        return FeatureSet(frame=features, columns=list(features.columns), meta=meta)


def season_onehot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df["target_date_local"])
    month = dates.dt.month
    season = pd.Series(index=df.index, dtype="string")
    season.loc[month.isin([12, 1, 2])] = "DJF"
    season.loc[month.isin([3, 4, 5])] = "MAM"
    season.loc[month.isin([6, 7, 8])] = "JJA"
    season.loc[month.isin([9, 10, 11])] = "SON"
    for label in ["DJF", "MAM", "JJA", "SON"]:
        df[f"season_{label}"] = (season == label).astype(int)
    return df


def predicted_deciles(series: pd.Series, train_mask: np.ndarray) -> tuple[pd.Series, np.ndarray]:
    train_vals = series[train_mask].to_numpy(dtype=float)
    thresholds = np.quantile(train_vals, np.linspace(0.1, 0.9, 9))
    bins = np.digitize(series.to_numpy(dtype=float), thresholds, right=True)
    return pd.Series(bins, index=series.index), thresholds
