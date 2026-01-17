"""MOS experiment suite (E01-E50) for time-feature experiments."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from weather_ml import metrics
from weather_ml import models_mean
from weather_ml import time_feature_library as tfl

LOGGER = logging.getLogger(__name__)
EPS = 1e-6
DEFAULT_TRUTH_LAG = 2
DEFAULT_BOOTSTRAP = 10000
CALENDAR_COLS = [
    "month",
    "sin_doy",
    "cos_doy",
    "is_weekend",
    "asof_sin_hour",
    "asof_cos_hour",
]


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    name: str
    base_cols: list[str]
    build_features: Callable[["ExperimentContext"], "DerivedFeatureSet"]


@dataclass
class DerivedFeatureSet:
    features: pd.DataFrame
    formulas: list[dict]
    train_fitted: list[dict]


@dataclass
class ExperimentContext:
    df: pd.DataFrame
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    group_station: pd.Series
    group_station_asof: pd.Series
    truth_lag: int
    seed: int

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_suite_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _log_dataset_stats(df: pd.DataFrame) -> None:
    LOGGER.info("CSV row count: %s", len(df))
    if df.empty:
        return
    stations = sorted(df["station_id"].dropna().unique().tolist())
    LOGGER.info("Stations: %s", stations)
    missing = df.isna().sum().to_dict()
    LOGGER.info("Missing counts: %s", missing)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    missing = [col for col in columns if col not in df.columns]
    if missing:
        for col in missing:
            df[col] = np.nan
        LOGGER.warning("Added missing columns with NaN: %s", missing)
    return df


def load_mos_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"station_id": "string"})
    if "actual_tmax_f" not in df.columns and "target_tmax_f" in df.columns:
        df = df.rename(columns={"target_tmax_f": "actual_tmax_f"})
    df["target_date_local"] = pd.to_datetime(
        df["target_date_local"], errors="coerce"
    ).dt.normalize()
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], errors="coerce", utc=True)
    numeric_cols = [
        col
        for col in df.columns
        if col not in ("station_id", "target_date_local", "asof_utc")
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["station_id"] = df["station_id"].astype("string")
    _log_dataset_stats(df)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = tfl.add_calendar_features(df)
    asof = pd.to_datetime(df["asof_utc"], errors="coerce", utc=True)
    df["asof_hour"] = asof.dt.hour.fillna(0).astype(int)
    radians = 2 * np.pi * df["asof_hour"] / 24.0
    df["asof_sin_hour"] = np.sin(radians)
    df["asof_cos_hour"] = np.cos(radians)
    return df


def _get_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _nanmean_two(a: pd.Series, b: pd.Series) -> pd.Series:
    arr = np.vstack([a.to_numpy(dtype=float), b.to_numpy(dtype=float)]).T
    valid = np.isfinite(arr)
    count = valid.sum(axis=1)
    sum_vals = np.where(valid, arr, 0.0).sum(axis=1)
    mean = np.full_like(sum_vals, np.nan, dtype=float)
    np.divide(sum_vals, count, out=mean, where=count > 0)
    return pd.Series(mean, index=a.index, dtype=float)


def add_ensemble_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nx_ens"] = _nanmean_two(_get_col(df, "gfs_n_x_mean"), _get_col(df, "nam_n_x_mean"))
    df["tmp_ens"] = _nanmean_two(_get_col(df, "gfs_tmp_mean"), _get_col(df, "nam_tmp_mean"))
    df["dpt_ens"] = _nanmean_two(_get_col(df, "gfs_dpt_mean"), _get_col(df, "nam_dpt_mean"))
    df["p06_ens"] = _nanmean_two(_get_col(df, "gfs_p06_mean"), _get_col(df, "nam_p06_mean"))
    df["p12_ens"] = _nanmean_two(_get_col(df, "gfs_p12_mean"), _get_col(df, "nam_p12_mean"))
    df["pos_ens"] = _nanmean_two(_get_col(df, "gfs_pos_mean"), _get_col(df, "nam_pos_mean"))
    df["poz_ens"] = _nanmean_two(_get_col(df, "gfs_poz_mean"), _get_col(df, "nam_poz_mean"))
    df["q06_ens"] = _nanmean_two(_get_col(df, "gfs_q06_mean"), _get_col(df, "nam_q06_mean"))
    df["q12_ens"] = _nanmean_two(_get_col(df, "gfs_q12_mean"), _get_col(df, "nam_q12_mean"))
    df["snw_ens"] = _nanmean_two(_get_col(df, "gfs_snw_mean"), _get_col(df, "nam_snw_mean"))
    df["t06_ens"] = _nanmean_two(_get_col(df, "gfs_t06_mean"), _get_col(df, "nam_t06_mean"))
    df["t06_1_ens"] = _nanmean_two(_get_col(df, "gfs_t06_1_mean"), _get_col(df, "nam_t06_1_mean"))
    df["t06_2_ens"] = _nanmean_two(_get_col(df, "gfs_t06_2_mean"), _get_col(df, "nam_t06_2_mean"))
    df["wdr_ens"] = _nanmean_two(_get_col(df, "gfs_wdr_mean"), _get_col(df, "nam_wdr_mean"))
    df["wsp_ens"] = _nanmean_two(_get_col(df, "gfs_wsp_mean"), _get_col(df, "nam_wsp_mean"))
    df["cig_ens_mean"] = _nanmean_two(_get_col(df, "gfs_cig_mean"), _get_col(df, "nam_cig_mean"))
    df["vis_ens_mean"] = _nanmean_two(_get_col(df, "gfs_vis_mean"), _get_col(df, "nam_vis_mean"))
    df["cig_ens_min"] = _nanmean_two(_get_col(df, "gfs_cig_min"), _get_col(df, "nam_cig_min"))
    df["vis_ens_min"] = _nanmean_two(_get_col(df, "gfs_vis_min"), _get_col(df, "nam_vis_min"))
    df["t12_ens"] = _nanmean_two(df["t06_1_ens"], df["t06_2_ens"])
    df["dep_ens"] = df["tmp_ens"] - df["dpt_ens"]
    df["wet_score"] = 0.5 * df["p12_ens"] + 100.0 * df["q12_ens"]
    return df


def add_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["resid_ens"] = df["actual_tmax_f"] - df["nx_ens"]
    df["resid_gfs"] = df["actual_tmax_f"] - _get_col(df, "gfs_n_x_mean")
    df["resid_nam"] = df["actual_tmax_f"] - _get_col(df, "nam_n_x_mean")
    df["sd_nx"] = _get_col(df, "gfs_n_x_mean") - _get_col(df, "nam_n_x_mean")
    df["ad_nx"] = df["sd_nx"].abs()
    return df


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_ensemble_columns(df)
    df = add_common_columns(df)
    df = df.sort_values(["station_id", "asof_hour", "target_date_local", "asof_utc"])
    return df


def min_periods(window: int) -> int:
    return int(np.ceil(window * 0.7))


def split_by_date(df: pd.DataFrame) -> dict:
    dates = pd.to_datetime(df["target_date_local"]).dropna()
    unique_dates = sorted(dates.unique())
    if not unique_dates:
        raise ValueError("No target_date_local values present.")

    years = sorted({pd.Timestamp(d).year for d in unique_dates})
    full_year = None
    for year in reversed(years):
        year_dates = [d for d in unique_dates if pd.Timestamp(d).year == year]
        months = {int(pd.Timestamp(d).month) for d in year_dates}
        if months == set(range(1, 13)):
            full_year = year
            break

    if full_year is not None:
        test_start = pd.Timestamp(year=full_year, month=1, day=1)
        test_end = pd.Timestamp(year=full_year, month=12, day=31)
        val_end = test_start - timedelta(days=1)
        val_start = val_end - timedelta(days=89)
        train_start = pd.Timestamp(unique_dates[0])
        train_end = val_start - timedelta(days=1)
        if train_end < train_start:
            full_year = None

    if full_year is None:
        n = len(unique_dates)
        train_end = unique_dates[int(n * 0.7) - 1]
        val_end = unique_dates[int(n * 0.85) - 1]
        train_start = unique_dates[0]
        val_start = unique_dates[int(n * 0.7)]
        test_start = unique_dates[int(n * 0.85)]
        test_end = unique_dates[-1]
    else:
        val_start = val_start
        test_start = test_start
        test_end = test_end

    date_series = pd.to_datetime(df["target_date_local"])
    train_mask = (date_series >= train_start) & (date_series <= train_end)
    val_mask = (date_series >= val_start) & (date_series <= val_end)
    test_mask = (date_series >= test_start) & (date_series <= test_end)

    return {
        "train_start": str(pd.Timestamp(train_start).date()),
        "train_end": str(pd.Timestamp(train_end).date()),
        "val_start": str(pd.Timestamp(val_start).date()),
        "val_end": str(pd.Timestamp(val_end).date()),
        "test_start": str(pd.Timestamp(test_start).date()),
        "test_end": str(pd.Timestamp(test_end).date()),
        "train_mask": train_mask.to_numpy(),
        "val_mask": val_mask.to_numpy(),
        "test_mask": test_mask.to_numpy(),
    }


def impute_features(features: pd.DataFrame, train_mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    cleaned = features.replace([np.inf, -np.inf], np.nan)
    train_means = cleaned.loc[train_mask].mean(axis=0, skipna=True)
    train_means = train_means.fillna(0.0)
    filled = cleaned.fillna(train_means)
    meta = {"method": "train_mean", "fill_values": train_means.to_dict()}
    return filled, meta


def train_model(model_name: str, seed: int) -> object:
    try:
        return models_mean.get_mean_model(model_name, seed=seed)
    except Exception as exc:
        LOGGER.warning("Falling back to ridge model: %s", exc)
        return models_mean.get_mean_model("ridge", seed=seed)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {}
    return metrics.regression_metrics(y_true, y_pred)


def per_station_metrics(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {}
    return metrics.per_station_metrics(df, y_true=y_true, y_pred=y_pred)


def write_predictions(path: Path, df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    out = df[["station_id", "target_date_local", "asof_utc"]].copy()
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out.to_csv(path, index=False)


def bootstrap_mae_delta(
    y_true: np.ndarray,
    pred_exp: np.ndarray,
    pred_base: np.ndarray,
    dates: pd.Series,
    samples: int,
    seed: int,
) -> dict:
    if len(y_true) == 0:
        return {"mean": 0.0, "p025": 0.0, "p975": 0.0, "p_lt_zero": 0.0}
    date_vals = pd.to_datetime(dates).dt.date
    unique_dates = sorted(date_vals.unique())
    indices_by_date = {d: np.where(date_vals == d)[0] for d in unique_dates}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        sample_dates = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        sample_idx = np.concatenate([indices_by_date[d] for d in sample_dates])
        mae_exp = float(np.mean(np.abs(pred_exp[sample_idx] - y_true[sample_idx])))
        mae_base = float(np.mean(np.abs(pred_base[sample_idx] - y_true[sample_idx])))
        deltas.append(mae_exp - mae_base)
    deltas_arr = np.array(deltas, dtype=float)
    return {
        "mean": float(np.mean(deltas_arr)),
        "p025": float(np.percentile(deltas_arr, 2.5)),
        "p975": float(np.percentile(deltas_arr, 97.5)),
        "p_lt_zero": float(np.mean(deltas_arr < 0.0)),
        "samples": int(samples),
    }


def add_feature(
    features: pd.DataFrame,
    formulas: list[dict],
    name: str,
    series: pd.Series,
    description: str,
) -> None:
    features[name] = series
    formulas.append({"name": name, "description": description})

def station_quantile(train_df: pd.DataFrame, column: str, q: float) -> tuple[dict, float]:
    grouped = train_df.groupby("station_id")[column].quantile(q)
    default = float(train_df[column].quantile(q)) if len(train_df) else np.nan
    return grouped.to_dict(), default


def map_station_threshold(
    df: pd.DataFrame, thresholds: dict, default: float
) -> pd.Series:
    return df["station_id"].map(thresholds).fillna(default).astype(float)


def station_standardize(
    df: pd.DataFrame, feature_cols: list[str], train_mask: np.ndarray
) -> tuple[np.ndarray, dict]:
    train_df = df.loc[train_mask, feature_cols]
    means = train_df.groupby(df.loc[train_mask, "station_id"]).mean()
    stds = train_df.groupby(df.loc[train_mask, "station_id"]).std(ddof=0)
    global_mean = train_df.mean(axis=0)
    global_std = train_df.std(axis=0, ddof=0).replace(0.0, 1.0)
    means = means.reindex(df["station_id"]).reset_index(drop=True)
    stds = stds.reindex(df["station_id"]).reset_index(drop=True)
    means = means.fillna(global_mean)
    stds = stds.fillna(global_std).replace(0.0, 1.0)
    scaled = (df[feature_cols].to_numpy(dtype=float) - means.to_numpy()) / stds.to_numpy()
    meta = {
        "per_station": True,
        "columns": feature_cols,
        "global_mean": global_mean.to_dict(),
        "global_std": global_std.to_dict(),
    }
    return scaled, meta


def global_standardize(
    df: pd.DataFrame, feature_cols: list[str], train_mask: np.ndarray
) -> tuple[np.ndarray, dict, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, feature_cols].to_numpy(dtype=float))
    scaled = scaler.transform(df[feature_cols].to_numpy(dtype=float))
    meta = {
        "columns": feature_cols,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    return scaled, meta, scaler


def rolling_mean_fallback(
    series: pd.Series,
    window: int,
    lag: int,
    group_key: pd.Series,
    fallback_key: pd.Series | None,
) -> pd.Series:
    primary = tfl.rolling_mean(
        series,
        window=window,
        min_periods=min_periods(window),
        lag=lag,
        group_key=group_key,
    )
    if fallback_key is None:
        return primary
    fallback = tfl.rolling_mean(
        series,
        window=window,
        min_periods=min_periods(window),
        lag=lag,
        group_key=fallback_key,
    )
    return primary.fillna(fallback)


def rolling_std_fallback(
    series: pd.Series,
    window: int,
    lag: int,
    group_key: pd.Series,
    fallback_key: pd.Series | None,
) -> pd.Series:
    primary = tfl.rolling_std(
        series,
        window=window,
        min_periods=min_periods(window),
        lag=lag,
        group_key=group_key,
    )
    if fallback_key is None:
        return primary
    fallback = tfl.rolling_std(
        series,
        window=window,
        min_periods=min_periods(window),
        lag=lag,
        group_key=fallback_key,
    )
    return primary.fillna(fallback)


def rolling_quantile_fallback(
    series: pd.Series,
    window: int,
    lag: int,
    q: float,
    group_key: pd.Series,
    fallback_key: pd.Series | None,
) -> pd.Series:
    primary = tfl.rolling_quantile(
        series,
        window=window,
        min_periods=min_periods(window),
        lag=lag,
        q=q,
        group_key=group_key,
    )
    if fallback_key is None:
        return primary
    fallback = tfl.rolling_quantile(
        series,
        window=window,
        min_periods=min_periods(window),
        lag=lag,
        q=q,
        group_key=fallback_key,
    )
    return primary.fillna(fallback)


def conditional_rolling_mean(
    values: pd.Series,
    indicator: pd.Series,
    window: int,
    lag: int,
    group_key: pd.Series,
    min_needed: int,
) -> pd.Series:
    if not isinstance(values, pd.Series):
        values = pd.Series(values, index=indicator.index)
    if not isinstance(indicator, pd.Series):
        indicator = pd.Series(indicator, index=values.index)
    mean = tfl.rolling_conditional_mean(
        values,
        indicator,
        window=window,
        min_periods=1,
        lag=lag,
        group_key=group_key,
    )
    count = tfl.rolling_sum(
        indicator.astype(float),
        window=window,
        min_periods=1,
        lag=lag,
        group_key=group_key,
    )
    return mean.where(count >= min_needed)


def safe_divide(numer: np.ndarray, denom: np.ndarray, default: float = np.nan) -> np.ndarray:
    out = np.full_like(numer, default, dtype=float)
    mask = np.isfinite(numer) & np.isfinite(denom) & (denom != 0)
    out[mask] = numer[mask] / denom[mask]
    return out


def rolling_ar1_phi(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    x = values[:-1]
    y = values[1:]
    denom = np.sum(x * x)
    if denom == 0:
        return 0.0
    return float(np.sum(x * y) / denom)


def compute_ewa_weights(
    df: pd.DataFrame,
    group_key: pd.Series,
    lag: int,
    eta: float,
    weight_floor: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gfs = _get_col(df, "gfs_n_x_mean").to_numpy(dtype=float)
    nam = _get_col(df, "nam_n_x_mean").to_numpy(dtype=float)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    groups = group_key.to_numpy()
    n = len(df)
    w_gfs = np.full(n, np.nan, dtype=float)
    w_nam = np.full(n, np.nan, dtype=float)
    nx_ewa = np.full(n, np.nan, dtype=float)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        wg = 0.5
        wn = 0.5
        for pos, row_idx in enumerate(idx):
            if pos >= lag:
                past_idx = idx[pos - lag]
                loss_gfs = np.abs(y[past_idx] - gfs[past_idx]) if np.isfinite(gfs[past_idx]) else 0.0
                loss_nam = np.abs(y[past_idx] - nam[past_idx]) if np.isfinite(nam[past_idx]) else 0.0
                wg = wg * np.exp(-eta * loss_gfs)
                wn = wn * np.exp(-eta * loss_nam)
                wg = max(wg, weight_floor)
                wn = max(wn, weight_floor)
                total = wg + wn
                if total > 0:
                    wg /= total
                    wn /= total
            w_gfs[row_idx] = wg
            w_nam[row_idx] = wn
            if np.isfinite(gfs[row_idx]) and np.isfinite(nam[row_idx]):
                nx_ewa[row_idx] = wg * gfs[row_idx] + wn * nam[row_idx]
            elif np.isfinite(gfs[row_idx]):
                nx_ewa[row_idx] = gfs[row_idx]
            elif np.isfinite(nam[row_idx]):
                nx_ewa[row_idx] = nam[row_idx]
    return w_gfs, w_nam, nx_ewa


def compute_cusum_features(
    df: pd.DataFrame,
    group_key: pd.Series,
    lag: int,
    k: float,
    h: float,
) -> dict:
    resid = df["resid_ens"].to_numpy(dtype=float)
    groups = group_key.to_numpy()
    n = len(df)
    cpos = np.full(n, np.nan, dtype=float)
    cneg = np.full(n, np.nan, dtype=float)
    alarm_pos = np.full(n, 0.0, dtype=float)
    alarm_neg = np.full(n, 0.0, dtype=float)
    tsa_pos = np.full(n, np.nan, dtype=float)
    tsa_neg = np.full(n, np.nan, dtype=float)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        pos_score = 0.0
        neg_score = 0.0
        since_pos = 0.0
        since_neg = 0.0
        for pos, row_idx in enumerate(idx):
            if pos >= lag:
                past_idx = idx[pos - lag]
                if np.isfinite(resid[past_idx]):
                    pos_score = max(0.0, pos_score + (resid[past_idx] - k))
                    neg_score = max(0.0, neg_score + (-resid[past_idx] - k))
            if pos_score > h:
                alarm_pos[row_idx] = 1.0
                since_pos = 0.0
            else:
                since_pos += 1.0
            if neg_score > h:
                alarm_neg[row_idx] = 1.0
                since_neg = 0.0
            else:
                since_neg += 1.0
            cpos[row_idx] = pos_score
            cneg[row_idx] = neg_score
            tsa_pos[row_idx] = since_pos
            tsa_neg[row_idx] = since_neg
    return {
        "cpos": cpos,
        "cneg": cneg,
        "alarm_pos": alarm_pos,
        "alarm_neg": alarm_neg,
        "tsa_pos": tsa_pos,
        "tsa_neg": tsa_neg,
    }


def compute_kalman_features(
    df: pd.DataFrame,
    group_key: pd.Series,
    lag: int,
    q: float,
    r: float,
    base_pred: np.ndarray,
) -> dict:
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    groups = group_key.to_numpy()
    n = len(df)
    b_hat = np.full(n, np.nan, dtype=float)
    p_var = np.full(n, np.nan, dtype=float)
    k_gain = np.full(n, np.nan, dtype=float)
    p_kf = np.full(n, np.nan, dtype=float)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        b = 0.0
        p = r
        k_last = 0.0
        for pos, row_idx in enumerate(idx):
            if pos >= lag:
                past_idx = idx[pos - lag]
                if np.isfinite(base_pred[past_idx]) and np.isfinite(y[past_idx]):
                    p = p + q
                    k = p / (p + r)
                    resid = (y[past_idx] - base_pred[past_idx]) - b
                    b = b + k * resid
                    p = (1.0 - k) * p
                    k_last = k
            b_hat[row_idx] = b
            p_var[row_idx] = p
            k_gain[row_idx] = k_last
            if np.isfinite(base_pred[row_idx]):
                p_kf[row_idx] = base_pred[row_idx] + b
    return {"b_hat": b_hat, "p_var": p_var, "k_gain": k_gain, "p_kf": p_kf}


def compute_isotonic_features(
    df: pd.DataFrame,
    group_key: pd.Series,
    lag: int,
    horizon_days: int,
    min_pairs: int,
    p: np.ndarray,
) -> dict:
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    dates = pd.to_datetime(df["target_date_local"]).to_numpy(dtype="datetime64[D]")
    groups = group_key.to_numpy()
    n = len(df)
    p_iso = np.full(n, np.nan, dtype=float)
    slope_local = np.full(n, np.nan, dtype=float)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        pairs = []
        for pos, row_idx in enumerate(idx):
            if pos >= lag:
                past_idx = idx[pos - lag]
                if np.isfinite(p[past_idx]) and np.isfinite(y[past_idx]):
                    pairs.append((dates[past_idx], p[past_idx], y[past_idx]))
            cutoff = dates[row_idx] - np.timedelta64(horizon_days, "D")
            if pairs:
                pairs = [item for item in pairs if item[0] >= cutoff]
            if len(pairs) >= min_pairs and np.isfinite(p[row_idx]):
                p_vals = np.array([item[1] for item in pairs], dtype=float)
                y_vals = np.array([item[2] for item in pairs], dtype=float)
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(p_vals, y_vals)
                p_iso[row_idx] = float(ir.predict([p[row_idx]])[0])
                p_hi = float(ir.predict([p[row_idx] + 1.0])[0])
                p_lo = float(ir.predict([p[row_idx] - 1.0])[0])
                slope_local[row_idx] = (p_hi - p_lo) / 2.0
            else:
                if np.isfinite(p[row_idx]):
                    p_iso[row_idx] = p[row_idx]
    return {"p_iso": p_iso, "slope_local": slope_local}


def expanding_time_splits(df: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_dates = sorted(pd.to_datetime(df["target_date_local"]).unique())
    if len(unique_dates) < n_splits + 2:
        return []
    blocks = np.array_split(unique_dates, n_splits + 1)
    splits = []
    for idx in range(1, len(blocks)):
        train_dates = np.concatenate(blocks[:idx])
        val_dates = blocks[idx]
        train_idx = df.index[df["target_date_local"].isin(train_dates)].to_numpy()
        val_idx = df.index[df["target_date_local"].isin(val_dates)].to_numpy()
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))
    return splits


def compute_knn_stats(
    features: np.ndarray,
    values: np.ndarray,
    group_key: pd.Series,
    lag: int,
    k: int,
    min_dims: int,
    fallback: np.ndarray | None = None,
) -> dict:
    n = len(values)
    mean = np.full(n, np.nan, dtype=float)
    median = np.full(n, np.nan, dtype=float)
    std = np.full(n, np.nan, dtype=float)
    wmean = np.full(n, np.nan, dtype=float)
    q25 = np.full(n, np.nan, dtype=float)
    q75 = np.full(n, np.nan, dtype=float)
    count = np.full(n, 0.0, dtype=float)
    mean_dist = np.full(n, np.nan, dtype=float)

    groups = group_key.to_numpy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        feats = features[idx]
        vals = values[idx]
        for pos, row_idx in enumerate(idx):
            if pos < lag:
                continue
            current = feats[pos]
            if not np.isfinite(current).any():
                continue
            cand_positions = np.arange(0, pos - lag + 1)
            if cand_positions.size == 0:
                continue
            dists = []
            cand_vals = []
            for cand_pos in cand_positions:
                cand = feats[cand_pos]
                mask = np.isfinite(current) & np.isfinite(cand)
                if mask.sum() < min_dims:
                    continue
                val = vals[cand_pos]
                if not np.isfinite(val):
                    continue
                dist = float(np.linalg.norm(current[mask] - cand[mask]))
                dists.append(dist)
                cand_vals.append(val)
            if not dists:
                continue
            order = np.argsort(dists)[:k]
            selected_vals = np.array([cand_vals[i] for i in order], dtype=float)
            selected_dists = np.array([dists[i] for i in order], dtype=float)
            if selected_vals.size == 0:
                continue
            weights = 1.0 / (selected_dists + EPS)
            sum_w = np.sum(weights)
            mean[row_idx] = float(np.mean(selected_vals))
            median[row_idx] = float(np.median(selected_vals))
            std[row_idx] = float(np.std(selected_vals, ddof=0))
            q25[row_idx] = float(np.quantile(selected_vals, 0.25))
            q75[row_idx] = float(np.quantile(selected_vals, 0.75))
            wmean[row_idx] = float(np.sum(selected_vals * weights) / sum_w) if sum_w > 0 else mean[row_idx]
            count[row_idx] = float(selected_vals.size)
            mean_dist[row_idx] = float(np.mean(selected_dists))

    if fallback is not None:
        mean = np.where(np.isfinite(mean), mean, fallback)
        median = np.where(np.isfinite(median), median, fallback)
        std = np.where(np.isfinite(std), std, np.nan)
        q25 = np.where(np.isfinite(q25), q25, fallback)
        q75 = np.where(np.isfinite(q75), q75, fallback)
        wmean = np.where(np.isfinite(wmean), wmean, fallback)
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "wmean": wmean,
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "count": count,
        "mean_dist": mean_dist,
    }


def exp_e01(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    r_ens = df["resid_ens"]
    nx_ens = df["nx_ens"]
    for w in [7, 15, 30, 60]:
        bias = rolling_mean_fallback(
            r_ens, w, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
        )
        mae = rolling_mean_fallback(
            r_ens.abs(), w, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
        )
        rmse = np.sqrt(
            rolling_mean_fallback(
                r_ens**2,
                w,
                ctx.truth_lag,
                ctx.group_station_asof,
                ctx.group_station,
            )
        )
        add_feature(features, formulas, f"bias_ens_{w}", bias, "rolling mean residual")
        add_feature(features, formulas, f"mae_ens_{w}", mae, "rolling mean abs residual")
        add_feature(features, formulas, f"rmse_ens_{w}", rmse, "rolling rmse residual")
        add_feature(
            features,
            formulas,
            f"nx_ens_bc_{w}",
            nx_ens + bias,
            "nx_ens + rolling bias",
        )
    lagged = r_ens.groupby(ctx.group_station_asof).shift(ctx.truth_lag)
    add_feature(features, formulas, "r_ens_lag2", lagged, "residual lagged by L")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e02(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    r_gfs = df["resid_gfs"]
    r_nam = df["resid_nam"]
    r_ens = df["resid_ens"]
    for h in [7, 14, 30]:
        bias_gfs = tfl.ewm_mean(
            r_gfs,
            halflife=h,
            min_periods=10,
            lag=ctx.truth_lag,
            group_key=ctx.group_station_asof,
        ).fillna(
            tfl.ewm_mean(
                r_gfs,
                halflife=h,
                min_periods=10,
                lag=ctx.truth_lag,
                group_key=ctx.group_station,
            )
        )
        bias_nam = tfl.ewm_mean(
            r_nam,
            halflife=h,
            min_periods=10,
            lag=ctx.truth_lag,
            group_key=ctx.group_station_asof,
        ).fillna(
            tfl.ewm_mean(
                r_nam,
                halflife=h,
                min_periods=10,
                lag=ctx.truth_lag,
                group_key=ctx.group_station,
            )
        )
        bias_ens = tfl.ewm_mean(
            r_ens,
            halflife=h,
            min_periods=10,
            lag=ctx.truth_lag,
            group_key=ctx.group_station_asof,
        ).fillna(
            tfl.ewm_mean(
                r_ens,
                halflife=h,
                min_periods=10,
                lag=ctx.truth_lag,
                group_key=ctx.group_station,
            )
        )
        mae_gfs = tfl.ewm_mean(
            r_gfs.abs(),
            halflife=h,
            min_periods=10,
            lag=ctx.truth_lag,
            group_key=ctx.group_station_asof,
        ).fillna(
            tfl.ewm_mean(
                r_gfs.abs(),
                halflife=h,
                min_periods=10,
                lag=ctx.truth_lag,
                group_key=ctx.group_station,
            )
        )
        mae_nam = tfl.ewm_mean(
            r_nam.abs(),
            halflife=h,
            min_periods=10,
            lag=ctx.truth_lag,
            group_key=ctx.group_station_asof,
        ).fillna(
            tfl.ewm_mean(
                r_nam.abs(),
                halflife=h,
                min_periods=10,
                lag=ctx.truth_lag,
                group_key=ctx.group_station,
            )
        )
        mae_ens = tfl.ewm_mean(
            r_ens.abs(),
            halflife=h,
            min_periods=10,
            lag=ctx.truth_lag,
            group_key=ctx.group_station_asof,
        ).fillna(
            tfl.ewm_mean(
                r_ens.abs(),
                halflife=h,
                min_periods=10,
                lag=ctx.truth_lag,
                group_key=ctx.group_station,
            )
        )
        add_feature(features, formulas, f"ewmbias_gfs_{h}", bias_gfs, "ewm bias gfs")
        add_feature(features, formulas, f"ewmbias_nam_{h}", bias_nam, "ewm bias nam")
        add_feature(features, formulas, f"ewmbias_ens_{h}", bias_ens, "ewm bias ens")
        add_feature(features, formulas, f"ewmae_gfs_{h}", mae_gfs, "ewm mae gfs")
        add_feature(features, formulas, f"ewmae_nam_{h}", mae_nam, "ewm mae nam")
        add_feature(features, formulas, f"ewmae_ens_{h}", mae_ens, "ewm mae ens")
        add_feature(
            features,
            formulas,
            f"nx_gfs_bc_{h}",
            _get_col(df, "gfs_n_x_mean") + bias_gfs,
            "gfs + ewmbias",
        )
        add_feature(
            features,
            formulas,
            f"nx_nam_bc_{h}",
            _get_col(df, "nam_n_x_mean") + bias_nam,
            "nam + ewmbias",
        )
        add_feature(
            features,
            formulas,
            f"nx_ens_bc_{h}",
            df["nx_ens"] + bias_ens,
            "ens + ewmbias",
        )
        skill_ratio = safe_divide(mae_gfs.to_numpy(), mae_nam.to_numpy() + EPS, default=np.nan)
        add_feature(
            features,
            formulas,
            f"skill_ratio_{h}",
            pd.Series(skill_ratio, index=df.index),
            "ewmae_gfs / ewmae_nam",
        )
        add_feature(
            features,
            formulas,
            f"log_skill_ratio_{h}",
            pd.Series(np.log(np.clip(skill_ratio, EPS, None)), index=df.index),
            "log skill ratio",
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e03(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    r_gfs = df["resid_gfs"].abs()
    r_nam = df["resid_nam"].abs()
    for w in [15, 60]:
        mae_gfs = rolling_mean_fallback(
            r_gfs, w, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
        )
        mae_nam = rolling_mean_fallback(
            r_nam, w, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
        )
        w_gfs = 1.0 / (mae_gfs + 0.25)
        w_nam = 1.0 / (mae_nam + 0.25)
        w_sum = w_gfs + w_nam
        w_gfs = safe_divide(w_gfs.to_numpy(), w_sum.to_numpy(), default=0.5)
        w_nam = safe_divide(w_nam.to_numpy(), w_sum.to_numpy(), default=0.5)
        w_gfs = pd.Series(w_gfs, index=df.index)
        w_nam = pd.Series(w_nam, index=df.index)
        nx_wt = w_gfs * _get_col(df, "gfs_n_x_mean") + w_nam * _get_col(df, "nam_n_x_mean")
        add_feature(features, formulas, f"mae_gfs_{w}", mae_gfs, "rolling mae gfs")
        add_feature(features, formulas, f"mae_nam_{w}", mae_nam, "rolling mae nam")
        add_feature(features, formulas, f"w_gfs_{w}", w_gfs, "inverse mae weight gfs")
        add_feature(features, formulas, f"w_nam_{w}", w_nam, "inverse mae weight nam")
        add_feature(features, formulas, f"nx_wt_{w}", nx_wt, "weighted ensemble")
        add_feature(
            features,
            formulas,
            f"delta_mae_{w}",
            mae_gfs - mae_nam,
            "mae_gfs - mae_nam",
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e04(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    etas = [0.02, 0.05, 0.1, 0.2]
    best_eta = etas[0]
    best_mae = float("inf")
    for eta in etas:
        _, _, nx_ewa = compute_ewa_weights(
            df, ctx.group_station_asof, ctx.truth_lag, eta
        )
        val_pred = nx_ewa[ctx.val_mask]
        val_true = df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
        if len(val_true) == 0:
            continue
        mae = float(np.mean(np.abs(val_pred - val_true)))
        if mae < best_mae:
            best_mae = mae
            best_eta = eta
    w_gfs, w_nam, nx_ewa = compute_ewa_weights(
        df, ctx.group_station_asof, ctx.truth_lag, best_eta
    )
    add_feature(features, formulas, "w_gfs", pd.Series(w_gfs, index=df.index), "ewa weight gfs")
    add_feature(features, formulas, "w_nam", pd.Series(w_nam, index=df.index), "ewa weight nam")
    logit_w = np.log((w_gfs + EPS) / (w_nam + EPS))
    add_feature(features, formulas, "logit_w", pd.Series(logit_w, index=df.index), "logit weight")
    add_feature(features, formulas, "nx_ewa", pd.Series(nx_ewa, index=df.index), "ewa ensemble")
    train_fitted.append({"eta": best_eta, "candidates": etas, "val_mae": best_mae})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e05(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    d0 = df["ad_nx"]
    dbar = rolling_mean_fallback(d0, 15, 1, ctx.group_station_asof, ctx.group_station)
    thresholds, default_thr = station_quantile(df.loc[ctx.train_mask], "ad_nx", 0.75)
    thr = map_station_threshold(df, thresholds, default_thr)
    regime = (dbar > thr).astype(float)
    mae_gfs = rolling_mean_fallback(
        df["resid_gfs"].abs(), 30, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    mae_nam = rolling_mean_fallback(
        df["resid_nam"].abs(), 30, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    nx_gate = np.where(
        regime.to_numpy() > 0.5,
        np.where(mae_gfs.to_numpy() < mae_nam.to_numpy(), _get_col(df, "gfs_n_x_mean"), _get_col(df, "nam_n_x_mean")),
        df["nx_ens"].to_numpy(),
    )
    add_feature(features, formulas, "dbar_15", dbar, "rolling mean disagreement")
    add_feature(features, formulas, "regime_flag", regime, "disagreement regime")
    add_feature(features, formulas, "nx_gate", pd.Series(nx_gate, index=df.index), "gated predictor")
    add_feature(features, formulas, "delta_mae_30", mae_gfs - mae_nam, "mae delta")
    train_fitted.append({"thr_q75": thresholds, "thr_default": default_thr})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e06(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    r_ens = df["resid_ens"]
    for w in [15, 60]:
        q25 = rolling_quantile_fallback(
            r_ens, w, ctx.truth_lag, 0.25, ctx.group_station_asof, ctx.group_station
        )
        q50 = rolling_quantile_fallback(
            r_ens, w, ctx.truth_lag, 0.50, ctx.group_station_asof, ctx.group_station
        )
        q75 = rolling_quantile_fallback(
            r_ens, w, ctx.truth_lag, 0.75, ctx.group_station_asof, ctx.group_station
        )
        iqr = q75 - q25
        skew = q75 + q25 - 2.0 * q50
        add_feature(features, formulas, f"q25_{w}", q25, "rolling q25")
        add_feature(features, formulas, f"q50_{w}", q50, "rolling q50")
        add_feature(features, formulas, f"q75_{w}", q75, "rolling q75")
        add_feature(features, formulas, f"iqr_{w}", iqr, "rolling iqr")
        add_feature(features, formulas, f"skewq_{w}", skew, "quantile skew")
        add_feature(
            features,
            formulas,
            f"nx_robust_{w}",
            df["nx_ens"] + q50,
            "nx_ens + q50 bias",
        )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e07(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    sin_doy = df["sin_doy"].to_numpy(dtype=float)
    cos_doy = df["cos_doy"].to_numpy(dtype=float)
    sin_2doy = np.sin(4 * np.pi * df["day_of_year"] / 365.25)
    cos_2doy = np.cos(4 * np.pi * df["day_of_year"] / 365.25)
    r_ens = df["resid_ens"].to_numpy(dtype=float)
    n = len(df)
    seas_bias = np.zeros(n, dtype=float)

    def fit_coeffs(mask: np.ndarray) -> tuple[float, np.ndarray] | None:
        if mask.sum() < 10:
            return None
        X = np.column_stack([sin_doy[mask], cos_doy[mask], sin_2doy[mask], cos_2doy[mask]])
        y = r_ens[mask]
        model = Ridge(alpha=1.0, fit_intercept=True)
        model.fit(X, y)
        return float(model.intercept_), model.coef_.astype(float)

    station_coeffs: dict[str, tuple[float, np.ndarray]] = {}
    for station in df["station_id"].unique():
        mask = ctx.train_mask & (df["station_id"] == station)
        coeffs = fit_coeffs(mask)
        if coeffs:
            station_coeffs[str(station)] = coeffs

    group_keys = ctx.group_station_asof.to_numpy()
    for g in np.unique(group_keys):
        mask = ctx.train_mask & (ctx.group_station_asof == g)
        coeffs = fit_coeffs(mask)
        if coeffs is None:
            station = str(df.loc[mask, "station_id"].iloc[0]) if mask.any() else str(g).split("_")[0]
            coeffs = station_coeffs.get(station)
        if coeffs is None:
            continue
        intercept, coef = coeffs
        idx = np.where(ctx.group_station_asof == g)[0]
        Xg = np.column_stack([sin_doy[idx], cos_doy[idx], sin_2doy[idx], cos_2doy[idx]])
        seas_bias[idx] = intercept + Xg @ coef

    seas_bias_series = pd.Series(seas_bias, index=df.index)
    drift = rolling_mean_fallback(
        df["resid_ens"], 30, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    ) - seas_bias_series
    add_feature(features, formulas, "seas_bias", seas_bias_series, "seasonal bias")
    add_feature(features, formulas, "drift_30", drift, "rolling drift")
    add_feature(
        features,
        formulas,
        "nx_seas_bc",
        df["nx_ens"] + seas_bias_series,
        "nx_ens + seasonal bias",
    )
    add_feature(
        features,
        formulas,
        "nx_seas_drift_bc",
        df["nx_ens"] + seas_bias_series + drift,
        "nx_ens + seasonal bias + drift",
    )
    train_fitted.append({"station_coeffs": list(station_coeffs.keys())})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e08(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    r_ens = df["resid_ens"]
    bias_a = rolling_mean_fallback(r_ens, 30, ctx.truth_lag, ctx.group_station, None)
    mae_a = rolling_mean_fallback(r_ens.abs(), 30, ctx.truth_lag, ctx.group_station, None)
    bias_b = rolling_mean_fallback(r_ens, 30, ctx.truth_lag, ctx.group_station_asof, ctx.group_station)
    mae_b = rolling_mean_fallback(r_ens.abs(), 30, ctx.truth_lag, ctx.group_station_asof, ctx.group_station)
    dbias = bias_b - bias_a
    dmae = mae_b - mae_a
    add_feature(features, formulas, "biasA_30", bias_a, "station bias")
    add_feature(features, formulas, "maeA_30", mae_a, "station mae")
    add_feature(features, formulas, "biasB_30", bias_b, "station-hour bias")
    add_feature(features, formulas, "maeB_30", mae_b, "station-hour mae")
    add_feature(features, formulas, "dbias_30", dbias, "biasB - biasA")
    add_feature(features, formulas, "dmae_30", dmae, "maeB - maeA")
    add_feature(
        features,
        formulas,
        "nx_bcA",
        df["nx_ens"] + bias_a,
        "nx_ens + station bias",
    )
    add_feature(
        features,
        formulas,
        "nx_bcB",
        df["nx_ens"] + bias_b,
        "nx_ens + station-hour bias",
    )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e09(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    series_list = [
        ("gfs_n_x_mean", _get_col(df, "gfs_n_x_mean")),
        ("nam_n_x_mean", _get_col(df, "nam_n_x_mean")),
        ("nx_ens", df["nx_ens"]),
        ("gfs_tmp_mean", _get_col(df, "gfs_tmp_mean")),
        ("nam_tmp_mean", _get_col(df, "nam_tmp_mean")),
        ("tmp_ens", df["tmp_ens"]),
    ]
    for name, series in series_list:
        delta = series.groupby(ctx.group_station_asof).diff()
        for w in [7, 30, 60]:
            vol_level = tfl.rolling_std(
                series,
                window=w,
                min_periods=min_periods(w),
                lag=1,
                group_key=ctx.group_station_asof,
            )
            vol_change = tfl.rolling_std(
                delta,
                window=w,
                min_periods=min_periods(w),
                lag=1,
                group_key=ctx.group_station_asof,
            )
            add_feature(
                features,
                formulas,
                f"{name}_vol_level_{w}",
                vol_level,
                "rolling std level",
            )
            add_feature(
                features,
                formulas,
                f"{name}_vol_change_{w}",
                vol_change,
                "rolling std change",
            )
        if name in {"nx_ens", "tmp_ens"}:
            vol_change_7 = features[f"{name}_vol_change_7"]
            vol_change_30 = features[f"{name}_vol_change_30"]
            vol_ratio = safe_divide(
                vol_change_7.to_numpy(), vol_change_30.to_numpy() + EPS, default=np.nan
            )
            add_feature(
                features,
                formulas,
                f"{name}_vol_ratio",
                pd.Series(vol_ratio, index=df.index),
                "vol_change_7 / vol_change_30",
            )
            mean_30 = tfl.rolling_mean(
                series,
                window=30,
                min_periods=min_periods(30),
                lag=1,
                group_key=ctx.group_station_asof,
            )
            std_30 = tfl.rolling_std(
                series,
                window=30,
                min_periods=min_periods(30),
                lag=1,
                group_key=ctx.group_station_asof,
            )
            level_anom = safe_divide(
                (series - mean_30).to_numpy(), std_30.to_numpy() + EPS, default=np.nan
            )
            add_feature(
                features,
                formulas,
                f"{name}_level_anom",
                pd.Series(level_anom, index=df.index),
                "level anomaly z",
            )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e10(ctx: ExperimentContext) -> DerivedFeatureSet:
    return DerivedFeatureSet(features=pd.DataFrame(index=ctx.df.index), formulas=[], train_fitted=[])


def exp_e11(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    series_list = [
        ("gfs_n_x_mean", _get_col(df, "gfs_n_x_mean")),
        ("nam_n_x_mean", _get_col(df, "nam_n_x_mean")),
        ("nx_ens", df["nx_ens"]),
        ("gfs_tmp_mean", _get_col(df, "gfs_tmp_mean")),
        ("nam_tmp_mean", _get_col(df, "nam_tmp_mean")),
        ("tmp_ens", df["tmp_ens"]),
    ]
    for name, series in series_list:
        for w in [7, 30, 60]:
            mu = tfl.rolling_mean(
                series,
                window=w,
                min_periods=min_periods(w),
                lag=1,
                group_key=ctx.group_station_asof,
            )
            sigma = tfl.rolling_std(
                series,
                window=w,
                min_periods=min_periods(w),
                lag=1,
                group_key=ctx.group_station_asof,
            )
            z = safe_divide((series - mu).to_numpy(), sigma.to_numpy() + EPS, default=np.nan)
            add_feature(features, formulas, f"{name}_z_{w}", pd.Series(z, index=df.index), "z-score")
        if name in {"nx_ens", "tmp_ens"}:
            z_short = features[f"{name}_z_7"]
            z_long = features[f"{name}_z_60"]
            add_feature(
                features,
                formulas,
                f"{name}_z_short_minus_long",
                z_short - z_long,
                "z7 - z60",
            )
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e12(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []

    def r2(values: np.ndarray) -> float:
        n = len(values)
        if n < 2:
            return np.nan
        x = np.arange(n, dtype=float)
        x_mean = np.mean(x)
        y_mean = np.mean(values)
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return 0.0
        slope = np.sum((x - x_mean) * (values - y_mean)) / denom
        intercept = y_mean - slope * x_mean
        y_pred = slope * x + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - y_mean) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)

    for name, series in [("nx_ens", df["nx_ens"]), ("tmp_ens", df["tmp_ens"])]:
        slope_7 = tfl.rolling_slope(
            series,
            window=7,
            min_periods=min_periods(7),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        slope_30 = tfl.rolling_slope(
            series,
            window=30,
            min_periods=min_periods(30),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        accel = slope_7 - slope_30
        r2_7 = tfl.rolling_apply(
            series,
            window=7,
            min_periods=min_periods(7),
            lag=1,
            func=r2,
            group_key=ctx.group_station_asof,
        )
        add_feature(features, formulas, f"{name}_slope_7", slope_7, "slope 7")
        add_feature(features, formulas, f"{name}_slope_30", slope_30, "slope 30")
        add_feature(features, formulas, f"{name}_accel", accel, "slope7 - slope30")
        add_feature(features, formulas, f"{name}_r2_7", r2_7, "r2 of slope 7")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e13(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    nx = df["nx_ens"]
    delta = nx.groupby(ctx.group_station_asof).diff()

    def sign_changes(values: np.ndarray) -> float:
        if len(values) < 2:
            return np.nan
        signs = np.sign(values)
        return float(np.sum(signs[1:] != signs[:-1]))

    sc = tfl.rolling_apply(
        delta,
        window=15,
        min_periods=min_periods(15),
        lag=1,
        func=sign_changes,
        group_key=ctx.group_station_asof,
    )
    sum_abs = tfl.rolling_sum(
        delta.abs(),
        window=15,
        min_periods=min_periods(15),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    nx_shift_1 = nx.groupby(ctx.group_station_asof).shift(1)
    nx_shift_w = nx.groupby(ctx.group_station_asof).shift(16)
    net_change = (nx_shift_1 - nx_shift_w).abs()
    choppiness = safe_divide(sum_abs.to_numpy(), net_change.to_numpy() + 1e-3, default=np.nan)
    last_delta = delta.groupby(ctx.group_station_asof).shift(1)
    last_sign = np.sign(last_delta)
    add_feature(features, formulas, "sign_changes", sc, "sign changes")
    add_feature(features, formulas, "choppiness", pd.Series(choppiness, index=df.index), "choppiness")
    add_feature(features, formulas, "last_delta", last_delta, "delta lag1")
    add_feature(features, formulas, "last_delta_sign", last_sign, "sign of delta")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e14(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    nx = df["nx_ens"]
    delta = nx.groupby(ctx.group_station_asof).diff()
    vol7 = tfl.rolling_std(
        delta,
        window=7,
        min_periods=min_periods(7),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    vol30 = tfl.rolling_std(
        delta,
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    vov30 = tfl.rolling_std(
        vol7,
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    vol_ratio = safe_divide(vol7.to_numpy(), vol30.to_numpy() + EPS, default=np.nan)
    vov_mean = tfl.rolling_mean(
        vov30,
        window=60,
        min_periods=min_periods(60),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    vov_std = tfl.rolling_std(
        vov30,
        window=60,
        min_periods=min_periods(60),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    vov_z = safe_divide((vov30 - vov_mean).to_numpy(), vov_std.to_numpy() + EPS, default=np.nan)
    add_feature(features, formulas, "vol7", vol7, "vol7")
    add_feature(features, formulas, "vol30", vol30, "vol30")
    add_feature(features, formulas, "vov30", vov30, "volatility of volatility")
    add_feature(features, formulas, "vol_ratio", pd.Series(vol_ratio, index=df.index), "vol7/vol30")
    add_feature(features, formulas, "vov_z", pd.Series(vov_z, index=df.index), "vov z")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e15(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []

    def range_and_skew(prefix: str, var: str) -> tuple[pd.Series, pd.Series]:
        vmin = _get_col(df, f"{prefix}_{var}_min")
        vmax = _get_col(df, f"{prefix}_{var}_max")
        vmean = _get_col(df, f"{prefix}_{var}_mean")
        vmed = _get_col(df, f"{prefix}_{var}_median")
        return vmax - vmin, vmean - vmed

    for var in ["tmp", "n_x"]:
        range_gfs, skew_gfs = range_and_skew("gfs", var)
        range_nam, skew_nam = range_and_skew("nam", var)
        range_ens = _nanmean_two(range_gfs, range_nam)
        skew_ens = _nanmean_two(skew_gfs, skew_nam)
        add_feature(features, formulas, f"{var}_range_ens", range_ens, "range ens")
        add_feature(features, formulas, f"{var}_skew_ens", skew_ens, "skew ens")
        mean_range_7 = tfl.rolling_mean(
            range_ens,
            window=7,
            min_periods=min_periods(7),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        mean_range_30 = tfl.rolling_mean(
            range_ens,
            window=30,
            min_periods=min_periods(30),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        slope_30 = tfl.rolling_slope(
            range_ens,
            window=30,
            min_periods=min_periods(30),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        anom_range = range_ens - mean_range_30
        add_feature(features, formulas, f"{var}_mean_range_7", mean_range_7, "mean range 7")
        add_feature(features, formulas, f"{var}_mean_range_30", mean_range_30, "mean range 30")
        add_feature(features, formulas, f"{var}_anom_range", anom_range, "range anomaly")
        add_feature(features, formulas, f"{var}_slope_range_30", slope_30, "range slope 30")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e16(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    sd = df["sd_nx"]
    ad = df["ad_nx"]
    for w in [7, 30, 60]:
        mean_sd = tfl.rolling_mean(
            sd,
            window=w,
            min_periods=min_periods(w),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        std_sd = tfl.rolling_std(
            sd,
            window=w,
            min_periods=min_periods(w),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        mean_ad = tfl.rolling_mean(
            ad,
            window=w,
            min_periods=min_periods(w),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        add_feature(features, formulas, f"mean_sd_{w}", mean_sd, "mean sd")
        add_feature(features, formulas, f"std_sd_{w}", std_sd, "std sd")
        add_feature(features, formulas, f"mean_ad_{w}", mean_ad, "mean ad")
    slope_ad_30 = tfl.rolling_slope(
        ad,
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    frac_pos = tfl.rolling_mean(
        (sd > 0).astype(float),
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    runlen = tfl.streak_length(
        np.sign(sd),
        lag=1,
        cap=60,
        group_key=ctx.group_station_asof,
    )
    add_feature(features, formulas, "slope_ad_30", slope_ad_30, "slope ad 30")
    add_feature(features, formulas, "frac_sd_pos_30", frac_pos, "frac sd positive")
    add_feature(features, formulas, "runlen_sd_sign", runlen, "run length sign")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e17(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    vars_list = ["n_x_mean", "tmp_mean", "dpt_mean", "wsp_mean", "p12_mean", "q12_mean"]
    n = len(df)
    ndv_matrix = np.full((n, len(vars_list)), np.nan, dtype=float)
    components_used = np.zeros(n, dtype=float)
    scales_meta = {}
    for j, var in enumerate(vars_list):
        gfs = _get_col(df, f"gfs_{var}")
        nam = _get_col(df, f"nam_{var}")
        dv = gfs - nam
        dv_name = f"dv_{var}"
        df[dv_name] = dv
        train_vals = df.loc[ctx.train_mask, ["station_id", dv_name]].copy()
        scales = train_vals.groupby("station_id")[dv_name].std(ddof=0)
        default_scale = float(train_vals[dv_name].std(ddof=0)) if len(train_vals) else 1.0
        scales_meta[var] = {"scales": scales.to_dict(), "default": default_scale}
        scale_series = df["station_id"].map(scales).fillna(default_scale).replace(0.0, 1.0)
        ndv_matrix[:, j] = (dv / scale_series).to_numpy(dtype=float)
    valid = np.isfinite(ndv_matrix)
    components_used = valid.sum(axis=1).astype(float)
    l2 = np.sqrt(np.nansum(ndv_matrix**2, axis=1))
    l1 = np.nansum(np.abs(ndv_matrix), axis=1)
    l2[components_used == 0] = np.nan
    l1[components_used == 0] = np.nan
    add_feature(features, formulas, "L2", pd.Series(l2, index=df.index), "L2 disagreement")
    add_feature(features, formulas, "L1", pd.Series(l1, index=df.index), "L1 disagreement")
    add_feature(
        features,
        formulas,
        "components_used",
        pd.Series(components_used, index=df.index),
        "components used",
    )
    for w in [15, 60]:
        mean_l2 = tfl.rolling_mean(
            pd.Series(l2, index=df.index),
            window=w,
            min_periods=min_periods(w),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        std_l2 = tfl.rolling_std(
            pd.Series(l2, index=df.index),
            window=w,
            min_periods=min_periods(w),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        add_feature(features, formulas, f"mean_L2_{w}", mean_l2, "mean L2")
        add_feature(features, formulas, f"std_L2_{w}", std_l2, "std L2")
    slope_l2 = tfl.rolling_slope(
        pd.Series(l2, index=df.index),
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    add_feature(features, formulas, "slope_L2_30", slope_l2, "slope L2 30")
    train_fitted.append({"scales": scales_meta})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e18(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    dep = df["dep_ens"]
    wet = df["wet_score"]
    corr_dep = tfl.rolling_corr(
        df["nx_ens"],
        dep,
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    corr_wet = tfl.rolling_corr(
        df["nx_ens"],
        wet,
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    dep_mean = tfl.rolling_mean(
        dep,
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    dep_std = tfl.rolling_std(
        dep,
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    wet_mean = tfl.rolling_mean(
        wet,
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    wet_std = tfl.rolling_std(
        wet,
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    dep_anom = safe_divide((dep - dep_mean).to_numpy(), dep_std.to_numpy() + EPS, default=np.nan)
    wet_anom = safe_divide((wet - wet_mean).to_numpy(), wet_std.to_numpy() + EPS, default=np.nan)
    add_feature(features, formulas, "corr_nx_dep", corr_dep, "corr nx dep")
    add_feature(features, formulas, "corr_nx_wet", corr_wet, "corr nx wet")
    add_feature(features, formulas, "dep_anom_z", pd.Series(dep_anom, index=df.index), "dep z")
    add_feature(features, formulas, "wet_anom_z", pd.Series(wet_anom, index=df.index), "wet z")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e19(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    cols = ["tmp_ens", "dpt_ens", "wsp_ens", "sin_doy", "cos_doy"]
    X_train = df.loc[ctx.train_mask, cols].to_numpy(dtype=float)
    y_train = df.loc[ctx.train_mask, "nx_ens"].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_all = scaler.transform(df[cols].to_numpy(dtype=float))
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_train, y_train)
    nx_ctx = model.predict(X_all)
    inc = df["nx_ens"] - nx_ctx
    inc_mean = tfl.rolling_mean(
        pd.Series(inc, index=df.index),
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    inc_std = tfl.rolling_std(
        pd.Series(inc, index=df.index),
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    inc_z = safe_divide((inc - inc_mean).to_numpy(), inc_std.to_numpy() + EPS, default=np.nan)
    add_feature(features, formulas, "nx_ctx", pd.Series(nx_ctx, index=df.index), "nx_ctx")
    add_feature(features, formulas, "inc", pd.Series(inc, index=df.index), "nx_ens - nx_ctx")
    add_feature(features, formulas, "inc_mean_30", inc_mean, "inc mean 30")
    add_feature(features, formulas, "inc_std_30", inc_std, "inc std 30")
    add_feature(features, formulas, "inc_z", pd.Series(inc_z, index=df.index), "inc z")
    train_fitted.append({"model": "ridge", "cols": cols})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e20(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    gfs_theta = np.deg2rad(_get_col(df, "gfs_wdr_mean").to_numpy(dtype=float))
    nam_theta = np.deg2rad(_get_col(df, "nam_wdr_mean").to_numpy(dtype=float))
    gfs_wsp = _get_col(df, "gfs_wsp_mean").to_numpy(dtype=float)
    nam_wsp = _get_col(df, "nam_wsp_mean").to_numpy(dtype=float)
    u_gfs = gfs_wsp * np.cos(gfs_theta)
    v_gfs = gfs_wsp * np.sin(gfs_theta)
    u_nam = nam_wsp * np.cos(nam_theta)
    v_nam = nam_wsp * np.sin(nam_theta)
    u_ens = np.nanmean(np.vstack([u_gfs, u_nam]).T, axis=1)
    v_ens = np.nanmean(np.vstack([v_gfs, v_nam]).T, axis=1)
    spd = np.sqrt(u_ens**2 + v_ens**2)
    u_mean = tfl.rolling_mean(
        pd.Series(u_ens, index=df.index),
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    v_mean = tfl.rolling_mean(
        pd.Series(v_ens, index=df.index),
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    spd_mean = np.sqrt(u_mean.to_numpy() ** 2 + v_mean.to_numpy() ** 2)
    cosang = safe_divide(u_ens * u_mean.to_numpy() + v_ens * v_mean.to_numpy(), spd * spd_mean + EPS, default=np.nan)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang_dev = np.arccos(cosang)
    spd_mean_30 = tfl.rolling_mean(
        pd.Series(spd, index=df.index),
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    spd_std_30 = tfl.rolling_std(
        pd.Series(spd, index=df.index),
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    spd_anom = safe_divide((spd - spd_mean_30).to_numpy(), spd_std_30.to_numpy() + EPS, default=np.nan)
    add_feature(features, formulas, "ang_dev", pd.Series(ang_dev, index=df.index), "angle deviation")
    add_feature(features, formulas, "spd_anom_z", pd.Series(spd_anom, index=df.index), "speed anomaly")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e21(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    dep = df["dep_ens"]
    q33_map, q33_default = station_quantile(df.loc[ctx.train_mask], "dep_ens", 0.33)
    q66_map, q66_default = station_quantile(df.loc[ctx.train_mask], "dep_ens", 0.66)
    thr33 = map_station_threshold(df, q33_map, q33_default)
    thr66 = map_station_threshold(df, q66_map, q66_default)
    bin_dep = np.where(dep <= thr33, 0, np.where(dep <= thr66, 1, 2)).astype(int)
    r_ens = df["resid_ens"]
    bias_bin = []
    mae_bin = []
    for b in [0, 1, 2]:
        indicator = (bin_dep == b).astype(float)
        bias = conditional_rolling_mean(
            r_ens,
            indicator,
            window=60,
            lag=ctx.truth_lag,
            group_key=ctx.group_station_asof,
            min_needed=15,
        )
        mae = conditional_rolling_mean(
            r_ens.abs(),
            indicator,
            window=60,
            lag=ctx.truth_lag,
            group_key=ctx.group_station_asof,
            min_needed=15,
        )
        bias_bin.append(bias)
        mae_bin.append(mae)
    bias_uncond = rolling_mean_fallback(
        r_ens, 60, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    mae_uncond = rolling_mean_fallback(
        r_ens.abs(), 60, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    bias_cond = np.full(len(df), np.nan, dtype=float)
    mae_cond = np.full(len(df), np.nan, dtype=float)
    for b in [0, 1, 2]:
        mask = bin_dep == b
        bias_cond[mask] = bias_bin[b][mask]
        mae_cond[mask] = mae_bin[b][mask]
    bias_cond = np.where(np.isfinite(bias_cond), bias_cond, bias_uncond.to_numpy())
    mae_cond = np.where(np.isfinite(mae_cond), mae_cond, mae_uncond.to_numpy())
    add_feature(features, formulas, "bin_dep", pd.Series(bin_dep, index=df.index), "dep bin")
    add_feature(features, formulas, "bias_depbin", pd.Series(bias_cond, index=df.index), "dep bin bias")
    add_feature(features, formulas, "mae_depbin", pd.Series(mae_cond, index=df.index), "dep bin mae")
    add_feature(
        features,
        formulas,
        "nx_depbin_bc",
        df["nx_ens"] + pd.Series(bias_cond, index=df.index),
        "nx_ens + dep bin bias",
    )
    train_fitted.append({"q33": q33_map, "q66": q66_map})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e22(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    q75_map, q75_default = station_quantile(df.loc[ctx.train_mask], "q12_ens", 0.75)
    q12_thr = map_station_threshold(df, q75_map, q75_default)
    wet = ((_get_col(df, "p12_ens") >= 50) | (df["q12_ens"] >= q12_thr)).astype(float)
    r_ens = df["resid_ens"]
    bias_wet = conditional_rolling_mean(
        r_ens,
        wet,
        window=60,
        lag=ctx.truth_lag,
        group_key=ctx.group_station_asof,
        min_needed=15,
    )
    bias_dry = conditional_rolling_mean(
        r_ens,
        1 - wet,
        window=60,
        lag=ctx.truth_lag,
        group_key=ctx.group_station_asof,
        min_needed=15,
    )
    mae_wet = conditional_rolling_mean(
        r_ens.abs(),
        wet,
        window=60,
        lag=ctx.truth_lag,
        group_key=ctx.group_station_asof,
        min_needed=15,
    )
    mae_dry = conditional_rolling_mean(
        r_ens.abs(),
        1 - wet,
        window=60,
        lag=ctx.truth_lag,
        group_key=ctx.group_station_asof,
        min_needed=15,
    )
    bias_uncond = rolling_mean_fallback(
        r_ens, 60, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    mae_uncond = rolling_mean_fallback(
        r_ens.abs(), 60, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    bias_cond = np.where(wet.to_numpy() > 0.5, bias_wet.to_numpy(), bias_dry.to_numpy())
    mae_cond = np.where(wet.to_numpy() > 0.5, mae_wet.to_numpy(), mae_dry.to_numpy())
    bias_cond = np.where(np.isfinite(bias_cond), bias_cond, bias_uncond.to_numpy())
    mae_cond = np.where(np.isfinite(mae_cond), mae_cond, mae_uncond.to_numpy())
    add_feature(features, formulas, "wet_flag", wet, "wet flag")
    add_feature(features, formulas, "bias_wetdry", pd.Series(bias_cond, index=df.index), "wet/dry bias")
    add_feature(features, formulas, "mae_wetdry", pd.Series(mae_cond, index=df.index), "wet/dry mae")
    add_feature(
        features,
        formulas,
        "nx_wetdry_bc",
        df["nx_ens"] + pd.Series(bias_cond, index=df.index),
        "nx_ens + wet/dry bias",
    )
    train_fitted.append({"q75_q12": q75_map})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e23(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    conv_score = df["t06_ens"] + df["t12_ens"]
    train_conv = conv_score[ctx.train_mask]
    q80_map = train_conv.groupby(df.loc[ctx.train_mask, "station_id"]).quantile(0.80).to_dict()
    q80_default = float(train_conv.quantile(0.80)) if len(train_conv) else 0.0
    thr_conv = df["station_id"].map(q80_map).fillna(q80_default).astype(float)
    conv = (conv_score >= thr_conv).astype(float)
    r_ens = df["resid_ens"]
    bias_conv = conditional_rolling_mean(
        r_ens,
        conv,
        window=60,
        lag=ctx.truth_lag,
        group_key=ctx.group_station_asof,
        min_needed=15,
    )
    bias_non = conditional_rolling_mean(
        r_ens,
        1 - conv,
        window=60,
        lag=ctx.truth_lag,
        group_key=ctx.group_station_asof,
        min_needed=15,
    )
    bias_uncond = rolling_mean_fallback(
        r_ens, 60, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    bias_cond = np.where(conv.to_numpy() > 0.5, bias_conv.to_numpy(), bias_non.to_numpy())
    bias_cond = np.where(np.isfinite(bias_cond), bias_cond, bias_uncond.to_numpy())
    add_feature(features, formulas, "conv_score", conv_score, "conv score")
    add_feature(features, formulas, "conv_flag", conv, "conv flag")
    add_feature(features, formulas, "bias_conv", pd.Series(bias_cond, index=df.index), "conv bias")
    add_feature(
        features,
        formulas,
        "nx_conv_bc",
        df["nx_ens"] + pd.Series(bias_cond, index=df.index),
        "nx_ens + conv bias",
    )
    train_fitted.append({"thr_conv": q80_map})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e24(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    q75_map, q75_default = station_quantile(df.loc[ctx.train_mask], "snw_ens", 0.75)
    snw_thr = map_station_threshold(df, q75_map, q75_default)
    wintry = ((df["pos_ens"].combine(df["poz_ens"], np.maximum) >= 20) | (df["snw_ens"] >= snw_thr)).astype(float)
    r_ens = df["resid_ens"]
    bias_win = conditional_rolling_mean(
        r_ens,
        wintry,
        window=90,
        lag=ctx.truth_lag,
        group_key=ctx.group_station_asof,
        min_needed=20,
    )
    bias_non = conditional_rolling_mean(
        r_ens,
        1 - wintry,
        window=90,
        lag=ctx.truth_lag,
        group_key=ctx.group_station_asof,
        min_needed=20,
    )
    bias_uncond = rolling_mean_fallback(
        r_ens, 90, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    bias_cond = np.where(wintry.to_numpy() > 0.5, bias_win.to_numpy(), bias_non.to_numpy())
    bias_cond = np.where(np.isfinite(bias_cond), bias_cond, bias_uncond.to_numpy())
    add_feature(features, formulas, "wintry_flag", wintry, "wintry flag")
    add_feature(features, formulas, "bias_wintry", pd.Series(bias_cond, index=df.index), "wintry bias")
    add_feature(
        features,
        formulas,
        "nx_wintry_bc",
        df["nx_ens"] + pd.Series(bias_cond, index=df.index),
        "nx_ens + wintry bias",
    )
    train_fitted.append({"snw_q75": q75_map})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e25(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    cig_map, cig_default = station_quantile(df.loc[ctx.train_mask], "cig_ens_min", 0.20)
    vis_map, vis_default = station_quantile(df.loc[ctx.train_mask], "vis_ens_min", 0.20)
    cig_thr = map_station_threshold(df, cig_map, cig_default)
    vis_thr = map_station_threshold(df, vis_map, vis_default)
    low_cig = (df["cig_ens_min"] <= cig_thr).astype(int)
    low_vis = (df["vis_ens_min"] <= vis_thr).astype(int)
    reg = (2 * low_cig + low_vis).astype(int)
    r_ens = df["resid_ens"]
    bias_reg = []
    for k in [0, 1, 2, 3]:
        bias_k = conditional_rolling_mean(
            r_ens,
            (reg == k).astype(float),
            window=60,
            lag=ctx.truth_lag,
            group_key=ctx.group_station_asof,
            min_needed=12,
        )
        bias_reg.append(bias_k)
    bias_uncond = rolling_mean_fallback(
        r_ens, 60, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    bias_cond = np.full(len(df), np.nan, dtype=float)
    for k in [0, 1, 2, 3]:
        mask = reg == k
        bias_cond[mask] = bias_reg[k][mask]
    bias_cond = np.where(np.isfinite(bias_cond), bias_cond, bias_uncond.to_numpy())
    add_feature(features, formulas, "reg_cig_vis", pd.Series(reg, index=df.index), "cig/vis reg")
    add_feature(features, formulas, "bias_cigvis", pd.Series(bias_cond, index=df.index), "regime bias")
    add_feature(
        features,
        formulas,
        "nx_cigvis_bc",
        df["nx_ens"] + pd.Series(bias_cond, index=df.index),
        "nx_ens + cig/vis bias",
    )
    train_fitted.append({"cig_q20": cig_map, "vis_q20": vis_map})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e26(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    q33_map, q33_default = station_quantile(df.loc[ctx.train_mask], "wsp_ens", 0.33)
    q66_map, q66_default = station_quantile(df.loc[ctx.train_mask], "wsp_ens", 0.66)
    thr33 = map_station_threshold(df, q33_map, q33_default)
    thr66 = map_station_threshold(df, q66_map, q66_default)
    wsp = df["wsp_ens"]
    bin_wsp = np.where(wsp <= thr33, 0, np.where(wsp <= thr66, 1, 2)).astype(int)
    r_ens = df["resid_ens"]
    bias_bin = []
    for b in [0, 1, 2]:
        bias = conditional_rolling_mean(
            r_ens,
            (bin_wsp == b).astype(float),
            window=60,
            lag=ctx.truth_lag,
            group_key=ctx.group_station_asof,
            min_needed=15,
        )
        bias_bin.append(bias)
    bias_uncond = rolling_mean_fallback(
        r_ens, 60, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    bias_cond = np.full(len(df), np.nan, dtype=float)
    for b in [0, 1, 2]:
        mask = bin_wsp == b
        bias_cond[mask] = bias_bin[b][mask]
    bias_cond = np.where(np.isfinite(bias_cond), bias_cond, bias_uncond.to_numpy())
    add_feature(features, formulas, "bin_wsp", pd.Series(bin_wsp, index=df.index), "wind bin")
    add_feature(features, formulas, "bias_wspbin", pd.Series(bias_cond, index=df.index), "wind bin bias")
    add_feature(
        features,
        formulas,
        "nx_wsp_bc",
        df["nx_ens"] + pd.Series(bias_cond, index=df.index),
        "nx_ens + wind bias",
    )
    train_fitted.append({"q33": q33_map, "q66": q66_map})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e27(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    dep = df["dep_ens"]
    q33_map, q33_default = station_quantile(df.loc[ctx.train_mask], "dep_ens", 0.33)
    dep_thr = map_station_threshold(df, q33_map, q33_default)
    humid = (dep <= dep_thr).astype(int)
    cig_map, cig_default = station_quantile(df.loc[ctx.train_mask], "cig_ens_min", 0.20)
    cig_thr = map_station_threshold(df, cig_map, cig_default)
    cloudy = ((_get_col(df, "p12_ens") >= 50) | (df["cig_ens_min"] <= cig_thr)).astype(int)
    reg = (2 * humid + cloudy).astype(int)
    r_ens = df["resid_ens"]
    bias_reg = []
    for k in [0, 1, 2, 3]:
        bias_k = conditional_rolling_mean(
            r_ens,
            (reg == k).astype(float),
            window=60,
            lag=ctx.truth_lag,
            group_key=ctx.group_station_asof,
            min_needed=12,
        )
        bias_reg.append(bias_k)
    bias_uncond = rolling_mean_fallback(
        r_ens, 60, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    bias_cond = np.full(len(df), np.nan, dtype=float)
    for k in [0, 1, 2, 3]:
        mask = reg == k
        bias_cond[mask] = bias_reg[k][mask]
    bias_cond = np.where(np.isfinite(bias_cond), bias_cond, bias_uncond.to_numpy())
    add_feature(features, formulas, "reg_humid_cloud", pd.Series(reg, index=df.index), "humid/cloud reg")
    add_feature(features, formulas, "bias_humid_cloud", pd.Series(bias_cond, index=df.index), "reg bias")
    add_feature(
        features,
        formulas,
        "nx_humid_cloud_bc",
        df["nx_ens"] + pd.Series(bias_cond, index=df.index),
        "nx_ens + humid/cloud bias",
    )
    train_fitted.append({"dep_q33": q33_map, "cig_q20": cig_map})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e28(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    q75_map, q75_default = station_quantile(df.loc[ctx.train_mask], "q12_ens", 0.75)
    q12_thr = map_station_threshold(df, q75_map, q75_default)
    wet = ((_get_col(df, "p12_ens") >= 50) | (df["q12_ens"] >= q12_thr)).astype(int)
    runlen_prev = tfl.streak_length(wet, lag=1, cap=180, group_key=ctx.group_station_asof)
    wet_prev = wet.groupby(ctx.group_station_asof).shift(1)
    runlen = np.where(wet.to_numpy() == wet_prev.to_numpy(), runlen_prev.to_numpy(), 0.0)
    frac_30 = tfl.rolling_mean(
        wet,
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    transitions_30 = tfl.switch_count(
        wet,
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    add_feature(features, formulas, "runlen_wet", pd.Series(runlen, index=df.index), "wet run length")
    add_feature(features, formulas, "frac_wet_30", frac_30, "wet fraction 30")
    add_feature(features, formulas, "transitions_30", transitions_30, "wet transitions 30")
    bucket_1_2 = ((runlen >= 1) & (runlen <= 2)).astype(float)
    bucket_3_5 = ((runlen >= 3) & (runlen <= 5)).astype(float)
    bucket_6_10 = ((runlen >= 6) & (runlen <= 10)).astype(float)
    bucket_gt10 = (runlen > 10).astype(float)
    add_feature(features, formulas, "wet_age_1_2", pd.Series(bucket_1_2, index=df.index), "wet age 1-2")
    add_feature(features, formulas, "wet_age_3_5", pd.Series(bucket_3_5, index=df.index), "wet age 3-5")
    add_feature(features, formulas, "wet_age_6_10", pd.Series(bucket_6_10, index=df.index), "wet age 6-10")
    add_feature(features, formulas, "wet_age_gt10", pd.Series(bucket_gt10, index=df.index), "wet age >10")
    train_fitted.append({"q75_q12": q75_map})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e29(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    vars_list = [
        ("p12_ens", df["p12_ens"]),
        ("q12_ens", df["q12_ens"]),
        ("cig_ens_min", df["cig_ens_min"]),
        ("vis_ens_min", df["vis_ens_min"]),
    ]
    z_abs_sum = np.zeros(len(df), dtype=float)
    for name, series in vars_list:
        mean_30 = tfl.rolling_mean(
            series,
            window=30,
            min_periods=min_periods(30),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        std_30 = tfl.rolling_std(
            series,
            window=30,
            min_periods=min_periods(30),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        z_30 = safe_divide((series - mean_30).to_numpy(), std_30.to_numpy() + EPS, default=np.nan)
        z_abs_sum += np.abs(z_30)
        prc_60 = tfl.percent_rank(
            series,
            window=60,
            min_periods=min_periods(60),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        add_feature(features, formulas, f"{name}_z_30", pd.Series(z_30, index=df.index), "z30")
        add_feature(features, formulas, f"{name}_prc_60", prc_60, "percent rank 60")
    add_feature(features, formulas, "extremeness", pd.Series(z_abs_sum, index=df.index), "sum abs z")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e30(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    for name, series in [("dep", df["dep_ens"]), ("wsp", df["wsp_ens"])]:
        m7 = tfl.rolling_mean(series, window=7, min_periods=min_periods(7), lag=1, group_key=ctx.group_station_asof)
        m60 = tfl.rolling_mean(series, window=60, min_periods=min_periods(60), lag=1, group_key=ctx.group_station_asof)
        s7 = tfl.rolling_std(series, window=7, min_periods=min_periods(7), lag=1, group_key=ctx.group_station_asof)
        s60 = tfl.rolling_std(series, window=60, min_periods=min_periods(60), lag=1, group_key=ctx.group_station_asof)
        shift_mean = m7 - m60
        shift_std = s7 - s60
        shift_z = safe_divide(shift_mean.to_numpy(), s60.to_numpy() + EPS, default=np.nan)
        add_feature(features, formulas, f"{name}_shift_mean", shift_mean, "shift mean")
        add_feature(features, formulas, f"{name}_shift_std", shift_std, "shift std")
        add_feature(features, formulas, f"{name}_shift_z", pd.Series(shift_z, index=df.index), "shift z")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e31(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    cols = ["nx_ens", "tmp_ens", "dpt_ens", "sin_doy", "cos_doy"]
    scaled, meta = station_standardize(df, cols, ctx.train_mask)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    fallback = tfl.rolling_mean(
        df["actual_tmax_f"],
        window=3650,
        min_periods=1,
        lag=ctx.truth_lag,
        group_key=ctx.group_station,
    ).to_numpy(dtype=float)
    best_k = 5
    best_mae = float("inf")
    for k in [5, 10, 20]:
        stats = compute_knn_stats(
            scaled,
            y,
            ctx.group_station_asof,
            ctx.truth_lag,
            k,
            min_dims=4,
            fallback=fallback,
        )
        pred = stats["mean"]
        val_pred = pred[ctx.val_mask]
        val_true = y[ctx.val_mask]
        if len(val_true) == 0:
            continue
        mae = float(np.mean(np.abs(val_pred - val_true)))
        if mae < best_mae:
            best_mae = mae
            best_k = k
    stats = compute_knn_stats(
        scaled,
        y,
        ctx.group_station_asof,
        ctx.truth_lag,
        best_k,
        min_dims=4,
        fallback=fallback,
    )
    add_feature(features, formulas, "analog_mean", pd.Series(stats["mean"], index=df.index), "analog mean")
    add_feature(features, formulas, "analog_median", pd.Series(stats["median"], index=df.index), "analog median")
    add_feature(features, formulas, "analog_std", pd.Series(stats["std"], index=df.index), "analog std")
    add_feature(features, formulas, "analog_wmean", pd.Series(stats["wmean"], index=df.index), "analog wmean")
    add_feature(features, formulas, "analog_count", pd.Series(stats["count"], index=df.index), "analog count")
    add_feature(features, formulas, "analog_mean_dist", pd.Series(stats["mean_dist"], index=df.index), "analog mean dist")
    train_fitted.append({"k": best_k, "cols": cols, "standardization": meta})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e32(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    cols = ["nx_ens", "tmp_ens", "dpt_ens", "sin_doy", "cos_doy"]
    scaled, meta = station_standardize(df, cols, ctx.train_mask)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = df["resid_ens"].to_numpy(dtype=float)
    fallback = tfl.rolling_mean(
        df["resid_ens"],
        window=3650,
        min_periods=1,
        lag=ctx.truth_lag,
        group_key=ctx.group_station,
    ).to_numpy(dtype=float)
    best_k = 5
    best_mae = float("inf")
    for k in [5, 10, 20]:
        stats = compute_knn_stats(
            scaled,
            resid,
            ctx.group_station_asof,
            ctx.truth_lag,
            k,
            min_dims=4,
            fallback=fallback,
        )
        pred = df["nx_ens"].to_numpy(dtype=float) + stats["wmean"]
        val_pred = pred[ctx.val_mask]
        val_true = y[ctx.val_mask]
        if len(val_true) == 0:
            continue
        mae = float(np.mean(np.abs(val_pred - val_true)))
        if mae < best_mae:
            best_mae = mae
            best_k = k
    stats = compute_knn_stats(
        scaled,
        resid,
        ctx.group_station_asof,
        ctx.truth_lag,
        best_k,
        min_dims=4,
        fallback=fallback,
    )
    add_feature(features, formulas, "analog_resid_mean", pd.Series(stats["mean"], index=df.index), "analog resid mean")
    add_feature(features, formulas, "analog_resid_wmean", pd.Series(stats["wmean"], index=df.index), "analog resid wmean")
    add_feature(features, formulas, "analog_resid_std", pd.Series(stats["std"], index=df.index), "analog resid std")
    add_feature(
        features,
        formulas,
        "nx_analog_bc",
        df["nx_ens"] + pd.Series(stats["wmean"], index=df.index),
        "nx_ens + analog resid",
    )
    train_fitted.append({"k": best_k, "cols": cols, "standardization": meta})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e33(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    cols = [
        "nx_ens",
        "tmp_ens",
        "dpt_ens",
        "p12_ens",
        "q12_ens",
        "cig_ens_min",
        "vis_ens_min",
        "wsp_ens",
        "sin_doy",
        "cos_doy",
    ]
    scaled_full, meta_full = station_standardize(df, cols, ctx.train_mask)
    simple_cols = ["nx_ens", "tmp_ens", "dpt_ens", "sin_doy", "cos_doy"]
    scaled_simple, meta_simple = station_standardize(df, simple_cols, ctx.train_mask)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    fallback = tfl.rolling_mean(
        df["actual_tmax_f"],
        window=3650,
        min_periods=1,
        lag=ctx.truth_lag,
        group_key=ctx.group_station,
    ).to_numpy(dtype=float)
    best_k = 10
    best_mae = float("inf")
    for k in [10, 20]:
        stats_full = compute_knn_stats(
            scaled_full,
            y,
            ctx.group_station_asof,
            ctx.truth_lag,
            k,
            min_dims=8,
            fallback=None,
        )
        stats_simple = compute_knn_stats(
            scaled_simple,
            y,
            ctx.group_station_asof,
            ctx.truth_lag,
            k,
            min_dims=4,
            fallback=fallback,
        )
        pred = np.where(np.isfinite(stats_full["mean"]), stats_full["mean"], stats_simple["mean"])
        val_pred = pred[ctx.val_mask]
        val_true = y[ctx.val_mask]
        if len(val_true) == 0:
            continue
        mae = float(np.mean(np.abs(val_pred - val_true)))
        if mae < best_mae:
            best_mae = mae
            best_k = k
    stats_full = compute_knn_stats(
        scaled_full,
        y,
        ctx.group_station_asof,
        ctx.truth_lag,
        best_k,
        min_dims=8,
        fallback=None,
    )
    stats_simple = compute_knn_stats(
        scaled_simple,
        y,
        ctx.group_station_asof,
        ctx.truth_lag,
        best_k,
        min_dims=4,
        fallback=fallback,
    )
    mean = np.where(np.isfinite(stats_full["mean"]), stats_full["mean"], stats_simple["mean"])
    median = np.where(np.isfinite(stats_full["median"]), stats_full["median"], stats_simple["median"])
    std = np.where(np.isfinite(stats_full["std"]), stats_full["std"], stats_simple["std"])
    wmean = np.where(np.isfinite(stats_full["wmean"]), stats_full["wmean"], stats_simple["wmean"])
    count = np.where(stats_full["count"] > 0, stats_full["count"], stats_simple["count"])
    mean_dist = np.where(
        np.isfinite(stats_full["mean_dist"]), stats_full["mean_dist"], stats_simple["mean_dist"]
    )
    add_feature(features, formulas, "analog_mean", pd.Series(mean, index=df.index), "analog mean")
    add_feature(features, formulas, "analog_median", pd.Series(median, index=df.index), "analog median")
    add_feature(features, formulas, "analog_std", pd.Series(std, index=df.index), "analog std")
    add_feature(features, formulas, "analog_wmean", pd.Series(wmean, index=df.index), "analog wmean")
    add_feature(features, formulas, "analog_count", pd.Series(count, index=df.index), "analog count")
    add_feature(features, formulas, "analog_mean_dist", pd.Series(mean_dist, index=df.index), "analog mean dist")
    train_fitted.append({"k": best_k, "cols": cols, "standardization": meta_full, "fallback_cols": simple_cols})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e34(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    cols = [
        "nx_ens",
        "tmp_ens",
        "dpt_ens",
        "p12_ens",
        "q12_ens",
        "cig_ens_min",
        "vis_ens_min",
        "wsp_ens",
        "sin_doy",
        "cos_doy",
    ]
    scaled, meta = station_standardize(df, cols, ctx.train_mask)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    doy = df["day_of_year"].to_numpy(dtype=int)
    fallback = tfl.rolling_mean(
        df["actual_tmax_f"],
        window=3650,
        min_periods=1,
        lag=ctx.truth_lag,
        group_key=ctx.group_station,
    ).to_numpy(dtype=float)

    def knn_doy(k: int, radius: int) -> dict:
        n = len(df)
        mean = np.full(n, np.nan, dtype=float)
        std = np.full(n, np.nan, dtype=float)
        wmean = np.full(n, np.nan, dtype=float)
        groups = ctx.group_station_asof.to_numpy()
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            feats = scaled[idx]
            vals = y[idx]
            for pos, row_idx in enumerate(idx):
                if pos < ctx.truth_lag:
                    continue
                current = feats[pos]
                if not np.isfinite(current).any():
                    continue
                cand_positions = np.arange(0, pos - ctx.truth_lag + 1)
                if cand_positions.size == 0:
                    continue
                dists = []
                cand_vals = []
                for cand_pos in cand_positions:
                    cd = abs(doy[idx[cand_pos]] - doy[idx[pos]])
                    cd = min(cd, 365 - cd)
                    if cd > radius:
                        continue
                    cand = feats[cand_pos]
                    mask = np.isfinite(current) & np.isfinite(cand)
                    if mask.sum() < 8:
                        continue
                    dist = float(np.linalg.norm(current[mask] - cand[mask]))
                    dists.append(dist)
                    cand_vals.append(vals[cand_pos])
                if not dists:
                    continue
                order = np.argsort(dists)[:k]
                sel_vals = np.array([cand_vals[i] for i in order], dtype=float)
                if sel_vals.size:
                    mean[row_idx] = float(np.mean(sel_vals))
                    std[row_idx] = float(np.std(sel_vals, ddof=0))
                    weights = 1.0 / (np.array([dists[i] for i in order], dtype=float) + EPS)
                    sum_w = float(np.sum(weights))
                    wmean[row_idx] = float(np.sum(sel_vals * weights) / sum_w) if sum_w > 0 else mean[row_idx]
        return {"mean": mean, "std": std, "wmean": wmean}

    best_k = 5
    best_mae = float("inf")
    for k in [5, 10, 20]:
        pred = None
        for radius in [30, 45, 90]:
            stats = knn_doy(k, radius)
            pred = stats["mean"]
            if np.isfinite(pred).any():
                break
        if pred is None:
            continue
        pred = np.where(np.isfinite(pred), pred, fallback)
        val_pred = pred[ctx.val_mask]
        val_true = y[ctx.val_mask]
        if len(val_true) == 0:
            continue
        mae = float(np.mean(np.abs(val_pred - val_true)))
        if mae < best_mae:
            best_mae = mae
            best_k = k
    pred = np.full(len(df), np.nan, dtype=float)
    pred_std = np.full(len(df), np.nan, dtype=float)
    pred_wmean = np.full(len(df), np.nan, dtype=float)
    for radius in [30, 45, 90]:
        stats = knn_doy(best_k, radius)
        pred = stats["mean"]
        pred_std = stats["std"]
        pred_wmean = stats["wmean"]
        if np.isfinite(pred).any():
            break
    pred = np.where(np.isfinite(pred), pred, fallback)
    pred_wmean = np.where(np.isfinite(pred_wmean), pred_wmean, pred)
    pred_std = np.where(np.isfinite(pred_std), pred_std, np.nan)
    add_feature(features, formulas, "analog_mean", pd.Series(pred, index=df.index), "doy analog mean")
    add_feature(features, formulas, "analog_wmean", pd.Series(pred_wmean, index=df.index), "doy analog wmean")
    add_feature(features, formulas, "analog_std", pd.Series(pred_std, index=df.index), "doy analog std")
    train_fitted.append({"k": best_k, "cols": cols, "standardization": meta})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e35(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    cols = [
        "nx_ens",
        "tmp_ens",
        "dpt_ens",
        "p12_ens",
        "q12_ens",
        "cig_ens_min",
        "vis_ens_min",
        "wsp_ens",
        "sin_doy",
        "cos_doy",
    ]
    scaled, meta = station_standardize(df, cols, ctx.train_mask)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    err_gfs = np.abs(y - _get_col(df, "gfs_n_x_mean").to_numpy(dtype=float))
    err_nam = np.abs(y - _get_col(df, "nam_n_x_mean").to_numpy(dtype=float))
    best_k = 10
    best_mae = float("inf")
    for k in [10, 20]:
        stats_gfs = compute_knn_stats(
            scaled,
            err_gfs,
            ctx.group_station_asof,
            ctx.truth_lag,
            k,
            min_dims=8,
            fallback=None,
        )
        stats_nam = compute_knn_stats(
            scaled,
            err_nam,
            ctx.group_station_asof,
            ctx.truth_lag,
            k,
            min_dims=8,
            fallback=None,
        )
        amae_gfs = stats_gfs["mean"]
        amae_nam = stats_nam["mean"]
        w_gfs = 1.0 / (amae_gfs + 0.25)
        w_nam = 1.0 / (amae_nam + 0.25)
        w_sum = w_gfs + w_nam
        w_gfs = safe_divide(w_gfs, w_sum, default=0.5)
        w_nam = safe_divide(w_nam, w_sum, default=0.5)
        pred = w_gfs * _get_col(df, "gfs_n_x_mean").to_numpy(dtype=float) + w_nam * _get_col(df, "nam_n_x_mean").to_numpy(dtype=float)
        val_pred = pred[ctx.val_mask]
        val_true = y[ctx.val_mask]
        if len(val_true) == 0:
            continue
        mae = float(np.mean(np.abs(val_pred - val_true)))
        if mae < best_mae:
            best_mae = mae
            best_k = k
    stats_gfs = compute_knn_stats(
        scaled,
        err_gfs,
        ctx.group_station_asof,
        ctx.truth_lag,
        best_k,
        min_dims=8,
        fallback=None,
    )
    stats_nam = compute_knn_stats(
        scaled,
        err_nam,
        ctx.group_station_asof,
        ctx.truth_lag,
        best_k,
        min_dims=8,
        fallback=None,
    )
    amae_gfs = stats_gfs["mean"]
    amae_nam = stats_nam["mean"]
    w_gfs = 1.0 / (amae_gfs + 0.25)
    w_nam = 1.0 / (amae_nam + 0.25)
    w_sum = w_gfs + w_nam
    w_gfs = safe_divide(w_gfs, w_sum, default=0.5)
    w_nam = safe_divide(w_nam, w_sum, default=0.5)
    nx_skill = w_gfs * _get_col(df, "gfs_n_x_mean").to_numpy(dtype=float) + w_nam * _get_col(df, "nam_n_x_mean").to_numpy(dtype=float)
    add_feature(features, formulas, "w_gfs", pd.Series(w_gfs, index=df.index), "analog weight gfs")
    add_feature(features, formulas, "w_nam", pd.Series(w_nam, index=df.index), "analog weight nam")
    add_feature(features, formulas, "nx_skillanalog", pd.Series(nx_skill, index=df.index), "skill analog blend")
    train_fitted.append({"k": best_k, "cols": cols, "standardization": meta})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e36(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    cols = [
        "nx_ens",
        "tmp_ens",
        "dpt_ens",
        "p12_ens",
        "q12_ens",
        "wsp_ens",
        "cig_ens_mean",
        "vis_ens_mean",
    ]
    scaled, meta, _ = global_standardize(df, cols, ctx.train_mask)
    resid = df["resid_ens"].to_numpy(dtype=float)
    best_k = 6
    best_mae = float("inf")
    for k in [6, 8, 12]:
        km = KMeans(n_clusters=k, random_state=ctx.seed, n_init=10)
        km.fit(scaled[ctx.train_mask])
        cluster_id = km.predict(scaled)
        bias = {}
        for cid in range(k):
            mask = ctx.train_mask & (cluster_id == cid)
            if mask.any():
                bias[cid] = float(np.mean(resid[mask]))
        bias_default = float(np.mean(resid[ctx.train_mask])) if ctx.train_mask.any() else 0.0
        bias_series = np.array([bias.get(cid, bias_default) for cid in cluster_id], dtype=float)
        pred = df["nx_ens"].to_numpy(dtype=float) + bias_series
        val_pred = pred[ctx.val_mask]
        val_true = df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
        if len(val_true) == 0:
            continue
        mae = float(np.mean(np.abs(val_pred - val_true)))
        if mae < best_mae:
            best_mae = mae
            best_k = k
    km = KMeans(n_clusters=best_k, random_state=ctx.seed, n_init=10)
    km.fit(scaled[ctx.train_mask])
    cluster_id = km.predict(scaled)
    bias = {}
    for cid in range(best_k):
        mask = ctx.train_mask & (cluster_id == cid)
        if mask.any():
            bias[cid] = float(np.mean(resid[mask]))
    bias_default = float(np.mean(resid[ctx.train_mask])) if ctx.train_mask.any() else 0.0
    bias_series = np.array([bias.get(cid, bias_default) for cid in cluster_id], dtype=float)
    dist = np.linalg.norm(scaled - km.cluster_centers_[cluster_id], axis=1)
    add_feature(features, formulas, "cluster_id", pd.Series(cluster_id, index=df.index), "cluster id")
    add_feature(features, formulas, "cluster_dist", pd.Series(dist, index=df.index), "dist to centroid")
    add_feature(features, formulas, "bias_cluster", pd.Series(bias_series, index=df.index), "cluster bias")
    add_feature(
        features,
        formulas,
        "nx_cluster_bc",
        df["nx_ens"] + pd.Series(bias_series, index=df.index),
        "nx_ens + cluster bias",
    )
    train_fitted.append({"k": best_k, "cols": cols, "standardization": meta, "bias": bias})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e37(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    nx = df["nx_ens"]
    pattern = np.column_stack([
        nx.groupby(ctx.group_station_asof).shift(k).to_numpy(dtype=float) for k in range(1, 8)
    ])
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    fallback = tfl.rolling_mean(
        df["actual_tmax_f"],
        window=3650,
        min_periods=1,
        lag=ctx.truth_lag,
        group_key=ctx.group_station,
    ).to_numpy(dtype=float)
    best_k = 5
    best_mae = float("inf")
    for k in [5, 10, 20]:
        stats = compute_knn_stats(
            pattern,
            y,
            ctx.group_station_asof,
            ctx.truth_lag,
            k,
            min_dims=7,
            fallback=fallback,
        )
        pred = stats["mean"]
        val_pred = pred[ctx.val_mask]
        val_true = y[ctx.val_mask]
        if len(val_true) == 0:
            continue
        mae = float(np.mean(np.abs(val_pred - val_true)))
        if mae < best_mae:
            best_mae = mae
            best_k = k
    stats = compute_knn_stats(
        pattern,
        y,
        ctx.group_station_asof,
        ctx.truth_lag,
        best_k,
        min_dims=7,
        fallback=fallback,
    )
    add_feature(features, formulas, "analog_traj_mean", pd.Series(stats["mean"], index=df.index), "traj mean")
    add_feature(features, formulas, "analog_traj_wmean", pd.Series(stats["wmean"], index=df.index), "traj wmean")
    add_feature(features, formulas, "analog_traj_std", pd.Series(stats["std"], index=df.index), "traj std")
    train_fitted.append({"k": best_k})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e38(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    nx = df["nx_ens"]
    pattern = np.column_stack([
        nx.groupby(ctx.group_station_asof).shift(k).to_numpy(dtype=float) for k in range(1, 8)
    ])
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    resid = df["resid_ens"].to_numpy(dtype=float)
    fallback = tfl.rolling_mean(
        df["resid_ens"],
        window=3650,
        min_periods=1,
        lag=ctx.truth_lag,
        group_key=ctx.group_station,
    ).to_numpy(dtype=float)
    best_k = 10
    best_mae = float("inf")
    for k in [10, 20]:
        stats = compute_knn_stats(
            pattern,
            resid,
            ctx.group_station_asof,
            ctx.truth_lag,
            k,
            min_dims=7,
            fallback=fallback,
        )
        pred = df["nx_ens"].to_numpy(dtype=float) + stats["wmean"]
        val_pred = pred[ctx.val_mask]
        val_true = y[ctx.val_mask]
        if len(val_true) == 0:
            continue
        mae = float(np.mean(np.abs(val_pred - val_true)))
        if mae < best_mae:
            best_mae = mae
            best_k = k
    stats = compute_knn_stats(
        pattern,
        resid,
        ctx.group_station_asof,
        ctx.truth_lag,
        best_k,
        min_dims=7,
        fallback=fallback,
    )
    add_feature(features, formulas, "analog_traj_resid_mean", pd.Series(stats["mean"], index=df.index), "traj resid mean")
    add_feature(features, formulas, "analog_traj_resid_wmean", pd.Series(stats["wmean"], index=df.index), "traj resid wmean")
    add_feature(features, formulas, "analog_traj_resid_iqr", pd.Series(stats["iqr"], index=df.index), "traj resid iqr")
    add_feature(
        features,
        formulas,
        "nx_traj_bc",
        df["nx_ens"] + pd.Series(stats["wmean"], index=df.index),
        "nx_ens + traj resid",
    )
    train_fitted.append({"k": best_k})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e39(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    q90_map, q90_default = station_quantile(df.loc[ctx.train_mask], "nx_ens", 0.90)
    q10_map, q10_default = station_quantile(df.loc[ctx.train_mask], "nx_ens", 0.10)
    hot_thr = map_station_threshold(df, q90_map, q90_default)
    cold_thr = map_station_threshold(df, q10_map, q10_default)
    hot_evt = (df["nx_ens"] >= hot_thr).astype(int)
    cold_evt = (df["nx_ens"] <= cold_thr).astype(int)
    days_hot = tfl.days_since_event(hot_evt, lag=1, cap=180, group_key=ctx.group_station_asof)
    days_cold = tfl.days_since_event(cold_evt, lag=1, cap=180, group_key=ctx.group_station_asof)
    count_hot = tfl.rolling_sum(hot_evt, window=30, min_periods=1, lag=1, group_key=ctx.group_station_asof)
    count_cold = tfl.rolling_sum(cold_evt, window=30, min_periods=1, lag=1, group_key=ctx.group_station_asof)
    hot_streak = tfl.streak_length(hot_evt, lag=1, cap=180, group_key=ctx.group_station_asof)
    cold_streak = tfl.streak_length(cold_evt, lag=1, cap=180, group_key=ctx.group_station_asof)
    add_feature(features, formulas, "days_since_hot", days_hot, "days since hot")
    add_feature(features, formulas, "days_since_cold", days_cold, "days since cold")
    add_feature(features, formulas, "count_hot_30", count_hot, "count hot 30")
    add_feature(features, formulas, "count_cold_30", count_cold, "count cold 30")
    add_feature(features, formulas, "hot_streak", hot_streak, "hot streak")
    add_feature(features, formulas, "cold_streak", cold_streak, "cold streak")
    train_fitted.append({"q90": q90_map, "q10": q10_map})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e40(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    wet_p_map, wet_p_default = station_quantile(df.loc[ctx.train_mask], "p12_ens", 0.90)
    wet_q_map, wet_q_default = station_quantile(df.loc[ctx.train_mask], "q12_ens", 0.90)
    cig_map, cig_default = station_quantile(df.loc[ctx.train_mask], "cig_ens_min", 0.10)
    vis_map, vis_default = station_quantile(df.loc[ctx.train_mask], "vis_ens_min", 0.10)
    wet_p_thr = map_station_threshold(df, wet_p_map, wet_p_default)
    wet_q_thr = map_station_threshold(df, wet_q_map, wet_q_default)
    low_cig_thr = map_station_threshold(df, cig_map, cig_default)
    low_vis_thr = map_station_threshold(df, vis_map, vis_default)
    wet_evt = ((_get_col(df, "p12_ens") >= wet_p_thr) | (df["q12_ens"] >= wet_q_thr)).astype(int)
    lowcig_evt = (df["cig_ens_min"] <= low_cig_thr).astype(int)
    lowvis_evt = (df["vis_ens_min"] <= low_vis_thr).astype(int)
    days_wet = tfl.days_since_event(wet_evt, lag=1, cap=180, group_key=ctx.group_station_asof)
    days_lowcig = tfl.days_since_event(lowcig_evt, lag=1, cap=180, group_key=ctx.group_station_asof)
    days_lowvis = tfl.days_since_event(lowvis_evt, lag=1, cap=180, group_key=ctx.group_station_asof)
    count_wet_30 = tfl.rolling_sum(wet_evt, window=30, min_periods=1, lag=1, group_key=ctx.group_station_asof)
    count_lowcig_30 = tfl.rolling_sum(lowcig_evt, window=30, min_periods=1, lag=1, group_key=ctx.group_station_asof)
    count_lowvis_30 = tfl.rolling_sum(lowvis_evt, window=30, min_periods=1, lag=1, group_key=ctx.group_station_asof)
    wet_streak = tfl.streak_length(wet_evt, lag=1, cap=180, group_key=ctx.group_station_asof)
    add_feature(features, formulas, "days_since_wet", days_wet, "days since wet")
    add_feature(features, formulas, "days_since_lowcig", days_lowcig, "days since low cig")
    add_feature(features, formulas, "days_since_lowvis", days_lowvis, "days since low vis")
    add_feature(features, formulas, "count_wet_30", count_wet_30, "count wet 30")
    add_feature(features, formulas, "count_lowcig_30", count_lowcig_30, "count lowcig 30")
    add_feature(features, formulas, "count_lowvis_30", count_lowvis_30, "count lowvis 30")
    add_feature(features, formulas, "wet_streak", wet_streak, "wet streak")
    train_fitted.append({"wet_p": wet_p_map, "wet_q": wet_q_map, "cig_q10": cig_map, "vis_q10": vis_map})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e41(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    resid_std = float(df.loc[ctx.train_mask, "resid_ens"].std(ddof=0)) if ctx.train_mask.any() else 1.0
    k = 0.25
    h = 5.0 * resid_std
    cusum = compute_cusum_features(df, ctx.group_station_asof, ctx.truth_lag, k=k, h=h)
    add_feature(features, formulas, "cpos", pd.Series(cusum["cpos"], index=df.index), "cusum pos")
    add_feature(features, formulas, "cneg", pd.Series(cusum["cneg"], index=df.index), "cusum neg")
    add_feature(features, formulas, "alarm_pos", pd.Series(cusum["alarm_pos"], index=df.index), "alarm pos")
    add_feature(features, formulas, "alarm_neg", pd.Series(cusum["alarm_neg"], index=df.index), "alarm neg")
    add_feature(features, formulas, "tsa_pos", pd.Series(cusum["tsa_pos"], index=df.index), "time since pos alarm")
    add_feature(features, formulas, "tsa_neg", pd.Series(cusum["tsa_neg"], index=df.index), "time since neg alarm")
    bias_shift = cusum["cpos"] - cusum["cneg"]
    add_feature(features, formulas, "bias_shift_score", pd.Series(bias_shift, index=df.index), "bias shift score")
    train_fitted.append({"k": k, "h": h})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e42(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    d = df["ad_nx"]
    m = tfl.rolling_mean(
        d,
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    delta = 0.05
    drift = np.full(len(df), np.nan, dtype=float)
    drift_min = np.full(len(df), np.nan, dtype=float)
    groups = ctx.group_station_asof.to_numpy()
    d_vals = d.to_numpy(dtype=float)
    m_vals = m.to_numpy(dtype=float)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        ph = 0.0
        ph_min = 0.0
        for pos, row_idx in enumerate(idx):
            if pos == 0:
                ph = 0.0
                ph_min = 0.0
            else:
                val = d_vals[row_idx]
                mean_val = m_vals[row_idx]
                if np.isfinite(val) and np.isfinite(mean_val):
                    ph = ph + (val - mean_val - delta)
                    ph_min = min(ph_min, ph)
            drift[row_idx] = ph
            drift_min[row_idx] = ph_min
    drift_score = drift - drift_min
    drift_series = pd.Series(drift_score, index=df.index)
    train_drift = drift_series[ctx.train_mask]
    q95_map = train_drift.groupby(df.loc[ctx.train_mask, "station_id"]).quantile(0.95).to_dict()
    q95_default = float(train_drift.quantile(0.95)) if len(train_drift) else 0.0
    lam = df["station_id"].map(q95_map).fillna(q95_default).astype(float)
    drift_flag = (drift_score > lam.to_numpy(dtype=float)).astype(float)
    add_feature(features, formulas, "drift_score", pd.Series(drift_score, index=df.index), "ph drift")
    add_feature(features, formulas, "drift_flag", pd.Series(drift_flag, index=df.index), "drift flag")
    train_fitted.append({"delta": delta, "lambda_q95": q95_map})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e43(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    s = np.sign(df["sd_nx"]).astype(float)
    frac_pos = tfl.rolling_mean(
        (s == 1).astype(float),
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    frac_neg = tfl.rolling_mean(
        (s == -1).astype(float),
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    runlen = tfl.streak_length(s, lag=1, cap=180, group_key=ctx.group_station_asof)
    dom = np.where(frac_pos.to_numpy() > 0.7, 1, np.where(frac_neg.to_numpy() > 0.7, -1, 0))
    add_feature(features, formulas, "frac_pos_30", frac_pos, "frac gfs>nam")
    add_feature(features, formulas, "frac_neg_30", frac_neg, "frac gfs<nam")
    add_feature(features, formulas, "runlen_sign", runlen, "runlen sign")
    add_feature(features, formulas, "dom_flag", pd.Series(dom, index=df.index), "dominance flag")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e44(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    r = df["resid_ens"]
    r_L = r.groupby(ctx.group_station_asof).shift(ctx.truth_lag)
    r_L1 = r.groupby(ctx.group_station_asof).shift(ctx.truth_lag + 1)
    r_L2 = r.groupby(ctx.group_station_asof).shift(ctx.truth_lag + 2)
    abs_r_L = r_L.abs()
    r_shift = r.groupby(ctx.group_station_asof).shift(ctx.truth_lag)
    phi = tfl.rolling_apply(
        r_shift,
        window=30,
        min_periods=min_periods(30),
        lag=0,
        func=rolling_ar1_phi,
        group_key=ctx.group_station_asof,
    )
    bias_ar = phi * r_L
    add_feature(features, formulas, "r_L", r_L, "resid lag L")
    add_feature(features, formulas, "r_L1", r_L1, "resid lag L+1")
    add_feature(features, formulas, "r_L2", r_L2, "resid lag L+2")
    add_feature(features, formulas, "abs_r_L", abs_r_L, "abs resid lag L")
    add_feature(features, formulas, "phi_30", phi, "ar1 phi")
    add_feature(features, formulas, "bias_ar", bias_ar, "phi * r_L")
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e45(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    cols = [
        "nx_ens",
        "tmp_ens",
        "dpt_ens",
        "p12_ens",
        "q12_ens",
        "wsp_ens",
        "cig_ens_mean",
        "vis_ens_mean",
    ]
    scaled, meta, _ = global_standardize(df, cols, ctx.train_mask)
    y = df["actual_tmax_f"].to_numpy(dtype=float)
    best_k = 2
    best_mae = float("inf")
    for k in [2, 3, 5]:
        pca = PCA(n_components=k, random_state=ctx.seed)
        pca.fit(scaled[ctx.train_mask])
        pcs = pca.transform(scaled)
        model = Ridge(alpha=1.0)
        model.fit(pcs[ctx.train_mask], y[ctx.train_mask])
        pred = model.predict(pcs[ctx.val_mask])
        if len(pred) == 0:
            continue
        mae = float(np.mean(np.abs(pred - y[ctx.val_mask])))
        if mae < best_mae:
            best_mae = mae
            best_k = k
    pca = PCA(n_components=best_k, random_state=ctx.seed)
    pca.fit(scaled[ctx.train_mask])
    pcs = pca.transform(scaled)
    for i in range(best_k):
        pc = pd.Series(pcs[:, i], index=df.index)
        add_feature(features, formulas, f"pc_{i+1}", pc, "pc")
        mean_30 = tfl.rolling_mean(
            pc,
            window=30,
            min_periods=min_periods(30),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        std_30 = tfl.rolling_std(
            pc,
            window=30,
            min_periods=min_periods(30),
            lag=1,
            group_key=ctx.group_station_asof,
        )
        z = safe_divide((pc - mean_30).to_numpy(), std_30.to_numpy() + EPS, default=np.nan)
        add_feature(features, formulas, f"pc_{i+1}_mean_30", mean_30, "pc mean 30")
        add_feature(features, formulas, f"pc_{i+1}_std_30", std_30, "pc std 30")
        add_feature(features, formulas, f"pc_{i+1}_z", pd.Series(z, index=df.index), "pc z")
    train_fitted.append({"k": best_k, "cols": cols, "standardization": meta})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e46(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    cols = [
        "nx_ens",
        "tmp_ens",
        "dpt_ens",
        "p12_ens",
        "q12_ens",
        "wsp_ens",
        "cig_ens_mean",
        "vis_ens_mean",
    ]
    scaled, meta, _ = global_standardize(df, cols, ctx.train_mask)
    resid = df["resid_ens"].to_numpy(dtype=float)
    best_k = 6
    best_mae = float("inf")
    for k in [6, 8, 12]:
        km = KMeans(n_clusters=k, random_state=ctx.seed, n_init=10)
        km.fit(scaled[ctx.train_mask])
        cluster_id = km.predict(scaled)
        bias = {}
        for cid in range(k):
            mask = ctx.train_mask & (cluster_id == cid)
            if mask.any():
                bias[cid] = float(np.mean(resid[mask]))
        bias_default = float(np.mean(resid[ctx.train_mask])) if ctx.train_mask.any() else 0.0
        bias_series = np.array([bias.get(cid, bias_default) for cid in cluster_id], dtype=float)
        pred = df["nx_ens"].to_numpy(dtype=float) + bias_series
        val_pred = pred[ctx.val_mask]
        val_true = df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
        if len(val_true) == 0:
            continue
        mae = float(np.mean(np.abs(val_pred - val_true)))
        if mae < best_mae:
            best_mae = mae
            best_k = k
    km = KMeans(n_clusters=best_k, random_state=ctx.seed, n_init=10)
    km.fit(scaled[ctx.train_mask])
    cluster_id = km.predict(scaled)
    prev_cluster = pd.Series(cluster_id, index=df.index).groupby(ctx.group_station_asof).shift(1)
    runlen = tfl.streak_length(pd.Series(cluster_id, index=df.index), lag=1, cap=180, group_key=ctx.group_station_asof)
    # Transition matrix on training
    trans_counts = {}
    for g in np.unique(ctx.group_station_asof.to_numpy()):
        idx = np.where(ctx.group_station_asof.to_numpy() == g)[0]
        for pos in range(1, len(idx)):
            if not ctx.train_mask[idx[pos]]:
                continue
            prev = cluster_id[idx[pos - 1]]
            curr = cluster_id[idx[pos]]
            key = (prev, curr)
            trans_counts[key] = trans_counts.get(key, 0) + 1
    totals = {}
    for (prev, _), count in trans_counts.items():
        totals[prev] = totals.get(prev, 0) + count
    surprise = np.full(len(df), np.nan, dtype=float)
    for i, cid in enumerate(cluster_id):
        prev = prev_cluster.iloc[i]
        if pd.isna(prev):
            continue
        prev = int(prev)
        key = (prev, cid)
        prob = trans_counts.get(key, 0) / max(totals.get(prev, 0), 1)
        surprise[i] = -np.log(prob + EPS)
    add_feature(features, formulas, "cluster_id", pd.Series(cluster_id, index=df.index), "cluster id")
    add_feature(features, formulas, "prev_cluster", prev_cluster, "prev cluster")
    add_feature(features, formulas, "runlen_cluster", runlen, "cluster runlen")
    add_feature(features, formulas, "cluster_surprise", pd.Series(surprise, index=df.index), "transition surprise")
    train_fitted.append({"k": best_k, "cols": cols, "standardization": meta})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e47(ctx: ExperimentContext) -> DerivedFeatureSet:
    return DerivedFeatureSet(features=pd.DataFrame(index=ctx.df.index), formulas=[], train_fitted=[])


def exp_e48(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    resid = df["resid_ens"]
    b_s = rolling_mean_fallback(
        resid, 30, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    count_s = tfl.rolling_sum(
        resid.notna().astype(float),
        window=30,
        min_periods=1,
        lag=ctx.truth_lag,
        group_key=ctx.group_station_asof,
    )
    b_g = tfl.rolling_mean(
        resid,
        window=30,
        min_periods=min_periods(30),
        lag=ctx.truth_lag,
        group_key=df["asof_hour"],
    )
    best_tau = 5
    best_mae = float("inf")
    for tau in [5, 10, 20, 50]:
        w = count_s.to_numpy() / (count_s.to_numpy() + tau)
        b_shrunk = w * b_s.to_numpy() + (1.0 - w) * b_g.to_numpy()
        pred = df["nx_ens"].to_numpy(dtype=float) + b_shrunk
        val_pred = pred[ctx.val_mask]
        val_true = df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
        if len(val_true) == 0:
            continue
        mae = float(np.mean(np.abs(val_pred - val_true)))
        if mae < best_mae:
            best_mae = mae
            best_tau = tau
    w = count_s.to_numpy() / (count_s.to_numpy() + best_tau)
    b_shrunk = w * b_s.to_numpy() + (1.0 - w) * b_g.to_numpy()
    add_feature(features, formulas, "b_shrunk", pd.Series(b_shrunk, index=df.index), "shrunk bias")
    add_feature(features, formulas, "nx_shrunk_bc", df["nx_ens"] + pd.Series(b_shrunk, index=df.index), "nx_ens + shrunk bias")
    train_fitted.append({"tau": best_tau})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e49(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    bias_ens = tfl.ewm_mean(
        df["resid_ens"],
        halflife=14,
        min_periods=10,
        lag=ctx.truth_lag,
        group_key=ctx.group_station_asof,
    ).fillna(
        tfl.ewm_mean(
            df["resid_ens"],
            halflife=14,
            min_periods=10,
            lag=ctx.truth_lag,
            group_key=ctx.group_station,
        )
    )
    p = df["nx_ens"].to_numpy(dtype=float) + bias_ens.to_numpy(dtype=float)
    iso = compute_isotonic_features(
        df,
        ctx.group_station_asof,
        ctx.truth_lag,
        horizon_days=180,
        min_pairs=50,
        p=p,
    )
    p_iso = iso["p_iso"]
    delta = p_iso - p
    add_feature(features, formulas, "p_base", pd.Series(p, index=df.index), "base predictor")
    add_feature(features, formulas, "p_iso", pd.Series(p_iso, index=df.index), "isotonic calibrated")
    add_feature(features, formulas, "calib_delta", pd.Series(delta, index=df.index), "p_iso - p")
    add_feature(features, formulas, "slope_local", pd.Series(iso["slope_local"], index=df.index), "local slope")
    train_fitted.append({"halflife": 14, "horizon_days": 180, "min_pairs": 50})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def exp_e50(ctx: ExperimentContext) -> DerivedFeatureSet:
    df = ctx.df
    features = pd.DataFrame(index=df.index)
    formulas: list[dict] = []
    train_fitted: list[dict] = []
    base_pred = df["nx_ens"].to_numpy(dtype=float)
    best_q = 0.01
    best_r = 1.0
    best_mae = float("inf")
    for q in [0.01, 0.05, 0.1, 0.25]:
        for r in [1.0, 2.0, 4.0, 9.0]:
            kf = compute_kalman_features(
                df, ctx.group_station_asof, ctx.truth_lag, q, r, base_pred
            )
            pred = kf["p_kf"]
            val_pred = pred[ctx.val_mask]
            val_true = df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
            if len(val_true) == 0:
                continue
            mae = float(np.mean(np.abs(val_pred - val_true)))
            if mae < best_mae:
                best_mae = mae
                best_q = q
                best_r = r
    kf = compute_kalman_features(
        df, ctx.group_station_asof, ctx.truth_lag, best_q, best_r, base_pred
    )
    add_feature(features, formulas, "b_hat", pd.Series(kf["b_hat"], index=df.index), "kalman bias")
    add_feature(features, formulas, "p_var", pd.Series(kf["p_var"], index=df.index), "kalman p")
    add_feature(features, formulas, "k_gain", pd.Series(kf["k_gain"], index=df.index), "kalman gain")
    add_feature(features, formulas, "p_kf", pd.Series(kf["p_kf"], index=df.index), "kalman predictor")
    train_fitted.append({"q": best_q, "r": best_r})
    return DerivedFeatureSet(features=features, formulas=formulas, train_fitted=train_fitted)


def merge_cols(*groups: list[str]) -> list[str]:
    seen = set()
    output: list[str] = []
    for group in groups:
        for col in group:
            if col not in seen:
                seen.add(col)
                output.append(col)
    return output


def with_calendar(cols: list[str]) -> list[str]:
    return merge_cols(cols, CALENDAR_COLS)


def base_columns_all_mos(df: pd.DataFrame) -> list[str]:
    cols = [
        col
        for col in df.columns
        if col.startswith(("gfs_", "nam_")) and col not in {"station_id", "target_date_local", "asof_utc"}
    ]
    cols = [col for col in cols if col != "actual_tmax_f"]
    return cols


def run_standard_experiment(
    exp_def: ExperimentDefinition,
    ctx: ExperimentContext,
    run_dir: Path,
    model_name: str,
) -> dict:
    df = ctx.df
    derived = exp_def.build_features(ctx)
    base_cols = exp_def.base_cols
    feature_df = df[base_cols].copy()
    feature_df = pd.concat([feature_df, derived.features], axis=1)
    feature_df, impute_meta = impute_features(feature_df, ctx.train_mask)

    X_train = feature_df.loc[ctx.train_mask].to_numpy(dtype=float)
    y_train = df.loc[ctx.train_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_val = feature_df.loc[ctx.val_mask].to_numpy(dtype=float)
    y_val = df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_test = feature_df.loc[ctx.test_mask].to_numpy(dtype=float)
    y_test = df.loc[ctx.test_mask, "actual_tmax_f"].to_numpy(dtype=float)

    model = train_model(model_name, ctx.seed)
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val) if len(X_val) else np.array([])
    pred_test = model.predict(X_test)

    metrics_summary = {
        "train": regression_metrics(y_train, pred_train),
        "validation": regression_metrics(y_val, pred_val) if len(y_val) else None,
        "test": regression_metrics(y_test, pred_test),
        "per_station_test": per_station_metrics(df.loc[ctx.test_mask], y_test, pred_test),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "metrics.json", metrics_summary)
    write_json(run_dir / "feature_list.json", list(feature_df.columns))
    write_json(run_dir / "derived_features.json", derived.formulas)
    write_json(run_dir / "train_fitted.json", derived.train_fitted)
    write_json(run_dir / "impute_meta.json", impute_meta)
    write_predictions(run_dir / "predictions_test.csv", df.loc[ctx.test_mask], y_test, pred_test)
    write_json(
        run_dir / "experiment_meta.json",
        {
            "experiment_id": exp_def.experiment_id,
            "name": exp_def.name,
            "base_cols": base_cols,
            "model": model_name,
        },
    )

    return {
        "experiment_id": exp_def.experiment_id,
        "name": exp_def.name,
        "metrics": metrics_summary,
        "run_dir": str(run_dir),
        "_y_test": y_test,
        "_pred_test": pred_test,
    }


def run_e10(ctx: ExperimentContext, run_dir: Path, model_name: str) -> dict:
    df = ctx.df
    stage1_cols = with_calendar(base_columns_all_mos(df))
    stage1_features = df[stage1_cols].copy()
    stage1_features, stage1_impute = impute_features(stage1_features, ctx.train_mask)

    X_train = stage1_features.loc[ctx.train_mask].to_numpy(dtype=float)
    y_train = df.loc[ctx.train_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_val = stage1_features.loc[ctx.val_mask].to_numpy(dtype=float)
    y_val = df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float)
    X_test = stage1_features.loc[ctx.test_mask].to_numpy(dtype=float)
    y_test = df.loc[ctx.test_mask, "actual_tmax_f"].to_numpy(dtype=float)

    model_stage1 = train_model(model_name, ctx.seed)
    model_stage1.fit(X_train, y_train)
    pred_train_stage1 = model_stage1.predict(X_train)
    pred_val_stage1 = model_stage1.predict(X_val) if len(X_val) else np.array([])

    X_train_full = (
        np.vstack([X_train, X_val]) if len(X_val) else X_train
    )
    y_train_full = (
        np.concatenate([y_train, y_val]) if len(y_val) else y_train
    )
    model_stage1_full = train_model(model_name, ctx.seed)
    model_stage1_full.fit(X_train_full, y_train_full)
    pred_test_stage1 = model_stage1_full.predict(X_test)

    oof_pred = np.full(len(df), np.nan, dtype=float)
    train_indices = np.where(ctx.train_mask)[0]
    index_to_pos = {idx: pos for pos, idx in enumerate(train_indices)}
    splits = expanding_time_splits(df.loc[ctx.train_mask])
    for train_idx, val_idx in splits:
        train_pos = [index_to_pos[i] for i in train_idx if i in index_to_pos]
        val_pos = [index_to_pos[i] for i in val_idx if i in index_to_pos]
        if not train_pos or not val_pos:
            continue
        model_fold = train_model(model_name, ctx.seed)
        model_fold.fit(X_train[train_pos], y_train[train_pos])
        oof_pred[val_idx] = model_fold.predict(X_train[val_pos])

    oof_pred[train_indices] = np.where(
        np.isfinite(oof_pred[train_indices]), oof_pred[train_indices], pred_train_stage1
    )

    # Stage2 features (E01 + E09 + E16 core)
    r_ens = df["resid_ens"]
    bias_ens_30 = rolling_mean_fallback(
        r_ens, 30, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    mae_ens_30 = rolling_mean_fallback(
        r_ens.abs(), 30, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    delta = df["nx_ens"].groupby(ctx.group_station_asof).diff()
    vol_change_7 = tfl.rolling_std(
        delta,
        window=7,
        min_periods=min_periods(7),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    vol_change_30 = tfl.rolling_std(
        delta,
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    mean_sd_30 = tfl.rolling_mean(
        df["sd_nx"],
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    std_sd_30 = tfl.rolling_std(
        df["sd_nx"],
        window=30,
        min_periods=min_periods(30),
        lag=1,
        group_key=ctx.group_station_asof,
    )

    stage2_features = pd.DataFrame(
        {
            "bias_ens_30": bias_ens_30,
            "mae_ens_30": mae_ens_30,
            "vol_change_7": vol_change_7,
            "vol_change_30": vol_change_30,
            "mean_sd_30": mean_sd_30,
            "std_sd_30": std_sd_30,
        },
        index=df.index,
    )
    stage2_features, stage2_impute = impute_features(stage2_features, ctx.train_mask)

    X2_train = stage2_features.loc[ctx.train_mask].to_numpy(dtype=float)
    X2_val = stage2_features.loc[ctx.val_mask].to_numpy(dtype=float)
    X2_test = stage2_features.loc[ctx.test_mask].to_numpy(dtype=float)

    resid_train = y_train - oof_pred[ctx.train_mask]
    model_stage2 = train_model(model_name, ctx.seed)
    model_stage2.fit(X2_train, resid_train)

    pred_train_stage2 = model_stage2.predict(X2_train)
    pred_val_stage2 = model_stage2.predict(X2_val) if len(X2_val) else np.array([])
    pred_test_stage2 = model_stage2.predict(X2_test)

    final_train = oof_pred[ctx.train_mask] + pred_train_stage2
    final_val = pred_val_stage1 + pred_val_stage2 if len(pred_val_stage1) else np.array([])
    final_test = pred_test_stage1 + pred_test_stage2

    metrics_summary = {
        "train": regression_metrics(y_train, final_train),
        "validation": regression_metrics(y_val, final_val) if len(y_val) else None,
        "test": regression_metrics(y_test, final_test),
        "per_station_test": per_station_metrics(df.loc[ctx.test_mask], y_test, final_test),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "metrics.json", metrics_summary)
    write_json(run_dir / "feature_list.json", list(stage2_features.columns))
    write_json(run_dir / "derived_features.json", [{"name": name} for name in stage2_features.columns])
    write_json(run_dir / "train_fitted.json", [])
    write_json(run_dir / "impute_meta.json", {"stage1": stage1_impute, "stage2": stage2_impute})
    write_predictions(run_dir / "predictions_test.csv", df.loc[ctx.test_mask], y_test, final_test)
    write_json(
        run_dir / "experiment_meta.json",
        {
            "experiment_id": "E10",
            "name": "Two-stage stacking",
            "stage1_cols": stage1_cols,
            "stage2_cols": list(stage2_features.columns),
            "model": model_name,
        },
    )

    return {
        "experiment_id": "E10",
        "name": "Two-stage stacking",
        "metrics": metrics_summary,
        "run_dir": str(run_dir),
        "_y_test": y_test,
        "_pred_test": final_test,
    }


def run_e47(ctx: ExperimentContext, run_dir: Path, model_name: str) -> dict:
    df = ctx.df
    nx = df["nx_ens"]
    delta = nx.groupby(ctx.group_station_asof).diff()
    vol = tfl.rolling_std(
        delta,
        window=7,
        min_periods=min_periods(7),
        lag=1,
        group_key=ctx.group_station_asof,
    )
    train_vol = vol[ctx.train_mask]
    q75_map = train_vol.groupby(df.loc[ctx.train_mask, "station_id"]).quantile(0.75).to_dict()
    q75_default = float(train_vol.quantile(0.75)) if len(train_vol) else 0.0
    v_thr = df["station_id"].map(q75_map).fillna(q75_default).astype(float)
    gate = (vol > v_thr).astype(int)

    r_ens = df["resid_ens"]
    bias_ens_30 = rolling_mean_fallback(
        r_ens, 30, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    mae_ens_30 = rolling_mean_fallback(
        r_ens.abs(), 30, ctx.truth_lag, ctx.group_station_asof, ctx.group_station
    )
    corr_features = pd.DataFrame(
        {
            "bias_ens_30": bias_ens_30,
            "mae_ens_30": mae_ens_30,
        },
        index=df.index,
    )
    corr_features, impute_meta = impute_features(corr_features, ctx.train_mask)

    X_train = corr_features.loc[ctx.train_mask].to_numpy(dtype=float)
    y_train = r_ens.loc[ctx.train_mask].to_numpy(dtype=float)
    X_val = corr_features.loc[ctx.val_mask].to_numpy(dtype=float)
    y_val = r_ens.loc[ctx.val_mask].to_numpy(dtype=float)
    X_test = corr_features.loc[ctx.test_mask].to_numpy(dtype=float)
    y_test = df.loc[ctx.test_mask, "actual_tmax_f"].to_numpy(dtype=float)

    gate_train = gate[ctx.train_mask].to_numpy(dtype=int)
    gate_val = gate[ctx.val_mask].to_numpy(dtype=int)
    gate_test = gate[ctx.test_mask].to_numpy(dtype=int)

    model_stable = train_model(model_name, ctx.seed)
    model_vol = train_model(model_name, ctx.seed)

    mask_stable = gate_train == 0
    mask_vol = gate_train == 1

    if mask_stable.sum() < 20 or mask_vol.sum() < 20:
        model_stable.fit(X_train, y_train)
        model_vol = model_stable
    else:
        model_stable.fit(X_train[mask_stable], y_train[mask_stable])
        model_vol.fit(X_train[mask_vol], y_train[mask_vol])

    resid_pred_train = np.where(
        mask_vol,
        model_vol.predict(X_train),
        model_stable.predict(X_train),
    )
    resid_pred_val = np.where(
        gate_val == 1,
        model_vol.predict(X_val),
        model_stable.predict(X_val),
    ) if len(X_val) else np.array([])
    resid_pred_test = np.where(
        gate_test == 1,
        model_vol.predict(X_test),
        model_stable.predict(X_test),
    )

    pred_train = df.loc[ctx.train_mask, "nx_ens"].to_numpy(dtype=float) + resid_pred_train
    pred_val = df.loc[ctx.val_mask, "nx_ens"].to_numpy(dtype=float) + resid_pred_val if len(resid_pred_val) else np.array([])
    pred_test = df.loc[ctx.test_mask, "nx_ens"].to_numpy(dtype=float) + resid_pred_test

    metrics_summary = {
        "train": regression_metrics(df.loc[ctx.train_mask, "actual_tmax_f"].to_numpy(dtype=float), pred_train),
        "validation": regression_metrics(df.loc[ctx.val_mask, "actual_tmax_f"].to_numpy(dtype=float), pred_val) if len(pred_val) else None,
        "test": regression_metrics(y_test, pred_test),
        "per_station_test": per_station_metrics(df.loc[ctx.test_mask], y_test, pred_test),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "metrics.json", metrics_summary)
    write_json(run_dir / "feature_list.json", list(corr_features.columns))
    write_json(run_dir / "derived_features.json", [{"name": name} for name in corr_features.columns])
    write_json(run_dir / "train_fitted.json", [])
    write_json(run_dir / "impute_meta.json", impute_meta)
    write_predictions(run_dir / "predictions_test.csv", df.loc[ctx.test_mask], y_test, pred_test)
    write_json(
        run_dir / "experiment_meta.json",
        {
            "experiment_id": "E47",
            "name": "Mixture of experts",
            "model": model_name,
        },
    )

    return {
        "experiment_id": "E47",
        "name": "Mixture of experts",
        "metrics": metrics_summary,
        "run_dir": str(run_dir),
        "_y_test": y_test,
        "_pred_test": pred_test,
    }


def build_experiments() -> list[ExperimentDefinition]:
    base_nx = ["gfs_n_x_mean", "nam_n_x_mean"]
    base_tmp = ["gfs_tmp_mean", "nam_tmp_mean"]
    base_dpt = ["gfs_dpt_mean", "nam_dpt_mean"]
    base_p06 = ["gfs_p06_mean", "nam_p06_mean"]
    base_p12 = ["gfs_p12_mean", "nam_p12_mean"]
    base_pos = ["gfs_pos_mean", "nam_pos_mean"]
    base_poz = ["gfs_poz_mean", "nam_poz_mean"]
    base_q06 = ["gfs_q06_mean", "nam_q06_mean"]
    base_q12 = ["gfs_q12_mean", "nam_q12_mean"]
    base_snw = ["gfs_snw_mean", "nam_snw_mean"]
    base_t06 = ["gfs_t06_mean", "nam_t06_mean"]
    base_t06_1 = ["gfs_t06_1_mean", "nam_t06_1_mean"]
    base_t06_2 = ["gfs_t06_2_mean", "nam_t06_2_mean"]
    base_wsp = ["gfs_wsp_mean", "nam_wsp_mean"]
    base_wdr = ["gfs_wdr_mean", "nam_wdr_mean"]
    base_cig_min = ["gfs_cig_min", "nam_cig_min"]
    base_cig_mean = ["gfs_cig_mean", "nam_cig_mean"]
    base_vis_min = ["gfs_vis_min", "nam_vis_min"]
    base_vis_mean = ["gfs_vis_mean", "nam_vis_mean"]
    base_tmp_stats = [
        "gfs_tmp_min",
        "gfs_tmp_max",
        "gfs_tmp_mean",
        "gfs_tmp_median",
        "nam_tmp_min",
        "nam_tmp_max",
        "nam_tmp_mean",
        "nam_tmp_median",
    ]
    base_nx_stats = [
        "gfs_n_x_min",
        "gfs_n_x_max",
        "gfs_n_x_mean",
        "gfs_n_x_median",
        "nam_n_x_min",
        "nam_n_x_max",
        "nam_n_x_mean",
        "nam_n_x_median",
    ]

    return [
        ExperimentDefinition("E01", "Rolling residual bias & skill", with_calendar(base_nx), exp_e01),
        ExperimentDefinition("E02", "EWMA residual bias filters", with_calendar(base_nx), exp_e02),
        ExperimentDefinition("E03", "Rolling-skill dynamic ensemble", with_calendar(base_nx), exp_e03),
        ExperimentDefinition("E04", "Online hedge weights", with_calendar(base_nx), exp_e04),
        ExperimentDefinition("E05", "Disagreement gated blend", with_calendar(base_nx), exp_e05),
        ExperimentDefinition("E06", "Rolling residual quantiles", with_calendar(base_nx), exp_e06),
        ExperimentDefinition("E07", "Seasonal bias + drift", with_calendar(base_nx), exp_e07),
        ExperimentDefinition("E08", "As-of conditional memory", with_calendar(base_nx), exp_e08),
        ExperimentDefinition("E09", "Forecast volatility proxies", with_calendar(merge_cols(base_nx, base_tmp)), exp_e09),
        ExperimentDefinition("E10", "Two-stage stacking", with_calendar(base_nx), exp_e10),
        ExperimentDefinition("E11", "Multi-scale anomalies", with_calendar(merge_cols(base_nx, base_tmp)), exp_e11),
        ExperimentDefinition("E12", "Rolling trend slopes", with_calendar(merge_cols(base_nx, base_tmp)), exp_e12),
        ExperimentDefinition("E13", "Reversal & choppiness", with_calendar(base_nx), exp_e13),
        ExperimentDefinition("E14", "Volatility of volatility", with_calendar(base_nx), exp_e14),
        ExperimentDefinition("E15", "Diurnal shape regimes", with_calendar(merge_cols(base_tmp_stats, base_nx_stats)), exp_e15),
        ExperimentDefinition("E16", "Disagreement dynamics", with_calendar(base_nx), exp_e16),
        ExperimentDefinition("E17", "Vector disagreement norms", with_calendar(merge_cols(base_nx, base_tmp, base_dpt, base_wsp, base_p12, base_q12)), exp_e17),
        ExperimentDefinition("E18", "Coupling correlations", with_calendar(merge_cols(base_nx, base_tmp, base_dpt, base_p12, base_q12)), exp_e18),
        ExperimentDefinition("E19", "Self-consistency residual", with_calendar(merge_cols(base_nx, base_tmp, base_dpt, base_wsp)), exp_e19),
        ExperimentDefinition("E20", "Wind regime deviation", with_calendar(merge_cols(base_nx, base_wdr, base_wsp)), exp_e20),
        ExperimentDefinition("E21", "Dewpoint dep bins", with_calendar(merge_cols(base_nx, base_tmp, base_dpt)), exp_e21),
        ExperimentDefinition("E22", "Wet/dry conditional bias", with_calendar(merge_cols(base_nx, base_p12, base_q12)), exp_e22),
        ExperimentDefinition("E23", "Convective regime bias", with_calendar(merge_cols(base_nx, base_t06, base_t06_1, base_t06_2)), exp_e23),
        ExperimentDefinition("E24", "Wintry regime bias", with_calendar(merge_cols(base_nx, base_pos, base_poz, base_snw)), exp_e24),
        ExperimentDefinition("E25", "Ceiling/visibility regime", with_calendar(merge_cols(base_nx, base_cig_min, base_vis_min)), exp_e25),
        ExperimentDefinition("E26", "Wind-speed bins", with_calendar(merge_cols(base_nx, base_wsp)), exp_e26),
        ExperimentDefinition("E27", "Humid/cloudy regime", with_calendar(merge_cols(base_nx, base_tmp, base_dpt, base_p12, base_cig_min)), exp_e27),
        ExperimentDefinition("E28", "Regime run-length", with_calendar(merge_cols(base_nx, base_p12, base_q12)), exp_e28),
        ExperimentDefinition("E29", "Precip/cloud anomalies", with_calendar(merge_cols(base_nx, base_p12, base_q12, base_cig_min, base_vis_min)), exp_e29),
        ExperimentDefinition("E30", "Humidity/wind shifts", with_calendar(merge_cols(base_nx, base_tmp, base_dpt, base_wsp)), exp_e30),
        ExperimentDefinition("E31", "KNN analog temp-only", with_calendar(base_nx), exp_e31),
        ExperimentDefinition("E32", "KNN analog residual", with_calendar(base_nx), exp_e32),
        ExperimentDefinition("E33", "KNN analog full MOS", with_calendar(base_nx), exp_e33),
        ExperimentDefinition("E34", "Seasonally constrained analogs", with_calendar(base_nx), exp_e34),
        ExperimentDefinition("E35", "Skill-analog weights", with_calendar(base_nx), exp_e35),
        ExperimentDefinition("E36", "Cluster residual correction", with_calendar(base_nx), exp_e36),
        ExperimentDefinition("E37", "Trajectory analog", with_calendar(base_nx), exp_e37),
        ExperimentDefinition("E38", "Trajectory residual", with_calendar(base_nx), exp_e38),
        ExperimentDefinition("E39", "Extreme temp memory", with_calendar(base_nx), exp_e39),
        ExperimentDefinition("E40", "Extreme wet/cloud memory", with_calendar(merge_cols(base_nx, base_p12, base_q12, base_cig_min, base_vis_min)), exp_e40),
        ExperimentDefinition("E41", "CUSUM bias shift", with_calendar(base_nx), exp_e41),
        ExperimentDefinition("E42", "Page-Hinkley drift", with_calendar(base_nx), exp_e42),
        ExperimentDefinition("E43", "Model dominance persistence", with_calendar(base_nx), exp_e43),
        ExperimentDefinition("E44", "Residual AR(1)", with_calendar(base_nx), exp_e44),
        ExperimentDefinition("E45", "PCA regime factors", with_calendar(merge_cols(base_nx, base_tmp, base_dpt, base_p12, base_q12, base_wsp, base_cig_mean, base_vis_mean)), exp_e45),
        ExperimentDefinition("E46", "Cluster transitions", with_calendar(merge_cols(base_nx, base_tmp, base_dpt, base_p12, base_q12, base_wsp, base_cig_mean, base_vis_mean)), exp_e46),
        ExperimentDefinition("E47", "Mixture of experts", with_calendar(base_nx), exp_e47),
        ExperimentDefinition("E48", "Shrinkage bias", with_calendar(base_nx), exp_e48),
        ExperimentDefinition("E49", "Isotonic calibration", with_calendar(base_nx), exp_e49),
        ExperimentDefinition("E50", "Kalman bias filter", with_calendar(base_nx), exp_e50),
    ]


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run MOS experiment suite E01-E50")
    parser.add_argument(
        "--csv",
        default=str(
            Path("ingestion-service")
            / "src"
            / "main"
            / "resources"
            / "trainingdata_output"
            / "KMIA_mos_training_data.csv"
        ),
        help="Path to MOS training CSV",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path("artifacts") / "experiments"),
        help="Root output directory",
    )
    parser.add_argument("--suite-id", help="Optional suite id override")
    parser.add_argument("--model", default="lgbm", help="Model name (lgbm, ridge, gbr)")
    parser.add_argument("--truth-lag", type=int, default=DEFAULT_TRUTH_LAG)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-ids", nargs="*", help="Optional experiment ids")
    args = parser.parse_args(argv)

    df = load_mos_csv(args.csv)
    required_cols = [
        "gfs_n_x_mean",
        "nam_n_x_mean",
        "gfs_tmp_mean",
        "nam_tmp_mean",
        "gfs_dpt_mean",
        "nam_dpt_mean",
        "gfs_p06_mean",
        "nam_p06_mean",
        "gfs_p12_mean",
        "nam_p12_mean",
        "gfs_pos_mean",
        "nam_pos_mean",
        "gfs_poz_mean",
        "nam_poz_mean",
        "gfs_q06_mean",
        "nam_q06_mean",
        "gfs_q12_mean",
        "nam_q12_mean",
        "gfs_snw_mean",
        "nam_snw_mean",
        "gfs_t06_mean",
        "nam_t06_mean",
        "gfs_t06_1_mean",
        "nam_t06_1_mean",
        "gfs_t06_2_mean",
        "nam_t06_2_mean",
        "gfs_cig_min",
        "nam_cig_min",
        "gfs_cig_mean",
        "nam_cig_mean",
        "gfs_vis_min",
        "nam_vis_min",
        "gfs_vis_mean",
        "nam_vis_mean",
        "gfs_wdr_mean",
        "nam_wdr_mean",
        "gfs_wsp_mean",
        "nam_wsp_mean",
        "gfs_tmp_min",
        "gfs_tmp_max",
        "gfs_tmp_median",
        "nam_tmp_min",
        "nam_tmp_max",
        "nam_tmp_median",
        "gfs_n_x_min",
        "gfs_n_x_max",
        "gfs_n_x_median",
        "nam_n_x_min",
        "nam_n_x_max",
        "nam_n_x_median",
    ]
    df = ensure_columns(df, required_cols)
    df = prepare_frame(df)

    split = split_by_date(df)
    train_mask = split.pop("train_mask")
    val_mask = split.pop("val_mask")
    test_mask = split.pop("test_mask")

    group_station = df["station_id"].astype(str)
    group_station_asof = df["station_id"].astype(str) + "_" + df["asof_hour"].astype(str)
    ctx = ExperimentContext(
        df=df,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        group_station=group_station,
        group_station_asof=group_station_asof,
        truth_lag=args.truth_lag,
        seed=args.seed,
    )

    suite_id = args.suite_id or default_suite_id()
    output_root = Path(args.output_root) / suite_id
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "split_info.json", split)

    experiments = build_experiments()
    if args.experiment_ids:
        exp_set = {eid.upper() for eid in args.experiment_ids}
        experiments = [exp for exp in experiments if exp.experiment_id.upper() in exp_set]

    results: list[dict] = []

    baseline_cols = [
        col
        for col in df.columns
        if col.startswith(("gfs_", "nam_")) and col.endswith("_mean")
    ]
    baseline_cols = with_calendar(baseline_cols)
    baseline_def = ExperimentDefinition("BASE", "Baseline MOS means", baseline_cols, lambda c: DerivedFeatureSet(pd.DataFrame(index=c.df.index), [], []))
    baseline_result = run_standard_experiment(baseline_def, ctx, output_root / "BASE", args.model)
    results.append(baseline_result)
    LOGGER.info("Completed %s - %s", baseline_result["experiment_id"], baseline_result["name"])

    for exp in experiments:
        run_dir = output_root / exp.experiment_id
        if exp.experiment_id == "E10":
            result = run_e10(ctx, run_dir, args.model)
            results.append(result)
            LOGGER.info("Completed %s - %s", result["experiment_id"], result["name"])
            continue
        if exp.experiment_id == "E47":
            result = run_e47(ctx, run_dir, args.model)
            results.append(result)
            LOGGER.info("Completed %s - %s", result["experiment_id"], result["name"])
            continue
        result = run_standard_experiment(exp, ctx, run_dir, args.model)
        results.append(result)
        LOGGER.info("Completed %s - %s", result["experiment_id"], result["name"])

    baseline_entry = results[0]
    base_mae = baseline_entry["metrics"]["test"]["mae"]
    base_pred = baseline_entry.get("_pred_test")
    base_y = baseline_entry.get("_y_test")
    for entry in results:
        entry["delta_test_mae"] = float(entry["metrics"]["test"]["mae"] - base_mae)
        if entry is baseline_entry:
            entry["bootstrap"] = None
        else:
            entry["bootstrap"] = bootstrap_mae_delta(
                entry.get("_y_test"),
                entry.get("_pred_test"),
                base_pred,
                df.loc[test_mask, "target_date_local"],
                args.bootstrap_samples,
                args.seed,
            )
        entry.pop("_y_test", None)
        entry.pop("_pred_test", None)

    best = min(
        [e for e in results if e["experiment_id"] != "BASE"],
        key=lambda e: e["metrics"]["test"]["mae"],
    )
    summary = {
        "suite_id": suite_id,
        "created_utc": utc_now_iso(),
        "csv_path": str(Path(args.csv).resolve()),
        "model": args.model,
        "truth_lag": args.truth_lag,
        "split": split,
        "baseline": "BASE",
        "best_experiment": {
            "experiment_id": best["experiment_id"],
            "name": best["name"],
            "test_mae": best["metrics"]["test"]["mae"],
        },
        "experiments": results,
    }
    write_json(output_root / "experiments_summary.json", summary)
    LOGGER.info("Suite complete. Output: %s", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

