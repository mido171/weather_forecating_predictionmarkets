"""45-experiment MOS suite (E01-E45) for KMIA Tmax next-day."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from weather_ml import metrics


LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_suite_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"station_id": "string"})
    df["target_date_local"] = pd.to_datetime(df["target_date_local"])
    df["asof_date_local"] = pd.to_datetime(df["asof_date_local"])
    return df


def split_by_date(
    df: pd.DataFrame,
    *,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> dict:
    date_series = pd.to_datetime(df["target_date_local"])
    train_mask = (date_series >= train_start) & (date_series <= train_end)
    val_mask = (date_series >= val_start) & (date_series <= val_end)
    test_mask = (date_series >= test_start) & (date_series <= test_end)
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Split masks are empty; adjust date ranges.")
    return {
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_mask": train_mask.to_numpy(),
        "val_mask": val_mask.to_numpy(),
        "test_mask": test_mask.to_numpy(),
    }


def impute_features(features: pd.DataFrame, train_mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    cleaned = features.replace([np.inf, -np.inf], np.nan)
    train_means = cleaned.loc[train_mask].mean(axis=0, skipna=True).fillna(0.0)
    filled = cleaned.fillna(train_means)
    meta = {"method": "train_mean", "fill_values": train_means.to_dict()}
    return filled, meta


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {}
    return metrics.regression_metrics(y_true, y_pred)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    missing = [col for col in columns if col not in df.columns]
    for col in missing:
        df[col] = np.nan
    if missing:
        LOGGER.warning("Added missing columns with NaN: %s", missing)
    return df


def _blend(a: pd.Series, b: pd.Series) -> pd.Series:
    blend = 0.5 * (a + b)
    blend = blend.where(~a.isna(), b)
    blend = blend.where(~b.isna(), a)
    return blend


def _half_life_alpha(days: float) -> float:
    if days <= 0:
        return 1.0
    return 1.0 - np.exp(np.log(0.5) / days)


def _ewma_bias(err: pd.Series, alpha: float, shift: int = 1) -> pd.Series:
    values = err.to_numpy(dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    prev = 0.0
    has_prev = False
    for i, val in enumerate(values):
        if not np.isnan(val):
            prev = (1 - alpha) * prev + alpha * val if has_prev else val
            has_prev = True
        out[i] = prev if has_prev else np.nan
    return pd.Series(out, index=err.index).shift(shift)


def _regime_bias(err: pd.Series, regime: pd.Series, alpha: float, shift: int = 1) -> pd.Series:
    values = err.to_numpy(dtype=float)
    mask = regime.fillna(False).to_numpy(dtype=bool)
    out = np.full_like(values, np.nan, dtype=float)
    prev = 0.0
    has_prev = False
    for i, val in enumerate(values):
        if mask[i] and not np.isnan(val):
            prev = (1 - alpha) * prev + alpha * val if has_prev else val
            has_prev = True
        out[i] = prev if has_prev else np.nan
    return pd.Series(out, index=err.index).shift(shift)


def _kalman_bias(err: pd.Series, q: float = 0.2, r: float = 2.0, shift: int = 1) -> pd.DataFrame:
    values = err.to_numpy(dtype=float)
    b = 0.0
    p = 1.0
    bias = []
    gain = []
    var = []
    for val in values:
        p = p + q
        if not np.isnan(val):
            k = p / (p + r)
            b = b + k * (val - b)
            p = (1 - k) * p
        else:
            k = np.nan
        bias.append(b)
        gain.append(k)
        var.append(p)
    df = pd.DataFrame(
        {"bias": bias, "gain": gain, "var": var},
        index=err.index,
    )
    return df.shift(shift)


def _rolling_linear_params(
    forecast: pd.Series,
    truth: pd.Series,
    window: int = 30,
    shift: int = 1,
) -> pd.DataFrame:
    x = forecast.astype(float)
    y = truth.astype(float)
    mean_x = x.rolling(window=window, min_periods=5).mean()
    mean_y = y.rolling(window=window, min_periods=5).mean()
    mean_xy = (x * y).rolling(window=window, min_periods=5).mean()
    mean_x2 = (x * x).rolling(window=window, min_periods=5).mean()
    cov = mean_xy - mean_x * mean_y
    var = mean_x2 - mean_x * mean_x
    b = cov / (var + 1e-6)
    a = mean_y - b * mean_x
    return pd.DataFrame({"a": a, "b": b}, index=forecast.index).shift(shift)


def build_feature_store(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("target_date_local").reset_index(drop=True)

    def col(name: str) -> pd.Series:
        return pd.to_numeric(df.get(name), errors="coerce")

    out = pd.DataFrame(
        {
            "target_date_local": df["target_date_local"],
            "asof_date_local": df["asof_date_local"],
            "y_actual_tmax_f": col("y_actual_tmax_f"),
            "cal_d_doy_sin": col("cal_d_doy_sin"),
            "cal_d_doy_cos": col("cal_d_doy_cos"),
        }
    )

    models = ["gfs", "nam"]
    buckets = [0, 12, 24, 36, 48]

    tmp_max = _blend(col("mos_gfs_tmp_max"), col("mos_nam_tmp_max"))
    tmp_mean = _blend(col("mos_gfs_tmp_mean"), col("mos_nam_tmp_mean"))
    tmp_median = _blend(col("mos_gfs_tmp_median"), col("mos_nam_tmp_median"))
    tmp_min = _blend(col("mos_gfs_tmp_min"), col("mos_nam_tmp_min"))

    dpt_mean = _blend(col("mos_gfs_dpt_mean"), col("mos_nam_dpt_mean"))
    dpt_max = _blend(col("mos_gfs_dpt_max"), col("mos_nam_dpt_max"))
    dpt_min = _blend(col("mos_gfs_dpt_min"), col("mos_nam_dpt_min"))

    wsp_mean = _blend(col("mos_gfs_wsp_mean"), col("mos_nam_wsp_mean"))
    wdr_mean = _blend(col("mos_gfs_wdr_mean"), col("mos_nam_wdr_mean"))
    wdr_min = _blend(col("mos_gfs_wdr_min"), col("mos_nam_wdr_min"))
    wdr_max = _blend(col("mos_gfs_wdr_max"), col("mos_nam_wdr_max"))

    cig_min = _blend(col("mos_gfs_cig_min"), col("mos_nam_cig_min"))
    vis_min = _blend(col("mos_gfs_vis_min"), col("mos_nam_vis_min"))
    p06_max = _blend(col("mos_gfs_p06_max"), col("mos_nam_p06_max"))
    p12_max = _blend(col("mos_gfs_p12_max"), col("mos_nam_p12_max"))
    q06_max = _blend(col("mos_gfs_q06_max"), col("mos_nam_q06_max"))
    q12_max = _blend(col("mos_gfs_q12_max"), col("mos_nam_q12_max"))
    n_x_mean = _blend(col("mos_gfs_n_x_mean"), col("mos_nam_n_x_mean"))

    out["feat_tmp_max_mean_models"] = tmp_max
    out["feat_tmp_mean_mean_models"] = tmp_mean
    out["feat_tmp_median_mean_models"] = tmp_median
    out["feat_tmp_min_mean_models"] = tmp_min
    out["feat_tmp_range_mean_models"] = tmp_max - tmp_min
    out["feat_tmp_proxy_blend"] = 0.5 * (tmp_max + tmp_mean)

    out["feat_dpt_mean_models"] = dpt_mean
    out["feat_dpt_range_mean_models"] = dpt_max - dpt_min
    out["feat_dd_models"] = tmp_max - dpt_mean

    out["feat_wsp_mean"] = wsp_mean
    out["feat_wdr_mean"] = wdr_mean
    wdr_rad = np.deg2rad(wdr_mean)
    out["feat_wdr_sin"] = np.sin(wdr_rad)
    out["feat_wdr_cos"] = np.cos(wdr_rad)
    out["feat_u"] = -wsp_mean * np.sin(wdr_rad)
    out["feat_v"] = -wsp_mean * np.cos(wdr_rad)

    out["feat_cig_min"] = cig_min
    out["feat_vis_min"] = vis_min
    out["feat_p06_max"] = p06_max
    out["feat_p12_max"] = p12_max
    out["feat_q06_max"] = q06_max
    out["feat_q12_max"] = q12_max
    out["feat_n_x_mean"] = n_x_mean

    out["feat_log_cig_min"] = np.log1p(cig_min)
    out["feat_log_vis_min"] = np.log1p(vis_min)
    out["feat_log_p06_max"] = np.log1p(p06_max)
    out["feat_log_p12_max"] = np.log1p(p12_max)
    out["feat_log_q06_max"] = np.log1p(q06_max)
    out["feat_log_q12_max"] = np.log1p(q12_max)

    tmp_spread = (col("mos_gfs_tmp_max") - col("mos_nam_tmp_max")).abs()
    out["feat_tmp_max_spread_models"] = tmp_spread

    out["feat_onshore"] = wdr_mean.between(45, 135).astype(float)
    out["feat_offshore"] = wdr_mean.between(225, 315).astype(float)
    out["feat_onshore_wsp"] = out["feat_onshore"] * wsp_mean
    out["feat_conv_proxy"] = (
        (p12_max >= 60) | (q12_max >= 3) | (cig_min <= 3) | (vis_min <= 3)
    ).astype(float)
    out["feat_humid"] = (dpt_mean >= 70).astype(float)
    out["feat_dry"] = (out["feat_dd_models"] >= 18).astype(float)

    for model in models:
        base = col(f"mos_{model}_tmp_max")
        out[f"feat_tmp_max_{model}_b0"] = base
        for b in buckets[1:]:
            out[f"feat_tmp_max_{model}_b{b}"] = col(f"mos_{model}_tmp_max_b{b}")

    le_members = [
        out[f"feat_tmp_max_{model}_b{b}"]
        for model in models
        for b in [0, 12, 24, 36]
    ]
    le_stack = np.vstack([m.to_numpy(dtype=float) for m in le_members])
    out["feat_le_mean"] = np.nanmean(le_stack, axis=0)
    out["feat_le_median"] = np.nanmedian(le_stack, axis=0)
    out["feat_le_spread"] = np.nanmax(le_stack, axis=0) - np.nanmin(le_stack, axis=0)
    out["feat_le_vol"] = np.nanstd(le_stack, axis=0)
    out["feat_le_count"] = np.sum(~np.isnan(le_stack), axis=0)
    out["feat_le_trend"] = np.nanmean(
        np.vstack(
            [
                out[f"feat_tmp_max_{model}_b0"] - out[f"feat_tmp_max_{model}_b24"]
                for model in models
            ]
        ),
        axis=0,
    )

    tmp_b24 = _blend(col("mos_gfs_tmp_max_b24"), col("mos_nam_tmp_max_b24"))
    tmp_b48 = _blend(col("mos_gfs_tmp_max_b48"), col("mos_nam_tmp_max_b48"))
    dpt_b24 = _blend(col("mos_gfs_dpt_mean_b24"), col("mos_nam_dpt_mean_b24"))
    dpt_b48 = _blend(col("mos_gfs_dpt_mean_b48"), col("mos_nam_dpt_mean_b48"))
    wdr_b24 = _blend(col("mos_gfs_wdr_mean_b24"), col("mos_nam_wdr_mean_b24"))
    wdr_b48 = _blend(col("mos_gfs_wdr_mean_b48"), col("mos_nam_wdr_mean_b48"))
    wsp_b24 = _blend(col("mos_gfs_wsp_mean_b24"), col("mos_nam_wsp_mean_b24"))
    wsp_b48 = _blend(col("mos_gfs_wsp_mean_b48"), col("mos_nam_wsp_mean_b48"))

    out["feat_tmp_tplus1"] = tmp_b24.shift(-1)
    out["feat_tmp_tplus2"] = tmp_b48.shift(-2)
    out["feat_tmp_delta_1d"] = out["feat_tmp_tplus1"] - out["feat_tmp_max_mean_models"]
    out["feat_tmp_delta_2d"] = out["feat_tmp_tplus2"] - out["feat_tmp_max_mean_models"]

    out["feat_dpt_tplus1"] = dpt_b24.shift(-1)
    out["feat_dpt_tplus2"] = dpt_b48.shift(-2)
    out["feat_dpt_delta"] = out["feat_dpt_tplus1"] - out["feat_dpt_mean_models"]

    wdr_rad_b24 = np.deg2rad(wdr_b24.shift(-1))
    wdr_rad_b48 = np.deg2rad(wdr_b48.shift(-2))
    out["feat_u_tplus1"] = -wsp_b24.shift(-1) * np.sin(wdr_rad_b24)
    out["feat_v_tplus1"] = -wsp_b24.shift(-1) * np.cos(wdr_rad_b24)
    out["feat_u_tplus2"] = -wsp_b48.shift(-2) * np.sin(wdr_rad_b48)
    out["feat_v_tplus2"] = -wsp_b48.shift(-2) * np.cos(wdr_rad_b48)

    out["feat_tmax_lag1"] = col("obs_tmax_last")
    out["feat_tmax_lag2"] = col("obs_tmax_prev")
    out["feat_tmax_roll3_mean"] = col("obs_tmax_roll_mean_3")
    out["feat_tmax_roll7_mean"] = col("obs_tmax_roll_mean_7")
    out["feat_tmax_roll30_mean"] = col("obs_tmax_roll_mean_30")
    out["feat_tmax_roll7_slope"] = col("obs_tmax_slope_7")
    out["feat_climo_mean_doy"] = col("obs_climo_d_mean")
    out["feat_climo_std_doy"] = col("obs_climo_d_std")
    out["feat_tmax_anom_lag1"] = col("obs_tmax_anom_last_vs_climo")
    out["feat_mos_tmp_anom"] = out["feat_tmp_max_mean_models"] - out["feat_climo_mean_doy"]
    out["feat_z_mos"] = out["feat_mos_tmp_anom"] / (out["feat_climo_std_doy"] + 1e-6)
    out["feat_z_truth_target"] = (
        out["y_actual_tmax_f"] - out["feat_climo_mean_doy"]
    ) / (out["feat_climo_std_doy"] + 1e-6)

    # IEM minute-derived features (optional)
    out["feat_iem_tmax_tminus1"] = col("iem_tminus1_iem_tmax")
    out["feat_iem_tmin_tminus1"] = col("iem_tminus1_iem_tmin")
    out["feat_iem_tmean_tminus1"] = col("iem_tminus1_iem_tmean")
    out["feat_iem_tmed_tminus1"] = col("iem_tminus1_iem_tmed")
    out["feat_iem_range_tminus1"] = col("iem_tminus1_iem_range")
    out["feat_iem_tmax_time_local_min_tminus1"] = col("iem_tminus1_iem_tmax_time_local_min")
    out["feat_iem_plateau_mins_eps_tminus1"] = col("iem_tminus1_iem_plateau_mins_eps")
    out["feat_iem_slope_09_12_tminus1"] = col("iem_tminus1_iem_slope_09_12")
    out["feat_iem_slope_12_15_tminus1"] = col("iem_tminus1_iem_slope_12_15")
    out["feat_iem_slope_15_18_tminus1"] = col("iem_tminus1_iem_slope_15_18")
    out["feat_iem_slope_18_21_tminus1"] = col("iem_tminus1_iem_slope_18_21")
    out["feat_iem_slope_21_24_tminus1"] = col("iem_tminus1_iem_slope_21_24")
    out["feat_iem_drop_cnt_30_tminus1"] = col("iem_tminus1_iem_drop_cnt_30")
    out["feat_iem_max_drop_30_tminus1"] = col("iem_tminus1_iem_max_drop_30")
    out["feat_iem_auc85_tminus1"] = col("iem_tminus1_iem_auc_85")
    out["feat_iem_auc88_tminus1"] = col("iem_tminus1_iem_auc_88")

    out["feat_iem_temp_00z"] = col("iem_temp_00z")
    out["feat_iem_temp_03z"] = col("iem_temp_03z")
    out["feat_iem_temp_06z"] = col("iem_temp_06z")
    out["feat_iem_slope_last60"] = col("iem_slope_last60")
    out["feat_iem_slope_last180"] = col("iem_slope_last180")
    out["feat_iem_std_last180"] = col("iem_std_last180")
    out["feat_iem_cool_00_06"] = col("iem_cool_00_06")
    out["feat_iem_night_warm_anom"] = col("iem_night_warm_anom")
    out["feat_iem_diff_tminus1"] = col("iem_diff_tminus1")
    out["feat_iem_diff_ewma_7_tminus1"] = col("iem_diff_ewma_7_tminus1")
    out["feat_iem_diff_ewma_30_tminus1"] = col("iem_diff_ewma_30_tminus1")
    out["feat_iem_diff_vol_30_tminus1"] = col("iem_diff_vol_30_tminus1")

    for h in range(24):
        out[f"feat_iem_hour_{h:02d}_tminus1"] = col(f"iem_tminus1_iem_hour_{h:02d}")

    alpha_diff = _half_life_alpha(30.0)
    out["feat_iem_diff_ewma_onshore"] = _regime_bias(
        out["feat_iem_diff_tminus1"], out["feat_onshore"] > 0.5, alpha_diff
    )
    out["feat_iem_diff_ewma_offshore"] = _regime_bias(
        out["feat_iem_diff_tminus1"], out["feat_offshore"] > 0.5, alpha_diff
    )

    date_idx = pd.to_datetime(out["target_date_local"])
    tmax_map = pd.Series(out["y_actual_tmax_f"].values, index=date_idx.dt.date)
    lag_dates = (date_idx - timedelta(days=365)).dt.date
    out["feat_tmax_lag365"] = lag_dates.map(tmax_map)

    err_blend = out["y_actual_tmax_f"] - out["feat_tmp_max_mean_models"]
    out["feat_err_blend_lag1"] = err_blend.shift(1)

    alpha_5 = _half_life_alpha(5.0)
    alpha_15 = _half_life_alpha(15.0)
    alpha_fast = 0.2
    alpha_slow = 0.05

    for model in models:
        f = out[f"feat_tmp_max_{model}_b0"]
        err = out["y_actual_tmax_f"] - f
        out[f"feat_err_{model}_lag1"] = err.shift(1)
        out[f"feat_bias_ewma_5d_{model}"] = _ewma_bias(err, alpha_5)
        out[f"feat_bias_ewma_15d_{model}"] = _ewma_bias(err, alpha_15)
        out[f"feat_bias_fast_{model}"] = _ewma_bias(err, alpha_fast)
        out[f"feat_bias_slow_{model}"] = _ewma_bias(err, alpha_slow)
        out[f"feat_tmp_corr_{model}_bias5"] = f + out[f"feat_bias_ewma_5d_{model}"]
        out[f"feat_tmp_corr_{model}_bias15"] = f + out[f"feat_bias_ewma_15d_{model}"]
        out[f"feat_tmp_corr_{model}_bias_fast"] = f + out[f"feat_bias_fast_{model}"]
        out[f"feat_tmp_corr_{model}_bias_slow"] = f + out[f"feat_bias_slow_{model}"]
        out[f"feat_bias_fast_minus_slow_{model}"] = (
            out[f"feat_bias_fast_{model}"] - out[f"feat_bias_slow_{model}"]
        )

        for b in [0, 12, 24, 36]:
            f_b = out[f"feat_tmp_max_{model}_b{b}"]
            err_b = out["y_actual_tmax_f"] - f_b
            out[f"feat_bias_b{b}_{model}"] = _ewma_bias(err_b, alpha_15)
            out[f"feat_tmp_corr_{model}_b{b}"] = f_b + out[f"feat_bias_b{b}_{model}"]

        conv = out["feat_conv_proxy"] > 0.5
        onshore = out["feat_onshore"] > 0.5
        cool = pd.to_datetime(out["target_date_local"]).dt.month.isin([11, 12, 1, 2, 3])

        out[f"feat_bias_conv_{model}"] = _regime_bias(err, conv, alpha_15)
        out[f"feat_bias_clear_{model}"] = _regime_bias(err, ~conv, alpha_15)
        out[f"feat_bias_onshore_{model}"] = _regime_bias(err, onshore, alpha_15)
        out[f"feat_bias_offshore_{model}"] = _regime_bias(err, ~onshore, alpha_15)
        out[f"feat_bias_cool_{model}"] = _regime_bias(err, cool, alpha_15)
        out[f"feat_bias_warm_{model}"] = _regime_bias(err, ~cool, alpha_15)

        out[f"feat_tmp_corr_{model}_conv"] = f + out[f"feat_bias_conv_{model}"]
        out[f"feat_tmp_corr_{model}_clear"] = f + out[f"feat_bias_clear_{model}"]
        out[f"feat_tmp_corr_{model}_onshore"] = f + out[f"feat_bias_onshore_{model}"]
        out[f"feat_tmp_corr_{model}_offshore"] = f + out[f"feat_bias_offshore_{model}"]
        out[f"feat_tmp_corr_{model}_cool"] = f + out[f"feat_bias_cool_{model}"]
        out[f"feat_tmp_corr_{model}_warm"] = f + out[f"feat_bias_warm_{model}"]

        kal = _kalman_bias(err)
        out[f"feat_kalman_bias_{model}"] = kal["bias"]
        out[f"feat_kalman_gain_{model}"] = kal["gain"]
        out[f"feat_kalman_var_{model}"] = kal["var"]
        out[f"feat_tmp_corr_{model}_kalman"] = f + out[f"feat_kalman_bias_{model}"]

        lin = _rolling_linear_params(f, out["y_actual_tmax_f"], window=30)
        out[f"feat_corr_a_{model}"] = lin["a"]
        out[f"feat_corr_b_{model}"] = lin["b"]
        out[f"feat_tmp_corr_{model}_lin"] = lin["a"] + lin["b"] * f

    corr_members = [
        out[f"feat_tmp_corr_{model}_b{b}"]
        for model in models
        for b in [0, 12, 24, 36]
    ]
    corr_stack = np.vstack([m.to_numpy(dtype=float) for m in corr_members])
    out["feat_le_median_biascorr"] = np.nanmedian(corr_stack, axis=0)

    wdr_span = (wdr_max - wdr_min).abs()
    wdr_span = np.where(wdr_span > 180, 360 - wdr_span, wdr_span)
    out["feat_wdr_span"] = wdr_span
    out["feat_wsp_range"] = (
        _blend(col("mos_gfs_wsp_max"), col("mos_nam_wsp_max"))
        - _blend(col("mos_gfs_wsp_min"), col("mos_nam_wsp_min"))
    )

    out["feat_precip_proxy"] = out["feat_log_p12_max"] + out["feat_log_q12_max"]
    out["feat_suppression"] = out["feat_precip_proxy"] * out["feat_tmp_max_mean_models"]
    out["feat_u_onshore"] = np.maximum(out["feat_u"], 0.0)
    out["feat_u_onshore_wsp"] = out["feat_u_onshore"] * out["feat_wsp_mean"]
    out["feat_marine"] = out["feat_onshore"] * out["feat_dpt_mean_models"]

    return out


def train_lgbm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    params: dict[str, Any] | None = None,
) -> lgb.LGBMRegressor:
    base_params = {
        "objective": "regression_l1",
        "learning_rate": 0.05,
        "n_estimators": 600,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
    }
    if params:
        base_params.update(params)
    model = lgb.LGBMRegressor(**base_params)
    if len(y_val):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def train_lgbm_quantile(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    alpha: float,
) -> lgb.LGBMRegressor:
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "learning_rate": 0.05,
        "n_estimators": 700,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
    }
    model = lgb.LGBMRegressor(**params)
    if len(y_val):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def train_lgbm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
) -> lgb.LGBMClassifier:
    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "n_estimators": 400,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
    }
    model = lgb.LGBMClassifier(**params)
    if len(y_val):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def save_predictions(
    path: Path,
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra: dict | None = None,
) -> None:
    payload = pd.DataFrame(
        {
            "target_date_local": df["target_date_local"].values,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    if extra:
        for key, val in extra.items():
            payload[key] = val
    payload.to_csv(path, index=False)


@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    features: list[str]
    metrics: dict[str, dict[str, float]]
    extras: dict[str, Any]


@dataclass
class SuiteContext:
    df: pd.DataFrame
    y: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    seed: int
    cache: dict[str, Any]


def run_point_model(
    ctx: SuiteContext,
    features: list[str],
    *,
    residual_base: str | None = None,
    target_transform: str | None = None,
    model_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    df = ensure_columns(ctx.df, features)
    features_df = df[features]
    filled, impute_meta = impute_features(features_df, ctx.train_mask)
    X = filled.to_numpy(dtype=float)

    y = ctx.y
    y_train = y[ctx.train_mask]
    y_val = y[ctx.val_mask]
    y_test = y[ctx.test_mask]

    base = None
    if residual_base:
        base_series = pd.to_numeric(ctx.df.get(residual_base), errors="coerce")
        base = base_series.to_numpy(dtype=float)
        train_mean = float(np.nanmean(y_train))
        base = np.where(np.isnan(base), train_mean, base)
        base_train = base[ctx.train_mask]
        base_val = base[ctx.val_mask]
        base_test = base[ctx.test_mask]
        y_train = y_train - base_train
        y_val = y_val - base_val
        y_test = y_test - base_test

    if target_transform == "climo_resid":
        climo = pd.to_numeric(ctx.df.get("feat_climo_mean_doy"), errors="coerce").to_numpy(dtype=float)
        train_mean = float(np.nanmean(y_train))
        climo = np.where(np.isnan(climo), train_mean, climo)
        y_train = y_train - climo[ctx.train_mask]
        y_val = y_val - climo[ctx.val_mask]
        y_test = y_test - climo[ctx.test_mask]
    elif target_transform == "zscore":
        climo = pd.to_numeric(ctx.df.get("feat_climo_mean_doy"), errors="coerce").to_numpy(dtype=float)
        std = pd.to_numeric(ctx.df.get("feat_climo_std_doy"), errors="coerce").to_numpy(dtype=float)
        train_mean = float(np.nanmean(y_train))
        climo = np.where(np.isnan(climo), train_mean, climo)
        std = np.where(np.isnan(std) | (std == 0), np.nanstd(y[ctx.train_mask]), std)
        z_target = (y - climo) / (std + 1e-6)
        y_train = z_target[ctx.train_mask]
        y_val = z_target[ctx.val_mask]
        y_test = z_target[ctx.test_mask]

    X_train = X[ctx.train_mask]
    X_val = X[ctx.val_mask]
    X_test = X[ctx.test_mask]

    model = train_lgbm_regressor(X_train, y_train, X_val, y_val, seed=ctx.seed, params=model_params)
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    if residual_base and base is not None:
        pred_train = pred_train + base[ctx.train_mask]
        pred_val = pred_val + base[ctx.val_mask]
        pred_test = pred_test + base[ctx.test_mask]

    if target_transform == "climo_resid":
        climo = pd.to_numeric(ctx.df.get("feat_climo_mean_doy"), errors="coerce").to_numpy(dtype=float)
        train_mean = float(np.nanmean(ctx.y[ctx.train_mask]))
        climo = np.where(np.isnan(climo), train_mean, climo)
        pred_train = pred_train + climo[ctx.train_mask]
        pred_val = pred_val + climo[ctx.val_mask]
        pred_test = pred_test + climo[ctx.test_mask]
    elif target_transform == "zscore":
        climo = pd.to_numeric(ctx.df.get("feat_climo_mean_doy"), errors="coerce").to_numpy(dtype=float)
        std = pd.to_numeric(ctx.df.get("feat_climo_std_doy"), errors="coerce").to_numpy(dtype=float)
        train_mean = float(np.nanmean(ctx.y[ctx.train_mask]))
        climo = np.where(np.isnan(climo), train_mean, climo)
        std = np.where(np.isnan(std) | (std == 0), np.nanstd(ctx.y[ctx.train_mask]), std)
        pred_train = pred_train * std[ctx.train_mask] + climo[ctx.train_mask]
        pred_val = pred_val * std[ctx.val_mask] + climo[ctx.val_mask]
        pred_test = pred_test * std[ctx.test_mask] + climo[ctx.test_mask]

    meta = {"impute": impute_meta}
    return pred_train, pred_val, pred_test, meta


def run_quantile_suite(
    ctx: SuiteContext,
    features: list[str],
    *,
    alphas: list[float],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[float, np.ndarray],
    dict[float, np.ndarray],
]:
    df = ensure_columns(ctx.df, features)
    filled, impute_meta = impute_features(df[features], ctx.train_mask)
    X = filled.to_numpy(dtype=float)
    X_train = X[ctx.train_mask]
    X_val = X[ctx.val_mask]
    X_test = X[ctx.test_mask]
    y = ctx.y
    y_train = y[ctx.train_mask]
    y_val = y[ctx.val_mask]
    q_models: dict[float, lgb.LGBMRegressor] = {}
    q_preds_test: dict[float, np.ndarray] = {}
    q_preds_val: dict[float, np.ndarray] = {}
    for alpha in alphas:
        model = train_lgbm_quantile(X_train, y_train, X_val, y_val, seed=ctx.seed, alpha=alpha)
        q_models[alpha] = model
        q_preds_test[alpha] = model.predict(X_test)
        q_preds_val[alpha] = model.predict(X_val)
    pred_train = q_models[0.5].predict(X_train)
    pred_val = q_models[0.5].predict(X_val)
    pred_test = q_models[0.5].predict(X_test)
    meta = {"impute": impute_meta}
    return pred_train, pred_val, pred_test, meta, q_preds_val, q_preds_test


def run_moe_gate(
    ctx: SuiteContext,
    *,
    gate_features: list[str],
    expert_features: list[str],
    gate_target: str,
    base_series: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = ctx.df.copy()
    gate = df[gate_target].fillna(0.0).to_numpy(dtype=int)
    base = pd.to_numeric(df.get(base_series), errors="coerce").to_numpy(dtype=float)
    base_mean = float(np.nanmean(ctx.y[ctx.train_mask]))
    base = np.where(np.isnan(base), base_mean, base)

    gate_df = ensure_columns(df, gate_features)
    gate_X, _ = impute_features(gate_df[gate_features], ctx.train_mask)
    X = gate_X.to_numpy(dtype=float)
    gate_model = train_lgbm_classifier(
        X[ctx.train_mask],
        gate[ctx.train_mask],
        X[ctx.val_mask],
        gate[ctx.val_mask],
        seed=ctx.seed,
    )
    p_gate = gate_model.predict_proba(X)[:, 1]

    expert_df = ensure_columns(df, expert_features)
    expert_X, _ = impute_features(expert_df[expert_features], ctx.train_mask)
    X_exp = expert_X.to_numpy(dtype=float)

    def fit_expert(mask: np.ndarray) -> lgb.LGBMRegressor | None:
        if not mask.any():
            return None
        return train_lgbm_regressor(
            X_exp[mask],
            (ctx.y[mask] - base[mask]),
            X_exp[ctx.val_mask & mask],
            (ctx.y[ctx.val_mask & mask] - base[ctx.val_mask & mask]),
            seed=ctx.seed,
        )

    expert_a = fit_expert(gate == 1)
    expert_b = fit_expert(gate == 0)

    def predict(model: lgb.LGBMRegressor | None, X_sub: np.ndarray) -> np.ndarray:
        if model is None:
            return np.full(len(X_sub), np.nan)
        return model.predict(X_sub)

    resid_a = predict(expert_a, X_exp)
    resid_b = predict(expert_b, X_exp)
    blend = p_gate * resid_a + (1 - p_gate) * resid_b
    pred = base + blend
    pred_train = pred[ctx.train_mask]
    pred_val = pred[ctx.val_mask]
    pred_test = pred[ctx.test_mask]
    return pred_train, pred_val, pred_test


def run_rule_moe(
    ctx: SuiteContext,
    *,
    rule_masks: dict[str, np.ndarray],
    expert_features: list[str],
    base_series: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = ctx.df.copy()
    base = pd.to_numeric(df.get(base_series), errors="coerce").to_numpy(dtype=float)
    base_mean = float(np.nanmean(ctx.y[ctx.train_mask]))
    base = np.where(np.isnan(base), base_mean, base)

    expert_df = ensure_columns(df, expert_features)
    expert_X, _ = impute_features(expert_df[expert_features], ctx.train_mask)
    X_exp = expert_X.to_numpy(dtype=float)

    preds = np.full(len(df), np.nan)
    for _, mask in rule_masks.items():
        train_mask = ctx.train_mask & mask
        val_mask = ctx.val_mask & mask
        if not train_mask.any():
            continue
        model = train_lgbm_regressor(
            X_exp[train_mask],
            (ctx.y[train_mask] - base[train_mask]),
            X_exp[val_mask],
            (ctx.y[val_mask] - base[val_mask]),
            seed=ctx.seed,
        )
        preds[mask] = base[mask] + model.predict(X_exp[mask])
    pred_train = preds[ctx.train_mask]
    pred_val = preds[ctx.val_mask]
    pred_test = preds[ctx.test_mask]
    return pred_train, pred_val, pred_test


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MOS 45-experiment suite.")
    parser.add_argument("--features", required=True, help="Path to features.csv")
    parser.add_argument("--suite-id", default=default_suite_id())
    parser.add_argument("--out-root", default="artifacts/MOS/experiments")
    parser.add_argument("--train-start", default="2002-01-22")
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--val-start", default="2020-01-01")
    parser.add_argument("--val-end", default="2022-12-31")
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()

    df_raw = load_csv(args.features)
    feature_store = build_feature_store(df_raw)
    feature_store = feature_store[feature_store["y_actual_tmax_f"].notna()].copy()

    split = split_by_date(
        feature_store,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    ctx = SuiteContext(
        df=feature_store,
        y=feature_store["y_actual_tmax_f"].to_numpy(dtype=float),
        train_mask=split.pop("train_mask"),
        val_mask=split.pop("val_mask"),
        test_mask=split.pop("test_mask"),
        seed=args.seed,
        cache={},
    )

    output_root = Path(args.out_root) / args.suite_id
    output_root.mkdir(parents=True, exist_ok=True)

    feature_store_path = output_root / "feature_store.csv"
    feature_store.to_csv(feature_store_path, index=False)
    write_json(output_root / "split_info.json", split)

    results: list[ExperimentResult] = []

    def record(
        exp_id: str,
        name: str,
        features: list[str],
        preds: tuple[np.ndarray, np.ndarray, np.ndarray],
        extra: dict[str, Any] | None = None,
    ) -> None:
        pred_train, pred_val, pred_test = preds
        metrics_payload = {
            "train": regression_metrics(ctx.y[ctx.train_mask], pred_train),
            "validation": regression_metrics(ctx.y[ctx.val_mask], pred_val),
            "test": regression_metrics(ctx.y[ctx.test_mask], pred_test),
        }
        extras = dict(extra or {})
        pred_cols = extras.pop("pred_cols", None)
        results.append(
            ExperimentResult(
                experiment_id=exp_id,
                name=name,
                features=features,
                metrics=metrics_payload,
                extras=extras,
            )
        )
        exp_dir = output_root / exp_id
        exp_dir.mkdir(exist_ok=True, parents=True)
        save_predictions(exp_dir / "predictions_train.csv", feature_store.loc[ctx.train_mask], ctx.y[ctx.train_mask], pred_train)
        save_predictions(exp_dir / "predictions_val.csv", feature_store.loc[ctx.val_mask], ctx.y[ctx.val_mask], pred_val)
        save_predictions(exp_dir / "predictions_test.csv", feature_store.loc[ctx.test_mask], ctx.y[ctx.test_mask], pred_test, extra=pred_cols)

    DOY = ["cal_d_doy_sin", "cal_d_doy_cos"]

    feature_store["feat_tmp_max_mean_models_shift_m1"] = feature_store["feat_tmp_max_mean_models"].shift(1)
    feature_store["feat_tmp_max_mean_models_shift_p1"] = feature_store["feat_tmp_max_mean_models"].shift(-1)

    # A) Data/target sanity
    pred = run_point_model(ctx, ["feat_tmp_max_mean_models_shift_m1", *DOY])
    record("E01", "Day alignment sweep (MOS shifted -1)", ["feat_tmp_max_mean_models_shift_m1", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_max_mean_models", *DOY])
    record("E02", "Day alignment sweep (no shift)", ["feat_tmp_max_mean_models", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_max_mean_models_shift_p1", *DOY])
    record("E03", "Day alignment sweep (MOS shifted +1)", ["feat_tmp_max_mean_models_shift_p1", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_mean_mean_models", *DOY])
    record("E04", "TMP statistic audit (mean)", ["feat_tmp_mean_mean_models", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_median_mean_models", *DOY])
    record("E05", "TMP statistic audit (median)", ["feat_tmp_median_mean_models", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_proxy_blend", "feat_tmp_range_mean_models", *DOY])
    record("E06", "TMP blended proxy", ["feat_tmp_proxy_blend", "feat_tmp_range_mean_models", *DOY], pred[:3])

    # B) Persistence + climatology
    pred = run_point_model(ctx, ["feat_tmax_lag1", *DOY])
    record("E07", "Persistence baseline", ["feat_tmax_lag1", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmax_lag1", "feat_tmp_max_mean_models", *DOY])
    record("E08", "Persistence + MOS anchor", ["feat_tmax_lag1", "feat_tmp_max_mean_models", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmax_lag1", "feat_tmp_max_mean_models", "feat_dpt_mean_models", "feat_dd_models", *DOY])
    record("E09", "Persistence + MOS + humidity", ["feat_tmax_lag1", "feat_tmp_max_mean_models", "feat_dpt_mean_models", "feat_dd_models", *DOY], pred[:3])
    pred = run_point_model(
        ctx,
        ["feat_tmax_lag1", "feat_tmp_max_mean_models", "feat_dpt_mean_models", "feat_dd_models", "feat_tmax_roll3_mean", "feat_tmax_roll7_mean", "feat_tmax_roll7_slope", *DOY],
    )
    record("E10", "Add short rolling memory", ["feat_tmax_lag1", "feat_tmp_max_mean_models", "feat_dpt_mean_models", "feat_dd_models", "feat_tmax_roll3_mean", "feat_tmax_roll7_mean", "feat_tmax_roll7_slope", *DOY], pred[:3])
    pred = run_point_model(
        ctx,
        ["feat_tmax_lag1", "feat_tmp_max_mean_models", "feat_dpt_mean_models", "feat_dd_models", "feat_tmax_roll3_mean", "feat_tmax_roll7_mean", "feat_tmax_roll7_slope", "feat_tmax_roll30_mean", *DOY],
    )
    record("E11", "Add long rolling memory", ["feat_tmax_lag1", "feat_tmp_max_mean_models", "feat_dpt_mean_models", "feat_dd_models", "feat_tmax_roll3_mean", "feat_tmax_roll7_mean", "feat_tmax_roll7_slope", "feat_tmax_roll30_mean", *DOY], pred[:3])
    pred = run_point_model(
        ctx,
        ["feat_mos_tmp_anom", "feat_tmax_anom_lag1", "feat_dd_models", *DOY],
        target_transform="climo_resid",
    )
    record("E12", "Climatology-anchored residual", ["feat_mos_tmp_anom", "feat_tmax_anom_lag1", "feat_dd_models", *DOY], pred[:3])
    pred = run_point_model(
        ctx,
        ["feat_z_mos", "feat_dd_models", "feat_dpt_mean_models", *DOY],
        target_transform="zscore",
    )
    record("E13", "Standardized anomaly z-score", ["feat_z_mos", "feat_dd_models", "feat_dpt_mean_models", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmax_lag365", "feat_tmax_lag1", "feat_tmp_max_mean_models", *DOY])
    record("E14", "Year-over-year analog anchor", ["feat_tmax_lag365", "feat_tmax_lag1", "feat_tmp_max_mean_models", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmax_lag1", "feat_tmp_max_mean_models", "feat_tmp_max_spread_models", "feat_le_vol", *DOY])
    record("E15", "Dynamic MOS vs persistence weighting", ["feat_tmax_lag1", "feat_tmp_max_mean_models", "feat_tmp_max_spread_models", "feat_le_vol", *DOY], pred[:3])

    # C) Lagged ensemble + multi-horizon context
    lagged_members = [f"feat_tmp_max_{model}_b{b}" for model in ["gfs", "nam"] for b in [0, 12, 24, 36]]
    pred = run_point_model(ctx, [*lagged_members, *DOY])
    record("E16", "Lagged ensemble members only", [*lagged_members, *DOY], pred[:3])
    pred = run_point_model(ctx, [*lagged_members, "feat_le_mean", "feat_le_median", "feat_le_spread", "feat_le_trend", "feat_le_vol", *DOY])
    record("E17", "Lagged ensemble summary + members", [*lagged_members, "feat_le_mean", "feat_le_median", "feat_le_spread", "feat_le_trend", "feat_le_vol", *DOY], pred[:3])
    pred = run_point_model(ctx, [*lagged_members, "feat_le_mean", "feat_le_median", "feat_le_spread", "feat_le_trend", "feat_le_vol", "feat_dd_models", "feat_p12_max", "feat_q12_max", "feat_cig_min", "feat_u", "feat_v", *DOY])
    record("E18", "Lagged ensemble + regime", [*lagged_members, "feat_le_mean", "feat_le_median", "feat_le_spread", "feat_le_trend", "feat_le_vol", "feat_dd_models", "feat_p12_max", "feat_q12_max", "feat_cig_min", "feat_u", "feat_v", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_dd_models", "feat_conv_proxy", "feat_onshore", "feat_offshore", "feat_tmp_range_mean_models", *DOY], residual_base="feat_le_median")
    record("E19", "Residual on lagged-ensemble median", ["feat_dd_models", "feat_conv_proxy", "feat_onshore", "feat_offshore", "feat_tmp_range_mean_models", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_dd_models", "feat_conv_proxy", "feat_onshore", "feat_offshore", "feat_tmp_range_mean_models", "feat_dpt_range_mean_models", *DOY], residual_base="feat_le_median")
    record("E20", "Residual + diurnal amplitude proxies", ["feat_dd_models", "feat_conv_proxy", "feat_onshore", "feat_offshore", "feat_tmp_range_mean_models", "feat_dpt_range_mean_models", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_max_mean_models", "feat_tmp_tplus1", "feat_tmp_tplus2", "feat_tmp_delta_1d", "feat_tmp_delta_2d", *DOY])
    record("E21", "Multi-horizon pattern momentum", ["feat_tmp_max_mean_models", "feat_tmp_tplus1", "feat_tmp_tplus2", "feat_tmp_delta_1d", "feat_tmp_delta_2d", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_max_mean_models", "feat_tmp_tplus1", "feat_tmp_tplus2", "feat_tmp_delta_1d", "feat_tmp_delta_2d", "feat_u", "feat_v", "feat_u_tplus1", "feat_v_tplus1", *DOY])
    record("E22", "Multi-horizon + wind shift", ["feat_tmp_max_mean_models", "feat_tmp_tplus1", "feat_tmp_tplus2", "feat_tmp_delta_1d", "feat_tmp_delta_2d", "feat_u", "feat_v", "feat_u_tplus1", "feat_v_tplus1", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_max_mean_models", "feat_tmp_tplus1", "feat_tmp_tplus2", "feat_tmp_delta_1d", "feat_tmp_delta_2d", "feat_u", "feat_v", "feat_u_tplus1", "feat_v_tplus1", "feat_dpt_mean_models", "feat_dpt_tplus1", "feat_dpt_delta", *DOY])
    record("E23", "Multi-horizon + dewpoint drop", ["feat_tmp_max_mean_models", "feat_tmp_tplus1", "feat_tmp_tplus2", "feat_tmp_delta_1d", "feat_tmp_delta_2d", "feat_u", "feat_v", "feat_u_tplus1", "feat_v_tplus1", "feat_dpt_mean_models", "feat_dpt_tplus1", "feat_dpt_delta", *DOY], pred[:3])
    pred = run_point_model(ctx, [*lagged_members, "feat_le_mean", "feat_le_median", "feat_le_spread", "feat_le_trend", "feat_le_vol", "feat_tmp_delta_1d", "feat_dpt_delta", "feat_u", "feat_v", *DOY])
    record("E24", "Lagged ensemble + multi-horizon hybrid", [*lagged_members, "feat_le_mean", "feat_le_median", "feat_le_spread", "feat_le_trend", "feat_le_vol", "feat_tmp_delta_1d", "feat_dpt_delta", "feat_u", "feat_v", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_dpt_delta", "feat_tmp_delta_1d", "feat_u", "feat_v", "feat_onshore", "feat_offshore", *DOY], residual_base="feat_le_median")
    record("E25", "Front timing residual model", ["feat_dpt_delta", "feat_tmp_delta_1d", "feat_u", "feat_v", "feat_onshore", "feat_offshore", *DOY], pred[:3])

    # D) Adaptive bias / state-space post-processing
    pred = run_point_model(ctx, ["feat_tmp_corr_gfs_bias15", "feat_tmp_corr_nam_bias15", "feat_dd_models", "feat_conv_proxy", *DOY])
    record("E26", "Per-model decaying-average bias", ["feat_tmp_corr_gfs_bias15", "feat_tmp_corr_nam_bias15", "feat_dd_models", "feat_conv_proxy", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_corr_gfs_bias_fast", "feat_tmp_corr_gfs_bias_slow", "feat_bias_fast_minus_slow_gfs", "feat_tmp_corr_nam_bias_fast", "feat_tmp_corr_nam_bias_slow", "feat_bias_fast_minus_slow_nam", "feat_dd_models", "feat_conv_proxy", *DOY])
    record("E27", "Two-timescale bias states", ["feat_tmp_corr_gfs_bias_fast", "feat_tmp_corr_gfs_bias_slow", "feat_bias_fast_minus_slow_gfs", "feat_tmp_corr_nam_bias_fast", "feat_tmp_corr_nam_bias_slow", "feat_bias_fast_minus_slow_nam", "feat_dd_models", "feat_conv_proxy", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_corr_gfs_b0", "feat_tmp_corr_gfs_b12", "feat_tmp_corr_gfs_b24", "feat_tmp_corr_gfs_b36", "feat_tmp_corr_nam_b0", "feat_tmp_corr_nam_b12", "feat_tmp_corr_nam_b24", "feat_tmp_corr_nam_b36", "feat_le_spread", *DOY])
    record("E28", "Bias per lead-bucket member", ["feat_tmp_corr_gfs_b0", "feat_tmp_corr_gfs_b12", "feat_tmp_corr_gfs_b24", "feat_tmp_corr_gfs_b36", "feat_tmp_corr_nam_b0", "feat_tmp_corr_nam_b12", "feat_tmp_corr_nam_b24", "feat_tmp_corr_nam_b36", "feat_le_spread", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_corr_gfs_conv", "feat_tmp_corr_gfs_clear", "feat_tmp_corr_nam_conv", "feat_tmp_corr_nam_clear", "feat_conv_proxy", *DOY])
    record("E29", "Regime-conditioned bias (convective)", ["feat_tmp_corr_gfs_conv", "feat_tmp_corr_gfs_clear", "feat_tmp_corr_nam_conv", "feat_tmp_corr_nam_clear", "feat_conv_proxy", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_corr_gfs_onshore", "feat_tmp_corr_gfs_offshore", "feat_tmp_corr_nam_onshore", "feat_tmp_corr_nam_offshore", "feat_onshore", "feat_offshore", "feat_u", "feat_v", *DOY])
    record("E30", "Onshore/offshore conditioned bias", ["feat_tmp_corr_gfs_onshore", "feat_tmp_corr_gfs_offshore", "feat_tmp_corr_nam_onshore", "feat_tmp_corr_nam_offshore", "feat_onshore", "feat_offshore", "feat_u", "feat_v", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_corr_gfs_cool", "feat_tmp_corr_gfs_warm", "feat_tmp_corr_nam_cool", "feat_tmp_corr_nam_warm", *DOY])
    record("E31", "Cool-season vs warm-season bias", ["feat_tmp_corr_gfs_cool", "feat_tmp_corr_gfs_warm", "feat_tmp_corr_nam_cool", "feat_tmp_corr_nam_warm", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_corr_gfs_kalman", "feat_kalman_gain_gfs", "feat_kalman_var_gfs", "feat_tmp_corr_nam_kalman", "feat_kalman_gain_nam", "feat_kalman_var_nam", *DOY])
    record("E32", "Kalman bias filter", ["feat_tmp_corr_gfs_kalman", "feat_kalman_gain_gfs", "feat_kalman_var_gfs", "feat_tmp_corr_nam_kalman", "feat_kalman_gain_nam", "feat_kalman_var_nam", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_tmp_corr_gfs_lin", "feat_corr_a_gfs", "feat_corr_b_gfs", "feat_tmp_corr_nam_lin", "feat_corr_a_nam", "feat_corr_b_nam", *DOY])
    record("E33", "Rolling linear correction", ["feat_tmp_corr_gfs_lin", "feat_corr_a_gfs", "feat_corr_b_gfs", "feat_tmp_corr_nam_lin", "feat_corr_a_nam", "feat_corr_b_nam", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_err_gfs_lag1", "feat_err_nam_lag1", "feat_err_blend_lag1", "feat_conv_proxy", "feat_onshore", "feat_offshore", *DOY])
    record("E34", "Error autoregression features", ["feat_err_gfs_lag1", "feat_err_nam_lag1", "feat_err_blend_lag1", "feat_conv_proxy", "feat_onshore", "feat_offshore", *DOY], pred[:3])
    pred = run_point_model(ctx, ["feat_dd_models", "feat_tmp_range_mean_models", "feat_cig_min", "feat_p12_max", "feat_q12_max", "feat_u", "feat_v", "feat_le_spread", *DOY], residual_base="feat_le_median_biascorr")
    record("E35", "Residual-of-residual (bias-corrected base)", ["feat_dd_models", "feat_tmp_range_mean_models", "feat_cig_min", "feat_p12_max", "feat_q12_max", "feat_u", "feat_v", "feat_le_spread", *DOY], pred[:3])

    # E) Regime gating
    gate_features = ["feat_p12_max", "feat_q12_max", "feat_cig_min", "feat_vis_min", "feat_dd_models", "feat_u", "feat_v", *DOY]
    expert_features = ["feat_dd_models", "feat_tmp_range_mean_models", "feat_p12_max", "feat_q12_max", "feat_cig_min", "feat_u", "feat_v", *DOY]
    pred_train, pred_val, pred_test = run_moe_gate(
        ctx,
        gate_features=gate_features,
        expert_features=expert_features,
        gate_target="feat_conv_proxy",
        base_series="feat_le_median_biascorr",
    )
    record("E36", "MoE convective suppression vs clear", [*gate_features, *expert_features], (pred_train, pred_val, pred_test))

    gate_features = ["feat_u", "feat_v", "feat_wsp_mean", *DOY]
    pred_train, pred_val, pred_test = run_moe_gate(
        ctx,
        gate_features=gate_features,
        expert_features=expert_features,
        gate_target="feat_onshore",
        base_series="feat_le_median_biascorr",
    )
    record("E37", "MoE onshore vs offshore", [*gate_features, *expert_features], (pred_train, pred_val, pred_test))

    conv_mask = feature_store["feat_conv_proxy"] > 0.5
    front_mask = (feature_store["feat_offshore"] > 0.5) & (feature_store["feat_dpt_delta"] < -5)
    fair_mask = ~(conv_mask | front_mask)
    pred_train, pred_val, pred_test = run_rule_moe(
        ctx,
        rule_masks={
            "front": front_mask.to_numpy(),
            "convective": conv_mask.to_numpy(),
            "fair": fair_mask.to_numpy(),
        },
        expert_features=expert_features,
        base_series="feat_le_median_biascorr",
    )
    record("E38", "Rule-based 3-regime experts", [*expert_features], (pred_train, pred_val, pred_test))

    pred = run_point_model(
        ctx,
        [
            "feat_tmp_max_mean_models",
            "feat_dd_models",
            "feat_precip_proxy",
            "feat_log_cig_min",
            "feat_u_onshore_wsp",
            "feat_suppression",
            "feat_marine",
            *DOY,
        ],
    )
    record("E39", "Explicit interaction features", ["feat_tmp_max_mean_models", "feat_dd_models", "feat_precip_proxy", "feat_log_cig_min", "feat_u_onshore_wsp", "feat_suppression", "feat_marine", *DOY], pred[:3])

    pred = run_point_model(
        ctx,
        ["feat_tmp_max_mean_models", "feat_wdr_span", "feat_wsp_range", "feat_dd_models", "feat_precip_proxy", *DOY],
    )
    record("E40", "Wind rotation / sea-breeze timing proxy", ["feat_tmp_max_mean_models", "feat_wdr_span", "feat_wsp_range", "feat_dd_models", "feat_precip_proxy", *DOY], pred[:3])

    # F) Probabilistic / distributional
    best_features = ["feat_dd_models", "feat_tmp_range_mean_models", "feat_cig_min", "feat_p12_max", "feat_q12_max", "feat_u", "feat_v", "feat_le_spread", *DOY]
    q50_train, q50_val, q50_test, _, q_preds_val, q_preds_test = run_quantile_suite(
        ctx, best_features, alphas=[0.1, 0.5, 0.9]
    )
    record("E41", "Quantile median LightGBM", best_features, (q50_train, q50_val, q50_test))

    coverage = float(
        np.mean((ctx.y[ctx.test_mask] >= q_preds_test[0.1]) & (ctx.y[ctx.test_mask] <= q_preds_test[0.9]))
    )
    record(
        "E42",
        "Quantile suite 0.1/0.5/0.9",
        best_features,
        (q50_train, q50_val, q50_test),
        extra={"coverage_80_test": coverage, "pred_cols": {"q10": q_preds_test[0.1], "q90": q_preds_test[0.9]}},
    )

    # E43: mean + sigma surrogate with shrinkage
    pred_train, pred_val, pred_test, _ = run_point_model(ctx, best_features)
    resid_train = np.abs(ctx.y[ctx.train_mask] - pred_train)
    resid_val = np.abs(ctx.y[ctx.val_mask] - pred_val)
    filled, _ = impute_features(ensure_columns(ctx.df, best_features)[best_features], ctx.train_mask)
    X = filled.to_numpy(dtype=float)
    sigma_model = train_lgbm_regressor(
        X[ctx.train_mask],
        resid_train,
        X[ctx.val_mask],
        resid_val,
        seed=ctx.seed,
        params={"objective": "regression_l2"},
    )
    sigma_train = sigma_model.predict(X[ctx.train_mask])
    sigma_val = sigma_model.predict(X[ctx.val_mask])
    sigma_test = sigma_model.predict(X[ctx.test_mask])
    s0 = np.nanmedian(sigma_train)
    s1 = np.nanstd(sigma_train) + 1e-6
    w_train = 1 / (1 + np.exp((sigma_train - s0) / s1))
    w_val = 1 / (1 + np.exp((sigma_val - s0) / s1))
    w_test = 1 / (1 + np.exp((sigma_test - s0) / s1))
    climo = pd.to_numeric(ctx.df.get("feat_climo_mean_doy"), errors="coerce").to_numpy(dtype=float)
    climo = np.where(np.isnan(climo), np.nanmean(ctx.y[ctx.train_mask]), climo)
    pred_train_shrink = w_train * pred_train + (1 - w_train) * climo[ctx.train_mask]
    pred_val_shrink = w_val * pred_val + (1 - w_val) * climo[ctx.val_mask]
    pred_test_shrink = w_test * pred_test + (1 - w_test) * climo[ctx.test_mask]
    record("E43", "NGR-style mean+sigma shrinkage", best_features, (pred_train_shrink, pred_val_shrink, pred_test_shrink))

    # E44: conformal calibration on top of E42
    calib_scores = np.maximum(
        np.maximum(q_preds_val[0.1] - ctx.y[ctx.val_mask], ctx.y[ctx.val_mask] - q_preds_val[0.9]),
        0.0,
    )
    s_star = float(np.quantile(calib_scores, 0.9))
    lower = q_preds_test[0.1] - s_star
    upper = q_preds_test[0.9] + s_star
    coverage = float(np.mean((ctx.y[ctx.test_mask] >= lower) & (ctx.y[ctx.test_mask] <= upper)))
    record(
        "E44",
        "Conformalized quantile intervals",
        best_features,
        (q50_train, q50_val, q50_test),
        extra={"coverage_80_test": coverage, "pred_cols": {"lower": lower, "upper": upper}},
    )

    # E45: spread-aware shrinkage blend
    persist = pd.to_numeric(ctx.df.get("feat_tmax_lag1"), errors="coerce").to_numpy(dtype=float)
    persist = np.where(np.isnan(persist), np.nanmean(ctx.y[ctx.train_mask]), persist)
    base_pred_full = np.full(len(ctx.y), np.nan)
    base_pred_full[ctx.train_mask] = q50_train
    base_pred_full[ctx.val_mask] = q50_val
    base_pred_full[ctx.test_mask] = q50_test
    w_target = (ctx.y - persist) / (base_pred_full - persist + 1e-6)
    w_target = np.clip(w_target, 0.0, 1.0)
    w_target = np.where(np.isnan(w_target), 0.5, w_target)
    w_features = ["feat_le_spread", "feat_le_vol", "feat_le_count", "feat_tmp_max_spread_models", *DOY]
    filled, _ = impute_features(ensure_columns(ctx.df, w_features)[w_features], ctx.train_mask)
    Xw = filled.to_numpy(dtype=float)
    w_model = train_lgbm_regressor(
        Xw[ctx.train_mask],
        w_target[ctx.train_mask],
        Xw[ctx.val_mask],
        w_target[ctx.val_mask],
        seed=ctx.seed,
        params={"objective": "regression_l2"},
    )
    w_pred = np.clip(w_model.predict(Xw), 0.0, 1.0)
    pred_all = w_pred * base_pred_full + (1 - w_pred) * persist
    pred_train = pred_all[ctx.train_mask]
    pred_val = pred_all[ctx.val_mask]
    pred_test = pred_all[ctx.test_mask]
    record("E45", "Spread-aware shrinkage blend", w_features, (pred_train, pred_val, pred_test))

    summary = {
        "suite_id": args.suite_id,
        "created_utc": utc_now_iso(),
        "split": split,
        "feature_store_path": str(feature_store_path),
        "experiments": [
            {
                "experiment_id": r.experiment_id,
                "name": r.name,
                "features": r.features,
                "metrics": r.metrics,
                "extras": r.extras,
            }
            for r in results
        ],
    }

    summary_path = output_root / "experiments_summary.json"
    write_json(summary_path, summary)

    rows = []
    for r in results:
        row = {
            "experiment_id": r.experiment_id,
            "name": r.name,
            "train_mae": r.metrics.get("train", {}).get("mae"),
            "val_mae": r.metrics.get("validation", {}).get("mae"),
            "test_mae": r.metrics.get("test", {}).get("mae"),
        }
        rows.append(row)
    pd.DataFrame(rows).sort_values("test_mae").to_csv(output_root / "experiments_summary.csv", index=False)

    LOGGER.info("Wrote MOS 45-experiment suite to %s", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
