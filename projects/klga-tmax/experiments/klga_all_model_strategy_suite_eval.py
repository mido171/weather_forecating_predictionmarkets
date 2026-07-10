"""All-model KLGA Tmax strategy-suite experiment.

This implements the GPT-Pro all-model recommendation as a reproducible local
experiment: wide eligible feature extraction, scalar construction, robust
half-life bias correction, NBM-dominant core blending, and sequential capped
family residual layers.

The script is read-only against Postgres and writes CSV artifacts only.
"""

# ruff: noqa: E402 - thread caps must be installed before numeric-library imports.

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd
import psycopg
from scipy.optimize import minimize, minimize_scalar


FEATURE_BUILD_VERSION = "TMAX_THIN_V1"
CUTOFF_ID = "T_1245UTC"
TARGET_STATION_ID = "KLGA"
SOURCE_BUFFER_HOURS = 1.5
LABEL_LAG_DAYS = 2

CORE_MODELS = ("nbm", "hrrr", "gfs", "rap")
DIRECT_MODELS = ("hrrr", "nbm", "rap", "gfs")
VALID_TIME_MODELS = ("gefsatmosmean", "ifsoper", "aifsoper", "aigfssfc")
ENSEMBLE_MODELS = ("gefsatmos", "ifsenfo", "aifsenfo", "aigefssfc")


def database_dsn() -> str:
    value = os.environ.get("KLGA_DB_URL", "").strip()
    if not value:
        raise RuntimeError("KLGA_DB_URL is required for this read-only experiment")
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def half_life_weight(age_days: np.ndarray, half_life_days: float) -> np.ndarray:
    return np.power(0.5, np.asarray(age_days, dtype=float) / float(half_life_days))


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(weights) & (weights > 0)
    weights = weights[mask]
    if len(weights) == 0 or weights.sum() <= 0:
        return 0.0
    return float((weights.sum() ** 2) / np.sum(weights**2))


def huber_loss(errors: np.ndarray, delta: float) -> np.ndarray:
    errors = np.asarray(errors, dtype=float)
    abs_errors = np.abs(errors)
    return np.where(
        abs_errors <= delta,
        0.5 * errors * errors,
        delta * (abs_errors - 0.5 * delta),
    )


def weighted_huber_location(errors: np.ndarray, weights: np.ndarray, delta: float = 3.0) -> float:
    errors = np.asarray(errors, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(errors) & np.isfinite(weights) & (weights > 0)
    errors = errors[mask]
    weights = weights[mask]
    if len(errors) == 0:
        return float("nan")
    if len(errors) == 1:
        return float(errors[0])

    def objective(value: float) -> float:
        return float(np.sum(weights * huber_loss(errors - value, delta)))

    lo = float(np.nanpercentile(errors, 5) - 5.0)
    hi = float(np.nanpercentile(errors, 95) + 5.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return float(np.average(errors, weights=weights))
    result = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
    if not result.success:
        return float(np.average(errors, weights=weights))
    return float(result.x)


def clip_or_none(value: float, cap: float | None) -> float:
    if cap is None:
        return float(value)
    return float(np.clip(value, -cap, cap))


def score_from_metrics(metrics: dict[str, float | int | str | None]) -> float:
    if not metrics or metrics["n"] == 0:
        return float("inf")
    return float(metrics["mae"] + 0.20 * metrics["rmse"] + 0.10 * abs(metrics["bias"]))


def load_wide_feature_frame() -> tuple[pd.DataFrame, dict[str, int]]:
    started = time.time()
    feature_sql = f"""
    with instances as (
        select
            target_instance_id,
            target_date,
            cutoff_utc
        from gold.target_instances
        where cutoff_id = %s
          and target_station_id = %s
    ), eligible as (
        select
            i.target_date,
            i.cutoff_utc,
            fv.feature_name,
            fv.feature_value,
            fv.source_latest_run_time_utc,
            fv.max_source_available_at_utc,
            row_number() over (
                partition by i.target_date, fv.feature_name
                order by fv.source_latest_run_time_utc desc,
                         fv.max_source_available_at_utc desc
            ) as rn
        from instances i
        join gold.feature_values fv
          on fv.target_instance_id = i.target_instance_id
        where fv.feature_build_version = %s
          and fv.feature_available = true
          and fv.source_latest_run_time_utc <= i.cutoff_utc - interval '{SOURCE_BUFFER_HOURS} hours'
          and fv.max_source_available_at_utc <= i.cutoff_utc
    )
    select
        target_date,
        cutoff_utc,
        feature_name,
        feature_value
    from eligible
    where rn = 1
    """
    labels_sql = """
    select
        ti.target_date,
        max(ti.cutoff_utc) as cutoff_utc,
        w.tmax_f::float8 as actual_tmax_f
    from gold.target_instances ti
    join public.wunderground_daily_tmax w
      on w.station_id = 'KLGA'
     and w.local_date = ti.target_date
     and w.validation_status in ('accepted', 'manual_confirmed', 'accepted/manual_confirmed')
     and w.tmax_f is not null
    where ti.cutoff_id = %s
      and ti.target_station_id = %s
    group by ti.target_date, w.tmax_f
    order by ti.target_date
    """
    leak_sql = f"""
    with instances as (
        select target_instance_id, cutoff_utc
        from gold.target_instances
        where cutoff_id = %s and target_station_id = %s
    )
    select
        count(*) filter (
            where fv.source_latest_run_time_utc > i.cutoff_utc - interval '{SOURCE_BUFFER_HOURS} hours'
        ) as run_buffer_violations,
        count(*) filter (
            where fv.max_source_available_at_utc > i.cutoff_utc
        ) as availability_violations
    from instances i
    join gold.feature_values fv
      on fv.target_instance_id = i.target_instance_id
    where fv.feature_build_version = %s
      and fv.feature_available = true
    """
    with psycopg.connect(database_dsn()) as conn:
        features = pd.read_sql_query(
            feature_sql,
            conn,
            params=(CUTOFF_ID, TARGET_STATION_ID, FEATURE_BUILD_VERSION),
        )
        labels = pd.read_sql_query(labels_sql, conn, params=(CUTOFF_ID, TARGET_STATION_ID))
        with conn.cursor() as cursor:
            cursor.execute(leak_sql, (CUTOFF_ID, TARGET_STATION_ID, FEATURE_BUILD_VERSION))
            run_violations, availability_violations = cursor.fetchone()

    features["target_date"] = pd.to_datetime(features["target_date"]).dt.normalize()
    labels["target_date"] = pd.to_datetime(labels["target_date"]).dt.normalize()
    labels["cutoff_utc"] = pd.to_datetime(labels["cutoff_utc"], utc=True)
    wide = features.pivot_table(
        index="target_date",
        columns="feature_name",
        values="feature_value",
        aggfunc="last",
    ).reset_index()
    wide.columns.name = None
    df = labels.merge(wide, on="target_date", how="left").sort_values("target_date")
    df = df.reset_index(drop=True)
    checks = {
        "feature_rows_loaded": int(len(features)),
        "wide_rows": int(len(df)),
        "wide_feature_columns": int(len(wide.columns) - 1),
        "source_run_buffer_violations": int(run_violations or 0),
        "source_availability_violations": int(availability_violations or 0),
        "load_seconds": int(round(time.time() - started)),
    }
    return df, checks


def feature_col(model: str, suffix: str) -> str:
    return f"grib_{model}_klga_core_{suffix}"


def existing_column(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index, dtype=float)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    doy = out["target_date"].dt.dayofyear.astype(float)
    month = out["target_date"].dt.month.astype(float)
    out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    out["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    return out


def add_direct_scalars(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for model in DIRECT_MODELS:
        mean = existing_column(out, feature_col(model, "tmax_proxy_mean_f"))
        median = existing_column(out, feature_col(model, "tmax_proxy_median_f"))
        member0 = existing_column(out, feature_col(model, "member_0_tmax_proxy_f"))
        p10 = existing_column(out, feature_col(model, "tmax_proxy_p10_f"))
        p25 = existing_column(out, feature_col(model, "tmax_proxy_p25_f"))
        p75 = existing_column(out, feature_col(model, "tmax_proxy_p75_f"))
        p90 = existing_column(out, feature_col(model, "tmax_proxy_p90_f"))
        out[f"raw_{model}"] = mean
        out[f"raw_{model}_median"] = median
        out[f"raw_{model}_member0"] = member0
        out[f"raw_{model}_mean_median"] = 0.5 * mean + 0.5 * median
        out[f"raw_{model}_trimmed"] = 0.10 * p10 + 0.20 * p25 + 0.40 * median + 0.20 * p75 + 0.10 * p90
        out[f"raw_{model}_upper_tilt"] = 0.50 * mean + 0.25 * p75 + 0.25 * p90
        out[f"raw_{model}_lower_tilt"] = 0.50 * mean + 0.25 * p25 + 0.25 * p10
    return out


def add_valid_time_scalars(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for model in VALID_TIME_MODELS:
        t18 = existing_column(out, feature_col(model, "valid_18z_tmp_2m_f"))
        t00 = existing_column(out, feature_col(model, "valid_00z_nextday_tmp_2m_f"))
        out[f"raw_{model}"] = pd.concat([t18, t00], axis=1).max(axis=1, skipna=True)
        out[f"{model}_valid_18z"] = t18
        out[f"{model}_valid_00z_nextday"] = t00
    return out


def member_peak_frame(df: pd.DataFrame, model: str) -> pd.DataFrame:
    pattern = re.compile(rf"^grib_{re.escape(model)}_klga_core_valid_(18z|00z_nextday)_member_(\d+)_tmp_2m_f$")
    members: dict[str, dict[str, str]] = {}
    for col in df.columns:
        match = pattern.match(col)
        if not match:
            continue
        valid_time, member = match.groups()
        members.setdefault(member, {})[valid_time] = col
    peak_columns: dict[str, pd.Series] = {}
    for member, cols in members.items():
        series = []
        if "18z" in cols:
            series.append(df[cols["18z"]])
        if "00z_nextday" in cols:
            series.append(df[cols["00z_nextday"]])
        if series:
            peak_columns[f"{model}_member_{member}_peak"] = pd.concat(series, axis=1).max(axis=1, skipna=True)
    return pd.DataFrame(peak_columns, index=df.index)


def add_ensemble_scalars(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for model in ENSEMBLE_MODELS:
        peaks = member_peak_frame(out, model)
        if peaks.empty:
            out[f"raw_{model}"] = np.nan
            out[f"{model}_ens_p10"] = np.nan
            out[f"{model}_ens_p25"] = np.nan
            out[f"{model}_ens_p50"] = np.nan
            out[f"{model}_ens_p70"] = np.nan
            out[f"{model}_ens_p75"] = np.nan
            out[f"{model}_ens_p90"] = np.nan
            out[f"{model}_ens_std"] = np.nan
            out[f"{model}_ens_spread"] = np.nan
            out[f"{model}_ens_iqr"] = np.nan
            out[f"{model}_ens_skew"] = np.nan
            continue
        out[f"raw_{model}"] = peaks.mean(axis=1, skipna=True)
        out[f"{model}_ens_p10"] = peaks.quantile(0.10, axis=1, interpolation="linear")
        out[f"{model}_ens_p25"] = peaks.quantile(0.25, axis=1, interpolation="linear")
        out[f"{model}_ens_p50"] = peaks.quantile(0.50, axis=1, interpolation="linear")
        out[f"{model}_ens_p70"] = peaks.quantile(0.70, axis=1, interpolation="linear")
        out[f"{model}_ens_p75"] = peaks.quantile(0.75, axis=1, interpolation="linear")
        out[f"{model}_ens_p90"] = peaks.quantile(0.90, axis=1, interpolation="linear")
        out[f"{model}_ens_std"] = peaks.std(axis=1, skipna=True)
        out[f"{model}_ens_spread"] = out[f"{model}_ens_p90"] - out[f"{model}_ens_p10"]
        out[f"{model}_ens_iqr"] = out[f"{model}_ens_p75"] - out[f"{model}_ens_p25"]
        out[f"{model}_ens_skew"] = ((out[f"{model}_ens_p90"] + out[f"{model}_ens_p25"]) / 2.0) - out[f"{model}_ens_p50"]
    return out


def qmd_bucket_expectation(df: pd.DataFrame) -> pd.Series:
    pieces = []
    weights = []
    pattern = re.compile(r"^grib_nbmqmd_klga_core_generic_bucket_prob_(\d+)_(\d+)$")
    for col in df.columns:
        match = pattern.match(col)
        if not match:
            continue
        lo, hi = (int(v) for v in match.groups())
        pieces.append(df[col] * ((lo + hi) / 2.0))
        weights.append(df[col])
    lt = "grib_nbmqmd_klga_core_generic_bucket_prob_lt_60"
    ge = "grib_nbmqmd_klga_core_generic_bucket_prob_ge_100"
    if lt in df.columns:
        pieces.append(df[lt] * 57.5)
        weights.append(df[lt])
    if ge in df.columns:
        pieces.append(df[ge] * 102.5)
        weights.append(df[ge])
    if not pieces:
        return pd.Series(np.nan, index=df.index)
    numerator = pd.concat(pieces, axis=1).sum(axis=1, skipna=True)
    denominator = pd.concat(weights, axis=1).sum(axis=1, skipna=True)
    expectation = numerator / denominator.replace(0, np.nan)
    valid_sum = denominator.between(0.70, 1.30)
    return expectation.where(valid_sum)


def qmd_threshold_expectation(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    cols: list[tuple[int, str]] = []
    pattern = re.compile(r"^grib_nbmqmd_klga_core_prob_tmax_ge_(\d+)f$")
    for col in df.columns:
        match = pattern.match(col)
        if match:
            cols.append((int(match.group(1)), col))
    if not cols:
        nan = pd.Series(np.nan, index=df.index)
        return nan, nan
    cols.sort()
    thresholds = np.array([item[0] for item in cols], dtype=float)
    values = df[[item[1] for item in cols]].to_numpy(float)
    expected = np.full(len(df), np.nan)
    width = np.full(len(df), np.nan)
    for idx in range(len(df)):
        surv = values[idx]
        mask = np.isfinite(surv)
        if mask.sum() < 5:
            continue
        tau = thresholds[mask]
        s = np.clip(surv[mask], 0.0, 1.0)
        s = np.minimum.accumulate(s)
        tau_aug = np.concatenate(([tau[0] - 1.0], tau, [tau[-1] + 1.0]))
        s_aug = np.concatenate(([1.0], s, [0.0]))
        expected[idx] = tau_aug[0] + np.sum(np.diff(tau_aug) * (s_aug[:-1] + s_aug[1:]) / 2.0)
        try:
            q10 = np.interp(0.90, s_aug[::-1], tau_aug[::-1])
            q90 = np.interp(0.10, s_aug[::-1], tau_aug[::-1])
            width[idx] = q90 - q10
        except Exception:
            width[idx] = np.nan
    return pd.Series(expected, index=df.index), pd.Series(width, index=df.index)


def add_qmd_scalars(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["qmd_mean_proxy"] = existing_column(out, "grib_nbmqmd_klga_core_mean_proxy_f")
    out["qmd_bucket_expectation"] = qmd_bucket_expectation(out)
    out["qmd_threshold_expectation"], out["qmd_distribution_width"] = qmd_threshold_expectation(out)
    components = out[["qmd_mean_proxy", "qmd_bucket_expectation", "qmd_threshold_expectation"]]
    blend = (
        0.50 * out["qmd_mean_proxy"]
        + 0.30 * out["qmd_bucket_expectation"]
        + 0.20 * out["qmd_threshold_expectation"]
    )
    out["raw_nbmqmd"] = blend.where(components.notna().sum(axis=1) >= 2, out["qmd_mean_proxy"])
    for threshold in (90, 95, 100):
        out[f"qmd_prob_ge_{threshold}"] = existing_column(out, f"grib_nbmqmd_klga_core_prob_tmax_ge_{threshold}f")
    return out


def add_rtma_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rtma_tmp"] = existing_column(out, "grib_rtma_klga_core_current_tmp_2m_f")
    out["rtma_dewpoint"] = existing_column(out, "grib_rtma_klga_core_current_dewpoint_2m_f")
    out["rtma_wind"] = existing_column(out, "grib_rtma_klga_core_current_wind_speed_10m_mph")
    return out


def build_scalar_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_calendar_features(df)
    out = add_direct_scalars(out)
    out = add_valid_time_scalars(out)
    out = add_ensemble_scalars(out)
    out = add_qmd_scalars(out)
    out = add_rtma_features(out)
    return out


@dataclass(frozen=True)
class BiasConfig:
    raw_col: str
    corr_col: str
    lookback_days: int
    half_life_days: float
    min_count: int
    min_eff_count: float
    cap: float | None


def bias_configs() -> list[BiasConfig]:
    return [
        BiasConfig("raw_nbm", "corr_nbm", 45, 15.0, 10, 5.0, None),
        BiasConfig("raw_hrrr", "corr_hrrr", 45, 15.0, 10, 5.0, 4.0),
        BiasConfig("raw_gfs", "corr_gfs", 60, 21.0, 15, 7.0, 4.0),
        BiasConfig("raw_rap", "corr_rap", 60, 21.0, 15, 7.0, 4.0),
        BiasConfig("raw_gefsatmos", "corr_gefsatmos", 90, 30.0, 30, 12.0, 4.0),
        BiasConfig("raw_gefsatmosmean", "corr_gefsatmosmean", 90, 30.0, 30, 12.0, 4.0),
        BiasConfig("raw_ifsoper", "corr_ifsoper", 120, 45.0, 60, 20.0, 4.0),
        BiasConfig("raw_ifsenfo", "corr_ifsenfo", 120, 45.0, 60, 20.0, 4.0),
        BiasConfig("raw_aifsoper", "corr_aifsoper", 120, 45.0, 90, 25.0, 2.5),
        BiasConfig("raw_aifsenfo", "corr_aifsenfo", 120, 45.0, 90, 25.0, 2.5),
        BiasConfig("raw_nbmqmd", "corr_nbmqmd", 60, 21.0, 80, 20.0, 2.5),
        BiasConfig("raw_aigefssfc", "corr_aigefssfc", 60, 21.0, 60, 20.0, 1.5),
        BiasConfig("raw_aigfssfc", "corr_aigfssfc", 60, 21.0, 60, 20.0, 1.5),
    ]


def add_online_bias_corrections(df: pd.DataFrame, configs: Iterable[BiasConfig]) -> pd.DataFrame:
    out = df.copy().sort_values("target_date").reset_index(drop=True)
    for config in configs:
        out[config.corr_col] = np.nan
        out[f"bias__{config.corr_col}"] = np.nan
        out[f"bias_n__{config.corr_col}"] = 0
        out[f"bias_neff__{config.corr_col}"] = 0.0
        out[f"bias_max_label_date__{config.corr_col}"] = pd.NaT

    for idx, row in out.iterrows():
        target_date = row["target_date"]
        for config in configs:
            raw_value = row.get(config.raw_col, np.nan)
            if not np.isfinite(raw_value):
                continue
            hist_start = target_date - pd.Timedelta(days=config.lookback_days)
            hist_end = target_date - pd.Timedelta(days=LABEL_LAG_DAYS)
            hist = out.loc[
                (out["target_date"] >= hist_start)
                & (out["target_date"] <= hist_end)
                & out[config.raw_col].notna()
                & out["actual_tmax_f"].notna(),
                ["target_date", "actual_tmax_f", config.raw_col],
            ]
            if len(hist) < config.min_count:
                continue
            age = (target_date - hist["target_date"]).dt.days.to_numpy(float)
            weights = half_life_weight(age, config.half_life_days)
            n_eff = effective_sample_size(weights)
            if n_eff < config.min_eff_count:
                continue
            errors_to_add = hist["actual_tmax_f"].to_numpy(float) - hist[config.raw_col].to_numpy(float)
            bias = weighted_huber_location(errors_to_add, weights, delta=3.0)
            if not np.isfinite(bias):
                continue
            bias = clip_or_none(bias, config.cap)
            out.at[idx, config.corr_col] = raw_value + bias
            out.at[idx, f"bias__{config.corr_col}"] = bias
            out.at[idx, f"bias_n__{config.corr_col}"] = len(hist)
            out.at[idx, f"bias_neff__{config.corr_col}"] = n_eff
            out.at[idx, f"bias_max_label_date__{config.corr_col}"] = hist["target_date"].max()
    return out


def fit_convex_weights(
    train: pd.DataFrame,
    forecast_cols: list[str],
    target_date: pd.Timestamp,
    *,
    nbm_col: str,
    nbm_floor: float = 0.70,
    half_life_days: float = 180.0,
    huber_delta: float = 2.0,
    ridge_lambda: float = 0.10,
    previous_weights: np.ndarray | None = None,
) -> np.ndarray:
    x = train[forecast_cols].to_numpy(float)
    y = train["actual_tmax_f"].to_numpy(float)
    age = (target_date - train["target_date"]).dt.days.to_numpy(float)
    sample_weight = half_life_weight(age, half_life_days)
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1) & np.isfinite(sample_weight) & (sample_weight > 0)
    x = x[valid]
    y = y[valid]
    sample_weight = sample_weight[valid]
    p = len(forecast_cols)
    nbm_idx = forecast_cols.index(nbm_col)
    if len(y) < 365:
        fallback = np.zeros(p)
        fallback[nbm_idx] = 1.0
        return fallback
    sample_weight = sample_weight / sample_weight.sum()
    prior_map = {col: 0.10 for col in forecast_cols}
    prior_map[nbm_col] = 0.75
    if "corr_rap" in prior_map:
        prior_map["corr_rap"] = 0.05
    prior = np.array([prior_map[col] for col in forecast_cols], dtype=float)
    prior = prior / prior.sum()
    x0 = previous_weights if previous_weights is not None else prior

    def objective(weights: np.ndarray) -> float:
        pred = x @ weights
        err = y - pred
        return float(np.sum(sample_weight * huber_loss(err, huber_delta)) + ridge_lambda * np.sum((weights - prior) ** 2))

    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        {"type": "ineq", "fun": lambda weights: weights[nbm_idx] - nbm_floor},
    ]
    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * p,
        constraints=constraints,
        options={"maxiter": 160, "ftol": 1e-8, "disp": False},
    )
    if not result.success:
        return prior
    weights = np.maximum(result.x, 0.0)
    weights = weights / weights.sum()
    if weights[nbm_idx] < nbm_floor:
        non = [idx for idx in range(p) if idx != nbm_idx]
        non_sum = weights[non].sum()
        weights[nbm_idx] = nbm_floor
        if non_sum > 0:
            weights[non] = weights[non] / non_sum * (1.0 - nbm_floor)
        else:
            weights[non] = 0.0
    return weights


def add_core_convex(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("target_date").reset_index(drop=True)
    forecast_cols = ["corr_nbm", "corr_hrrr", "corr_gfs", "corr_rap"]
    out["pred_core_convex"] = np.nan
    out["core_train_n"] = 0
    out["core_train_neff"] = np.nan
    out["core_train_start"] = pd.NaT
    out["core_train_end"] = pd.NaT
    for col in forecast_cols:
        out[f"weight_core__{col}"] = np.nan
    previous_weights: np.ndarray | None = None
    for idx, row in out.iterrows():
        target_date = row["target_date"]
        if any(not np.isfinite(row[col]) for col in forecast_cols):
            continue
        train_start = target_date - pd.Timedelta(days=730)
        train_end = target_date - pd.Timedelta(days=LABEL_LAG_DAYS)
        mask = (
            (out["target_date"] >= train_start)
            & (out["target_date"] <= train_end)
            & out["actual_tmax_f"].notna()
        )
        for col in forecast_cols:
            mask &= out[col].notna()
        train = out.loc[mask]
        if len(train) < 365:
            if np.isfinite(row["corr_nbm"]):
                out.at[idx, "pred_core_convex"] = row["corr_nbm"]
                out.at[idx, "weight_core__corr_nbm"] = 1.0
            continue
        weights = fit_convex_weights(
            train,
            forecast_cols,
            target_date,
            nbm_col="corr_nbm",
            previous_weights=previous_weights,
        )
        previous_weights = weights
        out.at[idx, "pred_core_convex"] = float(np.dot(row[forecast_cols].to_numpy(float), weights))
        ages = (target_date - train["target_date"]).dt.days.to_numpy(float)
        sample_weight = half_life_weight(ages, 180.0)
        out.at[idx, "core_train_n"] = len(train)
        out.at[idx, "core_train_neff"] = effective_sample_size(sample_weight)
        out.at[idx, "core_train_start"] = train["target_date"].min()
        out.at[idx, "core_train_end"] = train["target_date"].max()
        for col, weight in zip(forecast_cols, weights):
            out.at[idx, f"weight_core__{col}"] = float(weight)
    out["core_spread"] = out[forecast_cols].max(axis=1) - out[forecast_cols].min(axis=1)
    out["core_mean"] = out[forecast_cols].mean(axis=1)
    out["nbm_minus_core_mean"] = out["corr_nbm"] - out["core_mean"]
    out["max_abs_nbm_disagreement"] = out[["corr_hrrr", "corr_gfs", "corr_rap"]].sub(out["corr_nbm"], axis=0).abs().max(axis=1)
    out["high_core_spread_flag"] = 0.0
    for idx, row in out.iterrows():
        target_date = row["target_date"]
        hist = out.loc[
            (out["target_date"] <= target_date - pd.Timedelta(days=LABEL_LAG_DAYS))
            & out["core_spread"].notna(),
            "core_spread",
        ]
        if len(hist) >= 180 and np.isfinite(row["core_spread"]):
            out.at[idx, "high_core_spread_flag"] = float(row["core_spread"] >= hist.quantile(0.70))
    return out


def evaluate_predictions(df: pd.DataFrame, pred_col: str, mask: pd.Series | None = None) -> dict[str, object]:
    if mask is None:
        mask = pd.Series(True, index=df.index)
    valid = mask & df[pred_col].notna() & df["actual_tmax_f"].notna()
    d = df.loc[valid].copy()
    if len(d) == 0:
        return {
            "method": pred_col,
            "n": 0,
            "date_start": None,
            "date_end": None,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "median_abs_error": np.nan,
            "within_1f": np.nan,
            "within_2f": np.nan,
            "p90_abs_error": np.nan,
            "p95_abs_error": np.nan,
            "max_abs_error": np.nan,
            "score": np.nan,
        }
    err = d[pred_col].to_numpy(float) - d["actual_tmax_f"].to_numpy(float)
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    return {
        "method": pred_col,
        "n": int(len(d)),
        "date_start": str(d["target_date"].min().date()),
        "date_end": str(d["target_date"].max().date()),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "median_abs_error": float(np.median(abs_err)),
        "within_1f": float(np.mean(abs_err <= 1.0)),
        "within_2f": float(np.mean(abs_err <= 2.0)),
        "p90_abs_error": float(np.percentile(abs_err, 90)),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
        "max_abs_error": float(np.max(abs_err)),
        "score": mae + 0.20 * rmse + 0.10 * abs(bias),
    }


def fit_linear_residual_layer(
    train: pd.DataFrame,
    feature_cols: list[str],
    residual_col: str,
    target_date: pd.Timestamp,
    *,
    half_life_days: float,
    huber_delta: float,
    ridge_lambda: float,
    coefficient_bounds: dict[str, tuple[float, float]],
    adjustment_cap_f: float,
    previous_theta: np.ndarray | None = None,
) -> tuple[np.ndarray, float, bool]:
    z = train[feature_cols].to_numpy(float)
    residual = train[residual_col].to_numpy(float)
    age = (target_date - train["target_date"]).dt.days.to_numpy(float)
    sample_weight = half_life_weight(age, half_life_days)
    valid = np.isfinite(residual) & np.all(np.isfinite(z), axis=1) & np.isfinite(sample_weight) & (sample_weight > 0)
    z = z[valid]
    residual = residual[valid]
    sample_weight = sample_weight[valid]
    if len(residual) < 100:
        return np.zeros(1 + len(feature_cols)), 0.0, False
    sample_weight = sample_weight / sample_weight.sum()

    def objective(theta: np.ndarray) -> float:
        pred_residual = theta[0] + z @ theta[1:]
        err = residual - pred_residual
        return float(np.sum(sample_weight * huber_loss(err, huber_delta)) + ridge_lambda * np.sum(theta[1:] ** 2) + 0.01 * theta[0] ** 2)

    bounds = [(-adjustment_cap_f, adjustment_cap_f)]
    for col in feature_cols:
        bounds.append(coefficient_bounds.get(col, (-np.inf, np.inf)))
    x0 = previous_theta if previous_theta is not None else np.zeros(1 + len(feature_cols))
    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": 120, "ftol": 1e-8, "disp": False},
    )
    if not result.success or not np.isfinite(result.fun):
        return np.zeros(1 + len(feature_cols)), effective_sample_size(sample_weight), False
    return result.x, effective_sample_size(sample_weight), True


def compute_skill_gate(
    out: pd.DataFrame,
    target_date: pd.Timestamp,
    pred_before_col: str,
    pred_shadow_after_col: str,
    *,
    min_prior_scored_days: int,
    skill_window_days: int = 365,
    full_gate_improvement: float = 0.03,
) -> float:
    start = target_date - pd.Timedelta(days=skill_window_days)
    end = target_date - pd.Timedelta(days=LABEL_LAG_DAYS)
    hist = out.loc[
        (out["target_date"] >= start)
        & (out["target_date"] <= end)
        & out[pred_before_col].notna()
        & out[pred_shadow_after_col].notna()
        & out["actual_tmax_f"].notna()
    ]
    if len(hist) < min_prior_scored_days:
        return 0.0
    before = evaluate_predictions(hist, pred_before_col)
    after = evaluate_predictions(hist, pred_shadow_after_col)
    delta = score_from_metrics(before) - score_from_metrics(after)
    return float(np.clip(delta / full_gate_improvement, 0.0, 1.0))


@dataclass(frozen=True)
class FamilyLayerConfig:
    name: str
    before_col: str
    after_col: str
    shadow_col: str
    adj_col: str
    feature_cols: list[str]
    coefficient_bounds: dict[str, tuple[float, float]]
    cap: float
    min_training_days: int
    training_window_days: int | None
    half_life_days: float
    ridge_lambda: float
    shrink_k: float
    min_prior_scored_days: int


def add_family_layer(df: pd.DataFrame, config: FamilyLayerConfig) -> pd.DataFrame:
    out = df.copy().sort_values("target_date").reset_index(drop=True)
    out[config.after_col] = out[config.before_col]
    out[config.shadow_col] = out[config.before_col]
    out[config.adj_col] = 0.0
    out[f"{config.name}_shadow_adj"] = 0.0
    out[f"{config.name}_skill_gate"] = 0.0
    out[f"{config.name}_history_shrink"] = 0.0
    out[f"{config.name}_train_n"] = 0
    out[f"{config.name}_train_neff"] = 0.0
    out[f"{config.name}_available"] = out[config.feature_cols].notna().all(axis=1)
    out[f"{config.name}_fit_success"] = False
    previous_theta: np.ndarray | None = None
    for idx, row in out.iterrows():
        target_date = row["target_date"]
        before = row[config.before_col]
        if not np.isfinite(before):
            continue
        if any(not np.isfinite(row[col]) for col in config.feature_cols):
            continue
        if config.training_window_days is None:
            train_start = out["target_date"].min()
        else:
            train_start = target_date - pd.Timedelta(days=config.training_window_days)
        train_end = target_date - pd.Timedelta(days=LABEL_LAG_DAYS)
        mask = (
            (out["target_date"] >= train_start)
            & (out["target_date"] <= train_end)
            & out[config.before_col].notna()
            & out["actual_tmax_f"].notna()
        )
        for col in config.feature_cols:
            mask &= out[col].notna()
        train = out.loc[mask].copy()
        if len(train) < config.min_training_days:
            continue
        train["residual_before"] = train["actual_tmax_f"] - train[config.before_col]
        theta, n_eff, ok = fit_linear_residual_layer(
            train,
            config.feature_cols,
            "residual_before",
            target_date,
            half_life_days=config.half_life_days,
            huber_delta=2.0,
            ridge_lambda=config.ridge_lambda,
            coefficient_bounds=config.coefficient_bounds,
            adjustment_cap_f=config.cap,
            previous_theta=previous_theta,
        )
        if ok:
            previous_theta = theta
        features = row[config.feature_cols].to_numpy(float)
        raw_adjustment = float(theta[0] + np.dot(features, theta[1:]))
        capped_adjustment = float(np.clip(raw_adjustment, -config.cap, config.cap))
        shrink = float(n_eff / (n_eff + config.shrink_k)) if n_eff > 0 else 0.0
        shadow_adjustment = shrink * capped_adjustment
        out.at[idx, config.shadow_col] = before + shadow_adjustment
        gate = compute_skill_gate(
            out,
            target_date,
            config.before_col,
            config.shadow_col,
            min_prior_scored_days=config.min_prior_scored_days,
        )
        accepted_adjustment = gate * shadow_adjustment
        out.at[idx, config.after_col] = before + accepted_adjustment
        out.at[idx, config.adj_col] = accepted_adjustment
        out.at[idx, f"{config.name}_shadow_adj"] = shadow_adjustment
        out.at[idx, f"{config.name}_skill_gate"] = gate
        out.at[idx, f"{config.name}_history_shrink"] = shrink
        out.at[idx, f"{config.name}_train_n"] = len(train)
        out.at[idx, f"{config.name}_train_neff"] = n_eff
        out.at[idx, f"{config.name}_fit_success"] = bool(ok)
    return out


def prepare_family_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pred_core_plus"] = out["pred_core_convex"]
    out["gefs_delta_ens"] = out["corr_gefsatmos"] - out["pred_core_plus"]
    out["gefs_delta_mean"] = out["corr_gefsatmosmean"] - out["pred_core_plus"]
    out["gefs_family_disagreement"] = out["corr_gefsatmos"] - out["corr_gefsatmosmean"]
    out["gefs_high_spread_delta"] = out["high_core_spread_flag"] * out["gefs_delta_ens"]

    # These RTMA features are recalculated after GEFS, but initialize with core.
    out["rtma_tmp_minus_base"] = out["rtma_tmp"] - out["pred_core_plus"]
    out["rtma_soft_floor_gap"] = np.maximum(0.0, out["rtma_tmp_minus_base"])
    out["rtma_warmup_remaining"] = out["pred_core_plus"] - out["rtma_tmp"]
    out["rtma_dewpoint_minus_tmp"] = out["rtma_dewpoint"] - out["rtma_tmp"]

    out["ifsoper_delta"] = np.nan
    out["ifsenfo_delta"] = np.nan
    out["ifs_family_disagreement"] = out["corr_ifsoper"] - out["corr_ifsenfo"]
    out["ifs_high_spread_delta"] = np.nan

    out["aifsoper_delta"] = np.nan
    out["aifsenfo_delta"] = np.nan
    out["aifs_family_disagreement"] = out["corr_aifsoper"] - out["corr_aifsenfo"]

    out["qmd_delta"] = np.nan
    out["qmd_mean_minus_bucket"] = out["qmd_mean_proxy"] - out["qmd_bucket_expectation"]
    out["qmd_threshold_minus_mean"] = out["qmd_threshold_expectation"] - out["qmd_mean_proxy"]

    out["ai_delta"] = np.nan
    out["ai_spread"] = out["aigefssfc_ens_spread"]
    return out


def add_all_family_layers(df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_family_features(df)
    layers: list[FamilyLayerConfig] = [
        FamilyLayerConfig(
            name="gefs",
            before_col="pred_core_plus",
            after_col="pred_core_gefs",
            shadow_col="pred_core_gefs_shadow",
            adj_col="adj_gefs",
            feature_cols=[
                "gefs_delta_ens",
                "gefs_delta_mean",
                "gefs_family_disagreement",
                "gefsatmos_ens_spread",
                "gefsatmos_ens_skew",
                "gefs_high_spread_delta",
            ],
            coefficient_bounds={
                "gefs_delta_ens": (0.0, 0.20),
                "gefs_delta_mean": (0.0, 0.15),
                "gefs_family_disagreement": (-0.10, 0.10),
                "gefsatmos_ens_spread": (-0.05, 0.05),
                "gefsatmos_ens_skew": (-0.05, 0.05),
                "gefs_high_spread_delta": (0.0, 0.15),
            },
            cap=1.25,
            min_training_days=365,
            training_window_days=730,
            half_life_days=180.0,
            ridge_lambda=0.50,
            shrink_k=75.0,
            min_prior_scored_days=80,
        )
    ]
    out = add_family_layer(out, layers[0])

    out["rtma_tmp_minus_base"] = out["rtma_tmp"] - out["pred_core_gefs"]
    out["rtma_soft_floor_gap"] = np.maximum(0.0, out["rtma_tmp_minus_base"])
    out["rtma_warmup_remaining"] = out["pred_core_gefs"] - out["rtma_tmp"]
    out["rtma_dewpoint_minus_tmp"] = out["rtma_dewpoint"] - out["rtma_tmp"]
    rtma_layer = FamilyLayerConfig(
        name="rtma",
        before_col="pred_core_gefs",
        after_col="pred_core_gefs_rtma",
        shadow_col="pred_core_gefs_rtma_shadow",
        adj_col="adj_rtma",
        feature_cols=[
            "rtma_tmp_minus_base",
            "rtma_soft_floor_gap",
            "rtma_warmup_remaining",
            "rtma_dewpoint_minus_tmp",
            "rtma_wind",
        ],
        coefficient_bounds={
            "rtma_tmp_minus_base": (-0.20, 0.20),
            "rtma_soft_floor_gap": (0.0, 0.50),
            "rtma_warmup_remaining": (-0.10, 0.10),
            "rtma_dewpoint_minus_tmp": (-0.05, 0.05),
            "rtma_wind": (-0.05, 0.05),
        },
        cap=1.25,
        min_training_days=365,
        training_window_days=730,
        half_life_days=180.0,
        ridge_lambda=0.50,
        shrink_k=75.0,
        min_prior_scored_days=80,
    )
    out = add_family_layer(out, rtma_layer)
    out["adj_gefs_rtma_total"] = np.clip(out["adj_gefs"] + out["adj_rtma"], -2.0, 2.0)
    out["pred_core_gefs_rtma"] = out["pred_core_plus"] + out["adj_gefs_rtma_total"]

    out["ifsoper_delta"] = out["corr_ifsoper"] - out["pred_core_gefs_rtma"]
    out["ifsenfo_delta"] = out["corr_ifsenfo"] - out["pred_core_gefs_rtma"]
    out["ifs_high_spread_delta"] = out["high_core_spread_flag"] * out["ifsenfo_delta"]
    ifs_layer = FamilyLayerConfig(
        name="ifs",
        before_col="pred_core_gefs_rtma",
        after_col="pred_core_gefs_rtma_ifs",
        shadow_col="pred_core_gefs_rtma_ifs_shadow",
        adj_col="adj_ifs",
        feature_cols=[
            "ifsoper_delta",
            "ifsenfo_delta",
            "ifs_family_disagreement",
            "ifsenfo_ens_spread",
            "ifs_high_spread_delta",
        ],
        coefficient_bounds={
            "ifsoper_delta": (0.0, 0.12),
            "ifsenfo_delta": (0.0, 0.12),
            "ifs_family_disagreement": (-0.08, 0.08),
            "ifsenfo_ens_spread": (-0.04, 0.04),
            "ifs_high_spread_delta": (0.0, 0.10),
        },
        cap=1.00,
        min_training_days=365,
        training_window_days=730,
        half_life_days=180.0,
        ridge_lambda=1.00,
        shrink_k=150.0,
        min_prior_scored_days=80,
    )
    out = add_family_layer(out, ifs_layer)

    out["aifsoper_delta"] = out["corr_aifsoper"] - out["pred_core_gefs_rtma_ifs"]
    out["aifsenfo_delta"] = out["corr_aifsenfo"] - out["pred_core_gefs_rtma_ifs"]
    aifs_layer = FamilyLayerConfig(
        name="aifs",
        before_col="pred_core_gefs_rtma_ifs",
        after_col="pred_core_gefs_rtma_ifs_aifs",
        shadow_col="pred_core_gefs_rtma_ifs_aifs_shadow",
        adj_col="adj_aifs",
        feature_cols=[
            "aifsoper_delta",
            "aifsenfo_delta",
            "aifsenfo_ens_spread",
            "aifs_family_disagreement",
        ],
        coefficient_bounds={
            "aifsoper_delta": (0.0, 0.08),
            "aifsenfo_delta": (0.0, 0.08),
            "aifsenfo_ens_spread": (-0.04, 0.04),
            "aifs_family_disagreement": (-0.04, 0.04),
        },
        cap=0.75,
        min_training_days=250,
        training_window_days=None,
        half_life_days=180.0,
        ridge_lambda=2.00,
        shrink_k=250.0,
        min_prior_scored_days=60,
    )
    out = add_family_layer(out, aifs_layer)

    out["qmd_delta"] = out["corr_nbmqmd"] - out["pred_core_gefs_rtma_ifs_aifs"]
    qmd_layer = FamilyLayerConfig(
        name="nbmqmd",
        before_col="pred_core_gefs_rtma_ifs_aifs",
        after_col="pred_core_gefs_rtma_ifs_aifs_nbmqmd",
        shadow_col="pred_core_gefs_rtma_ifs_aifs_nbmqmd_shadow",
        adj_col="adj_nbmqmd",
        feature_cols=[
            "qmd_delta",
            "qmd_prob_ge_90",
            "qmd_prob_ge_95",
            "qmd_distribution_width",
        ],
        coefficient_bounds={
            "qmd_delta": (0.0, 0.08),
            "qmd_prob_ge_90": (-0.04, 0.04),
            "qmd_prob_ge_95": (-0.04, 0.04),
            "qmd_distribution_width": (-0.03, 0.03),
        },
        cap=0.75,
        min_training_days=100,
        training_window_days=None,
        half_life_days=60.0,
        ridge_lambda=4.00,
        shrink_k=300.0,
        min_prior_scored_days=40,
    )
    out = add_family_layer(out, qmd_layer)

    out["ai_delta"] = out["corr_aigefssfc"] - out["pred_core_gefs_rtma_ifs_aifs_nbmqmd"]
    ai_layer = FamilyLayerConfig(
        name="ai",
        before_col="pred_core_gefs_rtma_ifs_aifs_nbmqmd",
        after_col="pred_final_all_model",
        shadow_col="pred_final_all_model_shadow",
        adj_col="adj_ai",
        feature_cols=["ai_delta", "ai_spread"],
        coefficient_bounds={"ai_delta": (0.0, 0.03), "ai_spread": (-0.02, 0.02)},
        cap=0.35,
        min_training_days=60,
        training_window_days=None,
        half_life_days=60.0,
        ridge_lambda=8.00,
        shrink_k=500.0,
        min_prior_scored_days=40,
    )
    out = add_family_layer(out, ai_layer)

    out["total_non_core_adjustment"] = out[["adj_gefs", "adj_rtma", "adj_ifs", "adj_aifs", "adj_nbmqmd", "adj_ai"]].sum(axis=1)
    return out


def same_date_ablation_rows(df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("core_plus_gefs", "pred_core_plus", "pred_core_gefs", "gefs_available"),
        ("core_plus_gefs_rtma", "pred_core_gefs", "pred_core_gefs_rtma", "rtma_available"),
        ("add_ifs", "pred_core_gefs_rtma", "pred_core_gefs_rtma_ifs", "ifs_available"),
        ("add_aifs", "pred_core_gefs_rtma_ifs", "pred_core_gefs_rtma_ifs_aifs", "aifs_available"),
        ("add_nbmqmd", "pred_core_gefs_rtma_ifs_aifs", "pred_core_gefs_rtma_ifs_aifs_nbmqmd", "nbmqmd_available"),
        ("add_ai", "pred_core_gefs_rtma_ifs_aifs_nbmqmd", "pred_final_all_model", "ai_available"),
    ]
    rows = []
    for family, before, after, available_col in comparisons:
        mask = df[available_col].fillna(False) & df[before].notna() & df[after].notna()
        base = evaluate_predictions(df, before, mask)
        candidate = evaluate_predictions(df, after, mask)
        rows.append(
            {
                "family": family,
                "n": base["n"],
                "date_start": base["date_start"],
                "date_end": base["date_end"],
                "base_method": before,
                "candidate_method": after,
                "base_mae": base["mae"],
                "candidate_mae": candidate["mae"],
                "delta_mae": candidate["mae"] - base["mae"],
                "base_rmse": base["rmse"],
                "candidate_rmse": candidate["rmse"],
                "delta_rmse": candidate["rmse"] - base["rmse"],
                "base_bias": base["bias"],
                "candidate_bias": candidate["bias"],
                "accepted_by_score": bool(candidate["score"] < base["score"]) if base["n"] else False,
            }
        )
    return pd.DataFrame(rows)


def add_final_bias_calibration(
    df: pd.DataFrame,
    *,
    base_col: str = "pred_final_all_model",
    output_col: str = "pred_final_all_model_biascal_365d",
    lookback_days: int = 365,
    half_life_days: float = 90.0,
    cap: float = 0.75,
    min_count: int = 80,
) -> pd.DataFrame:
    out = df.copy().sort_values("target_date").reset_index(drop=True)
    out[output_col] = np.nan
    out[f"bias_add__{output_col}"] = 0.0
    dates = out["target_date"]
    actual = out["actual_tmax_f"].to_numpy(float)
    base = out[base_col].to_numpy(float)
    calibrated = np.full(len(out), np.nan)
    bias_add = np.zeros(len(out))
    for idx, target_date in enumerate(dates):
        if not np.isfinite(base[idx]):
            continue
        hist_start = target_date - pd.Timedelta(days=lookback_days)
        hist_end = target_date - pd.Timedelta(days=LABEL_LAG_DAYS)
        lo = np.searchsorted(dates.values, np.datetime64(hist_start), side="left")
        hi = np.searchsorted(dates.values, np.datetime64(hist_end), side="right")
        hist_idx = np.arange(lo, hi)
        valid = np.isfinite(base[hist_idx]) & np.isfinite(actual[hist_idx])
        hist_idx = hist_idx[valid]
        if len(hist_idx) < min_count:
            calibrated[idx] = base[idx]
            continue
        age = np.array([(target_date - dates.iloc[j]).days for j in hist_idx], dtype=float)
        weights = half_life_weight(age, half_life_days)
        errors_to_add = actual[hist_idx] - base[hist_idx]
        add = float(np.sum(errors_to_add * weights) / np.sum(weights))
        add = float(np.clip(add, -cap, cap))
        bias_add[idx] = add
        calibrated[idx] = base[idx] + add
    out[output_col] = calibrated
    out[f"bias_add__{output_col}"] = bias_add
    return out


def method_comparison(df: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "raw_nbm",
        "corr_nbm",
        "raw_hrrr",
        "corr_hrrr",
        "raw_gfs",
        "corr_gfs",
        "raw_rap",
        "corr_rap",
        "pred_core_convex",
        "pred_core_gefs",
        "pred_core_gefs_rtma",
        "pred_core_gefs_rtma_ifs",
        "pred_core_gefs_rtma_ifs_aifs",
        "pred_core_gefs_rtma_ifs_aifs_nbmqmd",
        "pred_final_all_model",
        "pred_final_all_model_biascal_365d",
    ]
    primary_mask = df["pred_core_convex"].notna()
    rows = [evaluate_predictions(df, method, primary_mask) for method in methods if method in df.columns]
    return pd.DataFrame(rows).sort_values(["mae", "rmse"], na_position="last")


def adjustment_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    components = [
        "weight_core__corr_nbm",
        "weight_core__corr_hrrr",
        "weight_core__corr_gfs",
        "weight_core__corr_rap",
        "adj_gefs",
        "adj_rtma",
        "adj_ifs",
        "adj_aifs",
        "adj_nbmqmd",
        "adj_ai",
        "total_non_core_adjustment",
    ]
    for col in components:
        if col not in df.columns:
            continue
        series = df.loc[df[col].notna(), col]
        if len(series) == 0:
            continue
        rows.append(
            {
                "component": col,
                "n": len(series),
                "mean": series.mean(),
                "std": series.std(),
                "p05": series.quantile(0.05),
                "p25": series.quantile(0.25),
                "median": series.quantile(0.50),
                "p75": series.quantile(0.75),
                "p95": series.quantile(0.95),
                "min": series.min(),
                "max": series.max(),
                "cap_hit_rate": float(np.mean(np.isclose(np.abs(series), np.nanmax(np.abs(series)), atol=1e-9))) if col.startswith("adj_") else np.nan,
            }
        )
    return pd.DataFrame(rows)


def regime_metrics(df: pd.DataFrame) -> pd.DataFrame:
    methods = ["corr_nbm", "pred_core_convex", "pred_core_gefs_rtma", "pred_final_all_model"]
    selected = df.loc[df["pred_core_convex"].notna()].copy()
    spread = selected["core_spread"]
    selected["spread_bucket"] = pd.cut(
        spread.rank(pct=True),
        bins=[0.0, 0.33, 0.67, 0.90, 1.0],
        labels=["low", "medium", "high", "very_high"],
        include_lowest=True,
    )
    selected["season"] = selected["target_date"].dt.month.map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "fall",
            10: "fall",
            11: "fall",
        }
    )
    rows = []
    for method in methods:
        for season, group in selected.groupby("season", observed=True):
            result = evaluate_predictions(group, method)
            rows.append({"dimension": "season", "bucket": season, **result})
        for bucket, group in selected.groupby("spread_bucket", observed=True):
            result = evaluate_predictions(group, method)
            rows.append({"dimension": "spread_bucket", "bucket": str(bucket), **result})
        for name, mask in {
            "actual_ge_90": selected["actual_tmax_f"] >= 90,
            "actual_ge_95": selected["actual_tmax_f"] >= 95,
            "actual_le_40": selected["actual_tmax_f"] <= 40,
            "actual_le_32": selected["actual_tmax_f"] <= 32,
        }.items():
            result = evaluate_predictions(selected, method, mask)
            rows.append({"dimension": "extreme", "bucket": name, **result})
    return pd.DataFrame(rows)


def leakage_checks(df: pd.DataFrame, load_checks: dict[str, int]) -> pd.DataFrame:
    rows = [
        {"check": "source_run_buffer_violations", "violations": load_checks["source_run_buffer_violations"]},
        {"check": "source_availability_violations", "violations": load_checks["source_availability_violations"]},
    ]
    for config in bias_configs():
        col = f"bias_max_label_date__{config.corr_col}"
        if col not in df.columns:
            continue
        valid = df[col].notna() & df[config.corr_col].notna()
        violations = valid & (df[col] > df["target_date"] - pd.Timedelta(days=LABEL_LAG_DAYS))
        rows.append({"check": f"bias_label_leak__{config.corr_col}", "violations": int(violations.sum())})
    core_train_valid = df["core_train_end"].notna() & df["pred_core_convex"].notna()
    core_violations = core_train_valid & (df["core_train_end"] > df["target_date"] - pd.Timedelta(days=LABEL_LAG_DAYS))
    rows.append({"check": "core_weight_label_leak", "violations": int(core_violations.sum())})
    return pd.DataFrame(rows)


def scalar_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in df.columns if c.startswith("raw_") or c.startswith("corr_")]:
        valid = df[col].notna()
        if not valid.any():
            rows.append({"column": col, "n": 0, "start": None, "end": None})
            continue
        rows.append(
            {
                "column": col,
                "n": int(valid.sum()),
                "start": str(df.loc[valid, "target_date"].min().date()),
                "end": str(df.loc[valid, "target_date"].max().date()),
            }
        )
    return pd.DataFrame(rows).sort_values(["n", "column"], ascending=[False, True])


def examples(df: pd.DataFrame) -> pd.DataFrame:
    selected = df.loc[df["pred_core_convex"].notna() & df["pred_final_all_model"].notna()].copy()
    selected["core_abs_error"] = (selected["pred_core_convex"] - selected["actual_tmax_f"]).abs()
    selected["final_abs_error"] = (selected["pred_final_all_model"] - selected["actual_tmax_f"]).abs()
    selected["final_minus_core_abs_improvement"] = selected["core_abs_error"] - selected["final_abs_error"]
    cols = [
        "target_date",
        "actual_tmax_f",
        "corr_nbm",
        "pred_core_convex",
        "pred_core_gefs_rtma",
        "pred_final_all_model",
        "final_minus_core_abs_improvement",
        "adj_gefs",
        "adj_rtma",
        "adj_ifs",
        "adj_aifs",
        "adj_nbmqmd",
        "adj_ai",
        "core_spread",
    ]
    return pd.concat(
        [
            selected.sort_values("final_minus_core_abs_improvement", ascending=False).head(10),
            selected.sort_values("final_minus_core_abs_improvement", ascending=True).head(10),
        ]
    )[cols]


def write_outputs(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Acknowledge the read-only database scan and bounded local experiment.",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("database experiment is disabled; re-run with --execute")

    started = time.time()
    print("Loading eligible feature rows and WU labels...")
    base_df, load_checks = load_wide_feature_frame()
    print(json.dumps(load_checks, sort_keys=True))

    print("Building scalar bank...")
    df = build_scalar_frame(base_df)

    print("Applying robust half-life bias corrections...")
    df = add_online_bias_corrections(df, bias_configs())

    print("Fitting rolling NBM-dominant core convex blend...")
    df = add_core_convex(df)

    print("Fitting sequential family residual layers...")
    df = add_all_family_layers(df)
    df = add_final_bias_calibration(df)

    tables = {
        "overall_method_comparison": method_comparison(df),
        "same_date_family_ablation": same_date_ablation_rows(df),
        "adjustment_summary": adjustment_summary(df),
        "regime_metrics": regime_metrics(df),
        "leakage_checks": leakage_checks(df, load_checks),
        "scalar_coverage": scalar_coverage(df),
        "help_hurt_examples": examples(df),
        "daily_predictions": df[
            [
                "target_date",
                "cutoff_utc",
                "actual_tmax_f",
                "raw_nbm",
                "corr_nbm",
                "corr_hrrr",
                "corr_gfs",
                "corr_rap",
                "pred_core_convex",
                "pred_core_gefs",
                "pred_core_gefs_rtma",
                "pred_core_gefs_rtma_ifs",
                "pred_core_gefs_rtma_ifs_aifs",
                "pred_core_gefs_rtma_ifs_aifs_nbmqmd",
                "pred_final_all_model",
                "pred_final_all_model_biascal_365d",
                "adj_gefs",
                "adj_rtma",
                "adj_ifs",
                "adj_aifs",
                "adj_nbmqmd",
                "adj_ai",
                "total_non_core_adjustment",
                "core_spread",
                "high_core_spread_flag",
            ]
        ],
    }
    write_outputs(Path(args.output_dir), tables)

    print("\n=== OVERALL METHOD COMPARISON ===")
    print(tables["overall_method_comparison"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== SAME-DATE FAMILY ABLATION ===")
    print(tables["same_date_family_ablation"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== LEAKAGE CHECKS ===")
    print(tables["leakage_checks"].to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")
    print(f"Elapsed seconds: {time.time() - started:.1f}")


if __name__ == "__main__":
    main()
