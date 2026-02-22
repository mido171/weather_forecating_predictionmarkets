"""MOS feature extraction helpers."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from .mos_config import MosDatasetConfig
from .mos_constants import DEFAULT_VARIABLES
from .mos_utils import normalize_sql, sha256_hex

import logging

LOGGER = logging.getLogger(__name__)


def _log_mos_debug(engine, station_id: str, start_target: date, end_target: date) -> None:
    try:
        sql_total = """
            SELECT COUNT(*) AS c
            FROM weather_predictionmarkets.mos_daily_value
            WHERE station_id = :station_id
              AND target_date_local BETWEEN :start_target AND :end_target
        """
        total = pd.read_sql(
            text(sql_total),
            engine,
            params={
                "station_id": station_id,
                "start_target": start_target,
                "end_target": end_target,
            },
        )
        if not total.empty:
            LOGGER.warning("MOS_DEBUG total_rows=%d", int(total.iloc[0]["c"]))

        sql_models = """
            SELECT model, COUNT(*) AS c
            FROM weather_predictionmarkets.mos_daily_value
            WHERE station_id = :station_id
              AND target_date_local BETWEEN :start_target AND :end_target
            GROUP BY model
            ORDER BY c DESC
        """
        models = pd.read_sql(
            text(sql_models),
            engine,
            params={
                "station_id": station_id,
                "start_target": start_target,
                "end_target": end_target,
            },
        )
        if not models.empty:
            LOGGER.warning("MOS_DEBUG models=%s", models.to_dict(orient="records")[:10])

        sql_vars = """
            SELECT variable_code, COUNT(*) AS c
            FROM weather_predictionmarkets.mos_daily_value
            WHERE station_id = :station_id
              AND target_date_local BETWEEN :start_target AND :end_target
            GROUP BY variable_code
            ORDER BY c DESC
        """
        vars_df = pd.read_sql(
            text(sql_vars),
            engine,
            params={
                "station_id": station_id,
                "start_target": start_target,
                "end_target": end_target,
            },
        )
        if not vars_df.empty:
            LOGGER.warning("MOS_DEBUG variable_codes=%s", vars_df.to_dict(orient="records")[:20])

        sql_hours = """
            SELECT HOUR(asof_utc) AS hr, COUNT(*) AS c
            FROM weather_predictionmarkets.mos_daily_value
            WHERE station_id = :station_id
              AND target_date_local BETWEEN :start_target AND :end_target
            GROUP BY hr
            ORDER BY hr
        """
        hours = pd.read_sql(
            text(sql_hours),
            engine,
            params={
                "station_id": station_id,
                "start_target": start_target,
                "end_target": end_target,
            },
        )
        if not hours.empty:
            LOGGER.warning("MOS_DEBUG asof_hours=%s", hours.to_dict(orient="records"))
    except Exception as exc:
        LOGGER.warning("MOS_DEBUG_FAILED %s", exc)

def fetch_mos_rows(
    engine,
    cfg: MosDatasetConfig,
    start_target: date,
    end_target: date,
) -> tuple[pd.DataFrame, str]:
    models = cfg.models or ["GFS", "NAM"]
    variables = cfg.variables or DEFAULT_VARIABLES
    model_placeholders = ", ".join([f":m{i}" for i in range(len(models))])
    var_placeholders = ", ".join([f":v{i}" for i in range(len(variables))])
    sql = f"""
        SELECT
          station_id, station_zoneid, model, asof_utc, runtime_utc, target_date_local, variable_code,
          value_min, value_max, value_mean, value_median, sample_count,
          first_forecast_time_utc, last_forecast_time_utc,
          raw_payload_hash_ref, retrieved_at_utc
        FROM weather_predictionmarkets.mos_daily_value
        WHERE station_id = :station_id
          AND target_date_local BETWEEN :start_target AND :end_target
          AND UPPER(model) IN ({model_placeholders})
          AND LOWER(variable_code) IN ({var_placeholders})
        ORDER BY target_date_local, model, variable_code, runtime_utc
    """
    params: dict[str, Any] = {
        "station_id": cfg.station_id,
        "start_target": start_target,
        "end_target": end_target,
    }
    for i, model in enumerate(models):
        params[f"m{i}"] = model
    for i, var in enumerate(variables):
        params[f"v{i}"] = var
    sql_hash = sha256_hex(normalize_sql(sql))
    df = pd.read_sql(text(sql), engine, params=params)
    if df.empty:
        LOGGER.warning(
            "MOS_FETCH_EMPTY station=%s start=%s end=%s models=%s vars=%s",
            cfg.station_id,
            start_target,
            end_target,
            models,
            variables,
        )
        _log_mos_debug(engine, cfg.station_id, start_target, end_target)
        return df, sql_hash
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["runtime_utc"] = pd.to_datetime(df["runtime_utc"], utc=True)
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
    df["retrieved_at_utc"] = pd.to_datetime(df["retrieved_at_utc"], utc=True)
    df["first_forecast_time_utc"] = pd.to_datetime(df["first_forecast_time_utc"], utc=True)
    df["last_forecast_time_utc"] = pd.to_datetime(df["last_forecast_time_utc"], utc=True)
    df["model"] = df["model"].astype(str).str.lower()
    df["variable_code"] = df["variable_code"].astype(str).str.lower()
    LOGGER.info("MOS_FETCH_ROWS %d", len(df))
    LOGGER.info("MOS_FETCH_MODELS %s", sorted(df["model"].dropna().unique().tolist()))
    LOGGER.info("MOS_FETCH_VARIABLES %s", sorted(df["variable_code"].dropna().unique().tolist())[:50])
    return df, sql_hash


def select_latest_mos(
    mos_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    cfg: MosDatasetConfig,
) -> pd.DataFrame:
    if mos_df.empty:
        return mos_df
    merged = mos_df.merge(
        cal_df[["target_date_local", "asof_utc"]],
        on="target_date_local",
        how="left",
        suffixes=("", "_cutoff"),
    )
    merged = merged.rename(columns={"asof_utc_cutoff": "asof_utc_cutoff"})
    target_dt = pd.to_datetime(merged["target_date_local"])
    expected_runtime = (
        target_dt
        + pd.to_timedelta(cfg.mos_runtime_hour_utc, unit="h")
        + pd.to_timedelta(cfg.mos_runtime_minute_utc, unit="m")
        - pd.Timedelta(days=1)
    ).dt.tz_localize("UTC")
    merged["runtime_expected_utc"] = expected_runtime
    merged["runtime_match"] = merged["runtime_utc"] == merged["runtime_expected_utc"]
    total_rows = len(merged)
    eligible_asof = merged[merged["asof_utc"] <= merged["asof_utc_cutoff"]]
    eligible_runtime = merged[merged["runtime_utc"] <= merged["asof_utc_cutoff"]]
    LOGGER.info(
        "MOS_SELECT_COUNTS total=%d eligible_asof=%d eligible_runtime=%d",
        total_rows,
        len(eligible_asof),
        len(eligible_runtime),
    )
    eligible = eligible_asof.copy()
    before = len(eligible)
    eligible = eligible[eligible["runtime_utc"] <= eligible["asof_utc"]]
    LOGGER.info("MOS_SELECT_RUNTIME_LEQ_ASOF before=%d after=%d", before, len(eligible))
    if cfg.include_retrieved_at_guard:
        before = len(eligible)
        eligible = eligible[eligible["retrieved_at_utc"] <= eligible["asof_utc_cutoff"]]
        LOGGER.info("MOS_SELECT_RETRIEVED_GUARD before=%d after=%d", before, len(eligible))
    if eligible.empty:
        LOGGER.warning("MOS_SELECT_EMPTY after guards")
        return eligible

    runtime_policy = (cfg.mos_runtime_policy or "latest_before_cutoff").lower()
    if runtime_policy == "exact":
        before = len(eligible)
        eligible = eligible[eligible["runtime_match"]]
        LOGGER.info("MOS_SELECT_RUNTIME_EXACT before=%d after=%d", before, len(eligible))
        if eligible.empty:
            LOGGER.warning("MOS_SELECT_RUNTIME_EXACT_EMPTY")
            return eligible

    sort_cols = ["target_date_local", "model", "variable_code", "asof_utc", "runtime_utc", "retrieved_at_utc"]
    ascending = [True, True, True, True, True, True]
    if runtime_policy == "prefer_exact":
        sort_cols.insert(3, "runtime_match")
        ascending.insert(3, True)
    eligible = eligible.sort_values(sort_cols, ascending=ascending)
    latest = eligible.drop_duplicates(
        subset=["target_date_local", "model", "variable_code"],
        keep="last",
    ).copy()
    LOGGER.info("MOS_SELECT_LATEST rows=%d", len(latest))

    latest["window_hours"] = (
        latest["last_forecast_time_utc"] - latest["first_forecast_time_utc"]
    ).dt.total_seconds() / 3600.0
    latest["first_lead_hours"] = (
        latest["first_forecast_time_utc"] - latest["runtime_utc"]
    ).dt.total_seconds() / 3600.0
    latest["last_lead_hours"] = (
        latest["last_forecast_time_utc"] - latest["runtime_utc"]
    ).dt.total_seconds() / 3600.0
    latest["runtime_hour"] = latest["runtime_utc"].dt.hour
    latest["runtime_age_hours"] = (
        latest["asof_utc_cutoff"] - latest["runtime_utc"]
    ).dt.total_seconds() / 3600.0
    latest["retrieval_lag_hours"] = (
        latest["retrieved_at_utc"] - latest["runtime_utc"]
    ).dt.total_seconds() / 3600.0

    return latest


def build_mos_pivots(
    latest: pd.DataFrame,
    *,
    col_suffix: str | None = None,
) -> pd.DataFrame:
    if latest.empty:
        return pd.DataFrame()
    index_cols = ["target_date_local"]
    output = None
    col_map = {
        "value_min": "min",
        "value_max": "max",
        "value_mean": "mean",
        "value_median": "median",
        "sample_count": "count",
        "window_hours": "window_hours",
        "first_lead_hours": "first_lead_hours",
        "last_lead_hours": "last_lead_hours",
        "runtime_hour": "runtime_hour",
        "runtime_age_hours": "runtime_age_hours",
        "retrieval_lag_hours": "retrieval_lag_hours",
    }
    for col, suffix in col_map.items():
        pivot = latest.pivot_table(
            index=index_cols,
            columns=["model", "variable_code"],
            values=col,
            aggfunc="first",
        )
        if pivot.empty:
            continue
        base_cols = [f"mos_{m}_{v}_{suffix}" for (m, v) in pivot.columns]
        if col_suffix:
            pivot.columns = [f"{col}_{col_suffix}" for col in base_cols]
        else:
            pivot.columns = base_cols
        pivot = pivot.reset_index()
        output = pivot if output is None else output.merge(pivot, on=index_cols, how="outer")
    return output if output is not None else pd.DataFrame(columns=index_cols)


def compute_baseline_medians(
    latest: pd.DataFrame,
    cfg: MosDatasetConfig,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    if latest.empty:
        return {}, {}
    if cfg.baseline_start and cfg.baseline_end:
        mask = (latest["target_date_local"] >= cfg.baseline_start) & (
            latest["target_date_local"] <= cfg.baseline_end
        )
        base = latest[mask]
    else:
        base = latest
    if base.empty:
        return {}, {}
    count_median = (
        base.groupby(["model", "variable_code"])["sample_count"].median().to_dict()
    )
    window_median = (
        base.groupby(["model", "variable_code"])["window_hours"].median().to_dict()
    )
    return count_median, window_median


def add_missing_flags(
    df: pd.DataFrame,
    cfg: MosDatasetConfig,
    count_median: dict[tuple[str, str], float],
    window_median: dict[tuple[str, str], float],
) -> pd.DataFrame:
    df = df.copy()
    models = [m.lower() for m in (cfg.models or ["GFS", "NAM"])]
    variables = cfg.variables or DEFAULT_VARIABLES
    for model in models:
        for var in variables:
            count_col = f"mos_{model}_{var}_count"
            window_col = f"mos_{model}_{var}_window_hours"
            missing_col = f"mos_{model}_{var}_is_missing"
            if count_col not in df.columns:
                df[count_col] = np.nan
            if window_col not in df.columns:
                df[window_col] = np.nan
            df[missing_col] = df[count_col].isna().astype(float)
            base_count = count_median.get((model, var))
            base_window = window_median.get((model, var))
            low_count_col = f"mos_{model}_{var}_count_is_low"
            short_window_col = f"mos_{model}_{var}_window_hours_is_short"
            if base_count is not None:
                df[low_count_col] = np.where(
                    df[count_col].isna(),
                    np.nan,
                    (df[count_col] < 0.5 * base_count).astype(float),
                )
            else:
                df[low_count_col] = np.nan
            if base_window is not None:
                df[short_window_col] = np.where(
                    df[window_col].isna(),
                    np.nan,
                    (df[window_col] < 0.8 * base_window).astype(float),
                )
            else:
                df[short_window_col] = np.nan
    return df


def add_shape_features(df: pd.DataFrame, cfg: MosDatasetConfig) -> pd.DataFrame:
    df = df.copy()
    models = [m.lower() for m in (cfg.models or ["GFS", "NAM"])]
    variables = cfg.variables or DEFAULT_VARIABLES
    c = 1.0
    for model in models:
        for var in variables:
            min_col = f"mos_{model}_{var}_min"
            max_col = f"mos_{model}_{var}_max"
            mean_col = f"mos_{model}_{var}_mean"
            median_col = f"mos_{model}_{var}_median"
            count_col = f"mos_{model}_{var}_count"
            for col in [min_col, max_col, mean_col, median_col, count_col]:
                if col not in df.columns:
                    df[col] = np.nan
            range_col = f"mos_shape_{model}_{var}_range"
            df[range_col] = df[max_col] - df[min_col]
            df[f"mos_shape_{model}_{var}_mean_minus_median"] = df[mean_col] - df[median_col]
            df[f"mos_shape_{model}_{var}_median_minus_min"] = df[median_col] - df[min_col]
            df[f"mos_shape_{model}_{var}_max_minus_median"] = df[max_col] - df[median_col]
            df[f"mos_shape_{model}_{var}_normalized_range"] = (
                df[max_col] - df[min_col]
            ) / (df[median_col].abs() + c)
            df[f"mos_shape_{model}_{var}_cov_proxy"] = (
                df[max_col] - df[min_col]
            ) / (df[mean_col].abs() + c)
            df[f"mos_shape_{model}_{var}_count_log"] = np.log1p(df[count_col])
    return df


def build_bucket_pivots(
    mos_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    cfg: MosDatasetConfig,
    bucket_hours: list[int],
) -> dict[int, pd.DataFrame]:
    pivots: dict[int, pd.DataFrame] = {}
    if mos_df.empty:
        return pivots
    hours = sorted({int(h) for h in bucket_hours if int(h) > 0})
    for hours_back in hours:
        cal_bucket = cal_df.copy()
        cal_bucket["asof_utc"] = cal_bucket["asof_utc"] - pd.to_timedelta(hours_back, unit="h")
        latest = select_latest_mos(mos_df, cal_bucket, cfg)
        if latest.empty:
            pivots[hours_back] = pd.DataFrame(columns=["target_date_local"])
            continue
        pivots[hours_back] = build_mos_pivots(latest, col_suffix=f"b{hours_back}")
    return pivots


def compute_update_counts(
    mos_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    cfg: MosDatasetConfig,
) -> pd.DataFrame:
    if mos_df.empty:
        return pd.DataFrame(columns=["target_date_local"])
    merged = mos_df.merge(
        cal_df[["target_date_local", "asof_utc"]],
        on="target_date_local",
        how="left",
        suffixes=("", "_cutoff"),
    ).rename(columns={"asof_utc_cutoff": "asof_utc_cutoff"})
    eligible = merged[merged["asof_utc"] <= merged["asof_utc_cutoff"]].copy()
    eligible = eligible[eligible["runtime_utc"] <= eligible["asof_utc"]]
    if cfg.include_retrieved_at_guard:
        eligible = eligible[eligible["retrieved_at_utc"] <= eligible["asof_utc_cutoff"]]
    if eligible.empty:
        return pd.DataFrame(columns=["target_date_local"])
    counts = (
        eligible.groupby(["target_date_local", "model", "variable_code"])["asof_utc"]
        .nunique()
        .reset_index()
    )
    pivot = counts.pivot_table(
        index=["target_date_local"],
        columns=["model", "variable_code"],
        values="asof_utc",
        aggfunc="first",
    )
    if pivot.empty:
        return pd.DataFrame(columns=["target_date_local"])
    pivot.columns = [f"mos_{m}_{v}_update_count" for (m, v) in pivot.columns]
    return pivot.reset_index()


def add_revision_features(
    df: pd.DataFrame,
    cfg: MosDatasetConfig,
    bucket_hours: list[int],
    *,
    stats: list[str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    hours = sorted({int(h) for h in bucket_hours if int(h) > 0})
    if not hours:
        return df
    models = [m.lower() for m in (cfg.models or ["GFS", "NAM"])]
    variables = cfg.variables or DEFAULT_VARIABLES
    stats = stats or ["min", "max", "mean", "median"]
    for hours_back in hours:
        suffix = f"b{hours_back}"
        for model in models:
            for var in variables:
                for stat in stats:
                    base_col = f"mos_{model}_{var}_{stat}"
                    bucket_col = f"{base_col}_{suffix}"
                    if base_col not in df.columns or bucket_col not in df.columns:
                        continue
                    rev_col = f"mos_rev_{model}_{var}_{stat}_{suffix}"
                    slope_col = f"mos_rev_slope_{model}_{var}_{stat}_{suffix}"
                    df[rev_col] = df[base_col] - df[bucket_col]
                    df[slope_col] = df[rev_col] / float(hours_back)
    return df


def add_cross_model_features(df: pd.DataFrame, cfg: MosDatasetConfig) -> pd.DataFrame:
    df = df.copy()
    variables = cfg.variables or DEFAULT_VARIABLES
    stats = ["min", "max", "mean", "median", "count"]
    eps = 1e-6

    def sanitize_temp(series: pd.Series) -> pd.Series:
        """Coerce to numeric and drop implausible Fahrenheit values.

        This guards against MOS missing/sentinel encodings (e.g. 999) that would
        otherwise poison derived features and KNN analog residuals downstream.
        """

        series = pd.to_numeric(series, errors="coerce")
        return series.where((series >= -100.0) & (series <= 200.0))

    temp_like_vars = {"n_x", "tmp", "dpt"}

    # Capture outlier/sentinel signals *before* we sanitize in-place below.
    raw_gfs_nx_max = pd.to_numeric(df.get("mos_gfs_n_x_max"), errors="coerce")
    raw_nam_nx_max = pd.to_numeric(df.get("mos_nam_n_x_max"), errors="coerce")
    df["base_tmax_gfs_is_outlier"] = (
        ((raw_gfs_nx_max < -100.0) | (raw_gfs_nx_max > 200.0)).fillna(False).astype(float)
    )
    df["base_tmax_nam_is_outlier"] = (
        ((raw_nam_nx_max < -100.0) | (raw_nam_nx_max > 200.0)).fillna(False).astype(float)
    )
    for var in variables:
        for stat in stats:
            a_col = f"mos_gfs_{var}_{stat}"
            b_col = f"mos_nam_{var}_{stat}"
            if a_col not in df.columns or b_col not in df.columns:
                continue
            if var in temp_like_vars and stat in {"min", "max", "mean", "median"}:
                df[a_col] = sanitize_temp(df[a_col])
                df[b_col] = sanitize_temp(df[b_col])
            a = df[a_col]
            b = df[b_col]
            df[f"mos_xmodel_spread_{var}_{stat}"] = a - b
            df[f"mos_xmodel_abs_spread_{var}_{stat}"] = (a - b).abs()
            df[f"mos_xmodel_ratio_{var}_{stat}"] = a / (b + eps)
            df[f"mos_xmodel_blend_{var}_{stat}"] = 0.5 * (a + b)
            abs_spread = (a - b).abs()
            df[f"mos_xmodel_agree_{var}_{stat}_lt_1"] = (abs_spread < 1).astype(float)
            df[f"mos_xmodel_agree_{var}_{stat}_lt_2"] = (abs_spread < 2).astype(float)
            df[f"mos_xmodel_agree_{var}_{stat}_lt_3"] = (abs_spread < 3).astype(float)

    gfs = df.get("mos_gfs_n_x_max", pd.Series(np.nan, index=df.index))
    nam = df.get("mos_nam_n_x_max", pd.Series(np.nan, index=df.index))
    gfs = sanitize_temp(gfs)
    nam = sanitize_temp(nam)
    df["base_tmax_gfs"] = gfs
    df["base_tmax_nam"] = nam

    # Robust blend: average when both present; otherwise fall back to the available model.
    blend = 0.5 * (gfs + nam)
    blend = blend.where(~gfs.isna(), nam)
    blend = blend.where(~nam.isna(), gfs)
    df["base_tmax_blend"] = blend
    df["base_tmax_abs_spread"] = (gfs - nam).abs()

    wdr_blend = df.get("mos_xmodel_blend_wdr_mean")
    if wdr_blend is not None:
        radians = np.deg2rad(wdr_blend)
        df["mos_xmodel_blend_wdr_mean_sin"] = np.sin(radians)
        df["mos_xmodel_blend_wdr_mean_cos"] = np.cos(radians)
    else:
        df["mos_xmodel_blend_wdr_mean_sin"] = np.nan
        df["mos_xmodel_blend_wdr_mean_cos"] = np.nan
    return df
