"""Leakage-safe KLGA Tmax NBM-residual stacking experiment.

Reads the model-ready GribStream feature values from gold.feature_values and
settled KLGA Tmax labels from public.wunderground_daily_tmax. It does not
mutate the database.
"""

# ruff: noqa: E402 - thread caps must be installed before numeric-library imports.

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

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
from scipy.optimize import minimize


FEATURE_BUILD_VERSION = "TMAX_THIN_V1"
CUTOFF_ID = "T_1245UTC"
BUFFER_HOURS = 1.5
LABEL_LAG_DAYS = 2

MODELS = ["hrrr", "nbm", "rap", "gfs"]
BLEND_ORDER = ["nbm", "hrrr", "gfs", "rap"]
DELTA_ORDER = ["hrrr", "gfs", "rap"]

RAW_FEATURES = {
    "hrrr": "grib_hrrr_klga_core_tmax_proxy_mean_f",
    "nbm": "grib_nbm_klga_core_tmax_proxy_mean_f",
    "rap": "grib_rap_klga_core_tmax_proxy_mean_f",
    "gfs": "grib_gfs_klga_core_tmax_proxy_mean_f",
}


def half_life_weights(age_days: np.ndarray, half_life_days: float) -> np.ndarray:
    return np.power(0.5, age_days / half_life_days)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def huber_loss(error: np.ndarray, delta: float) -> np.ndarray:
    abs_error = np.abs(error)
    return np.where(
        abs_error <= delta,
        0.5 * error * error,
        delta * (abs_error - 0.5 * delta),
    )


def load_core_dataset() -> pd.DataFrame:
    sql = f"""
    with eligible as (
        select
            ti.target_date,
            ti.cutoff_utc,
            fv.feature_name,
            fv.feature_value,
            fv.source_latest_run_time_utc,
            fv.max_source_available_at_utc
        from gold.feature_values fv
        join gold.target_instances ti
          on ti.target_instance_id = fv.target_instance_id
        where fv.feature_build_version = %s
          and ti.cutoff_id = %s
          and ti.target_station_id = 'KLGA'
          and fv.feature_available = true
          and fv.feature_name = any(%s)
          and fv.source_latest_run_time_utc <= ti.cutoff_utc - interval '{BUFFER_HOURS} hours'
          and fv.max_source_available_at_utc <= ti.cutoff_utc
    )
    select
        e.target_date,
        max(e.cutoff_utc) as cutoff_utc,
        w.tmax_f::float8 as actual_tmax_f,
        max(w.validation_status) as wu_validation_status,
        max(w.daily_high_source) as wu_daily_high_source,
        max(case when e.feature_name = '{RAW_FEATURES["hrrr"]}' then e.feature_value end) as raw_hrrr,
        max(case when e.feature_name = '{RAW_FEATURES["nbm"]}' then e.feature_value end) as raw_nbm,
        max(case when e.feature_name = '{RAW_FEATURES["rap"]}' then e.feature_value end) as raw_rap,
        max(case when e.feature_name = '{RAW_FEATURES["gfs"]}' then e.feature_value end) as raw_gfs,
        max(case when e.feature_name = '{RAW_FEATURES["hrrr"]}' then e.source_latest_run_time_utc end) as run_hrrr,
        max(case when e.feature_name = '{RAW_FEATURES["nbm"]}' then e.source_latest_run_time_utc end) as run_nbm,
        max(case when e.feature_name = '{RAW_FEATURES["rap"]}' then e.source_latest_run_time_utc end) as run_rap,
        max(case when e.feature_name = '{RAW_FEATURES["gfs"]}' then e.source_latest_run_time_utc end) as run_gfs,
        max(case when e.feature_name = '{RAW_FEATURES["hrrr"]}' then e.max_source_available_at_utc end) as avail_hrrr,
        max(case when e.feature_name = '{RAW_FEATURES["nbm"]}' then e.max_source_available_at_utc end) as avail_nbm,
        max(case when e.feature_name = '{RAW_FEATURES["rap"]}' then e.max_source_available_at_utc end) as avail_rap,
        max(case when e.feature_name = '{RAW_FEATURES["gfs"]}' then e.max_source_available_at_utc end) as avail_gfs
    from eligible e
    join public.wunderground_daily_tmax w
      on w.station_id = 'KLGA'
     and w.local_date = e.target_date
     and w.validation_status in ('accepted', 'manual_confirmed', 'accepted/manual_confirmed')
     and w.tmax_f is not null
    group by e.target_date, w.tmax_f
    order by e.target_date
    """
    database_url = os.environ.get("KLGA_DB_URL", "").strip()
    if not database_url:
        raise RuntimeError("KLGA_DB_URL is required for this read-only experiment")
    database_url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(database_url) as conn:
        df = pd.read_sql_query(
            sql,
            conn,
            params=(FEATURE_BUILD_VERSION, CUTOFF_ID, list(RAW_FEATURES.values())),
        )

    df["target_date"] = pd.to_datetime(df["target_date"]).dt.normalize()
    df["cutoff_utc"] = pd.to_datetime(df["cutoff_utc"], utc=True)
    for model in MODELS:
        df[f"run_{model}"] = pd.to_datetime(df[f"run_{model}"], utc=True)
        df[f"avail_{model}"] = pd.to_datetime(df[f"avail_{model}"], utc=True)
    return df.sort_values("target_date").reset_index(drop=True)


def compute_online_bias_corrected(
    df: pd.DataFrame,
    *,
    lookback_days: int = 45,
    half_life_days: float = 15.0,
    min_days: int = 10,
    label_lag_days: int = LABEL_LAG_DAYS,
) -> pd.DataFrame:
    out = df.copy().sort_values("target_date").reset_index(drop=True)
    for model in MODELS:
        out[f"corr_{model}"] = np.nan
        out[f"bias_add_{model}"] = np.nan
        out[f"bias_n_{model}"] = 0
        out[f"bias_max_label_date_{model}"] = pd.NaT

    for index, row in out.iterrows():
        target_date = row["target_date"]
        train_start = target_date - pd.Timedelta(days=lookback_days)
        train_end = target_date - pd.Timedelta(days=label_lag_days)
        base_mask = (
            (out["target_date"] >= train_start)
            & (out["target_date"] <= train_end)
            & out["actual_tmax_f"].notna()
        )
        for model in MODELS:
            raw_col = f"raw_{model}"
            if not np.isfinite(row[raw_col]):
                continue
            hist = out.loc[
                base_mask & out[raw_col].notna(),
                ["target_date", "actual_tmax_f", raw_col],
            ]
            if len(hist) < min_days:
                continue
            age_days = (target_date - hist["target_date"]).dt.days.to_numpy(float)
            weights = half_life_weights(age_days, half_life_days)
            errors_to_add = (
                hist["actual_tmax_f"].to_numpy(float) - hist[raw_col].to_numpy(float)
            )
            bias_add = weighted_mean(errors_to_add, weights)
            if not np.isfinite(bias_add):
                continue
            out.at[index, f"corr_{model}"] = row[raw_col] + bias_add
            out.at[index, f"bias_add_{model}"] = bias_add
            out.at[index, f"bias_n_{model}"] = len(hist)
            out.at[index, f"bias_max_label_date_{model}"] = hist["target_date"].max()
    return out


def fit_residual_stack(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    *,
    delta: float,
    lambda_beta: float,
    lambda_alpha: float,
    max_non_nbm_weight: float,
    previous_theta: np.ndarray | None,
) -> tuple[np.ndarray, bool]:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    valid = (
        np.isfinite(y_train)
        & np.isfinite(sample_weight)
        & (sample_weight > 0)
        & np.all(np.isfinite(x_train), axis=1)
    )
    x_train = x_train[valid]
    y_train = y_train[valid]
    sample_weight = sample_weight[valid]
    if len(y_train) < 60:
        return np.zeros(1 + len(DELTA_ORDER)), False

    sample_weight = sample_weight / np.sum(sample_weight)

    def objective(theta: np.ndarray) -> float:
        alpha = theta[0]
        beta = theta[1:]
        error = y_train - (alpha + x_train @ beta)
        return float(
            np.sum(sample_weight * huber_loss(error, delta))
            + lambda_beta * np.sum(beta * beta)
            + lambda_alpha * alpha * alpha
        )

    bounds = [(-2.0, 2.0)] + [(0.0, max_non_nbm_weight)] * len(DELTA_ORDER)
    constraints = [
        {"type": "ineq", "fun": lambda theta: max_non_nbm_weight - np.sum(theta[1:])}
    ]
    if previous_theta is None:
        x0 = np.zeros(1 + len(DELTA_ORDER))
    else:
        x0 = previous_theta.copy()
        x0[0] = np.clip(x0[0], -2.0, 2.0)
        x0[1:] = np.clip(x0[1:], 0.0, max_non_nbm_weight)
        beta_sum = x0[1:].sum()
        if beta_sum > max_non_nbm_weight and beta_sum > 0:
            x0[1:] *= max_non_nbm_weight / beta_sum

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 90, "ftol": 1e-8, "disp": False},
    )
    if not result.success or not np.isfinite(result.fun):
        return np.zeros(1 + len(DELTA_ORDER)), False

    theta = result.x.copy()
    theta[1:] = np.maximum(theta[1:], 0.0)
    beta_sum = theta[1:].sum()
    if beta_sum > max_non_nbm_weight and beta_sum > 0:
        theta[1:] *= max_non_nbm_weight / beta_sum
    return theta, True


def walk_forward_residual_stack(
    df: pd.DataFrame,
    *,
    training_window_days: int = 730,
    min_training_days: int = 365,
    stack_half_life_days: float = 180.0,
    delta: float = 2.0,
    lambda_beta: float = 0.10,
    lambda_alpha: float = 0.01,
    max_non_nbm_weight: float = 0.40,
    adjustment_cap_f: float = 2.5,
    use_final_bias: bool = True,
    label_lag_days: int = LABEL_LAG_DAYS,
) -> pd.DataFrame:
    out = df.copy().sort_values("target_date").reset_index(drop=True)
    for col in [
        "pred_stack",
        "pred_stack_final_bias",
        "stack_alpha",
        "stack_adjustment",
        "stack_train_n",
        "final_bias_add",
    ]:
        out[col] = np.nan
    out["stack_train_start"] = pd.NaT
    out["stack_train_end"] = pd.NaT
    out["stack_success"] = False
    for model in DELTA_ORDER:
        out[f"beta_{model}"] = np.nan

    previous_theta: np.ndarray | None = None
    for index, row in out.iterrows():
        target_date = row["target_date"]
        current_cols = ["corr_nbm"] + [f"corr_{model}" for model in DELTA_ORDER]
        if any(not np.isfinite(row[col]) for col in current_cols):
            continue

        train_start = target_date - pd.Timedelta(days=training_window_days)
        train_end = target_date - pd.Timedelta(days=label_lag_days)
        train_mask = (
            (out["target_date"] >= train_start)
            & (out["target_date"] <= train_end)
            & out["actual_tmax_f"].notna()
            & out["corr_nbm"].notna()
        )
        for model in DELTA_ORDER:
            train_mask &= out[f"corr_{model}"].notna()
        train = out.loc[train_mask]
        if len(train) < min_training_days:
            continue

        x_train = np.column_stack(
            [
                train[f"corr_{model}"].to_numpy(float)
                - train["corr_nbm"].to_numpy(float)
                for model in DELTA_ORDER
            ]
        )
        y_train = train["actual_tmax_f"].to_numpy(float) - train["corr_nbm"].to_numpy(
            float
        )
        age_days = (target_date - train["target_date"]).dt.days.to_numpy(float)
        sample_weight = half_life_weights(age_days, stack_half_life_days)
        theta, ok = fit_residual_stack(
            x_train,
            y_train,
            sample_weight,
            delta=delta,
            lambda_beta=lambda_beta,
            lambda_alpha=lambda_alpha,
            max_non_nbm_weight=max_non_nbm_weight,
            previous_theta=previous_theta,
        )
        if ok:
            previous_theta = theta

        x_current = np.array(
            [row[f"corr_{model}"] - row["corr_nbm"] for model in DELTA_ORDER],
            dtype=float,
        )
        adjustment = float(np.clip(theta[0] + np.dot(x_current, theta[1:]), -adjustment_cap_f, adjustment_cap_f))
        pred_stack = float(row["corr_nbm"] + adjustment)

        final_bias_add = 0.0
        if use_final_bias:
            fb_start = target_date - pd.Timedelta(days=45)
            fb_end = target_date - pd.Timedelta(days=label_lag_days)
            fb_mask = (
                (out["target_date"] >= fb_start)
                & (out["target_date"] <= fb_end)
                & out["actual_tmax_f"].notna()
                & out["pred_stack"].notna()
            )
            final_bias_rows = out.loc[fb_mask]
            if len(final_bias_rows) >= 20:
                fb_age = (target_date - final_bias_rows["target_date"]).dt.days.to_numpy(float)
                fb_weights = half_life_weights(fb_age, 15.0)
                fb_errors_to_add = (
                    final_bias_rows["actual_tmax_f"].to_numpy(float)
                    - final_bias_rows["pred_stack"].to_numpy(float)
                )
                final_bias_add = weighted_mean(fb_errors_to_add, fb_weights)
                if np.isfinite(final_bias_add):
                    final_bias_add = float(np.clip(final_bias_add, -0.75, 0.75))
                else:
                    final_bias_add = 0.0

        out.at[index, "pred_stack"] = pred_stack
        out.at[index, "pred_stack_final_bias"] = pred_stack + final_bias_add
        out.at[index, "stack_alpha"] = theta[0]
        out.at[index, "stack_adjustment"] = adjustment
        out.at[index, "stack_train_n"] = len(train)
        out.at[index, "stack_train_start"] = train["target_date"].min()
        out.at[index, "stack_train_end"] = train["target_date"].max()
        out.at[index, "stack_success"] = bool(ok)
        out.at[index, "final_bias_add"] = final_bias_add
        for model, beta in zip(DELTA_ORDER, theta[1:]):
            out.at[index, f"beta_{model}"] = beta
    return out


def convex_weight_candidates(min_nbm_weight: float, step: float = 0.05) -> np.ndarray:
    candidates: list[list[float]] = []
    scale = int(round(1.0 / step))
    min_nbm_units = int(math.ceil(min_nbm_weight / step - 1e-9))
    for nbm_units in range(min_nbm_units, scale + 1):
        rest_units = scale - nbm_units
        for hrrr_units in range(rest_units + 1):
            for gfs_units in range(rest_units - hrrr_units + 1):
                rap_units = rest_units - hrrr_units - gfs_units
                candidates.append(
                    [
                        nbm_units / scale,
                        hrrr_units / scale,
                        gfs_units / scale,
                        rap_units / scale,
                    ]
                )
    return np.array(candidates, dtype=float)


def add_rolling_convex_grid_blend(
    df: pd.DataFrame,
    *,
    min_nbm_weight: float,
    training_window_days: int = 730,
    min_training_days: int = 365,
    half_life_days: float = 180.0,
    label_lag_days: int = LABEL_LAG_DAYS,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    pred_col = f"pred_convex_grid_nbm{int(min_nbm_weight * 100)}"
    out[pred_col] = np.nan
    candidates = convex_weight_candidates(min_nbm_weight)
    corr_cols = [f"corr_{model}" for model in BLEND_ORDER]

    for index, row in df.iterrows():
        target_date = row["target_date"]
        if any(not np.isfinite(row[col]) for col in corr_cols):
            continue
        train_start = target_date - pd.Timedelta(days=training_window_days)
        train_end = target_date - pd.Timedelta(days=label_lag_days)
        mask = (
            (df["target_date"] >= train_start)
            & (df["target_date"] <= train_end)
            & df["actual_tmax_f"].notna()
        )
        for col in corr_cols:
            mask &= df[col].notna()
        train = df.loc[mask]
        if len(train) < min_training_days:
            continue
        x_train = train[corr_cols].to_numpy(float)
        y_train = train["actual_tmax_f"].to_numpy(float)
        age_days = (target_date - train["target_date"]).dt.days.to_numpy(float)
        sample_weight = half_life_weights(age_days, half_life_days)
        sample_weight = sample_weight / np.sum(sample_weight)
        preds = x_train @ candidates.T
        losses = np.sum(sample_weight[:, None] * huber_loss(y_train[:, None] - preds, 2.0), axis=0)
        best_weights = candidates[int(np.argmin(losses))]
        out.at[index, pred_col] = float(row[corr_cols].to_numpy(float) @ best_weights)
    return out


def add_online_performance_blends(
    df: pd.DataFrame,
    *,
    half_life_values: tuple[int, ...] = (30, 45, 60, 90),
    eta_values: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
    min_days: int = 30,
    label_lag_days: int = LABEL_LAG_DAYS,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    priors = {"nbm": 0.70, "hrrr": 0.10, "gfs": 0.10, "rap": 0.10}
    corr_cols = {model: f"corr_{model}" for model in BLEND_ORDER}

    for half_life in half_life_values:
        for eta in eta_values:
            out[f"pred_perf_hl{half_life}_eta{str(eta).replace('.', '')}"] = np.nan

    for index, row in df.iterrows():
        target_date = row["target_date"]
        if any(not np.isfinite(row[corr_cols[model]]) for model in BLEND_ORDER):
            continue
        train_start = target_date - pd.Timedelta(days=365)
        train_end = target_date - pd.Timedelta(days=label_lag_days)
        base_mask = (
            (df["target_date"] >= train_start)
            & (df["target_date"] <= train_end)
            & df["actual_tmax_f"].notna()
        )
        for half_life in half_life_values:
            losses: dict[str, float] = {}
            ok = True
            for model in BLEND_ORDER:
                col = corr_cols[model]
                hist = df.loc[base_mask & df[col].notna(), ["target_date", "actual_tmax_f", col]]
                if len(hist) < min_days:
                    ok = False
                    break
                age_days = (target_date - hist["target_date"]).dt.days.to_numpy(float)
                weights = half_life_weights(age_days, half_life)
                model_loss = np.abs(hist[col].to_numpy(float) - hist["actual_tmax_f"].to_numpy(float))
                losses[model] = weighted_mean(model_loss, weights)
            if not ok:
                continue
            for eta in eta_values:
                scores = np.array(
                    [math.log(priors[model]) - eta * losses[model] for model in BLEND_ORDER],
                    dtype=float,
                )
                scores -= np.max(scores)
                weights = np.exp(scores)
                weights = weights / weights.sum()
                pred = sum(weights[idx] * row[corr_cols[model]] for idx, model in enumerate(BLEND_ORDER))
                out.at[index, f"pred_perf_hl{half_life}_eta{str(eta).replace('.', '')}"] = float(pred)
    return out


def evaluate(df: pd.DataFrame, pred_col: str, mask: pd.Series | None = None) -> dict[str, object]:
    if mask is None:
        mask = pd.Series(True, index=df.index)
    valid = mask & df[pred_col].notna() & df["actual_tmax_f"].notna()
    evaluated = df.loc[valid]
    if len(evaluated) == 0:
        return {
            "method": pred_col,
            "n": 0,
            "start": None,
            "end": None,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "within_1f": np.nan,
            "within_2f": np.nan,
            "score": np.nan,
        }
    error = evaluated[pred_col].to_numpy(float) - evaluated["actual_tmax_f"].to_numpy(float)
    abs_error = np.abs(error)
    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(error * error)))
    return {
        "method": pred_col,
        "n": int(len(evaluated)),
        "start": str(evaluated["target_date"].min().date()),
        "end": str(evaluated["target_date"].max().date()),
        "mae": mae,
        "rmse": rmse,
        "bias": float(np.mean(error)),
        "within_1f": float(np.mean(abs_error <= 1.0)),
        "within_2f": float(np.mean(abs_error <= 2.0)),
        "score": mae + 0.20 * rmse,
    }


def summarize_weights(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = df.loc[mask]
    for col in [
        "beta_hrrr",
        "beta_gfs",
        "beta_rap",
        "stack_alpha",
        "stack_adjustment",
        "final_bias_add",
    ]:
        series = selected[col].dropna()
        if len(series) == 0:
            continue
        rows.append(
            {
                "field": col,
                "mean": series.mean(),
                "p05": series.quantile(0.05),
                "p50": series.quantile(0.50),
                "p95": series.quantile(0.95),
                "min": series.min(),
                "max": series.max(),
            }
        )
    beta_sum = selected[["beta_hrrr", "beta_gfs", "beta_rap"]].sum(axis=1)
    effective_nbm = 1.0 - beta_sum
    rows.append(
        {
            "field": "effective_nbm_weight",
            "mean": effective_nbm.mean(),
            "p05": effective_nbm.quantile(0.05),
            "p50": effective_nbm.quantile(0.50),
            "p95": effective_nbm.quantile(0.95),
            "min": effective_nbm.min(),
            "max": effective_nbm.max(),
        }
    )
    return pd.DataFrame(rows)


def regime_metrics(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    selected = df.loc[mask].copy()
    spread_threshold = selected["core_spread_corr"].quantile(0.70)
    masks = {
        "all": pd.Series(True, index=selected.index),
        "warm_season_may_sep": selected["target_date"].dt.month.isin([5, 6, 7, 8, 9]),
        "cool_season_oct_apr": ~selected["target_date"].dt.month.isin([5, 6, 7, 8, 9]),
        "actual_ge_90f": selected["actual_tmax_f"] >= 90,
        "actual_ge_95f": selected["actual_tmax_f"] >= 95,
        "actual_le_40f": selected["actual_tmax_f"] <= 40,
        "high_spread_top30": selected["core_spread_corr"] >= spread_threshold,
        "low_spread_bottom70": selected["core_spread_corr"] < spread_threshold,
    }
    rows: list[dict[str, object]] = []
    for regime, regime_mask in masks.items():
        for col in ["corr_nbm", "pred_stack_final_bias"]:
            result = evaluate(selected, col, regime_mask)
            rows.append(
                {
                    "regime": regime,
                    "method": col,
                    "n": result["n"],
                    "mae": result["mae"],
                    "rmse": result["rmse"],
                    "bias": result["bias"],
                    "within_1f": result["within_1f"],
                    "within_2f": result["within_2f"],
                }
            )
    return pd.DataFrame(rows)


def concrete_examples(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    selected = df.loc[mask].copy()
    selected["err_nbm"] = selected["corr_nbm"] - selected["actual_tmax_f"]
    selected["err_stack_final"] = selected["pred_stack_final_bias"] - selected["actual_tmax_f"]
    selected["stack_minus_nbm_abs_improvement"] = selected["err_nbm"].abs() - selected["err_stack_final"].abs()
    examples = pd.concat(
        [
            selected.sort_values("stack_minus_nbm_abs_improvement", ascending=False).head(5),
            selected.sort_values("stack_minus_nbm_abs_improvement", ascending=True).head(5),
        ]
    )
    return examples[
        [
            "target_date",
            "actual_tmax_f",
            "raw_nbm",
            "corr_nbm",
            "corr_hrrr",
            "corr_gfs",
            "corr_rap",
            "corr_core_equal",
            "pred_stack_final_bias",
            "err_nbm",
            "err_stack_final",
            "stack_minus_nbm_abs_improvement",
            "beta_hrrr",
            "beta_gfs",
            "beta_rap",
            "stack_alpha",
            "stack_adjustment",
            "final_bias_add",
            "core_spread_corr",
        ]
    ]


def write_artifacts(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for CSV result artifacts.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Acknowledge the read-only database scan and bounded local experiment.",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("database experiment is disabled; re-run with --execute")

    started = time.time()
    raw = load_core_dataset()
    common_raw_mask = raw[[f"raw_{model}" for model in MODELS]].notna().all(axis=1)
    print(
        f"Loaded {len(raw)} label-matched rows; common raw core rows="
        f"{int(common_raw_mask.sum())} "
        f"{raw.loc[common_raw_mask, 'target_date'].min().date()}.."
        f"{raw.loc[common_raw_mask, 'target_date'].max().date()}"
    )

    source_leak_check = {}
    for model in MODELS:
        available = raw[f"raw_{model}"].notna()
        run_violations = available & (
            raw[f"run_{model}"] > raw["cutoff_utc"] - pd.Timedelta(hours=BUFFER_HOURS)
        )
        availability_violations = available & (raw[f"avail_{model}"] > raw["cutoff_utc"])
        source_leak_check[model] = {
            "run_buffer_violations": int(run_violations.sum()),
            "availability_violations": int(availability_violations.sum()),
        }
    print("SOURCE_LEAK_CHECK", json.dumps(source_leak_check, sort_keys=True))

    df = compute_online_bias_corrected(raw)
    df["raw_core_equal"] = df[[f"raw_{model}" for model in MODELS]].mean(axis=1)
    df["corr_core_equal"] = df[[f"corr_{model}" for model in MODELS]].mean(axis=1)
    df["core_spread_corr"] = (
        df[[f"corr_{model}" for model in MODELS]].max(axis=1)
        - df[[f"corr_{model}" for model in MODELS]].min(axis=1)
    )
    corr_common_mask = df[[f"corr_{model}" for model in MODELS]].notna().all(axis=1)
    print(
        f"Corrected common rows before stacker={int(corr_common_mask.sum())} "
        f"{df.loc[corr_common_mask, 'target_date'].min().date()}.."
        f"{df.loc[corr_common_mask, 'target_date'].max().date()}"
    )

    bias_label_check = {}
    for model in MODELS:
        col = f"bias_max_label_date_{model}"
        valid = df[col].notna() & df[f"corr_{model}"].notna()
        violations = valid & (df[col] > df["target_date"] - pd.Timedelta(days=LABEL_LAG_DAYS))
        bias_label_check[model] = int(violations.sum())
    print("BIAS_LABEL_LEAK_CHECK", json.dumps(bias_label_check, sort_keys=True))

    print("Running default residual stacker...")
    stack_default = walk_forward_residual_stack(df)
    stack_cols = [
        "pred_stack",
        "pred_stack_final_bias",
        "stack_alpha",
        "stack_adjustment",
        "stack_train_n",
        "stack_train_start",
        "stack_train_end",
        "stack_success",
        "final_bias_add",
    ] + [f"beta_{model}" for model in DELTA_ORDER]
    for col in stack_cols:
        df[col] = stack_default[col]
    stack_mask = df["pred_stack"].notna()
    train_end = pd.to_datetime(df["stack_train_end"])
    stack_leaks = stack_mask & (train_end > df["target_date"] - pd.Timedelta(days=LABEL_LAG_DAYS))
    print(
        f"Default stacker rows={int(stack_mask.sum())} "
        f"{df.loc[stack_mask, 'target_date'].min().date()}.."
        f"{df.loc[stack_mask, 'target_date'].max().date()} "
        f"stack_train_label_leak_violations={int(stack_leaks.sum())}"
    )

    print("Running rolling convex-grid blend baselines...")
    convex_cols: list[str] = []
    for min_nbm in (0.50, 0.60, 0.70, 0.80):
        convex = add_rolling_convex_grid_blend(df, min_nbm_weight=min_nbm)
        df = pd.concat([df, convex], axis=1)
        convex_cols.extend(list(convex.columns))

    print("Running online performance-weighted baselines...")
    perf = add_online_performance_blends(df)
    df = pd.concat([df, perf], axis=1)
    perf_cols = list(perf.columns)

    print("Running compact stacker variants...")
    grid_specs = [
        ("stack_cap025", {"max_non_nbm_weight": 0.25}),
        ("stack_cap035", {"max_non_nbm_weight": 0.35}),
        ("stack_cap055", {"max_non_nbm_weight": 0.55}),
        ("stack_lam025", {"lambda_beta": 0.25}),
        ("stack_hl090", {"stack_half_life_days": 90.0}),
    ]
    variant_rows: list[dict[str, object]] = []
    for name, kwargs in grid_specs:
        variant = walk_forward_residual_stack(df, **kwargs)
        tmp = df[["target_date", "actual_tmax_f"]].copy()
        tmp[f"{name}_plain"] = variant["pred_stack"]
        tmp[f"{name}_final_bias"] = variant["pred_stack_final_bias"]
        own_mask = tmp[f"{name}_plain"].notna()
        row_plain = evaluate(tmp, f"{name}_plain", own_mask)
        row_bias = evaluate(tmp, f"{name}_final_bias", own_mask)
        variant_rows.append({"variant": name, "final_bias": False, **row_plain})
        variant_rows.append({"variant": name, "final_bias": True, **row_bias})
        print(f"  {name} done")

    baseline_cols = (
        [
            "raw_nbm",
            "raw_core_equal",
            "corr_hrrr",
            "corr_nbm",
            "corr_rap",
            "corr_gfs",
            "corr_core_equal",
            "pred_stack",
            "pred_stack_final_bias",
        ]
        + convex_cols
        + perf_cols
    )
    comparison = pd.DataFrame([evaluate(df, col, stack_mask) for col in baseline_cols]).sort_values(
        ["mae", "rmse"], na_position="last"
    )
    pre_stacker_baselines = pd.DataFrame(
        [
            evaluate(df, col, corr_common_mask)
            for col in [
                "raw_nbm",
                "raw_core_equal",
                "corr_hrrr",
                "corr_nbm",
                "corr_rap",
                "corr_gfs",
                "corr_core_equal",
            ]
        ]
    ).sort_values(["mae", "rmse"], na_position="last")
    variants = pd.DataFrame(variant_rows).sort_values(["mae", "rmse"], na_position="last")
    weights = summarize_weights(df, stack_mask)
    regimes = regime_metrics(df, stack_mask)
    examples = concrete_examples(df, stack_mask)

    print("\n=== STRICT T-2 BASELINES BEFORE STACKER WARMUP ===")
    print(pre_stacker_baselines.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== METHODS ON EXACT DEFAULT STACKER DATES ===")
    print(comparison.head(35).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== COMPACT STACKER VARIANTS ===")
    print(variants.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== DEFAULT STACKER WEIGHTS AND ADJUSTMENTS ===")
    print(weights.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== REGIME METRICS ===")
    print(regimes.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== BIGGEST DEFAULT STACKER HELP/HURT VS CORRECTED NBM ===")
    print(examples.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if args.output_dir:
        write_artifacts(
            Path(args.output_dir),
            {
                "pre_stacker_baselines": pre_stacker_baselines,
                "comparison_default_stacker_dates": comparison,
                "compact_stacker_variants": variants,
                "default_stacker_weights": weights,
                "regime_metrics": regimes,
                "examples": examples,
            },
        )
        print(f"\nWrote artifacts to {args.output_dir}")
    print(f"\nElapsed seconds: {time.time() - started:.1f}")


if __name__ == "__main__":
    main()
