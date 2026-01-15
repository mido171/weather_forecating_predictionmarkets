"""Global residual calibration helpers."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from weather_ml import artifacts


def resolve_forecast_columns(
    forecast_columns: Iterable[str], base_features: Iterable[str]
) -> list[str]:
    if forecast_columns:
        resolved = [str(col) for col in forecast_columns]
    else:
        resolved = [
            str(col)
            for col in base_features
            if "spread" not in str(col).lower()
        ]
    if not resolved:
        raise ValueError("No forecast columns available for baseline median.")
    return resolved


def select_calibration_rows(
    df: pd.DataFrame,
    *,
    cal_start: date,
    cal_end: date,
    station_scope: list[str] | None,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    df = df.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.normalize()
    cal_start_ts = pd.Timestamp(cal_start)
    cal_end_ts = pd.Timestamp(cal_end)

    if cal_end_ts < cal_start_ts:
        raise ValueError("Calibration window end is before start.")

    mask = (df["target_date_local"] >= cal_start_ts) & (
        df["target_date_local"] <= cal_end_ts
    )
    if station_scope:
        mask &= df["station_id"].isin(station_scope)
    cal_df = df.loc[mask].copy()

    missing_cols = [col for col in required_columns if col not in cal_df.columns]
    if missing_cols:
        raise ValueError(f"Calibration rows missing required columns: {missing_cols}")

    if cal_df.empty:
        raise ValueError("Calibration selection is empty.")

    duplicates = cal_df.duplicated(
        subset=["station_id", "target_date_local", "asof_utc"], keep=False
    )
    if duplicates.any():
        sample = cal_df.loc[duplicates, ["station_id", "target_date_local", "asof_utc"]]
        raise ValueError(
            "Duplicate calibration rows found: "
            f"{sample.head(5).to_dict(orient='records')}"
        )

    if cal_df["actual_tmax_f"].isna().any():
        raise ValueError("Calibration rows contain missing actual_tmax_f values.")

    return cal_df


def compute_median_forecast(
    df: pd.DataFrame, forecast_columns: Iterable[str]
) -> np.ndarray:
    values = df[list(forecast_columns)].to_numpy(dtype=float)
    return np.nanmedian(values, axis=1)


def compute_residual_stats(residuals: np.ndarray, *, ddof: int) -> dict:
    residuals = np.asarray(residuals, dtype=float)
    n = int(residuals.size)
    if n <= ddof:
        raise ValueError(
            f"Not enough calibration rows to compute std with ddof={ddof}."
        )
    bias = float(np.mean(residuals))
    sigma = float(np.std(residuals, ddof=ddof))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    quantiles = np.quantile(
        residuals, [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    )
    return {
        "n": n,
        "bias_mean_error_f": bias,
        "sigma_std_error_f": sigma,
        "mae_f": mae,
        "rmse_f": rmse,
        "residual_quantiles_f": {
            "p01": float(quantiles[0]),
            "p05": float(quantiles[1]),
            "p10": float(quantiles[2]),
            "p50": float(quantiles[3]),
            "p90": float(quantiles[4]),
            "p95": float(quantiles[5]),
            "p99": float(quantiles[6]),
        },
    }


def build_calibration_payload(
    *,
    residual_stats: dict,
    ddof: int,
    cal_start: date,
    cal_end: date,
    station_scope: list[str] | None,
    run_dir: Path,
    mean_model_path: Path,
    model_hash: str,
    dataset_id: str,
    dataset_hash: str,
    rows_used: int,
) -> dict:
    scope_mode = "ALL" if not station_scope else "LIST"
    return {
        "method": "global_normal_residual",
        "error_definition": "e = actual_tmax_f - mu_hat",
        "ddof": ddof,
        "n": residual_stats["n"],
        "bias_mean_error_f": residual_stats["bias_mean_error_f"],
        "sigma_std_error_f": residual_stats["sigma_std_error_f"],
        "mae_f": residual_stats["mae_f"],
        "rmse_f": residual_stats["rmse_f"],
        "residual_quantiles_f": residual_stats["residual_quantiles_f"],
        "calibration_window": {
            "start": cal_start.isoformat(),
            "end": cal_end.isoformat(),
        },
        "station_scope": {
            "mode": scope_mode,
            "stations": station_scope or [],
        },
        "model_ref": {
            "run_dir": str(run_dir),
            "mean_model_artifact": mean_model_path.name,
            "model_hash": model_hash,
        },
        "dataset_ref": {
            "dataset_id": dataset_id,
            "dataset_hash": dataset_hash,
            "rows_used": rows_used,
        },
        "created_utc": artifacts.utc_now_iso(),
    }


def build_baseline_payload(
    *,
    residual_stats: dict,
    ddof: int,
    cal_start: date,
    cal_end: date,
    station_scope: list[str] | None,
    forecast_columns: list[str],
    dataset_id: str,
    dataset_hash: str,
    rows_used: int,
) -> dict:
    scope_mode = "ALL" if not station_scope else "LIST"
    return {
        "method": "baseline_median_residual",
        "error_definition": "e = actual_tmax_f - median_forecast_f",
        "ddof": ddof,
        "n": residual_stats["n"],
        "bias_mean_error_f": residual_stats["bias_mean_error_f"],
        "sigma_std_error_f": residual_stats["sigma_std_error_f"],
        "mae_f": residual_stats["mae_f"],
        "rmse_f": residual_stats["rmse_f"],
        "residual_quantiles_f": residual_stats["residual_quantiles_f"],
        "calibration_window": {
            "start": cal_start.isoformat(),
            "end": cal_end.isoformat(),
        },
        "station_scope": {
            "mode": scope_mode,
            "stations": station_scope or [],
        },
        "forecast_columns": list(forecast_columns),
        "dataset_ref": {
            "dataset_id": dataset_id,
            "dataset_hash": dataset_hash,
            "rows_used": rows_used,
        },
        "created_utc": artifacts.utc_now_iso(),
    }


def write_residuals_csv(
    path: Path,
    *,
    cal_df: pd.DataFrame,
    mu_hat: np.ndarray,
    residuals: np.ndarray,
    include_abs: bool = True,
) -> None:
    output = cal_df[["station_id", "target_date_local", "asof_utc"]].copy()
    output["mu_hat_f"] = mu_hat
    output["actual_tmax_f"] = cal_df["actual_tmax_f"].to_numpy()
    output["residual_f"] = residuals
    if include_abs:
        output["abs_residual_f"] = np.abs(residuals)
    output.to_csv(path, index=False)


def write_baseline_residuals_csv(
    path: Path,
    *,
    cal_df: pd.DataFrame,
    median_forecast: np.ndarray,
    residuals: np.ndarray,
    include_abs: bool = True,
) -> None:
    output = cal_df[["station_id", "target_date_local", "asof_utc"]].copy()
    output["median_forecast_f"] = median_forecast
    output["actual_tmax_f"] = cal_df["actual_tmax_f"].to_numpy()
    output["residual_f"] = residuals
    if include_abs:
        output["abs_residual_f"] = np.abs(residuals)
    output.to_csv(path, index=False)


def check_calibration_window(
    *,
    cal_start: date | None,
    cal_end: date | None,
    train_end: date,
    val_end: date | None,
) -> None:
    if cal_start is None or cal_end is None:
        raise ValueError("Calibration window start/end must be set.")
    max_train_end = train_end
    if val_end and val_end > max_train_end:
        max_train_end = val_end
    if cal_start <= max_train_end:
        raise ValueError(
            "Calibration window must start after training/validation end. "
            f"cal_start={cal_start} max_train_end={max_train_end}"
        )
