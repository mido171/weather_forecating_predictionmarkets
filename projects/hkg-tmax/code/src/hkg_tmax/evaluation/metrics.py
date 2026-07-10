"""Metric and slicing helpers for residual-ML scoreboards."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def score_arrays(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(actual) & np.isfinite(prediction)
    actual = actual[mask]
    prediction = prediction[mask]
    if len(actual) == 0:
        return {
            "n_scored": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "median_absolute_error": float("nan"),
            "bias": float("nan"),
            "p80_absolute_error": float("nan"),
            "p90_absolute_error": float("nan"),
            "p95_absolute_error": float("nan"),
            "max_absolute_error": float("nan"),
            "mean_prediction": float("nan"),
            "mean_actual": float("nan"),
            "mean_anchor_forecast": float("nan"),
        }
    err = prediction - actual
    abs_err = np.abs(err)
    return {
        "n_scored": int(len(actual)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "median_absolute_error": float(np.median(abs_err)),
        "bias": float(np.mean(err)),
        "p80_absolute_error": float(np.quantile(abs_err, 0.80)),
        "p90_absolute_error": float(np.quantile(abs_err, 0.90)),
        "p95_absolute_error": float(np.quantile(abs_err, 0.95)),
        "max_absolute_error": float(np.max(abs_err)),
        "mean_prediction": float(np.mean(prediction)),
        "mean_actual": float(np.mean(actual)),
        "mean_anchor_forecast": float("nan"),
    }


def score_frame(
    predictions: pd.DataFrame,
    by: list[str],
    *,
    scope: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()
    grouped = predictions.groupby(by, dropna=False) if by else [((), predictions)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        metrics = score_arrays(
            pd.to_numeric(group["y_true_c"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(group["prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        if "anchor_forecast_max_c" in group:
            metrics["mean_anchor_forecast"] = float(pd.to_numeric(group["anchor_forecast_max_c"], errors="coerce").mean())
        record = {column: value for column, value in zip(by, key, strict=False)}
        record.update(metrics)
        record["scope"] = scope
        rows.append(record)
    return pd.DataFrame(rows)


def season_from_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "DJF"
    if month in {3, 4, 5}:
        return "MAM"
    if month in {6, 7, 8}:
        return "JJA"
    return "SON"


def add_score_slices(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    date = pd.to_datetime(out["target_date"], errors="coerce")
    out["score_month"] = date.dt.month
    out["score_quarter"] = date.dt.quarter
    out["score_season"] = out["score_month"].map(season_from_month)
    out["warm_season"] = out["score_month"].isin([4, 5, 6, 7, 8, 9, 10]).map({True: "warm", False: "not_warm"})
    out["hot_season"] = out["score_month"].isin([6, 7, 8, 9]).map({True: "hot", False: "not_hot"})
    out["typhoon_season"] = out["score_month"].isin([6, 7, 8, 9, 10]).map({True: "typhoon_season", False: "not_typhoon_season"})
    if "network_latest_temp_spread_c" in out:
        spread = pd.to_numeric(out["network_latest_temp_spread_c"], errors="coerce")
        threshold = spread.quantile(0.80)
        out["network_spread_regime"] = np.where(spread >= threshold, "network_spread_high", "network_spread_normal")
    if "inland_nt_mean_minus_coastal_marine_mean_c" in out:
        contrast = pd.to_numeric(out["inland_nt_mean_minus_coastal_marine_mean_c"], errors="coerce")
        threshold = contrast.quantile(0.80)
        out["inland_minus_coastal_regime"] = np.where(
            contrast >= threshold, "inland_minus_coastal_high", "inland_minus_coastal_normal"
        )
    for column in ("official_max_bin", "official_range_bin", "issue_hour_bucket"):
        if column not in out:
            out[column] = "missing"
    out["rain_thunderstorm_regime"] = np.where(
        out.get("fcst_flag_thunderstorm", 0).fillna(0).astype(int).eq(1)
        | out.get("hourly_any_thunderstorm_warning_24h", 0).fillna(0).astype(int).eq(1)
        | out.get("hourly_any_rainstorm_warning_24h", 0).fillna(0).astype(int).eq(1),
        "rain_thunderstorm",
        "normal_or_unknown",
    )
    return out

