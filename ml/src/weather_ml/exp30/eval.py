"""Evaluation helpers for exp30."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {}
    error = y_pred - y_true
    abs_err = np.abs(error)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "medianAE": float(np.median(abs_err)),
        "maxAE": float(np.max(abs_err)),
        "corr": corr,
        "n": int(len(y_true)),
    }


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def slice_mae_by_month(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    months = pd.to_datetime(df["target_date_local"]).dt.month
    results = {}
    for month in range(1, 13):
        mask = months == month
        if mask.any():
            results[str(month)] = float(np.mean(np.abs(y_pred[mask] - y_true[mask])))
    return results


def slice_mae_by_season(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    months = pd.to_datetime(df["target_date_local"]).dt.month
    results = {}
    for month in range(1, 13):
        season = _season_from_month(month)
        mask = months == month
        if mask.any():
            mae = float(np.mean(np.abs(y_pred[mask] - y_true[mask])))
            results.setdefault(season, []).append(mae)
    return {season: float(np.mean(vals)) for season, vals in results.items()}


def slice_mae_by_decile(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    preds = pd.Series(y_pred)
    deciles = pd.qcut(preds, 10, labels=False, duplicates="drop")
    results = {}
    for decile in sorted(deciles.dropna().unique()):
        mask = deciles == decile
        results[str(int(decile))] = float(np.mean(np.abs(y_pred[mask] - y_true[mask])))
    return results


def worth_testing(
    baseline_monthly: dict,
    baseline_seasonal: dict,
    exp_monthly: dict,
    exp_seasonal: dict,
    delta_mae: float,
    threshold: float = 0.03,
) -> dict:
    if delta_mae >= -threshold:
        return {
            "worth_testing": False,
            "reason": "delta_mae_below_threshold",
        }
    # Monthly improvement count
    improved_months = 0
    for month, base_val in baseline_monthly.items():
        exp_val = exp_monthly.get(month)
        if exp_val is None:
            continue
        if exp_val <= base_val:
            improved_months += 1
    improved_seasons = 0
    for season, base_val in baseline_seasonal.items():
        exp_val = exp_seasonal.get(season)
        if exp_val is None:
            continue
        if exp_val <= base_val:
            improved_seasons += 1
    month_ok = improved_months >= 8
    season_ok = improved_seasons >= 3
    return {
        "worth_testing": bool(month_ok or season_ok),
        "improved_months": improved_months,
        "improved_seasons": improved_seasons,
        "month_ok": month_ok,
        "season_ok": season_ok,
    }


def apply_deltas(baseline_metrics: dict, exp_metrics: dict) -> dict:
    deltas = {}
    for key in ["mae", "rmse", "bias", "medianAE", "maxAE", "corr"]:
        if key in baseline_metrics and key in exp_metrics:
            deltas[f"delta_{key}"] = exp_metrics[key] - baseline_metrics[key]
    return deltas
