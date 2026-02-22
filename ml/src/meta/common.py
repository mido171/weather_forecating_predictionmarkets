"""Shared utilities for meta stacking pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import json
import math

import numpy as np
import pandas as pd
import yaml

from weather_ml import config as config_module
from weather_ml import dataset as grib_dataset
from weather_ml import models_mean
from weather_ml import splits
from weather_ml import time_feature_library as tfl
from weather_ml import time_feature_sweep as tfs
from weather_ml import kalshi_tmax_train as kalshi_train


EVENT_SPECS_DEFAULT = [
    {"name": "lt_52", "type": "threshold", "lt": 52},
    {"name": "lt_70", "type": "threshold", "lt": 70},
    {"name": "lt_75", "type": "threshold", "lt": 75},
    {"name": "ge_85", "type": "threshold", "ge": 85},
    {"name": "ge_90", "type": "threshold", "ge": 90},
    {"name": "range_80_84", "type": "range", "start": 80, "end": 84},
    {"name": "range_85_89", "type": "range", "start": 85, "end": 89},
]


@dataclass(frozen=True)
class BasePredictions:
    frame: pd.DataFrame
    model_name: str


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_grib_config_path(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "time_feature_sweep_trees" / "20260129T200039Z" / "xgb" / "EX210" / "config_resolved.yaml"


def default_mos_features_path(repo_root: Path) -> Path:
    base = repo_root / "dataset" / "kmia_kalshi_tmax" / "kmia_kalshi_tmax_v1_full"
    if not base.exists():
        raise FileNotFoundError(f"KMIA Kalshi Tmax dataset not found: {base}")
    candidates = list(base.glob("*/features.csv"))
    if not candidates:
        raise FileNotFoundError(f"No features.csv found under {base}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def default_mos_train_config_path(repo_root: Path) -> Path:
    return repo_root / "ml" / "configs" / "kalshi_kmia_tmax_train_full_2007_2022.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _min_periods(window: int) -> int:
    return max(5, int(math.ceil(window * 0.7)))


def _quantile_key(q: float) -> str:
    return f"q{int(q * 100):02d}"


def _normal_quantiles(mu: np.ndarray, sigma: np.ndarray, qs: list[float]) -> dict[str, np.ndarray]:
    from scipy.stats import norm

    preds: dict[str, np.ndarray] = {}
    sigma = np.maximum(sigma, 1e-6)
    for q in qs:
        preds[_quantile_key(q)] = mu + sigma * norm.ppf(q)
    return preds


def _enforce_monotonic_quantiles(preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not preds:
        return preds
    keys = sorted(preds.keys())
    stack = np.vstack([preds[k] for k in keys])
    stack_sorted = np.sort(stack, axis=0)
    for idx, key in enumerate(keys):
        preds[key] = stack_sorted[idx, :]
    return preds


def load_gribstream_df(csv_path: Path) -> pd.DataFrame:
    df = grib_dataset.load_csv(csv_path)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
    return df


def load_mos_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
    return df


def build_gribstream_features(
    df: pd.DataFrame,
    *,
    train_index: pd.Index,
    config,
    truth_lag: int,
) -> tuple[pd.DataFrame, list[str], dict]:
    tfs._apply_model_cols(config)
    df = tfl.prepare_frame(df)
    train_df = df.loc[train_index]
    df = tfs._impute_base_columns(df, train_df)
    df = tfs._add_base_columns(df)
    train_df = df.loc[train_index]

    context = tfs.ExperimentContext(
        df=df,
        train_df=train_df,
        val_df=pd.DataFrame(),
        test_df=pd.DataFrame(),
        group_key=df["station_id"],
        truth_lag=int(truth_lag),
        seed=int(config.seeds.global_seed),
    )
    derived = tfs._exp_ex210(context)
    base_cols = tfs._base_feature_columns(df)
    tfs._ensure_columns_exist(df, base_cols, "gribstream_base")
    base_features = df[base_cols].astype(float).copy()
    feature_df = pd.concat([base_features, derived.features], axis=1)
    feature_cols = base_cols + list(derived.features.columns)
    filled, impute_meta = tfs._impute_features(feature_df, train_df.index)
    return filled, feature_cols, impute_meta


def train_gribstream_model(
    df: pd.DataFrame,
    *,
    train_index: pd.Index,
    config,
    truth_lag: int,
) -> tuple[Any, list[str], dict, float, pd.DataFrame]:
    features, feature_cols, impute_meta = build_gribstream_features(
        df, train_index=train_index, config=config, truth_lag=truth_lag
    )
    X_train = features.loc[train_index].to_numpy(dtype=float)
    y_train = df.loc[train_index, "actual_tmax_f"].to_numpy(dtype=float)
    model = models_mean.get_mean_model(config.models.mean.primary, seed=config.seeds.global_seed)
    params = tfs._fixed_params(config.models.mean.param_grid.get(config.models.mean.primary, {}))
    if params:
        model.set_params(**params)
    model.fit(X_train, y_train)
    resid = y_train - model.predict(X_train)
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 1.0
    return model, feature_cols, impute_meta, sigma, features


def predict_gribstream(
    df: pd.DataFrame,
    *,
    config,
    train_index: pd.Index,
    predict_index: pd.Index,
    truth_lag: int,
    quantiles: list[float],
) -> BasePredictions:
    model, feature_cols, impute_meta, sigma, features = train_gribstream_model(
        df, train_index=train_index, config=config, truth_lag=truth_lag
    )
    X_pred = features.loc[predict_index, feature_cols].to_numpy(dtype=float)
    mu = model.predict(X_pred)
    sigma_vec = np.full_like(mu, sigma, dtype=float)
    q_preds = _normal_quantiles(mu, sigma_vec, quantiles)
    q_preds = _enforce_monotonic_quantiles(q_preds)
    out = pd.DataFrame(
        {
            "target_date_local": df.loc[predict_index, "target_date_local"].to_numpy(),
            "asof_utc": df.loc[predict_index, "asof_utc"].to_numpy(),
            "station_id": df.loc[predict_index, "station_id"].to_numpy(),
            "y_true_f": df.loc[predict_index, "actual_tmax_f"].to_numpy(dtype=float),
            "mu_f": mu,
            "sigma_f": sigma_vec,
        }
    )
    for key, values in q_preds.items():
        out[key] = values
    return BasePredictions(frame=out, model_name="gribstream")


def train_mos_predict(
    df: pd.DataFrame,
    *,
    train_index: pd.Index,
    predict_index: pd.Index,
    quantiles: list[float],
    lgbm_params: dict[str, Any],
    quantile_params: dict[str, Any],
    recency_lambda: float,
) -> BasePredictions:
    feature_cols = kalshi_train._select_feature_columns(df)
    baseline, _ = kalshi_train._baseline_series(df)
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["y_actual_tmax_f"].to_numpy(dtype=float)
    baseline_np = baseline.to_numpy(dtype=float)

    train_mask = df.index.isin(train_index)
    X_train = X[train_mask]
    y_train = y[train_mask]
    baseline_train = baseline_np[train_mask]
    valid_train = np.isfinite(y_train) & np.isfinite(baseline_train)
    X_train = X_train[valid_train]
    y_train = y_train[valid_train]
    baseline_train = baseline_train[valid_train]

    weights = None
    if recency_lambda > 0:
        train_dates = pd.to_datetime(df.loc[train_mask, "target_date_local"])[valid_train]
        ages = (train_dates - train_dates.min()).dt.days
        weights = np.exp(-recency_lambda * ages.to_numpy(dtype=float))

    model = kalshi_train._fit_lgbm_residual_model(
        X_train,
        y_train,
        baseline_train,
        params=lgbm_params,
        sample_weight=weights,
    )

    pred_mask = df.index.isin(predict_index)
    X_pred = X[pred_mask]
    baseline_pred = baseline_np[pred_mask]
    mu = baseline_pred + model.predict(X_pred)
    mu = np.where(np.isfinite(baseline_pred), mu, np.nan)

    q_preds = {}
    if quantiles:
        split_preds = kalshi_train._fit_quantile_models_multi(
            X_train=X_train,
            y_train=y_train,
            baseline_train=baseline_train,
            eval_splits={"pred": (X_pred, baseline_pred)},
            params=quantile_params,
            quantiles=quantiles,
            sample_weight=weights,
        )
        q_preds = split_preds.get("pred", {})
        q_preds = _enforce_monotonic_quantiles(q_preds)

    sigma = None
    if _quantile_key(quantiles[0]) in q_preds and _quantile_key(quantiles[-1]) in q_preds:
        q10 = q_preds[_quantile_key(quantiles[0])]
        q90 = q_preds[_quantile_key(quantiles[-1])]
        sigma = (q90 - q10) / 2.563
    if sigma is None:
        resid = y_train - (baseline_train + model.predict(X_train))
        sigma = np.full_like(mu, float(np.std(resid, ddof=1)) if len(resid) > 1 else 1.0, dtype=float)
    else:
        sigma = np.maximum(sigma, 1e-6)

    out = pd.DataFrame(
        {
            "target_date_local": df.loc[predict_index, "target_date_local"].to_numpy(),
            "asof_utc": df.loc[predict_index, "asof_utc"].to_numpy(),
            "station_id": df.loc[predict_index, "station_id"].to_numpy(),
            "y_true_f": df.loc[predict_index, "y_actual_tmax_f"].to_numpy(dtype=float),
            "mu_f": mu,
            "sigma_f": sigma,
        }
    )
    for key, values in q_preds.items():
        out[key] = values
    if _quantile_key(0.5) not in out.columns and q_preds:
        mid = _quantile_key(0.5)
        if mid in q_preds:
            out[mid] = q_preds[mid]
    return BasePredictions(frame=out, model_name="mos")


def make_oof_folds(
    df: pd.DataFrame,
    *,
    start: date,
    end: date,
    n_splits: int,
    gap_days: int,
) -> list[tuple[set[date], set[date]]]:
    mask = (df["target_date_local"] >= start) & (df["target_date_local"] <= end)
    df_meta = df.loc[mask].copy()
    splits_idx = splits.make_time_cv_splits(
        df_meta, n_splits=n_splits, gap_days=gap_days
    )
    folds: list[tuple[set[date], set[date]]] = []
    for train_idx, val_idx in splits_idx:
        train_dates = set(df_meta.loc[train_idx, "target_date_local"].to_list())
        val_dates = set(df_meta.loc[val_idx, "target_date_local"].to_list())
        folds.append((train_dates, val_dates))
    return folds


def merge_base_predictions(
    gs: pd.DataFrame, mos: pd.DataFrame
) -> pd.DataFrame:
    gs = gs.copy()
    mos = mos.copy()
    for frame in (gs, mos):
        frame["target_date_local"] = pd.to_datetime(frame["target_date_local"]).dt.date
        frame["asof_utc"] = pd.to_datetime(frame["asof_utc"], utc=True)
    merged = pd.merge(
        gs,
        mos,
        on=["target_date_local", "asof_utc"],
        how="inner",
        suffixes=("_gs", "_mos"),
    )
    return merged


def add_meta_features(
    merged: pd.DataFrame,
    *,
    windows: list[int],
    lag_days: int,
) -> pd.DataFrame:
    df = merged.copy()
    df = df.sort_values("target_date_local").reset_index(drop=True)

    df["mu_gs"] = df["mu_f_gs"]
    df["mu_mos"] = df["mu_f_mos"]
    df["q10_gs"] = df[_quantile_key(0.1) + "_gs"]
    df["q50_gs"] = df[_quantile_key(0.5) + "_gs"]
    df["q90_gs"] = df[_quantile_key(0.9) + "_gs"]
    df["q10_mos"] = df[_quantile_key(0.1) + "_mos"]
    df["q50_mos"] = df[_quantile_key(0.5) + "_mos"]
    df["q90_mos"] = df[_quantile_key(0.9) + "_mos"]

    df["sigma_gs"] = df["sigma_f_gs"]
    df["sigma_mos"] = df["sigma_f_mos"]
    df["width80_gs"] = df["q90_gs"] - df["q10_gs"]
    df["width80_mos"] = df["q90_mos"] - df["q10_mos"]

    df["d_mu"] = df["mu_gs"] - df["mu_mos"]
    df["abs_d_mu"] = df["d_mu"].abs()
    df["d_sigma"] = df["sigma_gs"] - df["sigma_mos"]
    df["abs_d_sigma"] = df["d_sigma"].abs()

    y = df["y_true_f_gs"].to_numpy(dtype=float)
    df["y_true_f"] = y

    for base in ("gs", "mos"):
        err = df["y_true_f"] - df[f"mu_{base}"]
        err_shift = err.shift(lag_days)
        for window in windows:
            min_p = _min_periods(window)
            df[f"bias_{base}_{window}"] = err_shift.rolling(window, min_periods=min_p).mean()
            df[f"mae_{base}_{window}"] = err_shift.abs().rolling(window, min_periods=min_p).mean()

    dates = pd.to_datetime(df["target_date_local"])
    df["month"] = dates.dt.month.astype(int)
    radians = 2 * np.pi * dates.dt.dayofyear.to_numpy(dtype=float) / 365.25
    df["doy_sin"] = np.sin(radians)
    df["doy_cos"] = np.cos(radians)
    return df


def meta_feature_columns(windows: list[int]) -> list[str]:
    cols = [
        "mu_gs",
        "mu_mos",
        "q10_gs",
        "q50_gs",
        "q90_gs",
        "q10_mos",
        "q50_mos",
        "q90_mos",
        "sigma_gs",
        "sigma_mos",
        "width80_gs",
        "width80_mos",
        "d_mu",
        "abs_d_mu",
        "d_sigma",
        "abs_d_sigma",
        "month",
        "doy_sin",
        "doy_cos",
    ]
    for window in windows:
        cols.append(f"bias_gs_{window}")
        cols.append(f"mae_gs_{window}")
        cols.append(f"bias_mos_{window}")
        cols.append(f"mae_mos_{window}")
    return cols


def impute_medians(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    values = df[cols].to_numpy(dtype=float)
    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isnan(values), medians, values)
    meta = {col: float(med) for col, med in zip(cols, medians)}
    return filled, meta


def event_reliability(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(probs, edges, right=True) - 1
    idx = np.clip(idx, 0, bins - 1)
    rows = []
    ece = 0.0
    mce = 0.0
    total = len(probs)
    for b in range(bins):
        mask = idx == b
        if not np.any(mask):
            rows.append({"bin": b, "count": 0, "avg_pred": None, "emp_rate": None})
            continue
        avg_pred = float(np.mean(probs[mask]))
        emp_rate = float(np.mean(y_true[mask]))
        rows.append({"bin": b, "count": int(mask.sum()), "avg_pred": avg_pred, "emp_rate": emp_rate})
        gap = abs(avg_pred - emp_rate)
        ece += (mask.sum() / total) * gap
        mce = max(mce, gap)
    return {"bins": rows, "ece": float(ece), "mce": float(mce)}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
