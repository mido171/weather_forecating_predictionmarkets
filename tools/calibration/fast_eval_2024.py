#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from zoneinfo import ZoneInfo

# Ensure repo root is on sys.path for local imports.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml import run_mos_45_suite as base
from ml_live.python.calibration.emos_w45 import calibrate
from tools.live.run_kmia_live_v5plus8 import (
    _compute_daily_features,
    _compute_early_features,
    _half_life_alpha,
    _ewma_half_life,
    _fetch_iem_minute,
    build_mos_features,
)


UTC = timezone.utc


@dataclass(frozen=True)
class Config:
    station_id: str
    eval_start: date
    eval_end: date
    windows: list[int]
    truth_lag_days: int
    sigma_floor: float
    minute_dir: Path
    winner_dir: Path
    out_dir: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fast 2024 leakage-free calibration eval.")
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--eval-start", default="2024-01-01")
    parser.add_argument("--eval-end", default="2024-12-31")
    parser.add_argument("--windows", default="45,90,180")
    parser.add_argument("--truth-lag", type=int, default=2)
    parser.add_argument("--sigma-floor", type=float, default=0.5)
    parser.add_argument("--minute-dir", default="data/iem_minute_data/MIA/tmpf/UTC/yearly")
    parser.add_argument(
        "--winner-dir",
        default="artifacts/experiments/KMIA/classic/winners/V5_PLUS8_20260219T222321Z",
    )
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    eval_start = datetime.strptime(args.eval_start, "%Y-%m-%d").date()
    eval_end = datetime.strptime(args.eval_end, "%Y-%m-%d").date()
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    if not windows:
        raise ValueError("No windows provided.")
    out_dir = Path(args.out_dir) if args.out_dir else Path("artifacts/live_v5plus8/fast_eval_2024")
    out_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        station_id=args.station.upper(),
        eval_start=eval_start,
        eval_end=eval_end,
        windows=windows,
        truth_lag_days=args.truth_lag,
        sigma_floor=args.sigma_floor,
        minute_dir=_resolve_repo_path(args.minute_dir),
        winner_dir=_resolve_repo_path(args.winner_dir),
        out_dir=out_dir,
    )


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def _date_range(start_date: date, end_date: date) -> list[date]:
    if end_date < start_date:
        return []
    days = (end_date - start_date).days
    return [start_date + timedelta(days=i) for i in range(days + 1)]


def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0 or not np.isfinite(sigma):
        return float(x >= mu)
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _crps_normal(mu: float, sigma: float, y: float) -> float:
    if sigma <= 0 or not np.isfinite(sigma):
        return abs(mu - y)
    z = (y - mu) / sigma
    phi = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
    Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))


def _crps_empirical(samples: np.ndarray, y: float) -> float:
    if samples.size == 0:
        return float("nan")
    x = np.sort(samples)
    n = x.size
    term1 = float(np.mean(np.abs(x - y)))
    i = np.arange(1, n + 1)
    sum_abs = 2.0 * np.sum((2 * i - n - 1) * x)
    term2 = sum_abs / (2.0 * n * n)
    return float(term1 - term2)


def _empirical_cdf(x: float, samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("nan")
    xs = np.sort(samples)
    n = xs.size
    return float(np.interp(x, xs, np.linspace(1.0 / n, 1.0, n), left=0.0, right=1.0))


def _load_feature_store(winner_dir: Path) -> pd.DataFrame:
    manifest_path = winner_dir / "config_snapshot" / "manifest.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Winner manifest not found: {manifest_path}")
    feature_store_value = ""
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("feature_store="):
            feature_store_value = line.split("=", 1)[1].strip()
            break
    if not feature_store_value:
        raise ValueError(f"feature_store not found in manifest: {manifest_path}")
    feature_store_path = _resolve_repo_path(feature_store_value)
    if not feature_store_path.exists():
        raise FileNotFoundError(f"Feature store parquet not found: {feature_store_path}")
    df = pd.read_parquet(feature_store_path)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    return df


def _load_calibration_taus(winner_dir: Path) -> tuple[float, float]:
    calib_path = winner_dir / "calibration_report.json"
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration report not found: {calib_path}")
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    tau_vals = calib.get("cqr_hybrid_details", {}).get("tau_alpha_val", {})
    tau_05 = float(tau_vals.get("tau_0.05", 0.0))
    tau_10 = float(tau_vals.get("tau_0.10", 0.0))
    return tau_05, tau_10


def _train_models(
    feature_store: pd.DataFrame,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], np.ndarray, float]:
    gate_features = ["feat_u", "feat_v", "feat_wsp_mean", "cal_d_doy_sin", "cal_d_doy_cos"]
    expert_features = [
        "feat_dd_models",
        "feat_tmp_range_mean_models",
        "feat_p12_max",
        "feat_q12_max",
        "feat_cig_min",
        "feat_u",
        "feat_v",
        "cal_d_doy_sin",
        "cal_d_doy_cos",
        "iem_tmax_t1",
        "iem_tmin_t1",
        "iem_range_t1",
        "tmax_time_min_t1",
        "plateau_05_t1",
        "heat_12_15_t1",
        "heat_15_18_t1",
        "cool_18_21_t1",
        "max_drop_30_t1",
        "drop_cnt_15_19_t1",
        "T00",
        "T03",
        "T06",
        "night_drop_00_06",
        "slope_last180",
        "std_last180",
        "T06_adj",
        "diff_lag1",
        "diff_ewma_30",
        "diff_std_30",
        "MRI_suppress",
        "MRI_late",
    ]

    split = base.split_by_date(
        feature_store,
        train_start="2002-01-22",
        train_end="2019-12-31",
        val_start="2020-01-01",
        val_end="2022-12-31",
        test_start="2023-01-01",
        test_end="2025-12-31",
    )
    train_mask = split["train_mask"]

    gate_df = base.ensure_columns(feature_store, gate_features)
    gate_X, gate_meta = base.impute_features(gate_df[gate_features], train_mask)

    exp_df = base.ensure_columns(feature_store, expert_features)
    exp_X, exp_meta = base.impute_features(exp_df[expert_features], train_mask)

    y = pd.to_numeric(feature_store["y_actual_tmax_f"], errors="coerce").to_numpy(dtype=float)
    base_col = "feat_le_median_biascorr"
    base_vals = pd.to_numeric(feature_store.get(base_col), errors="coerce").to_numpy(dtype=float)
    base_mean = float(np.nanmean(y[train_mask]))
    base_vals = np.where(np.isnan(base_vals), base_mean, base_vals)

    gate_label = (pd.to_numeric(feature_store.get("feat_onshore"), errors="coerce") > 0.5).astype(int).to_numpy(dtype=int)
    gate_model = base.train_lgbm_classifier(
        gate_X.to_numpy(dtype=float)[train_mask],
        gate_label[train_mask],
        gate_X.to_numpy(dtype=float)[split["val_mask"]],
        gate_label[split["val_mask"]],
        seed=42,
    )

    def fit(mask: np.ndarray, alpha: float) -> Any:
        train_idx = mask & train_mask
        val_idx = mask & split["val_mask"]
        return base.train_lgbm_quantile(
            exp_X.to_numpy(dtype=float)[train_idx],
            y[train_idx] - base_vals[train_idx],
            exp_X.to_numpy(dtype=float)[val_idx],
            y[val_idx] - base_vals[val_idx],
            seed=42,
            alpha=alpha,
        )

    models = {
        "on_10": fit(gate_label == 1, 0.1),
        "on_50": fit(gate_label == 1, 0.5),
        "on_90": fit(gate_label == 1, 0.9),
        "off_10": fit(gate_label == 0, 0.1),
        "off_50": fit(gate_label == 0, 0.5),
        "off_90": fit(gate_label == 0, 0.9),
    }

    gate_np = gate_X.to_numpy(dtype=float)
    exp_np = exp_X.to_numpy(dtype=float)
    p_gate = gate_model.predict_proba(gate_np)[:, 1]

    def resid_pred(model_on: Any, model_off: Any) -> np.ndarray:
        r_on = model_on.predict(exp_np)
        r_off = model_off.predict(exp_np)
        return p_gate * r_on + (1 - p_gate) * r_off

    r10 = resid_pred(models["on_10"], models["off_10"])
    r50 = resid_pred(models["on_50"], models["off_50"])
    r90 = resid_pred(models["on_90"], models["off_90"])
    spread = r90 - r10

    val_mask = split["val_mask"]
    k_grid = [0.0, 0.2, 0.4, 0.6, 0.8]
    best_k = 0.0
    best_mae = float("inf")
    for k in k_grid:
        w = np.exp(-k * spread)
        pred = base_vals + w * r50
        mae = float(np.nanmean(np.abs(y[val_mask] - pred[val_mask])))
        if mae < best_mae:
            best_mae = mae
            best_k = k

    return gate_model, models, gate_meta["fill_values"], exp_meta["fill_values"], train_mask, float(best_k)


def _load_truth(engine_url: str, station_id: str, start: date, end: date) -> dict[date, float]:
    engine = create_engine(engine_url, pool_pre_ping=True)
    df = pd.read_sql(
        text(
            """
        SELECT date_local, tmax_f
        FROM station_daily_truth
        WHERE station_id = :station_id
          AND date_local BETWEEN :start_date AND :end_date
        """
        ),
        engine,
        params={"station_id": station_id, "start_date": start.isoformat(), "end_date": end.isoformat()},
    )
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.date
    df["tmax_f"] = pd.to_numeric(df["tmax_f"], errors="coerce")
    return {row["date_local"]: float(row["tmax_f"]) for _, row in df.iterrows() if pd.notna(row["tmax_f"])}


def _truth_from_feature_store(feature_store: pd.DataFrame, start: date, end: date) -> dict[date, float]:
    df = feature_store.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df = df[(df["target_date_local"] >= start) & (df["target_date_local"] <= end)]
    df["y_actual_tmax_f"] = pd.to_numeric(df["y_actual_tmax_f"], errors="coerce")
    df = df[df["y_actual_tmax_f"].notna()]
    return {
        row["target_date_local"]: float(row["y_actual_tmax_f"])
        for _, row in df.iterrows()
        if pd.notna(row["y_actual_tmax_f"])
    }


def _load_mos_range(engine_url: str, station_id: str, start: date, end: date) -> dict[date, pd.DataFrame]:
    engine = create_engine(engine_url, pool_pre_ping=True)
    sql = """
        SELECT id, station_id, model, variable_code, target_date_local, asof_utc, runtime_utc, retrieved_at_utc,
               value_mean, value_max, value_min
        FROM mos_daily_value
        WHERE station_id = :station_id
          AND target_date_local BETWEEN :start_date AND :end_date
          AND UPPER(model) IN ('GFS','NAM')
          AND LOWER(variable_code) IN ('tmp','dpt','wdr','wsp','p12','q12','cig','vis')
    """
    df = pd.read_sql(
        text(sql),
        engine,
        params={"station_id": station_id, "start_date": start.isoformat(), "end_date": end.isoformat()},
    )
    if df.empty:
        return {}
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    return {d: grp.copy() for d, grp in df.groupby("target_date_local")}


def _load_minute_data(minute_dir: Path, years: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in years:
        path = minute_dir / f"MIA_tmpf_1min_UTC_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing minute file: {path}")
        df = pd.read_csv(path, usecols=["station", "valid(UTC)", "tmpf"], dtype={"station": "string"})
        df = df[df["station"].str.upper().isin({"MIA", "KMIA"})]
        df = df.rename(columns={"valid(UTC)": "valid_utc"})
        df["ts_utc"] = pd.to_datetime(df["valid_utc"], errors="coerce", utc=True)
        df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
        df = df.dropna(subset=["ts_utc", "tmpf"])
        if not df.empty:
            frames.append(df[["ts_utc", "tmpf"]])
    if not frames:
        raise ValueError("No minute data loaded.")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("ts_utc")
    df = df.set_index("ts_utc").resample("5min").median().reset_index()
    tz = ZoneInfo("America/New_York")
    df["ts_local"] = df["ts_utc"].dt.tz_convert(tz)
    df["local_date"] = df["ts_local"].dt.date
    df["local_minute_of_day"] = df["ts_local"].dt.hour * 60 + df["ts_local"].dt.minute
    df["utc_date"] = df["ts_utc"].dt.date
    df["utc_minute_of_day"] = df["ts_utc"].dt.hour * 60 + df["ts_utc"].dt.minute
    return df


def _fetch_missing_minute_data(missing_t1: list[date], missing_early: list[date]) -> pd.DataFrame:
    tz = ZoneInfo("America/New_York")
    frames: list[pd.DataFrame] = []
    # Fetch full local day for missing T-1 dates.
    for day in missing_t1:
        start_local = datetime.combine(day, datetime.min.time(), tzinfo=tz)
        end_local = datetime.combine(day, datetime.max.time().replace(microsecond=0), tzinfo=tz)
        df = _fetch_iem_minute("MIA", start_local.astimezone(UTC), end_local.astimezone(UTC))
        if not df.empty:
            frames.append(df)
    # Fetch 00-06Z for missing early dates.
    for day in missing_early:
        start_utc = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
        end_utc = start_utc + timedelta(hours=6)
        df = _fetch_iem_minute("MIA", start_utc, end_utc)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ts_utc", "tmpf"])
    return pd.concat(frames, ignore_index=True)


def _precompute_minute_features(
    minute_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows = [_compute_daily_features(group) for _, group in minute_df.groupby("local_date", sort=True)]
    daily_df = pd.DataFrame([r.__dict__ for r in daily_rows])
    daily_df = daily_df.sort_values("local_date").reset_index(drop=True)
    if "iem_tmax" in daily_df.columns:
        daily_df = daily_df[daily_df["iem_tmax"].notna()].reset_index(drop=True)

    early_rows: list[dict[str, Any]] = []
    for utc_date, group in minute_df.groupby("utc_date", sort=True):
        feats = _compute_early_features(group)
        feats["utc_date"] = utc_date
        early_rows.append(feats)
    early_df = pd.DataFrame(early_rows)
    if not early_df.empty:
        early_df = early_df[
            ~(early_df[["T00", "T03", "T06"]].isna().all(axis=1))
        ].reset_index(drop=True)
    return daily_df, early_df


def _build_minute_feature_lookup(
    feature_store: pd.DataFrame,
    daily_df: pd.DataFrame,
    early_df: pd.DataFrame,
    truth: dict[date, float],
    pred_start: date,
    eval_end: date,
) -> dict[str, pd.Series]:
    fs_dates = pd.to_datetime(feature_store["target_date_local"]).dt.date
    diff_hist = pd.Series(
        pd.to_numeric(feature_store["diff_lag1"], errors="coerce").to_numpy(dtype=float),
        index=fs_dates - timedelta(days=1),
    )
    diff_hist = diff_hist[diff_hist.index.notnull()].sort_index()

    # Match the live script: only extend diff history for dates missing from the feature store.
    extra_dates = []
    for d in daily_df["local_date"]:
        v = float(diff_hist.get(d, np.nan))
        if (d not in diff_hist.index) or (not np.isfinite(v)):
            extra_dates.append(d)
    if extra_dates:
        extra_df = daily_df[daily_df["local_date"].isin(extra_dates)].copy()
        extra_df["y_tmax"] = extra_df["local_date"].map(truth)
        extra_df["diff"] = extra_df["y_tmax"] - extra_df["iem_tmax"]
        extra_df = extra_df[extra_df["diff"].notna()]
        extra_series = pd.Series(extra_df["diff"].to_numpy(dtype=float), index=extra_df["local_date"])
        diff_series = pd.concat([diff_hist, extra_series]).sort_index()
        diff_series = diff_series.groupby(diff_series.index).last()
    else:
        diff_series = diff_hist
    diff_series = diff_series[diff_series.index <= (eval_end - timedelta(days=1))]

    ewma_30 = _ewma_half_life(diff_series, 30)
    diff_std_30 = diff_series.rolling(30, min_periods=5).std()

    return {
        "diff_series": diff_series,
        "ewma_30": ewma_30,
        "std_30": diff_std_30,
        "extra_dates_used": pd.Series(extra_dates, dtype="object"),
    }


def _precompute_zscore_stats(
    feature_store: pd.DataFrame,
    train_mask: np.ndarray,
    cols: list[str],
) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for col in cols:
        series = pd.to_numeric(feature_store[col], errors="coerce")
        train_vals = series[train_mask]
        stats[col] = (float(train_vals.mean()), float(train_vals.std()))
    return stats


def _build_minute_features_for_day(
    target_date: date,
    daily_idx: pd.DataFrame,
    early_idx: pd.DataFrame,
    diff_lookup: dict[str, pd.Series],
    zstats: dict[str, tuple[float, float]],
) -> dict[str, float]:
    t1_date = target_date - timedelta(days=1)
    if t1_date not in daily_idx.index:
        raise ValueError(f"Missing T-1 minute data for {t1_date}")
    t1 = daily_idx.loc[t1_date]

    if target_date not in early_idx.index:
        raise ValueError(f"Missing early-minute data for {target_date}")
    early_feats = early_idx.loc[target_date]

    diff_series = diff_lookup["diff_series"]
    ewma_30 = diff_lookup["ewma_30"]
    diff_std_30 = diff_lookup["std_30"]

    diff_lag1 = float(diff_series.get(t1_date, np.nan))
    diff_ewma_30 = float(ewma_30.get(t1_date, np.nan))
    diff_std_30_val = float(diff_std_30.get(t1_date, np.nan))

    def zscore(val: float, col: str, negate: bool = False) -> float:
        key = f"{col}__neg" if negate else col
        mean, std = zstats.get(key, (0.0, 0.0))
        if std == 0 or np.isnan(std):
            return 0.0
        v = -val if negate else val
        return (v - mean) / std

    z_range = zscore(float(t1["iem_range"]), "iem_range_t1", negate=True)
    z_plateau = zscore(float(t1["plateau_05"]), "plateau_05_t1")
    z_drop_cnt = zscore(float(t1["drop_cnt_15_19"]), "drop_cnt_15_19_t1")
    z_max_drop = zscore(float(t1["max_drop_30"]), "max_drop_30_t1")
    z_heat_12_15 = zscore(float(t1["heat_12_15"]), "heat_12_15_t1", negate=True)
    z_heat_diff = zscore(float(t1["heat_12_15"] - t1["heat_15_18"]), "heat_diff_t1")
    mri_suppress = (
        1.2 * z_range
        + 1.0 * z_plateau
        + 1.0 * z_drop_cnt
        + 0.8 * z_max_drop
        + 0.6 * z_heat_12_15
        + 0.6 * z_heat_diff
    )
    z_tmax_time = zscore(float(t1["tmax_time_min"]), "tmax_time_min_t1")
    z_heat_15_18 = zscore(float(t1["heat_15_18"]), "heat_15_18_t1")
    mri_late = 1.0 * z_tmax_time + 0.8 * z_heat_15_18 - 0.6 * z_drop_cnt

    return {
        "iem_tmax_t1": float(t1["iem_tmax"]),
        "iem_tmin_t1": float(t1["iem_tmin"]),
        "iem_range_t1": float(t1["iem_range"]),
        "tmax_time_min_t1": float(t1["tmax_time_min"]),
        "plateau_05_t1": float(t1["plateau_05"]),
        "heat_12_15_t1": float(t1["heat_12_15"]),
        "heat_15_18_t1": float(t1["heat_15_18"]),
        "cool_18_21_t1": float(t1["cool_18_21"]),
        "max_drop_30_t1": float(t1["max_drop_30"]),
        "drop_cnt_15_19_t1": float(t1["drop_cnt_15_19"]),
        "T00": float(early_feats["T00"]),
        "T03": float(early_feats["T03"]),
        "T06": float(early_feats["T06"]),
        "night_drop_00_06": float(early_feats["night_drop_00_06"]),
        "slope_last180": float(early_feats["slope_last180"]),
        "std_last180": float(early_feats["std_last180"]),
        "diff_lag1": diff_lag1,
        "diff_ewma_30": diff_ewma_30,
        "diff_std_30": diff_std_30_val,
        "T06_adj": float(early_feats["T06"]) + diff_ewma_30 if np.isfinite(diff_ewma_30) else np.nan,
        "MRI_suppress": float(mri_suppress),
        "MRI_late": float(mri_late),
    }


def _prepare_feature_store_date_index(feature_store: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.to_datetime(feature_store["target_date_local"], errors="coerce")
    valid_mask = dates.notna().to_numpy()
    if not valid_mask.any():
        raise ValueError("feature_store has no valid target_date_local values")
    idx = np.nonzero(valid_mask)[0]
    dates64 = dates.to_numpy(dtype="datetime64[D]")[valid_mask]
    order_rel = np.argsort(dates64, kind="mergesort")
    return dates64[order_rel], idx[order_rel]


def _compute_base_fast(
    *,
    feature_store: pd.DataFrame,
    fs_dates_sorted: np.ndarray,
    fs_idx_sorted: np.ndarray,
    tmp_buckets: dict[str, float],
    target_date: date,
) -> float:
    cutoff = np.datetime64(target_date - timedelta(days=1))
    pos = int(np.searchsorted(fs_dates_sorted, cutoff, side="right") - 1)
    if pos < 0:
        raise ValueError("No historical rows available for baseline.")
    last = feature_store.iloc[int(fs_idx_sorted[pos])]

    alpha_15 = _half_life_alpha(15.0)
    corrected: list[float] = []
    for model in ["gfs", "nam"]:
        for bucket in [0, 12, 24, 36]:
            raw_prev = float(last.get(f"feat_tmp_max_{model}_b{bucket}", np.nan))
            corr_prev = float(last.get(f"feat_tmp_corr_{model}_b{bucket}", np.nan))
            bias_prev = corr_prev - raw_prev if np.isfinite(corr_prev) and np.isfinite(raw_prev) else np.nan
            err_prev = float(last["y_actual_tmax_f"]) - raw_prev if np.isfinite(raw_prev) else np.nan
            if np.isnan(bias_prev):
                bias_prev = 0.0
            if np.isnan(err_prev):
                bias_t = bias_prev
            else:
                bias_t = (1 - alpha_15) * bias_prev + alpha_15 * err_prev
            raw_t = tmp_buckets.get(f"feat_tmp_max_{model}_b{bucket}", np.nan)
            corrected.append(raw_t + bias_t if np.isfinite(raw_t) else np.nan)
    corrected_arr = np.array(corrected, dtype=float)
    if np.all(np.isnan(corrected_arr)):
        fallback = float(last.get("feat_le_median_biascorr", np.nan))
        if not np.isfinite(fallback):
            fallback = float(last.get("y_actual_tmax_f", np.nan))
        print("Warning: MOS buckets missing; falling back to last available baseline.", file=sys.stderr)
        return fallback
    return float(np.nanmedian(corrected_arr))


def _compute_predictions(
    *,
    feature_store: pd.DataFrame,
    fs_dates_sorted: np.ndarray,
    fs_idx_sorted: np.ndarray,
    mos_by_date: dict[date, pd.DataFrame],
    daily_idx: pd.DataFrame,
    early_idx: pd.DataFrame,
    diff_lookup: dict[str, pd.Series],
    zstats: dict[str, tuple[float, float]],
    pred_start: date,
    eval_end: date,
    train_mask: np.ndarray,
    gate_model: Any,
    models: dict[str, Any],
    gate_fill: dict[str, float],
    exp_fill: dict[str, float],
    best_k: float,
    tau_10: float,
    sigma_floor: float,
    blocked_dates: set[date],
) -> dict[date, dict[str, float]]:
    gate_features = ["feat_u", "feat_v", "feat_wsp_mean", "cal_d_doy_sin", "cal_d_doy_cos"]
    expert_features = [
        "feat_dd_models",
        "feat_tmp_range_mean_models",
        "feat_p12_max",
        "feat_q12_max",
        "feat_cig_min",
        "feat_u",
        "feat_v",
        "cal_d_doy_sin",
        "cal_d_doy_cos",
        "iem_tmax_t1",
        "iem_tmin_t1",
        "iem_range_t1",
        "tmax_time_min_t1",
        "plateau_05_t1",
        "heat_12_15_t1",
        "heat_15_18_t1",
        "cool_18_21_t1",
        "max_drop_30_t1",
        "drop_cnt_15_19_t1",
        "T00",
        "T03",
        "T06",
        "night_drop_00_06",
        "slope_last180",
        "std_last180",
        "T06_adj",
        "diff_lag1",
        "diff_ewma_30",
        "diff_std_30",
        "MRI_suppress",
        "MRI_late",
    ]

    preds: dict[date, dict[str, float]] = {}
    y_train = pd.to_numeric(feature_store["y_actual_tmax_f"], errors="coerce").to_numpy(dtype=float)
    base_mean = float(np.nanmean(y_train[train_mask]))

    for d in _date_range(pred_start, eval_end):
        if d in blocked_dates:
            continue
        mos_df = mos_by_date.get(d)
        if mos_df is None or mos_df.empty:
            raise ValueError(f"MOS missing for {d}")
        decision_utc = datetime.combine(d, datetime.min.time(), tzinfo=UTC) + timedelta(hours=6)
        mos_features, tmp_buckets, _ = build_mos_features(mos_df, decision_utc)
        base_val = _compute_base_fast(
            feature_store=feature_store,
            fs_dates_sorted=fs_dates_sorted,
            fs_idx_sorted=fs_idx_sorted,
            tmp_buckets=tmp_buckets,
            target_date=d,
        )

        minute_features = _build_minute_features_for_day(
            d, daily_idx, early_idx, diff_lookup, zstats
        )

        doy = d.timetuple().tm_yday
        cal_d_doy_sin = math.sin(2 * math.pi * doy / 365.0)
        cal_d_doy_cos = math.cos(2 * math.pi * doy / 365.0)

        row = {
            "target_date_local": d,
            "feat_le_median_biascorr": base_val,
            "cal_d_doy_sin": cal_d_doy_sin,
            "cal_d_doy_cos": cal_d_doy_cos,
        }
        row.update(mos_features)
        row.update(tmp_buckets)
        row.update(minute_features)

        X_gate = _impute_row(row, gate_features, gate_fill).reshape(1, -1)
        p_gate = float(gate_model.predict_proba(X_gate)[0, 1])

        X = _impute_row(row, expert_features, exp_fill).reshape(1, -1)
        r_on_10 = float(models["on_10"].predict(X)[0])
        r_on_50 = float(models["on_50"].predict(X)[0])
        r_on_90 = float(models["on_90"].predict(X)[0])
        r_off_10 = float(models["off_10"].predict(X)[0])
        r_off_50 = float(models["off_50"].predict(X)[0])
        r_off_90 = float(models["off_90"].predict(X)[0])

        r10 = p_gate * r_on_10 + (1 - p_gate) * r_off_10
        r50 = p_gate * r_on_50 + (1 - p_gate) * r_off_50
        r90 = p_gate * r_on_90 + (1 - p_gate) * r_off_90

        base_val = base_val if np.isfinite(base_val) else base_mean
        spread = r90 - r10
        w = float(np.exp(-best_k * spread))
        mu = float(base_val + w * r50)
        q10 = float(base_val + r10 - tau_10)
        q90 = float(base_val + r90 + tau_10)
        z10 = -1.281551565545
        z90 = 1.281551565545
        sigma_hat = float((q90 - q10) / (z90 - z10)) if q90 > q10 else float("nan")
        if not np.isfinite(sigma_hat) or sigma_hat < sigma_floor:
            sigma_hat = sigma_floor

        preds[d] = {"mu": mu, "sigma_hat": sigma_hat}

    return preds


def _build_leaderboard(
    *,
    preds: dict[date, dict[str, float]],
    truth_eval: dict[date, float],
    cfg: Config,
    blocked_dates: set[date],
) -> pd.DataFrame:
    truth_vals = np.array(
        [truth_eval[d] for d in truth_eval if cfg.eval_start <= d <= cfg.eval_end], dtype=float
    )
    if truth_vals.size == 0:
        thresholds = list(range(80, 101))
    else:
        p05 = float(np.nanpercentile(truth_vals, 5))
        p95 = float(np.nanpercentile(truth_vals, 95))
        low = max(int(math.floor(p05 - 5)), 40)
        high = min(int(math.ceil(p95 + 5)), 110)
        thresholds = list(range(low, high + 1))

    available_dates = sorted([d for d in preds if d in truth_eval])
    leaderboard: list[dict[str, Any]] = []
    for window in cfg.windows:
        for method in ["normal", "emos", "empirical"]:
            rows: list[dict[str, Any]] = []
            for d in _date_range(cfg.eval_start, cfg.eval_end):
                if d in blocked_dates:
                    continue
                hist_end = d - timedelta(days=cfg.truth_lag_days)
                import bisect

                idx = bisect.bisect_right(available_dates, hist_end) - 1
                if idx < window - 1:
                    continue
                hist_dates = available_dates[(idx - window + 1) : (idx + 1)]
                hist = [(preds[hd]["mu"], preds[hd]["sigma_hat"], truth_eval[hd]) for hd in hist_dates]
                if len(hist) < window or d not in preds or d not in truth_eval:
                    continue

                mu_hat = preds[d]["mu"]
                sigma_hat_day = preds[d]["sigma_hat"]
                actual = truth_eval[d]

                mu_hist = np.array([m for (m, s, a) in hist], dtype=float)
                sig_hist = np.array([s for (m, s, a) in hist], dtype=float)
                y_hist = np.array([a for (m, s, a) in hist], dtype=float)
                resid = mu_hist - y_hist

                if method == "normal":
                    bias = float(np.mean(resid))
                    sigma = float(np.std(resid, ddof=0))
                    if sigma < cfg.sigma_floor:
                        sigma = cfg.sigma_floor
                    mu = mu_hat - bias
                    crps = _crps_normal(mu, sigma, actual)
                    nll = 0.5 * math.log(2 * math.pi * sigma**2) + ((actual - mu) ** 2) / (
                        2 * sigma**2
                    )
                    briers = []
                    for t in thresholds:
                        p = 1.0 - _norm_cdf(t, mu, sigma)
                        y = 1.0 if actual >= t else 0.0
                        briers.append((p - y) ** 2)
                    rows.append(
                        {
                            "crps": crps,
                            "nll": float(nll),
                            "brier": float(np.mean(briers)),
                            "abs_err": abs(mu - actual),
                            "sq_err": (mu - actual) ** 2,
                        }
                    )

                elif method == "emos":
                    hist_df = pd.DataFrame(
                        {"mu_hat_f": mu_hist, "sigma_hat_f": sig_hist, "actual_tmax_f": y_hist}
                    )
                    emos_result = calibrate(hist_df, sigma_hat_day, sigma_floor=cfg.sigma_floor)
                    sigma = emos_result.sigma_emos
                    mu = mu_hat
                    crps = _crps_normal(mu, sigma, actual)
                    nll = 0.5 * math.log(2 * math.pi * sigma**2) + ((actual - mu) ** 2) / (
                        2 * sigma**2
                    )
                    briers = []
                    for t in thresholds:
                        p = 1.0 - _norm_cdf(t, mu, sigma)
                        y = 1.0 if actual >= t else 0.0
                        briers.append((p - y) ** 2)
                    rows.append(
                        {
                            "crps": crps,
                            "nll": float(nll),
                            "brier": float(np.mean(briers)),
                            "abs_err": abs(mu - actual),
                            "sq_err": (mu - actual) ** 2,
                        }
                    )

                else:  # empirical
                    samples = mu_hat - resid
                    crps = _crps_empirical(samples, actual)
                    briers = []
                    for t in thresholds:
                        p = _empirical_cdf(mu_hat - t, resid)
                        y = 1.0 if actual >= t else 0.0
                        briers.append((p - y) ** 2)
                    bias = float(np.mean(resid))
                    mu = mu_hat - bias
                    rows.append(
                        {
                            "crps": crps,
                            "nll": float("nan"),
                            "brier": float(np.mean(briers)),
                            "abs_err": abs(mu - actual),
                            "sq_err": (mu - actual) ** 2,
                        }
                    )

            if not rows:
                continue
            df = pd.DataFrame(rows)
            leaderboard.append(
                {
                    "method": method,
                    "window": window,
                    "n_days": int(len(df)),
                    "mae": float(df["abs_err"].mean()),
                    "rmse": float(math.sqrt(df["sq_err"].mean())),
                    "crps": float(df["crps"].mean()),
                    "brier": float(df["brier"].mean()),
                    "nll": float(df["nll"].mean()) if df["nll"].notna().any() else float("nan"),
                }
            )

    leader_df = pd.DataFrame(leaderboard)
    return leader_df.sort_values(["crps", "brier", "mae"], ascending=True)


def _leaderboard_to_text(leader_df: pd.DataFrame, *, blocked_days: int | None = None) -> str:
    lines = [
        "method  window  n_days  MAE    RMSE   CRPS   Brier  NLL",
        "------  ------  ------  -----  -----  -----  -----  -----",
    ]
    for _, r in leader_df.iterrows():
        nll = "" if not np.isfinite(r["nll"]) else f"{r['nll']:.3f}"
        lines.append(
            f"{r['method']:<6}  {int(r['window']):<6}  {int(r['n_days']):<6}  {r['mae']:.3f}  "
            f"{r['rmse']:.3f}  {r['crps']:.3f}  {r['brier']:.3f}  {nll}"
        )
    if blocked_days is not None:
        lines.append("")
        lines.append(f"Blocked days (missing minute data): {blocked_days}")
    return "\n".join(lines)


def _load_offline_preds(winner_dir: Path) -> pd.DataFrame:
    preds_path = winner_dir / "preds.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"preds.parquet not found: {preds_path}")
    df = pd.read_parquet(preds_path)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    return df


def _impute_row(row: dict[str, float], cols: list[str], fill_values: dict[str, float]) -> np.ndarray:
    values = []
    for col in cols:
        val = row.get(col, np.nan)
        if not np.isfinite(val):
            val = float(fill_values.get(col, 0.0))
        values.append(val)
    return np.array(values, dtype=float)


def main() -> int:
    cfg = parse_args()
    db_url = os.getenv("MYSQL_URL") or f"mysql+pymysql://{os.getenv('MYSQL_USER','root')}:{os.getenv('MYSQL_PASSWORD','root')}@{os.getenv('MYSQL_HOST','localhost')}:{os.getenv('MYSQL_PORT','3306')}/{os.getenv('MYSQL_DB','weather_predictionmarkets')}"

    feature_store = _load_feature_store(cfg.winner_dir)
    tau_05, tau_10 = _load_calibration_taus(cfg.winner_dir)

    gate_model, models, gate_fill, exp_fill, train_mask, best_k = _train_models(feature_store)

    pred_start = cfg.eval_start - timedelta(days=(max(cfg.windows) + cfg.truth_lag_days - 1))
    truth_station = _load_truth(db_url, cfg.station_id, pred_start, cfg.eval_end)
    truth_feature_store = _truth_from_feature_store(feature_store, pred_start, cfg.eval_end)
    mos_by_date = _load_mos_range(db_url, cfg.station_id, pred_start, cfg.eval_end)

    years = sorted({pred_start.year, cfg.eval_end.year})
    if pred_start.year != cfg.eval_end.year:
        years = list(range(pred_start.year, cfg.eval_end.year + 1))
    minute_df = _load_minute_data(cfg.minute_dir, years)

    daily_df, early_df = _precompute_minute_features(minute_df)

    # Ensure no missing minute data in required window. If missing, fetch from IEM.
    required_dates = set(_date_range(pred_start, cfg.eval_end))
    missing_t1 = sorted([d for d in required_dates if d - timedelta(days=1) not in set(daily_df["local_date"])])
    missing_early = sorted([d for d in required_dates if d not in set(early_df["utc_date"])])
    if missing_t1 or missing_early:
        fetched = _fetch_missing_minute_data(missing_t1, missing_early)
        if not fetched.empty:
            tz = ZoneInfo("America/New_York")
            fetched["ts_utc"] = pd.to_datetime(fetched["ts_utc"], errors="coerce", utc=True)
            fetched["tmpf"] = pd.to_numeric(fetched["tmpf"], errors="coerce")
            fetched = fetched.dropna(subset=["ts_utc", "tmpf"])
            if not fetched.empty:
                fetched = fetched.sort_values("ts_utc")
                fetched = fetched.set_index("ts_utc").resample("5min").median().reset_index()
                fetched["ts_local"] = fetched["ts_utc"].dt.tz_convert(tz)
                fetched["local_date"] = fetched["ts_local"].dt.date
                fetched["local_minute_of_day"] = fetched["ts_local"].dt.hour * 60 + fetched["ts_local"].dt.minute
                fetched["utc_date"] = fetched["ts_utc"].dt.date
                fetched["utc_minute_of_day"] = fetched["ts_utc"].dt.hour * 60 + fetched["ts_utc"].dt.minute
                minute_df = pd.concat([minute_df, fetched], ignore_index=True)
                minute_df = minute_df.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last")
                daily_df, early_df = _precompute_minute_features(minute_df)

        missing_t1 = sorted([d for d in required_dates if d - timedelta(days=1) not in set(daily_df["local_date"])])
        missing_early = sorted([d for d in required_dates if d not in set(early_df["utc_date"])])
        if missing_t1 or missing_early:
            missing_path = cfg.out_dir / "missing_minute_dates.txt"
            lines = []
            lines.append(f"T-1 missing ({len(missing_t1)}): {missing_t1[:20]}")
            lines.append(f"Early missing ({len(missing_early)}): {missing_early[:20]}")
            missing_path.write_text("\n".join(lines), encoding="utf-8")

    blocked_dates: set[date] = set(missing_t1) | set(missing_early)

    fs_dates_sorted, fs_idx_sorted = _prepare_feature_store_date_index(feature_store)

    daily_idx = daily_df.set_index("local_date")
    early_idx = early_df.set_index("utc_date")

    z_cols = [
        "iem_range_t1",
        "plateau_05_t1",
        "drop_cnt_15_19_t1",
        "max_drop_30_t1",
        "heat_12_15_t1",
        "tmax_time_min_t1",
        "heat_15_18_t1",
    ]
    zstats = _precompute_zscore_stats(feature_store, train_mask, z_cols)
    # Derived z-score stats to match the feature-store minute builder.
    iem_range = pd.to_numeric(feature_store.get("iem_range_t1"), errors="coerce")
    heat_12_15 = pd.to_numeric(feature_store.get("heat_12_15_t1"), errors="coerce")
    heat_15_18 = pd.to_numeric(feature_store.get("heat_15_18_t1"), errors="coerce")
    if iem_range is not None:
        vals = (-iem_range)[train_mask]
        zstats["iem_range_t1__neg"] = (float(vals.mean()), float(vals.std()))
    if heat_12_15 is not None:
        vals = (-heat_12_15)[train_mask]
        zstats["heat_12_15_t1__neg"] = (float(vals.mean()), float(vals.std()))
    if (heat_12_15 is not None) and (heat_15_18 is not None):
        heat_diff = heat_12_15 - heat_15_18
        vals = heat_diff[train_mask]
        zstats["heat_diff_t1"] = (float(vals.mean()), float(vals.std()))

    diff_lookup_station = _build_minute_feature_lookup(
        feature_store, daily_df, early_df, truth_station, pred_start, cfg.eval_end
    )
    if len(diff_lookup_station.get("extra_dates_used", [])) == 0:
        diff_lookup_fs = diff_lookup_station
    else:
        diff_lookup_fs = _build_minute_feature_lookup(
            feature_store, daily_df, early_df, truth_feature_store, pred_start, cfg.eval_end
        )

    preds_station = _compute_predictions(
        feature_store=feature_store,
        fs_dates_sorted=fs_dates_sorted,
        fs_idx_sorted=fs_idx_sorted,
        mos_by_date=mos_by_date,
        daily_idx=daily_idx,
        early_idx=early_idx,
        diff_lookup=diff_lookup_station,
        zstats=zstats,
        pred_start=pred_start,
        eval_end=cfg.eval_end,
        train_mask=train_mask,
        gate_model=gate_model,
        models=models,
        gate_fill=gate_fill,
        exp_fill=exp_fill,
        best_k=best_k,
        tau_10=tau_10,
        sigma_floor=cfg.sigma_floor,
        blocked_dates=blocked_dates,
    )

    if diff_lookup_fs is diff_lookup_station:
        preds_fs = preds_station
    else:
        preds_fs = _compute_predictions(
            feature_store=feature_store,
            fs_dates_sorted=fs_dates_sorted,
            fs_idx_sorted=fs_idx_sorted,
            mos_by_date=mos_by_date,
            daily_idx=daily_idx,
            early_idx=early_idx,
            diff_lookup=diff_lookup_fs,
            zstats=zstats,
            pred_start=pred_start,
            eval_end=cfg.eval_end,
            train_mask=train_mask,
            gate_model=gate_model,
            models=models,
            gate_fill=gate_fill,
            exp_fill=exp_fill,
            best_k=best_k,
            tau_10=tau_10,
            sigma_floor=cfg.sigma_floor,
            blocked_dates=blocked_dates,
        )

    leader_station = _build_leaderboard(
        preds=preds_station, truth_eval=truth_station, cfg=cfg, blocked_dates=blocked_dates
    )
    leader_fs = _build_leaderboard(
        preds=preds_fs, truth_eval=truth_feature_store, cfg=cfg, blocked_dates=blocked_dates
    )

    out_station_csv = cfg.out_dir / "leaderboard_2024_calibration_station_truth.csv"
    out_station_txt = cfg.out_dir / "leaderboard_2024_calibration_station_truth.txt"
    leader_station.to_csv(out_station_csv, index=False)
    out_station_txt.write_text(
        _leaderboard_to_text(leader_station, blocked_days=len(blocked_dates)), encoding="utf-8"
    )

    out_fs_csv = cfg.out_dir / "leaderboard_2024_calibration_featurestore_truth.csv"
    out_fs_txt = cfg.out_dir / "leaderboard_2024_calibration_featurestore_truth.txt"
    leader_fs.to_csv(out_fs_csv, index=False)
    out_fs_txt.write_text(
        _leaderboard_to_text(leader_fs, blocked_days=len(blocked_dates)), encoding="utf-8"
    )

    merged = leader_station.merge(
        leader_fs, on=["method", "window"], how="inner", suffixes=("_station", "_fs")
    )
    for metric in ["mae", "rmse", "crps", "brier", "nll"]:
        if f"{metric}_station" in merged.columns and f"{metric}_fs" in merged.columns:
            merged[f"delta_{metric}"] = merged[f"{metric}_station"] - merged[f"{metric}_fs"]

    merged = merged.sort_values(["crps_station", "brier_station", "mae_station"], ascending=True)
    out_delta_csv = cfg.out_dir / "leaderboard_2024_calibration_truth_delta.csv"
    out_delta_txt = cfg.out_dir / "leaderboard_2024_calibration_truth_delta.txt"
    merged.to_csv(out_delta_csv, index=False)

    lines = [
        "method  window  n_station  MAE_s  RMSE_s  CRPS_s  Brier_s  n_fs  MAE_f  RMSE_f  CRPS_f  Brier_f  dMAE  dCRPS",
        "------  ------  ---------  -----  ------  ------  -------  ----  -----  ------  ------  -------  ----  -----",
    ]
    for _, row in merged.iterrows():
        lines.append(
            f"{row['method']:<6}  {int(row['window']):<6}  {int(row['n_days_station']):<9}  "
            f"{row['mae_station']:.3f}  {row['rmse_station']:.3f}  {row['crps_station']:.3f}  {row['brier_station']:.3f}  "
            f"{int(row['n_days_fs']):<4}  {row['mae_fs']:.3f}  {row['rmse_fs']:.3f}  {row['crps_fs']:.3f}  {row['brier_fs']:.3f}  "
            f"{row['delta_mae']:.3f}  {row['delta_crps']:.3f}"
        )
    out_delta_txt.write_text("\n".join(lines), encoding="utf-8")

    # Compare live preds vs offline preds.parquet on 2024 dates.
    try:
        offline_df = _load_offline_preds(cfg.winner_dir)
        offline_df = offline_df[
            (offline_df["target_date_local"] >= cfg.eval_start)
            & (offline_df["target_date_local"] <= cfg.eval_end)
        ].copy()
        offline_df = offline_df.rename(columns={"V5+8": "mu_offline", "y": "y_offline"})
        live_rows = []
        for d, vals in preds_station.items():
            if cfg.eval_start <= d <= cfg.eval_end:
                live_rows.append(
                    {
                        "target_date_local": d,
                        "mu_live": float(vals["mu"]),
                        "sigma_live": float(vals["sigma_hat"]),
                        "y_station": truth_station.get(d, float("nan")),
                    }
                )
        live_df = pd.DataFrame(live_rows)
        compare = live_df.merge(offline_df[["target_date_local", "mu_offline"]], on="target_date_local", how="inner")
        compare["abs_err_live"] = (compare["mu_live"] - compare["y_station"]).abs()
        compare["abs_err_offline"] = (compare["mu_offline"] - compare["y_station"]).abs()
        compare["diff_live_minus_offline"] = compare["mu_live"] - compare["mu_offline"]

        out_compare_csv = cfg.out_dir / "preds_compare_2024.csv"
        compare.to_csv(out_compare_csv, index=False)

        summary_lines = [
            "preds_compare_2024 summary",
            f"best_k={best_k}",
            f"n_days={len(compare)}",
            f"mae_live={compare['abs_err_live'].mean():.6f}",
            f"mae_offline={compare['abs_err_offline'].mean():.6f}",
            f"mae_gap_live_minus_offline={(compare['abs_err_live'].mean() - compare['abs_err_offline'].mean()):.6f}",
            f"mean_mu_diff_live_minus_offline={compare['diff_live_minus_offline'].mean():.6f}",
        ]
        (cfg.out_dir / "preds_compare_2024_summary.txt").write_text(
            "\n".join(summary_lines), encoding="utf-8"
        )
    except Exception as exc:
        warn_path = cfg.out_dir / "preds_compare_2024_summary.txt"
        warn_path.write_text(f"compare_failed: {exc}", encoding="utf-8")

    print(out_station_txt)
    print(out_fs_txt)
    print(out_delta_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

