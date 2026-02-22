import argparse
import json
import math
import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors

import lightgbm as lgb
from sqlalchemy import create_engine, text

VERSION = "hit1830_v2_suite_v1"

LOCAL_TZ = "America/New_York"
STOCKHOLM_TZ = "Europe/Stockholm"

START_DATE = date(2002, 1, 1)
END_DATE = date(2026, 12, 31)

EPS = 0.01
KNN_K = 50

MOS_VARIABLES = [
    "n_x",
    "tmp",
    "dpt",
    "wdr",
    "wsp",
    "p06",
    "p12",
    "q06",
    "q12",
    "t06",
    "t06_1",
    "t06_2",
    "t12",
    "t12_1",
    "t12_2",
    "cig",
    "vis",
    "pos",
    "poz",
]

MOS_MODELS = ["GFS", "NAM"]
MOS_VARIABLES_SQL = ", ".join(f"'{v}'" for v in MOS_VARIABLES)


@dataclass
class DayWindow:
    target_date_local: date
    day_start_utc: datetime
    day_end_utc: datetime
    cutoff_utc: datetime


def _sha256_str(text_value: str) -> str:
    return hashlib.sha256(text_value.encode("utf-8")).hexdigest()


def _date_range(start: date, end: date) -> List[date]:
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def _build_day_windows() -> pd.DataFrame:
    local = ZoneInfo(LOCAL_TZ)
    stockholm = ZoneInfo(STOCKHOLM_TZ)
    rows = []
    for day in _date_range(START_DATE, END_DATE):
        day_start_local = datetime(day.year, day.month, day.day, 0, 0, tzinfo=local)
        day_end_local = day_start_local + timedelta(days=1)
        day_start_utc = day_start_local.astimezone(timezone.utc)
        day_end_utc = day_end_local.astimezone(timezone.utc)

        cutoff_local = datetime(day.year, day.month, day.day, 18, 30, tzinfo=stockholm)
        cutoff_utc = cutoff_local.astimezone(timezone.utc)

        if cutoff_utc < day_start_utc:
            cutoff_utc = day_start_utc
        if cutoff_utc >= day_end_utc:
            cutoff_utc = day_end_utc - timedelta(seconds=1)

        rows.append(
            {
                "target_date_local": day,
                "day_start_utc": day_start_utc,
                "day_end_utc": day_end_utc,
                "cutoff_utc": cutoff_utc,
            }
        )
    return pd.DataFrame(rows)


def _expected_points(start_utc: datetime, end_utc: datetime, freq_minutes: int, inclusive_end: bool) -> int:
    seconds = (end_utc - start_utc).total_seconds()
    if inclusive_end:
        return int(seconds // (freq_minutes * 60)) + 1
    return int(seconds // (freq_minutes * 60))


def _ols_slope(minutes: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return float("nan")
    x = minutes[mask] / 60.0
    y = values[mask].astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return float("nan")
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _count_sign_changes(values: np.ndarray) -> float:
    diffs = np.diff(values)
    if diffs.size == 0:
        return float("nan")
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if signs.size <= 1:
        return 0.0
    return float(np.sum(signs[1:] * signs[:-1] < 0))


def _longest_run(mask: np.ndarray) -> int:
    best = 0
    run = 0
    for val in mask:
        if val:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def _dct_matrix(n: int, kmax: int) -> np.ndarray:
    n_idx = np.arange(n)[:, None]
    k_idx = np.arange(kmax)[None, :]
    return np.cos(np.pi / n * (n_idx + 0.5) * k_idx)


def _segment_fit(prefix: Dict[str, np.ndarray], start: int, end: int) -> Tuple[float, float, float]:
    n = end - start
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    sx = prefix["sx"][end] - prefix["sx"][start]
    sy = prefix["sy"][end] - prefix["sy"][start]
    sxx = prefix["sxx"][end] - prefix["sxx"][start]
    sxy = prefix["sxy"][end] - prefix["sxy"][start]
    syy = prefix["syy"][end] - prefix["syy"][start]
    denom = n * sxx - sx * sx
    if denom == 0:
        return float("nan"), float("nan"), float("nan")
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    # SSE
    sse = (
        syy
        + slope * slope * sxx
        + n * intercept * intercept
        + 2 * slope * intercept * sx
        - 2 * slope * sxy
        - 2 * intercept * sy
    )
    return float(slope), float(intercept), float(sse)


def _two_break_piecewise(values: np.ndarray, step_minutes: int = 5, min_seg: int = 4) -> Dict[str, float]:
    n = len(values)
    if n < (min_seg * 3):
        return {
            "slope_1": float("nan"),
            "slope_2": float("nan"),
            "slope_3": float("nan"),
            "break1_min_before_cutoff": float("nan"),
            "break2_min_before_cutoff": float("nan"),
            "end_drop": float("nan"),
        }
    series = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    if not np.isfinite(series).any():
        series = np.zeros_like(series, dtype=float)
    x = np.arange(n, dtype=float) * (step_minutes / 60.0)
    prefix = {
        "sx": np.concatenate([[0.0], np.cumsum(x)]),
        "sy": np.concatenate([[0.0], np.cumsum(series)]),
        "sxx": np.concatenate([[0.0], np.cumsum(x * x)]),
        "sxy": np.concatenate([[0.0], np.cumsum(x * series)]),
        "syy": np.concatenate([[0.0], np.cumsum(series * series)]),
    }

    best = {
        "sse": float("inf"),
        "i": None,
        "j": None,
        "slope1": float("nan"),
        "int1": float("nan"),
        "slope2": float("nan"),
        "int2": float("nan"),
        "slope3": float("nan"),
        "int3": float("nan"),
    }

    for i in range(min_seg, n - 2 * min_seg + 1):
        for j in range(i + min_seg, n - min_seg + 1):
            slope1, int1, sse1 = _segment_fit(prefix, 0, i)
            slope2, int2, sse2 = _segment_fit(prefix, i, j)
            slope3, int3, sse3 = _segment_fit(prefix, j, n)
            if not np.isfinite(sse1 + sse2 + sse3):
                continue
            sse = sse1 + sse2 + sse3
            if sse < best["sse"]:
                best.update({
                    "sse": sse,
                    "i": i,
                    "j": j,
                    "slope1": slope1,
                    "int1": int1,
                    "slope2": slope2,
                    "int2": int2,
                    "slope3": slope3,
                    "int3": int3,
                })

    if best["i"] is None or best["j"] is None:
        return {
            "slope_1": float("nan"),
            "slope_2": float("nan"),
            "slope_3": float("nan"),
            "break1_min_before_cutoff": float("nan"),
            "break2_min_before_cutoff": float("nan"),
            "end_drop": float("nan"),
        }

    i = best["i"]
    j = best["j"]
    break1_min_before = (n - 1 - i) * step_minutes
    break2_min_before = (n - 1 - j) * step_minutes
    y_start = best["slope3"] * x[j] + best["int3"]
    y_end = best["slope3"] * x[-1] + best["int3"]
    end_drop = y_start - y_end

    return {
        "slope_1": best["slope1"],
        "slope_2": best["slope2"],
        "slope_3": best["slope3"],
        "break1_min_before_cutoff": float(break1_min_before),
        "break2_min_before_cutoff": float(break2_min_before),
        "end_drop": float(end_drop),
    }


def _compute_metrics(y_true: np.ndarray, p_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    y_hat = (p_pred >= threshold).astype(int)
    acc = accuracy_score(y_true, y_hat)
    bal = balanced_accuracy_score(y_true, y_hat)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", pos_label=1)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    try:
        auc = roc_auc_score(y_true, p_pred)
    except ValueError:
        auc = float("nan")
    try:
        brier = brier_score_loss(y_true, p_pred)
    except ValueError:
        brier = float("nan")
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "yes_precision": float(prec),
        "yes_recall": float(rec),
        "yes_f1": float(f1),
        "roc_auc": float(auc),
        "brier": float(brier),
        "cm_tn": int(cm[0, 0]),
        "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]),
        "cm_tp": int(cm[1, 1]),
    }


def _select_thresholds(y_true: np.ndarray, p_pred: np.ndarray) -> Tuple[float, float, float]:
    best_t = 0.5
    best_acc = -1.0
    best_t_bal = 0.5
    best_bal = -1.0
    best_t_profit = 0.5
    best_profit = -1e9
    for t in np.linspace(0.05, 0.95, 91):
        y_hat = p_pred >= t
        acc = accuracy_score(y_true, y_hat)
        bal = balanced_accuracy_score(y_true, y_hat)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
        if bal > best_bal:
            best_bal = bal
            best_t_bal = float(t)
        cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
        tp = int(cm[1, 1])
        fp = int(cm[0, 1])
        profit = tp - fp
        if profit > best_profit:
            best_profit = profit
            best_t_profit = float(t)
    return best_t, best_t_bal, best_t_profit


def _net_units_per_100(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    tp = int(cm[1, 1])
    fp = int(cm[0, 1])
    n = len(y_true)
    if n == 0:
        return float("nan")
    return (tp - fp) / n * 100.0

def _compute_cache_hash(minute_dir: Path, mos_version: str) -> str:
    meta = {
        "version": VERSION,
        "minute_files": [],
        "mos_version": mos_version,
        "mos_variables": MOS_VARIABLES,
        "mos_models": MOS_MODELS,
        "start_date": str(START_DATE),
        "end_date": str(END_DATE),
    }
    for path in sorted(minute_dir.glob("MIA_tmpf_1min_UTC_*.csv")):
        stat = path.stat()
        meta["minute_files"].append(
            {"name": path.name, "size": stat.st_size, "mtime": int(stat.st_mtime)}
        )
    return _sha256_str(json.dumps(meta, sort_keys=True))


def _read_cache(path: Path, meta_path: Path, expected_hash: str, reuse_cache: bool) -> pd.DataFrame | None:
    if not reuse_cache:
        return None
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if meta.get("hash") != expected_hash:
        return None
    return pd.read_parquet(path)


def _write_cache(path: Path, meta_path: Path, df: pd.DataFrame, hash_value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    meta = {
        "hash": hash_value,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "rows": int(len(df)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_minute_series(minute_dir: Path) -> pd.DataFrame:
    files = sorted(minute_dir.glob("MIA_tmpf_1min_UTC_*.csv"))
    if not files:
        raise FileNotFoundError(f"No minute files found in {minute_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path, usecols=["valid(UTC)", "tmpf"], dtype={"tmpf": "string"})
        df["ts_utc"] = pd.to_datetime(df["valid(UTC)"], utc=True, errors="coerce")
        df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
        df = df.dropna(subset=["ts_utc", "tmpf"])
        frames.append(df[["ts_utc", "tmpf"]])
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("ts_utc")
    all_df = all_df.drop_duplicates(subset=["ts_utc"], keep="last")
    return all_df


def _build_minute_features(minute_dir: Path, cache_dir: Path, reuse_cache: bool, rebuild_cache: bool, mos_version: str) -> pd.DataFrame:
    cache_path = cache_dir / "hit1830_minute_features.parquet"
    meta_path = cache_dir / "hit1830_minute_features.meta.json"
    cache_hash = _compute_cache_hash(minute_dir, mos_version)
    if not rebuild_cache:
        cached = _read_cache(cache_path, meta_path, cache_hash, reuse_cache)
        if cached is not None:
            return cached

    day_windows = _build_day_windows()
    df_1m = _load_minute_series(minute_dir)
    df_1m = df_1m.sort_values("ts_utc")
    df_1m = df_1m.set_index("ts_utc")

    series_5m = df_1m["tmpf"].resample("5min").median()

    dct_cos = _dct_matrix(144, 20)
    dct_cos6 = _dct_matrix(72, 21)

    rows = []
    minute_violations = 0
    for _, dw in day_windows.iterrows():
        day = dw["target_date_local"]
        day_start = dw["day_start_utc"]
        day_end = dw["day_end_utc"]
        cutoff = dw["cutoff_utc"]

        full_idx = pd.date_range(day_start, day_end - timedelta(minutes=5), freq="5min")
        full_series = series_5m.reindex(full_idx)
        if full_series.notna().sum() == 0:
            continue

        tmax_full = float(np.nanmax(full_series.to_numpy()))
        tmin_full = float(np.nanmin(full_series.to_numpy()))
        range_full = tmax_full - tmin_full if np.isfinite(tmax_full) and np.isfinite(tmin_full) else float("nan")

        partial_end = min(cutoff, day_end) - timedelta(minutes=5)
        partial_idx = pd.date_range(day_start, partial_end, freq="5min") if partial_end >= day_start else []
        if len(partial_idx) == 0:
            continue
        partial_series = series_5m.reindex(partial_idx)
        if partial_series.notna().sum() == 0:
            continue

        tmax_sofar = float(np.nanmax(partial_series.to_numpy()))

        max_time_full_utc = full_series[full_series == tmax_full].index.min()
        tmax_time_local_minute = float("nan")
        if pd.notna(max_time_full_utc):
            max_time_local = max_time_full_utc.tz_convert(LOCAL_TZ)
            tmax_time_local_minute = max_time_local.hour * 60 + max_time_local.minute

        max_time_partial_utc = partial_series[partial_series == tmax_sofar].index.min()
        minutes_since_max = float("nan")
        tmax_sofar_time_local_minute = float("nan")
        if pd.notna(max_time_partial_utc):
            minutes_since_max = (cutoff - max_time_partial_utc).total_seconds() / 60.0
            max_time_partial_local = max_time_partial_utc.tz_convert(LOCAL_TZ)
            tmax_sofar_time_local_minute = max_time_partial_local.hour * 60 + max_time_partial_local.minute

        temp_now = float(np.nanmedian(partial_series.to_numpy()[-2:])) if partial_series.size >= 2 else float("nan")
        drop_from_max = tmax_sofar - temp_now if np.isfinite(tmax_sofar) and np.isfinite(temp_now) else float("nan")

        # W12 and W3 windows ending at cutoff
        w12_end = cutoff - timedelta(minutes=5)
        w12_start = cutoff - timedelta(hours=12)
        w12_idx = pd.date_range(w12_start, w12_end, freq="5min")
        if len(w12_idx) and w12_idx.max() > cutoff:
            minute_violations += 1
        w12 = series_5m.reindex(w12_idx).to_numpy()

        w3_end = cutoff - timedelta(minutes=5)
        w3_start = cutoff - timedelta(hours=3)
        w3_idx = pd.date_range(w3_start, w3_end, freq="5min")
        if len(w3_idx) and w3_idx.max() > cutoff:
            minute_violations += 1
        w3 = series_5m.reindex(w3_idx).to_numpy()

        w6_end = cutoff - timedelta(minutes=5)
        w6_start = cutoff - timedelta(hours=6)
        w6_idx = pd.date_range(w6_start, w6_end, freq="5min")
        if len(w6_idx) and w6_idx.max() > cutoff:
            minute_violations += 1
        w6 = series_5m.reindex(w6_idx).to_numpy()

        # Slopes
        def slope_last(points: int, values: np.ndarray) -> float:
            if len(values) < points:
                return float("nan")
            y = values[-points:]
            mins = np.arange(points) * 5
            return _ols_slope(mins, y)

        slope_15 = slope_last(3, w12)
        slope_30 = slope_last(6, w12)
        slope_60 = slope_last(12, w12)
        slope_120 = slope_last(24, w12)
        accel = slope_30 - slope_120 if np.isfinite(slope_30) and np.isfinite(slope_120) else float("nan")

        slope_sign_changes_last180 = _count_sign_changes(w3)

        # Change-point detection on W3 (single break)
        w3_interp = pd.Series(w3).interpolate(limit_direction="both").to_numpy()
        best_sse = float("inf")
        best_split = None
        best_before = float("nan")
        best_after = float("nan")
        best_drop = float("nan")
        for s in range(8, 29):
            left = w3_interp[: s + 1]
            right = w3_interp[s:]
            mins_left = np.arange(len(left)) * 5
            mins_right = np.arange(len(right)) * 5
            slope_left = _ols_slope(mins_left, left)
            slope_right = _ols_slope(mins_right, right)
            if not np.isfinite(slope_left) or not np.isfinite(slope_right):
                continue
            intercept_left = np.nanmean(left) - slope_left * np.nanmean(mins_left / 60.0)
            intercept_right = np.nanmean(right) - slope_right * np.nanmean(mins_right / 60.0)
            pred_left = slope_left * (mins_left / 60.0) + intercept_left
            pred_right = slope_right * (mins_right / 60.0) + intercept_right
            sse = np.nansum((left - pred_left) ** 2) + np.nansum((right - pred_right) ** 2)
            if sse < best_sse:
                best_sse = sse
                best_split = s
                best_before = slope_left
                best_after = slope_right
                best_drop = pred_left[-1] - pred_right[0]

        cp_minute = float("nan")
        if best_split is not None:
            cp_minute = (len(w3_interp) - 1 - best_split) * 5

        # Two-break piecewise on W3
        two_break = _two_break_piecewise(w3_interp, step_minutes=5, min_seg=4)

        # Volatility / outflow
        std_30 = float(np.nanstd(w12[-6:])) if len(w12) >= 6 else float("nan")
        std_60 = float(np.nanstd(w12[-12:])) if len(w12) >= 12 else float("nan")
        std_180 = float(np.nanstd(w12[-36:])) if len(w12) >= 36 else float("nan")

        mean_abs_delta_60 = float("nan")
        if len(w12) >= 12:
            diffs = np.diff(w12[-12:])
            mean_abs_delta_60 = float(np.nanmean(np.abs(diffs)))

        max_drop_30_last6h = float("nan")
        max_drop_60_last6h = float("nan")
        drop_cnt_30_ge0p5 = float("nan")
        drop_cnt_30_ge1 = float("nan")
        drop_cnt_30_ge2 = float("nan")
        last6h = w12[-72:] if len(w12) >= 72 else w12
        if len(last6h) >= 13:
            drops30 = last6h[:-6] - last6h[6:]
            drops60 = last6h[:-12] - last6h[12:]
            max_drop_30_last6h = float(np.nanmax(drops30))
            max_drop_60_last6h = float(np.nanmax(drops60))
            drop_cnt_30_ge0p5 = float(np.sum(drops30 >= 0.5))
            drop_cnt_30_ge1 = float(np.sum(drops30 >= 1.0))
            drop_cnt_30_ge2 = float(np.sum(drops30 >= 2.0))

        plateau_frac_0p2 = float("nan")
        plateau_longest_run_0p2 = float("nan")
        if len(w12) >= 24 and np.isfinite(tmax_sofar):
            last120 = w12[-24:]
            mask_0p2 = last120 >= (np.nanmax(w12) - 0.2)
            plateau_frac_0p2 = float(np.nanmean(mask_0p2))
            plateau_longest_run_0p2 = float(_longest_run(mask_0p2) * 5)

        # DCT (12h)
        w12_filled = pd.Series(w12).interpolate(limit_direction="both").to_numpy()
        if not np.isfinite(w12_filled).any():
            w12_filled[:] = 0.0
        centered = w12_filled - np.nanmedian(w12_filled)
        centered = np.where(np.isfinite(centered), centered, 0.0)
        coeff = centered @ dct_cos
        energy_total = float(np.sum(coeff ** 2)) if coeff.size else float("nan")
        dct_hi_energy = float(np.sum(coeff[10:] ** 2) / energy_total) if energy_total else float("nan")

        # DCT (6h) for kNN shape
        w6_filled = pd.Series(w6).interpolate(limit_direction="both").to_numpy()
        if not np.isfinite(w6_filled).any():
            w6_filled[:] = 0.0
        centered6 = w6_filled - np.nanmedian(w6_filled)
        centered6 = np.where(np.isfinite(centered6), centered6, 0.0)
        coeff6 = centered6 @ dct_cos6

        # Lag features (filled later)
        rows.append(
            {
                "target_date_local": day,
                "cutoff_utc": cutoff,
                "tmax_full": tmax_full,
                "tmax_sofar": tmax_sofar,
                "y_hit_by_cutoff": int(tmax_sofar >= (tmax_full - EPS)),
                "temp_now": temp_now,
                "max_sofar": tmax_sofar,
                "minutes_since_max": minutes_since_max,
                "drop_from_max": drop_from_max,
                "tmax_sofar_time_local_minute": tmax_sofar_time_local_minute,
                "slope_15m": slope_15,
                "slope_30m": slope_30,
                "slope_60m": slope_60,
                "slope_120m": slope_120,
                "accel": accel,
                "slope_sign_changes_last180": slope_sign_changes_last180,
                "cp_minute": cp_minute,
                "cp_slope_before": best_before,
                "cp_slope_after": best_after,
                "cp_drop": best_drop,
                "tp_slope_1": two_break["slope_1"],
                "tp_slope_2": two_break["slope_2"],
                "tp_slope_3": two_break["slope_3"],
                "tp_break1_min_before_cutoff": two_break["break1_min_before_cutoff"],
                "tp_break2_min_before_cutoff": two_break["break2_min_before_cutoff"],
                "tp_end_drop": two_break["end_drop"],
                "std_30m": std_30,
                "std_60m": std_60,
                "std_180m": std_180,
                "mean_abs_delta_60m": mean_abs_delta_60,
                "max_drop_30m_last6h": max_drop_30_last6h,
                "max_drop_60m_last6h": max_drop_60_last6h,
                "drop_cnt_30m_ge0p5_last6h": drop_cnt_30_ge0p5,
                "drop_cnt_30m_ge1_last6h": drop_cnt_30_ge1,
                "drop_cnt_30m_ge2_last6h": drop_cnt_30_ge2,
                "plateau_frac_0p2_last120": plateau_frac_0p2,
                "plateau_longest_run_0p2_last120": plateau_longest_run_0p2,
                "dct_hi_energy": dct_hi_energy,
                "range_full": range_full,
                "tmax_time_local_minute": tmax_time_local_minute,
            }
        )

        for i in range(20):
            rows[-1][f"dct_{i}"] = float(coeff[i])

        for i in range(21):
            rows[-1][f"dct6_{i}"] = float(coeff6[i])

    minute_df = pd.DataFrame(rows).sort_values("target_date_local")

    # Lag features
    minute_df["tmax_time_local_lag1"] = minute_df["tmax_time_local_minute"].shift(1)
    minute_df["tmax_time_local_lag2"] = minute_df["tmax_time_local_minute"].shift(2)
    minute_df["range_lag1"] = minute_df["range_full"].shift(1)
    minute_df["range_lag2"] = minute_df["range_full"].shift(2)

    # Outflow drop count on full day (30m drops >=2)
    outflow_counts = []
    for _, dw in day_windows.iterrows():
        day = dw["target_date_local"]
        day_start = dw["day_start_utc"]
        day_end = dw["day_end_utc"]
        full_idx = pd.date_range(day_start, day_end - timedelta(minutes=5), freq="5min")
        full_series = series_5m.reindex(full_idx).to_numpy()
        if len(full_series) < 7:
            outflow_counts.append(float("nan"))
            continue
        drops = full_series[:-6] - full_series[6:]
        outflow_counts.append(float(np.nansum(drops >= 2.0)))

    outflow_df = pd.DataFrame({
        "target_date_local": day_windows["target_date_local"],
        "outflow_drop_cnt": outflow_counts,
    })

    minute_df = minute_df.merge(outflow_df, on="target_date_local", how="left")
    minute_df["outflow_drop_cnt_lag1"] = minute_df["outflow_drop_cnt"].shift(1)
    minute_df["delta_range_1d"] = minute_df["range_lag1"] - minute_df["range_lag2"]

    if minute_violations > 0:
        raise RuntimeError(f"Minute leakage audit failed: {minute_violations} violations")

    _write_cache(cache_path, meta_path, minute_df, cache_hash)
    return minute_df

def _mos_version_stamp(engine) -> str:
    query = text(
        f"""
        SELECT MAX(id) AS max_id, MAX(retrieved_at_utc) AS max_retrieved
        FROM mos_daily_value
        WHERE station_id='KMIA'
          AND model IN ('GFS','NAM')
          AND target_date_local BETWEEN '2002-01-01' AND '2026-12-31'
          AND variable_code IN ({MOS_VARIABLES_SQL})
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query).mappings().first()
    if not row:
        return "none"
    return f"{row['max_id']}|{row['max_retrieved']}"


def _load_mos_raw(cache_dir: Path, reuse_cache: bool, rebuild_cache: bool, engine) -> pd.DataFrame:
    cache_path = cache_dir / "mos_raw_kmia.parquet"
    meta_path = cache_dir / "mos_raw_kmia.meta.json"
    mos_stamp = _mos_version_stamp(engine)
    cache_hash = _sha256_str(json.dumps({"version": VERSION, "mos_stamp": mos_stamp, "vars": MOS_VARIABLES}, sort_keys=True))

    if not rebuild_cache:
        cached = _read_cache(cache_path, meta_path, cache_hash, reuse_cache)
        if cached is not None:
            return cached

    query = f"""
        SELECT target_date_local, model, variable_code, asof_utc, runtime_utc, retrieved_at_utc, id,
               value_min, value_max, value_mean, value_median, sample_count
        FROM mos_daily_value
        WHERE station_id='KMIA'
          AND model IN ('GFS','NAM')
          AND target_date_local BETWEEN '2002-01-01' AND '2026-12-31'
          AND variable_code IN ({MOS_VARIABLES_SQL})
    """

    frames = []
    with engine.connect() as conn:
        for chunk in pd.read_sql_query(text(query), conn, chunksize=250000):
            frames.append(chunk)
    mos_raw = pd.concat(frames, ignore_index=True)

    mos_raw["asof_utc"] = pd.to_datetime(mos_raw["asof_utc"], utc=True, errors="coerce")
    mos_raw["runtime_utc"] = pd.to_datetime(mos_raw["runtime_utc"], utc=True, errors="coerce")
    mos_raw["retrieved_at_utc"] = pd.to_datetime(mos_raw["retrieved_at_utc"], utc=True, errors="coerce")

    _write_cache(cache_path, meta_path, mos_raw, cache_hash)
    return mos_raw


def _select_latest_mos(mos_raw: pd.DataFrame, cutoff_map: Dict[date, datetime]) -> pd.DataFrame:
    mos = mos_raw.copy()
    mos["target_date_local"] = pd.to_datetime(mos["target_date_local"]).dt.date
    mos["cutoff_utc"] = mos["target_date_local"].map(cutoff_map)
    mos = mos[mos["cutoff_utc"].notna()].copy()
    mos = mos[mos["asof_utc"] <= mos["cutoff_utc"]].copy()
    if (mos["asof_utc"] > mos["cutoff_utc"]).any():
        raise RuntimeError("MOS leakage audit failed: asof_utc beyond cutoff_utc detected")
    mos = mos.sort_values(["asof_utc", "runtime_utc", "retrieved_at_utc", "id"])
    latest = mos.groupby(["target_date_local", "model", "variable_code"], as_index=False).tail(1)
    return latest


def _circular_mean_deg(deg_values: List[float]) -> float:
    vals = [v for v in deg_values if np.isfinite(v)]
    if not vals:
        return float("nan")
    radians = np.deg2rad(vals)
    sin_mean = np.mean(np.sin(radians))
    cos_mean = np.mean(np.cos(radians))
    angle = math.atan2(sin_mean, cos_mean)
    deg = np.rad2deg(angle)
    if deg < 0:
        deg += 360.0
    return float(deg)


def _mos_base_features(latest: pd.DataFrame) -> pd.DataFrame:
    latest = latest.copy()
    latest["target_date_local"] = pd.to_datetime(latest["target_date_local"]).dt.date

    pv_max = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_max")
    pv_min = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_min")
    pv_mean = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_mean")

    def get_col(pv: pd.DataFrame, model: str, var: str) -> pd.Series:
        try:
            return pv[(model, var)]
        except KeyError:
            return pd.Series(index=pv.index, dtype=float)

    tmax_gfs = get_col(pv_max, "GFS", "n_x")
    tmax_nam = get_col(pv_max, "NAM", "n_x")
    tmin_gfs = get_col(pv_min, "GFS", "n_x")
    tmin_nam = get_col(pv_min, "NAM", "n_x")

    mos_tmax_mean = pd.concat([tmax_gfs, tmax_nam], axis=1).mean(axis=1)
    mos_tmin_mean = pd.concat([tmin_gfs, tmin_nam], axis=1).mean(axis=1)
    mos_range_mean = mos_tmax_mean - mos_tmin_mean

    mos_tmax_disagree = (tmax_gfs - tmax_nam).abs()
    range_gfs = tmax_gfs - tmin_gfs
    range_nam = tmax_nam - tmin_nam
    mos_range_disagree = (range_gfs - range_nam).abs()

    dpt_gfs = get_col(pv_mean, "GFS", "dpt")
    dpt_nam = get_col(pv_mean, "NAM", "dpt")
    dpt_mean = pd.concat([dpt_gfs, dpt_nam], axis=1).mean(axis=1)

    wsp_gfs = get_col(pv_mean, "GFS", "wsp")
    wsp_nam = get_col(pv_mean, "NAM", "wsp")
    wsp_mean = pd.concat([wsp_gfs, wsp_nam], axis=1).mean(axis=1)

    wdr_gfs = get_col(pv_mean, "GFS", "wdr")
    wdr_nam = get_col(pv_mean, "NAM", "wdr")
    wdr_mean = pd.Series(index=pv_mean.index, dtype=float)
    for idx in wdr_mean.index:
        wdr_mean.loc[idx] = _circular_mean_deg([wdr_gfs.get(idx, np.nan), wdr_nam.get(idx, np.nan)])

    # u,v components (meteorological convention)
    wdr_rad = np.deg2rad(wdr_mean)
    u_mean = -wsp_mean * np.sin(wdr_rad)
    v_mean = -wsp_mean * np.cos(wdr_rad)

    dd_models = mos_tmax_mean - dpt_mean

    p06_max = pd.concat([get_col(pv_max, "GFS", "p06"), get_col(pv_max, "NAM", "p06")], axis=1).max(axis=1)
    p12_max = pd.concat([get_col(pv_max, "GFS", "p12"), get_col(pv_max, "NAM", "p12")], axis=1).max(axis=1)
    q06_max = pd.concat([get_col(pv_max, "GFS", "q06"), get_col(pv_max, "NAM", "q06")], axis=1).max(axis=1)
    q12_max = pd.concat([get_col(pv_max, "GFS", "q12"), get_col(pv_max, "NAM", "q12")], axis=1).max(axis=1)

    t06_max = pd.concat([get_col(pv_max, "GFS", "t06"), get_col(pv_max, "NAM", "t06")], axis=1).max(axis=1)
    t12_max = pd.concat([get_col(pv_max, "GFS", "t12"), get_col(pv_max, "NAM", "t12")], axis=1).max(axis=1)

    t06_1_max = pd.concat([get_col(pv_max, "GFS", "t06_1"), get_col(pv_max, "NAM", "t06_1")], axis=1).max(axis=1)
    t06_2_max = pd.concat([get_col(pv_max, "GFS", "t06_2"), get_col(pv_max, "NAM", "t06_2")], axis=1).max(axis=1)
    t12_1_max = pd.concat([get_col(pv_max, "GFS", "t12_1"), get_col(pv_max, "NAM", "t12_1")], axis=1).max(axis=1)
    t12_2_max = pd.concat([get_col(pv_max, "GFS", "t12_2"), get_col(pv_max, "NAM", "t12_2")], axis=1).max(axis=1)

    cig_min = pd.concat([get_col(pv_min, "GFS", "cig"), get_col(pv_min, "NAM", "cig")], axis=1).min(axis=1)
    vis_min = pd.concat([get_col(pv_min, "GFS", "vis"), get_col(pv_min, "NAM", "vis")], axis=1).min(axis=1)

    # Time-block and temp channel features
    block_codes = ["tmp", "t06", "t06_1", "t06_2", "t12", "t12_1", "t12_2"]
    block_feats: Dict[str, pd.Series] = {}
    for code in block_codes:
        gfs_mean = get_col(pv_mean, "GFS", code)
        nam_mean = get_col(pv_mean, "NAM", code)
        gfs_max = get_col(pv_max, "GFS", code)
        nam_max = get_col(pv_max, "NAM", code)
        gfs_min = get_col(pv_min, "GFS", code)
        nam_min = get_col(pv_min, "NAM", code)
        block_feats[f"mos_{code}_gfs_mean"] = gfs_mean
        block_feats[f"mos_{code}_nam_mean"] = nam_mean
        block_feats[f"mos_{code}_gfs_max"] = gfs_max
        block_feats[f"mos_{code}_nam_max"] = nam_max
        block_feats[f"mos_{code}_gfs_min"] = gfs_min
        block_feats[f"mos_{code}_nam_min"] = nam_min
        block_feats[f"mos_{code}_mean_models"] = pd.concat([gfs_mean, nam_mean], axis=1).mean(axis=1)
        block_feats[f"mos_{code}_max_models"] = pd.concat([gfs_max, nam_max], axis=1).mean(axis=1)
        block_feats[f"mos_{code}_min_models"] = pd.concat([gfs_min, nam_min], axis=1).mean(axis=1)
        block_feats[f"mos_{code}_mean_disagree"] = (gfs_mean - nam_mean).abs()

    # Latest asof per model
    asof_latest = latest.groupby(["target_date_local", "model"], as_index=False)["asof_utc"].max()
    asof_pivot = asof_latest.pivot(index="target_date_local", columns="model", values="asof_utc")

    out = pd.DataFrame({
        "target_date_local": pv_max.index,
        "mos_tmax_gfs": tmax_gfs,
        "mos_tmax_nam": tmax_nam,
        "mos_tmin_gfs": tmin_gfs,
        "mos_tmin_nam": tmin_nam,
        "mos_tmax_mean": mos_tmax_mean,
        "mos_tmin_mean": mos_tmin_mean,
        "mos_range_mean": mos_range_mean,
        "mos_tmax_disagree": mos_tmax_disagree,
        "mos_range_disagree": mos_range_disagree,
        "dpt_mean_models": dpt_mean,
        "dd_models": dd_models,
        "wsp_mean_models": wsp_mean,
        "wdr_mean_models": wdr_mean,
        "u_mean": u_mean,
        "v_mean": v_mean,
        "p06_max_models": p06_max,
        "p12_max_models": p12_max,
        "q06_max_models": q06_max,
        "q12_max_models": q12_max,
        "t06_max_models": t06_max,
        "t12_max_models": t12_max,
        "t06_1_max": t06_1_max,
        "t06_2_max": t06_2_max,
        "t12_1_max": t12_1_max,
        "t12_2_max": t12_2_max,
        "cig_min_models": cig_min,
        "vis_min_models": vis_min,
        "mos_latest_asof_utc_gfs": asof_pivot.get("GFS"),
        "mos_latest_asof_utc_nam": asof_pivot.get("NAM"),
    }).reset_index(drop=True)

    for key, series in block_feats.items():
        out[key] = series.values

    return out


def _build_mos_features(cache_dir: Path, reuse_cache: bool, rebuild_cache: bool, engine) -> pd.DataFrame:
    cache_path = cache_dir / "hit1830_mos_features.parquet"
    meta_path = cache_dir / "hit1830_mos_features.meta.json"

    mos_stamp = _mos_version_stamp(engine)
    cache_hash = _sha256_str(json.dumps({"version": VERSION, "mos_stamp": mos_stamp, "vars": MOS_VARIABLES}, sort_keys=True))

    if not rebuild_cache:
        cached = _read_cache(cache_path, meta_path, cache_hash, reuse_cache)
        if cached is not None:
            return cached

    day_windows = _build_day_windows()
    cutoff_map_0 = {row.target_date_local: row.cutoff_utc for row in day_windows.itertuples()}
    cutoff_map_24 = {row.target_date_local: row.cutoff_utc - timedelta(hours=24) for row in day_windows.itertuples()}

    mos_raw = _load_mos_raw(cache_dir, reuse_cache, rebuild_cache, engine)

    latest_0 = _select_latest_mos(mos_raw, cutoff_map_0)
    latest_24 = _select_latest_mos(mos_raw, cutoff_map_24)

    base_0 = _mos_base_features(latest_0)
    base_24 = _mos_base_features(latest_24).add_prefix("b24_")

    mos = base_0.merge(base_24, left_on="target_date_local", right_on="b24_target_date_local", how="left")
    mos = mos.drop(columns=["b24_target_date_local"], errors="ignore")

    # MOS age hours
    mos["mos_latest_asof_utc_gfs"] = pd.to_datetime(mos["mos_latest_asof_utc_gfs"], utc=True, errors="coerce")
    mos["mos_latest_asof_utc_nam"] = pd.to_datetime(mos["mos_latest_asof_utc_nam"], utc=True, errors="coerce")
    mos["mos_age_hours_gfs"] = (
        pd.to_datetime(mos["target_date_local"].astype(str))
        .dt.tz_localize(LOCAL_TZ)
        .dt.tz_convert("UTC")
    )
    # Use cutoff map for age calc
    mos["cutoff_utc"] = mos["target_date_local"].map(cutoff_map_0)
    mos["mos_age_hours_gfs"] = (mos["cutoff_utc"] - mos["mos_latest_asof_utc_gfs"]).dt.total_seconds() / 3600.0
    mos["mos_age_hours_nam"] = (mos["cutoff_utc"] - mos["mos_latest_asof_utc_nam"]).dt.total_seconds() / 3600.0

    # Revision features
    rev_features = [
        "mos_tmax_mean",
        "mos_range_mean",
        "cig_min_models",
        "p12_max_models",
        "t12_max_models",
        "u_mean",
        "v_mean",
    ]

    revision_vol = None
    for feat in rev_features:
        b24 = mos.get(f"b24_{feat}")
        b0 = mos.get(feat)
        if b24 is None or b0 is None:
            continue
        mos[f"rev_0_24_{feat}"] = b0 - b24
        mos[f"abs_rev_0_24_{feat}"] = (b0 - b24).abs()
        if revision_vol is None:
            revision_vol = mos[f"abs_rev_0_24_{feat}"].copy()
        else:
            revision_vol = revision_vol + mos[f"abs_rev_0_24_{feat}"]

    if revision_vol is not None:
        mos["revision_vol_index"] = revision_vol

    mos = mos.drop(columns=["cutoff_utc"], errors="ignore")

    _write_cache(cache_path, meta_path, mos, cache_hash)
    return mos

def _merge_dataset(minute_df: pd.DataFrame, mos_df: pd.DataFrame, cache_dir: Path, reuse_cache: bool, rebuild_cache: bool, cache_hash: str) -> pd.DataFrame:
    cache_path = cache_dir / "hit1830_v2_features.parquet"
    meta_path = cache_dir / "hit1830_v2_features.meta.json"
    if not rebuild_cache:
        cached = _read_cache(cache_path, meta_path, cache_hash, reuse_cache)
        if cached is not None:
            return cached

    df = minute_df.merge(mos_df, on="target_date_local", how="left")

    # Calendar features
    dt = pd.to_datetime(df["target_date_local"])
    doy = dt.dt.dayofyear
    df["doy"] = doy
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = dt.dt.month

    # Remaining heating capacity features
    df["gap_to_mos_tmax"] = df["mos_tmax_mean"] - df["max_sofar"]
    df["gap_to_mos_tmax_gfs"] = df["mos_tmax_gfs"] - df["max_sofar"]
    df["gap_to_mos_tmax_nam"] = df["mos_tmax_nam"] - df["max_sofar"]
    df["gap_frac"] = df["gap_to_mos_tmax"] / df["mos_range_mean"].clip(lower=1.0)

    # MOS time-block mismatches vs obs temp now
    for code in ["tmp", "t06", "t06_1", "t06_2", "t12", "t12_1", "t12_2"]:
        mean_col = f"mos_{code}_mean_models"
        if mean_col in df.columns:
            df[f"mis_{code}_mean"] = df["temp_now"] - df[mean_col]

    # Delta future for climo/knn
    df["delta_future"] = df["tmax_full"] - df["tmax_sofar"]

    # Conditional climatology (train-only)
    train_mask, _, _ = _prepare_splits(df)
    df = _add_climo_features(df, train_mask)

    # kNN analog prior (train library only)
    df = _add_knn_features(df, train_mask, KNN_K)

    _write_cache(cache_path, meta_path, df, cache_hash)
    return df


def _add_climo_features(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    train = df.loc[train_mask].copy()

    climo_p_yes = train.groupby("doy")["y_hit_by_cutoff"].mean()
    climo_tmax = train.groupby("doy")["tmax_full"].median()
    climo_peak_time = train.groupby("doy")["tmax_time_local_minute"].median()
    climo_extra = train.groupby("doy")["delta_future"].median()

    overall_p_yes = train["y_hit_by_cutoff"].mean()
    overall_tmax = train["tmax_full"].median()
    overall_peak_time = train["tmax_time_local_minute"].median()
    overall_extra = train["delta_future"].median()

    df["prior_p_yes"] = df["doy"].map(climo_p_yes).fillna(overall_p_yes)
    df["climo_tmax_median_doy"] = df["doy"].map(climo_tmax).fillna(overall_tmax)
    df["climo_peak_time_median_doy"] = df["doy"].map(climo_peak_time).fillna(overall_peak_time)
    df["climo_extra_warm_after_cutoff_median_doy"] = df["doy"].map(climo_extra).fillna(overall_extra)

    df["max_sofar_minus_climo_tmax"] = df["max_sofar"] - df["climo_tmax_median_doy"]
    df["minutes_since_max_vs_climo_peak_time"] = df["tmax_sofar_time_local_minute"] - df["climo_peak_time_median_doy"]
    df["gap_to_climo_extra"] = df["climo_extra_warm_after_cutoff_median_doy"]
    return df


def _add_knn_features(df: pd.DataFrame, train_mask: np.ndarray, k: int) -> pd.DataFrame:
    df = df.copy()
    knn_cols = [f"dct6_{i}" for i in range(21)] + ["slope_60m", "drop_from_max", "plateau_frac_0p2_last120"]
    knn_cols = [c for c in knn_cols if c in df.columns]
    if not knn_cols:
        return df

    imputer = SimpleImputer(strategy="median")
    X_all = imputer.fit_transform(df[knn_cols])
    train_idx = np.where(train_mask)[0]
    if len(train_idx) < 10:
        return df
    X_train = X_all[train_idx]

    k_eff = min(k + 1, len(train_idx))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(X_train)
    _, indices = nn.kneighbors(X_all, n_neighbors=k_eff)
    neighbor_global = train_idx[indices]

    knn_p_yes = np.full(len(df), np.nan)
    knn_mean_delta = np.full(len(df), np.nan)
    knn_mean_minutes = np.full(len(df), np.nan)

    y = df["y_hit_by_cutoff"].to_numpy()
    delta_future = df["delta_future"].to_numpy()
    minutes_since = df["minutes_since_max"].to_numpy()

    for i in range(len(df)):
        neigh = neighbor_global[i].tolist()
        if train_mask[i]:
            neigh = [idx for idx in neigh if idx != i]
        neigh = neigh[:k]
        if not neigh:
            continue
        knn_p_yes[i] = float(np.nanmean(y[neigh]))
        knn_mean_delta[i] = float(np.nanmean(delta_future[neigh]))
        knn_mean_minutes[i] = float(np.nanmean(minutes_since[neigh]))

    df["knn_p_yes"] = knn_p_yes
    df["knn_mean_delta_future"] = knn_mean_delta
    df["knn_mean_minutes_since_max"] = knn_mean_minutes
    return df


def _fit_isotonic(y_val: np.ndarray, p_val: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    return iso


def _build_monotone_constraints(feature_cols: List[str]) -> List[int]:
    constraint_map = {
        "gap_to_mos_tmax": -1,
        "gap_to_mos_tmax_gfs": -1,
        "gap_to_mos_tmax_nam": -1,
        "gap_frac": -1,
        "minutes_since_max": 1,
        "drop_from_max": 1,
        "slope_60m": -1,
    }
    return [constraint_map.get(col, 0) for col in feature_cols]


def _train_lgbm_classifier(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, params: Dict) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _train_lgbm_regressor(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, params: Dict) -> lgb.Booster:
    params = params.copy()
    params["objective"] = "regression"
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _prepare_splits(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = pd.to_datetime(df["target_date_local"]).dt.year
    train_mask = years <= 2019
    val_mask = (years >= 2020) & (years <= 2022)
    test_mask = (years >= 2023) & (years <= 2025)
    return train_mask.to_numpy(), val_mask.to_numpy(), test_mask.to_numpy()


def _save_experiment(out_dir: Path, name: str, feature_list: List[str],
                     preds_val: pd.DataFrame, preds_test: pd.DataFrame,
                     metrics: Dict, model: lgb.Booster | None, extra_files: Dict | None = None) -> None:
    exp_dir = out_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    preds_val.to_parquet(exp_dir / "preds_val.parquet", index=False)
    preds_test.to_parquet(exp_dir / "preds_test.parquet", index=False)

    (exp_dir / "features.json").write_text(json.dumps(feature_list, indent=2), encoding="utf-8")

    if model is not None:
        model.save_model(str(exp_dir / "model.txt"))

    if extra_files:
        for fname, content in extra_files.items():
            (exp_dir / fname).write_text(content, encoding="utf-8")

    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _build_experiment_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    minute_base = [
        "temp_now",
        "max_sofar",
        "minutes_since_max",
        "drop_from_max",
        "tmax_sofar_time_local_minute",
        "slope_15m",
        "slope_30m",
        "slope_60m",
        "slope_120m",
        "accel",
        "slope_sign_changes_last180",
        "cp_minute",
        "cp_slope_before",
        "cp_slope_after",
        "cp_drop",
        "std_30m",
        "std_60m",
        "std_180m",
        "mean_abs_delta_60m",
        "max_drop_30m_last6h",
        "max_drop_60m_last6h",
        "drop_cnt_30m_ge0p5_last6h",
        "drop_cnt_30m_ge1_last6h",
        "drop_cnt_30m_ge2_last6h",
        "plateau_frac_0p2_last120",
        "plateau_longest_run_0p2_last120",
        "dct_hi_energy",
        "tmax_time_local_lag1",
        "tmax_time_local_lag2",
        "range_lag1",
        "range_lag2",
        "outflow_drop_cnt_lag1",
        "delta_range_1d",
    ]
    minute_base += [f"dct_{i}" for i in range(20)]

    peak_v2 = [
        "tp_slope_1",
        "tp_slope_2",
        "tp_slope_3",
        "tp_break1_min_before_cutoff",
        "tp_break2_min_before_cutoff",
        "tp_end_drop",
    ]

    convective_block = [
        "p06_max_models",
        "p12_max_models",
        "q06_max_models",
        "q12_max_models",
        "t06_max_models",
        "t12_max_models",
        "t06_1_max",
        "t06_2_max",
        "t12_1_max",
        "t12_2_max",
        "cig_min_models",
        "vis_min_models",
    ]

    heating_gap_base = [
        "mos_tmax_mean",
        "mos_tmin_mean",
        "mos_range_mean",
        "mos_tmax_disagree",
        "mos_range_disagree",
        "gap_to_mos_tmax",
        "gap_frac",
    ]
    heat_gap_extra = [
        "gap_to_mos_tmax_gfs",
        "gap_to_mos_tmax_nam",
    ]

    tblock_features = []
    for code in ["tmp", "t06", "t06_1", "t06_2", "t12", "t12_1", "t12_2"]:
        tblock_features += [
            f"mos_{code}_gfs_mean",
            f"mos_{code}_nam_mean",
            f"mos_{code}_gfs_max",
            f"mos_{code}_nam_max",
            f"mos_{code}_gfs_min",
            f"mos_{code}_nam_min",
            f"mos_{code}_mean_models",
            f"mos_{code}_max_models",
            f"mos_{code}_min_models",
            f"mos_{code}_mean_disagree",
            f"mis_{code}_mean",
        ]

    climo_features = [
        "prior_p_yes",
        "climo_tmax_median_doy",
        "climo_peak_time_median_doy",
        "climo_extra_warm_after_cutoff_median_doy",
        "max_sofar_minus_climo_tmax",
        "minutes_since_max_vs_climo_peak_time",
        "gap_to_climo_extra",
    ]

    knn_features = [
        "knn_p_yes",
        "knn_mean_delta_future",
        "knn_mean_minutes_since_max",
    ]

    revision_v2 = [c for c in df.columns if c.startswith("rev_0_24_") or c.startswith("abs_rev_0_24_")]
    if "revision_vol_index" in df.columns:
        revision_v2.append("revision_vol_index")
    if "b24_mos_tmax_disagree" in df.columns:
        revision_v2.append("b24_mos_tmax_disagree")

    calendar = ["doy_sin", "doy_cos", "month"]

    exp = {}
    exp["EXP11_BASELINE_REPRO"] = minute_base + calendar + heating_gap_base + convective_block
    exp["EXP12_ADD_HEAT_GAP"] = exp["EXP11_BASELINE_REPRO"] + heat_gap_extra
    exp["EXP13_ADD_MOS_TBLOCK_MISMATCH"] = exp["EXP12_ADD_HEAT_GAP"] + tblock_features
    exp["EXP14_ADD_PEAK_CONFIDENCE_V2"] = exp["EXP13_ADD_MOS_TBLOCK_MISMATCH"] + peak_v2
    exp["EXP15_ADD_CONDITIONAL_CLIMO"] = exp["EXP14_ADD_PEAK_CONFIDENCE_V2"] + climo_features
    exp["EXP16_ADD_ANALOG_KNN_PRIOR"] = exp["EXP15_ADD_CONDITIONAL_CLIMO"] + knn_features
    exp["EXP17_MONOTONIC_CONSTRAINTS"] = exp["EXP16_ADD_ANALOG_KNN_PRIOR"]
    exp["EXP18_SEASONAL_2MODEL"] = exp["EXP16_ADD_ANALOG_KNN_PRIOR"]
    exp["EXP19_REVISION_V2_MINIMAL"] = exp["EXP16_ADD_ANALOG_KNN_PRIOR"] + revision_v2
    exp["EXP20_PROPER_OOF_STACK"] = []  # handled separately

    # Ensure all features exist
    for key, cols in exp.items():
        exp[key] = list(dict.fromkeys([c for c in cols if c in df.columns]))

    return exp

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minute-dir", default=r"C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets\data\iem_minute_data\MIA\tmpf\UTC\yearly")
    parser.add_argument("--out-dir", default=r"C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets\artifacts\experiments\early_maxout_strategy\B3")
    parser.add_argument("--cache-dir", default=r"C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets\cache")
    parser.add_argument("--reuse_cache", type=int, default=1)
    parser.add_argument("--rebuild_cache", type=int, default=0)
    args = parser.parse_args()

    minute_dir = Path(args.minute_dir)
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    reports_dir = Path("reports")

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine("mysql+pymysql://root:root@localhost/weather_predictionmarkets")
    mos_stamp = _mos_version_stamp(engine)

    minute_df = _build_minute_features(minute_dir, cache_dir, bool(args.reuse_cache), bool(args.rebuild_cache), mos_stamp)
    mos_df = _build_mos_features(cache_dir, bool(args.reuse_cache), bool(args.rebuild_cache), engine)

    cache_hash = _compute_cache_hash(minute_dir, mos_stamp)
    merged_df = _merge_dataset(minute_df, mos_df, cache_dir, bool(args.reuse_cache), bool(args.rebuild_cache), cache_hash)

    # Split masks
    train_mask, val_mask, test_mask = _prepare_splits(merged_df)
    y = merged_df["y_hit_by_cutoff"].to_numpy()
    years = pd.to_datetime(merged_df["target_date_local"]).dt.year.to_numpy()

    # Baselines
    always_no_acc = float(accuracy_score(y[test_mask], np.zeros_like(y[test_mask])))
    always_yes_acc = float(accuracy_score(y[test_mask], np.ones_like(y[test_mask])))

    # LightGBM params
    scale_pos_weight = (len(y[train_mask]) - y[train_mask].sum()) / max(y[train_mask].sum(), 1)
    lgb_params = {
        "objective": "binary",
        "boosting": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "seed": 42,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    feature_sets = _build_experiment_features(merged_df)

    results = {}
    def run_classifier(exp_name: str, feature_cols: List[str], params_override: Dict | None = None,
                       monotone_constraints: List[int] | None = None,
                       seasonal: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        df = merged_df
        feature_cols = [c for c in feature_cols if not df.loc[train_mask, c].isna().all()]
        X = df[feature_cols]
        imputer = SimpleImputer(strategy="median")
        X_all = imputer.fit_transform(X)
        if monotone_constraints is not None:
            monotone_constraints = _build_monotone_constraints(feature_cols)

        if not seasonal:
            X_train = X_all[train_mask]
            X_val = X_all[val_mask]
            X_test = X_all[test_mask]
            params = lgb_params.copy()
            if params_override:
                params.update(params_override)
            if monotone_constraints:
                params["monotone_constraints"] = monotone_constraints

            model = _train_lgbm_classifier(X_train, y[train_mask], X_val, y[val_mask], params)
            p_val = model.predict(X_val)
            p_test = model.predict(X_test)

            exp_dir = out_dir / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(exp_dir / "model.txt"))
        else:
            params = lgb_params.copy()
            if params_override:
                params.update(params_override)
            if monotone_constraints:
                params["monotone_constraints"] = monotone_constraints

            months = df["month"].to_numpy()
            warm_mask = (months >= 5) & (months <= 10)
            cool_mask = ~warm_mask

            p_val_full = np.full(len(df), np.nan)
            p_test_full = np.full(len(df), np.nan)

            # Warm model
            warm_train = train_mask & warm_mask
            warm_val = val_mask & warm_mask
            warm_test = test_mask & warm_mask
            if warm_train.sum() >= 50:
                X_train_warm = X_all[warm_train]
                y_train_warm = y[warm_train]
                X_val_warm = X_all[warm_val] if warm_val.sum() >= 20 else X_all[val_mask]
                y_val_warm = y[warm_val] if warm_val.sum() >= 20 else y[val_mask]
                model_warm = _train_lgbm_classifier(X_train_warm, y_train_warm, X_val_warm, y_val_warm, params)
                if warm_val.sum() > 0:
                    p_val_full[warm_val] = model_warm.predict(X_all[warm_val])
                if warm_test.sum() > 0:
                    p_test_full[warm_test] = model_warm.predict(X_all[warm_test])
            else:
                model_warm = None

            # Cool model
            cool_train = train_mask & cool_mask
            cool_val = val_mask & cool_mask
            cool_test = test_mask & cool_mask
            if cool_train.sum() >= 50:
                X_train_cool = X_all[cool_train]
                y_train_cool = y[cool_train]
                X_val_cool = X_all[cool_val] if cool_val.sum() >= 20 else X_all[val_mask]
                y_val_cool = y[cool_val] if cool_val.sum() >= 20 else y[val_mask]
                model_cool = _train_lgbm_classifier(X_train_cool, y_train_cool, X_val_cool, y_val_cool, params)
                if cool_val.sum() > 0:
                    p_val_full[cool_val] = model_cool.predict(X_all[cool_val])
                if cool_test.sum() > 0:
                    p_test_full[cool_test] = model_cool.predict(X_all[cool_test])
            else:
                model_cool = None

            p_val = p_val_full[val_mask]
            p_test = p_test_full[test_mask]

            exp_dir = out_dir / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            if model_warm is not None:
                model_warm.save_model(str(exp_dir / "model_warm.txt"))
            if model_cool is not None:
                model_cool.save_model(str(exp_dir / "model_cool.txt"))

        if np.isnan(p_val).any():
            fill_val = float(np.nanmean(p_val)) if np.isfinite(np.nanmean(p_val)) else 0.5
            p_val = np.nan_to_num(p_val, nan=fill_val)
        if np.isnan(p_test).any():
            fill_test = float(np.nanmean(p_val)) if np.isfinite(np.nanmean(p_val)) else 0.5
            p_test = np.nan_to_num(p_test, nan=fill_test)

        iso = _fit_isotonic(y[val_mask], p_val)
        p_val_cal = iso.transform(p_val)
        p_test_cal = iso.transform(p_test)

        t_acc, t_bal, t_profit = _select_thresholds(y[val_mask], p_val_cal)
        y_val_profit = (p_val_cal >= t_profit).astype(int)
        y_test_profit = (p_test_cal >= t_profit).astype(int)
        net_val = _net_units_per_100(y[val_mask], y_val_profit)
        net_test = _net_units_per_100(y[test_mask], y_test_profit)

        metrics = {
            "val": _compute_metrics(y[val_mask], p_val_cal, t_acc),
            "test": _compute_metrics(y[test_mask], p_test_cal, t_acc),
            "threshold_acc": t_acc,
            "threshold_bal": t_bal,
            "threshold_profit": t_profit,
            "net_units_per_100_val_profit": net_val,
            "net_units_per_100_test_profit": net_test,
            "metrics_raw_val": _compute_metrics(y[val_mask], p_val, t_acc),
            "metrics_raw_test": _compute_metrics(y[test_mask], p_test, t_acc),
            "always_no_acc": always_no_acc,
            "always_yes_acc": always_yes_acc,
        }

        preds_val = df.loc[val_mask, ["target_date_local", "cutoff_utc", "y_hit_by_cutoff"]].copy()
        preds_val["p_raw"] = p_val
        preds_val["p_cal"] = p_val_cal
        preds_val["y_pred_acc"] = (p_val_cal >= t_acc).astype(int)
        preds_val["y_pred_profit"] = y_val_profit

        preds_test = df.loc[test_mask, ["target_date_local", "cutoff_utc", "y_hit_by_cutoff"]].copy()
        preds_test["p_raw"] = p_test
        preds_test["p_cal"] = p_test_cal
        preds_test["y_pred_acc"] = (p_test_cal >= t_acc).astype(int)
        preds_test["y_pred_profit"] = y_test_profit

        _save_experiment(out_dir, exp_name, feature_cols, preds_val, preds_test, metrics, None)
        return preds_val, preds_test, metrics

    # EXP11-19
    for exp_name in [
        "EXP11_BASELINE_REPRO",
        "EXP12_ADD_HEAT_GAP",
        "EXP13_ADD_MOS_TBLOCK_MISMATCH",
        "EXP14_ADD_PEAK_CONFIDENCE_V2",
        "EXP15_ADD_CONDITIONAL_CLIMO",
        "EXP16_ADD_ANALOG_KNN_PRIOR",
        "EXP19_REVISION_V2_MINIMAL",
    ]:
        preds_val, preds_test, metrics = run_classifier(exp_name, feature_sets[exp_name])
        results[exp_name] = metrics

    # EXP17 monotonic constraints
    exp_name = "EXP17_MONOTONIC_CONSTRAINTS"
    mono_cols = feature_sets[exp_name]
    monotone = _build_monotone_constraints(mono_cols)
    preds_val, preds_test, metrics = run_classifier(exp_name, mono_cols, monotone_constraints=monotone)
    results[exp_name] = metrics

    # EXP18 seasonal models
    exp_name = "EXP18_SEASONAL_2MODEL"
    preds_val, preds_test, metrics = run_classifier(exp_name, feature_sets[exp_name], seasonal=True)
    results[exp_name] = metrics

    # EXP20 proper OOF stack
    exp_name = "EXP20_PROPER_OOF_STACK"
    base_models = {
        "exp14": feature_sets["EXP14_ADD_PEAK_CONFIDENCE_V2"],
        "exp16": feature_sets["EXP16_ADD_ANALOG_KNN_PRIOR"],
        "exp19": feature_sets["EXP19_REVISION_V2_MINIMAL"],
    }

    folds = [(2002, 2009), (2010, 2014), (2015, 2019)]
    oof_preds = {}
    for key, feats in base_models.items():
        feats = [c for c in feats if not merged_df.loc[train_mask, c].isna().all()]
        oof = np.full(len(merged_df), np.nan)
        for start, end in folds:
            fold_mask = (years >= start) & (years <= end)
            train_fold = train_mask & (~fold_mask)
            val_fold = train_mask & fold_mask
            if val_fold.sum() == 0:
                continue
            X = merged_df[feats]
            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X[train_fold])
            X_val = imputer.transform(X[val_fold])
            model = _train_lgbm_classifier(X_train, y[train_fold], X_val, y[val_fold], lgb_params)
            oof[val_fold] = model.predict(X_val)
        oof_preds[key] = oof

    oof_df = pd.DataFrame({k: v for k, v in oof_preds.items()})
    meta_train_mask = train_mask & oof_df.notna().all(axis=1).to_numpy()
    meta_X_train = oof_df.loc[meta_train_mask].to_numpy()
    meta_y_train = y[meta_train_mask]

    meta_model = _train_lgbm_classifier(meta_X_train, meta_y_train, meta_X_train, meta_y_train, lgb_params)

    # Fit base models on full train to score val/test
    base_val = {}
    base_test = {}
    for key, feats in base_models.items():
        feats = [c for c in feats if not merged_df.loc[train_mask, c].isna().all()]
        X = merged_df[feats]
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X[train_mask])
        X_val = imputer.transform(X[val_mask])
        X_test = imputer.transform(X[test_mask])
        model = _train_lgbm_classifier(X_train, y[train_mask], X_val, y[val_mask], lgb_params)
        base_val[key] = model.predict(X_val)
        base_test[key] = model.predict(X_test)

    stack_val = pd.DataFrame(base_val)
    stack_test = pd.DataFrame(base_test)

    p_val = meta_model.predict(stack_val.to_numpy())
    p_test = meta_model.predict(stack_test.to_numpy())

    iso = _fit_isotonic(y[val_mask], p_val)
    p_val_cal = iso.transform(p_val)
    p_test_cal = iso.transform(p_test)

    t_acc, t_bal, t_profit = _select_thresholds(y[val_mask], p_val_cal)
    y_val_profit = (p_val_cal >= t_profit).astype(int)
    y_test_profit = (p_test_cal >= t_profit).astype(int)
    net_val = _net_units_per_100(y[val_mask], y_val_profit)
    net_test = _net_units_per_100(y[test_mask], y_test_profit)

    metrics = {
        "val": _compute_metrics(y[val_mask], p_val_cal, t_acc),
        "test": _compute_metrics(y[test_mask], p_test_cal, t_acc),
        "threshold_acc": t_acc,
        "threshold_bal": t_bal,
        "threshold_profit": t_profit,
        "net_units_per_100_val_profit": net_val,
        "net_units_per_100_test_profit": net_test,
        "metrics_raw_val": _compute_metrics(y[val_mask], p_val, t_acc),
        "metrics_raw_test": _compute_metrics(y[test_mask], p_test, t_acc),
        "always_no_acc": always_no_acc,
        "always_yes_acc": always_yes_acc,
    }

    preds_val = merged_df.loc[val_mask, ["target_date_local", "cutoff_utc", "y_hit_by_cutoff"]].copy()
    preds_val["p_raw"] = p_val
    preds_val["p_cal"] = p_val_cal
    preds_val["y_pred_acc"] = (p_val_cal >= t_acc).astype(int)
    preds_val["y_pred_profit"] = y_val_profit

    preds_test = merged_df.loc[test_mask, ["target_date_local", "cutoff_utc", "y_hit_by_cutoff"]].copy()
    preds_test["p_raw"] = p_test
    preds_test["p_cal"] = p_test_cal
    preds_test["y_pred_acc"] = (p_test_cal >= t_acc).astype(int)
    preds_test["y_pred_profit"] = y_test_profit

    exp_dir = out_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    meta_model.save_model(str(exp_dir / "model.txt"))
    _save_experiment(out_dir, exp_name, list(stack_val.columns), preds_val, preds_test, metrics, None)
    results[exp_name] = metrics

    # Summary report
    summary_json_path = reports_dir / "hit1830_v2_experiments_summary.json"
    summary_md_path = reports_dir / "hit1830_v2_experiments_report.md"

    summary_json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = []
    for name, metrics in results.items():
        val_m = metrics["val"]
        test_m = metrics["test"]
        rows.append(
            {
                "exp": name,
                "val_acc": val_m["accuracy"],
                "val_bal": val_m["balanced_accuracy"],
                "val_yes_recall": val_m["yes_recall"],
                "test_acc": test_m["accuracy"],
                "test_bal": test_m["balanced_accuracy"],
                "test_yes_recall": test_m["yes_recall"],
                "net_units_val": metrics.get("net_units_per_100_val_profit"),
                "net_units_test": metrics.get("net_units_per_100_test_profit"),
            }
        )

    best_val_acc = max(rows, key=lambda r: r["val_acc"])
    best_val_bal = max(rows, key=lambda r: r["val_bal"])

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Hit 18:30 Stockholm V2 Experiments Report\n\n")
        f.write(f"Always-NO test accuracy: {always_no_acc:.3f}\n\n")
        f.write(f"Always-YES test accuracy: {always_yes_acc:.3f}\n\n")
        f.write("| Experiment | Val Acc | Val Bal Acc | Val YES Recall | Test Acc | Test Bal Acc | Test YES Recall | NetUnits/100 (Val) | NetUnits/100 (Test) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['exp']} | {row['val_acc']:.3f} | {row['val_bal']:.3f} | {row['val_yes_recall']:.3f} | {row['test_acc']:.3f} | {row['test_bal']:.3f} | {row['test_yes_recall']:.3f} | {row['net_units_val']:.2f} | {row['net_units_test']:.2f} |\n"
            )

        f.write("\n")
        f.write(f"Best by Val Accuracy: {best_val_acc['exp']} (Val Acc {best_val_acc['val_acc']:.3f})\n\n")
        f.write(f"Best by Val Balanced Accuracy: {best_val_bal['exp']} (Val Bal Acc {best_val_bal['val_bal']:.3f})\n\n")

        f.write("Overfit diagnostics (val-test acc gap > 0.08):\n")
        for row in rows:
            if row["val_acc"] - row["test_acc"] > 0.08:
                f.write(f"- {row['exp']}: val {row['val_acc']:.3f} vs test {row['test_acc']:.3f}\n")

    # Also copy summaries into out_dir for convenience
    (out_dir / summary_json_path.name).write_text(summary_json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / summary_md_path.name).write_text(summary_md_path.read_text(encoding="utf-8"), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
