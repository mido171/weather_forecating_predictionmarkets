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
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from sqlalchemy import create_engine, text

VERSION = "hit1830_v3_suite_v1"

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


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    # vectorized normal CDF using erf
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))

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

        partial_vals = partial_series.to_numpy()
        tmax_sofar = float(np.nanmax(partial_vals))
        tmin_sofar = float(np.nanmin(partial_vals))
        range_sofar = tmax_sofar - tmin_sofar if np.isfinite(tmax_sofar) and np.isfinite(tmin_sofar) else float("nan")

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

        # W4 for change-point (last 4 hours)
        w4_end = cutoff - timedelta(minutes=5)
        w4_start = cutoff - timedelta(hours=4)
        w4_idx = pd.date_range(w4_start, w4_end, freq="5min")
        if len(w4_idx) and w4_idx.max() > cutoff:
            minute_violations += 1
        w4 = series_5m.reindex(w4_idx).to_numpy()

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

        # Change-point on W4 (single break)
        w4_interp = pd.Series(w4).interpolate(limit_direction="both").to_numpy()
        best4_sse = float("inf")
        best4_split = None
        best4_before = float("nan")
        best4_after = float("nan")
        best4_drop = float("nan")
        if len(w4_interp) >= 10:
            for s in range(8, len(w4_interp) - 2):
                left = w4_interp[: s + 1]
                right = w4_interp[s:]
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
                if sse < best4_sse:
                    best4_sse = sse
                    best4_split = s
                    best4_before = slope_left
                    best4_after = slope_right
                    best4_drop = pred_left[-1] - pred_right[0]

        cp4_minute = float("nan")
        if best4_split is not None:
            cp4_minute = (len(w4_interp) - 1 - best4_split) * 5

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

        # Drop features from local midnight -> cutoff (partial series)
        partial_filled = pd.Series(partial_vals).interpolate(limit_direction="both").to_numpy()
        max_drop_10 = float("nan")
        max_drop_30 = float("nan")
        drop_cnt_30_ge_1f = float("nan")
        drop_cnt_30_ge_2f = float("nan")
        drop_cnt_10_ge_0p5 = float("nan")
        drop_cnt_20_ge_0p5 = float("nan")
        drop_cnt_30_ge_0p5 = float("nan")
        drop_cnt_10_ge_1f = float("nan")
        drop_cnt_10_ge_2f = float("nan")
        drop_cnt_20_ge_1f = float("nan")
        drop_cnt_20_ge_2f = float("nan")
        if len(partial_filled) >= 7:
            drops10 = partial_filled[:-2] - partial_filled[2:]
            drops20 = partial_filled[:-4] - partial_filled[4:]
            drops30_full = partial_filled[:-6] - partial_filled[6:]
            max_drop_10 = float(np.nanmax(drops10))
            max_drop_30 = float(np.nanmax(drops30_full))
            drop_cnt_10_ge_0p5 = float(np.sum(drops10 >= 0.5))
            drop_cnt_20_ge_0p5 = float(np.sum(drops20 >= 0.5))
            drop_cnt_30_ge_0p5 = float(np.sum(drops30_full >= 0.5))
            drop_cnt_10_ge_1f = float(np.sum(drops10 >= 1.0))
            drop_cnt_10_ge_2f = float(np.sum(drops10 >= 2.0))
            drop_cnt_20_ge_1f = float(np.sum(drops20 >= 1.0))
            drop_cnt_20_ge_2f = float(np.sum(drops20 >= 2.0))
            drop_cnt_30_ge_1f = float(np.sum(drops30_full >= 1.0))
            drop_cnt_30_ge_2f = float(np.sum(drops30_full >= 2.0))

        # Cooling run length (consecutive negative deltas)
        cooling_run_len = float("nan")
        if len(partial_filled) >= 3:
            deltas = np.diff(partial_filled)
            longest = 0
            run = 0
            for d in deltas:
                if d < 0:
                    run += 1
                    if run > longest:
                        longest = run
                else:
                    run = 0
            cooling_run_len = float(longest * 5)

        plateau_frac_0p2 = float("nan")
        plateau_longest_run_0p2 = float("nan")
        if len(w12) >= 24 and np.isfinite(tmax_sofar):
            last120 = w12[-24:]
            mask_0p2 = last120 >= (np.nanmax(w12) - 0.2)
            plateau_frac_0p2 = float(np.nanmean(mask_0p2))
            plateau_longest_run_0p2 = float(_longest_run(mask_0p2) * 5)

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
                "tmin_sofar": tmin_sofar,
                "range_sofar": range_sofar,
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
                "cp4_minute": cp4_minute,
                "cp4_slope_before": best4_before,
                "cp4_slope_after": best4_after,
                "cp4_drop": best4_drop,
                "std_30m": std_30,
                "std_60": std_60,
                "std_180": std_180,
                "mean_abs_delta_60m": mean_abs_delta_60,
                "max_drop_30m_last6h": max_drop_30_last6h,
                "max_drop_60m_last6h": max_drop_60_last6h,
                "drop_cnt_30m_ge0p5_last6h": drop_cnt_30_ge0p5,
                "drop_cnt_30m_ge1_last6h": drop_cnt_30_ge1,
                "drop_cnt_30m_ge2_last6h": drop_cnt_30_ge2,
                "max_drop_10": max_drop_10,
                "max_drop_30": max_drop_30,
                "drop_cnt_30_ge_1F": drop_cnt_30_ge_1f,
                "drop_cnt_30_ge_2F": drop_cnt_30_ge_2f,
                "drop_cnt_10_ge_0p5": drop_cnt_10_ge_0p5,
                "drop_cnt_20_ge_0p5": drop_cnt_20_ge_0p5,
                "drop_cnt_30_ge_0p5": drop_cnt_30_ge_0p5,
                "drop_cnt_10_ge_1F": drop_cnt_10_ge_1f,
                "drop_cnt_10_ge_2F": drop_cnt_10_ge_2f,
                "drop_cnt_20_ge_1F": drop_cnt_20_ge_1f,
                "drop_cnt_20_ge_2F": drop_cnt_20_ge_2f,
                "cooling_run_len": cooling_run_len,
                "plateau_frac_0p2_last120": plateau_frac_0p2,
                "plateau_longest_run_0p2_last120": plateau_longest_run_0p2,
                "range_full": range_full,
                "tmax_time_local_minute": tmax_time_local_minute,
            }
        )

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

def _angular_diff_deg(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    diff = abs(a - b) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return float(diff)

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

    out = pd.DataFrame({"target_date_local": pv_max.index}).reset_index(drop=True)

    # n_x special: MOS daytime max (X) and nighttime min (N)
    x_gfs = get_col(pv_max, "GFS", "n_x")
    x_nam = get_col(pv_max, "NAM", "n_x")
    n_gfs = get_col(pv_min, "GFS", "n_x")
    n_nam = get_col(pv_min, "NAM", "n_x")
    mos_x_mean = pd.concat([x_gfs, x_nam], axis=1).mean(axis=1)
    mos_n_mean = pd.concat([n_gfs, n_nam], axis=1).mean(axis=1)
    mos_range = mos_x_mean - mos_n_mean
    mos_x_disagree = (x_gfs - x_nam).abs()
    mos_range_disagree = (x_gfs - n_gfs - (x_nam - n_nam)).abs()

    out["mos_x_gfs"] = x_gfs.values
    out["mos_x_nam"] = x_nam.values
    out["mos_n_gfs"] = n_gfs.values
    out["mos_n_nam"] = n_nam.values
    out["mos_x_mean"] = mos_x_mean.values
    out["mos_n_mean"] = mos_n_mean.values
    out["mos_range"] = mos_range.values
    out["mos_x_disagree"] = mos_x_disagree.values
    out["mos_range_disagree"] = mos_range_disagree.values

    # Core vars (tmp, dpt, wsp, wdr, cig, vis)
    core_vars = ["tmp", "dpt", "wsp", "wdr", "cig", "vis"]
    for var in core_vars:
        gfs_mean = get_col(pv_mean, "GFS", var)
        nam_mean = get_col(pv_mean, "NAM", var)
        gfs_max = get_col(pv_max, "GFS", var)
        nam_max = get_col(pv_max, "NAM", var)
        gfs_min = get_col(pv_min, "GFS", var)
        nam_min = get_col(pv_min, "NAM", var)

        mean_models = pd.concat([gfs_mean, nam_mean], axis=1).mean(axis=1)
        max_models = pd.concat([gfs_max, nam_max], axis=1).max(axis=1)
        min_models = pd.concat([gfs_min, nam_min], axis=1).min(axis=1)

        if var == "wdr":
            wdr_mean = pd.Series(index=mean_models.index, dtype=float)
            for idx in wdr_mean.index:
                wdr_mean.loc[idx] = _circular_mean_deg([gfs_mean.get(idx, np.nan), nam_mean.get(idx, np.nan)])
            out["mos_wdr_mean_models"] = wdr_mean.values
            out["mos_wdr_mean_disagree"] = [
                _angular_diff_deg(gfs_mean.get(idx, np.nan), nam_mean.get(idx, np.nan)) for idx in wdr_mean.index
            ]
        else:
            out[f"mos_{var}_mean_models"] = mean_models.values
            out[f"mos_{var}_mean_disagree"] = (gfs_mean - nam_mean).abs().values

        out[f"mos_{var}_gfs_mean"] = gfs_mean.values
        out[f"mos_{var}_nam_mean"] = nam_mean.values
        out[f"mos_{var}_gfs_max"] = gfs_max.values
        out[f"mos_{var}_nam_max"] = nam_max.values
        out[f"mos_{var}_gfs_min"] = gfs_min.values
        out[f"mos_{var}_nam_min"] = nam_min.values
        out[f"mos_{var}_max_models"] = max_models.values
        out[f"mos_{var}_min_models"] = min_models.values

    # Convective / suppression vars
    conv_vars = ["p06", "p12", "q06", "q12", "t06", "t06_1", "t06_2", "t12", "t12_1", "t12_2"]
    for var in conv_vars:
        gfs_mean = get_col(pv_mean, "GFS", var)
        nam_mean = get_col(pv_mean, "NAM", var)
        gfs_max = get_col(pv_max, "GFS", var)
        nam_max = get_col(pv_max, "NAM", var)
        gfs_min = get_col(pv_min, "GFS", var)
        nam_min = get_col(pv_min, "NAM", var)

        mean_models = pd.concat([gfs_mean, nam_mean], axis=1).mean(axis=1)
        max_models = pd.concat([gfs_max, nam_max], axis=1).max(axis=1)
        min_models = pd.concat([gfs_min, nam_min], axis=1).min(axis=1)

        out[f"mos_{var}_gfs_mean"] = gfs_mean.values
        out[f"mos_{var}_nam_mean"] = nam_mean.values
        out[f"mos_{var}_gfs_max"] = gfs_max.values
        out[f"mos_{var}_nam_max"] = nam_max.values
        out[f"mos_{var}_gfs_min"] = gfs_min.values
        out[f"mos_{var}_nam_min"] = nam_min.values
        out[f"mos_{var}_mean_models"] = mean_models.values
        out[f"mos_{var}_max_models"] = max_models.values
        out[f"mos_{var}_min_models"] = min_models.values
        out[f"mos_{var}_mean_disagree"] = (gfs_mean - nam_mean).abs().values
        out[f"mos_{var}_max_disagree"] = (gfs_max - nam_max).abs().values

    # Latest asof per model
    asof_latest = latest.groupby(["target_date_local", "model"], as_index=False)["asof_utc"].max()
    asof_pivot = asof_latest.pivot(index="target_date_local", columns="model", values="asof_utc")
    out["mos_latest_asof_utc_gfs"] = asof_pivot.get("GFS").values
    out["mos_latest_asof_utc_nam"] = asof_pivot.get("NAM").values

    # Wind components (meteorological convention)
    wdr_mean_models = out.get("mos_wdr_mean_models")
    wsp_mean_models = out.get("mos_wsp_mean_models")
    if wdr_mean_models is not None and wsp_mean_models is not None:
        wdr_rad = np.deg2rad(wdr_mean_models)
        out["mos_u_mean"] = -wsp_mean_models * np.sin(wdr_rad)
        out["mos_v_mean"] = -wsp_mean_models * np.cos(wdr_rad)

    return out


def _build_mos_features(cache_dir: Path, reuse_cache: bool, rebuild_cache: bool, engine) -> pd.DataFrame:
    cache_path = cache_dir / "hit1830_v3_mos_features.parquet"
    meta_path = cache_dir / "hit1830_v3_mos_features.meta.json"

    mos_stamp = _mos_version_stamp(engine)
    cache_hash = _sha256_str(json.dumps({"version": VERSION, "mos_stamp": mos_stamp, "vars": MOS_VARIABLES}, sort_keys=True))

    if not rebuild_cache:
        cached = _read_cache(cache_path, meta_path, cache_hash, reuse_cache)
        if cached is not None:
            return cached

    day_windows = _build_day_windows()
    cutoff_map_0 = {row.target_date_local: row.cutoff_utc for row in day_windows.itertuples()}

    mos_raw = _load_mos_raw(cache_dir, reuse_cache, rebuild_cache, engine)
    latest_0 = _select_latest_mos(mos_raw, cutoff_map_0)
    base_0 = _mos_base_features(latest_0)

    cycle_feats = _mos_cycle_revision_features(mos_raw, day_windows)
    mos = base_0.merge(cycle_feats, on="target_date_local", how="left")

    _write_cache(cache_path, meta_path, mos, cache_hash)
    return mos


def _mos_cycle_revision_features(mos_raw: pd.DataFrame, day_windows: pd.DataFrame) -> pd.DataFrame:
    # Build cutoff maps for cycle-aligned revisions
    def cycle_time_map(hours: int, day_offset: int = 0) -> Dict[date, datetime]:
        mapping = {}
        for row in day_windows.itertuples():
            day = row.target_date_local + timedelta(days=day_offset)
            cycle_time = datetime(day.year, day.month, day.day, hours, 0, tzinfo=timezone.utc)
            mapping[row.target_date_local] = cycle_time
        return mapping

    cutoff_map_12 = cycle_time_map(12, 0)
    cutoff_map_00 = cycle_time_map(0, 0)
    cutoff_map_prev12 = cycle_time_map(12, -1)

    latest_12 = _select_latest_mos(mos_raw, cutoff_map_12)
    latest_00 = _select_latest_mos(mos_raw, cutoff_map_00)
    latest_prev12 = _select_latest_mos(mos_raw, cutoff_map_prev12)

    rev_vars = {
        "n_x": "max",
        "cig": "min",
        "vis": "min",
        "p12": "max",
        "q12": "max",
        "t12": "max",
        "wsp": "mean",
        "wdr": "mean",
        "dpt": "mean",
    }

    def cycle_values(latest: pd.DataFrame) -> pd.DataFrame:
        pv_max = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_max")
        pv_min = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_min")
        pv_mean = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_mean")

        def get_col(pv: pd.DataFrame, model: str, var: str) -> pd.Series:
            try:
                return pv[(model, var)]
            except KeyError:
                return pd.Series(index=pv.index, dtype=float)

        out = pd.DataFrame({"target_date_local": pv_max.index}).reset_index(drop=True)
        for var, stat in rev_vars.items():
            if stat == "max":
                gfs = get_col(pv_max, "GFS", var)
                nam = get_col(pv_max, "NAM", var)
            elif stat == "min":
                gfs = get_col(pv_min, "GFS", var)
                nam = get_col(pv_min, "NAM", var)
            else:
                gfs = get_col(pv_mean, "GFS", var)
                nam = get_col(pv_mean, "NAM", var)
            if var == "wdr":
                vals = []
                for idx in gfs.index:
                    vals.append(_circular_mean_deg([gfs.get(idx, np.nan), nam.get(idx, np.nan)]))
                out[var] = vals
            else:
                out[var] = pd.concat([gfs, nam], axis=1).mean(axis=1).values
        return out

    vals_12 = cycle_values(latest_12)
    vals_00 = cycle_values(latest_00)
    vals_prev12 = cycle_values(latest_prev12)

    out = pd.DataFrame({"target_date_local": day_windows["target_date_local"]})
    out = out.merge(vals_12, on="target_date_local", how="left", suffixes=("", "_12"))
    out = out.merge(vals_00, on="target_date_local", how="left", suffixes=("", "_00"))
    out = out.merge(vals_prev12, on="target_date_local", how="left", suffixes=("", "_prev12"))

    for var in rev_vars.keys():
        if var in out.columns and f"{var}_00" in out.columns:
            out[f"rev_12z_minus_00z_{var}"] = out[var] - out[f"{var}_00"]
            out[f"abs_rev_12z_minus_00z_{var}"] = (out[var] - out[f"{var}_00"]).abs()
        if var in out.columns and f"{var}_prev12" in out.columns:
            out[f"rev_12z_minus_prev12_{var}"] = out[var] - out[f"{var}_prev12"]
            out[f"abs_rev_12z_minus_prev12_{var}"] = (out[var] - out[f"{var}_prev12"]).abs()

    # Drop raw cycle columns to keep only revisions
    drop_cols = []
    for var in rev_vars.keys():
        drop_cols.extend([var, f"{var}_00", f"{var}_prev12"])
    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")
    return out

def _merge_dataset(minute_df: pd.DataFrame, mos_df: pd.DataFrame, cache_dir: Path, reuse_cache: bool, rebuild_cache: bool, cache_hash: str) -> pd.DataFrame:
    cache_path = cache_dir / "hit1830_v3_features.parquet"
    meta_path = cache_dir / "hit1830_v3_features.meta.json"
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

    # Cutoff local minute for time-of-max regression
    cutoff_local = pd.to_datetime(df["cutoff_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
    df["cutoff_local_minute"] = cutoff_local.dt.hour * 60 + cutoff_local.dt.minute

    # Heat gap features
    df["gap_to_mos_x"] = df["mos_x_mean"] - df["max_sofar"]
    df["gap_frac"] = df["gap_to_mos_x"] / df["mos_range"].clip(lower=1.0)
    df["completion_frac"] = (df["max_sofar"] - df["tmin_sofar"]) / df["mos_range"].clip(lower=1.0)

    # MOS time-block mismatches vs obs temp now
    for code in ["tmp", "t06", "t06_1", "t06_2", "t12", "t12_1", "t12_2"]:
        mean_col = f"mos_{code}_mean_models"
        if mean_col in df.columns:
            df[f"mis_{code}_mean"] = df["temp_now"] - df[mean_col]

    # QPF category shaping (q12 bins)
    if "mos_q12_max_models" in df.columns:
        for thr in [1, 2, 3, 4, 5]:
            df[f"q12_ge_{thr}"] = (df["mos_q12_max_models"] >= thr).astype(float)

    # Delta future for regression targets
    df["delta_future"] = df["tmax_full"] - df["tmax_sofar"]

    # Add climo range and heating fraction (train-only)
    train_mask, _, _ = _prepare_splits(df)
    df = _add_climo_range_features(df, train_mask)

    # Bias-correct MOS X (past-only EWMA)
    df = _add_bias_features(df)

    # MOS probability calibration features (train-only)
    df = _add_calibrated_mos_probs(df, train_mask)

    # p_onshore gate
    df = _add_onshore_probability(df, train_mask)

    # Front/cold-advection flag
    df = _add_front_flag(df)

    _write_cache(cache_path, meta_path, df, cache_hash)
    return df


def _add_climo_range_features(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    train = df.loc[train_mask]
    range_climo = train.groupby("doy")["range_full"].median()
    overall = train["range_full"].median()
    df["range_climo_doy"] = df["doy"].map(range_climo).fillna(overall)
    df["heating_fraction_obs"] = df["range_sofar"] / df["range_climo_doy"].clip(lower=1.0)
    return df


def _add_bias_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("target_date_local")
    err = (df["tmax_full"] - df["mos_x_mean"]).shift(1)
    bias = err.ewm(span=45, adjust=False).mean()
    bias = bias.fillna(0.0)
    df["bias_x_ewma_45"] = bias
    df["mos_x_bc"] = df["mos_x_mean"] + df["bias_x_ewma_45"]
    df["gap_to_mos_x_bc"] = df["mos_x_bc"] - df["max_sofar"]
    return df


def _add_calibrated_mos_probs(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    prob_vars = [
        "mos_p06_max_models",
        "mos_p12_max_models",
        "mos_t06_max_models",
        "mos_t12_max_models",
        "mos_t06_1_max_models",
        "mos_t06_2_max_models",
        "mos_t12_1_max_models",
        "mos_t12_2_max_models",
    ]
    for var in prob_vars:
        if var not in df.columns:
            continue
        x_train = df.loc[train_mask, var]
        y_train = df.loc[train_mask, "y_hit_by_cutoff"]
        mask = x_train.notna()
        if mask.sum() < 50:
            df[f"{var}_cal"] = df[var]
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x_train[mask], y_train[mask])
        df[f"{var}_cal"] = iso.transform(df[var].fillna(x_train[mask].median()))
    return df


def _add_onshore_probability(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    if "mos_wdr_mean_models" not in df.columns or "mos_wsp_mean_models" not in df.columns:
        df["p_onshore"] = np.nan
        return df
    wdr = df["mos_wdr_mean_models"]
    wsp = df["mos_wsp_mean_models"]
    onshore_label = ((wdr >= 45) & (wdr <= 160) & (wsp >= 5)).astype(int)
    X = pd.DataFrame({
        "wdr_sin": np.sin(np.deg2rad(wdr)),
        "wdr_cos": np.cos(np.deg2rad(wdr)),
        "wsp": wsp,
        "doy_sin": df["doy_sin"],
        "doy_cos": df["doy_cos"],
    })
    imputer = SimpleImputer(strategy="median")
    X_all = imputer.fit_transform(X)
    X_train = X_all[train_mask]
    y_train = onshore_label[train_mask]
    if y_train.sum() < 50 or y_train.sum() == len(y_train):
        df["p_onshore"] = onshore_label.astype(float)
        return df
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    df["p_onshore"] = lr.predict_proba(X_all)[:, 1]
    return df


def _add_front_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "mos_wdr_mean_models" not in df.columns or "mos_wsp_mean_models" not in df.columns:
        df["front_flag"] = 0.0
        return df
    wdr = df["mos_wdr_mean_models"]
    wsp = df["mos_wsp_mean_models"]
    northerly = (wdr >= 315) | (wdr <= 45)
    strong_wind = wsp >= 10
    cooling = (df["slope_60m"] <= 0) | (df["drop_from_max"] >= 0.3)
    df["front_flag"] = (northerly & strong_wind & cooling).astype(float)
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
        "gap_to_mos_x_bc": -1,
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
    min_block_1 = [
        "temp_now",
        "max_sofar",
        "tmax_sofar_time_local_minute",
        "minutes_since_max",
        "drop_from_max",
        "slope_15m",
        "slope_30m",
        "slope_60m",
        "slope_120m",
        "std_60",
        "std_180",
        "max_drop_10",
        "max_drop_30",
        "drop_cnt_30_ge_1F",
        "drop_cnt_30_ge_2F",
    ]

    min_block_2 = [
        "tmin_sofar",
        "range_sofar",
        "heating_fraction_obs",
    ]

    min_block_3 = [
        "cp4_minute",
        "cp4_slope_before",
        "cp4_slope_after",
        "cp4_drop",
    ]

    min_outflow_extra = [
        "drop_cnt_10_ge_0p5",
        "drop_cnt_20_ge_0p5",
        "drop_cnt_30_ge_0p5",
        "drop_cnt_10_ge_1F",
        "drop_cnt_10_ge_2F",
        "drop_cnt_20_ge_1F",
        "drop_cnt_20_ge_2F",
        "cooling_run_len",
    ]

    mos_core = [
        "mos_x_mean",
        "mos_n_mean",
        "mos_range",
        "mos_x_disagree",
        "mos_range_disagree",
        "mos_tmp_mean_models",
        "mos_dpt_mean_models",
        "mos_wsp_mean_models",
        "mos_wdr_mean_models",
        "mos_cig_min_models",
        "mos_vis_min_models",
        "mos_tmp_mean_disagree",
        "mos_dpt_mean_disagree",
        "mos_wdr_mean_disagree",
    ]

    convective_base = [
        "mos_p06_max_models",
        "mos_p12_max_models",
        "mos_q06_max_models",
        "mos_q12_max_models",
    ]

    convective_thunder_base = [
        "mos_t06_max_models",
        "mos_t12_max_models",
    ]

    convective_thunder_variants = [
        "mos_t06_1_max_models",
        "mos_t06_2_max_models",
        "mos_t12_1_max_models",
        "mos_t12_2_max_models",
    ]

    heat_gap = [
        "gap_to_mos_x",
        "gap_frac",
        "completion_frac",
        "mos_range_disagree",
        "mos_x_disagree",
    ]

    bias_block = [
        "bias_x_ewma_45",
        "mos_x_bc",
        "gap_to_mos_x_bc",
    ]

    qpf_bins = [c for c in df.columns if c.startswith("q12_ge_")]

    revision_block = [c for c in df.columns if c.startswith("rev_12z_minus_") or c.startswith("abs_rev_12z_minus_")]

    calibrated_prob_block = [c for c in df.columns if c.endswith("_cal") and c.startswith("mos_")]

    calendar = ["doy_sin", "doy_cos", "month"]

    base = min_block_1 + min_block_2 + min_block_3 + mos_core + convective_base + convective_thunder_base + heat_gap + calendar

    exp = {}
    exp["V3_EXP01"] = base
    exp["V3_EXP02"] = base + ["completion_frac", "gap_frac", "mos_range_disagree"]
    exp["V3_EXP03"] = exp["V3_EXP02"] + bias_block
    exp["V3_EXP04"] = exp["V3_EXP03"] + convective_thunder_base + convective_thunder_variants
    exp["V3_EXP05"] = exp["V3_EXP04"] + qpf_bins
    exp["V3_EXP06"] = exp["V3_EXP05"] + ["p_onshore"]
    exp["V3_EXP07"] = exp["V3_EXP05"]  # MoE handled separately
    exp["V3_EXP08"] = exp["V3_EXP05"]  # seasonal experts handled separately
    exp["V3_EXP09"] = exp["V3_EXP05"]  # 4-expert MoE handled separately
    exp["V3_EXP10"] = exp["V3_EXP05"]  # front specialist handled separately
    exp["V3_EXP11"] = exp["V3_EXP03"] + min_outflow_extra
    exp["V3_EXP12"] = exp["V3_EXP03"]  # monotonic constraints handled separately
    exp["V3_EXP13"] = exp["V3_EXP03"]  # delta_future regression
    exp["V3_EXP14"] = exp["V3_EXP03"]  # quantile regression
    exp["V3_EXP15"] = exp["V3_EXP03"]  # time-of-max regression
    exp["V3_EXP16"] = exp["V3_EXP03"] + revision_block
    exp["V3_EXP17"] = exp["V3_EXP03"] + calibrated_prob_block
    exp["V3_EXP18"] = exp["V3_EXP03"]  # time-weighted
    exp["V3_EXP19"] = exp["V3_EXP03"]  # diverse ensemble
    exp["V3_EXP20"] = exp["V3_EXP03"]  # profit-focused thresholds

    # Ensure all features exist and unique
    for key, cols in exp.items():
        exp[key] = list(dict.fromkeys([c for c in cols if c in df.columns]))

    return exp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minute-dir", default=r"C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets\data\iem_minute_data\MIA\tmpf\UTC\yearly")
    parser.add_argument("--out-dir", default=r"C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets\artifacts\experiments\early_maxout_strategy\B4")
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

    def prepare_matrices(feature_cols: List[str]):
        feature_cols = [c for c in feature_cols if not merged_df.loc[train_mask, c].isna().all()]
        X = merged_df[feature_cols]
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X[train_mask])
        X_val = imputer.transform(X[val_mask])
        X_test = imputer.transform(X[test_mask])
        X_all = imputer.transform(X)
        return feature_cols, X_train, X_val, X_test, X_all

    def train_lgb(X_train, y_train, X_val, y_val, params, sample_weight=None):
        train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        model = lgb.train(
            params,
            train_set,
            valid_sets=[val_set],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return model

    def evaluate(exp_name: str, p_val: np.ndarray, p_test: np.ndarray,
                 feature_cols: List[str], model: lgb.Booster | None,
                 extra_metrics: Dict | None = None, extra_files: Dict | None = None,
                 extra_thresholds: List[float] | None = None):
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
        trade_rate_val = float(np.mean(y_val_profit))
        trade_rate_test = float(np.mean(y_test_profit))

        metrics = {
            "val": _compute_metrics(y[val_mask], p_val_cal, t_acc),
            "test": _compute_metrics(y[test_mask], p_test_cal, t_acc),
            "threshold_acc": t_acc,
            "threshold_bal": t_bal,
            "threshold_profit": t_profit,
            "net_units_per_100_val_profit": net_val,
            "net_units_per_100_test_profit": net_test,
            "trade_rate_val_profit": trade_rate_val,
            "trade_rate_test_profit": trade_rate_test,
            "metrics_raw_val": _compute_metrics(y[val_mask], p_val, t_acc),
            "metrics_raw_test": _compute_metrics(y[test_mask], p_test, t_acc),
            "always_no_acc": always_no_acc,
            "always_yes_acc": always_yes_acc,
        }
        if extra_thresholds:
            for thr in extra_thresholds:
                y_val_thr = (p_val_cal >= thr).astype(int)
                y_test_thr = (p_test_cal >= thr).astype(int)
                metrics[f"net_units_val_thr_{thr}"] = _net_units_per_100(y[val_mask], y_val_thr)
                metrics[f"net_units_test_thr_{thr}"] = _net_units_per_100(y[test_mask], y_test_thr)
                metrics[f"trade_rate_val_thr_{thr}"] = float(np.mean(y_val_thr))
                metrics[f"trade_rate_test_thr_{thr}"] = float(np.mean(y_test_thr))
        if extra_metrics:
            metrics.update(extra_metrics)

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

        _save_experiment(out_dir, exp_name, feature_cols, preds_val, preds_test, metrics, model, extra_files)
        results[exp_name] = metrics

    # V3_EXP01-06,11,16,17 (standard classifiers)
    for exp_name in ["V3_EXP01", "V3_EXP02", "V3_EXP03", "V3_EXP04", "V3_EXP05", "V3_EXP06", "V3_EXP11", "V3_EXP16", "V3_EXP17"]:
        feats, X_train, X_val, X_test, _ = prepare_matrices(feature_sets[exp_name])
        model = train_lgb(X_train, y[train_mask], X_val, y[val_mask], lgb_params)
        p_val = model.predict(X_val)
        p_test = model.predict(X_test)
        evaluate(exp_name, p_val, p_test, feats, model)

    # V3_EXP07: Onshore vs Offshore MoE
    exp_name = "V3_EXP07"
    feats, X_train, X_val, X_test, X_all = prepare_matrices(feature_sets[exp_name])
    p_onshore = merged_df["p_onshore"].fillna(0.0).to_numpy()
    onshore_flag = p_onshore >= 0.5
    train_on = train_mask & onshore_flag
    train_off = train_mask & (~onshore_flag)
    if train_on.sum() < 50 or train_off.sum() < 50:
        model = train_lgb(X_train, y[train_mask], X_val, y[val_mask], lgb_params)
        p_val = model.predict(X_val)
        p_test = model.predict(X_test)
        evaluate(exp_name, p_val, p_test, feats, model, extra_metrics={"moe_fallback": True})
    else:
        model_on = train_lgb(X_all[train_on], y[train_on], X_val, y[val_mask], lgb_params)
        model_off = train_lgb(X_all[train_off], y[train_off], X_val, y[val_mask], lgb_params)
        p_val = p_onshore[val_mask] * model_on.predict(X_val) + (1 - p_onshore[val_mask]) * model_off.predict(X_val)
        p_test = p_onshore[test_mask] * model_on.predict(X_test) + (1 - p_onshore[test_mask]) * model_off.predict(X_test)
        evaluate(exp_name, p_val, p_test, feats, None, extra_metrics={"moe_fallback": False})

    # V3_EXP08: Wet vs Dry season experts
    exp_name = "V3_EXP08"
    feats, X_train, X_val, X_test, X_all = prepare_matrices(feature_sets[exp_name])
    months = merged_df["month"].to_numpy()
    wet = (months >= 5) & (months <= 10)
    dry = ~wet
    p_val = np.full(val_mask.sum(), np.nan)
    p_test = np.full(test_mask.sum(), np.nan)
    if (train_mask & wet).sum() >= 50:
        model_wet = train_lgb(X_all[train_mask & wet], y[train_mask & wet], X_val, y[val_mask], lgb_params)
        p_val[wet[val_mask]] = model_wet.predict(X_val[wet[val_mask]])
        p_test[wet[test_mask]] = model_wet.predict(X_test[wet[test_mask]])
    if (train_mask & dry).sum() >= 50:
        model_dry = train_lgb(X_all[train_mask & dry], y[train_mask & dry], X_val, y[val_mask], lgb_params)
        p_val[dry[val_mask]] = model_dry.predict(X_val[dry[val_mask]])
        p_test[dry[test_mask]] = model_dry.predict(X_test[dry[test_mask]])
    if np.isnan(p_val).any():
        fallback_model = train_lgb(X_train, y[train_mask], X_val, y[val_mask], lgb_params)
        p_val = np.where(np.isnan(p_val), fallback_model.predict(X_val), p_val)
        p_test = np.where(np.isnan(p_test), fallback_model.predict(X_test), p_test)
    evaluate(exp_name, p_val, p_test, feats, None)

    # V3_EXP09: Season x Onshore 4-expert MoE
    exp_name = "V3_EXP09"
    feats, X_train, X_val, X_test, X_all = prepare_matrices(feature_sets[exp_name])
    p_onshore = merged_df["p_onshore"].fillna(0.0).to_numpy()
    onshore_flag = p_onshore >= 0.5
    p_val = np.full(val_mask.sum(), np.nan)
    p_test = np.full(test_mask.sum(), np.nan)
    for season_mask in [wet, dry]:
        train_on = train_mask & season_mask & onshore_flag
        train_off = train_mask & season_mask & (~onshore_flag)
        if train_on.sum() < 30 or train_off.sum() < 30:
            continue
        model_on = train_lgb(X_all[train_on], y[train_on], X_val, y[val_mask], lgb_params)
        model_off = train_lgb(X_all[train_off], y[train_off], X_val, y[val_mask], lgb_params)
        season_val = season_mask[val_mask]
        season_test = season_mask[test_mask]
        p_val[season_val] = p_onshore[val_mask][season_val] * model_on.predict(X_val[season_val]) + (1 - p_onshore[val_mask][season_val]) * model_off.predict(X_val[season_val])
        p_test[season_test] = p_onshore[test_mask][season_test] * model_on.predict(X_test[season_test]) + (1 - p_onshore[test_mask][season_test]) * model_off.predict(X_test[season_test])
    if np.isnan(p_val).any():
        fallback_model = train_lgb(X_train, y[train_mask], X_val, y[val_mask], lgb_params)
        p_val = np.where(np.isnan(p_val), fallback_model.predict(X_val), p_val)
        p_test = np.where(np.isnan(p_test), fallback_model.predict(X_test), p_test)
    evaluate(exp_name, p_val, p_test, feats, None)

    # V3_EXP10: Front / cold advection specialist
    exp_name = "V3_EXP10"
    feats, X_train, X_val, X_test, X_all = prepare_matrices(feature_sets[exp_name])
    front_flag = merged_df["front_flag"].fillna(0.0).to_numpy().astype(bool)
    train_front = train_mask & front_flag
    train_base = train_mask & (~front_flag)
    if train_front.sum() < 30 or train_base.sum() < 50:
        model = train_lgb(X_train, y[train_mask], X_val, y[val_mask], lgb_params)
        p_val = model.predict(X_val)
        p_test = model.predict(X_test)
        evaluate(exp_name, p_val, p_test, feats, model, extra_metrics={"front_fallback": True})
    else:
        model_front = train_lgb(X_all[train_front], y[train_front], X_val, y[val_mask], lgb_params)
        model_base = train_lgb(X_all[train_base], y[train_base], X_val, y[val_mask], lgb_params)
        p_val = np.where(front_flag[val_mask], model_front.predict(X_val), model_base.predict(X_val))
        p_test = np.where(front_flag[test_mask], model_front.predict(X_test), model_base.predict(X_test))
        evaluate(exp_name, p_val, p_test, feats, None, extra_metrics={"front_fallback": False})

    # V3_EXP12: minimal monotone constraints
    exp_name = "V3_EXP12"
    feats, X_train, X_val, X_test, _ = prepare_matrices(feature_sets[exp_name])
    params = lgb_params.copy()
    params["monotone_constraints"] = _build_monotone_constraints(feats)
    model = train_lgb(X_train, y[train_mask], X_val, y[val_mask], params)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    evaluate(exp_name, p_val, p_test, feats, model)

    # V3_EXP13: delta_future regression -> probability
    exp_name = "V3_EXP13"
    feats, X_train, X_val, X_test, _ = prepare_matrices(feature_sets[exp_name])
    params_reg = lgb_params.copy()
    params_reg["objective"] = "regression"
    model_reg = train_lgb(X_train, merged_df.loc[train_mask, "delta_future"].to_numpy(), X_val, merged_df.loc[val_mask, "delta_future"].to_numpy(), params_reg)
    pred_train = model_reg.predict(X_train)
    residuals = np.abs(merged_df.loc[train_mask, "delta_future"].to_numpy() - pred_train)
    model_sigma = train_lgb(X_train, residuals, X_val, np.abs(merged_df.loc[val_mask, "delta_future"].to_numpy() - model_reg.predict(X_val)), params_reg)
    sigma_val = np.clip(model_sigma.predict(X_val), 0.1, None)
    sigma_test = np.clip(model_sigma.predict(X_test), 0.1, None)
    mu_val = model_reg.predict(X_val)
    mu_test = model_reg.predict(X_test)
    p_val = _norm_cdf((0 - mu_val) / sigma_val)
    p_test = _norm_cdf((0 - mu_test) / sigma_test)
    evaluate(exp_name, p_val, p_test, feats, model_reg)

    # V3_EXP14: quantile regression
    exp_name = "V3_EXP14"
    feats, X_train, X_val, X_test, _ = prepare_matrices(feature_sets[exp_name])
    q_models = {}
    for alpha in [0.1, 0.5, 0.9]:
        params_q = lgb_params.copy()
        params_q["objective"] = "quantile"
        params_q["alpha"] = alpha
        q_models[alpha] = train_lgb(X_train, merged_df.loc[train_mask, "delta_future"].to_numpy(), X_val, merged_df.loc[val_mask, "delta_future"].to_numpy(), params_q)
    q10_val = q_models[0.1].predict(X_val)
    q50_val = q_models[0.5].predict(X_val)
    q90_val = q_models[0.9].predict(X_val)
    q10_test = q_models[0.1].predict(X_test)
    q50_test = q_models[0.5].predict(X_test)
    q90_test = q_models[0.9].predict(X_test)

    def prob_from_quantiles(q10, q50, q90):
        p = np.zeros_like(q50)
        for i in range(len(p)):
            if 0 <= q10[i]:
                p[i] = 0.1
            elif 0 <= q50[i]:
                denom = q50[i] - q10[i] if q50[i] != q10[i] else 1.0
                p[i] = 0.1 + (0 - q10[i]) / denom * 0.4
            elif 0 <= q90[i]:
                denom = q90[i] - q50[i] if q90[i] != q50[i] else 1.0
                p[i] = 0.5 + (0 - q50[i]) / denom * 0.4
            else:
                p[i] = 0.9
        return np.clip(p, 0.0, 1.0)

    p_val = prob_from_quantiles(q10_val, q50_val, q90_val)
    p_test = prob_from_quantiles(q10_test, q50_test, q90_test)
    evaluate(exp_name, p_val, p_test, feats, q_models[0.5])

    # V3_EXP15: time-of-maximum regression
    exp_name = "V3_EXP15"
    feats, X_train, X_val, X_test, _ = prepare_matrices(feature_sets[exp_name])
    params_reg = lgb_params.copy()
    params_reg["objective"] = "regression"
    model_time = train_lgb(X_train, merged_df.loc[train_mask, "tmax_time_local_minute"].to_numpy(), X_val, merged_df.loc[val_mask, "tmax_time_local_minute"].to_numpy(), params_reg)
    pred_train = model_time.predict(X_train)
    resid = merged_df.loc[train_mask, "tmax_time_local_minute"].to_numpy() - pred_train
    sigma_time = np.clip(np.nanstd(resid), 30.0, None)
    cutoff_val = merged_df.loc[val_mask, "cutoff_local_minute"].to_numpy()
    cutoff_test = merged_df.loc[test_mask, "cutoff_local_minute"].to_numpy()
    mu_val = model_time.predict(X_val)
    mu_test = model_time.predict(X_test)
    p_val = _norm_cdf((cutoff_val - mu_val) / sigma_time)
    p_test = _norm_cdf((cutoff_test - mu_test) / sigma_time)
    evaluate(exp_name, p_val, p_test, feats, model_time)

    # V3_EXP18: time-weighted training
    exp_name = "V3_EXP18"
    feats, X_train, X_val, X_test, _ = prepare_matrices(feature_sets[exp_name])
    weights = 1.0 + (years[train_mask] - 2002) / max((2019 - 2002), 1)
    model = train_lgb(X_train, y[train_mask], X_val, y[val_mask], lgb_params, sample_weight=weights)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    evaluate(exp_name, p_val, p_test, feats, model)

    # V3_EXP19: diverse ensemble (5 models)
    exp_name = "V3_EXP19"
    feats, X_train, X_val, X_test, _ = prepare_matrices(feature_sets[exp_name])
    variants = [
        {"seed": 1, "feature_fraction": 0.8, "num_leaves": 63},
        {"seed": 2, "feature_fraction": 0.7, "num_leaves": 63},
        {"seed": 3, "feature_fraction": 0.9, "num_leaves": 31},
        {"seed": 4, "feature_fraction": 0.8, "num_leaves": 127},
        {"seed": 5, "feature_fraction": 0.6, "num_leaves": 63},
    ]
    preds_val = []
    preds_test = []
    for var in variants:
        params = lgb_params.copy()
        params.update(var)
        model = train_lgb(X_train, y[train_mask], X_val, y[val_mask], params)
        preds_val.append(model.predict(X_val))
        preds_test.append(model.predict(X_test))
    p_val = np.mean(preds_val, axis=0)
    p_test = np.mean(preds_test, axis=0)
    evaluate(exp_name, p_val, p_test, feats, None)

    # V3_EXP20: profit-focused thresholds + confidence gating
    exp_name = "V3_EXP20"
    feats, X_train, X_val, X_test, _ = prepare_matrices(feature_sets[exp_name])
    model = train_lgb(X_train, y[train_mask], X_val, y[val_mask], lgb_params)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    evaluate(exp_name, p_val, p_test, feats, model, extra_thresholds=[0.65, 0.70, 0.75])

    # Summary report
    summary_json_path = reports_dir / "hit1830_v3_experiments_summary.json"
    summary_md_path = reports_dir / "hit1830_v3_experiments_report.md"

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
                "trade_rate_val": metrics.get("trade_rate_val_profit"),
                "trade_rate_test": metrics.get("trade_rate_test_profit"),
            }
        )

    best_val_acc = max(rows, key=lambda r: r["val_acc"])
    best_val_bal = max(rows, key=lambda r: r["val_bal"])

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Hit 18:30 Stockholm V3 Experiments Report\n\n")
        f.write(f"Always-NO test accuracy: {always_no_acc:.3f}\n\n")
        f.write(f"Always-YES test accuracy: {always_yes_acc:.3f}\n\n")
        f.write("| Experiment | Val Acc | Val Bal Acc | Val YES Recall | Test Acc | Test Bal Acc | Test YES Recall | NetUnits/100 (Val) | NetUnits/100 (Test) | TradeRate (Val) | TradeRate (Test) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['exp']} | {row['val_acc']:.3f} | {row['val_bal']:.3f} | {row['val_yes_recall']:.3f} | {row['test_acc']:.3f} | {row['test_bal']:.3f} | {row['test_yes_recall']:.3f} | {row['net_units_val']:.2f} | {row['net_units_test']:.2f} | {row['trade_rate_val']:.3f} | {row['trade_rate_test']:.3f} |\n"
            )

        f.write("\n")
        f.write(f"Best by Val Accuracy: {best_val_acc['exp']} (Val Acc {best_val_acc['val_acc']:.3f})\n\n")
        f.write(f"Best by Val Balanced Accuracy: {best_val_bal['exp']} (Val Bal Acc {best_val_bal['val_bal']:.3f})\n\n")

    (out_dir / summary_json_path.name).write_text(summary_json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / summary_md_path.name).write_text(summary_md_path.read_text(encoding="utf-8"), encoding="utf-8")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
