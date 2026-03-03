from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from zoneinfo import ZoneInfo

UTC = timezone.utc
STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")
EPS = 1e-12


@dataclass(frozen=True)
class BucketInterval:
    label: str
    kind: str  # range | le | ge
    lo: Optional[int]
    hi: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive calibration audit for entry>=19:00 Stockholm.")
    parser.add_argument(
        "--kalshi-dir",
        default=r"D:\Ahmed\data\kalshi\kalshi_history\kxhighny_2025",
        help="Directory with daily Kalshi CSV files.",
    )
    parser.add_argument(
        "--predictions-parquet",
        default=(
            r"D:\Ahmed\data\kalshi\Experiments\E2_KNYC\Experiment_set_1\E2\20260228T153836Z\predictions\predictions_test.parquet"
        ),
        help="Predictions parquet.",
    )
    parser.add_argument(
        "--distribution-eval-parquet",
        default=(
            r"D:\Ahmed\data\kalshi\Experiments\E2_KNYC\Experiment_set_1\E2\20260228T153836Z\predictions\distribution_eval_test.parquet"
        ),
        help="Distribution eval parquet.",
    )
    parser.add_argument("--start-date", default="2025-01-01", help="Inclusive start date YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2025-12-31", help="Inclusive end date YYYY-MM-DD.")
    parser.add_argument("--entry-hour-stockholm", type=int, default=19)
    parser.add_argument("--entry-minute-stockholm", type=int, default=0)
    parser.add_argument("--trade-date-offset-days", type=int, default=0)
    parser.add_argument("--ev-threshold", type=float, default=0.10)
    parser.add_argument("--min-win-prob", type=float, default=0.650001)
    parser.add_argument("--output-dir", default=r"D:\Ahmed\data\kalshi\backtesting\results")
    return parser.parse_args()


def normalize_label(label: str) -> str:
    text = str(label)
    text = text.replace("Ãƒâ€šÃ‚Â°", "Â°").replace("Ã‚Â°", "Â°").replace("Ã‚Âº", "Â°").replace("Ã¢â‚¬â€œ", "-").replace("Ã¢Ë†â€™", "-")
    return re.sub(r"\s+", " ", text).strip()


def parse_bucket_interval(label: str) -> BucketInterval:
    raw = normalize_label(label)
    low = raw.lower().replace("Â°f", "").replace("°f", "").replace("Â°", "").replace("°", "")
    low = re.sub(r"\s+", " ", low).strip()

    m = re.search(r"(-?\d+)\s*-\s*(-?\d+)", low)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return BucketInterval(label=raw, kind="range", lo=lo, hi=hi)

    m = re.search(r"(-?\d+)\s*to\s*(-?\d+)", low)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return BucketInterval(label=raw, kind="range", lo=lo, hi=hi)

    m = re.search(r"(-?\d+)\s*(?:or)?\s*(?:below|less)", low)
    if m:
        return BucketInterval(label=raw, kind="le", lo=None, hi=int(m.group(1)))

    m = re.search(r"(-?\d+)\s*(?:or)?\s*(?:above|higher|more)", low)
    if m:
        return BucketInterval(label=raw, kind="ge", lo=int(m.group(1)), hi=None)

    raise ValueError(f"Unable to parse bucket label: {label}")


def bucket_contains(interval: BucketInterval, tmax: int) -> bool:
    if interval.kind == "range":
        assert interval.lo is not None and interval.hi is not None
        return interval.lo <= tmax <= interval.hi
    if interval.kind == "le":
        assert interval.hi is not None
        return tmax <= interval.hi
    if interval.kind == "ge":
        assert interval.lo is not None
        return tmax >= interval.lo
    return False


def _normalize_day_prices(day_df: pd.DataFrame, bucket_cols: List[str]) -> Tuple[pd.DataFrame, str]:
    out = day_df.copy()
    for c in bucket_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    vals = out[bucket_cols].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return out, "unknown"
    vmax = float(np.nanmax(finite))
    if vmax <= 1.5:
        out[bucket_cols] = out[bucket_cols] * 100.0
        return out, "0-1_to_0-100"
    return out, "0-100"


def build_model_frame(eval_path: Path, pred_path: Path) -> pd.DataFrame:
    eval_df = pd.read_parquet(eval_path)
    pred_df = pd.read_parquet(pred_path)
    df = eval_df.merge(pred_df, on="row_index", how="left", suffixes=("_eval", "_pred"))
    ny = pd.to_datetime(df["target_date_local"]) + pd.to_timedelta(df["cutoff_minutes"], unit="m")
    ny = ny.dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
    st = ny.dt.tz_convert(STOCKHOLM_TZ)
    df["st_timestamp"] = st
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    return df


def get_model_row_for_entry_time(
    model_df: pd.DataFrame,
    trade_date: date,
    entry_time_stockholm: datetime,
) -> Optional[pd.Series]:
    rows = model_df.loc[
        (model_df["target_date_local"] == trade_date)
        & (model_df["st_timestamp"].notna())
        & (model_df["st_timestamp"] <= pd.Timestamp(entry_time_stockholm))
    ]
    if rows.empty:
        return None
    return rows.sort_values("st_timestamp").iloc[-1]


def build_integer_pmf(row: pd.Series) -> Tuple[Dict[int, float], int, float]:
    p_peak = float(row["p_peak_pred"])
    tmax_sofar = int(row["tmax_sofar_round"])
    p_delta = np.array([float(row[f"p_delta_class_{k}"]) for k in range(1, 61)], dtype=float)
    if np.isfinite(p_delta).all() and p_delta.sum() > 0:
        p_delta = p_delta / p_delta.sum()
    else:
        p_delta = np.zeros(60, dtype=float)

    pmf: Dict[int, float] = {}
    pmf[tmax_sofar] = p_peak
    positive_mass = 1.0 - p_peak
    for k in range(1, 60):
        pmf[tmax_sofar + k] = positive_mass * float(p_delta[k - 1])
    tail_start = tmax_sofar + 60
    tail_prob = positive_mass * float(p_delta[59])
    return pmf, tail_start, tail_prob


def bucket_prob(interval: BucketInterval, pmf: Dict[int, float], tail_start: int, tail_prob: float) -> float:
    p = 0.0
    if interval.kind == "range":
        assert interval.lo is not None and interval.hi is not None
        for t, pt in pmf.items():
            if interval.lo <= t <= interval.hi:
                p += pt
        if interval.hi >= tail_start:
            p += tail_prob
        return float(p)
    if interval.kind == "le":
        assert interval.hi is not None
        for t, pt in pmf.items():
            if t <= interval.hi:
                p += pt
        return float(p)
    if interval.kind == "ge":
        assert interval.lo is not None
        for t, pt in pmf.items():
            if t >= interval.lo:
                p += pt
        if tail_start >= interval.lo:
            p += tail_prob
        elif interval.lo > tail_start:
            p += tail_prob
        return float(p)
    return float("nan")


def build_class_probs(row: pd.Series) -> Tuple[np.ndarray, int]:
    p_peak = float(row["p_peak_pred"])
    p_delta = np.array([float(row[f"p_delta_class_{k}"]) for k in range(1, 61)], dtype=float)
    if (not np.isfinite(p_delta).all()) or p_delta.sum() <= 0:
        p_delta = np.zeros(60, dtype=float)
        p_delta[-1] = 1.0
    else:
        p_delta = p_delta / p_delta.sum()

    q = np.zeros(61, dtype=float)
    q[0] = p_peak
    q[1:] = (1.0 - p_peak) * p_delta
    s = float(q.sum())
    if s > 0:
        q = q / s

    truth = int(row["tmax_truth"])
    sofar = int(row["tmax_sofar_round"])
    delta = truth - sofar
    true_class = 0 if delta <= 0 else (60 if delta >= 60 else delta)
    return q, int(true_class)


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int) -> Tuple[pd.DataFrame, float, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    idx = np.floor(p * n_bins).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)

    rows: List[Dict[str, float]] = []
    ece = 0.0
    mce = 0.0
    n = max(1, len(p))
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            rows.append(
                {
                    "bin": b,
                    "count": 0,
                    "pred_mean": float("nan"),
                    "empirical_rate": float("nan"),
                    "abs_gap": float("nan"),
                    "bin_left": b / n_bins,
                    "bin_right": (b + 1) / n_bins,
                }
            )
            continue
        pm = float(p[m].mean())
        ym = float(y[m].mean())
        gap = abs(pm - ym)
        ece += (c / n) * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin": b,
                "count": c,
                "pred_mean": pm,
                "empirical_rate": ym,
                "abs_gap": gap,
                "bin_left": b / n_bins,
                "bin_right": (b + 1) / n_bins,
            }
        )
    return pd.DataFrame(rows), float(ece), float(mce)


def binary_calibration_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)

    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    pred_mean = float(p.mean())
    emp_mean = float(y.mean())

    rel10, ece10, mce10 = reliability_table(y=y, p=p, n_bins=10)
    rel20, ece20, mce20 = reliability_table(y=y, p=p, n_bins=20)

    z = np.log(p / (1.0 - p))

    def _nll(params: np.ndarray) -> float:
        a = float(params[0])
        b = float(params[1])
        q = 1.0 / (1.0 + np.exp(-(a + b * z)))
        q = np.clip(q, EPS, 1.0 - EPS)
        return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))

    opt = minimize(_nll, x0=np.array([0.0, 1.0]), method="BFGS")
    cal_intercept = float(opt.x[0]) if opt.success else float("nan")
    cal_slope = float(opt.x[1]) if opt.success else float("nan")

    return {
        "count": int(len(y)),
        "brier": brier,
        "logloss": logloss,
        "pred_mean": pred_mean,
        "empirical_mean": emp_mean,
        "mean_gap_pred_minus_empirical": float(pred_mean - emp_mean),
        "ece_10": ece10,
        "mce_10": mce10,
        "ece_20": ece20,
        "mce_20": mce20,
        "calibration_intercept": cal_intercept,
        "calibration_slope": cal_slope,
        "reliability_10": rel10,
        "reliability_20": rel20,
    }


def find_quantile_class(cdf: np.ndarray, q: float) -> int:
    return int(np.searchsorted(cdf, q, side="left"))


def run_audit(args: argparse.Namespace) -> Tuple[Dict[str, object], Dict[str, pd.DataFrame]]:
    kalshi_dir = Path(args.kalshi_dir)
    pred_path = Path(args.predictions_parquet)
    eval_path = Path(args.distribution_eval_parquet)

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)

    model_df = build_model_frame(eval_path=eval_path, pred_path=pred_path)

    files = sorted(kalshi_dir.glob("*.csv"))
    parsed_files: List[Tuple[Path, date]] = []
    for p in files:
        token = p.stem.split("_")[-1]
        try:
            d = datetime.strptime(token, "%Y%m%d").date()
        except ValueError:
            continue
        if start_date <= d <= end_date:
            parsed_files.append((p, d))

    distribution_rows: List[Dict[str, object]] = []
    bucket_rows: List[Dict[str, object]] = []
    selected_trade_rows: List[Dict[str, object]] = []
    scale_mode_counts: Dict[str, int] = {}
    pit_values: List[float] = []

    rng = np.random.default_rng(42)

    for fp, file_date in parsed_files:
        trade_date = file_date + timedelta(days=int(args.trade_date_offset_days))
        day_df = pd.read_csv(fp)
        day_df["timestamp"] = pd.to_datetime(day_df["timestamp"], utc=True, errors="coerce")
        day_df = day_df[day_df["timestamp"].notna()].sort_values("timestamp")
        if day_df.empty:
            continue

        bucket_cols = [c for c in day_df.columns if c != "timestamp"]
        if not bucket_cols:
            continue

        parsed_intervals: Dict[str, BucketInterval] = {}
        for col in bucket_cols:
            parsed_intervals[col] = parse_bucket_interval(normalize_label(col))
        day_df, scale_mode = _normalize_day_prices(day_df, bucket_cols)
        scale_mode_counts[scale_mode] = int(scale_mode_counts.get(scale_mode, 0) + 1)
        day_df[bucket_cols] = day_df[bucket_cols].ffill()

        cutoff_st = datetime(
            trade_date.year,
            trade_date.month,
            trade_date.day,
            int(args.entry_hour_stockholm),
            int(args.entry_minute_stockholm),
            tzinfo=STOCKHOLM_TZ,
        )
        cutoff_utc = cutoff_st.astimezone(UTC)
        entry_candidates = day_df[day_df["timestamp"] >= pd.Timestamp(cutoff_utc)]
        if entry_candidates.empty:
            continue
        entry_row = entry_candidates.iloc[0]
        entry_time_st = entry_row["timestamp"].to_pydatetime().astimezone(STOCKHOLM_TZ)

        model_row = get_model_row_for_entry_time(model_df=model_df, trade_date=trade_date, entry_time_stockholm=entry_time_st)
        if model_row is None:
            continue

        q, true_class = build_class_probs(model_row)
        true_prob = float(np.clip(q[true_class], EPS, 1.0))
        nll = float(-np.log(true_prob))
        brier_mc = float(np.sum(q**2) - 2.0 * q[true_class] + 1.0)
        y_cdf = np.zeros_like(q)
        y_cdf[true_class:] = 1.0
        q_cdf = np.cumsum(q)
        rps = float(np.sum((q_cdf - y_cdf) ** 2) / (len(q) - 1))
        pred_class = int(np.argmax(q))
        top1_conf = float(np.max(q))
        top1_hit = 1 if pred_class == true_class else 0
        entropy = float(-np.sum(np.clip(q, EPS, 1.0) * np.log(np.clip(q, EPS, 1.0))))

        f_prev = float(np.sum(q[:true_class]))
        f_mass = float(q[true_class])
        pit = f_prev + float(rng.random()) * f_mass
        pit_values.append(float(pit))

        cdf = np.cumsum(q)
        coverage = {}
        for level in [0.50, 0.80, 0.90, 0.95]:
            low_q = (1.0 - level) / 2.0
            high_q = 1.0 - low_q
            lo = find_quantile_class(cdf, low_q)
            hi = find_quantile_class(cdf, high_q)
            coverage[f"cov_{int(level*100)}"] = 1 if (lo <= true_class <= hi) else 0
            coverage[f"width_{int(level*100)}"] = int(hi - lo + 1)

        distribution_rows.append(
            {
                "file_date": file_date.isoformat(),
                "trade_date": trade_date.isoformat(),
                "entry_time_stockholm": entry_time_st.isoformat(),
                "row_index": int(model_row["row_index"]),
                "true_class": true_class,
                "pred_class": pred_class,
                "top1_hit": top1_hit,
                "top1_conf": top1_conf,
                "true_prob": true_prob,
                "nll": nll,
                "brier_multiclass": brier_mc,
                "rps": rps,
                "entropy": entropy,
                "pit": pit,
                **coverage,
            }
        )

        actual_tmax = int(model_row["tmax_truth"])
        pmf, tail_start, tail_prob = build_integer_pmf(model_row)
        candidate_rows: List[Dict[str, object]] = []

        for col in bucket_cols:
            interval = parsed_intervals[col]
            label = normalize_label(col)

            entry_yes = pd.to_numeric(entry_row[col], errors="coerce")
            if not np.isfinite(entry_yes):
                continue
            entry_yes = float(entry_yes)
            if not (0.0 <= entry_yes <= 100.0):
                continue

            model_yes = float(bucket_prob(interval, pmf, tail_start, tail_prob))
            model_yes = float(np.clip(model_yes, 0.0, 1.0))
            market_yes = float(np.clip(entry_yes / 100.0, 0.0, 1.0))
            outcome_yes = 1 if bucket_contains(interval, actual_tmax) else 0
            model_no = 1.0 - model_yes
            market_no = 1.0 - market_yes
            ev_yes = model_yes - market_yes
            ev_no = model_no - market_no

            bucket_rows.append(
                {
                    "file_date": file_date.isoformat(),
                    "trade_date": trade_date.isoformat(),
                    "entry_time_stockholm": entry_time_st.isoformat(),
                    "bucket_label": label,
                    "actual_tmax_f": actual_tmax,
                    "model_yes_prob": model_yes,
                    "market_yes_prob": market_yes,
                    "outcome_yes": outcome_yes,
                    "ev_yes": ev_yes,
                    "ev_no": ev_no,
                }
            )

            if model_yes >= float(args.min_win_prob) and ev_yes >= float(args.ev_threshold):
                candidate_rows.append(
                    {
                        "side": "YES",
                        "bucket_label": label,
                        "model_win_prob": model_yes,
                        "market_win_prob": market_yes,
                        "ev": ev_yes,
                        "outcome_win": float(outcome_yes),
                    }
                )
            if model_no >= float(args.min_win_prob) and ev_no >= float(args.ev_threshold):
                candidate_rows.append(
                    {
                        "side": "NO",
                        "bucket_label": label,
                        "model_win_prob": model_no,
                        "market_win_prob": market_no,
                        "ev": ev_no,
                        "outcome_win": float(1 - outcome_yes),
                    }
                )

        if candidate_rows:
            best = max(
                candidate_rows,
                key=lambda r: (
                    float(r["ev"]),
                    float(r["model_win_prob"]),
                    -float(r["market_win_prob"]),
                    str(r["bucket_label"]),
                    str(r["side"]),
                ),
            )
            selected_trade_rows.append(
                {
                    "file_date": file_date.isoformat(),
                    "trade_date": trade_date.isoformat(),
                    "entry_time_stockholm": entry_time_st.isoformat(),
                    "bucket_label": str(best["bucket_label"]),
                    "side": str(best["side"]),
                    "model_win_prob": float(best["model_win_prob"]),
                    "market_win_prob": float(best["market_win_prob"]),
                    "ev": float(best["ev"]),
                    "outcome_win": float(best["outcome_win"]),
                }
            )

    dist_df = pd.DataFrame(distribution_rows)
    bucket_df = pd.DataFrame(bucket_rows)
    selected_df = pd.DataFrame(selected_trade_rows)

    if dist_df.empty:
        raise RuntimeError("No distribution rows were produced for the requested date/time range.")
    if bucket_df.empty:
        raise RuntimeError("No bucket rows were produced for the requested date/time range.")

    pit_arr = np.array(pit_values, dtype=float)
    ks = stats.kstest(pit_arr, "uniform")
    pit_hist_counts, pit_hist_edges = np.histogram(pit_arr, bins=20, range=(0.0, 1.0))
    pit_hist_df = pd.DataFrame(
        {
            "bin_left": pit_hist_edges[:-1],
            "bin_right": pit_hist_edges[1:],
            "count": pit_hist_counts,
        }
    )

    top1_rel_10, top1_ece_10, top1_mce_10 = reliability_table(
        y=dist_df["top1_hit"].to_numpy(dtype=float),
        p=dist_df["top1_conf"].to_numpy(dtype=float),
        n_bins=10,
    )

    bucket_metrics = binary_calibration_metrics(
        y=bucket_df["outcome_yes"].to_numpy(dtype=float),
        p=bucket_df["model_yes_prob"].to_numpy(dtype=float),
    )
    market_bucket_metrics = binary_calibration_metrics(
        y=bucket_df["outcome_yes"].to_numpy(dtype=float),
        p=bucket_df["market_yes_prob"].to_numpy(dtype=float),
    )

    selected_metrics = None
    if not selected_df.empty:
        selected_metrics = binary_calibration_metrics(
            y=selected_df["outcome_win"].to_numpy(dtype=float),
            p=selected_df["model_win_prob"].to_numpy(dtype=float),
        )

    coverage_summary = {}
    for lvl in [50, 80, 90, 95]:
        coverage_summary[f"coverage_{lvl}"] = float(dist_df[f"cov_{lvl}"].mean())
        coverage_summary[f"avg_width_{lvl}"] = float(dist_df[f"width_{lvl}"].mean())

    summary: Dict[str, object] = {
        "config": {
            "kalshi_dir": str(kalshi_dir),
            "predictions_parquet": str(pred_path),
            "distribution_eval_parquet": str(eval_path),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "entry_time_stockholm_min": f"{int(args.entry_hour_stockholm):02d}:{int(args.entry_minute_stockholm):02d}",
            "trade_date_offset_days": int(args.trade_date_offset_days),
            "trade_filter_ev_threshold": float(args.ev_threshold),
            "trade_filter_min_win_prob": float(args.min_win_prob),
        },
        "counts": {
            "files_considered": int(len(parsed_files)),
            "daily_entry_rows": int(len(dist_df)),
            "bucket_events": int(len(bucket_df)),
            "selected_trades": int(len(selected_df)),
            "price_scale_mode_counts": scale_mode_counts,
        },
        "distribution_entry_daily": {
            "nll_mean": float(dist_df["nll"].mean()),
            "brier_multiclass_mean": float(dist_df["brier_multiclass"].mean()),
            "rps_mean": float(dist_df["rps"].mean()),
            "top1_accuracy": float(dist_df["top1_hit"].mean()),
            "top1_confidence_mean": float(dist_df["top1_conf"].mean()),
            "top1_ece_10": float(top1_ece_10),
            "top1_mce_10": float(top1_mce_10),
            "pit_mean": float(pit_arr.mean()),
            "pit_variance": float(pit_arr.var()),
            "pit_ks_statistic": float(ks.statistic),
            "pit_ks_pvalue": float(ks.pvalue),
            **coverage_summary,
        },
        "bucket_yes_all_at_entry": {
            "model": {
                "count": bucket_metrics["count"],
                "brier": bucket_metrics["brier"],
                "logloss": bucket_metrics["logloss"],
                "pred_mean": bucket_metrics["pred_mean"],
                "empirical_mean": bucket_metrics["empirical_mean"],
                "mean_gap_pred_minus_empirical": bucket_metrics["mean_gap_pred_minus_empirical"],
                "ece_10": bucket_metrics["ece_10"],
                "mce_10": bucket_metrics["mce_10"],
                "ece_20": bucket_metrics["ece_20"],
                "mce_20": bucket_metrics["mce_20"],
                "calibration_intercept": bucket_metrics["calibration_intercept"],
                "calibration_slope": bucket_metrics["calibration_slope"],
            },
            "market": {
                "count": market_bucket_metrics["count"],
                "brier": market_bucket_metrics["brier"],
                "logloss": market_bucket_metrics["logloss"],
                "pred_mean": market_bucket_metrics["pred_mean"],
                "empirical_mean": market_bucket_metrics["empirical_mean"],
                "mean_gap_pred_minus_empirical": market_bucket_metrics["mean_gap_pred_minus_empirical"],
                "ece_10": market_bucket_metrics["ece_10"],
                "mce_10": market_bucket_metrics["mce_10"],
                "ece_20": market_bucket_metrics["ece_20"],
                "mce_20": market_bucket_metrics["mce_20"],
                "calibration_intercept": market_bucket_metrics["calibration_intercept"],
                "calibration_slope": market_bucket_metrics["calibration_slope"],
            },
            "model_minus_market": {
                "brier": float(bucket_metrics["brier"] - market_bucket_metrics["brier"]),
                "logloss": float(bucket_metrics["logloss"] - market_bucket_metrics["logloss"]),
                "ece_10": float(bucket_metrics["ece_10"] - market_bucket_metrics["ece_10"]),
            },
        },
    }

    if selected_metrics is not None:
        summary["selected_trade_side"] = {
            "count": selected_metrics["count"],
            "brier": selected_metrics["brier"],
            "logloss": selected_metrics["logloss"],
            "pred_mean": selected_metrics["pred_mean"],
            "empirical_mean": selected_metrics["empirical_mean"],
            "mean_gap_pred_minus_empirical": selected_metrics["mean_gap_pred_minus_empirical"],
            "ece_10": selected_metrics["ece_10"],
            "mce_10": selected_metrics["mce_10"],
            "ece_20": selected_metrics["ece_20"],
            "mce_20": selected_metrics["mce_20"],
            "calibration_intercept": selected_metrics["calibration_intercept"],
            "calibration_slope": selected_metrics["calibration_slope"],
        }

    outputs: Dict[str, pd.DataFrame] = {
        "distribution_daily_rows": dist_df,
        "bucket_events_all": bucket_df,
        "selected_trades": selected_df,
        "reliability_top1_10bins": top1_rel_10,
        "pit_hist_20bins": pit_hist_df,
        "reliability_bucket_model_10bins": bucket_metrics["reliability_10"],
        "reliability_bucket_model_20bins": bucket_metrics["reliability_20"],
        "reliability_bucket_market_10bins": market_bucket_metrics["reliability_10"],
        "reliability_bucket_market_20bins": market_bucket_metrics["reliability_20"],
    }
    if selected_metrics is not None:
        outputs["reliability_selected_10bins"] = selected_metrics["reliability_10"]
        outputs["reliability_selected_20bins"] = selected_metrics["reliability_20"]

    return summary, outputs


def main() -> int:
    args = parse_args()
    summary, outputs = run_audit(args)

    run_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"calibration_audit_1900_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for key, df in outputs.items():
        df.to_csv(out_dir / f"{key}.csv", index=False, encoding="utf-8")

    print(f"summary_json: {summary_path}")
    for key in outputs.keys():
        print(f"{key}_csv: {out_dir / f'{key}.csv'}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
