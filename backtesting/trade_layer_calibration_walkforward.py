from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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


@dataclass(frozen=True)
class PlattParams:
    a: float
    b: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-free trade-layer calibration with walk-forward OOF.")
    p.add_argument(
        "--kalshi-dir",
        default=r"D:\Ahmed\data\kalshi\kalshi_history\kxhighny_2025",
    )
    p.add_argument(
        "--predictions-parquet",
        default=r"D:\Ahmed\data\kalshi\Experiments\E2_KNYC\Experiment_set_1\E2\20260228T153836Z\predictions\predictions_test.parquet",
    )
    p.add_argument(
        "--distribution-eval-parquet",
        default=r"D:\Ahmed\data\kalshi\Experiments\E2_KNYC\Experiment_set_1\E2\20260228T153836Z\predictions\distribution_eval_test.parquet",
    )
    p.add_argument("--entry-hour-stockholm", type=int, default=19)
    p.add_argument("--entry-minute-stockholm", type=int, default=0)
    p.add_argument("--trade-date-offset-days", type=int, default=0)
    p.add_argument("--ev-threshold", type=float, default=0.10)
    p.add_argument("--min-win-prob", type=float, default=0.650001)
    p.add_argument("--oof-min-train-days", type=int, default=120)
    p.add_argument("--oof-block-days", type=int, default=30)
    p.add_argument("--holdout-start", default="2025-10-01")
    p.add_argument("--start-balance", type=float, default=2700.0)
    p.add_argument("--risk-fraction", type=float, default=0.04)
    p.add_argument("--output-dir", default=r"D:\Ahmed\data\kalshi\backtesting\results")
    return p.parse_args()


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


def get_model_row_for_entry_time(model_df: pd.DataFrame, trade_date: date, entry_time_stockholm: datetime) -> Optional[pd.Series]:
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
        p_delta[-1] = 1.0

    pmf: Dict[int, float] = {tmax_sofar: p_peak}
    pos_mass = 1.0 - p_peak
    for k in range(1, 60):
        pmf[tmax_sofar + k] = pos_mass * float(p_delta[k - 1])
    tail_start = tmax_sofar + 60
    tail_prob = pos_mass * float(p_delta[59])
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


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_platt(y: np.ndarray, p_raw: np.ndarray) -> PlattParams:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p_raw, dtype=float)
    z = logit(p)

    def nll(params: np.ndarray) -> float:
        a = float(params[0])
        b = float(params[1])
        q = sigmoid(a + b * z)
        q = np.clip(q, EPS, 1.0 - EPS)
        reg = 1e-4 * (a * a + (b - 1.0) * (b - 1.0))
        return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)) + reg)

    opt = minimize(nll, x0=np.array([0.0, 1.0]), method="L-BFGS-B", bounds=[(-10.0, 10.0), (0.01, 10.0)])
    if not opt.success:
        return PlattParams(a=0.0, b=1.0)
    return PlattParams(a=float(opt.x[0]), b=float(opt.x[1]))


def apply_platt(p_raw: np.ndarray, params: PlattParams) -> np.ndarray:
    return sigmoid(params.a + params.b * logit(np.asarray(p_raw, dtype=float)))


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> Tuple[pd.DataFrame, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    idx = np.floor(p * bins).astype(int)
    idx = np.clip(idx, 0, bins - 1)
    rows = []
    ece = 0.0
    n = max(1, len(p))
    for b in range(bins):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            rows.append({"bin": b, "count": 0, "pred_mean": np.nan, "empirical_rate": np.nan, "abs_gap": np.nan})
            continue
        pm = float(p[m].mean())
        ym = float(y[m].mean())
        gap = abs(pm - ym)
        ece += (c / n) * gap
        rows.append({"bin": b, "count": c, "pred_mean": pm, "empirical_rate": ym, "abs_gap": gap})
    return pd.DataFrame(rows), float(ece)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, object]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    rel, ece = reliability_table(y=y, p=p, bins=10)
    return {
        "count": int(len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "pred_mean": float(p.mean()),
        "empirical_mean": float(y.mean()),
        "mean_gap_pred_minus_empirical": float(p.mean() - y.mean()),
        "ece_10": ece,
        "reliability_10": rel,
    }


def build_candidate_universe(
    kalshi_dir: Path,
    model_df: pd.DataFrame,
    entry_hour: int,
    entry_minute: int,
    trade_offset_days: int,
) -> pd.DataFrame:
    files = sorted(kalshi_dir.glob("KNYC_*.csv"))
    rows: List[Dict[str, object]] = []
    for fp in files:
        try:
            file_date = datetime.strptime(fp.stem.split("_")[-1], "%Y%m%d").date()
        except Exception:
            continue
        trade_date = file_date + timedelta(days=int(trade_offset_days))

        day_df = pd.read_csv(fp)
        day_df["timestamp"] = pd.to_datetime(day_df["timestamp"], utc=True, errors="coerce")
        day_df = day_df[day_df["timestamp"].notna()].sort_values("timestamp")
        if day_df.empty:
            continue
        bucket_cols = [c for c in day_df.columns if c != "timestamp"]
        if not bucket_cols:
            continue

        intervals = {c: parse_bucket_interval(normalize_label(c)) for c in bucket_cols}
        day_df, scale_mode = _normalize_day_prices(day_df, bucket_cols)
        day_df[bucket_cols] = day_df[bucket_cols].ffill()

        cutoff_st = datetime(trade_date.year, trade_date.month, trade_date.day, entry_hour, entry_minute, tzinfo=STOCKHOLM_TZ)
        cutoff_utc = cutoff_st.astimezone(UTC)
        entry_candidates = day_df[day_df["timestamp"] >= pd.Timestamp(cutoff_utc)]
        if entry_candidates.empty:
            continue
        entry_row = entry_candidates.iloc[0]
        entry_time_st = entry_row["timestamp"].to_pydatetime().astimezone(STOCKHOLM_TZ)

        model_row = get_model_row_for_entry_time(model_df=model_df, trade_date=trade_date, entry_time_stockholm=entry_time_st)
        if model_row is None:
            continue

        actual_tmax = int(model_row["tmax_truth"])
        pmf, tail_start, tail_prob = build_integer_pmf(model_row)

        for col in bucket_cols:
            interval = intervals[col]
            label = normalize_label(col)
            entry_yes = pd.to_numeric(entry_row[col], errors="coerce")
            if not np.isfinite(entry_yes):
                continue
            entry_yes = float(entry_yes)
            if not (0.0 <= entry_yes <= 100.0):
                continue
            market_yes = entry_yes / 100.0
            model_yes = float(np.clip(bucket_prob(interval, pmf, tail_start, tail_prob), 0.0, 1.0))
            outcome_yes = 1.0 if bucket_contains(interval, actual_tmax) else 0.0

            for side in ["YES", "NO"]:
                if side == "YES":
                    model_win = model_yes
                    market_win = market_yes
                    outcome_win = outcome_yes
                else:
                    model_win = 1.0 - model_yes
                    market_win = 1.0 - market_yes
                    outcome_win = 1.0 - outcome_yes

                rows.append(
                    {
                        "file_date": file_date.isoformat(),
                        "trade_date": trade_date.isoformat(),
                        "entry_time_stockholm": entry_time_st.isoformat(),
                        "bucket_label": label,
                        "side": side,
                        "model_win_prob_raw": float(model_win),
                        "market_win_prob": float(market_win),
                        "ev_raw": float(model_win - market_win),
                        "outcome_win": float(outcome_win),
                        "actual_tmax_f": int(actual_tmax),
                        "price_scale_mode": scale_mode,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Candidate universe is empty.")
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date
    return out


def pick_daily_trades(
    df: pd.DataFrame,
    prob_col: str,
    ev_col: str,
    min_prob: float,
    min_ev: float,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    eligible = df[(df[prob_col] >= min_prob) & (df[ev_col] >= min_ev)].copy()
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(
        ["trade_date", ev_col, prob_col, "market_win_prob", "bucket_label", "side"],
        ascending=[True, False, False, True, True, True],
    )
    return eligible.groupby("trade_date", as_index=False).head(1).reset_index(drop=True)


def settlement_backtest(trades: pd.DataFrame, start_balance: float, risk_fraction: float) -> Dict[str, float]:
    bal = float(start_balance)
    eq = [bal]
    wins = 0
    for _, r in trades.sort_values("trade_date").iterrows():
        stake = bal * risk_fraction
        price = float(r["market_win_prob"])
        if price <= 0.0 or price >= 1.0:
            eq.append(bal)
            continue
        shares = stake / price
        win = float(r["outcome_win"]) > 0.5
        pnl = shares * (1.0 - price) if win else -stake
        bal += pnl
        wins += int(win)
        eq.append(bal)
    peak = eq[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for v in eq:
        peak = max(peak, v)
        dd = peak - v
        max_dd = max(max_dd, dd)
        max_dd_pct = max(max_dd_pct, dd / peak if peak > 0 else 0.0)
    n = len(trades)
    return {
        "entries": int(n),
        "wins": int(wins),
        "losses": int(n - wins),
        "win_rate": float(wins / n) if n else float("nan"),
        "start_balance": float(start_balance),
        "final_balance": float(bal),
        "total_pnl": float(bal - start_balance),
        "max_drawdown_abs": float(max_dd),
        "max_drawdown_pct": float(max_dd_pct),
    }


def run_walkforward_oof(cands: pd.DataFrame, min_train_days: int, block_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.Series(cands["trade_date"]).dropna().unique().tolist())
    fold_rows = []
    pred_rows = []
    for start in range(min_train_days, len(dates), block_days):
        train_dates = set(dates[:start])
        test_dates = set(dates[start : start + block_days])
        if not test_dates:
            continue
        tr = cands[cands["trade_date"].isin(train_dates)]
        te = cands[cands["trade_date"].isin(test_dates)].copy()
        if tr.empty or te.empty:
            continue
        params = fit_platt(y=tr["outcome_win"].to_numpy(dtype=float), p_raw=tr["model_win_prob_raw"].to_numpy(dtype=float))
        te["model_win_prob_cal"] = apply_platt(te["model_win_prob_raw"].to_numpy(dtype=float), params=params)
        te["ev_cal"] = te["model_win_prob_cal"] - te["market_win_prob"]
        pred_rows.append(te)
        fold_rows.append(
            {
                "fold_start_date": min(test_dates).isoformat(),
                "fold_end_date": max(test_dates).isoformat(),
                "train_days": len(train_dates),
                "test_days": len(test_dates),
                "train_rows": int(len(tr)),
                "test_rows": int(len(te)),
                "a": float(params.a),
                "b": float(params.b),
            }
        )
    if not pred_rows:
        raise RuntimeError("No OOF folds produced.")
    return pd.DataFrame(fold_rows), pd.concat(pred_rows, ignore_index=True)


def main() -> int:
    args = parse_args()
    model_df = build_model_frame(
        eval_path=Path(args.distribution_eval_parquet),
        pred_path=Path(args.predictions_parquet),
    )
    model_df = model_df[pd.to_datetime(model_df["target_date_local"]).dt.year == 2025].copy()

    cands = build_candidate_universe(
        kalshi_dir=Path(args.kalshi_dir),
        model_df=model_df,
        entry_hour=int(args.entry_hour_stockholm),
        entry_minute=int(args.entry_minute_stockholm),
        trade_offset_days=int(args.trade_date_offset_days),
    )
    cands = cands.sort_values(["trade_date", "bucket_label", "side"]).reset_index(drop=True)

    holdout_start = date.fromisoformat(args.holdout_start)
    train_dev = cands[cands["trade_date"] < holdout_start].copy()
    holdout = cands[cands["trade_date"] >= holdout_start].copy()
    if train_dev.empty or holdout.empty:
        raise RuntimeError("Train/holdout split is empty. Adjust --holdout-start.")

    # Walk-forward OOF on development period.
    fold_df, oof_preds = run_walkforward_oof(
        cands=train_dev,
        min_train_days=int(args.oof_min_train_days),
        block_days=int(args.oof_block_days),
    )
    oof_metrics_raw = binary_metrics(
        y=oof_preds["outcome_win"].to_numpy(dtype=float),
        p=oof_preds["model_win_prob_raw"].to_numpy(dtype=float),
    )
    oof_metrics_cal = binary_metrics(
        y=oof_preds["outcome_win"].to_numpy(dtype=float),
        p=oof_preds["model_win_prob_cal"].to_numpy(dtype=float),
    )

    # Fit final trade-layer calibrator on full dev, score untouched holdout.
    final_params = fit_platt(
        y=train_dev["outcome_win"].to_numpy(dtype=float),
        p_raw=train_dev["model_win_prob_raw"].to_numpy(dtype=float),
    )
    holdout = holdout.copy()
    holdout["model_win_prob_cal"] = apply_platt(holdout["model_win_prob_raw"].to_numpy(dtype=float), params=final_params)
    holdout["ev_cal"] = holdout["model_win_prob_cal"] - holdout["market_win_prob"]

    holdout_metrics_raw = binary_metrics(
        y=holdout["outcome_win"].to_numpy(dtype=float),
        p=holdout["model_win_prob_raw"].to_numpy(dtype=float),
    )
    holdout_metrics_cal = binary_metrics(
        y=holdout["outcome_win"].to_numpy(dtype=float),
        p=holdout["model_win_prob_cal"].to_numpy(dtype=float),
    )

    # Daily selected trades: raw vs calibrated thresholds, same constraints.
    holdout_raw_trades = pick_daily_trades(
        df=holdout,
        prob_col="model_win_prob_raw",
        ev_col="ev_raw",
        min_prob=float(args.min_win_prob),
        min_ev=float(args.ev_threshold),
    )
    holdout_cal_trades = pick_daily_trades(
        df=holdout,
        prob_col="model_win_prob_cal",
        ev_col="ev_cal",
        min_prob=float(args.min_win_prob),
        min_ev=float(args.ev_threshold),
    )
    bt_raw = settlement_backtest(
        trades=holdout_raw_trades,
        start_balance=float(args.start_balance),
        risk_fraction=float(args.risk_fraction),
    )
    bt_cal = settlement_backtest(
        trades=holdout_cal_trades,
        start_balance=float(args.start_balance),
        risk_fraction=float(args.risk_fraction),
    )

    summary = {
        "config": {
            "entry_time_stockholm_min": f"{int(args.entry_hour_stockholm):02d}:{int(args.entry_minute_stockholm):02d}",
            "ev_threshold": float(args.ev_threshold),
            "min_win_prob": float(args.min_win_prob),
            "oof_min_train_days": int(args.oof_min_train_days),
            "oof_block_days": int(args.oof_block_days),
            "holdout_start": args.holdout_start,
            "holdout_end": str(max(holdout["trade_date"])),
        },
        "counts": {
            "candidate_rows_2025": int(len(cands)),
            "dev_rows": int(len(train_dev)),
            "holdout_rows": int(len(holdout)),
            "oof_rows": int(len(oof_preds)),
            "oof_folds": int(len(fold_df)),
        },
        "calibrator_final_params": {"a": float(final_params.a), "b": float(final_params.b)},
        "oof_dev_metrics_raw": {k: v for k, v in oof_metrics_raw.items() if k != "reliability_10"},
        "oof_dev_metrics_cal": {k: v for k, v in oof_metrics_cal.items() if k != "reliability_10"},
        "oof_dev_delta": {
            "logloss": float(oof_metrics_cal["logloss"] - oof_metrics_raw["logloss"]),
            "brier": float(oof_metrics_cal["brier"] - oof_metrics_raw["brier"]),
            "ece_10": float(oof_metrics_cal["ece_10"] - oof_metrics_raw["ece_10"]),
            "mean_gap_pred_minus_empirical": float(
                oof_metrics_cal["mean_gap_pred_minus_empirical"] - oof_metrics_raw["mean_gap_pred_minus_empirical"]
            ),
        },
        "holdout_metrics_raw": {k: v for k, v in holdout_metrics_raw.items() if k != "reliability_10"},
        "holdout_metrics_cal": {k: v for k, v in holdout_metrics_cal.items() if k != "reliability_10"},
        "holdout_delta": {
            "logloss": float(holdout_metrics_cal["logloss"] - holdout_metrics_raw["logloss"]),
            "brier": float(holdout_metrics_cal["brier"] - holdout_metrics_raw["brier"]),
            "ece_10": float(holdout_metrics_cal["ece_10"] - holdout_metrics_raw["ece_10"]),
            "mean_gap_pred_minus_empirical": float(
                holdout_metrics_cal["mean_gap_pred_minus_empirical"] - holdout_metrics_raw["mean_gap_pred_minus_empirical"]
            ),
        },
        "holdout_selected_trades_raw": bt_raw,
        "holdout_selected_trades_cal": bt_cal,
    }

    run_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"trade_layer_calibration_walkforward_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cands.to_csv(out_dir / "candidate_universe_2025.csv", index=False, encoding="utf-8")
    fold_df.to_csv(out_dir / "oof_folds.csv", index=False, encoding="utf-8")
    oof_preds.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    holdout.to_csv(out_dir / "holdout_predictions.csv", index=False, encoding="utf-8")
    holdout_raw_trades.to_csv(out_dir / "holdout_selected_trades_raw.csv", index=False, encoding="utf-8")
    holdout_cal_trades.to_csv(out_dir / "holdout_selected_trades_cal.csv", index=False, encoding="utf-8")
    oof_metrics_raw["reliability_10"].to_csv(out_dir / "oof_reliability_raw_10bins.csv", index=False, encoding="utf-8")
    oof_metrics_cal["reliability_10"].to_csv(out_dir / "oof_reliability_cal_10bins.csv", index=False, encoding="utf-8")
    holdout_metrics_raw["reliability_10"].to_csv(out_dir / "holdout_reliability_raw_10bins.csv", index=False, encoding="utf-8")
    holdout_metrics_cal["reliability_10"].to_csv(out_dir / "holdout_reliability_cal_10bins.csv", index=False, encoding="utf-8")

    print(f"summary_json: {out_dir / 'summary.json'}")
    print(f"candidate_universe_csv: {out_dir / 'candidate_universe_2025.csv'}")
    print(f"oof_folds_csv: {out_dir / 'oof_folds.csv'}")
    print(f"oof_predictions_csv: {out_dir / 'oof_predictions.csv'}")
    print(f"holdout_predictions_csv: {out_dir / 'holdout_predictions.csv'}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
