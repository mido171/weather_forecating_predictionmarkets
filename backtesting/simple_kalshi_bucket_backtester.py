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
from zoneinfo import ZoneInfo

UTC = timezone.utc
STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")


@dataclass(frozen=True)
class BucketInterval:
    label: str
    kind: str  # range | le | ge
    lo: Optional[int]
    hi: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Kalshi bucket backtester using E2/E4 peak+delta outputs.")
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
        help="Predictions parquet (from experiment run).",
    )
    parser.add_argument(
        "--distribution-eval-parquet",
        default=(
            r"D:\Ahmed\data\kalshi\Experiments\E2_KNYC\Experiment_set_1\E2\20260228T153836Z\predictions\distribution_eval_test.parquet"
        ),
        help="Distribution eval parquet (from experiment run).",
    )
    parser.add_argument("--start-date", default="2025-01-01", help="Inclusive start date YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2025-12-31", help="Inclusive end date YYYY-MM-DD.")
    parser.add_argument("--entry-hour-stockholm", type=int, default=19)
    parser.add_argument("--entry-minute-stockholm", type=int, default=0)
    parser.add_argument("--ev-threshold", type=float, default=0.15)
    parser.add_argument("--min-win-prob", type=float, default=0.65)
    parser.add_argument("--trade-date-offset-days", type=int, default=0)
    parser.add_argument("--start-balance", type=float, default=2700.0)
    parser.add_argument("--risk-fraction", type=float, default=0.04)
    parser.add_argument(
        "--risk-mode",
        choices=["static", "fractional_kelly"],
        default="static",
        help="Position sizing mode.",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.15,
        help="Fractional Kelly scalar used when --risk-mode=fractional_kelly.",
    )
    parser.add_argument("--output-dir", default=r"D:\Ahmed\data\kalshi\backtesting\results")
    return parser.parse_args()


def normalize_label(label: str) -> str:
    text = str(label)
    text = text.replace("Ã‚Â°", "°").replace("Â°", "°").replace("Âº", "°").replace("â€“", "-").replace("âˆ’", "-")
    return re.sub(r"\s+", " ", text).strip()


def parse_bucket_interval(label: str) -> BucketInterval:
    raw = normalize_label(label)
    low = raw.lower().replace("°f", "").replace("°", "")
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


def build_model_frame(eval_path: Path, pred_path: Path) -> pd.DataFrame:
    eval_df = pd.read_parquet(eval_path)
    pred_df = pd.read_parquet(pred_path)
    df = eval_df.merge(pred_df, on="row_index", how="left", suffixes=("_eval", "_pred"))
    ny = pd.to_datetime(df["target_date_local"]) + pd.to_timedelta(df["cutoff_minutes"], unit="m")
    ny = ny.dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
    st = ny.dt.tz_convert(STOCKHOLM_TZ)
    df["st_timestamp"] = st
    df["st_hour"] = st.dt.hour
    df["st_min"] = st.dt.minute
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


def compute_risk_fraction(
    *,
    risk_mode: str,
    static_risk_fraction: float,
    kelly_fraction: float,
    model_win_prob: float,
    entry_price_cents: float,
) -> Tuple[float, Optional[float]]:
    if risk_mode == "static":
        rf = max(0.0, float(static_risk_fraction))
        return min(rf, 1.0), None

    p = float(entry_price_cents) / 100.0
    q = float(model_win_prob)
    if p <= 0.0 or p >= 1.0:
        return 0.0, 0.0

    # Binary contract Kelly fraction for long position:
    # f* = (q - p) / (1 - p)
    full_kelly = (q - p) / (1.0 - p)
    full_kelly = min(max(full_kelly, 0.0), 1.0)
    frac = float(kelly_fraction) * full_kelly
    frac = min(max(frac, 0.0), 1.0)
    return frac, full_kelly


def run_backtest(args: argparse.Namespace) -> Tuple[Dict[str, object], pd.DataFrame]:
    kalshi_dir = Path(args.kalshi_dir)
    pred_path = Path(args.predictions_parquet)
    eval_path = Path(args.distribution_eval_parquet)

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
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

    model_df = build_model_frame(eval_path=eval_path, pred_path=pred_path)

    balance = float(args.start_balance)
    mtm_balance = float(args.start_balance)
    trade_rows: List[Dict[str, object]] = []
    scale_mode_counts: Dict[str, int] = {}

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
            trade_date.year, trade_date.month, trade_date.day, int(args.entry_hour_stockholm), int(args.entry_minute_stockholm), tzinfo=STOCKHOLM_TZ
        )
        cutoff_utc = cutoff_st.astimezone(UTC)
        entry_candidates = day_df[day_df["timestamp"] >= pd.Timestamp(cutoff_utc)]
        if entry_candidates.empty:
            continue
        entry_row = entry_candidates.iloc[0]
        exit_row = day_df.iloc[-1]
        entry_time_stockholm = entry_row["timestamp"].to_pydatetime().astimezone(STOCKHOLM_TZ)

        model_row = get_model_row_for_entry_time(
            model_df=model_df,
            trade_date=trade_date,
            entry_time_stockholm=entry_time_stockholm,
        )
        if model_row is None:
            continue
        pmf, tail_start, tail_prob = build_integer_pmf(model_row)
        actual_tmax = int(model_row["tmax_truth"])

        candidate_rows: List[Dict[str, object]] = []
        for col in bucket_cols:
            label = normalize_label(col)
            interval = parsed_intervals[col]

            entry_yes = pd.to_numeric(entry_row[col], errors="coerce")
            exit_yes = pd.to_numeric(exit_row[col], errors="coerce")
            if not np.isfinite(entry_yes) or not np.isfinite(exit_yes):
                continue
            entry_yes = float(entry_yes)
            exit_yes = float(exit_yes)
            if not (0.0 <= entry_yes <= 100.0 and 0.0 <= exit_yes <= 100.0):
                continue

            model_yes = float(bucket_prob(interval, pmf, tail_start, tail_prob))
            model_no = 1.0 - model_yes
            mkt_yes = entry_yes / 100.0
            mkt_no = 1.0 - mkt_yes
            ev_yes = model_yes - mkt_yes
            ev_no = model_no - mkt_no

            if model_yes >= float(args.min_win_prob) and ev_yes >= float(args.ev_threshold):
                candidate_rows.append(
                    {
                        "side": "YES",
                        "bucket_label": label,
                        "interval": interval,
                        "entry_price": entry_yes,
                        "exit_price": exit_yes,
                        "model_win": model_yes,
                        "bucket_win": mkt_yes,
                        "ev": ev_yes,
                    }
                )
            if model_no >= float(args.min_win_prob) and ev_no >= float(args.ev_threshold):
                no_entry = 100.0 - entry_yes
                no_exit = 100.0 - exit_yes
                if not (0.0 <= no_entry <= 100.0 and 0.0 <= no_exit <= 100.0):
                    continue
                candidate_rows.append(
                    {
                        "side": "NO",
                        "bucket_label": label,
                        "interval": interval,
                        "entry_price": no_entry,
                        "exit_price": no_exit,
                        "model_win": model_no,
                        "bucket_win": mkt_no,
                        "ev": ev_no,
                    }
                )

        if not candidate_rows:
            continue

        best = max(
            candidate_rows,
            key=lambda r: (float(r["ev"]), float(r["model_win"]), -float(r["entry_price"]), str(r["bucket_label"]), str(r["side"])),
        )
        entry_price = float(best["entry_price"])
        exit_price = float(best["exit_price"])
        if entry_price <= 0.0 or entry_price >= 100.0:
            continue
        risk_fraction_used, kelly_full_fraction = compute_risk_fraction(
            risk_mode=str(args.risk_mode),
            static_risk_fraction=float(args.risk_fraction),
            kelly_fraction=float(args.kelly_fraction),
            model_win_prob=float(best["model_win"]),
            entry_price_cents=entry_price,
        )
        stake = balance * risk_fraction_used
        if stake <= 0.0:
            continue
        shares = 0.0 if entry_price <= 0 else stake / (entry_price / 100.0)

        interval = best["interval"]
        event_yes_true = bucket_contains(interval, actual_tmax)
        settlement_win = event_yes_true if best["side"] == "YES" else (not event_yes_true)
        if settlement_win:
            pnl_settlement = shares * ((100.0 - entry_price) / 100.0)
        else:
            pnl_settlement = -stake
        pnl_mtm = shares * ((exit_price - entry_price) / 100.0)
        balance += pnl_settlement
        mtm_balance += pnl_mtm
        settlement_result = "W" if settlement_win else "L"

        trade_rows.append(
            {
                "file_date": file_date.isoformat(),
                "trade_date": trade_date.isoformat(),
                "entry_time_stockholm": entry_time_stockholm.isoformat(),
                "model_time_stockholm": pd.Timestamp(model_row["st_timestamp"]).to_pydatetime().isoformat(),
                "bucket_label": best["bucket_label"],
                "side": best["side"],
                "actual_tmax_f": actual_tmax,
                "model_win_pct": round(float(best["model_win"]) * 100.0, 6),
                "bucket_win_pct": round(float(best["bucket_win"]) * 100.0, 6),
                "ev_produced": round(float(best["ev"]), 6),
                "entry_price": round(entry_price, 6),
                "exit_price": round(exit_price, 6),
                "stake": round(stake, 6),
                "risk_fraction_used": round(float(risk_fraction_used), 8),
                "kelly_full_fraction": None if kelly_full_fraction is None else round(float(kelly_full_fraction), 8),
                "shares": round(shares, 6),
                "pnl_settlement": round(float(pnl_settlement), 6),
                "pnl_mark_to_market": round(float(pnl_mtm), 6),
                "balance_after_settlement": round(float(balance), 6),
                "balance_after_mark_to_market": round(float(mtm_balance), 6),
                "result": settlement_result,
                "settlement_win": bool(settlement_win),
                "price_scale_mode": scale_mode,
            }
        )

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        summary = {
            "files_considered": len(parsed_files),
            "entries": 0,
            "wins_settlement": 0,
            "losses_settlement": 0,
            "win_rate_settlement": None,
            "wins_mark_to_market": 0,
            "losses_mark_to_market": 0,
            "win_rate_mark_to_market": None,
            "final_balance_settlement": balance,
            "final_balance_mark_to_market": mtm_balance,
            "assumptions": {
                "ev_threshold": float(args.ev_threshold),
                "min_win_prob": float(args.min_win_prob),
                "entry_time_stockholm_min": f"{int(args.entry_hour_stockholm):02d}:{int(args.entry_minute_stockholm):02d}",
                "one_trade_per_day": True,
                "risk_mode": str(args.risk_mode),
                "risk_fraction": float(args.risk_fraction),
                "kelly_fraction": float(args.kelly_fraction),
                "pnl_rule_settlement": "buy side at entry price; if side resolves true payout 1.00 else 0.00",
            },
        }
        return summary, trades

    wins_settlement = int(trades["settlement_win"].sum())
    losses_settlement = int(len(trades) - wins_settlement)
    win_rate_settlement = float(wins_settlement / len(trades))
    wins_mtm = int((trades["pnl_mark_to_market"] > 0).sum())
    losses_mtm = int((trades["pnl_mark_to_market"] <= 0).sum())
    win_rate_mtm = float(wins_mtm / len(trades))
    summary = {
        "files_considered": len(parsed_files),
        "entries": int(len(trades)),
        "wins_settlement": wins_settlement,
        "losses_settlement": losses_settlement,
        "win_rate_settlement": win_rate_settlement,
        "wins_mark_to_market": wins_mtm,
        "losses_mark_to_market": losses_mtm,
        "win_rate_mark_to_market": win_rate_mtm,
        "start_balance": float(args.start_balance),
        "final_balance_settlement": float(balance),
        "final_balance_mark_to_market": float(mtm_balance),
        "total_pnl_settlement": float(trades["pnl_settlement"].sum()),
        "total_pnl_mark_to_market": float(trades["pnl_mark_to_market"].sum()),
        "mean_ev": float(trades["ev_produced"].mean()),
        "price_scale_mode_counts": scale_mode_counts,
        "assumptions": {
            "ev_threshold": float(args.ev_threshold),
            "min_win_prob": float(args.min_win_prob),
            "entry_time_stockholm_min": f"{int(args.entry_hour_stockholm):02d}:{int(args.entry_minute_stockholm):02d}",
            "one_trade_per_day": True,
            "selection_rule": "max EV then max model_win_pct then min entry_price",
            "risk_mode": str(args.risk_mode),
            "risk_fraction": float(args.risk_fraction),
            "kelly_fraction": float(args.kelly_fraction),
            "pnl_rule_settlement": "stake = f(balance, model_win, entry_price); binary settlement payout",
            "pnl_rule_mark_to_market": "stake = f(balance, model_win, entry_price); pnl = shares * (exit-entry)/100",
        },
    }
    return summary, trades


def main() -> int:
    args = parse_args()
    summary, trades = run_backtest(args)

    run_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"simple_kalshi_bucket_backtest_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    entries_path = out_dir / "entries.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    trades.to_csv(entries_path, index=False, encoding="utf-8")
    print(f"summary_json: {summary_path}")
    print(f"entries_csv: {entries_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
