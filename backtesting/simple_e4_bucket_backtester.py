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
    parser = argparse.ArgumentParser(description="Simple E4 bucket backtester for Polymarket CSVs.")
    parser.add_argument(
        "--polymarket-dir",
        default=r"D:\Ahmed\data\early_peak_data\backtest_data\polymarket",
        help="Path to daily Polymarket CSV files.",
    )
    parser.add_argument(
        "--predictions-parquet",
        default=(
            r"D:\Ahmed\data\early_peak_data\results\experimentation_sets\Experiment_set_1\E4\20260228T081521Z\predictions\predictions_test.parquet"
        ),
        help="E4 predictions_test.parquet path.",
    )
    parser.add_argument(
        "--distribution-eval-parquet",
        default=(
            r"D:\Ahmed\data\early_peak_data\results\experimentation_sets\Experiment_set_1\E4\20260228T081521Z\predictions\distribution_eval_test.parquet"
        ),
        help="E4 distribution_eval_test.parquet path.",
    )
    parser.add_argument("--start-date", default="2025-12-01", help="Inclusive file date start YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2025-12-31", help="Inclusive file date end YYYY-MM-DD.")
    parser.add_argument("--entry-hour-stockholm", type=int, default=19, help="Entry hour Stockholm (>= this hour).")
    parser.add_argument("--entry-minute-stockholm", type=int, default=0, help="Entry minute Stockholm.")
    parser.add_argument("--ev-threshold", type=float, default=0.15, help="Minimum EV threshold.")
    parser.add_argument("--min-win-prob", type=float, default=0.65, help="Minimum model win probability.")
    parser.add_argument(
        "--file-date-to-trade-date-offset-days",
        type=int,
        default=-1,
        help=(
            "trade_date = file_date + offset_days. "
            "Default -1 for current CSV windows that end around 06:59 NY on file date."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="backtesting/results",
        help="Output directory root.",
    )
    return parser.parse_args()


def normalize_label(label: str) -> str:
    text = str(label).replace("Â°", "°").replace("º", "°").replace("−", "-")
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

    m = re.search(r"(-?\d+)\s*(?:or)?\s*(?:below|less)", low)
    if m:
        return BucketInterval(label=raw, kind="le", lo=None, hi=int(m.group(1)))

    m = re.search(r"(-?\d+)\s*(?:or)?\s*(?:above|higher|more)", low)
    if m:
        return BucketInterval(label=raw, kind="ge", lo=int(m.group(1)), hi=None)

    raise ValueError(f"Unable to parse bucket label: {label}")


def build_model_frame(eval_path: Path, pred_path: Path) -> pd.DataFrame:
    eval_df = pd.read_parquet(eval_path)
    pred_df = pd.read_parquet(pred_path)
    df = eval_df.merge(pred_df, on="row_index", how="left", suffixes=("_eval", "_pred"))

    ny = pd.to_datetime(df["target_date_local"]) + pd.to_timedelta(df["cutoff_minutes"], unit="m")
    ny = ny.dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
    st = ny.dt.tz_convert(STOCKHOLM_TZ)
    df["st_hour"] = st.dt.hour
    df["st_min"] = st.dt.minute
    return df


def get_model_row_for_trade_date(
    model_df: pd.DataFrame,
    trade_date: date,
    entry_hour_stockholm: int,
    entry_minute_stockholm: int,
) -> Optional[pd.Series]:
    m = (
        (pd.to_datetime(model_df["target_date_local"]).dt.date == trade_date)
        & (model_df["st_hour"] == entry_hour_stockholm)
        & (model_df["st_min"] == entry_minute_stockholm)
    )
    rows = model_df.loc[m]
    if rows.empty:
        return None
    return rows.sort_values("row_index").iloc[0]


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
            # Tail is coarse, include full tail for this simplified backtest.
            p += tail_prob
        return float(p)

    return float("nan")


def row_as_numeric_prices(row: pd.Series, col: str) -> Optional[float]:
    v = pd.to_numeric(row[col], errors="coerce")
    if not np.isfinite(v):
        return None
    return float(v)


def run_backtest(args: argparse.Namespace) -> Tuple[Dict[str, object], pd.DataFrame]:
    poly_dir = Path(args.polymarket_dir)
    pred_path = Path(args.predictions_parquet)
    eval_path = Path(args.distribution_eval_parquet)

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)

    files = sorted(poly_dir.glob("*.csv"))
    files = [p for p in files if start_date <= date.fromisoformat(p.stem) <= end_date]

    model_df = build_model_frame(eval_path=eval_path, pred_path=pred_path)

    records: List[Dict[str, object]] = []

    for fp in files:
        file_date = date.fromisoformat(fp.stem)
        trade_date = file_date + timedelta(days=int(args.file_date_to_trade_date_offset_days))
        cutoff_st = datetime(
            trade_date.year,
            trade_date.month,
            trade_date.day,
            int(args.entry_hour_stockholm),
            int(args.entry_minute_stockholm),
            tzinfo=STOCKHOLM_TZ,
        )
        cutoff_utc = cutoff_st.astimezone(UTC)

        model_row = get_model_row_for_trade_date(
            model_df=model_df,
            trade_date=trade_date,
            entry_hour_stockholm=int(args.entry_hour_stockholm),
            entry_minute_stockholm=int(args.entry_minute_stockholm),
        )
        if model_row is None:
            continue

        pmf, tail_start, tail_prob = build_integer_pmf(model_row)
        actual_tmax = int(model_row["tmax_truth"])

        day_df = pd.read_csv(fp)
        day_df["timestamp"] = pd.to_datetime(day_df["timestamp"], utc=True, errors="coerce")
        day_df = day_df[day_df["timestamp"].notna()].sort_values("timestamp")
        if day_df.empty:
            continue

        entry_candidates = day_df[day_df["timestamp"] >= pd.Timestamp(cutoff_utc)]
        if entry_candidates.empty:
            continue
        entry_row = entry_candidates.iloc[0]
        eod_row = day_df.iloc[-1]

        day_candidates: List[Dict[str, object]] = []
        yes_cols = [c for c in day_df.columns if c.endswith("__YES")]
        for ycol in yes_cols:
            base = ycol[:-5]
            ncol = base + "__NO"
            if ncol not in day_df.columns:
                continue
            try:
                interval = parse_bucket_interval(base)
            except Exception:
                continue

            entry_yes = row_as_numeric_prices(entry_row, ycol)
            entry_no = row_as_numeric_prices(entry_row, ncol)
            eod_yes = row_as_numeric_prices(eod_row, ycol)
            eod_no = row_as_numeric_prices(eod_row, ncol)
            if entry_yes is None or entry_no is None or eod_yes is None or eod_no is None:
                continue
            if not (0.0 <= entry_yes <= 100.0 and 0.0 <= entry_no <= 100.0 and 0.0 <= eod_yes <= 100.0 and 0.0 <= eod_no <= 100.0):
                continue

            p_yes = bucket_prob(interval, pmf, tail_start, tail_prob)
            p_no = 1.0 - p_yes

            # YES side
            model_win_yes = p_yes
            bucket_win_yes = entry_yes / 100.0
            ev_yes = model_win_yes - bucket_win_yes
            if model_win_yes >= float(args.min_win_prob) and ev_yes >= float(args.ev_threshold):
                result_yes = "W" if eod_yes > entry_yes else "L"
                day_candidates.append(
                    {
                        "file_date": file_date.isoformat(),
                        "trade_date": trade_date.isoformat(),
                        "bucket_label": normalize_label(base),
                        "side": "YES",
                        "model_win_pct": round(model_win_yes * 100.0, 6),
                        "bucket_win_pct": round(bucket_win_yes * 100.0, 6),
                        "ev_produced": round(ev_yes, 6),
                        "entry_price": round(entry_yes, 6),
                        "entry_time_stockholm": entry_row["timestamp"].to_pydatetime().astimezone(STOCKHOLM_TZ).isoformat(),
                        "eod_price": round(eod_yes, 6),
                        "actual_tmax_f": actual_tmax,
                        "result": result_yes,
                    }
                )

            # NO side
            model_win_no = p_no
            bucket_win_no = entry_no / 100.0
            ev_no = model_win_no - bucket_win_no
            if model_win_no >= float(args.min_win_prob) and ev_no >= float(args.ev_threshold):
                result_no = "W" if eod_no > entry_no else "L"
                day_candidates.append(
                    {
                        "file_date": file_date.isoformat(),
                        "trade_date": trade_date.isoformat(),
                        "bucket_label": normalize_label(base),
                        "side": "NO",
                        "model_win_pct": round(model_win_no * 100.0, 6),
                        "bucket_win_pct": round(bucket_win_no * 100.0, 6),
                        "ev_produced": round(ev_no, 6),
                        "entry_price": round(entry_no, 6),
                        "entry_time_stockholm": entry_row["timestamp"].to_pydatetime().astimezone(STOCKHOLM_TZ).isoformat(),
                        "eod_price": round(eod_no, 6),
                        "actual_tmax_f": actual_tmax,
                        "result": result_no,
                    }
                )

        if day_candidates:
            # Enforce exactly one trade per day by selecting the strongest edge candidate.
            best = max(
                day_candidates,
                key=lambda r: (
                    float(r["ev_produced"]),
                    float(r["model_win_pct"]),
                    -float(r["entry_price"]),
                    str(r["bucket_label"]),
                    str(r["side"]),
                ),
            )
            records.append(best)

    trades = pd.DataFrame(records)
    if trades.empty:
        summary = {
            "files_considered": len(files),
            "entries": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "assumptions": {
                "ev_threshold": float(args.ev_threshold),
                "min_win_prob": float(args.min_win_prob),
                "entry_time_stockholm_gte": f"{int(args.entry_hour_stockholm):02d}:{int(args.entry_minute_stockholm):02d}",
                "trade_date_mapping_offset_days": int(args.file_date_to_trade_date_offset_days),
                "exit_price": "end-of-day row in same CSV",
                "one_trade_per_day": True,
            },
        }
        return summary, trades

    wins = int((trades["result"] == "W").sum())
    losses = int((trades["result"] == "L").sum())
    win_rate = float(wins / len(trades))
    summary = {
        "files_considered": len(files),
        "entries": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "mean_ev": float(trades["ev_produced"].mean()),
        "median_ev": float(trades["ev_produced"].median()),
        "assumptions": {
            "ev_threshold": float(args.ev_threshold),
            "min_win_prob": float(args.min_win_prob),
            "entry_time_stockholm_gte": f"{int(args.entry_hour_stockholm):02d}:{int(args.entry_minute_stockholm):02d}",
            "trade_date_mapping_offset_days": int(args.file_date_to_trade_date_offset_days),
            "entry_row_rule": "first row with timestamp >= cutoff",
            "exit_price": "end-of-day row in same CSV",
            "result_rule": "W if eod_price > entry_price else L",
            "one_trade_per_day": True,
            "selection_rule": "max EV, then max model_win_pct, then min entry_price",
        },
    }
    return summary, trades


def main() -> int:
    args = parse_args()
    summary, trades = run_backtest(args)

    run_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"simple_e4_bucket_backtest_{run_tag}"
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
