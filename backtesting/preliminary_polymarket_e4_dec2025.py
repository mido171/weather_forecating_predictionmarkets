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

UTC = timezone.utc


@dataclass(frozen=True)
class BucketInterval:
    label: str
    kind: str  # "range" | "le" | "ge"
    lo: Optional[int]
    hi: Optional[int]


@dataclass
class TradeResult:
    file_date: date
    trade_date: date
    cutoff_utc: datetime
    entry_time_utc: Optional[datetime]
    entry_time_stockholm: Optional[datetime]
    side: Optional[str]
    bucket_label: Optional[str]
    actual_tmax_f: Optional[int]
    model_p_bucket: Optional[float]
    model_p_win: Optional[float]
    market_prob_side: Optional[float]
    edge: Optional[float]
    yes_price_c: Optional[float]
    no_price_c: Optional[float]
    entry_price_c: Optional[float]
    stake: float
    shares: Optional[float]
    win: Optional[bool]
    pnl: float
    balance_after: float
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preliminary Dec-2025 Polymarket backtest using E4 peak+delta probabilities."
    )
    parser.add_argument(
        "--polymarket-dir",
        default=r"D:\Ahmed\data\early_peak_data\backtest_data\polymarket",
        help="Directory containing daily Polymarket CSV files.",
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
    parser.add_argument("--start-date", default="2025-12-01", help="Inclusive file date start (YYYY-MM-DD).")
    parser.add_argument("--end-date", default="2025-12-31", help="Inclusive file date end (YYYY-MM-DD).")
    parser.add_argument(
        "--entry-hour-stockholm",
        type=int,
        default=19,
        help="Entry allowed at or after this Stockholm hour on trade_date.",
    )
    parser.add_argument("--entry-minute-stockholm", type=int, default=0, help="Stockholm minute for entry cutoff.")
    parser.add_argument("--edge-threshold", type=float, default=0.15, help="Required model edge vs market probability.")
    parser.add_argument("--min-win-prob", type=float, default=0.65, help="Required model win probability.")
    parser.add_argument("--start-balance", type=float, default=2700.0, help="Starting balance.")
    parser.add_argument("--risk-fraction", type=float, default=0.04, help="Risked fraction of current balance per trade.")
    parser.add_argument(
        "--file-date-to-trade-date-offset-days",
        type=int,
        default=-1,
        help=(
            "trade_date = file_date + offset_days. "
            "Default -1 because the supplied CSV windows are mostly aligned to the prior NY local day."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="backtesting/results",
        help="Output root directory for summary and trade log.",
    )
    return parser.parse_args()


def normalize_label_text(text: str) -> str:
    return (
        str(text)
        .replace("Â°", "°")
        .replace("º", "°")
        .replace("−", "-")
        .strip()
    )


def parse_bucket_interval(label: str) -> BucketInterval:
    raw = normalize_label_text(label)
    low = raw.lower()
    # Remove degree marks to simplify parsing.
    cleaned = low.replace("°f", "f").replace("°", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    m_range = re.search(r"(-?\d+)\s*-\s*(-?\d+)", cleaned)
    if m_range:
        lo = int(m_range.group(1))
        hi = int(m_range.group(2))
        if hi < lo:
            lo, hi = hi, lo
        return BucketInterval(label=raw, kind="range", lo=lo, hi=hi)

    m_le = re.search(r"(-?\d+)\s*(?:f)?\s*or\s*(?:below|less)", cleaned)
    if m_le:
        k = int(m_le.group(1))
        return BucketInterval(label=raw, kind="le", lo=None, hi=k)

    m_ge = re.search(r"(-?\d+)\s*(?:f)?\s*or\s*(?:above|higher|more)", cleaned)
    if m_ge:
        k = int(m_ge.group(1))
        return BucketInterval(label=raw, kind="ge", lo=k, hi=None)

    raise ValueError(f"Unable to parse bucket label: {label}")


def build_model_rows(eval_path: Path, pred_path: Path) -> pd.DataFrame:
    eval_df = pd.read_parquet(eval_path)
    pred_df = pd.read_parquet(pred_path)
    df = eval_df.merge(pred_df, on="row_index", how="left", suffixes=("_eval", "_pred"))

    ny = pd.to_datetime(df["target_date_local"]) + pd.to_timedelta(df["cutoff_minutes"], unit="m")
    ny = ny.dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
    st = ny.dt.tz_convert("Europe/Stockholm")
    df["cutoff_stockholm_hour"] = st.dt.hour
    df["cutoff_stockholm_minute"] = st.dt.minute
    return df


def get_model_row_for_cutoff(
    model_df: pd.DataFrame,
    trade_date: date,
    entry_hour_stockholm: int,
    entry_minute_stockholm: int,
) -> Optional[pd.Series]:
    m = (
        (pd.to_datetime(model_df["target_date_local"]).dt.date == trade_date)
        & (model_df["cutoff_stockholm_hour"] == entry_hour_stockholm)
        & (model_df["cutoff_stockholm_minute"] == entry_minute_stockholm)
    )
    rows = model_df.loc[m]
    if rows.empty:
        return None
    # There should be exactly one row for (date, cutoff), keep deterministic.
    return rows.sort_values("row_index").iloc[0]


def build_tmax_pmf_from_row(row: pd.Series) -> Tuple[Dict[int, float], int, float]:
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


def bucket_probability(interval: BucketInterval, pmf: Dict[int, float], tail_start: int, tail_prob: float) -> float:
    p = 0.0
    if interval.kind == "range":
        assert interval.lo is not None and interval.hi is not None
        for t, pt in pmf.items():
            if interval.lo <= t <= interval.hi:
                p += pt
        # No tail add unless range upper includes tail bucket start.
        if interval.hi >= tail_start:
            p += tail_prob
        return float(p)

    if interval.kind == "le":
        assert interval.hi is not None
        for t, pt in pmf.items():
            if t <= interval.hi:
                p += pt
        # tail always above finite values here, so no tail add.
        return float(p)

    if interval.kind == "ge":
        assert interval.lo is not None
        for t, pt in pmf.items():
            if t >= interval.lo:
                p += pt
        if tail_start >= interval.lo:
            p += tail_prob
        elif interval.lo > tail_start:
            # Tail bin is >= tail_start; with only tail aggregate available we cannot split further.
            # Conservative approximation for preliminary backtest: include full tail.
            p += tail_prob
        return float(p)

    raise ValueError(f"Unsupported interval kind: {interval.kind}")


def select_candidate_after_cutoff(
    day_df: pd.DataFrame,
    pmf: Dict[int, float],
    tail_start: int,
    tail_prob: float,
    cutoff_utc: datetime,
    edge_threshold: float,
    min_win_prob: float,
) -> Optional[Dict[str, object]]:
    work = day_df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work[work["timestamp"].notna()].sort_values("timestamp")
    work = work[work["timestamp"] >= pd.Timestamp(cutoff_utc)]
    if work.empty:
        return None

    # Bucket base names from *__YES columns.
    yes_cols = [c for c in work.columns if c.endswith("__YES")]
    if not yes_cols:
        return None

    parsed: List[Tuple[str, str, str, BucketInterval]] = []
    for ycol in yes_cols:
        base = ycol[:-5]
        ncol = base + "__NO"
        if ncol not in work.columns:
            continue
        try:
            interval = parse_bucket_interval(base)
        except Exception:
            continue
        parsed.append((base, ycol, ncol, interval))

    if not parsed:
        return None

    for _, row in work.iterrows():
        minute_candidates: List[Dict[str, object]] = []
        for base, ycol, ncol, interval in parsed:
            yes_c = pd.to_numeric(row[ycol], errors="coerce")
            no_c = pd.to_numeric(row[ncol], errors="coerce")
            if not np.isfinite(yes_c) or not np.isfinite(no_c):
                continue
            if yes_c < 0.0 or yes_c > 100.0 or no_c < 0.0 or no_c > 100.0:
                continue

            p_bucket = bucket_probability(interval, pmf, tail_start, tail_prob)

            p_mkt_yes = float(yes_c) / 100.0
            p_mkt_no = float(no_c) / 100.0
            p_win_yes = float(p_bucket)
            p_win_no = float(1.0 - p_bucket)

            edge_yes = p_win_yes - p_mkt_yes
            if p_win_yes >= min_win_prob and edge_yes >= edge_threshold:
                minute_candidates.append(
                    {
                        "timestamp": row["timestamp"].to_pydatetime(),
                        "side": "YES",
                        "bucket_label": base,
                        "p_bucket": p_bucket,
                        "p_win": p_win_yes,
                        "market_prob_side": p_mkt_yes,
                        "edge": edge_yes,
                        "yes_c": float(yes_c),
                        "no_c": float(no_c),
                        "entry_c": float(yes_c),
                    }
                )

            edge_no = p_win_no - p_mkt_no
            if p_win_no >= min_win_prob and edge_no >= edge_threshold:
                minute_candidates.append(
                    {
                        "timestamp": row["timestamp"].to_pydatetime(),
                        "side": "NO",
                        "bucket_label": base,
                        "p_bucket": p_bucket,
                        "p_win": p_win_no,
                        "market_prob_side": p_mkt_no,
                        "edge": edge_no,
                        "yes_c": float(yes_c),
                        "no_c": float(no_c),
                        "entry_c": float(no_c),
                    }
                )

        if minute_candidates:
            minute_candidates.sort(
                key=lambda c: (float(c["edge"]), float(c["p_win"])),
                reverse=True,
            )
            return minute_candidates[0]

    return None


def bucket_event_true(bucket_label: str, actual_tmax_f: int) -> bool:
    interval = parse_bucket_interval(bucket_label)
    if interval.kind == "range":
        assert interval.lo is not None and interval.hi is not None
        return interval.lo <= actual_tmax_f <= interval.hi
    if interval.kind == "le":
        assert interval.hi is not None
        return actual_tmax_f <= interval.hi
    if interval.kind == "ge":
        assert interval.lo is not None
        return actual_tmax_f >= interval.lo
    return False


def load_daily_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Keep original columns; enforce numeric parsing only where needed later.
    return df


def run_backtest(args: argparse.Namespace) -> Tuple[Dict[str, object], pd.DataFrame]:
    polymarket_dir = Path(args.polymarket_dir)
    pred_path = Path(args.predictions_parquet)
    eval_path = Path(args.distribution_eval_parquet)

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)

    files = sorted(polymarket_dir.glob("*.csv"))
    files = [p for p in files if start_date <= date.fromisoformat(p.stem) <= end_date]

    model_df = build_model_rows(eval_path=eval_path, pred_path=pred_path)

    stockholm_tz = datetime.now().astimezone().tzinfo
    # Force explicit zone names for deterministic conversion.
    from zoneinfo import ZoneInfo

    stockholm = ZoneInfo("Europe/Stockholm")

    balance = float(args.start_balance)
    risk_fraction = float(args.risk_fraction)

    trade_rows: List[TradeResult] = []

    for f in files:
        file_date = date.fromisoformat(f.stem)
        trade_date = file_date + timedelta(days=int(args.file_date_to_trade_date_offset_days))

        cutoff_local = datetime(
            trade_date.year,
            trade_date.month,
            trade_date.day,
            int(args.entry_hour_stockholm),
            int(args.entry_minute_stockholm),
            tzinfo=stockholm,
        )
        cutoff_utc = cutoff_local.astimezone(UTC)

        model_row = get_model_row_for_cutoff(
            model_df=model_df,
            trade_date=trade_date,
            entry_hour_stockholm=int(args.entry_hour_stockholm),
            entry_minute_stockholm=int(args.entry_minute_stockholm),
        )
        stake = balance * risk_fraction

        if model_row is None:
            trade_rows.append(
                TradeResult(
                    file_date=file_date,
                    trade_date=trade_date,
                    cutoff_utc=cutoff_utc,
                    entry_time_utc=None,
                    entry_time_stockholm=None,
                    side=None,
                    bucket_label=None,
                    actual_tmax_f=None,
                    model_p_bucket=None,
                    model_p_win=None,
                    market_prob_side=None,
                    edge=None,
                    yes_price_c=None,
                    no_price_c=None,
                    entry_price_c=None,
                    stake=stake,
                    shares=None,
                    win=None,
                    pnl=0.0,
                    balance_after=balance,
                    note="missing_model_row",
                )
            )
            continue

        pmf, tail_start, tail_prob = build_tmax_pmf_from_row(model_row)
        actual_tmax_f = int(model_row["tmax_truth"])

        day_df = load_daily_csv(f)
        candidate = select_candidate_after_cutoff(
            day_df=day_df,
            pmf=pmf,
            tail_start=tail_start,
            tail_prob=tail_prob,
            cutoff_utc=cutoff_utc,
            edge_threshold=float(args.edge_threshold),
            min_win_prob=float(args.min_win_prob),
        )

        if candidate is None:
            trade_rows.append(
                TradeResult(
                    file_date=file_date,
                    trade_date=trade_date,
                    cutoff_utc=cutoff_utc,
                    entry_time_utc=None,
                    entry_time_stockholm=None,
                    side=None,
                    bucket_label=None,
                    actual_tmax_f=actual_tmax_f,
                    model_p_bucket=None,
                    model_p_win=None,
                    market_prob_side=None,
                    edge=None,
                    yes_price_c=None,
                    no_price_c=None,
                    entry_price_c=None,
                    stake=stake,
                    shares=None,
                    win=None,
                    pnl=0.0,
                    balance_after=balance,
                    note="no_entry_signal",
                )
            )
            continue

        side = str(candidate["side"])
        bucket_label = str(candidate["bucket_label"])
        entry_price_c = float(candidate["entry_c"])
        yes_price_c = float(candidate["yes_c"])
        no_price_c = float(candidate["no_c"])
        model_p_bucket = float(candidate["p_bucket"])
        model_p_win = float(candidate["p_win"])
        market_prob_side = float(candidate["market_prob_side"])
        edge = float(candidate["edge"])
        entry_time_utc = candidate["timestamp"]
        assert isinstance(entry_time_utc, datetime)
        entry_time_stockholm = entry_time_utc.astimezone(stockholm)

        if entry_price_c <= 0.0 or entry_price_c > 100.0:
            trade_rows.append(
                TradeResult(
                    file_date=file_date,
                    trade_date=trade_date,
                    cutoff_utc=cutoff_utc,
                    entry_time_utc=entry_time_utc,
                    entry_time_stockholm=entry_time_stockholm,
                    side=side,
                    bucket_label=bucket_label,
                    actual_tmax_f=actual_tmax_f,
                    model_p_bucket=model_p_bucket,
                    model_p_win=model_p_win,
                    market_prob_side=market_prob_side,
                    edge=edge,
                    yes_price_c=yes_price_c,
                    no_price_c=no_price_c,
                    entry_price_c=entry_price_c,
                    stake=stake,
                    shares=None,
                    win=None,
                    pnl=0.0,
                    balance_after=balance,
                    note="invalid_entry_price",
                )
            )
            continue

        shares = stake / (entry_price_c / 100.0)
        event_true = bucket_event_true(bucket_label=bucket_label, actual_tmax_f=actual_tmax_f)
        win = event_true if side == "YES" else (not event_true)
        if win:
            pnl = shares * (1.0 - (entry_price_c / 100.0))
        else:
            pnl = -stake
        balance += pnl

        trade_rows.append(
            TradeResult(
                file_date=file_date,
                trade_date=trade_date,
                cutoff_utc=cutoff_utc,
                entry_time_utc=entry_time_utc,
                entry_time_stockholm=entry_time_stockholm,
                side=side,
                bucket_label=bucket_label,
                actual_tmax_f=actual_tmax_f,
                model_p_bucket=model_p_bucket,
                model_p_win=model_p_win,
                market_prob_side=market_prob_side,
                edge=edge,
                yes_price_c=yes_price_c,
                no_price_c=no_price_c,
                entry_price_c=entry_price_c,
                stake=stake,
                shares=shares,
                win=bool(win),
                pnl=float(pnl),
                balance_after=float(balance),
                note="trade",
            )
        )

    trade_df = pd.DataFrame([t.__dict__ for t in trade_rows])

    entered = trade_df[trade_df["note"] == "trade"].copy()
    wins = int((entered["win"] == True).sum()) if not entered.empty else 0  # noqa: E712
    losses = int((entered["win"] == False).sum()) if not entered.empty else 0  # noqa: E712
    n_trades = int(len(entered))
    win_rate = float(wins / n_trades) if n_trades > 0 else float("nan")
    total_pnl = float(entered["pnl"].sum()) if n_trades > 0 else 0.0
    ending_balance = float(args.start_balance) + total_pnl
    roi = (ending_balance / float(args.start_balance) - 1.0) if float(args.start_balance) > 0 else float("nan")

    summary: Dict[str, object] = {
        "run_generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "mode": "preliminary",
        "assumptions": {
            "model": "E4 peak+delta composed PMF",
            "entry_time_rule": f"timestamp >= {int(args.entry_hour_stockholm):02d}:{int(args.entry_minute_stockholm):02d} Europe/Stockholm",
            "file_date_to_trade_date_offset_days": int(args.file_date_to_trade_date_offset_days),
            "edge_threshold": float(args.edge_threshold),
            "min_win_prob": float(args.min_win_prob),
            "stake_rule": f"stake = {float(args.risk_fraction):.4f} * current_balance",
            "fees_slippage": "not included",
            "trade_selection": "earliest eligible minute after cutoff; if multiple, max edge then max model win prob",
            "one_trade_per_file": True,
        },
        "input": {
            "polymarket_dir": str(polymarket_dir),
            "predictions_parquet": str(pred_path),
            "distribution_eval_parquet": str(eval_path),
            "file_date_range": [str(start_date), str(end_date)],
        },
        "counts": {
            "files_considered": int(len(files)),
            "trades_executed": n_trades,
            "wins": wins,
            "losses": losses,
            "no_entry_or_skipped": int(len(trade_df) - n_trades),
        },
        "performance": {
            "start_balance": float(args.start_balance),
            "ending_balance": ending_balance,
            "total_pnl": total_pnl,
            "roi": roi,
            "win_rate": win_rate,
            "meets_win_rate_rule": bool((not math.isnan(win_rate)) and (win_rate >= float(args.min_win_prob))),
        },
    }
    return summary, trade_df


def main() -> int:
    args = parse_args()
    summary, trade_df = run_backtest(args)

    out_root = Path(args.output_dir)
    run_dir = out_root / f"prelim_poly_e4_dec2025_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trade_csv = run_dir / "trades.csv"
    summary_json = run_dir / "summary.json"
    summary_txt = run_dir / "summary.txt"

    trade_df.to_csv(trade_csv, index=False, encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    perf = summary["performance"]
    counts = summary["counts"]
    lines = [
        "Preliminary Polymarket E4 Backtest (Dec 2025)",
        f"files_considered: {counts['files_considered']}",
        f"trades_executed: {counts['trades_executed']}",
        f"wins: {counts['wins']}",
        f"losses: {counts['losses']}",
        f"win_rate: {perf['win_rate']}",
        f"start_balance: {perf['start_balance']}",
        f"ending_balance: {perf['ending_balance']}",
        f"total_pnl: {perf['total_pnl']}",
        f"roi: {perf['roi']}",
        f"meets_win_rate_rule: {perf['meets_win_rate_rule']}",
        "",
        f"trades_csv: {trade_csv}",
        f"summary_json: {summary_json}",
    ]
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
