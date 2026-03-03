from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from zoneinfo import ZoneInfo

from backtesting.ml_quantile_kalshi_backtester import (
    _normalize_day_prices,
    bucket_contains,
    build_integer_pmf_from_ml_quantiles,
    build_model_frame,
    get_model_row_for_entry_time,
    normalize_label,
    parse_bucket_interval,
)

UTC = timezone.utc
STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")


@dataclass(frozen=True)
class SideRecord:
    trade_date: date
    entry_time_stockholm: str
    bucket_label: str
    side: str
    actual_tmax_f: int
    raw_model_p: float
    market_p: float
    raw_ev: float
    entry_price: float
    result_int: int
    price_scale_mode: str


@dataclass(frozen=True)
class DayPacket:
    trade_date: date
    side_records: Sequence[SideRecord]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Leakage-free policy-conditional recalibration backtest (walk-forward isotonic)."
    )
    p.add_argument("--kalshi-dir", default=r"D:\Ahmed\data\kalshi\kalshi_history\kxhighny_2025")
    p.add_argument(
        "--predictions-parquet",
        default=r"D:\Ahmed\data\kalshi\Experiments\point_foreacast\E1\08_predictions\test_predictions_2024_2025.parquet",
    )
    p.add_argument("--start-date", default="2025-01-01")
    p.add_argument("--end-date", default="2025-12-31")
    p.add_argument("--entry-hour-stockholm", type=int, default=19)
    p.add_argument("--entry-minute-stockholm", type=int, default=0)
    p.add_argument("--ev-threshold", type=float, default=0.15)
    p.add_argument("--min-win-prob", type=float, default=0.65)
    p.add_argument("--start-balance", type=float, default=2700.0)
    p.add_argument("--risk-fraction", type=float, default=0.04)
    p.add_argument("--calibration-window", type=int, default=365)
    p.add_argument("--calibration-min-samples", type=int, default=60)
    p.add_argument("--output-dir", default=r"D:\Ahmed\data\kalshi\backtesting\results")
    return p.parse_args()


def _bucket_prob(interval, pmf: Dict[int, float]) -> float:
    out = 0.0
    for t, p in pmf.items():
        if interval.kind == "range" and interval.lo <= t <= interval.hi:  # type: ignore[operator]
            out += p
        elif interval.kind == "le" and t <= interval.hi:  # type: ignore[operator]
            out += p
        elif interval.kind == "ge" and t >= interval.lo:  # type: ignore[operator]
            out += p
    return float(out)


def _build_day_packets(args: argparse.Namespace) -> List[DayPacket]:
    kalshi_dir = Path(args.kalshi_dir)
    model_df = build_model_frame(Path(args.predictions_parquet))

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)

    files = sorted(kalshi_dir.glob("*.csv"))
    parsed_files: List[Tuple[Path, date]] = []
    for fp in files:
        token = fp.stem.split("_")[-1]
        try:
            d = datetime.strptime(token, "%Y%m%d").date()
        except ValueError:
            continue
        if start_date <= d <= end_date:
            parsed_files.append((fp, d))

    out: List[DayPacket] = []
    for fp, trade_date in parsed_files:
        day_df = pd.read_csv(fp)
        day_df["timestamp"] = pd.to_datetime(day_df["timestamp"], utc=True, errors="coerce")
        day_df = day_df[day_df["timestamp"].notna()].sort_values("timestamp")
        if day_df.empty:
            continue
        bucket_cols = [c for c in day_df.columns if c != "timestamp"]
        if not bucket_cols:
            continue
        parsed_intervals = {col: parse_bucket_interval(normalize_label(col)) for col in bucket_cols}

        day_df, scale_mode = _normalize_day_prices(day_df, bucket_cols)
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
        entry_time_stockholm = entry_row["timestamp"].to_pydatetime().astimezone(STOCKHOLM_TZ)

        model_row = get_model_row_for_entry_time(model_df, trade_date, entry_time_stockholm)
        if model_row is None:
            continue
        pmf = build_integer_pmf_from_ml_quantiles(model_row)
        actual_tmax = int(round(float(model_row["y_tmax"])))

        sides: List[SideRecord] = []
        for col in bucket_cols:
            interval = parsed_intervals[col]
            label = normalize_label(col)
            entry_yes = pd.to_numeric(entry_row[col], errors="coerce")
            if not np.isfinite(entry_yes):
                continue
            entry_yes = float(entry_yes)
            if not (0.0 <= entry_yes <= 100.0):
                continue
            market_yes = float(entry_yes / 100.0)
            model_yes = float(_bucket_prob(interval, pmf))
            yes_true = bool(bucket_contains(interval, actual_tmax))
            yes_ev = float(model_yes - market_yes)
            sides.append(
                SideRecord(
                    trade_date=trade_date,
                    entry_time_stockholm=entry_time_stockholm.isoformat(),
                    bucket_label=label,
                    side="YES",
                    actual_tmax_f=actual_tmax,
                    raw_model_p=model_yes,
                    market_p=market_yes,
                    raw_ev=yes_ev,
                    entry_price=entry_yes,
                    result_int=1 if yes_true else 0,
                    price_scale_mode=scale_mode,
                )
            )

            no_price = float(100.0 - entry_yes)
            if 0.0 <= no_price <= 100.0:
                model_no = float(1.0 - model_yes)
                market_no = float(1.0 - market_yes)
                no_ev = float(model_no - market_no)
                sides.append(
                    SideRecord(
                        trade_date=trade_date,
                        entry_time_stockholm=entry_time_stockholm.isoformat(),
                        bucket_label=label,
                        side="NO",
                        actual_tmax_f=actual_tmax,
                        raw_model_p=model_no,
                        market_p=market_no,
                        raw_ev=no_ev,
                        entry_price=no_price,
                        result_int=0 if yes_true else 1,
                        price_scale_mode=scale_mode,
                    )
                )
        if sides:
            out.append(DayPacket(trade_date=trade_date, side_records=sides))
    out.sort(key=lambda x: x.trade_date)
    return out


def _fit_isotonic(history: Sequence[Tuple[float, int]]) -> Optional[IsotonicRegression]:
    if not history:
        return None
    x = np.asarray([p for p, _ in history], dtype=float)
    y = np.asarray([r for _, r in history], dtype=float)
    if x.size < 2 or len(np.unique(y)) < 2:
        return None
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(x, y)
    return iso


def _predict_calibrated(iso: Optional[IsotonicRegression], p_raw: float) -> float:
    if iso is None:
        return float(p_raw)
    p = float(iso.predict(np.asarray([p_raw], dtype=float))[0])
    return min(max(p, 0.0), 1.0)


def run_backtest(args: argparse.Namespace) -> Tuple[Dict[str, object], pd.DataFrame]:
    day_packets = _build_day_packets(args)
    files_considered = len(day_packets)

    balance = float(args.start_balance)
    trade_rows: List[Dict[str, object]] = []
    scale_mode_counts: Dict[str, int] = {}
    history_same_rule: List[Tuple[float, int]] = []

    for packet in day_packets:
        hist_window = history_same_rule[-int(args.calibration_window) :]
        iso = _fit_isotonic(hist_window) if len(hist_window) >= int(args.calibration_min_samples) else None

        candidates = []
        for rec in packet.side_records:
            p_cal = _predict_calibrated(iso, rec.raw_model_p)
            ev_cal = float(p_cal - rec.market_p)
            if p_cal >= float(args.min_win_prob) and ev_cal >= float(args.ev_threshold):
                candidates.append((rec, p_cal, ev_cal))

        if candidates:
            best = max(
                candidates,
                key=lambda x: (float(x[2]), float(x[1]), -float(x[0].entry_price), str(x[0].bucket_label), str(x[0].side)),
            )
            rec, p_cal, ev_cal = best
            stake = float(balance * float(args.risk_fraction))
            entry_price = float(rec.entry_price)
            if entry_price > 0.0 and entry_price < 100.0 and stake > 0.0:
                shares = stake / (entry_price / 100.0)
                settlement_win = bool(rec.result_int == 1)
                pnl = shares * ((100.0 - entry_price) / 100.0) if settlement_win else -stake
                balance += float(pnl)
                scale_mode_counts[rec.price_scale_mode] = int(scale_mode_counts.get(rec.price_scale_mode, 0) + 1)
                trade_rows.append(
                    {
                        "trade_date": rec.trade_date.isoformat(),
                        "entry_time_stockholm": rec.entry_time_stockholm,
                        "bucket_label": rec.bucket_label,
                        "side": rec.side,
                        "actual_tmax_f": rec.actual_tmax_f,
                        "raw_model_win_pct": round(float(rec.raw_model_p) * 100.0, 6),
                        "cal_model_win_pct": round(float(p_cal) * 100.0, 6),
                        "bucket_win_pct": round(float(rec.market_p) * 100.0, 6),
                        "raw_ev": round(float(rec.raw_ev), 6),
                        "cal_ev": round(float(ev_cal), 6),
                        "entry_price": round(float(entry_price), 6),
                        "stake": round(float(stake), 6),
                        "shares": round(float(shares), 6),
                        "pnl_settlement": round(float(pnl), 6),
                        "balance_after": round(float(balance), 6),
                        "result": "W" if settlement_win else "L",
                        "calibrator_hist_size": len(hist_window),
                        "price_scale_mode": rec.price_scale_mode,
                    }
                )

        # Update calibration history with same-rule raw-policy candidates from the resolved day.
        for rec in packet.side_records:
            if rec.raw_model_p >= float(args.min_win_prob) and rec.raw_ev >= float(args.ev_threshold):
                history_same_rule.append((float(rec.raw_model_p), int(rec.result_int)))

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        summary = {
            "files_considered": files_considered,
            "entries": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "start_balance": float(args.start_balance),
            "final_balance": float(balance),
            "total_pnl": 0.0,
            "mean_cal_ev": None,
            "mean_raw_ev": None,
            "assumptions": {
                "method": "walk-forward isotonic recalibration on same-rule prior candidates",
                "ev_threshold": float(args.ev_threshold),
                "min_win_prob": float(args.min_win_prob),
                "entry_time_stockholm_min": f"{int(args.entry_hour_stockholm):02d}:{int(args.entry_minute_stockholm):02d}",
                "one_trade_per_day": True,
                "risk_fraction": float(args.risk_fraction),
                "calibration_window": int(args.calibration_window),
                "calibration_min_samples": int(args.calibration_min_samples),
            },
        }
        return summary, trades

    wins = int((trades["result"] == "W").sum())
    losses = int((trades["result"] == "L").sum())
    summary = {
        "files_considered": files_considered,
        "entries": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / len(trades)),
        "start_balance": float(args.start_balance),
        "final_balance": float(balance),
        "total_pnl": float(balance - float(args.start_balance)),
        "mean_cal_ev": float(pd.to_numeric(trades["cal_ev"], errors="coerce").mean()),
        "mean_raw_ev": float(pd.to_numeric(trades["raw_ev"], errors="coerce").mean()),
        "mean_raw_model_win": float(pd.to_numeric(trades["raw_model_win_pct"], errors="coerce").mean() / 100.0),
        "mean_cal_model_win": float(pd.to_numeric(trades["cal_model_win_pct"], errors="coerce").mean() / 100.0),
        "realized_win_rate": float(wins / len(trades)),
        "price_scale_mode_counts": scale_mode_counts,
        "assumptions": {
            "method": "walk-forward isotonic recalibration on same-rule prior candidates",
            "ev_threshold": float(args.ev_threshold),
            "min_win_prob": float(args.min_win_prob),
            "entry_time_stockholm_min": f"{int(args.entry_hour_stockholm):02d}:{int(args.entry_minute_stockholm):02d}",
            "one_trade_per_day": True,
            "selection_rule": "max calibrated EV then max calibrated model_win then min entry_price",
            "risk_fraction": float(args.risk_fraction),
            "pnl_rule_settlement": "binary settlement payout on chosen side",
            "calibration_window": int(args.calibration_window),
            "calibration_min_samples": int(args.calibration_min_samples),
            "history_update_rule": "append prior-day candidates passing raw same-rule thresholds",
        },
    }
    return summary, trades


def main() -> None:
    args = parse_args()
    summary, trades = run_backtest(args)

    run_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"policy_recal_isotonic_backtest_{run_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    entries_path = out_dir / "entries.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    trades.to_csv(entries_path, index=False)

    print(f"Summary written to: {summary_path}")
    print(f"Entries written to: {entries_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

