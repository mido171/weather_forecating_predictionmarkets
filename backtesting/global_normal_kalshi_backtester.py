from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from backtesting.ml_quantile_kalshi_backtester import (
    _normalize_day_prices,
    bucket_contains,
    build_model_frame,
    get_model_row_for_entry_time,
    normalize_label,
    parse_bucket_interval,
)

UTC = timezone.utc
STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")


@dataclass(frozen=True)
class DailySnapshot:
    target_date_local: date
    mu_point: float
    y: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest Kalshi using point forecast + global Normal (bias + sigma) fitted leakage-free."
    )
    p.add_argument("--kalshi-dir", default=r"D:\Ahmed\data\kalshi\kalshi_history\kxhighny_2025")
    p.add_argument(
        "--predictions-parquet",
        default=r"D:\Ahmed\data\kalshi\Experiments\point_foreacast\E1\08_predictions\test_predictions_2024_2025.parquet",
    )
    p.add_argument(
        "--calibration-parquet",
        default=None,
        help="Optional parquet used only for fitting global bias/sigma. If omitted, uses --predictions-parquet.",
    )
    p.add_argument("--start-date", default="2025-01-01")
    p.add_argument("--end-date", default="2025-12-31")
    p.add_argument("--entry-hour-stockholm", type=int, default=19)
    p.add_argument("--entry-minute-stockholm", type=int, default=0)
    p.add_argument("--ev-threshold", type=float, default=0.15)
    p.add_argument("--min-win-prob", type=float, default=0.65)
    p.add_argument("--start-balance", type=float, default=2700.0)
    p.add_argument("--risk-fraction", type=float, default=0.04)
    p.add_argument("--calibration-start", default="2024-01-01")
    p.add_argument("--calibration-end", default="2024-12-31")
    p.add_argument("--support-min", type=int, default=0)
    p.add_argument("--support-max", type=int, default=120)
    p.add_argument("--output-dir", default=r"D:\Ahmed\data\kalshi\backtesting\results")
    return p.parse_args()


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_integer_pmf(mu: float, sigma: float, support_min: int, support_max: int) -> Dict[int, float]:
    sigma = max(float(sigma), 1e-6)
    pmf: Dict[int, float] = {}
    for t in range(support_min, support_max + 1):
        lo = (t - 0.5 - mu) / sigma
        hi = (t + 0.5 - mu) / sigma
        p = max(_normal_cdf(hi) - _normal_cdf(lo), 0.0)
        pmf[t] = float(p)
    total = float(sum(pmf.values()))
    if total <= 0.0:
        center = int(round(mu))
        center = min(max(center, support_min), support_max)
        for t in pmf.keys():
            pmf[t] = 1.0 if t == center else 0.0
        return pmf
    inv = 1.0 / total
    for t in pmf.keys():
        pmf[t] = float(pmf[t] * inv)
    return pmf


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


def _build_daily_snapshots(model_df: pd.DataFrame, entry_hour: int, entry_minute: int) -> List[DailySnapshot]:
    rows: List[DailySnapshot] = []
    for target_date, g in model_df.groupby("target_date_local"):
        g2 = g.copy().sort_values("st_timestamp")
        st_ts = pd.to_datetime(g2["st_timestamp"], errors="coerce")
        cutoff = datetime(target_date.year, target_date.month, target_date.day, entry_hour, entry_minute, tzinfo=STOCKHOLM_TZ)
        elig = g2[st_ts >= pd.Timestamp(cutoff)]
        row = elig.iloc[0] if not elig.empty else g2.iloc[-1]
        mu = float(pd.to_numeric(row.get("ml_q_0.500"), errors="coerce"))
        y = float(pd.to_numeric(row.get("y_tmax"), errors="coerce"))
        if not np.isfinite(mu) or not np.isfinite(y):
            continue
        rows.append(DailySnapshot(target_date_local=target_date, mu_point=mu, y=y))
    rows.sort(key=lambda r: r.target_date_local)
    return rows


def _fit_global_bias_sigma(
    snapshots: List[DailySnapshot],
    cal_start: date,
    cal_end: date,
) -> Tuple[float, float, int]:
    cal = [s for s in snapshots if cal_start <= s.target_date_local <= cal_end]
    if not cal:
        raise ValueError("No calibration rows found in requested calibration range.")
    mu = np.asarray([s.mu_point for s in cal], dtype=float)
    y = np.asarray([s.y for s in cal], dtype=float)
    resid = y - mu
    bias = float(np.mean(resid))
    resid_centered = y - (mu + bias)
    sigma = float(np.std(resid_centered, ddof=1)) if resid_centered.size > 1 else float(np.std(resid_centered))
    sigma = max(sigma, 0.5)
    return bias, sigma, int(len(cal))


def run_backtest(args: argparse.Namespace) -> Tuple[Dict[str, object], pd.DataFrame]:
    kalshi_dir = Path(args.kalshi_dir)
    model_df = build_model_frame(Path(args.predictions_parquet))
    cal_df = build_model_frame(Path(args.calibration_parquet)) if args.calibration_parquet else model_df
    snapshots = _build_daily_snapshots(cal_df, int(args.entry_hour_stockholm), int(args.entry_minute_stockholm))
    cal_start = date.fromisoformat(args.calibration_start)
    cal_end = date.fromisoformat(args.calibration_end)
    bias, sigma, n_cal = _fit_global_bias_sigma(snapshots, cal_start, cal_end)

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

    balance = float(args.start_balance)
    trade_rows: List[Dict[str, object]] = []
    scale_mode_counts: Dict[str, int] = {}

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
        entry_time_stockholm = entry_row["timestamp"].to_pydatetime().astimezone(STOCKHOLM_TZ)

        model_row = get_model_row_for_entry_time(model_df, trade_date, entry_time_stockholm)
        if model_row is None:
            continue
        mu_raw = float(pd.to_numeric(model_row.get("ml_q_0.500"), errors="coerce"))
        y_today = float(pd.to_numeric(model_row.get("y_tmax"), errors="coerce"))
        if not np.isfinite(mu_raw) or not np.isfinite(y_today):
            continue
        mu_adj = mu_raw + bias
        pmf = _normal_integer_pmf(mu_adj, sigma, int(args.support_min), int(args.support_max))
        actual_tmax = int(round(y_today))

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
            market_yes = float(entry_yes / 100.0)
            model_yes = float(_bucket_prob(interval, pmf))
            model_no = float(1.0 - model_yes)
            market_no = float(1.0 - market_yes)
            ev_yes = float(model_yes - market_yes)
            ev_no = float(model_no - market_no)

            if model_yes >= float(args.min_win_prob) and ev_yes >= float(args.ev_threshold):
                candidate_rows.append(
                    {
                        "side": "YES",
                        "bucket_label": label,
                        "interval": interval,
                        "entry_price": entry_yes,
                        "model_win": model_yes,
                        "bucket_win": market_yes,
                        "ev": ev_yes,
                    }
                )
            no_entry = float(100.0 - entry_yes)
            if model_no >= float(args.min_win_prob) and ev_no >= float(args.ev_threshold) and 0.0 <= no_entry <= 100.0:
                candidate_rows.append(
                    {
                        "side": "NO",
                        "bucket_label": label,
                        "interval": interval,
                        "entry_price": no_entry,
                        "model_win": model_no,
                        "bucket_win": market_no,
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
        stake = float(balance * float(args.risk_fraction))
        if entry_price <= 0.0 or entry_price >= 100.0 or stake <= 0.0:
            continue
        shares = stake / (entry_price / 100.0)
        event_yes_true = bucket_contains(best["interval"], actual_tmax)
        settlement_win = bool(event_yes_true if best["side"] == "YES" else (not event_yes_true))
        pnl = shares * ((100.0 - entry_price) / 100.0) if settlement_win else -stake
        balance += float(pnl)

        trade_rows.append(
            {
                "trade_date": trade_date.isoformat(),
                "entry_time_stockholm": entry_time_stockholm.isoformat(),
                "bucket_label": best["bucket_label"],
                "side": best["side"],
                "actual_tmax_f": actual_tmax,
                "mu_raw": round(mu_raw, 6),
                "mu_adj": round(mu_adj, 6),
                "global_bias": round(bias, 8),
                "global_sigma": round(sigma, 8),
                "model_win_pct": round(float(best["model_win"]) * 100.0, 6),
                "bucket_win_pct": round(float(best["bucket_win"]) * 100.0, 6),
                "ev_produced": round(float(best["ev"]), 6),
                "entry_price": round(entry_price, 6),
                "stake": round(stake, 6),
                "shares": round(float(shares), 6),
                "pnl_settlement": round(float(pnl), 6),
                "balance_after": round(float(balance), 6),
                "result": "W" if settlement_win else "L",
                "price_scale_mode": scale_mode,
            }
        )

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        summary = {
            "files_considered": len(parsed_files),
            "entries": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "start_balance": float(args.start_balance),
            "final_balance": float(balance),
            "total_pnl": 0.0,
            "mean_ev": None,
            "global_params": {
                "calibration_start": cal_start.isoformat(),
                "calibration_end": cal_end.isoformat(),
                "calibration_n_days": n_cal,
                "bias": bias,
                "sigma": sigma,
            },
        }
        return summary, trades

    wins = int((trades["result"] == "W").sum())
    losses = int((trades["result"] == "L").sum())
    summary = {
        "files_considered": len(parsed_files),
        "entries": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / len(trades)),
        "start_balance": float(args.start_balance),
        "final_balance": float(balance),
        "total_pnl": float(balance - float(args.start_balance)),
        "mean_ev": float(pd.to_numeric(trades["ev_produced"], errors="coerce").mean()),
        "mean_model_win": float(pd.to_numeric(trades["model_win_pct"], errors="coerce").mean() / 100.0),
        "price_scale_mode_counts": scale_mode_counts,
        "global_params": {
            "calibration_start": cal_start.isoformat(),
            "calibration_end": cal_end.isoformat(),
            "calibration_n_days": n_cal,
            "bias": float(bias),
            "sigma": float(sigma),
        },
        "assumptions": {
            "method": "global Normal from point forecast (mu_adj = ml_q_0.5 + bias, sigma global)",
            "ev_threshold": float(args.ev_threshold),
            "min_win_prob": float(args.min_win_prob),
            "entry_time_stockholm_min": f"{int(args.entry_hour_stockholm):02d}:{int(args.entry_minute_stockholm):02d}",
            "one_trade_per_day": True,
            "selection_rule": "max EV then max model_win_pct then min entry_price",
            "risk_fraction": float(args.risk_fraction),
            "pnl_rule_settlement": "binary settlement payout on chosen side",
        },
    }
    return summary, trades


def main() -> None:
    args = parse_args()
    summary, trades = run_backtest(args)

    run_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"global_normal_backtest_{run_ts}"
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
