from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from pipelines.quantile_knn_conformal.cdf_bucket_mapper import quantile_rows_to_integer_pmf

UTC = timezone.utc
STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")

QUANTILES = [0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99]


@dataclass(frozen=True)
class BucketInterval:
    label: str
    kind: str  # range | le | ge
    lo: Optional[int]
    hi: Optional[int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest Kalshi using ML quantile model outputs.")
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
    p.add_argument("--output-dir", default=r"D:\Ahmed\data\kalshi\backtesting\results")
    return p.parse_args()


def normalize_label(label: str) -> str:
    text = str(label)
    text = text.replace("Ãƒâ€šÃ‚Â°", "Â°").replace("Ã‚Â°", "Â°").replace("Ã‚Âº", "Â°").replace("Ã¢â‚¬â€œ", "-").replace("Ã¢Ë†â€™", "-")
    return re.sub(r"\s+", " ", text).strip()


def parse_bucket_interval(label: str) -> BucketInterval:
    raw = normalize_label(label)
    low = raw.lower().replace("Â°f", "").replace("Â°", "").replace("°f", "").replace("°", "")
    low = re.sub(r"\s+", " ", low).strip()

    m = re.search(r"(-?\d+)\s*-\s*(-?\d+)", low) or re.search(r"(-?\d+)\s*to\s*(-?\d+)", low)
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
        return bool(interval.lo <= tmax <= interval.hi)  # type: ignore[operator]
    if interval.kind == "le":
        return bool(tmax <= interval.hi)  # type: ignore[operator]
    if interval.kind == "ge":
        return bool(tmax >= interval.lo)  # type: ignore[operator]
    return False


def bucket_prob(interval: BucketInterval, pmf: Dict[int, float]) -> float:
    out = 0.0
    for t, p in pmf.items():
        if interval.kind == "range" and interval.lo <= t <= interval.hi:  # type: ignore[operator]
            out += p
        elif interval.kind == "le" and t <= interval.hi:  # type: ignore[operator]
            out += p
        elif interval.kind == "ge" and t >= interval.lo:  # type: ignore[operator]
            out += p
    return float(out)


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


def build_model_frame(pred_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(pred_path)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["st_timestamp"] = pd.to_datetime(df["valid_time_stockholm"], errors="coerce")
    df["y_tmax"] = pd.to_numeric(df["y_tmax"], errors="coerce")
    return df[df["st_timestamp"].notna() & df["y_tmax"].notna()].copy()


def get_model_row_for_entry_time(model_df: pd.DataFrame, trade_date: date, entry_time_stockholm: datetime) -> Optional[pd.Series]:
    rows = model_df.loc[
        (model_df["target_date_local"] == trade_date)
        & (model_df["st_timestamp"] <= pd.Timestamp(entry_time_stockholm))
    ]
    if rows.empty:
        return None
    return rows.sort_values("st_timestamp").iloc[-1]


def build_integer_pmf_from_ml_quantiles(model_row: pd.Series) -> Dict[int, float]:
    q_df = pd.DataFrame(
        {
            f"q_{q:.3f}": [float(model_row[f"ml_q_{q:.3f}"])]
            for q in QUANTILES
        }
    )
    pmf_df = quantile_rows_to_integer_pmf(q_df, QUANTILES, support_min=0, support_max=120)
    pmf: Dict[int, float] = {}
    for t in range(0, 121):
        pmf[t] = float(pmf_df.iloc[0][f"p_int_{t}"])
    return pmf


def run_backtest(args: argparse.Namespace) -> Tuple[Dict[str, object], pd.DataFrame]:
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

        parsed_intervals: Dict[str, BucketInterval] = {}
        for col in bucket_cols:
            parsed_intervals[col] = parse_bucket_interval(normalize_label(col))

        day_df, scale_mode = _normalize_day_prices(day_df, bucket_cols)
        scale_mode_counts[scale_mode] = int(scale_mode_counts.get(scale_mode, 0) + 1)
        day_df[bucket_cols] = day_df[bucket_cols].ffill()

        cutoff_st = datetime(
            trade_date.year, trade_date.month, trade_date.day,
            int(args.entry_hour_stockholm), int(args.entry_minute_stockholm),
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

            model_yes = float(bucket_prob(interval, pmf))
            model_no = float(1.0 - model_yes)
            market_yes = float(entry_yes / 100.0)
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
            "assumptions": {
                "ev_threshold": float(args.ev_threshold),
                "min_win_prob": float(args.min_win_prob),
                "entry_time_stockholm_min": f"{int(args.entry_hour_stockholm):02d}:{int(args.entry_minute_stockholm):02d}",
                "one_trade_per_day": True,
                "risk_fraction": float(args.risk_fraction),
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
        "total_pnl": float(trades["pnl_settlement"].sum()),
        "mean_ev": float(trades["ev_produced"].mean()),
        "price_scale_mode_counts": scale_mode_counts,
        "assumptions": {
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


def main() -> int:
    args = parse_args()
    summary, trades = run_backtest(args)

    run_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"ml_quantile_kalshi_backtest_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    trades.to_csv(out_dir / "entries.csv", index=False, encoding="utf-8")
    print(f"summary_json: {out_dir / 'summary.json'}")
    print(f"entries_csv: {out_dir / 'entries.csv'}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
