from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


REPO = Path(__file__).resolve().parents[2]

MODEL_DIR = REPO / "artifacts" / "experiments" / "KMIA" / "early_maxout_strategy" / "B6" / "B6_EXP20_GAM_RESIDUAL"
PREDS_PATH = MODEL_DIR / "preds_test.parquet"
FEATURES_LIST_PATH = MODEL_DIR / "features.json"
FEATURES_PATH = REPO / "cache" / "hit1830_v6_features.parquet"
KALSHI_DIR = REPO / "data" / "kalshi_backtest_data"

OUT_DIR = REPO / "backtests" / "kmia_kalshi" / "b6_exp20_20251101_20251231"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = date(2025, 11, 1)
END_DATE = date(2025, 12, 31)

EDGE_PROB = 0.15  # 15 percentage points edge
RISK_FRACTION = 0.035
START_BALANCE = 2700.0

STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm") if ZoneInfo else None


@dataclass
class BucketInterval:
    label: str
    lower: float
    upper: float


@dataclass
class TradeResult:
    date: date
    cutoff_utc: datetime
    model_prob: float
    model_yes: bool
    bucket_label: Optional[str]
    bucket_price_at_cutoff: Optional[float]
    threshold_price: Optional[float]
    entry_time: Optional[datetime]
    entry_price: Optional[float]
    shares: Optional[float]
    win: Optional[bool]
    pnl: float
    balance_after: float
    tmax_full: Optional[float]
    note: str


def parse_bucket_interval(label: str) -> BucketInterval:
    cleaned = label.replace("\u00b0", "").strip().lower()
    # Range like "76 to 77" or "76-77"
    import re

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)", cleaned)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        lower = a - 0.5
        upper = b + 0.5
        return BucketInterval(label=label, lower=lower, upper=upper)

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:or\s+below|below)", cleaned)
    if m:
        k = float(m.group(1))
        return BucketInterval(label=label, lower=float("-inf"), upper=k + 0.5)

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:or\s+above|above)", cleaned)
    if m:
        k = float(m.group(1))
        return BucketInterval(label=label, lower=k - 0.5, upper=float("inf"))

    raise ValueError(f"Unable to parse bucket label: {label}")


def compute_cutoff_utc(day: date) -> datetime:
    if STOCKHOLM_TZ is None:
        raise RuntimeError("ZoneInfo not available; cannot compute cutoff.")
    local_dt = datetime(day.year, day.month, day.day, 18, 30, tzinfo=STOCKHOLM_TZ)
    return local_dt.astimezone(timezone.utc)


def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(PREDS_PATH)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df = df[(df["target_date_local"] >= START_DATE) & (df["target_date_local"] <= END_DATE)]
    return df


def load_tmax() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH, columns=["target_date_local", "tmax_full"])
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df = df[(df["target_date_local"] >= START_DATE) & (df["target_date_local"] <= END_DATE)]
    return df


def read_kalshi_day(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def price_at_or_before(df: pd.DataFrame, cutoff: datetime, bucket: str) -> Optional[float]:
    sub = df[df["timestamp"] <= cutoff][bucket].dropna()
    if sub.empty:
        return None
    return float(sub.iloc[-1])


def first_price_at_or_below(df: pd.DataFrame, cutoff: datetime, bucket: str, threshold: float) -> Tuple[Optional[datetime], Optional[float]]:
    sub = df[df["timestamp"] >= cutoff][["timestamp", bucket]].dropna()
    if sub.empty:
        return None, None
    hit = sub[sub[bucket] <= threshold]
    if hit.empty:
        return None, None
    row = hit.iloc[0]
    return row["timestamp"].to_pydatetime(), float(row[bucket])


def main() -> None:
    feature_list = json.loads(FEATURES_LIST_PATH.read_text(encoding="utf-8"))
    preds = load_predictions()
    tmax_df = load_tmax()
    tmax_map = dict(zip(tmax_df["target_date_local"], tmax_df["tmax_full"]))

    trades: List[TradeResult] = []
    balance = START_BALANCE

    missing_days = []
    no_pred_days = []

    for _, row in preds.iterrows():
        day = row["target_date_local"]
        p_cal = float(row.get("p_cal", row.get("p_raw")))
        model_yes = bool(p_cal >= 0.5)
        cutoff_utc = pd.to_datetime(row.get("cutoff_utc"), utc=True).to_pydatetime()

        # sanity check cutoff (not used further)
        try:
            computed_cutoff = compute_cutoff_utc(day)
            if abs((computed_cutoff - cutoff_utc).total_seconds()) > 60:
                cutoff_note = f"cutoff_mismatch:{computed_cutoff.isoformat()}"
            else:
                cutoff_note = ""
        except Exception:
            cutoff_note = ""

        file_path = KALSHI_DIR / f"KMIA_{day.strftime('%Y%m%d')}.csv"
        if not file_path.exists():
            missing_days.append(str(day))
            trades.append(TradeResult(day, cutoff_utc, p_cal, model_yes, None, None, None, None, None, None, None, 0.0, balance, tmax_map.get(day), "missing_kalshi_file"))
            continue

        if day not in tmax_map:
            no_pred_days.append(str(day))
            trades.append(TradeResult(day, cutoff_utc, p_cal, model_yes, None, None, None, None, None, None, None, 0.0, balance, None, "missing_tmax"))
            continue

        df = read_kalshi_day(file_path)
        bucket_cols = [c for c in df.columns if c != "timestamp"]

        # state at cutoff for each bucket
        bucket_prices = {}
        for col in bucket_cols:
            price = price_at_or_before(df, cutoff_utc, col)
            if price is not None and not math.isnan(price):
                bucket_prices[col] = price

        if not bucket_prices:
            trades.append(TradeResult(day, cutoff_utc, p_cal, model_yes, None, None, None, None, None, None, None, 0.0, balance, tmax_map.get(day), "no_prices_at_cutoff"))
            continue

        # Choose bucket with highest YES price at cutoff
        bucket_label, bucket_price = max(bucket_prices.items(), key=lambda kv: kv[1])

        if not model_yes:
            trades.append(TradeResult(day, cutoff_utc, p_cal, model_yes, bucket_label, bucket_price, None, None, None, None, None, 0.0, balance, tmax_map.get(day), "model_no"))
            continue

        threshold_price = (p_cal - EDGE_PROB) * 100.0
        if threshold_price <= 0:
            trades.append(TradeResult(day, cutoff_utc, p_cal, model_yes, bucket_label, bucket_price, threshold_price, None, None, None, None, 0.0, balance, tmax_map.get(day), "threshold_leq_zero"))
            continue

        # Entry check
        entry_time = None
        entry_price = None
        if bucket_price <= threshold_price:
            entry_time = cutoff_utc
            entry_price = bucket_price
        else:
            entry_time, entry_price = first_price_at_or_below(df, cutoff_utc, bucket_label, threshold_price)

        if entry_time is None or entry_price is None:
            trades.append(TradeResult(day, cutoff_utc, p_cal, model_yes, bucket_label, bucket_price, threshold_price, None, None, None, None, 0.0, balance, tmax_map.get(day), "no_entry"))
            continue

        # Determine win
        interval = parse_bucket_interval(bucket_label)
        tmax = float(tmax_map[day])
        win = (tmax > interval.lower) and (tmax <= interval.upper)

        stake = balance * RISK_FRACTION
        if entry_price <= 0:
            trades.append(TradeResult(day, cutoff_utc, p_cal, model_yes, bucket_label, bucket_price, threshold_price, entry_time, entry_price, None, win, 0.0, balance, tmax, "invalid_entry_price"))
            continue

        shares = stake / (entry_price / 100.0)
        if win:
            pnl = shares * (1.0 - entry_price / 100.0)
        else:
            pnl = -stake
        balance += pnl

        trades.append(TradeResult(day, cutoff_utc, p_cal, model_yes, bucket_label, bucket_price, threshold_price, entry_time, entry_price, shares, win, pnl, balance, tmax, cutoff_note))

    # Build output tables
    rows = []
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for t in trades:
        if t.win is True:
            wins += 1
        elif t.win is False:
            losses += 1
        if t.pnl > 0:
            gross_profit += t.pnl
        elif t.pnl < 0:
            gross_loss += -t.pnl

        rows.append({
            "date": t.date.isoformat(),
            "cutoff_utc": t.cutoff_utc.isoformat(),
            "model_prob": round(t.model_prob, 6),
            "model_yes": t.model_yes,
            "bucket_label": t.bucket_label,
            "bucket_price_at_cutoff": t.bucket_price_at_cutoff,
            "threshold_price": t.threshold_price,
            "entry_time": t.entry_time.isoformat() if t.entry_time else None,
            "entry_price": t.entry_price,
            "shares": t.shares,
            "win": t.win,
            "pnl": t.pnl,
            "balance_after": t.balance_after,
            "tmax_full": t.tmax_full,
            "note": t.note,
        })

    total_trades = wins + losses
    win_rate = (wins / total_trades) if total_trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    summary = {
        "model": "B6_EXP20_GAM_RESIDUAL",
        "features_count": len(feature_list),
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "start_balance": START_BALANCE,
        "end_balance": balance,
        "total_days": len(trades),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "missing_kalshi_files": missing_days,
        "missing_tmax_days": no_pred_days,
    }

    # Write outputs
    trades_path = OUT_DIR / "trades.csv"
    pd.DataFrame(rows).to_csv(trades_path, index=False)

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote", trades_path)
    print("Wrote", summary_path)


if __name__ == "__main__":
    main()
