from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Bucket:
    label_raw: str
    lo: Optional[int]
    hi: Optional[int]
    mode: str  # "range" | "or_below" | "or_above"

    def contains(self, temp_f: int) -> bool:
        if self.mode == "or_below" and self.hi is not None:
            return temp_f <= self.hi
        if self.mode == "or_above" and self.lo is not None:
            return temp_f >= self.lo
        if self.mode == "range" and self.lo is not None and self.hi is not None:
            return self.lo <= temp_f <= self.hi
        return False

    def canonical_label(self) -> str:
        if self.mode == "or_below" and self.hi is not None:
            return f"{self.hi}F or below"
        if self.mode == "or_above" and self.lo is not None:
            return f"{self.lo}F or above"
        if self.mode == "range" and self.lo is not None and self.hi is not None:
            return f"{self.lo}F to {self.hi}F"
        return self.label_raw


def normalize_price(v: float) -> float:
    if pd.isna(v):
        return np.nan
    x = float(v)
    if x < 0:
        return np.nan
    if x > 1.0:
        x = x / 100.0
    return float(np.clip(x, 0.0, 1.0))


def parse_bucket_label(label: str) -> Optional[Bucket]:
    s = str(label).strip().lower().replace(" to ", "-")
    s = re.sub(r"\s+", " ", s)
    # Bucket labels are temperature ranges; extract unsigned integers to avoid
    # interpreting range separators ("-") as negative signs.
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if ("or below" in s or "or less" in s) and nums:
        return Bucket(label_raw=str(label), lo=None, hi=nums[0], mode="or_below")
    if ("or above" in s or "or higher" in s) and nums:
        return Bucket(label_raw=str(label), lo=nums[0], hi=None, mode="or_above")
    if len(nums) >= 2:
        a, b = nums[0], nums[1]
        lo, hi = (a, b) if a <= b else (b, a)
        return Bucket(label_raw=str(label), lo=lo, hi=hi, mode="range")
    return None


def cdf_from_quantiles(qmap: Dict[float, float], x: float) -> float:
    taus = np.array(sorted(qmap.keys()), dtype=float)
    qvals = np.array([qmap[t] for t in taus], dtype=float)
    qvals = np.maximum.accumulate(qvals)
    return float(np.interp(x, qvals, taus, left=0.0, right=1.0))


def pmf_int_from_quantiles(qmap: Dict[float, float], support_lo: int = -20, support_hi: int = 130) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for t in range(support_lo, support_hi + 1):
        p = cdf_from_quantiles(qmap, t + 0.5) - cdf_from_quantiles(qmap, t - 0.5)
        out[t] = max(0.0, float(p))
    total = float(sum(out.values()))
    if total <= 0:
        width = support_hi - support_lo + 1
        return {t: 1.0 / width for t in range(support_lo, support_hi + 1)}
    return {k: v / total for k, v in out.items()}


def bucket_prob(pmf: Dict[int, float], b: Bucket) -> float:
    if b.mode == "or_below" and b.hi is not None:
        return float(sum(v for k, v in pmf.items() if k <= b.hi))
    if b.mode == "or_above" and b.lo is not None:
        return float(sum(v for k, v in pmf.items() if k >= b.lo))
    if b.mode == "range" and b.lo is not None and b.hi is not None:
        return float(sum(v for k, v in pmf.items() if b.lo <= k <= b.hi))
    return 0.0


def load_predictions(dev_path: Path, test_path: Path) -> pd.DataFrame:
    dev = pd.read_parquet(dev_path)
    test = pd.read_parquet(test_path)
    pred = pd.concat([dev, test], ignore_index=True)
    pred["target_date_local"] = pd.to_datetime(pred["target_date_local"]).dt.normalize()
    pred = pred.drop_duplicates(subset=["target_date_local"], keep="last")
    pred = pred.sort_values("target_date_local").reset_index(drop=True)
    return pred


def build_market_index(kalshi_root: Path) -> Tuple[Dict[str, Path], Dict[str, List[str]]]:
    preference = {
        "kxhighny_2022_2024": 1,
        "kxhighny_2025": 2,
        "kxhighny_2026_to_20260227": 3,
        "kxhighny_dec2025": 9,
    }
    selected: Dict[str, Tuple[int, Path]] = {}
    duplicates: Dict[str, List[str]] = {}
    for p in kalshi_root.rglob("KNYC_*.csv"):
        m = re.match(r"^KNYC_(\d{8})\.csv$", p.name)
        if not m:
            continue
        d = m.group(1)
        duplicates.setdefault(d, []).append(str(p))
        rank = preference.get(p.parent.name, 99)
        if d not in selected or rank < selected[d][0]:
            selected[d] = (rank, p)
    idx = {d: t[1] for d, t in selected.items()}
    dup_only = {d: paths for d, paths in duplicates.items() if len(paths) > 1}
    return idx, dup_only


def safe_iso_utc(ts: pd.Timestamp) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def compute_entry_cutoff_utc(target_date_local: pd.Timestamp, entry_hour_z: int, entry_minute_z: int) -> pd.Timestamp:
    return (target_date_local - pd.Timedelta(days=1) + pd.Timedelta(hours=entry_hour_z, minutes=entry_minute_z)).tz_localize(
        "UTC"
    )


def run_backtest(
    pred: pd.DataFrame,
    market_index: Dict[str, Path],
    ev_min: float,
    win_min: float,
    start_balance: float,
    risk_fraction: float,
    stake_cap_usd: float,
    entry_hour_z: int,
    entry_minute_z: int,
    min_entry_minutes_after_open: int = 0,
) -> Tuple[pd.DataFrame, Dict]:
    trades: List[Dict] = []
    counts = {
        "prediction_days": int(len(pred)),
        "market_days_available_total": int(len(market_index)),
        "market_days_with_predictions_overlap": 0,
        "days_without_market_file": 0,
        "days_with_no_trade_candidate": 0,
    }

    for _, row in pred.iterrows():
        tdate = pd.Timestamp(row["target_date_local"]).normalize()
        y = int(round(float(row["y_tmax"])))
        fpath = market_index.get(tdate.strftime("%Y%m%d"))
        if fpath is None:
            counts["days_without_market_file"] += 1
            continue
        counts["market_days_with_predictions_overlap"] += 1

        qmap = {
            0.05: float(row["q_0.05"]),
            0.10: float(row["q_0.10"]),
            0.25: float(row["q_0.25"]),
            0.50: float(row["q_0.50"]),
            0.75: float(row["q_0.75"]),
            0.90: float(row["q_0.90"]),
            0.95: float(row["q_0.95"]),
        }
        pmf = pmf_int_from_quantiles(qmap)

        df = pd.read_csv(fpath)
        if "timestamp" not in df.columns:
            counts["days_with_no_trade_candidate"] += 1
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if df.empty:
            counts["days_with_no_trade_candidate"] += 1
            continue

        entry_cut = compute_entry_cutoff_utc(tdate, entry_hour_z, entry_minute_z)
        market_open_ts = pd.Timestamp(df["timestamp"].iloc[0])
        open_delay_cut = market_open_ts + pd.Timedelta(minutes=int(min_entry_minutes_after_open))
        effective_entry_cut = max(entry_cut, open_delay_cut)
        entry_rows = df[df["timestamp"] >= effective_entry_cut]
        if entry_rows.empty:
            counts["days_with_no_trade_candidate"] += 1
            continue
        entry = entry_rows.iloc[0]
        entry_ts = pd.Timestamp(entry["timestamp"])

        best = None
        for col in df.columns:
            if col == "timestamp":
                continue
            b = parse_bucket_label(col)
            if b is None:
                continue
            p_mkt_yes = normalize_price(entry[col])
            if not np.isfinite(p_mkt_yes):
                continue
            p_model_yes = bucket_prob(pmf, b)
            p_model_no = 1.0 - p_model_yes
            p_mkt_no = 1.0 - p_mkt_yes

            ev_yes = p_model_yes - p_mkt_yes
            if p_model_yes >= win_min and ev_yes >= ev_min:
                win = 1 if b.contains(y) else 0
                c = {
                    "side": "YES",
                    "bucket_raw": col,
                    "bucket": b.canonical_label(),
                    "bucket_lo": b.lo,
                    "bucket_hi": b.hi,
                    "bucket_mode": b.mode,
                    "model_win_prob": p_model_yes,
                    "market_price": p_mkt_yes,
                    "ev": ev_yes,
                    "win": win,
                }
                if best is None or c["ev"] > best["ev"]:
                    best = c

            ev_no = p_model_no - p_mkt_no
            if p_model_no >= win_min and ev_no >= ev_min:
                win = 0 if b.contains(y) else 1
                c = {
                    "side": "NO",
                    "bucket_raw": col,
                    "bucket": b.canonical_label(),
                    "bucket_lo": b.lo,
                    "bucket_hi": b.hi,
                    "bucket_mode": b.mode,
                    "model_win_prob": p_model_no,
                    "market_price": p_mkt_no,
                    "ev": ev_no,
                    "win": win,
                }
                if best is None or c["ev"] > best["ev"]:
                    best = c

        if best is None:
            counts["days_with_no_trade_candidate"] += 1
            continue

        best["target_date_local"] = tdate.strftime("%Y-%m-%d")
        best["entry_timestamp_utc"] = safe_iso_utc(entry_ts)
        best["market_open_utc"] = safe_iso_utc(market_open_ts)
        best["entry_cutoff_utc"] = safe_iso_utc(entry_cut)
        best["effective_entry_cutoff_utc"] = safe_iso_utc(effective_entry_cut)
        best["y_tmax"] = y
        best["market_file"] = str(fpath)
        trades.append(best)

    trades_df = pd.DataFrame(trades)
    bal = float(start_balance)
    peak = bal
    if not trades_df.empty:
        for i, r in trades_df.iterrows():
            stake = min(bal * risk_fraction, stake_cap_usd)
            price = float(r["market_price"])
            shares = stake / price if price > 0 else 0.0
            pnl = shares * (1.0 - price) if int(r["win"]) == 1 else -stake
            bal_before = bal
            bal = bal + pnl
            peak = max(peak, bal)
            dd = 0.0 if peak <= 0 else (peak - bal) / peak
            trades_df.loc[i, "stake"] = stake
            trades_df.loc[i, "shares"] = shares
            trades_df.loc[i, "pnl"] = pnl
            trades_df.loc[i, "balance_before"] = bal_before
            trades_df.loc[i, "balance_after"] = bal
            trades_df.loc[i, "drawdown"] = dd
            trades_df.loc[i, "result"] = "W" if int(r["win"]) == 1 else "L"

    wins = trades_df[trades_df["win"] == 1] if not trades_df.empty else pd.DataFrame()
    losses = trades_df[trades_df["win"] == 0] if not trades_df.empty else pd.DataFrame()
    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(-losses["pnl"].sum()) if not losses.empty else 0.0

    if not trades_df.empty:
        entry_ts = pd.to_datetime(trades_df["entry_timestamp_utc"], utc=True)
        cut_ts = pd.to_datetime(trades_df["entry_cutoff_utc"], utc=True)
        lag_min = (entry_ts - cut_ts).dt.total_seconds() / 60.0
        side_counts = trades_df["side"].value_counts().to_dict()
        px = trades_df["market_price"].astype(float)
    else:
        lag_min = pd.Series(dtype=float)
        side_counts = {}
        px = pd.Series(dtype=float)

    summary = {
        "period_pred_min": str(pred["target_date_local"].min().date()) if len(pred) else None,
        "period_pred_max": str(pred["target_date_local"].max().date()) if len(pred) else None,
        **counts,
        "trades": int(len(trades_df)),
        "wins": int((trades_df["win"] == 1).sum()) if not trades_df.empty else 0,
        "losses": int((trades_df["win"] == 0).sum()) if not trades_df.empty else 0,
        "win_rate": float(trades_df["win"].mean()) if not trades_df.empty else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else None,
        "start_balance": float(start_balance),
        "final_balance": float(bal),
        "total_pnl": float(bal - start_balance),
        "risk_fraction": float(risk_fraction),
        "stake_cap_usd": float(stake_cap_usd),
        "avg_ev_at_trade": float(trades_df["ev"].mean()) if not trades_df.empty else 0.0,
        "median_ev_at_trade": float(trades_df["ev"].median()) if not trades_df.empty else 0.0,
        "avg_win_pnl": float(wins["pnl"].mean()) if not wins.empty else 0.0,
        "avg_loss_pnl": float(losses["pnl"].mean()) if not losses.empty else 0.0,
        "max_drawdown": float(trades_df["drawdown"].max()) if not trades_df.empty else 0.0,
        "entry_lag_minutes_min": float(lag_min.min()) if len(lag_min) else None,
        "entry_lag_minutes_median": float(lag_min.median()) if len(lag_min) else None,
        "entry_lag_minutes_max": float(lag_min.max()) if len(lag_min) else None,
        "side_counts": side_counts,
        "market_price_zero_count": int((px <= 0.0).sum()) if len(px) else 0,
        "market_price_lt_0p01_count": int((px < 0.01).sum()) if len(px) else 0,
        "market_price_gt_0p99_count": int((px > 0.99).sum()) if len(px) else 0,
        "entry_rule": (
            f"first quoted market row >= max(T-1 {entry_hour_z:02d}:{entry_minute_z:02d}:00Z, "
            f"market_open_utc + {int(min_entry_minutes_after_open)}m)"
        ),
        "min_entry_minutes_after_open": int(min_entry_minutes_after_open),
        "filters": {"ev_min": float(ev_min), "win_min": float(win_min)},
    }
    return trades_df, summary


def _eq(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def run_sanity_audit(
    trades_df: pd.DataFrame,
    pred: pd.DataFrame,
    entry_hour_z: int,
    entry_minute_z: int,
    stake_cap_usd: float,
    min_entry_minutes_after_open: int = 0,
) -> Dict:
    by_date = {pd.Timestamp(r["target_date_local"]).strftime("%Y-%m-%d"): r for _, r in pred.iterrows()}
    failures: Dict[str, int] = {
        "missing_market_file": 0,
        "target_date_file_mismatch": 0,
        "entry_before_effective_cutoff": 0,
        "entry_not_first_after_cutoff": 0,
        "bucket_not_found": 0,
        "bucket_unparseable": 0,
        "market_price_mismatch": 0,
        "model_prob_mismatch": 0,
        "ev_mismatch": 0,
        "win_label_mismatch": 0,
        "stake_cap_breach": 0,
        "pnl_mismatch": 0,
        "nan_in_critical_fields": 0,
    }
    checked = 0

    for _, tr in trades_df.iterrows():
        checked += 1
        tdate = str(tr["target_date_local"])
        fpath = Path(str(tr["market_file"]))
        if not fpath.exists():
            failures["missing_market_file"] += 1
            continue
        expected_name = f"KNYC_{pd.Timestamp(tdate).strftime('%Y%m%d')}.csv"
        if fpath.name != expected_name:
            failures["target_date_file_mismatch"] += 1

        df = pd.read_csv(fpath)
        if "timestamp" not in df.columns:
            failures["entry_not_first_after_cutoff"] += 1
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        entry_ts = pd.Timestamp(pd.to_datetime(tr["entry_timestamp_utc"], utc=True))
        gate_cutoff = compute_entry_cutoff_utc(pd.Timestamp(tdate), entry_hour_z, entry_minute_z)
        market_open_ts = pd.Timestamp(df["timestamp"].iloc[0])
        open_delay_cutoff = market_open_ts + pd.Timedelta(minutes=int(min_entry_minutes_after_open))
        cutoff = max(gate_cutoff, open_delay_cutoff)
        if entry_ts < cutoff:
            failures["entry_before_effective_cutoff"] += 1

        rows = df[df["timestamp"] >= cutoff]
        if rows.empty or pd.Timestamp(rows.iloc[0]["timestamp"]) != entry_ts:
            failures["entry_not_first_after_cutoff"] += 1
            continue
        entry_row = rows.iloc[0]

        raw_bucket = str(tr["bucket_raw"])
        if raw_bucket not in df.columns:
            failures["bucket_not_found"] += 1
            continue
        bucket = parse_bucket_label(raw_bucket)
        if bucket is None:
            failures["bucket_unparseable"] += 1
            continue

        p_yes = normalize_price(entry_row[raw_bucket])
        if not (
            math.isfinite(float(tr["market_price"]))
            and math.isfinite(float(tr["model_win_prob"]))
            and math.isfinite(float(tr["ev"]))
            and math.isfinite(float(tr["stake"]))
            and math.isfinite(float(tr["pnl"]))
        ):
            failures["nan_in_critical_fields"] += 1
        p_market_expected = p_yes if str(tr["side"]) == "YES" else (1.0 - p_yes)
        if not _eq(float(tr["market_price"]), float(p_market_expected), tol=1e-8):
            failures["market_price_mismatch"] += 1

        pr = by_date.get(tdate)
        if pr is None:
            failures["model_prob_mismatch"] += 1
            failures["ev_mismatch"] += 1
            continue
        qmap = {
            0.05: float(pr["q_0.05"]),
            0.10: float(pr["q_0.10"]),
            0.25: float(pr["q_0.25"]),
            0.50: float(pr["q_0.50"]),
            0.75: float(pr["q_0.75"]),
            0.90: float(pr["q_0.90"]),
            0.95: float(pr["q_0.95"]),
        }
        pmf = pmf_int_from_quantiles(qmap)
        p_yes_model = bucket_prob(pmf, bucket)
        p_model_expected = p_yes_model if str(tr["side"]) == "YES" else (1.0 - p_yes_model)
        if not _eq(float(tr["model_win_prob"]), float(p_model_expected), tol=1e-8):
            failures["model_prob_mismatch"] += 1

        ev_expected = float(p_model_expected - p_market_expected)
        if not _eq(float(tr["ev"]), ev_expected, tol=1e-8):
            failures["ev_mismatch"] += 1

        y = int(round(float(tr["y_tmax"])))
        win_expected = 1 if ((str(tr["side"]) == "YES" and bucket.contains(y)) or (str(tr["side"]) == "NO" and not bucket.contains(y))) else 0
        if int(tr["win"]) != win_expected:
            failures["win_label_mismatch"] += 1

        stake = float(tr["stake"])
        if stake > float(stake_cap_usd) + 1e-9:
            failures["stake_cap_breach"] += 1

        price = float(tr["market_price"])
        shares = float(tr["shares"])
        pnl_expected = shares * (1.0 - price) if int(tr["win"]) == 1 else -stake
        if not _eq(float(tr["pnl"]), float(pnl_expected), tol=1e-6):
            failures["pnl_mismatch"] += 1

    any_fail = any(v > 0 for v in failures.values())
    px = trades_df["market_price"].astype(float) if not trades_df.empty else pd.Series(dtype=float)
    additional = {
        "min_market_price": float(px.min()) if len(px) else None,
        "max_market_price": float(px.max()) if len(px) else None,
        "zero_or_negative_market_price_count": int((px <= 0.0).sum()) if len(px) else 0,
    }
    return {
        "checked_trades": int(checked),
        "passes_all_checks": (not any_fail),
        "failures": failures,
        "additional": additional,
    }


def build_june_dec_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "Entry time (Stockholm)",
                "Bucket",
                "Side",
                "Market win % (side)",
                "Model win %",
                "EV",
                "Amount invested ($)",
                "Profit made ($)",
                "Result",
            ]
        )
    t = trades_df.copy()
    t["target_date_local"] = pd.to_datetime(t["target_date_local"])
    t = t[(t["target_date_local"] >= pd.Timestamp("2025-06-01")) & (t["target_date_local"] <= pd.Timestamp("2025-12-31"))].copy()
    t["entry_ts_utc"] = pd.to_datetime(t["entry_timestamp_utc"], utc=True)
    t["entry_stockholm"] = t["entry_ts_utc"].dt.tz_convert(ZoneInfo("Europe/Stockholm")).dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    out = pd.DataFrame(
        {
            "Entry time (Stockholm)": t["entry_stockholm"],
            "Bucket": t["bucket"],
            "Side": t["side"],
            "Market win % (side)": (t["market_price"] * 100.0).round(2),
            "Model win %": (t["model_win_prob"] * 100.0).round(2),
            "EV": t["ev"].round(4),
            "Amount invested ($)": t["stake"].round(2),
            "Profit made ($)": t["pnl"].round(2),
            "Result": t["result"].map({"W": "Win", "L": "Loss"}).fillna(t["result"]),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audited MOS blend_00 backtest with entry gate and stake cap.")
    parser.add_argument("--pred-dev", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_00\dev_predictions.parquet")
    parser.add_argument("--pred-test", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_00\test_predictions.parquet")
    parser.add_argument("--kalshi-root", default=r"D:\Ahmed\data\kalshi\kalshi_history")
    parser.add_argument("--out-dir", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest")
    parser.add_argument("--table-out", default=r"D:\Ahmed\data\kalshi\plots\june_dec_2025_trade_table_sideaware_entry1530z_cap400.csv")
    parser.add_argument("--ev-min", type=float, default=0.10)
    parser.add_argument("--win-min", type=float, default=0.60)
    parser.add_argument("--start-balance", type=float, default=2700.0)
    parser.add_argument("--risk-fraction", type=float, default=0.055)
    parser.add_argument("--stake-cap-usd", type=float, default=400.0)
    parser.add_argument("--entry-hour-z", type=int, default=15)
    parser.add_argument("--entry-minute-z", type=int, default=30)
    parser.add_argument("--min-entry-minutes-after-open", type=int, default=0)
    parser.add_argument("--out-prefix", default="all_available_blend00_ev0p1_win60_risk5p5pct_entry1530z_cap400")
    args = parser.parse_args()

    pred = load_predictions(Path(args.pred_dev), Path(args.pred_test))
    market_index, duplicates = build_market_index(Path(args.kalshi_root))
    trades_df, summary = run_backtest(
        pred=pred,
        market_index=market_index,
        ev_min=float(args.ev_min),
        win_min=float(args.win_min),
        start_balance=float(args.start_balance),
        risk_fraction=float(args.risk_fraction),
        stake_cap_usd=float(args.stake_cap_usd),
        entry_hour_z=int(args.entry_hour_z),
        entry_minute_z=int(args.entry_minute_z),
        min_entry_minutes_after_open=int(args.min_entry_minutes_after_open),
    )
    sanity = run_sanity_audit(
        trades_df=trades_df,
        pred=pred,
        entry_hour_z=int(args.entry_hour_z),
        entry_minute_z=int(args.entry_minute_z),
        stake_cap_usd=float(args.stake_cap_usd),
        min_entry_minutes_after_open=int(args.min_entry_minutes_after_open),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / f"trades_{args.out_prefix}.csv"
    summary_path = out_dir / f"summary_{args.out_prefix}.json"
    sanity_path = out_dir / f"sanity_{args.out_prefix}.json"
    table_path = Path(args.table_out)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    trades_df.to_csv(trades_path, index=False)
    full_summary = {
        **summary,
        "duplicate_market_dates_detected": int(len(duplicates)),
        "duplicate_market_date_examples": {k: v for k, v in list(duplicates.items())[:10]},
    }
    summary_path.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    sanity_path.write_text(json.dumps(sanity, indent=2), encoding="utf-8")

    june_dec = build_june_dec_table(trades_df)
    june_dec.to_csv(table_path, index=False)

    print("WROTE_TRADES", trades_path)
    print("WROTE_SUMMARY", summary_path)
    print("WROTE_SANITY", sanity_path)
    print("WROTE_TABLE", table_path)
    print(json.dumps(full_summary, indent=2))
    print(json.dumps(sanity, indent=2))


if __name__ == "__main__":
    main()
