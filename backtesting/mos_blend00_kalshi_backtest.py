from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Bucket:
    label: str
    lo: int | None
    hi: int | None

    def contains(self, temp_f: int) -> bool:
        if self.lo is None and self.hi is None:
            return False
        if self.lo is None:
            return temp_f <= int(self.hi)
        if self.hi is None:
            return temp_f >= int(self.lo)
        return int(self.lo) <= temp_f <= int(self.hi)


def normalize_price(v: float) -> float:
    if pd.isna(v):
        return np.nan
    x = float(v)
    if x < 0:
        return np.nan
    # dataset may contain mixed 0-1 and 0-100; normalize to 0-1
    if x > 1.0:
        x = x / 100.0
    return float(np.clip(x, 0.0, 1.0))


def parse_bucket_label(label: str) -> Bucket | None:
    clean = (
        str(label)
        .replace("Â°", "°")
        .replace("º", "°")
        .replace(" to ", "-")
        .replace("°", "")
        .strip()
        .lower()
    )
    clean = re.sub(r"\s+", " ", clean)

    m = re.match(r"^(-?\d+)\s*or\s*below$", clean)
    if m:
        return Bucket(label=label, lo=None, hi=int(m.group(1)))
    m = re.match(r"^(-?\d+)\s*or\s*above$", clean)
    if m:
        return Bucket(label=label, lo=int(m.group(1)), hi=None)
    m = re.match(r"^(-?\d+)\s*-\s*(-?\d+)$", clean)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return Bucket(label=label, lo=lo, hi=hi)
    return None


def cdf_from_quantiles(qmap: dict[float, float], x: float) -> float:
    taus = np.array(sorted(qmap.keys()), dtype=float)
    qvals = np.array([qmap[t] for t in taus], dtype=float)
    qvals = np.maximum.accumulate(qvals)
    return float(np.interp(x, qvals, taus, left=0.0, right=1.0))


def pmf_int_from_quantiles(qmap: dict[float, float], support_lo: int = -20, support_hi: int = 130) -> dict[int, float]:
    out: dict[int, float] = {}
    for t in range(support_lo, support_hi + 1):
        p = cdf_from_quantiles(qmap, t + 0.5) - cdf_from_quantiles(qmap, t - 0.5)
        out[t] = max(0.0, float(p))
    total = float(sum(out.values()))
    if total <= 0:
        width = support_hi - support_lo + 1
        return {t: 1.0 / width for t in range(support_lo, support_hi + 1)}
    return {k: v / total for k, v in out.items()}


def bucket_prob(pmf: dict[int, float], b: Bucket) -> float:
    if b.lo is None and b.hi is not None:
        return float(sum(v for k, v in pmf.items() if k <= b.hi))
    if b.hi is None and b.lo is not None:
        return float(sum(v for k, v in pmf.items() if k >= b.lo))
    if b.lo is not None and b.hi is not None:
        return float(sum(v for k, v in pmf.items() if b.lo <= k <= b.hi))
    return 0.0


def run_backtest(
    model_pred_path: Path,
    kalshi_dir: Path,
    out_dir: Path,
    ev_min: float,
    win_min: float,
) -> tuple[pd.DataFrame, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(model_pred_path).copy()
    pred["target_date_local"] = pd.to_datetime(pred["target_date_local"]).dt.normalize()
    pred = pred[pred["target_date_local"].between(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))].copy()
    pred = pred.sort_values("target_date_local").reset_index(drop=True)

    trades: list[dict] = []
    skipped = 0

    for _, row in pred.iterrows():
        tdate = pd.Timestamp(row["target_date_local"]).normalize()
        y = int(round(float(row["y_tmax"])))
        fname = f"KNYC_{tdate.strftime('%Y%m%d')}.csv"
        fpath = kalshi_dir / fname
        if not fpath.exists():
            skipped += 1
            continue

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
            skipped += 1
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Entry aligned to model runtime: 00Z on T-1
        entry_cut = (tdate - pd.Timedelta(days=1)).tz_localize("UTC")
        entry_rows = df[df["timestamp"] >= entry_cut]
        if entry_rows.empty:
            skipped += 1
            continue
        entry = entry_rows.iloc[0]
        entry_ts = entry["timestamp"]

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

            cand = []
            # YES side
            ev_yes = p_model_yes - p_mkt_yes
            if p_model_yes >= win_min and ev_yes >= ev_min:
                win = 1 if b.contains(y) else 0
                pnl = (1.0 - p_mkt_yes) if win else (-p_mkt_yes)
                cand.append(
                    {
                        "side": "YES",
                        "bucket": col,
                        "bucket_lo": b.lo,
                        "bucket_hi": b.hi,
                        "model_win_prob": p_model_yes,
                        "market_price": p_mkt_yes,
                        "ev": ev_yes,
                        "win": win,
                        "pnl_per_1": pnl,
                    }
                )
            # NO side
            ev_no = p_model_no - p_mkt_no
            if p_model_no >= win_min and ev_no >= ev_min:
                win = 0 if b.contains(y) else 1
                pnl = (1.0 - p_mkt_no) if win else (-p_mkt_no)
                cand.append(
                    {
                        "side": "NO",
                        "bucket": col,
                        "bucket_lo": b.lo,
                        "bucket_hi": b.hi,
                        "model_win_prob": p_model_no,
                        "market_price": p_mkt_no,
                        "ev": ev_no,
                        "win": win,
                        "pnl_per_1": pnl,
                    }
                )

            for c in cand:
                if best is None or c["ev"] > best["ev"]:
                    best = c

        if best is None:
            continue

        best["target_date_local"] = tdate.strftime("%Y-%m-%d")
        best["entry_timestamp_utc"] = entry_ts.isoformat().replace("+00:00", "Z")
        best["y_tmax"] = y
        trades.append(best)

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(out_dir / "trades_2025_blend00_ev0p1_win65.csv", index=False)

    summary = {
        "trades": int(len(trades_df)),
        "skipped_days": int(skipped),
        "win_rate": float(trades_df["win"].mean()) if len(trades_df) else 0.0,
        "avg_ev": float(trades_df["ev"].mean()) if len(trades_df) else 0.0,
        "sum_pnl_per_1": float(trades_df["pnl_per_1"].sum()) if len(trades_df) else 0.0,
        "avg_pnl_per_1": float(trades_df["pnl_per_1"].mean()) if len(trades_df) else 0.0,
        "entry_rule": "entry at first market timestamp >= T-1 00:00:00Z (model runtime for blend_00)",
        "filters": {"ev_min": ev_min, "win_min": win_min},
    }
    (out_dir / "summary_2025_blend00_ev0p1_win65.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return trades_df, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pred", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_00\test_predictions.parquet")
    parser.add_argument("--kalshi-dir", default=r"D:\Ahmed\data\kalshi\kalshi_history\kxhighny_2025")
    parser.add_argument("--out-dir", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest")
    parser.add_argument("--ev-min", type=float, default=0.1)
    parser.add_argument("--win-min", type=float, default=0.65)
    args = parser.parse_args()
    run_backtest(Path(args.model_pred), Path(args.kalshi_dir), Path(args.out_dir), args.ev_min, args.win_min)


if __name__ == "__main__":
    main()

