from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pymysql

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


REPO = Path(__file__).resolve().parents[2]
UTC = timezone.utc

STATION_ID = "KMIA"
LOCAL_TZ = "America/New_York"
STOCKHOLM_TZ = "Europe/Stockholm"

KALSHI_DIR = REPO / "data" / "kalshi_backtest_data"
FEATURE_STORE = REPO / "cache" / "hit1830_v6_features.parquet"


def utc_now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def connect_db() -> pymysql.connections.Connection:
    host = os.environ.get("MYSQL_HOST", "localhost")
    port = int(os.environ.get("MYSQL_PORT", "3306"))
    user = os.environ.get("MYSQL_USER", "root")
    password = os.environ.get("MYSQL_PASSWORD", "root")
    db = os.environ.get("MYSQL_DB", "weather_predictionmarkets")
    return pymysql.connect(host=host, port=port, user=user, password=password, database=db, autocommit=True)


def compute_cutoff_utc(day: date) -> datetime:
    if ZoneInfo is None:
        raise RuntimeError("ZoneInfo not available; cannot compute cutoff.")
    stockholm = ZoneInfo(STOCKHOLM_TZ)
    cutoff_local = datetime(day.year, day.month, day.day, 18, 30, tzinfo=stockholm)
    return cutoff_local.astimezone(timezone.utc)


def day_start_end_utc(day: date) -> Tuple[datetime, datetime]:
    if ZoneInfo is None:
        raise RuntimeError("ZoneInfo not available; cannot compute local day boundaries.")
    local = ZoneInfo(LOCAL_TZ)
    start_local = datetime(day.year, day.month, day.day, 0, 0, tzinfo=local)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def load_minute_series_for_year(minute_dir: Path, year: int) -> pd.DataFrame:
    path = minute_dir / f"MIA_tmpf_1min_UTC_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing minute data file: {path}")
    df = pd.read_csv(path, usecols=["valid(UTC)", "tmpf"], dtype={"tmpf": "string"})
    df["ts_utc"] = pd.to_datetime(df["valid(UTC)"], utc=True, errors="coerce")
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    df = df.dropna(subset=["ts_utc", "tmpf"])
    df = df.sort_values("ts_utc")
    df = df.drop_duplicates(subset=["ts_utc"], keep="last")
    df = df.set_index("ts_utc")
    return df


def mos_x_mean_asof_le_cutoff(day: date, cutoff_utc: datetime) -> Tuple[float, Dict[str, str]]:
    """
    Recompute mos_x_mean (n_x / value_max) exactly as the hit1830 suite uses:
      - only rows with asof_utc <= cutoff_utc
      - choose latest per model by (asof_utc, runtime_utc, retrieved_at_utc, id)
      - average latest GFS and NAM (ignore NaNs)
    """
    sql = """
        SELECT model, variable_code, asof_utc, runtime_utc, retrieved_at_utc, id, value_max
        FROM mos_daily_value
        WHERE station_id=%s
          AND target_date_local=%s
          AND model IN ('GFS','NAM')
          AND variable_code='n_x'
        ORDER BY asof_utc, runtime_utc, retrieved_at_utc, id
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (STATION_ID, day.isoformat()))
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(
        rows,
        columns=["model", "variable_code", "asof_utc", "runtime_utc", "retrieved_at_utc", "id", "value_max"],
    )
    meta: Dict[str, str] = {}
    if df.empty:
        return float("nan"), {"mos_rows": "0"}

    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True, errors="coerce")
    df["runtime_utc"] = pd.to_datetime(df["runtime_utc"], utc=True, errors="coerce")
    df["retrieved_at_utc"] = pd.to_datetime(df["retrieved_at_utc"], utc=True, errors="coerce")
    df["value_max"] = pd.to_numeric(df["value_max"], errors="coerce")
    df = df.dropna(subset=["asof_utc", "value_max"])

    df = df[df["asof_utc"] <= cutoff_utc].copy()
    meta["mos_rows_asof_le_cutoff"] = str(len(df))
    if df.empty:
        return float("nan"), meta

    latest_rows: Dict[str, Optional[pd.Series]] = {}
    for model in ["GFS", "NAM"]:
        sub = df[df["model"] == model].copy()
        if sub.empty:
            latest_rows[model] = None
        else:
            latest_rows[model] = sub.iloc[-1]

    vals: List[float] = []
    for model in ["GFS", "NAM"]:
        r = latest_rows.get(model)
        if r is None:
            meta[f"{model}_present"] = "0"
            continue
        meta[f"{model}_present"] = "1"
        meta[f"{model}_asof_utc"] = str(r["asof_utc"])
        meta[f"{model}_runtime_utc"] = str(r["runtime_utc"])
        meta[f"{model}_retrieved_at_utc"] = str(r["retrieved_at_utc"])
        meta[f"{model}_value_max"] = str(float(r["value_max"]))
        vals.append(float(r["value_max"]))

    mos_x_mean = float(sum(vals) / len(vals)) if vals else float("nan")
    return mos_x_mean, meta


def price_at_or_before(kal: pd.DataFrame, cutoff: datetime, bucket: str) -> Optional[float]:
    sub = kal[kal["timestamp"] <= cutoff][bucket].dropna()
    if sub.empty:
        return None
    return float(sub.iloc[-1])


def first_at_or_below(kal: pd.DataFrame, cutoff: datetime, bucket: str, thr: float) -> Tuple[Optional[datetime], Optional[float]]:
    sub = kal[kal["timestamp"] >= cutoff][["timestamp", bucket]].dropna()
    if sub.empty:
        return None, None
    hit = sub[sub[bucket] <= thr]
    if hit.empty:
        return None, None
    row = hit.iloc[0]
    return row["timestamp"].to_pydatetime(), float(row[bucket])


def first_at_or_above(kal: pd.DataFrame, cutoff: datetime, bucket: str, thr: float) -> Tuple[Optional[datetime], Optional[float]]:
    sub = kal[kal["timestamp"] >= cutoff][["timestamp", bucket]].dropna()
    if sub.empty:
        return None, None
    hit = sub[sub[bucket] >= thr]
    if hit.empty:
        return None, None
    row = hit.iloc[0]
    return row["timestamp"].to_pydatetime(), float(row[bucket])


@dataclass(frozen=True)
class Violation:
    kind: str
    date: str
    detail: str


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Paranoid sanity + leakage audit for a Kalshi bridge backtest run.")
    parser.add_argument("--backtest-dir", required=True, help="Backtest output directory (must contain summary.json)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Report output directory (default: reports/live_trade_flow_audit_<timestamp>)",
    )
    args = parser.parse_args(argv)

    backtest_dir = Path(args.backtest_dir).resolve()
    summary_path = backtest_dir / "summary.json"
    trades_entered_path = backtest_dir / "trades_entered.csv"
    minute_dir = REPO / "data" / "iem_minute_data" / "MIA" / "tmpf" / "UTC" / "yearly"

    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")
    if not trades_entered_path.exists():
        raise SystemExit(f"Missing {trades_entered_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (REPO / "reports" / f"live_trade_flow_audit_{utc_now_tag()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    p_hit_gate = float(summary.get("p_hit_gate", float("nan")))
    edge_prob = float(summary.get("edge_prob", float("nan")))
    min_win_prob = float(summary.get("min_win_prob", float("nan")))
    bt_start = summary.get("backtest_start")
    bt_end = summary.get("backtest_end")
    bridge_train_end = summary.get("bridge_train_end")
    if bt_start and bridge_train_end:
        try:
            bt_start_d = date.fromisoformat(str(bt_start))
            bridge_train_end_d = date.fromisoformat(str(bridge_train_end))
            if bridge_train_end_d >= bt_start_d:
                raise RuntimeError("bridge_train_end must be < backtest_start for leakage safety.")
        except Exception as e:
            raise SystemExit(f"Invalid summary date fields in {summary_path}: {e}") from e

    # Optional: attach the latest exported deployable bundle metadata (if present).
    bundle_root = REPO / "artifacts" / "model_bundles" / "hit1830_v6" / str(summary.get("model", ""))
    latest_bundle: Optional[Dict[str, object]] = None
    if bundle_root.exists():
        candidates = [p for p in bundle_root.iterdir() if p.is_dir()]
        candidates.sort(key=lambda p: p.name)
        if candidates:
            bundle_dir = candidates[-1]
            meta_path = bundle_dir / "bundle_meta.json"
            if meta_path.exists():
                latest_bundle = {"bundle_dir": str(bundle_dir), "meta": json.loads(meta_path.read_text(encoding="utf-8"))}

    trades = pd.read_csv(trades_entered_path)
    if trades.empty:
        raise SystemExit("No trades found in trades_entered.csv; nothing to audit.")

    trades["date"] = pd.to_datetime(trades["date"]).dt.date
    trades["cutoff_utc"] = pd.to_datetime(trades["cutoff_utc"], utc=True)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)

    # Feature store rows for all trade days.
    feat_cols = ["target_date_local", "cutoff_utc", "tmax_sofar", "coverage_frac", "last_gap_minutes", "mos_x_mean"]
    feats = pd.read_parquet(FEATURE_STORE, columns=feat_cols)
    feats["target_date_local"] = pd.to_datetime(feats["target_date_local"]).dt.date
    feats["cutoff_utc"] = pd.to_datetime(feats["cutoff_utc"], utc=True)
    feats = feats.set_index("target_date_local")

    # Load minute data for years present in the trade set.
    years_needed = sorted({d.year for d in trades["date"]})
    minute_by_year: Dict[int, pd.DataFrame] = {}
    series5_by_year: Dict[int, pd.Series] = {}
    for y in years_needed:
        df_1m = load_minute_series_for_year(minute_dir, y)
        minute_by_year[y] = df_1m
        series5_by_year[y] = df_1m["tmpf"].resample("5min").median()

    violations: List[Violation] = []

    # --- Trade entry logic checks (pure, no DB) ---
    for r in trades.itertuples(index=False):
        day: date = getattr(r, "date")
        cutoff_utc: datetime = getattr(r, "cutoff_utc").to_pydatetime()
        entry_time: datetime = getattr(r, "entry_time").to_pydatetime()
        p_hit: float = float(getattr(r, "p_hit"))
        side: str = str(getattr(r, "trade_side"))
        bucket: str = str(getattr(r, "bucket_label"))
        p_bucket: float = float(getattr(r, "bucket_p_win"))
        trade_p_win: float = float(getattr(r, "trade_p_win"))
        cutoff_yes: float = float(getattr(r, "bucket_price_at_cutoff"))
        thr_price: float = float(getattr(r, "threshold_price"))
        entry_yes_price: float = float(getattr(r, "entry_yes_price"))
        entry_price: float = float(getattr(r, "entry_price"))
        ev_at_entry: float = float(getattr(r, "ev_at_entry"))

        if entry_time < cutoff_utc:
            violations.append(Violation("entry_time_before_cutoff", day.isoformat(), f"entry_time={entry_time} cutoff={cutoff_utc}"))

        if math.isfinite(p_hit_gate) and p_hit_gate >= 0 and p_hit < p_hit_gate - 1e-12:
            violations.append(Violation("p_hit_gate_failed", day.isoformat(), f"p_hit={p_hit} gate={p_hit_gate}"))

        if trade_p_win + 1e-12 < min_win_prob:
            violations.append(Violation("min_win_prob_failed", day.isoformat(), f"trade_p_win={trade_p_win} min={min_win_prob}"))

        # Threshold identity checks
        if side == "YES":
            expected_thr = (p_bucket - edge_prob) * 100.0
        else:
            expected_thr = (p_bucket + edge_prob) * 100.0
        if abs(expected_thr - thr_price) > 1e-6:
            violations.append(Violation("threshold_mismatch", day.isoformat(), f"expected={expected_thr} got={thr_price} side={side}"))

        # Entry price identity checks
        if side == "YES":
            expected_entry_price = entry_yes_price
        else:
            expected_entry_price = 100.0 - entry_yes_price
        if abs(expected_entry_price - entry_price) > 1e-6:
            violations.append(Violation("entry_price_mismatch", day.isoformat(), f"expected={expected_entry_price} got={entry_price} side={side}"))

        expected_ev = trade_p_win - (entry_price / 100.0)
        if abs(expected_ev - ev_at_entry) > 1e-9:
            violations.append(Violation("ev_mismatch", day.isoformat(), f"expected={expected_ev} got={ev_at_entry}"))

        # --- Kalshi integrity checks for this trade ---
        kal_path = KALSHI_DIR / f"{STATION_ID}_{day.strftime('%Y%m%d')}.csv"
        if not kal_path.exists():
            violations.append(Violation("missing_kalshi_file_for_trade", day.isoformat(), str(kal_path)))
            continue

        kal = pd.read_csv(kal_path)
        if "timestamp" not in kal.columns:
            violations.append(Violation("kalshi_missing_timestamp_col", day.isoformat(), str(kal_path)))
            continue
        if bucket not in kal.columns:
            violations.append(Violation("kalshi_missing_bucket_col", day.isoformat(), f"{bucket} not in {kal_path.name}"))
            continue

        kal["timestamp"] = pd.to_datetime(kal["timestamp"], utc=True, errors="coerce")
        kal = kal.dropna(subset=["timestamp"]).sort_values("timestamp")
        for c in kal.columns:
            if c == "timestamp":
                continue
            kal[c] = pd.to_numeric(kal[c], errors="coerce")

        # Prices must be within [0,100] when present.
        s = kal[bucket].dropna()
        if not s.empty:
            if (s < -1e-9).any() or (s > 100.0 + 1e-9).any():
                violations.append(Violation("kalshi_price_out_of_bounds", day.isoformat(), f"{bucket}"))

        cutoff_yes_recalc = price_at_or_before(kal, cutoff_utc, bucket)
        if cutoff_yes_recalc is None or abs(cutoff_yes_recalc - cutoff_yes) > 1e-6:
            violations.append(
                Violation(
                    "cutoff_price_mismatch",
                    day.isoformat(),
                    f"expected={cutoff_yes} recalculated={cutoff_yes_recalc} bucket={bucket}",
                )
            )

        # Entry must be first threshold crossing at/after cutoff.
        if side == "YES":
            exp_entry_time, exp_entry_yes = first_at_or_below(kal, cutoff_utc, bucket, thr_price)
            # Special case: filled immediately at cutoff.
            if cutoff_yes <= thr_price:
                exp_entry_time = cutoff_utc
                exp_entry_yes = cutoff_yes
        else:
            exp_entry_time, exp_entry_yes = first_at_or_above(kal, cutoff_utc, bucket, thr_price)
            if cutoff_yes >= thr_price:
                exp_entry_time = cutoff_utc
                exp_entry_yes = cutoff_yes

        if exp_entry_time is None or exp_entry_yes is None:
            violations.append(Violation("no_threshold_cross_found", day.isoformat(), f"side={side} bucket={bucket}"))
        else:
            if abs((exp_entry_time - entry_time).total_seconds()) > 1e-6:
                violations.append(
                    Violation(
                        "entry_time_not_first_crossing",
                        day.isoformat(),
                        f"expected={exp_entry_time.isoformat()} got={entry_time.isoformat()}",
                    )
                )
            if abs(float(exp_entry_yes) - float(entry_yes_price)) > 1e-6:
                violations.append(
                    Violation(
                        "entry_yes_price_mismatch",
                        day.isoformat(),
                        f"expected={float(exp_entry_yes)} got={float(entry_yes_price)}",
                    )
                )

        # --- Feature store leakage spot-checks for this trade day ---
        try:
            fr = feats.loc[day]
        except KeyError:
            violations.append(Violation("missing_feature_store_row", day.isoformat(), "no feature_store row"))
            continue

        cutoff_feat = pd.to_datetime(fr["cutoff_utc"], utc=True).to_pydatetime()
        expected_cutoff = compute_cutoff_utc(day)
        if abs((expected_cutoff - cutoff_feat).total_seconds()) > 60.0:
            violations.append(
                Violation(
                    "cutoff_utc_mismatch",
                    day.isoformat(),
                    f"expected={expected_cutoff.isoformat()} got={cutoff_feat.isoformat()}",
                )
            )

        # Recompute minute-derived stats using only <= cutoff.
        df_1m = minute_by_year[day.year]
        series_5m = series5_by_year[day.year]
        day_start_utc, day_end_utc = day_start_end_utc(day)

        partial_end = min(cutoff_feat, day_end_utc) - timedelta(minutes=5)
        if partial_end < day_start_utc:
            tmax_recalc = float("nan")
        else:
            idx = pd.date_range(day_start_utc, partial_end, freq="5min")
            s5 = series_5m.reindex(idx)
            tmax_recalc = float(s5.max(skipna=True))
        tmax_stored = float(fr["tmax_sofar"])
        if not (math.isfinite(tmax_recalc) and abs(tmax_recalc - tmax_stored) <= 1e-6):
            violations.append(Violation("tmax_sofar_mismatch", day.isoformat(), f"stored={tmax_stored} recalc={tmax_recalc}"))

        minute_slice = df_1m.loc[day_start_utc:cutoff_feat]
        expected_minutes = int(((cutoff_feat - day_start_utc).total_seconds() / 60.0) + 1)
        coverage_recalc = float(len(minute_slice) / expected_minutes) if expected_minutes > 0 else float("nan")
        coverage_stored = float(fr["coverage_frac"])
        if abs(coverage_recalc - coverage_stored) > 1e-9:
            violations.append(
                Violation("coverage_frac_mismatch", day.isoformat(), f"stored={coverage_stored} recalc={coverage_recalc}")
            )

        if len(minute_slice) > 0:
            last_gap_recalc = float((cutoff_feat - minute_slice.index.max()).total_seconds() / 60.0)
        else:
            last_gap_recalc = float("nan")
        last_gap_stored = float(fr["last_gap_minutes"])
        if not (math.isnan(last_gap_recalc) and math.isnan(last_gap_stored)) and abs(last_gap_recalc - last_gap_stored) > 1e-6:
            violations.append(
                Violation("last_gap_minutes_mismatch", day.isoformat(), f"stored={last_gap_stored} recalc={last_gap_recalc}")
            )

        # MOS as-of recompute (DB query). This is slow but we're only doing it for trade days.
        mos_recalc, mos_meta = mos_x_mean_asof_le_cutoff(day, cutoff_feat)
        mos_stored = float(fr["mos_x_mean"])
        mos_ok = (math.isfinite(mos_recalc) and abs(mos_recalc - mos_stored) <= 1e-6) or (
            (not math.isfinite(mos_recalc)) and (not math.isfinite(mos_stored))
        )
        if not mos_ok:
            violations.append(
                Violation(
                    "mos_x_mean_mismatch",
                    day.isoformat(),
                    f"stored={mos_stored} recalc={mos_recalc} meta={mos_meta}",
                )
            )

    audit = {
        "audit_type": "live_trade_flow_audit",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "backtest_dir": str(backtest_dir),
        "summary": summary,
        "latest_hit_model_bundle": latest_bundle,
        "n_trades_checked": int(len(trades)),
        "n_violations": int(len(violations)),
        "audit_pass": bool(len(violations) == 0),
        "violations": [v.__dict__ for v in violations],
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    md = []
    md.append("# Live Trade Flow Audit\n")
    md.append(f"- Backtest dir: `{backtest_dir}`")
    md.append(f"- Trades checked: `{len(trades)}`")
    md.append(f"- Audit pass: `{len(violations) == 0}`")
    md.append(f"- Violations: `{len(violations)}`\n")
    md.append("## Scope\n")
    md.append("- Trade entry timing/threshold rules (from `trades_entered.csv`)")
    md.append("- Kalshi cutoff + entry prices (from `data/kalshi_backtest_data/*.csv`)")
    md.append("- Cutoff-time features recomputed from raw 1m (tmax_sofar/coverage/last_gap)")
    md.append("- MOS `mos_x_mean` recomputed from DB with `asof_utc <= cutoff_utc`\n")

    md.append("## Summary\n")
    for k in [
        "station_id",
        "model",
        "backtest_start",
        "backtest_end",
        "bridge_mode",
        "bridge_train_start",
        "bridge_train_end",
        "p_hit_gate",
        "min_win_prob",
        "edge_prob",
        "risk_model",
        "fixed_risk_fraction",
        "kelly_fraction",
        "entered_trades",
        "win_rate",
        "profit_factor",
        "end_balance",
        "max_drawdown_pct",
    ]:
        if k in summary:
            md.append(f"- `{k}`: `{summary[k]}`")
    md.append("")

    md.append("## Hit-Model Bundle\n")
    if latest_bundle is None:
        md.append("- No exported bundle found under `artifacts/model_bundles/hit1830_v6/<model>/...`.")
        md.append("- Run: `python tools/early_maxout_strategy/export_b6_exp20_bundle.py`")
    else:
        md.append(f"- Latest bundle dir: `{latest_bundle['bundle_dir']}`")
        meta = latest_bundle.get("meta", {})
        validation = meta.get("validation", {}) if isinstance(meta, dict) else {}
        if isinstance(validation, dict):
            md.append(f"- Reproduction check: `{validation}`")
    md.append("")

    md.append("## Violations\n")
    if not violations:
        md.append("- None detected.")
    else:
        for v in violations:
            md.append(f"- `{v.kind}` on `{v.date}`: {v.detail}")
    md.append("")

    (out_dir / "FULL_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print("Wrote", out_dir / "FULL_REPORT.md")
    print("Wrote", out_dir / "audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
