#!/usr/bin/env python3
"""
Backtest KMIA Kalshi bucket strategy.

Required env vars:
  MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD
  GRIBSTREAM_TOKEN (used by tools/live/run_kmia_live.py)

Usage:
  python tools/backtest/backtest_kmia_kalshi_strategy.py --station KMIA \
    --global-json backtesting/out/KXHIGHMIA_2026-01-01_to_2026-01-18.json \
    --prices-dir backtesting/out --min-win 0.60 --min-ev 0.25 \
    --cutoff-local 17:30 --tz America/New_York --output-dir backtests/kmia_kalshi
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import create_engine, text

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback for older Python
    ZoneInfo = None  # type: ignore


DATE_KEYS = ["target_date_local", "target_date", "date", "date_local"]
STATION_KEYS = ["station_id", "station", "icao"]
DATE_COL_CANDIDATES = ["target_date_local", "date_local", "day", "valid_date", "obs_date", "date"]
TMAX_COL_CANDIDATES = ["tmax_f", "cli_tmax_f", "max_temp_f", "tmax", "high_temp"]


@dataclass(frozen=True)
class LiveParams:
    target_date: date
    asof_utc: datetime
    mu: float
    sigma: float


@dataclass(frozen=True)
class BucketInterval:
    label: str
    kind: str  # range, below, above
    a: Optional[int]
    b: Optional[int]
    lower: float
    upper: float


@dataclass(frozen=True)
class Candidate:
    timestamp_utc: datetime
    timestamp_local: datetime
    bucket_label: str
    bucket_lower: float
    bucket_upper: float
    side: str
    yes_price: float
    no_price: float
    entry_price: float
    win_prob: float
    ev: float


def parse_date(value: str) -> date:
    cleaned = value.strip()
    if len(cleaned) == 8 and cleaned.isdigit():
        return datetime.strptime(cleaned, "%Y%m%d").date()
    return datetime.strptime(cleaned, "%Y-%m-%d").date()


def load_global_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Global JSON is empty: {path}")
    stripped = text.lstrip()
    if stripped[0] in "[{":
        return json.loads(text)
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))
    return entries


def extract_target_dates(data: Any) -> List[date]:
    dates: List[date] = []

    def handle_entry(entry: Any) -> None:
        if entry is None:
            return
        if isinstance(entry, str):
            try:
                dates.append(parse_date(entry))
            except Exception:
                return
            return
        if isinstance(entry, dict):
            for key in DATE_KEYS:
                if key in entry and entry[key]:
                    try:
                        dates.append(parse_date(str(entry[key])))
                        return
                    except Exception:
                        pass
            return

    if isinstance(data, dict):
        if "days" in data and isinstance(data["days"], list):
            for item in data["days"]:
                handle_entry(item)
        else:
            handle_entry(data)
    elif isinstance(data, list):
        for item in data:
            handle_entry(item)
    else:
        handle_entry(data)

    unique = sorted({d for d in dates})
    if not unique:
        raise ValueError("No target dates could be extracted from global JSON.")
    return unique


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bucket_probability(mu: float, sigma: float, lower: float, upper: float) -> float:
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")
    if lower == float("-inf"):
        z_upper = (upper - mu) / sigma
        return normal_cdf(z_upper)
    if upper == float("inf"):
        z_lower = (lower - mu) / sigma
        return 1.0 - normal_cdf(z_lower)
    z_lower = (lower - mu) / sigma
    z_upper = (upper - mu) / sigma
    return max(0.0, normal_cdf(z_upper) - normal_cdf(z_lower))


def parse_bucket_interval(label: str) -> BucketInterval:
    if not label:
        raise ValueError("Bucket label is empty.")
    cleaned = label.strip()
    cleaned = cleaned.replace("\u00b0", "")
    text_lower = cleaned.lower()

    range_match = re.search(r"(-?\d+)\s*(?:to|-)\s*(-?\d+)", text_lower)
    if range_match:
        a = int(range_match.group(1))
        b = int(range_match.group(2))
        lower = a - 0.5
        upper = b + 0.5
        return BucketInterval(label=cleaned, kind="range", a=a, b=b, lower=lower, upper=upper)

    below_match = re.search(r"(-?\d+)\s*(?:or\s+below|below)", text_lower)
    if below_match:
        k = int(below_match.group(1))
        return BucketInterval(label=cleaned, kind="below", a=k, b=None, lower=float("-inf"), upper=k + 0.5)

    above_match = re.search(r"(-?\d+)\s*(?:or\s+above|above)", text_lower)
    if above_match:
        k = int(above_match.group(1))
        return BucketInterval(label=cleaned, kind="above", a=k, b=None, lower=k - 0.5, upper=float("inf"))

    raise ValueError(f"Unable to parse bucket label: {label}")

def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def detect_column(header: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized_map = {normalize_col(col): col for col in header}
    for cand in candidates:
        key = normalize_col(cand)
        if key in normalized_map:
            return normalized_map[key]
    return None


def parse_timestamp(value: Any, timestamps_local: bool, tz: ZoneInfo) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1e12:
            seconds = seconds / 1000.0
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if re.match(r"^\d{10,13}(\.\d+)?$", text):
        seconds = float(text)
        if seconds > 1e12:
            seconds = seconds / 1000.0
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if text.endswith("Z") or "+" in text or "T" in text:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz if timestamps_local else timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_yes_price(raw_value: Any) -> Optional[float]:
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except Exception:
        return None
    if value > 1.5:
        value = value / 100.0
    if value < 0 or value > 1:
        return None
    return value


def load_prices_from_kalshi_json(path: Path) -> List[Tuple[datetime, str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    buckets = data.get("buckets", [])
    if not buckets:
        return []
    rows: List[Tuple[datetime, str, float]] = []
    for bucket in buckets:
        label = bucket.get("subtitle") or bucket.get("title") or bucket.get("market_ticker")
        if not label:
            continue
        for row in bucket.get("yes_prices_1m", []):
            ts_val = row.get("end_period_ts") or row.get("end_period_utc")
            ts = parse_timestamp(ts_val, timestamps_local=False, tz=ZoneInfo("UTC"))
            if ts is None:
                continue
            price = normalize_yes_price(row.get("yes_price_dollars"))
            if price is None:
                continue
            rows.append((ts, label, price))
    return rows


def load_prices_from_csv(
    paths: List[Path],
    timestamps_local: bool,
    tz: ZoneInfo,
) -> List[Tuple[datetime, str, float]]:
    rows: List[Tuple[datetime, str, float]] = []

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            header = reader.fieldnames
            ts_col = detect_column(header, ["end_period_ts", "end_period_utc", "timestamp", "time", "datetime", "ts", "minute"])
            price_col = detect_column(header, ["yes_price", "yes", "price_yes", "close_yes", "yes_price_dollars", "close_dollars", "close", "price"])
            bucket_col = detect_column(header, ["bucket", "bucket_label", "market", "market_ticker", "subtitle", "title", "name", "ticker"])

            if ts_col is None or price_col is None:
                raise ValueError(
                    f"Unable to detect timestamp/price columns in {path}. Found: {header}"
                )

            for row in reader:
                ts = parse_timestamp(row.get(ts_col), timestamps_local, tz)
                if ts is None:
                    continue
                raw_price = row.get(price_col)
                price = normalize_yes_price(raw_price)
                if price is None:
                    continue

                label: Optional[str] = None
                if bucket_col:
                    label = row.get(bucket_col)
                if not label:
                    label = bucket_label_from_filename(path.name)
                if not label:
                    raise ValueError(
                        f"Unable to resolve bucket label for {path}. Columns: {header}"
                    )

                rows.append((ts, label, price))
    return rows


def bucket_label_from_filename(name: str) -> Optional[str]:
    text = name.replace("_", " ").replace("-", " ")
    match = re.search(r"(-?\d+)\s*(?:to|-)\s*(-?\d+)", text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} to {match.group(2)}"
    match = re.search(r"(-?\d+)\s*(?:or\s+below|below)", text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} or below"
    match = re.search(r"(-?\d+)\s*(?:or\s+above|above)", text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} or above"
    return None


def find_price_files(prices_dir: Path, target_date: date) -> Tuple[List[Path], List[Path]]:
    day_token = target_date.strftime("%y%b%d").lower()
    iso_token = target_date.isoformat()
    csv_paths = []
    json_paths = []
    for path in prices_dir.glob("*"):
        if path.is_dir():
            continue
        name_lower = path.name.lower()
        if path.suffix.lower() == ".csv" and (day_token in name_lower or iso_token in name_lower):
            csv_paths.append(path)
        if path.suffix.lower() == ".json" and (iso_token in name_lower or day_token in name_lower):
            json_paths.append(path)
    return csv_paths, json_paths


def load_price_rows_for_day(
    prices_dir: Path,
    target_date: date,
    timestamps_local: bool,
    tz: ZoneInfo,
) -> List[Tuple[datetime, str, float]]:
    csv_paths, json_paths = find_price_files(prices_dir, target_date)
    if json_paths:
        for path in json_paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            if "buckets" in data:
                return load_prices_from_kalshi_json(path)
    if not csv_paths:
        return []
    return load_prices_from_csv(csv_paths, timestamps_local, tz)


def build_run_id(min_win: float, min_ev: float, cutoff: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    win_tag = f"win{int(round(min_win * 100))}"
    ev_tag = f"ev{int(round(min_ev * 100))}"
    cut_tag = f"cut{cutoff.replace(':', '')}"
    return f"{ts}_{win_tag}_{ev_tag}_{cut_tag}"


def ensure_zoneinfo(tz_name: str) -> ZoneInfo:
    if ZoneInfo is None:
        raise RuntimeError("zoneinfo is required (Python 3.9+).")
    return ZoneInfo(tz_name)


def parse_cutoff_time(value: str) -> dtime:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError("cutoff-local must be in HH:MM")
    return dtime(hour=int(parts[0]), minute=int(parts[1]))


def parse_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def find_json_from_stdout(output: str) -> dict:
    decoder = json.JSONDecoder()
    last_obj: Optional[dict] = None
    idx = 0
    while True:
        brace = output.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(output[brace:])
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        if isinstance(obj, dict):
            last_obj = obj
        idx = brace + end
    if last_obj is None:
        raise ValueError("Live output JSON not found in stdout.")
    return last_obj


def run_live_params(
    station_id: str,
    target_date: date,
    cache_dir: Path,
    use_cache: bool,
) -> LiveParams:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{station_id}_{target_date.isoformat()}.json"
    if use_cache and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cmd = [
            sys.executable,
            str(Path("tools") / "live" / "run_kmia_live.py"),
            "--station",
            station_id,
            "--target-date",
            target_date.strftime("%Y%m%d"),
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(Path.cwd()))
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"run_kmia_live failed for {target_date}:\n{result.stdout}\n{result.stderr}"
            )
        data = find_json_from_stdout(result.stdout)
        cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    normal = data.get("normal_dist") or {}
    mu = normal.get("mu")
    sigma = normal.get("sigma")
    if mu is None or sigma is None:
        raise ValueError(f"Missing normal_dist.mu/sigma for {target_date}: {data}")
    asof_text = data.get("asof_utc")
    if not asof_text:
        raise ValueError(f"Missing asof_utc in live output for {target_date}")
    asof_dt = datetime.fromisoformat(asof_text.replace("Z", "+00:00")).astimezone(timezone.utc)
    return LiveParams(target_date=target_date, asof_utc=asof_dt, mu=float(mu), sigma=float(sigma))

def connect_db_from_env():
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    db = os.getenv("MYSQL_DB", "weather_predictionmarkets")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)


def detect_cli_columns(engine) -> Tuple[str, str, str, bool]:
    cols = []
    with engine.begin() as conn:
        res = conn.execute(text("DESCRIBE cli_daily"))
        for row in res:
            cols.append(str(row[0]))
    if not cols:
        raise ValueError("cli_daily has no columns.")

    station_col = next((c for c in cols if c in STATION_KEYS), None)
    date_col = next((c for c in cols if c in DATE_COL_CANDIDATES), None)
    tmax_col = next((c for c in cols if c in TMAX_COL_CANDIDATES), None)

    if station_col is None or date_col is None or tmax_col is None:
        raise ValueError(
            "Unable to detect cli_daily columns. "
            f"station={station_col}, date={date_col}, tmax={tmax_col}, columns={cols}"
        )

    with engine.begin() as conn:
        res = conn.execute(text("DESCRIBE cli_daily"))
        dtype_map = {str(row[0]): str(row[1]).lower() for row in res}
    date_is_datetime = "datetime" in dtype_map.get(date_col, "") or "timestamp" in dtype_map.get(date_col, "")
    return station_col, date_col, tmax_col, date_is_datetime


def fetch_truth(engine, station_id: str, target_date: date, station_col: str, date_col: str, tmax_col: str, date_is_datetime: bool) -> Optional[float]:
    if date_is_datetime:
        sql = f"SELECT {tmax_col} FROM cli_daily WHERE {station_col} = :station AND DATE({date_col}) = :date LIMIT 1"
    else:
        sql = f"SELECT {tmax_col} FROM cli_daily WHERE {station_col} = :station AND {date_col} = :date LIMIT 1"
    with engine.begin() as conn:
        value = conn.execute(text(sql), {"station": station_id, "date": target_date}).scalar()
    if value is None:
        return None
    return float(value)


def actual_int_from_value(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    rounded = int(round(value))
    return rounded


def bucket_contains(bucket: BucketInterval, actual_int: int) -> bool:
    if bucket.kind == "below" and bucket.a is not None:
        return actual_int <= bucket.a
    if bucket.kind == "above" and bucket.a is not None:
        return actual_int >= bucket.a
    if bucket.kind == "range" and bucket.a is not None and bucket.b is not None:
        return bucket.a <= actual_int <= bucket.b
    return False


def choose_better(candidate: Candidate, best: Optional[Candidate]) -> Candidate:
    if best is None:
        return candidate
    if candidate.ev > best.ev + 1e-9:
        return candidate
    if abs(candidate.ev - best.ev) <= 1e-9:
        if candidate.win_prob > best.win_prob + 1e-9:
            return candidate
        if abs(candidate.win_prob - best.win_prob) <= 1e-9:
            if candidate.timestamp_utc < best.timestamp_utc:
                return candidate
            if candidate.timestamp_utc == best.timestamp_utc:
                if (candidate.bucket_label, candidate.side) < (best.bucket_label, best.side):
                    return candidate
    return best


def compute_bin_counts(values: List[float]) -> Dict[str, int]:
    bins = [(0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.85),
            (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
    counts: Dict[str, int] = {f"{a:.2f}-{b:.2f}": 0 for a, b in bins}
    for v in values:
        for a, b in bins:
            if a <= v < b:
                counts[f"{a:.2f}-{b:.2f}"] += 1
                break
    return counts


def write_report_md(path: Path, summary: dict) -> None:
    lines = []
    lines.append("# KMIA Kalshi Backtest Report")
    lines.append("")
    lines.append("## Core Counts")
    for key in ["total_days", "days_with_truth", "trades_taken", "no_trade_days", "missing_price_days", "missing_truth_days"]:
        lines.append(f"- {key}: {summary.get(key)}")
    lines.append("")
    lines.append("## Risk Setup")
    for key in ["starting_balance", "risk_pct", "risk_per_trade", "ending_balance", "selection_mode"]:
        lines.append(f"- {key}: {summary.get(key)}")
    lines.append("")
    lines.append("## Performance")
    for key in [
        "realized_win_rate",
        "realized_total_profit",
        "realized_avg_profit_per_trade",
        "realized_median_profit_per_trade",
        "risk_realized_total_profit",
        "risk_realized_avg_profit_per_trade",
        "risk_realized_median_profit_per_trade",
        "average_rr",
        "profit_factor",
        "expected_total_ev",
        "expected_avg_ev_per_trade",
        "average_model_win_prob",
        "average_entry_price",
    ]:
        lines.append(f"- {key}: {summary.get(key)}")
    lines.append("")
    lines.append("## Daily Breakdown")
    lines.append(f"- daily_breakdown_path: {summary.get('daily_breakdown_path')}")
    lines.append("")
    lines.append("## Rule Compliance")
    lines.append(f"- minimum_win_prob: {summary.get('minimum_win_prob')}")
    lines.append(f"- minimum_ev: {summary.get('minimum_ev')}")
    lines.append("")
    lines.append("## Calibration Check")
    lines.append(f"- expected_wins: {summary.get('expected_wins')}")
    lines.append(f"- actual_wins: {summary.get('actual_wins')}")
    lines.append(f"- win_diff: {summary.get('win_diff')}")
    lines.append(f"- win_ratio: {summary.get('win_ratio')}")
    lines.append("")
    lines.append("## Top Trades by EV")
    for item in summary.get("top_trades_by_ev", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Worst Trades by Profit")
    for item in summary.get("worst_trades_by_profit", []):
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest KMIA Kalshi bucket strategy.")
    ap.add_argument("--station", default="KMIA")
    ap.add_argument("--global-json", required=True, help="Global JSON or JSONL with target dates.")
    ap.add_argument("--prices-dir", required=True, help="Folder with minute price CSVs or Kalshi JSONs.")
    ap.add_argument("--min-win", type=float, default=0.60)
    ap.add_argument("--min-ev", type=float, default=0.25)
    ap.add_argument("--cutoff-local", default="17:30")
    ap.add_argument("--cutoff-utc", default="16:00", help="Hard UTC cutoff on target day (HH:MM).")
    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--timestamps-local", action="store_true", help="Treat timestamps as local time.")
    ap.add_argument("--assume-no-complement", default="true", help="Use NO price = 1 - YES price.")
    ap.add_argument("--selection-mode", choices=["best", "first"], default="first")
    ap.add_argument("--output-dir", default="backtests/kmia_kalshi")
    ap.add_argument("--starting-balance", type=float, default=1500.0)
    ap.add_argument("--risk-pct", type=float, default=0.05, help="Risk per trade as fraction of balance.")
    ap.add_argument("--no-cache-live", action="store_true", help="Disable live params cache.")
    ap.add_argument("--trace-top-n", type=int, default=20)
    args = ap.parse_args()

    station_id = args.station
    tz = ensure_zoneinfo(args.tz)
    cutoff_time = parse_cutoff_time(args.cutoff_local)
    cutoff_time_utc = parse_cutoff_time(args.cutoff_utc)
    prices_dir = Path(args.prices_dir)
    global_json_path = Path(args.global_json)

    global_data = load_global_json(global_json_path)
    target_days = extract_target_dates(global_data)

    output_dir = Path(args.output_dir)
    run_id = build_run_id(args.min_win, args.min_ev, args.cutoff_local)
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = run_dir / "live_params"
    use_cache = not args.no_cache_live
    assume_no_complement = parse_bool(args.assume_no_complement)

    engine = connect_db_from_env()
    station_col, date_col, tmax_col, date_is_datetime = detect_cli_columns(engine)

    ledger_path = run_dir / "trades.csv"
    trace_path = run_dir / "daily_trace.jsonl"
    breakdown_path = run_dir / "daily_breakdown.csv"
    report_json_path = run_dir / "report.json"
    report_md_path = run_dir / "report.md"

    ledger_fields = [
        "target_date_local",
        "asof_utc",
        "mu",
        "sigma",
        "cutoff_local",
        "cutoff_utc",
        "cutoff_utc_rule",
        "cutoff_utc_effective",
        "trade_taken",
        "entry_timestamp_utc",
        "entry_timestamp_local",
        "bucket_label",
        "bucket_L",
        "bucket_U",
        "side",
        "yes_price",
        "no_price",
        "entry_price",
        "model_win_prob",
        "model_win_pct",
        "model_ev",
        "risk_per_trade",
        "contracts",
        "rr",
        "realized_profit_risked",
        "dollar_pnl",
        "actual_tmax_f",
        "outcome",
        "realized_profit",
    ]

    totals = {
        "total_days": len(target_days),
        "days_with_truth": 0,
        "trades_taken": 0,
        "no_trade_days": 0,
        "missing_price_days": 0,
        "missing_truth_days": 0,
    }
    trade_results: List[Dict[str, Any]] = []
    breakdown_rows: List[Dict[str, Any]] = []

    with (
        ledger_path.open("w", encoding="utf-8", newline="") as ledger_f,
        trace_path.open("w", encoding="utf-8") as trace_f,
        breakdown_path.open("w", encoding="utf-8", newline="") as breakdown_f,
    ):
        writer = csv.DictWriter(ledger_f, fieldnames=ledger_fields)
        writer.writeheader()
        breakdown_fields = [
            "target_date_local",
            "bucket_label",
            "bucket_L",
            "bucket_U",
            "mu",
            "sigma",
            "side",
            "entry_price",
            "yes_price",
            "no_price",
            "model_win_prob",
            "model_win_pct",
            "model_ev",
            "dollar_pnl",
            "outcome",
            "entry_timestamp_utc",
            "entry_timestamp_local",
        ]
        breakdown_writer = csv.DictWriter(breakdown_f, fieldnames=breakdown_fields)
        breakdown_writer.writeheader()

        risk_per_trade = args.starting_balance * args.risk_pct

        for day in target_days:
            cutoff_local = datetime.combine(day, cutoff_time, tzinfo=tz)
            cutoff_utc = cutoff_local.astimezone(timezone.utc)
            cutoff_utc_rule = datetime.combine(day, cutoff_time_utc, tzinfo=timezone.utc)
            cutoff_utc_effective = cutoff_utc if cutoff_utc < cutoff_utc_rule else cutoff_utc_rule

            live_params = run_live_params(station_id, day, cache_dir, use_cache)
            mu = live_params.mu
            sigma = live_params.sigma

            price_rows = load_price_rows_for_day(prices_dir, day, args.timestamps_local, tz)
            if not price_rows:
                totals["missing_price_days"] += 1

            buckets_cache: Dict[str, BucketInterval] = {}
            prob_cache: Dict[str, float] = {}

            best: Optional[Candidate] = None
            candidates_top: List[Candidate] = []

            price_rows_sorted = sorted(price_rows, key=lambda r: (r[0], str(r[1])))
            for ts_utc, label, yes_price in price_rows_sorted:
                if ts_utc > cutoff_utc_effective:
                    continue
                if label not in buckets_cache:
                    interval = parse_bucket_interval(label)
                    buckets_cache[label] = interval
                    prob_cache[label] = bucket_probability(mu, sigma, interval.lower, interval.upper)
                interval = buckets_cache[label]
                p_yes = prob_cache[label]
                p_no = 1.0 - p_yes

                no_price = 1.0 - yes_price if assume_no_complement else (1.0 - yes_price)

                ts_local = ts_utc.astimezone(tz)

                local_best: Optional[Candidate] = None
                for side, price, win_prob in (
                    ("YES", yes_price, p_yes),
                    ("NO", no_price, p_no),
                ):
                    ev = win_prob - price
                    candidate = Candidate(
                        timestamp_utc=ts_utc,
                        timestamp_local=ts_local,
                        bucket_label=label,
                        bucket_lower=interval.lower,
                        bucket_upper=interval.upper,
                        side=side,
                        yes_price=yes_price,
                        no_price=no_price,
                        entry_price=price,
                        win_prob=win_prob,
                        ev=ev,
                    )
                    candidates_top.append(candidate)
                    if win_prob >= args.min_win and ev >= args.min_ev:
                        if args.selection_mode == "best":
                            best = choose_better(candidate, best)
                        elif best is None:
                            local_best = choose_better(candidate, local_best)
                if args.selection_mode == "first" and best is None and local_best is not None:
                    best = local_best

            candidates_top = sorted(
                candidates_top,
                key=lambda c: (-c.ev, -c.win_prob, c.timestamp_utc, c.bucket_label, c.side),
            )[: args.trace_top_n]

            p_sum = sum(prob_cache.values())
            if abs(p_sum - 1.0) > 0.01:
                print(
                    f"WARNING {day}: bucket YES prob sum {p_sum:.4f} deviates from 1.0",
                    file=sys.stderr,
                )

            trade_taken = best is not None
            if trade_taken:
                totals["trades_taken"] += 1
            else:
                totals["no_trade_days"] += 1

            truth_value = fetch_truth(engine, station_id, day, station_col, date_col, tmax_col, date_is_datetime)
            if truth_value is None:
                totals["missing_truth_days"] += 1
            else:
                totals["days_with_truth"] += 1

            actual_int = actual_int_from_value(truth_value)
            outcome = "NO_TRADE" if not trade_taken else "NO_TRUTH" if truth_value is None else "LOSS"
            realized_profit = None
            realized_profit_risked = None
            rr = None
            contracts = None
            if trade_taken and truth_value is not None and actual_int is not None:
                interval = buckets_cache[best.bucket_label]  # type: ignore[arg-type]
                win = bucket_contains(interval, actual_int)
                if best.side == "NO":
                    win = not win
                outcome = "WIN" if win else "LOSS"
                realized_profit = (1.0 - best.entry_price) if win else (-best.entry_price)
                if best.entry_price > 0:
                    contracts = risk_per_trade / best.entry_price
                    realized_profit_risked = (1.0 - best.entry_price) * contracts if win else (-risk_per_trade)
                    rr = realized_profit_risked / risk_per_trade

            if trade_taken and realized_profit_risked is not None:
                dollar_pnl = realized_profit_risked
            elif not trade_taken:
                dollar_pnl = 0.0
            else:
                dollar_pnl = ""

            ledger_row = {
                "target_date_local": day.isoformat(),
                "asof_utc": live_params.asof_utc.isoformat().replace("+00:00", "Z"),
                "mu": mu,
                "sigma": sigma,
                "cutoff_local": cutoff_local.isoformat(),
                "cutoff_utc": cutoff_utc.isoformat().replace("+00:00", "Z"),
                "cutoff_utc_rule": cutoff_utc_rule.isoformat().replace("+00:00", "Z"),
                "cutoff_utc_effective": cutoff_utc_effective.isoformat().replace("+00:00", "Z"),
                "trade_taken": trade_taken,
                "entry_timestamp_utc": best.timestamp_utc.isoformat().replace("+00:00", "Z") if trade_taken else "",
                "entry_timestamp_local": best.timestamp_local.isoformat() if trade_taken else "",
                "bucket_label": best.bucket_label if trade_taken else "",
                "bucket_L": best.bucket_lower if trade_taken else "",
                "bucket_U": best.bucket_upper if trade_taken else "",
                "side": best.side if trade_taken else "",
                "yes_price": best.yes_price if trade_taken else "",
                "no_price": best.no_price if trade_taken else "",
                "entry_price": best.entry_price if trade_taken else "",
                "model_win_prob": best.win_prob if trade_taken else "",
                "model_win_pct": (best.win_prob * 100.0) if trade_taken else "",
                "model_ev": best.ev if trade_taken else "",
                "risk_per_trade": risk_per_trade if trade_taken else "",
                "contracts": contracts if trade_taken else "",
                "rr": rr if rr is not None else "",
                "realized_profit_risked": realized_profit_risked if realized_profit_risked is not None else "",
                "dollar_pnl": dollar_pnl,
                "actual_tmax_f": truth_value if truth_value is not None else "",
                "outcome": outcome,
                "realized_profit": realized_profit if realized_profit is not None else "",
            }
            writer.writerow(ledger_row)

            trade_results.append(ledger_row)
            breakdown_row = {
                "target_date_local": day.isoformat(),
                "bucket_label": best.bucket_label if trade_taken else "",
                "bucket_L": best.bucket_lower if trade_taken else "",
                "bucket_U": best.bucket_upper if trade_taken else "",
                "mu": mu,
                "sigma": sigma,
                "side": best.side if trade_taken else "",
                "entry_price": best.entry_price if trade_taken else "",
                "yes_price": best.yes_price if trade_taken else "",
                "no_price": best.no_price if trade_taken else "",
                "model_win_prob": best.win_prob if trade_taken else "",
                "model_win_pct": (best.win_prob * 100.0) if trade_taken else "",
                "model_ev": best.ev if trade_taken else "",
                "dollar_pnl": dollar_pnl,
                "outcome": outcome,
                "entry_timestamp_utc": best.timestamp_utc.isoformat().replace("+00:00", "Z") if trade_taken else "",
                "entry_timestamp_local": best.timestamp_local.isoformat() if trade_taken else "",
            }
            breakdown_writer.writerow(breakdown_row)
            breakdown_rows.append(breakdown_row)

            trace_obj = {
                "target_date_local": day.isoformat(),
                "trade": ledger_row,
                "top_candidates": [
                    {
                        "timestamp_utc": c.timestamp_utc.isoformat().replace("+00:00", "Z"),
                        "timestamp_local": c.timestamp_local.isoformat(),
                        "bucket_label": c.bucket_label,
                        "side": c.side,
                        "price": c.entry_price,
                        "win_prob": c.win_prob,
                        "ev": c.ev,
                    }
                    for c in candidates_top
                ],
            }
            trace_f.write(json.dumps(trace_obj) + "\n")

            if trade_taken:
                print(
                    f"{day} TRADE {best.bucket_label} {best.side} "
                    f"{best.timestamp_local.strftime('%H:%M')} price={best.entry_price:.4f} "
                    f"win%={best.win_prob:.3f} ev={best.ev:.3f} outcome={outcome}"
                )
            else:
                print(f"{day} NO TRADE outcome={outcome}")

    trades_with_truth = [row for row in trade_results if row["trade_taken"] and row["outcome"] in ("WIN", "LOSS")]
    realized_profits = [
        row["realized_profit"] for row in trades_with_truth if isinstance(row["realized_profit"], (int, float))
    ]
    risk_profits = [
        row["realized_profit_risked"] for row in trades_with_truth if isinstance(row["realized_profit_risked"], (int, float))
    ]
    rr_values = [row["rr"] for row in trades_with_truth if isinstance(row["rr"], (int, float))]
    model_evs = [row["model_ev"] for row in trade_results if isinstance(row["model_ev"], (int, float))]
    model_win_probs = [row["model_win_prob"] for row in trade_results if isinstance(row["model_win_prob"], (int, float))]
    entry_prices = [row["entry_price"] for row in trade_results if isinstance(row["entry_price"], (int, float))]

    wins = sum(1 for row in trades_with_truth if row["outcome"] == "WIN")
    expected_wins = sum(row["model_win_prob"] for row in trades_with_truth if isinstance(row["model_win_prob"], (int, float)))

    gross_profit = sum(p for p in risk_profits if p > 0) if risk_profits else 0.0
    gross_loss = abs(sum(p for p in risk_profits if p < 0)) if risk_profits else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    summary = {
        **totals,
        "starting_balance": args.starting_balance,
        "risk_pct": args.risk_pct,
        "risk_per_trade": risk_per_trade,
        "ending_balance": args.starting_balance + (sum(risk_profits) if risk_profits else 0.0),
        "selection_mode": args.selection_mode,
        "realized_win_rate": (wins / len(trades_with_truth)) if trades_with_truth else 0.0,
        "realized_total_profit": sum(realized_profits) if realized_profits else 0.0,
        "realized_avg_profit_per_trade": (sum(realized_profits) / len(realized_profits)) if realized_profits else 0.0,
        "realized_median_profit_per_trade": sorted(realized_profits)[len(realized_profits) // 2] if realized_profits else 0.0,
        "risk_realized_total_profit": sum(risk_profits) if risk_profits else 0.0,
        "risk_realized_avg_profit_per_trade": (sum(risk_profits) / len(risk_profits)) if risk_profits else 0.0,
        "risk_realized_median_profit_per_trade": sorted(risk_profits)[len(risk_profits) // 2] if risk_profits else 0.0,
        "average_rr": (sum(rr_values) / len(rr_values)) if rr_values else 0.0,
        "profit_factor": profit_factor,
        "expected_total_ev": sum(model_evs) if model_evs else 0.0,
        "expected_avg_ev_per_trade": (sum(model_evs) / len(model_evs)) if model_evs else 0.0,
        "average_model_win_prob": (sum(model_win_probs) / len(model_win_probs)) if model_win_probs else 0.0,
        "average_entry_price": (sum(entry_prices) / len(entry_prices)) if entry_prices else 0.0,
        "minimum_win_prob": min(model_win_probs) if model_win_probs else None,
        "minimum_ev": min(model_evs) if model_evs else None,
        "win_bins": compute_bin_counts(model_win_probs),
        "expected_wins": expected_wins,
        "actual_wins": wins,
        "win_diff": wins - expected_wins,
        "win_ratio": (wins / expected_wins) if expected_wins else None,
        "daily_breakdown_path": str(breakdown_path),
        "daily_breakdown": breakdown_rows,
    }

    top_trades = sorted(
        [row for row in trades_with_truth if isinstance(row["model_ev"], (int, float))],
        key=lambda r: r["model_ev"],
        reverse=True,
    )[:10]
    summary["top_trades_by_ev"] = [
        f"{row['target_date_local']} {row['bucket_label']} {row['side']} ev={row['model_ev']:.4f} outcome={row['outcome']}"
        for row in top_trades
    ]

    worst_trades = sorted(
        [row for row in trades_with_truth if isinstance(row["realized_profit"], (int, float))],
        key=lambda r: r["realized_profit"],
    )[:10]
    summary["worst_trades_by_profit"] = [
        f"{row['target_date_local']} {row['bucket_label']} {row['side']} profit={row['realized_profit']:.4f} win%={row['model_win_prob']:.3f}"
        for row in worst_trades
    ]

    report_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report_md(report_md_path, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
