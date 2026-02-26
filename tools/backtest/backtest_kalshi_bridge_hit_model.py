from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pymysql

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


REPO = Path(__file__).resolve().parents[2]

STATION_ID = "KMIA"

# NOTE: For this implementation we keep the existing "hit-by-cutoff" model (B6_EXP20),
# and build a bridge to NWS/settlement truth using DB `station_daily_truth`.
MODEL_DIR = REPO / "artifacts" / "experiments" / "KMIA" / "early_maxout_strategy" / "B6" / "B6_EXP20_GAM_RESIDUAL"
PREDS_TEST_PATH = MODEL_DIR / "preds_test.parquet"
FEATURES_PATH = REPO / "cache" / "hit1830_v6_features.parquet"

KALSHI_DIR = REPO / "data" / "kalshi_backtest_data"

DEFAULT_BACKTEST_START = date(2025, 1, 1)
DEFAULT_BACKTEST_END = date(2025, 12, 31)
DEFAULT_BRIDGE_MODE = "expanding"  # expanding|trailing
DEFAULT_TRAILING_YEARS = 10
DEFAULT_REQUIRE_P_HIT_GE = 0.5

DEFAULT_EDGE_PROB = 0.15
DEFAULT_MIN_WIN_PROB = 0.65
DEFAULT_FIXED_RISK_FRACTION = 0.035
DEFAULT_KELLY_FRACTION = 0.3
START_BALANCE = 2700.0

# Bridge context binning (simple + robust)
MIN_SAMPLES_BIN = 200

STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm") if ZoneInfo else None


@dataclass(frozen=True)
class BucketInterval:
    label: str
    lower: float
    upper: float


@dataclass(frozen=True)
class BridgeMeta:
    train_start: date
    train_end: date
    n_train_rows: int
    n_train_hit: int
    n_train_not_hit: int
    d_min: float
    d_max: float


@dataclass
class TradeRow:
    date: date
    cutoff_utc: datetime
    p_hit: float
    model_yes: bool
    max_sofar_iem: Optional[float]
    mos_x_mean: Optional[float]
    coverage_frac: Optional[float]
    truth_tmax_f: Optional[float]
    d_truth: Optional[float]
    bridge_scope: str
    bucket_label: Optional[str]
    bucket_p_win: Optional[float]
    bucket_price_at_cutoff: Optional[float]
    threshold_price: Optional[float]
    threshold_cmp: Optional[str]
    trade_side: Optional[str]
    trade_p_win: Optional[float]
    entry_time: Optional[datetime]
    entry_yes_price: Optional[float]
    entry_price: Optional[float]
    shares: Optional[float]
    win: Optional[bool]
    ev_at_entry: Optional[float]
    pnl: float
    balance_after: float
    note: str
    stake_fraction: Optional[float] = None
    stake: Optional[float] = None


def compute_cutoff_utc(day: date) -> datetime:
    if STOCKHOLM_TZ is None:
        raise RuntimeError("ZoneInfo not available; cannot compute cutoff.")
    cutoff_local = datetime(day.year, day.month, day.day, 18, 30, tzinfo=STOCKHOLM_TZ)
    return cutoff_local.astimezone(ZoneInfo("UTC"))


def parse_bucket_interval(label: str) -> BucketInterval:
    cleaned = label.replace("°", "").strip().lower()
    import re

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)", cleaned)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        return BucketInterval(label=label, lower=a - 0.5, upper=b + 0.5)

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:or\s+below|below)", cleaned)
    if m:
        k = float(m.group(1))
        return BucketInterval(label=label, lower=float("-inf"), upper=k + 0.5)

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:or\s+above|above)", cleaned)
    if m:
        k = float(m.group(1))
        return BucketInterval(label=label, lower=k - 0.5, upper=float("inf"))

    raise ValueError(f"Unable to parse bucket label: {label}")


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


def first_price_at_or_above(df: pd.DataFrame, cutoff: datetime, bucket: str, threshold: float) -> Tuple[Optional[datetime], Optional[float]]:
    sub = df[df["timestamp"] >= cutoff][["timestamp", bucket]].dropna()
    if sub.empty:
        return None, None
    hit = sub[sub[bucket] >= threshold]
    if hit.empty:
        return None, None
    row = hit.iloc[0]
    return row["timestamp"].to_pydatetime(), float(row[bucket])


def connect_db() -> pymysql.connections.Connection:
    host = os.environ.get("MYSQL_HOST", "localhost")
    port = int(os.environ.get("MYSQL_PORT", "3306"))
    user = os.environ.get("MYSQL_USER", "root")
    password = os.environ.get("MYSQL_PASSWORD", "root")
    db = os.environ.get("MYSQL_DB", "weather_predictionmarkets")
    return pymysql.connect(host=host, port=port, user=user, password=password, database=db, autocommit=True)


def fetch_truth_tmax(station_id: str, start: date, end: date) -> pd.DataFrame:
    sql = """
        SELECT date_local, tmax_f
        FROM station_daily_truth
        WHERE station_id=%s AND date_local >= %s AND date_local <= %s
        ORDER BY date_local
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (station_id, start, end))
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=["date_local", "tmax_f"])
    if df.empty:
        return df
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.date
    df["tmax_f"] = df["tmax_f"].astype(float)
    return df


def load_hit_features(columns: List[str]) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH, columns=columns)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    return df


def build_d_distributions(
    features: pd.DataFrame,
    truth: pd.DataFrame,
    train_start: date,
    train_end: date,
) -> Tuple[BridgeMeta, Dict[Tuple[int, Optional[int], Optional[int]], Counter], Dict[Tuple[int, Optional[int], Optional[int]], int]]:
    """
    Returns:
      - BridgeMeta
      - counters keyed by (hit_label, cover_bin)
      - counts keyed by same (for fast lookups/fallback)
    """
    feat = features.copy()
    feat = feat[(feat["target_date_local"] >= train_start) & (feat["target_date_local"] <= train_end)].copy()
    if feat.empty:
        raise RuntimeError("No feature rows available in the bridge training window.")

    truth_map = dict(zip(truth["date_local"], truth["tmax_f"]))
    feat["truth_tmax_f"] = feat["target_date_local"].map(truth_map)
    feat = feat[feat["truth_tmax_f"].notna()].copy()
    if feat.empty:
        raise RuntimeError("No joined rows between features and truth for bridge training.")

    # Coverage bin is the most important context feature: when IEM minute coverage is poor,
    # max_sofar at cutoff can be badly biased low, and D can be large.
    def cover_bin(c: float) -> int:
        c = float(c)
        if c >= 0.95:
            return 2  # good
        if c >= 0.80:
            return 1  # mid
        return 0  # bad

    feat["cover_bin"] = feat["coverage_frac"].map(cover_bin).astype(int)

    # D = settlement truth - IEM observed max_sofar at cutoff
    feat["d_truth"] = (feat["truth_tmax_f"].astype(float) - feat["tmax_sofar"].astype(float)).round(1)
    # Discretize to 0.5F bins (matches the occasional .5 in tmax_sofar).
    feat["d_bin"] = (feat["d_truth"] * 2.0).round().astype(int) / 2.0

    counters: Dict[Tuple[int, Optional[int], Optional[int]], Counter] = defaultdict(Counter)
    counts: Dict[Tuple[int, Optional[int], Optional[int]], int] = defaultdict(int)

    for r in feat.itertuples(index=False):
        hit = int(getattr(r, "y_hit_by_cutoff"))
        cb = int(getattr(r, "cover_bin"))
        d = float(getattr(r, "d_bin"))
        # Full key (hit + coverage)
        k_full = (hit, cb, None)
        counters[k_full][d] += 1
        counts[k_full] += 1
        # Global fallback key (hit only)
        k_global = (hit, None, None)
        counters[k_global][d] += 1
        counts[k_global] += 1

    hit_total = counts.get((1, None, None), 0)
    not_total = counts.get((0, None, None), 0)
    all_d = list(counters.get((0, None, None), Counter()).keys()) + list(counters.get((1, None, None), Counter()).keys())
    if not all_d:
        raise RuntimeError("Bridge training produced no D bins.")

    meta = BridgeMeta(
        train_start=min(feat["target_date_local"]),
        train_end=max(feat["target_date_local"]),
        n_train_rows=len(feat),
        n_train_hit=hit_total,
        n_train_not_hit=not_total,
        d_min=float(min(all_d)),
        d_max=float(max(all_d)),
    )
    return meta, counters, counts


def counter_to_probs(counter: Counter) -> Dict[float, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {float(k): v / total for k, v in counter.items()}


def write_bridge_tables(
    out_dir: Path,
    counters: Dict[Tuple[int, Optional[int], Optional[int]], Counter],
    counts: Dict[Tuple[int, Optional[int], Optional[int]], int],
) -> None:
    rows: List[Dict] = []
    for k, counter in counters.items():
        hit, cover_bin, _ = k
        n_total = int(counts.get(k, 0))
        if n_total <= 0:
            continue
        for d_bin, c in sorted(counter.items(), key=lambda kv: float(kv[0])):
            rows.append(
                {
                    "hit_label": int(hit),
                    "cover_bin": cover_bin if cover_bin is not None else "GLOBAL",
                    "d_bin": float(d_bin),
                    "count": int(c),
                    "n_total": n_total,
                    "prob": float(c) / float(n_total),
                }
            )

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.sort_values(by=["hit_label", "cover_bin", "d_bin"], inplace=True, kind="mergesort")
    df.to_csv(out_dir / "bridge_table.csv", index=False, encoding="utf-8")

    # A small summary table per distribution for quick inspection.
    summaries: List[Dict] = []
    for (hit_label, cover_bin), g in df.groupby(["hit_label", "cover_bin"], sort=False):
        mean_d = float((g["d_bin"] * g["prob"]).sum())
        summaries.append(
            {
                "hit_label": int(hit_label),
                "cover_bin": cover_bin,
                "n_total": int(g["n_total"].iloc[0]),
                "mean_d": mean_d,
                "d_min": float(g["d_bin"].min()),
                "d_max": float(g["d_bin"].max()),
            }
        )
    pd.DataFrame(summaries).to_csv(out_dir / "bridge_table_summary.csv", index=False, encoding="utf-8")


def df_to_markdown_simple(df: pd.DataFrame, max_col_width: int = 48) -> str:
    cols = list(df.columns)

    def fmt(v: object) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            s = ""
        else:
            s = str(v)
        s = s.replace("\n", " ").strip()
        if len(s) > max_col_width:
            s = s[: max_col_width - 1] + "…"
        return s

    rows = [[fmt(v) for v in df.iloc[i].tolist()] for i in range(len(df))]
    widths = [len(str(c)) for c in cols]
    for r in rows:
        for j, v in enumerate(r):
            widths[j] = max(widths[j], len(v))

    def make_row(values: List[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"

    header = make_row([str(c) for c in cols])
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = "\n".join(make_row(r) for r in rows)
    return "\n".join([header, sep, body]) + "\n"


def _kelly_full_fraction(p_win: float, entry_price_cents: float) -> float:
    """
    Full Kelly fraction for a binary contract where you pay c per share and
    receive $1 at settlement if you win.

    p_win is the win probability for the side you buy (YES or NO).
    entry_price_cents is the cost (in cents) for the side you buy.

    Returns a fraction of bankroll to stake. (May exceed 1 in theory; caller clamps.)
    """
    p = float(p_win)
    c = float(entry_price_cents) / 100.0
    if not (0.0 <= p <= 1.0):
        return 0.0
    if c <= 0.0 or c >= 1.0:
        return 0.0
    b = (1.0 - c) / c  # profit per $1 risked
    if b <= 0.0:
        return 0.0
    q = 1.0 - p
    return float((b * p - q) / b)


def _max_drawdown_pct(equity: pd.Series) -> float:
    s = pd.to_numeric(equity, errors="coerce").dropna()
    if s.empty:
        return 0.0
    peak = s.cummax()
    dd = (peak - s) / peak
    return float(dd.max() * 100.0)


def get_d_distribution(
    counters: Dict[Tuple[int, Optional[int], Optional[int]], Counter],
    counts: Dict[Tuple[int, Optional[int], Optional[int]], int],
    hit: int,
    cover_bin: int,
) -> Tuple[str, Dict[float, float]]:
    k_full = (hit, cover_bin, None)
    if counts.get(k_full, 0) >= MIN_SAMPLES_BIN:
        return "cover_bin", counter_to_probs(counters[k_full])
    return "global", counter_to_probs(counters[(hit, None, None)])


def mix_distributions(p_hit: float, d_hit: Dict[float, float], d_not: Dict[float, float]) -> Dict[float, float]:
    keys = set(d_hit.keys()) | set(d_not.keys())
    mixed: Dict[float, float] = {}
    for k in keys:
        mixed[k] = p_hit * d_hit.get(k, 0.0) + (1.0 - p_hit) * d_not.get(k, 0.0)
    s = sum(mixed.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in mixed.items()}


def bucket_win_prob(max_sofar: float, d_dist: Dict[float, float], interval: BucketInterval) -> float:
    p = 0.0
    for d, prob in d_dist.items():
        t = max_sofar + d
        if (t > interval.lower) and (t <= interval.upper):
            p += prob
    return float(p)


def _fmt_gate_tag(p_hit_gate: float) -> str:
    if p_hit_gate < 0:
        return "nogate"
    s = f"{float(p_hit_gate):.3f}".rstrip("0").rstrip(".")
    return "phit" + s.replace(".", "p")

def _fmt_prob_tag(prefix: str, value: float) -> str:
    s = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return prefix + s.replace(".", "p")


def _compute_trailing_start(train_end: date, years: int) -> date:
    # Exactly the last N years ending at train_end (inclusive).
    ts = pd.Timestamp(train_end) - pd.DateOffset(years=int(years)) + pd.Timedelta(days=1)
    return ts.date()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Backtest Kalshi KXHIGHMIA using a D-bridge + hit-by-cutoff model.")
    parser.add_argument("--backtest-start", default=DEFAULT_BACKTEST_START.isoformat(), help="Backtest start YYYY-MM-DD")
    parser.add_argument("--backtest-end", default=DEFAULT_BACKTEST_END.isoformat(), help="Backtest end YYYY-MM-DD")
    parser.add_argument(
        "--bridge-mode",
        default=DEFAULT_BRIDGE_MODE,
        choices=["expanding", "trailing"],
        help="Bridge training window type (default: expanding)",
    )
    parser.add_argument(
        "--trailing-years",
        type=int,
        default=DEFAULT_TRAILING_YEARS,
        help="Trailing window size in years (only used for --bridge-mode trailing)",
    )
    parser.add_argument(
        "--bridge-train-end",
        default=None,
        help="Bridge training end date YYYY-MM-DD (default: backtest_start - 1 day)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: backtests/kmia_kalshi/bridge_<mode>_b6_exp20_<start>_<end>_<gate>)",
    )
    parser.add_argument(
        "--p-hit-gate",
        type=float,
        default=DEFAULT_REQUIRE_P_HIT_GE,
        help="Only consider trades when p_hit >= gate. Set to -1 to disable.",
    )
    parser.add_argument(
        "--min-win-prob",
        type=float,
        default=DEFAULT_MIN_WIN_PROB,
        help="Minimum win probability required to consider a trade (default: 0.65)",
    )
    parser.add_argument(
        "--edge-prob",
        type=float,
        default=DEFAULT_EDGE_PROB,
        help="Required edge vs market probability (default: 0.15). Example: 0.10 means 10c edge.",
    )
    parser.add_argument(
        "--risk-model",
        choices=["fixed", "kelly"],
        default="fixed",
        help="Position sizing model (default: fixed).",
    )
    parser.add_argument(
        "--fixed-risk-fraction",
        type=float,
        default=DEFAULT_FIXED_RISK_FRACTION,
        help="Fixed risk fraction per trade (only used for --risk-model fixed). Default: 0.035",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=DEFAULT_KELLY_FRACTION,
        help="Fraction of full Kelly to use (only used for --risk-model kelly). Default: 0.3",
    )

    args = parser.parse_args(argv)

    backtest_start = date.fromisoformat(args.backtest_start)
    backtest_end = date.fromisoformat(args.backtest_end)
    if backtest_end < backtest_start:
        raise SystemExit("--backtest-end must be >= --backtest-start")

    bridge_train_end = (
        date.fromisoformat(args.bridge_train_end) if args.bridge_train_end else (backtest_start - timedelta(days=1))
    )
    if not (bridge_train_end < backtest_start):
        raise RuntimeError("Bridge train end must be < backtest start for leakage safety.")

    trailing_years: Optional[int] = None
    if args.bridge_mode == "expanding":
        bridge_train_start = date(2002, 1, 1)
    else:
        trailing_years = int(args.trailing_years)
        if trailing_years <= 0:
            raise SystemExit("--trailing-years must be > 0")
        bridge_train_start = _compute_trailing_start(bridge_train_end, trailing_years)

    if bridge_train_start > bridge_train_end:
        raise RuntimeError("Bridge train start must be <= bridge train end.")

    p_hit_gate = float(args.p_hit_gate)
    min_win_prob = float(args.min_win_prob)
    edge_prob = float(args.edge_prob)
    risk_model = str(args.risk_model)
    fixed_risk_fraction = float(args.fixed_risk_fraction)
    kelly_fraction = float(args.kelly_fraction)
    if not (0.0 <= min_win_prob <= 1.0):
        raise SystemExit("--min-win-prob must be between 0 and 1")
    if not (0.0 <= edge_prob <= 1.0):
        raise SystemExit("--edge-prob must be between 0 and 1")
    if not (0.0 < fixed_risk_fraction <= 1.0):
        raise SystemExit("--fixed-risk-fraction must be between (0, 1]")
    if not (0.0 <= kelly_fraction <= 1.0):
        raise SystemExit("--kelly-fraction must be between [0, 1]")

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        start_tag = backtest_start.strftime("%Y%m%d")
        end_tag = backtest_end.strftime("%Y%m%d")
        mode_tag = "expanding" if trailing_years is None else f"trailing{trailing_years}y"
        gate_tag = _fmt_gate_tag(p_hit_gate)
        win_tag = _fmt_prob_tag("win", min_win_prob)
        edge_tag = _fmt_prob_tag("edge", edge_prob)
        if risk_model == "fixed":
            risk_tag = _fmt_prob_tag("risk", fixed_risk_fraction)
            risk_tag = f"{risk_model}_{risk_tag}"
        else:
            risk_tag = _fmt_prob_tag("kelly", kelly_fraction)
            risk_tag = f"{risk_model}_{risk_tag}"
        out_dir = (
            REPO
            / "backtests"
            / "kmia_kalshi"
            / f"bridge_{mode_tag}_b6_exp20_{start_tag}_{end_tag}_{gate_tag}_{win_tag}_{edge_tag}_{risk_tag}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model probabilities for backtest days (already calibrated on 2020-2022 val).
    preds = pd.read_parquet(PREDS_TEST_PATH)
    preds["target_date_local"] = pd.to_datetime(preds["target_date_local"]).dt.date
    preds = preds[(preds["target_date_local"] >= backtest_start) & (preds["target_date_local"] <= backtest_end)].copy()
    preds["p_hit"] = preds["p_cal"].astype(float)

    # Load required cutoff-time features (max_sofar + MOS guidance), and IEM hit label for bridge training.
    feat_cols = ["target_date_local", "cutoff_utc", "tmax_sofar", "mos_x_mean", "coverage_frac", "y_hit_by_cutoff"]
    feats = load_hit_features(feat_cols)

    # Truth: settlement daily max temperature (Kalshi settles against this table in this repo).
    truth_train = fetch_truth_tmax(STATION_ID, bridge_train_start, bridge_train_end)
    truth_bt = fetch_truth_tmax(STATION_ID, backtest_start, backtest_end)
    truth_bt_map = dict(zip(truth_bt["date_local"], truth_bt["tmax_f"]))

    # Build bridge distributions (leakage-safe training window).
    meta, counters, counts = build_d_distributions(feats, truth_train, bridge_train_start, bridge_train_end)
    (out_dir / "bridge_meta.json").write_text(json.dumps(meta.__dict__, indent=2, default=str), encoding="utf-8")
    write_bridge_tables(out_dir, counters, counts)

    # Prepare backtest join.
    feats_bt = feats[(feats["target_date_local"] >= backtest_start) & (feats["target_date_local"] <= backtest_end)].copy()
    df_bt = preds.merge(feats_bt, on="target_date_local", how="inner", suffixes=("", "_feat"))
    if df_bt.empty:
        raise RuntimeError("No backtest rows after joining predictions and cutoff features.")

    # Ensure cutoffs are correct and UTC.
    df_bt["cutoff_utc"] = pd.to_datetime(df_bt["cutoff_utc"], utc=True)

    trades: List[TradeRow] = []
    balance = START_BALANCE

    missing_kalshi_files: List[str] = []
    missing_truth_days: List[str] = []

    for r in df_bt.itertuples(index=False):
        day: date = getattr(r, "target_date_local")
        cutoff_utc = getattr(r, "cutoff_utc").to_pydatetime()
        p_hit = float(getattr(r, "p_hit"))
        model_yes = bool(p_hit >= 0.5)

        max_sofar = float(getattr(r, "tmax_sofar")) if getattr(r, "tmax_sofar") is not None else None
        mos_x_mean = float(getattr(r, "mos_x_mean")) if getattr(r, "mos_x_mean") is not None else None
        coverage_frac = float(getattr(r, "coverage_frac")) if getattr(r, "coverage_frac") is not None else None
        truth_tmax = truth_bt_map.get(day)
        if truth_tmax is None:
            missing_truth_days.append(day.isoformat())
            trades.append(
                TradeRow(
                    date=day,
                    cutoff_utc=cutoff_utc,
                    p_hit=p_hit,
                    model_yes=model_yes,
                    max_sofar_iem=max_sofar,
                    mos_x_mean=mos_x_mean,
                    coverage_frac=coverage_frac,
                    truth_tmax_f=None,
                    d_truth=None,
                    bridge_scope="",
                    bucket_label=None,
                    bucket_p_win=None,
                    bucket_price_at_cutoff=None,
                    threshold_price=None,
                    threshold_cmp=None,
                    trade_side=None,
                    trade_p_win=None,
                    entry_time=None,
                    entry_yes_price=None,
                    entry_price=None,
                    shares=None,
                    win=None,
                    ev_at_entry=None,
                    pnl=0.0,
                    balance_after=balance,
                    note="missing_truth",
                )
            )
            continue

        d_truth = None
        if max_sofar is not None:
            d_truth = float(truth_tmax - max_sofar)

        if p_hit_gate >= 0.0 and p_hit < p_hit_gate:
            trades.append(
                TradeRow(
                    date=day,
                    cutoff_utc=cutoff_utc,
                    p_hit=p_hit,
                    model_yes=model_yes,
                    max_sofar_iem=max_sofar,
                    mos_x_mean=mos_x_mean,
                    coverage_frac=coverage_frac,
                    truth_tmax_f=float(truth_tmax),
                    d_truth=d_truth,
                    bridge_scope="",
                    bucket_label=None,
                    bucket_p_win=None,
                    bucket_price_at_cutoff=None,
                    threshold_price=None,
                    threshold_cmp=None,
                    trade_side=None,
                    trade_p_win=None,
                    entry_time=None,
                    entry_yes_price=None,
                    entry_price=None,
                    shares=None,
                    win=None,
                    ev_at_entry=None,
                    pnl=0.0,
                    balance_after=balance,
                    note="p_hit_below_gate",
                )
            )
            continue

        # Match the expected cutoff based on the repo's decision-time rule.
        cutoff_note = ""
        try:
            expected_cutoff = compute_cutoff_utc(day)
            if abs((expected_cutoff - cutoff_utc).total_seconds()) > 60:
                cutoff_note = f"cutoff_mismatch:{expected_cutoff.isoformat()}"
        except Exception:
            pass

        kalshi_path = KALSHI_DIR / f"KMIA_{day.strftime('%Y%m%d')}.csv"
        if not kalshi_path.exists():
            missing_kalshi_files.append(day.isoformat())
            trades.append(
                TradeRow(
                    date=day,
                    cutoff_utc=cutoff_utc,
                    p_hit=p_hit,
                    model_yes=model_yes,
                    max_sofar_iem=max_sofar,
                    mos_x_mean=mos_x_mean,
                    coverage_frac=coverage_frac,
                    truth_tmax_f=float(truth_tmax),
                    d_truth=d_truth,
                    bridge_scope="",
                    bucket_label=None,
                    bucket_p_win=None,
                    bucket_price_at_cutoff=None,
                    threshold_price=None,
                    threshold_cmp=None,
                    trade_side=None,
                    trade_p_win=None,
                    entry_time=None,
                    entry_yes_price=None,
                    entry_price=None,
                    shares=None,
                    win=None,
                    ev_at_entry=None,
                    pnl=0.0,
                    balance_after=balance,
                    note="missing_kalshi_file",
                )
            )
            continue

        # Need cutoff features to compute probabilities.
        if max_sofar is None or not math.isfinite(float(max_sofar)):
            trades.append(
                TradeRow(
                    date=day,
                    cutoff_utc=cutoff_utc,
                    p_hit=p_hit,
                    model_yes=model_yes,
                    max_sofar_iem=max_sofar,
                    mos_x_mean=mos_x_mean,
                    coverage_frac=coverage_frac,
                    truth_tmax_f=float(truth_tmax),
                    d_truth=d_truth,
                    bridge_scope="",
                    bucket_label=None,
                    bucket_p_win=None,
                    bucket_price_at_cutoff=None,
                    threshold_price=None,
                    threshold_cmp=None,
                    trade_side=None,
                    trade_p_win=None,
                    entry_time=None,
                    entry_yes_price=None,
                    entry_price=None,
                    shares=None,
                    win=None,
                    ev_at_entry=None,
                    pnl=0.0,
                    balance_after=balance,
                    note="missing_max_sofar",
                )
            )
            continue

        # Read Kalshi minute prices (updates table).
        kal = pd.read_csv(kalshi_path)
        kal["timestamp"] = pd.to_datetime(kal["timestamp"], utc=True)
        bucket_cols = [c for c in kal.columns if c != "timestamp"]

        if coverage_frac is None or not math.isfinite(float(coverage_frac)):
            cb = 0
        elif float(coverage_frac) >= 0.95:
            cb = 2
        elif float(coverage_frac) >= 0.80:
            cb = 1
        else:
            cb = 0

        scope_hit, d_hit = get_d_distribution(counters, counts, hit=1, cover_bin=cb)
        scope_not, d_not = get_d_distribution(counters, counts, hit=0, cover_bin=cb)
        d_mix = mix_distributions(p_hit, d_hit, d_not)
        bridge_scope = scope_hit if scope_hit == scope_not else f"{scope_hit}|{scope_not}"

        if not d_mix:
            trades.append(
                TradeRow(
                    date=day,
                    cutoff_utc=cutoff_utc,
                    p_hit=p_hit,
                    model_yes=model_yes,
                    max_sofar_iem=max_sofar,
                    mos_x_mean=mos_x_mean,
                    coverage_frac=coverage_frac,
                    truth_tmax_f=float(truth_tmax),
                    d_truth=d_truth,
                    bridge_scope=bridge_scope,
                    bucket_label=None,
                    bucket_p_win=None,
                    bucket_price_at_cutoff=None,
                    threshold_price=None,
                    threshold_cmp=None,
                    trade_side=None,
                    trade_p_win=None,
                    entry_time=None,
                    entry_yes_price=None,
                    entry_price=None,
                    shares=None,
                    win=None,
                    ev_at_entry=None,
                    pnl=0.0,
                    balance_after=balance,
                    note="empty_d_distribution",
                )
            )
            continue

        # Compute bucket probabilities (P(truth in bucket)) from the bridge distribution.
        bucket_probs: Dict[str, float] = {}
        for b in bucket_cols:
            try:
                interval = parse_bucket_interval(b)
            except Exception:
                continue
            bucket_probs[b] = bucket_win_prob(max_sofar, d_mix, interval)

        if not bucket_probs:
            trades.append(
                TradeRow(
                    date=day,
                    cutoff_utc=cutoff_utc,
                    p_hit=p_hit,
                    model_yes=model_yes,
                    max_sofar_iem=max_sofar,
                    mos_x_mean=mos_x_mean,
                    coverage_frac=coverage_frac,
                    truth_tmax_f=float(truth_tmax),
                    d_truth=d_truth,
                    bridge_scope=bridge_scope,
                    bucket_label=None,
                    bucket_p_win=None,
                    bucket_price_at_cutoff=None,
                    threshold_price=None,
                    threshold_cmp=None,
                    trade_side=None,
                    trade_p_win=None,
                    entry_time=None,
                    entry_yes_price=None,
                    entry_price=None,
                    shares=None,
                    win=None,
                    ev_at_entry=None,
                    pnl=0.0,
                    balance_after=balance,
                    note="no_parsable_buckets",
                )
            )
            continue

        # Build eligible YES/NO trade candidates across all buckets.
        # We only have YES-side prices in the CSV; we approximate NO price as (100 - YES).
        candidates: List[Tuple[float, str, str, float, float, float]] = []
        # tuple: (score_at_cutoff, side, bucket_label, p_bucket, yes_price_at_cutoff, threshold_yes_price)
        # score_at_cutoff is the EV if entering immediately at cutoff on that side (higher is better).
        for b, p_bucket in bucket_probs.items():
            yes_at_cutoff = price_at_or_before(kal, cutoff_utc, b)
            if yes_at_cutoff is None or not math.isfinite(float(yes_at_cutoff)):
                continue
            if float(yes_at_cutoff) < 0.0 or float(yes_at_cutoff) > 100.0:
                continue

            m_yes = float(yes_at_cutoff) / 100.0

            # YES trade on this bucket.
            if float(p_bucket) >= min_win_prob:
                threshold_yes = (float(p_bucket) - edge_prob) * 100.0
                if threshold_yes > 0.0:
                    score = float(p_bucket) - m_yes
                    candidates.append((score, "YES", b, float(p_bucket), float(yes_at_cutoff), float(threshold_yes)))

            # NO trade on this bucket: win prob is (1 - p_bucket).
            if (1.0 - float(p_bucket)) >= min_win_prob:
                # NO entry condition expressed as a YES-price threshold:
                # Buy NO when YES >= (p_bucket + EDGE)*100.
                threshold_yes = (float(p_bucket) + edge_prob) * 100.0
                if threshold_yes < 100.0:
                    score = m_yes - float(p_bucket)
                    candidates.append((score, "NO", b, float(p_bucket), float(yes_at_cutoff), float(threshold_yes)))

        if not candidates:
            trades.append(
                TradeRow(
                    date=day,
                    cutoff_utc=cutoff_utc,
                    p_hit=p_hit,
                    model_yes=model_yes,
                    max_sofar_iem=max_sofar,
                    mos_x_mean=mos_x_mean,
                    coverage_frac=coverage_frac,
                    truth_tmax_f=float(truth_tmax),
                    d_truth=d_truth,
                    bridge_scope=bridge_scope,
                    bucket_label=None,
                    bucket_p_win=None,
                    bucket_price_at_cutoff=None,
                    threshold_price=None,
                    threshold_cmp=None,
                    trade_side=None,
                    trade_p_win=None,
                    entry_time=None,
                    entry_yes_price=None,
                    entry_price=None,
                    shares=None,
                    win=None,
                    ev_at_entry=None,
                    pnl=0.0,
                    balance_after=balance,
                    note="no_trade_candidates",
                )
            )
            continue

        # Pick the "best" candidate by EV-at-cutoff score (deterministic, uses only cutoff info).
        candidates.sort(key=lambda t: t[0], reverse=True)
        score_at_cutoff, side, bucket_label, p_bucket, yes_price_at_cutoff, threshold_yes_price = candidates[0]

        threshold_cmp = "<=" if side == "YES" else ">="
        trade_p_win = float(p_bucket) if side == "YES" else float(1.0 - p_bucket)

        entry_time = None
        entry_yes_price = None
        entry_price = None
        if side == "YES":
            if yes_price_at_cutoff <= threshold_yes_price:
                entry_time = cutoff_utc
                entry_yes_price = yes_price_at_cutoff
            else:
                entry_time, entry_yes_price = first_price_at_or_below(kal, cutoff_utc, bucket_label, threshold_yes_price)

            if entry_yes_price is not None:
                entry_price = float(entry_yes_price)
        else:
            if yes_price_at_cutoff >= threshold_yes_price:
                entry_time = cutoff_utc
                entry_yes_price = yes_price_at_cutoff
            else:
                entry_time, entry_yes_price = first_price_at_or_above(kal, cutoff_utc, bucket_label, threshold_yes_price)

            if entry_yes_price is not None:
                entry_price = 100.0 - float(entry_yes_price)

        if entry_time is None or entry_price is None:
            trades.append(
                TradeRow(
                    date=day,
                    cutoff_utc=cutoff_utc,
                    p_hit=p_hit,
                    model_yes=model_yes,
                    max_sofar_iem=max_sofar,
                    mos_x_mean=mos_x_mean,
                    coverage_frac=coverage_frac,
                    truth_tmax_f=float(truth_tmax),
                    d_truth=d_truth,
                    bridge_scope=bridge_scope,
                    bucket_label=bucket_label,
                    bucket_p_win=float(p_bucket),
                    bucket_price_at_cutoff=float(yes_price_at_cutoff),
                    threshold_price=float(threshold_yes_price),
                    threshold_cmp=threshold_cmp,
                    trade_side=side,
                    trade_p_win=float(trade_p_win),
                    entry_time=None,
                    entry_yes_price=None,
                    entry_price=None,
                    shares=None,
                    win=None,
                    ev_at_entry=None,
                    pnl=0.0,
                    balance_after=balance,
                    note="no_entry",
                )
            )
            continue

        if entry_price <= 0:
            trades.append(
                TradeRow(
                    date=day,
                    cutoff_utc=cutoff_utc,
                    p_hit=p_hit,
                    model_yes=model_yes,
                    max_sofar_iem=max_sofar,
                    mos_x_mean=mos_x_mean,
                    coverage_frac=coverage_frac,
                    truth_tmax_f=float(truth_tmax),
                    d_truth=d_truth,
                    bridge_scope=bridge_scope,
                    bucket_label=bucket_label,
                    bucket_p_win=float(p_bucket),
                    bucket_price_at_cutoff=float(yes_price_at_cutoff),
                    threshold_price=float(threshold_yes_price),
                    threshold_cmp=threshold_cmp,
                    trade_side=side,
                    trade_p_win=float(trade_p_win),
                    entry_time=entry_time,
                    entry_yes_price=float(entry_yes_price) if entry_yes_price is not None else None,
                    entry_price=float(entry_price),
                    shares=None,
                    win=None,
                    ev_at_entry=None,
                    pnl=0.0,
                    balance_after=balance,
                    note="invalid_entry_price",
                )
            )
            continue

        interval = parse_bucket_interval(bucket_label)
        bucket_hit = (float(truth_tmax) > interval.lower) and (float(truth_tmax) <= interval.upper)
        win = bucket_hit if side == "YES" else (not bucket_hit)

        if risk_model == "fixed":
            stake_fraction = fixed_risk_fraction
        else:
            kelly_full = _kelly_full_fraction(float(trade_p_win), float(entry_price))
            stake_fraction = kelly_fraction * max(0.0, kelly_full)

        stake_fraction = float(max(0.0, min(1.0, stake_fraction)))
        stake = balance * stake_fraction
        if stake <= 0.0:
            trades.append(
                TradeRow(
                    date=day,
                    cutoff_utc=cutoff_utc,
                    p_hit=p_hit,
                    model_yes=model_yes,
                    max_sofar_iem=max_sofar,
                    mos_x_mean=mos_x_mean,
                    coverage_frac=coverage_frac,
                    truth_tmax_f=float(truth_tmax),
                    d_truth=d_truth,
                    bridge_scope=bridge_scope,
                    bucket_label=bucket_label,
                    bucket_p_win=float(p_bucket),
                    bucket_price_at_cutoff=float(yes_price_at_cutoff),
                    threshold_price=float(threshold_yes_price),
                    threshold_cmp=threshold_cmp,
                    trade_side=side,
                    trade_p_win=float(trade_p_win),
                    entry_time=entry_time,
                    entry_yes_price=float(entry_yes_price) if entry_yes_price is not None else None,
                    entry_price=float(entry_price),
                    shares=None,
                    win=None,
                    ev_at_entry=None,
                    pnl=0.0,
                    balance_after=balance,
                    note="stake_zero",
                    stake_fraction=stake_fraction,
                    stake=float(stake),
                )
            )
            continue

        shares = stake / (entry_price / 100.0)
        if win:
            pnl = shares * (1.0 - entry_price / 100.0)
        else:
            pnl = -stake
        balance += pnl

        ev_at_entry = float(trade_p_win) - float(entry_price) / 100.0
        note = cutoff_note if cutoff_note else "trade"

        trades.append(
            TradeRow(
                date=day,
                cutoff_utc=cutoff_utc,
                p_hit=p_hit,
                model_yes=model_yes,
                max_sofar_iem=max_sofar,
                mos_x_mean=mos_x_mean,
                coverage_frac=coverage_frac,
                truth_tmax_f=float(truth_tmax),
                d_truth=d_truth,
                bridge_scope=bridge_scope,
                bucket_label=bucket_label,
                bucket_p_win=float(p_bucket),
                bucket_price_at_cutoff=float(yes_price_at_cutoff),
                threshold_price=float(threshold_yes_price),
                threshold_cmp=threshold_cmp,
                trade_side=side,
                trade_p_win=float(trade_p_win),
                entry_time=entry_time,
                entry_yes_price=float(entry_yes_price) if entry_yes_price is not None else None,
                entry_price=float(entry_price),
                shares=float(shares),
                win=bool(win),
                ev_at_entry=ev_at_entry,
                pnl=float(pnl),
                balance_after=float(balance),
                note=note,
                stake_fraction=stake_fraction,
                stake=float(stake),
            )
        )

    # Summarize + write outputs.
    rows_out: List[Dict] = []
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    entered = 0
    entered_yes = 0
    entered_no = 0
    model_yes_days = 0
    model_no_days = 0

    for t in trades:
        if t.model_yes:
            model_yes_days += 1
        else:
            model_no_days += 1
        if t.entry_time is not None:
            entered += 1
            if t.trade_side == "YES":
                entered_yes += 1
            elif t.trade_side == "NO":
                entered_no += 1
        if t.win is True:
            wins += 1
        elif t.win is False:
            losses += 1
        if t.pnl > 0:
            gross_profit += t.pnl
        elif t.pnl < 0:
            gross_loss += -t.pnl

        rows_out.append(
            {
                "date": t.date.isoformat(),
                "cutoff_utc": t.cutoff_utc.isoformat(),
                "p_hit": round(float(t.p_hit), 6),
                "model_yes": t.model_yes,
                "max_sofar_iem": t.max_sofar_iem,
                "mos_x_mean": t.mos_x_mean,
                "coverage_frac": t.coverage_frac,
                "truth_tmax_f": t.truth_tmax_f,
                "d_truth": t.d_truth,
                "bridge_scope": t.bridge_scope,
                "bucket_label": t.bucket_label,
                "bucket_p_win": t.bucket_p_win,
                "bucket_price_at_cutoff": t.bucket_price_at_cutoff,
                "threshold_price": t.threshold_price,
                "threshold_cmp": t.threshold_cmp,
                "trade_side": t.trade_side,
                "trade_p_win": t.trade_p_win,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "entry_yes_price": t.entry_yes_price,
                "entry_price": t.entry_price,
                "shares": t.shares,
                "win": t.win,
                "ev_at_entry": t.ev_at_entry,
                "pnl": t.pnl,
                "balance_after": t.balance_after,
                "note": t.note,
                "stake_fraction": t.stake_fraction,
                "stake": t.stake,
            }
        )

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    trades_df = pd.DataFrame(rows_out)
    daily_equity = trades_df.copy()
    daily_equity["date"] = pd.to_datetime(daily_equity["date"])
    daily_equity["balance_after"] = pd.to_numeric(daily_equity["balance_after"], errors="coerce")
    daily_equity = daily_equity.groupby("date", as_index=True)["balance_after"].last()
    idx = pd.date_range(pd.to_datetime(backtest_start), pd.to_datetime(backtest_end), freq="D")
    daily_equity = daily_equity.reindex(idx).ffill().fillna(float(START_BALANCE))
    max_drawdown_pct = _max_drawdown_pct(daily_equity)

    summary = {
        "station_id": STATION_ID,
        "model": "B6_EXP20_GAM_RESIDUAL",
        "backtest_start": backtest_start.isoformat(),
        "backtest_end": backtest_end.isoformat(),
        "bridge_mode": args.bridge_mode,
        "bridge_train_start": bridge_train_start.isoformat(),
        "bridge_train_end": bridge_train_end.isoformat(),
        "bridge_trailing_years": trailing_years,
        "p_hit_gate": p_hit_gate,
        "edge_prob": edge_prob,
        "min_win_prob": min_win_prob,
        "risk_model": risk_model,
        "fixed_risk_fraction": fixed_risk_fraction,
        "kelly_fraction": kelly_fraction,
        "start_balance": START_BALANCE,
        "end_balance": balance,
        "max_drawdown_pct": max_drawdown_pct,
        "total_days_with_preds_and_features": len(trades),
        "model_yes_days": model_yes_days,
        "model_no_days": model_no_days,
        "entered_trades": entered,
        "entered_trades_yes": entered_yes,
        "entered_trades_no": entered_no,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "missing_kalshi_files": missing_kalshi_files,
        "missing_truth_days": missing_truth_days,
        "bridge_meta": meta.__dict__,
    }

    trades_path = out_dir / "trades.csv"
    trades_df = pd.DataFrame(rows_out)
    trades_df.to_csv(trades_path, index=False, encoding="utf-8")

    # Human-readable table: only the rows where we actually entered a trade.
    entered_df = trades_df[trades_df["entry_time"].notna()].copy()
    if not entered_df.empty:
        keep_cols = [
            "date",
            "cutoff_utc",
            "p_hit",
            "model_yes",
            "trade_side",
            "bucket_label",
            "bucket_p_win",
            "trade_p_win",
            "bucket_price_at_cutoff",
            "threshold_cmp",
            "threshold_price",
            "entry_time",
            "entry_yes_price",
            "entry_price",
            "stake_fraction",
            "stake",
            "win",
            "ev_at_entry",
            "pnl",
            "balance_after",
        ]
        keep_cols = [c for c in keep_cols if c in entered_df.columns]
        entered_df = entered_df[keep_cols]
        entered_df.to_csv(out_dir / "trades_entered.csv", index=False, encoding="utf-8")
        try:
            md = entered_df.to_markdown(index=False)
        except Exception:
            md = df_to_markdown_simple(entered_df)
        (out_dir / "trades_entered.md").write_text(md, encoding="utf-8")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("Wrote", trades_path)
    print("Wrote", summary_path)


if __name__ == "__main__":
    main()
