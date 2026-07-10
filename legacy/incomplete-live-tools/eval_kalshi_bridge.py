from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pymysql
import requests

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


REPO = Path(__file__).resolve().parents[2]

STATION_ID = "KMIA"

MODEL_NAME = "B6_EXP20_GAM_RESIDUAL"
FEATURE_STORE_PATH = REPO / "cache" / "hit1830_v6_features.parquet"
BUNDLE_ROOT = REPO / "artifacts" / "model_bundles" / "hit1830_v6" / MODEL_NAME

KALSHI_DIR = REPO / "data" / "kalshi_backtest_data"
KALSHI_DOWNLOADER_SCRIPT = (
    REPO / "apps" / "ingestion-service" / "scripts" / "kalshi_download_kxhighmia_minute.py"
)
DEFAULT_MINUTE_DIR = REPO / "data" / "iem_minute_data" / "MIA" / "tmpf" / "UTC" / "yearly"

DEFAULT_BRIDGE_MODE = "trailing"  # expanding|trailing
DEFAULT_TRAILING_YEARS = 10
DEFAULT_REQUIRE_P_HIT_GE = 0.5

DEFAULT_EDGE_PROB = 0.15
DEFAULT_MIN_WIN_PROB = 0.65

# Bridge context binning (simple + robust)
MIN_SAMPLES_BIN = 200

UTC = timezone.utc

STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm") if ZoneInfo else None
MIAMI_TZ = ZoneInfo("America/New_York") if ZoneInfo else None

MOS_MODELS = ["GFS", "NAM"]
MOS_VARIABLES = [
    "n_x",
    "tmp",
    "dpt",
    "wdr",
    "wsp",
    "p06",
    "p12",
    "q06",
    "q12",
    "t06",
    "t06_1",
    "t06_2",
    "t12",
    "t12_1",
    "t12_2",
    "cig",
    "vis",
    "pos",
    "poz",
]


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


@dataclass(frozen=True)
class BetaCalibratorParams:
    epsilon: float
    coef_log_p: float
    coef_log1m_p: float
    intercept: float


@dataclass(frozen=True)
class POnshoreParams:
    feature_cols: List[str]
    imputer_medians: Dict[str, float]
    coef: List[float]
    intercept: float


@dataclass(frozen=True)
class Bundle:
    bundle_dir: Path
    feature_list: List[str]
    imputer_medians: Dict[str, float]
    beta: BetaCalibratorParams
    p_onshore: POnshoreParams
    cp_penalty_value: float
    base_model: lgb.Booster
    residual_model: lgb.Booster


@dataclass(frozen=True)
class IemMosEntry:
    runtime_utc: datetime
    forecast_time_utc: Optional[datetime]
    values: Dict[str, Optional[float]]


def utc_now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_bundle_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.name)
    for p in reversed(candidates):
        if (p / "bundle_meta.json").exists() and (p / "features.json").exists():
            return p
    return None


def _clip_probs(p: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = _clip_probs(np.asarray(p, dtype=float), eps)
    return np.log(p / (1.0 - p))


def _beta_apply(params: BetaCalibratorParams, p: np.ndarray) -> np.ndarray:
    p = _clip_probs(np.asarray(p, dtype=float), params.epsilon)
    X = np.column_stack([np.log(p), np.log1p(-p)])
    z = X @ np.array([params.coef_log_p, params.coef_log1m_p], dtype=float) + params.intercept
    return _sigmoid(z)


def load_bundle(bundle_dir: Path) -> Bundle:
    features = json.loads((bundle_dir / "features.json").read_text(encoding="utf-8"))
    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
        raise RuntimeError("Invalid features.json in bundle.")

    imputer_medians_raw = _load_json(bundle_dir / "imputer_medians.json")
    imputer_medians: Dict[str, float] = {str(k): float(v) for k, v in imputer_medians_raw.items()}

    beta_raw = _load_json(bundle_dir / "beta_calibrator.json")
    beta = BetaCalibratorParams(
        epsilon=float(beta_raw["epsilon"]),
        coef_log_p=float(beta_raw["coef_log_p"]),
        coef_log1m_p=float(beta_raw["coef_log1m_p"]),
        intercept=float(beta_raw["intercept"]),
    )

    p_onshore_raw = _load_json(bundle_dir / "p_onshore_lr.json")
    p_onshore = POnshoreParams(
        feature_cols=[str(x) for x in p_onshore_raw.get("feature_cols", [])],
        imputer_medians={str(k): float(v) for k, v in dict(p_onshore_raw.get("imputer_medians", {})).items()},
        coef=[float(x) for x in p_onshore_raw.get("coef", [])],
        intercept=float(p_onshore_raw.get("intercept", float("nan"))),
    )

    meta_raw = _load_json(bundle_dir / "bundle_meta.json")
    cp_penalty_value = float(meta_raw.get("cp_penalty_value", float("nan")))
    if not math.isfinite(cp_penalty_value):
        raise RuntimeError("bundle_meta.json missing/invalid cp_penalty_value.")

    base_model = lgb.Booster(model_file=str(bundle_dir / "base_model.txt"))
    residual_model = lgb.Booster(model_file=str(bundle_dir / "residual_model.txt"))

    return Bundle(
        bundle_dir=bundle_dir,
        feature_list=[str(x) for x in features],
        imputer_medians=imputer_medians,
        beta=beta,
        p_onshore=p_onshore,
        cp_penalty_value=cp_penalty_value,
        base_model=base_model,
        residual_model=residual_model,
    )


def _compute_tmp_bias_features(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    # Mirrors tools/early_maxout_strategy/export_b6_exp20_bundle.py::_compute_tmp_bias_features
    df = df.copy()
    df = df.sort_values("target_date_local")
    for model in ["gfs", "nam"]:
        col = f"mos_tmax_{model}"
        if col not in df.columns:
            continue
        err = (pd.to_numeric(df[col], errors="coerce").astype(float) - pd.to_numeric(df["tmax_full"], errors="coerce").astype(float)).shift(1)
        bias = err.ewm(alpha=alpha, adjust=False).mean()
        df[f"mos_tmax_bias_{model}_a{alpha:.3f}"] = bias
        df[f"mos_tmax_bc_{model}_a{alpha:.3f}"] = pd.to_numeric(df[col], errors="coerce").astype(float) - bias

    gfs_bc = df.get(f"mos_tmax_bc_gfs_a{alpha:.3f}")
    nam_bc = df.get(f"mos_tmax_bc_nam_a{alpha:.3f}")
    if gfs_bc is not None and nam_bc is not None:
        df[f"mos_tmax_mean_bc_a{alpha:.3f}"] = pd.concat([gfs_bc, nam_bc], axis=1).mean(axis=1)
    return df


def _apply_p_onshore(df: pd.DataFrame, params: POnshoreParams) -> pd.Series:
    cols = list(params.feature_cols)
    if not cols:
        return pd.Series(np.nan, index=df.index)

    X = df.reindex(columns=cols).copy()
    for c in cols:
        if c not in X.columns:
            X[c] = np.nan
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        if c in params.imputer_medians:
            X[c] = X[c].fillna(float(params.imputer_medians[c]))

    w = np.array(params.coef, dtype=float).reshape(-1)
    z = X.to_numpy(dtype=float) @ w + float(params.intercept)
    return pd.Series(1.0 / (1.0 + np.exp(-z)), index=df.index)


def _make_feature_vector(bundle: Bundle, row: pd.Series) -> np.ndarray:
    x: List[float] = []
    for c in bundle.feature_list:
        v = row.get(c)
        try:
            fv = float(v)
        except Exception:
            fv = float("nan")
        if math.isfinite(fv):
            x.append(fv)
            continue
        med = bundle.imputer_medians.get(c)
        if med is None or not math.isfinite(float(med)):
            x.append(float("nan"))
        else:
            x.append(float(med))
    return np.asarray(x, dtype=float).reshape(1, -1)


def predict_p_hit(bundle: Bundle, feature_store: pd.DataFrame, eval_date: date) -> Tuple[float, Dict[str, object]]:
    """
    Computes p_hit for eval_date using the exported bundle.

    Leakage note: tmp-bias features use shift(1), so the eval_date row does not depend on eval_date truth.
    """
    df = feature_store.copy()
    df = df[df["target_date_local"] <= eval_date].copy()
    if df.empty:
        raise RuntimeError("No feature rows available <= eval_date.")

    # p_onshore overwrite (matches training/export)
    df["p_onshore"] = _apply_p_onshore(df, bundle.p_onshore)

    # cp_exists + gate cp columns (matches training/export)
    df["cp_exists"] = (pd.to_numeric(df.get("cp_improvement"), errors="coerce").astype(float) > bundle.cp_penalty_value).astype(float)
    for col in ["cp_time_since", "cp_drop_magnitude", "cp_slope_before_v6", "cp_slope_after_v6"]:
        if col in df.columns:
            df.loc[df["cp_exists"] < 0.5, col] = np.nan

    # tmp bias features (matches training/export)
    df = _compute_tmp_bias_features(df, 0.02)
    df = _compute_tmp_bias_features(df, 0.05)

    row = df[df["target_date_local"] == eval_date]
    if row.empty:
        raise RuntimeError(f"Missing eval_date row in feature store: {eval_date.isoformat()}")
    r0 = row.iloc[0]

    X = _make_feature_vector(bundle, r0)
    base_p = np.asarray(bundle.base_model.predict(X), dtype=float)
    base_raw = _logit(base_p)
    resid_raw = np.asarray(bundle.residual_model.predict(X, raw_score=True), dtype=float)
    p_raw = _sigmoid(base_raw + resid_raw)
    p_cal = _beta_apply(bundle.beta, p_raw)

    debug = {
        "p_raw": float(p_raw.reshape(-1)[0]),
        "p_cal": float(p_cal.reshape(-1)[0]),
        "cp_penalty_value": float(bundle.cp_penalty_value),
        "cp_improvement": float(r0.get("cp_improvement")) if r0.get("cp_improvement") is not None else None,
        "cp_exists": float(r0.get("cp_exists")) if r0.get("cp_exists") is not None else None,
    }
    return float(p_cal.reshape(-1)[0]), debug


def compute_cutoff_utc(day: date) -> datetime:
    if STOCKHOLM_TZ is None:
        raise RuntimeError("ZoneInfo not available; cannot compute cutoff.")
    cutoff_local = datetime(day.year, day.month, day.day, 18, 30, tzinfo=STOCKHOLM_TZ)
    return cutoff_local.astimezone(ZoneInfo("UTC"))


def _day_window_utc(day: date) -> Tuple[datetime, datetime]:
    if MIAMI_TZ is None:
        raise RuntimeError("ZoneInfo not available; cannot compute day window.")
    day_start_local = datetime(day.year, day.month, day.day, 0, 0, tzinfo=MIAMI_TZ)
    day_end_local = day_start_local + timedelta(days=1)
    return day_start_local.astimezone(UTC), day_end_local.astimezone(UTC)


def _ols_slope(minutes: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if int(mask.sum()) < 3:
        return float("nan")
    x = minutes[mask] / 60.0
    y = values[mask].astype(float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(np.sum((x - x_mean) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _mad(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan")
    med = float(np.median(vals))
    return float(np.median(np.abs(vals - med)))


def _gap_stats(is_present: np.ndarray, step_minutes: int = 5) -> Tuple[float, int, int]:
    if is_present.size == 0:
        return float("nan"), 0, 0
    max_gap = 0
    run = 0
    gap_cnt_15 = 0
    gap_cnt_30 = 0
    for ok in is_present:
        if not bool(ok):
            run += 1
        else:
            if run * step_minutes >= 15:
                gap_cnt_15 += 1
            if run * step_minutes >= 30:
                gap_cnt_30 += 1
            max_gap = max(max_gap, run)
            run = 0
    if run > 0:
        if run * step_minutes >= 15:
            gap_cnt_15 += 1
        if run * step_minutes >= 30:
            gap_cnt_30 += 1
        max_gap = max(max_gap, run)
    return float(max_gap * step_minutes), int(gap_cnt_15), int(gap_cnt_30)


def _longest_run(mask: np.ndarray) -> int:
    best = 0
    run = 0
    for val in mask:
        if bool(val):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def _read_minute_csv(path: Path) -> pd.DataFrame:
    usecols = ["valid(UTC)", "tmpf"]
    if path.exists():
        head = path.open("r", encoding="utf-8").readline()
        if "station" in head:
            usecols = ["station", "valid(UTC)", "tmpf"]

    df = pd.read_csv(path, usecols=usecols, dtype={"tmpf": "string"})
    if "station" in df.columns:
        df = df[df["station"].astype(str).str.upper().isin({STATION_ID.replace("K", ""), STATION_ID})]
    df["ts_utc"] = pd.to_datetime(df["valid(UTC)"], utc=True, errors="coerce")
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    df = df.dropna(subset=["ts_utc"]).copy()
    return df[["ts_utc", "tmpf"]]


def _load_minute_local(minute_dir: Path, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    years = sorted({start_utc.year, end_utc.year})
    frames: List[pd.DataFrame] = []
    for year in years:
        path = minute_dir / f"MIA_tmpf_1min_UTC_{year}.csv"
        if not path.exists():
            continue
        df = _read_minute_csv(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame({"ts_utc": pd.to_datetime([], utc=True), "tmpf": pd.Series([], dtype=float)})

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.dropna(subset=["ts_utc"]).copy()
    df_all["ts_utc"] = pd.to_datetime(df_all["ts_utc"], utc=True, errors="coerce")
    df_all["tmpf"] = pd.to_numeric(df_all["tmpf"], errors="coerce")
    df_all = df_all.dropna(subset=["ts_utc"]).copy()
    df_all = df_all[(df_all["ts_utc"] >= start_utc) & (df_all["ts_utc"] <= end_utc)].copy()
    df_all = df_all.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last")
    return df_all.reset_index(drop=True)


def _fetch_iem_minute(station: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"
    start_utc = start_utc.astimezone(UTC)
    end_utc = end_utc.astimezone(UTC)
    params = {
        "station": station.replace("K", ""),
        "vars": "tmpf",
        "sts": start_utc.strftime("%Y-%m-%dT%H:%MZ"),
        "ets": end_utc.strftime("%Y-%m-%dT%H:%MZ"),
        "what": "download",
        "tz": "UTC",
        "delim": "comma",
    }
    resp = requests.get(base_url, params=params, timeout=60)
    resp.raise_for_status()
    from io import StringIO

    df = pd.read_csv(StringIO(resp.text))
    if "valid(UTC)" not in df.columns or "tmpf" not in df.columns:
        raise ValueError("IEM response missing required columns.")
    df["ts_utc"] = pd.to_datetime(df["valid(UTC)"], utc=True, errors="coerce")
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    df = df.dropna(subset=["ts_utc"]).copy()
    df = df.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last")
    return df[["ts_utc", "tmpf"]].reset_index(drop=True)


def _read_local_minute_manifest_row(*, minute_dir: Path, station_id: str, year: int) -> Optional[Dict[str, str]]:
    """
    Returns the manifest CSV row for a given year if present, else None.

    This is best-effort diagnostics for why on-demand minute loads are empty.
    """
    station = station_id.replace("K", "")
    manifest_path = (minute_dir / ".." / "meta" / f"manifest_{station}_tmpf_1min_UTC_2002_2026.csv").resolve()
    if not manifest_path.exists():
        return None

    import csv

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if int(str(row.get("year", "")).strip()) == int(year):
                        return {str(k): str(v) for k, v in row.items()}
                except Exception:
                    continue
    except Exception:
        return None
    return None


def _load_minute_obs(
    *,
    minute_dir: Path,
    station_id: str,
    start_utc: datetime,
    end_utc: datetime,
    now_utc: datetime,
    minute_source: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns (df_1m, meta).

    Leakage guard: for current/future days, refuses to fetch beyond now_utc.
    """
    if end_utc > now_utc + timedelta(seconds=5):
        raise ValueError(f"Minute request exceeds current time: requested_end={end_utc.isoformat()} now={now_utc.isoformat()}")

    minute_dir = Path(minute_dir).resolve()
    meta: Dict[str, Any] = {
        "minute_source": minute_source,
        "minute_dir": str(minute_dir),
        "start_utc": start_utc.astimezone(UTC).isoformat(),
        "end_utc": end_utc.astimezone(UTC).isoformat(),
    }

    local_df = pd.DataFrame()
    if minute_source in ("auto", "local"):
        local_df = _load_minute_local(minute_dir, start_utc, end_utc)
        meta["local_rows"] = int(len(local_df))
        if minute_source == "local":
            return local_df, meta

    iem_df = pd.DataFrame()
    if minute_source in ("auto", "iem"):
        iem_df = _fetch_iem_minute(station_id, start_utc, end_utc)
        meta["iem_rows"] = int(len(iem_df))
        if minute_source == "iem":
            return iem_df, meta

    # auto: if local has near-complete coverage and reaches the end, prefer it (no network).
    expected_minutes = int(((end_utc - start_utc).total_seconds() / 60.0) + 1)
    meta["expected_minutes"] = int(expected_minutes)
    if not local_df.empty:
        coverage = float(len(local_df) / expected_minutes) if expected_minutes > 0 else float("nan")
        meta["local_coverage_frac"] = coverage
        local_max = pd.to_datetime(local_df["ts_utc"], utc=True, errors="coerce").max()
        meta["local_max_ts_utc"] = local_max.isoformat() if pd.notna(local_max) else None
        if pd.notna(local_max) and local_max.to_pydatetime() >= (end_utc - timedelta(minutes=2)) and coverage >= 0.95:
            return local_df, meta

    # Merge: prefer IEM timestamps when present; otherwise keep local.
    if local_df.empty:
        return iem_df, meta
    if iem_df.empty:
        return local_df, meta
    combined = pd.concat([local_df, iem_df], ignore_index=True)
    combined["ts_utc"] = pd.to_datetime(combined["ts_utc"], utc=True, errors="coerce")
    combined["tmpf"] = pd.to_numeric(combined["tmpf"], errors="coerce")
    combined = combined.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    combined = combined.drop_duplicates(subset=["ts_utc"], keep="last")
    combined = combined[(combined["ts_utc"] >= start_utc) & (combined["ts_utc"] <= end_utc)].copy()
    combined = combined.reset_index(drop=True)
    meta["merged_rows"] = int(len(combined))
    return combined, meta


def _circular_mean_deg(values: List[float]) -> float:
    vals = np.array([float(v) for v in values], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    rads = np.deg2rad(vals)
    x = float(np.mean(np.cos(rads)))
    y = float(np.mean(np.sin(rads)))
    deg = math.degrees(math.atan2(y, x))
    if deg < 0.0:
        deg += 360.0
    return float(deg)


def _angular_diff_deg(a: float, b: float) -> float:
    if not math.isfinite(float(a)) or not math.isfinite(float(b)):
        return float("nan")
    diff = abs(float(a) - float(b)) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return float(diff)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_epoch_millis(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                dt = datetime.fromisoformat(text)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return int(dt.timestamp() * 1000.0)
            except Exception:
                return None
    return None


def _parse_mos_numeric(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.upper() in {"M", "T"}:
        return None
    if "/" in text:
        text = text.split("/", 1)[0].strip()
        if not text or text.upper() in {"M", "T"}:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_mos_model(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text == "ETA":
        return "NAM"
    if text == "AVN":
        return "GFS"
    return text


def fetch_station_zoneid(station_id: str) -> str:
    sql = """
        SELECT zone_id
        FROM station_registry
        WHERE station_id=%s
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (station_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        raise ValueError(f"Station not found in station_registry: {station_id}")
    return str(row[0])


def compute_asof_utc(target_date: date, station_zoneid: str) -> datetime:
    if ZoneInfo is None:
        raise RuntimeError("ZoneInfo not available; cannot compute as-of time.")
    asof_local_time = datetime(target_date.year, target_date.month, target_date.day, 12, 0, tzinfo=UTC) - timedelta(days=1)
    asof_utc = asof_local_time.astimezone(UTC)
    return asof_utc


def fetch_iem_mos_payload(
    station_id: str,
    model: str,
    start_utc: datetime,
    end_utc: datetime,
) -> Tuple[List[IemMosEntry], str, str]:
    base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
    start_utc = start_utc.astimezone(UTC)
    end_utc = end_utc.astimezone(UTC)
    params = {
        "station": station_id,
        "model": model,
        "sts": start_utc.strftime("%Y-%m-%dT%H:%MZ"),
        "ets": end_utc.strftime("%Y-%m-%dT%H:%MZ"),
        "format": "json",
    }
    resp = requests.get(base_url, params=params, timeout=60)
    resp.raise_for_status()
    raw_bytes = resp.content
    raw_hash = _sha256_hex(raw_bytes)
    raw_text = raw_bytes.decode("utf-8", errors="replace")
    payload = json.loads(raw_text)
    if not isinstance(payload, list):
        raise ValueError("IEM MOS payload must be a JSON array.")

    normalized_station = station_id.strip().upper()
    alt_station = normalized_station[1:] if normalized_station.startswith("K") else normalized_station
    entries: List[IemMosEntry] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        station = str(item.get("station", "")).strip().upper()
        if station not in {normalized_station, alt_station}:
            continue
        model_norm = _normalize_mos_model(item.get("model"))
        if model_norm is None or model_norm != model:
            continue
        runtime_ms = _parse_epoch_millis(item.get("runtime"))
        if runtime_ms is None:
            continue
        runtime_utc = datetime.fromtimestamp(runtime_ms / 1000.0, tz=UTC)
        ftime_ms = _parse_epoch_millis(item.get("ftime"))
        forecast_utc = datetime.fromtimestamp(ftime_ms / 1000.0, tz=UTC) if ftime_ms is not None else None

        values: Dict[str, Optional[float]] = {}
        for key, val in item.items():
            if key in {"station", "model", "runtime", "ftime"}:
                continue
            norm_key = str(key).strip().lower()
            if not norm_key:
                continue
            values[norm_key] = _parse_mos_numeric(val)
        entries.append(IemMosEntry(runtime_utc=runtime_utc, forecast_time_utc=forecast_utc, values=values))

    return entries, raw_text, raw_hash


def select_mos_runtime_for_asof(
    entries: List[IemMosEntry],
    asof_utc: datetime,
    window_start_utc: datetime,
    window_end_utc: datetime,
    station_zone: ZoneInfo,
    target_date: date,
) -> Optional[datetime]:
    covered: List[datetime] = []
    for entry in entries:
        if entry.runtime_utc > asof_utc:
            continue
        forecast_time = entry.forecast_time_utc
        if forecast_time is None:
            continue
        if forecast_time < window_start_utc or forecast_time >= window_end_utc:
            continue
        if forecast_time.astimezone(station_zone).date() != target_date:
            continue
        covered.append(entry.runtime_utc)
    if not covered:
        return None
    return max(covered)


def build_mos_daily_rows_for_runtime(
    entries: List[IemMosEntry],
    *,
    station_id: str,
    station_zoneid: str,
    model: str,
    runtime_utc: datetime,
    target_date: date,
    window_start_utc: datetime,
    window_end_utc: datetime,
    asof_utc: datetime,
    retrieved_at_utc: datetime,
    raw_hash: str,
) -> List[Dict[str, Any]]:
    if ZoneInfo is None:
        raise RuntimeError("ZoneInfo not available; cannot compute station local date.")
    station_zone = ZoneInfo(station_zoneid)
    summaries: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if entry.runtime_utc != runtime_utc:
            continue
        forecast_time = entry.forecast_time_utc
        if forecast_time is None:
            continue
        if forecast_time < window_start_utc or forecast_time >= window_end_utc:
            continue
        if forecast_time.astimezone(station_zone).date() != target_date:
            continue
        for var, val in entry.values.items():
            if val is None:
                continue
            if var not in MOS_VARIABLES:
                continue
            stats = summaries.get(var)
            if stats is None:
                stats = {
                    "values": [],
                    "min": float("inf"),
                    "max": float("-inf"),
                    "sum": 0.0,
                    "count": 0,
                    "first": forecast_time,
                    "last": forecast_time,
                }
                summaries[var] = stats
            stats["values"].append(float(val))
            stats["min"] = min(stats["min"], float(val))
            stats["max"] = max(stats["max"], float(val))
            stats["sum"] = float(stats["sum"]) + float(val)
            stats["count"] = int(stats["count"]) + 1
            if forecast_time < stats["first"]:
                stats["first"] = forecast_time
            if forecast_time > stats["last"]:
                stats["last"] = forecast_time

    rows: List[Dict[str, Any]] = []
    for var, stats in summaries.items():
        if stats["count"] <= 0:
            continue
        mean = float(stats["sum"] / stats["count"])
        median = float(np.median(np.asarray(stats["values"], dtype=float)))
        rows.append(
            {
                "station_id": station_id,
                "station_zoneid": station_zoneid,
                "model": model,
                "asof_utc": asof_utc,
                "runtime_utc": runtime_utc,
                "target_date_local": target_date,
                "variable_code": var,
                "value_min": round(float(stats["min"]), 4),
                "value_max": round(float(stats["max"]), 4),
                "value_mean": round(mean, 4),
                "value_median": round(median, 4),
                "sample_count": int(stats["count"]),
                "first_forecast_time_utc": stats["first"],
                "last_forecast_time_utc": stats["last"],
                "raw_payload_hash_ref": raw_hash,
                "retrieved_at_utc": retrieved_at_utc,
            }
        )
    return rows


def _dt_for_db(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def upsert_mos_daily_rows(rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO mos_daily_value (
          station_id,
          station_zoneid,
          model,
          asof_utc,
          runtime_utc,
          target_date_local,
          variable_code,
          value_min,
          value_max,
          value_mean,
          value_median,
          sample_count,
          first_forecast_time_utc,
          last_forecast_time_utc,
          raw_payload_hash_ref,
          retrieved_at_utc
        ) VALUES (
          %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
          station_zoneid = VALUES(station_zoneid),
          asof_utc = VALUES(asof_utc),
          value_min = VALUES(value_min),
          value_max = VALUES(value_max),
          value_mean = VALUES(value_mean),
          value_median = VALUES(value_median),
          sample_count = VALUES(sample_count),
          first_forecast_time_utc = VALUES(first_forecast_time_utc),
          last_forecast_time_utc = VALUES(last_forecast_time_utc),
          raw_payload_hash_ref = VALUES(raw_payload_hash_ref),
          retrieved_at_utc = VALUES(retrieved_at_utc)
    """
    values = []
    for r in rows:
        values.append(
            (
                r["station_id"],
                r["station_zoneid"],
                r["model"],
                _dt_for_db(r["asof_utc"]),
                _dt_for_db(r["runtime_utc"]),
                r["target_date_local"],
                r["variable_code"],
                r["value_min"],
                r["value_max"],
                r["value_mean"],
                r["value_median"],
                r["sample_count"],
                _dt_for_db(r["first_forecast_time_utc"]),
                _dt_for_db(r["last_forecast_time_utc"]),
                r["raw_payload_hash_ref"],
                _dt_for_db(r["retrieved_at_utc"]),
            )
        )
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, values)
    finally:
        conn.close()
    return len(values)


def ensure_mos_daily_value_for_day(
    *,
    station_id: str,
    target_date: date,
    cutoff_utc: datetime,
    out_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    mos_raw = fetch_mos_raw_for_day(station_id, target_date)
    mos_latest = select_latest_mos_for_cutoff(mos_raw, cutoff_utc)

    expected = {(m.upper(), v.lower()) for m in MOS_MODELS for v in MOS_VARIABLES}
    got = {(str(r.model).upper(), str(r.variable_code).lower()) for r in mos_latest.itertuples(index=False)}
    missing = sorted(list(expected - got))

    meta: Dict[str, Any] = {
        "station_id": station_id,
        "target_date_local": target_date.isoformat(),
        "cutoff_utc": cutoff_utc.astimezone(UTC).isoformat(),
        "missing_before": missing,
        "ingest_attempted": False,
        "ingest": [],
    }

    if not missing:
        return mos_latest, meta

    station_zoneid = fetch_station_zoneid(station_id)
    if ZoneInfo is None:
        raise RuntimeError("ZoneInfo not available; cannot compute station window.")
    station_zone = ZoneInfo(station_zoneid)
    asof_utc = compute_asof_utc(target_date, station_zoneid)
    if asof_utc > cutoff_utc:
        raise ValueError(f"Leakage guard: computed asof_utc > cutoff_utc (asof_utc={asof_utc.isoformat()} cutoff_utc={cutoff_utc.isoformat()})")

    window_start_local = datetime(target_date.year, target_date.month, target_date.day, 0, 0, tzinfo=station_zone)
    window_start_utc = window_start_local.astimezone(UTC)
    window_end_utc = (window_start_local + timedelta(days=1)).astimezone(UTC)
    request_start_utc = min(asof_utc, window_start_utc)
    retrieved_at_utc = datetime.now(UTC)

    meta.update(
        {
            "station_zoneid": station_zoneid,
            "asof_utc": asof_utc.astimezone(UTC).isoformat(),
            "window_start_utc": window_start_utc.astimezone(UTC).isoformat(),
            "window_end_utc": window_end_utc.astimezone(UTC).isoformat(),
            "request_start_utc": request_start_utc.astimezone(UTC).isoformat(),
        }
    )

    meta["ingest_attempted"] = True
    for model in MOS_MODELS:
        model_upper = model.upper()
        try:
            entries, raw_text, raw_hash = fetch_iem_mos_payload(
                station_id=station_id,
                model=model_upper,
                start_utc=request_start_utc,
                end_utc=window_end_utc,
            )
            raw_path = out_dir / f"mos_raw_{model_upper}_{target_date.strftime('%Y%m%d')}.json"
            raw_path.write_text(raw_text, encoding="utf-8")
            runtime_utc = select_mos_runtime_for_asof(entries, asof_utc, window_start_utc, window_end_utc, station_zone, target_date)
            if runtime_utc is None:
                meta["ingest"].append(
                    {
                        "model": model_upper,
                        "rows_inserted": 0,
                        "runtime_utc": None,
                        "reason": "no_runtime<=asof_with_target_coverage",
                    }
                )
                continue
            if runtime_utc > asof_utc:
                raise ValueError(f"MOS runtime_utc > asof_utc for model={model_upper}: runtime_utc={runtime_utc} asof_utc={asof_utc}")
            rows = build_mos_daily_rows_for_runtime(
                entries,
                station_id=station_id,
                station_zoneid=station_zoneid,
                model=model_upper,
                runtime_utc=runtime_utc,
                target_date=target_date,
                window_start_utc=window_start_utc,
                window_end_utc=window_end_utc,
                asof_utc=asof_utc,
                retrieved_at_utc=retrieved_at_utc,
                raw_hash=raw_hash,
            )
            inserted = upsert_mos_daily_rows(rows)
            meta["ingest"].append(
                {
                    "model": model_upper,
                    "rows_inserted": inserted,
                    "runtime_utc": runtime_utc.astimezone(UTC).isoformat(),
                    "raw_payload_hash": raw_hash,
                }
            )
        except Exception as exc:
            meta["ingest"].append(
                {
                    "model": model_upper,
                    "rows_inserted": 0,
                    "runtime_utc": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    mos_raw = fetch_mos_raw_for_day(station_id, target_date)
    mos_latest = select_latest_mos_for_cutoff(mos_raw, cutoff_utc)
    got = {(str(r.model).upper(), str(r.variable_code).lower()) for r in mos_latest.itertuples(index=False)}
    missing_after = sorted(list(expected - got))
    meta["missing_after"] = missing_after
    return mos_latest, meta


def fetch_mos_raw_for_day(station_id: str, target_date: date) -> pd.DataFrame:
    placeholders_models = ", ".join(["%s"] * len(MOS_MODELS))
    placeholders_vars = ", ".join(["%s"] * len(MOS_VARIABLES))
    sql = f"""
        SELECT id, station_id, model, variable_code, target_date_local, asof_utc, runtime_utc, retrieved_at_utc,
               value_mean, value_max, value_min
        FROM mos_daily_value
        WHERE station_id=%s
          AND target_date_local=%s
          AND UPPER(model) IN ({placeholders_models})
          AND LOWER(variable_code) IN ({placeholders_vars})
    """
    params: List[object] = [station_id, target_date] + [m.upper() for m in MOS_MODELS] + [v.lower() for v in MOS_VARIABLES]
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(
        rows,
        columns=[
            "id",
            "station_id",
            "model",
            "variable_code",
            "target_date_local",
            "asof_utc",
            "runtime_utc",
            "retrieved_at_utc",
            "value_mean",
            "value_max",
            "value_min",
        ],
    )
    if df.empty:
        return df
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["model"] = df["model"].astype(str).str.upper()
    df["variable_code"] = df["variable_code"].astype(str).str.lower()
    for c in ["asof_utc", "runtime_utc", "retrieved_at_utc"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    for c in ["value_mean", "value_max", "value_min"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def select_latest_mos_for_cutoff(mos_raw: pd.DataFrame, cutoff_utc: datetime) -> pd.DataFrame:
    if mos_raw.empty:
        return mos_raw
    df = mos_raw.copy()
    df = df[df["asof_utc"] <= pd.Timestamp(cutoff_utc)].copy()
    if df.empty:
        return df
    df = df.sort_values(["model", "variable_code", "asof_utc", "runtime_utc", "retrieved_at_utc", "id"])
    latest = df.groupby(["model", "variable_code"], as_index=False).tail(1)
    return latest.reset_index(drop=True)


def mos_base_features(latest: pd.DataFrame) -> pd.DataFrame:
    latest = latest.copy()
    latest["target_date_local"] = pd.to_datetime(latest["target_date_local"]).dt.date
    pv_max = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_max")
    pv_min = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_min")
    pv_mean = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_mean")

    def get_col(pv: pd.DataFrame, model: str, var: str) -> pd.Series:
        try:
            return pv[(model, var)]
        except KeyError:
            return pd.Series(index=pv.index, dtype=float)

    out = pd.DataFrame({"target_date_local": pv_max.index}).reset_index(drop=True)

    x_gfs = get_col(pv_max, "GFS", "n_x")
    x_nam = get_col(pv_max, "NAM", "n_x")
    n_gfs = get_col(pv_min, "GFS", "n_x")
    n_nam = get_col(pv_min, "NAM", "n_x")
    mos_x_mean = pd.concat([x_gfs, x_nam], axis=1).mean(axis=1)
    mos_n_mean = pd.concat([n_gfs, n_nam], axis=1).mean(axis=1)
    mos_range = mos_x_mean - mos_n_mean
    mos_x_disagree = (x_gfs - x_nam).abs()
    mos_range_disagree = (x_gfs - n_gfs - (x_nam - n_nam)).abs()

    out["mos_x_gfs"] = x_gfs.values
    out["mos_x_nam"] = x_nam.values
    out["mos_n_gfs"] = n_gfs.values
    out["mos_n_nam"] = n_nam.values
    out["mos_x_mean"] = mos_x_mean.values
    out["mos_n_mean"] = mos_n_mean.values
    out["mos_range"] = mos_range.values
    out["mos_x_disagree"] = mos_x_disagree.values
    out["mos_range_disagree"] = mos_range_disagree.values

    tmp_max_gfs = get_col(pv_max, "GFS", "tmp")
    tmp_max_nam = get_col(pv_max, "NAM", "tmp")
    tmp_min_gfs = get_col(pv_min, "GFS", "tmp")
    tmp_min_nam = get_col(pv_min, "NAM", "tmp")
    mos_tmax_mean = pd.concat([tmp_max_gfs, tmp_max_nam], axis=1).mean(axis=1)
    mos_tmin_mean = pd.concat([tmp_min_gfs, tmp_min_nam], axis=1).mean(axis=1)
    mos_tmp_range = mos_tmax_mean - mos_tmin_mean
    out["mos_tmax_gfs"] = tmp_max_gfs.values
    out["mos_tmax_nam"] = tmp_max_nam.values
    out["mos_tmin_gfs"] = tmp_min_gfs.values
    out["mos_tmin_nam"] = tmp_min_nam.values
    out["mos_tmax_mean"] = mos_tmax_mean.values
    out["mos_tmin_mean"] = mos_tmin_mean.values
    out["mos_range_mean"] = mos_tmp_range.values
    out["mos_tmax_disagree"] = (tmp_max_gfs - tmp_max_nam).abs().values
    out["mos_range_disagree_tmp"] = (tmp_max_gfs - tmp_min_gfs - (tmp_max_nam - tmp_min_nam)).abs().values

    core_vars = ["tmp", "dpt", "wsp", "wdr", "cig", "vis"]
    for var in core_vars:
        gfs_mean = get_col(pv_mean, "GFS", var)
        nam_mean = get_col(pv_mean, "NAM", var)
        gfs_max = get_col(pv_max, "GFS", var)
        nam_max = get_col(pv_max, "NAM", var)
        gfs_min = get_col(pv_min, "GFS", var)
        nam_min = get_col(pv_min, "NAM", var)

        mean_models = pd.concat([gfs_mean, nam_mean], axis=1).mean(axis=1)
        max_models = pd.concat([gfs_max, nam_max], axis=1).max(axis=1)
        min_models = pd.concat([gfs_min, nam_min], axis=1).min(axis=1)

        if var == "wdr":
            wdr_mean = pd.Series(index=mean_models.index, dtype=float)
            for idx in wdr_mean.index:
                wdr_mean.loc[idx] = _circular_mean_deg([gfs_mean.get(idx, np.nan), nam_mean.get(idx, np.nan)])
            out["mos_wdr_mean_models"] = wdr_mean.values
            out["mos_wdr_mean_disagree"] = [_angular_diff_deg(gfs_mean.get(idx, np.nan), nam_mean.get(idx, np.nan)) for idx in wdr_mean.index]
        else:
            out[f"mos_{var}_mean_models"] = mean_models.values
            out[f"mos_{var}_mean_disagree"] = (gfs_mean - nam_mean).abs().values

        out[f"mos_{var}_gfs_mean"] = gfs_mean.values
        out[f"mos_{var}_nam_mean"] = nam_mean.values
        out[f"mos_{var}_gfs_max"] = gfs_max.values
        out[f"mos_{var}_nam_max"] = nam_max.values
        out[f"mos_{var}_gfs_min"] = gfs_min.values
        out[f"mos_{var}_nam_min"] = nam_min.values
        out[f"mos_{var}_max_models"] = max_models.values
        out[f"mos_{var}_min_models"] = min_models.values

    conv_vars = ["p06", "p12", "q06", "q12", "t06", "t06_1", "t06_2", "t12", "t12_1", "t12_2", "pos", "poz"]
    for var in conv_vars:
        gfs_mean = get_col(pv_mean, "GFS", var)
        nam_mean = get_col(pv_mean, "NAM", var)
        gfs_max = get_col(pv_max, "GFS", var)
        nam_max = get_col(pv_max, "NAM", var)
        gfs_min = get_col(pv_min, "GFS", var)
        nam_min = get_col(pv_min, "NAM", var)

        mean_models = pd.concat([gfs_mean, nam_mean], axis=1).mean(axis=1)
        max_models = pd.concat([gfs_max, nam_max], axis=1).max(axis=1)
        min_models = pd.concat([gfs_min, nam_min], axis=1).min(axis=1)

        out[f"mos_{var}_gfs_mean"] = gfs_mean.values
        out[f"mos_{var}_nam_mean"] = nam_mean.values
        out[f"mos_{var}_gfs_max"] = gfs_max.values
        out[f"mos_{var}_nam_max"] = nam_max.values
        out[f"mos_{var}_gfs_min"] = gfs_min.values
        out[f"mos_{var}_nam_min"] = nam_min.values
        out[f"mos_{var}_mean_models"] = mean_models.values
        out[f"mos_{var}_max_models"] = max_models.values
        out[f"mos_{var}_min_models"] = min_models.values
        out[f"mos_{var}_mean_disagree"] = (gfs_mean - nam_mean).abs().values
        out[f"mos_{var}_max_disagree"] = (gfs_max - nam_max).abs().values

    asof_latest = latest.groupby(["target_date_local", "model"], as_index=False)["asof_utc"].max()
    asof_pivot = asof_latest.pivot(index="target_date_local", columns="model", values="asof_utc")
    out["mos_latest_asof_utc_gfs"] = asof_pivot.get("GFS").values
    out["mos_latest_asof_utc_nam"] = asof_pivot.get("NAM").values

    tmp_rows = latest[latest["variable_code"] == "tmp"]
    if not tmp_rows.empty:
        tmp_asof = tmp_rows.pivot(index="target_date_local", columns="model", values="asof_utc")
        out["mos_tmp_asof_utc_gfs"] = tmp_asof.get("GFS").values
        out["mos_tmp_asof_utc_nam"] = tmp_asof.get("NAM").values
    for col in ["mos_latest_asof_utc_gfs", "mos_latest_asof_utc_nam", "mos_tmp_asof_utc_gfs", "mos_tmp_asof_utc_nam"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")

    wdr_mean_models = out.get("mos_wdr_mean_models")
    wsp_mean_models = out.get("mos_wsp_mean_models")
    if wdr_mean_models is not None and wsp_mean_models is not None:
        wdr_rad = np.deg2rad(wdr_mean_models)
        out["mos_u_mean"] = -wsp_mean_models * np.sin(wdr_rad)
        out["mos_v_mean"] = -wsp_mean_models * np.cos(wdr_rad)

    return out


def _compute_climo_range_map(train_df: pd.DataFrame) -> Tuple[Dict[int, float], float]:
    if train_df.empty:
        return {}, float("nan")
    dt = pd.to_datetime(train_df["target_date_local"])
    doy = dt.dt.dayofyear
    range_full = pd.to_numeric(train_df["range_full"], errors="coerce")
    tmp = pd.DataFrame({"doy": doy, "range_full": range_full})
    range_climo = tmp.groupby("doy")["range_full"].median()
    overall = float(np.nanmedian(range_full.to_numpy(dtype=float)))
    return {int(k): float(v) for k, v in range_climo.items()}, float(overall)


def _compute_suppression_stats(train_df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in ["mos_cig_min_models", "mos_vis_min_models", "mos_q12_max_models"]:
        vals = pd.to_numeric(train_df.get(col), errors="coerce")
        out[f"{col}_mean"] = float(vals.mean())
        out[f"{col}_std"] = float(vals.std())
    for k in list(out.keys()):
        if k.endswith("_std") and (not math.isfinite(out[k]) or out[k] <= 0.0):
            out[k] = 1.0
    return out


def compute_minute_features_for_day(df_1m: pd.DataFrame, day_start_utc: datetime, day_end_utc: datetime, cutoff_utc: datetime) -> Dict[str, float]:
    df_1m = df_1m.copy()
    df_1m["ts_utc"] = pd.to_datetime(df_1m["ts_utc"], utc=True, errors="coerce")
    df_1m["tmpf"] = pd.to_numeric(df_1m["tmpf"], errors="coerce")
    df_1m = df_1m.dropna(subset=["ts_utc"]).copy()
    df_1m = df_1m.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last")
    df_1m = df_1m.set_index("ts_utc")

    series_5m = df_1m["tmpf"].resample("5min").median()

    partial_end = min(cutoff_utc, day_end_utc) - timedelta(minutes=5)
    partial_idx = pd.date_range(day_start_utc, partial_end, freq="5min") if partial_end >= day_start_utc else []
    partial_series = series_5m.reindex(partial_idx)
    partial_vals = partial_series.to_numpy(dtype=float)

    if partial_series.notna().sum() == 0:
        raise ValueError("No minute observations available under strict as-of for this day (all NaN).")

    tmax_sofar = float(np.nanmax(partial_vals))
    tmin_sofar = float(np.nanmin(partial_vals))
    range_sofar = tmax_sofar - tmin_sofar if np.isfinite(tmax_sofar) and np.isfinite(tmin_sofar) else float("nan")

    max_time_partial_utc = partial_series[partial_series == tmax_sofar].index.min()
    minutes_since_max = float("nan")
    if pd.notna(max_time_partial_utc):
        minutes_since_max = float((cutoff_utc - max_time_partial_utc.to_pydatetime()).total_seconds() / 60.0)

    temp_last = float(partial_series.dropna().iloc[-1]) if not partial_series.dropna().empty else float("nan")
    temp_15m = float(np.nanmedian(partial_vals[-3:])) if len(partial_vals) >= 3 else float("nan")
    temp_30m = float(np.nanmedian(partial_vals[-6:])) if len(partial_vals) >= 6 else float("nan")
    temp_60m = float(np.nanmedian(partial_vals[-12:])) if len(partial_vals) >= 12 else float("nan")
    temp_120m = float(np.nanmedian(partial_vals[-24:])) if len(partial_vals) >= 24 else float("nan")
    drop_from_max = tmax_sofar - temp_last if np.isfinite(tmax_sofar) and np.isfinite(temp_last) else float("nan")

    minute_slice = df_1m.loc[day_start_utc:cutoff_utc]
    expected_minutes = int(((cutoff_utc - day_start_utc).total_seconds() / 60.0) + 1)
    coverage_frac = float(len(minute_slice) / expected_minutes) if expected_minutes > 0 else float("nan")
    if len(minute_slice) > 0:
        last_gap_minutes = float((cutoff_utc - minute_slice.index.max().to_pydatetime()).total_seconds() / 60.0)
    else:
        last_gap_minutes = float("nan")

    w12_end = cutoff_utc - timedelta(minutes=5)
    w12_start = cutoff_utc - timedelta(hours=12)
    w12_idx = pd.date_range(w12_start, w12_end, freq="5min")
    w12 = series_5m.reindex(w12_idx).to_numpy(dtype=float)

    w6_end = cutoff_utc - timedelta(minutes=5)
    w6_start = cutoff_utc - timedelta(hours=6)
    w6_idx = pd.date_range(w6_start, w6_end, freq="5min")
    w6 = series_5m.reindex(w6_idx).to_numpy(dtype=float)

    def slope_last(points: int, values: np.ndarray) -> float:
        if len(values) < points:
            return float("nan")
        y = values[-points:]
        mins = np.arange(points) * 5
        return _ols_slope(mins, y)

    slope_15 = slope_last(3, w12)
    slope_30 = slope_last(6, w12)
    slope_60 = slope_last(12, w12)
    slope_120 = slope_last(24, w12)
    slope_180 = slope_last(36, w12)
    curvature_30_180 = slope_30 - slope_180 if np.isfinite(slope_30) and np.isfinite(slope_180) else float("nan")
    curvature_60_180 = slope_60 - slope_180 if np.isfinite(slope_60) and np.isfinite(slope_180) else float("nan")

    # Change-point detection on W6 (single break) for cp_exists features
    cp_improvement = float("nan")
    cp_time_since = float("nan")
    cp_drop_magnitude = float("nan")
    cp_slope_before = float("nan")
    cp_slope_after = float("nan")
    w6_interp = pd.Series(w6).interpolate(limit_direction="both").to_numpy()
    if len(w6_interp) >= 10:
        mins = np.arange(len(w6_interp)) * 5
        base_slope = _ols_slope(mins, w6_interp)
        base_intercept = float(np.nanmean(w6_interp)) - base_slope * float(np.nanmean(mins / 60.0))
        base_pred = base_slope * (mins / 60.0) + base_intercept
        base_sse = float(np.nansum((w6_interp - base_pred) ** 2))
        best_sse = float("inf")
        best_idx = None
        best_before = float("nan")
        best_after = float("nan")
        best_drop = float("nan")
        for s in range(8, len(w6_interp) - 2):
            left = w6_interp[: s + 1]
            right = w6_interp[s:]
            mins_left = np.arange(len(left)) * 5
            mins_right = np.arange(len(right)) * 5
            slope_left = _ols_slope(mins_left, left)
            slope_right = _ols_slope(mins_right, right)
            if not np.isfinite(slope_left) or not np.isfinite(slope_right):
                continue
            intercept_left = float(np.nanmean(left)) - slope_left * float(np.nanmean(mins_left / 60.0))
            intercept_right = float(np.nanmean(right)) - slope_right * float(np.nanmean(mins_right / 60.0))
            pred_left = slope_left * (mins_left / 60.0) + intercept_left
            pred_right = slope_right * (mins_right / 60.0) + intercept_right
            sse = float(np.nansum((left - pred_left) ** 2) + np.nansum((right - pred_right) ** 2))
            if sse < best_sse:
                best_sse = sse
                best_idx = s
                best_before = slope_left
                best_after = slope_right
                best_drop = float(pred_left[-1] - pred_right[0])
        if best_idx is not None and np.isfinite(base_sse):
            cp_improvement = base_sse - best_sse
            cp_time_since = float((len(w6_interp) - 1 - int(best_idx)) * 5)
            cp_drop_magnitude = float(best_drop)
            cp_slope_before = float(best_before)
            cp_slope_after = float(best_after)

    std_30 = float(np.nanstd(w12[-6:])) if len(w12) >= 6 else float("nan")
    std_180 = float(np.nanstd(w12[-36:])) if len(w12) >= 36 else float("nan")
    mad_30 = _mad(w12[-6:]) if len(w12) >= 6 else float("nan")
    mad_60 = _mad(w12[-12:]) if len(w12) >= 12 else float("nan")

    mean_abs_delta_60 = float("nan")
    if len(w12) >= 12:
        diffs = np.diff(w12[-12:])
        mean_abs_delta_60 = float(np.nanmean(np.abs(diffs)))

    last6h = w12[-72:] if len(w12) >= 72 else w12
    max_drop_5m_last6h = float("nan")
    drop_cnt_5m_ge0p5_last6h = float("nan")
    drop_cnt_5m_ge1p0_last6h = float("nan")
    drop_cnt_5m_ge2p0_last6h = float("nan")
    time_since_last_drop_ge0p5 = float("nan")
    if len(last6h) >= 13:
        diffs5 = np.diff(last6h)
        if diffs5.size:
            max_drop_5m_last6h = float(np.nanmax(-diffs5))
            drop_cnt_5m_ge0p5_last6h = float(np.sum(diffs5 <= -0.5))
            drop_cnt_5m_ge1p0_last6h = float(np.sum(diffs5 <= -1.0))
            drop_cnt_5m_ge2p0_last6h = float(np.sum(diffs5 <= -2.0))
            if bool(np.any(diffs5 <= -0.5)):
                last_drop_idx = int(np.where(diffs5 <= -0.5)[0][-1])
                time_since_last_drop_ge0p5 = float((len(last6h) - 2 - last_drop_idx) * 5)

    is_present = partial_series.notna().to_numpy()
    frac_bins_present_total = float(np.nanmean(is_present)) if is_present.size else float("nan")
    max_gap_minutes_total, _, _ = _gap_stats(is_present, step_minutes=5)

    plateau = {
        "plateau_frac_0p1_last60": float("nan"),
        "plateau_longest_run_0p1_last60": float("nan"),
        "plateau_frac_0p1_last120": float("nan"),
        "plateau_longest_run_0p1_last120": float("nan"),
        "plateau_frac_0p1_last180": float("nan"),
        "plateau_longest_run_0p1_last180": float("nan"),
        "plateau_frac_0p2_last60": float("nan"),
        "plateau_longest_run_0p2_last60": float("nan"),
        "plateau_frac_0p2_last120": float("nan"),
        "plateau_longest_run_0p2_last120": float("nan"),
        "plateau_frac_0p2_last180": float("nan"),
        "plateau_longest_run_0p2_last180": float("nan"),
        "plateau_frac_0p3_last60": float("nan"),
        "plateau_longest_run_0p3_last60": float("nan"),
        "plateau_frac_0p3_last120": float("nan"),
        "plateau_longest_run_0p3_last120": float("nan"),
        "plateau_frac_0p3_last180": float("nan"),
        "plateau_longest_run_0p3_last180": float("nan"),
    }
    if len(w12) >= 24 and np.isfinite(tmax_sofar):
        last120 = w12[-24:]
        last60 = w12[-12:]
        last180 = w12[-36:]
        for thr in [0.1, 0.2, 0.3]:
            mask120 = last120 >= (tmax_sofar - thr)
            mask60 = last60 >= (tmax_sofar - thr)
            mask180 = last180 >= (tmax_sofar - thr)
            key = f"{thr:.1f}".replace(".", "p")
            plateau[f"plateau_frac_0p{key[2:]}_last120"] = float(np.nanmean(mask120))
            plateau[f"plateau_longest_run_0p{key[2:]}_last120"] = float(_longest_run(mask120) * 5)
            plateau[f"plateau_frac_0p{key[2:]}_last60"] = float(np.nanmean(mask60))
            plateau[f"plateau_longest_run_0p{key[2:]}_last60"] = float(_longest_run(mask60) * 5)
            plateau[f"plateau_frac_0p{key[2:]}_last180"] = float(np.nanmean(mask180))
            plateau[f"plateau_longest_run_0p{key[2:]}_last180"] = float(_longest_run(mask180) * 5)

    feats = {
        "temp_last": temp_last,
        "temp_15m": temp_15m,
        "temp_30m": temp_30m,
        "temp_60m": temp_60m,
        "temp_120m": temp_120m,
        "max_sofar": tmax_sofar,
        "tmin_sofar": tmin_sofar,
        "range_sofar": range_sofar,
        "minutes_since_max": minutes_since_max,
        "drop_from_max": drop_from_max,
        "slope_15m": slope_15,
        "slope_30m": slope_30,
        "slope_60m": slope_60,
        "slope_120m": slope_120,
        "slope_180m": slope_180,
        "curvature_30_180": curvature_30_180,
        "curvature_60_180": curvature_60_180,
        "cp_improvement": cp_improvement,
        "cp_time_since": cp_time_since,
        "cp_drop_magnitude": cp_drop_magnitude,
        "cp_slope_before_v6": cp_slope_before,
        "cp_slope_after_v6": cp_slope_after,
        "std_30m": std_30,
        "std_180": std_180,
        "mad_30m": mad_30,
        "mad_60m": mad_60,
        "mean_abs_delta_60m": mean_abs_delta_60,
        "max_drop_5m_last6h": max_drop_5m_last6h,
        "drop_cnt_5m_ge0p5_last6h": drop_cnt_5m_ge0p5_last6h,
        "drop_cnt_5m_ge1p0_last6h": drop_cnt_5m_ge1p0_last6h,
        "drop_cnt_5m_ge2p0_last6h": drop_cnt_5m_ge2p0_last6h,
        "time_since_last_drop_ge0p5": time_since_last_drop_ge0p5,
        "coverage_frac": coverage_frac,
        "last_gap_minutes": last_gap_minutes,
        "frac_bins_present_total": frac_bins_present_total,
        "max_gap_minutes_total": max_gap_minutes_total,
    }
    feats.update(plateau)
    return feats


def _maybe_download_kalshi_csv(target_date: date, out_dir: Path) -> Optional[Path]:
    kalshi_path = KALSHI_DIR / f"{STATION_ID}_{target_date.strftime('%Y%m%d')}.csv"
    if kalshi_path.exists():
        return kalshi_path
    if not KALSHI_DOWNLOADER_SCRIPT.exists():
        return None
    cmd = [
        sys.executable,
        str(KALSHI_DOWNLOADER_SCRIPT),
        "--start-date",
        target_date.isoformat(),
        "--end-date",
        target_date.isoformat(),
        "--out-dir",
        str(KALSHI_DIR),
    ]
    try:
        subprocess.run(cmd, cwd=str(REPO), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        (out_dir / "kalshi_download_stderr.txt").write_text(str(exc.stderr or ""), encoding="utf-8")
        (out_dir / "kalshi_download_stdout.txt").write_text(str(exc.stdout or ""), encoding="utf-8")
        return None
    return kalshi_path if kalshi_path.exists() else None


def _compute_mos_x_bc_for_date(feature_store: pd.DataFrame, eval_date: date, mos_x_mean_eval: float) -> float:
    df = feature_store[["target_date_local", "tmax_full", "mos_x_mean"]].copy()
    df = df[df["target_date_local"] <= (eval_date - timedelta(days=1))].copy()
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "target_date_local": eval_date,
                        "tmax_full": float("nan"),
                        "mos_x_mean": float(mos_x_mean_eval),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    df = df.sort_values("target_date_local")
    err = (pd.to_numeric(df["tmax_full"], errors="coerce").astype(float) - pd.to_numeric(df["mos_x_mean"], errors="coerce").astype(float)).shift(1)
    bias = err.ewm(span=45, adjust=False).mean().fillna(0.0)
    bias_eval = float(bias.iloc[-1]) if len(bias) else 0.0
    return float(mos_x_mean_eval + bias_eval)


def build_eval_row_on_demand(
    *,
    feature_store: pd.DataFrame,
    bundle: Bundle,
    eval_date: date,
    cutoff_utc: datetime,
    minute_dir: Path,
    minute_source: str,
    out_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    now_utc = datetime.now(UTC)
    day_start_utc, day_end_utc = _day_window_utc(eval_date)

    df_1m, minute_meta = _load_minute_obs(
        minute_dir=minute_dir,
        station_id=STATION_ID,
        start_utc=day_start_utc,
        end_utc=cutoff_utc,
        now_utc=now_utc,
        minute_source=minute_source,
    )
    (out_dir / "iem_minute_meta.json").write_text(json.dumps(minute_meta, indent=2, default=str), encoding="utf-8")
    df_1m.to_csv(out_dir / "iem_minute_used.csv", index=False, encoding="utf-8")

    try:
        minute_feats = compute_minute_features_for_day(df_1m, day_start_utc, day_end_utc, cutoff_utc)
    except ValueError as exc:
        if "No minute observations available under strict as-of" in str(exc):
            manifest_row = _read_local_minute_manifest_row(minute_dir=minute_dir, station_id=STATION_ID, year=int(eval_date.year))
            hint = (
                f"{exc} "
                f"(station_id={STATION_ID} date_local={eval_date.isoformat()} "
                f"start_utc={minute_meta.get('start_utc')} end_utc={minute_meta.get('end_utc')} "
                f"local_rows={minute_meta.get('local_rows')} iem_rows={minute_meta.get('iem_rows')} "
                f"expected_minutes={minute_meta.get('expected_minutes')} "
                f"local_manifest_last_ts={manifest_row.get('last_ts') if manifest_row else None} "
                f"local_manifest_status={manifest_row.get('status') if manifest_row else None})"
            )
            raise ValueError(hint) from exc
        raise

    mos_latest, mos_meta = ensure_mos_daily_value_for_day(
        station_id=STATION_ID,
        target_date=eval_date,
        cutoff_utc=cutoff_utc,
        out_dir=out_dir,
    )
    (out_dir / "mos_ingest_meta.json").write_text(json.dumps(mos_meta, indent=2, default=str), encoding="utf-8")
    mos_latest.to_csv(out_dir / "mos_latest_used.csv", index=False, encoding="utf-8")
    if mos_latest.empty:
        raise ValueError(f"MOS unavailable under strict as-of: date={eval_date.isoformat()} cutoff_utc={cutoff_utc.isoformat()}")

    # Sanity: ensure we got the expected (model, variable_code) coverage.
    expected = {(m.upper(), v.lower()) for m in MOS_MODELS for v in MOS_VARIABLES}
    got = {(str(r.model).upper(), str(r.variable_code).lower()) for r in mos_latest.itertuples(index=False)}
    missing = sorted(list(expected - got))
    if missing:
        raise ValueError(f"MOS missing required variables under strict as-of. missing={missing[:20]} (total_missing={len(missing)})")

    mos_df = mos_base_features(mos_latest)
    mos_row = mos_df[mos_df["target_date_local"] == eval_date]
    if mos_row.empty:
        raise ValueError("MOS base feature build produced no row for eval_date.")
    mos_row0 = mos_row.iloc[0]
    mos_feats: Dict[str, float] = {}
    for c in mos_df.columns:
        if c == "target_date_local":
            continue
        v = mos_row0.get(c)
        mos_feats[str(c)] = float(v) if v is not None and pd.notna(v) else float("nan")

    # Derived MOS age features (these are in the model feature list).
    mos_tmp_asof_gfs = mos_row0.get("mos_tmp_asof_utc_gfs")
    mos_tmp_asof_nam = mos_row0.get("mos_tmp_asof_utc_nam")
    mos_age_hours_tmp_gfs = float("nan")
    mos_age_hours_tmp_nam = float("nan")
    if pd.notna(mos_tmp_asof_gfs):
        mos_age_hours_tmp_gfs = float((cutoff_utc - pd.Timestamp(mos_tmp_asof_gfs).to_pydatetime()).total_seconds() / 3600.0)
    if pd.notna(mos_tmp_asof_nam):
        mos_age_hours_tmp_nam = float((cutoff_utc - pd.Timestamp(mos_tmp_asof_nam).to_pydatetime()).total_seconds() / 3600.0)

    # Lag features (from historical feature store).
    prev1 = feature_store[feature_store["target_date_local"] == (eval_date - timedelta(days=1))]
    prev2 = feature_store[feature_store["target_date_local"] == (eval_date - timedelta(days=2))]
    tmax_time_local_lag1 = float(prev1.iloc[0].get("tmax_time_local_minute")) if not prev1.empty else float("nan")
    tmax_time_local_lag2 = float(prev2.iloc[0].get("tmax_time_local_minute")) if not prev2.empty else float("nan")
    range_lag1 = float(prev1.iloc[0].get("range_full")) if not prev1.empty else float("nan")
    range_lag2 = float(prev2.iloc[0].get("range_full")) if not prev2.empty else float("nan")
    outflow_drop_cnt_lag1 = float(prev1.iloc[0].get("outflow_drop_cnt")) if not prev1.empty else float("nan")

    # Calendar features.
    doy = int(pd.Timestamp(eval_date).dayofyear)
    doy_sin = float(math.sin(2.0 * math.pi * float(doy) / 365.25))
    doy_cos = float(math.cos(2.0 * math.pi * float(doy) / 365.25))
    month = int(eval_date.month)

    # Climo range (train-only; years <= 2019).
    years = pd.to_datetime(feature_store["target_date_local"]).dt.year
    train_df = feature_store[years <= 2019].copy()
    range_climo_map, range_overall = _compute_climo_range_map(train_df)
    range_climo_doy = float(range_climo_map.get(doy, range_overall))
    max_sofar = float(minute_feats.get("max_sofar", float("nan")))
    tmin_sofar = float(minute_feats.get("tmin_sofar", float("nan")))
    range_sofar = float(minute_feats.get("range_sofar", float("nan")))
    heating_fraction_obs = float(range_sofar / max(1.0, range_climo_doy)) if math.isfinite(range_sofar) and math.isfinite(range_climo_doy) else float("nan")

    # Suppression index (train-only zscore).
    sup_stats = _compute_suppression_stats(train_df)
    cig = float(mos_feats.get("mos_cig_min_models", float("nan")))
    vis = float(mos_feats.get("mos_vis_min_models", float("nan")))
    q12 = float(mos_feats.get("mos_q12_max_models", float("nan")))
    suppression_index = float("nan")
    if math.isfinite(cig) and math.isfinite(vis) and math.isfinite(q12):
        z_cig_low = (sup_stats["mos_cig_min_models_mean"] - cig) / sup_stats["mos_cig_min_models_std"]
        z_vis_low = (sup_stats["mos_vis_min_models_mean"] - vis) / sup_stats["mos_vis_min_models_std"]
        z_q12_high = (q12 - sup_stats["mos_q12_max_models_mean"]) / sup_stats["mos_q12_max_models_std"]
        suppression_index = float(z_cig_low + z_vis_low + z_q12_high)

    # Derived heat-gap features (used by the model).
    mos_tmax_mean = float(mos_feats.get("mos_tmax_mean", float("nan")))
    mos_tmin_mean = float(mos_feats.get("mos_tmin_mean", float("nan")))
    mos_range_mean = float(mos_feats.get("mos_range_mean", mos_tmax_mean - mos_tmin_mean))
    mos_range_denom = max(1.0, float(mos_range_mean)) if math.isfinite(mos_range_mean) else float("nan")
    gap_to_mos_tmax = (mos_tmax_mean - max_sofar) if math.isfinite(mos_tmax_mean) and math.isfinite(max_sofar) else float("nan")
    gap_norm = (gap_to_mos_tmax / mos_range_denom) if math.isfinite(gap_to_mos_tmax) and math.isfinite(mos_range_denom) else float("nan")
    completion_frac = ((max_sofar - tmin_sofar) / mos_range_denom) if math.isfinite(max_sofar) and math.isfinite(tmin_sofar) and math.isfinite(mos_range_denom) else float("nan")
    dd_models = float("nan")
    if math.isfinite(mos_feats.get("mos_tmp_mean_models", float("nan"))) and math.isfinite(mos_feats.get("mos_dpt_mean_models", float("nan"))):
        dd_models = float(mos_feats["mos_tmp_mean_models"] - mos_feats["mos_dpt_mean_models"])

    # MOS X bias-corrected feature (EWMA using <= eval_date-1 truth only).
    mos_x_mean = float(mos_feats.get("mos_x_mean", float("nan")))
    mos_x_bc = _compute_mos_x_bc_for_date(feature_store, eval_date, mos_x_mean) if math.isfinite(mos_x_mean) else float("nan")

    row: Dict[str, Any] = {
        "target_date_local": eval_date,
        "cutoff_utc": cutoff_utc,
        # Training labels / full-day truth are unknown live; leave NaN.
        "y_hit_by_cutoff": float("nan"),
        "tmax_full": float("nan"),
        "range_full": float("nan"),
        "outflow_drop_cnt": float("nan"),
        "tmax_time_local_minute": float("nan"),
        # Bridge-required aliases.
        "tmax_sofar": float(max_sofar) if math.isfinite(max_sofar) else float("nan"),
        # Core minute + MOS features.
        **minute_feats,
        **mos_feats,
        # Derived model features.
        "mos_age_hours_tmp_gfs": mos_age_hours_tmp_gfs,
        "mos_age_hours_tmp_nam": mos_age_hours_tmp_nam,
        "tmax_time_local_lag1": tmax_time_local_lag1,
        "tmax_time_local_lag2": tmax_time_local_lag2,
        "range_lag1": range_lag1,
        "range_lag2": range_lag2,
        "outflow_drop_cnt_lag1": outflow_drop_cnt_lag1,
        "gap_to_mos_tmax": gap_to_mos_tmax,
        "gap_norm": gap_norm,
        "completion_frac": completion_frac,
        "dd_models": dd_models,
        "suppression_index": suppression_index,
        "mos_x_bc": mos_x_bc,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "month": month,
        "range_climo_doy": range_climo_doy,
        "heating_fraction_obs": heating_fraction_obs,
        # These are overwritten inside predict_p_hit, but add placeholders for clarity/audit.
        "p_onshore": float("nan"),
        "cp_exists": float("nan"),
    }

    # Ensure every bundle feature column exists for the eval row. Missing will be median-imputed, but
    # explicitly set to NaN so it's visible in the snapshot/debug.
    for c in bundle.feature_list:
        row.setdefault(c, float("nan"))

    eval_df = pd.DataFrame([row])
    return eval_df, {
        "minute": minute_meta,
        "mos_rows_latest": int(len(mos_latest)),
        "mos_ingest": mos_meta,
    }


def parse_bucket_interval(label: str) -> BucketInterval:
    # Handle both correct degree char and common double-encoded form (\u00c2\u00b0).
    cleaned = label.replace("\u00c2\u00b0", "").replace("\u00b0", "").strip().lower()
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
    df = pd.read_parquet(FEATURE_STORE_PATH, columns=columns)
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


def live_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe single-day evaluator for the KMIA Kalshi bridge strategy. "
            "Computes p_hit from the exported bundle, builds the bridge using only <date truth, "
            "and lists bucket trade candidates at cutoff."
        )
    )
    parser.add_argument("--date", required=True, help="Target settlement date (local) YYYY-MM-DD.")
    parser.add_argument("--kalshi-csv", default=None, help="Override Kalshi CSV (default: data/kalshi_backtest_data/KMIA_YYYYMMDD.csv).")
    parser.add_argument("--bundle-dir", default=None, help="Override bundle dir (default: latest under artifacts/model_bundles/hit1830_v6/B6_EXP20_GAM_RESIDUAL).")
    parser.add_argument("--out-dir", default=None, help="Output dir (default: artifacts/live_bridge_eval/<timestamp>_<date>).")
    parser.add_argument("--ensure-data", dest="ensure_data", action="store_true", default=True, help="Auto-download/build missing inputs (default).")
    parser.add_argument("--no-ensure-data", dest="ensure_data", action="store_false", help="Disable auto-download/build; fail fast if inputs missing.")
    parser.add_argument("--minute-dir", default=str(DEFAULT_MINUTE_DIR), help="Local IEM minute CSV folder (yearly files).")
    parser.add_argument("--minute-source", choices=["auto", "local", "iem"], default="auto", help="Minute data source for on-demand eval rows.")

    parser.add_argument("--bridge-mode", choices=["expanding", "trailing"], default=DEFAULT_BRIDGE_MODE)
    parser.add_argument("--trailing-years", type=int, default=DEFAULT_TRAILING_YEARS)
    parser.add_argument("--bridge-train-end", default=None, help="Bridge train end YYYY-MM-DD (default: date-1). Must be < date.")

    parser.add_argument("--p-hit-gate", type=float, default=DEFAULT_REQUIRE_P_HIT_GE, help="Require p_hit >= gate. Set to -1 to disable.")
    parser.add_argument("--min-win-prob", type=float, default=DEFAULT_MIN_WIN_PROB, help="Minimum win probability for the side you buy.")
    parser.add_argument("--edge-prob", type=float, default=DEFAULT_EDGE_PROB, help="Required edge vs market probability points (e.g., 0.15).")

    parser.add_argument("--simulate", action="store_true", help="(Historical) Scan forward for fills and compute win/loss using truth.")

    args = parser.parse_args(argv)

    if STOCKHOLM_TZ is None or MIAMI_TZ is None:
        raise RuntimeError("ZoneInfo not available; cannot compute local times.")

    eval_date = date.fromisoformat(str(args.date))

    bundle_dir = Path(args.bundle_dir).resolve() if args.bundle_dir else _find_latest_bundle_dir(BUNDLE_ROOT)
    if bundle_dir is None:
        raise SystemExit(
            f"No exported bundle found under: {BUNDLE_ROOT}\n"
            "Run: python tools/early_maxout_strategy/export_b6_exp20_bundle.py"
        )
    bundle = load_bundle(bundle_dir)

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (REPO / "artifacts" / "live_bridge_eval" / f"{utc_now_tag()}_{eval_date.strftime('%Y%m%d')}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the minimum feature-store columns needed for inference + bridge training.
    missing_cols = {
        "cp_exists",
        "mos_tmax_bias_gfs_a0.020",
        "mos_tmax_bc_gfs_a0.020",
        "mos_tmax_bias_nam_a0.020",
        "mos_tmax_bc_nam_a0.020",
        "mos_tmax_mean_bc_a0.020",
        "mos_tmax_bias_gfs_a0.050",
        "mos_tmax_bc_gfs_a0.050",
        "mos_tmax_bias_nam_a0.050",
        "mos_tmax_bc_nam_a0.050",
        "mos_tmax_mean_bc_a0.050",
    }
    read_cols = [c for c in bundle.feature_list if c not in missing_cols]
    read_cols += [
        "target_date_local",
        "cutoff_utc",
        "tmax_sofar",
        "coverage_frac",
        "y_hit_by_cutoff",
        "cp_improvement",
        "tmax_full",
        "range_full",
        "tmax_time_local_minute",
        "outflow_drop_cnt",
        "mos_tmax_gfs",
        "mos_tmax_nam",
        "mos_u_mean",
        "mos_v_mean",
        "mos_wsp_mean_models",
        "doy_sin",
        "doy_cos",
        "mos_x_mean",
        "last_gap_minutes",
    ]
    read_cols = sorted(set(read_cols))

    df = pd.read_parquet(FEATURE_STORE_PATH, columns=read_cols)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["cutoff_utc"] = pd.to_datetime(df["cutoff_utc"], utc=True, errors="coerce")

    cutoff_calc = compute_cutoff_utc(eval_date)

    row_eval = df[df["target_date_local"] == eval_date]
    if row_eval.empty:
        if not bool(args.ensure_data):
            raise SystemExit(f"Feature store has no row for date={eval_date} (needed for model inference).")

        # For true live usage, you should run at/after cutoff. If you run too early, refuse to
        # treat pre-cutoff observations/prices as "cutoff" inputs.
        now_utc = datetime.now(UTC)
        if now_utc + timedelta(seconds=5) < cutoff_calc:
            print("NO_TRADES")
            print(
                "- Reason: decision cutoff has not happened yet for this date under strict as-of.\n"
                f"  now_utc={now_utc.isoformat()} cutoff_utc={cutoff_calc.isoformat()}"
            )
            return 0

        try:
            eval_df, ensure_meta = build_eval_row_on_demand(
                feature_store=df,
                bundle=bundle,
                eval_date=eval_date,
                cutoff_utc=cutoff_calc,
                minute_dir=Path(args.minute_dir),
                minute_source=str(args.minute_source),
                out_dir=out_dir,
            )
        except Exception as exc:
            print("NO_TRADES")
            print(f"- Reason: unable to build eval-date features on-demand: {type(exc).__name__}: {exc}")
            return 0

        # Persist a readable snapshot of the feature row used for inference (not the full historical store).
        eval_row_obj = eval_df.iloc[0].to_dict()
        (out_dir / "eval_row_features.json").write_text(json.dumps(eval_row_obj, indent=2, default=str), encoding="utf-8")
        (out_dir / "ensure_meta.json").write_text(json.dumps(ensure_meta, indent=2, default=str), encoding="utf-8")

        df = pd.concat([df, eval_df], ignore_index=True)
        df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
        df["cutoff_utc"] = pd.to_datetime(df["cutoff_utc"], utc=True, errors="coerce")
        row_eval = df[df["target_date_local"] == eval_date]
        if row_eval.empty:
            raise SystemExit("Internal error: eval row build succeeded but did not materialize in feature store frame.")

    cutoff_feat = row_eval.iloc[0]["cutoff_utc"]
    cutoff_utc = cutoff_calc
    cutoff_mismatch_s = None
    if pd.notna(cutoff_feat):
        cutoff_feat_dt = pd.Timestamp(cutoff_feat).to_pydatetime()
        cutoff_mismatch_s = abs((cutoff_feat_dt - cutoff_calc).total_seconds())
        cutoff_utc = cutoff_feat_dt if cutoff_mismatch_s > 60.0 else cutoff_calc

    cutoff_miami = cutoff_utc.astimezone(MIAMI_TZ)
    cutoff_stockholm = cutoff_utc.astimezone(STOCKHOLM_TZ)

    p_hit, p_debug = predict_p_hit(bundle, df, eval_date)
    p_hit_gate = float(args.p_hit_gate)
    gate_ok = (p_hit_gate < 0.0) or (p_hit >= p_hit_gate)

    max_sofar = float(row_eval.iloc[0]["tmax_sofar"]) if row_eval.iloc[0].get("tmax_sofar") is not None else float("nan")
    mos_x_mean = float(row_eval.iloc[0]["mos_x_mean"]) if row_eval.iloc[0].get("mos_x_mean") is not None else float("nan")
    coverage_frac = float(row_eval.iloc[0]["coverage_frac"]) if row_eval.iloc[0].get("coverage_frac") is not None else float("nan")
    cover_bin = 0 if not math.isfinite(coverage_frac) else (2 if coverage_frac >= 0.95 else (1 if coverage_frac >= 0.80 else 0))

    snapshot = {
        "date_local": eval_date.isoformat(),
        "cutoff_utc": cutoff_utc.astimezone(UTC).isoformat(),
        "cutoff_miami": cutoff_miami.isoformat(),
        "cutoff_stockholm": cutoff_stockholm.isoformat(),
        "cutoff_mismatch_seconds": float(cutoff_mismatch_s) if cutoff_mismatch_s is not None else None,
        "station_id": STATION_ID,
        "model_name": MODEL_NAME,
        "bundle_dir": str(bundle.bundle_dir),
        "p_hit": float(p_hit),
        "p_hit_gate": float(p_hit_gate),
        "gate_ok": bool(gate_ok),
        "tmax_sofar_iem": float(max_sofar) if math.isfinite(max_sofar) else None,
        "mos_x_mean": float(mos_x_mean) if math.isfinite(mos_x_mean) else None,
        "coverage_frac": float(coverage_frac) if math.isfinite(coverage_frac) else None,
        "cover_bin": int(cover_bin),
        "debug": p_debug,
    }
    (out_dir / "snapshot.json").write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")

    print("LIVE_EVAL_KALSHI_BRIDGE")
    print(f"- date_local: {eval_date.isoformat()}")
    print(f"- cutoff_utc: {cutoff_utc.astimezone(UTC).isoformat()}")
    print(f"- cutoff_miami: {cutoff_miami.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"- cutoff_stockholm: {cutoff_stockholm.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"- bundle_dir: {bundle.bundle_dir}")
    print(f"- p_hit: {p_hit:.6f} (gate {p_hit_gate} -> {'PASS' if gate_ok else 'FAIL'})")
    print(f"- tmax_sofar_iem: {max_sofar if math.isfinite(max_sofar) else 'NaN'}")
    print(f"- coverage_frac: {coverage_frac if math.isfinite(coverage_frac) else 'NaN'} (cover_bin={cover_bin})")
    print(f"- outputs: {out_dir}")
    print("")

    if not gate_ok:
        print("NO_TRADES")
        print("- Reason: p_hit gate failed. Strategy stops here.")
        return 0

    bridge_train_end = date.fromisoformat(args.bridge_train_end) if args.bridge_train_end else (eval_date - timedelta(days=1))
    if not (bridge_train_end < eval_date):
        raise SystemExit("Leakage guard: --bridge-train-end must be < --date.")

    trailing_years: Optional[int] = None
    if args.bridge_mode == "expanding":
        bridge_train_start = date(2002, 1, 1)
    else:
        trailing_years = int(args.trailing_years)
        if trailing_years <= 0:
            raise SystemExit("--trailing-years must be > 0")
        bridge_train_start = _compute_trailing_start(bridge_train_end, trailing_years)

    truth_train = fetch_truth_tmax(STATION_ID, bridge_train_start, bridge_train_end)
    bridge_cols = ["target_date_local", "tmax_sofar", "coverage_frac", "y_hit_by_cutoff"]
    bridge_features = df[bridge_cols].copy()

    meta, counters, counts = build_d_distributions(bridge_features, truth_train, bridge_train_start, bridge_train_end)
    (out_dir / "bridge_meta.json").write_text(
        json.dumps(
            {
                **meta.__dict__,
                "bridge_mode": str(args.bridge_mode),
                "trailing_years": int(trailing_years) if trailing_years is not None else None,
                "bridge_train_start": bridge_train_start.isoformat(),
                "bridge_train_end": bridge_train_end.isoformat(),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_bridge_tables(out_dir, counters, counts)

    scope_hit, d_hit = get_d_distribution(counters, counts, hit=1, cover_bin=cover_bin)
    scope_not, d_not = get_d_distribution(counters, counts, hit=0, cover_bin=cover_bin)
    d_mix = mix_distributions(p_hit, d_hit, d_not)

    if not d_mix:
        print("NO_TRADES")
        print("- Reason: empty D distribution after mixing.")
        return 0

    kalshi_path = Path(args.kalshi_csv).resolve() if args.kalshi_csv else (KALSHI_DIR / f"{STATION_ID}_{eval_date.strftime('%Y%m%d')}.csv")
    if not kalshi_path.exists() and bool(args.ensure_data) and args.kalshi_csv is None:
        dl = _maybe_download_kalshi_csv(eval_date, out_dir)
        if dl is not None:
            kalshi_path = dl
    if not kalshi_path.exists():
        print("NO_TRADES")
        print(f"- Reason: missing Kalshi CSV: {kalshi_path}")
        if args.kalshi_csv is None and not bool(args.ensure_data):
            print("- Hint: run with --ensure-data to auto-download.")
        return 0

    kal = pd.read_csv(kalshi_path)
    if "timestamp" not in kal.columns:
        raise SystemExit(f"Kalshi CSV missing 'timestamp' col: {kalshi_path}")
    kal["timestamp"] = pd.to_datetime(kal["timestamp"], utc=True, errors="coerce")
    kal = kal.dropna(subset=["timestamp"]).sort_values("timestamp")
    bucket_cols = [c for c in kal.columns if c != "timestamp"]
    for c in bucket_cols:
        kal[c] = pd.to_numeric(kal[c], errors="coerce")

    bucket_rows: List[Dict[str, object]] = []
    for b in bucket_cols:
        try:
            interval = parse_bucket_interval(str(b))
        except Exception:
            continue
        p_bucket = bucket_win_prob(max_sofar, d_mix, interval) if math.isfinite(max_sofar) else float("nan")
        yes_at_cutoff = price_at_or_before(kal, cutoff_utc, str(b))
        bucket_rows.append(
            {
                "bucket_label": str(b),
                "lower": float(interval.lower),
                "upper": float(interval.upper),
                "bucket_p_win": float(p_bucket),
                "yes_price_at_cutoff": float(yes_at_cutoff) if yes_at_cutoff is not None else None,
            }
        )
    df_buckets = pd.DataFrame(bucket_rows)
    df_buckets.to_csv(out_dir / "bucket_probs.csv", index=False, encoding="utf-8")

    if df_buckets.empty:
        print("NO_TRADES")
        print("- Reason: no parsable buckets in Kalshi CSV.")
        return 0

    min_win_prob = float(args.min_win_prob)
    edge_prob = float(args.edge_prob)
    candidates: List[Dict[str, object]] = []
    for r in df_buckets.itertuples(index=False):
        bucket = str(getattr(r, "bucket_label"))
        p_bucket = float(getattr(r, "bucket_p_win"))
        yes_at_cutoff = getattr(r, "yes_price_at_cutoff")
        if yes_at_cutoff is None:
            continue
        yes_at_cutoff = float(yes_at_cutoff)
        if not math.isfinite(yes_at_cutoff) or yes_at_cutoff < 0.0 or yes_at_cutoff > 100.0:
            continue
        m_yes = yes_at_cutoff / 100.0

        if math.isfinite(p_bucket) and p_bucket >= min_win_prob:
            thr_yes = (p_bucket - edge_prob) * 100.0
            if thr_yes > 0.0:
                candidates.append(
                    {
                        "bucket_label": bucket,
                        "trade_side": "YES",
                        "bucket_p_win": p_bucket,
                        "trade_p_win": p_bucket,
                        "yes_price_at_cutoff": yes_at_cutoff,
                        "market_p_win": m_yes,
                        "edge_pp_at_cutoff": p_bucket - m_yes,
                        "threshold_yes_price": thr_yes,
                        "enter_now": bool(yes_at_cutoff <= thr_yes),
                        "entry_rule": f"BUY YES <= {thr_yes:.2f}c",
                    }
                )

        if math.isfinite(p_bucket) and (1.0 - p_bucket) >= min_win_prob:
            thr_yes = (p_bucket + edge_prob) * 100.0
            if thr_yes < 100.0:
                candidates.append(
                    {
                        "bucket_label": bucket,
                        "trade_side": "NO",
                        "bucket_p_win": p_bucket,
                        "trade_p_win": 1.0 - p_bucket,
                        "yes_price_at_cutoff": yes_at_cutoff,
                        "market_p_win": 1.0 - m_yes,
                        "edge_pp_at_cutoff": m_yes - p_bucket,
                        "threshold_yes_price": thr_yes,
                        "enter_now": bool(yes_at_cutoff >= thr_yes),
                        "entry_rule": f"BUY NO when YES >= {thr_yes:.2f}c (NO <= {100.0 - thr_yes:.2f}c)",
                    }
                )

    if not candidates:
        max_p = float(df_buckets["bucket_p_win"].max()) if df_buckets["bucket_p_win"].notna().any() else float("nan")
        min_p = float(df_buckets["bucket_p_win"].min()) if df_buckets["bucket_p_win"].notna().any() else float("nan")
        print("NO_TRADES")
        print("- Reason: no buckets met the win-probability rule for either side.")
        print(f"- min_win_prob={min_win_prob} implies: YES requires p_bucket>= {min_win_prob:.2f}, NO requires p_bucket<= {(1.0 - min_win_prob):.2f}")
        print(f"- observed p_bucket range: min={min_p:.4f}, max={max_p:.4f}")
        return 0

    df_cand = pd.DataFrame(candidates)
    df_cand = df_cand.sort_values(by=["edge_pp_at_cutoff", "trade_p_win"], ascending=False, kind="mergesort").reset_index(drop=True)
    df_cand.to_csv(out_dir / "candidates.csv", index=False, encoding="utf-8")
    (out_dir / "candidates.md").write_text(df_to_markdown_simple(df_cand), encoding="utf-8")

    best = df_cand.iloc[0].to_dict()
    (out_dir / "best_candidate.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")

    print("BRIDGE")
    print(f"- train: {bridge_train_start.isoformat()} .. {bridge_train_end.isoformat()} (n={meta.n_train_rows})")
    print(f"- D scopes: hit={scope_hit}, not_hit={scope_not}")
    print("")

    ev_cents_req = edge_prob * 100.0
    df_now = df_cand[df_cand["enter_now"] == True].copy()  # noqa: E712

    print("ELIGIBLE TRADES AT CUTOFF (EV + win% rules satisfied NOW)")
    print(f"- rule: win% >= {min_win_prob * 100.0:.0f}%, EV >= {ev_cents_req:.0f}c")
    print(f"- eligible_now: {len(df_now)} of {len(df_cand)} candidates")
    print("")

    if df_now.empty:
        print("NO ELIGIBLE TRADES AT CUTOFF")
        print(f"- Reason: market prices at cutoff do not offer EV >= {ev_cents_req:.0f}c for any win%-eligible bucket/side.")
        print("- You can still place limit orders at the printed thresholds and wait for a fill.")
        print("")
    else:
        top_n = min(5, len(df_now))
        for i, r in enumerate(df_now.head(top_n).itertuples(index=False), start=1):
            side = str(getattr(r, "trade_side"))
            bucket = str(getattr(r, "bucket_label"))
            yes_cutoff = float(getattr(r, "yes_price_at_cutoff"))
            no_cutoff = 100.0 - yes_cutoff
            market_win = float(getattr(r, "market_p_win"))
            model_win = float(getattr(r, "trade_p_win"))
            ev_cutoff = model_win - market_win
            entry_rule = str(getattr(r, "entry_rule"))

            print(f"{i}) {side} {bucket}")
            print(f"   - Market win% (implied): {market_win * 100.0:.2f}%")
            print(f"   - Model win%:           {model_win * 100.0:.2f}%")
            print(f"   - EV at cutoff:         {ev_cutoff * 100.0:+.2f}c (model - market)")
            print(f"   - Cutoff prices:        YES={yes_cutoff:.2f}c, NO={no_cutoff:.2f}c")
            print(f"   - Entry rule:           {entry_rule}")
            print("")

    if args.simulate:
        truth_day = fetch_truth_tmax(STATION_ID, eval_date, eval_date)
        truth_tmax = float(truth_day.iloc[0]["tmax_f"]) if not truth_day.empty else None
        sim_rows: List[Dict[str, object]] = []
        for r in df_cand.itertuples(index=False):
            bucket = str(getattr(r, "bucket_label"))
            side = str(getattr(r, "trade_side"))
            trade_p_win = float(getattr(r, "trade_p_win"))
            thr_yes = float(getattr(r, "threshold_yes_price"))
            if side == "YES":
                entry_time, entry_yes = first_price_at_or_below(kal, cutoff_utc, bucket, thr_yes)
                entry_price = entry_yes
            else:
                entry_time, entry_yes = first_price_at_or_above(kal, cutoff_utc, bucket, thr_yes)
                entry_price = (100.0 - entry_yes) if entry_yes is not None else None

            win = None
            if truth_tmax is not None:
                interval = parse_bucket_interval(bucket)
                bucket_hit = (float(truth_tmax) > float(interval.lower)) and (float(truth_tmax) <= float(interval.upper))
                win = bool(bucket_hit) if side == "YES" else bool(not bucket_hit)

            sim_rows.append(
                {
                    "bucket_label": bucket,
                    "trade_side": side,
                    "trade_p_win": trade_p_win,
                    "threshold_yes_price": thr_yes,
                    "entry_time": entry_time.astimezone(UTC).isoformat() if entry_time else None,
                    "entry_yes_price": float(entry_yes) if entry_yes is not None else None,
                    "entry_price": float(entry_price) if entry_price is not None else None,
                    "truth_tmax_f": truth_tmax,
                    "win": win,
                    "ev_at_entry": (trade_p_win - (float(entry_price) / 100.0)) if entry_price is not None else None,
                }
            )

        df_sim = pd.DataFrame(sim_rows)
        df_sim.to_csv(out_dir / "candidates_simulated.csv", index=False, encoding="utf-8")
        (out_dir / "candidates_simulated.md").write_text(df_to_markdown_simple(df_sim), encoding="utf-8")
        print("")
        print("SIMULATION")
        print(f"- wrote: {out_dir / 'candidates_simulated.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(live_main())
