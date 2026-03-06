
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


def find_repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in cur.parents:
        if (parent / "pom.xml").exists() and (parent / "ingestion-service").exists():
            return parent
    raise FileNotFoundError("Could not locate repo root from tools/live path")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_live.python.fetch.iem_mos import fetch_mos_payload, mos_window_utc
from tools.live import mos_blend12_bundle as blend12_bundle


DEFAULT_LIVE_ROOT = Path(r"D:\Ahmed\data\live\mos_quantile_live_inference")
DEFAULT_IEM_BASE_URL = "https://mesonet.agron.iastate.edu"

DEFAULT_MODEL_BUNDLE_KNYC = Path(r"D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_12\live_model_bundle_v2_20260302")
DEFAULT_MODEL_BUNDLE_KMIA = Path(r"D:\Ahmed\data\kalshi\Experiments\MOS_KMIA\03_blends\blend_12\live_model_bundle_v2_20260302")
DEFAULT_MODEL_BUNDLE_KMDW = Path(r"D:\Ahmed\data\kalshi\Experiments\MOS_KMDW\03_blends\blend_12\live_model_bundle_v2_20260304")
DEFAULT_MODEL_BUNDLE_KLAX = Path(r"D:\Ahmed\data\kalshi\Experiments\MOS_KLAX\03_blends\blend_12\live_model_bundle_v2_20260305")

DEFAULT_MOS_ARCHIVE_KNYC = Path(r"D:\Ahmed\data\kalshi\training_data\04_mos\archive_merged\KNYC_mos_archive_2000_2025.csv.gz")
DEFAULT_MOS_ARCHIVE_KMIA = Path(r"D:\Ahmed\data\kalshi\training_data\04_mos\archive_merged\KMIA_mos_archive_2000_2025.csv.gz")
DEFAULT_MOS_ARCHIVE_KMDW = Path(r"D:\Ahmed\data\kalshi\training_data\04_mos\archive_merged\KMDW_mos_archive_2002_2026.csv.gz")
DEFAULT_MOS_ARCHIVE_KLAX = Path(r"D:\Ahmed\data\kalshi\training_data\04_mos\archive_merged\KLAX_mos_archive_2002_2026.csv.gz")
DEFAULT_TRUTH_KNYC = Path(r"D:\Ahmed\data\kalshi\training_data\02_truth\KNYC_settled_tmax.csv")
DEFAULT_TRUTH_KMIA = Path(r"D:\Ahmed\data\kalshi\training_data\02_truth\KMIA_settled_tmax.csv")
DEFAULT_TRUTH_KMDW = Path(r"D:\Ahmed\data\kalshi\training_data\02_truth\KMDW_settled_tmax_2002_2026.csv")
DEFAULT_TRUTH_KLAX = Path(r"D:\Ahmed\data\kalshi\training_data\02_truth\KLAX_settled_tmax_2002_2026.csv")

DEFAULT_SERIES_BY_STATION = {"KNYC": "KXHIGHNY", "KMIA": "KXHIGHMIA", "KMDW": "KXHIGHCHI", "KLAX": "KXHIGHLAX"}
DEFAULT_FILE_PREFIX_BY_STATION = {"KNYC": "KNYC", "KMIA": "KMIA", "KMDW": "KMDW", "KLAX": "KLAX"}
DEFAULT_ZONE_BY_STATION = {
    "KNYC": "America/New_York",
    "KMIA": "America/New_York",
    "KMDW": "America/Chicago",
    "KLAX": "America/Los_Angeles",
}

QUANTILES = list(blend12_bundle.QUANTILES)
QUANTILE_COLUMNS = list(blend12_bundle.QUANTILE_COLUMNS)
FEATURE_COLUMNS = list(blend12_bundle.FEATURE_COLUMNS)

RAW_FIELD_NAMES = [
    "tmp",
    "dpt",
    "cld_raw",
    "sky",
    "wdr",
    "wsp",
    "gst",
    "p06",
    "p12",
    "t06",
    "t12",
    "n_x",
    "n_n",
]

TRAIN_DEV_START = "2022-01-01"
TRAIN_DEV_END = "2023-12-31"
TRAIN_FULL_END = "2023-12-31"
SEED = 42
EPS = 1e-9


@dataclass(frozen=True)
class SliceDef:
    sid: str
    model: str
    runtime_hour: int
    train_start: str


SLICE_DEFS = [
    SliceDef(s.sid, s.model, s.runtime_hour, s.train_start) for s in blend12_bundle.SLICE_DEFS
]


@dataclass(frozen=True)
class Bucket:
    label_raw: str
    lo: Optional[int]
    hi: Optional[int]
    mode: str

    def canonical_label(self) -> str:
        if self.mode == "or_below" and self.hi is not None:
            return f"{self.hi}F or below"
        if self.mode == "or_above" and self.lo is not None:
            return f"{self.lo}F or above"
        if self.mode == "range" and self.lo is not None and self.hi is not None:
            return f"{self.lo}F to {self.hi}F"
        return self.label_raw


@dataclass
class StationConfig:
    station_id: str
    zoneid: str
    series: str
    file_prefix: str
    market_root: Path
    bundle_dir: Path
    mos_archive_path: Path
    truth_csv_path: Path


@dataclass
class StationBundle:
    point_models: Dict[str, Any]
    quantile_models: Dict[str, Dict[float, Any]]
    medians: Dict[str, Dict[str, float]]
    blend_point_weight: float
    blend_quantile_weights: Dict[float, float]
    manifest: Dict[str, Any]
    artifact_hashes: Dict[str, str]


@dataclass
class StationContext:
    station_id: str
    target_date_local: str
    runtime_utc: pd.Timestamp
    quote_asof_utc: pd.Timestamp
    gate_cutoff_utc: pd.Timestamp
    effective_cutoff_utc: pd.Timestamp
    market_open_utc: pd.Timestamp
    market_file: Path
    market_file_sha256: str
    market_file_day_local: str
    bucket_columns: List[str]
    qmap: Dict[float, float]
    pmf: Dict[int, float]
    quantiles_monotonic: bool


@dataclass(frozen=True)
class RuntimeGateFailure:
    station_id: str
    slice_id: str
    model: str
    target_date_local: str
    quote_asof_utc: str
    expected_runtime_utc: str
    selected_runtime_utc: Optional[str]
    latest_available_runtime_utc: Optional[str]
    available_target_runtimes_utc: List[str]
    expected_runtime_present_for_target: bool
    reason: str


class RequiredRuntimeNotReadyError(RuntimeError):
    def __init__(self, failure: RuntimeGateFailure) -> None:
        self.failure = failure
        latest = failure.latest_available_runtime_utc or "NONE"
        selected = failure.selected_runtime_utc or "NONE"
        msg = (
            "Required runtime not ready: "
            f"station={failure.station_id} slice={failure.slice_id} model={failure.model} "
            f"reason={failure.reason} expected={failure.expected_runtime_utc} "
            f"selected={selected} latest_available={latest}"
        )
        super().__init__(msg)


def parse_target_date(value: str) -> date:
    raw = value.strip()
    if not raw:
        raise ValueError("--target-date is required")
    normalized = raw.replace(":", "-")

    if len(normalized) == 8 and normalized.isdigit():
        return datetime.strptime(normalized, "%Y%m%d").date()

    m_short = re.match(r"^(\d{2})-(\d{2})-(\d{2})$", normalized)
    if m_short:
        yy = int(m_short.group(1))
        mm = int(m_short.group(2))
        dd = int(m_short.group(3))
        return date(2000 + yy, mm, dd)

    return datetime.strptime(normalized, "%Y-%m-%d").date()


def parse_utc_timestamp(value: str) -> pd.Timestamp:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    ts = pd.Timestamp(cleaned)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def safe_iso_utc(ts: pd.Timestamp | datetime) -> str:
    return pd.Timestamp(ts).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_logger(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("mos_quantile_live_inference")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")

    # Keep logs on stderr so stdout can be consumed as a clean JSON channel by script invokers.
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ----------------------------
# Probability helpers
# ----------------------------
def parse_bucket_label(label: str) -> Optional[Bucket]:
    s = str(label).strip().lower().replace(" to ", "-")
    s = re.sub(r"\s+", " ", s)
    lt_match = re.search(r"<\s*(\d+)", s)
    if lt_match:
        bound = int(lt_match.group(1)) - 1
        return Bucket(label_raw=str(label), lo=None, hi=bound, mode="or_below")
    le_match = re.search(r"<=\s*(\d+)", s)
    if le_match:
        return Bucket(label_raw=str(label), lo=None, hi=int(le_match.group(1)), mode="or_below")
    gt_match = re.search(r">\s*(\d+)", s)
    if gt_match:
        bound = int(gt_match.group(1)) + 1
        return Bucket(label_raw=str(label), lo=bound, hi=None, mode="or_above")
    ge_match = re.search(r">=\s*(\d+)", s)
    if ge_match:
        return Bucket(label_raw=str(label), lo=int(ge_match.group(1)), hi=None, mode="or_above")
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if ("or below" in s or "or less" in s) and nums:
        return Bucket(label_raw=str(label), lo=None, hi=nums[0], mode="or_below")
    if ("or above" in s or "or higher" in s) and nums:
        return Bucket(label_raw=str(label), lo=nums[0], hi=None, mode="or_above")
    if len(nums) >= 2:
        lo, hi = sorted([nums[0], nums[1]])
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


def is_quantiles_monotonic(qmap: Dict[float, float]) -> bool:
    arr = np.array([qmap[q] for q in QUANTILES], dtype=float)
    return bool(np.all(np.diff(arr) >= -1e-12))


def enforce_non_cross(qmap: Dict[float, float]) -> Dict[float, float]:
    running = -1e18
    out: Dict[float, float] = {}
    for q in QUANTILES:
        running = max(running, float(qmap[q]))
        out[q] = float(running)
    return out


def pmf_mean(pmf: Dict[int, float]) -> float:
    return float(sum(float(k) * float(v) for k, v in pmf.items()))


def pmf_median(pmf: Dict[int, float]) -> float:
    running = 0.0
    for k in sorted(pmf.keys()):
        running += float(pmf[k])
        if running >= 0.5:
            return float(k)
    return float(max(pmf.keys()))


# ----------------------------
# Runtime rules
# ----------------------------
def compute_runtime_utc_for_target(target_date_local: date, policy: str) -> pd.Timestamp:
    if policy != "blend12_tminus1_1200z":
        raise ValueError(f"Unsupported runtime policy: {policy}")
    ts = pd.Timestamp(
        datetime(target_date_local.year, target_date_local.month, target_date_local.day, 12, 0, tzinfo=timezone.utc)
    ) - pd.Timedelta(days=1)
    return ts.tz_convert("UTC")


def compute_gate_cutoff_utc(target_date_local: date, entry_hour_z: int, entry_minute_z: int) -> pd.Timestamp:
    ts = datetime(target_date_local.year, target_date_local.month, target_date_local.day, 0, 0, tzinfo=timezone.utc)
    return pd.Timestamp(ts - timedelta(days=1) + timedelta(hours=entry_hour_z, minutes=entry_minute_z))


def market_file_date_from_path(path: Path) -> Optional[str]:
    m = re.search(r"_(\d{8})\.csv$", path.name)
    if not m:
        return None
    ymd = m.group(1)
    return f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"


# ----------------------------
# Model training + bundle
# ----------------------------
def load_truth(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path).rename(columns={"date": "target_date_local", "settled_tmax": "y_tmax"})
    t["target_date_local"] = pd.to_datetime(t["target_date_local"], errors="coerce").dt.normalize()
    t["y_tmax"] = pd.to_numeric(t["y_tmax"], errors="coerce")
    t = t.dropna(subset=["target_date_local", "y_tmax"])[["target_date_local", "y_tmax"]].drop_duplicates("target_date_local")
    return t


def load_mos(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path)
    m["runtime_utc"] = pd.to_datetime(m["runtime_utc"], errors="coerce", utc=True)
    m["forecast_time_utc"] = pd.to_datetime(m["forecast_time_utc"], errors="coerce", utc=True)
    m = m.dropna(subset=["runtime_utc", "forecast_time_utc", "model"]).copy()
    m["runtime_ny"] = m["runtime_utc"].dt.tz_convert("America/New_York")
    m["forecast_ny"] = m["forecast_time_utc"].dt.tz_convert("America/New_York")
    m["runtime_hour_utc"] = m["runtime_utc"].dt.hour.astype(int)
    m["target_date_local"] = m["forecast_ny"].dt.tz_localize(None).dt.normalize()
    m["runtime_date_local"] = m["runtime_ny"].dt.tz_localize(None).dt.normalize()
    m["forecast_hour_local"] = m["forecast_ny"].dt.hour + m["forecast_ny"].dt.minute / 60.0
    for c in ["tmp", "dpt", "sky", "wdr", "wsp", "gst", "p06", "p12", "t06", "t12", "n_x", "n_n"]:
        if c not in m.columns:
            m[c] = np.nan
        m[c] = pd.to_numeric(m[c], errors="coerce")
    if "cld_raw" not in m.columns and "cld" in m.columns:
        m["cld_raw"] = m["cld"]
    return m


def cloud_frac(row: pd.Series) -> float:
    sky = row.get("sky", np.nan)
    if pd.notna(sky):
        return float(np.clip(float(sky) / 100.0, 0.0, 1.0))
    raw = str(row.get("cld_raw", "")).strip().upper()
    mp = {"CL": 0.05, "FW": 0.20, "SC": 0.40, "BK": 0.75, "OV": 0.95}
    return float(mp.get(raw, np.nan))


def interp(h: np.ndarray, v: np.ndarray, target: float) -> float:
    mask = np.isfinite(h) & np.isfinite(v)
    if mask.sum() == 0:
        return float("nan")
    h2, v2 = h[mask], v[mask]
    idx = np.argsort(h2)
    h2, v2 = h2[idx], v2[idx]
    if len(h2) == 1:
        return float(v2[0])
    return float(np.interp(target, h2, v2))


def _build_slice_features_only(mos: pd.DataFrame, sdef: SliceDef) -> pd.DataFrame:
    d = mos[(mos["model"] == sdef.model) & (mos["runtime_hour_utc"] == sdef.runtime_hour)].copy()
    if d.empty:
        return pd.DataFrame()
    d = d[d["runtime_date_local"] == (d["target_date_local"] - pd.Timedelta(days=1))]
    if d.empty:
        return pd.DataFrame()
    d["cloud_frac"] = d.apply(cloud_frac, axis=1)

    rows: List[Dict[str, Any]] = []
    for (runtime_utc, tdate), g in d.groupby(["runtime_utc", "target_date_local"], sort=False):
        h = g["forecast_hour_local"].to_numpy(dtype=float)
        tmp = g["tmp"].to_numpy(dtype=float)
        if np.isfinite(tmp).sum() == 0:
            continue
        dpt = g["dpt"].to_numpy(dtype=float)
        cloud = g["cloud_frac"].to_numpy(dtype=float)
        wsp = g["wsp"].to_numpy(dtype=float)
        wdr = g["wdr"].to_numpy(dtype=float)
        tmax_i = int(np.nanargmax(tmp))

        tmp_09, tmp_12, tmp_15, tmp_18, tmp_21 = [interp(h, tmp, x) for x in [9, 12, 15, 18, 21]]
        dpt_09, dpt_15, dpt_21 = [interp(h, dpt, x) for x in [9, 15, 21]]

        mid = (h >= 12) & (h <= 21)
        wdr_mid = wdr[mid]
        if np.isfinite(wdr_mid).any():
            rad = np.deg2rad(wdr_mid[np.isfinite(wdr_mid)])
            wsin = float(np.mean(np.sin(rad)))
            wcos = float(np.mean(np.cos(rad)))
        else:
            wsin, wcos = float("nan"), float("nan")

        noon = pd.Timestamp(tdate).tz_localize("America/New_York") + pd.Timedelta(hours=12)
        lead = (noon.tz_convert("UTC") - runtime_utc).total_seconds() / 3600.0
        doy = pd.Timestamp(tdate).dayofyear
        rad_doy = 2 * math.pi * doy / 365.25

        rows.append(
            {
                "target_date_local": pd.Timestamp(tdate).normalize(),
                "runtime_utc": runtime_utc,
                "runtime_hour_utc": sdef.runtime_hour,
                "mos_tmax_raw": float(np.nanmax(tmp)),
                "mos_tmin_raw": float(np.nanmin(tmp)),
                "mos_dtr_raw": float(np.nanmax(tmp) - np.nanmin(tmp)),
                "mos_tmax_hour_local": float(h[tmax_i]) if np.isfinite(h[tmax_i]) else np.nan,
                "tmp_09": tmp_09,
                "tmp_12": tmp_12,
                "tmp_15": tmp_15,
                "tmp_18": tmp_18,
                "tmp_21": tmp_21,
                "heat_09_15": tmp_15 - tmp_09,
                "heat_12_18": tmp_18 - tmp_12,
                "cool_18_21": tmp_21 - tmp_18,
                "nx_high": float(np.nanmax(g["n_x"].to_numpy(dtype=float))) if np.isfinite(g["n_x"].to_numpy(dtype=float)).any() else np.nan,
                "nx_low": float(np.nanmin(g["n_n"].to_numpy(dtype=float))) if np.isfinite(g["n_n"].to_numpy(dtype=float)).any() else np.nan,
                "dpt_09": dpt_09,
                "dpt_15": dpt_15,
                "dpt_21": dpt_21,
                "dep_15": tmp_15 - dpt_15,
                "dep_21": tmp_21 - dpt_21,
                "dpt_change_09_15": dpt_15 - dpt_09,
                "cloud_mean_12_21": float(np.nanmean(cloud[mid])) if np.any(mid) else np.nan,
                "cloud_max_12_21": float(np.nanmax(cloud[mid])) if np.any(mid) else np.nan,
                "cloud_change_12_18": interp(h, cloud, 18) - interp(h, cloud, 12),
                "p06_max_day": float(np.nanmax(g["p06"].to_numpy(dtype=float))) if np.isfinite(g["p06"].to_numpy(dtype=float)).any() else np.nan,
                "p12_max_day": float(np.nanmax(g["p12"].to_numpy(dtype=float))) if np.isfinite(g["p12"].to_numpy(dtype=float)).any() else np.nan,
                "t06_max_day": float(np.nanmax(g["t06"].to_numpy(dtype=float))) if np.isfinite(g["t06"].to_numpy(dtype=float)).any() else np.nan,
                "t12_max_day": float(np.nanmax(g["t12"].to_numpy(dtype=float))) if np.isfinite(g["t12"].to_numpy(dtype=float)).any() else np.nan,
                "wsp_mean_12_21": float(np.nanmean(wsp[mid])) if np.any(mid) else np.nan,
                "wsp_max_day": float(np.nanmax(wsp)) if np.isfinite(wsp).any() else np.nan,
                "wdr_sin_mean_12_21": wsin,
                "wdr_cos_mean_12_21": wcos,
                "gst_max_day": float(np.nanmax(g["gst"].to_numpy(dtype=float))) if np.isfinite(g["gst"].to_numpy(dtype=float)).any() else np.nan,
                "doy_sin": math.sin(rad_doy),
                "doy_cos": math.cos(rad_doy),
                "lead_hours_to_local_noon": float(lead),
                "source_forecast_rows": int(len(g)),
                "source_forecast_min_utc": pd.Timestamp(g["forecast_time_utc"].min()).tz_convert("UTC"),
                "source_forecast_max_utc": pd.Timestamp(g["forecast_time_utc"].max()).tz_convert("UTC"),
            }
        )

    return pd.DataFrame(rows)


def build_slice(mos: pd.DataFrame, truth: pd.DataFrame, sdef: SliceDef) -> pd.DataFrame:
    out = _build_slice_features_only(mos, sdef)
    if out.empty:
        return out
    out = out.merge(truth, on="target_date_local", how="inner").sort_values("target_date_local").reset_index(drop=True)
    out = out[out["target_date_local"] >= pd.Timestamp(sdef.train_start)].reset_index(drop=True)
    return out


def month_starts(start: str, end: str) -> List[pd.Timestamp]:
    s = pd.Timestamp(start).replace(day=1)
    e = pd.Timestamp(end)
    out: List[pd.Timestamp] = []
    while s <= e:
        out.append(s)
        s = (s + pd.offsets.MonthBegin(1)).normalize()
    return out

def fit_point(X: pd.DataFrame, y: np.ndarray, seed: int) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(
        objective="l1",
        n_estimators=320,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=20,
        random_state=seed,
        n_jobs=-1,
    )
    m.fit(X, y)
    return m


def fit_q(X: pd.DataFrame, y: np.ndarray, seed: int, q: float) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(
        objective="quantile",
        alpha=q,
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=20,
        random_state=seed + int(q * 1000),
        n_jobs=-1,
    )
    m.fit(X, y)
    return m


def pinball(y: np.ndarray, qv: np.ndarray, a: float) -> float:
    e = y - qv
    return float(np.mean(np.maximum(a * e, (a - 1.0) * e)))


def tune_w(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    best_w, best_mae = 0.5, 1e18
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mae = float(np.mean(np.abs(y - (w * a + (1 - w) * b))))
        if mae < best_mae:
            best_w, best_mae = w, mae
    return float(best_w)


def _median_map(train_df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    med = train_df[cols].median(axis=0, skipna=True).fillna(0.0).to_dict()
    return {c: float(med.get(c, 0.0)) for c in cols}


def _fill_with_median(df: pd.DataFrame, cols: List[str], med: Dict[str, float]) -> pd.DataFrame:
    return df[cols].fillna(med)


def train_slice_bundle(df: pd.DataFrame, sdef: SliceDef, logger: logging.Logger) -> Dict[str, Any]:
    w = df.sort_values("target_date_local").reset_index(drop=True).copy()
    w["resid"] = w["y_tmax"] - w["mos_tmax_raw"]
    d = w["target_date_local"]

    dev = w[d.between(pd.Timestamp(TRAIN_DEV_START), pd.Timestamp(TRAIN_DEV_END))][["target_date_local", "y_tmax", "mos_tmax_raw"]].copy()
    for c in ["pred_point", *QUANTILE_COLUMNS]:
        dev[c] = np.nan

    for ms in month_starts(TRAIN_DEV_START, TRAIN_DEV_END):
        me = (ms + pd.offsets.MonthEnd(1)).normalize()
        msk = d.between(ms, me)
        trn = d.between(pd.Timestamp(sdef.train_start), ms - pd.Timedelta(days=1))
        if msk.sum() == 0 or trn.sum() < 200:
            continue

        med = _median_map(w.loc[trn], FEATURE_COLUMNS)
        Xtr = _fill_with_median(w.loc[trn], FEATURE_COLUMNS, med)
        Xv = _fill_with_median(w.loc[msk], FEATURE_COLUMNS, med)
        ytr = w.loc[trn, "resid"].to_numpy(dtype=float)

        p = fit_point(Xtr, ytr, SEED)
        block_pred = w.loc[msk, "mos_tmax_raw"].to_numpy(dtype=float) + p.predict(Xv)
        dev.loc[dev["target_date_local"].isin(w.loc[msk, "target_date_local"]), "pred_point"] = block_pred

        for q in QUANTILES:
            qm = fit_q(Xtr, ytr, SEED, q)
            qv = w.loc[msk, "mos_tmax_raw"].to_numpy(dtype=float) + qm.predict(Xv)
            dev.loc[dev["target_date_local"].isin(w.loc[msk, "target_date_local"]), f"q_{q:.2f}"] = qv

    trn_full = d.between(pd.Timestamp(sdef.train_start), pd.Timestamp(TRAIN_FULL_END))
    if trn_full.sum() < 400:
        raise ValueError(
            f"Insufficient full-train rows for slice={sdef.sid}. rows={int(trn_full.sum())} train_start={sdef.train_start}"
        )

    med_full = _median_map(w.loc[trn_full], FEATURE_COLUMNS)
    Xtr_full = _fill_with_median(w.loc[trn_full], FEATURE_COLUMNS, med_full)
    ytr_full = w.loc[trn_full, "resid"].to_numpy(dtype=float)

    point_model = fit_point(Xtr_full, ytr_full, SEED)
    quantile_models: Dict[float, Any] = {}
    for q in QUANTILES:
        quantile_models[q] = fit_q(Xtr_full, ytr_full, SEED, q)

    logger.info("TRAINED_SLICE station_slice=%s full_train_rows=%d", sdef.sid, int(trn_full.sum()))
    return {
        "dev": dev,
        "medians": med_full,
        "point_model": point_model,
        "quantile_models": quantile_models,
        "train_rows": int(trn_full.sum()),
    }


def train_and_write_bundle(cfg: StationConfig, logger: logging.Logger) -> None:
    blend12_bundle.train_and_write_bundle(
        station_id=cfg.station_id,
        station_zoneid=cfg.zoneid,
        mos_archive_path=cfg.mos_archive_path,
        truth_csv_path=cfg.truth_csv_path,
        bundle_dir=cfg.bundle_dir,
        logger=logger,
    )


def load_bundle(cfg: StationConfig, auto_train_if_missing: bool, logger: logging.Logger) -> StationBundle:
    manifest_path = cfg.bundle_dir / "manifest.json"
    if not manifest_path.exists():
        if not auto_train_if_missing:
            raise FileNotFoundError(f"Missing model bundle manifest: {manifest_path}")
        train_and_write_bundle(cfg, logger)
    b = blend12_bundle.load_bundle(cfg.bundle_dir)
    return StationBundle(
        point_models=b.point_models,
        quantile_models=b.quantile_models,
        medians=b.medians,
        blend_point_weight=b.blend_point_weight,
        blend_quantile_weights=b.blend_quantile_weights,
        manifest=b.manifest,
        artifact_hashes=b.artifact_hashes,
    )


# ----------------------------
# Live MOS fetch + feature compute
# ----------------------------
def _expected_runtime_utc(target_date_local: date) -> datetime:
    return (datetime(target_date_local.year, target_date_local.month, target_date_local.day, 12, 0, tzinfo=timezone.utc) - timedelta(days=1))


def _covered_target_runtimes(
    entries: List[Any],
    zoneid: str,
    target_date_local: date,
    asof_utc: datetime,
    window_start_utc: datetime,
    window_end_utc: datetime,
) -> List[datetime]:
    zone = ZoneInfo(zoneid)
    covered: set[datetime] = set()
    for entry in entries:
        runtime_utc = entry.runtime_utc
        if runtime_utc is None or runtime_utc > asof_utc:
            continue
        ftime = entry.forecast_time_utc
        if ftime is None or ftime < window_start_utc or ftime >= window_end_utc:
            continue
        if ftime.astimezone(zone).date() != target_date_local:
            continue
        covered.add(runtime_utc)
    return sorted(covered)


def fetch_live_rows_for_slice(
    station_id: str,
    zoneid: str,
    sdef: SliceDef,
    target_date_local: date,
    asof_utc: datetime,
    iem_base_url: str,
    out_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    window_start_utc, window_end_utc = mos_window_utc(target_date_local, zoneid)
    zone = ZoneInfo(zoneid)
    expected_runtime = _expected_runtime_utc(target_date_local)
    logger.info(
        "MOS_FETCH_START station=%s slice=%s model=%s target=%s asof=%s expected_runtime=%s window_start=%s window_end=%s",
        station_id,
        sdef.sid,
        sdef.model,
        target_date_local.isoformat(),
        safe_iso_utc(asof_utc),
        safe_iso_utc(expected_runtime),
        safe_iso_utc(window_start_utc),
        safe_iso_utc(window_end_utc),
    )
    payload = fetch_mos_payload(
        base_url=iem_base_url,
        station_id=station_id,
        model=sdef.model,
        start=window_start_utc - timedelta(days=2),
        end=window_end_utc + timedelta(days=1),
    )

    raw_json_path = out_dir / f"raw_mos_{station_id}_{sdef.model}.json"
    raw_json_path.write_text(payload.raw_json, encoding="utf-8")
    logger.info(
        "MOS_FETCH_DONE station=%s slice=%s model=%s payload_entries=%d raw_json=%s",
        station_id,
        sdef.sid,
        sdef.model,
        int(len(payload.entries)),
        str(raw_json_path),
    )

    available_target_runtimes = _covered_target_runtimes(
        entries=payload.entries,
        zoneid=zoneid,
        target_date_local=target_date_local,
        asof_utc=asof_utc,
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
    )
    expected_present = expected_runtime in available_target_runtimes
    latest_available = available_target_runtimes[-1] if available_target_runtimes else None
    logger.info(
        "RUNTIME_CHECK station=%s slice=%s model=%s expected=%s expected_present=%s available_count=%d latest_available=%s",
        station_id,
        sdef.sid,
        sdef.model,
        safe_iso_utc(expected_runtime),
        str(bool(expected_present)).lower(),
        int(len(available_target_runtimes)),
        safe_iso_utc(latest_available) if latest_available is not None else "NONE",
    )

    if not expected_present:
        if not available_target_runtimes:
            reason = "no_target_runtime_available_asof"
        elif expected_runtime > asof_utc:
            reason = "required_runtime_in_future_relative_to_run"
        else:
            reason = "required_runtime_not_available_yet"
        failure = RuntimeGateFailure(
            station_id=station_id,
            slice_id=sdef.sid,
            model=sdef.model,
            target_date_local=target_date_local.isoformat(),
            quote_asof_utc=safe_iso_utc(asof_utc),
            expected_runtime_utc=safe_iso_utc(expected_runtime),
            selected_runtime_utc=None,
            latest_available_runtime_utc=safe_iso_utc(latest_available) if latest_available is not None else None,
            available_target_runtimes_utc=[safe_iso_utc(rt) for rt in available_target_runtimes],
            expected_runtime_present_for_target=bool(expected_present),
            reason=reason,
        )
        raise RequiredRuntimeNotReadyError(failure)

    selected_runtime = expected_runtime
    logger.info(
        "RUNTIME_SELECTED station=%s slice=%s model=%s runtime=%s",
        station_id,
        sdef.sid,
        sdef.model,
        safe_iso_utc(selected_runtime),
    )

    rows: List[Dict[str, Any]] = []
    retrieved_at = datetime.now(timezone.utc)

    for entry in payload.entries:
        if entry.runtime_utc != selected_runtime:
            continue
        ftime = entry.forecast_time_utc
        if ftime is None or ftime < window_start_utc or ftime >= window_end_utc:
            continue
        if ftime.astimezone(zone).date() != target_date_local:
            continue

        values = entry.values

        def num(code: str) -> float:
            mv = values.get(code)
            return float(mv.numeric) if (mv is not None and mv.numeric is not None) else float("nan")

        def raw(code: str) -> str:
            mv = values.get(code)
            if mv is None or mv.raw is None:
                return ""
            return str(mv.raw)

        rows.append(
            {
                "station_id": station_id,
                "model": sdef.model,
                "runtime_utc": selected_runtime,
                "forecast_time_utc": ftime,
                "retrieved_at_utc": retrieved_at,
                "response_sha256": payload.raw_payload_hash,
                "tmp": num("tmp"),
                "dpt": num("dpt"),
                "cld_raw": raw("cld"),
                "sky": num("sky"),
                "wdr": num("wdr"),
                "wsp": num("wsp"),
                "gst": num("gst"),
                "p06": num("p06"),
                "p12": num("p12"),
                "t06": num("t06"),
                "t12": num("t12"),
                "n_x": num("n_x"),
                "n_n": num("n_n"),
            }
        )

    if not rows:
        raise ValueError(f"No target-day forecast rows for station={station_id} model={sdef.model}")

    df = pd.DataFrame(rows)
    row_csv = out_dir / f"live_mos_rows_{station_id}_{sdef.sid}.csv"
    tmp = df.copy()
    for col in ["runtime_utc", "forecast_time_utc", "retrieved_at_utc"]:
        tmp[col] = pd.to_datetime(tmp[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp.to_csv(row_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    logger.info("LIVE_MOS_FETCH station=%s slice=%s rows=%d", station_id, sdef.sid, int(len(df)))

    return {
        "slice_id": sdef.sid,
        "model": sdef.model,
        "runtime_utc": selected_runtime,
        "payload_sha256": payload.raw_payload_hash,
        "payload_json_path": str(raw_json_path),
        "row_csv_path": str(row_csv),
        "rows_df": df,
    }


def build_live_feature_row(rows_df: pd.DataFrame, sdef: SliceDef, target_date_local: date) -> Dict[str, Any]:
    mos = rows_df.copy()
    mos["runtime_utc"] = pd.to_datetime(mos["runtime_utc"], utc=True)
    mos["forecast_time_utc"] = pd.to_datetime(mos["forecast_time_utc"], utc=True)
    mos["runtime_ny"] = mos["runtime_utc"].dt.tz_convert("America/New_York")
    mos["forecast_ny"] = mos["forecast_time_utc"].dt.tz_convert("America/New_York")
    mos["runtime_hour_utc"] = mos["runtime_utc"].dt.hour.astype(int)
    mos["target_date_local"] = mos["forecast_ny"].dt.tz_localize(None).dt.normalize()
    mos["runtime_date_local"] = mos["runtime_ny"].dt.tz_localize(None).dt.normalize()
    mos["forecast_hour_local"] = mos["forecast_ny"].dt.hour + mos["forecast_ny"].dt.minute / 60.0
    mos["model"] = mos["model"].astype(str).str.upper()

    for c in ["tmp", "dpt", "sky", "wdr", "wsp", "gst", "p06", "p12", "t06", "t12", "n_x", "n_n"]:
        mos[c] = pd.to_numeric(mos[c], errors="coerce")

    feat_df = _build_slice_features_only(mos, sdef)
    feat_df = feat_df[feat_df["target_date_local"] == pd.Timestamp(target_date_local)].copy()
    if feat_df.empty:
        raise ValueError(f"Feature row missing target_date for slice={sdef.sid} target_date={target_date_local.isoformat()}")

    row = feat_df.sort_values("runtime_utc").iloc[-1].to_dict()
    row["target_date_local"] = pd.Timestamp(row["target_date_local"]).strftime("%Y-%m-%d")
    row["runtime_utc"] = pd.Timestamp(row["runtime_utc"]).tz_convert("UTC")
    row["source_forecast_min_utc"] = pd.Timestamp(row["source_forecast_min_utc"]).tz_convert("UTC")
    row["source_forecast_max_utc"] = pd.Timestamp(row["source_forecast_max_utc"]).tz_convert("UTC")
    return row


def predict_slice(feature_row: Dict[str, Any], slice_id: str, bundle: StationBundle) -> Dict[str, Any]:
    X = pd.DataFrame([{c: feature_row.get(c, np.nan) for c in FEATURE_COLUMNS}]).fillna(bundle.medians[slice_id])
    mos_tmax_raw = float(feature_row["mos_tmax_raw"])
    point = mos_tmax_raw + float(bundle.point_models[slice_id].predict(X)[0])
    qmap: Dict[float, float] = {}
    for q in QUANTILES:
        qmap[q] = mos_tmax_raw + float(bundle.quantile_models[slice_id][q].predict(X)[0])
    return {"point": float(point), "qmap": enforce_non_cross(qmap)}


def blend_station_quantiles(gfs_pred: Dict[str, Any], nam_pred: Dict[str, Any], bundle: StationBundle) -> Dict[float, float]:
    out: Dict[float, float] = {}
    for q in QUANTILES:
        w = float(bundle.blend_quantile_weights.get(q, bundle.blend_point_weight))
        out[q] = w * float(gfs_pred["qmap"][q]) + (1.0 - w) * float(nam_pred["qmap"][q])
    return enforce_non_cross(out)


def blend_station_point(gfs_pred: Dict[str, Any], nam_pred: Dict[str, Any], bundle: StationBundle) -> float:
    w = float(bundle.blend_point_weight)
    return float(w * float(gfs_pred["point"]) + (1.0 - w) * float(nam_pred["point"]))


# ----------------------------
# Market handling
# ----------------------------
def build_market_file_path(root: Path, file_prefix: str, target_date_local: date) -> Path:
    ymd = target_date_local.strftime("%Y%m%d")
    return root / f"{file_prefix}_{ymd}.csv"


def download_market_day(series: str, target_date_local: date, out_dir: Path, logger: logging.Logger) -> None:
    script = REPO_ROOT / "ingestion-service" / "scripts" / "kalshi_download_temperature_minute.py"
    out_dir.mkdir(parents=True, exist_ok=True)
    day = target_date_local.strftime("%Y-%m-%d")
    cmd = [
        sys.executable,
        str(script),
        "--series",
        series,
        "--start-date",
        day,
        "--end-date",
        day,
        "--out-dir",
        str(out_dir),
        "--skip-existing",
    ]
    completed = subprocess.run(cmd, cwd=str(script.parent), text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Downloader failed with code={completed.returncode} for series={series} day={day}")


def ensure_market_file(cfg: StationConfig, target_date_local: date, auto_download_market: bool, logger: logging.Logger) -> Path:
    market_file = build_market_file_path(cfg.market_root, cfg.file_prefix, target_date_local)
    if market_file.exists():
        return market_file
    if not auto_download_market:
        raise FileNotFoundError(f"Missing market file for {cfg.station_id} day={target_date_local}: {market_file}")
    download_market_day(cfg.series, target_date_local, cfg.market_root, logger)
    if not market_file.exists():
        raise FileNotFoundError(f"Market file still missing after download: {market_file}")
    return market_file


def load_market_rows(market_file: Path) -> pd.DataFrame:
    df = pd.read_csv(market_file)
    if "timestamp" not in df.columns:
        raise ValueError(f"Market file missing timestamp column: {market_file}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        raise ValueError(f"Market file has no timestamp rows: {market_file}")
    if df["timestamp"].duplicated().any():
        df = df.groupby("timestamp", as_index=False).last()
    return df

def build_bucket_probabilities(ctx: StationContext) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for raw in ctx.bucket_columns:
        b = parse_bucket_label(raw)
        if b is None:
            continue
        p_yes = bucket_prob(ctx.pmf, b)
        records.append(
            {
                "target_date_local": ctx.target_date_local,
                "station_id": ctx.station_id,
                "bucket_raw": raw,
                "bucket": b.canonical_label(),
                "bucket_yes_prob": float(p_yes),
                "bucket_no_prob": float(1.0 - p_yes),
            }
        )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out.sort_values(["bucket_raw"]).reset_index(drop=True)


def build_station_context(
    cfg: StationConfig,
    target_date_local: date,
    quote_asof_utc: pd.Timestamp,
    runtime_utc: pd.Timestamp,
    qmap: Dict[float, float],
    market_file: Path,
    entry_hour_z: int,
    entry_minute_z: int,
    min_entry_minutes_after_open: int,
) -> StationContext:
    day_key = target_date_local.strftime("%Y-%m-%d")
    gate_cutoff_utc = compute_gate_cutoff_utc(target_date_local, entry_hour_z, entry_minute_z)
    if runtime_utc != gate_cutoff_utc:
        raise ValueError(f"Runtime and gate mismatch for station={cfg.station_id}")

    market_df = load_market_rows(market_file)
    market_open_utc = pd.Timestamp(market_df["timestamp"].iloc[0]).tz_convert("UTC")
    effective_cutoff_utc = max(gate_cutoff_utc, market_open_utc + pd.Timedelta(minutes=min_entry_minutes_after_open))

    market_file_day = market_file_date_from_path(market_file)
    if market_file_day != day_key:
        raise ValueError(f"Market file day mismatch for station={cfg.station_id}: target_date={day_key} market_file_day={market_file_day}")

    bucket_columns = [c for c in market_df.columns if c != "timestamp" and parse_bucket_label(c) is not None]
    if not bucket_columns:
        raise ValueError(f"No parseable bucket columns found in market file: {market_file}")

    return StationContext(
        station_id=cfg.station_id,
        target_date_local=day_key,
        runtime_utc=runtime_utc,
        quote_asof_utc=quote_asof_utc,
        gate_cutoff_utc=gate_cutoff_utc,
        effective_cutoff_utc=effective_cutoff_utc,
        market_open_utc=market_open_utc,
        market_file=market_file,
        market_file_sha256=sha256_file(market_file),
        market_file_day_local=market_file_day,
        bucket_columns=bucket_columns,
        qmap=qmap,
        pmf=pmf_int_from_quantiles(qmap),
        quantiles_monotonic=is_quantiles_monotonic(qmap),
    )


# ----------------------------
# Evidence
# ----------------------------
def build_raw_input_evidence(
    station_id: str,
    slice_id: str,
    payload_sha256: str,
    payload_json_path: str,
    rows_df: pd.DataFrame,
    asof_utc: pd.Timestamp,
    expected_runtime_utc: pd.Timestamp,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _, row in rows_df.sort_values("forecast_time_utc").iterrows():
        runtime_utc = pd.Timestamp(row["runtime_utc"]).tz_convert("UTC")
        ftime_utc = pd.Timestamp(row["forecast_time_utc"]).tz_convert("UTC")
        for field in RAW_FIELD_NAMES:
            value = row.get(field, np.nan)
            if isinstance(value, str):
                value_json: Any = value
            elif pd.isna(value):
                value_json = None
            else:
                value_json = float(value)
            records.append(
                {
                    "station_id": station_id,
                    "slice_id": slice_id,
                    "runtime_utc": safe_iso_utc(runtime_utc),
                    "forecast_time_utc": safe_iso_utc(ftime_utc),
                    "field_name": field,
                    "field_value": value_json,
                    "payload_sha256": payload_sha256,
                    "payload_json_path": payload_json_path,
                    "runtime_lte_quote_asof": bool(runtime_utc <= asof_utc),
                    "runtime_equals_expected": bool(runtime_utc == expected_runtime_utc),
                    "leakage_free": bool(runtime_utc <= asof_utc and runtime_utc == expected_runtime_utc),
                }
            )
    return records


def build_feature_evidence(
    station_id: str,
    slice_id: str,
    feature_row: Dict[str, Any],
    payload_sha256: str,
    payload_json_path: str,
    asof_utc: pd.Timestamp,
    expected_runtime_utc: pd.Timestamp,
) -> List[Dict[str, Any]]:
    runtime_utc = pd.Timestamp(feature_row["runtime_utc"]).tz_convert("UTC")
    runtime_lte_asof = bool(runtime_utc <= asof_utc)
    runtime_match = bool(runtime_utc == expected_runtime_utc)
    leakage_free = bool(runtime_lte_asof and runtime_match)

    out: List[Dict[str, Any]] = []
    for f in FEATURE_COLUMNS:
        val = feature_row.get(f, np.nan)
        out.append(
            {
                "station_id": station_id,
                "slice_id": slice_id,
                "feature_name": f,
                "feature_value": None if pd.isna(val) else float(val),
                "runtime_utc": safe_iso_utc(runtime_utc),
                "quote_asof_utc": safe_iso_utc(asof_utc),
                "runtime_lte_quote_asof": runtime_lte_asof,
                "runtime_equals_expected": runtime_match,
                "source_forecast_rows": int(feature_row.get("source_forecast_rows", 0)),
                "source_forecast_min_utc": safe_iso_utc(feature_row["source_forecast_min_utc"]),
                "source_forecast_max_utc": safe_iso_utc(feature_row["source_forecast_max_utc"]),
                "payload_sha256": payload_sha256,
                "payload_json_path": payload_json_path,
                "leakage_free": leakage_free,
            }
        )
    return out


def _require_arg(name: str, value: Optional[str]) -> str:
    out = str(value or "").strip()
    if not out:
        raise ValueError(f"--{name} is required for single-station mode")
    return out


def _station_config_from_obj(payload: Dict[str, Any], live_root: Path) -> StationConfig:
    station_id = str(payload.get("station_id") or "").strip().upper()
    zoneid = str(payload.get("zoneid") or "").strip()
    series = str(payload.get("series") or "").strip().upper()
    file_prefix = str(payload.get("file_prefix") or station_id).strip().upper()
    if not station_id or not zoneid or not series:
        raise ValueError("station config requires station_id, zoneid, and series")
    bundle_raw = str(payload.get("bundle_dir") or "").strip()
    mos_raw = str(payload.get("mos_archive_path") or payload.get("mos_archive") or "").strip()
    truth_raw = str(payload.get("truth_csv_path") or payload.get("truth_csv") or "").strip()
    market_root_raw = str(payload.get("market_root") or "").strip()
    market_root = Path(market_root_raw).expanduser() if market_root_raw else (live_root / "kalshi" / station_id.lower())
    if not bundle_raw:
        raise ValueError(f"station config missing bundle_dir for station={station_id}")
    if not mos_raw:
        raise ValueError(f"station config missing mos_archive_path for station={station_id}")
    if not truth_raw:
        raise ValueError(f"station config missing truth_csv_path for station={station_id}")
    bundle_dir = Path(bundle_raw).expanduser()
    mos_archive_path = Path(mos_raw).expanduser()
    truth_csv_path = Path(truth_raw).expanduser()
    return StationConfig(
        station_id=station_id,
        zoneid=zoneid,
        series=series,
        file_prefix=file_prefix,
        market_root=market_root,
        bundle_dir=bundle_dir,
        mos_archive_path=mos_archive_path,
        truth_csv_path=truth_csv_path,
    )


def resolve_station_configs(args: argparse.Namespace, live_root: Path) -> List[StationConfig]:
    if args.station_configs_json:
        path = Path(args.station_configs_json)
        if not path.exists():
            raise FileNotFoundError(f"--station-configs-json not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            for sid, cfg in payload.items():
                if isinstance(cfg, dict):
                    row = dict(cfg)
                    row.setdefault("station_id", sid)
                    rows.append(row)
        elif isinstance(payload, list):
            rows = [x for x in payload if isinstance(x, dict)]
        else:
            raise ValueError("--station-configs-json must contain a JSON object or array")
        if not rows:
            raise ValueError("--station-configs-json did not contain any station configs")
        configs = [_station_config_from_obj(x, live_root) for x in rows]
    elif args.station_id:
        station_id = str(args.station_id).strip().upper()
        zoneid = _require_arg("station-zoneid", args.station_zoneid)
        series = _require_arg("series", args.series).upper()
        file_prefix = _require_arg("file-prefix", args.file_prefix).upper()
        bundle_dir = Path(_require_arg("bundle-dir", args.bundle_dir))
        mos_archive_path = Path(_require_arg("mos-archive", args.mos_archive))
        truth_csv_path = Path(_require_arg("truth-csv", args.truth_csv))
        market_root = Path(str(args.market_root).strip()) if args.market_root else (live_root / "kalshi" / station_id.lower())
        configs = [
            StationConfig(
                station_id=station_id,
                zoneid=zoneid,
                series=series,
                file_prefix=file_prefix,
                market_root=market_root,
                bundle_dir=bundle_dir,
                mos_archive_path=mos_archive_path,
                truth_csv_path=truth_csv_path,
            )
        ]
    else:
        market_root_knyc = Path(args.market_root_knyc) if args.market_root_knyc else (live_root / "kalshi" / "knyc")
        market_root_kmia = Path(args.market_root_kmia) if args.market_root_kmia else (live_root / "kalshi" / "kmia")
        market_root_kmdw = Path(args.market_root_kmdw) if args.market_root_kmdw else (live_root / "kalshi" / "kmdw")
        market_root_klax = Path(args.market_root_klax) if args.market_root_klax else (live_root / "kalshi" / "klax")
        configs = [
            StationConfig(
                station_id="KMIA",
                zoneid=DEFAULT_ZONE_BY_STATION["KMIA"],
                series=DEFAULT_SERIES_BY_STATION["KMIA"],
                file_prefix=DEFAULT_FILE_PREFIX_BY_STATION["KMIA"],
                market_root=market_root_kmia,
                bundle_dir=Path(args.bundle_dir_kmia),
                mos_archive_path=Path(args.mos_archive_kmia),
                truth_csv_path=Path(args.truth_csv_kmia),
            ),
            StationConfig(
                station_id="KMDW",
                zoneid=DEFAULT_ZONE_BY_STATION["KMDW"],
                series=DEFAULT_SERIES_BY_STATION["KMDW"],
                file_prefix=DEFAULT_FILE_PREFIX_BY_STATION["KMDW"],
                market_root=market_root_kmdw,
                bundle_dir=Path(args.bundle_dir_kmdw),
                mos_archive_path=Path(args.mos_archive_kmdw),
                truth_csv_path=Path(args.truth_csv_kmdw),
            ),
            StationConfig(
                station_id="KLAX",
                zoneid=DEFAULT_ZONE_BY_STATION["KLAX"],
                series=DEFAULT_SERIES_BY_STATION["KLAX"],
                file_prefix=DEFAULT_FILE_PREFIX_BY_STATION["KLAX"],
                market_root=market_root_klax,
                bundle_dir=Path(args.bundle_dir_klax),
                mos_archive_path=Path(args.mos_archive_klax),
                truth_csv_path=Path(args.truth_csv_klax),
            ),
            StationConfig(
                station_id="KNYC",
                zoneid=DEFAULT_ZONE_BY_STATION["KNYC"],
                series=DEFAULT_SERIES_BY_STATION["KNYC"],
                file_prefix=DEFAULT_FILE_PREFIX_BY_STATION["KNYC"],
                market_root=market_root_knyc,
                bundle_dir=Path(args.bundle_dir_knyc),
                mos_archive_path=Path(args.mos_archive_knyc),
                truth_csv_path=Path(args.truth_csv_knyc),
            ),
        ]

    seen: set[str] = set()
    deduped: List[StationConfig] = []
    for cfg in configs:
        if cfg.station_id in seen:
            raise ValueError(f"Duplicate station_id in config set: {cfg.station_id}")
        seen.add(cfg.station_id)
        deduped.append(cfg)
    return deduped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live leakage-safe MOS blend12 inference (station-generic).")
    p.add_argument("--target-date", required=True)
    p.add_argument("--runtime-policy", choices=["blend12_tminus1_1200z"], default="blend12_tminus1_1200z")
    p.add_argument("--entry-hour-z", type=int, default=12)
    p.add_argument("--entry-minute-z", type=int, default=0)
    p.add_argument("--min-entry-minutes-after-open", type=int, default=30)

    p.add_argument("--live-root", default=str(DEFAULT_LIVE_ROOT))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--iem-base-url", default=DEFAULT_IEM_BASE_URL)

    p.add_argument("--bundle-dir-knyc", default=str(DEFAULT_MODEL_BUNDLE_KNYC))
    p.add_argument("--bundle-dir-kmia", default=str(DEFAULT_MODEL_BUNDLE_KMIA))
    p.add_argument("--bundle-dir-kmdw", default=str(DEFAULT_MODEL_BUNDLE_KMDW))
    p.add_argument("--bundle-dir-klax", default=str(DEFAULT_MODEL_BUNDLE_KLAX))
    p.add_argument("--mos-archive-knyc", default=str(DEFAULT_MOS_ARCHIVE_KNYC))
    p.add_argument("--mos-archive-kmia", default=str(DEFAULT_MOS_ARCHIVE_KMIA))
    p.add_argument("--mos-archive-kmdw", default=str(DEFAULT_MOS_ARCHIVE_KMDW))
    p.add_argument("--mos-archive-klax", default=str(DEFAULT_MOS_ARCHIVE_KLAX))
    p.add_argument("--truth-csv-knyc", default=str(DEFAULT_TRUTH_KNYC))
    p.add_argument("--truth-csv-kmia", default=str(DEFAULT_TRUTH_KMIA))
    p.add_argument("--truth-csv-kmdw", default=str(DEFAULT_TRUTH_KMDW))
    p.add_argument("--truth-csv-klax", default=str(DEFAULT_TRUTH_KLAX))

    p.add_argument("--auto-train-bundle", dest="auto_train_bundle", action="store_true")
    p.add_argument("--no-auto-train-bundle", dest="auto_train_bundle", action="store_false")
    p.set_defaults(auto_train_bundle=False)

    p.add_argument("--market-root-knyc", default=None)
    p.add_argument("--market-root-kmia", default=None)
    p.add_argument("--market-root-kmdw", default=None)
    p.add_argument("--market-root-klax", default=None)
    p.add_argument("--station-configs-json", default=None)
    p.add_argument("--station-id", default=None, help="Single-station mode station id (e.g. KATL)")
    p.add_argument("--station-zoneid", default=None, help="Single-station mode timezone (e.g. America/New_York)")
    p.add_argument("--series", default=None, help="Single-station mode Kalshi series ticker")
    p.add_argument("--file-prefix", default=None, help="Single-station mode market file prefix (e.g. KATL)")
    p.add_argument("--bundle-dir", default=None, help="Single-station mode bundle directory")
    p.add_argument("--mos-archive", default=None, help="Single-station mode MOS archive CSV path")
    p.add_argument("--truth-csv", default=None, help="Single-station mode truth CSV path")
    p.add_argument("--market-root", default=None, help="Single-station mode Kalshi minute root")
    p.add_argument("--auto-download-market", dest="auto_download_market", action="store_true")
    p.add_argument("--no-auto-download-market", dest="auto_download_market", action="store_false")
    p.set_defaults(auto_download_market=True)

    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument(
        "--stdout-json",
        default="full",
        choices=["full", "summary", "none"],
        help="Structured JSON object emitted on stdout: full report, summary view, or disabled.",
    )
    return p.parse_args()


def ensure_args_valid(args: argparse.Namespace) -> None:
    if args.min_entry_minutes_after_open < 0:
        raise ValueError("--min-entry-minutes-after-open must be >= 0")


def main() -> int:
    args = parse_args()
    ensure_args_valid(args)

    target_date = parse_target_date(args.target_date)
    quote_asof_utc = pd.Timestamp(datetime.now(timezone.utc))
    runtime_utc = compute_runtime_utc_for_target(target_date, args.runtime_policy)

    live_root = Path(args.live_root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else (live_root / f"{run_id}_target_{target_date.strftime('%Y%m%d')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "inference.log"
    logger = configure_logger(log_file, args.log_level)
    logger.info(
        "RUN_START target_date=%s asof_utc=%s runtime_policy=%s required_runtime_utc=%s out_dir=%s",
        target_date.isoformat(),
        safe_iso_utc(quote_asof_utc),
        args.runtime_policy,
        safe_iso_utc(runtime_utc),
        str(out_dir),
    )
    if args.auto_train_bundle:
        logger.warning("AUTO_TRAIN_BUNDLE is enabled; missing bundles will be trained.")

    station_cfgs = resolve_station_configs(args, live_root)

    expected_runtime_utc = pd.Timestamp(runtime_utc)
    inference_blocks: Dict[str, Any] = {}
    leakage_station_evidence: Dict[str, Any] = {}
    feature_rollup_by_station: Dict[str, Dict[str, int]] = {}
    feature_evidence_by_station: Dict[str, List[Dict[str, Any]]] = {}
    raw_input_evidence_by_station: Dict[str, List[Dict[str, Any]]] = {}
    feature_csv_paths: Dict[str, str] = {}
    raw_input_csv_paths: Dict[str, str] = {}
    runtime_gate_failures: List[Dict[str, Any]] = []

    for cfg in station_cfgs:
        logger.info("STATION_START station=%s zone=%s bundle_dir=%s", cfg.station_id, cfg.zoneid, str(cfg.bundle_dir))
        station_out = out_dir / cfg.station_id.lower()
        station_out.mkdir(parents=True, exist_ok=True)

        bundle = load_bundle(cfg, args.auto_train_bundle, logger)
        logger.info("BUNDLE_READY station=%s", cfg.station_id)

        slice_fetch: Dict[str, Dict[str, Any]] = {}
        slice_preds: Dict[str, Dict[str, Any]] = {}
        raw_rows_all: List[Dict[str, Any]] = []
        feature_rows_all: List[Dict[str, Any]] = []
        station_has_runtime_failure = False

        for sdef in SLICE_DEFS:
            logger.info("SLICE_START station=%s slice=%s model=%s", cfg.station_id, sdef.sid, sdef.model)
            try:
                fetched = fetch_live_rows_for_slice(
                    station_id=cfg.station_id,
                    zoneid=cfg.zoneid,
                    sdef=sdef,
                    target_date_local=target_date,
                    asof_utc=quote_asof_utc.to_pydatetime(),
                    iem_base_url=args.iem_base_url,
                    out_dir=station_out,
                    logger=logger,
                )
            except RequiredRuntimeNotReadyError as exc:
                station_has_runtime_failure = True
                failure = asdict(exc.failure)
                runtime_gate_failures.append(failure)
                logger.error(
                    "RUNTIME_NOT_READY station=%s slice=%s model=%s reason=%s expected=%s selected=%s latest_available=%s",
                    failure["station_id"],
                    failure["slice_id"],
                    failure["model"],
                    failure["reason"],
                    failure["expected_runtime_utc"],
                    failure["selected_runtime_utc"],
                    failure["latest_available_runtime_utc"],
                )
                continue
            slice_fetch[sdef.sid] = fetched

            feature_row = build_live_feature_row(fetched["rows_df"], sdef, target_date)
            logger.info(
                "FEATURE_ROW_READY station=%s slice=%s runtime=%s source_rows=%d src_min=%s src_max=%s",
                cfg.station_id,
                sdef.sid,
                safe_iso_utc(feature_row["runtime_utc"]),
                int(feature_row.get("source_forecast_rows", 0)),
                safe_iso_utc(feature_row["source_forecast_min_utc"]),
                safe_iso_utc(feature_row["source_forecast_max_utc"]),
            )
            slice_preds[sdef.sid] = predict_slice(feature_row, sdef.sid, bundle)
            logger.info(
                "SLICE_PRED_DONE station=%s slice=%s point=%.4f q50=%.4f",
                cfg.station_id,
                sdef.sid,
                float(slice_preds[sdef.sid]["point"]),
                float(slice_preds[sdef.sid]["qmap"][0.50]),
            )

            raw_rows_all.extend(
                build_raw_input_evidence(
                    station_id=cfg.station_id,
                    slice_id=sdef.sid,
                    payload_sha256=fetched["payload_sha256"],
                    payload_json_path=fetched["payload_json_path"],
                    rows_df=fetched["rows_df"],
                    asof_utc=quote_asof_utc,
                    expected_runtime_utc=expected_runtime_utc,
                )
            )
            feature_rows_all.extend(
                build_feature_evidence(
                    station_id=cfg.station_id,
                    slice_id=sdef.sid,
                    feature_row=feature_row,
                    payload_sha256=fetched["payload_sha256"],
                    payload_json_path=fetched["payload_json_path"],
                    asof_utc=quote_asof_utc,
                    expected_runtime_utc=expected_runtime_utc,
                )
            )

        if station_has_runtime_failure:
            logger.info("STATION_ABORTED station=%s reason=runtime_gate_failure", cfg.station_id)
            continue

        blend_qmap = blend_station_quantiles(slice_preds["gfs_12"], slice_preds["nam_12"], bundle)
        blend_point = blend_station_point(slice_preds["gfs_12"], slice_preds["nam_12"], bundle)
        logger.info(
            "STATION_BLEND_DONE station=%s blend_point=%.4f blend_q50=%.4f",
            cfg.station_id,
            float(blend_point),
            float(blend_qmap[0.50]),
        )

        raw_df = pd.DataFrame(raw_rows_all)
        raw_path = station_out / f"raw_input_evidence_{cfg.station_id}.csv"
        raw_df.to_csv(raw_path, index=False, quoting=csv.QUOTE_MINIMAL)
        raw_input_csv_paths[cfg.station_id] = str(raw_path)

        feat_df = pd.DataFrame(feature_rows_all)
        feat_path = station_out / f"feature_evidence_{cfg.station_id}.csv"
        feat_df.to_csv(feat_path, index=False, quoting=csv.QUOTE_MINIMAL)
        feature_csv_paths[cfg.station_id] = str(feat_path)

        raw_input_evidence_by_station[cfg.station_id] = raw_rows_all
        feature_evidence_by_station[cfg.station_id] = feature_rows_all

        total_features = int(len(feat_df))
        leakage_free_features = int(feat_df["leakage_free"].fillna(False).sum()) if total_features else 0
        feature_rollup_by_station[cfg.station_id] = {
            "total_features": total_features,
            "leakage_free_features": leakage_free_features,
            "non_leakage_free_features": int(total_features - leakage_free_features),
        }
        quantiles_monotonic = is_quantiles_monotonic(blend_qmap)
        runtime_lte_quote_asof = bool(expected_runtime_utc <= quote_asof_utc)
        runtime_equals_expected_policy = bool(expected_runtime_utc == compute_runtime_utc_for_target(target_date, args.runtime_policy))

        inference_blocks[cfg.station_id] = {
            "station_id": cfg.station_id,
            "target_date_local": target_date.isoformat(),
            "runtime_utc": safe_iso_utc(expected_runtime_utc),
            "prediction_point_tmax_f": float(blend_point),
            "quantiles": {f"q_{q:.2f}": float(blend_qmap[q]) for q in QUANTILES},
        }

        leakage_station_evidence[cfg.station_id] = {
            "station_id": cfg.station_id,
            "target_date_local": target_date.isoformat(),
            "runtime_utc": safe_iso_utc(expected_runtime_utc),
            "quote_asof_utc": safe_iso_utc(quote_asof_utc),
            "runtime_lte_quote_asof": runtime_lte_quote_asof,
            "runtime_expected_from_policy_utc": safe_iso_utc(compute_runtime_utc_for_target(target_date, args.runtime_policy)),
            "runtime_equals_expected_policy_runtime": runtime_equals_expected_policy,
            "inference_quantiles_monotonic": quantiles_monotonic,
            "bundle_dir": str(cfg.bundle_dir),
            "bundle_manifest": bundle.manifest,
            "bundle_artifact_hashes": bundle.artifact_hashes,
            "feature_level_rollup": feature_rollup_by_station[cfg.station_id],
        }
        logger.info("STATION_DONE station=%s", cfg.station_id)

    if runtime_gate_failures:
        lagging_slices = [
            f
            for f in runtime_gate_failures
            if f["reason"] in {"required_runtime_not_available_yet", "no_target_runtime_available_asof", "required_runtime_in_future_relative_to_run"}
        ]
        error_code = "required_runtime_not_ready"
        message = "Required T-1 12:00Z runtime is missing for one or more station/model slices. No fallback was used; inference aborted."
        failure_report = {
            "script": "tools/live/mos_quantile_live_inference.py",
            "error": error_code,
            "message": message,
            "target_date_local": target_date.isoformat(),
            "quote_asof_utc": safe_iso_utc(quote_asof_utc),
            "required_runtime_policy": args.runtime_policy,
            "required_runtime_utc": safe_iso_utc(expected_runtime_utc),
            "lagging_slices": lagging_slices,
            "all_runtime_gate_failures": runtime_gate_failures,
        }
        failure_path = out_dir / "runtime_gate_failure.json"
        failure_path.write_text(json.dumps(failure_report, indent=2), encoding="utf-8")
        logger.error(
            "RUNTIME_GATE_FAILURE report=%s lagging_count=%d",
            str(failure_path),
            int(len(lagging_slices)),
        )
        if args.stdout_json != "none":
            print(json.dumps(failure_report, indent=2))
        return 2

    total_feature_rows = int(sum(v["total_features"] for v in feature_rollup_by_station.values()))
    total_leakage_free_feature_rows = int(sum(v["leakage_free_features"] for v in feature_rollup_by_station.values()))
    total_non_leakage_free = int(sum(v["non_leakage_free_features"] for v in feature_rollup_by_station.values()))

    guardrail_counters = {
        "runtime_after_quote_asof": int(sum(1 for v in leakage_station_evidence.values() if not v["runtime_lte_quote_asof"])),
        "runtime_policy_mismatch": int(sum(1 for v in leakage_station_evidence.values() if not v["runtime_equals_expected_policy_runtime"])),
        "prediction_quantiles_monotonic_violation": int(
            sum(1 for v in leakage_station_evidence.values() if not v["inference_quantiles_monotonic"])
        ),
        "non_leakage_free_feature_rows": total_non_leakage_free,
        "truth_columns_used_for_inference": 0,
    }
    passes_all_guardrails = all(v == 0 for v in guardrail_counters.values())

    leakage_proof = {
        "proof_standard": "Inference uses only live-fetched MOS payload rows and derived features where runtime_utc is fixed by policy to T-1 12:00Z and runtime_utc <= quote_asof_utc.",
        "global_guardrail_counters": guardrail_counters,
        "per_station_evidence": leakage_station_evidence,
        "feature_level_rollup_global": {
            "total_features": total_feature_rows,
            "leakage_free_features": total_leakage_free_feature_rows,
            "non_leakage_free_features": total_non_leakage_free,
            "all_feature_rows_leakage_free": bool(total_non_leakage_free == 0),
        },
        "feature_level_rollup_by_station": feature_rollup_by_station,
        "feature_level_evidence_by_station": feature_evidence_by_station,
        "raw_input_evidence_by_station": raw_input_evidence_by_station,
        "passes_all_guardrails": bool(passes_all_guardrails),
    }

    report = {
        "script": "tools/live/mos_quantile_live_inference.py",
        "mode": "inference_only_quantiles",
        "run_id": run_id,
        "target_date_local": target_date.isoformat(),
        "quote_asof_utc": safe_iso_utc(quote_asof_utc),
        "inference_by_station": {sid: inference_blocks[sid] for sid in sorted(inference_blocks)},
        "leakage_proof": leakage_proof,
        "artifacts": {
            "feature_evidence_csv_by_station": feature_csv_paths,
            "raw_input_evidence_csv_by_station": raw_input_csv_paths,
            "log_file": str(log_file),
        },
    }
    for sid in sorted(inference_blocks):
        report[f"inference_{sid.lower()}"] = inference_blocks[sid]

    report_path = out_dir / "inference_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("REPORT_WRITTEN path=%s", str(report_path))

    summary_payload = {
        "target_date_local": target_date.isoformat(),
        "inference_by_station": report["inference_by_station"],
        "leakage_proof_summary": {
            "passes_all_guardrails": report["leakage_proof"]["passes_all_guardrails"],
            "global_guardrail_counters": report["leakage_proof"]["global_guardrail_counters"],
            "feature_level_rollup_global": report["leakage_proof"]["feature_level_rollup_global"],
        },
        "report_path": str(report_path),
    }
    if args.stdout_json == "full":
        print(json.dumps(report, indent=2))
    elif args.stdout_json == "summary":
        print(json.dumps(summary_payload, indent=2))

    logger.info("RUN_DONE target_date=%s", target_date.isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
