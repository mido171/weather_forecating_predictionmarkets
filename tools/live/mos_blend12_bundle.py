from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
QUANTILE_COLUMNS = [f"q_{q:.2f}" for q in QUANTILES]

# Keep this feature contract exactly aligned with live inference.
FEATURE_COLUMNS = [
    "mos_tmax_raw",
    "mos_tmin_raw",
    "mos_dtr_raw",
    "mos_tmax_hour_local",
    "tmp_09",
    "tmp_12",
    "tmp_15",
    "tmp_18",
    "tmp_21",
    "heat_09_15",
    "heat_12_18",
    "cool_18_21",
    "nx_high",
    "nx_low",
    "dpt_09",
    "dpt_15",
    "dpt_21",
    "dep_15",
    "dep_21",
    "dpt_change_09_15",
    "cloud_mean_12_21",
    "cloud_max_12_21",
    "cloud_change_12_18",
    "p06_max_day",
    "p12_max_day",
    "t06_max_day",
    "t12_max_day",
    "wsp_mean_12_21",
    "wsp_max_day",
    "wdr_sin_mean_12_21",
    "wdr_cos_mean_12_21",
    "gst_max_day",
    "doy_sin",
    "doy_cos",
    "runtime_hour_utc",
    "lead_hours_to_local_noon",
]

TRAIN_DEV_START = "2022-01-01"
TRAIN_DEV_END = "2023-12-31"
TRAIN_FULL_END = "2023-12-31"
SEED = 42


@dataclass(frozen=True)
class SliceDef:
    sid: str
    model: str
    runtime_hour: int
    train_start: str


# Live strategy is locked to blend_12 = gfs_12 + nam_12.
SLICE_DEFS = [
    SliceDef("gfs_12", "GFS", 12, "2009-01-01"),
    SliceDef("nam_12", "NAM", 12, "2009-01-01"),
]


@dataclass
class StationBundle:
    point_models: Dict[str, Any]
    quantile_models: Dict[str, Dict[float, Any]]
    medians: Dict[str, Dict[str, float]]
    blend_point_weight: float
    blend_quantile_weights: Dict[float, float]
    manifest: Dict[str, Any]
    artifact_hashes: Dict[str, str]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _cloud_frac(row: pd.Series) -> float:
    sky = row.get("sky", np.nan)
    if pd.notna(sky):
        return float(np.clip(float(sky) / 100.0, 0.0, 1.0))
    raw = str(row.get("cld_raw", "")).strip().upper()
    mp = {"CL": 0.05, "FW": 0.20, "SC": 0.40, "BK": 0.75, "OV": 0.95}
    return float(mp.get(raw, np.nan))


def _interp(h: np.ndarray, v: np.ndarray, target: float) -> float:
    mask = np.isfinite(h) & np.isfinite(v)
    if mask.sum() == 0:
        return float("nan")
    h2, v2 = h[mask], v[mask]
    idx = np.argsort(h2)
    h2, v2 = h2[idx], v2[idx]
    if len(h2) == 1:
        return float(v2[0])
    return float(np.interp(target, h2, v2))


def load_truth(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path).rename(columns={"date": "target_date_local", "settled_tmax": "y_tmax"})
    t["target_date_local"] = pd.to_datetime(t["target_date_local"], errors="coerce").dt.normalize()
    t["y_tmax"] = pd.to_numeric(t["y_tmax"], errors="coerce")
    t = (
        t.dropna(subset=["target_date_local", "y_tmax"])[["target_date_local", "y_tmax"]]
        .drop_duplicates("target_date_local")
        .sort_values("target_date_local")
        .reset_index(drop=True)
    )
    return t


def load_mos(path: Path, station_zoneid: str) -> pd.DataFrame:
    m = pd.read_csv(path)
    m["runtime_utc"] = pd.to_datetime(m["runtime_utc"], errors="coerce", utc=True)
    m["forecast_time_utc"] = pd.to_datetime(m["forecast_time_utc"], errors="coerce", utc=True)
    m = m.dropna(subset=["runtime_utc", "forecast_time_utc", "model"]).copy()
    m["runtime_local"] = m["runtime_utc"].dt.tz_convert(station_zoneid)
    m["forecast_local"] = m["forecast_time_utc"].dt.tz_convert(station_zoneid)
    m["runtime_hour_utc"] = m["runtime_utc"].dt.hour.astype(int)
    m["target_date_local"] = m["forecast_local"].dt.tz_localize(None).dt.normalize()
    m["runtime_date_local"] = m["runtime_local"].dt.tz_localize(None).dt.normalize()
    m["forecast_hour_local"] = m["forecast_local"].dt.hour + m["forecast_local"].dt.minute / 60.0
    m["model"] = m["model"].astype(str).str.upper()
    for c in ["tmp", "dpt", "sky", "wdr", "wsp", "gst", "p06", "p12", "t06", "t12", "n_x", "n_n"]:
        if c not in m.columns:
            m[c] = np.nan
        m[c] = pd.to_numeric(m[c], errors="coerce")
    return m


def _build_slice_features_only(mos: pd.DataFrame, sdef: SliceDef, station_zoneid: str) -> pd.DataFrame:
    d = mos[(mos["model"] == sdef.model) & (mos["runtime_hour_utc"] == sdef.runtime_hour)].copy()
    if d.empty:
        return d
    d = d[d["runtime_date_local"] == (d["target_date_local"] - pd.Timedelta(days=1))]
    d["cloud_frac"] = d.apply(_cloud_frac, axis=1)

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
        tmp_09, tmp_12, tmp_15, tmp_18, tmp_21 = [_interp(h, tmp, x) for x in [9, 12, 15, 18, 21]]
        dpt_09, dpt_15, dpt_21 = [_interp(h, dpt, x) for x in [9, 15, 21]]
        mid = (h >= 12) & (h <= 21)
        wdr_mid = wdr[mid]
        if np.isfinite(wdr_mid).any():
            rad = np.deg2rad(wdr_mid[np.isfinite(wdr_mid)])
            wsin = float(np.mean(np.sin(rad)))
            wcos = float(np.mean(np.cos(rad)))
        else:
            wsin, wcos = float("nan"), float("nan")
        noon = pd.Timestamp(tdate).tz_localize(station_zoneid) + pd.Timedelta(hours=12)
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
                "nx_high": float(np.nanmax(g["n_x"].to_numpy(dtype=float)))
                if np.isfinite(g["n_x"].to_numpy(dtype=float)).any()
                else np.nan,
                "nx_low": float(np.nanmin(g["n_n"].to_numpy(dtype=float)))
                if np.isfinite(g["n_n"].to_numpy(dtype=float)).any()
                else np.nan,
                "dpt_09": dpt_09,
                "dpt_15": dpt_15,
                "dpt_21": dpt_21,
                "dep_15": tmp_15 - dpt_15,
                "dep_21": tmp_21 - dpt_21,
                "dpt_change_09_15": dpt_15 - dpt_09,
                "cloud_mean_12_21": float(np.nanmean(cloud[mid])) if np.any(mid) else np.nan,
                "cloud_max_12_21": float(np.nanmax(cloud[mid])) if np.any(mid) else np.nan,
                "cloud_change_12_18": _interp(h, cloud, 18) - _interp(h, cloud, 12),
                "p06_max_day": float(np.nanmax(g["p06"].to_numpy(dtype=float)))
                if np.isfinite(g["p06"].to_numpy(dtype=float)).any()
                else np.nan,
                "p12_max_day": float(np.nanmax(g["p12"].to_numpy(dtype=float)))
                if np.isfinite(g["p12"].to_numpy(dtype=float)).any()
                else np.nan,
                "t06_max_day": float(np.nanmax(g["t06"].to_numpy(dtype=float)))
                if np.isfinite(g["t06"].to_numpy(dtype=float)).any()
                else np.nan,
                "t12_max_day": float(np.nanmax(g["t12"].to_numpy(dtype=float)))
                if np.isfinite(g["t12"].to_numpy(dtype=float)).any()
                else np.nan,
                "wsp_mean_12_21": float(np.nanmean(wsp[mid])) if np.any(mid) else np.nan,
                "wsp_max_day": float(np.nanmax(wsp)) if np.isfinite(wsp).any() else np.nan,
                "wdr_sin_mean_12_21": wsin,
                "wdr_cos_mean_12_21": wcos,
                "gst_max_day": float(np.nanmax(g["gst"].to_numpy(dtype=float)))
                if np.isfinite(g["gst"].to_numpy(dtype=float)).any()
                else np.nan,
                "doy_sin": math.sin(rad_doy),
                "doy_cos": math.cos(rad_doy),
                "lead_hours_to_local_noon": float(lead),
            }
        )

    return pd.DataFrame(rows)


def build_slice(mos: pd.DataFrame, truth: pd.DataFrame, sdef: SliceDef, station_zoneid: str) -> pd.DataFrame:
    out = _build_slice_features_only(mos, sdef, station_zoneid)
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

    dev = w[d.between(pd.Timestamp(TRAIN_DEV_START), pd.Timestamp(TRAIN_DEV_END))][
        ["target_date_local", "y_tmax", "mos_tmax_raw"]
    ].copy()
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


def train_and_write_bundle(
    *,
    station_id: str,
    station_zoneid: str,
    mos_archive_path: Path,
    truth_csv_path: Path,
    bundle_dir: Path,
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    log = logger or logging.getLogger("mos_blend12_bundle")
    if not mos_archive_path.exists():
        raise FileNotFoundError(f"Missing MOS archive CSV for bundle training: {mos_archive_path}")
    if not truth_csv_path.exists():
        raise FileNotFoundError(f"Missing truth CSV for bundle training: {truth_csv_path}")

    log.info("TRAIN_BUNDLE_START station=%s out=%s", station_id, str(bundle_dir))
    truth = load_truth(truth_csv_path)
    mos = load_mos(mos_archive_path, station_zoneid)

    per_slice: Dict[str, Dict[str, Any]] = {}
    for sdef in SLICE_DEFS:
        sdf = build_slice(mos, truth, sdef, station_zoneid)
        if sdf.empty:
            raise ValueError(f"No rows after slice build for station={station_id} slice={sdef.sid}")
        per_slice[sdef.sid] = train_slice_bundle(sdf, sdef, log)

    gd = per_slice["gfs_12"]["dev"]
    nd = per_slice["nam_12"]["dev"]
    dev = gd.merge(nd, on=["target_date_local", "y_tmax"], suffixes=("_g", "_n"))
    if dev.empty:
        raise ValueError(f"Empty merged dev set for station={station_id} while tuning blend weights")

    yv = dev["y_tmax"].to_numpy(dtype=float)
    blend_point_weight = tune_w(yv, dev["pred_point_g"].to_numpy(dtype=float), dev["pred_point_n"].to_numpy(dtype=float))

    blend_quantile_weights: Dict[float, float] = {}
    for q in QUANTILES:
        c0, c1 = f"q_{q:.2f}_g", f"q_{q:.2f}_n"
        best_w, best_pin = blend_point_weight, 1e18
        for wv in [0.0, 0.25, 0.5, 0.75, 1.0]:
            blended = wv * dev[c0].to_numpy(dtype=float) + (1 - wv) * dev[c1].to_numpy(dtype=float)
            pb = pinball(yv, blended, q)
            if pb < best_pin:
                best_w, best_pin = wv, pb
        blend_quantile_weights[q] = float(best_w)

    bundle_dir.mkdir(parents=True, exist_ok=True)
    artifact_hashes: Dict[str, str] = {}
    for sid in ["gfs_12", "nam_12"]:
        sdir = bundle_dir / sid
        sdir.mkdir(parents=True, exist_ok=True)

        point_path = sdir / "point_model.joblib"
        joblib.dump(per_slice[sid]["point_model"], point_path)
        artifact_hashes[str(point_path)] = sha256_file(point_path)

        med_path = sdir / "feature_medians.json"
        med_path.write_text(json.dumps(per_slice[sid]["medians"], indent=2, sort_keys=True), encoding="utf-8")
        artifact_hashes[str(med_path)] = sha256_file(med_path)

        for q in QUANTILES:
            q_path = sdir / f"q_{q:.2f}.joblib"
            joblib.dump(per_slice[sid]["quantile_models"][q], q_path)
            artifact_hashes[str(q_path)] = sha256_file(q_path)

    blend_weights_path = bundle_dir / "blend_weights.json"
    blend_weights_path.write_text(
        json.dumps(
            {
                "point": blend_point_weight,
                "quantiles": {f"{q:.2f}": blend_quantile_weights[q] for q in QUANTILES},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifact_hashes[str(blend_weights_path)] = sha256_file(blend_weights_path)

    manifest = {
        "schema": "mos_blend12_live_bundle_v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "station_id": station_id,
        "station_zoneid": station_zoneid,
        "inputs": {
            "mos_archive_path": str(mos_archive_path),
            "mos_archive_sha256": sha256_file(mos_archive_path),
            "truth_csv_path": str(truth_csv_path),
            "truth_csv_sha256": sha256_file(truth_csv_path),
        },
        "blend_weights": {
            "point": blend_point_weight,
            "quantiles": {f"{q:.2f}": blend_quantile_weights[q] for q in QUANTILES},
        },
        "artifacts_sha256": artifact_hashes,
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_bundle(bundle_dir: Path) -> StationBundle:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing model bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    point_models: Dict[str, Any] = {}
    quantile_models: Dict[str, Dict[float, Any]] = {}
    medians: Dict[str, Dict[str, float]] = {}
    artifact_hashes: Dict[str, str] = {}

    for sid in ["gfs_12", "nam_12"]:
        sdir = bundle_dir / sid
        point_path = sdir / "point_model.joblib"
        point_models[sid] = joblib.load(point_path)
        artifact_hashes[str(point_path)] = sha256_file(point_path)

        med_path = sdir / "feature_medians.json"
        medians[sid] = {k: float(v) for k, v in json.loads(med_path.read_text(encoding="utf-8")).items()}
        artifact_hashes[str(med_path)] = sha256_file(med_path)

        qmods: Dict[float, Any] = {}
        for q in QUANTILES:
            q_path = sdir / f"q_{q:.2f}.joblib"
            qmods[q] = joblib.load(q_path)
            artifact_hashes[str(q_path)] = sha256_file(q_path)
        quantile_models[sid] = qmods

    bw = json.loads((bundle_dir / "blend_weights.json").read_text(encoding="utf-8"))
    blend_point = float(bw.get("point", 0.5))
    bq_raw = bw.get("quantiles", {}) or {}
    blend_quantiles = {q: float(bq_raw.get(f"{q:.2f}", blend_point)) for q in QUANTILES}

    return StationBundle(
        point_models=point_models,
        quantile_models=quantile_models,
        medians=medians,
        blend_point_weight=blend_point,
        blend_quantile_weights=blend_quantiles,
        manifest=manifest,
        artifact_hashes=artifact_hashes,
    )
