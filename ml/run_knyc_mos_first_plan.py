from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import kstest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


LOGGER = logging.getLogger("knyc_mos")
EPS = 1e-9
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
K_GRID = [16, 32, 64]
ALPHA_GRID = [0.25, 0.50, 0.75]


@dataclass(frozen=True)
class SliceDef:
    sid: str
    model: str
    runtime_hour: int
    train_start: str


DEFAULT_SLICE_DEFS = [
    SliceDef("gfs_00", "GFS", 0, "2009-01-01"),
    SliceDef("nam_00", "NAM", 0, "2009-01-01"),
    SliceDef("gfs_12", "GFS", 12, "2009-01-01"),
    SliceDef("nam_12", "NAM", 12, "2009-01-01"),
    SliceDef("gfs_06", "GFS", 6, "2004-01-01"),
    SliceDef("gfs_18", "GFS", 18, "2004-01-01"),
]

COMMON_SLICE_IDS_BY_HOUR = {
    0: ("gfs_00", "nam_00"),
    12: ("gfs_12", "nam_12"),
}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_truth(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path).rename(columns={"date": "target_date_local", "settled_tmax": "y_tmax"})
    t["target_date_local"] = pd.to_datetime(t["target_date_local"], errors="coerce").dt.normalize()
    t["y_tmax"] = pd.to_numeric(t["y_tmax"], errors="coerce")
    t = t.dropna(subset=["target_date_local", "y_tmax"])[["target_date_local", "y_tmax"]].drop_duplicates("target_date_local")
    return t


def load_mos(path: Path, station_zoneid: str) -> pd.DataFrame:
    m = pd.read_csv(path)
    m["runtime_utc"] = pd.to_datetime(m["runtime_utc"], errors="coerce", utc=True)
    m["forecast_time_utc"] = pd.to_datetime(m["forecast_time_utc"], errors="coerce", utc=True)
    m = m.dropna(subset=["runtime_utc", "forecast_time_utc", "model"])
    m["runtime_local"] = m["runtime_utc"].dt.tz_convert(station_zoneid)
    m["forecast_local"] = m["forecast_time_utc"].dt.tz_convert(station_zoneid)
    m["runtime_hour_utc"] = m["runtime_utc"].dt.hour.astype(int)
    m["target_date_local"] = m["forecast_local"].dt.tz_localize(None).dt.normalize()
    m["runtime_date_local"] = m["runtime_local"].dt.tz_localize(None).dt.normalize()
    m["forecast_hour_local"] = m["forecast_local"].dt.hour + m["forecast_local"].dt.minute / 60.0
    for c in ["tmp", "dpt", "sky", "wdr", "wsp", "gst", "p06", "p12", "t06", "t12", "n_x", "n_n"]:
        if c not in m.columns:
            m[c] = np.nan
        m[c] = pd.to_numeric(m[c], errors="coerce")
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


def resolve_slice_defs(
    mos: pd.DataFrame,
    *,
    common_train_start_policy: str,
    common_train_start: str | None,
) -> list[SliceDef]:
    out = [SliceDef(s.sid, s.model, s.runtime_hour, s.train_start) for s in DEFAULT_SLICE_DEFS]
    if common_train_start_policy == "legacy":
        if common_train_start:
            out = [SliceDef(s.sid, s.model, s.runtime_hour, common_train_start if s.sid in {x for pair in COMMON_SLICE_IDS_BY_HOUR.values() for x in pair} else s.train_start) for s in out]
        return out
    if common_train_start_policy == "manual":
        if not common_train_start:
            raise ValueError("--common-train-start is required when --common-train-start-policy=manual")
        return [
            SliceDef(
                s.sid,
                s.model,
                s.runtime_hour,
                common_train_start if s.sid in {x for pair in COMMON_SLICE_IDS_BY_HOUR.values() for x in pair} else s.train_start,
            )
            for s in out
        ]
    if common_train_start_policy != "earliest-common":
        raise ValueError(f"Unsupported common train-start policy: {common_train_start_policy}")

    common_start_by_sid: dict[str, str] = {}
    for runtime_hour, pair in COMMON_SLICE_IDS_BY_HOUR.items():
        starts: list[pd.Timestamp] = []
        for sid in pair:
            sdef = next(s for s in out if s.sid == sid)
            d = mos[(mos["model"] == sdef.model) & (mos["runtime_hour_utc"] == sdef.runtime_hour)].copy()
            d = d[d["runtime_date_local"] == (d["target_date_local"] - pd.Timedelta(days=1))]
            if d.empty:
                raise ValueError(f"No eligible rows found while resolving earliest-common train start for slice={sid}")
            starts.append(pd.Timestamp(d["target_date_local"].min()).normalize())
        common_start_ts = max(starts)
        if common_train_start:
            common_start_ts = max(common_start_ts, pd.Timestamp(common_train_start).normalize())
        for sid in pair:
            common_start_by_sid[sid] = common_start_ts.strftime("%Y-%m-%d")
    return [
        SliceDef(s.sid, s.model, s.runtime_hour, common_start_by_sid.get(s.sid, s.train_start))
        for s in out
    ]


def build_slice(mos: pd.DataFrame, truth: pd.DataFrame, s: SliceDef, station_zoneid: str) -> pd.DataFrame:
    d = mos[(mos["model"] == s.model) & (mos["runtime_hour_utc"] == s.runtime_hour)].copy()
    if d.empty:
        return d
    d = d[d["runtime_date_local"] == (d["target_date_local"] - pd.Timedelta(days=1))]
    d["cloud_frac"] = d.apply(cloud_frac, axis=1)
    rows = []
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
        noon = pd.Timestamp(tdate).tz_localize(station_zoneid) + pd.Timedelta(hours=12)
        lead = (noon.tz_convert("UTC") - runtime_utc).total_seconds() / 3600.0
        doy = pd.Timestamp(tdate).dayofyear
        rad = 2 * math.pi * doy / 365.25
        rows.append(
            {
                "target_date_local": pd.Timestamp(tdate).normalize(),
                "runtime_utc": runtime_utc,
                "runtime_hour_utc": s.runtime_hour,
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
                "doy_sin": math.sin(rad),
                "doy_cos": math.cos(rad),
                "lead_hours_to_local_noon": float(lead),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(truth, on="target_date_local", how="inner").sort_values("target_date_local").reset_index(drop=True)
    out = out[out["target_date_local"] >= pd.Timestamp(s.train_start)].reset_index(drop=True)
    return out


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    s = pd.Timestamp(start).replace(day=1)
    e = pd.Timestamp(end)
    out = []
    while s <= e:
        out.append(s)
        s = (s + pd.offsets.MonthBegin(1)).normalize()
    return out


def fill_train_median(train_df: pd.DataFrame, pred_df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = train_df[cols].median(axis=0, skipna=True).fillna(0.0).to_dict()
    return train_df[cols].fillna(med), pred_df[cols].fillna(med)


def fit_point(X: pd.DataFrame, y: np.ndarray, seed: int) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(objective="l1", n_estimators=320, learning_rate=0.04, num_leaves=31, subsample=0.85, colsample_bytree=0.85, min_child_samples=20, random_state=seed, n_jobs=-1)
    m.fit(X, y)
    return m


def fit_q(X: pd.DataFrame, y: np.ndarray, seed: int, q: float) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=300, learning_rate=0.04, num_leaves=31, subsample=0.85, colsample_bytree=0.85, min_child_samples=20, random_state=seed + int(q * 1000), n_jobs=-1)
    m.fit(X, y)
    return m


def non_cross(qmap: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    if not qmap:
        return qmap
    t = sorted(qmap)
    m = np.column_stack([qmap[k] for k in t])
    m = np.maximum.accumulate(m, axis=1)
    return {k: m[:, i] for i, k in enumerate(t)}


def pinball(y: np.ndarray, q: np.ndarray, a: float) -> float:
    e = y - q
    return float(np.mean(np.maximum(a * e, (a - 1.0) * e)))


def reg(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = p - y
    ae = np.abs(e)
    return {"mae": float(np.mean(ae)), "rmse": float(np.sqrt(np.mean(e**2))), "bias": float(np.mean(e)), "median_abs_error": float(np.median(ae)), "p90_abs_error": float(np.quantile(ae, 0.9))}


def interval_metrics(y: np.ndarray, q: dict[float, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for lo, hi, name in [(0.25, 0.75, "50"), (0.10, 0.90, "80"), (0.05, 0.95, "90")]:
        if lo in q and hi in q:
            l, h = q[lo], q[hi]
            out[f"coverage_{name}"] = float(np.mean((y >= l) & (y <= h)))
            out[f"width_{name}_avg"] = float(np.mean(h - l))
    return out


def pit(y: np.ndarray, q: dict[float, np.ndarray]) -> tuple[float, float]:
    if not q:
        return float("nan"), float("nan")
    t = sorted(q)
    m = np.column_stack([q[k] for k in t])
    m = np.maximum.accumulate(m, axis=1)
    pv = np.array([np.interp(y[i], m[i], np.array(t), left=0.0, right=1.0) for i in range(len(y))], dtype=float)
    pv = np.clip(pv, 0.0, 1.0)
    ks = kstest(pv, "uniform")
    return float(ks.statistic), float(ks.pvalue)


def prob_ge(q: dict[float, np.ndarray], thr: float) -> np.ndarray:
    t = sorted(q)
    m = np.column_stack([q[k] for k in t])
    m = np.maximum.accumulate(m, axis=1)
    cdf = np.array([np.interp(thr, m[i], np.array(t), left=0.0, right=1.0) for i in range(m.shape[0])], dtype=float)
    return np.clip(1.0 - cdf, EPS, 1.0 - EPS)


def ece10(y: np.ndarray, p: np.ndarray) -> float:
    df = pd.DataFrame({"y": y, "p": p})
    try:
        df["bin"] = pd.qcut(df["p"], q=10, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["p"], bins=10)
    g = df.groupby("bin", observed=False).agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean")).reset_index()
    g = g[g["n"] > 0]
    if g.empty:
        return float("nan")
    return float((((g["obs"] - g["pred"]).abs()) * g["n"]).sum() / g["n"].sum())


def cal_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(p, EPS, 1.0 - EPS)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(max_iter=400)
        lr.fit(x, y.astype(int))
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        return float("nan"), float("nan")


def eval_dist(y: np.ndarray, point: np.ndarray, qmap: dict[float, np.ndarray], train_y: np.ndarray) -> dict[str, float]:
    out = reg(y, point)
    if len(train_y):
        out["skill_vs_train_median_mae"] = float(1.0 - out["mae"] / (np.mean(np.abs(y - np.median(train_y))) + EPS))
    if qmap:
        qmap = non_cross(qmap)
        pb = [pinball(y, qmap[q], q) for q in sorted(qmap)]
        for q, v in zip(sorted(qmap), pb):
            out[f"pinball_{int(q*100):02d}"] = float(v)
        out["avg_pinball"] = float(np.mean(pb))
        out.update(interval_metrics(y, qmap))
        ks_stat, ks_p = pit(y, qmap)
        out["pit_ks_stat"] = ks_stat
        out["pit_ks_pvalue"] = ks_p
        thresholds = [70, 75, 80, 85, 90, 95]
        if len(train_y):
            thresholds += [int(round(np.quantile(train_y, 0.5))), int(round(np.quantile(train_y, 0.75))), int(round(np.quantile(train_y, 0.9)))]
        rows = []
        for thr in sorted(set(thresholds)):
            p = prob_ge(qmap, float(thr))
            yb = (y >= thr).astype(int)
            rows.append(
                {
                    "brier": float(np.mean((p - yb) ** 2)),
                    "logloss": float(log_loss(yb, p, labels=[0, 1])),
                    "ece10": ece10(yb, p),
                    "slope": cal_slope_intercept(yb, p)[0],
                    "intercept": cal_slope_intercept(yb, p)[1],
                }
            )
        d = pd.DataFrame(rows)
        out["threshold_brier_avg"] = float(d["brier"].mean()) if len(d) else float("nan")
        out["threshold_logloss_avg"] = float(d["logloss"].mean()) if len(d) else float("nan")
        out["threshold_ece10_avg"] = float(d["ece10"].mean()) if len(d) else float("nan")
    return out


def run_residual(df: pd.DataFrame, cols: list[str], train_start: str, dev_start: str, dev_end: str, test_start: str, test_end: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = df.sort_values("target_date_local").reset_index(drop=True).copy()
    w["resid"] = w["y_tmax"] - w["mos_tmax_raw"]
    d = w["target_date_local"]
    dev = w[d.between(pd.Timestamp(dev_start), pd.Timestamp(dev_end))][["target_date_local", "y_tmax", "mos_tmax_raw"]].copy()
    for c in ["pred_point"] + [f"q_{q:.2f}" for q in QUANTILES]:
        dev[c] = np.nan
    for ms in month_starts(dev_start, dev_end):
        me = (ms + pd.offsets.MonthEnd(1)).normalize()
        msk = d.between(ms, me)
        trn = d.between(pd.Timestamp(train_start), ms - pd.Timedelta(days=1))
        if msk.sum() == 0 or trn.sum() < 200:
            continue
        Xtr, Xv = fill_train_median(w.loc[trn], w.loc[msk], cols)
        ytr = w.loc[trn, "resid"].to_numpy(dtype=float)
        p = fit_point(Xtr, ytr, seed)
        block_pred = w.loc[msk, "mos_tmax_raw"].to_numpy(dtype=float) + p.predict(Xv)
        dev.loc[dev["target_date_local"].isin(w.loc[msk, "target_date_local"]), "pred_point"] = block_pred
        for q in QUANTILES:
            qm = fit_q(Xtr, ytr, seed, q)
            qv = w.loc[msk, "mos_tmax_raw"].to_numpy(dtype=float) + qm.predict(Xv)
            dev.loc[dev["target_date_local"].isin(w.loc[msk, "target_date_local"]), f"q_{q:.2f}"] = qv
    tst_mask = d.between(pd.Timestamp(test_start), pd.Timestamp(test_end))
    trn_full = d.between(pd.Timestamp(train_start), pd.Timestamp("2023-12-31"))
    Xtr, Xte = fill_train_median(w.loc[trn_full], w.loc[tst_mask], cols)
    ytr = w.loc[trn_full, "resid"].to_numpy(dtype=float)
    pfull = fit_point(Xtr, ytr, seed)
    test = w.loc[tst_mask, ["target_date_local", "y_tmax", "mos_tmax_raw"]].copy()
    test["pred_point"] = w.loc[tst_mask, "mos_tmax_raw"].to_numpy(dtype=float) + pfull.predict(Xte)
    for q in QUANTILES:
        qm = fit_q(Xtr, ytr, seed, q)
        test[f"q_{q:.2f}"] = w.loc[tst_mask, "mos_tmax_raw"].to_numpy(dtype=float) + qm.predict(Xte)
    return dev, test


def wquant(values: np.ndarray, qs: list[float], w: np.ndarray) -> np.ndarray:
    idx = np.argsort(values)
    v = values[idx]
    ww = w[idx]
    c = np.cumsum(ww)
    if c[-1] <= 0:
        return np.full(len(qs), np.nan)
    c = c / c[-1]
    return np.array([np.interp(q, c, v) for q in qs], dtype=float)


def knn_all(df: pd.DataFrame, train_start: str, state_cols: list[str], k: int) -> pd.DataFrame:
    w = df.sort_values("target_date_local").reset_index(drop=True).copy()
    w["resid"] = w["y_tmax"] - w["mos_tmax_raw"]
    out = w[["target_date_local", "y_tmax", "mos_tmax_raw"]].copy()
    for c in ["analog_resid_med", "analog_resid_q10", "analog_resid_q90", "analog_resid_iqr", "analog_dist_min", "analog_dist_mean", "analog_eff_k", "pred_point"]:
        out[c] = np.nan
    for q in QUANTILES:
        out[f"q_{q:.2f}"] = np.nan
    X = w[state_cols].to_numpy(dtype=float)
    y = w["resid"].to_numpy(dtype=float)
    d = w["target_date_local"].to_numpy()
    start = pd.Timestamp(train_start)
    for i in range(len(w)):
        di = pd.Timestamp(d[i]).normalize()
        cm = (w["target_date_local"] >= start) & (w["target_date_local"] < di)
        idx = np.where(cm.to_numpy())[0]
        if len(idx) < 20:
            continue
        Xc = X[idx]
        yc = y[idx]
        ok = np.isfinite(Xc).all(axis=1) & np.isfinite(yc)
        Xc = Xc[ok]
        yc = yc[ok]
        if len(Xc) < 20 or not np.isfinite(X[i]).all():
            continue
        mu = np.mean(Xc, axis=0)
        sd = np.std(Xc, axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        zc = (Xc - mu) / sd
        zq = (X[i] - mu) / sd
        dist = np.sqrt(np.sum((zc - zq) ** 2, axis=1))
        kk = min(k, len(dist))
        sel = np.argpartition(dist, kk - 1)[:kk]
        ds = dist[sel]
        ys = yc[sel]
        s = float(np.median(ds)) if np.isfinite(ds).any() else 1.0
        s = 1.0 if s <= 1e-8 else s
        ww = np.exp(-ds / s)
        sw = float(np.sum(ww))
        if sw <= 0:
            continue
        ww = ww / sw
        qv = wquant(ys, QUANTILES, ww)
        qmap = {q: qv[j] for j, q in enumerate(QUANTILES)}
        out.at[i, "analog_resid_med"] = qmap[0.5]
        out.at[i, "analog_resid_q10"] = qmap[0.1]
        out.at[i, "analog_resid_q90"] = qmap[0.9]
        out.at[i, "analog_resid_iqr"] = qmap[0.75] - qmap[0.25]
        out.at[i, "analog_dist_min"] = float(np.min(ds))
        out.at[i, "analog_dist_mean"] = float(np.mean(ds))
        out.at[i, "analog_eff_k"] = float(1.0 / np.sum(ww**2))
        out.at[i, "pred_point"] = w.at[i, "mos_tmax_raw"] + qmap[0.5]
        for q in QUANTILES:
            out.at[i, f"q_{q:.2f}"] = w.at[i, "mos_tmax_raw"] + qmap[q]
    return out


def tune_w(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    bw, bm = 0.5, 1e18
    for w in W_GRID:
        m = float(np.mean(np.abs(y - (w * a + (1 - w) * b))))
        if m < bm:
            bw, bm = w, m
    return bw


def main() -> None:
    parser = argparse.ArgumentParser(description="Station MOS-first full run")
    parser.add_argument("--station-id", default="KNYC")
    parser.add_argument("--station-zoneid", default="America/New_York")
    parser.add_argument("--mos-csv", default=r"D:\Ahmed\data\kalshi\training_data\KNYC_mos_archive_2000_2025.csv.gz")
    parser.add_argument("--truth-csv", default=r"D:\Ahmed\data\kalshi\training_data\KNYC_settled_tmax.csv")
    parser.add_argument("--out-root", default=r"D:\Ahmed\data\kalshi\Experiments\MOS")
    parser.add_argument("--dev-start", default="2022-01-01")
    parser.add_argument("--dev-end", default="2023-12-31")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument("--common-train-start-policy", choices=["legacy", "manual", "earliest-common"], default="legacy")
    parser.add_argument("--common-train-start", default=None, help="Optional ISO date floor for common GFS/NAM slices.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    out = ensure_dir(Path(args.out_root))
    d0 = ensure_dir(out / "00_data")
    d1 = ensure_dir(out / "01_phaseA_raw")
    d2 = ensure_dir(out / "02_phaseB_residual")
    d3 = ensure_dir(out / "03_blends")
    d4 = ensure_dir(out / "04_knn_ablation")
    d9 = ensure_dir(out / "09_reports")

    truth = load_truth(Path(args.truth_csv))
    mos = load_mos(Path(args.mos_csv), str(args.station_zoneid))
    slice_defs = resolve_slice_defs(
        mos,
        common_train_start_policy=str(args.common_train_start_policy),
        common_train_start=str(args.common_train_start) if args.common_train_start else None,
    )
    LOGGER.info("Resolved slice train starts: %s", {s.sid: s.train_start for s in slice_defs})
    features = [
        "mos_tmax_raw", "mos_tmin_raw", "mos_dtr_raw", "mos_tmax_hour_local", "tmp_09", "tmp_12", "tmp_15", "tmp_18", "tmp_21",
        "heat_09_15", "heat_12_18", "cool_18_21", "nx_high", "nx_low", "dpt_09", "dpt_15", "dpt_21", "dep_15", "dep_21", "dpt_change_09_15",
        "cloud_mean_12_21", "cloud_max_12_21", "cloud_change_12_18", "p06_max_day", "p12_max_day", "t06_max_day", "t12_max_day",
        "wsp_mean_12_21", "wsp_max_day", "wdr_sin_mean_12_21", "wdr_cos_mean_12_21", "gst_max_day", "doy_sin", "doy_cos", "runtime_hour_utc", "lead_hours_to_local_noon",
    ]
    state = ["doy_sin", "doy_cos", "mos_tmax_raw", "mos_tmin_raw", "mos_dtr_raw", "tmp_15", "tmp_18", "dep_15", "cloud_mean_12_21", "p06_max_day", "wsp_mean_12_21", "wdr_sin_mean_12_21", "wdr_cos_mean_12_21"]

    slices: dict[str, pd.DataFrame] = {}
    residual: dict[str, dict[str, Any]] = {}
    raw_rows = []

    for s in slice_defs:
        LOGGER.info("Building %s", s.sid)
        sdf = build_slice(mos, truth, s, str(args.station_zoneid))
        slices[s.sid] = sdf
        ensure_dir(d0 / "slices")
        sdf.to_parquet(d0 / "slices" / f"{s.sid}.parquet", index=False)
        devm = sdf["target_date_local"].between(pd.Timestamp(args.dev_start), pd.Timestamp(args.dev_end))
        testm = sdf["target_date_local"].between(pd.Timestamp(args.test_start), pd.Timestamp(args.test_end))
        if devm.sum() and testm.sum():
            raw_rows.append(
                {
                    "slice_id": s.sid,
                    "dev_mae": float(np.mean(np.abs(sdf.loc[devm, "mos_tmax_raw"] - sdf.loc[devm, "y_tmax"]))),
                    "test_mae": float(np.mean(np.abs(sdf.loc[testm, "mos_tmax_raw"] - sdf.loc[testm, "y_tmax"]))),
                }
            )
        if len(sdf) < 500:
            continue
        LOGGER.info("Residual run %s rows=%s", s.sid, len(sdf))
        dev, test = run_residual(sdf, features, s.train_start, args.dev_start, args.dev_end, args.test_start, args.test_end, args.seed)
        qd = non_cross({q: dev[f"q_{q:.2f}"].to_numpy(dtype=float) for q in QUANTILES})
        qt = non_cross({q: test[f"q_{q:.2f}"].to_numpy(dtype=float) for q in QUANTILES})
        for q in QUANTILES:
            dev[f"q_{q:.2f}"] = qd[q]
            test[f"q_{q:.2f}"] = qt[q]
        trn_y = sdf[sdf["target_date_local"].between(pd.Timestamp(s.train_start), pd.Timestamp("2021-12-31"))]["y_tmax"].to_numpy(dtype=float)
        mdev = eval_dist(dev["y_tmax"].to_numpy(dtype=float), dev["pred_point"].to_numpy(dtype=float), qd, trn_y)
        mtst = eval_dist(test["y_tmax"].to_numpy(dtype=float), test["pred_point"].to_numpy(dtype=float), qt, trn_y)
        ensure_dir(d2 / s.sid)
        dev.to_parquet(d2 / s.sid / "dev_predictions.parquet", index=False)
        test.to_parquet(d2 / s.sid / "test_predictions.parquet", index=False)
        write_json(d2 / s.sid / "metrics_dev.json", mdev)
        write_json(d2 / s.sid / "metrics_test.json", mtst)
        residual[s.sid] = {"sdef": s, "sdf": sdf, "dev": dev, "test": test, "mdev": mdev, "mtst": mtst, "train_y": trn_y}

    pd.DataFrame(raw_rows).to_csv(d1 / "raw_metrics.csv", index=False)

    blend_rows = []
    for g, n, b in [("gfs_00", "nam_00", "blend_00"), ("gfs_12", "nam_12", "blend_12")]:
        if g not in residual or n not in residual:
            continue
        gd = residual[g]["dev"]
        nd = residual[n]["dev"]
        gt = residual[g]["test"]
        nt = residual[n]["test"]
        dev = gd.merge(nd, on=["target_date_local", "y_tmax"], suffixes=("_g", "_n"))
        test = gt.merge(nt, on=["target_date_local", "y_tmax"], suffixes=("_g", "_n"))
        if dev.empty or test.empty:
            continue
        wp = tune_w(dev["y_tmax"].to_numpy(), dev["pred_point_g"].to_numpy(), dev["pred_point_n"].to_numpy())
        bd = pd.DataFrame({"target_date_local": dev["target_date_local"], "y_tmax": dev["y_tmax"]})
        bt = pd.DataFrame({"target_date_local": test["target_date_local"], "y_tmax": test["y_tmax"]})
        bd["pred_point"] = wp * dev["pred_point_g"] + (1 - wp) * dev["pred_point_n"]
        bt["pred_point"] = wp * test["pred_point_g"] + (1 - wp) * test["pred_point_n"]
        for q in QUANTILES:
            c0, c1 = f"q_{q:.2f}_g", f"q_{q:.2f}_n"
            bw, bl = wp, 1e18
            yv = dev["y_tmax"].to_numpy(dtype=float)
            for w in W_GRID:
                pl = pinball(yv, w * dev[c0].to_numpy(dtype=float) + (1 - w) * dev[c1].to_numpy(dtype=float), q)
                if pl < bl:
                    bw, bl = w, pl
            bd[f"q_{q:.2f}"] = bw * dev[c0] + (1 - bw) * dev[c1]
            bt[f"q_{q:.2f}"] = bw * test[c0] + (1 - bw) * test[c1]
        qd = non_cross({q: bd[f"q_{q:.2f}"].to_numpy(dtype=float) for q in QUANTILES})
        qt = non_cross({q: bt[f"q_{q:.2f}"].to_numpy(dtype=float) for q in QUANTILES})
        for q in QUANTILES:
            bd[f"q_{q:.2f}"] = qd[q]
            bt[f"q_{q:.2f}"] = qt[q]
        trn = np.concatenate([residual[g]["train_y"], residual[n]["train_y"]])
        md, mt = eval_dist(bd["y_tmax"].to_numpy(dtype=float), bd["pred_point"].to_numpy(dtype=float), qd, trn), eval_dist(bt["y_tmax"].to_numpy(dtype=float), bt["pred_point"].to_numpy(dtype=float), qt, trn)
        ensure_dir(d3 / b)
        bd.to_parquet(d3 / b / "dev_predictions.parquet", index=False)
        bt.to_parquet(d3 / b / "test_predictions.parquet", index=False)
        write_json(d3 / b / "metrics_dev.json", md)
        write_json(d3 / b / "metrics_test.json", mt)
        write_json(d3 / b / "blend_weights.json", {"point": wp})
        blend_rows.append({"slice_id": b, "dev_mae": md.get("mae"), "test_mae": mt.get("mae"), "dev_avg_pinball": md.get("avg_pinball"), "test_avg_pinball": mt.get("avg_pinball")})
    if blend_rows:
        pd.DataFrame(blend_rows).to_csv(d3 / "blend_summary.csv", index=False)

    top2 = sorted([(k, v["mdev"]["mae"]) for k, v in residual.items()], key=lambda x: x[1])[:2]
    knn_rows = []
    for sid, _ in top2:
        sdef = residual[sid]["sdef"]
        sdf = residual[sid]["sdf"]
        best_k, best_mae, best = None, 1e18, None
        for k in K_GRID:
            ka = knn_all(sdf, sdef.train_start, state, k)
            dm = ka["target_date_local"].between(pd.Timestamp(args.dev_start), pd.Timestamp(args.dev_end))
            d = ka[dm].dropna(subset=["pred_point"])
            if len(d) < 100:
                continue
            mae = float(np.mean(np.abs(d["pred_point"] - d["y_tmax"])))
            if mae < best_mae:
                best_k, best_mae, best = k, mae, ka
        if best_k is None or best is None:
            continue
        ensure_dir(d4 / sid)
        write_json(d4 / sid / "knn_selection.json", {"best_k": best_k, "dev_mae": best_mae})
        best.to_parquet(d4 / sid / "knn_all_rows.parquet", index=False)
        dm = best["target_date_local"].between(pd.Timestamp(args.dev_start), pd.Timestamp(args.dev_end))
        tm = best["target_date_local"].between(pd.Timestamp(args.test_start), pd.Timestamp(args.test_end))
        k1d, k1t = best[dm].copy(), best[tm].copy()
        trn = residual[sid]["train_y"]
        q1d = non_cross({q: k1d[f"q_{q:.2f}"].to_numpy(dtype=float) for q in QUANTILES})
        q1t = non_cross({q: k1t[f"q_{q:.2f}"].to_numpy(dtype=float) for q in QUANTILES})
        mk1d, mk1t = eval_dist(k1d["y_tmax"].to_numpy(dtype=float), k1d["pred_point"].to_numpy(dtype=float), q1d, trn), eval_dist(k1t["y_tmax"].to_numpy(dtype=float), k1t["pred_point"].to_numpy(dtype=float), q1t, trn)
        aug = features + ["analog_resid_med", "analog_resid_q10", "analog_resid_q90", "analog_resid_iqr", "analog_dist_min", "analog_dist_mean", "analog_eff_k"]
        merged = sdf.merge(best[["target_date_local", *aug[len(features):]]], on="target_date_local", how="left")
        k2d, k2t = run_residual(merged, aug, sdef.train_start, args.dev_start, args.dev_end, args.test_start, args.test_end, args.seed + 77)
        q2d = non_cross({q: k2d[f"q_{q:.2f}"].to_numpy(dtype=float) for q in QUANTILES})
        q2t = non_cross({q: k2t[f"q_{q:.2f}"].to_numpy(dtype=float) for q in QUANTILES})
        mk2d, mk2t = eval_dist(k2d["y_tmax"].to_numpy(dtype=float), k2d["pred_point"].to_numpy(dtype=float), q2d, trn), eval_dist(k2t["y_tmax"].to_numpy(dtype=float), k2t["pred_point"].to_numpy(dtype=float), q2t, trn)
        k0d = residual[sid]["dev"].merge(k1d[["target_date_local", "pred_point", *[f"q_{q:.2f}" for q in QUANTILES]]], on="target_date_local", suffixes=("_k0", "_k1"))
        k0t = residual[sid]["test"].merge(k1t[["target_date_local", "pred_point", *[f"q_{q:.2f}" for q in QUANTILES]]], on="target_date_local", suffixes=("_k0", "_k1"))
        ba, bm = 0.5, 1e18
        yv = k0d["y_tmax"].to_numpy(dtype=float)
        for a in ALPHA_GRID:
            mae = float(np.mean(np.abs(yv - (a * k0d["pred_point_k0"].to_numpy(dtype=float) + (1 - a) * k0d["pred_point_k1"].to_numpy(dtype=float)))))
            if mae < bm:
                ba, bm = a, mae
        k3d = pd.DataFrame({"target_date_local": k0d["target_date_local"], "y_tmax": k0d["y_tmax"]})
        k3t = pd.DataFrame({"target_date_local": k0t["target_date_local"], "y_tmax": k0t["y_tmax"]})
        k3d["pred_point"] = ba * k0d["pred_point_k0"] + (1 - ba) * k0d["pred_point_k1"]
        k3t["pred_point"] = ba * k0t["pred_point_k0"] + (1 - ba) * k0t["pred_point_k1"]
        for q in QUANTILES:
            c0, c1 = f"q_{q:.2f}_k0", f"q_{q:.2f}_k1"
            k3d[f"q_{q:.2f}"] = ba * k0d[c0] + (1 - ba) * k0d[c1]
            k3t[f"q_{q:.2f}"] = ba * k0t[c0] + (1 - ba) * k0t[c1]
        q3d = non_cross({q: k3d[f"q_{q:.2f}"].to_numpy(dtype=float) for q in QUANTILES})
        q3t = non_cross({q: k3t[f"q_{q:.2f}"].to_numpy(dtype=float) for q in QUANTILES})
        mk3d, mk3t = eval_dist(k3d["y_tmax"].to_numpy(dtype=float), k3d["pred_point"].to_numpy(dtype=float), q3d, trn), eval_dist(k3t["y_tmax"].to_numpy(dtype=float), k3t["pred_point"].to_numpy(dtype=float), q3t, trn)
        k1d.to_parquet(d4 / sid / "k1_dev_predictions.parquet", index=False)
        k1t.to_parquet(d4 / sid / "k1_test_predictions.parquet", index=False)
        k2d.to_parquet(d4 / sid / "k2_dev_predictions.parquet", index=False)
        k2t.to_parquet(d4 / sid / "k2_test_predictions.parquet", index=False)
        k3d.to_parquet(d4 / sid / "k3_dev_predictions.parquet", index=False)
        k3t.to_parquet(d4 / sid / "k3_test_predictions.parquet", index=False)
        write_json(d4 / sid / "metrics_k1_dev.json", mk1d)
        write_json(d4 / sid / "metrics_k1_test.json", mk1t)
        write_json(d4 / sid / "metrics_k2_dev.json", mk2d)
        write_json(d4 / sid / "metrics_k2_test.json", mk2t)
        write_json(d4 / sid / "metrics_k3_dev.json", mk3d)
        write_json(d4 / sid / "metrics_k3_test.json", mk3t)
        write_json(d4 / sid / "k3_alpha.json", {"alpha": ba})
        knn_rows += [
            {"slice_id": sid, "variant": "K0", "dev_mae": residual[sid]["mdev"]["mae"], "test_mae": residual[sid]["mtst"]["mae"], "dev_avg_pinball": residual[sid]["mdev"].get("avg_pinball"), "test_avg_pinball": residual[sid]["mtst"].get("avg_pinball")},
            {"slice_id": sid, "variant": "K1", "dev_mae": mk1d.get("mae"), "test_mae": mk1t.get("mae"), "dev_avg_pinball": mk1d.get("avg_pinball"), "test_avg_pinball": mk1t.get("avg_pinball")},
            {"slice_id": sid, "variant": "K2", "dev_mae": mk2d.get("mae"), "test_mae": mk2t.get("mae"), "dev_avg_pinball": mk2d.get("avg_pinball"), "test_avg_pinball": mk2t.get("avg_pinball")},
            {"slice_id": sid, "variant": "K3", "dev_mae": mk3d.get("mae"), "test_mae": mk3t.get("mae"), "dev_avg_pinball": mk3d.get("avg_pinball"), "test_avg_pinball": mk3t.get("avg_pinball")},
        ]
    if knn_rows:
        pd.DataFrame(knn_rows).to_csv(d4 / "knn_summary.csv", index=False)

    comp = []
    for sid, r in residual.items():
        comp.append({"family": "residual_ml", "slice_id": sid, "dev_mae": r["mdev"].get("mae"), "test_mae": r["mtst"].get("mae"), "dev_avg_pinball": r["mdev"].get("avg_pinball"), "test_avg_pinball": r["mtst"].get("avg_pinball"), "test_cov80": r["mtst"].get("coverage_80"), "test_cov90": r["mtst"].get("coverage_90"), "test_threshold_ece10_avg": r["mtst"].get("threshold_ece10_avg")})
    for r in blend_rows:
        comp.append({"family": "blend_ml", "slice_id": r["slice_id"], "dev_mae": r["dev_mae"], "test_mae": r["test_mae"], "dev_avg_pinball": r["dev_avg_pinball"], "test_avg_pinball": r["test_avg_pinball"]})
    for r in knn_rows:
        comp.append({"family": f"knn_{r['variant']}", "slice_id": r["slice_id"], "dev_mae": r["dev_mae"], "test_mae": r["test_mae"], "dev_avg_pinball": r["dev_avg_pinball"], "test_avg_pinball": r["test_avg_pinball"]})
    cdf = pd.DataFrame(comp).sort_values(["test_mae", "dev_mae"], na_position="last")
    cdf.to_csv(d9 / "final_comparison.csv", index=False)
    best = cdf.iloc[0].to_dict() if len(cdf) else {}
    write_json(
        d9 / "summary.json",
        {
            "generated_at_utc": utc_now_iso(),
            "inputs": {"mos_csv": args.mos_csv, "truth_csv": args.truth_csv},
            "splits": {"dev_start": args.dev_start, "dev_end": args.dev_end, "test_start": args.test_start, "test_end": args.test_end},
            "train_start_policy": {
                "common_train_start_policy": args.common_train_start_policy,
                "common_train_start_floor": args.common_train_start,
                "resolved_train_starts_by_slice": {s.sid: s.train_start for s in slice_defs},
            },
            "top2_knn": [s for s, _ in top2],
            "best_overall": best,
        },
    )
    (d9 / "results_executive.md").write_text(
        "\n".join(
            [
                "# KNYC MOS-First Results",
                f"- Generated UTC: {utc_now_iso()}",
                f"- Best row: {best}",
                "- Chronology: monthly OOF on dev, frozen design to test.",
                f"- Train starts by slice: {{ {', '.join(f'{s.sid}: {s.train_start}' for s in slice_defs)} }}",
                "- KNN candidates always candidate_date < query_date.",
                "- Blend weights tuned on dev only.",
            ]
        ),
        encoding="utf-8",
    )
    LOGGER.info("Completed. Outputs: %s", out)


if __name__ == "__main__":
    main()
