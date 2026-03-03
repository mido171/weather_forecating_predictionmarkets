from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kstest
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

from .cdf_bucket_mapper import Bucket
from .train_quantiles import avg_pinball


def _season_of_date(d: pd.Timestamp) -> str:
    m = d.month
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"


def point_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    m = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(m):
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "median_ae": float("nan"),
            "mean_signed_error": float("nan"),
            "p90_ae": float("nan"),
        }
    y = y[m]
    yhat = yhat[m]
    e = yhat - y
    ae = np.abs(e)
    return {
        "mae": float(mean_absolute_error(y, yhat)),
        "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
        "median_ae": float(np.median(ae)),
        "mean_signed_error": float(np.mean(e)),
        "p90_ae": float(np.quantile(ae, 0.9)),
    }


def quantile_metrics(y: np.ndarray, pred_q: pd.DataFrame, quantiles: list[float]) -> dict[str, Any]:
    arr = pred_q[[f"q_{q:.3f}" for q in quantiles]].to_numpy(dtype=float)
    m = np.isfinite(y) & np.all(np.isfinite(arr), axis=1)
    if not np.any(m):
        return {
            "avg_pinball": float("nan"),
            "pinball_by_quantile": {f"{q:.3f}": float("nan") for q in quantiles},
            "quantile_crossing_count_before_repair": 0,
        }
    y = y[m]
    arr = arr[m]
    crossing_before = int(np.sum(np.any(np.diff(arr, axis=1) < 0, axis=1)))
    pinball_by_q = {}
    pin_vals = []
    for qi, q in enumerate(quantiles):
        e = y - arr[:, qi]
        pin = np.maximum(q * e, (q - 1.0) * e)
        pmean = float(np.mean(pin))
        pinball_by_q[f"{q:.3f}"] = pmean
        pin_vals.append(pmean)
    return {
        "avg_pinball": float(np.mean(pin_vals)),
        "pinball_by_quantile": pinball_by_q,
        "quantile_crossing_count_before_repair": crossing_before,
    }


def interval_coverage_metrics(rows: pd.DataFrame, pred_q: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out_rows = []
    y = rows["y_tmax"].to_numpy(dtype=float)

    qcols = [c for c in pred_q.columns if c.startswith("q_")]
    qlevels = np.array([float(c.split("_")[1]) for c in qcols], dtype=float) if qcols else np.array([], dtype=float)
    qarr = pred_q[qcols].to_numpy(dtype=float) if qcols else np.empty((len(pred_q), 0), dtype=float)

    def q_at(tau: float) -> np.ndarray:
        cname = f"q_{tau:.3f}"
        if cname in pred_q.columns:
            return pred_q[cname].to_numpy(dtype=float)
        if qarr.shape[1] == 0:
            return np.full(len(pred_q), np.nan, dtype=float)
        out = np.empty(len(pred_q), dtype=float)
        for i in range(len(pred_q)):
            out[i] = float(np.interp(tau, qlevels, qarr[i]))
        return out

    specs = {
        "50": ("q_0.250", "q_0.750", 0.50),
        "80": ("q_0.100", "q_0.900", 0.80),
        "90": ("q_0.050", "q_0.950", 0.90),
        "95": ("q_0.025", "q_0.975", 0.95),
        "98": ("q_0.010", "q_0.990", 0.98),
    }
    summary = {}
    for name, (lo_col, hi_col, nominal) in specs.items():
        lo_tau = float(lo_col.split("_")[1])
        hi_tau = float(hi_col.split("_")[1])
        lo = q_at(lo_tau)
        hi = q_at(hi_tau)
        m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        if not np.any(m):
            hit = np.array([], dtype=float)
            width = np.array([], dtype=float)
            cov = float("nan")
            avg_width = float("nan")
            med_width = float("nan")
        else:
            hit = ((y[m] >= lo[m]) & (y[m] <= hi[m])).astype(float)
            width = hi[m] - lo[m]
            cov = float(np.mean(hit))
            avg_width = float(np.mean(width))
            med_width = float(np.median(width))
        out_rows.append(
            {
                "interval": name,
                "nominal": nominal,
                "empirical": cov,
                "coverage_error": cov - nominal,
                "avg_width": avg_width,
                "median_width": med_width,
            }
        )
        summary[f"cov_{name}"] = cov
        summary[f"avg_width_{name}"] = avg_width
    return pd.DataFrame(out_rows), summary


def pit_metrics(pmf_df: pd.DataFrame, y: np.ndarray, support_min: int = 0, support_max: int = 120, seed: int = 42) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    temps = np.arange(support_min, support_max + 1, dtype=int)
    pcols = [f"p_int_{t}" for t in temps]

    pits = []
    top1 = []
    top3 = []
    for i, yi in enumerate(y):
        yi_int = int(round(float(yi)))
        pmf = pmf_df.iloc[i][pcols].to_numpy(dtype=float)
        pmf = np.clip(pmf, 0.0, None)
        s = np.sum(pmf)
        pmf = pmf / s if s > 0 else np.full_like(pmf, 1.0 / len(pmf))

        cdf = np.cumsum(pmf)
        idx = int(np.clip(yi_int - support_min, 0, len(temps) - 1))
        left = float(cdf[idx - 1]) if idx > 0 else 0.0
        mass = float(pmf[idx])
        u = float(left + rng.random() * mass)
        pits.append(u)

        order = np.argsort(-pmf)
        top1.append(1.0 if temps[order[0]] == yi_int else 0.0)
        top3.append(1.0 if yi_int in temps[order[:3]] else 0.0)

    ks_stat, ks_p = kstest(np.array(pits, dtype=float), "uniform")
    hist, edges = np.histogram(np.array(pits), bins=20, range=(0, 1))
    hist_df = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist})
    summary = {
        "pit_ks_stat": float(ks_stat),
        "pit_ks_pvalue": float(ks_p),
        "top1_integer_accuracy": float(np.mean(top1)),
        "top3_integer_coverage": float(np.mean(top3)),
    }
    return hist_df, summary


def _calibration_table(prob: np.ndarray, obs: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(prob, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            rows.append({"bin": b, "count": 0, "pred_mean": np.nan, "obs_rate": np.nan, "abs_gap": np.nan})
            continue
        pm = float(np.mean(prob[mask]))
        orate = float(np.mean(obs[mask]))
        rows.append({"bin": b, "count": int(np.sum(mask)), "pred_mean": pm, "obs_rate": orate, "abs_gap": abs(pm - orate)})
    return pd.DataFrame(rows)


def _ece_mce(cal_df: pd.DataFrame, total_n: int) -> tuple[float, float]:
    valid = cal_df[cal_df["count"] > 0].copy()
    if valid.empty or total_n <= 0:
        return 0.0, 0.0
    w = valid["count"].to_numpy(dtype=float) / total_n
    gap = valid["abs_gap"].to_numpy(dtype=float)
    return float(np.sum(w * gap)), float(np.max(gap))


def bucket_calibration_metrics(
    bucket_prob_df: pd.DataFrame,
    realized_bucket_df: pd.DataFrame,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cols = [c for c in bucket_prob_df.columns if c.startswith("bucket_yes::")]
    rows = []
    rel_rows = []

    all_prob = []
    all_obs = []

    for c in cols:
        p = bucket_prob_df[c].to_numpy(dtype=float)
        o = realized_bucket_df[c].to_numpy(dtype=float)
        p = np.where(np.isfinite(p), p, 0.5)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        all_prob.append(p)
        all_obs.append(o)

        brier = float(brier_score_loss(o, p))
        ll = float(log_loss(o, np.vstack([1 - p, p]).T, labels=[0, 1]))
        cal = _calibration_table(p, o, n_bins=n_bins)
        ece, mce = _ece_mce(cal, len(o))
        slope, intercept = np.polyfit(p, o, 1) if len(np.unique(p)) > 1 else (np.nan, np.nan)

        rows.append(
            {
                "bucket": c,
                "brier": brier,
                "logloss": ll,
                "ece10": ece,
                "mce": mce,
                "calibration_slope": float(slope) if np.isfinite(slope) else np.nan,
                "calibration_intercept": float(intercept) if np.isfinite(intercept) else np.nan,
            }
        )
        cal["bucket"] = c
        rel_rows.append(cal)

    p_all = np.concatenate(all_prob) if all_prob else np.array([], dtype=float)
    o_all = np.concatenate(all_obs) if all_obs else np.array([], dtype=float)

    if len(p_all) > 0:
        all_brier = float(brier_score_loss(o_all, p_all))
        all_ll = float(log_loss(o_all, np.vstack([1 - p_all, p_all]).T, labels=[0, 1]))
        all_cal = _calibration_table(p_all, o_all, n_bins=n_bins)
        all_ece, all_mce = _ece_mce(all_cal, len(o_all))
    else:
        all_brier, all_ll, all_ece, all_mce = np.nan, np.nan, np.nan, np.nan
        all_cal = pd.DataFrame(columns=["bin", "count", "pred_mean", "obs_rate", "abs_gap"])

    all_cal["bucket"] = "__overall__"
    rel_rows.append(all_cal)

    summary = {
        "overall_brier": all_brier,
        "overall_logloss": all_ll,
        "overall_ece10": all_ece,
        "overall_mce": all_mce,
    }
    return pd.DataFrame(rows), pd.concat(rel_rows, ignore_index=True), summary


def slice_metrics(
    rows: pd.DataFrame,
    pred_q: pd.DataFrame,
    pmf_df: pd.DataFrame,
    bucket_prob_df: pd.DataFrame,
    realized_bucket_df: pd.DataFrame,
) -> pd.DataFrame:
    out = rows.copy()
    out["season"] = pd.to_datetime(out["target_date_local"]).map(_season_of_date)
    out["cloud_regime"] = np.where(pd.to_numeric(out.get("clds_norm"), errors="coerce") >= 0.6, "cloudy", "clear")
    wdir = pd.to_numeric(out.get("wdir"), errors="coerce")
    out["wind_regime"] = np.where((wdir >= 70) & (wdir <= 160), "E_SE", np.where((wdir >= 260) & (wdir <= 340), "W_NW", "OTHER"))
    out["precip_regime"] = np.where(pd.to_numeric(out.get("precip_flag"), errors="coerce") > 0, "precip", "no_precip")
    out["sea_breeze_proxy"] = np.where(pd.to_numeric(out.get("coastal_minus_inland_temp"), errors="coerce") < -1.0, "high", "low")
    out["analog_gap_abs"] = (pred_q["q_0.500"] - pred_q["q_0.500"]).abs()  # placeholder constant zero for schema stability

    rows_out = []
    for col in ["season", "cloud_regime", "wind_regime", "precip_regime", "sea_breeze_proxy"]:
        for key, sub_idx in out.groupby(col).groups.items():
            idx = list(sub_idx)
            y = out.loc[idx, "y_tmax"].to_numpy(dtype=float)
            q50 = pred_q.loc[idx, "q_0.500"].to_numpy(dtype=float)
            mae = float(np.mean(np.abs(y - q50)))
            rows_out.append({"slice": col, "value": str(key), "rows": len(idx), "mae_q50": mae})
    return pd.DataFrame(rows_out)
