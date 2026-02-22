from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.isotonic import IsotonicRegression

import run_v5_plus_suite as v5
from ml_live.calibration.emos_w45 import calibrate as emos_calibrate


def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.date


def split_masks(dates: pd.Series) -> dict[str, np.ndarray]:
    dt = pd.to_datetime(dates)
    train_mask = (dt >= "2002-01-22") & (dt <= "2019-12-31")
    val_mask = (dt >= "2020-01-01") & (dt <= "2022-12-31")
    test_mask = (dt >= "2023-01-01") & (dt <= "2025-12-31")
    return {
        "train_mask": train_mask.to_numpy(),
        "val_mask": val_mask.to_numpy(),
        "test_mask": test_mask.to_numpy(),
    }


def _pinball(y: np.ndarray, q_hat: np.ndarray, q: float) -> float:
    diff = y - q_hat
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def _interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return {"coverage": float("nan"), "avg_width": float("nan")}
    cov = float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])))
    width = float(np.mean(hi[mask] - lo[mask]))
    return {"coverage": cov, "avg_width": width}


def _pit_stats(pit: np.ndarray) -> dict[str, float]:
    pit = pit[np.isfinite(pit)]
    if pit.size == 0:
        return {}
    hist, _ = np.histogram(pit, bins=10, range=(0.0, 1.0))
    expected = pit.size / 10.0
    chi2 = float(np.sum((hist - expected) ** 2 / max(expected, 1.0)))
    return {
        "count": int(pit.size),
        "mean": float(np.mean(pit)),
        "std": float(np.std(pit)),
        "chi2": chi2,
        "hist_bins": hist.tolist(),
    }


def _reliability_table(probs: np.ndarray, y_true: np.ndarray, bins: int = 10) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(probs, edges, right=True) - 1
    idx = np.clip(idx, 0, bins - 1)
    rows = []
    ece = 0.0
    mce = 0.0
    total = len(probs)
    for b in range(bins):
        mask = idx == b
        if not np.any(mask):
            rows.append({"bin": b, "count": 0, "avg_pred": None, "emp_rate": None})
            continue
        avg_pred = float(np.mean(probs[mask]))
        emp_rate = float(np.mean(y_true[mask]))
        rows.append({"bin": b, "count": int(mask.sum()), "avg_pred": avg_pred, "emp_rate": emp_rate})
        gap = abs(avg_pred - emp_rate)
        ece += (mask.sum() / total) * gap
        mce = max(mce, gap)
    return {"bins": rows, "ece": float(ece), "mce": float(mce)}


def _brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(probs, 0.0, 1.0)
    return float(np.mean((y_true - probs) ** 2))


def _log_loss_binary(y_true: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _crps_empirical(resid_samples: np.ndarray, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    r = np.asarray(resid_samples, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.full_like(y, np.nan, dtype=float)
    r_sorted = np.sort(r)
    n = r_sorted.size
    coeff = (2 * np.arange(1, n + 1) - n - 1) / (n * n)
    pairwise_mean = float(np.sum(coeff * r_sorted))
    dy = (y - pred).reshape(-1, 1)
    mean_abs = np.mean(np.abs(r_sorted.reshape(1, -1) - dy), axis=1)
    return mean_abs - 0.5 * pairwise_mean


def _cdf_from_quantiles(y: np.ndarray, q_vals: np.ndarray, q_levels: np.ndarray) -> np.ndarray:
    q_vals = np.asarray(q_vals, dtype=float)
    q_levels = np.asarray(q_levels, dtype=float)
    q_vals = np.maximum.accumulate(q_vals)
    out = np.zeros_like(y, dtype=float)
    for i, yi in enumerate(y):
        if yi <= q_vals[0]:
            if q_vals[1] == q_vals[0]:
                out[i] = q_levels[0]
            else:
                out[i] = q_levels[0] + (yi - q_vals[0]) * (q_levels[1] - q_levels[0]) / (
                    q_vals[1] - q_vals[0]
                )
        elif yi >= q_vals[-1]:
            if q_vals[-1] == q_vals[-2]:
                out[i] = q_levels[-1]
            else:
                out[i] = q_levels[-1] + (yi - q_vals[-1]) * (q_levels[-1] - q_levels[-2]) / (
                    q_vals[-1] - q_vals[-2]
                )
        else:
            idx = np.searchsorted(q_vals, yi) - 1
            x0, x1 = q_vals[idx], q_vals[idx + 1]
            p0, p1 = q_levels[idx], q_levels[idx + 1]
            if x1 == x0:
                out[i] = p1
            else:
                out[i] = p0 + (p1 - p0) * (yi - x0) / (x1 - x0)
        out[i] = float(np.clip(out[i], 0.0, 1.0))
    return out


def _crps_from_quantiles(y: np.ndarray, q_preds: dict[float, np.ndarray]) -> float:
    qs = np.array(sorted(q_preds.keys()), dtype=float)
    pin = []
    for q in qs:
        pin.append(_pinball(y, q_preds[q], q))
    pin = np.array(pin, dtype=float)
    weights = np.zeros_like(pin)
    weights[0] = qs[1] - qs[0]
    weights[-1] = qs[-1] - qs[-2]
    weights[1:-1] = (qs[2:] - qs[:-2]) / 2.0
    return float(2.0 * np.sum(weights * pin))


def _ensure_mu_oof_manifest(folds: list[dict[str, Any]], violations: int, out_path: Path) -> None:
    payload = {
        "oof_years": "2008-2019",
        "folds": folds,
        "violations": violations,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_mu_predictions(
    df: pd.DataFrame,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], int]:
    dates = pd.to_datetime(df["target_date_local"]).dt.date.to_numpy()
    masks = split_masks(pd.to_datetime(df["target_date_local"]))
    train_mask = masks["train_mask"]
    val_mask = masks["val_mask"]

    # Final frozen mu model (train 2002-2019, early stop on 2020-2022)
    mu_final = v5.compute_v5p8_predictions(df, train_mask=train_mask, val_mask=val_mask, seed=seed)["pred_v5p8"]

    # OOF mu for train years 2008-2019 (expanding)
    mu_oof = np.full(len(df), np.nan, dtype=float)
    folds: list[dict[str, Any]] = []
    violations = 0
    for year in range(2008, 2020):
        train_end = date(year - 2, 12, 31)
        val_start = date(year - 1, 1, 1)
        val_end = date(year - 1, 12, 31)
        pred_start = date(year, 1, 1)
        pred_end = date(year, 12, 31)

        train_mask_y = (dates >= date(2002, 1, 22)) & (dates <= train_end)
        val_mask_y = (dates >= val_start) & (dates <= val_end)
        pred_mask_y = (dates >= pred_start) & (dates <= pred_end)

        if train_mask_y.any():
            if max(dates[train_mask_y]) > train_end:
                violations += 1
        if pred_mask_y.any():
            if min(dates[pred_mask_y]) < pred_start or max(dates[pred_mask_y]) > pred_end:
                violations += 1
        if np.any(train_mask_y & pred_mask_y) or np.any(val_mask_y & pred_mask_y):
            violations += 1

        fold_pred = v5.compute_v5p8_predictions(
            df,
            train_mask=train_mask_y,
            val_mask=val_mask_y,
            seed=seed,
        )["pred_v5p8"]
        mu_oof[pred_mask_y] = fold_pred[pred_mask_y]

        folds.append(
            {
                "year": year,
                "train_start": "2002-01-22",
                "train_end": train_end.isoformat(),
                "val_start": val_start.isoformat(),
                "val_end": val_end.isoformat(),
                "pred_start": pred_start.isoformat(),
                "pred_end": pred_end.isoformat(),
                "train_count": int(train_mask_y.sum()),
                "val_count": int(val_mask_y.sum()),
                "pred_count": int(pred_mask_y.sum()),
            }
        )
    return mu_final, mu_oof, folds, violations


def _studentt_quantiles(mu: np.ndarray, sigma: np.ndarray, nu: int, levels: list[float]) -> dict[float, np.ndarray]:
    q = {}
    for a in levels:
        q[a] = mu + sigma * stats.t.ppf(a, nu)
    return q


def _studentt_metrics(
    *,
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: int,
    mask: np.ndarray,
) -> dict[str, Any]:
    idx = np.where(mask)[0]
    levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    q_preds = _studentt_quantiles(mu, sigma, nu, levels)

    metrics: dict[str, Any] = {}
    metrics["crps"] = _crps_from_quantiles(y[idx], {q: q_preds[q][idx] for q in levels})
    pit_vals = []
    for i in idx:
        q_vals = np.array([q_preds[q][i] for q in levels], dtype=float)
        pit_vals.append(_cdf_from_quantiles(np.array([y[i]]), q_vals, np.array(levels))[0])
    pit = _pit_stats(np.array(pit_vals))
    metrics["pit_chi2"] = pit.get("chi2")
    metrics["pit_bins"] = pit.get("hist_bins")

    intervals = {
        "p50": _interval_metrics(y[idx], q_preds[0.25][idx], q_preds[0.75][idx]),
        "p80": _interval_metrics(y[idx], q_preds[0.10][idx], q_preds[0.90][idx]),
        "p90": _interval_metrics(y[idx], q_preds[0.05][idx], q_preds[0.95][idx]),
    }
    metrics["intervals"] = intervals

    def cdf_at(thr: float) -> np.ndarray:
        probs = []
        for i in idx:
            q_vals = np.array([q_preds[q][i] for q in levels], dtype=float)
            probs.append(_cdf_from_quantiles(np.array([thr]), q_vals, np.array(levels))[0])
        return np.array(probs, dtype=float)

    metrics["events"] = _event_metrics_from_cdf(
        y[idx], mu[idx], cdf_at, thresholds=[80.0, 85.0, 90.0]
    )

    # sigma diagnostics
    z = (y[idx] - mu[idx]) / sigma[idx]
    abs_err = np.abs(y[idx] - mu[idx])
    metrics["sigma_diagnostics"] = {
        "z_mean": float(np.nanmean(z)),
        "z_std": float(np.nanstd(z)),
        "p_abs_z_le_1_28": float(np.mean(np.abs(z) <= 1.28)),
        "p_abs_z_le_1_64": float(np.mean(np.abs(z) <= 1.64)),
        "corr_sigma_abs_error": float(np.corrcoef(sigma[idx], abs_err)[0, 1]) if idx.size > 1 else float("nan"),
        "sharpness_mean": float(np.nanmean(sigma[idx])),
        "sharpness_median": float(np.nanmedian(sigma[idx])),
        "sharpness_p90": float(np.nanquantile(sigma[idx], 0.90)),
    }
    return metrics


def _normal_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(math.pi))


def _normal_sigma_metrics(
    *,
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    idx = np.where(mask)[0]
    metrics: dict[str, Any] = {}
    if idx.size == 0:
        return metrics
    crps_vals = _normal_crps(y[idx], mu[idx], sigma[idx])
    metrics["crps"] = float(np.nanmean(crps_vals))

    z = (y[idx] - mu[idx]) / sigma[idx]
    pit = stats.norm.cdf(z)
    pit_stats = _pit_stats(pit)
    metrics["pit_chi2"] = pit_stats.get("chi2")
    metrics["pit_bins"] = pit_stats.get("hist_bins")

    z50 = stats.norm.ppf(0.75)
    z80 = stats.norm.ppf(0.90)
    z90 = stats.norm.ppf(0.95)
    intervals = {
        "p50": _interval_metrics(y[idx], mu[idx] - z50 * sigma[idx], mu[idx] + z50 * sigma[idx]),
        "p80": _interval_metrics(y[idx], mu[idx] - z80 * sigma[idx], mu[idx] + z80 * sigma[idx]),
        "p90": _interval_metrics(y[idx], mu[idx] - z90 * sigma[idx], mu[idx] + z90 * sigma[idx]),
    }
    metrics["intervals"] = intervals

    def cdf_at(thr: float) -> np.ndarray:
        zthr = (thr - mu[idx]) / sigma[idx]
        return stats.norm.cdf(zthr)

    metrics["events"] = _event_metrics_from_cdf(
        y[idx], mu[idx], cdf_at, thresholds=[80.0, 85.0, 90.0]
    )

    abs_err = np.abs(y[idx] - mu[idx])
    metrics["sigma_diagnostics"] = {
        "z_mean": float(np.nanmean(z)),
        "z_std": float(np.nanstd(z)),
        "p_abs_z_le_1_28": float(np.mean(np.abs(z) <= 1.28)),
        "p_abs_z_le_1_64": float(np.mean(np.abs(z) <= 1.64)),
        "corr_sigma_abs_error": float(np.corrcoef(sigma[idx], abs_err)[0, 1]) if idx.size > 1 else float("nan"),
        "sharpness_mean": float(np.nanmean(sigma[idx])),
        "sharpness_median": float(np.nanmedian(sigma[idx])),
        "sharpness_p90": float(np.nanquantile(sigma[idx], 0.90)),
    }
    return metrics


def _softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _trim_mean(values: np.ndarray, trim: float = 0.1) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan")
    n = v.size
    k = int(n * trim)
    if n - 2 * k <= 0:
        return float(np.mean(v))
    v = np.sort(v)
    return float(np.mean(v[k : n - k]))


def _sigma_nll_obj_factory(sigma_floor: float):
    def _obj(preds: np.ndarray, train_data: lgb.Dataset):
        err = train_data.get_label().astype(float)
        s = _softplus(preds) + sigma_floor
        a = _sigmoid(preds)
        b = a * (1.0 - a)
        g_s = 1.0 / s - (err**2) / (s**3)
        dg_s_ds = -1.0 / (s**2) + 3.0 * (err**2) / (s**4)
        grad = g_s * a
        hess = dg_s_ds * a * a + g_s * b
        hess = np.maximum(hess, 1e-6)
        return grad, hess

    return _obj


def _sigma_nll_eval_factory(sigma_floor: float):
    def _eval(preds: np.ndarray, train_data: lgb.Dataset):
        err = train_data.get_label().astype(float)
        s = _softplus(preds) + sigma_floor
        nll = np.log(s) + (err**2) / (2.0 * s**2)
        return "nll", float(np.mean(nll)), False

    return _eval


def _train_sigma_core_model(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sigma_floor: float,
    seed: int,
    weights: np.ndarray | None = None,
) -> lgb.Booster:
    params = {
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_depth": -1,
        "verbosity": -1,
        "seed": seed,
        "objective": _sigma_nll_obj_factory(sigma_floor),
    }
    train_data = lgb.Dataset(X_train, label=y_train, weight=weights, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
    booster = lgb.train(
        params,
        train_set=train_data,
        num_boost_round=5000,
        valid_sets=[val_data],
        feval=_sigma_nll_eval_factory(sigma_floor),
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    return booster


def _predict_sigma_raw(model: lgb.Booster, X: np.ndarray, sigma_floor: float) -> np.ndarray:
    raw = model.predict(X, num_iteration=model.best_iteration)
    return _softplus(raw) + sigma_floor


def _mixture_crps(y: np.ndarray, mu: np.ndarray, s1: np.ndarray, s2: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    mu = mu.astype(float)
    s1 = np.maximum(s1.astype(float), 1e-6)
    s2 = np.maximum(s2.astype(float), 1e-6)
    p = np.clip(p.astype(float), 0.0, 1.0)
    w1 = 1.0 - p
    w2 = p
    d = y - mu

    def _A(dval: np.ndarray, s: np.ndarray) -> np.ndarray:
        z = dval / s
        return 2.0 * s * stats.norm.pdf(z) + dval * (2.0 * stats.norm.cdf(z) - 1.0)

    term1 = w1 * _A(d, s1) + w2 * _A(d, s2)
    a11 = _A(np.zeros_like(d), np.sqrt(2.0) * s1)
    a22 = _A(np.zeros_like(d), np.sqrt(2.0) * s2)
    a12 = _A(np.zeros_like(d), np.sqrt(s1**2 + s2**2))
    term2 = 0.5 * (w1**2 * a11 + 2.0 * w1 * w2 * a12 + w2**2 * a22)
    return term1 - term2


def _mixture_cdf(x: np.ndarray, mu: np.ndarray, s1: np.ndarray, s2: np.ndarray, p: np.ndarray) -> np.ndarray:
    z1 = (x - mu) / s1
    z2 = (x - mu) / s2
    return (1.0 - p) * stats.norm.cdf(z1) + p * stats.norm.cdf(z2)


def _mixture_quantile_vectorized(
    mu: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    p: np.ndarray,
    alpha: float,
    *,
    iterations: int = 60,
) -> np.ndarray:
    mu = mu.astype(float)
    s1 = np.maximum(s1.astype(float), 1e-6)
    s2 = np.maximum(s2.astype(float), 1e-6)
    p = np.clip(p.astype(float), 0.0, 1.0)
    scale = np.maximum(s1, s2)
    lo = mu - 20.0 * scale
    hi = mu + 20.0 * scale
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        cdf = _mixture_cdf(mid, mu, s1, s2, p)
        lo = np.where(cdf < alpha, mid, lo)
        hi = np.where(cdf >= alpha, mid, hi)
    return (lo + hi) / 2.0


def _mixture_metrics(
    *,
    y: np.ndarray,
    mu: np.ndarray,
    sigma_core: np.ndarray,
    sigma_tail: np.ndarray,
    p_bust: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    idx = np.where(mask)[0]
    metrics: dict[str, Any] = {}
    if idx.size == 0:
        return metrics
    y_m = y[idx]
    mu_m = mu[idx]
    s1 = sigma_core[idx]
    s2 = sigma_tail[idx]
    p = p_bust[idx]

    crps_vals = _mixture_crps(y_m, mu_m, s1, s2, p)
    metrics["crps"] = float(np.nanmean(crps_vals))

    pit_vals = _mixture_cdf(y_m, mu_m, s1, s2, p)
    pit_stats = _pit_stats(pit_vals)
    metrics["pit_chi2"] = pit_stats.get("chi2")
    metrics["pit_bins"] = pit_stats.get("hist_bins")

    q05 = _mixture_quantile_vectorized(mu_m, s1, s2, p, 0.05)
    q10 = _mixture_quantile_vectorized(mu_m, s1, s2, p, 0.10)
    q25 = _mixture_quantile_vectorized(mu_m, s1, s2, p, 0.25)
    q50 = _mixture_quantile_vectorized(mu_m, s1, s2, p, 0.50)
    q75 = _mixture_quantile_vectorized(mu_m, s1, s2, p, 0.75)
    q90 = _mixture_quantile_vectorized(mu_m, s1, s2, p, 0.90)
    q95 = _mixture_quantile_vectorized(mu_m, s1, s2, p, 0.95)

    metrics["intervals"] = {
        "p50": _interval_metrics(y_m, q25, q75),
        "p80": _interval_metrics(y_m, q10, q90),
        "p90": _interval_metrics(y_m, q05, q95),
    }

    def cdf_at(thr: float) -> np.ndarray:
        x = np.full_like(mu_m, thr, dtype=float)
        return _mixture_cdf(x, mu_m, s1, s2, p)

    metrics["events"] = _event_metrics_from_cdf(y_m, mu_m, cdf_at, thresholds=[80.0, 85.0, 90.0])

    err = y_m - mu_m
    sigma_eff = np.sqrt((1.0 - p) * s1**2 + p * s2**2)
    z = err / sigma_eff
    abs_err = np.abs(err)
    metrics["sigma_diagnostics"] = {
        "z_mean": float(np.nanmean(z)),
        "z_std": float(np.nanstd(z)),
        "p_abs_z_le_1_28": float(np.mean(np.abs(z) <= 1.28)),
        "p_abs_z_le_1_64": float(np.mean(np.abs(z) <= 1.64)),
        "corr_sigma_abs_error": float(
            np.corrcoef(sigma_eff, abs_err)[0, 1] if idx.size > 1 else float("nan")
        ),
        "sharpness_mean": float(np.nanmean(sigma_eff)),
        "sharpness_median": float(np.nanmedian(sigma_eff)),
        "sharpness_p90": float(np.nanquantile(sigma_eff, 0.90)),
    }
    return metrics


def _online_rms_scale(
    *,
    err: np.ndarray,
    sigma_raw: np.ndarray,
    dates: np.ndarray,
    window_days: int,
    trim: float,
    k_min: float,
    k_max: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    n = len(dates)
    k_vals = np.ones(n, dtype=float)
    max_window_date = np.array([None] * n, dtype=object)
    violations = 0
    for i, d in enumerate(dates):
        start = d - timedelta(days=window_days)
        end = d - timedelta(days=1)
        mask = (dates >= start) & (dates <= end)
        if mask.sum() < window_days:
            max_window_date[i] = max(dates[mask]) if mask.any() else None
            continue
        window_err = err[mask]
        window_sig = sigma_raw[mask]
        e2 = window_err**2
        s2 = window_sig**2
        e_mean = _trim_mean(e2, trim=trim)
        s_mean = _trim_mean(s2, trim=trim)
        if np.isfinite(e_mean) and np.isfinite(s_mean) and s_mean > 0:
            k = math.sqrt(e_mean / max(s_mean, 1e-6))
            k = float(np.clip(k, k_min, k_max))
            k_vals[i] = k
        max_d = max(dates[mask]) if mask.any() else None
        max_window_date[i] = max_d
        if max_d is not None and max_d >= d:
            violations += 1
    return k_vals, max_window_date, violations


def _prepare_sigma_v2_features(
    df: pd.DataFrame,
    *,
    mu_used: np.ndarray,
    y_vals: np.ndarray,
    dates: np.ndarray,
    rev_cols: list[str],
    minute_cols: list[str],
) -> tuple[pd.DataFrame, list[str], int]:
    n = len(df)
    err = y_vals - mu_used
    abs_err = np.abs(err)
    sq_err = err**2

    order = np.argsort(dates)
    err_s = pd.Series(err[order])
    abs_err_s = pd.Series(abs_err[order])
    sq_err_s = pd.Series(sq_err[order])

    err_lag1 = err_s.shift(1).to_numpy()
    abs_err_lag1 = abs_err_s.shift(1).to_numpy()
    abs_err_ewma_15 = abs_err_s.shift(1).ewm(span=15, adjust=False).mean().to_numpy()
    abs_err_ewma_45 = abs_err_s.shift(1).ewm(span=45, adjust=False).mean().to_numpy()
    sq_err_ewma_45 = sq_err_s.shift(1).ewm(span=45, adjust=False).mean().to_numpy()

    def _scatter(arr: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan, dtype=float)
        out[order] = arr
        return out

    df["err_lag1"] = _scatter(err_lag1)
    df["abs_err_lag1"] = _scatter(abs_err_lag1)
    df["abs_err_ewma_15"] = _scatter(abs_err_ewma_15)
    df["abs_err_ewma_45"] = _scatter(abs_err_ewma_45)
    df["sq_err_ewma_45"] = _scatter(sq_err_ewma_45)

    if rev_cols:
        rev_num = df[rev_cols].apply(pd.to_numeric, errors="coerce")
        df["has_mos_revision_feats"] = rev_num.notna().any(axis=1).astype(int)
    else:
        df["has_mos_revision_feats"] = 0

    if minute_cols:
        minute_num = df[minute_cols].apply(pd.to_numeric, errors="coerce")
        df["has_minute_feats"] = minute_num.notna().any(axis=1).astype(int)
    else:
        df["has_minute_feats"] = 0

    exclude = {
        "target_date_local",
        "y",
        "y_actual_tmax_f",
        "V5+8",
        "mu_pred_final",
        "mu_pred_oof",
        "err",
        "abs_err",
        "sq_err",
        "asof_date_local",
        "decision_utc",
        "max_ts_utc_t1",
        "max_ts_utc_early",
        "max_minute_ts_used_utc",
        "leak_violation",
    }
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    sigma_features = []
    for c in numeric_cols:
        if c in exclude:
            continue
        if c.startswith("pred_"):
            continue
        sigma_features.append(c)

    # lag leakage audit (err_lag1 must come from past day)
    lag_dates = pd.Series(dates[order]).shift(1).to_numpy()
    lag_dates_full = np.array([None] * n, dtype=object)
    lag_dates_full[order] = lag_dates
    violations = 0
    for i, d in enumerate(dates):
        src = lag_dates_full[i]
        if src is None:
            continue
        if src >= d:
            violations += 1

    return df, sigma_features, violations

@dataclass
class MethodResult:
    name: str
    val: dict[str, Any]
    test: dict[str, Any]


def _event_metrics_from_cdf(
    y: np.ndarray, pred: np.ndarray, cdf_fn, thresholds: list[float]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for thr in thresholds:
        cdf = cdf_fn(thr)
        prob_ge = 1.0 - cdf
        y_event = (y >= thr).astype(int)
        out[f"ge_{int(thr)}"] = {
            "brier": _brier(y_event, prob_ge),
            "log_loss": _log_loss_binary(y_event, prob_ge),
            "reliability": _reliability_table(prob_ge, y_event, bins=10),
        }
    return out


def build_conformal_distribution(
    df: pd.DataFrame,
    *,
    pred_col: str,
    y_col: str,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> MethodResult:
    y = df[y_col].to_numpy(dtype=float)
    pred = df[pred_col].to_numpy(dtype=float)
    onshore = (pd.to_numeric(df["feat_onshore"], errors="coerce") > 0.5).to_numpy(dtype=bool)
    suppress = pd.to_numeric(df["MRI_suppress"], errors="coerce").to_numpy(dtype=float)
    suppress_thr = float(np.nanquantile(suppress[val_mask], 0.70))
    high_suppress = suppress >= suppress_thr

    residuals = y - pred
    calib_resid = residuals[val_mask]
    regimes = {
        "onshore_high": (onshore & high_suppress),
        "onshore_low": (onshore & ~high_suppress),
        "offshore_high": (~onshore & high_suppress),
        "offshore_low": (~onshore & ~high_suppress),
    }

    resid_by_regime: dict[str, np.ndarray] = {}
    for name, mask in regimes.items():
        r = residuals[mask & val_mask]
        r = r[np.isfinite(r)]
        resid_by_regime[name] = r

    global_resid = calib_resid[np.isfinite(calib_resid)]
    min_needed = 50

    def _select_resid(idx: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return global_resid
        if idx.sum() < min_needed:
            return global_resid
        r = residuals[idx & val_mask]
        r = r[np.isfinite(r)]
        return r if len(r) >= min_needed else global_resid

    def _metrics(mask: np.ndarray) -> dict[str, Any]:
        idx = np.where(mask)[0]
        q_levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        q_preds = {q: np.full(idx.size, np.nan) for q in q_levels}
        pits = []
        crps_vals = []
        for i, row_idx in enumerate(idx):
            reg = None
            for name, m in regimes.items():
                if m[row_idx]:
                    reg = name
                    break
            r = resid_by_regime.get(reg, global_resid)
            if r is None or len(r) < min_needed:
                r = global_resid
            if len(r) == 0:
                continue
            qs = np.quantile(r, q_levels)
            for q, val in zip(q_levels, qs):
                q_preds[q][i] = pred[row_idx] + val
            val = y[row_idx] - pred[row_idx]
            rank = np.searchsorted(np.sort(r), val, side="right")
            pit = (rank - 0.5) / max(len(r), 1)
            pits.append(pit)
            crps_vals.append(_crps_empirical(r, y[row_idx : row_idx + 1], pred[row_idx : row_idx + 1])[0])

        metrics: dict[str, Any] = {
            "pinball": {str(q): _pinball(y[idx], q_preds[q], q) for q in q_levels},
            "intervals": {
                "p50": _interval_metrics(y[idx], q_preds[0.25], q_preds[0.75]),
                "p80": _interval_metrics(y[idx], q_preds[0.10], q_preds[0.90]),
                "p90": _interval_metrics(y[idx], q_preds[0.05], q_preds[0.95]),
            },
            "pit": _pit_stats(np.array(pits)),
            "crps": float(np.nanmean(crps_vals)) if len(crps_vals) else float("nan"),
        }

        def cdf_at(thr: float) -> np.ndarray:
            out = np.zeros(idx.size, dtype=float)
            for j, row_idx in enumerate(idx):
                reg = None
                for name, m in regimes.items():
                    if m[row_idx]:
                        reg = name
                        break
                r = resid_by_regime.get(reg, global_resid)
                if r is None or len(r) < min_needed:
                    r = global_resid
                if len(r) == 0:
                    out[j] = np.nan
                    continue
                out[j] = np.mean(r <= (thr - pred[row_idx]))
            return out

        metrics["events"] = _event_metrics_from_cdf(
            y[idx], pred[idx], cdf_at, thresholds=[80.0, 85.0, 90.0]
        )
        return metrics

    val_metrics = _metrics(val_mask)
    test_metrics = _metrics(test_mask)
    return MethodResult("conformal_residual", val_metrics, test_metrics)


def build_quantile_models(
    df: pd.DataFrame,
    *,
    pred_col: str,
    y_col: str,
    feature_cols: list[str],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    seed: int,
) -> tuple[MethodResult, dict[float, np.ndarray], list[float]]:
    import sys

    sys.path.append("ml")
    import run_mos_45_suite as base

    df_feat = base.ensure_columns(df, feature_cols)
    X_imp, _ = base.impute_features(df_feat[feature_cols], train_mask)
    X = X_imp.to_numpy(dtype=float)

    y = df[y_col].to_numpy(dtype=float)
    pred = df[pred_col].to_numpy(dtype=float)
    resid = y - pred

    q_levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    q_preds: dict[float, np.ndarray] = {}
    for q in q_levels:
        model = base.train_lgbm_quantile(
            X[train_mask],
            resid[train_mask],
            X[val_mask],
            resid[val_mask],
            seed=seed,
            alpha=q,
        )
        q_preds[q] = pred + model.predict(X)

    def _metrics(mask: np.ndarray) -> dict[str, Any]:
        idx = np.where(mask)[0]
        metrics: dict[str, Any] = {
            "pinball": {str(q): _pinball(y[idx], q_preds[q][idx], q) for q in q_levels},
            "intervals": {
                "p50": _interval_metrics(y[idx], q_preds[0.25][idx], q_preds[0.75][idx]),
                "p80": _interval_metrics(y[idx], q_preds[0.10][idx], q_preds[0.90][idx]),
                "p90": _interval_metrics(y[idx], q_preds[0.05][idx], q_preds[0.95][idx]),
            },
            "crps": _crps_from_quantiles(y[idx], {q: q_preds[q][idx] for q in q_levels}),
        }
        pit_vals = []
        for i in idx:
            q_vals = np.array([q_preds[q][i] for q in q_levels], dtype=float)
            pit_vals.append(_cdf_from_quantiles(np.array([y[i]]), q_vals, np.array(q_levels))[0])
        metrics["pit"] = _pit_stats(np.array(pit_vals))

        def cdf_at(thr: float) -> np.ndarray:
            probs = []
            for i in idx:
                q_vals = np.array([q_preds[q][i] for q in q_levels], dtype=float)
                probs.append(_cdf_from_quantiles(np.array([thr]), q_vals, np.array(q_levels))[0])
            return np.array(probs, dtype=float)

        metrics["events"] = _event_metrics_from_cdf(
            y[idx], pred[idx], cdf_at, thresholds=[80.0, 85.0, 90.0]
        )
        return metrics

    return MethodResult("quantile_residual", _metrics(val_mask), _metrics(test_mask)), q_preds, q_levels


def build_cqr_hybrid(
    df: pd.DataFrame,
    *,
    pred_col: str,
    y_col: str,
    q_preds: dict[float, np.ndarray],
    q_levels: list[float],
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    alphas: list[float],
    rearrange: bool = True,
) -> tuple[MethodResult, dict[str, Any]]:
    y = df[y_col].to_numpy(dtype=float)
    dates = pd.to_datetime(df["target_date_local"]).dt.date
    max_val_date = max(dates[val_mask]) if np.any(val_mask) else None
    if max_val_date is None:
        raise ValueError("Validation set is empty; cannot compute CQR taus.")
    if max_val_date > date(2022, 12, 31):
        raise ValueError(f"CQR calibration leakage: max val date {max_val_date} exceeds 2022-12-31.")

    taus: dict[float, float] = {}
    for alpha in alphas:
        lo = q_preds[alpha]
        hi = q_preds[1 - alpha]
        s = np.maximum(lo - y, y - hi)
        s = np.where(s < 0, 0, s)
        s_val = s[val_mask]
        delta = 2 * alpha
        tau = float(np.nanquantile(s_val, 1 - delta))
        taus[alpha] = tau

    def _adjust_preds() -> dict[float, np.ndarray]:
        adj = {q: q_preds[q].copy() for q in q_levels}
        for alpha, tau in taus.items():
            adj[alpha] = adj[alpha] - tau
            adj[1 - alpha] = adj[1 - alpha] + tau
        if rearrange:
            mat = np.vstack([adj[q] for q in q_levels]).T
            mat_sorted = np.sort(mat, axis=1)
            for i, q in enumerate(q_levels):
                adj[q] = mat_sorted[:, i]
        return adj

    adj = _adjust_preds()

    def _metrics(mask: np.ndarray) -> dict[str, Any]:
        idx = np.where(mask)[0]
        metrics: dict[str, Any] = {}
        metrics["crps"] = _crps_from_quantiles(y[idx], {q: adj[q][idx] for q in q_levels})
        pit_vals = []
        for i in idx:
            q_vals = np.array([adj[q][i] for q in q_levels], dtype=float)
            pit_vals.append(_cdf_from_quantiles(np.array([y[i]]), q_vals, np.array(q_levels))[0])
        pit = _pit_stats(np.array(pit_vals))
        metrics["pit_chi2"] = pit.get("chi2")
        metrics["pit_bins"] = pit.get("hist_bins")
        cov80 = _interval_metrics(y[idx], adj[0.10][idx], adj[0.90][idx])
        cov90 = _interval_metrics(y[idx], adj[0.05][idx], adj[0.95][idx])
        metrics["coverage_p80"] = cov80["coverage"]
        metrics["coverage_p90"] = cov90["coverage"]
        return metrics

    details = {
        "alphas_used": alphas,
        "tau_alpha_val": {f"tau_{alpha:.2f}": taus[alpha] for alpha in alphas},
        "rearrangement": rearrange,
    }
    return MethodResult("cqr_hybrid", _metrics(val_mask), _metrics(test_mask)), details


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate V5+8 point forecast to distributions.")
    parser.add_argument(
        "--preds",
        default="artifacts/experiments/winners/V5_PLUS8_20260219T222321Z/preds.parquet",
    )
    parser.add_argument(
        "--feature-store",
        default="artifacts/experiments/winners/E37_V5_MINUTE_CONDENSED_V1/feature_store_e37_minute_condensed.parquet",
    )
    parser.add_argument(
        "--mos-revisions",
        default="artifacts/experiments/winners/V5_PLUS8_20260219T222321Z/mos_revision_features.parquet",
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("artifacts/experiments") / f"V5_PLUS8_CALIB_{utc_now_tag()}"
    ensure_dir(out_dir)

    preds = pd.read_parquet(Path(args.preds))
    preds["target_date_local"] = to_date(preds["target_date_local"])

    fs = pd.read_parquet(Path(args.feature_store))
    fs["target_date_local"] = to_date(fs["target_date_local"])

    df = preds.merge(fs, on="target_date_local", how="left")

    rev_path = Path(args.mos_revisions)
    if rev_path.exists():
        rev = pd.read_parquet(rev_path)
        rev["target_date_local"] = to_date(rev["target_date_local"])
        df = df.merge(rev, on="target_date_local", how="left")

    masks = split_masks(pd.to_datetime(df["target_date_local"]))
    train_mask = masks["train_mask"]
    val_mask = masks["val_mask"]
    test_mask = masks["test_mask"]

    pred_col = "V5+8"
    y_col = "y"
    if pred_col not in df.columns or y_col not in df.columns:
        raise ValueError("preds.parquet must contain columns 'y' and 'V5+8'.")

    # Build mu predictions (final + OOF) for sigma modeling
    mu_final, mu_oof, oof_folds, oof_violations = _build_mu_predictions(df, seed=args.seed)
    df["mu_pred_final"] = mu_final
    df["mu_pred_oof"] = mu_oof

    dates = pd.to_datetime(df["target_date_local"]).dt.date.to_numpy()
    mu_used = np.where(dates <= date(2019, 12, 31), mu_oof, mu_final)

    expert_features_base = [
        "feat_dd_models",
        "feat_tmp_range_mean_models",
        "feat_p12_max",
        "feat_q12_max",
        "feat_cig_min",
        "feat_u",
        "feat_v",
        "cal_d_doy_sin",
        "cal_d_doy_cos",
    ]
    minute_all = [
        "iem_tmax_t1",
        "iem_tmin_t1",
        "iem_range_t1",
        "tmax_time_min_t1",
        "plateau_05_t1",
        "heat_12_15_t1",
        "heat_15_18_t1",
        "cool_18_21_t1",
        "max_drop_30_t1",
        "drop_cnt_15_19_t1",
        "T00",
        "T03",
        "T06",
        "night_drop_00_06",
        "slope_last180",
        "std_last180",
        "T06_adj",
        "diff_lag1",
        "diff_ewma_30",
        "diff_std_30",
        "MRI_suppress",
        "MRI_late",
    ]
    rev_features = [
        "abs_rev24_tmp_gfs",
        "abs_rev24_tmp_nam",
        "abs_rev24_q12_gfs",
        "abs_rev24_q12_nam",
        "abs_rev24_cig_gfs",
        "abs_rev24_cig_nam",
        "abs_disc0_tmp",
        "abs_disc0_q12",
        "abs_disc0_cig",
    ]
    feature_cols = expert_features_base + minute_all + rev_features

    conf = build_conformal_distribution(
        df,
        pred_col=pred_col,
        y_col=y_col,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    quant, q_preds, q_levels = build_quantile_models(
        df,
        pred_col=pred_col,
        y_col=y_col,
        feature_cols=feature_cols,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        seed=args.seed,
    )
    cqr, cqr_details = build_cqr_hybrid(
        df,
        pred_col=pred_col,
        y_col=y_col,
        q_preds=q_preds,
        q_levels=q_levels,
        val_mask=val_mask,
        test_mask=test_mask,
        alphas=[0.05, 0.10],
        rearrange=True,
    )

    # --- Student-t sigma model (w=45) ---
    if "feat_le_median_biascorr" not in df.columns:
        raise ValueError("feat_le_median_biascorr missing from feature store.")

    df["feat_base"] = pd.to_numeric(df["feat_le_median_biascorr"], errors="coerce")
    df["feat_mu"] = mu_used
    df["feat_corr"] = df["feat_mu"] - df["feat_base"]
    df["feat_abs_corr"] = df["feat_corr"].abs()

    y_vals = df[y_col].to_numpy(dtype=float)

    minute_cols = [
        "iem_tmax_t1",
        "iem_tmin_t1",
        "iem_range_t1",
        "tmax_time_min_t1",
        "plateau_05_t1",
        "heat_12_15_t1",
        "heat_15_18_t1",
        "cool_18_21_t1",
        "max_drop_30_t1",
        "drop_cnt_15_19_t1",
        "T00",
        "T03",
        "T06",
        "night_drop_00_06",
        "slope_last180",
        "std_last180",
        "T06_adj",
        "diff_lag1",
        "diff_ewma_30",
        "diff_std_30",
        "MRI_suppress",
        "MRI_late",
    ]
    minute_cols = [c for c in minute_cols if c in df.columns]
    rev_cols = [c for c in df.columns if c.startswith("rev_") or c.startswith("disc_")]

    df, sigma_v2_features, lag_violation = _prepare_sigma_v2_features(
        df,
        mu_used=mu_used,
        y_vals=y_vals,
        dates=dates,
        rev_cols=rev_cols,
        minute_cols=minute_cols,
    )

    drop_cols = {
        "y",
        "base",
        "V5",
        "V5+1",
        "V5+2",
        "V5+3",
        "V5+4",
        "V5+5",
        "V5+6",
        "V5+7",
        "V5+8",
        "y_actual_tmax_f",
        "mu_pred_final",
        "mu_pred_oof",
    }
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols]
    sigma_features = []
    for col in numeric_cols + ["feat_mu", "feat_base", "feat_abs_corr", "feat_corr"]:
        if col not in sigma_features:
            sigma_features.append(col)

    # sigma targets
    sigma_train_mask = (dates >= date(2008, 1, 1)) & (dates <= date(2019, 12, 31)) & np.isfinite(mu_oof)
    sigma_val_mask = (dates >= date(2020, 1, 1)) & (dates <= date(2022, 12, 31)) & np.isfinite(mu_final)
    sigma_test_mask = (dates >= date(2023, 1, 1)) & (dates <= date(2025, 12, 31)) & np.isfinite(mu_final)

    resid_used = y_vals - mu_used
    z_all = np.log(np.abs(resid_used) + 0.1)

    X = df[sigma_features].copy()
    X_train = X.loc[sigma_train_mask]
    y_train = z_all[sigma_train_mask]
    X_val = X.loc[sigma_val_mask]
    y_val = z_all[sigma_val_mask]

    sigma_params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_depth": -1,
        "verbosity": -1,
        "seed": 1337,
    }

    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=sigma_features)
    val_ds = lgb.Dataset(X_val, label=y_val, feature_name=sigma_features)
    sigma_model = lgb.train(
        sigma_params,
        train_ds,
        num_boost_round=5000,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )

    sigma_raw = np.exp(sigma_model.predict(X, num_iteration=sigma_model.best_iteration))
    sigma_raw = np.where(np.isfinite(sigma_raw), sigma_raw, np.nan)

    r_vals = np.abs(resid_used) / np.maximum(0.05, sigma_raw)

    # Normal + EMOS sigma calibration (w=45)
    sigma_hat = np.maximum(0.5, sigma_raw)
    sigma_emos = np.full(len(df), np.nan, dtype=float)
    emos_c = np.full(len(df), np.nan, dtype=float)
    emos_d = np.full(len(df), np.nan, dtype=float)
    emos_roll_bias = np.full(len(df), np.nan, dtype=float)
    emos_roll_rmse = np.full(len(df), np.nan, dtype=float)
    emos_window_violations = 0
    max_emos_window_date = np.array([None] * len(df), dtype=object)

    for i, d in enumerate(dates):
        if not (sigma_val_mask[i] or sigma_test_mask[i]):
            continue
        start = d - timedelta(days=45)
        end = d - timedelta(days=1)
        mask = (dates >= start) & (dates <= end)
        if not mask.any():
            continue
        max_d = max(dates[mask])
        max_emos_window_date[i] = max_d
        if max_d >= d:
            emos_window_violations += 1
        hist_df = pd.DataFrame(
            {
                "actual_tmax_f": y_vals[mask],
                "mu_hat_f": mu_used[mask],
                "sigma_hat_f": sigma_hat[mask],
            }
        ).dropna()
        if len(hist_df) < 45:
            sigma_emos[i] = sigma_hat[i]
            continue
        result = emos_calibrate(hist_df, float(sigma_hat[i]), sigma_floor=0.5)
        sigma_emos[i] = result.sigma_emos
        emos_c[i] = result.c
        emos_d[i] = result.d
        emos_roll_bias[i] = result.rolling_bias
        emos_roll_rmse[i] = result.rolling_rmse

    sigma_emos_final = np.where(np.isfinite(sigma_emos), sigma_emos, sigma_hat)

    # rolling median ratio for calibration
    median_r = np.full(len(df), np.nan, dtype=float)
    max_window_date = np.array([None] * len(df), dtype=object)
    window_violations = 0
    for i, d in enumerate(dates):
        if not (sigma_val_mask[i] or sigma_test_mask[i]):
            continue
        start = d - timedelta(days=45)
        end = d - timedelta(days=1)
        mask = (dates >= start) & (dates <= end)
        if not mask.any():
            continue
        window_r = r_vals[mask]
        window_r = window_r[np.isfinite(window_r)]
        if window_r.size == 0:
            continue
        median_r[i] = float(np.nanmedian(window_r))
        max_d = max(dates[mask])
        max_window_date[i] = max_d
        if max_d >= d:
            window_violations += 1

    # df selection via validation CRPS
    candidate_dfs = [4, 6, 8, 10, 15]
    df_table: list[dict[str, Any]] = []
    best_df = None
    best_key = None
    for nu in candidate_dfs:
        m0 = float(stats.t.ppf(0.75, nu))
        g = median_r / m0
        sigma_cal = np.maximum(0.10, g * sigma_raw)
        metrics_val = _studentt_metrics(
            y=y_vals,
            mu=mu_final,
            sigma=sigma_cal,
            nu=nu,
            mask=sigma_val_mask,
        )
        crps = metrics_val.get("crps", float("inf"))
        cov90 = metrics_val.get("intervals", {}).get("p90", {}).get("coverage", np.nan)
        pit_chi2 = metrics_val.get("pit_chi2", np.nan)
        df_table.append(
            {
                "df": nu,
                "val_crps": crps,
                "val_coverage_p90": cov90,
                "val_pit_chi2": pit_chi2,
            }
        )
        key = (crps, abs(cov90 - 0.90), pit_chi2 if pit_chi2 is not None else 1e9)
        if best_key is None or key < best_key:
            best_key = key
            best_df = nu

    df_selected = int(best_df) if best_df is not None else 8

    # final sigma calibration with selected df
    m0_sel = float(stats.t.ppf(0.75, df_selected))
    g_sel = median_r / m0_sel
    sigma_cal = np.maximum(0.10, g_sel * sigma_raw)

    studentt_val = _studentt_metrics(
        y=y_vals,
        mu=mu_final,
        sigma=sigma_cal,
        nu=df_selected,
        mask=sigma_val_mask,
    )
    studentt_test = _studentt_metrics(
        y=y_vals,
        mu=mu_final,
        sigma=sigma_cal,
        nu=df_selected,
        mask=sigma_test_mask,
    )

    studentt = MethodResult("studentt_sigma_w45", studentt_val, studentt_test)

    normal_emos_val = _normal_sigma_metrics(
        y=y_vals,
        mu=mu_final,
        sigma=sigma_emos_final,
        mask=sigma_val_mask,
    )
    normal_emos_test = _normal_sigma_metrics(
        y=y_vals,
        mu=mu_final,
        sigma=sigma_emos_final,
        mask=sigma_test_mask,
    )
    normal_emos = MethodResult("normal_sigma_emos_w45", normal_emos_val, normal_emos_test)

    # Sigma V2: bust-aware mixture model
    sigma_v2_candidates: list[dict[str, Any]] = []
    best_cfg: dict[str, Any] | None = None
    best_val_metrics: dict[str, Any] | None = None
    best_core_model: lgb.Booster | None = None
    best_bust_model: lgb.LGBMClassifier | None = None
    best_iso: IsotonicRegression | None = None
    best_sigma_core_cal: np.ndarray | None = None
    best_sigma_tail_cal: np.ndarray | None = None
    best_p_bust_cal: np.ndarray | None = None
    best_window_violation = 0
    best_max_window_date: np.ndarray | None = None

    sigma_train_mask = (dates >= date(2008, 1, 1)) & (dates <= date(2019, 12, 31)) & np.isfinite(mu_used)
    sigma_val_mask = val_mask
    sigma_test_mask = test_mask

    X_all = df[sigma_v2_features].to_numpy(dtype=float)
    X_train = X_all[sigma_train_mask]
    X_val = X_all[sigma_val_mask]
    abs_err_all = np.abs(resid_used)
    err_all = resid_used

    bust_thresholds = [1.5, 2.0, 2.5, 3.0]
    tail_multipliers = [1.5, 2.0, 2.5, 3.0, 3.5]
    p_max_list = [0.30, 0.45, 0.60]
    k_clamps = [(0.7, 1.4), (0.6, 1.6)]
    sigma_floors = [0.1, 0.2]
    core_modes = ["nonbust", "weighted"]

    def _choose_best(cands: list[dict[str, Any]]) -> dict[str, Any] | None:
        valid = [
            c
            for c in cands
            if c["val_pit_chi2"] <= 60.0 and 0.88 <= c["val_p90_coverage"] <= 0.92
        ]
        if not valid:
            valid = [
                c
                for c in cands
                if c["val_pit_chi2"] <= 90.0 and 0.87 <= c["val_p90_coverage"] <= 0.93
            ]
        if not valid:
            return None
        return min(valid, key=lambda c: c["val_crps"])

    for B in bust_thresholds:
        is_bust = (abs_err_all >= B).astype(int)
        y_bust_train = is_bust[sigma_train_mask]
        y_bust_val = is_bust[sigma_val_mask]
        bust_model = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=100,
            feature_fraction=0.85,
            bagging_fraction=0.85,
            bagging_freq=1,
            n_estimators=2000,
            random_state=args.seed,
        )
        bust_model.fit(
            X_train,
            y_bust_train,
            eval_set=[(X_val, y_bust_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        p_bust_raw = bust_model.predict_proba(X_all)[:, 1]

        iso = IsotonicRegression(out_of_bounds="clip")
        if np.unique(y_bust_val).size >= 2:
            iso.fit(p_bust_raw[sigma_val_mask], y_bust_val)
            p_bust_cal = iso.predict(p_bust_raw)
        else:
            p_bust_cal = p_bust_raw.copy()

        for core_mode in core_modes:
            for sigma_floor in sigma_floors:
                if core_mode == "nonbust":
                    train_mask_core = sigma_train_mask & (abs_err_all < B)
                    weights = None
                else:
                    train_mask_core = sigma_train_mask
                    weights = 1.0 - p_bust_raw[sigma_train_mask]

                X_train_core = X_all[train_mask_core]
                y_train_core = err_all[train_mask_core]
                X_val_core = X_all[sigma_val_mask]
                y_val_core = err_all[sigma_val_mask]

                sigma_core_model = _train_sigma_core_model(
                    X_train=X_train_core,
                    y_train=y_train_core,
                    X_val=X_val_core,
                    y_val=y_val_core,
                    sigma_floor=sigma_floor,
                    seed=args.seed,
                    weights=weights,
                )
                sigma_core_raw = _predict_sigma_raw(sigma_core_model, X_all, sigma_floor)

                for k_min, k_max in k_clamps:
                    k_vals, max_window_date, window_viol = _online_rms_scale(
                        err=err_all,
                        sigma_raw=sigma_core_raw,
                        dates=dates,
                        window_days=45,
                        trim=0.10,
                        k_min=k_min,
                        k_max=k_max,
                    )
                    sigma_core_cal = sigma_core_raw * k_vals

                    for p_max in p_max_list:
                        p_bust_clip = np.clip(p_bust_cal, 0.01, p_max)
                        for m_tail in tail_multipliers:
                            sigma_tail_raw = np.maximum(sigma_core_raw, m_tail * sigma_core_raw)
                            sigma_tail_cal = sigma_tail_raw * k_vals

                            val_idx = np.where(sigma_val_mask)[0]
                            y_m = y_vals[val_idx]
                            mu_m = mu_final[val_idx]
                            s1_m = sigma_core_cal[val_idx]
                            s2_m = sigma_tail_cal[val_idx]
                            p_m = p_bust_clip[val_idx]
                            val_crps = float(np.nanmean(_mixture_crps(y_m, mu_m, s1_m, s2_m, p_m)))
                            pit_vals = _mixture_cdf(y_m, mu_m, s1_m, s2_m, p_m)
                            pit_stats = _pit_stats(pit_vals)
                            val_pit = float(pit_stats.get("chi2", float("inf")))
                            q05 = _mixture_quantile_vectorized(mu_m, s1_m, s2_m, p_m, 0.05, iterations=25)
                            q95 = _mixture_quantile_vectorized(mu_m, s1_m, s2_m, p_m, 0.95, iterations=25)
                            val_cov = float(_interval_metrics(y_m, q05, q95).get("coverage", float("nan")))
                            sigma_v2_candidates.append(
                                {
                                    "B": B,
                                    "core_mode": core_mode,
                                    "sigma_floor": sigma_floor,
                                    "k_min": k_min,
                                    "k_max": k_max,
                                    "p_max": p_max,
                                    "m_tail": m_tail,
                                    "val_crps": val_crps,
                                    "val_pit_chi2": val_pit,
                                    "val_p90_coverage": val_cov,
                                }
                            )

                            cfg = _choose_best(sigma_v2_candidates)
                            if cfg is None:
                                continue
                            if best_cfg is None or cfg["val_crps"] < best_cfg["val_crps"]:
                                best_cfg = cfg
                                best_val_metrics = {
                                    "crps": val_crps,
                                    "pit_chi2": val_pit,
                                    "p90_coverage": val_cov,
                                }
                                best_core_model = sigma_core_model
                                best_bust_model = bust_model
                                best_iso = iso
                                best_sigma_core_cal = sigma_core_cal
                                best_sigma_tail_cal = sigma_tail_cal
                                best_p_bust_cal = p_bust_clip
                                best_window_violation = window_viol
                                best_max_window_date = max_window_date

    if best_cfg is None or best_sigma_core_cal is None or best_sigma_tail_cal is None or best_p_bust_cal is None:
        raise ValueError("Sigma V2 grid search failed to find a valid configuration.")

    sigma_v2_val = _mixture_metrics(
        y=y_vals,
        mu=mu_final,
        sigma_core=best_sigma_core_cal,
        sigma_tail=best_sigma_tail_cal,
        p_bust=best_p_bust_cal,
        mask=sigma_val_mask,
    )
    sigma_v2_test = _mixture_metrics(
        y=y_vals,
        mu=mu_final,
        sigma_core=best_sigma_core_cal,
        sigma_tail=best_sigma_tail_cal,
        p_bust=best_p_bust_cal,
        mask=sigma_test_mask,
    )
    mog_sigma = MethodResult("mog_sigma_bust_w45", sigma_v2_val, sigma_v2_test)

    gaussian_val = _normal_sigma_metrics(
        y=y_vals,
        mu=mu_final,
        sigma=best_sigma_core_cal,
        mask=sigma_val_mask,
    )
    gaussian_test = _normal_sigma_metrics(
        y=y_vals,
        mu=mu_final,
        sigma=best_sigma_core_cal,
        mask=sigma_test_mask,
    )
    gaussian_sigma = MethodResult("gaussian_sigma_nll_rms_w45", gaussian_val, gaussian_test)

    methods = [conf, quant, cqr, studentt, normal_emos, gaussian_sigma, mog_sigma]
    best = min(methods, key=lambda m: m.val.get("crps", float("inf")))

    # write sigma artifacts
    if oof_violations > 0:
        raise ValueError(f"OOF μ leakage violations detected: {oof_violations}")
    if window_violations > 0:
        raise ValueError(f"Online sigma calibration window violations: {window_violations}")
    if emos_window_violations > 0:
        raise ValueError(f"EMOS window violations: {emos_window_violations}")

    sigma_dir = Path("artifacts/experiments/winners") / f"STUDENTT_SIGMA_W45_{utc_now_tag()}"
    ensure_dir(sigma_dir)
    sigma_model.save_model(str(sigma_dir / "sigma_lgbm.txt"))
    (sigma_dir / "sigma_features.json").write_text(
        json.dumps({"features": sigma_features}, indent=2), encoding="utf-8"
    )
    _ensure_mu_oof_manifest(oof_folds, oof_violations, sigma_dir / "oof_mu_manifest.json")
    (sigma_dir / "df_selection_table.json").write_text(
        json.dumps(df_table, indent=2), encoding="utf-8"
    )

    sigma_daily = pd.DataFrame(
        {
            "target_date_local": df["target_date_local"],
            "mu_used": mu_used,
            "y": y_vals,
            "sigma_raw": sigma_raw,
            "sigma_cal": sigma_cal,
            "g": g_sel,
            "max_cal_window_date_used": max_window_date,
        }
    )
    sigma_daily = sigma_daily.loc[sigma_val_mask | sigma_test_mask]
    sigma_daily.to_parquet(sigma_dir / "sigma_daily.parquet", index=False)

    sigma_v2_dir = Path("artifacts/experiments/winners") / f"SIGMA_V2_{utc_now_tag()}"
    ensure_dir(sigma_v2_dir)
    if best_core_model is not None:
        joblib.dump(best_core_model, sigma_v2_dir / "sigma_core_model.joblib")
    if best_bust_model is not None:
        joblib.dump(best_bust_model, sigma_v2_dir / "bust_prob_model.joblib")
    if best_iso is not None:
        joblib.dump(best_iso, sigma_v2_dir / "bust_prob_isotonic.joblib")
    (sigma_v2_dir / "sigma_feature_columns.json").write_text(
        json.dumps({"features": sigma_v2_features}, indent=2), encoding="utf-8"
    )
    (sigma_v2_dir / "sigma_tail_multiplier.json").write_text(
        json.dumps({"m_tail": best_cfg["m_tail"]}, indent=2), encoding="utf-8"
    )
    (sigma_v2_dir / "sigma_v2_config.json").write_text(
        json.dumps(best_cfg, indent=2), encoding="utf-8"
    )
    (sigma_v2_dir / "sigma_v2_config_search.json").write_text(
        json.dumps(sigma_v2_candidates, indent=2), encoding="utf-8"
    )
    (sigma_v2_dir / "README.md").write_text(
        "\n".join(
            [
                "# Sigma V2 (Bust-Aware Mixture)",
                "",
                "Decision time: 06Z day T",
                f"Feature store: {Path(args.feature_store).resolve()}",
                f"Preds: {Path(args.preds).resolve()}",
                "",
                f"Selected config: {json.dumps(best_cfg)}",
            ]
        ),
        encoding="utf-8",
    )

    print("Sigma V2 candidate configs (val):")
    for row in sigma_v2_candidates:
        print(
            "B={B} core={core_mode} floor={sigma_floor} k=({k_min},{k_max}) p_max={p_max} "
            "m_tail={m_tail} val_crps={val_crps:.6f} val_pit_chi2={val_pit_chi2:.2f} "
            "val_p90_cov={val_p90_coverage:.3f}".format(**row)
        )
    print(f"Sigma V2 selected config: {best_cfg}")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "preds": str(Path(args.preds).resolve()),
            "feature_store": str(Path(args.feature_store).resolve()),
            "mos_revisions": str(rev_path.resolve()) if rev_path.exists() else None,
        },
        "calibration_split": {
            "train": "2002-01-22..2019-12-31",
            "val": "2020-01-01..2022-12-31",
            "test": "2023-01-01..2025-12-31",
        },
        "selection": {
            "criterion": "val.crps",
            "best_method": best.name,
        },
        "cqr_hybrid_details": cqr_details,
        "studentt_sigma_w45_details": {
            "window_days": 45,
            "eps_abs": 0.1,
            "sigma_floor": 0.10,
            "df_selected": df_selected,
            "oof_years": "2008-2019",
            "calibration_method": "rolling_median_abs_ratio",
        },
        "normal_sigma_emos_w45_details": {
            "calibration_method": "ml_live.calibration.emos_w45.calibrate",
            "window_days": 45,
            "sigma_floor": 0.5,
        },
        "sigma_v2_details": {
            "model": "mog_sigma_bust_w45",
            "gaussian_ablation": "gaussian_sigma_nll_rms_w45",
            "selected_config": best_cfg,
            "sigma_floor": best_cfg["sigma_floor"],
            "window_days": 45,
            "p_min": 0.01,
        },
        "leakage_audit_studentt_sigma_w45": {
            "oof_mu": {"oof_years": "2008-2019", "violations": oof_violations},
            "online_sigma_calibration": {
                "window_days": 45,
                "max_future_window_violation_days": int(window_violations),
                "violations": int(window_violations),
            },
        },
        "leakage_audit_normal_sigma_emos_w45": {
            "window_days": 45,
            "max_future_window_violation_days": int(emos_window_violations),
            "violations": int(emos_window_violations),
        },
        "leakage_audit_sigma_v2": {
            "lag_feature_violations": int(lag_violation),
            "online_sigma_calibration": {
                "window_days": 45,
                "max_future_window_violation_days": int(best_window_violation),
                "violations": int(best_window_violation),
            },
        },
        "methods": {
            m.name: {
                "val": m.val,
                "test": m.test,
            }
            for m in methods
        },
    }

    (out_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(out_dir / "calibration_report.md", report)
    print(out_dir)
    return 0


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# V5+8 Calibration Comparison", ""]
    lines.append(f"- best_method: {report['selection']['best_method']}")
    lines.append("")
    for name, payload in report["methods"].items():
        lines.append(f"## {name}")
        for split in ["val", "test"]:
            m = payload.get(split, {})
            lines.append(f"### {split}")
            lines.append(f"- crps: {m.get('crps'):.6f}")
            pit = m.get("pit", {})
            if pit:
                lines.append(f"- PIT mean={pit.get('mean'):.4f} std={pit.get('std'):.4f} chi2={pit.get('chi2'):.2f}")
            intervals = m.get("intervals", {})
            for key, vals in intervals.items():
                lines.append(
                    f"- {key}: coverage={vals.get('coverage'):.3f} avg_width={vals.get('avg_width'):.3f}"
                )
            pin = m.get("pinball", {})
            if pin:
                q_list = ", ".join([f"{k}:{v:.4f}" for k, v in pin.items()])
                lines.append(f"- pinball: {q_list}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
