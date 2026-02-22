from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

sys.path.append("ml")
import run_mos_45_suite as base


def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class SuiteContext:
    df: pd.DataFrame
    y: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    seed: int
    cache: dict[str, Any]


@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    features: list[str]
    metrics: dict[str, dict[str, float]]
    extras: dict[str, Any]


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return base.ensure_columns(df, cols)


def _prepare_matrix(ctx: SuiteContext, features: list[str]) -> np.ndarray:
    df = _ensure_cols(ctx.df, features)
    filled, _ = base.impute_features(df[features], ctx.train_mask)
    return filled.to_numpy(dtype=float)


def _get_base(ctx: SuiteContext, base_series: str) -> np.ndarray:
    base_vals = pd.to_numeric(ctx.df.get(base_series), errors="coerce").to_numpy(dtype=float)
    base_mean = float(np.nanmean(ctx.y[ctx.train_mask]))
    return np.where(np.isnan(base_vals), base_mean, base_vals)


def _train_gate(
    ctx: SuiteContext,
    features: list[str],
    target: np.ndarray,
) -> np.ndarray:
    X = _prepare_matrix(ctx, features)
    model = base.train_lgbm_classifier(
        X[ctx.train_mask],
        target[ctx.train_mask],
        X[ctx.val_mask],
        target[ctx.val_mask],
        seed=ctx.seed,
    )
    return model.predict_proba(X)[:, 1]


def _fit_residual_model(
    ctx: SuiteContext,
    X: np.ndarray,
    base_vals: np.ndarray,
    mask: np.ndarray,
) -> Any:
    train_mask = ctx.train_mask & mask
    val_mask = ctx.val_mask & mask
    if not train_mask.any():
        return None
    return base.train_lgbm_regressor(
        X[train_mask],
        ctx.y[train_mask] - base_vals[train_mask],
        X[val_mask],
        ctx.y[val_mask] - base_vals[val_mask],
        seed=ctx.seed,
    )


def _predict_or_zero(model: Any, X: np.ndarray) -> np.ndarray:
    if model is None:
        return np.zeros(len(X), dtype=float)
    return model.predict(X)


def _moe_two_expert(
    ctx: SuiteContext,
    *,
    gate_features: list[str],
    gate_label: np.ndarray,
    expert_features: list[str],
    base_series: str,
) -> tuple[np.ndarray, np.ndarray]:
    base_vals = _get_base(ctx, base_series)
    p_gate = _train_gate(ctx, gate_features, gate_label)
    X_exp = _prepare_matrix(ctx, expert_features)
    expert_on = _fit_residual_model(ctx, X_exp, base_vals, gate_label == 1)
    expert_off = _fit_residual_model(ctx, X_exp, base_vals, gate_label == 0)
    resid_on = _predict_or_zero(expert_on, X_exp)
    resid_off = _predict_or_zero(expert_off, X_exp)
    pred = base_vals + p_gate * resid_on + (1.0 - p_gate) * resid_off
    pred = np.where(~np.isfinite(pred), base_vals, pred)
    return pred, p_gate


def _moe_four_expert(
    ctx: SuiteContext,
    *,
    p_onshore: np.ndarray,
    p_suppress: np.ndarray,
    expert_features: list[str],
    base_series: str,
    label_onshore: np.ndarray,
    label_suppress: np.ndarray,
) -> np.ndarray:
    base_vals = _get_base(ctx, base_series)
    X_exp = _prepare_matrix(ctx, expert_features)

    mask_on = label_onshore == 1
    mask_off = label_onshore == 0
    mask_sup = label_suppress == 1
    mask_clear = label_suppress == 0

    masks = {
        "on_clear": mask_on & mask_clear,
        "on_sup": mask_on & mask_sup,
        "off_clear": mask_off & mask_clear,
        "off_sup": mask_off & mask_sup,
    }
    experts = {k: _fit_residual_model(ctx, X_exp, base_vals, m) for k, m in masks.items()}
    resid = {k: _predict_or_zero(model, X_exp) for k, model in experts.items()}

    w_on_clear = p_onshore * (1.0 - p_suppress)
    w_on_sup = p_onshore * p_suppress
    w_off_clear = (1.0 - p_onshore) * (1.0 - p_suppress)
    w_off_sup = (1.0 - p_onshore) * p_suppress
    pred = base_vals + (
        w_on_clear * resid["on_clear"]
        + w_on_sup * resid["on_sup"]
        + w_off_clear * resid["off_clear"]
        + w_off_sup * resid["off_sup"]
    )
    pred = np.where(~np.isfinite(pred), base_vals, pred)
    return pred


def _train_quantile_models(
    ctx: SuiteContext,
    features: list[str],
    base_series: str,
    alphas: list[float],
) -> dict[float, np.ndarray]:
    base_vals = _get_base(ctx, base_series)
    X = _prepare_matrix(ctx, features)
    q_preds: dict[float, np.ndarray] = {}
    for alpha in alphas:
        model = base.train_lgbm_quantile(
            X[ctx.train_mask],
            ctx.y[ctx.train_mask] - base_vals[ctx.train_mask],
            X[ctx.val_mask],
            ctx.y[ctx.val_mask] - base_vals[ctx.val_mask],
            seed=ctx.seed,
            alpha=alpha,
        )
        q_preds[alpha] = base_vals + model.predict(X)
    return q_preds


def _v5p8_like(
    ctx: SuiteContext,
    *,
    gate_features: list[str],
    gate_label: np.ndarray,
    expert_features: list[str],
    base_series: str,
    k_grid: list[float] | None = None,
) -> tuple[np.ndarray, float]:
    base_vals = _get_base(ctx, base_series)
    X = _prepare_matrix(ctx, expert_features)
    p_gate = _train_gate(ctx, gate_features, gate_label)
    k_grid = k_grid or [0.0, 0.2, 0.4, 0.6, 0.8]

    def fit_quantile(alpha: float) -> np.ndarray:
        mask_on = gate_label == 1
        mask_off = gate_label == 0
        model_on = base.train_lgbm_quantile(
            X[ctx.train_mask & mask_on],
            ctx.y[ctx.train_mask & mask_on] - base_vals[ctx.train_mask & mask_on],
            X[ctx.val_mask & mask_on],
            ctx.y[ctx.val_mask & mask_on] - base_vals[ctx.val_mask & mask_on],
            seed=ctx.seed,
            alpha=alpha,
        )
        model_off = base.train_lgbm_quantile(
            X[ctx.train_mask & mask_off],
            ctx.y[ctx.train_mask & mask_off] - base_vals[ctx.train_mask & mask_off],
            X[ctx.val_mask & mask_off],
            ctx.y[ctx.val_mask & mask_off] - base_vals[ctx.val_mask & mask_off],
            seed=ctx.seed,
            alpha=alpha,
        )
        resid_on = model_on.predict(X)
        resid_off = model_off.predict(X)
        pred = base_vals + p_gate * resid_on + (1 - p_gate) * resid_off
        return pred - base_vals

    r10 = fit_quantile(0.1)
    r50 = fit_quantile(0.5)
    r90 = fit_quantile(0.9)
    spread = r90 - r10
    best_k = 0.0
    best_mae = 1e9
    for k in k_grid:
        w = np.exp(-k * spread)
        pred = base_vals + w * r50
        mae = base.regression_metrics(ctx.y[ctx.val_mask], pred[ctx.val_mask]).get("mae", float("inf"))
        if mae < best_mae:
            best_mae = mae
            best_k = k
    w_final = np.exp(-best_k * spread)
    pred_final = base_vals + w_final * r50
    pred_final = np.where(~np.isfinite(pred_final), base_vals, pred_final)
    return pred_final, best_k


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


def _pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    diff = y - q
    return float(np.nanmean(np.maximum(alpha * diff, (alpha - 1) * diff)))


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


def _pit_stats(pit: np.ndarray) -> dict[str, float]:
    pit = pit[np.isfinite(pit)]
    if pit.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "chi2": float("nan"), "hist_bins": []}
    hist, _ = np.histogram(pit, bins=10, range=(0.0, 1.0))
    expected = pit.size / 10.0
    chi2 = float(np.nansum((hist - expected) ** 2 / expected)) if expected > 0 else float("nan")
    return {
        "count": int(pit.size),
        "mean": float(np.mean(pit)),
        "std": float(np.std(pit)),
        "chi2": chi2,
        "hist_bins": hist.tolist(),
    }


def _coverage_width(y: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(q_low) & np.isfinite(q_high)
    if not mask.any():
        return {"coverage": float("nan"), "avg_width": float("nan")}
    cov = float(np.mean((y[mask] >= q_low[mask]) & (y[mask] <= q_high[mask])))
    width = float(np.nanmean(q_high[mask] - q_low[mask]))
    return {"coverage": cov, "avg_width": width}


def _sharpness_metrics_from_quantiles(q_preds: dict[float, np.ndarray]) -> dict[str, float]:
    q50 = q_preds[0.5]
    q05 = q_preds[0.05]
    q95 = q_preds[0.95]
    idx = np.isfinite(q50) & np.isfinite(q05) & np.isfinite(q95)
    if not idx.any():
        return {}
    ranges = q95[idx] - q05[idx]
    return {
        "sharpness_mean": float(np.nanmean(ranges)),
        "sharpness_median": float(np.nanmedian(ranges)),
        "sharpness_p90": float(np.nanquantile(ranges, 0.90)),
    }


def _integer_prob_metrics(q_preds: dict[float, np.ndarray]) -> dict[str, float]:
    q_levels = np.array(sorted(q_preds.keys()))
    q_vals = np.stack([q_preds[q] for q in q_levels], axis=1)
    p_top1 = []
    p_top2 = []
    entropy = []
    p_within1 = []
    for i in range(q_vals.shape[0]):
        vals = q_vals[i]
        if not np.isfinite(vals).any():
            continue
        qmin = int(np.floor(np.nanmin(vals))) - 2
        qmax = int(np.ceil(np.nanmax(vals))) + 2
        probs = []
        for k in range(qmin, qmax + 1):
            p = _cdf_from_quantiles(np.array([k + 0.5]), vals, q_levels)[0] - _cdf_from_quantiles(
                np.array([k - 0.5]), vals, q_levels
            )[0]
            probs.append(max(p, 0.0))
        probs = np.array(probs, dtype=float)
        if probs.sum() <= 0:
            continue
        probs = probs / probs.sum()
        order = np.sort(probs)[::-1]
        p_top1.append(float(order[0]))
        p_top2.append(float(order[:2].sum()))
        entropy.append(float(-np.sum(probs * np.log(probs + 1e-12))))
        median = int(np.round(q_preds[0.5][i]))
        idxs = [median - 1 - qmin, median - qmin, median + 1 - qmin]
        idxs = [j for j in idxs if 0 <= j < len(probs)]
        p_within1.append(float(np.sum(probs[idxs])))
    return {
        "p_top1": float(np.nanmean(p_top1)) if p_top1 else float("nan"),
        "p_top2": float(np.nanmean(p_top2)) if p_top2 else float("nan"),
        "entropy": float(np.nanmean(entropy)) if entropy else float("nan"),
        "p_within_1F_of_median": float(np.nanmean(p_within1)) if p_within1 else float("nan"),
    }


def _compute_suppression_labels(df: pd.DataFrame, train_mask: np.ndarray, threshold: float) -> np.ndarray:
    tmp = df.copy()
    tmp["doy"] = pd.to_datetime(tmp["target_date_local"]).dt.dayofyear
    climo = (
        tmp.loc[train_mask]
        .groupby("doy")["iem_range_day0"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "range_mean", "std": "range_std"})
    )
    tmp = tmp.merge(climo, on="doy", how="left")
    range_z = (pd.to_numeric(tmp["iem_range_day0"], errors="coerce") - tmp["range_mean"]) / (tmp["range_std"] + 1e-6)
    return (range_z <= threshold).astype(int).to_numpy()


def _oof_v6_predictions(
    df: pd.DataFrame,
    *,
    gate_features: list[str],
    expert_features: list[str],
    base_series: str,
    threshold: float,
    years: list[int],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    preds = np.full(len(df), np.nan, dtype=float)
    folds: list[dict[str, Any]] = []
    for year in years:
        train_end = f"{year-2}-12-31"
        val_start = f"{year-1}-01-01"
        val_end = f"{year-1}-12-31"
        pred_start = f"{year}-01-01"
        pred_end = f"{year}-12-31"
        split = base.split_by_date(
            df,
            train_start="2002-01-22",
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=pred_start,
            test_end=pred_end,
        )
        train_mask = split["train_mask"]
        val_mask = split["val_mask"]
        pred_mask = split["test_mask"]
        if not train_mask.any() or not pred_mask.any():
            continue
        ctx_fold = SuiteContext(
            df=df,
            y=pd.to_numeric(df["y_actual_tmax_f"], errors="coerce").to_numpy(dtype=float),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=pred_mask,
            seed=seed,
            cache={},
        )
        gate_label = (pd.to_numeric(df.get("feat_onshore"), errors="coerce") > 0.5).astype(int).to_numpy(dtype=int)
        label_supp = _compute_suppression_labels(df, train_mask, threshold)
        p_onshore = _train_gate(ctx_fold, gate_features, gate_label)
        p_suppress = _train_gate(ctx_fold, expert_features, label_supp)
        pred_full = _moe_four_expert(
            ctx_fold,
            p_onshore=p_onshore,
            p_suppress=p_suppress,
            expert_features=expert_features,
            base_series=base_series,
            label_onshore=gate_label,
            label_suppress=label_supp,
        )
        preds[pred_mask] = pred_full[pred_mask]
        folds.append(
            {
                "year": year,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "pred_start": pred_start,
                "pred_end": pred_end,
                "pred_count": int(pred_mask.sum()),
            }
        )
    return preds, folds


def _apply_cqr_hybrid(
    y_val: np.ndarray,
    q_preds_val: dict[float, np.ndarray],
    q_preds: dict[float, np.ndarray],
    *,
    alphas: list[float],
) -> tuple[dict[float, np.ndarray], dict[str, float]]:
    tau_values = {}
    adjusted = {q: q_preds[q].copy() for q in q_preds}
    for alpha in alphas:
        lo = q_preds_val[alpha]
        hi = q_preds_val[1 - alpha]
        s = np.maximum(lo - y_val, y_val - hi)
        s = np.where(s < 0, 0.0, s)
        delta = 2 * alpha
        tau = float(np.nanquantile(s, 1 - delta))
        tau_values[f"tau_{alpha:.2f}"] = tau
        adjusted[alpha] = adjusted[alpha] - tau
        adjusted[1 - alpha] = adjusted[1 - alpha] + tau
    # rearrange
    q_levels = sorted(adjusted.keys())
    q_vals = np.stack([adjusted[q] for q in q_levels], axis=1)
    q_vals = np.sort(q_vals, axis=1)
    for i, q in enumerate(q_levels):
        adjusted[q] = q_vals[:, i]
    return adjusted, tau_values


def _compute_distribution_metrics(
    y: np.ndarray,
    q_preds: dict[float, np.ndarray],
) -> dict[str, Any]:
    q_levels = np.array(sorted(q_preds.keys()))
    q_vals = np.stack([q_preds[q] for q in q_levels], axis=1)
    pit_vals = []
    for i in range(len(y)):
        pit_vals.append(_cdf_from_quantiles(np.array([y[i]]), q_vals[i], q_levels)[0])
    pit = _pit_stats(np.array(pit_vals))
    metrics = {
        "crps": _crps_from_quantiles(y, q_preds),
        "pit_chi2": pit.get("chi2"),
        "pit_bins": pit.get("hist_bins"),
        "intervals": {
            "p80": _coverage_width(y, q_preds[0.1], q_preds[0.9]),
            "p90": _coverage_width(y, q_preds[0.05], q_preds[0.95]),
        },
    }
    metrics.update(_sharpness_metrics_from_quantiles(q_preds))
    metrics.update(_integer_prob_metrics(q_preds))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V6 suite for KMIA.")
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline-preds",
        default="artifacts/experiments/winners/V5_PLUS8_20260219T222321Z/preds.parquet",
        help="Optional preds.parquet with V5+8 column to anchor baseline reproduction.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("artifacts/experiments") / f"V6_SUITE_{utc_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.feature_store)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"])
    split = base.split_by_date(
        df,
        train_start="2002-01-22",
        train_end="2019-12-31",
        val_start="2020-01-01",
        val_end="2022-12-31",
        test_start="2023-01-01",
        test_end="2025-12-31",
    )
    ctx = SuiteContext(
        df=df,
        y=pd.to_numeric(df["y_actual_tmax_f"], errors="coerce").to_numpy(dtype=float),
        train_mask=split["train_mask"],
        val_mask=split["val_mask"],
        test_mask=split["test_mask"],
        seed=args.seed,
        cache={},
    )

    # Baseline V5+8 reproduction
    DOY = ["cal_d_doy_sin", "cal_d_doy_cos"]
    gate_features = ["feat_u", "feat_v", "feat_wsp_mean", *DOY]
    expert_features_base = [
        "feat_dd_models",
        "feat_tmp_range_mean_models",
        "feat_p12_max",
        "feat_q12_max",
        "feat_cig_min",
        "feat_u",
        "feat_v",
        *DOY,
    ]
    base_series = "feat_le_median_biascorr"
    gate_label = (pd.to_numeric(df.get("feat_onshore"), errors="coerce") > 0.5).astype(int).to_numpy(dtype=int)
    minute_base = [
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

    base_vals_full = _get_base(ctx, base_series)
    pred_full_v5 = None
    baseline_source = "trained"
    baseline_missing = 0
    v5_out: dict[str, Any] | None = None

    baseline_path = Path(args.baseline_preds)
    if baseline_path.exists():
        try:
            bdf = pd.read_parquet(baseline_path)
            if "V5+8" in bdf.columns and "target_date_local" in bdf.columns:
                bdf["target_date_local"] = pd.to_datetime(bdf["target_date_local"])
                baseline_map = bdf.set_index("target_date_local")["V5+8"]
                pred_full_v5 = df["target_date_local"].map(baseline_map).to_numpy(dtype=float)
                baseline_source = str(baseline_path)
                baseline_missing = int(np.isnan(pred_full_v5).sum())
        except Exception:
            pred_full_v5 = None

    if pred_full_v5 is None:
        import run_v5_plus_suite as v5suite

        v5_out = v5suite.compute_v5p8_predictions(
            df, train_mask=ctx.train_mask, val_mask=ctx.val_mask, seed=ctx.seed
        )
        pred_full_v5 = v5_out["pred_v5p8"]

    pred_full_v5 = np.where(~np.isfinite(pred_full_v5), base_vals_full, pred_full_v5)
    p_onshore = _train_gate(ctx, gate_features, gate_label)
    pred_train = pred_full_v5[ctx.train_mask]
    pred_val = pred_full_v5[ctx.val_mask]
    pred_test = pred_full_v5[ctx.test_mask]

    results: list[ExperimentResult] = []
    preds_full_map: dict[str, np.ndarray] = {}

    def record(exp_id: str, name: str, features: list[str], preds: tuple[np.ndarray, np.ndarray, np.ndarray], extra: dict | None = None):
        pred_tr, pred_va, pred_te = preds
        metrics_payload = {
            "train": base.regression_metrics(ctx.y[ctx.train_mask], pred_tr),
            "validation": base.regression_metrics(ctx.y[ctx.val_mask], pred_va),
            "test": base.regression_metrics(ctx.y[ctx.test_mask], pred_te),
        }
        results.append(
            ExperimentResult(
                experiment_id=exp_id,
                name=name,
                features=features,
                metrics=metrics_payload,
                extras=extra or {},
            )
        )
        exp_dir = out_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        pred_full = np.full(len(ctx.y), np.nan, dtype=float)
        pred_full[ctx.train_mask] = pred_tr
        pred_full[ctx.val_mask] = pred_va
        pred_full[ctx.test_mask] = pred_te
        all_rows = pd.DataFrame(
            {
                "target_date_local": ctx.df["target_date_local"].astype(str),
                "y": ctx.y,
                "pred": pred_full,
                "split": np.where(ctx.train_mask, "train", np.where(ctx.val_mask, "val", "test")),
            }
        )
        all_rows.to_parquet(exp_dir / "preds.parquet", index=False)
        preds_full_map[exp_id] = pred_full
        pd.DataFrame(
            {
                "target_date_local": ctx.df.loc[ctx.train_mask, "target_date_local"].astype(str),
                "y": ctx.y[ctx.train_mask],
                "pred": pred_tr,
            }
        ).to_parquet(exp_dir / "preds_train.parquet", index=False)
        pd.DataFrame(
            {
                "target_date_local": ctx.df.loc[ctx.val_mask, "target_date_local"].astype(str),
                "y": ctx.y[ctx.val_mask],
                "pred": pred_va,
            }
        ).to_parquet(exp_dir / "preds_val.parquet", index=False)
        pd.DataFrame(
            {
                "target_date_local": ctx.df.loc[ctx.test_mask, "target_date_local"].astype(str),
                "y": ctx.y[ctx.test_mask],
                "pred": pred_te,
            }
        ).to_parquet(exp_dir / "preds_test.parquet", index=False)

    record("E601", "Reproduce V5+8", expert_features_base + minute_base, (pred_train, pred_val, pred_test))

    # Baseline snapshot
    baseline_dir = out_dir / "E601"
    baseline_snapshot = {
        "features": expert_features_base + minute_base,
        "train_mae": base.regression_metrics(ctx.y[ctx.train_mask], pred_train).get("mae"),
        "val_mae": base.regression_metrics(ctx.y[ctx.val_mask], pred_val).get("mae"),
        "test_mae": base.regression_metrics(ctx.y[ctx.test_mask], pred_test).get("mae"),
        "best_k": v5_out.get("best_k") if v5_out else None,
        "baseline_source": baseline_source,
        "baseline_missing": baseline_missing,
    }
    (baseline_dir / "baseline_snapshot.json").write_text(json.dumps(baseline_snapshot, indent=2), encoding="utf-8")
    if baseline_snapshot["test_mae"] is not None and abs(baseline_snapshot["test_mae"] - 0.6904) > 0.05:
        raise ValueError("Baseline V5+8 reproduction failed; aborting V6 suite.")

    # Phase A minute representation ablations
    local_bins = np.arange(0, 1440, 5)
    utc_bins = np.arange(0, 360, 5)
    raw_tminus1 = [f"iem_tminus1_profile_{b:04d}" for b in local_bins]
    raw_utc = [f"iem_utc00_06_profile_{b:04d}" for b in utc_bins]
    dct_tminus1 = [f"iem_tminus1_dct_{i:02d}" for i in range(20)] + ["iem_tminus1_dct_energy_hi"]
    dct_utc = [f"iem_utc00_06_dct_{i:02d}" for i in range(10)] + ["iem_utc00_06_dct_energy_hi"]

    def run_ablation(extra_feats: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_full, _ = _v5p8_like(
            ctx,
            gate_features=gate_features,
            gate_label=gate_label,
            expert_features=expert_features_base + minute_base + extra_feats,
            base_series=base_series,
        )
        return pred_full[ctx.train_mask], pred_full[ctx.val_mask], pred_full[ctx.test_mask]

    pred_e602 = run_ablation(raw_tminus1)
    record("E602", "V5+8 + T-1 raw profile", expert_features_base + minute_base + raw_tminus1, pred_e602)

    pred_e603 = run_ablation(raw_utc)
    record("E603", "V5+8 + UTC 00-06 raw profile", expert_features_base + minute_base + raw_utc, pred_e603)

    pred_e604 = run_ablation(raw_tminus1 + raw_utc)
    record("E604", "V5+8 + raw profiles", expert_features_base + minute_base + raw_tminus1 + raw_utc, pred_e604)

    pred_e605 = run_ablation(dct_tminus1 + dct_utc)
    record("E605", "V5+8 + DCT only", expert_features_base + minute_base + dct_tminus1 + dct_utc, pred_e605)

    pred_e606 = run_ablation(raw_tminus1 + raw_utc + dct_tminus1 + dct_utc)
    record("E606", "V5+8 + raw + DCT", expert_features_base + minute_base + raw_tminus1 + raw_utc + dct_tminus1 + dct_utc, pred_e606)

    # Pick best minute representation by val MAE
    best_phase_a = min(
        [("E602", pred_e602), ("E603", pred_e603), ("E604", pred_e604), ("E605", pred_e605), ("E606", pred_e606)],
        key=lambda item: base.regression_metrics(ctx.y[ctx.val_mask], item[1][1]).get("mae", float("inf")),
    )
    best_minute_feats = {
        "E602": raw_tminus1,
        "E603": raw_utc,
        "E604": raw_tminus1 + raw_utc,
        "E605": dct_tminus1 + dct_utc,
        "E606": raw_tminus1 + raw_utc + dct_tminus1 + dct_utc,
    }[best_phase_a[0]]

    # Suppression labels
    best_thr = None
    best_e611 = None
    best_e611_pred = None
    for thr in [-0.5, -0.7, -0.9]:
        label_supp = _compute_suppression_labels(df, ctx.train_mask, thr)
        p_suppress = _train_gate(ctx, expert_features_base + minute_base + best_minute_feats, label_supp)
        pred_full_e611 = _moe_four_expert(
            ctx,
            p_onshore=p_onshore,
            p_suppress=p_suppress,
            expert_features=expert_features_base + minute_base + best_minute_feats,
            base_series=base_series,
            label_onshore=gate_label,
            label_suppress=label_supp,
        )
        mae_val = base.regression_metrics(ctx.y[ctx.val_mask], pred_full_e611[ctx.val_mask]).get("mae", float("inf"))
        if best_thr is None or mae_val < best_e611:
            best_thr = thr
            best_e611 = mae_val
            best_e611_pred = pred_full_e611

    label_supp = _compute_suppression_labels(df, ctx.train_mask, best_thr)
    p_suppress = _train_gate(ctx, expert_features_base + minute_base + best_minute_feats, label_supp)
    record(
        "E611",
        f"4 experts + suppression gate (thr={best_thr})",
        expert_features_base + minute_base + best_minute_feats,
        (
            best_e611_pred[ctx.train_mask],
            best_e611_pred[ctx.val_mask],
            best_e611_pred[ctx.test_mask],
        ),
        {"suppression_threshold": best_thr},
    )

    # E612: add multi-day lags
    lag_feats = [
        "iem_tmax_t2",
        "iem_tmin_t2",
        "iem_range_t2",
        "tmax_time_min_t2",
        "max_drop_30_t2",
        "drop_cnt_15_19_t2",
        "iem_tmax_t3",
        "iem_tmin_t3",
        "iem_range_t3",
        "tmax_time_min_t3",
        "max_drop_30_t3",
        "drop_cnt_15_19_t3",
        "delta_tmax_1d",
        "delta_range_1d",
        "trend_tmax_3d",
    ]
    pred_full_e612 = _moe_four_expert(
        ctx,
        p_onshore=p_onshore,
        p_suppress=p_suppress,
        expert_features=expert_features_base + minute_base + best_minute_feats + lag_feats,
        base_series=base_series,
        label_onshore=gate_label,
        label_suppress=label_supp,
    )
    record(
        "E612",
        "4 experts + multi-day lags",
        expert_features_base + minute_base + best_minute_feats + lag_feats,
        (
            pred_full_e612[ctx.train_mask],
            pred_full_e612[ctx.val_mask],
            pred_full_e612[ctx.test_mask],
        ),
    )

    # E613: add MOS revisions
    rev_feats = [c for c in df.columns if c.startswith("v6_abs_rev_") or c.startswith("v6_trend_") or c.startswith("v6_disc_") or c.startswith("v6_abs_disc_")]
    pred_full_e613 = _moe_four_expert(
        ctx,
        p_onshore=p_onshore,
        p_suppress=p_suppress,
        expert_features=expert_features_base + minute_base + best_minute_feats + lag_feats + rev_feats,
        base_series=base_series,
        label_onshore=gate_label,
        label_suppress=label_supp,
    )
    record(
        "E613",
        "4 experts + lags + MOS revisions",
        expert_features_base + minute_base + best_minute_feats + lag_feats + rev_feats,
        (
            pred_full_e613[ctx.train_mask],
            pred_full_e613[ctx.val_mask],
            pred_full_e613[ctx.test_mask],
        ),
    )

    # E614: add MOS vs obs mismatch
    mismatch_feats = ["obs06z_minus_mos_tmpmin", "obs06z_minus_mos_tmpmax", "obs00z_minus_mos_tmpmin"]
    pred_full_e614 = _moe_four_expert(
        ctx,
        p_onshore=p_onshore,
        p_suppress=p_suppress,
        expert_features=expert_features_base + minute_base + best_minute_feats + lag_feats + rev_feats + mismatch_feats,
        base_series=base_series,
        label_onshore=gate_label,
        label_suppress=label_supp,
    )
    record(
        "E614",
        "4 experts + lags + revisions + mismatch",
        expert_features_base + minute_base + best_minute_feats + lag_feats + rev_feats + mismatch_feats,
        (
            pred_full_e614[ctx.train_mask],
            pred_full_e614[ctx.val_mask],
            pred_full_e614[ctx.test_mask],
        ),
    )

    # Bust specialist (OOF on train, tune on val)
    best_bust = None
    best_e621 = None
    bust_feats = expert_features_base + minute_base + best_minute_feats + lag_feats + rev_feats + mismatch_feats
    oof_preds, oof_folds = _oof_v6_predictions(
        df,
        gate_features=gate_features,
        expert_features=bust_feats,
        base_series=base_series,
        threshold=best_thr,
        years=list(range(2008, 2020)),
        seed=ctx.seed,
    )
    oof_mask = ctx.train_mask & np.isfinite(oof_preds)
    X_bust = _prepare_matrix(ctx, bust_feats)
    for B in [2.0, 2.5, 3.0]:
        label_bust = (np.abs(ctx.y - oof_preds) >= B).astype(int)
        model_bust = base.train_lgbm_classifier(
            X_bust[oof_mask],
            label_bust[oof_mask],
            X_bust[ctx.val_mask],
            label_bust[ctx.val_mask],
            seed=ctx.seed,
        )
        p_bust = model_bust.predict_proba(X_bust)[:, 1]
        for cap in [1.5, 2.0, 2.5]:
            mask_bust = oof_mask & (label_bust == 1)
            if not mask_bust.any():
                continue
            delta_model = base.train_lgbm_regressor(
                X_bust[mask_bust],
                (ctx.y - oof_preds)[mask_bust],
                X_bust[ctx.val_mask],
                (ctx.y - pred_full_e614)[ctx.val_mask],
                seed=ctx.seed,
            )
            delta_hat = delta_model.predict(X_bust)
            adj = p_bust * np.clip(delta_hat, -cap, cap)
            pred_all = pred_full_e614 + adj
            mae_val = base.regression_metrics(ctx.y[ctx.val_mask], pred_all[ctx.val_mask]).get("mae", float("inf"))
            if best_bust is None or mae_val < best_bust:
                best_bust = mae_val
                best_e621 = (pred_all[ctx.train_mask], pred_all[ctx.val_mask], pred_all[ctx.test_mask])

    if best_e621 is not None:
        record(
            "E621",
            "E614 + bust specialist",
            expert_features_base + minute_base + best_minute_feats + lag_feats + rev_feats + mismatch_feats,
            best_e621,
        )

    # Distribution densification for all experiments (dense quantiles + CQR hybrid)
    alpha_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    dist_metrics_map: dict[str, dict[str, Any]] = {}
    for exp in results:
        features = exp.features
        q_preds = _train_quantile_models(ctx, features, base_series, alpha_levels)
        q_val = {q: q_preds[q][ctx.val_mask] for q in q_preds}
        q_adj, tau_vals = _apply_cqr_hybrid(ctx.y[ctx.val_mask], q_val, q_preds, alphas=[0.05, 0.10])
        q_adj_val = {q: q_adj[q][ctx.val_mask] for q in q_adj}
        q_adj_test = {q: q_adj[q][ctx.test_mask] for q in q_adj}
        metrics_val = _compute_distribution_metrics(ctx.y[ctx.val_mask], q_adj_val)
        metrics_test = _compute_distribution_metrics(ctx.y[ctx.test_mask], q_adj_test)
        exp_dir = out_dir / exp.experiment_id
        report = {
            "experiment_id": exp.experiment_id,
            "quantiles": alpha_levels,
            "tau": tau_vals,
            "val": metrics_val,
            "test": metrics_test,
        }
        (exp_dir / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        dist_metrics_map[exp.experiment_id] = {"val": metrics_val, "test": metrics_test, "tau": tau_vals}

        # per-experiment report
        pred_full = preds_full_map.get(exp.experiment_id)
        if pred_full is None:
            continue
        abs_err = np.abs(ctx.y[ctx.test_mask] - pred_full[ctx.test_mask])
        order = np.argsort(abs_err)[::-1][:20]
        dates = ctx.df.loc[ctx.test_mask, "target_date_local"].astype(str).to_numpy()
        onshore_label = (pd.to_numeric(ctx.df.get("feat_onshore"), errors="coerce") > 0.5).astype(int).to_numpy()
        worst_lines = ["| date | y | pred | abs_err | onshore | suppressed |", "|---|---:|---:|---:|---:|---:|"]
        for idx in order:
            worst_lines.append(
                f"| {dates[idx]} | {ctx.y[ctx.test_mask][idx]:.2f} | {pred_full[ctx.test_mask][idx]:.2f} | {abs_err[idx]:.2f} | {onshore_label[ctx.test_mask][idx]} | {label_supp[ctx.test_mask][idx]} |"
            )
        md = [
            f"# {exp.experiment_id} — {exp.name}",
            "",
            "## Point Metrics",
            f"- train_mae: {exp.metrics['train'].get('mae'):.4f}",
            f"- val_mae: {exp.metrics['validation'].get('mae'):.4f}",
            f"- test_mae: {exp.metrics['test'].get('mae'):.4f}",
            "",
            "## Distribution Metrics (CQR hybrid)",
            f"- val_crps: {metrics_val.get('crps'):.6f}",
            f"- test_crps: {metrics_test.get('crps'):.6f}",
            f"- test_p90_coverage: {metrics_test.get('intervals',{}).get('p90',{}).get('coverage')}",
            f"- test_p90_width: {metrics_test.get('intervals',{}).get('p90',{}).get('avg_width')}",
            f"- p_top1: {metrics_test.get('p_top1')}",
            f"- entropy: {metrics_test.get('entropy')}",
            "",
            "## Worst 20 Test Errors",
            *worst_lines,
        ]
        (exp_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    summary = {
        "experiments": [
            {
                "experiment_id": r.experiment_id,
                "name": r.name,
                "features": r.features,
                "metrics": r.metrics,
                "extras": r.extras,
            }
            for r in results
        ]
    }
    (out_dir / "experiments_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "split_info.json").write_text(json.dumps(split, indent=2, default=str), encoding="utf-8")

    # Leakage audit
    decision_utc = pd.to_datetime(df["target_date_local"]).dt.tz_localize(timezone.utc) + pd.Timedelta(hours=6)
    mos_max = pd.to_datetime(df.get("mos_max_asof_used"), utc=True, errors="coerce")
    minute_max_t1 = pd.to_datetime(df.get("minute_max_ts_utc_tminus1"), utc=True, errors="coerce")
    minute_max_early = pd.to_datetime(df.get("minute_max_ts_utc_early"), utc=True, errors="coerce")
    minute_max = minute_max_t1.combine(minute_max_early, func=lambda a, b: b if pd.isna(a) else max(a, b))
    mos_viol = (mos_max > decision_utc).sum()
    minute_viol = (minute_max > decision_utc).sum()
    audit = {
        "mos_asof_violations": int(mos_viol),
        "minute_ts_violations": int(minute_viol),
        "mos_max_asof_used_max": str(mos_max.max()),
        "minute_max_ts_used_max": str(minute_max.max()),
    }
    (out_dir / "leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # Report
    lines = ["V6 Suite Summary", ""]
    for r in results:
        mae_val = r.metrics["validation"].get("mae")
        mae_test = r.metrics["test"].get("mae")
        lines.append(f"- {r.experiment_id} {r.name}: val_mae={mae_val:.4f} test_mae={mae_test:.4f}")
    lines.append("")
    lines.append(f"Leakage audit: mos_violations={audit['mos_asof_violations']} minute_violations={audit['minute_ts_violations']}")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote V6 suite to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
