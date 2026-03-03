from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from zoneinfo import ZoneInfo

UTC = timezone.utc
STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")
EPS = 1e-12


@dataclass(frozen=True)
class CalibParams:
    a_peak: float
    b_peak: float
    log_gamma_delta: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-free OOF calibration for peak+delta distributions.")
    p.add_argument(
        "--experiment-dir",
        default=r"D:\Ahmed\data\kalshi\Experiments\E2_KNYC\Experiment_set_1\E2\20260228T153836Z",
    )
    p.add_argument("--oof-min-train-days", type=int, default=180)
    p.add_argument("--oof-block-days", type=int, default=30)
    p.add_argument("--output-dir", default=r"D:\Ahmed\data\kalshi\backtesting\results")
    return p.parse_args()


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_model_frame(eval_path: Path, pred_path: Path) -> pd.DataFrame:
    eval_df = pd.read_parquet(eval_path)
    pred_df = pd.read_parquet(pred_path)
    df = eval_df.merge(pred_df, on="row_index", how="left", suffixes=("_eval", "_pred"))
    ny = pd.to_datetime(df["target_date_local"]) + pd.to_timedelta(df["cutoff_minutes"], unit="m")
    ny = ny.dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
    df["st_timestamp"] = ny.dt.tz_convert(STOCKHOLM_TZ)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    return df


def add_arrays(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    delta_cols = [f"p_delta_class_{k}" for k in range(1, 61)]
    peak = out["p_peak_pred"].to_numpy(dtype=float)
    delta = out[delta_cols].to_numpy(dtype=float)
    delta = np.where(np.isfinite(delta), delta, 0.0)
    dsum = delta.sum(axis=1, keepdims=True)
    zero_rows = dsum[:, 0] <= 0
    if np.any(zero_rows):
        delta[zero_rows, -1] = 1.0
        dsum = delta.sum(axis=1, keepdims=True)
    delta = delta / np.clip(dsum, EPS, None)

    truth = out["tmax_truth"].to_numpy(dtype=int)
    sofar = out["tmax_sofar_round"].to_numpy(dtype=int)
    dtruth = truth - sofar
    true_class = np.where(dtruth <= 0, 0, np.where(dtruth >= 60, 60, dtruth)).astype(int)

    out["_peak_raw"] = peak
    out["_true_class"] = true_class
    out["_delta_json"] = [json.dumps(row.tolist()) for row in delta]
    return out


def parse_delta(df: pd.DataFrame) -> np.ndarray:
    return np.array([np.array(json.loads(x), dtype=float) for x in df["_delta_json"].tolist()], dtype=float)


def fit_calibrator(peak_raw: np.ndarray, delta_probs: np.ndarray, true_class: np.ndarray) -> CalibParams:
    z = logit(peak_raw)

    def nll(params: np.ndarray) -> float:
        a, b, lg = float(params[0]), float(params[1]), float(params[2])
        gamma = float(np.exp(lg))
        p0 = sigmoid(a + b * z)
        d_adj = np.power(np.clip(delta_probs, EPS, 1.0), gamma)
        d_adj = d_adj / np.clip(d_adj.sum(axis=1, keepdims=True), EPS, None)

        idx = np.arange(len(true_class))
        q_true = np.where(true_class == 0, p0, (1.0 - p0) * d_adj[idx, np.clip(true_class - 1, 0, 59)])
        q_true = np.clip(q_true, EPS, 1.0)
        reg = 1e-4 * (a * a + (b - 1.0) * (b - 1.0) + lg * lg)
        return float(-np.mean(np.log(q_true)) + reg)

    opt = minimize(
        nll,
        x0=np.array([0.0, 1.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0), (0.05, 5.0), (-3.0, 3.0)],
    )
    if not opt.success:
        return CalibParams(a_peak=0.0, b_peak=1.0, log_gamma_delta=0.0)
    return CalibParams(a_peak=float(opt.x[0]), b_peak=float(opt.x[1]), log_gamma_delta=float(opt.x[2]))


def build_q_raw(peak_raw: np.ndarray, delta_probs: np.ndarray) -> np.ndarray:
    q = np.zeros((len(peak_raw), 61), dtype=float)
    q[:, 0] = peak_raw
    q[:, 1:] = (1.0 - peak_raw)[:, None] * delta_probs
    q = q / np.clip(q.sum(axis=1, keepdims=True), EPS, None)
    return q


def build_q_cal(peak_raw: np.ndarray, delta_probs: np.ndarray, params: CalibParams) -> np.ndarray:
    z = logit(peak_raw)
    p0 = sigmoid(params.a_peak + params.b_peak * z)
    gamma = float(np.exp(params.log_gamma_delta))
    d_adj = np.power(np.clip(delta_probs, EPS, 1.0), gamma)
    d_adj = d_adj / np.clip(d_adj.sum(axis=1, keepdims=True), EPS, None)
    q = np.zeros((len(peak_raw), 61), dtype=float)
    q[:, 0] = p0
    q[:, 1:] = (1.0 - p0)[:, None] * d_adj
    q = q / np.clip(q.sum(axis=1, keepdims=True), EPS, None)
    return q


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> Tuple[pd.DataFrame, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    idx = np.floor(p * bins).astype(int)
    idx = np.clip(idx, 0, bins - 1)
    rows = []
    ece = 0.0
    n = max(1, len(p))
    for b in range(bins):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            rows.append({"bin": b, "count": 0, "pred_mean": np.nan, "empirical_rate": np.nan, "abs_gap": np.nan})
            continue
        pm = float(p[m].mean())
        ym = float(y[m].mean())
        gap = abs(pm - ym)
        ece += (c / n) * gap
        rows.append({"bin": b, "count": c, "pred_mean": pm, "empirical_rate": ym, "abs_gap": gap})
    return pd.DataFrame(rows), float(ece)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, object]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    rel, ece = reliability(y=y, p=p, bins=10)
    return {
        "count": int(len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "pred_mean": float(p.mean()),
        "empirical_mean": float(y.mean()),
        "mean_gap_pred_minus_empirical": float(p.mean() - y.mean()),
        "ece_10": ece,
        "reliability_10": rel,
    }


def quantile_index(cdf: np.ndarray, q: float) -> int:
    return int(np.searchsorted(cdf, q, side="left"))


def eval_distribution(true_class: np.ndarray, q: np.ndarray, seed: int) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    tc = np.asarray(true_class, dtype=int)
    q = np.asarray(q, dtype=float)
    n = len(tc)
    idx = np.arange(n)
    q_true = np.clip(q[idx, tc], EPS, 1.0)
    nll = float(-np.mean(np.log(q_true)))

    one = np.zeros_like(q)
    one[idx, tc] = 1.0
    brier = float(np.mean(np.sum((q - one) ** 2, axis=1)))
    q_cdf = np.cumsum(q, axis=1)
    one_cdf = np.cumsum(one, axis=1)
    rps = float(np.mean(np.sum((q_cdf - one_cdf) ** 2, axis=1) / (q.shape[1] - 1)))

    top1_pred = np.argmax(q, axis=1)
    top1_hit = (top1_pred == tc).astype(float)
    top1_conf = np.max(q, axis=1)
    rel_top1, ece_top1 = reliability(y=top1_hit, p=top1_conf, bins=10)

    rng = np.random.default_rng(seed)
    f_prev = np.array([np.sum(q[i, : tc[i]]) for i in range(n)], dtype=float)
    pit = f_prev + rng.random(n) * q_true
    ks = stats.kstest(pit, "uniform")
    hist, edges = np.histogram(pit, bins=20, range=(0.0, 1.0))
    pit_hist = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist})

    cov = {}
    for level in [0.50, 0.80, 0.90, 0.95]:
        lq = (1.0 - level) / 2.0
        hq = 1.0 - lq
        inside = []
        widths = []
        for i in range(n):
            cdf = q_cdf[i]
            lo = quantile_index(cdf, lq)
            hi = quantile_index(cdf, hq)
            inside.append(1.0 if lo <= tc[i] <= hi else 0.0)
            widths.append(float(hi - lo + 1))
        cov[f"coverage_{int(level*100)}"] = float(np.mean(inside))
        cov[f"avg_width_{int(level*100)}"] = float(np.mean(widths))

    peak = binary_metrics(y=(tc == 0).astype(float), p=q[:, 0])
    summary = {
        "count": int(n),
        "nll": nll,
        "brier_multiclass": brier,
        "rps": rps,
        "top1_accuracy": float(top1_hit.mean()),
        "top1_confidence_mean": float(top1_conf.mean()),
        "top1_ece_10": float(ece_top1),
        "pit_mean": float(pit.mean()),
        "pit_variance": float(pit.var()),
        "pit_ks_statistic": float(ks.statistic),
        "pit_ks_pvalue": float(ks.pvalue),
        **cov,
        "peak_binary": {k: v for k, v in peak.items() if k != "reliability_10"},
    }
    return summary, rel_top1, pit_hist


def run_oof(val_df: pd.DataFrame, min_days: int, block_days: int) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    dates = sorted(pd.Series(val_df["target_date_local"]).dropna().unique().tolist())
    fold_rows: List[Dict[str, object]] = []
    q_map: Dict[int, np.ndarray] = {}

    for start in range(min_days, len(dates), block_days):
        tr_dates = set(dates[:start])
        te_dates = set(dates[start : start + block_days])
        if not te_dates:
            continue
        tr = val_df[val_df["target_date_local"].isin(tr_dates)]
        te = val_df[val_df["target_date_local"].isin(te_dates)]
        if tr.empty or te.empty:
            continue

        params = fit_calibrator(
            peak_raw=tr["_peak_raw"].to_numpy(dtype=float),
            delta_probs=parse_delta(tr),
            true_class=tr["_true_class"].to_numpy(dtype=int),
        )
        q_te = build_q_cal(
            peak_raw=te["_peak_raw"].to_numpy(dtype=float),
            delta_probs=parse_delta(te),
            params=params,
        )
        for rid, q in zip(te["row_index"].astype(int).tolist(), q_te):
            q_map[int(rid)] = q

        fold_rows.append(
            {
                "fold_start_date": min(te_dates).isoformat(),
                "fold_end_date": max(te_dates).isoformat(),
                "train_days": len(tr_dates),
                "test_days": len(te_dates),
                "train_rows": int(len(tr)),
                "test_rows": int(len(te)),
                "a_peak": float(params.a_peak),
                "b_peak": float(params.b_peak),
                "gamma_delta": float(np.exp(params.log_gamma_delta)),
            }
        )
    if not q_map:
        raise RuntimeError("OOF map empty. Reduce --oof-min-train-days.")
    return pd.DataFrame(fold_rows), q_map


def main() -> int:
    args = parse_args()
    exp_dir = Path(args.experiment_dir)
    pred_dir = exp_dir / "predictions"
    metrics_path = exp_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    val_df = add_arrays(
        build_model_frame(
            eval_path=pred_dir / "distribution_eval_val.parquet",
            pred_path=pred_dir / "predictions_val.parquet",
        )
    )
    test_df = add_arrays(
        build_model_frame(
            eval_path=pred_dir / "distribution_eval_test.parquet",
            pred_path=pred_dir / "predictions_test.parquet",
        )
    )

    fold_df, q_oof_map = run_oof(
        val_df=val_df,
        min_days=int(args.oof_min_train_days),
        block_days=int(args.oof_block_days),
    )
    oof_idx = sorted(q_oof_map.keys())
    oof_df = val_df[val_df["row_index"].isin(oof_idx)].copy().sort_values("row_index")
    q_oof_cal = np.array([q_oof_map[int(r)] for r in oof_df["row_index"].astype(int).tolist()], dtype=float)
    q_oof_raw = build_q_raw(
        peak_raw=oof_df["_peak_raw"].to_numpy(dtype=float),
        delta_probs=parse_delta(oof_df),
    )
    true_oof = oof_df["_true_class"].to_numpy(dtype=int)

    oof_raw_summary, oof_rel_raw, oof_pit_raw = eval_distribution(true_class=true_oof, q=q_oof_raw, seed=42)
    oof_cal_summary, oof_rel_cal, oof_pit_cal = eval_distribution(true_class=true_oof, q=q_oof_cal, seed=42)

    final_params = fit_calibrator(
        peak_raw=val_df["_peak_raw"].to_numpy(dtype=float),
        delta_probs=parse_delta(val_df),
        true_class=val_df["_true_class"].to_numpy(dtype=int),
    )
    q_test_raw = build_q_raw(
        peak_raw=test_df["_peak_raw"].to_numpy(dtype=float),
        delta_probs=parse_delta(test_df),
    )
    q_test_cal = build_q_cal(
        peak_raw=test_df["_peak_raw"].to_numpy(dtype=float),
        delta_probs=parse_delta(test_df),
        params=final_params,
    )
    true_test = test_df["_true_class"].to_numpy(dtype=int)

    test_raw_summary, test_rel_raw, test_pit_raw = eval_distribution(true_class=true_test, q=q_test_raw, seed=43)
    test_cal_summary, test_rel_cal, test_pit_cal = eval_distribution(true_class=true_test, q=q_test_cal, seed=43)

    summary = {
        "model_split_from_metrics": metrics.get("split", {}),
        "oof_config": {
            "oof_min_train_days": int(args.oof_min_train_days),
            "oof_block_days": int(args.oof_block_days),
            "oof_rows_scored": int(len(oof_df)),
            "val_rows_total": int(len(val_df)),
        },
        "calibrator_final_params": {
            "a_peak": float(final_params.a_peak),
            "b_peak": float(final_params.b_peak),
            "gamma_delta": float(np.exp(final_params.log_gamma_delta)),
        },
        "oof_val_uncalibrated": oof_raw_summary,
        "oof_val_calibrated": oof_cal_summary,
        "oof_val_delta": {
            "nll": float(oof_cal_summary["nll"] - oof_raw_summary["nll"]),
            "brier_multiclass": float(oof_cal_summary["brier_multiclass"] - oof_raw_summary["brier_multiclass"]),
            "rps": float(oof_cal_summary["rps"] - oof_raw_summary["rps"]),
            "peak_logloss": float(oof_cal_summary["peak_binary"]["logloss"] - oof_raw_summary["peak_binary"]["logloss"]),
            "peak_ece_10": float(oof_cal_summary["peak_binary"]["ece_10"] - oof_raw_summary["peak_binary"]["ece_10"]),
        },
        "test_uncalibrated": test_raw_summary,
        "test_calibrated": test_cal_summary,
        "test_delta": {
            "nll": float(test_cal_summary["nll"] - test_raw_summary["nll"]),
            "brier_multiclass": float(test_cal_summary["brier_multiclass"] - test_raw_summary["brier_multiclass"]),
            "rps": float(test_cal_summary["rps"] - test_raw_summary["rps"]),
            "peak_logloss": float(test_cal_summary["peak_binary"]["logloss"] - test_raw_summary["peak_binary"]["logloss"]),
            "peak_ece_10": float(test_cal_summary["peak_binary"]["ece_10"] - test_raw_summary["peak_binary"]["ece_10"]),
        },
    }

    run_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"leakage_free_calibration_oof_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fold_df.to_csv(out_dir / "oof_folds.csv", index=False, encoding="utf-8")
    oof_df[["row_index", "target_date_local", "_true_class"]].to_csv(out_dir / "oof_rows.csv", index=False, encoding="utf-8")
    oof_rel_raw.to_csv(out_dir / "oof_reliability_top1_uncal_10bins.csv", index=False, encoding="utf-8")
    oof_rel_cal.to_csv(out_dir / "oof_reliability_top1_cal_10bins.csv", index=False, encoding="utf-8")
    oof_pit_raw.to_csv(out_dir / "oof_pit_hist_uncal_20bins.csv", index=False, encoding="utf-8")
    oof_pit_cal.to_csv(out_dir / "oof_pit_hist_cal_20bins.csv", index=False, encoding="utf-8")
    test_rel_raw.to_csv(out_dir / "test_reliability_top1_uncal_10bins.csv", index=False, encoding="utf-8")
    test_rel_cal.to_csv(out_dir / "test_reliability_top1_cal_10bins.csv", index=False, encoding="utf-8")
    test_pit_raw.to_csv(out_dir / "test_pit_hist_uncal_20bins.csv", index=False, encoding="utf-8")
    test_pit_cal.to_csv(out_dir / "test_pit_hist_cal_20bins.csv", index=False, encoding="utf-8")

    print(f"summary_json: {out_dir / 'summary.json'}")
    print(f"oof_folds_csv: {out_dir / 'oof_folds.csv'}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
