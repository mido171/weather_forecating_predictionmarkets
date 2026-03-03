from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

UTC = timezone.utc
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-free selected-trade stream calibration eval.")
    p.add_argument(
        "--candidate-universe-csv",
        default=r"D:\Ahmed\data\kalshi\backtesting\results\trade_layer_calibration_walkforward_20260228T204616Z\candidate_universe_2025.csv",
    )
    p.add_argument("--holdout-start", default="2025-10-01")
    p.add_argument("--oof-min-train-days", type=int, default=120)
    p.add_argument("--oof-block-days", type=int, default=30)
    p.add_argument("--min-win-prob", type=float, default=0.650001)
    p.add_argument("--ev-threshold", type=float, default=0.10)
    p.add_argument("--output-dir", default=r"D:\Ahmed\data\kalshi\backtesting\results")
    return p.parse_args()


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_platt(y: np.ndarray, p_raw: np.ndarray) -> Tuple[float, float]:
    z = logit(p_raw)
    y = np.asarray(y, dtype=float)

    def nll(params: np.ndarray) -> float:
        a = float(params[0])
        b = float(params[1])
        q = np.clip(sigmoid(a + b * z), EPS, 1.0 - EPS)
        reg = 1e-4 * (a * a + (b - 1.0) * (b - 1.0))
        return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)) + reg)

    opt = minimize(
        nll,
        x0=np.array([0.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0), (0.01, 10.0)],
    )
    if not opt.success:
        return 0.0, 1.0
    return float(opt.x[0]), float(opt.x[1])


def apply_platt(p_raw: np.ndarray, a: float, b: float) -> np.ndarray:
    return sigmoid(a + b * logit(p_raw))


def pick_daily(
    df: pd.DataFrame,
    prob_col: str,
    ev_col: str,
    min_prob: float,
    min_ev: float,
) -> pd.DataFrame:
    e = df[(df[prob_col] >= min_prob) & (df[ev_col] >= min_ev)].copy()
    if e.empty:
        return e
    e = e.sort_values(
        ["trade_date", ev_col, prob_col, "market_win_prob", "bucket_label", "side"],
        ascending=[True, False, False, True, True, True],
    )
    return e.groupby("trade_date", as_index=False).head(1).reset_index(drop=True)


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    if len(y) == 0:
        return {"count": 0, "win_rate": float("nan"), "mean_p": float("nan"), "gap": float("nan"), "logloss": float("nan"), "brier": float("nan")}
    return {
        "count": int(len(y)),
        "win_rate": float(y.mean()),
        "mean_p": float(p.mean()),
        "gap": float(p.mean() - y.mean()),
        "logloss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "brier": float(np.mean((p - y) ** 2)),
    }


def run() -> int:
    args = parse_args()
    cands = pd.read_csv(args.candidate_universe_csv)
    cands["trade_date"] = pd.to_datetime(cands["trade_date"]).dt.date
    holdout_start = date.fromisoformat(args.holdout_start)

    dev = cands[cands["trade_date"] < holdout_start].copy()
    hold = cands[cands["trade_date"] >= holdout_start].copy()

    dates = sorted(dev["trade_date"].dropna().unique().tolist())
    oof_rows: List[pd.DataFrame] = []
    fold_rows: List[Dict[str, object]] = []

    for start in range(int(args.oof_min_train_days), len(dates), int(args.oof_block_days)):
        tr_dates = set(dates[:start])
        te_dates = set(dates[start : start + int(args.oof_block_days)])
        if not te_dates:
            continue
        tr = dev[dev["trade_date"].isin(tr_dates)].copy()
        te = dev[dev["trade_date"].isin(te_dates)].copy()
        tr_sel = pick_daily(tr, "model_win_prob_raw", "ev_raw", float(args.min_win_prob), float(args.ev_threshold))
        te_sel = pick_daily(te, "model_win_prob_raw", "ev_raw", float(args.min_win_prob), float(args.ev_threshold))
        if len(tr_sel) < 30 or te_sel.empty:
            continue
        a, b = fit_platt(
            y=tr_sel["outcome_win"].to_numpy(dtype=float),
            p_raw=tr_sel["model_win_prob_raw"].to_numpy(dtype=float),
        )
        te_sel = te_sel.copy()
        te_sel["model_win_prob_cal"] = apply_platt(te_sel["model_win_prob_raw"].to_numpy(dtype=float), a, b)
        te_sel["ev_cal"] = te_sel["model_win_prob_cal"] - te_sel["market_win_prob"]
        oof_rows.append(te_sel)
        fold_rows.append(
            {
                "fold_start_date": min(te_dates).isoformat(),
                "fold_end_date": max(te_dates).isoformat(),
                "train_selected_rows": int(len(tr_sel)),
                "test_selected_rows": int(len(te_sel)),
                "a": float(a),
                "b": float(b),
            }
        )

    if not oof_rows:
        raise RuntimeError("No OOF rows produced.")
    oof = pd.concat(oof_rows, ignore_index=True)
    oof_raw = metrics(oof["outcome_win"].to_numpy(dtype=float), oof["model_win_prob_raw"].to_numpy(dtype=float))
    oof_cal = metrics(oof["outcome_win"].to_numpy(dtype=float), oof["model_win_prob_cal"].to_numpy(dtype=float))

    dev_sel = pick_daily(dev, "model_win_prob_raw", "ev_raw", float(args.min_win_prob), float(args.ev_threshold))
    hold_sel = pick_daily(hold, "model_win_prob_raw", "ev_raw", float(args.min_win_prob), float(args.ev_threshold))
    a_final, b_final = fit_platt(
        y=dev_sel["outcome_win"].to_numpy(dtype=float),
        p_raw=dev_sel["model_win_prob_raw"].to_numpy(dtype=float),
    )
    hold_sel = hold_sel.copy()
    hold_sel["model_win_prob_cal"] = apply_platt(hold_sel["model_win_prob_raw"].to_numpy(dtype=float), a_final, b_final)
    hold_sel["ev_cal"] = hold_sel["model_win_prob_cal"] - hold_sel["market_win_prob"]

    hold_raw = metrics(hold_sel["outcome_win"].to_numpy(dtype=float), hold_sel["model_win_prob_raw"].to_numpy(dtype=float))
    hold_cal = metrics(hold_sel["outcome_win"].to_numpy(dtype=float), hold_sel["model_win_prob_cal"].to_numpy(dtype=float))
    hold_cal_filtered = hold_sel[
        (hold_sel["model_win_prob_cal"] >= float(args.min_win_prob)) & (hold_sel["ev_cal"] >= float(args.ev_threshold))
    ].copy()
    hold_cal_filtered_metrics = metrics(
        hold_cal_filtered["outcome_win"].to_numpy(dtype=float),
        hold_cal_filtered["model_win_prob_cal"].to_numpy(dtype=float),
    )

    summary = {
        "config": {
            "holdout_start": args.holdout_start,
            "oof_min_train_days": int(args.oof_min_train_days),
            "oof_block_days": int(args.oof_block_days),
            "selection_min_win_prob_raw": float(args.min_win_prob),
            "selection_ev_threshold_raw": float(args.ev_threshold),
        },
        "counts": {
            "dev_candidate_rows": int(len(dev)),
            "holdout_candidate_rows": int(len(hold)),
            "dev_selected_rows": int(len(dev_sel)),
            "holdout_selected_rows": int(len(hold_sel)),
            "oof_selected_rows": int(len(oof)),
            "oof_folds": int(len(fold_rows)),
        },
        "final_selected_stream_calibrator": {"a": float(a_final), "b": float(b_final)},
        "oof_selected_raw": oof_raw,
        "oof_selected_cal": oof_cal,
        "oof_selected_delta": {
            "logloss": float(oof_cal["logloss"] - oof_raw["logloss"]),
            "brier": float(oof_cal["brier"] - oof_raw["brier"]),
            "gap": float(oof_cal["gap"] - oof_raw["gap"]),
        },
        "holdout_selected_raw": hold_raw,
        "holdout_selected_cal": hold_cal,
        "holdout_selected_delta": {
            "logloss": float(hold_cal["logloss"] - hold_raw["logloss"]),
            "brier": float(hold_cal["brier"] - hold_raw["brier"]),
            "gap": float(hold_cal["gap"] - hold_raw["gap"]),
        },
        "holdout_selected_after_cal_thresholds": hold_cal_filtered_metrics,
    }

    run_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"trade_selected_stream_calibration_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(out_dir / "oof_folds.csv", index=False, encoding="utf-8")
    oof.to_csv(out_dir / "oof_selected_predictions.csv", index=False, encoding="utf-8")
    hold_sel.to_csv(out_dir / "holdout_selected_predictions.csv", index=False, encoding="utf-8")
    hold_cal_filtered.to_csv(out_dir / "holdout_selected_after_cal_thresholds.csv", index=False, encoding="utf-8")

    print(f"summary_json: {out_dir / 'summary.json'}")
    print(f"oof_folds_csv: {out_dir / 'oof_folds.csv'}")
    print(f"oof_selected_predictions_csv: {out_dir / 'oof_selected_predictions.csv'}")
    print(f"holdout_selected_predictions_csv: {out_dir / 'holdout_selected_predictions.csv'}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
