from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist

from backtesting.trade_layer_calibration_walkforward import (
    binary_metrics,
    build_candidate_universe,
    build_model_frame,
    parse_bucket_interval,
    pick_daily_trades,
    settlement_backtest,
)

UTC = timezone.utc
EPS = 1e-12


@dataclass(frozen=True)
class OffsetModel:
    feature_cols: List[str]
    means: np.ndarray
    stds: np.ndarray
    beta: np.ndarray  # includes intercept as beta[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dynamic-bucket market-offset walk-forward calibration + policy backtest."
    )
    p.add_argument("--kalshi-dir", default=r"D:\Ahmed\data\kalshi\kalshi_history\kxhighny_2025")
    p.add_argument(
        "--predictions-parquet",
        default=r"D:\Ahmed\data\kalshi\Experiments\E2_KNYC\Experiment_set_1\E2\20260228T153836Z\predictions\predictions_test.parquet",
    )
    p.add_argument(
        "--distribution-eval-parquet",
        default=r"D:\Ahmed\data\kalshi\Experiments\E2_KNYC\Experiment_set_1\E2\20260228T153836Z\predictions\distribution_eval_test.parquet",
    )
    p.add_argument("--entry-hour-stockholm", type=int, default=19)
    p.add_argument("--entry-minute-stockholm", type=int, default=0)
    p.add_argument("--trade-date-offset-days", type=int, default=0)
    p.add_argument("--holdout-start", default="2025-10-01")
    p.add_argument("--oof-min-train-days", type=int, default=120)
    p.add_argument("--oof-block-days", type=int, default=30)
    p.add_argument("--l2", type=float, default=0.05)
    p.add_argument("--min-win-prob", type=float, default=0.65)
    p.add_argument("--ev-threshold", type=float, default=0.15)
    p.add_argument("--start-balance", type=float, default=2700.0)
    p.add_argument("--risk-fraction", type=float, default=0.04)
    p.add_argument("--lcb-alpha", type=float, default=0.10)
    p.add_argument("--min-cell-count", type=int, default=80)
    p.add_argument("--output-dir", default=r"D:\Ahmed\data\kalshi\backtesting\results")
    return p.parse_args()


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _bucket_features(label: str) -> Tuple[float, float, float]:
    interval = parse_bucket_interval(label)
    if interval.kind == "range":
        assert interval.lo is not None and interval.hi is not None
        width = float(interval.hi - interval.lo + 1)
        center = 0.5 * (float(interval.lo) + float(interval.hi))
        tail = 0.0
    elif interval.kind == "le":
        assert interval.hi is not None
        width = float(interval.hi + 1)  # integer support from 0..hi
        center = float(interval.hi)
        tail = 1.0
    else:
        assert interval.lo is not None
        width = 30.0  # proxy tail width on upper side
        center = float(interval.lo)
        tail = 1.0
    return width, center, tail


def enrich_candidate_features(cands: pd.DataFrame) -> pd.DataFrame:
    out = cands.copy()
    widths: List[float] = []
    centers: List[float] = []
    tails: List[float] = []
    for lbl in out["bucket_label"].astype(str).tolist():
        w, c, t = _bucket_features(lbl)
        widths.append(w)
        centers.append(c)
        tails.append(t)
    out["bucket_width"] = np.asarray(widths, dtype=float)
    out["bucket_center"] = np.asarray(centers, dtype=float)
    out["tail_flag"] = np.asarray(tails, dtype=float)
    out["side_yes"] = (out["side"].astype(str) == "YES").astype(float)

    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date
    trade_ts = pd.to_datetime(out["trade_date"])
    day_of_year = trade_ts.dt.dayofyear.to_numpy(dtype=float)
    out["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)

    p_model = np.clip(pd.to_numeric(out["model_win_prob_raw"], errors="coerce").to_numpy(dtype=float), EPS, 1.0 - EPS)
    p_mkt = np.clip(pd.to_numeric(out["market_win_prob"], errors="coerce").to_numpy(dtype=float), EPS, 1.0 - EPS)
    l_model = _logit(p_model)
    l_mkt = _logit(p_mkt)
    d_raw = l_model - l_mkt
    out["d_raw"] = d_raw
    out["abs_d_raw"] = np.abs(d_raw)
    out["market_prob"] = p_mkt
    out["market_prob_sq"] = p_mkt ** 2
    out["model_prob_raw"] = p_model
    out["center_distance_proxy"] = np.abs(out["bucket_center"] - 0.5 * (out["bucket_center"].median() + out["bucket_center"]))
    return out


def _feature_columns() -> List[str]:
    return [
        "d_raw",
        "abs_d_raw",
        "side_yes",
        "market_prob",
        "market_prob_sq",
        "bucket_width",
        "tail_flag",
        "doy_sin",
        "doy_cos",
    ]


def _prepare_matrix(df: pd.DataFrame, cols: Sequence[str], means: Optional[np.ndarray], stds: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[list(cols)].to_numpy(dtype=float)
    if means is None or stds is None:
        means = np.nanmean(x, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        stds = np.nanstd(x, axis=0)
        stds = np.where((np.isfinite(stds)) & (stds > 1e-8), stds, 1.0)
    x = np.where(np.isfinite(x), x, means[None, :])
    x_std = (x - means[None, :]) / stds[None, :]
    l_mkt = _logit(df["market_win_prob"].to_numpy(dtype=float))
    y = df["outcome_win"].to_numpy(dtype=float)
    return x_std, l_mkt, y


def fit_market_offset_model(df: pd.DataFrame, l2: float) -> OffsetModel:
    cols = _feature_columns()
    x, l_mkt, y = _prepare_matrix(df, cols, None, None)
    means = np.nanmean(df[cols].to_numpy(dtype=float), axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    stds = np.nanstd(df[cols].to_numpy(dtype=float), axis=0)
    stds = np.where((np.isfinite(stds)) & (stds > 1e-8), stds, 1.0)

    def nll(beta: np.ndarray) -> float:
        z = l_mkt + beta[0] + x @ beta[1:]
        p = np.clip(_sigmoid(z), EPS, 1.0 - EPS)
        loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        reg = float(l2) * float(np.sum(beta[1:] ** 2))
        return float(loss + reg)

    init = np.zeros(x.shape[1] + 1, dtype=float)
    opt = minimize(nll, init, method="L-BFGS-B")
    beta = init if (not opt.success) else np.asarray(opt.x, dtype=float)
    return OffsetModel(feature_cols=cols, means=means, stds=stds, beta=beta)


def predict_market_offset(model: OffsetModel, df: pd.DataFrame) -> np.ndarray:
    x, l_mkt, _ = _prepare_matrix(df, model.feature_cols, model.means, model.stds)
    z = l_mkt + model.beta[0] + x @ model.beta[1:]
    return np.clip(_sigmoid(z), EPS, 1.0 - EPS)


def run_walkforward_oof_market_offset(
    cands: pd.DataFrame,
    min_train_days: int,
    block_days: int,
    l2: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.Series(cands["trade_date"]).dropna().unique().tolist())
    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[pd.DataFrame] = []
    for start in range(int(min_train_days), len(dates), int(block_days)):
        tr_dates = set(dates[:start])
        te_dates = set(dates[start : start + int(block_days)])
        if not te_dates:
            continue
        tr = cands[cands["trade_date"].isin(tr_dates)].copy()
        te = cands[cands["trade_date"].isin(te_dates)].copy()
        if tr.empty or te.empty:
            continue
        model = fit_market_offset_model(tr, l2=l2)
        te["model_win_prob_cal"] = predict_market_offset(model, te)
        te["ev_cal"] = te["model_win_prob_cal"] - te["market_win_prob"]
        pred_rows.append(te)
        fold_rows.append(
            {
                "fold_start_date": min(te_dates).isoformat(),
                "fold_end_date": max(te_dates).isoformat(),
                "train_days": int(len(tr_dates)),
                "test_days": int(len(te_dates)),
                "train_rows": int(len(tr)),
                "test_rows": int(len(te)),
                "beta_intercept": float(model.beta[0]),
            }
        )
    if not pred_rows:
        raise RuntimeError("No OOF folds produced in market-offset walk-forward.")
    return pd.DataFrame(fold_rows), pd.concat(pred_rows, ignore_index=True)


def _cell_keys(df: pd.DataFrame) -> pd.Series:
    p_bin = np.clip(np.floor(df["model_win_prob_cal"] * 10.0).astype(int), 0, 9)
    m_bin = np.clip(np.floor(df["market_win_prob"] * 10.0).astype(int), 0, 9)
    key = (
        df["side"].astype(str)
        + "|p"
        + p_bin.astype(str)
        + "|m"
        + m_bin.astype(str)
        + "|t"
        + df["tail_flag"].astype(int).astype(str)
    )
    return key


def build_lcb_table(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    g = df.groupby("cell_key", as_index=False).agg(
        n=("outcome_win", "size"),
        wins=("outcome_win", "sum"),
        mean_cal=("model_win_prob_cal", "mean"),
        mean_mkt=("market_win_prob", "mean"),
    )
    g["alpha_post"] = g["wins"] + 1.0
    g["beta_post"] = g["n"] - g["wins"] + 1.0
    g["p_lcb"] = beta_dist.ppf(float(alpha), g["alpha_post"], g["beta_post"])
    g["p_lcb"] = np.clip(g["p_lcb"], 0.0, 1.0)
    return g


def apply_lcb(
    holdout: pd.DataFrame,
    lcb_table: pd.DataFrame,
    min_cell_count: int,
) -> pd.DataFrame:
    out = holdout.copy()
    out = out.merge(lcb_table[["cell_key", "n", "p_lcb"]], on="cell_key", how="left")
    # Conservative fallback for sparse/unknown cells: shrink to market probability.
    sparse = out["n"].fillna(0) < float(min_cell_count)
    out["model_win_prob_lcb"] = np.where(sparse, out["market_win_prob"], out["p_lcb"])
    out["model_win_prob_lcb"] = np.clip(out["model_win_prob_lcb"], 0.0, 1.0)
    out["ev_lcb"] = out["model_win_prob_lcb"] - out["market_win_prob"]
    return out


def _metrics_block(df: pd.DataFrame, p_col: str) -> Dict[str, object]:
    return binary_metrics(df["outcome_win"].to_numpy(dtype=float), df[p_col].to_numpy(dtype=float))


def main() -> int:
    args = parse_args()
    model_df = build_model_frame(
        eval_path=Path(args.distribution_eval_parquet),
        pred_path=Path(args.predictions_parquet),
    )
    model_df = model_df[pd.to_datetime(model_df["target_date_local"]).dt.year == 2025].copy()

    cands = build_candidate_universe(
        kalshi_dir=Path(args.kalshi_dir),
        model_df=model_df,
        entry_hour=int(args.entry_hour_stockholm),
        entry_minute=int(args.entry_minute_stockholm),
        trade_offset_days=int(args.trade_date_offset_days),
    )
    cands = enrich_candidate_features(cands).sort_values(["trade_date", "bucket_label", "side"]).reset_index(drop=True)

    holdout_start = date.fromisoformat(args.holdout_start)
    dev = cands[cands["trade_date"] < holdout_start].copy()
    hold = cands[cands["trade_date"] >= holdout_start].copy()
    if dev.empty or hold.empty:
        raise RuntimeError("Train/holdout split is empty. Adjust --holdout-start.")

    fold_df, oof = run_walkforward_oof_market_offset(
        cands=dev,
        min_train_days=int(args.oof_min_train_days),
        block_days=int(args.oof_block_days),
        l2=float(args.l2),
    )

    oof_raw_m = _metrics_block(oof, "model_win_prob_raw")
    oof_cal_m = _metrics_block(oof, "model_win_prob_cal")

    final_model = fit_market_offset_model(dev, l2=float(args.l2))
    hold["model_win_prob_cal"] = predict_market_offset(final_model, hold)
    hold["ev_cal"] = hold["model_win_prob_cal"] - hold["market_win_prob"]
    hold["cell_key"] = _cell_keys(hold)

    dev_scored = dev.copy()
    dev_scored["model_win_prob_cal"] = predict_market_offset(final_model, dev_scored)
    dev_scored["ev_cal"] = dev_scored["model_win_prob_cal"] - dev_scored["market_win_prob"]
    dev_scored["cell_key"] = _cell_keys(dev_scored)
    lcb_table = build_lcb_table(dev_scored, alpha=float(args.lcb_alpha))
    hold_lcb = apply_lcb(hold, lcb_table=lcb_table, min_cell_count=int(args.min_cell_count))

    hold_raw_m = _metrics_block(hold, "model_win_prob_raw")
    hold_cal_m = _metrics_block(hold, "model_win_prob_cal")
    hold_lcb_m = _metrics_block(hold_lcb, "model_win_prob_lcb")

    raw_trades = pick_daily_trades(
        df=hold,
        prob_col="model_win_prob_raw",
        ev_col="ev_raw",
        min_prob=float(args.min_win_prob),
        min_ev=float(args.ev_threshold),
    )
    cal_trades = pick_daily_trades(
        df=hold,
        prob_col="model_win_prob_cal",
        ev_col="ev_cal",
        min_prob=float(args.min_win_prob),
        min_ev=float(args.ev_threshold),
    )
    lcb_trades = pick_daily_trades(
        df=hold_lcb,
        prob_col="model_win_prob_lcb",
        ev_col="ev_lcb",
        min_prob=float(args.min_win_prob),
        min_ev=float(args.ev_threshold),
    )

    bt_raw = settlement_backtest(raw_trades, float(args.start_balance), float(args.risk_fraction))
    bt_cal = settlement_backtest(cal_trades, float(args.start_balance), float(args.risk_fraction))
    bt_lcb = settlement_backtest(lcb_trades, float(args.start_balance), float(args.risk_fraction))

    summary = {
        "config": {
            "entry_time_stockholm": f"{int(args.entry_hour_stockholm):02d}:{int(args.entry_minute_stockholm):02d}",
            "holdout_start": args.holdout_start,
            "ev_threshold": float(args.ev_threshold),
            "min_win_prob": float(args.min_win_prob),
            "oof_min_train_days": int(args.oof_min_train_days),
            "oof_block_days": int(args.oof_block_days),
            "l2": float(args.l2),
            "lcb_alpha": float(args.lcb_alpha),
            "min_cell_count": int(args.min_cell_count),
        },
        "counts": {
            "candidate_rows_2025": int(len(cands)),
            "dev_rows": int(len(dev)),
            "holdout_rows": int(len(hold)),
            "oof_rows": int(len(oof)),
            "oof_folds": int(len(fold_df)),
        },
        "final_model": {
            "feature_cols": final_model.feature_cols,
            "beta": [float(x) for x in final_model.beta.tolist()],
        },
        "oof_metrics_raw": {k: v for k, v in oof_raw_m.items() if k != "reliability_10"},
        "oof_metrics_cal": {k: v for k, v in oof_cal_m.items() if k != "reliability_10"},
        "holdout_metrics_raw": {k: v for k, v in hold_raw_m.items() if k != "reliability_10"},
        "holdout_metrics_cal": {k: v for k, v in hold_cal_m.items() if k != "reliability_10"},
        "holdout_metrics_lcb": {k: v for k, v in hold_lcb_m.items() if k != "reliability_10"},
        "holdout_selected_trades_raw": bt_raw,
        "holdout_selected_trades_cal": bt_cal,
        "holdout_selected_trades_lcb": bt_lcb,
    }

    run_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"market_offset_policy_backtest_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cands.to_csv(out_dir / "candidate_universe_2025.csv", index=False, encoding="utf-8")
    fold_df.to_csv(out_dir / "oof_folds.csv", index=False, encoding="utf-8")
    oof.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    hold.to_csv(out_dir / "holdout_predictions.csv", index=False, encoding="utf-8")
    hold_lcb.to_csv(out_dir / "holdout_predictions_lcb.csv", index=False, encoding="utf-8")
    lcb_table.to_csv(out_dir / "lcb_table.csv", index=False, encoding="utf-8")
    raw_trades.to_csv(out_dir / "holdout_selected_trades_raw.csv", index=False, encoding="utf-8")
    cal_trades.to_csv(out_dir / "holdout_selected_trades_cal.csv", index=False, encoding="utf-8")
    lcb_trades.to_csv(out_dir / "holdout_selected_trades_lcb.csv", index=False, encoding="utf-8")
    oof_raw_m["reliability_10"].to_csv(out_dir / "oof_reliability_raw_10bins.csv", index=False, encoding="utf-8")
    oof_cal_m["reliability_10"].to_csv(out_dir / "oof_reliability_cal_10bins.csv", index=False, encoding="utf-8")
    hold_raw_m["reliability_10"].to_csv(out_dir / "holdout_reliability_raw_10bins.csv", index=False, encoding="utf-8")
    hold_cal_m["reliability_10"].to_csv(out_dir / "holdout_reliability_cal_10bins.csv", index=False, encoding="utf-8")
    hold_lcb_m["reliability_10"].to_csv(out_dir / "holdout_reliability_lcb_10bins.csv", index=False, encoding="utf-8")

    print(f"summary_json: {out_dir / 'summary.json'}")
    print(f"candidate_universe_csv: {out_dir / 'candidate_universe_2025.csv'}")
    print(f"oof_predictions_csv: {out_dir / 'oof_predictions.csv'}")
    print(f"holdout_predictions_csv: {out_dir / 'holdout_predictions.csv'}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

