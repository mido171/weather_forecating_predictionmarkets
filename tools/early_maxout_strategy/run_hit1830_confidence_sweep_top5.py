import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def _load_summary(summary_path: Path) -> Dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _rank_top5(summary: Dict) -> List[Tuple[str, float, float]]:
    rows = []
    for exp, metrics in summary.items():
        net = metrics.get("net_units_per_100_test_profit")
        test_acc = metrics.get("test", {}).get("accuracy")
        if net is None or test_acc is None:
            continue
        rows.append((exp, float(net), float(test_acc)))
    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return rows[:5]


def _resolve_pred_paths(base_dir: Path, exp_name: str) -> Tuple[Path, Path]:
    exp_dir = base_dir / exp_name
    val_path = exp_dir / "preds_val.parquet"
    test_path = exp_dir / "preds_test.parquet"
    if not val_path.exists():
        raise FileNotFoundError(f"Missing val preds: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test preds: {test_path}")
    return val_path, test_path


def _extract_columns(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    if "y_true" in df.columns:
        y = df["y_true"]
    elif "y_hit_by_cutoff" in df.columns:
        y = df["y_hit_by_cutoff"]
    else:
        raise ValueError("No y_true/y_hit_by_cutoff column found.")
    y = y.astype(int)
    if not set(np.unique(y.dropna())).issubset({0, 1}):
        raise ValueError("y_true must be binary 0/1.")

    probs = {}
    if "p_cal" in df.columns:
        probs["p_cal"] = df["p_cal"]
    if "p_raw" in df.columns:
        probs["p_raw"] = df["p_raw"]
    if not probs:
        raise ValueError("No p_raw or p_cal columns found.")
    return y, probs


def _compute_drawdown(outcomes: np.ndarray) -> float:
    cum = outcomes.cumsum()
    running_max = np.maximum.accumulate(cum)
    drawdown = running_max - cum
    return float(np.max(drawdown)) if len(drawdown) else 0.0


def _longest_loss_streak(trade_flags: np.ndarray, outcomes: np.ndarray) -> int:
    streak = 0
    best = 0
    for trade, out in zip(trade_flags, outcomes):
        if not trade:
            continue
        if out == -1:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return int(best)


def _sweep(df: pd.DataFrame, y: pd.Series, p: pd.Series, thresholds: List[float]) -> List[Dict]:
    out = []
    df = df.copy()
    df["y_true"] = y.values
    df["p_yes"] = p.values

    # Drop NaNs in p_yes and report count
    before = len(df)
    df = df.dropna(subset=["p_yes", "y_true"]).copy()
    dropped = before - len(df)

    # Ensure ordering for streaks/drawdown
    if "target_date_local" in df.columns:
        df = df.sort_values("target_date_local")

    y_arr = df["y_true"].astype(int).to_numpy()
    p_arr = df["p_yes"].to_numpy()

    for thr in thresholds:
        trade = p_arr >= thr
        wins = (trade & (y_arr == 1))
        losses = (trade & (y_arr == 0))
        tp = int(wins.sum())
        fp = int(losses.sum())
        n = len(y_arr)
        n_trades = tp + fp
        trade_rate = n_trades / n if n else 0.0
        net_units = tp - fp
        net_per_100_days = net_units / n * 100 if n else 0.0
        win_rate = tp / n_trades if n_trades else 0.0
        net_per_100_trades = net_units / n_trades * 100 if n_trades else 0.0

        outcomes = np.where(wins, 1, np.where(losses, -1, 0))
        max_drawdown = _compute_drawdown(outcomes)
        max_loss_streak = _longest_loss_streak(trade, outcomes)

        out.append({
            "threshold": float(thr),
            "N": int(n),
            "dropped_nan": int(dropped),
            "TP": tp,
            "FP": fp,
            "n_trades": int(n_trades),
            "trade_rate": float(trade_rate),
            "win_rate_on_trades": float(win_rate),
            "net_units_per_100_days": float(net_per_100_days),
            "net_units_per_100_trades": float(net_per_100_trades),
            "max_drawdown": float(max_drawdown),
            "longest_loss_streak": int(max_loss_streak),
        })
    return out


def _best_threshold(sweep: List[Dict], key: str) -> float:
    best = None
    for row in sweep:
        val = row.get(key)
        if best is None or val > best[1] or (val == best[1] and row["threshold"] > best[0]):
            best = (row["threshold"], val)
    return float(best[0]) if best else 0.5


def _metric_at_threshold(sweep: List[Dict], thr: float) -> Dict:
    for row in sweep:
        if abs(row["threshold"] - thr) < 1e-9:
            return row
    return {}


def main() -> int:
    base_dir = Path(r"C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets\artifacts\experiments\early_maxout_strategy\B4")
    summary_path = base_dir / "hit1830_v3_experiments_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")

    summary = _load_summary(summary_path)
    ranked = _rank_top5(summary)
    if len(ranked) < 5:
        raise RuntimeError("Fewer than 5 experiments available in summary.")

    top5 = []
    experiments = {}

    for exp_name, net, acc in ranked:
        val_path, test_path = _resolve_pred_paths(base_dir, exp_name)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)

        if len(test_df) < 500:
            raise RuntimeError(f"Test rows < 500 for {exp_name}: {len(test_df)}")

        y_val, probs_val = _extract_columns(val_df)
        y_test, probs_test = _extract_columns(test_df)

        exp_entry = {
            "rank_net_units_per_100_test": net,
            "rank_test_accuracy": acc,
            "paths": {
                "preds_val": str(val_path),
                "preds_test": str(test_path),
            },
            "prob_variants": {},
        }

        for prob_key in probs_val.keys():
            p_val = probs_val[prob_key]
            p_test = probs_test.get(prob_key)
            if p_test is None:
                # fallback to raw if cal missing in test
                p_test = probs_test.get("p_raw")
            val_sweep = _sweep(val_df, y_val, p_val, THRESHOLDS)
            test_sweep = _sweep(test_df, y_test, p_test, THRESHOLDS)

            thr_best_netdays = _best_threshold(val_sweep, "net_units_per_100_days")
            thr_best_nettrades = _best_threshold(val_sweep, "net_units_per_100_trades")

            test_at_best_netdays = _metric_at_threshold(test_sweep, thr_best_netdays)
            test_at_best_nettrades = _metric_at_threshold(test_sweep, thr_best_nettrades)

            best_test_netdays = max(test_sweep, key=lambda r: r["net_units_per_100_days"])

            exp_entry["prob_variants"][prob_key] = {
                "val_sweep": val_sweep,
                "test_sweep": test_sweep,
                "best_val_netdays_threshold": thr_best_netdays,
                "best_val_nettrades_threshold": thr_best_nettrades,
                "test_at_best_val_netdays": test_at_best_netdays,
                "test_at_best_val_nettrades": test_at_best_nettrades,
                "best_test_netdays": best_test_netdays,
            }

        top5.append({"exp": exp_name, "rank_net_units_per_100_test": net, "rank_test_accuracy": acc})
        experiments[exp_name] = exp_entry

    out_json = {
        "top5": top5,
        "experiments": experiments,
        "thresholds": THRESHOLDS,
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = reports_dir / "hit1830_confidence_sweep_top5.json"
    out_md_path = reports_dir / "hit1830_confidence_sweep_top5.md"
    out_json_path.write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # Build Markdown report
    lines = []
    lines.append("# HIT1830 Confidence Sweep — Top 5 Models\n")

    # Winner summary
    lines.append("## Winner Summary (Best Test Net Units/100 Days)\n")
    lines.append("| Experiment | Prob Variant | Best Test Net/100 Days | Threshold | Trade Rate | TP | FP | Win Rate | Max Drawdown | Longest Loss Streak |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for exp in top5:
        exp_name = exp["exp"]
        exp_entry = experiments[exp_name]
        for prob_key, details in exp_entry["prob_variants"].items():
            best = details["best_test_netdays"]
            lines.append(
                f"| {exp_name} | {prob_key} | {best['net_units_per_100_days']:.2f} | {best['threshold']:.2f} | {best['trade_rate']:.3f} | {best['TP']} | {best['FP']} | {best['win_rate_on_trades']:.3f} | {best['max_drawdown']:.2f} | {best['longest_loss_streak']} |\n"
            )
    lines.append("\n")

    # Per-experiment tables
    for exp in top5:
        exp_name = exp["exp"]
        exp_entry = experiments[exp_name]
        lines.append(f"## {exp_name}\n")
        for prob_key, details in exp_entry["prob_variants"].items():
            lines.append(f"### Prob Variant: {prob_key}\n")
            lines.append("| Threshold | Trade Rate | TP | FP | Win Rate | Net/100 Days | Net/100 Trades | Max Drawdown | Longest Loss Streak |\n")
            lines.append("|---|---|---|---|---|---|---|---|---|\n")
            for row in details["test_sweep"]:
                lines.append(
                    f"| {row['threshold']:.2f} | {row['trade_rate']:.3f} | {row['TP']} | {row['FP']} | {row['win_rate_on_trades']:.3f} | {row['net_units_per_100_days']:.2f} | {row['net_units_per_100_trades']:.2f} | {row['max_drawdown']:.2f} | {row['longest_loss_streak']} |\n"
                )

            lines.append("\n")
            lines.append(f"Best threshold by VAL net/100 days: {details['best_val_netdays_threshold']:.2f}\n\n")
            lines.append(
                f"TEST at that threshold: net/100 days {details['test_at_best_val_netdays']['net_units_per_100_days']:.2f}, "
                f"trade rate {details['test_at_best_val_netdays']['trade_rate']:.3f}, "
                f"win rate {details['test_at_best_val_netdays']['win_rate_on_trades']:.3f}\n\n"
            )
            lines.append(f"Best threshold by VAL net/100 trades: {details['best_val_nettrades_threshold']:.2f}\n\n")
            lines.append(
                f"TEST at that threshold: net/100 trades {details['test_at_best_val_nettrades']['net_units_per_100_trades']:.2f}, "
                f"trade rate {details['test_at_best_val_nettrades']['trade_rate']:.3f}, "
                f"win rate {details['test_at_best_val_nettrades']['win_rate_on_trades']:.3f}\n\n"
            )

    out_md_path.write_text("".join(lines), encoding="utf-8")

    # Also copy into B4 for convenience
    (base_dir / out_json_path.name).write_text(out_json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (base_dir / out_md_path.name).write_text(out_md_path.read_text(encoding="utf-8"), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
