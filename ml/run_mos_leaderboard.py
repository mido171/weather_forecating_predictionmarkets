"""Build a MOS experiment leaderboard with calibration metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err)) if len(abs_err) else float("nan")
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": float(np.mean(err)) if len(err) else float("nan"),
        "p50_abs_error": float(np.quantile(abs_err, 0.5)) if len(abs_err) else float("nan"),
        "p90_abs_error": float(np.quantile(abs_err, 0.9)) if len(abs_err) else float("nan"),
        "p95_abs_error": float(np.quantile(abs_err, 0.95)) if len(abs_err) else float("nan"),
    }


def calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    if not len(abs_err):
        return {}
    return {
        "abs_error_mean": float(np.mean(abs_err)),
        "abs_error_std": float(np.std(abs_err)),
        "coverage_abs_1": float(np.mean(abs_err <= 1.0)),
        "coverage_abs_2": float(np.mean(abs_err <= 2.0)),
        "coverage_abs_3": float(np.mean(abs_err <= 3.0)),
        "coverage_abs_5": float(np.mean(abs_err <= 5.0)),
    }


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, *, label: str) -> dict[str, float]:
    mask = ~np.isnan(lower) & ~np.isnan(upper)
    if not mask.any():
        return {}
    cov = float(np.mean((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])))
    width = float(np.mean(upper[mask] - lower[mask]))
    return {
        f"{label}_coverage": cov,
        f"{label}_avg_width": width,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MOS experiment leaderboard JSON.")
    parser.add_argument("--experiments-root", required=True, help="Path to suite output folder.")
    parser.add_argument("--out-dir", default="artifacts/MOS/leaderboards", help="Output directory for leaderboard JSON.")
    args = parser.parse_args()

    root = Path(args.experiments_root)
    if not root.exists():
        raise FileNotFoundError(f"Experiments root not found: {root}")

    summary_path = root / "experiments_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing experiments_summary.json at {summary_path}")
    summary = load_json(summary_path)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for exp in summary.get("experiments", []):
        exp_id = exp.get("experiment_id")
        name = exp.get("name")
        exp_dir = root / exp_id
        pred_path = exp_dir / "predictions_test.csv"
        if not pred_path.exists():
            warnings.append(f"Missing predictions_test.csv for {exp_id}")
            continue
        df_pred = pd.read_csv(pred_path)
        y_true = df_pred["y_true"].to_numpy(dtype=float)
        y_pred = df_pred["y_pred"].to_numpy(dtype=float)
        test_metrics = regression_metrics(y_true, y_pred)
        calib = calibration_metrics(y_true, y_pred)
        if "q10" in df_pred.columns and "q90" in df_pred.columns:
            q10 = df_pred["q10"].to_numpy(dtype=float)
            q90 = df_pred["q90"].to_numpy(dtype=float)
            calib.update(interval_metrics(y_true, q10, q90, label="quantile_80"))
        if "lower" in df_pred.columns and "upper" in df_pred.columns:
            lower = df_pred["lower"].to_numpy(dtype=float)
            upper = df_pred["upper"].to_numpy(dtype=float)
            calib.update(interval_metrics(y_true, lower, upper, label="conformal_80"))
        extras = exp.get("extras", {})
        if isinstance(extras, dict):
            for key in ["coverage_80_test"]:
                if key in extras:
                    calib[key] = extras[key]
        rows.append(
            {
                "experiment_id": exp_id,
                "name": name,
                "n_test": int(len(y_true)),
                "metrics": {
                    "train": exp.get("metrics", {}).get("train"),
                    "validation": exp.get("metrics", {}).get("validation"),
                    "test": test_metrics,
                },
                "calibration": calib,
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["metrics"]["test"]["mae"])
    for rank, row in enumerate(rows_sorted, start=1):
        row["rank"] = rank
        row["delta_mae_vs_best"] = float(row["metrics"]["test"]["mae"] - rows_sorted[0]["metrics"]["test"]["mae"])

    payload = {
        "created_utc": utc_now_iso(),
        "suite_id": summary.get("suite_id"),
        "source": {
            "experiments_root": str(root.resolve()),
            "summary_json": str(summary_path.resolve()),
            "csv_path": summary.get("csv_path"),
            "split": summary.get("split"),
        },
        "leaderboard": {
            "sort_metric": "test.mae",
            "rows": rows_sorted,
        },
        "warnings": warnings,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"leaderboard_{summary.get('suite_id','unknown')}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
