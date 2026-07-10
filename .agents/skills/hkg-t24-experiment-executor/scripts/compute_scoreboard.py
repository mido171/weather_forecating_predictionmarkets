#!/usr/bin/env python3
"""Compute an identical-row HKG T+24 candidate-versus-baseline scoreboard.

The input must contain one row per target date and out-of-fold predictions.
This utility never fits a model. It only scores precomputed predictions after
forcing the baseline and candidate onto the same non-null rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "compute_scoreboard.py requires pandas and numpy in the project environment"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--date-column", default="target_date")
    parser.add_argument("--actual-column", default="actual_tmax_c")
    parser.add_argument("--baseline-column", required=True)
    parser.add_argument("--candidate-column", required=True)
    parser.add_argument("--fold-column", default="fold_id")
    parser.add_argument("--source-column", default="forecast_source")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--development-end-exclusive", default="2024-01-01")
    parser.add_argument(
        "--owner-authorized-confirmation",
        action="store_true",
        help="Must stay false during ordinary development.",
    )
    return parser.parse_args()


def load_table(path: Path) -> "pd.DataFrame":
    name = path.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    if name.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    if name.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Predictions must be parquet, csv, or csv.gz")


def metrics(actual: "pd.Series", pred: "pd.Series") -> dict[str, float | int]:
    error = pred.astype(float) - actual.astype(float)
    absolute = error.abs()
    return {
        "n": int(len(error)),
        "mae_c": float(absolute.mean()),
        "rmse_c": float(np.sqrt(np.mean(np.square(error)))),
        "bias_c": float(error.mean()),
        "median_ae_c": float(absolute.median()),
        "p90_ae_c": float(absolute.quantile(0.90)),
        "p95_ae_c": float(absolute.quantile(0.95)),
        "max_ae_c": float(absolute.max()),
        "mean_underforecast_c": float((-error[error < 0]).mean()) if (error < 0).any() else 0.0,
        "mean_overforecast_c": float(error[error > 0].mean()) if (error > 0).any() else 0.0,
        "underforecast_rate": float((error < 0).mean()),
        "overforecast_rate": float((error > 0).mean()),
    }


def season(month: int) -> str:
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    if month in (9, 10, 11):
        return "SON"
    return "DJF"


def score_group(frame: "pd.DataFrame", group_type: str, group_value: str, args: argparse.Namespace) -> list[dict]:
    if frame.empty:
        return []
    base = metrics(frame[args.actual_column], frame[args.baseline_column])
    cand = metrics(frame[args.actual_column], frame[args.candidate_column])
    return [
        {
            "group_type": group_type,
            "group_value": group_value,
            "model_id": args.baseline_id,
            **base,
            "mae_delta_vs_baseline_c": 0.0,
        },
        {
            "group_type": group_type,
            "group_value": group_value,
            "model_id": args.candidate_id,
            **cand,
            "mae_delta_vs_baseline_c": cand["mae_c"] - base["mae_c"],
        },
    ]


def main() -> int:
    args = parse_args()
    frame = load_table(args.predictions)
    required = {
        args.date_column,
        args.actual_column,
        args.baseline_column,
        args.candidate_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="raise")
    if not args.owner_authorized_confirmation:
        end = pd.Timestamp(args.development_end_exclusive)
        offending = frame[frame[args.date_column] >= end]
        if not offending.empty:
            raise ValueError(
                f"Found {len(offending)} sealed confirmation rows at or after "
                f"{args.development_end_exclusive}. Filter them before scoring."
            )

    # One target date must identify one score row.
    if frame[args.date_column].duplicated().any():
        duplicated = frame.loc[frame[args.date_column].duplicated(False), args.date_column]
        raise ValueError(
            "Duplicate target dates are prohibited in a daily score table; examples: "
            + ", ".join(str(v.date()) for v in duplicated.head(5))
        )

    common = frame.dropna(
        subset=[args.actual_column, args.baseline_column, args.candidate_column]
    ).copy()
    if common.empty:
        raise ValueError("No identical non-null rows are available for scoring")
    common = common.sort_values(args.date_column).reset_index(drop=True)
    key_text = "\n".join(
        pd.to_datetime(common[args.date_column]).dt.strftime("%Y-%m-%d").tolist()
    )
    common_row_hash = hashlib.sha256(key_text.encode("utf-8")).hexdigest()

    baseline_metrics = metrics(common[args.actual_column], common[args.baseline_column])
    candidate_metrics = metrics(common[args.actual_column], common[args.candidate_column])
    scoreboard = pd.DataFrame([
        {
            "candidate_id": args.baseline_id,
            "role": "baseline",
            **baseline_metrics,
            "mae_delta_vs_baseline_c": 0.0,
            "common_row_hash": common_row_hash,
        },
        {
            "candidate_id": args.candidate_id,
            "role": "candidate",
            **candidate_metrics,
            "mae_delta_vs_baseline_c": (
                candidate_metrics["mae_c"] - baseline_metrics["mae_c"]
            ),
            "common_row_hash": common_row_hash,
        },
    ])

    common["year"] = common[args.date_column].dt.year
    common["month"] = common[args.date_column].dt.month
    common["season"] = common["month"].map(season)
    yearly_rows: list[dict] = []
    for year, group in common.groupby("year", sort=True):
        yearly_rows.extend(score_group(group, "year", str(year), args))
    slice_rows: list[dict] = []
    for label, group in common.groupby("season", sort=True):
        slice_rows.extend(score_group(group, "season", str(label), args))
    for month, group in common.groupby("month", sort=True):
        slice_rows.extend(score_group(group, "month", f"{int(month):02d}", args))
    if args.source_column in common.columns:
        for source, group in common.groupby(args.source_column, dropna=False, sort=True):
            slice_rows.extend(score_group(group, "source", str(source), args))

    fold_rows: list[dict] = []
    if args.fold_column in common.columns:
        for fold, group in common.groupby(args.fold_column, dropna=False, sort=True):
            fold_rows.extend(score_group(group, "fold", str(fold), args))
    else:
        fold_rows.extend(score_group(common, "fold", "UNSPECIFIED", args))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    scoreboard.to_csv(out / "scoreboard.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(out / "yearly_metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(out / "slice_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)

    coverage = pd.DataFrame([
        {
            "input_rows": len(frame),
            "common_rows": len(common),
            "dropped_any_score_null": len(frame) - len(common),
            "date_start": common[args.date_column].min().date().isoformat(),
            "date_end": common[args.date_column].max().date().isoformat(),
            "common_row_hash": common_row_hash,
            "confirmation_rows_used": int(
                (common[args.date_column] >= pd.Timestamp("2024-01-01")).sum()
            ),
        }
    ])
    coverage.to_csv(out / "row_coverage.csv", index=False)

    scored = common[
        [
            args.date_column,
            args.actual_column,
            args.baseline_column,
            args.candidate_column,
        ]
        + ([args.fold_column] if args.fold_column in common.columns else [])
        + ([args.source_column] if args.source_column in common.columns else [])
    ].copy()
    scored["baseline_error_c"] = (
        scored[args.baseline_column] - scored[args.actual_column]
    )
    scored["candidate_error_c"] = (
        scored[args.candidate_column] - scored[args.actual_column]
    )
    try:
        scored.to_parquet(out / "predictions.parquet", index=False)
        prediction_path = "predictions.parquet"
    except Exception:
        scored.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
        prediction_path = "predictions.csv.gz"

    fragment = {
        "date_start": common[args.date_column].min().date().isoformat(),
        "date_end": common[args.date_column].max().date().isoformat(),
        "n_candidate": int(common[args.candidate_column].notna().sum()),
        "n_common": int(len(common)),
        "baseline_id": args.baseline_id,
        "baseline_mae_c": baseline_metrics["mae_c"],
        "candidate_id": args.candidate_id,
        "candidate_mae_c": candidate_metrics["mae_c"],
        "mae_delta_c": candidate_metrics["mae_c"] - baseline_metrics["mae_c"],
        "candidate_rmse_c": candidate_metrics["rmse_c"],
        "candidate_bias_c": candidate_metrics["bias_c"],
        "common_row_hash": common_row_hash,
        "baseline_n": int(len(common)),
        "candidate_n": int(len(common)),
        "confirmation_rows_used": int(
            (common[args.date_column] >= pd.Timestamp("2024-01-01")).sum()
        ),
        "prediction_artifact": prediction_path,
    }
    (out / "score_summary_fragment.json").write_text(
        json.dumps(fragment, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(fragment, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
