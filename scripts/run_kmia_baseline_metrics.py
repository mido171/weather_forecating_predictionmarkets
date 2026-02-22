from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BLEND_COLS = [
    "nbm_tmax_f",
    "hrrr_tmax_f",
    "rap_tmax_f",
    "gefsatmosmean_tmax_f",
    "gfs_n_x_max",
    "nam_n_x_max",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = (
        repo_root
        / "ingestion-service"
        / "src"
        / "main"
        / "resources"
        / "trainingdata_output"
        / "KMIA_mos_training_data.csv"
    )
    output_dir = (
        repo_root
        / "ingestion-service"
        / "src"
        / "main"
        / "resources"
        / "baseline"
    )
    return csv_path, output_dir


def _blend_mean(values: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(values)
    counts = valid.sum(axis=1)
    sums = np.where(valid, values, 0.0).sum(axis=1)
    return np.where(counts > 0, sums / counts, np.nan)


def _parse_blend_cols(raw: str) -> list[str]:
    if not raw:
        return DEFAULT_BLEND_COLS
    return [col.strip() for col in raw.split(",") if col.strip()]


def compute_metrics(
    df: pd.DataFrame,
    target_col: str,
    start_date: str,
    end_date: str,
    blend_cols: list[str],
) -> dict:
    df = df.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.normalize()
    missing_cols = [column for column in blend_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing blend columns in CSV: {missing_cols}")
    for column in blend_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    df = df[(df["target_date_local"] >= start_ts) & (df["target_date_local"] <= end_ts)]
    if df.empty:
        raise ValueError("No rows remain after date filtering.")

    df["blend_mean"] = _blend_mean(df[blend_cols].to_numpy(dtype=float))
    df["target_date_local"] = df["target_date_local"].dt.date

    daily = (
        df.groupby("target_date_local", as_index=False)
        .agg(blend_mean=("blend_mean", "mean"), target=(target_col, "mean"))
        .dropna(subset=["blend_mean", "target"])
    )
    if daily.empty:
        raise ValueError("No daily rows available after aggregation.")

    errors = daily["blend_mean"].to_numpy() - daily["target"].to_numpy()
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    bias = float(np.mean(errors))
    p50 = float(np.percentile(abs_errors, 50))
    p90 = float(np.percentile(abs_errors, 90))
    p95 = float(np.percentile(abs_errors, 95))

    return {
        "metrics": {
            "bias": bias,
            "mae": mae,
            "p50_abs_error": p50,
            "p90_abs_error": p90,
            "p95_abs_error": p95,
            "rmse": rmse,
        },
        "blend_columns": blend_cols,
        "daily_rows": int(len(daily)),
        "start_date": str(daily["target_date_local"].min()),
        "end_date": str(daily["target_date_local"].max()),
    }


def main() -> int:
    default_csv, default_output = _resolve_default_paths()
    parser = argparse.ArgumentParser(
        description="Compute KMIA blended max baseline metrics.",
    )
    parser.add_argument("--csv-path", default=str(default_csv))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--output-name", default="kmia_blended_model_mean_metrics.json")
    parser.add_argument(
        "--blend-cols",
        default="",
        help="Comma-separated columns to blend. Defaults to the standard model set.",
    )
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"station_id": "string"})
    if "target_tmax_f" in df.columns:
        target_col = "target_tmax_f"
    elif "actual_tmax_f" in df.columns:
        target_col = "actual_tmax_f"
    else:
        raise ValueError("Missing target column: expected target_tmax_f or actual_tmax_f.")

    blend_cols = _parse_blend_cols(args.blend_cols)
    payload = compute_metrics(df, target_col, args.start_date, args.end_date, blend_cols)
    payload.update(
        {
            "created_utc": utc_now_iso(),
            "csv_path": str(csv_path),
            "target_col": target_col,
            "blend_method": "mean_of_selected_model_columns",
            "daily_aggregation": "mean_over_asof",
            "filter_start": args.start_date,
            "filter_end": args.end_date,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / args.output_name
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
