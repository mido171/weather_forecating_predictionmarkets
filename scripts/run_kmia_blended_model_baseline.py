from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "ml" / "data" / "gribstream" / "KMIA_gribstream_cli_training_data.csv"
    output_dir = (
        repo_root
        / "ingestion-service"
        / "src"
        / "main"
        / "resources"
        / "blended_baseline"
    )
    return csv_path, output_dir


def _detect_target_col(df: pd.DataFrame) -> str:
    if "target_tmax_f" in df.columns:
        return "target_tmax_f"
    if "actual_tmax_f" in df.columns:
        return "actual_tmax_f"
    raise ValueError("Missing target column: expected target_tmax_f or actual_tmax_f.")


def _detect_model_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    model_cols = [
        col
        for col in df.columns
        if col.endswith("_tmax_f") and col not in {target_col, "actual_tmax_f"}
    ]
    if not model_cols:
        raise ValueError("No model forecast columns detected for blending.")
    return model_cols


def compute_metrics(
    df: pd.DataFrame,
    *,
    station_id: str,
    target_col: str,
    model_cols: list[str],
    start_date: str,
    end_date: str,
) -> dict:
    df = df.copy()
    df = df[df["station_id"].astype(str) == station_id]
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.normalize()

    for col in model_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    df = df[(df["target_date_local"] >= start_ts) & (df["target_date_local"] <= end_ts)]
    if df.empty:
        raise ValueError("No rows remain after date filtering.")

    df["blend_mean"] = np.nanmean(df[model_cols].to_numpy(dtype=float), axis=1)
    df["target_date_local"] = df["target_date_local"].dt.date

    daily = (
        df.groupby("target_date_local", as_index=False)
        .agg(blend_mean=("blend_mean", "mean"), target=(target_col, "mean"))
        .dropna(subset=["blend_mean", "target"])
    )
    if daily.empty:
        raise ValueError("No daily rows available after aggregation.")

    errors = daily["blend_mean"].to_numpy() - daily["target"].to_numpy()
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    bias = float(np.mean(errors))

    return {
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
        },
        "daily_rows": int(len(daily)),
        "start_date": str(daily["target_date_local"].min()),
        "end_date": str(daily["target_date_local"].max()),
    }


def main() -> int:
    default_csv, default_output = _resolve_default_paths()
    parser = argparse.ArgumentParser(
        description="Compute KMIA blended model-mean baseline metrics.",
    )
    parser.add_argument("--csv-path", default=str(default_csv))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--station-id", default="KMIA")
    parser.add_argument("--start-date", default="2021-02-23")
    parser.add_argument("--end-date", default="2025-12-31")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"station_id": "string"})
    target_col = _detect_target_col(df)
    model_cols = _detect_model_cols(df, target_col)

    payload = compute_metrics(
        df,
        station_id=args.station_id,
        target_col=target_col,
        model_cols=model_cols,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    payload.update(
        {
            "created_utc": utc_now_iso(),
            "csv_path": str(csv_path),
            "station_id": args.station_id,
            "target_col": target_col,
            "model_columns": model_cols,
            "blend_method": "mean_of_model_tmax_columns",
            "daily_aggregation": "mean_over_asof",
            "filter_start": args.start_date,
            "filter_end": args.end_date,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.station_id.lower()}_blended_model_mean_baseline_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
