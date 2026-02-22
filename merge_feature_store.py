from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge E37 feature store with minute features.")
    parser.add_argument("--features-csv", required=True, help="Path to MOS features.csv")
    parser.add_argument("--minute-features", required=True, help="Path to minute features parquet")
    parser.add_argument("--out", required=True, help="Output parquet path")
    args = parser.parse_args()

    # Build E37 feature store exactly as run_e37_only does
    import sys

    sys.path.append("ml")
    import run_mos_45_suite as base

    df_raw = base.load_csv(args.features_csv)
    feature_store = base.build_feature_store(df_raw)
    feature_store = feature_store[feature_store["y_actual_tmax_f"].notna()].copy()

    minute = pd.read_parquet(args.minute_features)

    feature_store["target_date_local"] = pd.to_datetime(feature_store["target_date_local"]).dt.date
    minute["target_date_local"] = pd.to_datetime(minute["target_date_local"]).dt.date

    merged = feature_store.merge(minute, on="target_date_local", how="left", suffixes=("", "_minute"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
