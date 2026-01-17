from __future__ import annotations

import argparse
from pathlib import Path

from weather_ml import train


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train NAM/GFS MOS base features for daily TMAX."
    )
    parser.add_argument("--run-id", help="Optional run id override.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    csv_path = repo_root / "data" / "mos" / "mos_training_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {csv_path}. Copy mos_training_data.csv into ml/data/mos."
        )

    config_path = repo_root / "configs" / "train_nam_gfs_sweep_01_base.yaml"
    argv = ["--config", str(config_path)]
    if args.run_id:
        argv.extend(["--run-id", args.run_id])
    return train.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
