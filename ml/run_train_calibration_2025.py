from __future__ import annotations

import argparse
from pathlib import Path

from weather_ml import train


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train mean model and run 2025 global normal calibration."
    )
    parser.add_argument("--run-id", help="Optional run id override.")
    args = parser.parse_args()

    config_path = (
        Path(__file__).resolve().parent
        / "configs"
        / "train_mean_sigma_lgbm_cal2025.yaml"
    )
    argv = ["--config", str(config_path)]
    if args.run_id:
        argv.extend(["--run-id", args.run_id])
    return train.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
