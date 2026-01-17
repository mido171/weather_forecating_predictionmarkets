from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from weather_ml import train


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run NAM/GFS baseline (TOD) for KMIA."
    )
    parser.add_argument("--run-id", help="Optional run id override.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    config_path = (
        repo_root / "configs" / "train_nam_gfs_sweep_01_base_kmia_tod.yaml"
    )

    run_id = args.run_id
    if not run_id:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"nam_gfs_sweep_01_base_tod_kmia_{timestamp}"

    argv = ["--config", str(config_path), "--run-id", run_id]
    return train.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
