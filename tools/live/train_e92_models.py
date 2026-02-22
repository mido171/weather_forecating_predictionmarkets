from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import yaml

from ml_live.modeling.train_e92 import TrainingConfig, train_e92_models
from ml_live.runtime.logging import configure_logging
from ml_live.runtime.paths import config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train E92 mean and sigma models for KMIA.")
    parser.add_argument("--station", default="KMIA", help="Station ID (default: KMIA).")
    parser.add_argument("--train-start", default="2021-03-22", help="Training start date (YYYY-MM-DD).")
    parser.add_argument("--train-end", default="2024-12-31", help="Training end date (YYYY-MM-DD).")
    parser.add_argument("--dataset-path", help="Override dataset parquet path.")
    parser.add_argument("--feature-list-path", help="Override feature_list.json path.")
    parser.add_argument("--config", help="Path to live_kmia.yaml config.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> int:
    args = parse_args()
    logger = configure_logging(verbose=args.verbose)

    cfg_path = Path(args.config) if args.config else config_path()
    config = load_yaml(cfg_path)
    station_id = args.station or (config.get("station_id") or "KMIA")

    cfg = TrainingConfig(
        station_id=station_id,
        train_start=date.fromisoformat(args.train_start),
        train_end=date.fromisoformat(args.train_end),
    )

    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    feature_list_path = Path(args.feature_list_path) if args.feature_list_path else None

    artifacts = train_e92_models(cfg, dataset_path=dataset_path, feature_list_path=feature_list_path)
    logger.info("Training complete: %s", artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
