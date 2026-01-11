"""Training CLI entrypoint."""

from __future__ import annotations

import argparse

from weather_ml import config as config_module
from weather_ml import dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Weather ML training pipeline (scaffold)."
    )
    parser.add_argument(
        "--config",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--stage",
        choices=["dataset", "train", "eval", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.config:
        raise SystemExit("Config is required. Pass --config <path>.")

    config = config_module.load_config(args.config)

    if args.stage in ("dataset", "all"):
        snapshot = dataset.build_dataset_snapshot(
            config.data.stations,
            config.data.start_date_local,
            config.data.end_date_local,
            config.data.asof_policy_id,
            missing_strategy=config.data.missing_strategy,
            datasets_dir=config.output.datasets_dir,
            db_url=config.db.url,
        )
        print(
            "Dataset snapshot complete: "
            f"id={snapshot.dataset_id}, path={snapshot.data_path}"
        )

    if args.stage in ("train", "eval", "all"):
        print("Training/evaluation stages are not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
