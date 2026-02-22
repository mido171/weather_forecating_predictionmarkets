from __future__ import annotations

import argparse
from pathlib import Path

from weather_ml import config as config_module
from weather_ml import experiment_db
from weather_ml import time_feature_sweep


EXPERIMENT_IDS = [f"EX{i:02d}" for i in range(1, 51)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EX01-EX50 time feature sweep and persist results to DB."
    )
    parser.add_argument("--config", required=True, help="Path to base YAML config.")
    parser.add_argument("--sweep-id", help="Optional sweep id override.")
    parser.add_argument("--sweep-root", help="Optional sweep output root.")
    parser.add_argument(
        "--truth-lag",
        type=int,
        default=2,
        help="Truth lag in days (default: 2).",
    )
    parser.add_argument(
        "--allow-tuning",
        action="store_true",
        help="Enable hyperparameter tuning.",
    )
    parser.add_argument(
        "--experiment-ids",
        nargs="*",
        help="Optional list of experiment ids to run.",
    )
    parser.add_argument(
        "--db-url",
        help="SQLAlchemy URL for results DB (default: MYSQL_* env vars).",
    )
    parser.add_argument(
        "--no-persist-predictions",
        action="store_true",
        help="Skip persisting prediction rows.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = time_feature_sweep._resolve_repo_root()
    config = config_module.load_config(args.config)
    config = config_module.resolve_paths(config, repo_root=repo_root)
    sweep_id = args.sweep_id or time_feature_sweep._default_sweep_id()
    sweep_root = (
        Path(args.sweep_root)
        if args.sweep_root
        else Path(config.artifacts.root_dir) / "time_feature_sweep" / sweep_id
    )

    experiment_ids = args.experiment_ids or EXPERIMENT_IDS
    argv = [
        "--config",
        args.config,
        "--sweep-id",
        sweep_id,
        "--sweep-root",
        str(sweep_root),
        "--truth-lag",
        str(args.truth_lag),
        "--experiment-ids",
        *experiment_ids,
    ]
    if args.allow_tuning:
        argv.append("--allow-tuning")
    time_feature_sweep.main(argv)

    db_url = args.db_url or experiment_db.default_mysql_url()
    engine = experiment_db.create_db_engine(db_url)
    experiment_db.persist_sweep(
        engine,
        sweep_root / "time_feature_sweep.json",
        persist_predictions=not args.no_persist_predictions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
