"""End-to-end KMIA Kalshi Tmax pipeline (dataset build + train + eval)."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from weather_ml.kalshi_tmax_dataset import build_dataset as build_kalshi_dataset
from weather_ml.kalshi_tmax_dataset import write_outputs as write_kalshi_outputs
from weather_ml.kalshi_tmax_train import load_train_config, train_and_evaluate
from weather_ml.mos_config import load_config as load_mos_config
from weather_ml.mos_db import create_engine_from_config, load_db_config, load_db_config_from_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KMIA Kalshi Tmax pipeline.")
    parser.add_argument("--dataset-config", required=True, help="Path to MOS dataset config JSON.")
    parser.add_argument("--train-config", required=True, help="Path to Kalshi train config YAML.")
    parser.add_argument("--db-config", help="Path to DB config JSON.")
    parser.add_argument("--db-env", action="store_true", help="Load DB config from env vars.")
    parser.add_argument("--dataset-csv", help="Use an existing dataset CSV instead of building from DB.")
    parser.add_argument("--output-root", default="artifacts/kalshi_tmax", help="Root output directory.")
    parser.add_argument("--run-id", help="Override run id.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = load_mos_config(args.dataset_config)
    train_cfg = load_train_config(args.train_config)

    (run_dir / "dataset_config.json").write_text(
        Path(args.dataset_config).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (run_dir / "train_config.yaml").write_text(
        Path(args.train_config).read_text(encoding="utf-8"), encoding="utf-8"
    )

    if args.dataset_csv:
        df = pd.read_csv(args.dataset_csv)
        logging.info("Loaded dataset CSV %s rows=%d", args.dataset_csv, len(df))
    else:
        if args.db_env:
            db_cfg = load_db_config_from_env()
        elif args.db_config:
            db_cfg = load_db_config(args.db_config)
        else:
            raise ValueError("Provide --db-config or --db-env for database credentials.")
        engine = create_engine_from_config(db_cfg)
        df, metadata = build_kalshi_dataset(dataset_cfg, engine)
        dataset_dir = write_kalshi_outputs(df, metadata, dataset_cfg)
        logging.info("Dataset written to %s", dataset_dir)

    metrics_payload = train_and_evaluate(df, train_cfg, output_dir=run_dir)
    logging.info("Training complete. Run dir: %s", run_dir)
    logging.info(
        "Test MAE: %.4f",
        metrics_payload["metrics"]["model"]["point"]["test"]["mae"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
