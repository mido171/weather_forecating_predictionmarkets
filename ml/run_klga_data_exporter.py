from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from weather_ml.data_exporter.klga_training_data_exporter import (
    ExportConfig,
    default_output_dir,
    export_klga_training_eval_csvs,
)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export KLGA same-day Tmax training/evaluation raw inputs to CSV files."
    )
    p.add_argument(
        "--mysql-url",
        default=None,
        help="Optional SQLAlchemy MySQL URL. If omitted, MYSQL_* env vars are used.",
    )
    p.add_argument(
        "--start-date",
        default="1973-01-01",
        help="Local start date (YYYY-MM-DD). Default: 1973-01-01",
    )
    p.add_argument(
        "--end-date",
        default="2025-12-31",
        help="Local end date (YYYY-MM-DD). Default: 2025-12-31",
    )
    p.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Output directory under repo exports/.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=250000,
        help="Observation export SQL chunk size.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = ExportConfig(
        start_date_local=_parse_date(str(args.start_date)),
        end_date_local=_parse_date(str(args.end_date)),
        output_dir=Path(args.output_dir).resolve(),
        chunk_size=max(int(args.chunk_size), 1),
        mysql_url=args.mysql_url,
    )
    manifest = export_klga_training_eval_csvs(cfg=cfg)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

