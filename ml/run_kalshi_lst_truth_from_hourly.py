"""CLI: build LST-aligned daily truth from hourly CSV and upsert into MySQL."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from weather_ml.kalshi_truth_lst import (
    LstTruthConfig,
    compute_lst_daily_from_hourly,
    ensure_lst_truth_table,
    upsert_lst_truth,
)
from weather_ml.mos_db import create_engine_from_config, load_db_config, load_db_config_from_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest LST daily truth from hourly CSV.")
    parser.add_argument("--csv", required=True, help="Path to hourly CSV file.")
    parser.add_argument("--timestamp-col", default="valid_utc", help="UTC timestamp column.")
    parser.add_argument("--temp-col", default="tmpf", help="Temperature column (F).")
    parser.add_argument("--station-id", default="KMIA")
    parser.add_argument("--station-zoneid", default="America/New_York")
    parser.add_argument("--offset-hours", type=int, default=-5, help="LST UTC offset hours (default -5).")
    parser.add_argument("--db-config", help="Path to DB config JSON.")
    parser.add_argument("--db-env", action="store_true", help="Load DB config from env vars.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    hourly = pd.read_csv(csv_path)
    cfg = LstTruthConfig(
        station_id=args.station_id,
        station_zoneid=args.station_zoneid,
        climate_day_utc_offset_hours=args.offset_hours,
        source_station=args.station_id,
    )

    if args.db_env:
        db_cfg = load_db_config_from_env()
    elif args.db_config:
        db_cfg = load_db_config(args.db_config)
    else:
        raise ValueError("Provide --db-config or --db-env for database credentials.")

    engine = create_engine_from_config(db_cfg)
    ensure_lst_truth_table(engine)

    daily = compute_lst_daily_from_hourly(
        hourly,
        timestamp_col=args.timestamp_col,
        temp_col=args.temp_col,
        cfg=cfg,
    )
    if daily.empty:
        logging.warning("No daily rows produced; check input CSV.")
        return 0
    upsert_lst_truth(engine, daily.to_dict(orient="records"))
    logging.info("Upserted %d LST daily truth rows.", len(daily))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
