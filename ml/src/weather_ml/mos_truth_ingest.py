"""CLI to ingest IEM daily truth into station_daily_truth."""

from __future__ import annotations

import argparse
import logging
from datetime import date

import pandas as pd

from .mos_db import create_engine_from_config, load_db_config, load_db_config_from_env
from .mos_truth import IemDailyConfig, ingest_iem_daily


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest IEM daily truth into station_daily_truth.")
    parser.add_argument("--station-id", required=True)
    parser.add_argument("--station-zoneid", required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--source-station", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--db-config")
    parser.add_argument("--db-env", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.db_env:
        db_cfg = load_db_config_from_env()
    elif args.db_config:
        db_cfg = load_db_config(args.db_config)
    else:
        raise ValueError("Provide --db-config or --db-env for database credentials.")

    cfg = IemDailyConfig(
        station_id=args.station_id,
        station_zoneid=args.station_zoneid,
        source_network=args.network,
        source_station=args.source_station,
    )

    start_date = pd.to_datetime(args.start_date).date()
    end_date = pd.to_datetime(args.end_date).date()

    engine = create_engine_from_config(db_cfg)
    result = ingest_iem_daily(engine, cfg, start_date, end_date)
    LOGGER.info("Ingested %s rows with query hash %s", result["row_count"], result["query_hash"])


if __name__ == "__main__":
    main()
