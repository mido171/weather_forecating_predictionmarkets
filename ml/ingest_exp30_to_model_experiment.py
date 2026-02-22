"""CLI to ingest exp30 sweep results into model_experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from weather_ml.exp30 import db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest exp30 sweep into model_experiment.")
    parser.add_argument("--summary", required=True, help="Path to exp30_sweep_summary.json")
    parser.add_argument("--db-url", default=None, help="Optional SQLAlchemy DB URL override")
    args = parser.parse_args(argv)

    payload = db.load_sweep_summary(Path(args.summary))
    sweep_id = payload.get("sweep_id", "")
    url = args.db_url or db.default_mysql_url()
    engine = db.create_db_engine(url)
    db.upsert_model_experiments(engine, payload, sweep_id=sweep_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
