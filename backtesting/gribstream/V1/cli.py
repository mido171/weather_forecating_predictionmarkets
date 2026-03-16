from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Sequence

from . import DB_PATH
from . import db
from .backtest import compute_coverage_summary, compute_metrics_summary, run_backtest
from .blend import compute_daily_predictions
from .config import (
    DEFAULT_FETCH_THREADS,
    DEFAULT_TRUTH_THREADS,
    EVALUATION_END_DATE,
    EVALUATION_START_DATE,
    STATION,
    TRUTH_END_DATE,
    TRUTH_START_DATE,
)
from .derive_daily_tmax import derive_daily_products
from .exports import export_all
from .fetch import fetch_historical_forecasts, fetch_prediction_date_forecasts
from .nws_truth import ingest_truth_range
from .weights import compute_daily_model_weights

LOGGER = logging.getLogger(__name__)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _run_init_db(connection) -> None:
    db.initialize_database(connection)
    db.seed_model_catalog(connection)
    LOGGER.info("Initialized database path=%s", DB_PATH)


def _run_fetch_truth(connection, threads: int) -> None:
    ingest_truth_range(
        connection,
        station_id=STATION.station_id,
        start_date=TRUTH_START_DATE,
        end_date=TRUTH_END_DATE,
        max_workers=threads,
    )


def _run_fetch_forecasts(connection, threads: int) -> None:
    fetch_historical_forecasts(
        connection,
        start_date=TRUTH_START_DATE,
        end_date=TRUTH_END_DATE,
        max_workers=threads,
    )


def _run_derive_daily(connection) -> None:
    derive_daily_products(
        connection,
        start_date=TRUTH_START_DATE,
        end_date=TRUTH_END_DATE,
        include_live_only=False,
        require_truth=True,
    )


def _run_backtest(connection) -> None:
    run_backtest(
        connection,
        evaluation_start=EVALUATION_START_DATE,
        evaluation_end=EVALUATION_END_DATE,
        include_live_only=False,
    )


def _run_export(connection, output_dir: Path) -> None:
    export_all(connection, output_dir=output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KNYC Gribstream V1 non-ML Tmax pipeline")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db")

    truth_parser = subparsers.add_parser("fetch-truth")
    truth_parser.add_argument("--threads", type=int, default=DEFAULT_TRUTH_THREADS)

    forecast_parser = subparsers.add_parser("fetch-forecasts")
    forecast_parser.add_argument("--threads", type=int, default=DEFAULT_FETCH_THREADS)

    subparsers.add_parser("derive-daily")
    subparsers.add_parser("backtest")

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--output-dir", type=Path, default=DB_PATH.parent)

    run_all_parser = subparsers.add_parser("run-all")
    run_all_parser.add_argument("--truth-threads", type=int, default=DEFAULT_TRUTH_THREADS)
    run_all_parser.add_argument("--forecast-threads", type=int, default=DEFAULT_FETCH_THREADS)
    run_all_parser.add_argument("--output-dir", type=Path, default=DB_PATH.parent)

    predict_parser = subparsers.add_parser("predict-date")
    predict_parser.add_argument("--date", required=True, type=_parse_date)
    predict_parser.add_argument("--threads", type=int, default=DEFAULT_FETCH_THREADS)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    connection = db.connect(args.db_path)
    try:
        if args.command == "init-db":
            _run_init_db(connection)
            return 0
        if args.command == "fetch-truth":
            _run_fetch_truth(connection, args.threads)
            return 0
        if args.command == "fetch-forecasts":
            _run_fetch_forecasts(connection, args.threads)
            return 0
        if args.command == "derive-daily":
            _run_derive_daily(connection)
            return 0
        if args.command == "backtest":
            _run_backtest(connection)
            return 0
        if args.command == "export":
            _run_export(connection, args.output_dir)
            return 0
        if args.command == "run-all":
            _run_init_db(connection)
            _run_fetch_truth(connection, args.truth_threads)
            _run_fetch_forecasts(connection, args.forecast_threads)
            _run_derive_daily(connection)
            _run_backtest(connection)
            _run_export(connection, args.output_dir)
            return 0
        if args.command == "predict-date":
            _run_init_db(connection)
            fetch_prediction_date_forecasts(
                connection,
                args.date,
                include_live_only=True,
                max_workers=args.threads,
            )
            derive_daily_products(
                connection,
                start_date=args.date,
                end_date=args.date,
                include_live_only=True,
                require_truth=False,
            )
            compute_daily_model_weights(
                connection,
                start_date=args.date,
                end_date=args.date,
                include_live_only=True,
            )
            prediction_result = compute_daily_predictions(
                connection,
                start_date=args.date,
                end_date=args.date,
                require_truth=False,
            )
            preview = prediction_result.previews_by_date.get(args.date)
            if preview is None:
                LOGGER.error("No prediction preview generated for date=%s", args.date)
                return 1
            LOGGER.info(
                "Prediction preview date=%s family_capped_blend_f=%s equal_weight_blend_f=%s inverse_rmse_blend_f=%s actual_tmax_f=%s",
                args.date,
                preview["family_capped_blend_f"],
                preview["equal_weight_blend_f"],
                preview["inverse_rmse_blend_f"],
                preview["actual_tmax_f"],
            )
            if args.date <= EVALUATION_END_DATE:
                compute_metrics_summary(
                    connection,
                    evaluation_start=EVALUATION_START_DATE,
                    evaluation_end=EVALUATION_END_DATE,
                )
                compute_coverage_summary(
                    connection,
                    evaluation_start=EVALUATION_START_DATE,
                    evaluation_end=EVALUATION_END_DATE,
                )
            return 0
        parser.error(f"Unknown command: {args.command}")
        return 2
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
