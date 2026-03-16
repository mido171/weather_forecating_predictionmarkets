from __future__ import annotations

import csv
import logging
from pathlib import Path

from .config import SQLITE_DIR

LOGGER = logging.getLogger(__name__)

EXPORT_ORDER_BY = {
    "model_catalog": "ORDER BY model_code",
    "nws_daily_settlements": "ORDER BY station_id, settlement_date_local",
    "gribstream_requests": "ORDER BY station_id, settlement_date_local, model_code",
    "gribstream_raw_forecasts": (
        "ORDER BY station_id, settlement_date_local, model_code, forecasted_time_utc, "
        "forecasted_at_utc, variable_name, variable_level"
    ),
    "daily_model_tmax": "ORDER BY station_id, settlement_date_local, model_code",
    "model_daily_errors": "ORDER BY station_id, settlement_date_local, model_code",
    "daily_model_weights": "ORDER BY station_id, settlement_date_local, model_code",
    "daily_prediction_components": "ORDER BY station_id, settlement_date_local, model_code",
    "daily_predictions": "ORDER BY station_id, settlement_date_local",
    "metrics_summary": "ORDER BY metric_scope, metric_name, evaluation_start, evaluation_end",
    "coverage_summary": "ORDER BY model_code",
}

EXPORT_FILES = {
    "nws_daily_settlements": "nws_daily_settlements.csv",
    "gribstream_raw_forecasts": "gribstream_raw_forecasts.csv",
    "daily_model_tmax": "daily_model_tmax.csv",
    "model_daily_errors": "model_daily_errors.csv",
    "daily_model_weights": "daily_model_weights.csv",
    "daily_prediction_components": "daily_prediction_components.csv",
    "daily_predictions": "daily_predictions.csv",
    "metrics_summary": "metrics_summary.csv",
    "coverage_summary": "coverage_summary.csv",
}


def export_table(
    connection,
    table_name: str,
    output_path: Path,
) -> Path:
    order_by_clause = EXPORT_ORDER_BY.get(table_name, "")
    cursor = connection.execute(f"SELECT * FROM {table_name} {order_by_clause}")
    columns = [description[0] for description in cursor.description or ()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in cursor:
            writer.writerow([row[column] for column in columns])
    LOGGER.info("Exported table=%s path=%s", table_name, output_path)
    return output_path


def export_all(
    connection,
    *,
    output_dir: Path = SQLITE_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_paths: list[Path] = []
    for table_name, file_name in EXPORT_FILES.items():
        exported_paths.append(export_table(connection, table_name, output_dir / file_name))
    return exported_paths
