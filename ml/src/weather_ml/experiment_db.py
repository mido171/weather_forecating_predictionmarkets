from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class ExperimentDbConfig:
    url: str


def default_mysql_url() -> str:
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    db = os.getenv("MYSQL_DB", "weather_predictionmarkets")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"


def create_db_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)


def ensure_tables(engine: Engine) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS tfs_sweeps (
            sweep_id VARCHAR(64) PRIMARY KEY,
            created_utc DATETIME,
            csv_path TEXT,
            csv_hash VARCHAR(64),
            schema_version INT,
            baseline_experiment_id VARCHAR(64),
            sweep_root TEXT,
            sweep_json_path TEXT,
            leaderboard_test_mae_json LONGTEXT,
            leaderboard_val_mae_json LONGTEXT,
            split_ref_json LONGTEXT,
            model_ref_json LONGTEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tfs_experiments (
            sweep_id VARCHAR(64) NOT NULL,
            experiment_id VARCHAR(64) NOT NULL,
            description TEXT,
            run_dir TEXT,
            dataset_id VARCHAR(64),
            num_features INT,
            uses_spread_feature TINYINT,
            calendar_enabled TINYINT,
            raw_model_cols_json LONGTEXT,
            final_feature_columns_json LONGTEXT,
            derived_features_json LONGTEXT,
            artifact_hashes_json LONGTEXT,
            deltas_json LONGTEXT,
            worst_days_json LONGTEXT,
            metrics_json LONGTEXT,
            PRIMARY KEY (sweep_id, experiment_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tfs_metrics (
            sweep_id VARCHAR(64) NOT NULL,
            experiment_id VARCHAR(64) NOT NULL,
            split VARCHAR(16) NOT NULL,
            mae DOUBLE,
            rmse DOUBLE,
            bias DOUBLE,
            medianAE DOUBLE,
            maxAE DOUBLE,
            corr DOUBLE,
            n INT,
            PRIMARY KEY (sweep_id, experiment_id, split)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tfs_predictions (
            sweep_id VARCHAR(64) NOT NULL,
            experiment_id VARCHAR(64) NOT NULL,
            station_id VARCHAR(16) NOT NULL,
            target_date_local DATE NOT NULL,
            asof_utc DATETIME NOT NULL,
            mu_hat_f DOUBLE,
            sigma_hat_f DOUBLE,
            p_bins_json LONGTEXT,
            p_temp_json LONGTEXT,
            support_min_f INT,
            support_max_f INT,
            PRIMARY KEY (sweep_id, experiment_id, station_id, target_date_local, asof_utc)
        )
        """,
    ]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def persist_sweep(
    engine: Engine,
    sweep_path: Path,
    *,
    persist_predictions: bool = True,
    chunk_size: int = 5000,
) -> None:
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep JSON not found: {sweep_path}")
    payload = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep_id = payload.get("sweep_id")
    created_utc = _parse_timestamp(payload.get("created_utc"))
    dataset_ref = payload.get("dataset_ref", {})
    sweep_root = str(sweep_path.parent)

    ensure_tables(engine)

    sweep_row = {
        "sweep_id": sweep_id,
        "created_utc": created_utc,
        "csv_path": dataset_ref.get("csv_path"),
        "csv_hash": dataset_ref.get("csv_hash"),
        "schema_version": dataset_ref.get("schema_version"),
        "baseline_experiment_id": payload.get("baseline_experiment_id"),
        "sweep_root": sweep_root,
        "sweep_json_path": str(sweep_path),
        "leaderboard_test_mae_json": _json_dumps(payload.get("leaderboard_test_mae")),
        "leaderboard_val_mae_json": _json_dumps(payload.get("leaderboard_val_mae")),
        "split_ref_json": _json_dumps(payload.get("split_ref")),
        "model_ref_json": _json_dumps(payload.get("model_ref")),
    }
    _upsert_rows(engine, "tfs_sweeps", [sweep_row])

    experiments = payload.get("experiments", [])
    exp_rows = []
    metric_rows = []
    prediction_jobs = []
    for exp in experiments:
        run_dir = exp.get("run_dir")
        dataset_id = _read_dataset_id(run_dir)
        exp_rows.append(
            {
                "sweep_id": sweep_id,
                "experiment_id": exp.get("experiment_id"),
                "description": exp.get("description"),
                "run_dir": run_dir,
                "dataset_id": dataset_id,
                "num_features": exp.get("num_features"),
                "uses_spread_feature": int(bool(exp.get("uses_spread_feature"))),
                "calendar_enabled": int(bool(exp.get("calendar_enabled"))),
                "raw_model_cols_json": _json_dumps(exp.get("raw_model_cols_used")),
                "final_feature_columns_json": _json_dumps(exp.get("final_feature_columns")),
                "derived_features_json": _json_dumps(exp.get("derived_features")),
                "artifact_hashes_json": _json_dumps(exp.get("artifact_hashes")),
                "deltas_json": _json_dumps(exp.get("deltas_vs_baseline")),
                "worst_days_json": _json_dumps(exp.get("worst_test_days")),
                "metrics_json": _json_dumps(exp.get("metrics")),
            }
        )
        metrics = exp.get("metrics", {})
        for split, data in metrics.items():
            if not data:
                continue
            metric_rows.append(
                {
                    "sweep_id": sweep_id,
                    "experiment_id": exp.get("experiment_id"),
                    "split": split,
                    "mae": data.get("mae"),
                    "rmse": data.get("rmse"),
                    "bias": data.get("bias"),
                    "medianAE": data.get("medianAE"),
                    "maxAE": data.get("maxAE"),
                    "corr": data.get("corr"),
                    "n": data.get("n"),
                }
            )
        if persist_predictions and run_dir:
            prediction_jobs.append((exp.get("experiment_id"), Path(run_dir) / "predictions_test.parquet"))

    if exp_rows:
        _upsert_rows(engine, "tfs_experiments", exp_rows)
    if metric_rows:
        _upsert_rows(engine, "tfs_metrics", metric_rows)
    if persist_predictions:
        for experiment_id, prediction_path in prediction_jobs:
            _persist_predictions(
                engine,
                sweep_id,
                experiment_id,
                prediction_path,
                chunk_size=chunk_size,
            )


def _persist_predictions(
    engine: Engine,
    sweep_id: str,
    experiment_id: str,
    path: Path,
    *,
    chunk_size: int,
) -> None:
    if not path.exists():
        return
    df = pd.read_parquet(path)
    df = df.copy()
    df["sweep_id"] = sweep_id
    df["experiment_id"] = experiment_id
    records = df.to_dict(orient="records")
    for chunk in _chunked(records, chunk_size):
        _upsert_rows(engine, "tfs_predictions", chunk)


def _upsert_rows(engine: Engine, table: str, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    columns = list(rows[0].keys())
    col_clause = ", ".join(columns)
    param_clause = ", ".join(f":{col}" for col in columns)
    if engine.dialect.name == "mysql":
        updates = ", ".join(f"{col}=VALUES({col})" for col in columns if col not in {"sweep_id"})
        sql = f"INSERT INTO {table} ({col_clause}) VALUES ({param_clause}) ON DUPLICATE KEY UPDATE {updates}"
    else:
        sql = f"INSERT OR REPLACE INTO {table} ({col_clause}) VALUES ({param_clause})"
    with engine.begin() as conn:
        conn.execute(text(sql), rows)


def _read_dataset_id(run_dir: str | None) -> str | None:
    if not run_dir:
        return None
    dataset_path = Path(run_dir) / "dataset_id.txt"
    if dataset_path.exists():
        return dataset_path.read_text(encoding="utf-8").strip()
    return None


def _json_dumps(payload) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _parse_timestamp(value: str | None):
    if not value:
        return None
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc).replace(tzinfo=None)
    except ValueError:
        return None


def _chunked(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for idx in range(0, len(rows), size):
        yield rows[idx : idx + size]
