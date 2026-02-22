from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


DEFAULT_DB_NAME = "weather_predictionmarkets_experiments"


@dataclass(frozen=True)
class ExperimentResultsDbConfig:
    url: str
    db_name: str


def default_db_name() -> str:
    return os.getenv("EXPERIMENTS_DB_NAME", DEFAULT_DB_NAME)


def default_mysql_url(db_name: str | None = None) -> str:
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    db = db_name or default_db_name()
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"


def default_mysql_server_url() -> str:
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}"


def create_db_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)


def ensure_database(db_name: str) -> None:
    engine = create_db_engine(default_mysql_server_url())
    stmt = (
        "CREATE DATABASE IF NOT EXISTS "
        f"`{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    with engine.begin() as conn:
        conn.execute(text(stmt))


def ensure_tables(engine: Engine) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS experiment_sweeps (
            sweep_id VARCHAR(64) PRIMARY KEY,
            station_id VARCHAR(16),
            sweep_kind VARCHAR(64) NOT NULL,
            created_utc DATETIME,
            sweep_root TEXT,
            sweep_json_path TEXT,
            baseline_experiment_id VARCHAR(64),
            dataset_ref_json LONGTEXT,
            split_ref_json LONGTEXT,
            model_ref_json LONGTEXT,
            leaderboard_test_mae_json LONGTEXT,
            leaderboard_val_mae_json LONGTEXT,
            payload_json LONGTEXT,
            payload_hash VARCHAR(64),
            ingested_utc DATETIME NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        """
        CREATE TABLE IF NOT EXISTS experiment_variants (
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
            payload_hash VARCHAR(64),
            ingested_utc DATETIME NOT NULL,
            PRIMARY KEY (sweep_id, experiment_id),
            CONSTRAINT fk_variants_sweep
                FOREIGN KEY (sweep_id) REFERENCES experiment_sweeps(sweep_id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        """
        CREATE TABLE IF NOT EXISTS experiment_metrics (
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
            PRIMARY KEY (sweep_id, experiment_id, split),
            CONSTRAINT fk_metrics_variant
                FOREIGN KEY (sweep_id, experiment_id)
                REFERENCES experiment_variants(sweep_id, experiment_id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        """
        CREATE TABLE IF NOT EXISTS experiment_calibrations (
            calibration_id BIGINT AUTO_INCREMENT PRIMARY KEY,
            sweep_id VARCHAR(64) NOT NULL,
            experiment_id VARCHAR(64) NOT NULL,
            station_id VARCHAR(16) NOT NULL,
            asof_utc DATETIME NOT NULL,
            target_date_local DATE NOT NULL,
            calibration_source VARCHAR(64) NOT NULL,
            emos_window_days INT,
            mu_hat_f DOUBLE,
            sigma_hat_f DOUBLE,
            sigma_emos_f DOUBLE,
            emos_c DOUBLE,
            emos_d DOUBLE,
            rolling_bias_45 DOUBLE,
            rolling_rmse_45 DOUBLE,
            normal_dist_json LONGTEXT,
            threshold_probs_json LONGTEXT,
            payload_json LONGTEXT,
            payload_hash VARCHAR(64),
            ingested_utc DATETIME NOT NULL,
            UNIQUE KEY uq_calibration (
                sweep_id,
                experiment_id,
                station_id,
                asof_utc,
                target_date_local,
                calibration_source
            ),
            CONSTRAINT fk_calibration_variant
                FOREIGN KEY (sweep_id, experiment_id)
                REFERENCES experiment_variants(sweep_id, experiment_id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
    ]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def persist_sweep(
    engine: Engine,
    sweep_path: Path,
    *,
    sweep_kind: str = "time_feature_sweep",
    station_id: str | None = None,
) -> None:
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep JSON not found: {sweep_path}")
    payload_text = sweep_path.read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    sweep_id = payload.get("sweep_id")
    created_utc = _parse_timestamp(payload.get("created_utc"))
    dataset_ref = payload.get("dataset_ref", {})
    sweep_root = str(sweep_path.parent)
    station_id = station_id or _infer_station_id(
        dataset_ref.get("csv_path"),
        sweep_root,
    )
    ingested_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    ensure_tables(engine)

    sweep_row = {
        "sweep_id": sweep_id,
        "station_id": station_id,
        "sweep_kind": sweep_kind,
        "created_utc": created_utc,
        "sweep_root": sweep_root,
        "sweep_json_path": str(sweep_path),
        "baseline_experiment_id": payload.get("baseline_experiment_id"),
        "dataset_ref_json": _json_dumps(dataset_ref),
        "split_ref_json": _json_dumps(payload.get("split_ref")),
        "model_ref_json": _json_dumps(payload.get("model_ref")),
        "leaderboard_test_mae_json": _json_dumps(payload.get("leaderboard_test_mae")),
        "leaderboard_val_mae_json": _json_dumps(payload.get("leaderboard_val_mae")),
        "payload_json": _json_dumps(payload),
        "payload_hash": _hash_payload(payload_text),
        "ingested_utc": ingested_utc,
    }
    _upsert_rows(
        engine,
        "experiment_sweeps",
        [sweep_row],
        key_columns={"sweep_id"},
    )

    experiments = payload.get("experiments", [])
    exp_rows = []
    metric_rows = []
    for exp in experiments:
        exp_payload_hash = _hash_payload(_json_dumps(exp) or "")
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
                "payload_hash": exp_payload_hash,
                "ingested_utc": ingested_utc,
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

    if exp_rows:
        _upsert_rows(
            engine,
            "experiment_variants",
            exp_rows,
            key_columns={"sweep_id", "experiment_id"},
        )
    if metric_rows:
        _upsert_rows(
            engine,
            "experiment_metrics",
            metric_rows,
            key_columns={"sweep_id", "experiment_id", "split"},
        )


def persist_calibration(
    engine: Engine,
    payload: dict,
    *,
    sweep_id: str,
    experiment_id: str,
    calibration_source: str = "live_emos_w45",
) -> None:
    ensure_tables(engine)
    payload_text = _json_dumps(payload) or ""
    ingested_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    asof_utc = _parse_timestamp(payload.get("asof_utc"))
    target_date = _parse_date(payload.get("target_date_local"))
    row = {
        "sweep_id": sweep_id,
        "experiment_id": experiment_id,
        "station_id": payload.get("station_id"),
        "asof_utc": asof_utc,
        "target_date_local": target_date,
        "calibration_source": calibration_source,
        "emos_window_days": payload.get("emos_window_days"),
        "mu_hat_f": payload.get("mu_hat_f"),
        "sigma_hat_f": payload.get("sigma_hat_f"),
        "sigma_emos_f": payload.get("sigma_emos_f"),
        "emos_c": payload.get("emos_c"),
        "emos_d": payload.get("emos_d"),
        "rolling_bias_45": payload.get("rolling_bias_45"),
        "rolling_rmse_45": payload.get("rolling_rmse_45"),
        "normal_dist_json": _json_dumps(payload.get("normal_dist")),
        "threshold_probs_json": _json_dumps(payload.get("threshold_probs")),
        "payload_json": payload_text,
        "payload_hash": _hash_payload(payload_text),
        "ingested_utc": ingested_utc,
    }
    _upsert_rows(
        engine,
        "experiment_calibrations",
        [row],
        key_columns={
            "sweep_id",
            "experiment_id",
            "station_id",
            "asof_utc",
            "target_date_local",
            "calibration_source",
        },
    )


def _upsert_rows(engine: Engine, table: str, rows: Iterable[dict], *, key_columns: set[str]) -> None:
    rows = list(rows)
    if not rows:
        return
    columns = list(rows[0].keys())
    col_clause = ", ".join(columns)
    param_clause = ", ".join(f":{col}" for col in columns)
    if engine.dialect.name == "mysql":
        updates = ", ".join(f"{col}=VALUES({col})" for col in columns if col not in key_columns)
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


def _infer_station_id(csv_path: str | None, sweep_root: str | None) -> str | None:
    for candidate in (csv_path, sweep_root):
        if not candidate:
            continue
        upper = candidate.upper()
        match = re.search(r"(K[A-Z]{3})", upper)
        if match:
            return match.group(1)
        match = re.search(r"(KNYC|KPHL|KMIA)", upper)
        if match:
            return match.group(1)
    return None


def _json_dumps(payload) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _hash_payload(payload: str) -> str | None:
    if not payload:
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_timestamp(value: str | None):
    if not value:
        return None
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc).replace(tzinfo=None)
    except ValueError:
        return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None
