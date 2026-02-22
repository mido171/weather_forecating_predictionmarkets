"""DB ingestion for TFS2 results into model_experiment."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .experiments import EXPERIMENT_DESCRIPTIONS


def default_mysql_url() -> str:
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    db = os.getenv("MYSQL_DB", "weather_predictionmarkets")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"


def create_db_engine(url: str | None = None) -> Engine:
    return create_engine(url or default_mysql_url(), pool_pre_ping=True, pool_recycle=3600)


def upsert_model_experiments(engine: Engine, payload: dict, *, sweep_id: str) -> None:
    experiments = payload.get("experiments", [])
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for exp in experiments:
        experiment_id = exp.get("experiment_id")
        if not experiment_id or "error" in exp:
            continue
        description = EXPERIMENT_DESCRIPTIONS.get(experiment_id, exp.get("description", ""))
        description = description.strip() if description else ""
        word_count = len([w for w in description.split() if w.strip()])
        if word_count < 50 or word_count > 80:
            raise ValueError(
                f"Description for {experiment_id} out of bounds: {word_count} words"
            )
        experiment_key = f"tfs2::{sweep_id}::{experiment_id}"
        run_dir = exp.get("run_dir")
        source_path = str(Path(run_dir) / "experiment_meta.json") if run_dir else experiment_key
        metrics = exp.get("metrics", {})
        train = metrics.get("train", {})
        validation = metrics.get("validation", {})
        test = metrics.get("test", {})

        metadata = {
            "schema_version": 1,
            "primary_path": source_path,
            "raw": {"primary": exp},
            "extras": {
                "sweep_id": sweep_id,
                "run_dir": run_dir,
            },
        }
        metadata_json = json.dumps(metadata, sort_keys=True, ensure_ascii=True)
        raw_hash = hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()
        rows.append(
            {
                "experiment_key": experiment_key,
                "experiment_name": exp.get("name"),
                "station_id": payload.get("station"),
                "model_name": exp.get("model_family"),
                "model_family": exp.get("model_family"),
                "source_path": source_path,
                "metadata_json": metadata_json,
                "metrics_train_json": json.dumps(train, ensure_ascii=True) if train else None,
                "metrics_validation_json": json.dumps(validation, ensure_ascii=True) if validation else None,
                "metrics_test_json": json.dumps(test, ensure_ascii=True) if test else None,
                "train_mae": train.get("mae"),
                "train_rmse": train.get("rmse"),
                "train_bias": train.get("bias"),
                "train_median_ae": train.get("medianAE"),
                "train_max_ae": train.get("maxAE"),
                "train_corr": train.get("corr"),
                "train_n": train.get("n"),
                "validation_mae": validation.get("mae"),
                "validation_rmse": validation.get("rmse"),
                "validation_bias": validation.get("bias"),
                "validation_median_ae": validation.get("medianAE"),
                "validation_max_ae": validation.get("maxAE"),
                "validation_corr": validation.get("corr"),
                "validation_n": validation.get("n"),
                "test_mae": test.get("mae"),
                "test_rmse": test.get("rmse"),
                "test_bias": test.get("bias"),
                "test_median_ae": test.get("medianAE"),
                "test_max_ae": test.get("maxAE"),
                "test_corr": test.get("corr"),
                "test_n": test.get("n"),
                "description_text": description or "(missing description)",
                "raw_payload_hash": raw_hash,
                "retrieved_at_utc": now,
                "created_at_utc": now,
                "updated_at_utc": now,
            }
        )

    if not rows:
        return

    cols = list(rows[0].keys())
    col_clause = ", ".join(cols)
    param_clause = ", ".join(f":{col}" for col in cols)
    updates = ", ".join(
        f"{col}=VALUES({col})" for col in cols if col not in {"experiment_key"}
    )
    sql = (
        f"INSERT INTO model_experiment ({col_clause}) VALUES ({param_clause}) "
        f"ON DUPLICATE KEY UPDATE {updates}"
    )
    with engine.begin() as conn:
        conn.execute(text(sql), rows)


def load_sweep_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Sweep summary not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
