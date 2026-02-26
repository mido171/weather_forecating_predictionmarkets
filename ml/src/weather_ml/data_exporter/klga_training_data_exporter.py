from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
import json
import logging

import pandas as pd
from sqlalchemy import text

from weather_ml.klga_daily_tmax_dist.config import (
    ALL_STATION_IDS,
    NEIGHBOR_STATION_IDS,
    OBS_ALLOWED_COLUMNS,
    TARGET_STATION_ID,
    find_repo_root,
)
from weather_ml.klga_daily_tmax_dist.db import create_engine_from_url


@dataclass(frozen=True)
class ExportConfig:
    start_date_local: date
    end_date_local: date
    output_dir: Path
    chunk_size: int = 250_000
    mysql_url: str | None = None


def default_output_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return find_repo_root() / "exports" / f"klga_same_day_tmax_dist_{ts}"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _obs_window_utc(start_date_local: date, end_date_local: date) -> tuple[datetime, datetime]:
    # Wider-than-necessary UTC window to safely include local-day boundaries across DST.
    start_utc = datetime.combine(start_date_local - timedelta(days=1), time(0, 0, 0))
    end_utc_exclusive = datetime.combine(end_date_local + timedelta(days=2), time(0, 0, 0))
    return start_utc, end_utc_exclusive


def _write_station_universe_csv(output_dir: Path) -> Path:
    rows: list[dict[str, Any]] = [
        {"request_location_id": TARGET_STATION_ID, "role": "target"},
    ]
    rows.extend({"request_location_id": s, "role": "neighbor"} for s in NEIGHBOR_STATION_IDS)
    df = pd.DataFrame(rows)
    path = output_dir / "station_universe.csv"
    df.to_csv(path, index=False)
    return path


def _export_daily_truth_csv(
    *,
    output_dir: Path,
    start_date_local: date,
    end_date_local: date,
    mysql_url: str | None,
    logger: logging.Logger,
) -> tuple[Path, int]:
    engine = create_engine_from_url(mysql_url)
    sql = text(
        """
        SELECT
            request_location_id,
            obs_id,
            station_zoneid,
            target_date_local,
            max_temp_f
        FROM wunderground_ml.wunderground_station_daily_max_temperature
        WHERE request_location_id = :station_id
          AND target_date_local BETWEEN :start_date AND :end_date
        ORDER BY target_date_local
        """
    )
    df = pd.read_sql(
        sql,
        engine,
        params={
            "station_id": TARGET_STATION_ID,
            "start_date": start_date_local,
            "end_date": end_date_local,
        },
    )
    path = output_dir / "daily_max_truth_klga.csv"
    df.to_csv(path, index=False)
    logger.info("EXPORT_DAILY_TRUTH_DONE rows=%d path=%s", len(df), path)
    return path, int(len(df))


def _export_observations_csv(
    *,
    output_dir: Path,
    start_date_local: date,
    end_date_local: date,
    mysql_url: str | None,
    chunk_size: int,
    logger: logging.Logger,
) -> tuple[Path, int]:
    engine = create_engine_from_url(mysql_url)
    station_ids = tuple(ALL_STATION_IDS)
    colset = tuple(OBS_ALLOWED_COLUMNS)
    cols_sql = ", ".join(colset)
    placeholders = ", ".join(f":loc_{i}" for i in range(len(station_ids)))
    start_utc, end_utc_exclusive = _obs_window_utc(start_date_local, end_date_local)

    sql = text(
        f"""
        SELECT {cols_sql}
        FROM wunderground_ml.wunderground_station_observation_30m
        WHERE request_location_id IN ({placeholders})
          AND valid_time_utc >= :start_utc
          AND valid_time_utc < :end_utc_exclusive
        ORDER BY request_location_id, valid_time_utc
        """
    )
    params: dict[str, Any] = {f"loc_{i}": sid for i, sid in enumerate(station_ids)}
    params.update(
        {
            "start_utc": start_utc,
            "end_utc_exclusive": end_utc_exclusive,
        }
    )

    out_path = output_dir / "observations_30m_required_columns.csv"
    if out_path.exists():
        out_path.unlink()

    total_rows = 0
    first = True
    for chunk_i, chunk in enumerate(
        pd.read_sql(sql, engine, params=params, chunksize=max(int(chunk_size), 1)),
        start=1,
    ):
        chunk.to_csv(
            out_path,
            mode="w" if first else "a",
            index=False,
            header=first,
        )
        first = False
        total_rows += int(len(chunk))
        logger.info(
            "EXPORT_OBS_CHUNK chunk=%d rows=%d total_rows=%d",
            chunk_i,
            len(chunk),
            total_rows,
        )

    logger.info("EXPORT_OBS_DONE rows=%d path=%s", total_rows, out_path)
    return out_path, total_rows


def export_klga_training_eval_csvs(
    *,
    cfg: ExportConfig,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    active_logger = logger or logging.getLogger(__name__)
    out_dir = _ensure_dir(cfg.output_dir)
    active_logger.info(
        "EXPORT_START output_dir=%s start_date=%s end_date=%s chunk_size=%d",
        out_dir,
        cfg.start_date_local,
        cfg.end_date_local,
        cfg.chunk_size,
    )

    station_csv = _write_station_universe_csv(out_dir)
    daily_csv, daily_rows = _export_daily_truth_csv(
        output_dir=out_dir,
        start_date_local=cfg.start_date_local,
        end_date_local=cfg.end_date_local,
        mysql_url=cfg.mysql_url,
        logger=active_logger,
    )
    obs_csv, obs_rows = _export_observations_csv(
        output_dir=out_dir,
        start_date_local=cfg.start_date_local,
        end_date_local=cfg.end_date_local,
        mysql_url=cfg.mysql_url,
        chunk_size=cfg.chunk_size,
        logger=active_logger,
    )

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "period_local": {
            "start_date_local": str(cfg.start_date_local),
            "end_date_local": str(cfg.end_date_local),
        },
        "station_universe": list(ALL_STATION_IDS),
        "files": {
            "station_universe_csv": str(station_csv),
            "daily_max_truth_klga_csv": str(daily_csv),
            "observations_30m_required_columns_csv": str(obs_csv),
        },
        "row_counts": {
            "daily_max_truth_klga": int(daily_rows),
            "observations_30m_required_columns": int(obs_rows),
        },
        "notes": [
            "These are the exact raw input tables/columns required by the KLGA same-day Tmax peak/delta pipeline.",
            "daily_max_truth_klga.csv is the label/prior source.",
            "observations_30m_required_columns.csv contains all stations and only leakage-safe instantaneous columns used by the model pipeline.",
        ],
    }
    manifest_path = out_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    active_logger.info("EXPORT_DONE manifest=%s", manifest_path)
    return manifest

