from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.backfills.pooled_strategy.iem_registry import ResolvedStationMetadata
from tools.backfills.pooled_strategy.station_universe import StationSeed


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS station_registry(
    station_id TEXT PRIMARY KEY,
    metadata_lookup_station_id TEXT NOT NULL,
    tier TEXT NOT NULL,
    group_name TEXT NOT NULL,
    active_flag INTEGER NOT NULL,
    traded_station_flag INTEGER NOT NULL,
    kalshi_series TEXT,
    iem_station_id TEXT NOT NULL,
    iem_network TEXT NOT NULL,
    station_zoneid TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    elevation_m REAL,
    display_name TEXT NOT NULL,
    archive_begin TEXT,
    archive_end TEXT,
    climate_site TEXT,
    wfo TEXT,
    nws_usw TEXT,
    ncei91 TEXT,
    ghcnh_id TEXT,
    synop_wban TEXT,
    metar_reset_minute INTEGER,
    wu_location_id TEXT NOT NULL,
    seed_json TEXT NOT NULL,
    raw_feature_json TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_runs(
    run_id TEXT PRIMARY KEY,
    stage_name TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    status TEXT NOT NULL,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS ingest_events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    event_time_utc TEXT NOT NULL,
    level TEXT NOT NULL,
    component TEXT NOT NULL,
    message TEXT NOT NULL,
    detail_json TEXT
);

CREATE TABLE IF NOT EXISTS ingest_source_status(
    run_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    status TEXT NOT NULL,
    detail_json TEXT,
    PRIMARY KEY(run_id, source_name)
);

CREATE TABLE IF NOT EXISTS nws_raw_snapshots(
    station_id TEXT NOT NULL,
    station_usw TEXT NOT NULL,
    window_start_date TEXT NOT NULL,
    window_end_date TEXT NOT NULL,
    request_url TEXT,
    response_path TEXT,
    headers_path TEXT,
    retrieved_at_utc TEXT,
    http_status INTEGER,
    body_sha256 TEXT,
    byte_count INTEGER,
    inserted_at_utc TEXT NOT NULL,
    PRIMARY KEY(station_id, window_start_date, window_end_date)
);

CREATE TABLE IF NOT EXISTS nws_truth_canonical(
    station_id TEXT NOT NULL,
    station_usw TEXT NOT NULL,
    target_date_local TEXT NOT NULL,
    tmax_f INTEGER,
    truth_source TEXT,
    source_record_id TEXT,
    retrieved_at_utc TEXT,
    PRIMARY KEY(station_id, target_date_local)
);

CREATE TABLE IF NOT EXISTS nws_truth_enriched(
    station_id TEXT NOT NULL,
    station_usw TEXT NOT NULL,
    target_date_local TEXT NOT NULL,
    tmax_f INTEGER,
    truth_source TEXT,
    source_record_id TEXT,
    retrieved_at_utc TEXT,
    attribute_measurement_flag TEXT,
    attribute_quality_flag TEXT,
    attribute_source_flag TEXT,
    attribute_obs_time_hhmm TEXT,
    attribute_raw TEXT,
    source_station_field TEXT
);

CREATE TABLE IF NOT EXISTS nws_qa_reports(
    station_id TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    rows_count INTEGER,
    duplicate_station_date_rows INTEGER,
    missing_dates_count INTEGER,
    qa_json TEXT,
    qa_md_path TEXT,
    inserted_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nws_run_meta(
    station_id TEXT NOT NULL,
    meta_key TEXT NOT NULL,
    meta_value_json TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY(station_id, meta_key)
);

CREATE TABLE IF NOT EXISTS mos_raw_payloads(
    station_id TEXT NOT NULL,
    model TEXT NOT NULL,
    year INTEGER NOT NULL,
    request_params_json TEXT,
    retrieved_at_utc TEXT,
    response_sha256 TEXT,
    row_count INTEGER,
    runtime_hour_counts_json TEXT,
    yearly_csv_gz TEXT,
    raw_json_gz TEXT,
    meta_file TEXT,
    PRIMARY KEY(station_id, model, year)
);

CREATE TABLE IF NOT EXISTS mos_hourly_values(
    station_id TEXT NOT NULL,
    model TEXT NOT NULL,
    year INTEGER NOT NULL,
    runtime_utc TEXT NOT NULL,
    forecast_time_utc TEXT NOT NULL,
    retrieved_at_utc TEXT,
    response_sha256 TEXT,
    tmp REAL,
    dpt REAL,
    cld REAL,
    sky REAL,
    wdr REAL,
    wsp REAL,
    gst REAL,
    p06 REAL,
    p12 REAL,
    t06 REAL,
    t12 REAL,
    q06 REAL,
    q12 REAL,
    n_x REAL,
    n_n REAL,
    cig REAL,
    vis REAL,
    tmp_raw TEXT,
    dpt_raw TEXT,
    cld_raw TEXT,
    sky_raw TEXT,
    wdr_raw TEXT,
    wsp_raw TEXT,
    gst_raw TEXT,
    p06_raw TEXT,
    p12_raw TEXT,
    t06_raw TEXT,
    t12_raw TEXT,
    q06_raw TEXT,
    q12_raw TEXT,
    n_x_raw TEXT,
    n_n_raw TEXT,
    cig_raw TEXT,
    vis_raw TEXT,
    PRIMARY KEY(station_id, model, runtime_utc, forecast_time_utc)
);

CREATE TABLE IF NOT EXISTS mos_download_manifest(
    station_id TEXT NOT NULL,
    model TEXT NOT NULL,
    year INTEGER NOT NULL,
    status TEXT,
    row_count INTEGER,
    column_count INTEGER,
    yearly_file TEXT,
    raw_file TEXT,
    meta_file TEXT,
    PRIMARY KEY(station_id, model, year)
);

CREATE TABLE IF NOT EXISTS mos_run_meta(
    station_id TEXT NOT NULL,
    meta_key TEXT NOT NULL,
    meta_value_json TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY(station_id, meta_key)
);

CREATE TABLE IF NOT EXISTS wu_fetch_manifest(
    station_id TEXT NOT NULL,
    location_id TEXT NOT NULL,
    window_start_date TEXT NOT NULL,
    window_end_date TEXT NOT NULL,
    request_url TEXT,
    retrieved_at_utc TEXT,
    status_code INTEGER,
    bytes INTEGER,
    sha256 TEXT,
    attempts INTEGER,
    observations_count INTEGER,
    window_csv_path TEXT,
    raw_dir TEXT,
    skipped_existing INTEGER,
    error TEXT,
    PRIMARY KEY(station_id, window_start_date, window_end_date)
);

CREATE TABLE IF NOT EXISTS wu_observations_30m(
    station_id TEXT NOT NULL,
    request_location_id TEXT NOT NULL,
    valid_time_utc TEXT NOT NULL,
    valid_time_local TEXT,
    target_date_local TEXT,
    cutoff_minutes_local INTEGER,
    temp REAL,
    dew_pt REAL,
    rh REAL,
    pressure REAL,
    vis REAL,
    wspd REAL,
    wdir REAL,
    gust REAL,
    precip_hrly REAL,
    clds TEXT,
    wx_phrase TEXT,
    uv_index REAL,
    uv_desc TEXT,
    wdir_cardinal TEXT,
    PRIMARY KEY(station_id, valid_time_utc)
);

CREATE TABLE IF NOT EXISTS wu_run_meta(
    station_id TEXT NOT NULL,
    meta_key TEXT NOT NULL,
    meta_value_json TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY(station_id, meta_key)
);

CREATE TABLE IF NOT EXISTS kalshi_download_manifest(
    station_id TEXT NOT NULL,
    target_date_local TEXT NOT NULL,
    event_ticker TEXT,
    market_tickers_json TEXT,
    start_time_utc TEXT,
    end_time_utc TEXT,
    start_ts INTEGER,
    end_ts INTEGER,
    rows_written INTEGER,
    errors_json TEXT,
    csv_path TEXT,
    downloaded_at_utc TEXT NOT NULL,
    PRIMARY KEY(station_id, target_date_local)
);

CREATE TABLE IF NOT EXISTS kalshi_minute_prices(
    station_id TEXT NOT NULL,
    target_date_local TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    bucket_label TEXT NOT NULL,
    yes_price REAL,
    market_ticker TEXT,
    source_csv_path TEXT,
    PRIMARY KEY(station_id, target_date_local, timestamp_utc, bucket_label)
);

CREATE TABLE IF NOT EXISTS kalshi_run_meta(
    station_id TEXT NOT NULL,
    meta_key TEXT NOT NULL,
    meta_value_json TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY(station_id, meta_key)
);

CREATE INDEX IF NOT EXISTS idx_nws_truth_canonical_date ON nws_truth_canonical(target_date_local);
CREATE INDEX IF NOT EXISTS idx_mos_hourly_values_runtime ON mos_hourly_values(model, runtime_utc, forecast_time_utc);
CREATE INDEX IF NOT EXISTS idx_wu_observations_target_date ON wu_observations_30m(target_date_local, cutoff_minutes_local);
CREATE INDEX IF NOT EXISTS idx_kalshi_minute_prices_ts ON kalshi_minute_prices(target_date_local, timestamp_utc);
"""


def connect_station_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn


def upsert_station_registry(
    conn: sqlite3.Connection,
    *,
    seed: StationSeed,
    lookup_station_id: str,
    metadata: ResolvedStationMetadata,
) -> None:
    conn.execute(
        """
        INSERT INTO station_registry(
            station_id, metadata_lookup_station_id, tier, group_name, active_flag, traded_station_flag, kalshi_series,
            iem_station_id, iem_network, station_zoneid, latitude, longitude, elevation_m, display_name,
            archive_begin, archive_end, climate_site, wfo, nws_usw, ncei91, ghcnh_id, synop_wban,
            metar_reset_minute, wu_location_id, seed_json, raw_feature_json, updated_at_utc
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(station_id) DO UPDATE SET
            metadata_lookup_station_id=excluded.metadata_lookup_station_id,
            tier=excluded.tier,
            group_name=excluded.group_name,
            active_flag=excluded.active_flag,
            traded_station_flag=excluded.traded_station_flag,
            kalshi_series=excluded.kalshi_series,
            iem_station_id=excluded.iem_station_id,
            iem_network=excluded.iem_network,
            station_zoneid=excluded.station_zoneid,
            latitude=excluded.latitude,
            longitude=excluded.longitude,
            elevation_m=excluded.elevation_m,
            display_name=excluded.display_name,
            archive_begin=excluded.archive_begin,
            archive_end=excluded.archive_end,
            climate_site=excluded.climate_site,
            wfo=excluded.wfo,
            nws_usw=excluded.nws_usw,
            ncei91=excluded.ncei91,
            ghcnh_id=excluded.ghcnh_id,
            synop_wban=excluded.synop_wban,
            metar_reset_minute=excluded.metar_reset_minute,
            wu_location_id=excluded.wu_location_id,
            seed_json=excluded.seed_json,
            raw_feature_json=excluded.raw_feature_json,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            seed.station_id,
            lookup_station_id,
            seed.tier,
            seed.group_name,
            int(seed.active),
            int(seed.traded_station),
            seed.kalshi_series,
            metadata.iem_station_id,
            metadata.iem_network,
            metadata.station_zoneid,
            metadata.latitude,
            metadata.longitude,
            metadata.elevation_m,
            metadata.display_name,
            metadata.archive_begin,
            metadata.archive_end,
            metadata.climate_site,
            metadata.wfo,
            metadata.nws_usw,
            metadata.ncei91,
            metadata.ghcnh_id,
            metadata.synop_wban,
            metadata.metar_reset_minute,
            metadata.wu_location_id,
            json.dumps(asdict(seed), sort_keys=True),
            json.dumps(metadata.raw_feature, sort_keys=True),
            now_utc(),
        ),
    )


def begin_ingest_run(conn: sqlite3.Connection, *, run_id: str, stage_name: str) -> None:
    conn.execute(
        """
        INSERT INTO ingest_runs(run_id, stage_name, started_at_utc, finished_at_utc, status, summary_json)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(run_id) DO UPDATE SET
            stage_name=excluded.stage_name,
            started_at_utc=excluded.started_at_utc,
            finished_at_utc=excluded.finished_at_utc,
            status=excluded.status,
            summary_json=excluded.summary_json
        """,
        (run_id, stage_name, now_utc(), None, "RUNNING", None),
    )


def finish_ingest_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    status: str,
    summary: dict[str, Any],
) -> None:
    conn.execute(
        "UPDATE ingest_runs SET finished_at_utc=?, status=?, summary_json=? WHERE run_id=?",
        (now_utc(), status, json.dumps(summary, sort_keys=True), run_id),
    )


def log_ingest_event(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    level: str,
    component: str,
    message: str,
    detail: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO ingest_events(run_id, event_time_utc, level, component, message, detail_json)
        VALUES(?,?,?,?,?,?)
        """,
        (run_id, now_utc(), level.upper(), component, message, json.dumps(detail, sort_keys=True) if detail else None),
    )


def upsert_source_status(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    source_name: str,
    status: str,
    started_at_utc: str,
    finished_at_utc: str | None,
    detail: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO ingest_source_status(run_id, source_name, started_at_utc, finished_at_utc, status, detail_json)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(run_id, source_name) DO UPDATE SET
            started_at_utc=excluded.started_at_utc,
            finished_at_utc=excluded.finished_at_utc,
            status=excluded.status,
            detail_json=excluded.detail_json
        """,
        (
            run_id,
            source_name,
            started_at_utc,
            finished_at_utc,
            status,
            json.dumps(detail, sort_keys=True) if detail else None,
        ),
    )
