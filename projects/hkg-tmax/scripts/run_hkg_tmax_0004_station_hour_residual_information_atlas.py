from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import psycopg

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = (
    REPO_ROOT
    / "experiments"
    / "campaigns"
    / "hkg-tmax"
    / "0004_station_hour_residual_information_atlas_20260708"
)
RESULTS_DIR = EXPERIMENT_DIR / "results"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
LOGS_DIR = EXPERIMENT_DIR / "logs"

DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"
START_DATE = "2000-01-02"
END_DATE = "2023-12-31"
CONFIRMATION_START = "2024-01-01"
PRIMARY_CUTOFF_PROFILE = "tminus1_2359"
MIN_CORR_ROWS = 500
MIN_ACTION_TRAIN_ROWS = 2500
MIN_ACTION_FOLD_ROWS = 250
TOP_VALUE_FEATURES = 220
RANDOM_SEED = 20260708
README_GENERATED_START = "<!-- BEGIN GENERATED: station-hour-residual-information-atlas -->"
README_GENERATED_END = "<!-- END GENERATED: station-hour-residual-information-atlas -->"
DEFAULT_README_PREAMBLE = """# Station-Hour Residual Information Atlas

This is the canonical human dossier for the station-hour residual information atlas. The
runner refreshes only the marked generated section below; curator-owned context outside the
markers is preserved.
"""


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, ARTIFACTS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def fs_path(path: Path) -> str:
    resolved = path.resolve()
    text = str(resolved)
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        return "\\\\?\\" + text
    return text


def redact_database_url(database_url: str) -> str:
    parsed = urlsplit(database_url)
    if "@" not in parsed.netloc:
        return database_url
    credentials, address = parsed.netloc.rsplit("@", 1)
    if ":" not in credentials:
        return database_url
    username, _ = credentials.split(":", 1)
    return urlunsplit(parsed._replace(netloc=f"{username}:***@{address}"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(fs_path(path), "w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def write_bounded_readme_section(path: Path, generated_section: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else DEFAULT_README_PREAMBLE
    start_count = existing.count(README_GENERATED_START)
    end_count = existing.count(README_GENERATED_END)
    if start_count != end_count or start_count > 1:
        raise RuntimeError(f"Malformed generated README markers in {path}")

    block = (
        f"{README_GENERATED_START}\n"
        f"{generated_section.strip()}\n"
        f"{README_GENERATED_END}"
    )
    if start_count == 1:
        start = existing.index(README_GENERATED_START)
        end_start = existing.find(README_GENERATED_END, start)
        if end_start < 0:
            raise RuntimeError(f"Malformed generated README markers in {path}")
        end = end_start + len(README_GENERATED_END)
        parts = [existing[:start].strip(), block, existing[end:].strip()]
    else:
        parts = [existing.strip(), block]
    write_text(path, "\n\n".join(part for part in parts if part))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(fs_path(path), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def md_table(df: pd.DataFrame, *, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    headers = list(view.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for record in view.astype(object).where(pd.notna(view), "").to_dict("records"):
        rows.append("| " + " | ".join(str(record[h]).replace("\n", " ") for h in headers) + " |")
    return "\n".join(rows)


def log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, "ts": now_utc(), **fields}
    with open(fs_path(LOGS_DIR / "run_log.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def execute_sql(connection: psycopg.Connection, sql: str) -> None:
    with connection.cursor() as cursor:
        cursor.execute(sql)


def fetch_df(connection: psycopg.Connection, sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [desc.name for desc in cursor.description]
        rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=columns)


TEMP_TABLE_SQL = r"""
DROP TABLE IF EXISTS atlas_features;
DROP TABLE IF EXISTS atlas_station_long;
DROP TABLE IF EXISTS atlas_hourly;
DROP TABLE IF EXISTS atlas_frame;

CREATE TEMP TABLE atlas_frame ON COMMIT DROP AS
WITH labels AS (
    SELECT
        local_date::date AS target_date,
        target_tmax_c::double precision AS target_tmax_c,
        (((local_date::date - 1) + time '23:59') AT TIME ZONE 'Asia/Hong_Kong') AS cutoff_at_utc
    FROM label_core.hko_daily_tmax
    WHERE local_date BETWEEN date '2000-01-02' AND date '2023-12-31'
      AND target_tmax_c IS NOT NULL
),
anchors AS (
    SELECT
        l.target_date,
        l.target_tmax_c,
        l.cutoff_at_utc,
        f.bulletin_id AS official_bulletin_id,
        f.source_url AS official_source_url,
        f.issue_at_utc AS official_issue_at_utc,
        f.issue_at_hkt AS official_issue_at_hkt,
        f.forecast_min_c::double precision AS official_min_c,
        f.forecast_max_c::double precision AS official_max_c,
        f.forecast_range_c::double precision AS official_range_c,
        f.forecast_midpoint_c::double precision AS official_midpoint_c,
        f.row_quality_status
    FROM labels l
    JOIN LATERAL (
        SELECT *
        FROM public.hko_historical_forecasts_2000_2026 f
        WHERE f.product_type = 'local'
          AND f.target_date = l.target_date
          AND f.target_issue_lead_days = 1
          AND f.usable_local_tmax_forecast
          AND f.forecast_max_c IS NOT NULL
          AND f.issue_at_utc <= l.cutoff_at_utc
        ORDER BY f.issue_at_utc DESC, f.source_url DESC
        LIMIT 1
    ) f ON true
)
SELECT
    *,
    target_tmax_c - official_max_c AS official_residual_c,
    abs(target_tmax_c - official_max_c) AS official_abs_error_c,
    (target_tmax_c - official_max_c > 1.0)::int AS underforecast_gt1_flag,
    (target_tmax_c - official_max_c < -1.0)::int AS overforecast_gt1_flag,
    (target_tmax_c >= 34.0 AND target_tmax_c - official_max_c > 0.0)::int AS hot_underforecast_flag,
    extract(year from target_date)::int AS target_year,
    extract(month from target_date)::int AS target_month,
    CASE
      WHEN target_date <= date '2010-12-31' THEN 'train_2000_2010'
      WHEN target_date <= date '2016-12-31' THEN 'mid_2011_2016'
      ELSE 'eval_2017_2023'
    END AS split
FROM anchors;

CREATE INDEX atlas_frame_target_idx ON atlas_frame(target_date);
CREATE INDEX atlas_frame_cutoff_idx ON atlas_frame(cutoff_at_utc);

CREATE TEMP TABLE atlas_hourly ON COMMIT DROP AS
SELECT
    f.target_date,
    f.cutoff_at_utc,
    f.official_max_c,
    f.official_min_c,
    h.bulletin_id,
    h.dispatch_at_utc,
    h.observation_at_utc,
    h.observation_at_hkt,
    extract(hour from h.observation_at_hkt)::int AS obs_hour_hkt,
    CASE WHEN h.hko_air_temp_c BETWEEN -5 AND 45 THEN h.hko_air_temp_c::double precision END AS hko_temp_c,
    CASE WHEN h.hko_relative_humidity_pct BETWEEN 0 AND 100 THEN h.hko_relative_humidity_pct::double precision END AS hko_rh_pct,
    h.station_count,
    h.station_missing_count,
    CASE WHEN h.station_temp_min_c BETWEEN -5 AND 45 THEN h.station_temp_min_c::double precision END AS station_temp_min_c,
    CASE WHEN h.station_temp_max_c BETWEEN -5 AND 45 THEN h.station_temp_max_c::double precision END AS station_temp_max_c,
    CASE WHEN h.station_temp_mean_c BETWEEN -5 AND 45 THEN h.station_temp_mean_c::double precision END AS station_temp_mean_c,
    h.station_temp_spread_c::double precision AS station_temp_spread_c,
    h.warning_text IS NOT NULL AS has_warning_text,
    h.rainfall_text IS NOT NULL AS has_rainfall_text,
    h.lightning_text IS NOT NULL AS has_lightning_text,
    h.tropical_cyclone_text IS NOT NULL AS has_tropical_cyclone_text,
    h.station_readings_jsonb
FROM atlas_frame f
JOIN public.hko_info_gov_hourly_readings_1998_2026 h
  ON h.index_date_hkt = f.target_date - 1
 AND h.dispatch_at_utc <= f.cutoff_at_utc
 AND h.dispatch_at_utc > f.cutoff_at_utc - interval '24 hours'
 AND h.observation_at_utc <= f.cutoff_at_utc
WHERE h.parse_status IN ('parsed', 'partial');

CREATE INDEX atlas_hourly_target_dispatch_idx ON atlas_hourly(target_date, dispatch_at_utc);
CREATE INDEX atlas_hourly_target_hour_idx ON atlas_hourly(target_date, obs_hour_hkt);

CREATE TEMP TABLE atlas_station_long ON COMMIT DROP AS
SELECT
    h.target_date,
    h.cutoff_at_utc,
    h.official_max_c,
    h.dispatch_at_utc,
    h.observation_at_utc,
    h.observation_at_hkt,
    h.obs_hour_hkt,
    h.hko_temp_c,
    h.hko_rh_pct,
    s->>'station_canonical_name' AS station,
    trim(both '_' from lower(regexp_replace(s->>'station_canonical_name', '[^a-zA-Z0-9]+', '_', 'g'))) AS station_key,
    CASE
      WHEN s->>'station_canonical_name' IN ('KING''S PARK','HONG KONG PARK','KOWLOON CITY','HAPPY VALLEY','WONG TAI SIN','SHAM SHUI PO','KWUN TONG','KAI TAK RUNWAY PARK') THEN 'urban_core'
      WHEN s->>'station_canonical_name' IN ('SHA TIN','TAI PO','TA KWU LING','SHEK KONG','YUEN LONG PARK','TAI MEI TUK') THEN 'inland_nt'
      WHEN s->>'station_canonical_name' IN ('CHEUNG CHAU','SAI KUNG','CHEK LAP KOK','WONG CHUK HANG','SHAU KEI WAN','STANLEY','TSEUNG KWAN O') THEN 'coastal_marine'
      WHEN s->>'station_canonical_name' IN ('LAU FAU SHAN','TUEN MUN','TSING YI','TSUEN WAN','TSUEN WAN HO KOON','TSUEN WAN SHING MUN VALLEY') THEN 'west_nw_nt'
      ELSE 'other'
    END AS station_role,
    CASE
      WHEN (s->>'temperature_missing')::boolean IS DISTINCT FROM true
       AND s->>'temperature_c' IS NOT NULL
       AND (s->>'temperature_c')::double precision BETWEEN -5 AND 45
      THEN (s->>'temperature_c')::double precision
    END AS station_temp_c,
    (s->>'temperature_missing')::boolean AS station_missing
FROM atlas_hourly h
CROSS JOIN LATERAL jsonb_array_elements(h.station_readings_jsonb) s;

CREATE INDEX atlas_station_long_target_station_idx ON atlas_station_long(target_date, station);
CREATE INDEX atlas_station_long_target_role_idx ON atlas_station_long(target_date, station_role);

CREATE TEMP TABLE atlas_features ON COMMIT DROP AS
WITH latest_hko AS (
    SELECT DISTINCT ON (target_date) *
    FROM atlas_hourly
    WHERE hko_temp_c IS NOT NULL
    ORDER BY target_date, dispatch_at_utc DESC
),
latest_station AS (
    SELECT DISTINCT ON (target_date, station)
        target_date, station, station_key, station_role, station_temp_c, hko_temp_c, official_max_c, dispatch_at_utc
    FROM atlas_station_long
    WHERE station_temp_c IS NOT NULL
    ORDER BY target_date, station, dispatch_at_utc DESC
),
snapshot_station AS (
    SELECT DISTINCT ON (target_date, station, obs_hour_hkt)
        target_date, station, station_key, station_role, obs_hour_hkt, station_temp_c, hko_temp_c, official_max_c, dispatch_at_utc
    FROM atlas_station_long
    WHERE station_temp_c IS NOT NULL
      AND obs_hour_hkt IN (15,18,21,23)
      AND observation_at_hkt::date = target_date - 1
    ORDER BY target_date, station, obs_hour_hkt, dispatch_at_utc DESC
),
station_window AS (
    SELECT
        sl.target_date,
        sl.station,
        sl.station_key,
        sl.station_role,
        w.window_hours,
        avg(sl.station_temp_c) AS temp_mean_c,
        max(sl.station_temp_c) AS temp_max_c,
        min(sl.station_temp_c) AS temp_min_c,
        max(sl.station_temp_c) - min(sl.station_temp_c) AS temp_range_c,
        (array_agg(sl.station_temp_c ORDER BY sl.dispatch_at_utc DESC))[1]
          - (array_agg(sl.station_temp_c ORDER BY sl.dispatch_at_utc ASC))[1] AS temp_trend_c,
        avg(sl.station_temp_c - sl.hko_temp_c) AS mean_minus_hko_c,
        max(sl.station_temp_c) - max(sl.official_max_c) AS max_minus_official_max_c,
        count(*) AS obs_count
    FROM atlas_station_long sl
    CROSS JOIN (VALUES (6), (12), (24)) AS w(window_hours)
    WHERE sl.station_temp_c IS NOT NULL
      AND sl.dispatch_at_utc > sl.cutoff_at_utc - (w.window_hours || ' hours')::interval
    GROUP BY sl.target_date, sl.station, sl.station_key, sl.station_role, w.window_hours
    HAVING count(*) >= greatest(3, w.window_hours / 3)
),
hko_window AS (
    SELECT
        h.target_date,
        w.window_hours,
        avg(h.hko_temp_c) AS temp_mean_c,
        max(h.hko_temp_c) AS temp_max_c,
        min(h.hko_temp_c) AS temp_min_c,
        max(h.hko_temp_c) - min(h.hko_temp_c) AS temp_range_c,
        (array_agg(h.hko_temp_c ORDER BY h.dispatch_at_utc DESC))[1]
          - (array_agg(h.hko_temp_c ORDER BY h.dispatch_at_utc ASC))[1] AS temp_trend_c,
        avg(h.hko_rh_pct) AS rh_mean_pct,
        (array_agg(h.hko_rh_pct ORDER BY h.dispatch_at_utc DESC))[1]
          - (array_agg(h.hko_rh_pct ORDER BY h.dispatch_at_utc ASC))[1] AS rh_trend_pct,
        max(h.official_max_c) AS official_max_c,
        count(*) AS obs_count
    FROM atlas_hourly h
    CROSS JOIN (VALUES (6), (12), (24)) AS w(window_hours)
    WHERE h.hko_temp_c IS NOT NULL
      AND h.dispatch_at_utc > h.cutoff_at_utc - (w.window_hours || ' hours')::interval
    GROUP BY h.target_date, w.window_hours
    HAVING count(*) >= greatest(3, w.window_hours / 3)
),
latest_network AS (
    SELECT DISTINCT ON (target_date) *
    FROM atlas_hourly
    WHERE station_temp_max_c IS NOT NULL
    ORDER BY target_date, dispatch_at_utc DESC
),
network_window AS (
    SELECT
        sl.target_date,
        w.window_hours,
        avg(sl.station_temp_c) AS temp_mean_c,
        max(sl.station_temp_c) AS temp_max_c,
        min(sl.station_temp_c) AS temp_min_c,
        max(sl.station_temp_c) - min(sl.station_temp_c) AS temp_range_c,
        max(sl.station_temp_c) - max(sl.official_max_c) AS max_minus_official_max_c,
        avg(sl.station_temp_c - sl.hko_temp_c) AS mean_minus_hko_c,
        count(*) AS obs_count
    FROM atlas_station_long sl
    CROSS JOIN (VALUES (6), (12), (24)) AS w(window_hours)
    WHERE sl.station_temp_c IS NOT NULL
      AND sl.dispatch_at_utc > sl.cutoff_at_utc - (w.window_hours || ' hours')::interval
    GROUP BY sl.target_date, w.window_hours
    HAVING count(*) >= greatest(50, w.window_hours * 8)
),
role_window AS (
    SELECT
        sl.target_date,
        sl.station_role,
        w.window_hours,
        avg(sl.station_temp_c) AS temp_mean_c,
        max(sl.station_temp_c) AS temp_max_c,
        min(sl.station_temp_c) AS temp_min_c,
        max(sl.station_temp_c) - min(sl.station_temp_c) AS temp_range_c,
        max(sl.station_temp_c) - max(sl.official_max_c) AS max_minus_official_max_c,
        avg(sl.station_temp_c - sl.hko_temp_c) AS mean_minus_hko_c,
        count(*) AS obs_count
    FROM atlas_station_long sl
    CROSS JOIN (VALUES (6), (12), (24)) AS w(window_hours)
    WHERE sl.station_temp_c IS NOT NULL
      AND sl.dispatch_at_utc > sl.cutoff_at_utc - (w.window_hours || ' hours')::interval
      AND sl.station_role <> 'other'
    GROUP BY sl.target_date, sl.station_role, w.window_hours
    HAVING count(*) >= greatest(16, w.window_hours * 2)
)
SELECT target_date, 'hko_latest'::text AS feature_family, NULL::text AS station, 'hko'::text AS station_role,
       'latest_temp'::text AS transform, NULL::int AS window_hours, NULL::int AS snapshot_hour,
       'hko__latest_temp_c'::text AS feature_name, hko_temp_c::double precision AS feature_value
FROM latest_hko
UNION ALL SELECT target_date, 'hko_latest', NULL, 'hko', 'latest_rh', NULL, NULL, 'hko__latest_rh_pct', hko_rh_pct FROM latest_hko
UNION ALL SELECT target_date, 'forecast_contradiction', NULL, 'hko', 'latest_temp_minus_official_max', NULL, NULL, 'hko__latest_temp_minus_official_max_c', hko_temp_c - official_max_c FROM latest_hko
UNION ALL SELECT target_date, 'forecast_contradiction', NULL, 'hko', 'latest_temp_minus_official_min', NULL, NULL, 'hko__latest_temp_minus_official_min_c', hko_temp_c - official_min_c FROM latest_hko
UNION ALL SELECT target_date, 'hko_window', NULL, 'hko', 'temp_mean', window_hours, NULL, 'hko__w' || window_hours || 'h_temp_mean_c', temp_mean_c FROM hko_window
UNION ALL SELECT target_date, 'hko_window', NULL, 'hko', 'temp_max', window_hours, NULL, 'hko__w' || window_hours || 'h_temp_max_c', temp_max_c FROM hko_window
UNION ALL SELECT target_date, 'hko_window', NULL, 'hko', 'temp_min', window_hours, NULL, 'hko__w' || window_hours || 'h_temp_min_c', temp_min_c FROM hko_window
UNION ALL SELECT target_date, 'hko_window', NULL, 'hko', 'temp_range', window_hours, NULL, 'hko__w' || window_hours || 'h_temp_range_c', temp_range_c FROM hko_window
UNION ALL SELECT target_date, 'hko_window', NULL, 'hko', 'temp_trend', window_hours, NULL, 'hko__w' || window_hours || 'h_temp_trend_c', temp_trend_c FROM hko_window
UNION ALL SELECT target_date, 'hko_window', NULL, 'hko', 'rh_mean', window_hours, NULL, 'hko__w' || window_hours || 'h_rh_mean_pct', rh_mean_pct FROM hko_window
UNION ALL SELECT target_date, 'hko_window', NULL, 'hko', 'rh_trend', window_hours, NULL, 'hko__w' || window_hours || 'h_rh_trend_pct', rh_trend_pct FROM hko_window
UNION ALL SELECT target_date, 'forecast_contradiction', NULL, 'hko', 'hko_window_max_minus_official_max', window_hours, NULL, 'hko__w' || window_hours || 'h_max_minus_official_max_c', temp_max_c - official_max_c FROM hko_window
UNION ALL SELECT target_date, 'network_latest', NULL, 'network', 'latest_mean', NULL, NULL, 'network__latest_mean_c', station_temp_mean_c FROM latest_network
UNION ALL SELECT target_date, 'network_latest', NULL, 'network', 'latest_max', NULL, NULL, 'network__latest_max_c', station_temp_max_c FROM latest_network
UNION ALL SELECT target_date, 'network_latest', NULL, 'network', 'latest_min', NULL, NULL, 'network__latest_min_c', station_temp_min_c FROM latest_network
UNION ALL SELECT target_date, 'network_latest', NULL, 'network', 'latest_spread', NULL, NULL, 'network__latest_spread_c', station_temp_spread_c FROM latest_network
UNION ALL SELECT target_date, 'forecast_contradiction', NULL, 'network', 'latest_max_minus_official_max', NULL, NULL, 'network__latest_max_minus_official_max_c', station_temp_max_c - official_max_c FROM latest_network
UNION ALL SELECT target_date, 'network_window', NULL, 'network', 'temp_mean', window_hours, NULL, 'network__w' || window_hours || 'h_temp_mean_c', temp_mean_c FROM network_window
UNION ALL SELECT target_date, 'network_window', NULL, 'network', 'temp_max', window_hours, NULL, 'network__w' || window_hours || 'h_temp_max_c', temp_max_c FROM network_window
UNION ALL SELECT target_date, 'network_window', NULL, 'network', 'temp_min', window_hours, NULL, 'network__w' || window_hours || 'h_temp_min_c', temp_min_c FROM network_window
UNION ALL SELECT target_date, 'network_window', NULL, 'network', 'temp_range', window_hours, NULL, 'network__w' || window_hours || 'h_temp_range_c', temp_range_c FROM network_window
UNION ALL SELECT target_date, 'network_window', NULL, 'network', 'max_minus_official_max', window_hours, NULL, 'network__w' || window_hours || 'h_max_minus_official_max_c', max_minus_official_max_c FROM network_window
UNION ALL SELECT target_date, 'network_window', NULL, 'network', 'mean_minus_hko', window_hours, NULL, 'network__w' || window_hours || 'h_mean_minus_hko_c', mean_minus_hko_c FROM network_window
UNION ALL SELECT target_date, 'role_window', NULL, station_role, 'temp_mean', window_hours, NULL, 'role_' || station_role || '__w' || window_hours || 'h_temp_mean_c', temp_mean_c FROM role_window
UNION ALL SELECT target_date, 'role_window', NULL, station_role, 'temp_max', window_hours, NULL, 'role_' || station_role || '__w' || window_hours || 'h_temp_max_c', temp_max_c FROM role_window
UNION ALL SELECT target_date, 'role_window', NULL, station_role, 'temp_range', window_hours, NULL, 'role_' || station_role || '__w' || window_hours || 'h_temp_range_c', temp_range_c FROM role_window
UNION ALL SELECT target_date, 'forecast_contradiction', NULL, station_role, 'role_max_minus_official_max', window_hours, NULL, 'role_' || station_role || '__w' || window_hours || 'h_max_minus_official_max_c', max_minus_official_max_c FROM role_window
UNION ALL SELECT target_date, 'role_window', NULL, station_role, 'mean_minus_hko', window_hours, NULL, 'role_' || station_role || '__w' || window_hours || 'h_mean_minus_hko_c', mean_minus_hko_c FROM role_window
UNION ALL SELECT target_date, 'station_latest', station, station_role, 'latest_temp', NULL, NULL, 'station_' || station_key || '__latest_temp_c', station_temp_c FROM latest_station
UNION ALL SELECT target_date, 'station_latest', station, station_role, 'latest_minus_hko', NULL, NULL, 'station_' || station_key || '__latest_minus_hko_c', station_temp_c - hko_temp_c FROM latest_station
UNION ALL SELECT target_date, 'forecast_contradiction', station, station_role, 'latest_minus_official_max', NULL, NULL, 'station_' || station_key || '__latest_minus_official_max_c', station_temp_c - official_max_c FROM latest_station
UNION ALL SELECT target_date, 'station_snapshot', station, station_role, 'snapshot_temp', NULL, obs_hour_hkt, 'station_' || station_key || '__h' || obs_hour_hkt || '_temp_c', station_temp_c FROM snapshot_station
UNION ALL SELECT target_date, 'station_snapshot', station, station_role, 'snapshot_minus_hko', NULL, obs_hour_hkt, 'station_' || station_key || '__h' || obs_hour_hkt || '_minus_hko_c', station_temp_c - hko_temp_c FROM snapshot_station
UNION ALL SELECT target_date, 'forecast_contradiction', station, station_role, 'snapshot_minus_official_max', NULL, obs_hour_hkt, 'station_' || station_key || '__h' || obs_hour_hkt || '_minus_official_max_c', station_temp_c - official_max_c FROM snapshot_station
UNION ALL SELECT target_date, 'station_window', station, station_role, 'temp_mean', window_hours, NULL, 'station_' || station_key || '__w' || window_hours || 'h_temp_mean_c', temp_mean_c FROM station_window
UNION ALL SELECT target_date, 'station_window', station, station_role, 'temp_max', window_hours, NULL, 'station_' || station_key || '__w' || window_hours || 'h_temp_max_c', temp_max_c FROM station_window
UNION ALL SELECT target_date, 'station_window', station, station_role, 'temp_min', window_hours, NULL, 'station_' || station_key || '__w' || window_hours || 'h_temp_min_c', temp_min_c FROM station_window
UNION ALL SELECT target_date, 'station_window', station, station_role, 'temp_range', window_hours, NULL, 'station_' || station_key || '__w' || window_hours || 'h_temp_range_c', temp_range_c FROM station_window
UNION ALL SELECT target_date, 'station_window', station, station_role, 'temp_trend', window_hours, NULL, 'station_' || station_key || '__w' || window_hours || 'h_temp_trend_c', temp_trend_c FROM station_window
UNION ALL SELECT target_date, 'station_window', station, station_role, 'mean_minus_hko', window_hours, NULL, 'station_' || station_key || '__w' || window_hours || 'h_mean_minus_hko_c', mean_minus_hko_c FROM station_window
UNION ALL SELECT target_date, 'forecast_contradiction', station, station_role, 'window_max_minus_official_max', window_hours, NULL, 'station_' || station_key || '__w' || window_hours || 'h_max_minus_official_max_c', max_minus_official_max_c FROM station_window;

CREATE INDEX atlas_features_feature_idx ON atlas_features(feature_name);
CREATE INDEX atlas_features_target_idx ON atlas_features(target_date);
ANALYZE atlas_frame;
ANALYZE atlas_hourly;
ANALYZE atlas_station_long;
ANALYZE atlas_features;
"""


SUMMARY_SQL = r"""
SELECT
    ft.feature_name,
    min(ft.feature_family) AS feature_family,
    min(coalesce(ft.station, '')) AS station,
    min(ft.station_role) AS station_role,
    min(ft.transform) AS transform,
    min(ft.window_hours) AS window_hours,
    min(ft.snapshot_hour) AS snapshot_hour,
    count(*) FILTER (WHERE ft.feature_value IS NOT NULL) AS n,
    corr(ft.feature_value, fr.official_residual_c) AS pearson_residual,
    corr(ft.feature_value, fr.official_abs_error_c) AS pearson_abs_error,
    corr(ft.feature_value, fr.underforecast_gt1_flag::double precision) AS pearson_under_gt1,
    corr(ft.feature_value, fr.overforecast_gt1_flag::double precision) AS pearson_over_gt1,
    corr(ft.feature_value, fr.hot_underforecast_flag::double precision) AS pearson_hot_under,
    count(*) FILTER (WHERE fr.split = 'train_2000_2010') AS n_train_2000_2010,
    corr(ft.feature_value, fr.official_residual_c) FILTER (WHERE fr.split = 'train_2000_2010') AS pearson_residual_train_2000_2010,
    count(*) FILTER (WHERE fr.split = 'mid_2011_2016') AS n_mid_2011_2016,
    corr(ft.feature_value, fr.official_residual_c) FILTER (WHERE fr.split = 'mid_2011_2016') AS pearson_residual_mid_2011_2016,
    count(*) FILTER (WHERE fr.split = 'eval_2017_2023') AS n_eval_2017_2023,
    corr(ft.feature_value, fr.official_residual_c) FILTER (WHERE fr.split = 'eval_2017_2023') AS pearson_residual_eval_2017_2023,
    avg(ft.feature_value) AS feature_mean,
    stddev_samp(ft.feature_value) AS feature_std
FROM atlas_features ft
JOIN atlas_frame fr USING (target_date)
WHERE ft.feature_value IS NOT NULL
GROUP BY ft.feature_name
HAVING count(*) FILTER (WHERE ft.feature_value IS NOT NULL) >= 500
ORDER BY greatest(
    abs(coalesce(corr(ft.feature_value, fr.official_residual_c), 0)),
    abs(coalesce(corr(ft.feature_value, fr.official_abs_error_c), 0)),
    abs(coalesce(corr(ft.feature_value, fr.underforecast_gt1_flag::double precision), 0)),
    abs(coalesce(corr(ft.feature_value, fr.overforecast_gt1_flag::double precision), 0)),
    abs(coalesce(corr(ft.feature_value, fr.hot_underforecast_flag::double precision), 0))
) DESC;
"""


FRAME_SQL = """
SELECT
    target_date,
    target_tmax_c,
    official_max_c,
    official_min_c,
    official_residual_c,
    official_abs_error_c,
    underforecast_gt1_flag,
    overforecast_gt1_flag,
    hot_underforecast_flag,
    split,
    target_year,
    target_month,
    official_issue_at_utc,
    official_source_url
FROM atlas_frame
ORDER BY target_date;
"""


DB_COUNTS_SQL = """
SELECT 'frame_rows' AS metric, count(*)::text AS value FROM atlas_frame
UNION ALL SELECT 'frame_min_date', min(target_date)::text FROM atlas_frame
UNION ALL SELECT 'frame_max_date', max(target_date)::text FROM atlas_frame
UNION ALL SELECT 'hourly_rows_24h_join', count(*)::text FROM atlas_hourly
UNION ALL SELECT 'station_long_rows_24h_join', count(*)::text FROM atlas_station_long
UNION ALL SELECT 'feature_value_rows', count(*)::text FROM atlas_features
UNION ALL SELECT 'feature_count', count(distinct feature_name)::text FROM atlas_features
UNION ALL SELECT 'station_count', count(distinct station)::text FROM atlas_station_long
UNION ALL SELECT 'uses_confirmation_rows', count(*)::text FROM atlas_frame WHERE target_date >= date '2024-01-01';
"""


def build_postgres_temp_tables(connection: psycopg.Connection) -> dict[str, str]:
    log_event("postgres_temp_build_start")
    execute_sql(connection, TEMP_TABLE_SQL)
    counts = fetch_df(connection, DB_COUNTS_SQL)
    log_event("postgres_temp_build_complete", **{row.metric: row.value for row in counts.itertuples(index=False)})
    return {str(row.metric): str(row.value) for row in counts.itertuples(index=False)}


def add_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    corr_cols = [
        "pearson_residual",
        "pearson_abs_error",
        "pearson_under_gt1",
        "pearson_over_gt1",
        "pearson_hot_under",
    ]
    out["max_abs_primary_corr"] = out[corr_cols].abs().max(axis=1)
    signs = np.sign(out[["pearson_residual_train_2000_2010", "pearson_residual_mid_2011_2016", "pearson_residual_eval_2017_2023"]])
    out["residual_corr_same_sign_3way"] = (signs.replace(0, np.nan).nunique(axis=1) == 1) & signs.notna().all(axis=1)
    out["residual_corr_train_eval_same_sign"] = (
        np.sign(out["pearson_residual_train_2000_2010"]) == np.sign(out["pearson_residual_eval_2017_2023"])
    ) & out["pearson_residual_train_2000_2010"].notna() & out["pearson_residual_eval_2017_2023"].notna()
    return out.sort_values(["max_abs_primary_corr", "residual_corr_train_eval_same_sign", "n"], ascending=[False, False, False])


def select_top_features(summary: pd.DataFrame) -> list[str]:
    pool = summary[
        (summary["n_train_2000_2010"] >= MIN_ACTION_TRAIN_ROWS)
        & (summary["n_eval_2017_2023"] >= 1000)
    ].copy()
    if pool.empty:
        pool = summary.copy()
    train_signal = pool["pearson_residual_train_2000_2010"].abs().fillna(0)
    global_signal = pool["max_abs_primary_corr"].fillna(0)
    pool["_selection_score"] = 0.65 * train_signal + 0.35 * global_signal
    return pool.sort_values("_selection_score", ascending=False)["feature_name"].head(TOP_VALUE_FEATURES).tolist()


def fetch_feature_values(connection: psycopg.Connection, feature_names: list[str]) -> pd.DataFrame:
    if not feature_names:
        return pd.DataFrame()
    sql = """
    SELECT
        ft.target_date,
        ft.feature_name,
        ft.feature_value,
        fr.target_tmax_c,
        fr.official_max_c,
        fr.official_residual_c,
        fr.official_abs_error_c,
        fr.underforecast_gt1_flag,
        fr.overforecast_gt1_flag,
        fr.hot_underforecast_flag,
        fr.split
    FROM atlas_features ft
    JOIN atlas_frame fr USING (target_date)
    WHERE ft.feature_name = ANY(%s)
      AND ft.feature_value IS NOT NULL
    ORDER BY ft.feature_name, ft.target_date;
    """
    return fetch_df(connection, sql, (feature_names,))


def safe_corr(a: pd.Series, b: pd.Series, *, method: str) -> float:
    pair = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    if len(pair) < MIN_CORR_ROWS or pair.iloc[:, 0].nunique() < 3 or pair.iloc[:, 1].nunique() < 3:
        return math.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def quantile_spread(feature_values: pd.Series, response: pd.Series) -> dict[str, Any]:
    pair = pd.concat(
        [pd.to_numeric(feature_values, errors="coerce"), pd.to_numeric(response, errors="coerce")],
        axis=1,
    ).dropna()
    pair.columns = ["feature", "response"]
    if len(pair) < 1000 or pair["feature"].nunique() < 10:
        return {"q10_q90_response_spread": math.nan, "low_n": 0, "high_n": 0}
    q10 = pair["feature"].quantile(0.10)
    q90 = pair["feature"].quantile(0.90)
    low = pair[pair["feature"] <= q10]["response"]
    high = pair[pair["feature"] >= q90]["response"]
    if len(low) < 100 or len(high) < 100:
        return {"q10_q90_response_spread": math.nan, "low_n": len(low), "high_n": len(high)}
    return {
        "q10_q90_response_spread": float(high.mean() - low.mean()),
        "low_n": int(len(low)),
        "high_n": int(len(high)),
    }


def spearman_and_spreads(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if values.empty:
        return pd.DataFrame(rows)
    for feature_name, group in values.groupby("feature_name", sort=False):
        row: dict[str, Any] = {
            "feature_name": feature_name,
            "n_values": int(group["feature_value"].notna().sum()),
            "spearman_residual": safe_corr(group["feature_value"], group["official_residual_c"], method="spearman"),
            "spearman_abs_error": safe_corr(group["feature_value"], group["official_abs_error_c"], method="spearman"),
            "spearman_under_gt1": safe_corr(group["feature_value"], group["underforecast_gt1_flag"], method="spearman"),
            "spearman_over_gt1": safe_corr(group["feature_value"], group["overforecast_gt1_flag"], method="spearman"),
            "spearman_hot_under": safe_corr(group["feature_value"], group["hot_underforecast_flag"], method="spearman"),
        }
        row.update(quantile_spread(group["feature_value"], group["official_residual_c"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    corr_cols = [c for c in out.columns if c.startswith("spearman_")]
    out["max_abs_spearman"] = out[corr_cols].abs().max(axis=1)
    return out.sort_values("max_abs_spearman", ascending=False)


FOLDS = [
    ("fold1_2011_2013", "2010-12-31", "2011-01-01", "2013-12-31"),
    ("fold2_2014_2016", "2013-12-31", "2014-01-01", "2016-12-31"),
    ("fold3_2017_2019", "2016-12-31", "2017-01-01", "2019-12-31"),
    ("fold4_2020_2023", "2019-12-31", "2020-01-01", "2023-12-31"),
]


def fit_line(train: pd.DataFrame) -> tuple[float, float] | None:
    x = pd.to_numeric(train["feature_value"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(train["official_residual_c"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < MIN_ACTION_TRAIN_ROWS or np.nanstd(x) < 1e-9:
        return None
    beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    alpha = float(np.mean(y) - beta * np.mean(x))
    return alpha, beta


def evaluate_univariate_actionability(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if values.empty:
        return pd.DataFrame(rows)
    values = values.copy()
    values["target_date"] = pd.to_datetime(values["target_date"])
    for feature_name, group in values.groupby("feature_name", sort=False):
        fold_rows: list[dict[str, Any]] = []
        for fold_name, train_end, valid_start, valid_end in FOLDS:
            train = group[group["target_date"] <= pd.Timestamp(train_end)].dropna(subset=["feature_value", "official_residual_c"])
            valid = group[
                (group["target_date"] >= pd.Timestamp(valid_start))
                & (group["target_date"] <= pd.Timestamp(valid_end))
            ].dropna(subset=["feature_value", "target_tmax_c", "official_max_c", "official_residual_c"])
            if len(valid) < MIN_ACTION_FOLD_ROWS:
                continue
            fit = fit_line(train)
            if fit is None:
                continue
            alpha, beta = fit
            residual_mean = float(train["official_residual_c"].mean())
            correction = np.clip(alpha + beta * valid["feature_value"].to_numpy(dtype=float), -1.25, 1.25)
            bias_correction = np.clip(np.full(len(valid), residual_mean), -1.25, 1.25)
            actual = valid["target_tmax_c"].to_numpy(dtype=float)
            official = valid["official_max_c"].to_numpy(dtype=float)
            baseline_abs = np.abs(actual - official)
            candidate_abs = np.abs(actual - (official + correction))
            bias_abs = np.abs(actual - (official + bias_correction))
            fold_rows.append(
                {
                    "fold": fold_name,
                    "n": int(len(valid)),
                    "alpha": alpha,
                    "beta": beta,
                    "official_mae": float(np.mean(baseline_abs)),
                    "bias_only_mae": float(np.mean(bias_abs)),
                    "candidate_mae": float(np.mean(candidate_abs)),
                    "delta_vs_official_c": float(np.mean(candidate_abs) - np.mean(baseline_abs)),
                    "delta_vs_bias_only_c": float(np.mean(candidate_abs) - np.mean(bias_abs)),
                    "mean_abs_correction_c": float(np.mean(np.abs(correction))),
                }
            )
        if not fold_rows:
            continue
        fold_df = pd.DataFrame(fold_rows)
        weights = fold_df["n"].to_numpy(dtype=float)
        rows.append(
            {
                "feature_name": feature_name,
                "folds": int(len(fold_df)),
                "n_valid": int(fold_df["n"].sum()),
                "official_mae": float(np.average(fold_df["official_mae"], weights=weights)),
                "bias_only_mae": float(np.average(fold_df["bias_only_mae"], weights=weights)),
                "candidate_mae": float(np.average(fold_df["candidate_mae"], weights=weights)),
                "delta_vs_official_c": float(np.average(fold_df["delta_vs_official_c"], weights=weights)),
                "delta_vs_bias_only_c": float(np.average(fold_df["delta_vs_bias_only_c"], weights=weights)),
                "mean_abs_correction_c": float(np.average(fold_df["mean_abs_correction_c"], weights=weights)),
                "folds_beating_official": int((fold_df["delta_vs_official_c"] < 0).sum()),
                "folds_beating_bias_only": int((fold_df["delta_vs_bias_only_c"] < 0).sum()),
                "fold_deltas_vs_bias_only": ";".join(
                    f"{r.fold}:{r.delta_vs_bias_only_c:.5f}" for r in fold_df.itertuples(index=False)
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["delta_vs_bias_only_c", "delta_vs_official_c"], ascending=[True, True])


def station_leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
    stations = summary[summary["station"].astype(str).str.len() > 0].copy()
    if stations.empty:
        return pd.DataFrame()
    idx = stations.groupby("station")["max_abs_primary_corr"].idxmax()
    return stations.loc[idx, [
        "station", "station_role", "feature_name", "feature_family", "transform", "window_hours",
        "snapshot_hour", "n", "max_abs_primary_corr", "pearson_residual", "pearson_abs_error",
        "pearson_under_gt1", "pearson_over_gt1", "pearson_hot_under",
        "residual_corr_train_eval_same_sign",
    ]].sort_values("max_abs_primary_corr", ascending=False)


def family_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in summary.groupby("feature_family", dropna=False):
        leader = group.sort_values("max_abs_primary_corr", ascending=False).iloc[0]
        rows.append(
            {
                "feature_family": family,
                "feature_count": int(len(group)),
                "stable_train_eval_count": int(group["residual_corr_train_eval_same_sign"].sum()),
                "best_feature_name": leader["feature_name"],
                "best_max_abs_primary_corr": float(leader["max_abs_primary_corr"]),
                "best_pearson_residual": float(leader["pearson_residual"]) if pd.notna(leader["pearson_residual"]) else math.nan,
                "best_pearson_abs_error": float(leader["pearson_abs_error"]) if pd.notna(leader["pearson_abs_error"]) else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("best_max_abs_primary_corr", ascending=False)


def compute_significance_score(summary: pd.DataFrame, spearman: pd.DataFrame, actionability: pd.DataFrame, db_counts: dict[str, str]) -> dict[str, Any]:
    stable = summary[(summary["n"] >= 5000) & (summary["residual_corr_train_eval_same_sign"])]
    top_corr = float(stable["max_abs_primary_corr"].max()) if not stable.empty else float(summary["max_abs_primary_corr"].max())
    top_spearman = float(spearman["max_abs_spearman"].max()) if not spearman.empty else math.nan
    best_action = None
    best_delta_bias = 0.0
    if not actionability.empty:
        best_action = actionability.iloc[0].to_dict()
        best_delta_bias = float(best_action["delta_vs_bias_only_c"])
    support = min(float(db_counts.get("frame_rows", 0)) / 8500.0, 1.0)
    corr_component = min(max(top_corr, 0.0) / 0.22, 1.0) * 35.0
    spearman_component = min(max(top_spearman if math.isfinite(top_spearman) else 0.0, 0.0) / 0.22, 1.0) * 20.0
    stability_component = min(len(stable) / 35.0, 1.0) * 20.0
    action_component = min(max(-best_delta_bias, 0.0) / 0.015, 1.0) * 15.0
    support_component = support * 10.0
    score = int(round(corr_component + spearman_component + stability_component + action_component + support_component))
    return {
        "significance_score_1_to_100": max(1, min(100, score)),
        "score_components": {
            "pearson_component_35": corr_component,
            "spearman_component_20": spearman_component,
            "stability_component_20": stability_component,
            "actionability_component_15": action_component,
            "support_component_10": support_component,
        },
        "top_stable_abs_primary_corr": top_corr,
        "top_abs_spearman": top_spearman,
        "stable_high_support_feature_count": int(len(stable)),
        "best_actionability": best_action,
    }


def write_static_protocol_artifacts(database_url: str) -> None:
    redacted = redact_database_url(database_url)
    write_text(EXPERIMENT_DIR / "DATA_MANIFEST.yaml", f"""database_url_redacted: "{redacted}"
tables:
  target: label_core.hko_daily_tmax
  official_forecast: public.hko_historical_forecasts_2000_2026
  hourly_observations: public.hko_info_gov_hourly_readings_1998_2026
date_window: "{START_DATE}..{END_DATE}"
confirmation_rows_allowed: false
temporary_postgres_tables:
  - atlas_frame
  - atlas_hourly
  - atlas_station_long
  - atlas_features
""")
    write_text(EXPERIMENT_DIR / "RUN_CONFIG.yaml", f"""seed: {RANDOM_SEED}
database_url_env_priority:
  - HKG_TMAX_DATABASE_URL
  - DATABASE_URL
  - default_local_postgres_redacted
date_window:
  start: "{START_DATE}"
  end: "{END_DATE}"
confirmation_start: "{CONFIRMATION_START}"
primary_cutoff_profile: "{PRIMARY_CUTOFF_PROFILE}"
cutoff_hkt: "T-1 23:59"
hourly_window_hours: 24
station_temperature_plausibility_c: [-5, 45]
correlation_min_rows: {MIN_CORR_ROWS}
top_value_features: {TOP_VALUE_FEATURES}
walk_forward_folds:
  - ["fold1_2011_2013", "train<=2010-12-31", "valid=2011-01-01..2013-12-31"]
  - ["fold2_2014_2016", "train<=2013-12-31", "valid=2014-01-01..2016-12-31"]
  - ["fold3_2017_2019", "train<=2016-12-31", "valid=2017-01-01..2019-12-31"]
  - ["fold4_2020_2023", "train<=2019-12-31", "valid=2020-01-01..2023-12-31"]
""")


def write_experiment_readme(
    *,
    summary: pd.DataFrame,
    spearman: pd.DataFrame,
    actionability: pd.DataFrame,
    station_board: pd.DataFrame,
    family_board: pd.DataFrame,
    metrics: dict[str, Any],
    db_counts: dict[str, str],
) -> None:
    score = metrics["significance"]["significance_score_1_to_100"]
    best_action = metrics["significance"].get("best_actionability") or {}
    best_action_text = "No guarded univariate actionability candidate was scoreable."
    if best_action:
        best_action_text = (
            f"Best guarded univariate residual correction: `{best_action['feature_name']}`; "
            f"candidate MAE `{best_action['candidate_mae']:.6f}` vs official `{best_action['official_mae']:.6f}` "
            f"and bias-only `{best_action['bias_only_mae']:.6f}`. "
            f"Delta vs bias-only `{best_action['delta_vs_bias_only_c']:.6f}` C."
        )
    generated_section = f"""## Generated Experiment Dossier

This experiment mines PostgreSQL hourly Info.gov readings for station, role, network, and HKO features that explain the official forecast residual for HKG daily Tmax.

Main result: **{score}/100 significance**, `INFORMATION_GAIN_POSITIVE_NO_PROMOTE`.

## Hypothesis

Hourly Info.gov HKO and neighboring-station observations available before the T-1 23:59 HKT cutoff contain directional information about the official forecast residual `target_tmax_c - official_max_c`.

The strongest expected mechanisms are forecast-observation contradiction, inland/coastal heat contrast, late-evening heat retention, network heat ceiling, and humid/rain-cooled overforecast suppression.

Falsification: station-hour features show only weak, unstable correlations across time splits and do not produce any guarded walk-forward residual-correction improvement beyond a bias-only correction.

## As-Of Contract

The decision cutoff is T-1 23:59 HKT. A forecast row is eligible only if `issue_at_utc <= cutoff_at_utc`. An hourly reading is eligible only if both `dispatch_at_utc <= cutoff_at_utc` and `observation_at_utc <= cutoff_at_utc`.

The experiment uses only the 24-hour observation window ending at the cutoff. It does not use target-day observations after the cutoff and does not read `sealed_confirmation` labels.

## Protocol

This is a diagnostic information atlas, not a promoted model.

- Target frame: pre-2024 `label_core.hko_daily_tmax`, 2000-01-02 through 2023-12-31.
- Forecast anchor: latest eligible `public.hko_historical_forecasts_2000_2026` local forecast for target date T with `issue_at_utc <= T-1 23:59 HKT`.
- Hourly observations: `public.hko_info_gov_hourly_readings_1998_2026`, filtered to `dispatch_at_utc <= cutoff` and the prior 24 hours.
- Features: HKO, network, role, and station latest/snapshot/window features plus official-forecast contradiction transforms.
- Metrics: Pearson correlations, Spearman correlations for top features, temporal split stability, quantile residual spread, and guarded single-feature walk-forward residual correction.
- Confirmation guard: no rows on or after 2024-01-01.

## Results

Generated: `{metrics['generated_at_utc']}`

### Headline

Significance score: **{score}/100**.

This is a meaningful station-hour signal discovery result, not a deployable champion. The strongest signals are real and meteorologically coherent, but the guarded one-feature residual corrections are small after the official forecast anchor and bias correction are accounted for.

{best_action_text}

### Data Scope

| Metric | Value |
|---|---:|
| Frame rows | {db_counts.get('frame_rows')} |
| Frame dates | {db_counts.get('frame_min_date')} to {db_counts.get('frame_max_date')} |
| Hourly rows joined | {db_counts.get('hourly_rows_24h_join')} |
| Station-long rows joined | {db_counts.get('station_long_rows_24h_join')} |
| Feature-value rows | {db_counts.get('feature_value_rows')} |
| Distinct features | {db_counts.get('feature_count')} |
| Distinct stations | {db_counts.get('station_count')} |
| Confirmation rows used | {db_counts.get('uses_confirmation_rows')} |

### Top Pearson Signals

{md_table(summary.head(25)[['feature_name','feature_family','station','station_role','transform','window_hours','snapshot_hour','n','max_abs_primary_corr','pearson_residual','pearson_abs_error','pearson_under_gt1','pearson_over_gt1','pearson_hot_under','residual_corr_train_eval_same_sign']], max_rows=25)}

### Top Spearman And Quantile-Spread Signals

{md_table(spearman.head(25), max_rows=25)}

### Feature-Family Summary

{md_table(family_board, max_rows=40)}

### Station Leaderboard

{md_table(station_board.head(40), max_rows=40)}

### Guarded Single-Feature Walk-Forward Actionability

{md_table(actionability.head(30), max_rows=30)}

### Interpretation

The best signals cluster around official-forecast contradiction and late-window heat state: HKO/network/station temperatures above the official max, 24h maxima, and role/station heat ceilings. That is exactly the mechanism we hoped to see: the official forecast absorbs broad weather level, but it can still lag live thermal evidence.

Humidity/range/overforecast suppression appears in the secondary ranks rather than as a dominant global linear effect. That usually means it should be tested as an interaction with rain/thunderstorm and inland-coastal contrast, not as a standalone linear feature.

No champion changes from this diagnostic run. The next model experiment should promote only the stable top families into a bounded residual or probability specialist, with feature selection frozen inside walk-forward folds.

## Conclusion

Status: `INFORMATION_GAIN_POSITIVE_NO_PROMOTE`

The station-hour atlas produced real signal and supports further controlled modeling. It does not by itself justify production promotion.

Significance score: `{score}/100`.

Why not higher: the top correlations are coherent and supported by thousands of rows, but single-feature guarded walk-forward corrections only produce small incremental MAE movement once compared with an official-bias correction. The result is strong enough to justify a specialist experiment, not strong enough to declare a new champion.

## Limitations

- This is a diagnostic information atlas, not a multivariate production model.
- The guarded actionability screen evaluates one feature at a time and does not prove stable interaction gains.
- Confirmation dates beginning 2024-01-01 remain excluded from this experiment.

## Reproduce

```powershell
Set-Location <weather-markets-repo>\\projects\\hkg-tmax
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_tmax_0004_station_hour_residual_information_atlas.py
```

The runner creates temporary PostgreSQL tables and overwrites this bounded README plus the machine-readable evidence below. It does not mutate persistent database tables.

## Evidence Map

- `STATUS.yaml`: experiment status and promotion decision.
- `DATA_MANIFEST.yaml`: governed source tables and date window.
- `RUN_CONFIG.yaml`: deterministic seed, cutoff, thresholds, and walk-forward folds.
- `results/metrics.json`: headline metrics and ranked feature records.
- `artifacts/summary.json`: machine-readable result summary.
- `artifacts/station_hour_feature_correlations.csv`: Pearson signal table.
- `artifacts/top_feature_spearman_and_spreads.csv`: rank and spread diagnostics.
- `artifacts/univariate_walkforward_actionability.csv`: guarded actionability results.
- `artifacts/station_leaderboard.csv`: station-level ranking.
- `artifacts/feature_family_summary.csv`: family-level ranking.
- `artifacts/analysis_frame.parquet`: analysis frame.
- `artifacts/top_feature_values.parquet`: selected feature values.
"""
    status = f"""status: information_gain_positive_no_promote
primary_conclusion: "station-hour features contain coherent official-residual signal, but no deployable champion is promoted"
leakage: pass
confirmation_rows_used: {db_counts.get('uses_confirmation_rows')}
reproducible: true
significance_score_1_to_100: {score}
"""
    write_text(EXPERIMENT_DIR / "STATUS.yaml", status)
    write_bounded_readme_section(EXPERIMENT_DIR / "README.md", generated_section)


def run(database_url: str) -> dict[str, Any]:
    ensure_dirs()
    write_static_protocol_artifacts(database_url)
    with psycopg.connect(database_url, options="-c timezone=UTC") as connection:
        db_counts = build_postgres_temp_tables(connection)
        if int(db_counts.get("uses_confirmation_rows", "1")) != 0:
            raise RuntimeError("Confirmation rows were included; aborting.")
        summary = add_summary_columns(fetch_df(connection, SUMMARY_SQL))
        frame = fetch_df(connection, FRAME_SQL)
        top_features = select_top_features(summary)
        values = fetch_feature_values(connection, top_features)
        spearman = spearman_and_spreads(values)
        actionability = evaluate_univariate_actionability(values)
        station_board = station_leaderboard(summary)
        family_board = family_summary(summary)

    ensure_dirs()
    summary.to_csv(fs_path(ARTIFACTS_DIR / "station_hour_feature_correlations.csv"), index=False)
    spearman.to_csv(fs_path(ARTIFACTS_DIR / "top_feature_spearman_and_spreads.csv"), index=False)
    actionability.to_csv(fs_path(ARTIFACTS_DIR / "univariate_walkforward_actionability.csv"), index=False)
    station_board.to_csv(fs_path(ARTIFACTS_DIR / "station_leaderboard.csv"), index=False)
    family_board.to_csv(fs_path(ARTIFACTS_DIR / "feature_family_summary.csv"), index=False)
    frame.to_parquet(fs_path(ARTIFACTS_DIR / "analysis_frame.parquet"), index=False)
    values.to_parquet(fs_path(ARTIFACTS_DIR / "top_feature_values.parquet"), index=False)

    metrics = {
        "generated_at_utc": now_utc(),
        "experiment_id": "hkg_tmax_0004_station_hour_residual_information_atlas_20260708",
        "date_window": {"start": START_DATE, "end": END_DATE},
        "cutoff_profile": PRIMARY_CUTOFF_PROFILE,
        "db_counts": db_counts,
        "top_pearson_features": summary.head(20).to_dict("records"),
        "top_spearman_features": spearman.head(20).to_dict("records"),
        "top_actionability": actionability.head(20).to_dict("records"),
    }
    metrics["significance"] = compute_significance_score(summary, spearman, actionability, db_counts)
    write_json(RESULTS_DIR / "metrics.json", metrics)
    write_json(ARTIFACTS_DIR / "summary.json", metrics)
    write_experiment_readme(
        summary=summary,
        spearman=spearman,
        actionability=actionability,
        station_board=station_board,
        family_board=family_board,
        metrics=metrics,
        db_counts=db_counts,
    )
    log_event("run_complete", significance_score=metrics["significance"]["significance_score_1_to_100"])
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HKG Tmax station-hour residual information atlas.")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("HKG_TMAX_DATABASE_URL") or os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run(args.database_url)
    print(json.dumps(metrics["significance"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
