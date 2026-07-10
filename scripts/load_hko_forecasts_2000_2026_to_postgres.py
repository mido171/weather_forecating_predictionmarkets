"""Load the HKO official press forecast archive into one Postgres table.

This script intentionally collapses the HKO forecast archive into one canonical
table instead of preserving the previous split export tables. It reads the live
SQLite archive at C:\\hko_press_2000_2026 and streams rows to Postgres through
psql COPY, so it does not need an additional Python Postgres driver.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE = Path(r"C:\hko_press_2000_2026\metadata\archive.sqlite3")
DEFAULT_PSQL = Path(r"C:\Program Files\PostgreSQL\16\bin\psql.exe")
DEFAULT_TABLE = "public.hko_historical_forecasts_2000_2026"
HKT = ZoneInfo("Asia/Hong_Kong")

SOURCE_COLUMNS = [
    "bulletin_id",
    "source",
    "source_url",
    "product_type",
    "title",
    "index_date",
    "snapshot_at_hkt",
    "issue_at_hkt",
    "issue_parse_method",
    "raw_sha256",
    "raw_path",
    "text",
    "target_date",
    "target_date_confidence",
    "forecast_min_c",
    "forecast_max_c",
    "temperature_text",
    "stale_snapshot_flag",
    "stale_hours",
    "parse_status",
    "parse_notes",
]

COPY_COLUMNS = [
    "bulletin_id",
    "source",
    "source_url",
    "product_type",
    "title",
    "index_date",
    "snapshot_at_hkt",
    "snapshot_at_utc",
    "issue_at_hkt",
    "issue_at_utc",
    "issue_parse_method",
    "target_date",
    "target_issue_lead_days",
    "target_date_confidence",
    "forecast_min_c",
    "forecast_max_c",
    "forecast_range_c",
    "forecast_midpoint_c",
    "has_target_date",
    "has_forecast_min",
    "has_forecast_max",
    "has_forecast_minmax",
    "temperature_valid",
    "usable_local_tmax_forecast",
    "row_quality_status",
    "temperature_text",
    "stale_snapshot_flag",
    "stale_hours",
    "parse_status",
    "parse_notes",
    "full_text",
    "raw_sha256",
    "raw_path",
    "source_archive_path",
    "source_archive_mtime_utc",
    "ingested_at_utc",
]

OLD_HKO_FORECAST_OBJECTS = [
    "DROP VIEW IF EXISTS feature_safe.hko_t24_official_anchor CASCADE;",
    "DROP TABLE IF EXISTS operational_anchor.hko_t24_official_anchor_rows CASCADE;",
    "DROP TABLE IF EXISTS operational_anchor.codex_audit_ds_05_hko_historical_rss_forecasts_hko_his_bedb970a CASCADE;",
    "DROP TABLE IF EXISTS operational_archive_raw.codex_audit_ds_05_hko_historical_rss_forecasts_hko_his_ad6e1592 CASCADE;",
    "DROP TABLE IF EXISTS operational_archive_raw.codex_audit_ds_05_hko_historical_rss_forecasts_hko_pre_e8bb96fa CASCADE;",
    "DROP TABLE IF EXISTS operational_archive_normalized.codex_audit_ds_05_hko_historical_rss_forecasts_hko_pre_8b9efa12 CASCADE;",
    "DROP TABLE IF EXISTS operational_archive_normalized.codex_audit_ds_05_hko_historical_rss_forecasts_hko_pre_db0d0932 CASCADE;",
    "DROP TABLE IF EXISTS research_supervised.codex_audit_ds_05_hko_historical_rss_forecasts_hko_off_40ef37f0 CASCADE;",
    "DROP TABLE IF EXISTS acquisition_quality.codex_audit_ds_05_hko_historical_rss_forecasts_hko_pre_8461ed7a CASCADE;",
    "DROP TABLE IF EXISTS acquisition_quality.codex_audit_ds_05_hko_historical_rss_forecasts_hko_pre_9f83c23e CASCADE;",
    "DROP TABLE IF EXISTS quality_monitoring.codex_audit_ds_05_hko_historical_rss_forecasts_hko_pre_a1f818d5 CASCADE;",
]


def log(event: str, **fields: object) -> None:
    payload = {"event": event, "ts": datetime.now(UTC).replace(microsecond=0).isoformat()}
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True), flush=True)


def find_psql(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(f"psql not found at {path}")
    if DEFAULT_PSQL.exists():
        return DEFAULT_PSQL
    discovered = shutil.which("psql")
    if discovered:
        return Path(discovered)
    raise FileNotFoundError("Could not find psql.exe")


def split_table_name(table: str) -> tuple[str, str]:
    parts = table.split(".")
    if len(parts) != 2 or not all(parts):
        raise ValueError("table must be schema.table")
    return parts[0], parts[1]


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def qualified_table(schema: str, table: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(table)}"


def psql_base(args: argparse.Namespace) -> list[str]:
    return [
        str(args.psql_path),
        "-h",
        args.pg_host,
        "-p",
        str(args.pg_port),
        "-U",
        args.pg_user,
        "-d",
        args.pg_database,
        "-v",
        "ON_ERROR_STOP=1",
        "-X",
        "-q",
    ]


def psql_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PGPASSWORD"] = args.pg_password
    env["PGCLIENTENCODING"] = "UTF8"
    return env


def run_psql(args: argparse.Namespace, sql: str) -> str:
    proc = subprocess.run(
        [*psql_base(args), "-f", "-"],
        input=sql,
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=psql_env(args),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"psql failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout


def copy_with_psql(args: argparse.Namespace, fq_stage: str, sqlite_path: Path) -> int:
    source_mtime = datetime.fromtimestamp(sqlite_path.stat().st_mtime, UTC).isoformat()
    ingested_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    copy_sql = (
        f"COPY {fq_stage} ({', '.join(quote_ident(col) for col in COPY_COLUMNS)}) "
        "FROM STDIN WITH (FORMAT csv, HEADER true, NULL '');"
    )
    proc = subprocess.Popen(
        [*psql_base(args), "-c", copy_sql],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=psql_env(args),
    )
    assert proc.stdin is not None
    writer = csv.writer(proc.stdin, lineterminator="\n")
    writer.writerow(COPY_COLUMNS)

    con = sqlite3.connect(sqlite_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    sql = f"select {', '.join(SOURCE_COLUMNS)} from bulletins order by index_date, product_type, issue_at_hkt, source_url"
    count = 0
    try:
        for row in cur.execute(sql):
            writer.writerow(to_copy_row(row, sqlite_path, source_mtime, ingested_at))
            count += 1
            if count % args.progress_interval_rows == 0:
                log("copy_progress", rows=count)
    finally:
        con.close()
        proc.stdin.close()
        proc.stdin = None

    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"psql COPY failed after {count} rows\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
    return count


def parse_date(value: object) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def parse_hkt_timestamp(value: object) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, None
    try:
        normalized = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=HKT)
        local_hkt = dt.astimezone(HKT).replace(tzinfo=None)
        utc = dt.astimezone(UTC)
        return local_hkt.isoformat(sep=" "), utc.isoformat()
    except ValueError:
        return None, None


def as_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def pg_bool(value: bool) -> str:
    return "true" if value else "false"


def pg_value(value: object) -> object:
    if value is None:
        return ""
    return value


def row_quality_status(
    product_type: str,
    parse_status: str,
    target_date_value: date | None,
    target_issue_lead_days: int | None,
    forecast_min: float | None,
    forecast_max: float | None,
    temperature_valid: bool,
) -> str:
    if parse_status not in {"ok", "partial"}:
        return "parse_not_ok"
    if product_type != "local":
        return "bulletin_only_multiday_product"
    if target_date_value is None:
        return "missing_target_date"
    if target_issue_lead_days not in {0, 1}:
        return "invalid_target_lead"
    if forecast_max is None:
        return "missing_forecast_max"
    if not temperature_valid:
        return "invalid_temperature_range"
    if forecast_min is None:
        return "usable_local_tmax_only"
    return "usable_local_minmax"


def to_copy_row(row: sqlite3.Row, sqlite_path: Path, source_mtime: str, ingested_at: str) -> list[object]:
    index_date = parse_date(row["index_date"])
    target_date = parse_date(row["target_date"])
    snapshot_hkt, snapshot_utc = parse_hkt_timestamp(row["snapshot_at_hkt"])
    issue_hkt, issue_utc = parse_hkt_timestamp(row["issue_at_hkt"])
    forecast_min = as_float(row["forecast_min_c"])
    forecast_max = as_float(row["forecast_max_c"])
    forecast_range = None
    forecast_midpoint = None
    if forecast_min is not None and forecast_max is not None:
        forecast_range = forecast_max - forecast_min
        forecast_midpoint = (forecast_max + forecast_min) / 2.0
    target_issue_lead_days = None
    if index_date is not None and target_date is not None:
        target_issue_lead_days = (target_date - index_date).days
    has_target_date = target_date is not None
    has_forecast_min = forecast_min is not None
    has_forecast_max = forecast_max is not None
    has_forecast_minmax = has_forecast_min and has_forecast_max
    temperature_valid = (
        (forecast_min is None or -20.0 <= forecast_min <= 60.0)
        and (forecast_max is None or -20.0 <= forecast_max <= 60.0)
        and (forecast_min is None or forecast_max is None or forecast_min <= forecast_max)
    )
    product_type = str(row["product_type"])
    parse_status = str(row["parse_status"])
    quality_status = row_quality_status(
        product_type,
        parse_status,
        target_date,
        target_issue_lead_days,
        forecast_min,
        forecast_max,
        temperature_valid,
    )
    usable_local_tmax = quality_status in {"usable_local_tmax_only", "usable_local_minmax"}
    return [
        row["bulletin_id"],
        row["source"],
        row["source_url"],
        product_type,
        row["title"],
        index_date.isoformat() if index_date else None,
        snapshot_hkt,
        snapshot_utc,
        issue_hkt,
        issue_utc,
        row["issue_parse_method"],
        target_date.isoformat() if target_date else None,
        target_issue_lead_days,
        row["target_date_confidence"],
        forecast_min,
        forecast_max,
        forecast_range,
        forecast_midpoint,
        pg_bool(has_target_date),
        pg_bool(has_forecast_min),
        pg_bool(has_forecast_max),
        pg_bool(has_forecast_minmax),
        pg_bool(temperature_valid),
        pg_bool(usable_local_tmax),
        quality_status,
        row["temperature_text"],
        pg_bool(bool(row["stale_snapshot_flag"])),
        as_float(row["stale_hours"]),
        parse_status,
        row["parse_notes"],
        row["text"],
        row["raw_sha256"],
        row["raw_path"],
        str(sqlite_path),
        source_mtime,
        ingested_at,
    ]


def validate_sqlite(sqlite_path: Path) -> int:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite archive not found: {sqlite_path}")
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    try:
        table_exists = cur.execute(
            "select 1 from sqlite_master where type='table' and name='bulletins'"
        ).fetchone()
        if not table_exists:
            raise RuntimeError("SQLite archive does not contain bulletins table")
        observed_columns = [row[1] for row in cur.execute("pragma table_info(bulletins)").fetchall()]
        if observed_columns != SOURCE_COLUMNS:
            raise RuntimeError(f"Unexpected bulletins schema: {observed_columns}")
        count = int(cur.execute("select count(*) from bulletins").fetchone()[0])
        if count <= 0:
            raise RuntimeError("SQLite bulletins table is empty")
        return count
    finally:
        con.close()


def create_stage_sql(fq_stage: str) -> str:
    return f"""
DROP TABLE IF EXISTS {fq_stage};
CREATE TABLE {fq_stage} (
    bulletin_id text PRIMARY KEY,
    source text NOT NULL,
    source_url text NOT NULL,
    product_type text NOT NULL,
    title text,
    index_date date,
    snapshot_at_hkt timestamp without time zone,
    snapshot_at_utc timestamp with time zone,
    issue_at_hkt timestamp without time zone,
    issue_at_utc timestamp with time zone,
    issue_parse_method text,
    target_date date,
    target_issue_lead_days integer,
    target_date_confidence text,
    forecast_min_c double precision,
    forecast_max_c double precision,
    forecast_range_c double precision,
    forecast_midpoint_c double precision,
    has_target_date boolean NOT NULL,
    has_forecast_min boolean NOT NULL,
    has_forecast_max boolean NOT NULL,
    has_forecast_minmax boolean NOT NULL,
    temperature_valid boolean NOT NULL,
    usable_local_tmax_forecast boolean NOT NULL,
    row_quality_status text NOT NULL,
    temperature_text text,
    stale_snapshot_flag boolean NOT NULL,
    stale_hours double precision,
    parse_status text NOT NULL,
    parse_notes text,
    full_text text NOT NULL,
    raw_sha256 text NOT NULL,
    raw_path text NOT NULL,
    source_archive_path text NOT NULL,
    source_archive_mtime_utc timestamp with time zone NOT NULL,
    ingested_at_utc timestamp with time zone NOT NULL,
    CHECK (product_type in ('local', '5day', '7day', '9day')),
    CHECK (
        (forecast_min_c is null or forecast_min_c between -20 and 60)
        and (forecast_max_c is null or forecast_max_c between -20 and 60)
        and (forecast_min_c is null or forecast_max_c is null or forecast_min_c <= forecast_max_c)
    )
);
"""


def finalize_sql(fq_stage: str, schema: str, table: str, drop_old: bool) -> str:
    fq_final = qualified_table(schema, table)
    old_drop_sql = "\n".join(OLD_HKO_FORECAST_OBJECTS) if drop_old else ""
    return f"""
BEGIN;
{old_drop_sql}
DROP TABLE IF EXISTS {fq_final} CASCADE;
ALTER TABLE {fq_stage} RENAME TO {quote_ident(table)};
CREATE INDEX hko_forecasts_2000_2026_product_issue_idx
    ON {fq_final} (product_type, issue_at_utc);
CREATE INDEX hko_forecasts_2000_2026_target_idx
    ON {fq_final} (target_date)
    WHERE target_date IS NOT NULL;
CREATE INDEX hko_forecasts_2000_2026_local_usable_idx
    ON {fq_final} (target_date, issue_at_utc)
    WHERE usable_local_tmax_forecast;
CREATE INDEX hko_forecasts_2000_2026_raw_sha_idx
    ON {fq_final} (raw_sha256);
COMMENT ON TABLE {fq_final} IS
    'Canonical one-table HKO official press weather forecast archive loaded from C:\\hko_press_2000_2026\\metadata\\archive.sqlite3.';
COMMIT;
ANALYZE {fq_final};
"""


def fetch_scalar(args: argparse.Namespace, sql: str) -> str:
    proc = subprocess.run(
        [*psql_base(args), "-A", "-t", "-c", sql],
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=psql_env(args),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"psql scalar failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout.strip()


def write_summary(args: argparse.Namespace, fq_final: str, source_count: int, copied_count: int) -> Path:
    summary_sql = f"""
select jsonb_build_object(
    'table', '{fq_final}',
    'total_rows', count(*),
    'source_rows', {source_count},
    'copied_rows', {copied_count},
    'product_counts', (
        select jsonb_object_agg(product_type, rows)
        from (
            select product_type, count(*) rows
            from {fq_final}
            group by product_type
            order by product_type
        ) x
    ),
    'index_date_min', min(index_date),
    'index_date_max', max(index_date),
    'target_date_min', min(target_date),
    'target_date_max', max(target_date),
    'usable_local_tmax_rows', count(*) filter (where usable_local_tmax_forecast),
    'usable_local_tmax_target_dates', count(distinct target_date) filter (where usable_local_tmax_forecast),
    'row_quality_counts', (
        select jsonb_object_agg(row_quality_status, rows)
        from (
            select row_quality_status, count(*) rows
            from {fq_final}
            group by row_quality_status
            order by row_quality_status
        ) q
    )
)::text
from {fq_final};
"""
    summary = json.loads(fetch_scalar(args, summary_sql))
    out_path = PROJECT_ROOT / "run_logs" / "hko_historical_forecasts_2000_2026_load_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite-path", default=str(DEFAULT_SQLITE))
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--psql-path")
    parser.add_argument("--pg-host", default="127.0.0.1")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--pg-password", default="root")
    parser.add_argument("--pg-database", default="hkg_tmax_research")
    parser.add_argument("--progress-interval-rows", type=int, default=25000)
    parser.add_argument(
        "--keep-old-split-objects",
        action="store_true",
        help="Keep previous split HKO forecast tables/views instead of dropping them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.psql_path = find_psql(args.psql_path)
    sqlite_path = Path(args.sqlite_path)
    schema, table = split_table_name(args.table)
    stage_table = f"{table}__stage"
    fq_stage = qualified_table(schema, stage_table)
    fq_final = qualified_table(schema, table)

    log(
        "run_start",
        sqlite_path=str(sqlite_path),
        pg_database=args.pg_database,
        table=args.table,
        drop_old_split_objects=not args.keep_old_split_objects,
    )
    source_count = validate_sqlite(sqlite_path)
    log("sqlite_validated", source_rows=source_count)

    run_psql(args, f"CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)};\n{create_stage_sql(fq_stage)}")
    log("stage_created", stage_table=fq_stage)

    copied_count = copy_with_psql(args, fq_stage, sqlite_path)
    log("copy_complete", copied_rows=copied_count)
    if copied_count != source_count:
        raise RuntimeError(f"Copied {copied_count} rows, expected {source_count}")

    stage_count = int(fetch_scalar(args, f"select count(*) from {fq_stage};"))
    if stage_count != source_count:
        raise RuntimeError(f"Stage has {stage_count} rows, expected {source_count}")
    log("stage_verified", stage_rows=stage_count)

    run_psql(args, finalize_sql(fq_stage, schema, table, not args.keep_old_split_objects))
    log("finalized", table=fq_final)

    final_count = int(fetch_scalar(args, f"select count(*) from {fq_final};"))
    if final_count != source_count:
        raise RuntimeError(f"Final table has {final_count} rows, expected {source_count}")
    summary_path = write_summary(args, fq_final, source_count, copied_count)
    log("run_complete", final_rows=final_count, summary_path=str(summary_path))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log("run_failed", error=str(exc))
        raise
