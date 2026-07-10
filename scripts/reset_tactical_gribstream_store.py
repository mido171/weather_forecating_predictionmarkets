from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hkg_tmax_db.connection import apply_migration, import_psycopg, redact_database_url


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"
TACTICAL_MIGRATION = REPO_ROOT / "migrations/postgres/20260625_0007_tactical_gribstream_h24n_schema.sql"
EXPERIMENT_ROOT = REPO_ROOT / "experiments/0214_tactical_h24n_gribstream_backfill"
SUMMARY_PATH = EXPERIMENT_ROOT / "legacy_purge_summary.json"
RAW_DIRS = (
    REPO_ROOT / "data/_pipeline_internal/raw/gribstream",
    REPO_ROOT / "data/raw/gribstream",
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def fs_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.lstrip("\\")
    return "\\\\?\\" + resolved


def ensure_directory(path: Path) -> None:
    os.makedirs(fs_path(path), exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def is_under(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    root_resolved = root.resolve()
    return resolved == root_resolved or root_resolved in resolved.parents


def table_exists(cursor: Any, table_name: str) -> bool:
    cursor.execute("SELECT to_regclass(%s)", (table_name,))
    return cursor.fetchone()[0] is not None


def scalar(cursor: Any, sql: str) -> int:
    cursor.execute(sql)
    value = cursor.fetchone()[0]
    return int(value or 0)


def collect_counts(cursor: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in (
        "raw_audit.acquisition_request",
        "raw_audit.response_object",
        "nwp_core.model_run",
        "nwp_core.point_value",
        "catalog.weather_model",
        "catalog.variable_selector_snapshot",
        "quarantine.rejected_payload",
        "nwp_tactical.acquisition_chunk",
        "nwp_tactical.raw_response_object",
        "nwp_tactical.forecast_wide",
    ):
        if table_exists(cursor, table):
            counts[table] = scalar(cursor, f"SELECT count(*) FROM {table}")
    if table_exists(cursor, "raw_audit.acquisition_request"):
        counts["raw_audit.acquisition_request.gribstream"] = scalar(
            cursor,
            "SELECT count(*) FROM raw_audit.acquisition_request WHERE provider = 'GribStream'",
        )
    if table_exists(cursor, "nwp_core.point_value"):
        counts["nwp_core.point_value.gribstream_linked"] = scalar(
            cursor,
            """
            SELECT count(*)
            FROM nwp_core.point_value pv
            JOIN raw_audit.response_object ro ON ro.response_object_id = pv.response_object_id
            JOIN raw_audit.acquisition_request ar ON ar.request_id = ro.request_id
            WHERE ar.provider = 'GribStream'
            """,
        )
    return counts


def purge_database(database_url: str, *, execute: bool) -> dict[str, Any]:
    psycopg = import_psycopg()
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            before = collect_counts(cursor)
            if not execute:
                connection.rollback()
                return {"before": before, "after": before, "executed": False}

            if table_exists(cursor, "nwp_tactical.forecast_wide"):
                cursor.execute("TRUNCATE nwp_tactical.forecast_wide RESTART IDENTITY CASCADE")
            if table_exists(cursor, "nwp_tactical.validation_issue"):
                cursor.execute("TRUNCATE nwp_tactical.validation_issue RESTART IDENTITY CASCADE")
            if table_exists(cursor, "nwp_tactical.raw_response_object"):
                cursor.execute("TRUNCATE nwp_tactical.raw_response_object RESTART IDENTITY CASCADE")
            if table_exists(cursor, "nwp_tactical.acquisition_chunk"):
                cursor.execute("TRUNCATE nwp_tactical.acquisition_chunk RESTART IDENTITY CASCADE")

            if table_exists(cursor, "raw_audit.acquisition_request"):
                cursor.execute(
                    """
                    CREATE TEMP TABLE _gribstream_request_ids AS
                    SELECT request_id
                    FROM raw_audit.acquisition_request
                    WHERE provider = 'GribStream'
                    """
                )
                cursor.execute(
                    """
                    CREATE TEMP TABLE _gribstream_response_ids AS
                    SELECT ro.response_object_id
                    FROM raw_audit.response_object ro
                    JOIN _gribstream_request_ids ids ON ids.request_id = ro.request_id
                    """
                )

                if table_exists(cursor, "quarantine.rejected_payload"):
                    cursor.execute(
                        """
                        DELETE FROM quarantine.rejected_payload q
                        USING _gribstream_request_ids ids
                        WHERE q.request_id = ids.request_id
                        """
                    )
                    cursor.execute(
                        """
                        DELETE FROM quarantine.rejected_payload q
                        USING _gribstream_response_ids ids
                        WHERE q.response_object_id = ids.response_object_id
                        """
                    )

                if table_exists(cursor, "nwp_core.point_value"):
                    cursor.execute(
                        """
                        DELETE FROM nwp_core.point_value pv
                        USING _gribstream_response_ids ids
                        WHERE pv.response_object_id = ids.response_object_id
                        """
                    )

                if table_exists(cursor, "raw_audit.response_object"):
                    cursor.execute(
                        """
                        DELETE FROM raw_audit.response_object ro
                        USING _gribstream_request_ids ids
                        WHERE ro.request_id = ids.request_id
                        """
                    )

                cursor.execute(
                    """
                    DELETE FROM raw_audit.acquisition_request ar
                    USING _gribstream_request_ids ids
                    WHERE ar.request_id = ids.request_id
                    """
                )

            if table_exists(cursor, "nwp_core.model_run") and table_exists(cursor, "catalog.weather_model"):
                cursor.execute(
                    """
                    DELETE FROM nwp_core.model_run mr
                    USING catalog.weather_model wm
                    WHERE mr.model_id = wm.model_id
                      AND wm.disposition LIKE 'T%_%'
                      AND NOT EXISTS (
                          SELECT 1 FROM nwp_core.point_value pv WHERE pv.model_run_id = mr.model_run_id
                      )
                    """
                )

            if table_exists(cursor, "catalog.variable_selector_snapshot") and table_exists(cursor, "catalog.weather_model"):
                cursor.execute(
                    """
                    DELETE FROM catalog.variable_selector_snapshot vss
                    USING catalog.weather_model wm
                    WHERE vss.model_id = wm.model_id
                      AND wm.disposition LIKE 'T%_%'
                      AND NOT EXISTS (
                          SELECT 1 FROM nwp_core.point_value pv WHERE pv.selector_id = vss.selector_id
                      )
                    """
                )

            if table_exists(cursor, "catalog.weather_model"):
                cursor.execute(
                    """
                    DELETE FROM catalog.weather_model wm
                    WHERE wm.disposition LIKE 'T%_%'
                      AND NOT EXISTS (
                          SELECT 1 FROM nwp_core.model_run mr WHERE mr.model_id = wm.model_id
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM catalog.variable_selector_snapshot vss WHERE vss.model_id = wm.model_id
                      )
                    """
                )

            after = collect_counts(cursor)
        connection.commit()
    return {"before": before, "after": after, "executed": True}


def purge_raw_dirs(*, execute: bool) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in RAW_DIRS:
        if not is_under(path, REPO_ROOT / "data"):
            raise RuntimeError(f"Refusing to purge outside data directory: {path}")
        exists = path.exists()
        file_count = 0
        byte_count = 0
        if exists:
            for dirpath, _dirnames, filenames in os.walk(fs_path(path)):
                for filename in filenames:
                    file_count += 1
                    try:
                        byte_count += os.path.getsize(os.path.join(dirpath, filename))
                    except OSError:
                        pass
        if execute and exists:
            shutil.rmtree(fs_path(path))
            ensure_directory(path)
        results.append(
            {
                "path": str(path),
                "existed": exists,
                "files_before": file_count,
                "bytes_before": byte_count,
                "executed": execute,
            }
        )
    return results


def run(args: argparse.Namespace) -> int:
    ensure_directory(EXPERIMENT_ROOT)
    if args.apply_schema:
        apply_migration(args.database_url, TACTICAL_MIGRATION)
    db_result = purge_database(args.database_url, execute=args.execute)
    raw_result = purge_raw_dirs(execute=args.execute and args.purge_raw)
    summary = {
        "status": "executed" if args.execute else "dry_run",
        "updated_at_utc": utc_now_iso(),
        "database": redact_database_url(args.database_url),
        "applied_schema": bool(args.apply_schema),
        "purged_raw": bool(args.execute and args.purge_raw),
        "database_purge": db_result,
        "raw_purge": raw_result,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset legacy GribStream rows and prepare tactical H24N schema.")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    parser.add_argument("--apply-schema", action="store_true")
    parser.add_argument("--purge-raw", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually delete legacy GribStream rows/raw files.")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
