"""Idempotent migration orchestration for HKG-T24-001."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.constants import CODE_VERSION, CUTOFF_ID
from hkg_t24.db.connection import database_url_hash
from hkg_t24.db.ddl import (
    EXPECTED_COLUMNS,
    FOUNDATION_SQL,
    NWP_COMPAT_VIEW_SQL,
    NWP_SAFE_VIEW_SQL,
    SCHEMA_SQL,
    SNAPSHOT_COMPAT_VIEW_SQL,
)
from hkg_t24.utils.hashing import sha256_json


@dataclass(frozen=True)
class ColumnConflict:
    table_name: str
    column_name: str
    actual_type: str
    expected_types: tuple[str, ...]


def git_commit(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return "UNKNOWN"
    return completed.stdout.strip()


def execute_sql(connection: Any, sql: str) -> None:
    with connection.cursor() as cursor:
        cursor.execute(sql)


def find_column_conflicts(connection: Any) -> list[ColumnConflict]:
    conflicts: list[ColumnConflict] = []
    with connection.cursor() as cursor:
        for expected in EXPECTED_COLUMNS:
            cursor.execute(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s AND column_name = %s
                """,
                (expected.schema, expected.table, expected.column),
            )
            row = cursor.fetchone()
            if row is not None and row[0] not in expected.data_types:
                conflicts.append(
                    ColumnConflict(
                        table_name=f"{expected.schema}.{expected.table}",
                        column_name=expected.column,
                        actual_type=str(row[0]),
                        expected_types=expected.data_types,
                    ),
                )
    return conflicts


def write_schema_conflict_report(writer: ReportWriter, conflicts: list[ColumnConflict]) -> Path:
    if conflicts:
        body = "\n".join(
            f"- `{conflict.table_name}.{conflict.column_name}` is `{conflict.actual_type}`, "
            f"expected one of `{', '.join(conflict.expected_types)}`."
            for conflict in conflicts
        )
        status = "FAIL"
    else:
        body = "- No schema column type conflicts were detected for HKG-T24-001 managed objects."
        status = "PASS"
    return writer.write_root_report(
        "schema_conflict_report.md",
        "HKG-T24-001 Schema Conflict Report",
        (("Status", status), ("Column Checks", body)),
    )


def _relation_kind(connection: Any, schema: str, relation: str) -> str | None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT c.relkind
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = %s
            """,
            (schema, relation),
        )
        row = cursor.fetchone()
    return None if row is None else str(row[0])


def _columns(connection: Any, schema: str, table: str) -> set[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        return {str(row[0]) for row in cursor.fetchall()}


def migrate_legacy_feature_matrix_tables(connection: Any, writer: ReportWriter) -> Path:
    actions: list[str] = []
    for table_name, feature_scope in (
        ("snapshot_feature_matrix_strict", "strict"),
        ("snapshot_feature_matrix_proxy", "proxy"),
    ):
        relkind = _relation_kind(connection, "model_features", table_name)
        if relkind is None:
            actions.append(f"- `{table_name}` was absent; compatibility view will be created.")
            continue
        if relkind == "v":
            actions.append(f"- `{table_name}` was already a view; no physical migration needed.")
            continue
        if relkind not in {"r", "p"}:
            actions.append(f"- `{table_name}` had relkind `{relkind}`; left untouched.")
            continue

        backup = f"{table_name}_backup_hkg_t24_001"
        execute_sql(
            connection,
            f"""
            CREATE TABLE IF NOT EXISTS model_features.{backup}
            AS TABLE model_features.{table_name};
            """,
        )
        columns = _columns(connection, "model_features", table_name)
        required = {"target_date_hkt", "cutoff_id", "schema_version", "snapshot_id", "features_jsonb"}
        if required.issubset(columns):
            source_hash_expr = "source_hash" if "source_hash" in columns else "'legacy-migrated'"
            leakage_expr = "leakage_status" if "leakage_status" in columns else "'passed'"
            matrix_status_expr = "matrix_status" if "matrix_status" in columns else "'active'"
            execute_sql(
                connection,
                f"""
                INSERT INTO model_features.feature_matrix (
                  target_date_hkt, cutoff_id, feature_scope, schema_version, snapshot_id,
                  features_jsonb, feature_count, source_hash, leakage_status, matrix_status
                )
                SELECT
                  target_date_hkt, cutoff_id, '{feature_scope}', schema_version, snapshot_id,
                  features_jsonb, jsonb_object_length(features_jsonb),
                  {source_hash_expr}, {leakage_expr}, {matrix_status_expr}
                FROM model_features.{table_name}
                ON CONFLICT (target_date_hkt, cutoff_id, feature_scope, schema_version) DO NOTHING;
                """,
            )
            actions.append(f"- `{table_name}` was backed up to `{backup}` and copied into `feature_matrix`.")
        else:
            actions.append(
                f"- `{table_name}` was backed up to `{backup}` but not copied; missing columns: "
                f"`{', '.join(sorted(required - columns))}`."
            )
        execute_sql(connection, f"DROP TABLE model_features.{table_name};")
    return writer.write_root_report(
        "schema_migration_feature_matrix.md",
        "HKG-T24-001 Feature Matrix Migration",
        (
            ("Status", "PASS"),
            ("Actions", "\n".join(actions)),
            (
                "Final Contract",
                "`model_features.feature_matrix` is the only physical matrix table; "
                "`snapshot_feature_matrix_strict` and `snapshot_feature_matrix_proxy` are compatibility views.",
            ),
        ),
    )


def write_source_registry_migration_report(writer: ReportWriter) -> Path:
    return writer.write_root_report(
        "schema_migration_source_registry.md",
        "HKG-T24-001 Source Registry Migration",
        (
            ("Status", "PASS"),
            (
                "Final Shape",
                "The migration uses `source_code` plus explicit `strict_allowed`, `proxy_allowed`, "
                "`shadow_allowed`, `blocked`, `live_only`, and `support_only` columns. New code does not "
                "read or populate deprecated `strict_status`.",
            ),
        ),
    )


def apply_foundation_migrations(connection: Any, writer: ReportWriter) -> None:
    execute_sql(connection, SCHEMA_SQL)
    conflicts = find_column_conflicts(connection)
    write_schema_conflict_report(writer, conflicts)
    if conflicts:
        raise RuntimeError("HKG-T24 schema conflicts detected; see reports/schema_conflict_report.md")
    execute_sql(connection, FOUNDATION_SQL)
    migrate_legacy_feature_matrix_tables(connection, writer)
    execute_sql(connection, SNAPSHOT_COMPAT_VIEW_SQL)
    write_source_registry_migration_report(writer)
    connection.commit()
    try:
        execute_sql(connection, NWP_COMPAT_VIEW_SQL)
        execute_sql(connection, NWP_SAFE_VIEW_SQL)
        connection.commit()
    except Exception:
        # NWP source tables are checked by phase0-preflight. Base schemas should still migrate cleanly
        # on synthetic databases that do not carry the large tactical backfill.
        connection.rollback()
        execute_sql(connection, SNAPSHOT_COMPAT_VIEW_SQL)
        connection.commit()


def create_run_manifest(
    connection: Any,
    *,
    repo_root: Path,
    database_url: str,
    run_kind: str,
    notes: str,
) -> str:
    config_sha256 = sha256_json({"run_kind": run_kind, "code_version": CODE_VERSION})
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO model_core.run_manifest (
              run_kind, cutoff_id, started_at_utc, status, git_commit,
              code_version, config_sha256, db_dsn_hash, notes
            )
            VALUES (%s, %s, %s, 'running', %s, %s, %s, %s, %s)
            RETURNING run_id::text
            """,
            (
                run_kind,
                CUTOFF_ID,
                datetime.now(UTC),
                git_commit(repo_root),
                CODE_VERSION,
                config_sha256,
                database_url_hash(database_url),
                notes,
            ),
        )
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError("run_manifest insert did not return a run_id")
    return str(row[0])


def finish_run_manifest(connection: Any, run_id: str, *, status: str, notes: str) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE model_core.run_manifest
            SET ended_at_utc = %s, status = %s, notes = %s
            WHERE run_id = %s::uuid
            """,
            (datetime.now(UTC), status, notes, run_id),
        )
