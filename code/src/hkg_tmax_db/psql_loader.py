"""psql-backed governed corpus loader.

This module intentionally uses the installed PostgreSQL client binary instead
of a Python DB driver. That keeps the ingestion executable on Windows machines
where PostgreSQL is installed but the project virtualenv has not yet been
resynced with psycopg.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import AuditBundle, load_contract_tables
from .cutoff import CUTOFF_RULE_VERSION
from .hashing import sha256_file
from .reconciliation import (
    OBJECT_ACTIONS,
    SourceReconciliation,
    sanitize_identifier,
    table_name_for_source,
)

CONTRACT_VERSION = "audit_bundle_2026_06_23_v1"
LOADER_VERSION = "hkg_tmax_db_psql_loader_20260623_v1"
GENERATED_TABLE_PREFIX = "codex_audit_"
ROW_CHUNK_SIZE = 100_000
MAX_IDENTIFIER_LENGTH = 63


@dataclass(frozen=True)
class PsqlConfig:
    psql_path: Path
    host: str
    port: int
    admin_user: str
    admin_password: str
    database: str


@dataclass(frozen=True)
class LoadResult:
    batch_id: str
    rows_loaded_by_layer: dict[str, int]
    rows_quarantined: int
    duplicate_formats_skipped: int
    objects_registered: int
    files_succeeded: int
    files_skipped: int
    sealed_confirmation_enforced: bool
    live_role_label_access_denied: bool
    strict_validation_passed: bool
    idempotency_signature: dict[str, int]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def emit_event(event: str, message: str, **fields: Any) -> None:
    payload = {
        "ts": utc_now_iso(),
        "level": "INFO",
        "event": event,
        "message": message,
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def find_psql() -> Path:
    found = shutil.which("psql")
    if found:
        return Path(found)
    candidates = [
        Path("C:/Program Files/PostgreSQL/16/bin/psql.exe"),
        Path("C:/Program Files/PostgreSQL/15/bin/psql.exe"),
        Path("C:/Program Files/PostgreSQL/14/bin/psql.exe"),
        Path("C:/Program Files/PostgreSQL/16/pgAdmin 4/runtime/psql.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find psql.exe")


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def parse_optional_timestamptz(value: str | None) -> str | None:
    if not value:
        return None
    return value


def run_psql(
    config: PsqlConfig,
    sql: str,
    *,
    database: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PGPASSWORD"] = config.admin_password
    command = [
        str(config.psql_path),
        "-h",
        config.host,
        "-p",
        str(config.port),
        "-U",
        config.admin_user,
        "-d",
        database or config.database,
        "-v",
        "ON_ERROR_STOP=1",
        "-t",
        "-A",
        "-X",
        "-q",
        "-c",
        sql,
    ]
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"psql failed with exit {completed.returncode}\nSQL:\n{sql[:4000]}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
        )
    return completed


def run_psql_file(config: PsqlConfig, sql_path: Path, *, database: str | None = None) -> None:
    env = os.environ.copy()
    env["PGPASSWORD"] = config.admin_password
    command = [
        str(config.psql_path),
        "-h",
        config.host,
        "-p",
        str(config.port),
        "-U",
        config.admin_user,
        "-d",
        database or config.database,
        "-v",
        "ON_ERROR_STOP=1",
        "-X",
        "-q",
        "-f",
        str(sql_path),
    ]
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"psql file failed with exit {completed.returncode}: {sql_path}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
        )


def ensure_database(config: PsqlConfig) -> None:
    emit_event("database_check", "Checking target PostgreSQL database", database=config.database)
    exists = run_psql(
        config,
        f"SELECT 1 FROM pg_database WHERE datname = {sql_literal(config.database)};",
        database="postgres",
    )
    if "1" not in exists.stdout:
        emit_event("database_create", "Creating target PostgreSQL database", database=config.database)
        run_psql(config, f"CREATE DATABASE {quote_ident(config.database)};", database="postgres")
    else:
        emit_event("database_exists", "Target PostgreSQL database already exists", database=config.database)


def psql_copy(config: PsqlConfig, table: str, columns: Sequence[str], csv_path: Path) -> None:
    column_sql = ", ".join(quote_ident(column) for column in columns)
    source = csv_path.resolve().as_posix().replace("'", "''")
    sql = (
        f"\\copy {table} ({column_sql}) FROM '{source}' "
        "WITH (FORMAT csv, HEADER true, NULL '')"
    )
    run_psql(config, sql)


def clean_column_names(columns: Iterable[str]) -> list[str]:
    used: set[str] = set()
    cleaned: list[str] = []
    for index, column in enumerate(columns, start=1):
        base = sanitize_identifier(str(column), fallback=f"col_{index}")
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base[:58]}_{suffix}"
            suffix += 1
        used.add(candidate)
        cleaned.append(candidate)
    return cleaned


def postgres_type_for_arrow(arrow_type: pa.DataType) -> str:
    if pa.types.is_boolean(arrow_type):
        return "boolean"
    if pa.types.is_integer(arrow_type):
        return "bigint"
    if pa.types.is_floating(arrow_type):
        return "double precision"
    if pa.types.is_date(arrow_type):
        return "date"
    if pa.types.is_timestamp(arrow_type):
        return "timestamptz"
    return "text"


def postgres_type_for_csv_column(name: str, sample: pd.Series) -> str:
    non_null = sample.dropna()
    if non_null.empty:
        return "text"
    lowered = name.lower()
    if pd.api.types.is_bool_dtype(non_null) or set(non_null.astype(str).str.lower().unique()) <= {"true", "false"}:
        return "boolean"
    if lowered == "date" or lowered.endswith("_date"):
        parsed = pd.to_datetime(non_null.astype(str), errors="coerce", utc=True)
        if parsed.notna().all():
            return "date"
    if lowered.endswith("_utc") or lowered.endswith("_hkt") or lowered.endswith("_at"):
        parsed = pd.to_datetime(non_null.astype(str), errors="coerce", utc=True)
        if parsed.notna().all():
            return "timestamptz"
    if pd.api.types.is_integer_dtype(non_null):
        return "bigint"
    if pd.api.types.is_float_dtype(non_null):
        return "double precision"
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().all():
        if (numeric % 1 == 0).all():
            return "bigint"
        return "double precision"
    return "text"


def dataframe_for_copy(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        series = output[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            output[column] = pd.to_datetime(series, errors="coerce", utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ",
            )
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            output[column] = series.map(format_cell)
    return output


def format_cell(value: Any) -> Any:
    if pd.isna(value):
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_rows_csv(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def bounded_identifier(identifier: str, *, fallback: str = "identifier") -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", identifier).strip("_").lower()
    if not clean:
        clean = fallback
    if clean[0].isdigit():
        clean = f"{fallback}_{clean}"
    if len(clean) <= MAX_IDENTIFIER_LENGTH:
        return clean
    digest = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:8]
    return f"{clean[: MAX_IDENTIFIER_LENGTH - 9]}_{digest}"


def staging_table_name(table: str) -> str:
    return "ingestion." + quote_ident(
        bounded_identifier(f"_stage_{table.replace('.', '_')}", fallback="stage"),
    )


def create_staging_and_upsert(
    config: PsqlConfig,
    *,
    table: str,
    columns: Sequence[tuple[str, str]],
    csv_path: Path,
    conflict_targets: Sequence[str],
) -> None:
    staging = staging_table_name(table)
    column_defs = ", ".join(f"{quote_ident(name)} {kind}" for name, kind in columns)
    column_names = [name for name, _ in columns]
    conflict_set = set(conflict_targets)
    run_psql(
        config,
        f"""
        DROP TABLE IF EXISTS {staging};
        CREATE UNLOGGED TABLE {staging} ({column_defs});
        """,
    )
    try:
        psql_copy(config, staging, column_names, csv_path)
        insert_cols = ", ".join(quote_ident(name) for name in column_names)
        updates = ", ".join(
            f"{quote_ident(name)} = EXCLUDED.{quote_ident(name)}"
            for name in column_names
            if name not in conflict_set
        )
        conflict_sql = ", ".join(quote_ident(name) for name in conflict_targets)
        action = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
        sql = (
            f"INSERT INTO {table} ({insert_cols}) SELECT {insert_cols} FROM {staging} "
            f"ON CONFLICT ({conflict_sql}) {action};"
        )
        run_psql(config, sql)
    finally:
        run_psql(config, f"DROP TABLE IF EXISTS {staging};", check=False)


def load_audit_contracts(
    config: PsqlConfig,
    *,
    bundle: AuditBundle,
    bundle_zip: Path,
    source_reconciliation: list[SourceReconciliation],
    temp_dir: Path,
    git_commit: str,
) -> None:
    emit_event("contracts_load_start", "Loading audit contracts and registries")
    contracts = load_contract_tables(bundle)
    audit_snapshot_id = "audit_2026_06_23"
    bundle_manifest = json.dumps(bundle.summary, ensure_ascii=False).replace("'", "''")
    run_psql(
        config,
        f"""
        INSERT INTO catalog.audit_snapshot (
            audit_snapshot_id, bundle_sha256, bundle_bytes, original_local_path,
            repository_uri, extracted_uri, generated_at_utc, extracted_at_utc,
            git_commit_before, git_commit_after, manifest
        ) VALUES (
            {sql_literal(audit_snapshot_id)}, {sql_literal(sha256_file(bundle_zip))},
            {bundle_zip.stat().st_size}, {sql_literal(str(bundle_zip.resolve()))},
            'repo://data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT_BUNDLE.zip',
            'repo://data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT',
            {sql_literal(bundle.generated_at_utc)}::timestamptz, now(),
            {sql_literal(git_commit)}, {sql_literal(git_commit)}, '{bundle_manifest}'::jsonb
        )
        ON CONFLICT (audit_snapshot_id) DO UPDATE SET
            bundle_sha256 = EXCLUDED.bundle_sha256,
            manifest = EXCLUDED.manifest;
        """,
    )

    dataset_rows = [
        {
            "dataset_id": row["dataset_id"],
            "db_inclusion": row["db_inclusion"],
            "recommended_layer": row["recommended_layer"],
            "current_operational_value": row["current_operational_predictor_value_0_100"],
            "diagnostic_research_value": row["diagnostic_research_value_0_100"],
            "future_potential": row["future_potential_0_100"],
            "verdict": row["verdict"],
            "audit_snapshot_id": audit_snapshot_id,
            "contract_version": CONTRACT_VERSION,
        }
        for row in contracts["datasets"]
    ]
    dataset_csv = temp_dir / "dataset_registry.csv"
    dataset_columns = [
        ("dataset_id", "text"),
        ("db_inclusion", "text"),
        ("recommended_layer", "text"),
        ("current_operational_value", "smallint"),
        ("diagnostic_research_value", "smallint"),
        ("future_potential", "smallint"),
        ("verdict", "text"),
        ("audit_snapshot_id", "text"),
        ("contract_version", "text"),
    ]
    write_rows_csv(dataset_csv, dataset_rows, [name for name, _ in dataset_columns])
    create_staging_and_upsert(
        config,
        table="catalog.dataset_registry",
        columns=dataset_columns,
        csv_path=dataset_csv,
        conflict_targets=["dataset_id"],
    )

    table_rows = []
    for row in contracts["tables"]:
        table_rows.append(
            {
                **row,
                "audit_snapshot_id": audit_snapshot_id,
                "contract_version": CONTRACT_VERSION,
            },
        )
    table_csv = temp_dir / "table_load_contract.csv"
    table_columns = [
        ("dataset_id", "text"),
        ("source_file", "text"),
        ("file_type", "text"),
        ("row_count", "bigint"),
        ("byte_size", "bigint"),
        ("attribute_count", "integer"),
        ("data_min", "timestamptz"),
        ("data_max", "timestamptz"),
        ("db_action", "text"),
        ("db_layer", "text"),
        ("model_status", "text"),
        ("priority", "text"),
        ("notes", "text"),
        ("audit_snapshot_id", "text"),
        ("contract_version", "text"),
    ]
    write_rows_csv(table_csv, table_rows, [name for name, _ in table_columns])
    create_staging_and_upsert(
        config,
        table="governance.table_load_contract",
        columns=table_columns,
        csv_path=table_csv,
        conflict_targets=["source_file", "contract_version"],
    )

    source_rows = [
        {
            "dataset_id": row.dataset_id,
            "source_file": row.source_file,
            "repository_uri": row.repository_uri,
            "original_local_path": row.original_local_path,
            "file_type": row.file_type,
            "physical_sha256": row.physical_sha256,
            "byte_size": row.observed_bytes,
            "source_row_count": row.observed_rows,
            "attribute_count": row.observed_attributes,
            "data_min": row.data_min,
            "data_max": row.data_max,
            "metadata_min": row.metadata_min or "",
            "metadata_max": row.metadata_max or "",
            "ingestion_action": row.db_action,
            "target_database_layer": row.db_layer,
            "model_status": row.model_status,
            "priority": row.priority,
            "ingestion_version": CONTRACT_VERSION,
            "audit_snapshot_id": audit_snapshot_id,
            "status": row.status,
        }
        for row in source_reconciliation
    ]
    source_csv = temp_dir / "source_file_registry.csv"
    source_columns = [
        ("dataset_id", "text"),
        ("source_file", "text"),
        ("repository_uri", "text"),
        ("original_local_path", "text"),
        ("file_type", "text"),
        ("physical_sha256", "char(64)"),
        ("byte_size", "bigint"),
        ("source_row_count", "bigint"),
        ("attribute_count", "integer"),
        ("data_min", "timestamptz"),
        ("data_max", "timestamptz"),
        ("metadata_min", "timestamptz"),
        ("metadata_max", "timestamptz"),
        ("ingestion_action", "text"),
        ("target_database_layer", "text"),
        ("model_status", "text"),
        ("priority", "text"),
        ("ingestion_version", "text"),
        ("audit_snapshot_id", "text"),
        ("status", "text"),
    ]
    write_rows_csv(source_csv, source_rows, [name for name, _ in source_columns])
    source_staging = staging_table_name("catalog.source_file_registry")
    run_psql(
        config,
        f"""
        DROP TABLE IF EXISTS {source_staging};
        CREATE UNLOGGED TABLE {source_staging} (
            dataset_id text, source_file text, repository_uri text, original_local_path text,
            file_type text, physical_sha256 char(64), byte_size bigint, source_row_count bigint,
            attribute_count integer, data_min timestamptz, data_max timestamptz,
            metadata_min timestamptz, metadata_max timestamptz, ingestion_action text,
            target_database_layer text, model_status text, priority text, ingestion_version text,
            audit_snapshot_id text, status text
        );
        """,
    )
    try:
        psql_copy(
            config,
            source_staging,
            [name for name, _ in source_columns],
            source_csv,
        )
        run_psql(
            config,
            f"""
            INSERT INTO catalog.source_file_registry (
                dataset_id, source_file, repository_uri, original_local_path, file_type,
                physical_sha256, byte_size, source_row_count, attribute_count, data_min, data_max,
                metadata_min, metadata_max, ingestion_action, target_database_layer, model_status,
                priority, ingestion_version, audit_snapshot_id, status
            )
            SELECT dataset_id, source_file, repository_uri, original_local_path, file_type,
                physical_sha256, byte_size, source_row_count, attribute_count, data_min, data_max,
                metadata_min, metadata_max, ingestion_action, target_database_layer, model_status,
                priority, ingestion_version, audit_snapshot_id, status
            FROM {source_staging}
            ON CONFLICT (source_file, physical_sha256, ingestion_version) DO UPDATE SET
                source_row_count = EXCLUDED.source_row_count,
                attribute_count = EXCLUDED.attribute_count,
                status = EXCLUDED.status;
            """,
        )
    finally:
        run_psql(config, f"DROP TABLE IF EXISTS {source_staging};", check=False)

    attribute_rows = []
    for row in contracts["attributes"]:
        attribute_rows.append(
            {
                "dataset_id": row["dataset_id"],
                "source_file": row["source_file"],
                "file_type": row["file_type"],
                "attribute_name": row["attribute"],
                "source_dtype": row["source_dtype"],
                "semantic_class": row["semantic_class"],
                "row_count": row["row_count"],
                "non_null_count": row["non_null_count"],
                "null_count": row["null_count"],
                "null_pct": row["null_pct"],
                "storage_decision": row["storage_decision"],
                "db_layer": row["db_layer"],
                "model_role": row["model_role"],
                "operational_status": row["operational_status"],
                "quality_action": row["quality_action"],
                "usefulness_score": row["usefulness_score_0_100"],
                "rationale": row["rationale"],
                "profile_min": row["profile_min"],
                "profile_max": row["profile_max"],
                "audit_snapshot_id": audit_snapshot_id,
                "contract_version": CONTRACT_VERSION,
                "reconciliation_status": "CONTRACTED",
                "physical_destination": row["db_layer"],
            },
        )
    attr_csv = temp_dir / "attribute_contract.csv"
    attr_columns = [
        ("dataset_id", "text"),
        ("source_file", "text"),
        ("file_type", "text"),
        ("attribute_name", "text"),
        ("source_dtype", "text"),
        ("semantic_class", "text"),
        ("row_count", "bigint"),
        ("non_null_count", "bigint"),
        ("null_count", "bigint"),
        ("null_pct", "double precision"),
        ("storage_decision", "text"),
        ("db_layer", "text"),
        ("model_role", "text"),
        ("operational_status", "text"),
        ("quality_action", "text"),
        ("usefulness_score", "smallint"),
        ("rationale", "text"),
        ("profile_min", "text"),
        ("profile_max", "text"),
        ("audit_snapshot_id", "text"),
        ("contract_version", "text"),
        ("reconciliation_status", "text"),
        ("physical_destination", "text"),
    ]
    write_rows_csv(attr_csv, attribute_rows, [name for name, _ in attr_columns])
    run_psql(config, "TRUNCATE catalog.attribute_contract RESTART IDENTITY;")
    psql_copy(
        config,
        "catalog.attribute_contract",
        [name for name, _ in attr_columns],
        attr_csv,
    )

    dataset_ids = {row["dataset_id"] for row in contracts["datasets"]}
    quality_rows = []
    for index, row in enumerate(contracts["quality_issues"], start=1):
        dataset_scope = row["dataset_id"]
        normalized_dataset_id = dataset_scope if dataset_scope in dataset_ids else ""
        scope_note = (
            ""
            if normalized_dataset_id
            else f" Audit dataset scope was {dataset_scope!r}, not a concrete dataset id."
        )
        quality_rows.append(
            {
                "quality_issue_id": f"QI-{index:03d}",
                **row,
                "dataset_id": normalized_dataset_id,
                "current_status": "OPEN",
                "remediation_implementation_path": "",
                "validation_evidence_uri": "",
                "resolution_timestamp": "",
                "resolution_commit": "",
                "notes": "Loaded from audit contract; not resolved without validation evidence." + scope_note,
            },
        )
    quality_csv = temp_dir / "quality_issue.csv"
    quality_columns = [
        ("quality_issue_id", "text"),
        ("severity", "text"),
        ("dataset_id", "text"),
        ("source_table", "text"),
        ("attributes", "text"),
        ("evidence", "text"),
        ("required_action", "text"),
        ("current_status", "text"),
        ("remediation_implementation_path", "text"),
        ("validation_evidence_uri", "text"),
        ("resolution_timestamp", "timestamptz"),
        ("resolution_commit", "text"),
        ("notes", "text"),
    ]
    write_rows_csv(quality_csv, quality_rows, [name for name, _ in quality_columns])
    create_staging_and_upsert(
        config,
        table="governance.quality_issue",
        columns=quality_columns,
        csv_path=quality_csv,
        conflict_targets=["quality_issue_id"],
    )

    station_rows = []
    for row in contracts["stations"]:
        station_rows.append(
            {
                "station_id": row["station_id"],
                "station_name": row["STATION NAME"],
                "country_code": row["CTRY"],
                "icao": row["ICAO"],
                "valid_from": yyyymmdd_to_date(row["BEGIN"]),
                "valid_to": yyyymmdd_to_date(row["END"]),
                "latitude": row["LAT"],
                "longitude": row["LON"],
                "elevation_m": row["ELEV(M)"],
                "distance_to_hko_km": row["distance_km"],
                "bearing_from_hko_deg": row["bearing_deg"],
                "tier": row["tier"],
                "meteorological_role": row["role"],
                "research_note": row["research_note"],
                "dossier_version": CONTRACT_VERSION,
                "audit_snapshot_id": audit_snapshot_id,
            },
        )
    station_csv = temp_dir / "station_dim.csv"
    station_columns = [
        ("station_id", "text"),
        ("station_name", "text"),
        ("country_code", "text"),
        ("icao", "text"),
        ("valid_from", "date"),
        ("valid_to", "date"),
        ("latitude", "double precision"),
        ("longitude", "double precision"),
        ("elevation_m", "double precision"),
        ("distance_to_hko_km", "double precision"),
        ("bearing_from_hko_deg", "double precision"),
        ("tier", "text"),
        ("meteorological_role", "text"),
        ("research_note", "text"),
        ("dossier_version", "text"),
        ("audit_snapshot_id", "text"),
    ]
    write_rows_csv(station_csv, station_rows, [name for name, _ in station_columns])
    run_psql(config, "TRUNCATE catalog.station_dim RESTART IDENTITY;")
    psql_copy(config, "catalog.station_dim", [name for name, _ in station_columns], station_csv)

    register_assets(config, bundle=bundle, bundle_zip=bundle_zip, source_reconciliation=source_reconciliation)
    emit_event(
        "contracts_load_done",
        "Loaded audit contracts and registries",
        datasets=len(contracts["datasets"]),
        tables=len(contracts["tables"]),
        attributes=len(contracts["attributes"]),
        quality_issues=len(contracts["quality_issues"]),
        stations=len(contracts["stations"]),
    )


def yyyymmdd_to_date(value: str) -> str:
    if not value:
        return ""
    return f"{value[0:4]}-{value[4:6]}-{value[6:8]}"


def register_assets(
    config: PsqlConfig,
    *,
    bundle: AuditBundle,
    bundle_zip: Path,
    source_reconciliation: list[SourceReconciliation],
) -> None:
    rows = [
        {
            "asset_uri": "repo://data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT_BUNDLE.zip",
            "original_local_path": str(bundle_zip.resolve()),
            "content_sha256": sha256_file(bundle_zip),
            "byte_size": bundle_zip.stat().st_size,
            "media_type": "application/zip",
            "dataset_id": "",
            "source_file_id": "",
            "asset_role": "AUDIT_BUNDLE",
            "extraction_status": "SNAPSHOTTED",
            "metadata": json.dumps({"generated_at_utc": bundle.generated_at_utc}, sort_keys=True),
        },
    ]
    for name in bundle.summary["files"]:
        path = bundle.root / name
        rows.append(
            {
                "asset_uri": f"repo://data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/{name}",
                "original_local_path": str(path.resolve()),
                "content_sha256": sha256_file(path),
                "byte_size": path.stat().st_size,
                "media_type": media_type_for_path(path),
                "dataset_id": "",
                "source_file_id": "",
                "asset_role": "AUDIT_CONTRACT_FILE",
                "extraction_status": "SNAPSHOTTED",
                "metadata": "{}",
            },
        )
    for item in source_reconciliation:
        if item.db_action in OBJECT_ACTIONS:
            rows.append(
                {
                    "asset_uri": item.repository_uri,
                    "original_local_path": item.original_local_path,
                    "content_sha256": item.physical_sha256,
                    "byte_size": item.observed_bytes,
                    "media_type": media_type_for_path(Path(item.source_file)),
                    "dataset_id": item.dataset_id,
                    "source_file_id": "",
                    "asset_role": item.db_action,
                    "extraction_status": "REGISTERED_ONLY",
                    "metadata": json.dumps({"model_status": item.model_status, "db_layer": item.db_layer}),
                },
            )
    with tempfile.TemporaryDirectory(prefix="hkg_tmax_assets_") as temp:
        csv_path = Path(temp) / "assets.csv"
        columns = [
            "asset_uri",
            "original_local_path",
            "content_sha256",
            "byte_size",
            "media_type",
            "dataset_id",
            "source_file_id",
            "asset_role",
            "extraction_status",
            "metadata",
        ]
        write_rows_csv(csv_path, rows, columns)
        asset_staging = staging_table_name("object_catalog.asset")
        run_psql(
            config,
            f"""
            DROP TABLE IF EXISTS {asset_staging};
            CREATE UNLOGGED TABLE {asset_staging} (
                asset_uri text, original_local_path text, content_sha256 char(64),
                byte_size bigint, media_type text, dataset_id text, source_file_id bigint,
                asset_role text, extraction_status text, metadata jsonb
            );
            """,
        )
        try:
            psql_copy(config, asset_staging, columns, csv_path)
            run_psql(
                config,
                f"""
                INSERT INTO object_catalog.asset (
                    asset_uri, original_local_path, content_sha256, byte_size, media_type,
                    dataset_id, source_file_id, asset_role, extraction_status, metadata
                )
                SELECT asset_uri, original_local_path, content_sha256, byte_size, media_type,
                    NULLIF(dataset_id, ''), source_file_id, asset_role, extraction_status, metadata
                FROM {asset_staging}
                ON CONFLICT (asset_uri, content_sha256) DO UPDATE SET
                    byte_size = EXCLUDED.byte_size,
                    extraction_status = EXCLUDED.extraction_status,
                    metadata = EXCLUDED.metadata;
                """,
            )
        finally:
            run_psql(config, f"DROP TABLE IF EXISTS {asset_staging};", check=False)


def media_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".zip": "application/zip",
        ".json": "application/json",
        ".csv": "text/csv",
        ".md": "text/markdown",
        ".sql": "application/sql",
        ".parquet": "application/vnd.apache.parquet",
    }.get(suffix, "application/octet-stream")


def source_file_id(config: PsqlConfig, source_file: str) -> int:
    completed = run_psql(
        config,
        f"SELECT source_file_id FROM catalog.source_file_registry WHERE source_file = {sql_literal(source_file)} ORDER BY source_file_id DESC LIMIT 1;",
    )
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError(f"No source_file_id found for {source_file}")
    return int(value)


def create_dynamic_table(
    config: PsqlConfig,
    *,
    schema: str,
    table_name: str,
    columns: Sequence[tuple[str, str]],
) -> str:
    table = f"{quote_ident(schema)}.{quote_ident(table_name)}"
    index_name = bounded_identifier(f"{table_name}_source_idx", fallback="source_idx")
    column_defs = [
        "ingest_source_file text NOT NULL",
        "ingest_source_file_id bigint REFERENCES catalog.source_file_registry(source_file_id)",
        "ingest_source_row_number bigint NOT NULL",
        "ingested_at_utc timestamptz NOT NULL",
        "ingestion_batch_id text NOT NULL REFERENCES ingestion.batch(batch_id)",
    ]
    column_defs.extend(f"{quote_ident(name)} {pg_type}" for name, pg_type in columns)
    run_psql(
        config,
        f"""
        DROP TABLE IF EXISTS {table} CASCADE;
        CREATE TABLE {table} (
            {", ".join(column_defs)}
        );
        CREATE INDEX {quote_ident(index_name)} ON {table} (ingest_source_file_id, ingest_source_row_number);
        """,
    )
    return table


def schema_and_columns_for_file(path: Path, file_type: str) -> tuple[list[str], list[tuple[str, str]], dict[str, str]]:
    if file_type == "parquet":
        schema = pq.ParquetFile(path).schema_arrow
        source_names = [field.name for field in schema]
        clean_names = clean_column_names(source_names)
        return (
            source_names,
            [(clean_names[index], postgres_type_for_arrow(field.type)) for index, field in enumerate(schema)],
            dict(zip(source_names, clean_names, strict=True)),
        )
    try:
        sample = pd.read_csv(path, nrows=1000)
    except pd.errors.EmptyDataError:
        return [], [], {}
    source_names = [str(column) for column in sample.columns]
    clean_names = clean_column_names(source_names)
    return (
        source_names,
        [
            (clean_names[index], postgres_type_for_csv_column(source_name, sample[source_name]))
            for index, source_name in enumerate(source_names)
        ],
        dict(zip(source_names, clean_names, strict=True)),
    )


def csv_for_source(
    *,
    path: Path,
    file_type: str,
    output_csv: Path,
    source_names: Sequence[str],
    clean_map: dict[str, str],
    source_file: str,
    source_id: int,
    batch_id: str,
) -> int:
    output_columns = [
        "ingest_source_file",
        "ingest_source_file_id",
        "ingest_source_row_number",
        "ingested_at_utc",
        "ingestion_batch_id",
        *[clean_map[name] for name in source_names],
    ]
    rows = 0
    header_written = False
    if file_type == "parquet":
        parquet_file = pq.ParquetFile(path)
        batches = (batch.to_pandas() for batch in parquet_file.iter_batches(batch_size=ROW_CHUNK_SIZE))
    elif not source_names:
        pd.DataFrame(columns=output_columns).to_csv(output_csv, index=False, encoding="utf-8")
        return 0
    else:
        batches = pd.read_csv(path, chunksize=ROW_CHUNK_SIZE)
    for frame in batches:
        frame = frame.rename(columns=clean_map)
        frame = dataframe_for_copy(frame)
        row_numbers = range(rows + 1, rows + len(frame) + 1)
        frame.insert(0, "ingestion_batch_id", batch_id)
        frame.insert(0, "ingested_at_utc", utc_now_iso())
        frame.insert(0, "ingest_source_row_number", list(row_numbers))
        frame.insert(0, "ingest_source_file_id", source_id)
        frame.insert(0, "ingest_source_file", source_file)
        frame = frame[output_columns]
        frame.to_csv(
            output_csv,
            mode="a",
            header=not header_written,
            index=False,
            na_rep="",
            encoding="utf-8",
        )
        header_written = True
        rows += len(frame)
        emit_event(
            "source_csv_progress",
            "Prepared source rows for COPY",
            source_file=source_file,
            rows_prepared=rows,
        )
    if not header_written:
        pd.DataFrame(columns=output_columns).to_csv(output_csv, index=False, encoding="utf-8")
    return rows


def load_label_split(
    config: PsqlConfig,
    *,
    path: Path,
    source_file_id_value: int,
    batch_id: str,
    temp_dir: Path,
) -> tuple[int, int]:
    emit_event("label_split_start", "Loading target-label table into pre-2024 and sealed 2024+ tables")
    frame = pd.read_parquet(path)
    frame["local_date"] = pd.to_datetime(frame["local_date"]).dt.date.astype(str)
    frame["retrieved_at_utc"] = frame["raw_retrieved_at_utc"]
    frame["quality_status"] = "VALID"
    frame["source_file_id"] = source_file_id_value
    frame["ingestion_batch_id"] = batch_id
    columns = [
        "local_date",
        "target_tmax_c",
        "target_station",
        "target_source_id",
        "content_sha256",
        "retrieved_at_utc",
        "quality_status",
        "source_file_id",
        "ingestion_batch_id",
    ]
    pre = frame[pd.to_datetime(frame["local_date"]) < pd.Timestamp("2024-01-01")][columns]
    sealed = frame[pd.to_datetime(frame["local_date"]) >= pd.Timestamp("2024-01-01")][columns]
    pre_csv = temp_dir / "label_core_hko_daily_tmax.csv"
    sealed_csv = temp_dir / "sealed_confirmation_hko_daily_tmax.csv"
    pre.to_csv(pre_csv, index=False, na_rep="", encoding="utf-8")
    sealed.to_csv(sealed_csv, index=False, na_rep="", encoding="utf-8")
    run_psql(config, "TRUNCATE label_core.hko_daily_tmax; TRUNCATE sealed_confirmation.hko_daily_tmax;")
    psql_copy(config, "label_core.hko_daily_tmax", columns, pre_csv)
    psql_copy(config, "sealed_confirmation.hko_daily_tmax", columns, sealed_csv)
    emit_event("label_split_done", "Loaded target-label split", label_core_rows=len(pre), sealed_rows=len(sealed))
    return len(pre), len(sealed)


def build_anchor_rows(config: PsqlConfig, *, source_table: str, batch_id: str) -> int:
    emit_event("anchor_build_start", "Building leakage-safe T-24 official anchor rows")
    run_psql(
        config,
        f"""
        TRUNCATE operational_anchor.hko_t24_official_anchor_rows RESTART IDENTITY;
        INSERT INTO operational_anchor.hko_t24_official_anchor_rows (
            target_date, cutoff_utc, forecast_min_c, forecast_max_c, forecast_range_c,
            source_era, source_product, issue_time_utc, published_at_utc, available_at_utc,
            selected_source_row_id, selection_rule_version, quality_status, eligibility_status,
            source_file_id, ingestion_batch_id
        )
        SELECT DISTINCT ON (forecast_date::date)
            forecast_date::date AS target_date,
            governance.hkg_t24_cutoff_utc(forecast_date::date) AS cutoff_utc,
            forecast_min_temperature_c,
            forecast_max_temperature_c,
            forecast_max_temperature_c - forecast_min_temperature_c AS forecast_range_c,
            'rss_exact_vintage' AS source_era,
            feed_type AS source_product,
            published_at_utc::timestamptz,
            published_at_utc::timestamptz,
            available_at_hkt::timestamptz AS available_at_utc,
            guid AS selected_source_row_id,
            'latest_available_before_hkg_t24_cutoff_v1' AS selection_rule_version,
            'VALID' AS quality_status,
            'ELIGIBLE' AS eligibility_status,
            ingest_source_file_id,
            {sql_literal(batch_id)}
        FROM {source_table}
        WHERE forecast_date IS NOT NULL
          AND forecast_max_temperature_c IS NOT NULL
          AND available_at_hkt::timestamptz <= governance.hkg_t24_cutoff_utc(forecast_date::date)
        ORDER BY forecast_date::date, available_at_hkt::timestamptz DESC, published_at_utc DESC;
        """,
    )
    completed = run_psql(config, "SELECT count(*) FROM operational_anchor.hko_t24_official_anchor_rows;")
    row_count = int(completed.stdout.strip())
    emit_event("anchor_build_done", "Built leakage-safe T-24 official anchor rows", rows=row_count)
    return row_count


def load_sources(
    config: PsqlConfig,
    *,
    source_reconciliation: list[SourceReconciliation],
    datasets_root: Path,
    batch_id: str,
    temp_dir: Path,
) -> tuple[dict[str, int], int, int, int, str | None]:
    rows_by_layer: Counter[str] = Counter()
    rows_quarantined = 0
    duplicate_skipped = 0
    objects_registered = 0
    files_succeeded = 0
    rss_temperature_table: str | None = None
    emit_event("sources_load_start", "Starting source data load", source_files=len(source_reconciliation))
    for item in source_reconciliation:
        started = utc_now_iso()
        source_id = source_file_id(config, item.source_file)
        emit_event(
            "source_file_start",
            "Processing source file",
            source_file=item.source_file,
            db_action=item.db_action,
            db_layer=item.db_layer,
            expected_rows=item.expected_rows,
            observed_rows=item.observed_rows,
        )
        if item.db_action == "SKIP_DUPLICATE_FORMAT":
            duplicate_skipped += 1
            insert_file_result(
                config,
                batch_id=batch_id,
                source_file_id_value=source_id,
                item=item,
                status="SKIPPED_DUPLICATE",
                rows_inserted=0,
                rows_skipped=item.observed_rows,
                rows_quarantined=0,
                started_at=started,
            )
            emit_event("source_file_skipped_duplicate", "Skipped duplicate-format source file", source_file=item.source_file)
            continue
        if item.db_action in OBJECT_ACTIONS:
            objects_registered += 1
            insert_file_result(
                config,
                batch_id=batch_id,
                source_file_id_value=source_id,
                item=item,
                status="REGISTERED_OBJECT_ONLY",
                rows_inserted=0,
                rows_skipped=item.observed_rows,
                rows_quarantined=0,
                started_at=started,
            )
            emit_event("source_file_registered_object", "Registered object/artifact without row load", source_file=item.source_file)
            continue
        path = datasets_root / item.source_file
        if item.source_file == "01_hko_daily_tmax_target/hko_daily_tmax_target_labels.parquet":
            pre_count, sealed_count = load_label_split(
                config,
                path=path,
                source_file_id_value=source_id,
                batch_id=batch_id,
                temp_dir=temp_dir,
            )
            rows_by_layer["label_core"] += pre_count
            rows_by_layer["sealed_confirmation"] += sealed_count
            insert_file_result(
                config,
                batch_id=batch_id,
                source_file_id_value=source_id,
                item=item,
                status="LOADED_SPLIT_SEALED",
                rows_inserted=pre_count + sealed_count,
                rows_skipped=0,
                rows_quarantined=0,
                started_at=started,
            )
            files_succeeded += 1
            emit_event(
                "source_file_done",
                "Loaded source file",
                source_file=item.source_file,
                rows_inserted=pre_count + sealed_count,
                rows_quarantined=0,
            )
            continue

        schema = item.db_layer if item.db_layer not in {"none", "live_object_catalog", "nwp_object_catalog", "static_object_catalog"} else "object_catalog"
        table_name = bounded_identifier(GENERATED_TABLE_PREFIX + table_name_for_source(item.source_file))
        source_names, columns, clean_map = schema_and_columns_for_file(path, item.file_type)
        table = create_dynamic_table(config, schema=schema, table_name=table_name, columns=columns)
        csv_path = temp_dir / f"{table_name}.csv"
        rows_inserted = csv_for_source(
            path=path,
            file_type=item.file_type,
            output_csv=csv_path,
            source_names=source_names,
            clean_map=clean_map,
            source_file=item.source_file,
            source_id=source_id,
            batch_id=batch_id,
        )
        copy_columns = [
            "ingest_source_file",
            "ingest_source_file_id",
            "ingest_source_row_number",
            "ingested_at_utc",
            "ingestion_batch_id",
            *[name for name, _ in columns],
        ]
        psql_copy(config, table, copy_columns, csv_path)
        emit_event(
            "source_copy_done",
            "Copied source rows into target table",
            source_file=item.source_file,
            target_table=table,
            rows_inserted=rows_inserted,
        )
        quarantine_count = quarantine_known_invalids(
            config,
            table=table,
            item=item,
            batch_id=batch_id,
            source_file_id_value=source_id,
        )
        rows_quarantined += quarantine_count
        rows_by_layer[schema] += rows_inserted
        insert_file_result(
            config,
            batch_id=batch_id,
            source_file_id_value=source_id,
            item=item,
            status="LOADED",
            rows_inserted=rows_inserted,
            rows_skipped=0,
            rows_quarantined=quarantine_count,
            started_at=started,
        )
        files_succeeded += 1
        if item.source_file == "05_hko_historical_rss_forecasts/hko_historical_rss_temperature_forecasts.parquet":
            rss_temperature_table = table
        emit_event(
            "source_file_done",
            "Loaded source file",
            source_file=item.source_file,
            target_table=table,
            rows_inserted=rows_inserted,
            rows_quarantined=quarantine_count,
        )
    emit_event(
        "sources_load_done",
        "Finished source data load",
        files_succeeded=files_succeeded,
        duplicate_formats_skipped=duplicate_skipped,
        objects_registered=objects_registered,
        rows_quarantined=rows_quarantined,
    )
    return dict(rows_by_layer), rows_quarantined, duplicate_skipped, objects_registered, rss_temperature_table


def quarantine_known_invalids(
    config: PsqlConfig,
    *,
    table: str,
    item: SourceReconciliation,
    batch_id: str,
    source_file_id_value: int,
) -> int:
    if "hko_press_archive_temperature_forecast_days.parquet" not in item.source_file:
        return 0
    run_psql(
        config,
        f"""
        DELETE FROM ingestion.row_rejection
        WHERE batch_id = {sql_literal(batch_id)}
          AND source_file_id = {source_file_id_value}
          AND target_table = {sql_literal(table)};

        INSERT INTO ingestion.row_rejection (
            batch_id, source_file_id, source_row_number, dataset_id, target_table,
            reason_code, reason_detail, raw_row_payload, raw_content_hash
        )
        SELECT {sql_literal(batch_id)}, {source_file_id_value}, ingest_source_row_number,
            {sql_literal(item.dataset_id)}, {sql_literal(table)},
            CASE
                WHEN forecast_max_c > 60 OR forecast_min_c < -20 THEN 'IMPOSSIBLE_TEMPERATURE'
                ELSE 'INVALID_LEAD_OR_TARGET_DATE'
            END,
            'Audit-driven quarantine from scoreable/temperature plausibility flags',
            to_jsonb(t),
            md5(to_jsonb(t)::text)
        FROM {table} AS t
        WHERE COALESCE(scoreable_row_valid, true) = false
           OR COALESCE(temperature_row_valid, true) = false
           OR COALESCE(target_date_plausible, true) = false
           OR forecast_max_c > 60
           OR forecast_min_c < -20;
        """,
    )
    completed = run_psql(
        config,
        f"""
        SELECT count(*)
        FROM {table}
        WHERE COALESCE(scoreable_row_valid, true) = false
           OR COALESCE(temperature_row_valid, true) = false
           OR COALESCE(target_date_plausible, true) = false
           OR forecast_max_c > 60
           OR forecast_min_c < -20;
        """,
    )
    return int(completed.stdout.strip())


def insert_file_result(
    config: PsqlConfig,
    *,
    batch_id: str,
    source_file_id_value: int,
    item: SourceReconciliation,
    status: str,
    rows_inserted: int,
    rows_skipped: int,
    rows_quarantined: int,
    started_at: str,
) -> None:
    expected_schema = json.dumps(
        {"expected_attributes": item.expected_attributes},
        sort_keys=True,
    ).replace("'", "''")
    observed_schema = json.dumps(
        {"observed_attributes": item.observed_attributes},
        sort_keys=True,
    ).replace("'", "''")
    run_psql(
        config,
        f"""
        INSERT INTO ingestion.file_result (
            batch_id, source_file_id, source_file, expected_hash, observed_hash,
            expected_row_count, observed_row_count, expected_schema, observed_schema,
            load_action, started_at_utc, finished_at_utc, rows_staged, rows_inserted,
            rows_updated_versioned, rows_quarantined, rows_skipped_as_duplicate,
            status, error_text, reconciliation_artifact_uri
        ) VALUES (
            {sql_literal(batch_id)}, {source_file_id_value}, {sql_literal(item.source_file)},
            {sql_literal(item.physical_sha256)}, {sql_literal(item.physical_sha256)},
            {item.expected_rows}, {item.observed_rows}, '{expected_schema}'::jsonb, '{observed_schema}'::jsonb,
            {sql_literal(item.db_action)}, {sql_literal(started_at)}::timestamptz, now(),
            {rows_inserted}, {rows_inserted}, 0, {rows_quarantined}, {rows_skipped},
            {sql_literal(status)}, NULL,
            'repo://experiments/0206_audit_driven_database_ingestion/DB_SOURCE_FILE_RECONCILIATION_ALL_52.csv'
        )
        ON CONFLICT (batch_id, source_file) DO UPDATE SET
            finished_at_utc = EXCLUDED.finished_at_utc,
            rows_inserted = EXCLUDED.rows_inserted,
            rows_quarantined = EXCLUDED.rows_quarantined,
            rows_skipped_as_duplicate = EXCLUDED.rows_skipped_as_duplicate,
            status = EXCLUDED.status;
        """,
    )


def create_batch(config: PsqlConfig, *, batch_id: str, audit_hash: str, command_line: str) -> None:
    host_metadata = json.dumps(
        {"hostname": socket.gethostname(), "pid": os.getpid()},
        sort_keys=True,
    ).replace("'", "''")
    run_psql(
        config,
        f"""
        INSERT INTO ingestion.batch (
            batch_id, started_at_utc, status, code_commit, audit_snapshot_hash,
            dataset_root_uri, cutoff_rule_version, database_target_redacted,
            loader_version, command_line, host_metadata
        ) VALUES (
            {sql_literal(batch_id)}, now(), 'STARTED', NULL, {sql_literal(audit_hash)},
            'repo://data/datasets', {sql_literal(CUTOFF_RULE_VERSION)},
            {sql_literal(config.host + ':' + str(config.port) + '/' + config.database)},
            {sql_literal(LOADER_VERSION)}, {sql_literal(command_line)}, '{host_metadata}'::jsonb
        )
        ON CONFLICT (batch_id) DO UPDATE SET
            started_at_utc = now(),
            status = 'STARTED',
            command_line = EXCLUDED.command_line;
        """,
    )


def finish_batch(
    config: PsqlConfig,
    *,
    batch_id: str,
    files_succeeded: int,
    files_skipped: int,
    status: str = "SUCCEEDED",
) -> None:
    run_psql(
        config,
        f"""
        UPDATE ingestion.batch
        SET finished_at_utc = now(),
            status = {sql_literal(status)},
            files_succeeded = {files_succeeded},
            files_failed = 0,
            files_skipped = {files_skipped},
            files_resumed = 0
        WHERE batch_id = {sql_literal(batch_id)};
        """,
    )


def load_reconciliation(
    config: PsqlConfig,
    *,
    source_reconciliation: list[SourceReconciliation],
    batch_id: str,
) -> None:
    emit_event("reconciliation_load_start", "Loading source-file reconciliation rows")
    rows = []
    for item in source_reconciliation:
        rows.append(
            {
                "batch_id": batch_id,
                "reconciliation_scope": "SOURCE_FILE",
                "dataset_id": item.dataset_id,
                "source_file": item.source_file,
                "attribute_name": "",
                "expected_disposition": item.db_action,
                "actual_disposition": item.disposition,
                "physical_destination": item.db_layer,
                "count_hash_evidence": json.dumps(
                    {
                        "expected_rows": item.expected_rows,
                        "observed_rows": item.observed_rows,
                        "sha256": item.physical_sha256,
                    },
                    sort_keys=True,
                ),
                "status": item.status,
                "exception_explanation": item.exception,
            },
        )
    with tempfile.TemporaryDirectory(prefix="hkg_tmax_recon_") as temp:
        csv_path = Path(temp) / "reconciliation.csv"
        columns = [
            "batch_id",
            "reconciliation_scope",
            "dataset_id",
            "source_file",
            "attribute_name",
            "expected_disposition",
            "actual_disposition",
            "physical_destination",
            "count_hash_evidence",
            "status",
            "exception_explanation",
        ]
        write_rows_csv(csv_path, rows, columns)
        reconciliation_staging = staging_table_name("ingestion.reconciliation")
        run_psql(
            config,
            f"""
            DROP TABLE IF EXISTS {reconciliation_staging};
            CREATE UNLOGGED TABLE {reconciliation_staging} (
                batch_id text, reconciliation_scope text, dataset_id text, source_file text,
                attribute_name text, expected_disposition text, actual_disposition text,
                physical_destination text, count_hash_evidence jsonb, status text,
                exception_explanation text
            );
            """,
        )
        try:
            psql_copy(config, reconciliation_staging, columns, csv_path)
            run_psql(
                config,
                f"""
                INSERT INTO ingestion.reconciliation (
                    batch_id, reconciliation_scope, dataset_id, source_file, attribute_name,
                    expected_disposition, actual_disposition, physical_destination,
                    count_hash_evidence, status, exception_explanation
                )
                SELECT batch_id, reconciliation_scope, dataset_id, source_file, COALESCE(attribute_name, ''),
                    expected_disposition, actual_disposition, physical_destination,
                    count_hash_evidence, status, exception_explanation
                FROM {reconciliation_staging}
                ON CONFLICT (batch_id, reconciliation_scope, source_file, attribute_name) DO UPDATE SET
                    actual_disposition = EXCLUDED.actual_disposition,
                    count_hash_evidence = EXCLUDED.count_hash_evidence,
                    status = EXCLUDED.status,
                    exception_explanation = EXCLUDED.exception_explanation;
                """,
            )
        finally:
            run_psql(config, f"DROP TABLE IF EXISTS {reconciliation_staging};", check=False)
    emit_event("reconciliation_load_done", "Loaded source-file reconciliation rows", rows=len(rows))


def validate_database_firewalls(config: PsqlConfig) -> tuple[bool, bool, bool]:
    sealed = run_psql(
        config,
        "SELECT count(*) FROM sealed_confirmation.hko_daily_tmax WHERE local_date < date '2024-01-01';",
    )
    sealed_ok = sealed.stdout.strip() == "0"
    denied = run_psql(
        config,
        "SET ROLE hkg_tmax_live_inference; SELECT count(*) FROM label_core.hko_daily_tmax; RESET ROLE;",
        check=False,
    )
    live_denied = denied.returncode != 0 and "permission denied" in (denied.stderr + denied.stdout).lower()
    safe_view = run_psql(
        config,
        """
        SELECT count(*) FROM feature_safe.hko_t24_official_anchor
        WHERE available_at_utc > cutoff_utc;
        """,
    )
    strict_ok = safe_view.stdout.strip() == "0"
    return sealed_ok, live_denied, strict_ok


def table_counts_signature(config: PsqlConfig) -> dict[str, int]:
    completed = run_psql(
        config,
        """
        SELECT schemaname || '.' || relname
        FROM pg_stat_user_tables
        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY 1;
        """,
    )
    tables = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    signature: dict[str, int] = {}
    for table in tables:
        if table.startswith("pg_"):
            continue
        count = run_psql(config, f"SELECT count(*) FROM {table};")
        signature[table] = int(count.stdout.strip())
    return signature


def run_full_psql_load(
    *,
    config: PsqlConfig,
    migration_path: Path,
    bundle: AuditBundle,
    bundle_zip: Path,
    source_reconciliation: list[SourceReconciliation],
    datasets_root: Path,
    command_line: str,
    git_commit: str,
    run_suffix: str = "primary",
) -> LoadResult:
    emit_event(
        "psql_load_start",
        "Starting governed direct PostgreSQL load",
        database=config.database,
        run_suffix=run_suffix,
    )
    ensure_database(config)
    emit_event("migration_apply_start", "Applying PostgreSQL migration", migration_path=str(migration_path))
    run_psql_file(config, migration_path)
    emit_event("migration_apply_done", "Applied PostgreSQL migration")
    batch_id = f"audit-ingest-{sha256_file(bundle_zip)[:12]}-{run_suffix}"
    with tempfile.TemporaryDirectory(prefix="hkg_tmax_db_load_") as temp:
        temp_dir = Path(temp)
        create_batch(
            config,
            batch_id=batch_id,
            audit_hash=sha256_file(bundle_zip),
            command_line=command_line,
        )
        load_audit_contracts(
            config,
            bundle=bundle,
            bundle_zip=bundle_zip,
            source_reconciliation=source_reconciliation,
            temp_dir=temp_dir,
            git_commit=git_commit,
        )
        rows_by_layer, rows_quarantined, duplicate_skipped, objects_registered, rss_table = load_sources(
            config,
            source_reconciliation=source_reconciliation,
            datasets_root=datasets_root,
            batch_id=batch_id,
            temp_dir=temp_dir,
        )
        anchor_rows = build_anchor_rows(config, source_table=rss_table, batch_id=batch_id) if rss_table else 0
        rows_by_layer["operational_anchor"] = rows_by_layer.get("operational_anchor", 0) + anchor_rows
        load_reconciliation(config, source_reconciliation=source_reconciliation, batch_id=batch_id)
        sealed_ok, live_denied, strict_ok = validate_database_firewalls(config)
        finish_batch(
            config,
            batch_id=batch_id,
            files_succeeded=sum(1 for row in source_reconciliation if row.db_action not in {"SKIP_DUPLICATE_FORMAT", *OBJECT_ACTIONS}),
            files_skipped=duplicate_skipped + objects_registered,
        )
        emit_event(
            "psql_load_done",
            "Finished governed direct PostgreSQL load",
            batch_id=batch_id,
            rows_loaded_by_layer=dict(rows_by_layer),
            rows_quarantined=rows_quarantined,
            duplicate_formats_skipped=duplicate_skipped,
            objects_registered=objects_registered,
            sealed_confirmation_enforced=sealed_ok,
            live_role_label_access_denied=live_denied,
            strict_validation_passed=strict_ok,
        )
    return LoadResult(
        batch_id=batch_id,
        rows_loaded_by_layer=rows_by_layer,
        rows_quarantined=rows_quarantined,
        duplicate_formats_skipped=duplicate_skipped,
        objects_registered=objects_registered,
        files_succeeded=sum(1 for row in source_reconciliation if row.db_action not in {"SKIP_DUPLICATE_FORMAT", *OBJECT_ACTIONS}),
        files_skipped=duplicate_skipped + objects_registered,
        sealed_confirmation_enforced=sealed_ok,
        live_role_label_access_denied=live_denied,
        strict_validation_passed=strict_ok,
        idempotency_signature=table_counts_signature(config),
    )
