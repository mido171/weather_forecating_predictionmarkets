#!/usr/bin/env python3
"""Load normalized repo datasets into the repo-default MySQL database.

The loader treats ``data/datasets`` as the source of truth for v1:

* every ``.parquet`` file is loaded as a MySQL table;
* every ``.csv`` file is loaded as a MySQL table;
* every ``.zip`` file is registered in metadata only;
* raw blobs and the live HKO SQLite archive are deliberately out of scope.

Default mode is ``replace``. Rerunning the script drops and recreates each data
table, then records a new ingest run in the metadata tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
DEFAULT_DATABASE = "hkg_tmax_research"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 3306
DEFAULT_USER = "root"
DEFAULT_PASSWORD = "root"
DEFAULT_BATCH_SIZE = 5000
MAX_IDENTIFIER_LENGTH = 64

METADATA_TABLES = {
    "dataset_ingest_runs",
    "dataset_files",
    "dataset_tables",
    "dataset_columns",
}
INGEST_METADATA_COLUMNS = {
    "ingest_row_id",
    "ingest_source_path",
    "ingest_source_sha256",
    "ingested_at_utc",
}
MYSQL_RESERVED_WORDS = {
    "accessible",
    "add",
    "all",
    "alter",
    "and",
    "as",
    "between",
    "bigint",
    "binary",
    "blob",
    "boolean",
    "by",
    "call",
    "case",
    "change",
    "check",
    "column",
    "condition",
    "constraint",
    "create",
    "cross",
    "current_date",
    "current_time",
    "current_timestamp",
    "database",
    "databases",
    "date",
    "datetime",
    "day_hour",
    "day_microsecond",
    "day_minute",
    "day_second",
    "dec",
    "decimal",
    "default",
    "delete",
    "desc",
    "describe",
    "distinct",
    "double",
    "drop",
    "else",
    "exists",
    "false",
    "float",
    "for",
    "force",
    "foreign",
    "from",
    "fulltext",
    "generated",
    "group",
    "having",
    "if",
    "in",
    "index",
    "inner",
    "insert",
    "int",
    "integer",
    "interval",
    "into",
    "is",
    "join",
    "json",
    "key",
    "keys",
    "left",
    "like",
    "limit",
    "lock",
    "long",
    "longtext",
    "match",
    "not",
    "null",
    "on",
    "or",
    "order",
    "outer",
    "primary",
    "range",
    "read",
    "real",
    "references",
    "regexp",
    "rename",
    "replace",
    "right",
    "schema",
    "select",
    "set",
    "show",
    "table",
    "then",
    "to",
    "true",
    "union",
    "unique",
    "update",
    "using",
    "values",
    "varchar",
    "when",
    "where",
    "with",
    "write",
}


class LoaderError(RuntimeError):
    """Base error for expected loader failures."""


class ConfigurationError(LoaderError):
    """Raised when CLI/configuration is unsafe or incomplete."""


class ParseError(LoaderError):
    """Raised when a source dataset cannot be read or normalized."""


class MySQLError(LoaderError):
    """Raised when a database operation fails."""


@dataclass(frozen=True)
class DatasetFile:
    path: Path
    relative_path: str
    file_type: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ColumnPlan:
    source_name: str
    mysql_name: str
    mysql_type: str
    logical_kind: str
    nullable: bool
    ordinal_position: int


@dataclass(frozen=True)
class TablePlan:
    source_file: DatasetFile
    table_name: str
    row_count: int
    columns: tuple[ColumnPlan, ...]


@dataclass(frozen=True)
class LoaderConfig:
    datasets_root: Path
    host: str
    port: int
    user: str
    password: str
    database: str
    mode: str
    batch_size: int
    connect_retries: int
    connect_retry_delay_seconds: float
    connect_timeout_seconds: float
    include_csv: bool
    dry_run: bool


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def emit_log(event: str, message: str, *, level: str = "INFO", **fields: Any) -> None:
    payload = {
        "ts": utc_now_iso(),
        "level": level,
        "event": event,
        "message": message,
        **fields,
    }
    print(json.dumps(payload, default=str, sort_keys=True), flush=True)


def short_hash(value: str, length: int = 8) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def truncate_identifier(base: str, suffix: str, max_length: int = MAX_IDENTIFIER_LENGTH) -> str:
    reserved = len(suffix) + 1
    if reserved >= max_length:
        raise ValueError("suffix is too long for a MySQL identifier")
    stem = base[: max_length - reserved].rstrip("_")
    if not stem:
        stem = "x"
    return f"{stem}_{suffix}"


def sanitize_identifier(
    raw: object,
    *,
    fallback: str,
    max_length: int = MAX_IDENTIFIER_LENGTH,
) -> str:
    text = "" if raw is None else str(raw)
    text = text.strip().lower()
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"{fallback}_{text}"
    if text in MYSQL_RESERVED_WORDS:
        text = f"{text}_{fallback}"
    if len(text) > max_length:
        text = truncate_identifier(text, short_hash(str(raw)), max_length=max_length)
    return text


def unique_identifier(
    base: str,
    *,
    used: set[str],
    salt: str,
    max_length: int = MAX_IDENTIFIER_LENGTH,
) -> str:
    if base not in used:
        used.add(base)
        return base
    suffix = short_hash(salt)
    candidate = truncate_identifier(base, suffix, max_length=max_length)
    counter = 2
    while candidate in used:
        candidate = truncate_identifier(base, short_hash(f"{salt}:{counter}"), max_length=max_length)
        counter += 1
    used.add(candidate)
    return candidate


def quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def validate_database_name(database: str) -> str:
    sanitized = sanitize_identifier(database, fallback="db")
    if sanitized != database:
        raise ConfigurationError(
            f"MySQL database name must already be a safe identifier, got {database!r}",
        )
    return database


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_dataset_files(root: Path, *, include_csv: bool = True) -> list[DatasetFile]:
    if not root.exists():
        raise ConfigurationError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise ConfigurationError(f"Dataset root is not a directory: {root}")

    suffixes = {".parquet", ".zip"}
    if include_csv:
        suffixes.add(".csv")

    files: list[DatasetFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in suffixes:
            continue
        relative_path = path.relative_to(root).as_posix()
        files.append(
            DatasetFile(
                path=path,
                relative_path=relative_path,
                file_type=suffix.lstrip("."),
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
            ),
        )
    return files


def plan_table_names(files: Sequence[DatasetFile]) -> dict[str, str]:
    used = set(METADATA_TABLES)
    table_names: dict[str, str] = {}
    for dataset_file in files:
        if dataset_file.file_type not in {"parquet", "csv"}:
            continue
        base = sanitize_identifier(dataset_file.path.stem, fallback="dataset")
        table_names[dataset_file.relative_path] = unique_identifier(
            base,
            used=used,
            salt=dataset_file.relative_path,
        )
    return table_names


def clean_column_names(source_names: Sequence[object]) -> tuple[str, ...]:
    used = set(INGEST_METADATA_COLUMNS)
    mysql_names: list[str] = []
    for index, source_name in enumerate(source_names):
        base = sanitize_identifier(source_name, fallback=f"col_{index + 1}")
        if base in INGEST_METADATA_COLUMNS:
            base = truncate_identifier(base, "source")
        mysql_names.append(
            unique_identifier(
                base,
                used=used,
                salt=f"{index}:{source_name}",
            ),
        )
    return tuple(mysql_names)


def import_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ConfigurationError(
            "pandas is required to load dataset files. Install the research dependencies.",
        ) from exc
    return pd


def import_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ConfigurationError(
            "pyarrow is required to inspect and stream Parquet files. "
            "Install the research dependencies.",
        ) from exc
    return pa, pq


def mysql_type_for_arrow_type(arrow_type: Any) -> str:
    pa, _ = import_pyarrow()
    if pa.types.is_integer(arrow_type):
        return "BIGINT"
    if pa.types.is_floating(arrow_type) or pa.types.is_decimal(arrow_type):
        return "DOUBLE"
    if pa.types.is_boolean(arrow_type):
        return "BOOLEAN"
    if pa.types.is_date(arrow_type):
        return "DATE"
    if pa.types.is_timestamp(arrow_type):
        return "DATETIME(6)"
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return "JSON"
    if pa.types.is_struct(arrow_type) or pa.types.is_map(arrow_type):
        return "JSON"
    return "TEXT"


def logical_kind_for_mysql_type(mysql_type: str) -> str:
    normalized = mysql_type.upper()
    if normalized == "BIGINT":
        return "integer"
    if normalized == "DOUBLE":
        return "float"
    if normalized == "BOOLEAN":
        return "boolean"
    if normalized == "DATE":
        return "date"
    if normalized == "DATETIME(6)":
        return "datetime"
    if normalized == "JSON":
        return "json"
    return "text"


def csv_string_values(values: Sequence[object]) -> list[str]:
    pd = import_pandas()
    parsed: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        try:
            missing = pd.isna(value)
        except TypeError:
            missing = False
        if isinstance(missing, bool) and missing:
            continue
        text = str(value).strip()
        if text:
            parsed.append(text)
    return parsed


def looks_like_date_column(name: str) -> bool:
    normalized = sanitize_identifier(name, fallback="column")
    return normalized == "date" or normalized.endswith("_date") or normalized.startswith("date_")


def looks_like_datetime_column(name: str) -> bool:
    normalized = sanitize_identifier(name, fallback="column")
    return (
        normalized.endswith("_at")
        or normalized.endswith("_at_utc")
        or normalized.endswith("_at_hkt")
        or normalized.endswith("_time")
        or normalized.endswith("_time_utc")
        or normalized.endswith("_time_hkt")
        or normalized.endswith("_timestamp")
        or normalized.endswith("_timestamp_utc")
        or normalized.endswith("_timestamp_hkt")
        or normalized.endswith("_datetime")
        or normalized in {"timestamp", "datetime", "time"}
    )


def all_parse_as_datetime(values: Sequence[str]) -> bool:
    if not values:
        return False
    pd = import_pandas()
    parsed = pd.to_datetime(list(values), errors="coerce", utc=False, format="mixed")
    return bool(parsed.notna().all())


def all_parse_as_date(values: Sequence[str]) -> bool:
    if not all_parse_as_datetime(values):
        return False
    pd = import_pandas()
    parsed = pd.to_datetime(list(values), errors="coerce", utc=False, format="mixed")
    return all(value.hour == 0 and value.minute == 0 and value.second == 0 for value in parsed)


def all_parse_as_integer(values: Sequence[str]) -> bool:
    return bool(values) and all(re.fullmatch(r"[+-]?\d+", value) for value in values)


def all_parse_as_float(values: Sequence[str]) -> bool:
    if not values:
        return False
    for value in values:
        try:
            float(value)
        except ValueError:
            return False
    return True


def all_parse_as_bool(values: Sequence[str]) -> bool:
    tokens = {"true", "false", "t", "f", "yes", "no", "y", "n", "0", "1"}
    return bool(values) and all(value.lower() in tokens for value in values)


def all_parse_as_json(values: Sequence[str]) -> bool:
    if not values:
        return False
    for value in values:
        text = value.strip()
        if not (text.startswith("{") or text.startswith("[")):
            return False
        try:
            json.loads(text)
        except json.JSONDecodeError:
            return False
    return True


def infer_csv_mysql_type(column_name: str, values: Sequence[object]) -> str:
    string_values = csv_string_values(values)
    if looks_like_datetime_column(column_name) and all_parse_as_datetime(string_values):
        return "DATETIME(6)"
    if looks_like_date_column(column_name) and all_parse_as_date(string_values):
        return "DATE"
    if all_parse_as_bool(string_values):
        return "BOOLEAN"
    if all_parse_as_integer(string_values):
        return "BIGINT"
    if all_parse_as_float(string_values):
        return "DOUBLE"
    if all_parse_as_json(string_values):
        return "JSON"
    return "TEXT"


def plan_parquet_table(dataset_file: DatasetFile, table_name: str) -> TablePlan:
    pa, pq = import_pyarrow()
    try:
        parquet_file = pq.ParquetFile(dataset_file.path)
    except Exception as exc:
        raise ParseError(f"Could not inspect Parquet file {dataset_file.path}: {exc}") from exc

    schema = parquet_file.schema_arrow
    source_names = [field.name for field in schema]
    mysql_names = clean_column_names(source_names)
    columns: list[ColumnPlan] = []
    for index, field in enumerate(schema):
        mysql_type = mysql_type_for_arrow_type(field.type)
        columns.append(
            ColumnPlan(
                source_name=field.name,
                mysql_name=mysql_names[index],
                mysql_type=mysql_type,
                logical_kind=logical_kind_for_mysql_type(mysql_type),
                nullable=field.nullable,
                ordinal_position=index + 1,
            ),
        )
    metadata = parquet_file.metadata
    row_count = 0 if metadata is None else metadata.num_rows
    return TablePlan(
        source_file=dataset_file,
        table_name=table_name,
        row_count=row_count,
        columns=tuple(columns),
    )


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def plan_csv_table(dataset_file: DatasetFile, table_name: str) -> TablePlan:
    pd = import_pandas()
    try:
        sample = pd.read_csv(dataset_file.path, nrows=1000)
    except pd.errors.EmptyDataError:
        sample = pd.DataFrame()
    except Exception as exc:
        raise ParseError(f"Could not inspect CSV file {dataset_file.path}: {exc}") from exc

    source_names = [str(column) for column in sample.columns]
    mysql_names = clean_column_names(source_names)
    columns: list[ColumnPlan] = []
    for index, source_name in enumerate(source_names):
        mysql_type = infer_csv_mysql_type(source_name, sample[source_name].tolist())
        columns.append(
            ColumnPlan(
                source_name=source_name,
                mysql_name=mysql_names[index],
                mysql_type=mysql_type,
                logical_kind=logical_kind_for_mysql_type(mysql_type),
                nullable=True,
                ordinal_position=index + 1,
            ),
        )
    try:
        row_count = count_csv_rows(dataset_file.path)
    except Exception as exc:
        raise ParseError(f"Could not count CSV rows in {dataset_file.path}: {exc}") from exc
    return TablePlan(
        source_file=dataset_file,
        table_name=table_name,
        row_count=row_count,
        columns=tuple(columns),
    )


def plan_table(dataset_file: DatasetFile, table_name: str) -> TablePlan:
    if dataset_file.file_type == "parquet":
        return plan_parquet_table(dataset_file, table_name)
    if dataset_file.file_type == "csv":
        return plan_csv_table(dataset_file, table_name)
    raise ParseError(f"Unsupported row-table file type: {dataset_file.file_type}")


def build_create_table_sql(table_plan: TablePlan) -> str:
    column_lines = [
        "  `ingest_row_id` BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY",
        "  `ingest_source_path` TEXT NOT NULL",
        "  `ingest_source_sha256` CHAR(64) NOT NULL",
        "  `ingested_at_utc` DATETIME(6) NOT NULL",
    ]
    for column in table_plan.columns:
        column_lines.append(f"  {quote_identifier(column.mysql_name)} {column.mysql_type} NULL")
    joined_columns = ",\n".join(column_lines)
    return (
        f"CREATE TABLE {quote_identifier(table_plan.table_name)} (\n"
        f"{joined_columns}\n"
        ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci"
    )


def metadata_table_sql() -> list[str]:
    suffix = "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci"
    return [
        f"""
        CREATE TABLE IF NOT EXISTS dataset_ingest_runs (
          run_id CHAR(36) NOT NULL PRIMARY KEY,
          started_at_utc DATETIME(6) NOT NULL,
          completed_at_utc DATETIME(6) NULL,
          status VARCHAR(32) NOT NULL,
          datasets_root TEXT NOT NULL,
          mode VARCHAR(16) NOT NULL,
          table_count BIGINT NULL,
          row_count BIGINT NULL,
          error_message LONGTEXT NULL
        ) {suffix}
        """,
        f"""
        CREATE TABLE IF NOT EXISTS dataset_files (
          dataset_file_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
          run_id CHAR(36) NOT NULL,
          relative_path TEXT NOT NULL,
          file_type VARCHAR(32) NOT NULL,
          file_bytes BIGINT NOT NULL,
          sha256 CHAR(64) NOT NULL,
          table_name VARCHAR(64) NULL,
          load_status VARCHAR(32) NOT NULL,
          row_count BIGINT NULL,
          registered_at_utc DATETIME(6) NOT NULL,
          updated_at_utc DATETIME(6) NULL,
          INDEX idx_dataset_files_run_id (run_id),
          INDEX idx_dataset_files_table_name (table_name)
        ) {suffix}
        """,
        f"""
        CREATE TABLE IF NOT EXISTS dataset_tables (
          dataset_table_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
          run_id CHAR(36) NOT NULL,
          table_name VARCHAR(64) NOT NULL,
          source_relative_path TEXT NOT NULL,
          source_type VARCHAR(32) NOT NULL,
          row_count BIGINT NOT NULL,
          column_count BIGINT NOT NULL,
          created_at_utc DATETIME(6) NOT NULL,
          INDEX idx_dataset_tables_run_id (run_id),
          UNIQUE KEY uq_dataset_tables_run_table (run_id, table_name)
        ) {suffix}
        """,
        f"""
        CREATE TABLE IF NOT EXISTS dataset_columns (
          dataset_column_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
          run_id CHAR(36) NOT NULL,
          table_name VARCHAR(64) NOT NULL,
          source_column_name TEXT NOT NULL,
          mysql_column_name VARCHAR(64) NOT NULL,
          ordinal_position BIGINT NOT NULL,
          mysql_type VARCHAR(64) NOT NULL,
          nullable BOOLEAN NOT NULL,
          INDEX idx_dataset_columns_run_table (run_id, table_name)
        ) {suffix}
        """,
    ]


def import_mysql_connector() -> Any:
    try:
        import mysql.connector
    except ModuleNotFoundError as exc:
        raise ConfigurationError(
            "mysql-connector-python is required for MySQL loading. "
            "Install dependencies with the repo environment first.",
        ) from exc
    return mysql.connector


@contextmanager
def mysql_connection(config: LoaderConfig, *, database: str | None) -> Iterator[Any]:
    mysql = import_mysql_connector()
    connection = None
    location = f"{config.host}:{config.port}"
    last_error: Exception | None = None
    for attempt in range(1, config.connect_retries + 1):
        try:
            connection = mysql.connect(
                host=config.host,
                port=config.port,
                user=config.user,
                password=config.password,
                database=database,
                charset="utf8mb4",
                autocommit=False,
                connection_timeout=math.ceil(config.connect_timeout_seconds),
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt >= config.connect_retries:
                break
            emit_log(
                "mysql_connect_retry",
                "MySQL connection attempt failed; retrying",
                level="WARNING",
                location=location,
                database=database,
                attempt=attempt,
                max_attempts=config.connect_retries,
                retry_delay_seconds=config.connect_retry_delay_seconds,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            time.sleep(config.connect_retry_delay_seconds)
    if connection is None:
        raise MySQLError(f"Could not connect to MySQL at {location}: {last_error}") from last_error
    try:
        yield connection
    finally:
        connection.close()


def create_database(config: LoaderConfig) -> None:
    database = validate_database_name(config.database)
    with mysql_connection(config, database=None) as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {quote_identifier(database)} "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci",
            )
            connection.commit()
        finally:
            cursor.close()


def ensure_metadata_tables(cursor: Any) -> None:
    for statement in metadata_table_sql():
        cursor.execute(statement)


def insert_ingest_run(cursor: Any, run_id: str, config: LoaderConfig) -> None:
    cursor.execute(
        """
        INSERT INTO dataset_ingest_runs
          (run_id, started_at_utc, status, datasets_root, mode)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (run_id, utc_now(), "running", str(config.datasets_root), config.mode),
    )


def finish_ingest_run(
    cursor: Any,
    run_id: str,
    *,
    status: str,
    table_count: int | None,
    row_count: int | None,
    error_message: str | None = None,
) -> None:
    cursor.execute(
        """
        UPDATE dataset_ingest_runs
        SET completed_at_utc = %s,
            status = %s,
            table_count = %s,
            row_count = %s,
            error_message = %s
        WHERE run_id = %s
        """,
        (utc_now(), status, table_count, row_count, error_message, run_id),
    )


def insert_dataset_file(
    cursor: Any,
    run_id: str,
    dataset_file: DatasetFile,
    *,
    table_name: str | None,
    load_status: str,
    row_count: int | None,
) -> int:
    cursor.execute(
        """
        INSERT INTO dataset_files
          (run_id, relative_path, file_type, file_bytes, sha256, table_name,
           load_status, row_count, registered_at_utc)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            run_id,
            dataset_file.relative_path,
            dataset_file.file_type,
            dataset_file.bytes,
            dataset_file.sha256,
            table_name,
            load_status,
            row_count,
            utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def update_dataset_file_status(
    cursor: Any,
    dataset_file_id: int,
    *,
    load_status: str,
    row_count: int | None,
) -> None:
    cursor.execute(
        """
        UPDATE dataset_files
        SET load_status = %s,
            row_count = %s,
            updated_at_utc = %s
        WHERE dataset_file_id = %s
        """,
        (load_status, row_count, utc_now(), dataset_file_id),
    )


def insert_table_metadata(cursor: Any, run_id: str, table_plan: TablePlan) -> None:
    cursor.execute(
        """
        INSERT INTO dataset_tables
          (run_id, table_name, source_relative_path, source_type, row_count,
           column_count, created_at_utc)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            run_id,
            table_plan.table_name,
            table_plan.source_file.relative_path,
            table_plan.source_file.file_type,
            table_plan.row_count,
            len(table_plan.columns),
            utc_now(),
        ),
    )
    rows = [
        (
            run_id,
            table_plan.table_name,
            column.source_name,
            column.mysql_name,
            column.ordinal_position,
            column.mysql_type,
            column.nullable,
        )
        for column in table_plan.columns
    ]
    if rows:
        cursor.executemany(
            """
            INSERT INTO dataset_columns
              (run_id, table_name, source_column_name, mysql_column_name,
               ordinal_position, mysql_type, nullable)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    pd = import_pandas()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, bool) and missing


def normalize_datetime(value: Any) -> datetime | None:
    if is_missing_value(value):
        return None
    pd = import_pandas()
    parsed = pd.to_datetime(value, errors="raise", utc=False)
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    if hasattr(parsed, "to_pydatetime"):
        return parsed.to_pydatetime()
    if isinstance(parsed, datetime):
        return parsed
    raise ParseError(f"Could not normalize datetime value {value!r}")


def normalize_date(value: Any) -> date | None:
    if is_missing_value(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = normalize_datetime(value)
    return None if parsed is None else parsed.date()


def normalize_bool(value: Any) -> bool | None:
    if is_missing_value(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    raise ParseError(f"Could not normalize boolean value {value!r}")


def normalize_json(value: Any) -> str | None:
    if is_missing_value(value):
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), default=str)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def normalize_value(value: Any, logical_kind: str) -> Any:
    value = maybe_numpy_scalar(value)
    if is_missing_value(value):
        return None
    if isinstance(value, Decimal):
        return float(value)
    if logical_kind == "datetime":
        return normalize_datetime(value)
    if logical_kind == "date":
        return normalize_date(value)
    if logical_kind == "boolean":
        return normalize_bool(value)
    if logical_kind == "json":
        return normalize_json(value)
    if logical_kind == "integer":
        return int(value)
    if logical_kind == "float":
        return float(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return str(value) if not isinstance(value, str) else value


def maybe_numpy_scalar(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray, str)):
        try:
            return value.item()
        except (AttributeError, ValueError, TypeError):
            return value
    return value


def normalized_rows_from_frame(frame: Any, table_plan: TablePlan, ingested_at: datetime) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    source_columns = [column.source_name for column in table_plan.columns]
    for raw_values in frame[source_columns].itertuples(index=False, name=None):
        source_values = [
            normalize_value(maybe_numpy_scalar(value), column.logical_kind)
            for value, column in zip(raw_values, table_plan.columns, strict=True)
        ]
        rows.append(
            (
                table_plan.source_file.relative_path,
                table_plan.source_file.sha256,
                ingested_at,
                *source_values,
            ),
        )
    return rows


def parquet_frame_chunks(table_plan: TablePlan, batch_size: int) -> Iterator[Any]:
    _, pq = import_pyarrow()
    parquet_file = pq.ParquetFile(table_plan.source_file.path)
    source_columns = [column.source_name for column in table_plan.columns]
    for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=source_columns):
        yield record_batch.to_pandas()


def csv_frame_chunks(table_plan: TablePlan, batch_size: int) -> Iterator[Any]:
    pd = import_pandas()
    if not table_plan.columns and table_plan.row_count == 0:
        return
    yield from pd.read_csv(table_plan.source_file.path, chunksize=batch_size)


def frame_chunks(table_plan: TablePlan, batch_size: int) -> Iterator[Any]:
    if table_plan.source_file.file_type == "parquet":
        yield from parquet_frame_chunks(table_plan, batch_size)
        return
    if table_plan.source_file.file_type == "csv":
        yield from csv_frame_chunks(table_plan, batch_size)
        return
    raise ParseError(f"Unsupported row-table file type: {table_plan.source_file.file_type}")


def insert_rows(cursor: Any, table_plan: TablePlan, rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return
    column_names = [
        "ingest_source_path",
        "ingest_source_sha256",
        "ingested_at_utc",
        *[column.mysql_name for column in table_plan.columns],
    ]
    placeholders = ", ".join(["%s"] * len(column_names))
    quoted_columns = ", ".join(quote_identifier(column_name) for column_name in column_names)
    sql = (
        f"INSERT INTO {quote_identifier(table_plan.table_name)} "
        f"({quoted_columns}) VALUES ({placeholders})"
    )
    cursor.executemany(sql, rows)


def count_table_rows(cursor: Any, table_name: str) -> int:
    cursor.execute(f"SELECT COUNT(*) FROM {quote_identifier(table_name)}")
    row = cursor.fetchone()
    return int(row[0])


def load_table(connection: Any, run_id: str, table_plan: TablePlan, batch_size: int) -> int:
    cursor = connection.cursor()
    dataset_file_id = insert_dataset_file(
        cursor,
        run_id,
        table_plan.source_file,
        table_name=table_plan.table_name,
        load_status="loading",
        row_count=None,
    )
    connection.commit()
    try:
        emit_log(
            "table_create_start",
            "Dropping and recreating data table",
            table=table_plan.table_name,
            source=table_plan.source_file.relative_path,
            source_rows=table_plan.row_count,
            source_columns=len(table_plan.columns),
        )
        cursor.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_plan.table_name)}")
        cursor.execute(build_create_table_sql(table_plan))
        connection.commit()

        inserted = 0
        ingested_at = utc_now()
        for chunk_number, frame in enumerate(frame_chunks(table_plan, batch_size), start=1):
            rows = normalized_rows_from_frame(frame, table_plan, ingested_at)
            insert_rows(cursor, table_plan, rows)
            connection.commit()
            inserted += len(rows)
            emit_log(
                "table_load_progress",
                "Inserted source chunk",
                table=table_plan.table_name,
                chunk=chunk_number,
                inserted_rows=inserted,
                expected_rows=table_plan.row_count,
            )

        actual_rows = count_table_rows(cursor, table_plan.table_name)
        if actual_rows != table_plan.row_count:
            raise MySQLError(
                f"Row-count mismatch for {table_plan.table_name}: "
                f"source={table_plan.row_count}, mysql={actual_rows}",
            )
        update_dataset_file_status(
            cursor,
            dataset_file_id,
            load_status="loaded",
            row_count=actual_rows,
        )
        insert_table_metadata(cursor, run_id, table_plan)
        connection.commit()
        emit_log(
            "table_load_done",
            "Loaded and verified data table",
            table=table_plan.table_name,
            rows=actual_rows,
            source=table_plan.source_file.relative_path,
        )
        return actual_rows
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()


def register_zip_file(connection: Any, run_id: str, dataset_file: DatasetFile) -> None:
    cursor = connection.cursor()
    try:
        insert_dataset_file(
            cursor,
            run_id,
            dataset_file,
            table_name=None,
            load_status="registered_only",
            row_count=None,
        )
        connection.commit()
        emit_log(
            "file_registered",
            "Registered ZIP payload without loading rows",
            source=dataset_file.relative_path,
            bytes=dataset_file.bytes,
            sha256=dataset_file.sha256,
        )
    finally:
        cursor.close()


def plan_all_tables(files: Sequence[DatasetFile]) -> list[TablePlan]:
    table_names = plan_table_names(files)
    plans: list[TablePlan] = []
    for dataset_file in files:
        table_name = table_names.get(dataset_file.relative_path)
        if table_name is None:
            continue
        emit_log(
            "plan_file",
            "Inspecting source table schema",
            source=dataset_file.relative_path,
            file_type=dataset_file.file_type,
            table=table_name,
        )
        plans.append(plan_table(dataset_file, table_name))
    return plans


def summarize_plan(files: Sequence[DatasetFile], table_plans: Sequence[TablePlan]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for dataset_file in files:
        by_type[dataset_file.file_type] = by_type.get(dataset_file.file_type, 0) + 1
    return {
        "dataset_files": len(files),
        "files_by_type": by_type,
        "tables": len(table_plans),
        "source_rows": sum(plan.row_count for plan in table_plans),
        "zip_payloads_registered_only": sum(1 for file in files if file.file_type == "zip"),
        "table_preview": [
            {
                "table": plan.table_name,
                "source": plan.source_file.relative_path,
                "rows": plan.row_count,
                "columns": len(plan.columns),
            }
            for plan in table_plans[:10]
        ],
    }


def run_loader(config: LoaderConfig) -> dict[str, Any]:
    emit_log(
        "run_start",
        "Starting dataset MySQL load",
        datasets_root=str(config.datasets_root),
        host=config.host,
        port=config.port,
        database=config.database,
        mode=config.mode,
        batch_size=config.batch_size,
        connect_retries=config.connect_retries,
        connect_retry_delay_seconds=config.connect_retry_delay_seconds,
        connect_timeout_seconds=config.connect_timeout_seconds,
        include_csv=config.include_csv,
        dry_run=config.dry_run,
    )
    files = discover_dataset_files(config.datasets_root, include_csv=config.include_csv)
    table_plans = plan_all_tables(files)
    plan_summary = summarize_plan(files, table_plans)
    emit_log("plan_done", "Prepared dataset load plan", **plan_summary)

    if config.dry_run:
        print(json.dumps({"status": "dry_run", **plan_summary}, indent=2, sort_keys=True))
        return {"status": "dry_run", **plan_summary}

    create_database(config)
    run_id = str(uuid.uuid4())
    table_rows_loaded = 0
    tables_loaded = 0
    with mysql_connection(config, database=config.database) as connection:
        cursor = connection.cursor()
        try:
            ensure_metadata_tables(cursor)
            insert_ingest_run(cursor, run_id, config)
            connection.commit()
        finally:
            cursor.close()

        try:
            for dataset_file in files:
                if dataset_file.file_type == "zip":
                    register_zip_file(connection, run_id, dataset_file)
                    continue
                table_plan = next(
                    plan for plan in table_plans if plan.source_file.relative_path == dataset_file.relative_path
                )
                rows = load_table(connection, run_id, table_plan, config.batch_size)
                table_rows_loaded += rows
                tables_loaded += 1
            cursor = connection.cursor()
            try:
                finish_ingest_run(
                    cursor,
                    run_id,
                    status="succeeded",
                    table_count=tables_loaded,
                    row_count=table_rows_loaded,
                )
                connection.commit()
            finally:
                cursor.close()
        except Exception as exc:
            cursor = connection.cursor()
            try:
                finish_ingest_run(
                    cursor,
                    run_id,
                    status="failed",
                    table_count=tables_loaded,
                    row_count=table_rows_loaded,
                    error_message=str(exc),
                )
                connection.commit()
            finally:
                cursor.close()
            raise

    result = {
        "status": "succeeded",
        "run_id": run_id,
        "tables_loaded": tables_loaded,
        "rows_loaded": table_rows_loaded,
        **plan_summary,
    }
    emit_log("run_done", "Dataset MySQL load completed", **result)
    return result


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load every normalized dataset file under data/datasets into MySQL.",
    )
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--host", default=os.environ.get("MYSQL_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MYSQL_PORT", DEFAULT_PORT)))
    parser.add_argument("--user", default=os.environ.get("MYSQL_USER", DEFAULT_USER))
    parser.add_argument("--password", default=os.environ.get("MYSQL_PASSWORD", DEFAULT_PASSWORD))
    parser.add_argument("--database", default=os.environ.get("MYSQL_DATABASE", DEFAULT_DATABASE))
    parser.add_argument("--mode", choices=["replace"], default="replace")
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--connect-retries", type=positive_int, default=30)
    parser.add_argument("--connect-retry-delay-seconds", type=positive_float, default=2.0)
    parser.add_argument("--connect-timeout-seconds", type=positive_float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-csv", dest="include_csv", action="store_true", default=True)
    parser.add_argument("--no-include-csv", dest="include_csv", action="store_false")
    return parser


def config_from_args(args: argparse.Namespace) -> LoaderConfig:
    datasets_root = args.datasets_root.resolve()
    if args.port <= 0 or args.port > 65535:
        raise ConfigurationError("--port must be between 1 and 65535")
    if not args.user:
        raise ConfigurationError("--user must not be empty")
    validate_database_name(args.database)
    return LoaderConfig(
        datasets_root=datasets_root,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        mode=args.mode,
        batch_size=args.batch_size,
        connect_retries=args.connect_retries,
        connect_retry_delay_seconds=args.connect_retry_delay_seconds,
        connect_timeout_seconds=args.connect_timeout_seconds,
        include_csv=args.include_csv,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        run_loader(config)
        return 0
    except LoaderError as exc:
        emit_log(
            "process_error",
            str(exc),
            level="ERROR",
            error_type=type(exc).__name__,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
