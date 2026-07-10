"""Source-file and attribute reconciliation against the audit contracts."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .contracts import AuditBundle, load_contract_tables
from .hashing import sha256_file

OBJECT_ACTIONS = {
    "REGISTER_OBJECT_METADATA",
    "REGISTER_OBJECT_METADATA_AND_REBUILD_INVENTORY",
    "REGISTER_RESEARCH_ARTIFACT",
    "REGISTER_OBJECT_ONLY",
}


@dataclass(frozen=True)
class SourceReconciliation:
    dataset_id: str
    source_file: str
    file_type: str
    db_action: str
    db_layer: str
    model_status: str
    priority: str
    expected_rows: int
    observed_rows: int
    expected_bytes: int
    observed_bytes: int
    expected_attributes: int
    observed_attributes: int
    physical_sha256: str
    repository_uri: str
    original_local_path: str
    data_min: str
    data_max: str
    metadata_min: str | None
    metadata_max: str | None
    disposition: str
    status: str
    exception: str


def portable_uri(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/")
    return f"repo://data/datasets/{normalized}"


def sanitize_identifier(value: str, *, fallback: str = "table") -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    if not sanitized:
        sanitized = fallback
    if sanitized[0].isdigit():
        sanitized = f"{fallback}_{sanitized}"
    return sanitized[:63]


def table_name_for_source(source_file: str) -> str:
    path = Path(source_file)
    dataset = sanitize_identifier(path.parts[0] if len(path.parts) > 1 else "root", fallback="ds")
    stem = sanitize_identifier(path.stem, fallback="source")
    return f"{dataset}__{stem}"[:63]


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def observed_table_shape(path: Path, file_type: str) -> tuple[int, int]:
    if file_type == "parquet":
        parquet_file = pq.ParquetFile(path)
        return int(parquet_file.metadata.num_rows), len(parquet_file.schema_arrow)
    if file_type == "csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return 0, 0
        return csv_row_count(path), len(header)
    return 0, 0


def load_profile_index(profile_json: Path) -> dict[str, dict[str, Any]]:
    if not profile_json.exists():
        return {}
    profile = json.loads(profile_json.read_text(encoding="utf-8"))
    tables: dict[str, dict[str, Any]] = {}
    for dataset in profile.get("datasets", []):
        for table in dataset.get("tables", []):
            tables[str(table["source_file"])] = table
    return tables


def reconcile_sources(
    bundle: AuditBundle,
    *,
    datasets_root: Path,
    profile_json: Path,
) -> list[SourceReconciliation]:
    contracts = load_contract_tables(bundle)
    profile_index = load_profile_index(profile_json)
    results: list[SourceReconciliation] = []
    for row in contracts["tables"]:
        relative = row["source_file"]
        path = datasets_root / relative
        exists = path.exists()
        observed_rows, observed_attributes = (
            observed_table_shape(path, row["file_type"]) if exists else (0, 0)
        )
        observed_bytes = path.stat().st_size if exists else 0
        physical_sha256 = sha256_file(path) if exists else ""
        profile_table = profile_index.get(relative, {})
        metadata_range = profile_table.get("metadata_timestamp_range", {})
        expected_rows = int(row["row_count"])
        expected_bytes = int(row["byte_size"])
        expected_attributes = int(row["attribute_count"])
        mismatches = []
        if not exists:
            mismatches.append("source file missing")
        if observed_rows != expected_rows:
            mismatches.append(f"row_count {observed_rows} != {expected_rows}")
        if observed_attributes != expected_attributes:
            mismatches.append(f"attribute_count {observed_attributes} != {expected_attributes}")
        if observed_bytes != expected_bytes:
            mismatches.append(f"byte_size {observed_bytes} != {expected_bytes}")

        if row["db_action"] == "SKIP_DUPLICATE_FORMAT":
            disposition = "SKIPPED_DUPLICATE_FORMAT"
        elif row["db_action"] in OBJECT_ACTIONS:
            disposition = "REGISTER_OBJECT_OR_ARTIFACT"
        elif row["db_action"] in {"RECOMPUTE_BEFORE_CANONICAL_LOAD", "REBUILD_AFTER_RAW_FIX"}:
            disposition = "BLOCKED_PENDING_REBUILD"
        else:
            disposition = "LOAD_OR_REGISTER_BY_CONTRACT"

        results.append(
            SourceReconciliation(
                dataset_id=row["dataset_id"],
                source_file=relative,
                file_type=row["file_type"],
                db_action=row["db_action"],
                db_layer=row["db_layer"],
                model_status=row["model_status"],
                priority=row["priority"],
                expected_rows=expected_rows,
                observed_rows=observed_rows,
                expected_bytes=expected_bytes,
                observed_bytes=observed_bytes,
                expected_attributes=expected_attributes,
                observed_attributes=observed_attributes,
                physical_sha256=physical_sha256,
                repository_uri=portable_uri(relative),
                original_local_path=str(path.resolve()) if exists else "",
                data_min=row["data_min"],
                data_max=row["data_max"],
                metadata_min=metadata_range.get("min"),
                metadata_max=metadata_range.get("max"),
                disposition=disposition,
                status="PASS" if not mismatches else "FAIL",
                exception="; ".join(mismatches),
            ),
        )
    return results


def attribute_reconciliation_rows(bundle: AuditBundle) -> list[dict[str, str]]:
    rows = load_contract_tables(bundle)["attributes"]
    reconciled: list[dict[str, str]] = []
    for row in rows:
        storage = row["storage_decision"]
        quality = row["quality_action"]
        if storage == "DROP" or quality.startswith(("QUARANTINE", "BLOCK")):
            status = "BLOCKED_OR_QUARANTINE_CONTRACTED"
        else:
            status = "CONTRACTED"
        reconciled.append({**row, "reconciliation_status": status})
    return reconciled
