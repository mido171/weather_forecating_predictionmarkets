"""Audit-bundle contracts and validation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import sha256_file

REQUIRED_AUDIT_FILES = (
    "README.md",
    "AUDIT_SUMMARY.json",
    "HKG_TMAX_DATASET_DB_AND_MODEL_VALUE_AUDIT.md",
    "HKG_TMAX_DATASET_DECISION_MATRIX.csv",
    "HKG_TMAX_TABLE_DECISIONS_ALL_52.csv",
    "HKG_TMAX_ATTRIBUTE_DECISIONS_ALL_1869.csv",
    "HKG_TMAX_DATA_QUALITY_ISSUES.csv",
    "HKG_TMAX_ISD_STATION_DOSSIER_36.csv",
    "HKG_TMAX_DB_SCHEMA_BLUEPRINT.sql",
)

EXPECTED_COUNTS = {
    "dataset_count": 13,
    "table_decision_count": 52,
    "attribute_decision_count": 1869,
    "quality_issue_count": 22,
    "station_dossier_count": 36,
}


@dataclass(frozen=True)
class AuditBundle:
    root: Path
    summary: dict[str, Any]
    read_order_bytes: dict[str, int]

    @property
    def generated_at_utc(self) -> str:
        return str(self.summary["generated_at_utc"])

    @property
    def profile_summary(self) -> dict[str, Any]:
        return dict(self.summary["profile_summary"])


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_audit_bundle(root: Path) -> AuditBundle:
    root = root.resolve()
    missing = [name for name in REQUIRED_AUDIT_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Audit bundle is missing required files: {missing}")

    read_order_bytes = {name: len((root / name).read_bytes()) for name in REQUIRED_AUDIT_FILES}
    summary = json.loads((root / "AUDIT_SUMMARY.json").read_text(encoding="utf-8"))

    for name, expected in summary["files"].items():
        path = root / name
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        if actual_hash != expected["sha256"] or actual_size != expected["bytes"]:
            raise ValueError(
                f"Audit file verification failed for {name}: "
                f"{actual_hash}/{actual_size} != {expected['sha256']}/{expected['bytes']}",
            )

    for key, expected_count in EXPECTED_COUNTS.items():
        actual_count = int(summary[key])
        if actual_count != expected_count:
            raise ValueError(f"Audit count mismatch for {key}: {actual_count} != {expected_count}")

    profile = summary["profile_summary"]
    expected_profile = {
        "datasets_profiled": 13,
        "files_profiled": 52,
        "row_tables_profiled": 51,
        "row_table_rows_total": 7_219_745,
        "attributes_profiled": 1869,
    }
    for key, expected_count in expected_profile.items():
        actual_count = int(profile[key])
        if actual_count != expected_count:
            raise ValueError(f"Profile count mismatch for {key}: {actual_count} != {expected_count}")

    csv_counts = {
        "dataset_count": len(read_csv_rows(root / "HKG_TMAX_DATASET_DECISION_MATRIX.csv")),
        "table_decision_count": len(read_csv_rows(root / "HKG_TMAX_TABLE_DECISIONS_ALL_52.csv")),
        "attribute_decision_count": len(
            read_csv_rows(root / "HKG_TMAX_ATTRIBUTE_DECISIONS_ALL_1869.csv"),
        ),
        "quality_issue_count": len(read_csv_rows(root / "HKG_TMAX_DATA_QUALITY_ISSUES.csv")),
        "station_dossier_count": len(read_csv_rows(root / "HKG_TMAX_ISD_STATION_DOSSIER_36.csv")),
    }
    for key, actual_count in csv_counts.items():
        if actual_count != int(summary[key]):
            raise ValueError(f"CSV count mismatch for {key}: {actual_count} != {summary[key]}")

    return AuditBundle(root=root, summary=summary, read_order_bytes=read_order_bytes)


def load_contract_tables(bundle: AuditBundle) -> dict[str, list[dict[str, str]]]:
    root = bundle.root
    return {
        "datasets": read_csv_rows(root / "HKG_TMAX_DATASET_DECISION_MATRIX.csv"),
        "tables": read_csv_rows(root / "HKG_TMAX_TABLE_DECISIONS_ALL_52.csv"),
        "attributes": read_csv_rows(root / "HKG_TMAX_ATTRIBUTE_DECISIONS_ALL_1869.csv"),
        "quality_issues": read_csv_rows(root / "HKG_TMAX_DATA_QUALITY_ISSUES.csv"),
        "stations": read_csv_rows(root / "HKG_TMAX_ISD_STATION_DOSSIER_36.csv"),
    }
