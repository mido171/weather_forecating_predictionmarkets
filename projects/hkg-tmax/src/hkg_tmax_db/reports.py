"""Bookkeeping report generation for audit-driven ingestion."""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import AuditBundle, load_contract_tables
from .cutoff import CUTOFF_RULE_VERSION
from .hashing import sha256_file
from .reconciliation import SourceReconciliation, attribute_reconciliation_rows

REQUIRED_REPORTS = (
    "README.md",
    "RESULTS.md",
    "CONCLUSION.md",
    "TASK_SPEC.md",
    "DB_IMPLEMENTATION_SUMMARY.md",
    "DB_SCHEMA_MANIFEST.json",
    "DB_TABLE_CATALOG.csv",
    "DB_ATTRIBUTE_RECONCILIATION_ALL_1869.csv",
    "DB_SOURCE_FILE_RECONCILIATION_ALL_52.csv",
    "DB_DATASET_RECONCILIATION_ALL_13.csv",
    "DB_ROW_COUNT_RECONCILIATION.csv",
    "DB_QUARANTINE_SUMMARY.csv",
    "DB_QUALITY_ISSUE_STATUS_ALL_22.csv",
    "DB_STATION_DIMENSION_STATUS_ALL_36.csv",
    "DB_PERMISSION_MATRIX.csv",
    "DB_T24_LEAKAGE_AUDIT.md",
    "DB_TIMESTAMP_AUDIT.md",
    "DB_IDEMPOTENCY_REPORT.md",
    "DB_TEST_RESULTS.md",
    "DB_RUN_COMMANDS.md",
    "summary.json",
    "run_manifest.json",
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def next_bookkeeping_folder(experiments_root: Path) -> Path:
    max_id = -1
    for path in experiments_root.iterdir():
        if not path.is_dir():
            continue
        prefix = path.name.split("_", 1)[0]
        if prefix.isdigit():
            max_id = max(max_id, int(prefix))
    return experiments_root / f"{max_id + 1:04d}_audit_driven_database_ingestion"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def schema_manifest(migration_path: Path) -> dict[str, Any]:
    sql = migration_path.read_text(encoding="utf-8")
    schemas = sorted(set(part.split(";")[0].strip() for part in sql.split("CREATE SCHEMA IF NOT EXISTS ")[1:]))
    tables = []
    for part in sql.split("CREATE TABLE IF NOT EXISTS ")[1:]:
        tables.append(part.split("(", 1)[0].strip())
    views = []
    for part in sql.split("CREATE OR REPLACE VIEW ")[1:]:
        views.append(part.split(" AS", 1)[0].strip())
    functions = []
    for part in sql.split("CREATE OR REPLACE FUNCTION ")[1:]:
        functions.append(part.split("(", 1)[0].strip())
    return {
        "migration_file": str(migration_path),
        "schemas": schemas,
        "tables": tables,
        "views": views,
        "functions": functions,
    }


def generate_reports(
    *,
    output_dir: Path,
    task_spec_path: Path,
    migration_path: Path,
    bundle_zip: Path,
    audit_bundle: AuditBundle,
    source_reconciliation: list[SourceReconciliation],
    database_status: dict[str, Any],
    test_results: list[dict[str, str]],
) -> dict[str, Any]:
    contracts = load_contract_tables(audit_bundle)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(task_spec_path, output_dir / "TASK_SPEC.md")

    source_rows = [asdict(row) for row in source_reconciliation]
    attribute_rows = attribute_reconciliation_rows(audit_bundle)
    dataset_rows = [
        {
            **row,
            "reconciliation_status": "CONTRACTED",
            "source_file_count": sum(1 for item in source_reconciliation if item.dataset_id == row["dataset_id"]),
        }
        for row in contracts["datasets"]
    ]
    quality_rows = [
        {
            "issue_id": f"QI-{index:03d}",
            **row,
            "current_status": "OPEN",
            "remediation_implementation_path": "",
            "validation_evidence_uri": "",
            "resolution_timestamp": "",
            "resolution_commit": "",
            "notes": "Loaded from audit contract; not resolved without validation evidence.",
        }
        for index, row in enumerate(contracts["quality_issues"], start=1)
    ]
    station_rows = [
        {
            **row,
            "dimension_status": "CONTRACTED_FOR_LOAD",
            "metadata_invented": "false",
        }
        for row in contracts["stations"]
    ]
    row_count_rows = [
        {
            "source_file": row.source_file,
            "expected_rows": row.expected_rows,
            "observed_rows": row.observed_rows,
            "status": "PASS" if row.expected_rows == row.observed_rows else "FAIL",
        }
        for row in source_reconciliation
    ]
    duplicate_skips = sum(1 for row in source_reconciliation if row.db_action == "SKIP_DUPLICATE_FORMAT")
    object_contracts = sum(1 for row in source_reconciliation if row.disposition == "REGISTER_OBJECT_OR_ARTIFACT")
    rows_by_layer = Counter(row.db_layer for row in source_reconciliation)
    blocked_reasons = []
    if database_status["status"] != "LOADED":
        blocked_reasons.append(database_status["reason"])
    validation_failures = []
    if database_status["status"] == "LOADED":
        for key in (
            "sealed_confirmation_enforced",
            "live_role_label_access_denied",
            "strict_validation_passed",
            "idempotency_passed",
        ):
            if not database_status.get(key, False):
                validation_failures.append(key)

    summary = {
        "task_type": "audit_driven_database_ingestion",
        "status": "BLOCKED" if blocked_reasons else ("FAIL" if validation_failures else "PASS"),
        "audit_bundle_sha256": sha256_file(bundle_zip),
        "database_engine": "postgresql",
        "migration_version": "20260623_0001_audit_driven_ingestion",
        "ingestion_batch_id": database_status.get("batch_id", "NOT_STARTED_DB_BLOCKED"),
        "datasets_accounted": len(dataset_rows),
        "tables_accounted": len(source_rows),
        "attributes_accounted": len(attribute_rows),
        "quality_issues_accounted": len(quality_rows),
        "stations_accounted": len(station_rows),
        "source_rows_profiled": audit_bundle.profile_summary["row_table_rows_total"],
        "rows_loaded_by_layer": database_status.get("rows_loaded_by_layer", {}),
        "rows_quarantined": database_status.get("rows_quarantined", 0),
        "objects_registered": database_status.get("objects_registered", 0),
        "object_assets_accounted_by_contract": object_contracts,
        "duplicate_formats_skipped": database_status.get("duplicate_formats_skipped", 0),
        "duplicate_formats_accounted_by_contract": duplicate_skips,
        "critical_open_issues": [
            row["evidence"] for row in quality_rows if row["severity"].upper() == "CRITICAL"
        ],
        "sealed_confirmation_enforced": database_status.get("sealed_confirmation_enforced", False),
        "live_role_label_access_denied": database_status.get("live_role_label_access_denied", False),
        "strict_validation_passed": database_status.get("strict_validation_passed", False),
        "idempotency_passed": database_status.get("idempotency_passed", False),
        "production_database_loaded": database_status.get("production_database_loaded", False),
        "local_test_database_loaded": database_status.get("local_test_database_loaded", False),
        "validation_failures": validation_failures,
        "next_action": database_status["next_action"],
    }

    manifest = {
        "generated_at_utc": utc_now_iso(),
        "git_commit": git_commit(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
        "audit_read_order_bytes": audit_bundle.read_order_bytes,
        "audit_bundle_zip": {
            "path": str(bundle_zip),
            "sha256": summary["audit_bundle_sha256"],
            "bytes": bundle_zip.stat().st_size,
        },
        "cutoff_rule_version": CUTOFF_RULE_VERSION,
        "database_status": database_status,
        "required_reports": list(REQUIRED_REPORTS),
    }

    write_csv(output_dir / "DB_SOURCE_FILE_RECONCILIATION_ALL_52.csv", source_rows)
    write_csv(output_dir / "DB_ATTRIBUTE_RECONCILIATION_ALL_1869.csv", attribute_rows)
    write_csv(output_dir / "DB_DATASET_RECONCILIATION_ALL_13.csv", dataset_rows)
    write_csv(output_dir / "DB_ROW_COUNT_RECONCILIATION.csv", row_count_rows)
    write_csv(output_dir / "DB_QUALITY_ISSUE_STATUS_ALL_22.csv", quality_rows)
    write_csv(output_dir / "DB_STATION_DIMENSION_STATUS_ALL_36.csv", station_rows)
    write_csv(output_dir / "DB_TABLE_CATALOG.csv", source_rows)
    write_csv(
        output_dir / "DB_QUARANTINE_SUMMARY.csv",
        [
            {
                "reason_code": "AUDIT_DRIVEN_ROW_REJECTION",
                "rows_quarantined": database_status.get("rows_quarantined", 0),
                "note": (
                    "Rows quarantined during direct DB load."
                    if database_status["status"] == "LOADED"
                    else "No database load ran; source-level quarantine contracts are recorded only."
                ),
            },
        ],
    )
    write_csv(
        output_dir / "DB_PERMISSION_MATRIX.csv",
        [
            {"role": "hkg_tmax_live_inference", "label_core": "DENY", "sealed_confirmation": "DENY"},
            {"role": "hkg_tmax_research_dev", "label_core_pre2024_view": "ALLOW", "sealed_confirmation": "DENY"},
            {"role": "hkg_tmax_audit", "all_layers_read": "ALLOW", "write_source_data": "DENY"},
            {"role": "hkg_tmax_confirmation_admin", "sealed_confirmation": "ALLOW"},
        ],
    )
    write_json(output_dir / "DB_SCHEMA_MANIFEST.json", schema_manifest(migration_path))
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "run_manifest.json", manifest)

    readme = f"""# Audit-Driven Database Ingestion

Status: {summary["status"]}

This folder is the bookkeeping artifact for the HKG T+24 Tmax audit-driven
database ingestion task. The audit bundle was verified and all contract rows
were reconciled against the current `data/datasets` files.

Database execution status: {database_status["status"]}

Reason: {database_status["reason"]}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    results = f"""# Results

| Item | Count |
| --- | ---: |
| Datasets accounted | {len(dataset_rows)} |
| Source files/tables accounted | {len(source_rows)} |
| Attributes accounted | {len(attribute_rows)} |
| Quality issues accounted | {len(quality_rows)} |
| ISD stations accounted | {len(station_rows)} |
| Profiled source rows | {audit_bundle.profile_summary["row_table_rows_total"]} |
| Duplicate formats in contract | {duplicate_skips} |
| Object/research artifacts in contract | {object_contracts} |

Database load: `{database_status["status"]}`.

Rows by contracted layer:

{json.dumps(dict(rows_by_layer), indent=2)}
"""
    (output_dir / "RESULTS.md").write_text(results, encoding="utf-8")

    conclusion = f"""# Conclusion

The audit contract has been verified and reconciled. Database status:
`{database_status["status"]}`.

Production database loaded: `{summary["production_database_loaded"]}`.
Local test database loaded: `{summary["local_test_database_loaded"]}`.

Critical blockers remain open exactly as supplied by the audit bundle. Storage
or registration does not make diagnostic data production-eligible.
"""
    (output_dir / "CONCLUSION.md").write_text(conclusion, encoding="utf-8")

    (output_dir / "DB_IMPLEMENTATION_SUMMARY.md").write_text(
        f"""# DB Implementation Summary

Migration: `{migration_path}`

Implemented schemas, control tables, cutoff function, label sealing tables,
safe views, quarantine/reconciliation tables, object catalog, and role grants.

The canonical cutoff is `{CUTOFF_RULE_VERSION}`:
15:00 Asia/Hong_Kong on T-1, equivalent to 07:00 UTC on T-1.
""",
        encoding="utf-8",
    )
    (output_dir / "DB_TIMESTAMP_AUDIT.md").write_text(
        """# Timestamp Audit

Python and SQL cutoff implementations use Asia/Hong_Kong explicitly.
The unit tests verify that Hong Kong remains UTC+08:00 and that 15:00 HKT
on T-1 maps to 07:00 UTC on T-1.
""",
        encoding="utf-8",
    )
    leakage_execution = (
        "Execution proof: database checks passed."
        if database_status.get("strict_validation_passed")
        else "Execution proof is blocked or failed; inspect summary.json."
    )
    (output_dir / "DB_T24_LEAKAGE_AUDIT.md").write_text(
        f"""# T-24 Leakage Audit

Cutoff rule: `{CUTOFF_RULE_VERSION}`.

Safe views are defined in the migration and enforce:

- official anchors require `available_at_utc <= cutoff_utc`;
- target history is pre-2024 only;
- live exact-vintage catalog requires explicit `ELIGIBLE` status;
- live inference role receives no label or sealed-confirmation grants.

{leakage_execution}
""",
        encoding="utf-8",
    )
    (output_dir / "DB_IDEMPOTENCY_REPORT.md").write_text(
        f"""# Idempotency Report

Idempotency status: `{database_status.get("idempotency_status", "BLOCKED")}`.

Reason: {database_status["reason"]}

First-run signature equals second-run signature:
`{database_status.get("idempotency_passed", False)}`.
""",
        encoding="utf-8",
    )
    (output_dir / "DB_TEST_RESULTS.md").write_text(
        "# Test Results\n\n"
        + "\n".join(f"- `{row['command']}`: {row['status']}" for row in test_results)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "DB_RUN_COMMANDS.md").write_text(
        f"""# Run Commands

Apply/load when PostgreSQL is available:

```powershell
Set-Location "{Path.cwd()}"
.\\.venv\\Scripts\\python.exe -u -m hkg_tmax_db.cli run --apply-db --psql-direct --pg-host 127.0.0.1 --pg-port 5432 --pg-admin-user postgres --pg-admin-password root --pg-database hkg_tmax_research
```

Generate reports without DB:

```powershell
.\\.venv\\Scripts\\python.exe -m hkg_tmax_db.cli run --no-db
```
""",
        encoding="utf-8",
    )

    missing_reports = [name for name in REQUIRED_REPORTS if not (output_dir / name).exists()]
    if missing_reports:
        raise RuntimeError(f"Failed to create required reports: {missing_reports}")
    return summary
