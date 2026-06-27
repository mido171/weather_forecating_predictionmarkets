"""CLI entrypoint for HKG-T24-001 foundation commands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.audit.source_registry import (
    populate_source_registry,
    validate_source_registry_contract,
)
from hkg_t24.constants import CODE_VERSION, END_TARGET_DATE, REPORT_NAMES, START_TARGET_DATE
from hkg_t24.db.connection import (
    DatabaseConfigError,
    connect,
    get_database_url,
    redact_database_url,
)
from hkg_t24.db.migrations import (
    apply_foundation_migrations,
    create_run_manifest,
    finish_run_manifest,
)
from hkg_t24.features.gribstream_safe_rows import refresh_nwp_safe_row_ledger
from hkg_t24.features.snapshot_builder import build_snapshots
from hkg_t24.features.source_contracts import (
    run_source_contract_checks,
    verify_lightgbm_required,
    write_source_contract_reports,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {value}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("phase0-preflight", help="Run DB/schema/source/lightgbm preflight.")
    subparsers.add_parser("build-source-registry", help="Apply migrations and populate source registry.")
    snapshots = subparsers.add_parser("build-h24n-snapshots", help="Build cutoff calendar and H24N snapshots.")
    snapshots.add_argument("--from-date", type=_parse_date, default=START_TARGET_DATE)
    snapshots.add_argument("--to-date", type=_parse_date, default=END_TARGET_DATE)
    return parser


def _connect_and_migrate(
    *,
    database_url: str,
    writer: ReportWriter,
    messages: list[str],
) -> Any:
    connection = connect(database_url)
    apply_foundation_migrations(connection, writer)
    messages.append(f"database={redact_database_url(database_url)}")
    return connection


def _write_phase0_report(
    writer: ReportWriter,
    *,
    status: str,
    messages: list[str],
    failures: tuple[str, ...],
    warnings: tuple[str, ...],
) -> None:
    writer.write_root_report(
        "phase0_preflight_report.md",
        "HKG-T24-001 Phase 0 Preflight Report",
        (
            ("Status", status),
            ("Messages", "\n".join(f"- {message}" for message in messages) or "- None."),
            ("Warnings", "\n".join(f"- {warning}" for warning in warnings) or "- None."),
            ("Failures", "\n".join(f"- {failure}" for failure in failures) or "- None."),
        ),
    )


def _write_contract_coverage(
    writer: ReportWriter,
    *,
    status: str,
    command: str,
    details: tuple[str, ...],
) -> None:
    report_lines = [
        f"- `{report_name}`: {'present' if (writer.paths.reports_dir / report_name).exists() else 'missing'}"
        for report_name in REPORT_NAMES
    ]
    writer.write_root_report(
        "jira_001_contract_coverage.md",
        "HKG-T24-001 Contract Coverage",
        (
            ("Status", status),
            ("Command", f"`{command}`"),
            (
                "Binding Precedence",
                "Final consistency patch, final clarifications, completion specification, original blueprint, Jira packet.",
            ),
            (
                "Implemented Foundation",
                "\n".join(
                    [
                        "- Dedicated `code/src/hkg_t24` package and `code/tests/hkg_t24` tests.",
                        "- Exact DSN priority and missing-DSN fail-closed behavior.",
                        "- Final source registry booleans and blocked/support-only statuses.",
                        "- H24N cutoff calendar, snapshot IDs, and availability flags.",
                        "- Strict GribStream full-run source filter and 6-hour H24N buffer.",
                        "- Final physical `feature_matrix` with strict/proxy compatibility views.",
                        "- Raw GribStream response-object compatibility view with final hash/timestamp aliases.",
                        "- Validation scoreboard and negative-control result scaffolds without producing model outputs.",
                        "- Live/eval prediction-component table scaffolding required by Jira 001.",
                    ]
                ),
            ),
            (
                "Superseded Contract Items",
                "\n".join(
                    [
                        "- `lag1` finalized daily target-memory feature names are forbidden; lag2 is canonical.",
                        "- `snapshot_feature_matrix_strict/proxy` are views only, not physical tables.",
                        "- `HKG_TMAX_DATABASE_URL` wins over `HKG_TMAX_DB_DSN`.",
                        "- LightGBM is mandatory; there is no HistGradientBoosting fallback.",
                    ]
                ),
            ),
            ("Details", "\n".join(f"- {detail}" for detail in details) or "- None."),
            ("Required Reports", "\n".join(report_lines)),
        ),
    )


def _run_with_manifest(
    *,
    run_kind: str,
    command_name: str,
    operation: Callable[[Any, ReportWriter, list[str]], tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> int:
    writer = ReportWriter(REPO_ROOT)
    messages: list[str] = []
    try:
        database_url = get_database_url(message_sink=messages.append)
    except DatabaseConfigError as exc:
        print(str(exc), file=sys.stderr)
        _write_phase0_report(
            writer,
            status="FAILED_CLOSED",
            messages=messages,
            failures=(str(exc),),
            warnings=(),
        )
        _write_contract_coverage(
            writer,
            status="FAILED_CLOSED",
            command=command_name,
            details=("Database DSN missing; command stopped before DB mutation as required.",),
        )
        return 1

    connection = _connect_and_migrate(database_url=database_url, writer=writer, messages=messages)
    run_id = create_run_manifest(
        connection,
        repo_root=REPO_ROOT,
        database_url=database_url,
        run_kind=run_kind,
        notes=f"{command_name} started by hkg_t24.cli {CODE_VERSION}",
    )
    connection.commit()
    try:
        status, warnings, failures = operation(connection, writer, messages)
    except Exception as exc:
        connection.rollback()
        finish_run_manifest(connection, run_id, status="failed_closed", notes=str(exc))
        connection.commit()
        _write_phase0_report(
            writer,
            status="FAILED_CLOSED",
            messages=messages,
            failures=(str(exc),),
            warnings=(),
        )
        _write_contract_coverage(
            writer,
            status="FAILED_CLOSED",
            command=command_name,
            details=(str(exc),),
        )
        print(str(exc), file=sys.stderr)
        return 1

    manifest_status = "passed" if status == "PASS" else "failed_closed"
    finish_run_manifest(
        connection,
        run_id,
        status=manifest_status,
        notes=f"{command_name} completed with status {status}",
    )
    connection.commit()
    _write_contract_coverage(
        writer,
        status=status,
        command=command_name,
        details=tuple(messages) + tuple(warnings) + tuple(failures),
    )
    print(f"{command_name}: {status}")
    return 0 if status == "PASS" else 1


def _phase0_operation(
    connection: Any,
    writer: ReportWriter,
    messages: list[str],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    validate_source_registry_contract()
    verify_lightgbm_required()
    result = run_source_contract_checks(connection)
    write_source_contract_reports(writer, result)
    _write_phase0_report(
        writer,
        status="PASS" if result.passed else "FAIL",
        messages=messages + ["LightGBM import passed.", "Source registry constants validated."],
        failures=result.failures,
        warnings=result.warnings,
    )
    return ("PASS" if result.passed else "FAIL", result.warnings, result.failures)


def _source_registry_operation(
    connection: Any,
    writer: ReportWriter,
    messages: list[str],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    validate_source_registry_contract()
    path = populate_source_registry(connection, writer)
    messages.append(f"wrote {path}")
    return "PASS", (), ()


def _snapshot_operation_factory(
    start_date: date,
    end_date: date,
) -> Callable[[Any, ReportWriter, list[str]], tuple[str, tuple[str, ...], tuple[str, ...]]]:
    def operation(
        connection: Any,
        writer: ReportWriter,
        messages: list[str],
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        if end_date < start_date:
            raise ValueError("--to-date must be >= --from-date")
        result = run_source_contract_checks(connection)
        write_source_contract_reports(writer, result)
        if not result.passed:
            return "FAIL", result.warnings, result.failures
        refresh_nwp_safe_row_ledger(connection, writer)
        build_snapshots(
            connection,
            writer,
            target_table=result.target_labels,
            start_date=start_date,
            end_date=end_date,
        )
        messages.append(f"built snapshots for {start_date.isoformat()}..{end_date.isoformat()}")
        return "PASS", result.warnings, result.failures

    return operation


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "phase0-preflight":
        return _run_with_manifest(
            run_kind="phase0_preflight",
            command_name="phase0-preflight",
            operation=_phase0_operation,
        )
    if args.command == "build-source-registry":
        return _run_with_manifest(
            run_kind="build_source_registry",
            command_name="build-source-registry",
            operation=_source_registry_operation,
        )
    if args.command == "build-h24n-snapshots":
        return _run_with_manifest(
            run_kind="build_h24n_snapshots",
            command_name="build-h24n-snapshots",
            operation=_snapshot_operation_factory(args.from_date, args.to_date),
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
