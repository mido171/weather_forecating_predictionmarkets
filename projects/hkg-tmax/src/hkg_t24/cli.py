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
from hkg_t24.features.db_feature_builders import build_feature_scope, build_online_state_family
from hkg_t24.features.gribstream_safe_rows import refresh_nwp_safe_row_ledger
from hkg_t24.features.snapshot_builder import build_snapshots
from hkg_t24.features.source_contracts import (
    run_source_contract_checks,
    verify_lightgbm_required,
    write_source_contract_reports,
)
from hkg_t24.models.db_expert_factory import run_expert_factory
from hkg_t24.models.router import run_router_training
from hkg_t24.models.system_replay import run_jira003_replay
from hkg_tmax.paths import find_project_root

REPO_ROOT = find_project_root(Path(__file__))


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
    features = subparsers.add_parser("build-features", help="Build Jira 002 feature-family tables and feature matrix.")
    features.add_argument("--scope", choices=("strict", "proxy", "live_shadow"), required=True)
    features.add_argument("--from-date", type=_parse_date, default=START_TARGET_DATE)
    features.add_argument("--to-date", type=_parse_date, default=END_TARGET_DATE)
    features.add_argument("--smoke", action="store_true", help="Accept short date ranges for smoke verification.")
    online = subparsers.add_parser("build-online-states", help="Build online residual state rows.")
    online.add_argument("--cutoff-id", default="H24N")
    online.add_argument("--scope", choices=("strict",), default="strict")
    online.add_argument("--from-date", type=_parse_date, required=True)
    online.add_argument("--to-date", type=_parse_date, required=True)
    replay = subparsers.add_parser("replay-online-states-oof", help="Replay OOF online residual states.")
    replay.add_argument("--cutoff-id", default="H24N")
    replay.add_argument("--fold-spec", required=True)
    replay.add_argument("--from-date", type=_parse_date, default=START_TARGET_DATE)
    replay.add_argument("--to-date", type=_parse_date, default=date(2023, 12, 31))
    settlement = subparsers.add_parser(
        "update-online-states-after-settlement",
        help="Update online residual states after one settlement label is visible.",
    )
    settlement.add_argument("--cutoff-id", default="H24N")
    settlement.add_argument("--target-date", type=_parse_date, required=True)
    train = subparsers.add_parser("train-experts", help="Train/select Jira 002 expert artifacts.")
    train.add_argument("--scope", choices=("strict-pre2024",), required=True)
    train.add_argument("--from-date", type=_parse_date, default=START_TARGET_DATE)
    train.add_argument("--to-date", type=_parse_date, default=date(2023, 12, 31))
    train.add_argument("--smoke", action="store_true")
    generate = subparsers.add_parser("generate-oof", help="Generate and persist Jira 002 expert OOF predictions.")
    generate.add_argument("--scope", choices=("strict-pre2024",), required=True)
    generate.add_argument("--from-date", type=_parse_date, default=START_TARGET_DATE)
    generate.add_argument("--to-date", type=_parse_date, default=date(2023, 12, 31))
    generate.add_argument("--smoke", action="store_true")
    router = subparsers.add_parser("train-router", help="Train Jira 003 R0/R1 router artifacts.")
    router.add_argument("--router", choices=("R0", "R1", "R2", "R3", "R4"), required=True)
    router.add_argument("--scope", choices=("strict-pre2024",), default="strict-pre2024")
    router.add_argument("--from-date", type=_parse_date, default=START_TARGET_DATE)
    router.add_argument("--to-date", type=_parse_date, default=date(2023, 12, 31))
    router.add_argument("--smoke", action="store_true")
    specialists = subparsers.add_parser("train-specialists", help="Train Jira 003 specialists.")
    specialists.add_argument("--scope", choices=("strict-pre2024",), required=True)
    specialists.add_argument("--from-date", type=_parse_date, default=START_TARGET_DATE)
    specialists.add_argument("--to-date", type=_parse_date, default=date(2023, 12, 31))
    specialists.add_argument("--smoke", action="store_true")
    distribution = subparsers.add_parser("train-distribution", help="Train Jira 003 distributional layer.")
    distribution.add_argument("--scope", choices=("strict-pre2024",), required=True)
    distribution.add_argument("--from-date", type=_parse_date, default=START_TARGET_DATE)
    distribution.add_argument("--to-date", type=_parse_date, default=date(2023, 12, 31))
    distribution.add_argument("--smoke", action="store_true")
    system_replay = subparsers.add_parser("run-system-replay", help="Run Jira 003 strict system replay.")
    system_replay.add_argument("--scope", choices=("strict-pre2024",), required=True)
    system_replay.add_argument("--from-date", type=_parse_date, default=START_TARGET_DATE)
    system_replay.add_argument("--to-date", type=_parse_date, default=date(2023, 12, 31))
    system_replay.add_argument("--smoke", action="store_true")
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
    jira003_commands = {"train-router", "train-specialists", "train-distribution", "run-system-replay"}
    if command in jira003_commands:
        required_reports = (
            "router_scoreboard_strict.csv",
            "router_weight_diagnostics.csv",
            "router_promotion_decisions.csv",
            "specialist_scoreboard_strict.csv",
            "specialist_activation_report.csv",
            "specialist_no_harm_report.csv",
            "specialist_promotion_decisions.csv",
            "distribution_scoreboard.csv",
            "distribution_calibration_report.csv",
            "distribution_calibration_report.md",
            "calibration_report.md",
            "threshold_probability_scoreboard.csv",
            "prediction_interval_coverage_report.csv",
            "system_scoreboard_strict.csv",
            "system_scoreboard_proxy.csv",
            "system_ablation_matrix.csv",
        )
        report_lines = [
            f"- `{report_name}`: {'present' if (writer.paths.reports_dir / report_name).exists() else 'missing'}"
            for report_name in required_reports
        ]
        writer.write_root_report(
            "jira_003_contract_coverage.md",
            "HKG-T24-003 Contract Coverage",
            (
                ("Status", status),
                ("Command", f"`{command}`"),
                (
                    "Binding Precedence",
                    "Final consistency patch, final clarifications, completion specification, original blueprint, Jira packet.",
                ),
                (
                    "Implemented Jira003 Surface",
                    "\n".join(
                        [
                            "- OOF-only R0/R1 router training with static/dynamic weights, promotion gates, demotion records, and zero strict shadow/proxy impact.",
                            "- Six specialist correction modules with fold-local prior scoring, neutral missing components, support/no-harm gates, activation records, and bounded corrections.",
                            "- Final strict formula with R1/R0/E0/E2 fallback, specialist total cap, official +/-1.20C clipping, and component provenance.",
                            "- Distributional replay with LightGBM attempt, empirical fallback, monotonic quantile repair, exact 41 threshold probabilities, confidence states, and no-trade flags.",
                            "- Strict synthetic smoke path that does not require a database DSN and DB-backed path through the existing run manifest/migration boundary.",
                        ]
                    ),
                ),
                ("Details", "\n".join(f"- {detail}" for detail in details) or "- None."),
                ("Required Jira003 Reports", "\n".join(report_lines)),
            ),
        )
        return
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
                        "- Dedicated `src/hkg_t24` package and `tests/hkg_t24` tests.",
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


def _run_smoke_without_manifest(
    *,
    command_name: str,
    operation: Callable[[Any, ReportWriter, list[str]], tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> int:
    writer = ReportWriter(REPO_ROOT)
    messages: list[str] = ["smoke=true", "database=not_required"]
    try:
        status, warnings, failures = operation(None, writer, messages)
    except Exception as exc:
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


def _feature_operation_factory(
    scope: str,
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
        summary = build_feature_scope(
            connection,
            writer,
            scope=scope,
            start_date=start_date,
            end_date=end_date,
        )
        connection.commit()
        messages.append(
            f"built {summary.feature_matrix_rows} {scope} feature rows for "
            f"{start_date.isoformat()}..{end_date.isoformat()}"
        )
        return "PASS", result.warnings, result.failures

    return operation


def _online_operation_factory(
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
        rows = build_online_state_family(
            connection,
            writer,
            start_date=start_date,
            end_date=end_date,
        )
        connection.commit()
        messages.append(f"built online residual states for {len(rows)} target dates")
        return "PASS", (), ()

    return operation


def _expert_operation_factory(
    scope: str,
    start_date: date,
    end_date: date,
    *,
    smoke: bool,
    persist_predictions: bool,
) -> Callable[[Any, ReportWriter, list[str]], tuple[str, tuple[str, ...], tuple[str, ...]]]:
    def operation(
        connection: Any,
        writer: ReportWriter,
        messages: list[str],
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        if end_date < start_date:
            raise ValueError("--to-date must be >= --from-date")
        summary = run_expert_factory(
            connection,
            writer,
            scope=scope,
            start_date=start_date,
            end_date=end_date,
            smoke=smoke,
            persist_predictions=persist_predictions,
        )
        connection.commit()
        messages.append(
            f"expert factory rows={summary.prediction_rows}, active={summary.active_rows}, "
            f"placeholders={summary.placeholder_rows}, artifacts={summary.artifact_rows}"
        )
        return "PASS", (), ()

    return operation


def _router_operation_factory(
    router: str,
    scope: str,
    start_date: date,
    end_date: date,
    *,
    smoke: bool,
) -> Callable[[Any, ReportWriter, list[str]], tuple[str, tuple[str, ...], tuple[str, ...]]]:
    def operation(
        connection: Any,
        writer: ReportWriter,
        messages: list[str],
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        if end_date < start_date:
            raise ValueError("--to-date must be >= --from-date")
        status, warnings, details = run_router_training(
            connection,
            writer,
            router=router,
            scope=scope,
            start_date=start_date,
            end_date=end_date,
            smoke=smoke,
        )
        messages.extend(details)
        if connection is not None:
            connection.commit()
        return status, warnings, ()

    return operation


def _jira003_replay_operation_factory(
    scope: str,
    start_date: date,
    end_date: date,
    *,
    smoke: bool,
    persist: bool,
) -> Callable[[Any, ReportWriter, list[str]], tuple[str, tuple[str, ...], tuple[str, ...]]]:
    def operation(
        connection: Any,
        writer: ReportWriter,
        messages: list[str],
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        if end_date < start_date:
            raise ValueError("--to-date must be >= --from-date")
        status, warnings, details = run_jira003_replay(
            connection,
            writer,
            scope=scope,
            start_date=start_date,
            end_date=end_date,
            smoke=smoke,
            persist=persist,
        )
        messages.extend(details)
        if connection is not None:
            connection.commit()
        return status, warnings, ()

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
    if args.command == "build-features":
        return _run_with_manifest(
            run_kind=f"build_features_{args.scope}",
            command_name="build-features",
            operation=_feature_operation_factory(args.scope, args.from_date, args.to_date),
        )
    if args.command == "build-online-states":
        if args.cutoff_id != "H24N":
            raise ValueError("--cutoff-id must be H24N")
        return _run_with_manifest(
            run_kind="build_online_states",
            command_name="build-online-states",
            operation=_online_operation_factory(args.from_date, args.to_date),
        )
    if args.command == "replay-online-states-oof":
        if args.cutoff_id != "H24N":
            raise ValueError("--cutoff-id must be H24N")
        return _run_with_manifest(
            run_kind="replay_online_states_oof",
            command_name="replay-online-states-oof",
            operation=_online_operation_factory(args.from_date, args.to_date),
        )
    if args.command == "update-online-states-after-settlement":
        if args.cutoff_id != "H24N":
            raise ValueError("--cutoff-id must be H24N")
        return _run_with_manifest(
            run_kind="update_online_states_after_settlement",
            command_name="update-online-states-after-settlement",
            operation=_online_operation_factory(args.target_date, args.target_date),
        )
    if args.command == "train-experts":
        return _run_with_manifest(
            run_kind="train_experts",
            command_name="train-experts",
            operation=_expert_operation_factory(
                args.scope,
                args.from_date,
                args.to_date,
                smoke=args.smoke,
                persist_predictions=False,
            ),
        )
    if args.command == "generate-oof":
        return _run_with_manifest(
            run_kind="generate_oof",
            command_name="generate-oof",
            operation=_expert_operation_factory(
                args.scope,
                args.from_date,
                args.to_date,
                smoke=args.smoke,
                persist_predictions=True,
            ),
        )
    if args.command == "train-router":
        operation = _router_operation_factory(
            args.router,
            args.scope,
            args.from_date,
            args.to_date,
            smoke=args.smoke,
        )
        if args.smoke:
            return _run_smoke_without_manifest(command_name="train-router", operation=operation)
        return _run_with_manifest(
            run_kind=f"train_router_{args.router}",
            command_name="train-router",
            operation=operation,
        )
    if args.command == "train-specialists":
        operation = _jira003_replay_operation_factory(
            args.scope,
            args.from_date,
            args.to_date,
            smoke=args.smoke,
            persist=not args.smoke,
        )
        if args.smoke:
            return _run_smoke_without_manifest(command_name="train-specialists", operation=operation)
        return _run_with_manifest(
            run_kind="train_specialists",
            command_name="train-specialists",
            operation=operation,
        )
    if args.command == "train-distribution":
        operation = _jira003_replay_operation_factory(
            args.scope,
            args.from_date,
            args.to_date,
            smoke=args.smoke,
            persist=not args.smoke,
        )
        if args.smoke:
            return _run_smoke_without_manifest(command_name="train-distribution", operation=operation)
        return _run_with_manifest(
            run_kind="train_distribution",
            command_name="train-distribution",
            operation=operation,
        )
    if args.command == "run-system-replay":
        operation = _jira003_replay_operation_factory(
            args.scope,
            args.from_date,
            args.to_date,
            smoke=args.smoke,
            persist=not args.smoke,
        )
        if args.smoke:
            return _run_smoke_without_manifest(command_name="run-system-replay", operation=operation)
        return _run_with_manifest(
            run_kind="run_system_replay",
            command_name="run-system-replay",
            operation=operation,
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
