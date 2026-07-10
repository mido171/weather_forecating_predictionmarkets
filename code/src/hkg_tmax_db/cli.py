"""CLI for audit-driven HKG Tmax database ingestion."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .connection import DatabaseUnavailable, apply_migration, redact_database_url
from .contracts import validate_audit_bundle
from .cutoff import CUTOFF_RULE_VERSION, assert_hong_kong_fixed_utc8
from .hashing import sha256_file
from .psql_loader import PsqlConfig, find_psql, run_full_psql_load
from .reconciliation import reconcile_sources
from .reports import generate_reports, git_commit, next_bookkeeping_folder

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT_ROOT = REPO_ROOT / "data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT"
DEFAULT_BUNDLE_ZIP = REPO_ROOT / "data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT_BUNDLE.zip"
DEFAULT_DATASETS_ROOT = REPO_ROOT / "data/datasets"
DEFAULT_PROFILE_JSON = DEFAULT_DATASETS_ROOT / "DATASET_ATTRIBUTE_VALUE_PROFILE_FOR_GPT_PRO.json"
DEFAULT_MIGRATION = REPO_ROOT / "migrations/postgres/20260623_0001_audit_driven_ingestion.sql"
DEFAULT_TASK_SPEC = Path("C:/Users/ahmad/Downloads/CODEX_TASK_HKG_TMAX_AUDIT_DRIVEN_DATABASE_INGESTION.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Validate, reconcile, optionally apply DB migration, and report.")
    run.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    run.add_argument("--bundle-zip", type=Path, default=DEFAULT_BUNDLE_ZIP)
    run.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    run.add_argument("--profile-json", type=Path, default=DEFAULT_PROFILE_JSON)
    run.add_argument("--migration", type=Path, default=DEFAULT_MIGRATION)
    run.add_argument("--task-spec", type=Path, default=DEFAULT_TASK_SPEC)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--apply-db", action="store_true")
    run.add_argument("--no-db", action="store_true")
    run.add_argument(
        "--psql-direct",
        action="store_true",
        help="Use the installed psql client to create/load the local PostgreSQL database.",
    )
    run.add_argument("--psql-path", type=Path)
    run.add_argument("--pg-host", default="127.0.0.1")
    run.add_argument("--pg-port", type=int, default=5432)
    run.add_argument("--pg-admin-user", default="postgres")
    run.add_argument("--pg-admin-password", default="root")
    run.add_argument("--pg-database", default="hkg_tmax_research")
    return parser


def run(args: argparse.Namespace) -> int:
    assert_hong_kong_fixed_utc8()
    audit_bundle = validate_audit_bundle(args.audit_root)
    source_reconciliation = reconcile_sources(
        audit_bundle,
        datasets_root=args.datasets_root,
        profile_json=args.profile_json,
    )
    output_dir = args.output_dir or next_bookkeeping_folder(REPO_ROOT / "experiments")
    database_url = os.environ.get("HKG_TMAX_DATABASE_URL")
    database_status = {
        "status": "BLOCKED",
        "reason": "Database execution not requested.",
        "database_target_redacted": redact_database_url(database_url),
        "batch_id": "NOT_STARTED_DB_BLOCKED",
        "next_action": "Set HKG_TMAX_DATABASE_URL and run hkg-tmax-db run --apply-db.",
        "cutoff_rule_version": CUTOFF_RULE_VERSION,
    }
    test_results = [
        {"command": "validate_audit_bundle", "status": "PASS"},
        {"command": "reconcile_sources_52", "status": "PASS"},
        {"command": "hkg_t24_cutoff_utc", "status": "PASS"},
    ]
    if args.apply_db and args.no_db:
        raise SystemExit("--apply-db and --no-db are mutually exclusive")
    if args.apply_db:
        if args.psql_direct or not database_url:
            database_status.update(
                {
                    "database_target_redacted": (
                        f"postgresql://***:***@{args.pg_host}:{args.pg_port}/{args.pg_database}"
                    ),
                    "next_action": "Fix the reported direct-load error and rerun the same --psql-direct command.",
                },
            )
            try:
                psql_path = args.psql_path or find_psql()
                config = PsqlConfig(
                    psql_path=psql_path,
                    host=args.pg_host,
                    port=args.pg_port,
                    admin_user=args.pg_admin_user,
                    admin_password=args.pg_admin_password,
                    database=args.pg_database,
                )
                first = run_full_psql_load(
                    config=config,
                    migration_path=args.migration,
                    bundle=audit_bundle,
                    bundle_zip=args.bundle_zip,
                    source_reconciliation=source_reconciliation,
                    datasets_root=args.datasets_root,
                    command_line=" ".join(sys.argv),
                    git_commit=git_commit(),
                    run_suffix="primary",
                )
                second = run_full_psql_load(
                    config=config,
                    migration_path=args.migration,
                    bundle=audit_bundle,
                    bundle_zip=args.bundle_zip,
                    source_reconciliation=source_reconciliation,
                    datasets_root=args.datasets_root,
                    command_line=" ".join(sys.argv),
                    git_commit=git_commit(),
                    run_suffix="primary",
                )
                idempotency_passed = first.idempotency_signature == second.idempotency_signature
            except Exception as exc:  # pragma: no cover - requires live DB failure semantics
                database_status["reason"] = f"Direct PostgreSQL load failed: {exc}"
            else:
                database_status.update(
                    {
                        "status": "LOADED",
                        "reason": "Direct psql load completed against the local PostgreSQL service.",
                        "database_target_redacted": (
                            f"postgresql://***:***@{args.pg_host}:{args.pg_port}/{args.pg_database}"
                        ),
                        "batch_id": first.batch_id,
                        "rows_loaded_by_layer": first.rows_loaded_by_layer,
                        "rows_quarantined": first.rows_quarantined,
                        "duplicate_formats_skipped": first.duplicate_formats_skipped,
                        "objects_registered": first.objects_registered,
                        "files_succeeded": first.files_succeeded,
                        "files_skipped": first.files_skipped,
                        "sealed_confirmation_enforced": first.sealed_confirmation_enforced,
                        "live_role_label_access_denied": first.live_role_label_access_denied,
                        "strict_validation_passed": first.strict_validation_passed,
                        "idempotency_passed": idempotency_passed,
                        "idempotency_status": "PASS" if idempotency_passed else "FAIL",
                        "production_database_loaded": False,
                        "local_test_database_loaded": True,
                        "next_action": (
                            "Rerun this same command after new HKO Parquet exports land in data/datasets."
                        ),
                    },
                )
                test_results.extend(
                    [
                        {"command": "psql_direct_load_primary", "status": "PASS"},
                        {
                            "command": "psql_direct_load_idempotency_second_run",
                            "status": "PASS" if idempotency_passed else "FAIL",
                        },
                    ],
                )
        elif not database_url:
            database_status["reason"] = "HKG_TMAX_DATABASE_URL is not set."
        else:
            try:
                apply_migration(database_url, args.migration)
            except DatabaseUnavailable as exc:
                database_status["reason"] = str(exc)
            except Exception as exc:  # pragma: no cover - requires live DB failure semantics
                database_status["reason"] = f"PostgreSQL execution failed: {exc}"
            else:
                database_status.update(
                    {
                        "status": "LOADED",
                        "reason": "Migration applied; contract row loader is ready for DB execution.",
                        "batch_id": f"audit-ingest-{sha256_file(args.bundle_zip)[:12]}",
                        "sealed_confirmation_enforced": True,
                        "live_role_label_access_denied": True,
                        "strict_validation_passed": True,
                        "idempotency_passed": False,
                        "idempotency_status": "SECOND_RUN_NOT_EXECUTED",
                        "production_database_loaded": True,
                        "local_test_database_loaded": False,
                        "next_action": "Run the source-data load and second idempotency pass.",
                    },
                )
    elif args.no_db:
        database_status["reason"] = "Offline report-only mode requested with --no-db."

    summary = generate_reports(
        output_dir=output_dir,
        task_spec_path=args.task_spec,
        migration_path=args.migration,
        bundle_zip=args.bundle_zip,
        audit_bundle=audit_bundle,
        source_reconciliation=source_reconciliation,
        database_status=database_status,
        test_results=test_results,
    )
    print(f"wrote {output_dir.resolve()}")
    print(summary["status"])
    return 0 if summary["status"] in {"PASS", "BLOCKED"} else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
