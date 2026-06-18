from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

from .acquisition import ensure_data_root, inspect_data_root
from .bronze import build_bronze_latest
from .collector import (
    collect_source_ids,
    run_due_schedules,
    write_health_report,
    write_inventory_reports,
    write_machine_source_catalog,
)
from .config import SourceCatalog, find_repo_root
from .doctor import doctor
from .experiments import create_experiment, generate_index
from .manifest import write_manifest
from .market import snapshot_polymarket_event
from .milestones import render_milestones
from .settlement import load_bucket_set, rules_hash
from .sources import fetch_sources, write_source_inventory
from .validation import (
    ValidationError,
    validate_bucket_fixture,
    validate_configs,
    validate_experiment_template,
    validate_repository,
    validate_yaml_tree,
)


def _load_env_file(root: Path) -> None:
    path = root / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _print_checks(checks: Sequence[str], warnings: Sequence[str] = ()) -> None:
    for check in checks:
        print(f"PASS  {check}")
    for warning in warnings:
        print(f"WARN  {warning}")


def _root_from_args(args: argparse.Namespace) -> Path:
    root = Path(args.root).expanduser().resolve() if args.root else find_repo_root()
    _load_env_file(root)
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hkg-tmax",
        description="Leakage-safe HKO Tmax research infrastructure",
    )
    parser.add_argument("--root", help="Repository root; auto-detected by default")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Check local environment and repository")

    validate_parser = subparsers.add_parser("validate", help="Validate contracts/configuration")
    validate_parser.add_argument(
        "scope",
        choices=("all", "configs", "buckets", "templates", "yaml"),
        default="all",
        nargs="?",
    )

    sources_parser = subparsers.add_parser("sources", help="List or fetch configured sources")
    sources_sub = sources_parser.add_subparsers(dest="sources_command", required=True)
    sources_list = sources_sub.add_parser("list", help="List source catalog")
    sources_list.add_argument("--tag")
    sources_fetch = sources_sub.add_parser("fetch", help="Archive configured HTTP sources")
    sources_fetch.add_argument("--id", action="append", dest="source_ids")
    sources_fetch.add_argument("--tag")
    sources_fetch.add_argument("--continue-on-error", action="store_true")
    sources_sub.add_parser("report", help="Render reports/source_inventory.md")

    experiments_parser = subparsers.add_parser("experiments", help="Manage experiment ledger")
    experiments_sub = experiments_parser.add_subparsers(
        dest="experiments_command", required=True
    )
    create_parser = experiments_sub.add_parser("create", help="Create immutable experiment folder")
    create_parser.add_argument("--title", required=True)
    experiments_sub.add_parser("index", help="Regenerate EXPERIMENT_INDEX.md")

    milestones_parser = subparsers.add_parser("milestones", help="Manage milestone dashboard")
    milestones_sub = milestones_parser.add_subparsers(
        dest="milestones_command", required=True
    )
    milestones_sub.add_parser("render", help="Regenerate MILESTONES.md from accepted statuses")

    subparsers.add_parser("manifest", help="Write repository MANIFEST.json")

    settlement_parser = subparsers.add_parser("settlement", help="Test bucket mapping")
    settlement_sub = settlement_parser.add_subparsers(
        dest="settlement_command", required=True
    )
    map_parser = settlement_sub.add_parser("map", help="Map a decimal Tmax to a bucket")
    map_parser.add_argument("--temperature", required=True)
    map_parser.add_argument(
        "--buckets",
        default="config/example_market_buckets.yaml",
        help="Bucket YAML path relative to root",
    )

    rules_parser = subparsers.add_parser("rules", help="Hash market rules")
    rules_sub = rules_parser.add_subparsers(dest="rules_command", required=True)
    hash_parser = rules_sub.add_parser("hash")
    hash_parser.add_argument("--file", required=True)
    hash_parser.add_argument("--exact", action="store_true")

    market_parser = subparsers.add_parser("market", help="Archive Polymarket metadata")
    market_sub = market_parser.add_subparsers(dest="market_command", required=True)
    snapshot_parser = market_sub.add_parser("snapshot-event")
    snapshot_parser.add_argument("--slug", required=True)

    acquisition_parser = subparsers.add_parser(
        "acquisition", help="Manage weather-only data acquisition infrastructure"
    )
    acquisition_sub = acquisition_parser.add_subparsers(
        dest="acquisition_command", required=True
    )
    acquisition_sub.add_parser("init", help="Create/verify HKG_TMAX_DATA_ROOT layout")
    collect_parser = acquisition_sub.add_parser(
        "collect", help="Collect selected non-market sources into the acquisition data root"
    )
    collect_parser.add_argument("--source-id", action="append", required=True)
    collect_parser.add_argument("--continue-on-error", action="store_true")
    acquisition_sub.add_parser("run-due", help="Run due sources from config/collector_schedules.yaml")
    acquisition_sub.add_parser("catalog", help="Write metadata/source_catalog.parquet")
    acquisition_sub.add_parser("reports", help="Refresh acquisition inventory and coverage reports")
    acquisition_sub.add_parser("health", help="Refresh reports/live_collector_health.md")
    bronze_parser = acquisition_sub.add_parser(
        "build-bronze", help="Build bronze Parquet for the latest successful raw retrieval"
    )
    bronze_parser.add_argument("--source-id", action="append", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root = _root_from_args(args)

        if args.command == "doctor":
            checks, warnings = doctor(root)
            _print_checks(checks, warnings)
            return

        if args.command == "validate":
            if args.scope == "all":
                report = validate_repository(root)
                _print_checks(report.checks, report.warnings)
            elif args.scope == "configs":
                checks, warnings = validate_configs(root)
                _print_checks(checks, warnings)
            elif args.scope == "buckets":
                _print_checks(validate_bucket_fixture(root))
            elif args.scope == "templates":
                _print_checks(validate_experiment_template(root))
            elif args.scope == "yaml":
                _print_checks(validate_yaml_tree(root))
            return

        if args.command == "sources":
            catalog = SourceCatalog.from_path(root / "config" / "data_sources.yaml")
            if args.sources_command == "list":
                selected = catalog.select(tag=args.tag)
                print("id\tprovider\tpoint_in_time_status\trole\turl")
                for source in selected:
                    print(
                        f"{source.id}\t{source.provider}\t{source.point_in_time_status}"
                        f"\t{source.role}\t{source.url}"
                    )
                return
            if args.sources_command == "report":
                print(write_source_inventory(root, catalog))
                return
            if args.sources_command == "fetch":
                if not args.source_ids and not args.tag:
                    parser.error("sources fetch requires --id or --tag")
                snapshots, errors = fetch_sources(
                    root=root,
                    catalog=catalog,
                    source_ids=args.source_ids,
                    tag=args.tag,
                    continue_on_error=args.continue_on_error,
                )
                for snapshot in snapshots:
                    print(
                        json.dumps(
                            {
                                "source_id": snapshot.source_id,
                                "path": str(snapshot.content_path),
                                "sha256": snapshot.sha256,
                                "retrieved_at": snapshot.retrieved_at.isoformat(),
                                "content_length": snapshot.content_length,
                            },
                            sort_keys=True,
                        )
                    )
                for error in errors:
                    print(f"ERROR {error}", file=sys.stderr)
                if errors:
                    raise SystemExit(1)
                return

        if args.command == "experiments":
            if args.experiments_command == "create":
                destination = create_experiment(root, args.title)
                generate_index(root)
                print(destination)
                return
            if args.experiments_command == "index":
                print(generate_index(root))
                return

        if args.command == "milestones" and args.milestones_command == "render":
            print(render_milestones(root))
            return

        if args.command == "manifest":
            print(write_manifest(root))
            return

        if args.command == "settlement" and args.settlement_command == "map":
            bucket_path = Path(args.buckets)
            if not bucket_path.is_absolute():
                bucket_path = root / bucket_path
            bucket_set = load_bucket_set(bucket_path)
            winner = bucket_set.winner(Decimal(args.temperature))
            print(
                json.dumps(
                    {
                        "temperature": str(Decimal(args.temperature)),
                        "winner": winner.label,
                        "lower_inclusive": (
                            None
                            if winner.lower_inclusive is None
                            else str(winner.lower_inclusive)
                        ),
                        "upper_exclusive": (
                            None
                            if winner.upper_exclusive is None
                            else str(winner.upper_exclusive)
                        ),
                    },
                    ensure_ascii=False,
                )
            )
            return

        if args.command == "rules" and args.rules_command == "hash":
            path = Path(args.file)
            if not path.is_absolute():
                path = root / path
            text = path.read_text(encoding="utf-8")
            print(rules_hash(text, normalized=not args.exact))
            return

        if args.command == "market" and args.market_command == "snapshot-event":
            snapshot = snapshot_polymarket_event(root, args.slug)
            print(
                json.dumps(
                    {
                        "path": str(snapshot.content_path),
                        "sha256": snapshot.sha256,
                        "retrieved_at": snapshot.retrieved_at.isoformat(),
                    }
                )
            )
            return

        if args.command == "acquisition":
            if args.acquisition_command == "init":
                data_root = ensure_data_root(root)
                status = inspect_data_root(root)
                print(
                    json.dumps(
                        {
                            "data_root": str(data_root),
                            "path_length": status.path_length,
                            "free_bytes": status.free_bytes,
                            "total_bytes": status.total_bytes,
                            "long_path_risk": status.long_path_risk,
                        },
                        sort_keys=True,
                    )
                )
                return
            if args.acquisition_command == "collect":
                outcomes = collect_source_ids(
                    root,
                    source_ids=args.source_id,
                    continue_on_error=args.continue_on_error,
                )
                for outcome in outcomes:
                    payload: dict[str, object] = {
                        "source_id": outcome.source_id,
                        "status": outcome.status,
                        "message": outcome.message,
                    }
                    if outcome.record is not None:
                        payload.update(
                            {
                                "content_sha256": outcome.record.content_sha256,
                                "content_length": outcome.record.content_length,
                                "content_path": str(outcome.record.content_path),
                                "deduplicated": outcome.record.deduplicated,
                            }
                        )
                    print(json.dumps(payload, sort_keys=True))
                return
            if args.acquisition_command == "run-due":
                outcomes = run_due_schedules(root)
                for outcome in outcomes:
                    print(
                        json.dumps(
                            {
                                "source_id": outcome.source_id,
                                "status": outcome.status,
                                "message": outcome.message,
                            },
                            sort_keys=True,
                        )
                    )
                return
            if args.acquisition_command == "catalog":
                print(write_machine_source_catalog(root))
                return
            if args.acquisition_command == "reports":
                for path in write_inventory_reports(root):
                    print(path)
                return
            if args.acquisition_command == "health":
                print(write_health_report(root))
                return
            if args.acquisition_command == "build-bronze":
                data_root = ensure_data_root(root)
                for source_id in args.source_id:
                    dataset = build_bronze_latest(data_root, source_id=source_id)
                    print(
                        json.dumps(
                            {
                                "source_id": dataset.source_id,
                                "content_sha256": dataset.content_sha256,
                                "row_count": dataset.row_count,
                                "parquet_path": str(dataset.parquet_path),
                                "metadata_path": str(dataset.metadata_path),
                            },
                            sort_keys=True,
                        )
                    )
                return

        parser.error("Unhandled command")
    except (ValidationError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
