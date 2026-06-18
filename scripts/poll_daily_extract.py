from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from hkg_tmax.config import find_repo_root
from hkg_tmax.fetch import FetchPolicy
from hkg_tmax.publication import (
    build_daily_extract_publication_ledger,
    daily_extract_month_source_id,
    fetch_daily_extract_month,
    summarize_publication_rows,
    write_publication_ledger,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Poll HKO Daily Extract and build first-seen ledger.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gold/target_publication/daily_extract_first_seen.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/daily_extract_publication.md"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("reports/generated/daily_extract_publication_metrics.json"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--fetch-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--interval-seconds", type=float, default=0.0)
    parser.add_argument(
        "--active-polling-start-at",
        help='Timezone-aware ISO timestamp, or "now"; required for provider-first candidates.',
    )
    parser.add_argument(
        "--watch-candidate-date",
        action="append",
        default=[],
        help="YYYY-MM-DD local date eligible for provider-first-publication candidate status.",
    )
    return parser


def _parse_active_start(value: str | None) -> datetime | None:
    if value is None:
        return None
    if value == "now":
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--active-polling-start-at must be timezone-aware")
    return parsed.astimezone(UTC)


def main() -> None:
    args = build_parser().parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if args.interval_seconds < 0:
        raise ValueError("--interval-seconds must be >= 0")
    if args.fetch_attempts < 1:
        raise ValueError("--fetch-attempts must be >= 1")
    if args.retry_sleep_seconds < 0:
        raise ValueError("--retry-sleep-seconds must be >= 0")
    root = args.root.resolve() if args.root else find_repo_root()
    source_id = daily_extract_month_source_id(args.year, args.month)
    active_start = _parse_active_start(args.active_polling_start_at)
    policy = FetchPolicy(
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.fetch_attempts,
        retry_sleep_seconds=args.retry_sleep_seconds,
        user_agent="HKG-Tmax-Research/0.1 (+research-contact-required)",
    )

    catalog_snapshot = None
    monthly_snapshot = None
    poll_snapshots = []
    for index in range(args.iterations):
        catalog_snapshot, monthly_snapshot = fetch_daily_extract_month(
            root=root,
            year=args.year,
            month=args.month,
            policy=policy,
        )
        poll_snapshots.append(
            {
                "iteration": index + 1,
                "catalog_snapshot": {
                    "sha256": catalog_snapshot.sha256,
                    "retrieved_at": catalog_snapshot.retrieved_at.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "path": str(catalog_snapshot.content_path),
                },
                "monthly_snapshot": {
                    "sha256": monthly_snapshot.sha256,
                    "retrieved_at": monthly_snapshot.retrieved_at.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "path": str(monthly_snapshot.content_path),
                },
            }
        )
        if index < args.iterations - 1:
            time.sleep(args.interval_seconds)
    if catalog_snapshot is None or monthly_snapshot is None:
        raise RuntimeError("No Daily Extract poll iterations completed")

    rows = build_daily_extract_publication_ledger(
        raw_root=root / "data" / "raw",
        year=args.year,
        month=args.month,
        source_id=source_id,
        provider_first_candidate_after=active_start,
        watched_candidate_dates=args.watch_candidate_date,
    )

    output_path = (root / args.output).resolve()
    metrics_path = (root / args.metrics).resolve()
    report_path = (root / args.report).resolve()
    row_count = write_publication_ledger(output_path, rows)
    summary = summarize_publication_rows(rows)
    observed_dates = {row.local_date for row in rows}
    watched_present = [date for date in args.watch_candidate_date if date in observed_dates]
    watched_missing = [date for date in args.watch_candidate_date if date not in observed_dates]
    metrics = {
        **summary,
        "year": args.year,
        "month": args.month,
        "source_id": source_id,
        "poll_iterations_completed": args.iterations,
        "poll_snapshot_count": len(poll_snapshots),
        "poll_snapshots": poll_snapshots,
        "fetch_attempts": args.fetch_attempts,
        "retry_sleep_seconds": args.retry_sleep_seconds,
        "interval_seconds": args.interval_seconds,
        "active_polling_start_at": (
            None if active_start is None else active_start.isoformat().replace("+00:00", "Z")
        ),
        "watched_candidate_dates": args.watch_candidate_date,
        "watched_candidate_dates_present": watched_present,
        "watched_candidate_dates_missing": watched_missing,
        "catalog_snapshot": {
            "sha256": catalog_snapshot.sha256,
            "retrieved_at": catalog_snapshot.retrieved_at.isoformat().replace("+00:00", "Z"),
            "path": str(catalog_snapshot.content_path),
        },
        "monthly_snapshot": {
            "sha256": monthly_snapshot.sha256,
            "retrieved_at": monthly_snapshot.retrieved_at.isoformat().replace("+00:00", "Z"),
            "path": str(monthly_snapshot.content_path),
        },
        "ledger_path": str(output_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Daily Extract Publication Ledger",
        "",
        f"Generated for HKO Daily Extract `{args.year:04d}-{args.month:02d}`.",
        "",
        "## Gate Status",
        "",
        "**G1 remains blocked.** This run proves polling/ledger mechanics and archive-first-observed evidence. It does not by itself prove provider first publication.",
        "",
        "## Latest Poll",
        "",
        f"- catalog hash: `{catalog_snapshot.sha256}`",
        f"- catalog retrieved_at: `{catalog_snapshot.retrieved_at.isoformat().replace('+00:00', 'Z')}`",
        f"- monthly source: `{source_id}`",
        f"- monthly hash: `{monthly_snapshot.sha256}`",
        f"- monthly retrieved_at: `{monthly_snapshot.retrieved_at.isoformat().replace('+00:00', 'Z')}`",
        "",
        "## Ledger Summary",
        "",
        f"- row count: `{row_count}`",
        f"- evidence counts: `{summary['evidence_counts']}`",
        f"- revision count: `{summary['revision_count']}`",
        f"- provider first publication proven: `{summary['provider_first_publication_proven']}`",
        f"- poll snapshot count: `{len(poll_snapshots)}`",
        f"- fetch attempts per request: `{args.fetch_attempts}`",
        f"- retry sleep seconds: `{args.retry_sleep_seconds}`",
        f"- active polling start: `{metrics['active_polling_start_at']}`",
        f"- watched candidate dates: `{args.watch_candidate_date}`",
        f"- watched candidate dates present: `{watched_present}`",
        f"- watched candidate dates missing: `{watched_missing}`",
        "",
        "## Artifacts",
        "",
        f"- ledger CSV: `{output_path}`",
        f"- metrics JSON: `{metrics_path}`",
        "",
        "## Limitations",
        "",
        "- Rows already visible before active polling are only first observed by this archive.",
        "- Provider first-publication candidate status requires active absent-before-present evidence.",
        "- No predictive modelling or market backtesting was run.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
