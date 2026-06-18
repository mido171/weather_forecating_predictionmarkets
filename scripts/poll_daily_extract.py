from __future__ import annotations

import argparse
import json
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
        default=Path(
            "experiments/EXP-0003-g1-daily-extract-first-publication-polling/results/metrics.json"
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve() if args.root else find_repo_root()
    source_id = daily_extract_month_source_id(args.year, args.month)
    policy = FetchPolicy(
        timeout_seconds=args.timeout_seconds,
        user_agent="HKG-Tmax-Research/0.1 (+research-contact-required)",
    )

    catalog_snapshot, monthly_snapshot = fetch_daily_extract_month(
        root=root,
        year=args.year,
        month=args.month,
        policy=policy,
    )
    rows = build_daily_extract_publication_ledger(
        raw_root=root / "data" / "raw",
        year=args.year,
        month=args.month,
        source_id=source_id,
    )

    output_path = (root / args.output).resolve()
    metrics_path = (root / args.metrics).resolve()
    report_path = (root / args.report).resolve()
    row_count = write_publication_ledger(output_path, rows)
    summary = summarize_publication_rows(rows)
    metrics = {
        **summary,
        "year": args.year,
        "month": args.month,
        "source_id": source_id,
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
        "",
        "## Artifacts",
        "",
        f"- ledger CSV: `{output_path}`",
        f"- metrics JSON: `{metrics_path}`",
        "",
        "## Limitations",
        "",
        "- Rows already visible before active polling are only first observed by this archive.",
        "- Provider first-publication status requires repeated near-publication polling and review.",
        "- No predictive modelling or Polymarket backtesting was run.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
