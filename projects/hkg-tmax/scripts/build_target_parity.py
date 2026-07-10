from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from hkg_tmax.config import find_repo_root
from hkg_tmax.hko import DailyClimateRow, parse_daily_climate_csv, parse_daily_extract_json
from hkg_tmax.target import TargetError, require_daily_extract_target


@dataclass(frozen=True)
class Snapshot:
    source_id: str
    content_path: Path
    sha256: str
    retrieved_at: str
    final_url: str | None


def _latest_snapshot(root: Path, source_id: str) -> Snapshot:
    sidecars = sorted((root / "data" / "raw" / source_id).glob("**/*.metadata.json"))
    if not sidecars:
        raise FileNotFoundError(f"No raw metadata sidecars found for {source_id}")

    latest_data: dict[str, object] | None = None
    latest_sidecar: Path | None = None
    for sidecar in sidecars:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if latest_data is None or str(data["retrieved_at"]) > str(latest_data["retrieved_at"]):
            latest_data = data
            latest_sidecar = sidecar
    if latest_data is None or latest_sidecar is None:
        raise FileNotFoundError(f"No usable raw metadata sidecars found for {source_id}")

    content_path = Path(str(latest_data["content_path"]))
    if not content_path.exists():
        extension = (
            latest_data.get("metadata", {})
            if isinstance(latest_data.get("metadata"), dict)
            else {}
        ).get("extension_inferred", "bin")
        content_path = latest_sidecar.with_name(
            latest_sidecar.name.replace(".metadata.json", f".{extension}")
        )
    if not content_path.exists():
        raise FileNotFoundError(f"Raw content missing for {source_id}: {content_path}")

    metadata = latest_data.get("metadata", {})
    final_url = metadata.get("final_url") if isinstance(metadata, dict) else None
    return Snapshot(
        source_id=str(latest_data["source_id"]),
        content_path=content_path,
        sha256=str(latest_data["content_sha256"]),
        retrieved_at=str(latest_data["retrieved_at"]),
        final_url=str(final_url) if final_url is not None else None,
    )


def _clmmaxt_by_date(rows: list[DailyClimateRow]) -> dict[date, DailyClimateRow]:
    dated: dict[date, DailyClimateRow] = {}
    for row in rows:
        if row.local_date is not None:
            dated[row.local_date] = row
    return dated


def _decimal_delta(left: Decimal | None, right: Decimal | None) -> str:
    if left is None or right is None:
        return ""
    return str(left - right)


def build_parity(
    *,
    root: Path,
    year: int,
    month: int,
    daily_source_id: str,
    clmmaxt_source_id: str,
    output_path: Path,
    report_path: Path,
    metrics_path: Path,
) -> None:
    daily_snapshot = _latest_snapshot(root, daily_source_id)
    clmmaxt_snapshot = _latest_snapshot(root, clmmaxt_source_id)
    daily_rows = parse_daily_extract_json(
        daily_snapshot.content_path.read_bytes(),
        year=year,
        month=month,
    )
    climate_rows = parse_daily_climate_csv(clmmaxt_snapshot.content_path.read_bytes())
    climate_by_date = _clmmaxt_by_date(climate_rows)

    output_rows: list[dict[str, str]] = []
    compared = 0
    matches = 0
    mismatches = 0
    for daily_row in sorted(daily_rows, key=lambda row: row.local_date):
        clmmaxt_row = climate_by_date.get(daily_row.local_date)
        clmmaxt_value = clmmaxt_row.value if clmmaxt_row is not None else None
        notes = "latest Daily Extract vs latest CLMMAXT only; first-publication proof absent"
        quality_state = "MATCH_LATEST_ONLY"
        try:
            observation = require_daily_extract_target(
                daily_rows,
                target_date=daily_row.local_date,
                source_id=daily_snapshot.source_id,
                source_sha256=daily_snapshot.sha256,
            )
            daily_value = observation.value_c
        except TargetError as exc:
            daily_value = daily_row.absolute_daily_max_c
            quality_state = "ADAPTER_REJECTED"
            notes = f"{notes}; adapter rejected row: {exc}"

        if clmmaxt_row is None or clmmaxt_value is None:
            quality_state = "MISSING_CLMMAXT"
        elif daily_value is not None:
            compared += 1
            if daily_value == clmmaxt_value:
                matches += 1
            else:
                mismatches += 1
                quality_state = "CLMMAXT_DIFFERENCE"

        output_rows.append(
            {
                "local_date": daily_row.local_date.isoformat(),
                "event_slug": "",
                "rules_hash": "",
                "daily_extract_first_value": "",
                "daily_extract_first_available_at": "",
                "daily_extract_latest_value": "" if daily_value is None else str(daily_value),
                "daily_extract_source_id": daily_snapshot.source_id,
                "daily_extract_sha256": daily_snapshot.sha256,
                "daily_extract_retrieved_at": daily_snapshot.retrieved_at,
                "clmmaxt_value": "" if clmmaxt_value is None else str(clmmaxt_value),
                "clmmaxt_retrieved_at": clmmaxt_snapshot.retrieved_at,
                "clmmaxt_sha256": clmmaxt_snapshot.sha256,
                "actual_winner": "",
                "computed_winner": "",
                "first_vs_clmmaxt_delta": _decimal_delta(daily_value, clmmaxt_value),
                "computed_winner_matches": "",
                "quality_state": quality_state,
                "notes": notes,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    metrics = {
        "year": year,
        "month": month,
        "row_count": len(output_rows),
        "compared_count": compared,
        "latest_match_count": matches,
        "latest_mismatch_count": mismatches,
        "latest_match_rate": None if compared == 0 else matches / compared,
        "first_publication_proven": False,
        "g1_gate_status": "BLOCKED_PENDING_FIRST_PUBLICATION_EVIDENCE",
        "daily_extract": {
            "source_id": daily_snapshot.source_id,
            "sha256": daily_snapshot.sha256,
            "retrieved_at": daily_snapshot.retrieved_at,
            "path": str(daily_snapshot.content_path),
            "final_url": daily_snapshot.final_url,
        },
        "clmmaxt": {
            "source_id": clmmaxt_snapshot.source_id,
            "sha256": clmmaxt_snapshot.sha256,
            "retrieved_at": clmmaxt_snapshot.retrieved_at,
            "path": str(clmmaxt_snapshot.content_path),
            "final_url": clmmaxt_snapshot.final_url,
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Target Parity Report",
        "",
        f"Generated for HKO Daily Extract {year:04d}-{month:02d} against CLMMAXT HKO.",
        "",
        "## Gate status",
        "",
        "**G1 is not passed.** This table proves latest-payload equality for the archived "
        "May 2026 sample only. It does not prove first-publication parity.",
        "",
        "## Sources",
        "",
        f"- Daily Extract: `{daily_snapshot.source_id}`",
        f"  - retrieved_at: `{daily_snapshot.retrieved_at}`",
        f"  - sha256: `{daily_snapshot.sha256}`",
        f"  - path: `{daily_snapshot.content_path}`",
        f"  - final_url: `{daily_snapshot.final_url}`",
        f"- CLMMAXT HKO: `{clmmaxt_snapshot.source_id}`",
        f"  - retrieved_at: `{clmmaxt_snapshot.retrieved_at}`",
        f"  - sha256: `{clmmaxt_snapshot.sha256}`",
        f"  - path: `{clmmaxt_snapshot.content_path}`",
        f"  - final_url: `{clmmaxt_snapshot.final_url}`",
        "",
        "## Results",
        "",
        f"- parity rows: `{len(output_rows)}`",
        f"- compared rows: `{compared}`",
        f"- latest-payload matches: `{matches}`",
        f"- latest-payload mismatches: `{mismatches}`",
        f"- latest-payload match rate: `{metrics['latest_match_rate']}`",
        "",
        "## Limitations",
        "",
        "- `daily_extract_first_value` is intentionally blank because first-publication "
        "capture has not yet been observed for these historical dates.",
        "- Polymarket backtesting, price history, order books, trades, liquidity, execution, "
        "and market replay are deferred by user instruction.",
        "- CLMMAXT remains a proxy until first-publication Daily Extract parity is proven.",
        "",
        "## Artifacts",
        "",
        f"- parity CSV: `{output_path}`",
        f"- metrics JSON: `{metrics_path}`",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build HKO target parity artifacts.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--daily-source-id", default="hko_daily_extract_202605")
    parser.add_argument("--clmmaxt-source-id", default="hko_clmmaxt_hko")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gold/target_parity/target_parity.csv"),
    )
    parser.add_argument("--report", type=Path, default=Path("reports/target_parity.md"))
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "experiments/EXP-0002-g1-daily-extract-and-clmmaxt-target-parity/results/metrics.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve() if args.root else find_repo_root()
    build_parity(
        root=root,
        year=args.year,
        month=args.month,
        daily_source_id=args.daily_source_id,
        clmmaxt_source_id=args.clmmaxt_source_id,
        output_path=(root / args.output).resolve(),
        report_path=(root / args.report).resolve(),
        metrics_path=(root / args.metrics).resolve(),
    )


if __name__ == "__main__":
    main()
