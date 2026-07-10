#!/usr/bin/env python3
"""Audit row-level point-in-time eligibility against a declared cutoff.

This utility does not infer historical publication latency. It accepts only an
explicit available-at timestamp column or a pre-audited latency contract
already materialized into that column. Naive timestamps are rejected by
default because timezone ambiguity is itself a leakage risk.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="CSV or JSONL input")
    parser.add_argument("--cutoff-column", required=True)
    parser.add_argument("--available-at-column", required=True)
    parser.add_argument("--target-date-column")
    parser.add_argument("--valid-time-column")
    parser.add_argument("--id-column")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-naive",
        action="store_true",
        help="Only use when a separate audited file proves one timezone for both columns.",
    )
    parser.add_argument("--max-violations", type=int, default=1000)
    return parser.parse_args()


def rows(path: Path) -> Iterator[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
    elif suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"Line {number} is not an object")
                    yield value
    else:
        raise ValueError("Input must be CSV, JSONL, or NDJSON")


def parse_timestamp(value: object, allow_naive: bool) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty timestamp")
    text = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        if not allow_naive:
            raise ValueError("naive timestamp")
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    counts = {
        "rows": 0,
        "eligible": 0,
        "after_cutoff": 0,
        "missing_cutoff": 0,
        "missing_available_at": 0,
        "parse_error": 0,
    }
    violations: list[dict] = []
    for index, row in enumerate(rows(args.input), start=1):
        counts["rows"] += 1
        cutoff_raw = row.get(args.cutoff_column)
        available_raw = row.get(args.available_at_column)
        if cutoff_raw in (None, ""):
            counts["missing_cutoff"] += 1
            reason = "MISSING_CUTOFF"
        elif available_raw in (None, ""):
            counts["missing_available_at"] += 1
            reason = "MISSING_AVAILABLE_AT"
        else:
            try:
                cutoff = parse_timestamp(cutoff_raw, args.allow_naive)
                available = parse_timestamp(available_raw, args.allow_naive)
            except Exception as exc:
                counts["parse_error"] += 1
                reason = f"TIMESTAMP_PARSE_ERROR:{exc}"
            else:
                if available <= cutoff:
                    counts["eligible"] += 1
                    continue
                counts["after_cutoff"] += 1
                reason = "AVAILABLE_AFTER_CUTOFF"
        if len(violations) < args.max_violations:
            violations.append({
                "row_number": index,
                "row_id": row.get(args.id_column) if args.id_column else None,
                "target_date": row.get(args.target_date_column) if args.target_date_column else None,
                "valid_time": row.get(args.valid_time_column) if args.valid_time_column else None,
                "cutoff": cutoff_raw,
                "available_at": available_raw,
                "reason": reason,
            })

    hard_failures = (
        counts["after_cutoff"] + counts["missing_cutoff"]
        + counts["missing_available_at"] + counts["parse_error"]
    )
    report = {
        "input": str(args.input.resolve()),
        "cutoff_column": args.cutoff_column,
        "available_at_column": args.available_at_column,
        "allow_naive": args.allow_naive,
        "counts": counts,
        "all_rows_eligible": hard_failures == 0 and counts["rows"] > 0,
        "violations_truncated": hard_failures > len(violations),
        "violations": violations,
        "interpretation": (
            "PASS: all rows have available_at <= cutoff."
            if hard_failures == 0 and counts["rows"] > 0
            else "FAIL/BLOCK: rows without proven pre-cutoff availability cannot be deployable."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))
    return 0 if report["all_rows_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
