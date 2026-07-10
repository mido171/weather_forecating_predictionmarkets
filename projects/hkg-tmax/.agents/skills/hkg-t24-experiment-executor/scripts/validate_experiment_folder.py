#!/usr/bin/env python3
"""Validate a completed or rejected HKG T+24 experiment folder.

The validator is intentionally conservative. It cannot prove that every
meteorological value was historically available, but it can reject incomplete
artifacts, inconsistent score claims, confirmation contamination, row-count
mismatches, placeholder documents, and missing audit evidence.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from _common import ALL_STATUSES, COMPLETED_STATUSES, REJECTED_STATUSES, load_json

MANDATORY_ALWAYS = [
    "README.md",
    "RESULTS.md",
    "CONCLUSION.md",
    "experiment_spec.json",
    "leakage_audit.md",
    "data_manifest.csv",
    "feature_definitions.csv",
    "summary.json",
    "run_manifest.json",
    "REPRODUCE.md",
    "src",
]
MANDATORY_SCORED = [
    "scoreboard.csv",
    "slice_metrics.csv",
    "yearly_metrics.csv",
    "fold_metrics.csv",
    "row_coverage.csv",
]
MANDATORY_REJECTED = ["REJECTION.md"]
REQUIRED_README_HEADINGS = [
    "hypothesis",
    "why",
    "cutoff",
    "dataset",
    "feature",
    "baseline",
    "walk-forward",
    "acceptance",
]
REQUIRED_RESULTS_HEADINGS = [
    "headline",
    "coverage",
    "global",
    "fold",
    "year",
    "season",
    "tail",
    "leakage",
]
REQUIRED_CONCLUSION_HEADINGS = [
    "verdict",
    "learned",
    "mae",
    "robust",
    "failure",
    "promotion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_folder", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require all optional consistency evidence used for promotion.",
    )
    return parser.parse_args()


def normalized_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").lower()


def has_any_prediction_file(folder: Path) -> bool:
    candidates = [
        folder / "predictions.parquet",
        folder / "predictions.csv",
        folder / "predictions.csv.gz",
    ]
    return any(path.is_file() and path.stat().st_size > 0 for path in candidates)


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_or_none(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def get_scoreboard_row(rows: list[dict[str, str]], candidate_id: str | None) -> dict[str, str] | None:
    if not rows:
        return None
    for row in rows:
        row_id = row.get("candidate_id") or row.get("model") or row.get("candidate")
        if candidate_id and row_id == candidate_id:
            return row
    return rows[0]


def parse_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def check_document(
    folder: Path,
    filename: str,
    required_terms: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    path = folder / filename
    if not path.is_file():
        return
    text = normalized_text(path)
    words = re.findall(r"\b[\w.-]+\b", text)
    if len(words) < 120:
        errors.append(f"{filename} is too short ({len(words)} words)")
    placeholder_patterns = [
        "to be completed",
        "{{experiment",
        "replace this",
        "pending analysis",
        "todo:",
    ]
    for pattern in placeholder_patterns:
        if pattern in text:
            errors.append(f"{filename} contains unresolved placeholder: {pattern!r}")
    for term in required_terms:
        if term not in text:
            warnings.append(f"{filename} does not visibly cover required topic {term!r}")


def main() -> int:
    args = parse_args()
    folder = args.experiment_folder.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if not folder.is_dir():
        errors.append(f"Experiment folder does not exist: {folder}")
    else:
        for rel in MANDATORY_ALWAYS:
            path = folder / rel
            if not path.exists():
                errors.append(f"Missing mandatory artifact: {rel}")
            elif path.is_file() and path.stat().st_size == 0:
                errors.append(f"Mandatory artifact is empty: {rel}")

    spec: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    if (folder / "experiment_spec.json").is_file():
        try:
            spec = load_json(folder / "experiment_spec.json")
        except Exception as exc:
            errors.append(f"Invalid experiment_spec.json: {exc}")
    if (folder / "summary.json").is_file():
        try:
            summary = load_json(folder / "summary.json")
        except Exception as exc:
            errors.append(f"Invalid summary.json: {exc}")

    status = summary.get("status")
    if status not in ALL_STATUSES:
        errors.append(f"Invalid or missing summary status: {status!r}")
    scored = status in COMPLETED_STATUSES
    rejected = status in REJECTED_STATUSES

    if scored:
        for rel in MANDATORY_SCORED:
            path = folder / rel
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"Scored experiment missing non-empty artifact: {rel}")
        if not has_any_prediction_file(folder):
            errors.append("Scored experiment has no predictions.parquet/csv/csv.gz")
    if rejected:
        for rel in MANDATORY_REJECTED:
            if not (folder / rel).is_file():
                errors.append(f"Rejected/blocked experiment missing {rel}")

    if folder.is_dir():
        check_document(folder, "README.md", REQUIRED_README_HEADINGS, errors, warnings)
        check_document(folder, "RESULTS.md", REQUIRED_RESULTS_HEADINGS, errors, warnings)
        check_document(folder, "CONCLUSION.md", REQUIRED_CONCLUSION_HEADINGS, errors, warnings)

    # Identity checks.
    match = re.match(r"^(?P<id>\d{4})_(?P<slug>.+)$", folder.name)
    if not match:
        errors.append("Folder name must be NNNN_slug")
    else:
        if summary.get("experiment_id") != match.group("id"):
            errors.append("summary.experiment_id does not match folder ID")
        if spec.get("experiment_id") != match.group("id"):
            errors.append("spec.experiment_id does not match folder ID")
        if summary.get("slug") != match.group("slug"):
            errors.append("summary.slug does not match folder slug")
        if spec.get("slug") != match.group("slug"):
            errors.append("spec.slug does not match folder slug")

    # Contract checks.
    target = spec.get("target", {})
    if target.get("horizon") != "T-24":
        errors.append("spec target horizon must be exactly T-24")
    if target.get("timezone") != "Asia/Hong_Kong":
        errors.append("spec target timezone must be Asia/Hong_Kong")
    cutoff_path = str(target.get("cutoff_contract_path") or "").strip()
    if not cutoff_path:
        errors.append("spec has no cutoff_contract_path")
    frame = spec.get("frame", {})
    if frame.get("development_end_exclusive") != "2024-01-01":
        errors.append("development_end_exclusive must remain 2024-01-01")
    if frame.get("confirmation_locked") is not True:
        errors.append("confirmation_locked must be true in development specs")
    if spec.get("owner_authorized_confirmation") is not False:
        errors.append("owner_authorized_confirmation must be false for development experiments")
    if summary.get("confirmation_rows_used") not in (0, None):
        errors.append("Confirmation rows were used in a development experiment")
    if summary.get("owner_authorized_confirmation") not in (False, None):
        errors.append("Summary claims confirmation authorization in a development folder")

    # Score integrity checks.
    for key in (
        "baseline_mae_c",
        "candidate_mae_c",
        "mae_delta_c",
        "candidate_rmse_c",
        "candidate_bias_c",
    ):
        if not finite_or_none(summary.get(key)):
            errors.append(f"summary.{key} must be finite or null")

    n_candidate = summary.get("n_candidate")
    n_common = summary.get("n_common")
    baseline_n = summary.get("baseline_n")
    candidate_n = summary.get("candidate_n")
    for key, value in (
        ("n_candidate", n_candidate),
        ("n_common", n_common),
        ("baseline_n", baseline_n),
        ("candidate_n", candidate_n),
    ):
        if value is not None and (not isinstance(value, int) or value < 0):
            errors.append(f"summary.{key} must be a nonnegative integer or null")
    if scored:
        if not isinstance(n_common, int) or n_common <= 0:
            errors.append("Scored experiment must have positive n_common")
        if baseline_n is not None and baseline_n != n_common:
            errors.append("baseline_n differs from n_common; identical-row comparison failed")
        if candidate_n is not None and candidate_n != n_common:
            errors.append("candidate_n differs from n_common; identical-row comparison failed")
        baseline_mae = summary.get("baseline_mae_c")
        candidate_mae = summary.get("candidate_mae_c")
        delta = summary.get("mae_delta_c")
        if baseline_mae is None or candidate_mae is None or delta is None:
            errors.append("Scored experiment must report baseline MAE, candidate MAE, and delta")
        elif abs((candidate_mae - baseline_mae) - delta) > 1e-8:
            errors.append(
                "mae_delta_c is inconsistent; expected candidate_mae_c - baseline_mae_c"
            )
        if summary.get("leakage_status") != "PASS":
            errors.append("Scored experiment cannot be complete without leakage_status PASS")
        if not summary.get("common_row_hash"):
            warnings.append("No common_row_hash was saved")
            if args.strict:
                errors.append("Strict promotion validation requires common_row_hash")

    # Date seal.
    date_end = summary.get("date_end")
    if date_end:
        try:
            parsed = date.fromisoformat(str(date_end)[:10])
            if parsed >= date(2024, 1, 1):
                errors.append("Development experiment date_end reaches sealed 2024+ period")
        except ValueError:
            errors.append(f"Could not parse summary.date_end: {date_end!r}")

    # Check score table against summary when possible.
    scoreboard = csv_rows(folder / "scoreboard.csv")
    if scored and scoreboard:
        row = get_scoreboard_row(scoreboard, summary.get("candidate_id"))
        if row:
            table_mae = parse_float(row.get("mae_c") or row.get("mae"))
            if table_mae is not None and summary.get("candidate_mae_c") is not None:
                if abs(table_mae - float(summary["candidate_mae_c"])) > 1e-8:
                    errors.append("Candidate MAE differs between scoreboard.csv and summary.json")
            table_n = row.get("n") or row.get("row_count") or row.get("n_common")
            if table_n:
                try:
                    if int(float(table_n)) != int(summary.get("n_common", -1)):
                        errors.append("Candidate row count differs between scoreboard and summary")
                except ValueError:
                    warnings.append("Could not parse scoreboard candidate row count")

    # Audit document must contain explicit disposition.
    if (folder / "leakage_audit.md").is_file():
        audit = normalized_text(folder / "leakage_audit.md")
        if status in COMPLETED_STATUSES and "pass" not in audit:
            errors.append("leakage_audit.md does not state PASS for scored experiment")
        for phrase in ("cutoff", "available", "target", "rolling", "confirmation"):
            if phrase not in audit:
                warnings.append(f"leakage_audit.md omits {phrase!r} discussion")

    # Promotion cannot be claimed from diagnostic-only sources.
    diagnostic_sources = [
        source for source in spec.get("data_sources", [])
        if source.get("eligibility") in {"DIAGNOSTIC_ONLY", "PROSPECTIVE_ONLY", "BLOCKED", "REJECTED"}
    ]
    if status == "COMPLETED_PROMOTION_CANDIDATE" and diagnostic_sources:
        errors.append(
            "Promotion candidate uses at least one non-deployable data source: "
            + ", ".join(str(s.get("source_id")) for s in diagnostic_sources)
        )

    # Basic data manifest check.
    manifest_rows = csv_rows(folder / "data_manifest.csv")
    if not manifest_rows:
        errors.append("data_manifest.csv contains no source rows")
    else:
        required_manifest_fields = {
            "source_id", "path", "sha256", "size_bytes",
            "timestamp_fields", "availability_class"
        }
        missing_fields = required_manifest_fields - set(manifest_rows[0])
        if missing_fields:
            errors.append(
                "data_manifest.csv missing columns: " + ", ".join(sorted(missing_fields))
            )
        for idx, row in enumerate(manifest_rows, start=2):
            if not row.get("source_id") or not row.get("path"):
                errors.append(f"data_manifest.csv row {idx} lacks source_id/path")
            if row.get("availability_class") not in {
                "DEPLOYABLE_PROVEN","DEPLOYABLE_LAGGED_ONLY","DIAGNOSTIC_ONLY",
                "PROSPECTIVE_ONLY","BLOCKED","REJECTED"
            }:
                errors.append(f"data_manifest.csv row {idx} has invalid availability class")

    feature_rows = csv_rows(folder / "feature_definitions.csv")
    if scored and not feature_rows:
        errors.append("Scored experiment has no feature definitions")
    if feature_rows:
        required_feature_fields = {
            "feature_name","formula","input_columns","fit_scope","availability_rule"
        }
        missing_fields = required_feature_fields - set(feature_rows[0])
        if missing_fields:
            errors.append(
                "feature_definitions.csv missing columns: " + ", ".join(sorted(missing_fields))
            )

    report = {
        "folder": str(folder),
        "status": status,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Experiment folder: {folder}")
        print(f"Status: {status}")
        print(f"VALID: {'YES' if not errors else 'NO'}")
        for item in errors:
            print(f"ERROR: {item}")
        for item in warnings:
            print(f"WARNING: {item}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
