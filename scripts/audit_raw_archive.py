from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit of the canonical acquisition raw archive."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("HKG_TMAX_DATA_ROOT", r"C:\hkg_tmax_data"),
        help="Canonical acquisition data root.",
    )
    parser.add_argument(
        "--output",
        default="reports/raw_archive_audit.md",
        help="Markdown audit report path to write inside the repository.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="Maximum detailed errors to include in the report.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_int(value: str | None) -> int:
    try:
        return int(value or "0")
    except ValueError:
        return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sidecar root is not a JSON object")
    return value


def sidecar_has_http_metadata(sidecar: dict[str, Any]) -> bool:
    metadata = sidecar.get("metadata")
    if not isinstance(metadata, dict):
        return False
    response_headers = metadata.get("response_headers")
    return (
        isinstance(metadata.get("http_status"), int)
        and isinstance(metadata.get("requested_url"), str)
        and isinstance(metadata.get("final_url"), str)
        and isinstance(response_headers, dict)
    )


def audit_raw_archive(data_root: Path) -> tuple[dict[str, int], list[str]]:
    ledger_rows = read_csv_rows(data_root / "manifests" / "retrieval_ledger.csv")
    file_manifest_rows = read_csv_rows(data_root / "manifests" / "file_manifest.csv")
    dataset_lineage_rows = read_csv_rows(data_root / "manifests" / "dataset_lineage.csv")
    success_rows = [row for row in ledger_rows if row.get("status") == "success"]
    errors: list[str] = []

    success_hashes = {row.get("content_sha256", "") for row in success_rows}
    success_hashes.discard("")
    file_manifest_hashes = {
        row.get("content_sha256", "") for row in file_manifest_rows if row.get("content_sha256")
    }
    if file_manifest_hashes != success_hashes:
        missing = sorted(success_hashes - file_manifest_hashes)
        extra = sorted(file_manifest_hashes - success_hashes)
        if missing:
            errors.append(f"file_manifest missing {len(missing)} successful content hashes")
        if extra:
            errors.append(f"file_manifest has {len(extra)} hashes absent from successful ledger rows")

    lineage_keys = {
        (row.get("source_id", ""), row.get("content_sha256", ""), row.get("retrieved_at", ""))
        for row in dataset_lineage_rows
    }
    for row in success_rows:
        lineage_key = (
            row.get("source_id", ""),
            row.get("content_sha256", ""),
            row.get("retrieved_at", ""),
        )
        if lineage_key not in lineage_keys:
            errors.append(
                "dataset_lineage missing "
                f"{lineage_key[0]} {lineage_key[1]} {lineage_key[2]}"
            )

    audited_hashes: set[str] = set()
    for row in success_rows:
        digest = row.get("content_sha256", "")
        content_path = Path(row.get("content_path", ""))
        sidecar_path = Path(row.get("sidecar_path", ""))
        expected_length = as_int(row.get("content_length"))
        if not digest:
            errors.append(f"missing content_sha256 for retrieval {row.get('retrieval_id', '')}")
            continue
        if content_path.name.split(".")[0] != digest:
            errors.append(f"content path does not use digest filename: {content_path}")
        if not content_path.exists():
            errors.append(f"missing content object: {content_path}")
            continue
        actual_length = content_path.stat().st_size
        if actual_length != expected_length:
            errors.append(
                f"length mismatch for {digest}: ledger={expected_length} actual={actual_length}"
            )
        if digest not in audited_hashes:
            actual_digest = sha256_file(content_path)
            audited_hashes.add(digest)
            if actual_digest != digest:
                errors.append(f"sha256 mismatch for {content_path}: actual={actual_digest}")
        if not sidecar_path.exists():
            errors.append(f"missing sidecar: {sidecar_path}")
            continue
        try:
            sidecar = load_json(sidecar_path)
        except Exception as exc:
            errors.append(f"invalid sidecar JSON {sidecar_path}: {exc}")
            continue
        if sidecar.get("content_sha256") != digest:
            errors.append(f"sidecar digest mismatch for {sidecar_path}")
        if as_int(str(sidecar.get("content_length", ""))) != expected_length:
            errors.append(f"sidecar content_length mismatch for {sidecar_path}")
        if not sidecar_has_http_metadata(sidecar):
            errors.append(f"sidecar missing HTTP metadata: {sidecar_path}")

    metrics = {
        "ledger_rows": len(ledger_rows),
        "success_rows": len(success_rows),
        "failure_rows": len(ledger_rows) - len(success_rows),
        "unique_success_hashes": len(success_hashes),
        "file_manifest_rows": len(file_manifest_rows),
        "dataset_lineage_rows": len(dataset_lineage_rows),
        "audited_unique_hashes": len(audited_hashes),
        "errors": len(errors),
    }
    return metrics, errors


def write_report(output: Path, data_root: Path, metrics: dict[str, int], errors: list[str], max_errors: int) -> None:
    status = "PASS" if not errors else "FAIL"
    lines = [
        "# Raw Archive Audit",
        "",
        f"- status: `{status}`",
        f"- data root: `{data_root}`",
        f"- retrieval ledger rows: `{metrics['ledger_rows']:,}`",
        f"- successful retrieval rows: `{metrics['success_rows']:,}`",
        f"- failed retrieval rows: `{metrics['failure_rows']:,}`",
        f"- unique successful content hashes: `{metrics['unique_success_hashes']:,}`",
        f"- file manifest rows: `{metrics['file_manifest_rows']:,}`",
        f"- dataset lineage rows: `{metrics['dataset_lineage_rows']:,}`",
        f"- audited unique content objects: `{metrics['audited_unique_hashes']:,}`",
        f"- errors: `{metrics['errors']:,}`",
        "",
    ]
    if errors:
        lines.extend(["## Errors", ""])
        for error in errors[:max_errors]:
            lines.append(f"- {error}")
        if len(errors) > max_errors:
            lines.append(f"- ... {len(errors) - max_errors:,} additional errors omitted")
    else:
        lines.extend(
            [
                "## Verified",
                "",
                "- every successful ledger row points to an existing content object;",
                "- every audited content object hash matches its digest filename and ledger hash;",
                "- every successful ledger row length matches the object length;",
                "- every successful ledger row has a metadata sidecar with HTTP metadata;",
                "- file manifest hashes match successful ledger hashes;",
                "- dataset lineage covers every successful ledger row.",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output = Path(args.output)
    metrics, errors = audit_raw_archive(data_root)
    write_report(output, data_root, metrics, errors, args.max_errors)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
