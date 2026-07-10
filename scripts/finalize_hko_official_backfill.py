from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# isort: off
from scripts.monitor_hko_official_backfill import (  # noqa: E402
    DEFAULT_TARGET_END,
    DEFAULT_TARGET_START,
    run as run_monitor,
)
from scripts.run_hkg_t24_forecast_archive_continuous_scored_export import (  # noqa: E402
    SCORED_EXPORT_MANIFEST_PATH,
    run as run_scored_export,
)
# isort: on

DEFAULT_DATA_ROOT = Path(r"C:\hko_press_2000_2026")
DEFAULT_ARCHIVE_DB = DEFAULT_DATA_ROOT / "metadata" / "archive.sqlite3"
DEFAULT_DETAILS_LOG = DEFAULT_DATA_ROOT / "run_logs" / "official_details_20000101_20260620.out.log"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "datasets"
DEFAULT_MONITOR_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments"
    / "0000_research_state_and_data_contract"
    / "hko_official_backfill_monitor"
    / "artifacts"
)
DEFAULT_BUNDLE_STEM = "hko_official_press_weather_forecasts_20000101_20260620"


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Missing archive DB: {db_path}")
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def details_complete(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]
    return any(line.startswith("official-details complete ") for line in tail)


def quote_identifier(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Unsafe SQLite identifier: {name!r}")
    return f'"{name}"'


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    rows = connection.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    columns = [str(row["name"]) for row in rows]
    if not columns:
        raise ValueError(f"Table not found or has no columns: {table}")
    return columns


def stream_table_exports(
    connection: sqlite3.Connection,
    *,
    table: str,
    output_dir: Path,
    stem: str,
    order_by: str | None = None,
) -> list[Path]:
    columns = table_columns(connection, table)
    select_sql = f"SELECT {', '.join(quote_identifier(col) for col in columns)} FROM {quote_identifier(table)}"
    if order_by:
        select_sql += f" ORDER BY {order_by}"

    csv_path = output_dir / f"{stem}_{table}.csv"
    jsonl_path = output_dir / f"{stem}_{table}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    cursor = connection.execute(select_sql)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_handle, jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=columns)
        writer.writeheader()
        while True:
            rows = cursor.fetchmany(5000)
            if not rows:
                break
            for row in rows:
                item = {column: row[column] for column in columns}
                writer.writerow(item)
                jsonl_handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return [csv_path, jsonl_path]


def package_data_root(data_root: Path, zip_path: Path) -> Path:
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(data_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(data_root.parent))
    return zip_path


def archive_counts(connection: sqlite3.Connection) -> dict[str, Any]:
    tables = ["candidates", "retrievals", "bulletins", "forecast_days"]
    counts = {table: int(connection.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}").fetchone()[0]) for table in tables}
    retrievals = [
        dict(row)
        for row in connection.execute(
            """
            SELECT source, status_code, COUNT(*) AS rows,
                   SUM(CASE WHEN error IS NULL OR error='' THEN 0 ELSE 1 END) AS error_rows,
                   MAX(attempted_at_utc) AS last_attempted_at_utc
            FROM retrievals
            GROUP BY source, status_code
            ORDER BY source, status_code
            """
        )
    ]
    products = [
        dict(row)
        for row in connection.execute(
            """
            SELECT product_type, COUNT(*) AS bulletins,
                   MIN(index_date) AS first_index_date,
                   MAX(index_date) AS last_index_date,
                   MIN(issue_at_hkt) AS first_issue_at_hkt,
                   MAX(issue_at_hkt) AS last_issue_at_hkt
            FROM bulletins
            GROUP BY product_type
            ORDER BY product_type
            """
        )
    ]
    return {"counts": counts, "retrievals": retrievals, "bulletins_by_product": products}


def file_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.exists() and path.is_file():
            records.append(
                {
                    "path": str(path),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return records


def run(
    *,
    data_root: Path,
    archive_db: Path,
    details_log: Path,
    output_dir: Path,
    monitor_output_dir: Path,
    bundle_stem: str,
    require_details_complete: bool,
    skip_zip: bool,
    allow_gaps: bool,
) -> dict[str, Any]:
    if require_details_complete and not details_complete(details_log):
        raise RuntimeError(f"official-details is not complete according to {details_log}")

    generated_at = now_utc()
    with connect_readonly(archive_db) as connection:
        counts = archive_counts(connection)
        export_paths: list[Path] = []
        export_specs = {
            "candidates": "source, url",
            "retrievals": "id",
            "bulletins": "issue_at_hkt, source_url",
            "forecast_days": "target_date, issue_at_hkt, source_url",
        }
        for table, order_by in export_specs.items():
            export_paths.extend(
                stream_table_exports(
                    connection,
                    table=table,
                    output_dir=output_dir,
                    stem=bundle_stem,
                    order_by=order_by,
                )
            )

    scored_manifest = run_scored_export(archive_db=archive_db, raw_root=data_root / "raw" / "info_gov_bulletin")
    monitor_summary = run_monitor(
        archive_db,
        output_dir=monitor_output_dir,
        target_start=pd.Timestamp(DEFAULT_TARGET_START),
        target_end=pd.Timestamp(DEFAULT_TARGET_END),
    )
    if not allow_gaps and monitor_summary.get("completion_status") != "complete_no_gap":
        raise RuntimeError(
            "Refusing to finalize with remaining scored target-date gaps: "
            f"{monitor_summary.get('largest_missing_scored_gap')}"
        )

    zip_path = output_dir / f"{bundle_stem}.zip"
    package_paths: list[Path] = []
    if not skip_zip:
        package_paths.append(package_data_root(data_root, zip_path))

    hash_paths = [
        *export_paths,
        *package_paths,
        output_dir / "05_hko_historical_rss_forecasts" / "hko_press_archive_temperature_forecast_days.parquet",
        output_dir / "05_hko_historical_rss_forecasts" / "hko_official_t15_scored_pre2024.parquet",
        SCORED_EXPORT_MANIFEST_PATH,
        monitor_output_dir / "monitor_summary.json",
        monitor_output_dir / "missing_scored_gaps.csv",
    ]
    manifest = {
        "generated_at_utc": generated_at,
        "data_root": str(data_root),
        "archive_db": str(archive_db),
        "details_log": str(details_log),
        "details_complete": details_complete(details_log),
        "bundle_stem": bundle_stem,
        "archive_counts": counts,
        "scored_manifest": scored_manifest,
        "monitor_summary": monitor_summary,
        "files": file_records(hash_paths),
    }
    manifest_path = output_dir / f"{bundle_stem}_final_manifest.json"
    write_json(manifest_path, manifest)
    manifest["files"].append(
        {
            "path": str(manifest_path),
            "bytes": int(manifest_path.stat().st_size),
            "sha256": sha256_file(manifest_path),
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize the HKO official press forecast backfill.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--archive-db", type=Path, default=DEFAULT_ARCHIVE_DB)
    parser.add_argument("--details-log", type=Path, default=DEFAULT_DETAILS_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--monitor-output-dir", type=Path, default=DEFAULT_MONITOR_OUTPUT_DIR)
    parser.add_argument("--bundle-stem", default=DEFAULT_BUNDLE_STEM)
    parser.add_argument("--allow-incomplete-details", action="store_true")
    parser.add_argument("--allow-gaps", action="store_true")
    parser.add_argument("--skip-zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(
        data_root=args.data_root,
        archive_db=args.archive_db,
        details_log=args.details_log,
        output_dir=args.output_dir,
        monitor_output_dir=args.monitor_output_dir,
        bundle_stem=args.bundle_stem,
        require_details_complete=not args.allow_incomplete_details,
        skip_zip=args.skip_zip,
        allow_gaps=args.allow_gaps,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
