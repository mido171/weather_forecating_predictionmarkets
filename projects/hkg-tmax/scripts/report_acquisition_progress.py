from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict
from urllib.parse import parse_qs, urlparse


class SatelliteProgressRow(TypedDict):
    product: str
    expected: int
    downloaded: int
    downloaded_matching: int
    remaining: int
    first_missing: str
    last_missing: str


class HistoricalFeedProgressRow(TypedDict):
    source_id: str
    archives: int
    start: str
    end: str


class StationProgressRow(TypedDict):
    station: str
    start: int
    end: int
    files: int


SATELLITE_PRODUCTS: tuple[tuple[str, str, str], ...] = (
    (
        "H8 infrared",
        "hko_satellite_current_infrared_h8_manifest",
        "hko_satellite_current_infrared_h8_image",
    ),
    (
        "FY4B deep convection",
        "hko_satellite_current_deepconvection_fy4b_manifest",
        "hko_satellite_current_deepconvection_fy4b_image",
    ),
    (
        "H8 deep convection",
        "hko_satellite_current_deepconvection_h8_manifest",
        "hko_satellite_current_deepconvection_h8_image",
    ),
    (
        "FY4B infrared",
        "hko_satellite_current_infrared_fy4b_manifest",
        "hko_satellite_current_infrared_fy4b_image",
    ),
    (
        "H8 true colour",
        "hko_satellite_current_truecolour_h8_manifest",
        "hko_satellite_current_truecolour_h8_image",
    ),
    (
        "FY4B true colour",
        "hko_satellite_current_truecolour_fy4b_manifest",
        "hko_satellite_current_truecolour_fy4b_image",
    ),
    (
        "H8 all-day visible",
        "hko_satellite_current_alldayvisible_h8_manifest",
        "hko_satellite_current_alldayvisible_h8_image",
    ),
    (
        "GK2B aerosol optical depth",
        "hko_satellite_current_aerosolopticaldepth_gk2b_manifest",
        "hko_satellite_current_aerosolopticaldepth_gk2b_image",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an offline acquisition progress report from the raw ledger."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("HKG_TMAX_DATA_ROOT", r"C:\hkg_tmax_data"),
        help="Canonical acquisition data root.",
    )
    parser.add_argument(
        "--output",
        default="reports/acquisition_progress_snapshot.md",
        help="Markdown report path to write inside the repository.",
    )
    return parser.parse_args()


def read_ledger(data_root: Path) -> list[dict[str, str]]:
    ledger_path = data_root / "manifests" / "retrieval_ledger.csv"
    with ledger_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_int(value: str | None) -> int:
    try:
        return int(value or "0")
    except ValueError:
        return 0


def row_url(row: dict[str, str]) -> str:
    return row.get("request_url") or row.get("final_url") or ""


def latest_success(rows: list[dict[str, str]], source_id: str) -> dict[str, str] | None:
    matches = [
        row
        for row in rows
        if row.get("source_id") == source_id and row.get("status") == "success"
    ]
    if not matches:
        return None
    return max(matches, key=lambda row: row.get("retrieved_at", ""))


def extract_satellite_filenames(js_text: str) -> set[str]:
    return set(re.findall(r'"([^"\\]+\.(?:png|jpg|gif))"', js_text, flags=re.IGNORECASE))


def successful_filenames(rows: list[dict[str, str]], source_id: str) -> set[str]:
    filenames: set[str] = set()
    for row in rows:
        if row.get("source_id") != source_id or row.get("status") != "success":
            continue
        filename = Path(urlparse(row_url(row)).path).name
        if filename:
            filenames.add(filename)
    return filenames


def satellite_progress(rows: list[dict[str, str]]) -> list[SatelliteProgressRow]:
    progress: list[SatelliteProgressRow] = []
    for label, manifest_source_id, image_source_id in SATELLITE_PRODUCTS:
        manifest = latest_success(rows, manifest_source_id)
        expected: set[str] = set()
        if manifest:
            content_path = manifest.get("content_path", "")
            if content_path:
                expected = extract_satellite_filenames(
                    Path(content_path).read_text(encoding="utf-8", errors="replace")
                )
        downloaded = successful_filenames(rows, image_source_id)
        missing = sorted(expected - downloaded)
        downloaded_matching = expected & downloaded
        progress.append(
            {
                "product": label,
                "expected": len(expected),
                "downloaded": len(downloaded),
                "downloaded_matching": len(downloaded_matching),
                "remaining": len(missing),
                "first_missing": missing[0] if missing else "",
                "last_missing": missing[-1] if missing else "",
            }
        )
    return progress


def hko_high_frequency_progress(rows: list[dict[str, str]]) -> list[HistoricalFeedProgressRow]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        source_id = row.get("source_id", "")
        if not (
            source_id.startswith("datagov_hko_historical_latest_")
            and source_id.endswith("_archive")
            and row.get("status") == "success"
        ):
            continue
        parsed = urlparse(row_url(row))
        time_values = parse_qs(parsed.query).get("time", [])
        if time_values:
            grouped[source_id].append(time_values[0])
    return [
        {
            "source_id": source_id,
            "archives": len(times),
            "start": min(times) if times else "",
            "end": max(times) if times else "",
        }
        for source_id, times in sorted(grouped.items())
    ]


def noaa_isd_station_progress(rows: list[dict[str, str]]) -> tuple[list[StationProgressRow], int]:
    station_years: dict[tuple[str, str], list[int]] = defaultdict(list)
    pattern = re.compile(r"/((?:19|20)\d{2})/(\d{6})-(\d{5})-((?:19|20)\d{2})\.gz$")
    for row in rows:
        if row.get("source_id") != "noaa_isd_nearby_station_year":
            continue
        match = pattern.search(urlparse(row_url(row)).path)
        if not match:
            continue
        year = int(match.group(1))
        station_key = (match.group(2), match.group(3))
        station_years[station_key].append(year)
    table: list[StationProgressRow] = [
        {
            "station": f"{usaf}-{wban}",
            "start": min(years),
            "end": max(years),
            "files": len(years),
        }
        for (usaf, wban), years in sorted(station_years.items())
    ]
    return table, sum(item["files"] for item in table)


def source_count(rows: list[dict[str, str]], source_id: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("source_id") == source_id and row.get("status") == "success"
    )


def write_markdown(data_root: Path, output: Path, rows: list[dict[str, str]]) -> None:
    status_counts = Counter(row.get("status", "") for row in rows)
    success_rows = [row for row in rows if row.get("status") == "success"]
    successful_bytes = sum(as_int(row.get("content_length")) for row in success_rows)
    source_ids = {row.get("source_id", "") for row in rows if row.get("source_id")}
    unique_hashes = {row.get("content_sha256", "") for row in success_rows}
    satellite_rows = satellite_progress(rows)
    hko_hf_rows = hko_high_frequency_progress(rows)
    isd_rows, isd_file_count = noaa_isd_station_progress(rows)
    current_satellite_remaining = sum(
        satellite_row["remaining"] for satellite_row in satellite_rows
    )
    current_satellite_expected = sum(
        satellite_row["expected"] for satellite_row in satellite_rows
    )
    current_satellite_downloaded_matching = sum(
        satellite_row["downloaded_matching"] for satellite_row in satellite_rows
    )

    lines = [
        "# Acquisition Progress Snapshot",
        "",
        "Generated offline from the canonical raw ledger. This report is read-only with",
        f"respect to `{data_root}`.",
        "",
        "## Ledger Totals",
        "",
        f"- retrieval attempts: `{len(rows):,}`",
        f"- successful retrievals: `{status_counts.get('success', 0):,}`",
        f"- failed retrievals: `{len(rows) - status_counts.get('success', 0):,}`",
        f"- logical source IDs: `{len(source_ids):,}`",
        f"- unique successful content hashes: `{len(unique_hashes):,}`",
        f"- successful archived bytes: `{successful_bytes:,}`",
        "",
        "## HKO Current Satellite Progress",
        "",
        (
            f"Latest archived HKO current-satellite manifests list `{current_satellite_expected:,}` "
            "frames. Of those manifest-listed filenames, "
            f"`{current_satellite_downloaded_matching:,}` are already archived and "
            f"`{current_satellite_remaining:,}` are not yet present in the ledger."
        ),
        "",
        (
            "Note: these manifests are rolling operational windows and can list candidate "
            "filenames that are no longer provider-resolvable. The `satellite-current` batch "
            "is authoritative for live collection because it preflights current URLs and "
            "archives only resolvable non-2xx-skipped frames without overwriting immutable raw "
            "objects."
        ),
        "",
        "| Product | Manifest-listed | Downloaded matching manifest | Downloaded total | Manifest-listed missing | First missing | Last missing |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for satellite_row in satellite_rows:
        lines.append(
            "| {product} | {expected:,} | {downloaded_matching:,} | {downloaded:,} | {remaining:,} | {first_missing} | {last_missing} |".format(
                **satellite_row
            )
        )

    lines.extend(
        [
            "",
            "Current satellite acquisition state:",
            "",
            (
                f"- `{current_satellite_downloaded_matching:,}/{current_satellite_expected:,}` "
                "manifest-listed filenames are archived in the current ledger snapshot."
            ),
            (
                f"- `{current_satellite_remaining:,}` manifest-listed filenames are not in the "
                "ledger snapshot; treat them as rolling-window/persistent-miss candidates until "
                "the live preflight proves they are still resolvable."
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## HKO High-Frequency Historical Feeds",
            "",
            "| Feed archive source | Archive files | Start | End |",
            "|---|---:|---:|---:|",
        ]
    )
    for hko_hf_row in hko_hf_rows:
        lines.append(
            "| {source_id} | {archives:,} | {start} | {end} |".format(**hko_hf_row)
        )

    lines.extend(
        [
            "",
            "## Nearby / Surrounding Station Coverage",
            "",
            (
                f"NOAA ISD nearby station-year files: `{isd_file_count:,}` across "
                f"`{len(isd_rows):,}` station histories."
            ),
            "",
            "| Station | Start | End | Files |",
            "|---|---:|---:|---:|",
        ]
    )
    for isd_row in isd_rows:
        lines.append("| {station} | {start} | {end} | {files:,} |".format(**isd_row))

    lines.extend(
        [
            "",
            "## Other Current Coverage Counts",
            "",
            f"- HKO ARWF station/grid forecast payloads: `{source_count(rows, 'hko_arwf_station_forecast'):,}`",
            f"- HKO radar current frames: `{source_count(rows, 'hko_radar_current_frame'):,}`",
            f"- NCEP GFS regional subset files: `{source_count(rows, 'ncep_gfs_hk_subset_grib2'):,}`",
            f"- NCEP GEFS regional subset files: `{source_count(rows, 'ncep_gefs_hk_subset_grib2'):,}`",
            f"- HKO tropical cyclone best-track annual CSVs: `{source_count(rows, 'hko_tropical_cyclone_best_track'):,}`",
            "",
            "## Immediate Remaining Fetch Queue",
            "",
            "1. Keep the scheduled live collector enabled and monitor changed-payload health.",
            "2. Rerun the live satellite batch when a fresh current-window audit is required:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe -m hkg_tmax acquisition hko-backfill --batch satellite-current --continue-on-error --delay-seconds 0 --skip-existing-successes",
            "```",
            "",
            "3. Investigate persistent manifest-listed satellite misses only if the live preflight still returns resolvable 2xx URLs.",
            "4. Continue the remaining credential-gated or byte-budgeted historical families through the gridded acquisition policy.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output = Path(args.output)
    rows = read_ledger(data_root)
    write_markdown(data_root, output, rows)


if __name__ == "__main__":
    main()
