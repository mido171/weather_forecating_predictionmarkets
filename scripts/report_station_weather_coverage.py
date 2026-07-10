from __future__ import annotations

import argparse
import csv
import io
import os
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

HKO_HF_PREFIX = "datagov_hko_historical_latest_"


class StationYearRow(TypedDict):
    station: str
    start: int
    end: int
    files: int


@dataclass
class FeedCoverage:
    source_id: str
    archive_count: int = 0
    csv_snapshot_count: int = 0
    row_count: int = 0
    archive_start: str | None = None
    archive_end: str | None = None
    observation_start: str | None = None
    observation_end: str | None = None
    stations: set[str] = field(default_factory=set)
    headers: set[tuple[str, ...]] = field(default_factory=set)

    def update_archive_time(self, value: str) -> None:
        self.archive_start = value if self.archive_start is None else min(self.archive_start, value)
        self.archive_end = value if self.archive_end is None else max(self.archive_end, value)

    def update_observation_time(self, value: str) -> None:
        self.observation_start = (
            value if self.observation_start is None else min(self.observation_start, value)
        )
        self.observation_end = value if self.observation_end is None else max(self.observation_end, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate offline station/weather coverage from downloaded acquisition archives."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("HKG_TMAX_DATA_ROOT", r"C:\hkg_tmax_data"),
        help="Canonical acquisition data root.",
    )
    parser.add_argument(
        "--output",
        default="reports/station_weather_coverage.md",
        help="Markdown report path to write inside the repository.",
    )
    return parser.parse_args()


def read_ledger(data_root: Path) -> list[dict[str, str]]:
    with (data_root / "manifests" / "retrieval_ledger.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


def query_time_from_url(url: str) -> str | None:
    match = re.search(r"[?&]time=(\d{8})", url)
    return match.group(1) if match else None


def csv_entry_name_time(name: str) -> str | None:
    match = re.search(r"/(\d{8})-(\d{4})-[^/]+\.csv$", name)
    if not match:
        return None
    return match.group(1) + match.group(2)


def decode_text(bytes_value: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "big5", "latin-1"):
        try:
            return bytes_value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return bytes_value.decode("utf-8", errors="replace")


def normalized_station(value: str) -> str:
    return " ".join(value.strip().split())


def is_station_name(value: str) -> bool:
    return bool(re.search(r"[A-Za-z]", value)) and not re.fullmatch(r"\d+(?:\.\d+)?", value)


def parse_csv_snapshot(coverage: FeedCoverage, text: str, entry_time: str | None) -> None:
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return
    header = [column.strip() for column in header]
    coverage.headers.add(tuple(header))
    timestamp_col = 0 if header and header[0].lower().replace(" ", "") == "datetime" else None
    station_col = None
    for index, column in enumerate(header):
        if column.lower() == "automatic weather station":
            station_col = index
            break
    if entry_time:
        coverage.update_observation_time(entry_time)
    for row in reader:
        if not row:
            continue
        if timestamp_col is not None and (
            len(row) <= timestamp_col or not re.fullmatch(r"\d{12}", row[timestamp_col].strip())
        ):
            continue
        coverage.row_count += 1
        if timestamp_col is not None and len(row) > timestamp_col:
            value = row[timestamp_col].strip()
            if re.fullmatch(r"\d{12}", value):
                coverage.update_observation_time(value)
        if station_col is not None and len(row) > station_col:
            station = normalized_station(row[station_col])
            if station and station.upper() != "N/A" and is_station_name(station):
                coverage.stations.add(station)


def hko_high_frequency_coverage(rows: list[dict[str, str]]) -> list[FeedCoverage]:
    coverage_by_source: dict[str, FeedCoverage] = {}
    for row in rows:
        source_id = row.get("source_id", "")
        if not (
            row.get("status") == "success"
            and source_id.startswith(HKO_HF_PREFIX)
            and source_id.endswith("_archive")
        ):
            continue
        coverage = coverage_by_source.setdefault(source_id, FeedCoverage(source_id=source_id))
        coverage.archive_count += 1
        archive_time = query_time_from_url(row.get("request_url", ""))
        if archive_time:
            coverage.update_archive_time(archive_time)
        content_path = Path(row.get("content_path", ""))
        with zipfile.ZipFile(content_path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                coverage.csv_snapshot_count += 1
                entry_time = csv_entry_name_time(name)
                with archive.open(name) as handle:
                    text = decode_text(handle.read())
                parse_csv_snapshot(coverage, text, entry_time)
    return [coverage_by_source[key] for key in sorted(coverage_by_source)]


def noaa_isd_station_coverage(rows: list[dict[str, str]]) -> list[StationYearRow]:
    station_years: dict[str, list[int]] = defaultdict(list)
    pattern = re.compile(r"/((?:19|20)\d{2})/(\d{6}-\d{5})-((?:19|20)\d{2})\.gz$")
    for row in rows:
        if row.get("status") != "success" or row.get("source_id") != "noaa_isd_nearby_station_year":
            continue
        match = pattern.search(row.get("request_url", ""))
        if not match:
            continue
        station_years[match.group(2)].append(int(match.group(1)))
    return [
        {
            "station": station,
            "start": min(years),
            "end": max(years),
            "files": len(years),
        }
        for station, years in sorted(station_years.items())
    ]


def write_report(output: Path, data_root: Path, rows: list[dict[str, str]]) -> None:
    hko_feeds = hko_high_frequency_coverage(rows)
    noaa_stations = noaa_isd_station_coverage(rows)
    lines = [
        "# Station Weather Coverage",
        "",
        "Generated offline from downloaded raw archives. This report does not write to",
        f"`{data_root}` and does not perform modelling.",
        "",
        "## HKO High-Frequency Historical Station Feeds",
        "",
        "| Feed | Archives | CSV snapshots | Rows | Archive start | Archive end | Observation start | Observation end | Stations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for feed in hko_feeds:
        lines.append(
            "| {source_id} | {archives:,} | {snapshots:,} | {rows:,} | {archive_start} | {archive_end} | {obs_start} | {obs_end} | {stations:,} |".format(
                source_id=feed.source_id,
                archives=feed.archive_count,
                snapshots=feed.csv_snapshot_count,
                rows=feed.row_count,
                archive_start=feed.archive_start or "",
                archive_end=feed.archive_end or "",
                obs_start=feed.observation_start or "",
                obs_end=feed.observation_end or "",
                stations=len(feed.stations),
            )
        )

    lines.extend(["", "## HKO Station Names Observed By Feed", ""])
    for feed in hko_feeds:
        station_text = ", ".join(sorted(feed.stations)) if feed.stations else "(feed is not station-row based)"
        lines.extend([f"### {feed.source_id}", "", station_text, ""])

    lines.extend(
        [
            "## NOAA ISD Nearby / Surrounding Station Files",
            "",
            f"Station histories: `{len(noaa_stations):,}`; station-year files: `{sum(row['files'] for row in noaa_stations):,}`.",
            "",
            "| Station | Start | End | Files |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in noaa_stations:
        lines.append("| {station} | {start} | {end} | {files:,} |".format(**row))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    rows = read_ledger(data_root)
    write_report(Path(args.output), data_root, rows)


if __name__ == "__main__":
    main()
