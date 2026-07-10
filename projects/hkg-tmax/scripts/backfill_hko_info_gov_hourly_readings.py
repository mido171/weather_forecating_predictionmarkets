"""Backfill Info.gov HKO hourly-reading dispatches and load one Postgres table.

This source is separate from the historical forecast archive. It targets only
Info.gov pages titled "PRESS WEATHER NO. ### - HOURLY READINGS" and preserves
every discovered dispatch page, including days with more than 24 dispatches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup

from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
DEFAULT_DATASET_ROOT = PROJECT_PATHS.data_root / "datasets" / "13_hko_info_gov_hourly_readings"
DEFAULT_TABLE = "public.hko_info_gov_hourly_readings_1998_2026"
INFO_GOV_INDEX = "https://www.info.gov.hk/gia/wr/{year_month}/{day}.htm"
HKT = ZoneInfo("Asia/Hong_Kong")
_THREAD_LOCAL = threading.local()

HOURLY_TITLE_RE = re.compile(r"\bPRESS WEATHER NO\.\s*(\d+)\s*-\s*HOURLY READINGS\b", re.I)
DISPATCH_RE = re.compile(
    r"DISPATCHED BY HONG KONG OBSERVATORY AT\s*"
    r"(?P<clock>\d{1,2}:\d{2})\s*HKT\s+ON\s+"
    r"(?P<day>\d{1,2})[./-](?P<month>\d{1,2})[./-](?P<year>\d{2,4})",
    re.I,
)
HKO_OBS_RE = re.compile(
    r"AT\s+(?P<obs_label>NOON|MIDNIGHT|\d{1,2}(?::\d{2})?\s*(?:A\.?M\.?|P\.?M\.?))"
    r"\s+AT\s+THE\s+HONG\s+KONG\s+OBSERVATORY\s+THE\s+AIR\s+TEMPERATURE\s+WAS\s+"
    r"(?P<temp>-?\d+(?:\.\d+)?)\s+DEGREES\s+CELSIUS\s+AND\s+THE\s+RELATIVE\s+HUMIDITY\s+"
    r"(?P<rh>\d+(?:\.\d+)?)\s+PER\s+CENT",
    re.I | re.S,
)
STATION_TEMP_RE = re.compile(
    r"(?P<name>[A-Z][A-Z0-9'(). /-]*?)\s+(?P<value>//|-?\d+(?:\.\d+)?)\s+DEGREES\b",
    re.I,
)
NON_STATION_NAME_RE = re.compile(
    r"\b(?:BETWEEN|CENTRE|CENTER|DEPRESSION|LAST NIGHT|MAXIMUM TEMPERATURE|MINIMUM TEMPERATURE|"
    r"STORM|TROPICAL|TYPHOON|WAS NEAR)\b",
    re.I,
)
TC_LOCATION_RE = re.compile(
    r"LOCATION:\s*(?P<lat>\d+(?:\.\d+)?)\s+DEGREES\s+(?P<lat_dir>NORTH|SOUTH),\s*"
    r"(?P<lon>\d+(?:\.\d+)?)\s+DEGREES\s+(?P<lon_dir>EAST|WEST)",
    re.I,
)
TC_OLD_LOCATION_RE = re.compile(
    r"\bCENT(?:RE|ER)\s+OF\s+(?:[A-Z ]+?)\s+WAS\s+NEAR\s+"
    r"(?P<lat>\d+(?:\.\d+)?)\s+DEGREES\s+(?P<lat_dir>NORTH|SOUTH)\s+"
    r"(?P<lon>\d+(?:\.\d+)?)\s+DEGREES\s+(?P<lon_dir>EAST|WEST)",
    re.I,
)
TC_NAME_RE = re.compile(
    r"HERE IS THE INFORMATION ON\s+(?:SEVERE\s+)?(?:TROPICAL\s+STORM|TYPHOON|SUPER\s+TYPHOON|"
    r"TROPICAL\s+DEPRESSION|SEVERE\s+TYPHOON)\s+(?P<name>[A-Z][A-Z -]+?)\s+AT\b",
    re.I,
)

COPY_COLUMNS = [
    "bulletin_id",
    "source",
    "source_url",
    "index_date_hkt",
    "title",
    "press_weather_no",
    "dispatch_at_hkt",
    "dispatch_at_utc",
    "observation_at_hkt",
    "observation_at_utc",
    "available_at_utc",
    "retrieved_at_utc",
    "hko_air_temp_c",
    "hko_relative_humidity_pct",
    "rainfall_text",
    "warning_text",
    "lightning_text",
    "tropical_cyclone_text",
    "tropical_cyclone_name",
    "tropical_cyclone_lat",
    "tropical_cyclone_lon",
    "station_readings_jsonb",
    "station_count",
    "station_missing_count",
    "station_temp_min_c",
    "station_temp_max_c",
    "station_temp_mean_c",
    "station_temp_spread_c",
    "target_station_present",
    "full_text",
    "raw_html_path",
    "raw_sha256",
    "parse_status",
    "parse_notes",
    "ingested_at_utc",
]


@dataclass(frozen=True)
class HourlyCandidate:
    index_date_hkt: date
    title: str
    press_weather_no: int
    source_url: str


@dataclass(frozen=True)
class FetchOutcome:
    url: str
    status_code: int | None
    content: bytes | None
    error: str | None
    headers: dict[str, str]
    retrieved_at_utc: str


@dataclass(frozen=True)
class ParsedDispatch:
    bulletin_id: str
    source: str
    source_url: str
    index_date_hkt: str | None
    title: str | None
    press_weather_no: int | None
    dispatch_at_hkt: str | None
    dispatch_at_utc: str | None
    observation_at_hkt: str | None
    observation_at_utc: str | None
    available_at_utc: str | None
    retrieved_at_utc: str | None
    hko_air_temp_c: float | None
    hko_relative_humidity_pct: float | None
    rainfall_text: str | None
    warning_text: str | None
    lightning_text: str | None
    tropical_cyclone_text: str | None
    tropical_cyclone_name: str | None
    tropical_cyclone_lat: float | None
    tropical_cyclone_lon: float | None
    station_readings_jsonb: list[dict[str, Any]]
    station_count: int
    station_missing_count: int
    station_temp_min_c: float | None
    station_temp_max_c: float | None
    station_temp_mean_c: float | None
    station_temp_spread_c: float | None
    target_station_present: bool
    full_text: str
    raw_html_path: str
    raw_sha256: str
    parse_status: str
    parse_notes: str | None
    ingested_at_utc: str

    def csv_row(self) -> dict[str, object]:
        def clean(value: object) -> object:
            if isinstance(value, str):
                return value.replace("\x00", "")
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            return value

        values = {key: clean(value) for key, value in self.__dict__.items()}
        values["station_readings_jsonb"] = json.dumps(
            clean(self.station_readings_jsonb), ensure_ascii=False, sort_keys=True
        )
        values["target_station_present"] = "true" if self.target_station_present else "false"
        return values


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def default_end_hkt() -> date:
    return datetime.now(UTC).astimezone(HKT).date()


def iter_dates(start: date, end: date) -> Iterator[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def normalize_text(value: str) -> str:
    value = value.replace("\x00", "")
    value = value.replace("\xa0", " ")
    value = value.replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def html_to_text(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    return normalize_text(soup.get_text("\n", strip=True))


def canonical_station_name(value: str) -> str:
    value = value.replace("’", "'").replace("`", "'")
    value = re.sub(r"\s+", " ", value.upper()).strip(" ;.")
    aliases = {
        "SHATIN": "SHA TIN",
    }
    return aliases.get(value, value)


def is_station_name(value: str) -> bool:
    if not value:
        return False
    if any(char.isdigit() for char in value):
        return False
    if value in {"NEAR", "NORTH", "SOUTH", "EAST", "WEST"}:
        return False
    return NON_STATION_NAME_RE.search(value) is None


def bulletin_id_for(source_url: str) -> str:
    return hashlib.sha256(f"info_gov_hourly_readings|{source_url}".encode("utf-8")).hexdigest()


def parse_index_candidates(content: bytes, index_date_hkt: date, index_url: str) -> list[HourlyCandidate]:
    soup = BeautifulSoup(content, "html.parser")
    candidates: dict[str, HourlyCandidate] = {}
    for link in soup.find_all("a"):
        title = normalize_text(link.get_text(" ", strip=True))
        match = HOURLY_TITLE_RE.search(title)
        href = link.get("href")
        if not match or not href:
            continue
        source_url = urljoin(index_url, href)
        candidates[source_url] = HourlyCandidate(
            index_date_hkt=index_date_hkt,
            title=match.group(0).upper().replace("  ", " "),
            press_weather_no=int(match.group(1)),
            source_url=source_url,
        )
    return sorted(candidates.values(), key=lambda item: (item.index_date_hkt, item.press_weather_no, item.source_url))


def parse_dispatch_timestamp(text: str) -> datetime | None:
    match = DISPATCH_RE.search(text)
    if not match:
        return None
    year = int(match.group("year"))
    if year < 100:
        year += 2000 if year < 70 else 1900
    hour, minute = map(int, match.group("clock").split(":"))
    return datetime(
        year,
        int(match.group("month")),
        int(match.group("day")),
        hour,
        minute,
        tzinfo=HKT,
    )


def parse_observation_clock(label: str, dispatch_at_hkt: datetime | None) -> datetime | None:
    if dispatch_at_hkt is None:
        return None
    cleaned = label.upper().replace(" ", "")
    if cleaned == "NOON":
        return datetime.combine(dispatch_at_hkt.date(), dt_time(12, 0), HKT)
    if cleaned == "MIDNIGHT":
        return datetime.combine(dispatch_at_hkt.date(), dt_time(0, 0), HKT)
    match = re.match(r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<period>A\.?M\.?|P\.?M\.?)", cleaned)
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or "0")
    period = match.group("period")[0]
    if period == "P" and hour != 12:
        hour += 12
    if period == "A" and hour == 12:
        hour = 0
    return datetime.combine(dispatch_at_hkt.date(), dt_time(hour, minute), HKT)


def extract_block(text: str, start_pattern: str, end_patterns: Iterable[str]) -> str | None:
    start = re.search(start_pattern, text, re.I | re.S)
    if not start:
        return None
    begin = start.start()
    end = len(text)
    for pattern in end_patterns:
        match = re.search(pattern, text[start.end() :], re.I | re.S)
        if match:
            end = min(end, start.end() + match.start())
    return normalize_text(text[begin:end]) or None


def extract_station_section(text: str) -> str | None:
    start = re.search(r"THE AIR TEMPERATURES AT OTHER PLACES WERE\s*:", text, re.I)
    if not start:
        return None
    tail = text[start.end() :]
    stops = [
        r"\n\s*PLEASE BE REMINDED THAT\s*:",
        r"\n\s*BETWEEN\b",
        r"\n\s*AT\s+(?:NOON|MIDNIGHT|\d{1,2}(?::\d{2})?\s*(?:A\.?M\.?|P\.?M\.?))(?:\s+LAST\s+NIGHT)?\s+THE\s+CENT(?:RE|ER)\b",
        r"\n\s*[A-Z][A-Z -]+\s+WAS\s+NEAR\b",
        r"\n\s*HERE IS THE INFORMATION ON\b",
        r"\n\s*DISPATCHED BY HONG KONG OBSERVATORY\b",
        r"\n\s*Archives\b",
        r"\n\s*Back to Index Page\b",
    ]
    end = len(tail)
    for pattern in stops:
        match = re.search(pattern, tail, re.I)
        if match:
            end = min(end, match.start())
    return normalize_text(tail[:end]) or None


def parse_station_readings(text: str) -> list[dict[str, Any]]:
    section = extract_station_section(text)
    if not section:
        return []
    readings: list[dict[str, Any]] = []
    for match in STATION_TEMP_RE.finditer(section):
        display = canonical_station_name(match.group("name"))
        if not is_station_name(display):
            continue
        raw_value = normalize_text(match.group("value") + " DEGREES")
        missing = match.group("value") == "//"
        temperature = None if missing else float(match.group("value"))
        if temperature is not None and not -20 <= temperature <= 60:
            continue
        raw_line = normalize_text(match.group(0)).rstrip(";.")
        readings.append(
            {
                "station_display_name": display,
                "station_canonical_name": canonical_station_name(display),
                "temperature_c": temperature,
                "temperature_missing": missing,
                "raw_temperature_text": raw_value,
                "raw_station_line": raw_line,
                "station_order": len(readings) + 1,
            }
        )
    return readings


def extract_rainfall_text(text: str) -> str | None:
    return extract_block(
        text,
        r"(?:\d+(?:\.\d+)?\s+MILLIMETRES OF RAINFALL|RAINFALL)",
        [
            r"\n\s*PLEASE BE REMINDED THAT\s*:",
            r"\n\s*THE AIR TEMPERATURES AT OTHER PLACES WERE\s*:",
            r"\n\s*DISPATCHED BY HONG KONG OBSERVATORY\b",
        ],
    )


def extract_warning_text(text: str) -> str | None:
    return extract_block(
        text,
        r"PLEASE BE REMINDED THAT\s*:",
        [
            r"\n\s*THE AIR TEMPERATURES AT OTHER PLACES WERE\s*:",
            r"\n\s*BETWEEN\s+\d",
            r"\n\s*HERE IS THE INFORMATION ON\b",
            r"\n\s*DISPATCHED BY HONG KONG OBSERVATORY\b",
        ],
    )


def extract_lightning_text(text: str) -> str | None:
    return extract_block(
        text,
        r"BETWEEN\s+.*?LIGHTNING WAS DETECTED",
        [r"\n\s*HERE IS THE INFORMATION ON\b", r"\n\s*DISPATCHED BY HONG KONG OBSERVATORY\b"],
    )


def extract_tropical_cyclone_text(text: str) -> str | None:
    return extract_block(
        text,
        r"(?:HERE IS THE INFORMATION ON\b|AT\s+(?:NOON|MIDNIGHT|\d{1,2}(?::\d{2})?\s*(?:A\.?M\.?|P\.?M\.?))(?:\s+LAST\s+NIGHT)?\s+THE\s+CENT(?:RE|ER)\s+OF\b)",
        [r"\n\s*DISPATCHED BY HONG KONG OBSERVATORY\b"],
    )


def parse_tropical_cyclone_fields(tc_text: str | None) -> tuple[str | None, float | None, float | None]:
    if not tc_text:
        return None, None, None
    name_match = TC_NAME_RE.search(tc_text)
    name = canonical_station_name(name_match.group("name")) if name_match else None
    location = TC_LOCATION_RE.search(tc_text) or TC_OLD_LOCATION_RE.search(tc_text)
    if not location:
        return name, None, None
    lat = float(location.group("lat"))
    lon = float(location.group("lon"))
    if location.group("lat_dir").upper() == "SOUTH":
        lat = -lat
    if location.group("lon_dir").upper() == "WEST":
        lon = -lon
    return name, lat, lon


def parse_dispatch(
    *,
    content: bytes,
    source_url: str,
    raw_html_path: str,
    raw_sha256: str,
    index_date_hkt: date | None,
    title_hint: str | None,
    press_weather_no_hint: int | None,
    retrieved_at_utc: str | None,
) -> ParsedDispatch:
    text = html_to_text(content)
    title_match = HOURLY_TITLE_RE.search(text)
    title = title_match.group(0).upper().replace("  ", " ") if title_match else title_hint
    press_weather_no = int(title_match.group(1)) if title_match else press_weather_no_hint
    dispatch_at = parse_dispatch_timestamp(text)
    dispatch_utc = dispatch_at.astimezone(UTC) if dispatch_at else None

    hko_temp = hko_rh = None
    observation_at = None
    hko_match = HKO_OBS_RE.search(text)
    if hko_match:
        hko_temp = float(hko_match.group("temp"))
        hko_rh = float(hko_match.group("rh"))
        observation_at = parse_observation_clock(hko_match.group("obs_label"), dispatch_at)
    observation_utc = observation_at.astimezone(UTC) if observation_at else None

    station_readings = parse_station_readings(text)
    station_temps = [
        float(item["temperature_c"])
        for item in station_readings
        if item.get("temperature_c") is not None
    ]
    station_min = min(station_temps) if station_temps else None
    station_max = max(station_temps) if station_temps else None
    station_mean = sum(station_temps) / len(station_temps) if station_temps else None
    station_spread = station_max - station_min if station_min is not None and station_max is not None else None
    station_missing = sum(1 for item in station_readings if item["temperature_missing"])

    tc_text = extract_tropical_cyclone_text(text)
    tc_name, tc_lat, tc_lon = parse_tropical_cyclone_fields(tc_text)
    notes: list[str] = []
    if not title:
        notes.append("missing_title")
    if dispatch_at is None:
        notes.append("missing_dispatch_at")
    if hko_match is None:
        notes.append("missing_hko_temp_rh")
    if not station_readings:
        notes.append("missing_station_readings")
    parse_status = "parsed" if not notes else "partial"

    return ParsedDispatch(
        bulletin_id=bulletin_id_for(source_url),
        source="info_gov",
        source_url=source_url,
        index_date_hkt=index_date_hkt.isoformat() if index_date_hkt else None,
        title=title,
        press_weather_no=press_weather_no,
        dispatch_at_hkt=dispatch_at.replace(tzinfo=None).isoformat(sep=" ") if dispatch_at else None,
        dispatch_at_utc=dispatch_utc.isoformat().replace("+00:00", "Z") if dispatch_utc else None,
        observation_at_hkt=observation_at.replace(tzinfo=None).isoformat(sep=" ") if observation_at else None,
        observation_at_utc=observation_utc.isoformat().replace("+00:00", "Z") if observation_utc else None,
        available_at_utc=dispatch_utc.isoformat().replace("+00:00", "Z") if dispatch_utc else None,
        retrieved_at_utc=retrieved_at_utc,
        hko_air_temp_c=hko_temp,
        hko_relative_humidity_pct=hko_rh,
        rainfall_text=extract_rainfall_text(text),
        warning_text=extract_warning_text(text),
        lightning_text=extract_lightning_text(text),
        tropical_cyclone_text=tc_text,
        tropical_cyclone_name=tc_name,
        tropical_cyclone_lat=tc_lat,
        tropical_cyclone_lon=tc_lon,
        station_readings_jsonb=station_readings,
        station_count=len(station_readings),
        station_missing_count=station_missing,
        station_temp_min_c=station_min,
        station_temp_max_c=station_max,
        station_temp_mean_c=station_mean,
        station_temp_spread_c=station_spread,
        target_station_present=hko_temp is not None and hko_rh is not None,
        full_text=text,
        raw_html_path=raw_html_path,
        raw_sha256=raw_sha256,
        parse_status=parse_status,
        parse_notes=",".join(notes) if notes else None,
        ingested_at_utc=utc_now(),
    )


class HourlyArchiveStore:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root
        self.raw_root = dataset_root / "raw_html"
        self.normalized_root = dataset_root / "normalized"
        self.reports_root = dataset_root / "reports"
        self.metadata_root = dataset_root / "metadata"
        for path in (self.raw_root, self.normalized_root, self.reports_root, self.metadata_root):
            path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.metadata_root / "hourly_readings_archive.sqlite3"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def close(self) -> None:
        self.conn.close()

    def _init_db(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS index_pages (
                index_date_hkt TEXT PRIMARY KEY,
                index_url TEXT NOT NULL,
                status_code INTEGER,
                error TEXT,
                raw_html_path TEXT,
                raw_sha256 TEXT,
                retrieved_at_utc TEXT,
                discovered_hourly_count INTEGER NOT NULL DEFAULT 0,
                parsed_at_utc TEXT
            );
            CREATE TABLE IF NOT EXISTS detail_pages (
                source_url TEXT PRIMARY KEY,
                index_date_hkt TEXT,
                title TEXT,
                press_weather_no INTEGER,
                status_code INTEGER,
                error TEXT,
                raw_html_path TEXT,
                raw_sha256 TEXT,
                retrieved_at_utc TEXT,
                bulletin_id TEXT,
                parse_status TEXT,
                parse_notes TEXT,
                parsed_at_utc TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_detail_pages_index_date ON detail_pages(index_date_hkt);
            CREATE INDEX IF NOT EXISTS idx_detail_pages_status ON detail_pages(status_code, parse_status);
            """
        )
        self.conn.commit()

    def successful_index(self, day: date) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM index_pages WHERE index_date_hkt=? AND status_code BETWEEN 200 AND 299 AND raw_html_path IS NOT NULL",
            (day.isoformat(),),
        ).fetchone()

    def reusable_index(self, day: date) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT * FROM index_pages
            WHERE index_date_hkt=?
              AND (
                (status_code BETWEEN 200 AND 299 AND raw_html_path IS NOT NULL)
                OR status_code BETWEEN 400 AND 499
              )
            """,
            (day.isoformat(),),
        ).fetchone()

    def successful_detail(self, source_url: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM detail_pages WHERE source_url=? AND status_code BETWEEN 200 AND 299 AND raw_html_path IS NOT NULL",
            (source_url,),
        ).fetchone()

    def record_index(
        self,
        *,
        day: date,
        index_url: str,
        outcome: FetchOutcome,
        raw_path: str | None,
        digest: str | None,
        discovered_count: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO index_pages
            (index_date_hkt, index_url, status_code, error, raw_html_path, raw_sha256, retrieved_at_utc, discovered_hourly_count, parsed_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(index_date_hkt) DO UPDATE SET
                index_url=excluded.index_url,
                status_code=excluded.status_code,
                error=excluded.error,
                raw_html_path=excluded.raw_html_path,
                raw_sha256=excluded.raw_sha256,
                retrieved_at_utc=excluded.retrieved_at_utc,
                discovered_hourly_count=excluded.discovered_hourly_count,
                parsed_at_utc=excluded.parsed_at_utc
            """,
            (
                day.isoformat(),
                index_url,
                outcome.status_code,
                outcome.error,
                raw_path,
                digest,
                outcome.retrieved_at_utc,
                discovered_count,
                utc_now(),
            ),
        )
        self.conn.commit()

    def add_candidate(self, candidate: HourlyCandidate) -> None:
        self.conn.execute(
            """
            INSERT INTO detail_pages (source_url, index_date_hkt, title, press_weather_no)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_url) DO UPDATE SET
                index_date_hkt=coalesce(detail_pages.index_date_hkt, excluded.index_date_hkt),
                title=coalesce(detail_pages.title, excluded.title),
                press_weather_no=coalesce(detail_pages.press_weather_no, excluded.press_weather_no)
            """,
            (
                candidate.source_url,
                candidate.index_date_hkt.isoformat(),
                candidate.title,
                candidate.press_weather_no,
            ),
        )
        self.conn.commit()

    def record_detail_fetch(
        self,
        *,
        candidate: HourlyCandidate,
        outcome: FetchOutcome,
        raw_path: str | None,
        digest: str | None,
        commit: bool = True,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO detail_pages
            (source_url, index_date_hkt, title, press_weather_no, status_code, error, raw_html_path, raw_sha256, retrieved_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_url) DO UPDATE SET
                index_date_hkt=excluded.index_date_hkt,
                title=excluded.title,
                press_weather_no=excluded.press_weather_no,
                status_code=excluded.status_code,
                error=excluded.error,
                raw_html_path=excluded.raw_html_path,
                raw_sha256=excluded.raw_sha256,
                retrieved_at_utc=excluded.retrieved_at_utc
            """,
            (
                candidate.source_url,
                candidate.index_date_hkt.isoformat(),
                candidate.title,
                candidate.press_weather_no,
                outcome.status_code,
                outcome.error,
                raw_path,
                digest,
                outcome.retrieved_at_utc,
            ),
        )
        if commit:
            self.conn.commit()

    def record_parse_result(self, parsed: ParsedDispatch, *, commit: bool = True) -> None:
        self.conn.execute(
            """
            UPDATE detail_pages
            SET bulletin_id=?, parse_status=?, parse_notes=?, parsed_at_utc=?
            WHERE source_url=?
            """,
            (
                parsed.bulletin_id,
                parsed.parse_status,
                parsed.parse_notes,
                utc_now(),
                parsed.source_url,
            ),
        )
        if commit:
            self.conn.commit()

    def detail_rows(self) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT * FROM detail_pages
                WHERE status_code BETWEEN 200 AND 299 AND raw_html_path IS NOT NULL
                ORDER BY index_date_hkt, press_weather_no, source_url
                """
            )
        )

    def detail_failure_rows(self) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT source_url, index_date_hkt, title, press_weather_no, status_code, error, retrieved_at_utc
                FROM detail_pages
                WHERE NOT (status_code BETWEEN 200 AND 299 AND raw_html_path IS NOT NULL)
                ORDER BY index_date_hkt, press_weather_no, source_url
                """
            )
        )

    def discovered_detail_count(self) -> int:
        return int(self.conn.execute("SELECT count(*) FROM detail_pages").fetchone()[0])

    def fetched_detail_count(self) -> int:
        return int(
            self.conn.execute(
                "SELECT count(*) FROM detail_pages WHERE status_code BETWEEN 200 AND 299 AND raw_html_path IS NOT NULL"
            ).fetchone()[0]
        )

    def save_raw(self, *, kind: str, url: str, content: bytes, retrieved_at_utc: str, date_hint: date) -> tuple[str, str]:
        digest = sha256_bytes(content)
        if kind == "index":
            raw_path = self.raw_root / "index" / f"{date_hint:%Y}" / f"{date_hint:%m}" / f"{date_hint:%d}" / "index.html"
        else:
            basename = Path(urlparse(url).path).name or f"{digest}.html"
            if not basename.lower().endswith(".htm") and not basename.lower().endswith(".html"):
                basename = f"{basename}.html"
            raw_path = self.raw_root / "detail" / f"{date_hint:%Y}" / f"{date_hint:%m}" / f"{date_hint:%d}" / basename
        atomic_write(raw_path, content)
        write_json(
            raw_path.with_suffix(raw_path.suffix + ".metadata.json"),
            {
                "kind": kind,
                "url": url,
                "sha256": digest,
                "content_length": len(content),
                "retrieved_at_utc": retrieved_at_utc,
            },
        )
        return str(raw_path), digest


def extract_info_gov_challenge_cookies(text: str) -> dict[str, str] | None:
    if "__tst_status" not in text or "EO_Bot_Ssid" not in text or "location.href" not in text:
        return None
    numeric_assignments = [
        int(value)
        for value in re.findall(r"\b[A-Za-z_$][A-Za-z0-9_$]*\s*:\s*(\d{5,})\b", text)
    ]
    ssid_match = re.search(r"\]\(\s*t\s*,\s*(\d{5,})\s*\).*?case\s*[\"']4[\"']", text, re.S)
    if not numeric_assignments or ssid_match is None:
        return None
    return {
        "__tst_status": f"{sum(numeric_assignments)}#",
        "EO_Bot_Ssid": ssid_match.group(1),
    }


def fetch_url(client: httpx.Client, url: str, retries: int, timeout_sleep: float) -> FetchOutcome:
    last_error: str | None = None
    challenge_retried = False
    for attempt in range(retries + 1):
        retrieved_at = utc_now()
        try:
            response = client.get(url)
            headers = dict(response.headers)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                sleep_for = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 2.0**attempt)
                time.sleep(sleep_for)
                continue
            if response.status_code >= 400:
                return FetchOutcome(url, response.status_code, None, f"HTTP {response.status_code}", headers, retrieved_at)
            challenge_cookies = extract_info_gov_challenge_cookies(response.text)
            if challenge_cookies and not challenge_retried:
                for name, value in challenge_cookies.items():
                    client.cookies.set(name, value, domain="www.info.gov.hk", path="/")
                challenge_retried = True
                continue
            if timeout_sleep > 0:
                time.sleep(timeout_sleep)
            return FetchOutcome(url, response.status_code, response.content, None, headers, retrieved_at)
        except Exception as exc:  # noqa: BLE001 - final error is recorded in the ledger.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(60.0, 2.0**attempt))
                continue
            return FetchOutcome(url, None, None, last_error, {}, retrieved_at)
    return FetchOutcome(url, None, None, last_error or "unknown_error", {}, utc_now())


def thread_client(timeout_seconds: float, user_agent: str) -> httpx.Client:
    key = (timeout_seconds, user_agent)
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None or getattr(_THREAD_LOCAL, "client_key", None) != key:
        if client is not None:
            client.close()
        client = httpx.Client(
            follow_redirects=True,
            timeout=timeout_seconds,
            headers={"User-Agent": user_agent},
        )
        _THREAD_LOCAL.client = client
        _THREAD_LOCAL.client_key = key
    return client


def _fetch_index_worker(
    day: date,
    retries: int,
    timeout_seconds: float,
    delay_seconds: float,
    user_agent: str,
) -> tuple[date, str, FetchOutcome]:
    index_url = INFO_GOV_INDEX.format(year_month=f"{day:%Y%m}", day=f"{day:%d}")
    outcome = fetch_url(thread_client(timeout_seconds, user_agent), index_url, retries, delay_seconds)
    return day, index_url, outcome


def discover_candidates(args: argparse.Namespace, store: HourlyArchiveStore) -> list[HourlyCandidate]:
    start = parse_date(args.start)
    end = parse_date(args.end)
    candidates: dict[str, HourlyCandidate] = {}
    days = list(iter_dates(start, end))
    pending_days: list[date] = []
    completed_days = 0

    def record_day(
        *,
        day: date,
        index_url: str,
        outcome: FetchOutcome,
        content: bytes | None,
        raw_path: str | None,
        digest: str | None,
    ) -> None:
        nonlocal completed_days
        if content is not None and raw_path is None:
            raw_path, digest = store.save_raw(
                kind="index",
                url=index_url,
                content=content,
                retrieved_at_utc=outcome.retrieved_at_utc,
                date_hint=day,
            )
        day_candidates: list[HourlyCandidate] = []
        if content is not None:
            day_candidates = parse_index_candidates(content, day, index_url)
            for candidate in day_candidates:
                store.add_candidate(candidate)
                candidates[candidate.source_url] = candidate
        store.record_index(
            day=day,
            index_url=index_url,
            outcome=outcome,
            raw_path=raw_path,
            digest=digest,
            discovered_count=len(day_candidates),
        )
        completed_days += 1
        if completed_days % args.progress_interval_days == 0 or completed_days == len(days):
            print(
                json.dumps(
                    {
                        "event": "index_progress",
                        "days": completed_days,
                        "date": day.isoformat(),
                        "candidate_count": len(candidates),
                        "ts": utc_now(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    for day in days:
        index_url = INFO_GOV_INDEX.format(year_month=f"{day:%Y%m}", day=f"{day:%d}")
        row = store.reusable_index(day) if not args.force_index else None
        if row is None:
            pending_days.append(day)
            continue
        raw_path = str(row["raw_html_path"]) if row["raw_html_path"] else None
        content = Path(raw_path).read_bytes() if raw_path else None
        outcome = FetchOutcome(
            index_url,
            int(row["status_code"]) if row["status_code"] is not None else None,
            content,
            str(row["error"]) if row["error"] else None,
            {},
            str(row["retrieved_at_utc"]),
        )
        record_day(
            day=day,
            index_url=index_url,
            outcome=outcome,
            content=content,
            raw_path=raw_path,
            digest=str(row["raw_sha256"]) if row["raw_sha256"] else None,
        )

    if pending_days:
        print(
            json.dumps(
                {
                    "event": "index_fetch_start",
                    "cached_days": completed_days,
                    "pending_days": len(pending_days),
                    "ts": utc_now(),
                    "workers": args.workers,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    _fetch_index_worker,
                    day,
                    args.retries,
                    args.timeout_seconds,
                    args.delay_seconds,
                    args.user_agent,
                ): day
                for day in pending_days
            }
            for future in as_completed(future_map):
                day, index_url, outcome = future.result()
                record_day(
                    day=day,
                    index_url=index_url,
                    outcome=outcome,
                    content=outcome.content,
                    raw_path=None,
                    digest=None,
                )
    else:
        print(
            json.dumps(
                {
                    "event": "index_fetch_start",
                    "cached_days": completed_days,
                    "pending_days": 0,
                    "ts": utc_now(),
                    "workers": args.workers,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return sorted(candidates.values(), key=lambda item: (item.index_date_hkt, item.press_weather_no, item.source_url))


def _fetch_detail_worker(
    candidate: HourlyCandidate,
    retries: int,
    timeout_seconds: float,
    user_agent: str,
    delay_seconds: float,
) -> tuple[HourlyCandidate, FetchOutcome]:
    outcome = fetch_url(thread_client(timeout_seconds, user_agent), candidate.source_url, retries, delay_seconds)
    return candidate, outcome


def fetch_details(args: argparse.Namespace, store: HourlyArchiveStore, candidates: list[HourlyCandidate]) -> None:
    pending = [candidate for candidate in candidates if args.force_details or not store.successful_detail(candidate.source_url)]
    commit_every = 500
    print(
        json.dumps(
            {
                "event": "detail_fetch_start",
                "candidates": len(candidates),
                "pending": len(pending),
                "workers": args.workers,
                "ts": utc_now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending_iter = iter(pending)
        futures = {}
        max_inflight = max(args.workers, args.workers * 4)

        def submit_next() -> bool:
            try:
                candidate = next(pending_iter)
            except StopIteration:
                return False
            future = executor.submit(
                _fetch_detail_worker,
                candidate,
                args.retries,
                args.timeout_seconds,
                args.user_agent,
                args.delay_seconds,
            )
            futures[future] = candidate
            return True

        for _ in range(min(max_inflight, len(pending))):
            submit_next()

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                candidate, outcome = future.result()
                raw_path = digest = None
                if outcome.content is not None:
                    raw_path, digest = store.save_raw(
                        kind="detail",
                        url=candidate.source_url,
                        content=outcome.content,
                        retrieved_at_utc=outcome.retrieved_at_utc,
                        date_hint=candidate.index_date_hkt,
                    )
                store.record_detail_fetch(
                    candidate=candidate,
                    outcome=outcome,
                    raw_path=raw_path,
                    digest=digest,
                    commit=False,
                )
                completed += 1
                if completed % commit_every == 0:
                    store.conn.commit()
                if completed % args.progress_interval_details == 0 or completed == len(pending):
                    store.conn.commit()
                    print(
                        json.dumps(
                            {
                                "event": "detail_progress",
                                "completed": completed,
                                "pending": len(pending),
                                "fetched_total": store.fetched_detail_count(),
                                "ts": utc_now(),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                submit_next()
    store.conn.commit()


def normalize_dispatches(args: argparse.Namespace, store: HourlyArchiveStore) -> Path:
    output_path = store.normalized_root / "hko_info_gov_hourly_readings.csv"
    rows = store.detail_rows()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parse_failures: list[dict[str, object]] = []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COPY_COLUMNS)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            raw_path = Path(str(row["raw_html_path"]))
            try:
                content = raw_path.read_bytes()
                parsed = parse_dispatch(
                    content=content,
                    source_url=str(row["source_url"]),
                    raw_html_path=str(raw_path),
                    raw_sha256=str(row["raw_sha256"]),
                    index_date_hkt=parse_date(str(row["index_date_hkt"])) if row["index_date_hkt"] else None,
                    title_hint=str(row["title"]) if row["title"] else None,
                    press_weather_no_hint=int(row["press_weather_no"]) if row["press_weather_no"] is not None else None,
                    retrieved_at_utc=str(row["retrieved_at_utc"]) if row["retrieved_at_utc"] else None,
                )
            except Exception as exc:  # noqa: BLE001 - failed raw pages are still represented.
                parsed = ParsedDispatch(
                    bulletin_id=bulletin_id_for(str(row["source_url"])),
                    source="info_gov",
                    source_url=str(row["source_url"]),
                    index_date_hkt=str(row["index_date_hkt"]) if row["index_date_hkt"] else None,
                    title=str(row["title"]) if row["title"] else None,
                    press_weather_no=int(row["press_weather_no"]) if row["press_weather_no"] is not None else None,
                    dispatch_at_hkt=None,
                    dispatch_at_utc=None,
                    observation_at_hkt=None,
                    observation_at_utc=None,
                    available_at_utc=None,
                    retrieved_at_utc=str(row["retrieved_at_utc"]) if row["retrieved_at_utc"] else None,
                    hko_air_temp_c=None,
                    hko_relative_humidity_pct=None,
                    rainfall_text=None,
                    warning_text=None,
                    lightning_text=None,
                    tropical_cyclone_text=None,
                    tropical_cyclone_name=None,
                    tropical_cyclone_lat=None,
                    tropical_cyclone_lon=None,
                    station_readings_jsonb=[],
                    station_count=0,
                    station_missing_count=0,
                    station_temp_min_c=None,
                    station_temp_max_c=None,
                    station_temp_mean_c=None,
                    station_temp_spread_c=None,
                    target_station_present=False,
                    full_text="",
                    raw_html_path=str(raw_path),
                    raw_sha256=str(row["raw_sha256"]),
                    parse_status="failed",
                    parse_notes=f"{type(exc).__name__}: {exc}",
                    ingested_at_utc=utc_now(),
                )
                parse_failures.append(
                    {
                        "source_url": row["source_url"],
                        "raw_html_path": str(raw_path),
                        "error": parsed.parse_notes,
                    }
                )
            writer.writerow(parsed.csv_row())
            store.record_parse_result(parsed, commit=False)
            if index % 1000 == 0:
                store.conn.commit()
            if index % args.progress_interval_details == 0:
                print(
                    json.dumps(
                        {"event": "normalize_progress", "rows": index, "total": len(rows), "ts": utc_now()},
                        sort_keys=True,
                    ),
                    flush=True,
                )
    store.conn.commit()
    write_json(store.reports_root / "parse_failures.json", parse_failures)
    write_reports(store, output_path)
    return output_path


def _csv_number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def write_reports(store: HourlyArchiveStore, normalized_csv: Path) -> None:
    daily = Counter()
    hourly = Counter()
    parse_status = Counter()
    station_seen: dict[str, Counter] = defaultdict(Counter)
    missing_by_station = Counter()
    station_count_by_year: dict[str, Counter] = defaultdict(Counter)
    first_dispatch = last_dispatch = None
    row_count = 0
    with normalized_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            index_date = row["index_date_hkt"] or "UNKNOWN"
            year = index_date[:4]
            daily[index_date] += 1
            if row["dispatch_at_hkt"]:
                hhmm = row["dispatch_at_hkt"][11:16]
                hourly[hhmm] += 1
                first_dispatch = row["dispatch_at_hkt"] if first_dispatch is None else min(first_dispatch, row["dispatch_at_hkt"])
                last_dispatch = row["dispatch_at_hkt"] if last_dispatch is None else max(last_dispatch, row["dispatch_at_hkt"])
            parse_status[row["parse_status"]] += 1
            try:
                stations = json.loads(row["station_readings_jsonb"])
            except json.JSONDecodeError:
                stations = []
            for station in stations:
                name = str(station.get("station_canonical_name") or "")
                if not name:
                    continue
                station_seen[name][year] += 1
                station_count_by_year[year][name] += 1
                if station.get("temperature_missing"):
                    missing_by_station[name] += 1

    detail_failure_count = write_detail_fetch_failure_reports(store)

    with (store.reports_root / "daily_dispatch_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index_date_hkt", "dispatch_count"])
        writer.writeheader()
        for index_date, count in sorted(daily.items()):
            writer.writerow({"index_date_hkt": index_date, "dispatch_count": count})
    with (store.reports_root / "issue_time_cadence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dispatch_hhmm_hkt", "count"])
        writer.writeheader()
        for hhmm, count in sorted(hourly.items()):
            writer.writerow({"dispatch_hhmm_hkt": hhmm, "count": count})
    with (store.reports_root / "station_coverage_by_year.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["year", "station_canonical_name", "dispatch_count"])
        writer.writeheader()
        for year, station_counts in sorted(station_count_by_year.items()):
            for station, count in sorted(station_counts.items()):
                writer.writerow({"year": year, "station_canonical_name": station, "dispatch_count": count})
    with (store.reports_root / "station_missing_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["station_canonical_name", "missing_count"])
        writer.writeheader()
        for station, count in sorted(missing_by_station.items()):
            writer.writerow({"station_canonical_name": station, "missing_count": count})

    summary = {
        "generated_at_utc": utc_now(),
        "normalized_csv": str(normalized_csv),
        "row_count": row_count,
        "first_dispatch_at_hkt": first_dispatch,
        "last_dispatch_at_hkt": last_dispatch,
        "first_index_date_hkt": min(daily) if daily else None,
        "last_index_date_hkt": max(daily) if daily else None,
        "daily_dispatch_count_min": min(daily.values()) if daily else None,
        "daily_dispatch_count_max": max(daily.values()) if daily else None,
        "daily_dispatch_count_mode": daily.most_common(1)[0][1] if daily else None,
        "parse_status_counts": dict(parse_status),
        "detail_fetch_failure_count": detail_failure_count,
        "unique_station_count": len(station_seen),
        "stations": sorted(station_seen),
    }
    write_json(store.reports_root / "structure_pattern_report.json", summary)
    (store.reports_root / "structure_pattern_report.md").write_text(
        "\n".join(
            [
                "# HKO Info.gov Hourly Readings Structure Pattern Report",
                "",
                f"- Generated at UTC: `{summary['generated_at_utc']}`",
                f"- Normalized dispatch rows: `{row_count}`",
                f"- First dispatch HKT: `{first_dispatch}`",
                f"- Last dispatch HKT: `{last_dispatch}`",
                f"- First index date HKT: `{summary['first_index_date_hkt']}`",
                f"- Last index date HKT: `{summary['last_index_date_hkt']}`",
                f"- Min dispatches per indexed day: `{summary['daily_dispatch_count_min']}`",
                f"- Max dispatches per indexed day: `{summary['daily_dispatch_count_max']}`",
                f"- Parse status counts: `{json.dumps(dict(parse_status), sort_keys=True)}`",
                f"- Detail URLs discovered but not loaded: `{detail_failure_count}`",
                f"- Unique stations in per-dispatch station JSON: `{len(station_seen)}`",
                "",
                "All station temperatures are stored per dispatch in `station_readings_jsonb`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_detail_fetch_failure_reports(store: HourlyArchiveStore) -> int:
    rows = [
        {
            "source_url": str(row["source_url"]),
            "index_date_hkt": str(row["index_date_hkt"]) if row["index_date_hkt"] else None,
            "title": str(row["title"]) if row["title"] else None,
            "press_weather_no": int(row["press_weather_no"]) if row["press_weather_no"] is not None else None,
            "status_code": int(row["status_code"]) if row["status_code"] is not None else None,
            "error": str(row["error"]) if row["error"] else None,
            "retrieved_at_utc": str(row["retrieved_at_utc"]) if row["retrieved_at_utc"] else None,
        }
        for row in store.detail_failure_rows()
    ]
    csv_path = store.reports_root / "detail_fetch_failures.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_url",
                "index_date_hkt",
                "title",
                "press_weather_no",
                "status_code",
                "error",
                "retrieved_at_utc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_json(store.reports_root / "detail_fetch_failures.json", rows)
    return len(rows)


def split_table_name(table: str) -> tuple[str, str]:
    parts = table.split(".")
    if len(parts) != 2 or not all(parts):
        raise ValueError("table must be schema.table")
    return parts[0], parts[1]


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def qualified_table(schema: str, table: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(table)}"


def table_ddl(fq_table: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {fq_table} (
    bulletin_id text PRIMARY KEY,
    source text NOT NULL,
    source_url text NOT NULL UNIQUE,
    index_date_hkt date,
    title text,
    press_weather_no integer,
    dispatch_at_hkt timestamp without time zone,
    dispatch_at_utc timestamp with time zone,
    observation_at_hkt timestamp without time zone,
    observation_at_utc timestamp with time zone,
    available_at_utc timestamp with time zone,
    retrieved_at_utc timestamp with time zone,
    hko_air_temp_c double precision,
    hko_relative_humidity_pct double precision,
    rainfall_text text,
    warning_text text,
    lightning_text text,
    tropical_cyclone_text text,
    tropical_cyclone_name text,
    tropical_cyclone_lat double precision,
    tropical_cyclone_lon double precision,
    station_readings_jsonb jsonb NOT NULL DEFAULT '[]'::jsonb,
    station_count integer NOT NULL,
    station_missing_count integer NOT NULL,
    station_temp_min_c double precision,
    station_temp_max_c double precision,
    station_temp_mean_c double precision,
    station_temp_spread_c double precision,
    target_station_present boolean NOT NULL,
    full_text text NOT NULL,
    raw_html_path text NOT NULL,
    raw_sha256 text NOT NULL,
    parse_status text NOT NULL,
    parse_notes text,
    ingested_at_utc timestamp with time zone NOT NULL,
    CHECK (source = 'info_gov'),
    CHECK (hko_air_temp_c IS NULL OR hko_air_temp_c BETWEEN -20 AND 60),
    CHECK (hko_relative_humidity_pct IS NULL OR hko_relative_humidity_pct BETWEEN 0 AND 100),
    CHECK (station_count >= 0),
    CHECK (station_missing_count >= 0),
    CHECK (station_missing_count <= station_count),
    CHECK (station_temp_min_c IS NULL OR station_temp_min_c BETWEEN -20 AND 60),
    CHECK (station_temp_max_c IS NULL OR station_temp_max_c BETWEEN -20 AND 60),
    CHECK (station_temp_min_c IS NULL OR station_temp_max_c IS NULL OR station_temp_min_c <= station_temp_max_c),
    CHECK (parse_status IN ('parsed', 'partial', 'failed'))
);
"""


def connect_db(database_url: str):
    import psycopg

    return psycopg.connect(database_url, options="-c timezone=UTC")


def load_postgres(args: argparse.Namespace, csv_path: Path) -> None:
    schema, table = split_table_name(args.table)
    fq_final = qualified_table(schema, table)
    stage = f"{table}_stage"
    fq_stage = qualified_table(schema, stage)
    with connect_db(args.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)}")
            cur.execute(f"DROP TABLE IF EXISTS {fq_stage}")
            cur.execute(table_ddl(fq_stage))
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                with cur.copy(
                    f"COPY {fq_stage} ({', '.join(quote_ident(col) for col in COPY_COLUMNS)}) "
                    "FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
                ) as copy:
                    while True:  # repo-doctor: allow-unsafe-default - exits at CSV EOF
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            break
                        chunk = chunk.replace("\x00", "")
                        copy.write(chunk)
            cur.execute(f"DROP TABLE IF EXISTS {fq_final} CASCADE")
            cur.execute(f"ALTER TABLE {fq_stage} RENAME TO {quote_ident(table)}")
            cur.execute(f"CREATE INDEX {quote_ident(table + '_dispatch_utc_idx')} ON {fq_final} (dispatch_at_utc)")
            cur.execute(f"CREATE INDEX {quote_ident(table + '_obs_utc_idx')} ON {fq_final} (observation_at_utc)")
            cur.execute(f"CREATE INDEX {quote_ident(table + '_index_date_idx')} ON {fq_final} (index_date_hkt)")
            cur.execute(f"CREATE INDEX {quote_ident(table + '_raw_sha_idx')} ON {fq_final} (raw_sha256)")
            cur.execute(f"CREATE INDEX {quote_ident(table + '_station_jsonb_gin_idx')} ON {fq_final} USING gin (station_readings_jsonb)")
            cur.execute(
                f"COMMENT ON TABLE {fq_final} IS "
                "'Canonical one-table Info.gov HKO hourly readings archive. One row per PRESS WEATHER HOURLY READINGS dispatch; station readings preserved in station_readings_jsonb.'"
            )
            cur.execute(f"ANALYZE {fq_final}")
        conn.commit()


def db_summary(args: argparse.Namespace) -> dict[str, object]:
    schema, table = split_table_name(args.table)
    fq = qualified_table(schema, table)
    with connect_db(args.database_url) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT jsonb_build_object(
                'table', %s::text,
                'rows', count(*),
                'first_dispatch_at_utc', min(dispatch_at_utc),
                'last_dispatch_at_utc', max(dispatch_at_utc),
                'first_index_date_hkt', min(index_date_hkt),
                'last_index_date_hkt', max(index_date_hkt),
                'parsed_rows', count(*) FILTER (WHERE parse_status='parsed'),
                'partial_rows', count(*) FILTER (WHERE parse_status='partial'),
                'failed_rows', count(*) FILTER (WHERE parse_status='failed'),
                'target_station_present_rows', count(*) FILTER (WHERE target_station_present),
                'max_station_count', max(station_count),
                'max_station_missing_count', max(station_missing_count)
            )::text
            FROM {fq}
            """,
            (args.table,),
        )
        return json.loads(cur.fetchone()[0])


def write_readme(dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    readme = dataset_root / "README.md"
    if readme.exists():
        return
    readme.write_text(
        "\n".join(
            [
                "# HKO Info.gov Hourly Readings",
                "",
                "Dedicated archive for Info.gov `PRESS WEATHER ... HOURLY READINGS` dispatch pages.",
                "",
                "- `raw_html/` contains downloaded index/detail HTML and metadata sidecars.",
                "- `metadata/hourly_readings_archive.sqlite3` is the resumable download ledger.",
                "- `normalized/hko_info_gov_hourly_readings.csv` is the one-row-per-dispatch normalized export used for DB loading.",
                "- `reports/` contains coverage, cadence, station, missingness, and parse-status reports.",
                "",
                "Canonical Postgres target: `public.hko_info_gov_hourly_readings_1998_2026`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root).resolve()
    write_readme(dataset_root)
    store = HourlyArchiveStore(dataset_root)
    try:
        if not args.skip_discovery:
            candidates = discover_candidates(args, store)
        else:
            candidates = [
                HourlyCandidate(
                    index_date_hkt=parse_date(str(row["index_date_hkt"])),
                    title=str(row["title"]),
                    press_weather_no=int(row["press_weather_no"]),
                    source_url=str(row["source_url"]),
                )
                for row in store.conn.execute(
                    "SELECT source_url, index_date_hkt, title, press_weather_no FROM detail_pages ORDER BY index_date_hkt, press_weather_no, source_url"
                )
            ]
        if not args.skip_details:
            fetch_details(args, store, candidates)
        normalized_csv = normalize_dispatches(args, store)
        if args.load_db:
            load_postgres(args, normalized_csv)
            summary = db_summary(args)
            write_json(dataset_root / "reports" / "postgres_load_summary.json", summary)
            print(json.dumps({"event": "postgres_loaded", **summary}, sort_keys=True), flush=True)
        print(
            json.dumps(
                {
                    "event": "done",
                    "dataset_root": str(dataset_root),
                    "discovered_detail_count": store.discovered_detail_count(),
                    "fetched_detail_count": store.fetched_detail_count(),
                    "normalized_csv": str(normalized_csv),
                    "ts": utc_now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    finally:
        store.close()


def build_parser() -> argparse.ArgumentParser:
    default_db_url = (
        os.environ.get("HKG_TMAX_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or ""
    )
    parser = argparse.ArgumentParser(description="Backfill Info.gov HKO hourly reading dispatches")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--start", default="1997-01-01")
    parser.add_argument("--end", default=default_end_hkt().isoformat())
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--retries", type=int, choices=(0, 1, 2, 3), default=2)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--max-days", type=int, default=31)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--progress-interval-days", type=int, default=250)
    parser.add_argument("--progress-interval-details", type=int, default=1000)
    parser.add_argument("--user-agent", default="HKG-Tmax-Research/0.1 hourly-readings backfill")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--force-details", action="store_true")
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--skip-details", action="store_true")
    parser.add_argument("--load-db", action="store_true")
    parser.add_argument("--database-url", default=default_db_url)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.execute:
        print("DRY RUN: no requests made; pass --execute with a <=31-day range.")
        return 2
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start or (end - start).days + 1 > args.max_days or args.max_days > 31:
        raise ValueError("Backfill range must be ordered and no more than 31 days")
    if args.load_db and not args.database_url:
        raise ValueError("--database-url or HKG_TMAX_DATABASE_URL is required with --load-db")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
