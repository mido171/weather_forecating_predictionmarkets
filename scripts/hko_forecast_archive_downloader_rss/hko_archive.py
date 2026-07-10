#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

HKT = ZoneInfo("Asia/Hong_Kong")
UTC = UTC

OFFICIAL_INDEX_TEMPLATE = "https://www.info.gov.hk/gia/wr/{year_month}/{day}.htm"
HKUST_TEMPLATE = "https://envf.ust.hk/dataview/hko_weather/current/display_winfo.py?ymdh={key}"
CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
DATA_GOV_LIST_VERSIONS = "https://app.data.gov.hk/v1/historical-archive/list-file-versions"
DATA_GOV_GET_FILE = "https://app.data.gov.hk/v1/historical-archive/get-file"

DATA_GOV_RESOURCES = {
    "local_weather_forecast": "https://rss.weather.gov.hk/rss/LocalWeatherForecast.xml",
    "nine_day_weather_forecast": "https://rss.weather.gov.hk/rss/SeveralDaysWeatherForecast.xml",
}

WAYBACK_URLS = [
    "http://www.weather.gov.hk/wxinfo/currwx/flw.htm",
    "https://www.weather.gov.hk/wxinfo/currwx/flw.htm",
    "http://www.hko.gov.hk/wxinfo/currwx/flw.htm",
    "https://www.hko.gov.hk/wxinfo/currwx/flw.htm",
    "http://www.weather.gov.hk/wxinfo/currwx/f5d.htm",
    "http://www.hko.gov.hk/wxinfo/currwx/f5d.htm",
    "http://www.weather.gov.hk/wxinfo/currwx/f7d.htm",
    "http://www.hko.gov.hk/wxinfo/currwx/f7d.htm",
    "http://www.weather.gov.hk/wxinfo/currwx/fnd.htm",
    "https://www.weather.gov.hk/wxinfo/currwx/fnd.htm",
    "http://www.hko.gov.hk/wxinfo/currwx/fnd.htm",
    "https://www.hko.gov.hk/wxinfo/currwx/fnd.htm",
    "http://www.weather.gov.hk/textonly/forecast/englishwx.htm",
    "http://www.hko.gov.hk/textonly/forecast/englishwx.htm",
    "http://data.weather.gov.hk/textonly/forecast/englishwx.htm",
    "http://www.weather.gov.hk/textonly/forecast/nday.htm",
    "http://www.hko.gov.hk/textonly/forecast/nday.htm",
    "http://data.weather.gov.hk/textonly/forecast/nday.htm",
    "http://data.weather.gov.hk/textonly/v2/forecast/local.htm",
    "http://data.weather.gov.hk/textonly/v2/forecast/nday.htm",
]

PRODUCT_PATTERNS = [
    ("9day", re.compile(r"\b9[\s-]*DAY WEATHER FORECAST\b", re.I)),
    ("7day", re.compile(r"\b7[\s-]*DAY WEATHER FORECAST\b", re.I)),
    ("5day", re.compile(r"\b5[\s-]*DAY WEATHER FORECAST\b", re.I)),
    ("local", re.compile(r"\bLOCAL WEATHER FORECAST\b", re.I)),
]

BOUNDARY_HEADERS = [
    "MARINE FORECAST",
    "SOUTH CHINA COASTAL WATERS",
    "WEATHER FORECAST FOR LOCAL AVIATION",
    "TROPICAL CYCLONE WARNING",
    "YESTERDAY'S WEATHER",
    "SIGNIFICANT WEATHER INFORMATION",
]

@dataclass(frozen=True)
class FetchResult:
    url: str
    status_code: int
    content: bytes | None
    raw_path: str | None
    sha256: str | None
    headers: dict[str, str]
    retrieved_at_utc: str
    error: str | None = None

@dataclass
class Bulletin:
    bulletin_id: str
    source: str
    source_url: str
    product_type: str
    title: str | None
    index_date: str | None
    snapshot_at_hkt: str | None
    issue_at_hkt: str | None
    issue_parse_method: str | None
    raw_sha256: str
    raw_path: str
    text: str
    target_date: str | None
    target_date_confidence: str | None
    forecast_min_c: float | None
    forecast_max_c: float | None
    temperature_text: str | None
    stale_snapshot_flag: int
    stale_hours: float | None
    parse_status: str
    parse_notes: str | None

@dataclass
class ForecastDay:
    bulletin_id: str
    source: str
    source_url: str
    product_type: str
    issue_at_hkt: str | None
    target_date: str
    forecast_min_c: float | None
    forecast_max_c: float | None
    rh_min_pct: float | None
    rh_max_pct: float | None
    wind_text: str | None
    weather_text: str | None
    raw_sha256: str

def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()

def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)

def daterange(start: date, end: date) -> Iterator[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)

def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def normalize_text(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()

def html_text(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    return normalize_text(soup.get_text("\n", strip=True))

def classify_product(text: str) -> str | None:
    for name, pattern in PRODUCT_PATTERNS:
        if pattern.search(text):
            return name
    return None

def parse_hours_utc(spec: str) -> list[int]:
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            for hour in range(int(left), int(right) + 1):
                result.add(hour)
        else:
            result.add(int(part))
    if not result or min(result) < 0 or max(result) > 23:
        raise ValueError("hours must be in the range 0-23")
    return sorted(result)

def parse_types(spec: str) -> set[str]:
    allowed = {"local", "5day", "7day", "9day"}
    result = {part.strip().lower() for part in spec.split(",") if part.strip()}
    unknown = result - allowed
    if unknown:
        raise ValueError(f"unknown types: {sorted(unknown)}")
    return result

def infer_product_from_url(url: str) -> str | None:
    lower = url.lower()
    if "f5d" in lower:
        return "5day"
    if "f7d" in lower:
        return "7day"
    if "fnd" in lower or "nday" in lower or "severaldays" in lower:
        return "9day"
    if "flw" in lower or "local" in lower or "englishwx" in lower:
        return "local"
    return None

def extract_local_section(full_text: str) -> str:
    upper = full_text.upper()
    issue_positions = [m.start() for m in re.finditer(r"BULLETIN (?:UPDATED|ISSUED) AT", upper)]
    local_positions = [m.start() for m in re.finditer(r"LOCAL WEATHER FORECAST", upper)]
    if not issue_positions or not local_positions:
        return full_text
    best_start = None
    best_distance = None
    for start in local_positions:
        future_issues = [pos for pos in issue_positions if pos >= start]
        if not future_issues:
            continue
        distance = min(future_issues) - start
        if best_distance is None or distance < best_distance:
            best_start = start
            best_distance = distance
    if best_start is None:
        return full_text
    end = len(full_text)
    for header in BOUNDARY_HEADERS:
        pos = upper.find(header, best_start + 50)
        if pos != -1:
            end = min(end, pos)
    return normalize_text(full_text[best_start:end])

def _parse_date_fragment(fragment: str, fallback_year: int | None = None) -> date | None:
    fragment = normalize_text(fragment)
    try:
        dt = date_parser.parse(fragment, dayfirst=True, fuzzy=True, default=datetime(fallback_year or 2000, 1, 1))
        return dt.date()
    except (ValueError, OverflowError):
        return None

def extract_issue_datetime(text: str, fallback_date: date | None) -> tuple[datetime | None, str | None]:
    patterns = [
        re.compile(
            r"(?:DISPATCHED BY HONG KONG OBSERVATORY AT|BULLETIN (?:UPDATED|ISSUED) AT)"
            r"\s*(\d{1,2}:\d{2})\s*HKT(?:\s+ON)?\s*"
            r"(\d{1,2}(?:[./-]|\s+)(?:[A-Z]{3,9}|\d{1,2})(?:[./-]|\s+)\d{2,4})",
            re.I,
        ),
        re.compile(
            r"ISSUED BY THE HONG KONG OBSERVATORY AT\s*(\d{1,2}:\d{2})\s*(?:A\.?M\.?|P\.?M\.?)?\s*"
            r"(?:ON\s*)?(\d{1,2}\s+[A-Z]{3,9}\s+\d{4})",
            re.I,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        clock = match.group(1)
        day = _parse_date_fragment(match.group(2), fallback_date.year if fallback_date else None)
        if day is None:
            continue
        hour, minute = map(int, clock.split(":"))
        return datetime.combine(day, dt_time(hour, minute), HKT), "embedded_full_timestamp"

    time_match = re.search(r"ISSUED AT HKT\s*(\d{1,2}):(\d{2})", text, re.I)
    if time_match:
        ending_match = re.search(
            r"ENDS?/[A-Z]+,\s*([A-Z]+\s+\d{1,2},\s+\d{4})",
            text,
            re.I,
        )
        issue_day = _parse_date_fragment(ending_match.group(1)) if ending_match else fallback_date
        if issue_day:
            return datetime.combine(
                issue_day,
                dt_time(int(time_match.group(1)), int(time_match.group(2))),
                HKT,
            ), "issued_at_plus_page_date"

    if fallback_date:
        generic_match = re.search(
            r"(?:BULLETIN (?:UPDATED|ISSUED) AT|DISPATCHED BY HONG KONG OBSERVATORY AT)"
            r"\s*(\d{1,2}):(\d{2})\s*HKT",
            text,
            re.I,
        )
        if generic_match:
            return datetime.combine(
                fallback_date,
                dt_time(int(generic_match.group(1)), int(generic_match.group(2))),
                HKT,
            ), "time_plus_index_date"

    return None, None

def infer_target_date(section: str, issue_at: datetime | None) -> tuple[date | None, str | None]:
    if issue_at is None:
        return None, None
    explicit = re.search(r"WEATHER FORECAST FOR HONG KONG\s*\(([^)]+)\)", section, re.I)
    if explicit:
        target = _parse_date_fragment(explicit.group(1), issue_at.year)
        if target:
            return target, "explicit_date"
    if re.search(r"WEATHER FORECAST FOR (?:TONIGHT AND )?TOMORROW", section, re.I):
        return issue_at.date() + timedelta(days=1), "tomorrow_phrase"
    if re.search(r"WEATHER FORECAST FOR TODAY", section, re.I):
        return issue_at.date(), "today_phrase"
    return None, None

def extract_temperature_range(section: str) -> tuple[float | None, float | None, str | None]:
    patterns = [
        re.compile(r"TEMPERATURES?\s+(?:(?:WILL\s+)?(?:RANGE|BE)|RANGING)\s+BETWEEN\s+(-?\d+(?:\.\d+)?)\s+AND\s+(-?\d+(?:\.\d+)?)\s+DEGREES", re.I),
        re.compile(r"TEMPERATURES?\s+(?:WILL\s+)?RANGE\s+FROM\s+(-?\d+(?:\.\d+)?)\s+TO\s+(-?\d+(?:\.\d+)?)\s+DEGREES", re.I),
        re.compile(r"TEMPS?\s*:\s*(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s+DEGREES", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(section)
        if match:
            low, high = float(match.group(1)), float(match.group(2))
            return min(low, high), max(low, high), normalize_text(match.group(0))

    min_match = re.search(
        r"MIN(?:IMUM)?\s+(?:AIR\s+)?TEMPERATURE(?:\s+WILL\s+BE)?(?:\s+OF)?(?:\s+ABOUT|\s+AROUND)?\s+(-?\d+(?:\.\d+)?)\s+DEGREES",
        section,
        re.I,
    )
    max_match = re.search(
        r"MAX(?:IMUM)?\s+(?:AIR\s+)?TEMPERATURE(?:\s+WILL\s+BE)?(?:\s+OF)?(?:\s+ABOUT|\s+AROUND)?\s+(-?\d+(?:\.\d+)?)\s+DEGREES",
        section,
        re.I,
    )
    minimum = float(min_match.group(1)) if min_match else None
    maximum = float(max_match.group(1)) if max_match else None
    evidence = None
    if min_match or max_match:
        evidence = " | ".join(m.group(0) for m in (min_match, max_match) if m)
    return minimum, maximum, evidence

def _choose_target_year(issue_date: date, month: int, day: int) -> int:
    candidates = []
    for year in (issue_date.year - 1, issue_date.year, issue_date.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        distance = abs((candidate - issue_date).days)
        penalty = 1000 if (candidate - issue_date).days < -2 or (candidate - issue_date).days > 20 else 0
        candidates.append((distance + penalty, year))
    return min(candidates)[1]

def parse_multiday_rows(
    section: str,
    issue_at: datetime | None,
    bulletin_id: str,
    source: str,
    source_url: str,
    product_type: str,
    raw_sha256: str,
) -> list[ForecastDay]:
    if issue_at is None:
        return []
    starts = list(re.finditer(r"DATE/MONTH\s+(\d{1,2})/(\d{1,2})(?:\s*\([^)]+\))?", section, re.I))
    result: list[ForecastDay] = []
    for index, match in enumerate(starts):
        block_end = starts[index + 1].start() if index + 1 < len(starts) else len(section)
        block = section[match.start():block_end]
        day_num, month_num = int(match.group(1)), int(match.group(2))
        target_year = _choose_target_year(issue_at.date(), month_num, day_num)
        try:
            target = date(target_year, month_num, day_num)
        except ValueError:
            continue

        temp_match = re.search(r"TEMP(?:ERATURE)?\s+RANGE\s*:\s*(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*C", block, re.I)
        rh_match = re.search(r"R\.?H\.?\s+RANGE\s*:\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", block, re.I)
        wind_match = re.search(r"WIND\s*:\s*(.*?)(?=\nWEATHER\s*:|\nTEMP(?:ERATURE)?\s+RANGE\s*:|$)", block, re.I | re.S)
        weather_match = re.search(r"WEATHER\s*:\s*(.*?)(?=\nTEMP(?:ERATURE)?\s+RANGE\s*:|\nR\.?H\.?\s+RANGE\s*:|$)", block, re.I | re.S)

        tmin = tmax = None
        if temp_match:
            first, second = float(temp_match.group(1)), float(temp_match.group(2))
            tmin, tmax = min(first, second), max(first, second)

        rh_min = rh_max = None
        if rh_match:
            first, second = float(rh_match.group(1)), float(rh_match.group(2))
            rh_min, rh_max = min(first, second), max(first, second)

        result.append(
            ForecastDay(
                bulletin_id=bulletin_id,
                source=source,
                source_url=source_url,
                product_type=product_type,
                issue_at_hkt=issue_at.isoformat(),
                target_date=target.isoformat(),
                forecast_min_c=tmin,
                forecast_max_c=tmax,
                rh_min_pct=rh_min,
                rh_max_pct=rh_max,
                wind_text=normalize_text(wind_match.group(1)) if wind_match else None,
                weather_text=normalize_text(weather_match.group(1)) if weather_match else None,
                raw_sha256=raw_sha256,
            )
        )
    return result

class ArchiveStore:
    def __init__(self, root: Path):
        self.root = root
        self.raw_root = root / "raw"
        self.metadata_root = root / "metadata"
        self.bronze_root = root / "bronze"
        self.reports_root = root / "reports"
        for path in (self.raw_root, self.metadata_root, self.bronze_root, self.reports_root):
            path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.metadata_root / "archive.sqlite3"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS retrievals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                url TEXT NOT NULL,
                attempted_at_utc TEXT NOT NULL,
                status_code INTEGER,
                error TEXT,
                content_sha256 TEXT,
                raw_path TEXT,
                headers_json TEXT,
                UNIQUE(source, url, attempted_at_utc)
            );
            CREATE INDEX IF NOT EXISTS idx_retrievals_source_url ON retrievals(source, url);
            CREATE TABLE IF NOT EXISTS candidates (
                source TEXT NOT NULL,
                index_date TEXT,
                title TEXT NOT NULL,
                product_type TEXT NOT NULL,
                url TEXT NOT NULL,
                discovered_at_utc TEXT NOT NULL,
                PRIMARY KEY(source, url)
            );
            CREATE TABLE IF NOT EXISTS bulletins (
                bulletin_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_url TEXT NOT NULL,
                product_type TEXT NOT NULL,
                title TEXT,
                index_date TEXT,
                snapshot_at_hkt TEXT,
                issue_at_hkt TEXT,
                issue_parse_method TEXT,
                raw_sha256 TEXT NOT NULL,
                raw_path TEXT NOT NULL,
                text TEXT NOT NULL,
                target_date TEXT,
                target_date_confidence TEXT,
                forecast_min_c REAL,
                forecast_max_c REAL,
                temperature_text TEXT,
                stale_snapshot_flag INTEGER NOT NULL,
                stale_hours REAL,
                parse_status TEXT NOT NULL,
                parse_notes TEXT
            );
            CREATE TABLE IF NOT EXISTS forecast_days (
                bulletin_id TEXT NOT NULL,
                source TEXT NOT NULL,
                source_url TEXT NOT NULL,
                product_type TEXT NOT NULL,
                issue_at_hkt TEXT,
                target_date TEXT NOT NULL,
                forecast_min_c REAL,
                forecast_max_c REAL,
                rh_min_pct REAL,
                rh_max_pct REAL,
                wind_text TEXT,
                weather_text TEXT,
                raw_sha256 TEXT NOT NULL,
                PRIMARY KEY(bulletin_id, target_date)
            );
            CREATE TABLE IF NOT EXISTS wayback_captures (
                original_url TEXT NOT NULL,
                capture_timestamp TEXT NOT NULL,
                digest TEXT,
                mime_type TEXT,
                status_code TEXT,
                length TEXT,
                replay_url TEXT NOT NULL,
                PRIMARY KEY(original_url, capture_timestamp)
            );
            CREATE TABLE IF NOT EXISTS data_gov_versions (
                resource_name TEXT NOT NULL,
                resource_url TEXT NOT NULL,
                version_time TEXT NOT NULL,
                PRIMARY KEY(resource_url, version_time)
            );
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def latest_success(self, source: str, url: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT * FROM retrievals
            WHERE source=? AND url=? AND status_code BETWEEN 200 AND 299 AND raw_path IS NOT NULL
            ORDER BY id DESC LIMIT 1
            """,
            (source, url),
        ).fetchone()

    def record_retrieval(self, result: FetchResult, source: str) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO retrievals
            (source, url, attempted_at_utc, status_code, error, content_sha256, raw_path, headers_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                result.url,
                result.retrieved_at_utc,
                result.status_code,
                result.error,
                result.sha256,
                result.raw_path,
                json.dumps(result.headers, ensure_ascii=False, sort_keys=True),
            ),
        )
        self.conn.commit()

    def save_raw(
        self,
        source: str,
        url: str,
        content: bytes,
        headers: dict[str, str],
        retrieved_at_utc: str,
        date_hint: date | datetime | None,
    ) -> tuple[str, str]:
        digest = sha256_bytes(content)
        if isinstance(date_hint, datetime):
            hint_date = date_hint.date()
        elif isinstance(date_hint, date):
            hint_date = date_hint
        else:
            hint_date = datetime.now(UTC).date()
        suffix = ".xml" if "xml" in headers.get("content-type", "").lower() or url.lower().endswith(".xml") else ".html"
        directory = self.raw_root / source / f"{hint_date.year:04d}" / f"{hint_date.month:02d}" / f"{hint_date.day:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        raw_path = directory / f"{digest}{suffix}"
        if not raw_path.exists():
            raw_path.write_bytes(content)
        sidecar = raw_path.with_suffix(raw_path.suffix + ".metadata.json")
        if not sidecar.exists():
            sidecar.write_text(
                json.dumps(
                    {
                        "source": source,
                        "url": url,
                        "retrieved_at_utc": retrieved_at_utc,
                        "sha256": digest,
                        "content_length": len(content),
                        "headers": headers,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return str(raw_path), digest

    def add_candidate(self, source: str, index_date: date | None, title: str, product_type: str, url: str) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO candidates
            (source, index_date, title, product_type, url, discovered_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                index_date.isoformat() if index_date else None,
                title,
                product_type,
                url,
                utc_now_iso(),
            ),
        )
        self.conn.commit()

    def upsert_bulletin(self, bulletin: Bulletin, days: Sequence[ForecastDay]) -> None:
        fields = list(asdict(bulletin).keys())
        placeholders = ", ".join("?" for _ in fields)
        updates = ", ".join(f"{field}=excluded.{field}" for field in fields if field != "bulletin_id")
        self.conn.execute(
            f"""
            INSERT INTO bulletins ({", ".join(fields)})
            VALUES ({placeholders})
            ON CONFLICT(bulletin_id) DO UPDATE SET {updates}
            """,
            tuple(asdict(bulletin)[field] for field in fields),
        )
        for day_row in days:
            values = asdict(day_row)
            day_fields = list(values.keys())
            day_updates = ", ".join(f"{field}=excluded.{field}" for field in day_fields if field not in {"bulletin_id", "target_date"})
            self.conn.execute(
                f"""
                INSERT INTO forecast_days ({", ".join(day_fields)})
                VALUES ({", ".join("?" for _ in day_fields)})
                ON CONFLICT(bulletin_id, target_date) DO UPDATE SET {day_updates}
                """,
                tuple(values[field] for field in day_fields),
            )
        self.conn.commit()

class PoliteFetcher:
    def __init__(
        self,
        store: ArchiveStore,
        delay_seconds: float,
        timeout_seconds: float,
        max_retries: int,
        force: bool,
    ):
        self.store = store
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.force = force
        self.last_request_by_host: dict[str, float] = {}
        self.client = httpx.Client(
            follow_redirects=True,
            timeout=timeout_seconds,
            headers={
                "User-Agent": "LundUniversity-HKOForecastResearch/1.0 (+academic research; polite archival client)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )

    def close(self) -> None:
        self.client.close()

    def _wait(self, url: str) -> None:
        host = urlparse(url).netloc
        last = self.last_request_by_host.get(host)
        if last is not None:
            remaining = self.delay_seconds - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_by_host[host] = time.monotonic()

    def fetch(self, source: str, url: str, date_hint: date | datetime | None = None) -> FetchResult:
        if not self.force:
            existing = self.store.latest_success(source, url)
            if existing and existing["raw_path"] and Path(existing["raw_path"]).exists():
                content = Path(existing["raw_path"]).read_bytes()
                return FetchResult(
                    url=url,
                    status_code=int(existing["status_code"]),
                    content=content,
                    raw_path=existing["raw_path"],
                    sha256=existing["content_sha256"],
                    headers=json.loads(existing["headers_json"] or "{}"),
                    retrieved_at_utc=existing["attempted_at_utc"],
                )

        last_error = None
        for attempt in range(self.max_retries + 1):
            self._wait(url)
            attempted_at = utc_now_iso()
            try:
                response = self.client.get(url)
                headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
                if 200 <= response.status_code < 300:
                    raw_path, digest = self.store.save_raw(source, url, response.content, headers, attempted_at, date_hint)
                    result = FetchResult(
                        url=url,
                        status_code=response.status_code,
                        content=response.content,
                        raw_path=raw_path,
                        sha256=digest,
                        headers=headers,
                        retrieved_at_utc=attempted_at,
                    )
                    self.store.record_retrieval(result, source)
                    return result

                error = f"HTTP {response.status_code}"
                result = FetchResult(
                    url=url,
                    status_code=response.status_code,
                    content=None,
                    raw_path=None,
                    sha256=None,
                    headers=headers,
                    retrieved_at_utc=attempted_at,
                    error=error,
                )
                self.store.record_retrieval(result, source)
                if response.status_code in {404, 410}:
                    return result
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else min(120, 2 ** (attempt + 2)))
                elif 500 <= response.status_code < 600:
                    time.sleep(min(120, 2 ** (attempt + 1)))
                else:
                    return result
                last_error = error
            except httpx.HTTPError as exc:
                last_error = repr(exc)
                result = FetchResult(
                    url=url,
                    status_code=0,
                    content=None,
                    raw_path=None,
                    sha256=None,
                    headers={},
                    retrieved_at_utc=attempted_at,
                    error=last_error,
                )
                self.store.record_retrieval(result, source)
                time.sleep(min(120, 2 ** (attempt + 1)))

        return FetchResult(
            url=url,
            status_code=0,
            content=None,
            raw_path=None,
            sha256=None,
            headers={},
            retrieved_at_utc=utc_now_iso(),
            error=last_error or "unknown failure",
        )

def parse_bulletin(
    source: str,
    source_url: str,
    content: bytes,
    raw_path: str,
    raw_sha256: str,
    title_hint: str | None,
    index_date: date | None,
    snapshot_at_hkt: datetime | None,
) -> tuple[Bulletin, list[ForecastDay]]:
    full_text = html_text(content)
    product_type = classify_product(title_hint or "") or classify_product(full_text[:5000]) or infer_product_from_url(source_url) or "unknown"
    section = extract_local_section(full_text) if product_type == "local" and source == "hkust" else full_text
    issue_at, issue_method = extract_issue_datetime(section, index_date)
    target_date, target_confidence = infer_target_date(section, issue_at)
    tmin, tmax, temp_text = extract_temperature_range(section) if product_type == "local" else (None, None, None)

    stale_hours = None
    stale_flag = 0
    if snapshot_at_hkt and issue_at:
        stale_hours = (snapshot_at_hkt - issue_at).total_seconds() / 3600.0
        stale_flag = int(stale_hours > 12 or stale_hours < -1)

    bulletin_id = hashlib.sha256(
        f"{source}|{product_type}|{issue_at.isoformat() if issue_at else ''}|{raw_sha256}|{normalize_text(section)}".encode()
    ).hexdigest()

    days: list[ForecastDay] = []
    if product_type in {"5day", "7day", "9day"}:
        days = parse_multiday_rows(section, issue_at, bulletin_id, source, source_url, product_type, raw_sha256)
    elif product_type == "local" and target_date is not None:
        days = [
            ForecastDay(
                bulletin_id=bulletin_id,
                source=source,
                source_url=source_url,
                product_type=product_type,
                issue_at_hkt=issue_at.isoformat() if issue_at else None,
                target_date=target_date.isoformat(),
                forecast_min_c=tmin,
                forecast_max_c=tmax,
                rh_min_pct=None,
                rh_max_pct=None,
                wind_text=None,
                weather_text=section,
                raw_sha256=raw_sha256,
            )
        ]

    notes = []
    if issue_at is None:
        notes.append("issue timestamp not parsed")
    if product_type == "unknown":
        notes.append("product type not identified")
    if product_type in {"5day", "7day", "9day"} and not days:
        notes.append("no daily forecast rows parsed")
    parse_status = "ok" if not notes else "partial"

    bulletin = Bulletin(
        bulletin_id=bulletin_id,
        source=source,
        source_url=source_url,
        product_type=product_type,
        title=title_hint,
        index_date=index_date.isoformat() if index_date else None,
        snapshot_at_hkt=snapshot_at_hkt.isoformat() if snapshot_at_hkt else None,
        issue_at_hkt=issue_at.isoformat() if issue_at else None,
        issue_parse_method=issue_method,
        raw_sha256=raw_sha256,
        raw_path=raw_path,
        text=section,
        target_date=target_date.isoformat() if target_date else None,
        target_date_confidence=target_confidence,
        forecast_min_c=tmin,
        forecast_max_c=tmax,
        temperature_text=temp_text,
        stale_snapshot_flag=stale_flag,
        stale_hours=stale_hours,
        parse_status=parse_status,
        parse_notes="; ".join(notes) if notes else None,
    )
    return bulletin, days

def command_official_index(args: argparse.Namespace) -> None:
    store = ArchiveStore(Path(args.data_root))
    fetcher = PoliteFetcher(store, args.delay_seconds, args.timeout_seconds, args.max_retries, args.force)
    start, end = parse_iso_date(args.start), parse_iso_date(args.end)
    counts = {"days": 0, "success": 0, "candidates": 0}
    try:
        for current in daterange(start, end):
            counts["days"] += 1
            url = OFFICIAL_INDEX_TEMPLATE.format(year_month=current.strftime("%Y%m"), day=current.strftime("%d"))
            result = fetcher.fetch("info_gov_index", url, current)
            if not result.content:
                continue
            counts["success"] += 1
            soup = BeautifulSoup(result.content, "html.parser")
            for anchor in soup.find_all("a", href=True):
                title = normalize_text(anchor.get_text(" ", strip=True))
                product = classify_product(title)
                if not product:
                    continue
                detail_url = urljoin(url, anchor["href"])
                store.add_candidate("info_gov", current, title, product, detail_url)
                counts["candidates"] += 1
            if counts["days"] % 100 == 0:
                print(json.dumps(counts, sort_keys=True))
    finally:
        fetcher.close()
        store.close()
    print("official-index complete", json.dumps(counts, sort_keys=True))

def candidate_date_in_range(value: str | None, start: date | None, end: date | None) -> bool:
    if not start and not end:
        return True
    if not value:
        return False
    current = date.fromisoformat(value)
    if start and current < start:
        return False
    return not (end and current > end)

def command_official_details(args: argparse.Namespace) -> None:
    store = ArchiveStore(Path(args.data_root))
    fetcher = PoliteFetcher(store, args.delay_seconds, args.timeout_seconds, args.max_retries, args.force)
    selected_types = parse_types(args.types)
    start = parse_iso_date(args.start) if args.start else None
    end = parse_iso_date(args.end) if args.end else None
    rows = store.conn.execute(
        "SELECT * FROM candidates WHERE source='info_gov' ORDER BY index_date, url"
    ).fetchall()
    processed = parsed = skipped_existing = 0
    try:
        for row in rows:
            if row["product_type"] not in selected_types:
                continue
            if not candidate_date_in_range(row["index_date"], start, end):
                continue
            current = date.fromisoformat(row["index_date"]) if row["index_date"] else None
            if args.missing_success_only:
                existing = store.latest_success("info_gov_bulletin", row["url"])
                if existing and existing["raw_path"] and Path(existing["raw_path"]).exists():
                    skipped_existing += 1
                    continue
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1
            result = fetcher.fetch("info_gov_bulletin", row["url"], current)
            if not result.content or not result.raw_path or not result.sha256:
                continue
            bulletin, days = parse_bulletin(
                source="info_gov",
                source_url=row["url"],
                content=result.content,
                raw_path=result.raw_path,
                raw_sha256=result.sha256,
                title_hint=row["title"],
                index_date=current,
                snapshot_at_hkt=None,
            )
            store.upsert_bulletin(bulletin, days)
            parsed += 1
            if processed % 100 == 0:
                print(
                    "official-details "
                    f"processed={processed} parsed={parsed} skipped_existing={skipped_existing}"
                )
    finally:
        fetcher.close()
        store.close()
    print(
        "official-details complete "
        f"processed={processed} parsed={parsed} skipped_existing={skipped_existing}"
    )

def command_hkust(args: argparse.Namespace) -> None:
    if not args.acknowledge_research_only:
        raise SystemExit("HKUST requires --acknowledge-research-only")
    hours = parse_hours_utc(args.hours_utc)
    if len(hours) > 8 and not args.acknowledge_large_crawl:
        raise SystemExit("More than 8 snapshots/day requires --acknowledge-large-crawl")
    store = ArchiveStore(Path(args.data_root))
    fetcher = PoliteFetcher(store, args.delay_seconds, args.timeout_seconds, args.max_retries, args.force)
    start, end = parse_iso_date(args.start), parse_iso_date(args.end)
    processed = parsed = 0
    try:
        for current in daterange(start, end):
            for hour in hours:
                snapshot_utc = datetime.combine(current, dt_time(hour, 0), UTC)
                snapshot_hkt = snapshot_utc.astimezone(HKT)
                key = snapshot_utc.strftime("%Y%m%d%H")
                url = HKUST_TEMPLATE.format(key=key)
                processed += 1
                result = fetcher.fetch("hkust", url, snapshot_hkt)
                if not result.content or not result.raw_path or not result.sha256:
                    continue
                text = html_text(result.content)
                if "LOCAL WEATHER FORECAST" not in text.upper() or "HONG KONG OBSERVATORY" not in text.upper():
                    continue
                bulletin, days = parse_bulletin(
                    source="hkust",
                    source_url=url,
                    content=result.content,
                    raw_path=result.raw_path,
                    raw_sha256=result.sha256,
                    title_hint="LOCAL WEATHER FORECAST",
                    index_date=snapshot_hkt.date(),
                    snapshot_at_hkt=snapshot_hkt,
                )
                store.upsert_bulletin(bulletin, days)
                parsed += 1
                if processed % 100 == 0:
                    print(f"hkust processed={processed} parsed={parsed}")
    finally:
        fetcher.close()
        store.close()
    print(f"hkust complete processed={processed} parsed={parsed}")

def _cdx_rows(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, list) or not payload:
        return []
    header = payload[0]
    return [dict(zip(header, row, strict=False)) for row in payload[1:]]

def command_wayback(args: argparse.Namespace) -> None:
    store = ArchiveStore(Path(args.data_root))
    fetcher = PoliteFetcher(store, args.delay_seconds, args.timeout_seconds, args.max_retries, args.force)
    try:
        for original in WAYBACK_URLS:
            params = {
                "url": original,
                "output": "json",
                "fl": "timestamp,original,mimetype,statuscode,digest,length",
                "filter": "statuscode:200",
                "collapse": "digest",
                "from": str(args.from_year),
                "to": str(args.to_year),
            }
            query = str(httpx.URL(CDX_ENDPOINT, params=params))
            result = fetcher.fetch("wayback_cdx", query, date(args.from_year, 1, 1))
            if not result.content:
                continue
            try:
                payload = json.loads(result.content)
            except json.JSONDecodeError:
                continue
            for row in _cdx_rows(payload):
                replay = f"https://web.archive.org/web/{row['timestamp']}id_/{row['original']}"
                store.conn.execute(
                    """
                    INSERT OR IGNORE INTO wayback_captures
                    (original_url, capture_timestamp, digest, mime_type, status_code, length, replay_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["original"],
                        row["timestamp"],
                        row.get("digest"),
                        row.get("mimetype"),
                        row.get("statuscode"),
                        row.get("length"),
                        replay,
                    ),
                )
        store.conn.commit()

        captures = store.conn.execute(
            "SELECT * FROM wayback_captures ORDER BY capture_timestamp, original_url"
        ).fetchall()
        processed = parsed = 0
        for row in captures:
            processed += 1
            capture_dt = datetime.strptime(row["capture_timestamp"], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
            result = fetcher.fetch("wayback", row["replay_url"], capture_dt)
            if not result.content or not result.raw_path or not result.sha256:
                continue
            bulletin, days = parse_bulletin(
                source="wayback",
                source_url=row["replay_url"],
                content=result.content,
                raw_path=result.raw_path,
                raw_sha256=result.sha256,
                title_hint=infer_product_from_url(row["original_url"]),
                index_date=capture_dt.astimezone(HKT).date(),
                snapshot_at_hkt=capture_dt.astimezone(HKT),
            )
            store.upsert_bulletin(bulletin, days)
            parsed += 1
            if processed % 100 == 0:
                print(f"wayback processed={processed} parsed={parsed}")
    finally:
        fetcher.close()
        store.close()

def _extract_version_times(payload: Any) -> list[str]:
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = []
        for key in ("timestamps", "versions", "files", "result", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates = value
                break
    else:
        candidates = []
    result: list[str] = []
    for item in candidates:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            for key in ("time", "timestamp", "datetime", "version"):
                if item.get(key):
                    result.append(str(item[key]))
                    break
    return result

def command_data_gov(args: argparse.Namespace) -> None:
    store = ArchiveStore(Path(args.data_root))
    fetcher = PoliteFetcher(store, args.delay_seconds, args.timeout_seconds, args.max_retries, args.force)
    start, end = parse_iso_date(args.start), parse_iso_date(args.end)
    try:
        for name, resource_url in DATA_GOV_RESOURCES.items():
            params = {
                "url": resource_url,
                "start": start.strftime("%Y%m%d"),
                "end": end.strftime("%Y%m%d"),
            }
            list_url = str(httpx.URL(DATA_GOV_LIST_VERSIONS, params=params))
            result = fetcher.fetch("data_gov_versions", list_url, start)
            if not result.content:
                continue
            try:
                payload = json.loads(result.content)
            except json.JSONDecodeError:
                continue
            version_times = _extract_version_times(payload)
            for version_time in version_times:
                store.conn.execute(
                    """
                    INSERT OR IGNORE INTO data_gov_versions
                    (resource_name, resource_url, version_time) VALUES (?, ?, ?)
                    """,
                    (name, resource_url, version_time),
                )
        store.conn.commit()

        versions = store.conn.execute(
            "SELECT * FROM data_gov_versions ORDER BY version_time, resource_name"
        ).fetchall()
        for index, row in enumerate(versions, 1):
            params = {"url": row["resource_url"], "time": row["version_time"]}
            url = str(httpx.URL(DATA_GOV_GET_FILE, params=params))
            try:
                hint = date_parser.parse(row["version_time"]).date()
            except (ValueError, TypeError):
                hint = start
            result = fetcher.fetch("data_gov", url, hint)
            if not result.content or not result.raw_path or not result.sha256:
                continue
            bulletin, days = parse_bulletin(
                source="data_gov",
                source_url=url,
                content=result.content,
                raw_path=result.raw_path,
                raw_sha256=result.sha256,
                title_hint=row["resource_name"].replace("_", " "),
                index_date=hint,
                snapshot_at_hkt=None,
            )
            store.upsert_bulletin(bulletin, days)
            if index % 100 == 0:
                print(f"data-gov processed={index}/{len(versions)}")
    finally:
        fetcher.close()
        store.close()

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

def command_export(args: argparse.Namespace) -> None:
    store = ArchiveStore(Path(args.data_root))
    try:
        bulletin_rows = [dict(row) for row in store.conn.execute("SELECT * FROM bulletins ORDER BY issue_at_hkt, source_url")]
        day_rows = [dict(row) for row in store.conn.execute("SELECT * FROM forecast_days ORDER BY target_date, issue_at_hkt")]
        retrieval_rows = [dict(row) for row in store.conn.execute("SELECT * FROM retrievals ORDER BY id")]

        _write_jsonl(store.bronze_root / "forecast_bulletins.jsonl", bulletin_rows)
        _write_jsonl(store.bronze_root / "forecast_days.jsonl", day_rows)
        _write_csv(store.bronze_root / "forecast_bulletins.csv", bulletin_rows)
        _write_csv(store.bronze_root / "forecast_days.csv", day_rows)

        coverage: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in bulletin_rows:
            year = (row.get("issue_at_hkt") or row.get("index_date") or "unknown")[:4]
            key = (row["source"], row["product_type"], year)
            item = coverage.setdefault(
                key,
                {
                    "source": row["source"],
                    "product_type": row["product_type"],
                    "year": year,
                    "bulletin_count": 0,
                    "parsed_issue_count": 0,
                    "target_date_count": 0,
                    "stale_count": 0,
                },
            )
            item["bulletin_count"] += 1
            item["parsed_issue_count"] += int(bool(row.get("issue_at_hkt")))
            item["target_date_count"] += int(bool(row.get("target_date")))
            item["stale_count"] += int(bool(row.get("stale_snapshot_flag")))
        _write_csv(store.reports_root / "coverage_by_source_product_year.csv", list(coverage.values()))

        issue_coverage: dict[tuple[str, str], dict[str, Any]] = {}
        for row in bulletin_rows:
            issue_date = (row.get("issue_at_hkt") or "unknown")[:10]
            key = (row["source"], issue_date)
            item = issue_coverage.setdefault(
                key,
                {"source": row["source"], "issue_date": issue_date, "bulletin_count": 0},
            )
            item["bulletin_count"] += 1
        _write_csv(store.reports_root / "coverage_by_issue_date.csv", list(issue_coverage.values()))

        failed = [row for row in retrieval_rows if not (200 <= int(row.get("status_code") or 0) < 300)]
        _write_csv(store.reports_root / "failed_requests.csv", failed)
        _write_csv(
            store.reports_root / "parse_failures.csv",
            [row for row in bulletin_rows if row.get("parse_status") != "ok"],
        )
        _write_csv(
            store.reports_root / "stale_hkust_snapshots.csv",
            [row for row in bulletin_rows if row.get("source") == "hkust" and row.get("stale_snapshot_flag")],
        )

        candidates = [
            dict(row)
            for row in store.conn.execute(
                """
                SELECT source, product_type, substr(index_date, 1, 4) AS year, COUNT(*) AS candidate_count
                FROM candidates
                GROUP BY source, product_type, substr(index_date, 1, 4)
                ORDER BY source, year, product_type
                """
            )
        ]
        _write_csv(store.reports_root / "candidate_link_counts.csv", candidates)

        summary = {
            "bulletins": len(bulletin_rows),
            "forecast_days": len(day_rows),
            "retrievals": len(retrieval_rows),
            "failed_retrievals": len(failed),
            "parse_failures": sum(row.get("parse_status") != "ok" for row in bulletin_rows),
            "stale_hkust": sum(bool(row.get("stale_snapshot_flag")) for row in bulletin_rows if row.get("source") == "hkust"),
            "generated_at_utc": utc_now_iso(),
        }
        (store.reports_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        store.close()

def add_common_fetch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--delay-seconds", type=float, default=1.25)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--force", action="store_true")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and normalize historical HKO forecast bulletins")
    subparsers = parser.add_subparsers(dest="command", required=True)

    official_index = subparsers.add_parser("official-index")
    official_index.add_argument("--start", required=True)
    official_index.add_argument("--end", required=True)
    add_common_fetch_args(official_index)
    official_index.set_defaults(func=command_official_index)

    official_details = subparsers.add_parser("official-details")
    official_details.add_argument("--types", default="local,5day,7day,9day")
    official_details.add_argument("--start")
    official_details.add_argument("--end")
    official_details.add_argument("--limit", type=int)
    official_details.add_argument("--missing-success-only", action="store_true")
    add_common_fetch_args(official_details)
    official_details.set_defaults(func=command_official_details)

    hkust = subparsers.add_parser("hkust")
    hkust.add_argument("--start", required=True)
    hkust.add_argument("--end", required=True)
    hkust.add_argument("--hours-utc", default="6,7")
    hkust.add_argument("--acknowledge-research-only", action="store_true")
    hkust.add_argument("--acknowledge-large-crawl", action="store_true")
    add_common_fetch_args(hkust)
    hkust.set_defaults(func=command_hkust)

    wayback = subparsers.add_parser("wayback")
    wayback.add_argument("--from-year", type=int, required=True)
    wayback.add_argument("--to-year", type=int, required=True)
    add_common_fetch_args(wayback)
    wayback.set_defaults(func=command_wayback)

    data_gov = subparsers.add_parser("data-gov")
    data_gov.add_argument("--start", required=True)
    data_gov.add_argument("--end", required=True)
    add_common_fetch_args(data_gov)
    data_gov.set_defaults(func=command_data_gov)

    export = subparsers.add_parser("export")
    export.add_argument("--data-root", required=True)
    export.set_defaults(func=command_export)

    return parser

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
