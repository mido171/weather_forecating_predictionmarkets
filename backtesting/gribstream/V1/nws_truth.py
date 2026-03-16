from __future__ import annotations

import concurrent.futures
import logging
import re
from datetime import date, datetime
from typing import Any

import requests

from . import db
from .config import (
    DEFAULT_TRUTH_THREADS,
    IEM_CLI_BASE_URL,
    STATION,
    TRUTH_NATIVE_UNIT,
    TRUTH_SOURCE_NAME,
    isoformat_utc,
    local_day_window_utc,
    safe_float,
    utc_now,
)

LOGGER = logging.getLogger(__name__)
ISSUE_TIMESTAMP_RE = re.compile(r"(\d{12})")


def _parse_report_issued_at(entry: dict[str, Any]) -> str | None:
    for field_name in ("product", "link"):
        value = str(entry.get(field_name) or "").strip()
        match = ISSUE_TIMESTAMP_RE.search(value)
        if not match:
            continue
        try:
            parsed = datetime.strptime(match.group(1), "%Y%m%d%H%M")
        except ValueError:
            continue
        return isoformat_utc(parsed)
    return None


def _year_url(station_id: str, year: int) -> str:
    return f"{IEM_CLI_BASE_URL}?station={station_id}&year={year}&fmt=json"


def _parse_year_payload(
    station_id: str,
    year: int,
    payload: dict[str, Any],
    start_date: date,
    end_date: date,
) -> list[dict[str, object]]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"IEM CLI payload missing results array for {station_id} {year}")
    rows: list[dict[str, object]] = []
    ingested_at = isoformat_utc(utc_now())
    for entry in results:
        if not isinstance(entry, dict):
            continue
        entry_station = str(entry.get("station") or "").strip().upper()
        if entry_station != station_id:
            raise ValueError(
                f"IEM CLI station mismatch for year={year}: expected={station_id} got={entry_station}"
            )
        date_text = str(entry.get("valid") or "").strip()
        if not date_text:
            continue
        settlement_date_local = date.fromisoformat(date_text)
        if settlement_date_local < start_date or settlement_date_local > end_date:
            continue
        actual_tmax_f = safe_float(entry.get("high"))
        if actual_tmax_f is None:
            continue
        local_start_utc, local_end_utc = local_day_window_utc(
            settlement_date_local,
            STATION.timezone_name,
        )
        truth_source = str(entry.get("link") or _year_url(station_id, year)).strip()
        rows.append(
            {
                "station_id": station_id,
                "settlement_date_local": settlement_date_local.isoformat(),
                "timezone": STATION.timezone_name,
                "local_day_start_utc": isoformat_utc(local_start_utc),
                "local_day_end_utc": isoformat_utc(local_end_utc),
                "actual_tmax_native": actual_tmax_f,
                "actual_tmax_native_unit": TRUTH_NATIVE_UNIT,
                "actual_tmax_f": actual_tmax_f,
                "source": truth_source or TRUTH_SOURCE_NAME,
                "ingested_at_utc": ingested_at,
                "_report_issued_at_utc": _parse_report_issued_at(entry),
            }
        )
    rows.sort(key=lambda row: str(row["settlement_date_local"]))
    return rows


def _fetch_year(
    station_id: str,
    year: int,
    start_date: date,
    end_date: date,
    timeout_seconds: tuple[int, int],
) -> list[dict[str, object]]:
    url = _year_url(station_id, year)
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    return _parse_year_payload(station_id, year, payload, start_date, end_date)


def fetch_truth_rows(
    station_id: str = STATION.station_id,
    start_date: date | None = None,
    end_date: date | None = None,
    max_workers: int = DEFAULT_TRUTH_THREADS,
    timeout_seconds: tuple[int, int] = (10, 60),
) -> list[dict[str, object]]:
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date are required")
    station_id = station_id.strip().upper()
    years = list(range(start_date.year, end_date.year + 1))
    rows: list[dict[str, object]] = []
    LOGGER.info(
        "Fetching truth rows station=%s range=%s..%s years=%d workers=%d",
        station_id,
        start_date,
        end_date,
        len(years),
        max_workers,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_year = {
            executor.submit(
                _fetch_year,
                station_id,
                year,
                start_date,
                end_date,
                timeout_seconds,
            ): year
            for year in years
        }
        for future in concurrent.futures.as_completed(future_to_year):
            year = future_to_year[future]
            year_rows = future.result()
            LOGGER.info("Fetched truth year station=%s year=%s rows=%d", station_id, year, len(year_rows))
            rows.extend(year_rows)
    rows.sort(key=lambda row: str(row["settlement_date_local"]))
    return rows


def ingest_truth_range(
    connection,
    station_id: str,
    start_date: date,
    end_date: date,
    max_workers: int = DEFAULT_TRUTH_THREADS,
) -> list[dict[str, object]]:
    rows = fetch_truth_rows(
        station_id=station_id,
        start_date=start_date,
        end_date=end_date,
        max_workers=max_workers,
    )
    db.upsert_nws_daily_settlements(connection, rows)
    LOGGER.info(
        "Persisted truth rows station=%s range=%s..%s rows=%d",
        station_id,
        start_date,
        end_date,
        len(rows),
    )
    return rows
