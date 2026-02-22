from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Any

import requests


_ISSUE_TIMESTAMP = re.compile(r"(\\d{12})")


@dataclass(frozen=True)
class CliDaily:
    station_id: str
    target_date_local: date
    tmax_f: float | None
    tmin_f: float | None
    report_issued_at_utc: datetime | None
    truth_source_url: str | None


@dataclass(frozen=True)
class CliPayload:
    station_id: str
    days: list[CliDaily]
    raw_json: str
    raw_payload_hash: str


def fetch_cli_year(base_url: str, station_id: str, year: int) -> CliPayload:
    if year < 1800 or year > 2500:
        raise ValueError(f"CLI year out of range: {year}")
    url = f"{base_url}/json/cli.py"
    params = {"station": station_id.upper(), "year": str(year), "fmt": "json"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    raw_json = resp.text
    if not raw_json.strip():
        raise ValueError("CLI response is empty")
    payload_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    data = json.loads(raw_json)
    if not isinstance(data, dict):
        raise ValueError("CLI response is not a JSON object")
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("CLI response missing results array")
    days: list[CliDaily] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        station = entry.get("station")
        if not station:
            continue
        station_norm = str(station).strip().upper()
        if station_norm != station_id.upper():
            continue
        valid = entry.get("valid")
        if not valid:
            continue
        target_date = _parse_date(str(valid))
        tmax = _parse_numeric(entry.get("high"))
        tmin = _parse_numeric(entry.get("low"))
        report_issued = _parse_issued_at(entry)
        truth_url = _parse_truth_source_url(entry)
        days.append(
            CliDaily(
                station_id=station_norm,
                target_date_local=target_date,
                tmax_f=tmax,
                tmin_f=tmin,
                report_issued_at_utc=report_issued,
                truth_source_url=truth_url,
            )
        )
    return CliPayload(
        station_id=station_id.upper(),
        days=days,
        raw_json=raw_json,
        raw_payload_hash=payload_hash,
    )


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid CLI date: {value}") from exc


def _parse_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.upper() in {"M", "T"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_issued_at(entry: dict[str, Any]) -> datetime | None:
    token = entry.get("product") or entry.get("link")
    if not token:
        return None
    match = _ISSUE_TIMESTAMP.search(str(token))
    if not match:
        return None
    stamp = match.group(1)
    try:
        dt = datetime.strptime(stamp, "%Y%m%d%H%M")
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc)


def _parse_truth_source_url(entry: dict[str, Any]) -> str | None:
    value = entry.get("link")
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
