from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import requests


@dataclass(frozen=True)
class MosValue:
    raw: str | None
    numeric: float | None


@dataclass(frozen=True)
class MosEntry:
    runtime_utc: datetime
    forecast_time_utc: datetime | None
    values: dict[str, MosValue]


@dataclass(frozen=True)
class MosPayload:
    station_id: str
    model: str
    entries: list[MosEntry]
    raw_json: str
    raw_payload_hash: str


@dataclass
class SummaryStats:
    values: list[float] = field(default_factory=list)
    sum_value: float = 0.0
    count: int = 0
    min_value: float | None = None
    max_value: float | None = None
    first_forecast_time_utc: datetime | None = None
    last_forecast_time_utc: datetime | None = None

    def add(self, value: float, forecast_time_utc: datetime) -> None:
        if self.count == 0:
            self.min_value = value
            self.max_value = value
            self.first_forecast_time_utc = forecast_time_utc
            self.last_forecast_time_utc = forecast_time_utc
        else:
            if self.min_value is None or value < self.min_value:
                self.min_value = value
            if self.max_value is None or value > self.max_value:
                self.max_value = value
            if self.first_forecast_time_utc and forecast_time_utc < self.first_forecast_time_utc:
                self.first_forecast_time_utc = forecast_time_utc
            if self.last_forecast_time_utc and forecast_time_utc > self.last_forecast_time_utc:
                self.last_forecast_time_utc = forecast_time_utc
        self.values.append(value)
        self.sum_value += value
        self.count += 1

    def mean(self, scale: int = 4) -> float:
        if self.count == 0:
            return float("nan")
        return round(self.sum_value / self.count, scale)

    def median(self, scale: int = 4) -> float:
        if self.count == 0:
            return float("nan")
        return round(statistics.median(self.values), scale)


def fetch_mos_payload(
    base_url: str,
    station_id: str,
    model: str,
    start: datetime,
    end: datetime,
) -> MosPayload:
    params = {
        "station": station_id.upper(),
        "model": model.upper(),
        "sts": start.strftime("%Y-%m-%dT%H:%MZ"),
        "ets": end.strftime("%Y-%m-%dT%H:%MZ"),
        "format": "json",
    }
    url = f"{base_url}/cgi-bin/request/mos.py?{urlencode(params)}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    raw_json = resp.text
    if not raw_json.strip():
        raise ValueError("MOS response is empty")
    payload_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    data = json.loads(raw_json)
    if not isinstance(data, list):
        raise ValueError("MOS response is not a JSON array")

    expected_station = station_id.strip().upper()
    expected_model = _normalize_model(model)
    entries: list[MosEntry] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        station = row.get("station")
        entry_station = str(station).strip().upper() if station else None
        if entry_station != expected_station:
            raise ValueError(
                f"Station mismatch in MOS payload: expected {expected_station} got {entry_station}"
            )
        entry_model = _normalize_model(row.get("model"))
        if entry_model != expected_model:
            raise ValueError(
                f"Model mismatch in MOS payload: expected {expected_model} got {entry_model}"
            )
        runtime_utc = _parse_time_utc(row.get("runtime"))
        if runtime_utc is None:
            raise ValueError("MOS entry missing runtime")
        forecast_time_utc = _parse_time_utc(row.get("ftime"))
        values = _parse_values(row)
        entries.append(MosEntry(runtime_utc=runtime_utc, forecast_time_utc=forecast_time_utc, values=values))

    return MosPayload(
        station_id=expected_station,
        model=expected_model,
        entries=entries,
        raw_json=raw_json,
        raw_payload_hash=payload_hash,
    )


def resolve_daily_tmax(
    payload: MosPayload,
    asof_utc: datetime,
    target_date_local: date,
    zone_id: str,
    var_code: str = "n_x",
) -> float | None:
    window_start, window_end = mos_window_utc(target_date_local, zone_id)
    runtime = select_runtime(
        payload.entries, asof_utc, window_start, window_end, ZoneInfo(zone_id), target_date_local
    )
    if runtime is None:
        return None
    summaries = summarize_entries_for_runtime(
        payload.entries,
        runtime,
        target_date_local,
        window_start,
        window_end,
        ZoneInfo(zone_id),
    )
    stats = summaries.get(var_code.lower())
    if not stats or stats.count == 0:
        return None
    return float(stats.max_value) if stats.max_value is not None else None


def mos_window_utc(target_date_local: date, zone_id: str) -> tuple[datetime, datetime]:
    zone = ZoneInfo(zone_id)
    start_local = datetime.combine(target_date_local, datetime.min.time(), tzinfo=zone)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def select_runtime(
    entries: list[MosEntry],
    asof_utc: datetime,
    window_start_utc: datetime,
    window_end_utc: datetime,
    station_zone: ZoneInfo,
    target_date_local: date,
) -> datetime | None:
    if not entries:
        return None
    candidates: set[datetime] = set()
    covered: set[datetime] = set()
    for entry in entries:
        runtime_utc = entry.runtime_utc
        if runtime_utc > asof_utc:
            continue
        candidates.add(runtime_utc)
        forecast_time = entry.forecast_time_utc
        if (
            forecast_time is None
            or forecast_time < window_start_utc
            or forecast_time >= window_end_utc
        ):
            continue
        entry_target = forecast_time.astimezone(station_zone).date()
        if entry_target == target_date_local:
            covered.add(runtime_utc)
    if not covered:
        return None
    return max(covered)


def summarize_entries_for_runtime(
    entries: list[MosEntry],
    runtime_utc: datetime,
    target_date_local: date,
    window_start_utc: datetime,
    window_end_utc: datetime,
    station_zone: ZoneInfo,
) -> dict[str, SummaryStats]:
    summaries: dict[str, SummaryStats] = {}
    for entry in entries:
        if entry.runtime_utc != runtime_utc:
            continue
        forecast_time = entry.forecast_time_utc
        if (
            forecast_time is None
            or forecast_time < window_start_utc
            or forecast_time >= window_end_utc
        ):
            continue
        entry_target = forecast_time.astimezone(station_zone).date()
        if entry_target != target_date_local:
            continue
        for code, value in entry.values.items():
            if value.numeric is None:
                continue
            stats = summaries.setdefault(code.lower(), SummaryStats())
            stats.add(float(value.numeric), forecast_time)
    return summaries


def build_daily_rows_for_runtime(
    payload: MosPayload,
    station_zoneid: str,
    asof_utc: datetime,
    runtime_utc: datetime,
    target_date_local: date,
    window_start_utc: datetime,
    window_end_utc: datetime,
    retrieved_at_utc: datetime,
    variable_filter: set[str] | None = None,
) -> list[dict]:
    summaries = summarize_entries_for_runtime(
        payload.entries,
        runtime_utc,
        target_date_local,
        window_start_utc,
        window_end_utc,
        ZoneInfo(station_zoneid),
    )
    rows: list[dict] = []
    for variable_code, stats in summaries.items():
        if stats.count == 0:
            continue
        if variable_filter and variable_code not in variable_filter:
            continue
        row = {
            "station_id": payload.station_id,
            "station_zoneid": station_zoneid,
            "model": payload.model,
            "asof_utc": asof_utc,
            "runtime_utc": runtime_utc,
            "target_date_local": target_date_local,
            "variable_code": variable_code,
            "value_min": stats.min_value,
            "value_max": stats.max_value,
            "value_mean": stats.mean(),
            "value_median": stats.median(),
            "sample_count": stats.count,
            "first_forecast_time_utc": stats.first_forecast_time_utc,
            "last_forecast_time_utc": stats.last_forecast_time_utc,
            "raw_payload_hash_ref": payload.raw_payload_hash,
            "retrieved_at_utc": retrieved_at_utc,
        }
        rows.append(row)
    return rows


def _normalize_model(value: Any) -> str:
    if value is None:
        raise ValueError("MOS model is required")
    text = str(value).strip().upper()
    if text == "ETA":
        return "NAM"
    return text


def _parse_time_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned.isdigit():
            return datetime.fromtimestamp(int(cleaned) / 1000.0, tz=timezone.utc)
        if cleaned.endswith("Z"):
            cleaned = cleaned.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(cleaned)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _parse_values(entry: dict[str, Any]) -> dict[str, MosValue]:
    values: dict[str, MosValue] = {}
    for key, value in entry.items():
        if key is None:
            continue
        normalized = str(key).strip().lower()
        if normalized in {"station", "model", "runtime", "ftime"}:
            continue
        parsed = _parse_value(value)
        values[normalized] = parsed
    return values


def _parse_value(value: Any) -> MosValue:
    if value is None:
        return MosValue(None, None)
    if isinstance(value, (int, float)):
        return MosValue(str(value), float(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return MosValue(None, None)
        numeric = _parse_numeric(text)
        return MosValue(text, numeric)
    return MosValue(str(value), None)


def _parse_numeric(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in {"M", "T"}:
        return None
    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[0].strip()
        if not cleaned or cleaned.upper() in {"M", "T"}:
            return None
    try:
        return float(cleaned)
    except ValueError:
        return None
