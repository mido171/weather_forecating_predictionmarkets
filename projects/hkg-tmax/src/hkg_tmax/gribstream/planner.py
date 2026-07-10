"""Request planning for GribStream `/runs` acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hkg_tmax.gribstream.catalog import ResolvedSelector
from hkg_tmax.gribstream.client import request_sha256
from hkg_tmax_db.connection import import_psycopg


@dataclass(frozen=True)
class Coordinate:
    location_id: int
    location_code: str
    lat: float
    lon: float
    name: str

    def as_gribstream_coordinate(self) -> dict[str, Any]:
        return {"lat": self.lat, "lon": self.lon, "name": self.location_code}


@dataclass(frozen=True)
class RunsRequestPlan:
    dataset: str
    endpoint: str
    forecasted_from: str
    forecasted_until: str
    min_lead_time: str
    max_lead_time: str
    selector: ResolvedSelector
    coordinates: tuple[Coordinate, ...]
    payload: dict[str, Any]
    request_sha256: str
    estimated_credits: int
    estimated_rows: int


def load_canonical_locations(database_url: str, *, limit: int | None = None) -> list[Coordinate]:
    psycopg = import_psycopg()
    sql = """
        SELECT location_id, location_code, latitude, longitude, name
        FROM catalog.location
        ORDER BY location_code
    """
    if limit is not None:
        sql += " LIMIT %s"
        params: tuple[Any, ...] = (limit,)
    else:
        params = ()
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
    return [
        Coordinate(
            location_id=int(row[0]),
            location_code=str(row[1]),
            lat=float(row[2]),
            lon=float(row[3]),
            name=str(row[4]),
        )
        for row in rows
    ]


def parse_duration_hours(value: str) -> int:
    text = value.strip().lower()
    if text.endswith("h"):
        return int(text[:-1])
    if text.endswith("hour") or text.endswith("hours"):
        return int(text.split()[0])
    raise ValueError(f"Unsupported lead duration: {value}")


def estimate_hourly_credits(min_lead_time: str, max_lead_time: str, variable_count: int = 1) -> int:
    low = parse_duration_hours(min_lead_time)
    high = parse_duration_hours(max_lead_time)
    if high < low:
        raise ValueError("max lead must be >= min lead")
    return (high - low + 1) * variable_count


def build_runs_plan(
    *,
    selector: ResolvedSelector,
    locations: list[Coordinate],
    forecasted_from: str,
    forecasted_until: str,
    min_lead_time: str,
    max_lead_time: str,
    dataset: str = "gfs",
) -> RunsRequestPlan:
    if not locations:
        raise ValueError("At least one canonical location is required")
    datetime.fromisoformat(forecasted_from.replace("Z", "+00:00"))
    datetime.fromisoformat(forecasted_until.replace("Z", "+00:00"))
    payload = {
        "forecastedFrom": forecasted_from,
        "forecastedUntil": forecasted_until,
        "minLeadTime": min_lead_time,
        "maxLeadTime": max_lead_time,
        "coordinates": [location.as_gribstream_coordinate() for location in locations],
        "variables": [selector.as_request_variable()],
    }
    estimated_credits = estimate_hourly_credits(min_lead_time, max_lead_time, 1)
    return RunsRequestPlan(
        dataset=dataset,
        endpoint="runs",
        forecasted_from=forecasted_from,
        forecasted_until=forecasted_until,
        min_lead_time=min_lead_time,
        max_lead_time=max_lead_time,
        selector=selector,
        coordinates=tuple(locations),
        payload=payload,
        request_sha256=request_sha256(payload),
        estimated_credits=estimated_credits,
        estimated_rows=estimated_credits * len(locations),
    )
