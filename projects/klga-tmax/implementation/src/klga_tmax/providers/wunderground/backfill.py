from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.engine import Engine

from klga_tmax.constants import TARGET_TZ
from klga_tmax.ingestion.hash_keys import sha256_hex
from klga_tmax.providers.wunderground.client import (
    RateLimiter,
    WundergroundHistoricalClient,
    weathercom_location_id,
)
from klga_tmax.providers.wunderground.config import WundergroundSettings
from klga_tmax.providers.wunderground.models import (
    ParsedWundergroundResponse,
    PersistedWundergroundWindow,
    Units,
    WundergroundFetchTask,
    WundergroundRawDayResponse,
)
from klga_tmax.providers.wunderground.parser import parse_wunderground_response
from klga_tmax.providers.wunderground.persistence import (
    mark_station_dates_not_fetched,
    persist_wunderground_response,
    window_already_complete,
)
from klga_tmax.registry.materialize_targets import iter_dates
from klga_tmax.registry.station_universe import (
    MANDATORY_STATION_REGISTRY,
    StationRegistryEntry,
    registry_entry_by_station_id,
)

DEFAULT_BACKFILL_START_DATE = date(1973, 1, 1)
CONTRACT_START_DATE = date(2000, 1, 1)


@dataclass(frozen=True)
class FetchedWindow:
    task: WundergroundFetchTask
    response: WundergroundRawDayResponse
    parsed: ParsedWundergroundResponse


def latest_complete_local_date() -> date:
    return datetime.now(ZoneInfo(TARGET_TZ)).date() - timedelta(days=1)


def parse_station_selection(stations: str | None) -> tuple[StationRegistryEntry, ...]:
    if not stations or stations.strip().lower() in {"all", "*"}:
        return tuple(entry for entry in MANDATORY_STATION_REGISTRY if entry.wunderground_station_id)
    selected: list[StationRegistryEntry] = []
    for raw_station_id in stations.split(","):
        station_id = raw_station_id.strip().upper()
        if not station_id:
            continue
        try:
            entry = registry_entry_by_station_id(station_id)
        except KeyError as exc:
            raise ValueError(f"unknown station {station_id}") from exc
        if entry.is_pseudo_point or not entry.wunderground_station_id:
            raise ValueError(f"station {station_id} is not fetchable from Wunderground")
        selected.append(entry)
    if not selected:
        raise ValueError("no Wunderground-fetchable station IDs were selected")
    return tuple(selected)


def build_fetch_tasks(
    *,
    stations: Iterable[StationRegistryEntry],
    start_date: date,
    end_date: date,
    chunk_days: int,
    units: Units = "e",
) -> tuple[WundergroundFetchTask, ...]:
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    tasks: list[WundergroundFetchTask] = []
    for station in stations:
        assert station.wunderground_station_id is not None
        cursor = start_date
        while cursor <= end_date:
            window_end = min(end_date, cursor + timedelta(days=chunk_days - 1))
            tasks.append(
                WundergroundFetchTask(
                    station_id=station.station_id,
                    wunderground_station_id=station.wunderground_station_id,
                    weathercom_location_id=weathercom_location_id(station.wunderground_station_id),
                    start_date=cursor,
                    end_date=window_end,
                    units=units,
                )
            )
            cursor = window_end + timedelta(days=1)
    return tuple(tasks)


def _fetch_and_parse(
    *,
    task: WundergroundFetchTask,
    settings: WundergroundSettings,
    rate_limiter: RateLimiter,
) -> FetchedWindow:
    client = WundergroundHistoricalClient(settings, rate_limiter=rate_limiter)
    response = client.fetch_registry_station_range(
        canonical_station_id=task.station_id,
        wunderground_station_id=task.wunderground_station_id,
        start_local_date=task.start_date,
        end_local_date=task.end_date,
        units=task.units,
    )
    parsed = parse_wunderground_response(
        response,
        canonical_station_id=task.station_id,
        intraday_lag_minutes=settings.intraday_available_lag_minutes,
    )
    return FetchedWindow(task=task, response=response, parsed=parsed)


def fetch_window_dry_run(
    *,
    settings: WundergroundSettings,
    station_id: str,
    local_date: date,
    units: Units = "e",
) -> FetchedWindow:
    station = registry_entry_by_station_id(station_id.upper())
    if station.wunderground_station_id is None:
        raise ValueError(f"station {station_id} has no Wunderground provider ID")
    task = WundergroundFetchTask(
        station_id=station.station_id,
        wunderground_station_id=station.wunderground_station_id,
        weathercom_location_id=weathercom_location_id(station.wunderground_station_id),
        start_date=local_date,
        end_date=local_date,
        units=units,
    )
    return _fetch_and_parse(
        task=task,
        settings=settings,
        rate_limiter=RateLimiter(permits_per_minute=settings.rate_limit_per_minute),
    )


def persist_single_window(
    *,
    engine: Engine,
    settings: WundergroundSettings,
    station_id: str,
    start_date: date,
    end_date: date,
    job_id: str,
    units: Units = "e",
) -> PersistedWundergroundWindow:
    station = registry_entry_by_station_id(station_id.upper())
    if station.wunderground_station_id is None:
        raise ValueError(f"station {station_id} has no Wunderground provider ID")
    task = WundergroundFetchTask(
        station_id=station.station_id,
        wunderground_station_id=station.wunderground_station_id,
        weathercom_location_id=weathercom_location_id(station.wunderground_station_id),
        start_date=start_date,
        end_date=end_date,
        units=units,
    )
    fetched = _fetch_and_parse(
        task=task,
        settings=settings,
        rate_limiter=RateLimiter(permits_per_minute=settings.rate_limit_per_minute),
    )
    with engine.begin() as connection:
        return persist_wunderground_response(
            connection,
            job_id=job_id,
            response=fetched.response,
            parsed=fetched.parsed,
        )


def backfill_wunderground(
    *,
    engine: Engine,
    settings: WundergroundSettings,
    start_date: date,
    end_date: date,
    stations: str | None,
    job_id: str,
    chunk_days: int,
    workers: int,
    resume: bool,
    units: Units = "e",
) -> dict[str, int | str]:
    selected_stations = parse_station_selection(stations)
    tasks = list(
        build_fetch_tasks(
            stations=selected_stations,
            start_date=start_date,
            end_date=end_date,
            chunk_days=chunk_days,
            units=units,
        )
    )
    skipped = 0
    prepared_dates = 0
    with engine.begin() as connection:
        for station in selected_stations:
            assert station.wunderground_station_id is not None
            prepared_dates += mark_station_dates_not_fetched(
                connection,
                station_id=station.station_id,
                wunderground_station_id=station.wunderground_station_id,
                weathercom_location_id=weathercom_location_id(station.wunderground_station_id),
                start_date=start_date,
                end_date=end_date,
            )
        if resume:
            filtered_tasks: list[WundergroundFetchTask] = []
            for task in tasks:
                if window_already_complete(
                    connection,
                    station_id=task.station_id,
                    start_date=task.start_date,
                    end_date=task.end_date,
                ):
                    skipped += 1
                else:
                    filtered_tasks.append(task)
            tasks = filtered_tasks

    rate_limiter = RateLimiter(permits_per_minute=settings.rate_limit_per_minute)
    persisted: list[PersistedWundergroundWindow] = []
    failures = 0
    max_workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        task_iter = iter(tasks)
        future_map = {}

        def submit_next() -> None:
            try:
                next_task = next(task_iter)
            except StopIteration:
                return
            future_map[
                executor.submit(
                    _fetch_and_parse,
                    task=next_task,
                    settings=settings,
                    rate_limiter=rate_limiter,
                )
            ] = next_task

        for _ in range(max_workers):
            submit_next()

        while future_map:
            done_futures, _ = wait(future_map, return_when=FIRST_COMPLETED)
            for future in done_futures:
                task = future_map.pop(future)
                try:
                    fetched = future.result()
                except Exception as exc:
                    failures += 1
                    with engine.begin() as connection:
                        response = WundergroundRawDayResponse(
                            station_id=task.station_id,
                            wunderground_station_id=task.wunderground_station_id,
                            weathercom_location_id=task.weathercom_location_id,
                            start_local_date=task.start_date,
                            end_local_date=task.end_date,
                            units=task.units,
                            endpoint_url_redacted="unbuilt_request_failed_before_fetch",
                            retrieved_at_utc=datetime.now(ZoneInfo("UTC")),
                            http_status=None,
                            content_type=None,
                            response_body_text="",
                            response_body_sha256=sha256_hex(""),
                            response_size_bytes=0,
                            payload_json=None,
                            attempts=0,
                            error_type="WORKER_EXCEPTION",
                            error_message=str(exc),
                        )
                        persisted.append(
                            persist_wunderground_response(
                                connection,
                                job_id=job_id,
                                response=response,
                                parsed=ParsedWundergroundResponse(
                                    daily_actuals=(),
                                    intraday_observations=(),
                                ),
                            )
                        )
                    submit_next()
                    continue
                with engine.begin() as connection:
                    persisted_window = persist_wunderground_response(
                        connection,
                        job_id=job_id,
                        response=fetched.response,
                        parsed=fetched.parsed,
                    )
                    persisted.append(persisted_window)
                    if persisted_window.status == "failed":
                        failures += 1
                submit_next()

    daily_rows = sum(item.daily_rows_upserted for item in persisted)
    intraday_rows = sum(item.intraday_rows_upserted for item in persisted)
    coverage_rows = sum(item.coverage_rows_updated for item in persisted)
    revisions = sum(item.revisions_inserted for item in persisted)
    succeeded = sum(1 for item in persisted if item.status == "succeeded")
    no_data = sum(1 for item in persisted if item.status == "no_data")
    return {
        "job_id": job_id,
        "stations": len(selected_stations),
        "windows_planned": len(tasks) + skipped,
        "windows_fetched": len(tasks),
        "windows_skipped": skipped,
        "windows_succeeded": succeeded,
        "windows_no_data": no_data,
        "windows_failed": failures,
        "prepared_not_fetched_rows": prepared_dates,
        "daily_rows_upserted": daily_rows,
        "intraday_rows_upserted": intraday_rows,
        "coverage_rows_updated": coverage_rows,
        "revisions_inserted": revisions,
    }


def coverage_summary(
    *,
    engine: Engine,
    start_date: date,
    end_date: date,
    station_id: str | None,
) -> list[dict[str, object]]:
    where_station = "AND station_id = :station_id" if station_id else ""
    params: dict[str, object] = {"start_date": start_date, "end_date": end_date}
    if station_id:
        params["station_id"] = station_id.upper()
    with engine.begin() as connection:
        rows = connection.execute(
            text(
                f"""
                SELECT station_id, status, count(*) AS rows
                FROM audit.wu_station_date_coverage
                WHERE local_date BETWEEN :start_date AND :end_date
                {where_station}
                GROUP BY station_id, status
                ORDER BY station_id, status
                """
            ),
            params,
        ).mappings().all()
    return [dict(row) for row in rows]


def count_dates(start_date: date, end_date: date) -> int:
    return sum(1 for _ in iter_dates(start_date, end_date))
