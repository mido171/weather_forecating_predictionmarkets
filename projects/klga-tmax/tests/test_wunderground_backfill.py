from __future__ import annotations

from datetime import date

import pytest

from klga_tmax.providers.wunderground.backfill import (
    build_fetch_tasks,
    count_dates,
    parse_station_selection,
)
from klga_tmax.registry.station_universe import registry_entry_by_station_id


def test_parse_station_selection_excludes_pseudo_points() -> None:
    station = registry_entry_by_station_id("KLGA")
    assert parse_station_selection("KLGA") == (station,)

    with pytest.raises(ValueError, match="not fetchable"):
        parse_station_selection("GP_KLGA_EXACT")


def test_build_fetch_tasks_chunks_inclusive_date_windows() -> None:
    station = registry_entry_by_station_id("KLGA")
    tasks = build_fetch_tasks(
        stations=(station,),
        start_date=date(2021, 8, 1),
        end_date=date(2021, 8, 31),
        chunk_days=10,
    )

    assert [(task.start_date, task.end_date) for task in tasks] == [
        (date(2021, 8, 1), date(2021, 8, 10)),
        (date(2021, 8, 11), date(2021, 8, 20)),
        (date(2021, 8, 21), date(2021, 8, 30)),
        (date(2021, 8, 31), date(2021, 8, 31)),
    ]
    assert all(task.weathercom_location_id == "KLGA:9:US" for task in tasks)


def test_count_dates_is_inclusive() -> None:
    assert count_dates(date(2021, 8, 1), date(2021, 8, 31)) == 31
