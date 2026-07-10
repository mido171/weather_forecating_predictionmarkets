from __future__ import annotations

from typing import Any

import typer
from typer.testing import CliRunner

from klga_tmax.cli import app
from klga_tmax.constants import EXIT_VALIDATION_ERROR
from klga_tmax.registry.station_universe import (
    BACKDOOR_FRONT_STATIONS,
    COASTAL_MARINE_STATIONS,
    INLAND_HOT_REFERENCE_STATIONS,
    LONG_ISLAND_SOUND_STATIONS,
    MANDATORY_PSEUDO_POINT_REGISTRY,
    MANDATORY_STATION_REGISTRY,
    NYC_CORE_STATIONS,
    STATION_GROUPS,
    TARGET_STATION,
    TIER_A_POINT_IDS,
    TIER_B_POINT_IDS,
    UPSTREAM_SOUTHWEST_STATIONS,
    coordinate_tier,
    provider_station_id,
    station_group,
    tier_c_points,
)
from klga_tmax.validation.station_universe import validate_station_universe


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def mappings(self) -> "_FakeResult":
        return self

    def all(self) -> list[dict[str, Any]]:
        return self._rows


class _EmptyStationConnection:
    def execute(self, *_args: Any, **_kwargs: Any) -> _FakeResult:
        return _FakeResult([])


def test_task01_station_and_pseudo_point_counts_are_exact() -> None:
    assert len(MANDATORY_STATION_REGISTRY) == 19
    assert len(MANDATORY_PSEUDO_POINT_REGISTRY) == 10
    assert len({entry.station_id for entry in MANDATORY_STATION_REGISTRY}) == 19
    assert len({entry.grid_point_id for entry in MANDATORY_PSEUDO_POINT_REGISTRY}) == 10


def test_klga_provider_ids_are_exact() -> None:
    assert TARGET_STATION == "KLGA"
    assert provider_station_id("KLGA", "iem_asos") == "LGA"
    assert provider_station_id("KLGA", "wunderground") == "KLGA"
    assert provider_station_id("KLGA", "mos") == "LGA"


def test_coordinate_tiers_match_spec_order() -> None:
    assert tuple(entry.grid_point_id for entry in coordinate_tier("A")) == TIER_A_POINT_IDS
    assert tuple(entry.grid_point_id for entry in coordinate_tier("B")) == TIER_B_POINT_IDS


def test_tier_c_grid_is_deterministic_and_contains_base_point() -> None:
    tier_c = tier_c_points()
    assert len(tier_c) == 25
    assert len({entry.grid_point_id for entry in tier_c}) == 25
    base = [
        entry
        for entry in tier_c
        if entry.grid_point_id == "GP_KLGA_GRID_DLAT_+0.00_DLON_+0.00"
    ]
    assert len(base) == 1
    assert base[0].lat == 40.77945
    assert base[0].lon == -73.88027


def test_station_groups_match_spec() -> None:
    expected = {
        "TARGET_STATION": ("KLGA",),
        "NYC_CORE_STATIONS": NYC_CORE_STATIONS,
        "COASTAL_MARINE_STATIONS": COASTAL_MARINE_STATIONS,
        "INLAND_HOT_REFERENCE_STATIONS": INLAND_HOT_REFERENCE_STATIONS,
        "UPSTREAM_SOUTHWEST_STATIONS": UPSTREAM_SOUTHWEST_STATIONS,
        "BACKDOOR_FRONT_STATIONS": BACKDOOR_FRONT_STATIONS,
        "LONG_ISLAND_SOUND_STATIONS": LONG_ISLAND_SOUND_STATIONS,
    }
    assert STATION_GROUPS == expected
    for group_name, station_ids in expected.items():
        assert station_group(group_name) == station_ids


def test_station_universe_validation_fails_on_empty_database_rows() -> None:
    inspection = validate_station_universe(_EmptyStationConnection())  # type: ignore[arg-type]
    assert not inspection.ok
    assert any("registry.station_registry rows" in failure for failure in inspection.failures)


def test_validate_station_universe_cli_uses_validation_exit_code(monkeypatch) -> None:
    def fake_run_audited(**kwargs: Any) -> None:
        assert kwargs["command_name"] == "validate station-universe"
        assert kwargs["failure_exit_code"] == EXIT_VALIDATION_ERROR
        raise typer.Exit(kwargs["failure_exit_code"])

    monkeypatch.setattr("klga_tmax.cli._run_audited", fake_run_audited)
    result = CliRunner().invoke(app, ["validate", "station-universe"])
    assert result.exit_code == EXIT_VALIDATION_ERROR

