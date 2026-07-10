from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATH = REPO_ROOT / "scripts/run_tactical_gribstream_h24n_smoke.py"
MIGRATION_PATH = REPO_ROOT / "migrations/postgres/20260625_0007_tactical_gribstream_h24n_schema.sql"


def load_smoke_runner():
    spec = importlib.util.spec_from_file_location("run_tactical_gribstream_h24n_smoke", SMOKE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tactical_payload_uses_exact_times_list_not_broad_run_range() -> None:
    runner = load_smoke_runner()
    payload = runner.build_payload(runner.MODEL_SPECS["gfs"])

    assert payload["timesList"] == ["2021-03-23T00:00:00Z"]
    assert "forecastedFrom" not in payload
    assert "forecastedUntil" not in payload
    assert payload["minLeadTime"] == "15h"
    assert payload["maxLeadTime"] == "39h"
    assert len(payload["coordinates"]) == 12
    assert len(payload["variables"]) == 13


def test_full_ensemble_payload_is_hko_center_only_with_all_members() -> None:
    runner = load_smoke_runner()
    payload = runner.build_payload(runner.MODEL_SPECS["ifsenfo"])

    assert payload["timesList"] == ["2024-03-01T18:00:00Z"]
    assert len(payload["coordinates"]) == 1
    assert payload["coordinates"][0]["name"] == "hko_center"
    assert payload["members"] == list(range(51))
    assert payload["variables"] == [
        {"name": "2t", "level": "sfc", "info": "", "alias": "member_temperature_2m_k"},
    ]


def test_tactical_credit_estimate_uses_coordinate_blocks_not_location_count() -> None:
    runner = load_smoke_runner()
    gfs = runner.MODEL_SPECS["gfs"]
    payload = runner.build_payload(gfs)

    assert runner.expected_credits(gfs, payload) == 25 * 13 * 1 * 1


def test_migration_creates_wide_tactical_schema_and_blocks_broad_ranges() -> None:
    sql = MIGRATION_PATH.read_text(encoding="utf-8")

    assert "CREATE SCHEMA IF NOT EXISTS nwp_tactical" in sql
    assert "CREATE TABLE IF NOT EXISTS nwp_tactical.forecast_wide" in sql
    assert "temperature_2m_k double precision" in sql
    assert "CHECK (NOT (request_json ? 'forecastedFrom'))" in sql
    assert "CHECK (request_json ? 'timesList')" in sql
