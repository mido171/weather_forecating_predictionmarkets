from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from hkg_tmax.paths import ProjectPaths
from hkg_tmax_db.cutoff import (
    CUTOFF_RULE_VERSION,
    AvailabilityGrade,
    hkg_t24_cutoff_utc,
    is_strictly_eligible,
)

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
MIGRATION = (
    PROJECT_PATHS.db_root
    / "migrations/postgres/20260624_0002_t24_time_availability_contract.sql"
)


def test_cutoff_contract_version_matches_a_to_z_package() -> None:
    assert CUTOFF_RULE_VERSION == "hkg_t24_1500hkt_v1"


def test_cutoff_handles_leap_day_and_year_boundary() -> None:
    assert hkg_t24_cutoff_utc(date(2024, 2, 29)).isoformat() == "2024-02-28T07:00:00+00:00"
    assert hkg_t24_cutoff_utc(date(2026, 1, 1)).isoformat() == "2025-12-31T07:00:00+00:00"


def test_strict_eligibility_requires_grade_and_cutoff_order() -> None:
    cutoff = datetime(2026, 6, 22, 7, 0, tzinfo=UTC)

    assert is_strictly_eligible(
        available_at_utc=cutoff - timedelta(seconds=1),
        cutoff_utc=cutoff,
        grade=AvailabilityGrade.A_EXACT_FIRST_SEEN,
    )
    assert is_strictly_eligible(
        available_at_utc=cutoff,
        cutoff_utc=cutoff,
        grade=AvailabilityGrade.B_PROVIDER_SCHEDULE_PROVEN,
    )
    assert not is_strictly_eligible(
        available_at_utc=cutoff + timedelta(seconds=1),
        cutoff_utc=cutoff,
        grade=AvailabilityGrade.A_EXACT_FIRST_SEEN,
    )
    assert not is_strictly_eligible(
        available_at_utc=cutoff - timedelta(hours=1),
        cutoff_utc=cutoff,
        grade=AvailabilityGrade.C_RUN_TIME_ONLY,
    )
    assert not is_strictly_eligible(
        available_at_utc=None,
        cutoff_utc=cutoff,
        grade=AvailabilityGrade.A_EXACT_FIRST_SEEN,
    )


def test_strict_eligibility_rejects_naive_datetimes() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        is_strictly_eligible(
            available_at_utc=datetime(2026, 6, 22, 7, 0),
            cutoff_utc=datetime(2026, 6, 22, 7, 0, tzinfo=UTC),
            grade=AvailabilityGrade.A_EXACT_FIRST_SEEN,
        )


def test_contract_migration_declares_required_governance_objects() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS governance.operational_contract" in sql
    assert "CREATE TABLE IF NOT EXISTS governance.availability_grade" in sql
    assert "CREATE TABLE IF NOT EXISTS governance.sealed_period" in sql
    assert "CREATE OR REPLACE FUNCTION governance.is_available_for_cutoff" in sql
    assert "CREATE OR REPLACE FUNCTION governance.hkg_t24_is_eligible" in sql
    assert "REVOKE ALL ON SCHEMA sealed_confirmation FROM hkg_tmax_live_inference" in sql
