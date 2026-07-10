from __future__ import annotations

from datetime import date

from hkg_tmax.hkg_t24.governance import (
    check_four_year_oof_feasibility,
    operational_allowed_for_tier,
    tier_for_point_in_time_status,
)


def test_four_year_oof_feasibility_passes_long_window() -> None:
    result = check_four_year_oof_feasibility(
        date(2018, 1, 1),
        date(2023, 12, 31),
        reason_context="unit test",
    )

    assert result.status == "PASS"
    assert result.available_years >= 4.0


def test_four_year_oof_feasibility_blocks_short_window() -> None:
    result = check_four_year_oof_feasibility(
        date(2021, 12, 30),
        date(2023, 12, 31),
        reason_context="unit test",
    )

    assert result.status == "BLOCKED"
    assert result.available_years < 4.0


def test_point_in_time_status_to_tier_mapping() -> None:
    assert tier_for_point_in_time_status("OPERATIONAL_POINT_IN_TIME") == "GOLD_EXACT_VINTAGE"
    assert tier_for_point_in_time_status("RETROSPECTIVE_ONLY") == "MECHANISM_ONLY"
    assert tier_for_point_in_time_status("TARGET_ONLY") == "TARGET_ONLY"
    assert tier_for_point_in_time_status("MARKET_ONLY", role="event_market_rules_metadata") == "FORBIDDEN"


def test_operational_allowed_only_for_gold_and_silver() -> None:
    assert operational_allowed_for_tier("GOLD_EXACT_VINTAGE")
    assert operational_allowed_for_tier("SILVER_OPERATIONAL_REPLAY")
    assert not operational_allowed_for_tier("MECHANISM_ONLY")
    assert not operational_allowed_for_tier("FORBIDDEN")
