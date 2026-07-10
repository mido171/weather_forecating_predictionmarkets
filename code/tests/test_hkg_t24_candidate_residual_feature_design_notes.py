from __future__ import annotations

import math

from scripts.run_hkg_t24_candidate_residual_feature_design_notes import (
    deployable_status_for_family,
    finite_float,
    priority_tier,
)


def test_deployable_status_rejects_residual_and_label_fields() -> None:
    assert (
        deployable_status_for_family("station_attribute", "official_error_c")
        == "diagnostic_only_outcome_or_residual"
    )
    assert (
        deployable_status_for_family("target", "target_tmax_c")
        == "diagnostic_only_outcome_or_residual"
    )
    assert (
        deployable_status_for_family("station_trajectory", "air_temperature_c_latest_before_1500__delta_1d")
        == "deployable_input_candidate"
    )


def test_deployable_status_marks_upper_air_for_timestamp_audit() -> None:
    assert (
        deployable_status_for_family("upper_air", "igra_thickness_1000_500_m_change_48h")
        == "deployable_after_timestamp_audit"
    )


def test_priority_tier_boundaries() -> None:
    assert priority_tier(5.0) == "tier_1"
    assert priority_tier(3.0) == "tier_2"
    assert priority_tier(2.99) == "tier_3"
    assert priority_tier(math.nan) == "review"


def test_finite_float_returns_nan_for_non_numeric() -> None:
    assert finite_float("1.25") == 1.25
    assert math.isnan(finite_float("not-a-number"))
