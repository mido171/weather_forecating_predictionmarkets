from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_candidate_timestamp_eligibility_audit import (
    audit_candidates,
    source_family_tokens,
    timing_status_for_candidate,
)


def test_source_family_tokens_splits_composite_families() -> None:
    assert source_family_tokens("station_pair_spread+station_trajectory") == {
        "station_pair_spread",
        "station_trajectory",
    }


def test_timing_status_allows_station_latest_before_cutoff() -> None:
    row = pd.Series(
        {
            "deployable_status": "deployable_input_candidate",
            "source_family": "station_trajectory",
            "candidate_type": "station_trajectory",
            "deployable_feature_text": "air_temperature_c_latest_before_1500__current_minus_rolling_mean_14d",
        }
    )

    status = timing_status_for_candidate(row)

    assert status["timestamp_audit_status"] == "eligible_proven_pre_cutoff_station"
    assert status["allowed_for_future_walkforward"] is True


def test_timing_status_blocks_upper_air_until_available_at_proof() -> None:
    row = pd.Series(
        {
            "deployable_status": "deployable_input_candidate",
            "source_family": "isd_station_network+upper_air",
            "candidate_type": "cross_family_joint_regime",
            "deployable_feature_text": "isd_pressure_plane_lat_slope_hpa_per_deg; igra_thickness_1000_500_m_change_48h",
        }
    )

    status = timing_status_for_candidate(row)

    assert status["timestamp_audit_status"] == "timestamp_audit_required"
    assert status["allowed_for_future_walkforward"] is False


def test_timing_status_blocks_hko_daily_until_publication_lag_proof() -> None:
    row = pd.Series(
        {
            "deployable_status": "deployable_input_candidate",
            "source_family": "upper_air+hko_daily_climate",
            "candidate_type": "cross_family_joint_regime",
            "deployable_feature_text": "daily_hong_kong_observatory_mean_sea_level_pressure_lag7",
        }
    )

    status = timing_status_for_candidate(row)

    assert status["timestamp_audit_status"] == "timestamp_audit_required"
    assert "available" in str(status["required_proof_before_model"])


def test_timing_status_forbids_outcome_or_residual_fields() -> None:
    row = pd.Series(
        {
            "deployable_status": "diagnostic_only_outcome_or_residual",
            "source_family": "station_attribute",
            "candidate_type": "station_attribute",
            "deployable_feature_text": "official_error_c",
        }
    )

    status = timing_status_for_candidate(row)

    assert status["timestamp_audit_status"] == "forbidden_diagnostic_or_outcome"
    assert status["allowed_for_future_walkforward"] is False


def test_audit_candidates_sorts_allowed_rows_first() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "blocked",
                "deployable_status": "deployable_input_candidate",
                "source_family": "upper_air",
                "candidate_type": "upper_air",
                "deployable_feature_text": "igra_thickness_1000_500_m_change_48h",
                "primary_score": 10.0,
                "official_error_score": 1.0,
            },
            {
                "candidate_id": "allowed",
                "deployable_status": "deployable_input_candidate",
                "source_family": "station_attribute",
                "candidate_type": "station_attribute",
                "deployable_feature_text": "wind_speed_mps_latest_before_1500",
                "primary_score": 1.0,
                "official_error_score": 0.0,
            },
        ]
    )

    audit = audit_candidates(candidates)

    assert audit.iloc[0]["candidate_id"] == "allowed"
    assert bool(audit.iloc[0]["allowed_for_future_walkforward"])
