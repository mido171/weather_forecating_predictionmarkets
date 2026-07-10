from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_station_official_family_router import (
    FamilyRouterSpec,
    absdiff_bucket,
    active_count_bucket,
    apply_candidate,
    group_key,
    promotion_gate,
    signeddiff_bucket,
)


def test_fixed_blend_uses_declared_station_weight() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01"]),
            "target_tmax_c": [25.0],
            "fold_id": ["fold"],
            "forecast_source_family": ["rss"],
            "official_family_prediction_c": [20.0],
            "station_family_prediction_c": [24.0],
        }
    )
    spec = FamilyRouterSpec("blend", "fixed_blend", 0.25, "global", 0, 0.0, 0.25, 1.0)

    out = apply_candidate(frame, spec)

    assert out["candidate_prediction_c"].tolist() == [21.0]
    assert out["station_weight"].tolist() == [0.25]


def test_prior_choice_router_excludes_current_row_from_decision() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "target_tmax_c": [24.0, 30.0],
            "fold_id": ["fold", "fold"],
            "forecast_source_family": ["rss", "rss"],
            "official_family_prediction_c": [20.0, 20.0],
            "station_family_prediction_c": [24.0, 30.0],
            "family_disagreement_c": [4.0, 10.0],
            "abs_family_disagreement_c": [4.0, 10.0],
            "active_member_count": [1, 1],
            "official_abs_error_c": [4.0, 10.0],
            "station_abs_error_c": [0.0, 0.0],
        }
    )
    spec = FamilyRouterSpec("router", "prior_choice", 0.0, "global", 1, 0.0, 1.0, 1.0)

    out = apply_candidate(frame, spec)

    assert out["station_weight"].tolist() == [0.0, 1.0]
    assert out["candidate_prediction_c"].tolist() == [20.0, 30.0]


def test_group_key_uses_only_pre_target_features() -> None:
    row = pd.Series(
        {
            "forecast_source_family": "rss",
            "abs_family_disagreement_c": 2.0,
            "family_disagreement_c": -2.0,
            "active_member_count": 3,
            "target_tmax_c": 99.0,
        }
    )

    assert group_key(row, "source_absdiff") == "rss|absdiff_1p50_2p50"
    assert group_key(row, "source_signeddiff") == "rss|station_cooler_ge_1c"
    assert group_key(row, "source_active_count") == "rss|station_stack_three_plus_members"
    assert absdiff_bucket(0.4) == "absdiff_le_0p75"
    assert signeddiff_bucket(0.0) == "families_close_lt_1c"
    assert active_count_bucket(0) == "station_stack_inactive"


def test_promotion_gate_requires_fold_and_late_improvement() -> None:
    row = {
        "delta_mae_vs_official": -0.01,
        "fold_delta_max_vs_official": 0.01,
        "late_delta_mae_vs_official": -0.01,
        "mean_station_weight": 0.25,
    }

    assert promotion_gate(row) is False

    row["fold_delta_max_vs_official"] = -0.001

    assert promotion_gate(row) is True
