from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_0094_expanded_high_error_interaction_lab import (
    InteractionSpec,
    active_mask_for_gate,
    apply_targeted_interaction,
    interaction_group,
    select_features,
)


def test_interaction_group_promotes_marine_and_upper_air_ceiling() -> None:
    assert interaction_group("daily_waglan_island_sea_temperature_lag7", "hko_daily_climate") == "marine_proxy"
    assert interaction_group("ua_layer_1000_925_ceiling_minus_isd_temp_c", "upper_air") == "upper_air_ceiling"
    assert interaction_group("target_roll120_mean_lag7_c", "target_memory") == "target_memory"


def test_select_features_keeps_requested_signal_groups() -> None:
    contrasts = pd.DataFrame(
        [
            {
                "feature": "target_roll120_mean_lag7_c",
                "family": "target_memory",
                "contrast_priority": 0.9,
            },
            {
                "feature": "isd_morning_to_midday_temp_rise_c",
                "family": "isd_station_network",
                "contrast_priority": 0.8,
            },
            {
                "feature": "ua_layer_1000_925_ceiling_minus_isd_temp_c",
                "family": "upper_air",
                "contrast_priority": 0.7,
            },
            {
                "feature": "daily_waglan_island_sea_temperature_lag7",
                "family": "hko_daily_climate",
                "contrast_priority": 0.6,
            },
        ]
    )
    features = pd.DataFrame({row["feature"]: [1.0, 2.0] for row in contrasts.to_dict("records")})

    selected = select_features(contrasts, features)

    assert selected["interaction_group"].tolist() == [
        "target_memory",
        "isd_station_network",
        "upper_air_ceiling",
        "marine_proxy",
    ]


def test_active_mask_for_gate_targets_mam_slices() -> None:
    frame = pd.DataFrame(
        {
            "season": ["MAM", "MAM", "DJF"],
            "forecast_source_family": ["press_archive", "rss_archive", "press_archive"],
            "frame_segment": ["newly_available_official_frame", "current_0081_frame", "newly_available_official_frame"],
        }
    )

    assert active_mask_for_gate(frame, "mam_new_frame").tolist() == [True, False, False]
    assert active_mask_for_gate(frame, "mam_press_archive").tolist() == [True, False, False]
    assert active_mask_for_gate(frame, "mam_all").tolist() == [True, True, False]


def test_targeted_interaction_uses_prior_active_rows_only() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-02-28", "2020-03-01", "2020-03-02", "2020-03-03"]),
            "forecast_source_family": ["press_archive"] * 4,
            "season": ["DJF", "MAM", "MAM", "MAM"],
            "frame_segment": ["newly_available_official_frame"] * 4,
            "era_bucket": ["a"] * 4,
            "target_tmax_c": [10.0, 18.0, 19.0, 18.0],
            "forecast_max_c": [30.0, 20.0, 21.0, 21.0],
            "candidate_prediction_c": [30.0, 20.0, 21.0, 21.0],
            "base_residual_c": [20.0, 2.0, 2.0, 3.0],
            "feature_a__x__feature_b__bucket": [0.0, 0.0, 0.0, 0.0],
        }
    )
    spec = InteractionSpec(
        candidate_id="test",
        pair_name="feature_a__x__feature_b",
        feature_a="feature_a",
        feature_b="feature_b",
        group_a="target_memory",
        group_b="marine_proxy",
        active_gate="mam_all",
        min_history=1,
        shrink_rows=0.0,
        correction_cap_c=10.0,
    )

    prediction, diagnostics = apply_targeted_interaction(frame, spec)

    assert prediction.tolist() == [30.0, 20.0, 19.0, 19.0]
    assert diagnostics["prior_rows"].tolist() == [0, 0, 1, 2]
    assert diagnostics["gate_active_row"].tolist() == [False, True, True, True]
