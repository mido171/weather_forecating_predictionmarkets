from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_pressure_regime_interaction_atlas import (
    PairExpertSpec,
    add_interaction_features,
    pair_expert_predictions,
)


def test_add_interaction_features_builds_spreads_and_phase_labels() -> None:
    frame = pd.DataFrame(
        {
            "target_date": ["2020-07-15"],
            "isd_station_air_temperature_c_590960_99999": [31.0],
            "isd_station_air_temperature_c_596730_99999": [29.5],
            "isd_station_dew_point_c_590960_99999": [25.0],
            "isd_station_dew_point_c_596730_99999": [24.25],
            "isd_wind_u_mean_mps": [-3.0],
            "isd_wind_v_mean_mps": [-4.0],
        }
    )

    out = add_interaction_features(frame)

    assert out.loc[0, "thermal_590960_minus_596730_c"] == pytest.approx(1.5)
    assert out.loc[0, "dew_590960_minus_596730_c"] == pytest.approx(0.75)
    assert out.loc[0, "season"] == "JJA"
    assert out.loc[0, "monsoon_phase"] == "southwest_monsoon"
    assert out.loc[0, "isd_wind_vector_speed_mps"] == pytest.approx(5.0)
    assert out.loc[0, "isd_onshore_easterly_proxy_mps"] == pytest.approx(3.0)


def test_pair_expert_predictions_excludes_current_target_date_label() -> None:
    frame = add_interaction_features(
        pd.DataFrame(
            {
                "target_date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "forecast_source_family": ["rss"] * 5,
                "target_tmax_c": [10.0, 10.0, 100.0, 10.0, 10.0],
                "forecast_max_c": [10.0] * 5,
                "pressure_feature": [0.0, 1.0, 100.0, 1.0, 1.0],
                "modifier_feature": [0.0, 1.0, 100.0, 1.0, 1.0],
            }
        )
    )
    spec = PairExpertSpec(
        pressure_feature="pressure_feature",
        modifier_feature="modifier_feature",
        bins=2,
        same_source=False,
        phase_conditioned=False,
        shrinkage=0.0,
        min_history=2,
        min_match_rows=1,
    )

    out = pair_expert_predictions(frame, spec)

    assert out.loc[2, "past_rows_used"] == 1
    assert out.loc[2, "residual_correction_c"] == pytest.approx(0.0)
    assert out.loc[2, "candidate_prediction_c"] == pytest.approx(10.0)


def test_pair_expert_predictions_same_source_isolates_history() -> None:
    frame = add_interaction_features(
        pd.DataFrame(
            {
                "target_date": pd.date_range("2020-01-01", periods=6, freq="D"),
                "forecast_source_family": ["rss", "press", "rss", "press", "rss", "press"],
                "target_tmax_c": [30.0, 11.0, 30.0, 11.0, 30.0, 11.0],
                "forecast_max_c": [10.0] * 6,
                "pressure_feature": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                "modifier_feature": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
    )
    spec = PairExpertSpec(
        pressure_feature="pressure_feature",
        modifier_feature="modifier_feature",
        bins=2,
        same_source=True,
        phase_conditioned=False,
        shrinkage=0.0,
        correction_clip_c=25.0,
        min_history=2,
        min_match_rows=1,
    )

    out = pair_expert_predictions(frame, spec)

    assert out.loc[4, "forecast_source_family"] == "rss"
    assert out.loc[4, "past_rows_used"] == 1
    assert out.loc[4, "residual_correction_c"] == pytest.approx(20.0)
    assert out.loc[4, "candidate_prediction_c"] == pytest.approx(30.0)
