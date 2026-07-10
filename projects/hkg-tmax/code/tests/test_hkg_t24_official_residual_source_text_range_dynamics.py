from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_official_residual_source_text_range_dynamics import (
    BucketExpertSpec,
    add_forecast_dynamics,
    add_source_phase_features,
    add_text_flags,
    past_only_bucket_predictions,
)


def test_add_forecast_dynamics_uses_prior_same_source_only() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            ),
            "forecast_source_family": ["rss", "press", "rss", "press"],
            "forecast_max_c": [20.0, 100.0, 22.0, 104.0],
            "forecast_min_c": [15.0, 95.0, 16.0, 98.0],
            "forecast_range_c": [5.0, 5.0, 6.0, 6.0],
            "forecast_midpoint_c": [17.5, 97.5, 19.0, 101.0],
            "issue_to_cutoff_hours": [3.0, 3.0, 4.0, 4.0],
        }
    )

    out = add_forecast_dynamics(frame)
    rss_second = out[(out["forecast_source_family"].eq("rss")) & (out["target_date"].eq(pd.Timestamp("2020-01-03")))].iloc[0]
    press_second = out[(out["forecast_source_family"].eq("press")) & (out["target_date"].eq(pd.Timestamp("2020-01-04")))].iloc[0]

    assert rss_second["forecast_max_lag1_source_c"] == pytest.approx(20.0)
    assert rss_second["forecast_max_change_1_source_c"] == pytest.approx(2.0)
    assert press_second["forecast_max_lag1_source_c"] == pytest.approx(100.0)
    assert press_second["forecast_max_change_1_source_c"] == pytest.approx(4.0)


def test_add_text_flags_extracts_weather_and_description_keywords() -> None:
    frame = pd.DataFrame(
        {
            "weather_text": ["Mainly cloudy with showers and isolated thunderstorms."],
            "wind_text": ["Moderate easterly winds."],
            "description_text": ["Very hot and humid during the day."],
        }
    )

    out = add_text_flags(frame)

    assert out.loc[0, "text_showers"] == pytest.approx(1.0)
    assert out.loc[0, "text_thunder"] == pytest.approx(1.0)
    assert out.loc[0, "text_cloud"] == pytest.approx(1.0)
    assert out.loc[0, "text_very_hot"] == pytest.approx(1.0)
    assert out.loc[0, "text_humid"] == pytest.approx(1.0)
    assert out.loc[0, "text_easterly"] == pytest.approx(1.0)
    assert out.loc[0, "text_keyword_count"] >= 6.0


def test_past_only_bucket_predictions_excludes_current_target_date_label() -> None:
    frame = add_source_phase_features(
        pd.DataFrame(
            {
                "target_date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "forecast_source_family": ["rss"] * 5,
                "target_tmax_c": [10.0, 10.0, 100.0, 10.0, 10.0],
                "forecast_max_c": [10.0] * 5,
                "feature_a": [0.0, 1.0, 100.0, 1.0, 1.0],
            }
        )
    )
    spec = BucketExpertSpec(
        feature="feature_a",
        bins=2,
        exact_match=False,
        same_source=False,
        month_conditioned=False,
        phase_conditioned=False,
        shrinkage=0.0,
        min_history=2,
        min_match_rows=1,
    )

    out = past_only_bucket_predictions(frame, spec)

    assert out.loc[2, "past_rows_used"] == 1
    assert out.loc[2, "residual_correction_c"] == pytest.approx(0.0)
    assert out.loc[2, "candidate_prediction_c"] == pytest.approx(10.0)


def test_past_only_bucket_predictions_same_source_isolates_history() -> None:
    frame = add_source_phase_features(
        pd.DataFrame(
            {
                "target_date": pd.date_range("2020-01-01", periods=6, freq="D"),
                "forecast_source_family": ["rss", "press", "rss", "press", "rss", "press"],
                "target_tmax_c": [30.0, 11.0, 30.0, 11.0, 30.0, 11.0],
                "forecast_max_c": [10.0] * 6,
                "feature_a": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
    )
    spec = BucketExpertSpec(
        feature="feature_a",
        bins=2,
        exact_match=True,
        same_source=True,
        month_conditioned=False,
        phase_conditioned=False,
        shrinkage=0.0,
        correction_clip_c=25.0,
        min_history=2,
        min_match_rows=1,
    )

    out = past_only_bucket_predictions(frame, spec)

    assert out.loc[4, "forecast_source_family"] == "rss"
    assert out.loc[4, "past_rows_used"] == 1
    assert out.loc[4, "residual_correction_c"] == pytest.approx(20.0)
    assert out.loc[4, "candidate_prediction_c"] == pytest.approx(30.0)
