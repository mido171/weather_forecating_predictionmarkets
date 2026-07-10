from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_forecast_history_state_deepening import (
    add_forecast_history_state_features,
    forecast_history_feature_sets,
    same_source_unchanged_streak,
    sign_bucket,
)
from scripts.run_hkg_t24_stack_trust_meta_features import past_only_meta_trust_predictions


def test_same_source_unchanged_streak_isolates_source_families() -> None:
    streak = same_source_unchanged_streak(
        pd.Series([30.0, 31.0, 30.0, 30.0, 31.0, 31.0]),
        pd.Series(["press", "rss", "press", "press", "rss", "rss"]),
    )

    assert streak.to_list() == [1.0, 1.0, 2.0, 3.0, 2.0, 3.0]


def test_sign_bucket_uses_fixed_temperature_thresholds() -> None:
    buckets = sign_bucket(pd.Series([-3.0, -0.5, 0.0, 0.5, 2.5, None]), near=0.25, large=2.0)

    assert buckets.to_list() == ["large_down", "down", "flat", "up", "large_up", "missing"]


def test_add_forecast_history_state_features_builds_current_source_states() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "forecast_source_family": ["press", "press", "press"],
            "forecast_max_c": [25.0, 25.0, 26.0],
            "forecast_max_vs_prior7_mean_source_c": [-2.5, 0.1, 2.5],
            "forecast_max_change_1_source_c": [-1.0, 0.0, 1.0],
            "forecast_range_c": [3.0, 4.0, 5.0],
            "forecast_range_change_1_source_c": [-1.0, 0.0, 1.0],
            "forecast_range_vs_prior7_mean_source_c": [-0.5, 0.0, 0.5],
            "forecast_midpoint_change_1_source_c": [-0.5, 0.0, 0.5],
            "issue_to_cutoff_change_1_source_c": [-6.0, 0.0, 6.0],
            "forecast_max_prior7_std_source_c": [0.2, 0.6, 1.2],
            "prediction_0018_c_prior90_source_residual_mean_c": [-0.3, 0.0, 0.3],
            "text_hot": [0.0, 1.0, 1.0],
            "text_very_hot": [0.0, 0.0, 1.0],
            "text_any_rain": [1.0, 0.0, 0.0],
            "text_cloud": [1.0, 0.0, 0.0],
            "text_sunny_or_fine": [0.0, 1.0, 1.0],
            "text_thunder": [0.0, 0.0, 0.0],
            "text_humid": [0.0, 0.0, 0.0],
            "text_hot_change_1_source": [0.0, 1.0, 0.0],
            "text_cloud_change_1_source": [0.0, -1.0, 0.0],
            "text_any_rain_change_1_source": [0.0, -1.0, 0.0],
            "text_hot_turned_on_source": [0.0, 1.0, 0.0],
            "text_cloud_turned_off_source": [0.0, 1.0, 0.0],
            "text_any_rain_turned_off_source": [0.0, 1.0, 0.0],
        }
    )

    enriched, catalog = add_forecast_history_state_features(frame)

    assert enriched.loc[1, "forecast_max_unchanged_streak_source"] == 2.0
    assert enriched.loc[1, "text_turnover_count_source"] == 3.0
    assert enriched.loc[2, "meta_forecast_vs_prior7_sign"] == "large_up"
    assert enriched.loc[2, "meta_text_signal_state"] == "very_hot"
    assert "meta_forecast_history_state" in catalog["meta_feature"].to_list()


def test_forecast_history_feature_sets_include_deep_composites() -> None:
    features = [
        "meta_forecast_vs_prior7_fine_bin",
        "meta_forecast_vs_prior7_abs_bin",
        "meta_forecast_vs_prior7_sign",
        "meta_forecast_jump_fine_bin",
        "meta_forecast_jump_sign",
        "meta_forecast_history_state",
        "meta_revision_range_state",
        "meta_forecast_confidence_state",
    ]

    sets = forecast_history_feature_sets(features)

    assert sets["forecast_vs_prior_deep"] == tuple(features[:5])
    assert sets["forecast_history_core"] == tuple(features[5:])


def test_forecast_history_state_works_with_prior_only_selector() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "forecast_source_family": ["press", "press"],
            "target_tmax_c": [10.0, 100.0],
            "official_raw": [10.0, 10.0],
            "family_0033_smooth": [20.0, 100.0],
            "family_0034_centroid": [10.0, 10.0],
            "family_0035_revision": [20.0, 100.0],
            "meta_forecast_history_state": ["flat|flat|<= 1", "flat|flat|<= 1"],
        }
    )

    predictions = past_only_meta_trust_predictions(
        frame,
        feature_names=("meta_forecast_history_state",),
        mode="best",
        same_source=False,
        min_bucket_history=1,
        min_global_history=1,
    )

    assert predictions.loc[1, "selected_family"] == "family_0034_centroid"
    assert predictions.loc[1, "expert_prediction_c"] == 10.0
