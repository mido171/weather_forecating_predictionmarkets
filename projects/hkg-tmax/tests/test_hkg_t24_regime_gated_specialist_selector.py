from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_regime_gated_specialist_selector import (
    SelectorSpec,
    classify_primary_regime,
    expert_active_for_regime,
    past_only_regime_selector,
)


def test_classify_primary_regime_prioritizes_official_rain_cloud_text() -> None:
    row = pd.Series(
        {
            "text_any_rain": 1.0,
            "text_hot": 1.0,
            "forecast_max_c": 33.0,
            "isd_dew_point_mean_c_change_1d": 2.0,
        }
    )

    assert classify_primary_regime(row) == "rain_cloud"


def test_expert_active_for_regime_supports_all_and_exact_regimes() -> None:
    assert expert_active_for_regime("rain_cloud,hot_sunny", "rain_cloud")
    assert not expert_active_for_regime("rain_cloud,hot_sunny", "dry_mixing")
    assert expert_active_for_regime("all", "dry_mixing")


def test_past_only_regime_selector_excludes_current_target_label() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=4, freq="D"),
            "forecast_source_family": ["rss"] * 4,
            "primary_regime": ["rain_cloud"] * 4,
            "target_tmax_c": [10.0, 10.0, 10.0, 100.0],
            "official_raw": [10.0] * 4,
            "expert_a": [10.0, 10.0, 10.0, 10.0],
            "expert_b": [20.0, 20.0, 20.0, 100.0],
        }
    )
    mapping = pd.DataFrame(
        {
            "expert_id": ["expert_a", "expert_b"],
            "active_regimes": ["rain_cloud", "rain_cloud"],
        }
    )

    out = past_only_regime_selector(
        frame,
        mapping,
        SelectorSpec(mode="best", same_source=False, match_regime=True, min_history=2),
    )

    assert out.loc[3, "selected_expert"] == "expert_a"
    assert out.loc[3, "selector_prediction_c"] == pytest.approx(10.0)


def test_past_only_regime_selector_same_source_uses_same_source_history() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "forecast_source_family": ["press", "rss", "press", "rss", "rss"],
            "primary_regime": ["default"] * 5,
            "target_tmax_c": [10.0, 30.0, 10.0, 30.0, 30.0],
            "official_raw": [10.0] * 5,
            "expert_press_good": [10.0, 10.0, 10.0, 10.0, 10.0],
            "expert_rss_good": [30.0, 30.0, 30.0, 30.0, 30.0],
        }
    )
    mapping = pd.DataFrame(
        {
            "expert_id": ["expert_press_good", "expert_rss_good"],
            "active_regimes": ["all", "all"],
        }
    )

    out = past_only_regime_selector(
        frame,
        mapping,
        SelectorSpec(mode="best", same_source=True, match_regime=False, min_history=2),
    )

    assert out.loc[4, "forecast_source_family"] == "rss"
    assert out.loc[4, "selected_expert"] == "expert_rss_good"
    assert out.loc[4, "selector_prediction_c"] == pytest.approx(30.0)


def test_past_only_regime_selector_filters_to_active_regime() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=4, freq="D"),
            "forecast_source_family": ["rss"] * 4,
            "primary_regime": ["rain_cloud"] * 4,
            "target_tmax_c": [10.0, 10.0, 10.0, 10.0],
            "official_raw": [10.0] * 4,
            "hot_expert": [10.0, 10.0, 10.0, 10.0],
            "rain_expert": [11.0, 11.0, 11.0, 11.0],
        }
    )
    mapping = pd.DataFrame(
        {
            "expert_id": ["hot_expert", "rain_expert"],
            "active_regimes": ["hot_sunny", "rain_cloud"],
        }
    )

    out = past_only_regime_selector(
        frame,
        mapping,
        SelectorSpec(mode="best", same_source=False, match_regime=True, min_history=2),
    )

    assert out.loc[3, "selected_expert"] == "rain_expert"
