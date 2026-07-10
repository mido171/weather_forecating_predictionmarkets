from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_online_residual_memory_halflife import (
    MemoryState,
    OnlineMemorySpec,
    apply_online_memory_spec,
    context_keys_for_row,
    eligible_decision,
    estimate_context_correction,
    half_life_factor,
)


def make_spec(**overrides: object) -> OnlineMemorySpec:
    values = {
        "candidate_id": "test",
        "context_set": "source",
        "min_history": 1,
        "min_perf_history": 1,
        "halflife_rows": 100.0,
        "support_shrink": 0.0,
        "min_prior_lift_c": -1.0,
        "correction_cap_c": 5.0,
        "combine_mode": "best_lift",
        "max_contexts": 4,
    }
    values.update(overrides)
    return OnlineMemorySpec(**values)  # type: ignore[arg-type]


def row(**overrides: object) -> pd.Series:
    values = {
        "forecast_source_family": "rss_archive",
        "season": "MAM",
        "month": 5,
        "signeddiff_bucket": "station_warmer_ge_1c",
        "forecast_range_bucket": "range_le_3c",
        "weather_bucket": "weather_sunny",
        "active_count_bucket": "station_stack_inactive",
    }
    values.update(overrides)
    return pd.Series(values)


def test_context_keys_include_source_season_behavior_cells() -> None:
    keys = context_keys_for_row(row(), "all")

    assert "source=rss_archive" in keys
    assert "source=rss_archive|season=MAM|signed=station_warmer_ge_1c|range=range_le_3c" in keys
    assert (
        "source=rss_archive|season=MAM|signed=station_warmer_ge_1c|range=range_le_3c|active=station_stack_inactive"
        in keys
    )


def test_online_memory_does_not_use_current_row_for_its_own_prediction() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "target_tmax_c": [25.0, 25.0],
            "fold_id": ["fold_a", "fold_a"],
            "forecast_source_family": ["rss_archive", "rss_archive"],
            "season": ["DJF", "DJF"],
            "month": [1, 1],
            "signeddiff_bucket": ["neutral", "neutral"],
            "forecast_range_bucket": ["range_le_3c", "range_le_3c"],
            "weather_bucket": ["weather_other", "weather_other"],
            "active_count_bucket": ["station_stack_inactive", "station_stack_inactive"],
            "base_0069_prediction_c": [20.0, 20.0],
        }
    )

    predictions = apply_online_memory_spec(frame, make_spec())

    assert predictions.loc[0, "memory_correction_c"] == 0.0
    assert predictions.loc[1, "memory_correction_c"] > 0.0


def test_context_decision_requires_prior_performance_support() -> None:
    state = MemoryState(residual_sum=3.0, residual_weight=3.0, residual_count=3)
    spec = make_spec(min_history=1, min_perf_history=2)

    assert eligible_decision("source=rss_archive", state, spec) is None


def test_estimate_context_correction_applies_support_shrink_and_cap() -> None:
    state = MemoryState(residual_sum=10.0, residual_weight=2.0, residual_count=2)
    spec = make_spec(support_shrink=2.0, correction_cap_c=2.0)

    assert estimate_context_correction(state, spec) == 2.0


def test_half_life_factor_is_between_zero_and_one() -> None:
    factor = half_life_factor(10.0)

    assert 0.0 < factor < 1.0
