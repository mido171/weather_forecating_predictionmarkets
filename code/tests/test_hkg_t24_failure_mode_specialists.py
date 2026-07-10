from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_failure_mode_specialists import (
    FailureSpecialistSpec,
    FeatureCondition,
    fold_local_threshold,
    past_only_failure_specialist_prediction,
)


def base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "forecast_source_family": ["press_archive"] * 5,
            "month": [1] * 5,
            "target_tmax_c": [11.0, 11.0, 100.0, 11.0, 11.0],
            "forecast_max_c": [10.0, 10.0, 10.0, 10.0, 10.0],
            "feature_a": [1.0, 2.0, 100.0, 1.5, 1.6],
        }
    )


def test_fold_local_threshold_excludes_current_row() -> None:
    values = pd.Series([1.0, 2.0, 100.0]).to_numpy(dtype=float)
    prior_mask = pd.Series([True, True, False]).to_numpy(dtype=bool)
    condition = FeatureCondition("feature_a", "high", 0.80)

    threshold = fold_local_threshold(values, prior_mask, condition, min_history=2)

    assert threshold == pytest.approx(1.8)


def test_failure_specialist_uses_prior_residual_not_current_target_label() -> None:
    spec = FailureSpecialistSpec(
        family="test",
        name="high_feature",
        conditions=(FeatureCondition("feature_a", "high", 0.80),),
        statistic="shrunk_mean",
        shrinkage=0.0,
        min_history=2,
        min_match_rows=1,
    )

    out = past_only_failure_specialist_prediction(base_frame(), spec)

    assert out.loc[2, "candidate_prediction_c"] == pytest.approx(11.0)
    assert out.loc[2, "past_rows_used"] == 1
    assert bool(out.loc[2, "triggered"])


def test_failure_specialist_falls_back_when_current_row_does_not_trigger() -> None:
    frame = base_frame()
    frame.loc[2, "feature_a"] = 1.1
    spec = FailureSpecialistSpec(
        family="test",
        name="high_feature",
        conditions=(FeatureCondition("feature_a", "high", 0.80),),
        statistic="shrunk_mean",
        shrinkage=0.0,
        min_history=2,
        min_match_rows=1,
    )

    out = past_only_failure_specialist_prediction(frame, spec)

    assert out.loc[2, "candidate_prediction_c"] == pytest.approx(10.0)
    assert out.loc[2, "past_rows_used"] == 0
    assert not bool(out.loc[2, "triggered"])


def test_failure_specialist_same_source_uses_only_prior_same_source() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "forecast_source_family": [
                "press_archive",
                "rss_archive",
                "press_archive",
                "rss_archive",
                "press_archive",
                "rss_archive",
            ],
            "month": [1] * 6,
            "target_tmax_c": [11.0, 30.0, 11.0, 30.0, 11.0, 100.0],
            "forecast_max_c": [10.0] * 6,
            "feature_a": [1.0, 1.0, 1.1, 2.0, 1.2, 3.0],
        }
    )
    spec = FailureSpecialistSpec(
        family="test",
        name="high_feature",
        conditions=(FeatureCondition("feature_a", "high", 0.50),),
        same_source=True,
        statistic="shrunk_mean",
        shrinkage=0.0,
        min_history=2,
        min_match_rows=1,
    )

    out = past_only_failure_specialist_prediction(frame, spec)

    assert out.loc[5, "candidate_prediction_c"] == pytest.approx(30.0)
    assert out.loc[5, "past_rows_used"] == 1
