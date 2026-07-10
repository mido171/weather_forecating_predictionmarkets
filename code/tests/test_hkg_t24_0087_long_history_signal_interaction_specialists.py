from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_0087_long_history_signal_interaction_specialists import (
    InteractionSpec,
    apply_interaction_specialist,
    context_key,
)


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "forecast_source_family": ["press_archive", "press_archive", "press_archive"],
            "season": ["DJF", "DJF", "DJF"],
            "frame_segment": ["current_0081_frame", "current_0081_frame", "current_0081_frame"],
            "era_bucket": ["press_2000_2001", "press_2000_2001", "press_2000_2001"],
            "target_tmax_c": [18.0, 19.0, 20.0],
            "forecast_max_c": [20.0, 20.0, 30.0],
            "candidate_prediction_c": [20.0, 21.0, 22.0],
            "base_residual_c": [2.0, 2.0, 2.0],
            "feature_a__x__feature_b__bucket": [12.0, 12.0, 12.0],
        }
    )


def test_context_key_can_include_source_dimension() -> None:
    spec = InteractionSpec(
        candidate_id="test",
        feature_a="feature_a",
        feature_b="feature_b",
        context_mode="source_interaction",
        min_history=1,
        shrink_rows=0.0,
        correction_cap_c=10.0,
    )

    assert context_key(make_frame().iloc[0], spec) == ("feature_a__x__feature_b", 12, "press_archive")


def test_interaction_specialist_uses_prior_dates_not_same_date_residuals() -> None:
    spec = InteractionSpec(
        candidate_id="test",
        feature_a="feature_a",
        feature_b="feature_b",
        context_mode="interaction",
        min_history=1,
        shrink_rows=0.0,
        correction_cap_c=10.0,
    )

    prediction, diagnostics = apply_interaction_specialist(make_frame(), spec)

    assert prediction.tolist() == [20.0, 19.0, 20.0]
    assert diagnostics["prior_rows"].tolist() == [0, 1, 1]
    assert diagnostics["specialist_active"].tolist() == [False, True, True]
