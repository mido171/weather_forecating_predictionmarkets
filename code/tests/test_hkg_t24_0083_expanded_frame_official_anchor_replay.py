from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_0083_expanded_frame_official_anchor_replay import (
    OnlineBiasSpec,
    apply_online_bias,
    apply_past_performance_selector,
)


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "forecast_source_family": ["press_archive", "press_archive", "press_archive"],
            "season": ["DJF", "DJF", "DJF"],
            "month": [1, 1, 1],
            "forecast_max_c": [20.0, 20.0, 30.0],
            "target_tmax_c": [18.0, 19.0, 20.0],
            "official_error_c": [2.0, 1.0, 10.0],
        }
    )


def test_online_bias_uses_prior_dates_not_same_date_rows() -> None:
    spec = OnlineBiasSpec(
        candidate_id="test",
        group_mode="global",
        group_cols=(),
        half_life_days=365.0,
        min_history=1,
        shrink_rows=0.0,
        correction_cap_c=20.0,
    )

    prediction = apply_online_bias(make_frame(), spec)

    assert prediction[0] == 20.0
    assert np.isclose(prediction[1], 18.0)
    assert np.isclose(prediction[2], 28.0)


def test_past_performance_selector_does_not_use_same_date_errors() -> None:
    frame = make_frame()
    candidate_ids = ["official_raw", "candidate_b"]
    prediction_matrix = np.array(
        [
            [20.0, 18.0],
            [20.0, 21.0],
            [30.0, 20.0],
        ],
        dtype=float,
    )

    prediction, selected = apply_past_performance_selector(
        frame,
        candidate_ids=candidate_ids,
        prediction_matrix=prediction_matrix,
        mode="global",
        min_prior_rows=1,
    )

    assert selected == ["official_raw", "candidate_b", "candidate_b"]
    assert prediction.tolist() == [20.0, 21.0, 20.0]
