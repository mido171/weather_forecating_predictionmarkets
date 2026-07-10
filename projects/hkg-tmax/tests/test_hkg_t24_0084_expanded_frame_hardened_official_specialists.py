from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_0084_expanded_frame_hardened_official_specialists import (
    HardenSpec,
    apply_hardened_gate,
    context_key,
)


def make_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "forecast_source_family": ["press_archive", "press_archive", "press_archive"],
            "season": ["DJF", "DJF", "DJF"],
            "month": [1, 1, 1],
            "frame_segment": ["current_0081_frame", "current_0081_frame", "current_0081_frame"],
            "era_bucket": ["press_2000_2001", "press_2000_2001", "press_2000_2001"],
            "forecast_max_c": [20.0, 20.0, 30.0],
            "candidate_prediction_c": [18.0, 19.0, 20.0],
            "target_tmax_c": [18.0, 19.0, 20.0],
        }
    )
    frame["raw_abs_error_c"] = (frame["forecast_max_c"] - frame["target_tmax_c"]).abs()
    frame["corrected_abs_error_c"] = (frame["candidate_prediction_c"] - frame["target_tmax_c"]).abs()
    return frame


def test_context_key_uses_requested_dimensions() -> None:
    row = make_frame().iloc[0]

    assert context_key(row, "source_season_era") == ("press_archive", "DJF", "press_2000_2001")
    assert context_key(row, "source_frame") == ("press_archive", "current_0081_frame")


def test_hard_gate_uses_prior_dates_not_same_date_rows() -> None:
    spec = HardenSpec(
        candidate_id="test",
        context_mode="global",
        min_history=1,
        margin_c=0.0,
        action="hard_gate",
    )

    prediction, diagnostics = apply_hardened_gate(make_frame(), spec)

    assert prediction.tolist() == [20.0, 19.0, 20.0]
    assert diagnostics["prior_rows"].tolist() == [0, 1, 1]
    assert diagnostics["selected_candidate_id"].tolist() == [
        "official_raw",
        "0083_prior_blend_source_top5_min90",
        "0083_prior_blend_source_top5_min90",
    ]


def test_soft_blend_uses_prior_weight_after_history_gate() -> None:
    spec = HardenSpec(
        candidate_id="test",
        context_mode="global",
        min_history=1,
        margin_c=0.0,
        action="soft_blend",
    )

    prediction, diagnostics = apply_hardened_gate(make_frame(), spec)

    assert prediction[0] == 20.0
    assert np.isclose(prediction[1], 19.023809523809526)
    assert np.isclose(prediction[2], 20.238095238095237)
    assert diagnostics["corrected_weight"].tolist()[0] == 0.0
