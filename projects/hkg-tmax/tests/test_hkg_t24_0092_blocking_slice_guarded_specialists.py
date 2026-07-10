from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_0092_blocking_slice_guarded_specialists import (
    apply_no_correction_guards,
    failed_slices_for_candidate,
    mask_for_slice,
)


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_source_family": ["press_archive", "rss_archive", "press_archive"],
            "frame_segment": ["current_0081_frame", "newly_available_official_frame", "current_0081_frame"],
            "season": ["DJF", "MAM", "JJA"],
            "candidate_prediction_c": [10.0, 20.0, 30.0],
        }
    )


def test_mask_for_slice_resolves_source_frame_and_season() -> None:
    frame = make_frame()

    assert mask_for_slice(frame, "rss").tolist() == [False, True, False]
    assert mask_for_slice(frame, "old_frame").tolist() == [True, False, True]
    assert mask_for_slice(frame, "season_JJA").tolist() == [False, False, True]


def test_failed_slices_for_candidate_extracts_positive_deltas() -> None:
    scoreboard = pd.DataFrame(
        {
            "candidate_id": ["candidate_a"],
            "rss_delta_mae_vs_0088_base": [0.1],
            "season_MAM_delta_mae_vs_0088_base": [-0.2],
            "season_JJA_delta_mae_vs_0088_base": [0.3],
            "delta_mae_vs_0088_base": [-0.01],
        }
    )

    assert failed_slices_for_candidate(scoreboard, "candidate_a") == ["rss", "season_JJA"]


def test_apply_no_correction_guards_falls_back_to_base_prediction() -> None:
    frame = make_frame()
    prediction = np.array([11.0, 21.0, 31.0])

    guarded, guard_rows = apply_no_correction_guards(
        frame=frame,
        prediction=prediction,
        failed_slices=["rss", "season_JJA"],
    )

    assert guarded.tolist() == [11.0, 20.0, 30.0]
    assert guard_rows["guarded_rows"].tolist() == [1, 1]
