from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_0090_guarded_specialists_from_error_autopsy import (
    evaluation_masks,
    make_specs,
    score_candidate,
)


def test_make_specs_uses_smaller_calendar_correction_cap() -> None:
    leads = pd.DataFrame(
        {
            "feature": ["doy_sin", "target_roll120_mean_lag7_c"],
            "lead_rank": [1, 2],
            "contrast_priority": [0.5, 0.4],
            "recommended_action": ["calendar guard", "memory guard"],
        }
    )
    thresholds = pd.DataFrame({"feature": ["doy_sin", "target_roll120_mean_lag7_c"]})

    specs = make_specs(leads, thresholds)

    calendar_caps = {spec.correction_cap_c for spec in specs if spec.feature == "doy_sin"}
    memory_caps = {spec.correction_cap_c for spec in specs if spec.feature == "target_roll120_mean_lag7_c"}
    assert calendar_caps == {0.30}
    assert memory_caps == {0.55}


def test_score_candidate_rejects_season_regression_even_when_full_mae_improves() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-04-01", "2020-04-02"]),
            "forecast_source_family": ["press_archive"] * 4,
            "frame_segment": ["current_0081_frame", "current_0081_frame", "newly_available_official_frame", "newly_available_official_frame"],
            "season": ["DJF", "DJF", "MAM", "MAM"],
            "target_tmax_c": [0.0, 0.0, 0.0, 0.0],
            "forecast_max_c": [1.0, 1.0, 1.0, 1.0],
            "candidate_prediction_c": [1.0, 1.0, 1.0, 1.0],
        }
    )
    prediction = np.array([0.0, 0.0, 1.5, 1.5])

    row = score_candidate(
        frame,
        candidate_id="test",
        candidate_class="test",
        prediction=prediction,
        mask_map=evaluation_masks(frame),
    )

    assert row["delta_mae_vs_0088_base"] < 0.0
    assert row["season_MAM_delta_mae_vs_0088_base"] > 0.0
    assert row["season_no_regression_passed"] is False
    assert row["hardened_gate_passed"] is False
