from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_r07_transition_detection import (
    active_cols,
    add_transition_scores,
    long_report,
)


def test_add_transition_scores_uses_fixed_scale_components() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "target_tmax_c": [20.0, 17.0, 18.0],
            "hko_mslp_cutoff_hpa": [1010.0, 1015.0, 1018.0],
            "hko_mslp_3h_change_to_cutoff_hpa": [1.5, 3.0, 0.0],
            "hko_temp_change_360m_to_latest_c": [-2.0, -1.0, 1.0],
            "hko_dew_point_change_6h_c": [-2.0, -0.5, 0.5],
            "network_median_wind_speed_3h_change_kmh": [10.0, 0.0, -5.0],
            "hko_dewpoint_depression_c": [5.0, 8.0, 3.0],
        }
    )

    scored = add_transition_scores(frame)

    expected_first = 1.5 / 3.0 + 2.0 / 2.0 + 2.0 / 2.0 + 10.0 / 10.0
    assert scored.loc[0, "cold_surge_score"] == expected_first
    assert scored.loc[1, "target_tmax_change_1d_c"] == -3.0
    assert scored.loc[1, "aux_transition_label"] == 1.0


def test_active_cols_removes_constant_and_all_missing_training_columns() -> None:
    train = pd.DataFrame(
        {
            "usable": [1.0, 2.0, 3.0],
            "constant": [5.0, 5.0, 5.0],
            "all_missing": [np.nan, np.nan, np.nan],
        }
    )

    assert active_cols(train, ["usable", "constant", "all_missing", "missing"]) == ["usable"]


def test_r07_long_report_exceeds_required_experiment_narrative_length() -> None:
    report = long_report(
        {
            "champion": {
                "model_id": "r07_baseline_temp_calendar",
                "n": 911,
                "mae": 1.4723,
                "rmse": 1.8861,
                "bias": 0.0298,
                "crps_normal": 1.0512,
            },
            "oof_feasibility": {
                "status": "BLOCKED",
                "reason": "synthetic four-year OOF blocker",
            },
            "feature_min": "2020-07-02",
            "feature_max": "2023-12-31",
            "prediction_min": "2021-07-01",
            "prediction_max": "2023-12-31",
        }
    )

    assert len(report) >= 7500
    assert "wind direction" in report
