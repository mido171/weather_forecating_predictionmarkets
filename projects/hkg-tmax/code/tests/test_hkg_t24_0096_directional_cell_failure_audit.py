from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_0096_directional_cell_failure_audit import (
    build_analysis_frame,
    cell_status,
    mam_submonth,
    summarize_groups,
)


def test_mam_submonth_labels_only_march_april_may() -> None:
    assert mam_submonth(pd.Timestamp("2020-03-15")) == "march"
    assert mam_submonth(pd.Timestamp("2020-04-15")) == "april"
    assert mam_submonth(pd.Timestamp("2020-05-15")) == "may"
    assert mam_submonth(pd.Timestamp("2020-06-01")) == "non_mam"


def test_cell_status_requires_rows_and_meaningful_delta() -> None:
    assert cell_status(-0.01, 20) == "stable_improving"
    assert cell_status(0.01, 20) == "damaging"
    assert cell_status(0.001, 20) == "neutral"
    assert cell_status(-0.01, 19) == "too_sparse"


def test_build_analysis_frame_computes_row_level_improvement() -> None:
    top_0095 = pd.DataFrame(
        {
            "target_date": ["2020-03-01"],
            "forecast_source_family": ["press_archive"],
            "target_tmax_c": [20.0],
            "forecast_max_c": [22.0],
            "season": ["MAM"],
            "frame_segment": ["current_0081_frame"],
            "era_bucket": ["a"],
            "candidate_id": ["c"],
            "candidate_prediction_c": [21.0],
            "candidate_error_c": [1.0],
        }
    )
    top_0094 = pd.DataFrame(
        {
            "target_date": ["2020-03-01"],
            "forecast_source_family": ["press_archive"],
            "candidate_prediction_c": [22.0],
            "candidate_error_c": [2.0],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "target_date": ["2020-03-01"],
            "forecast_source_family": ["press_archive"],
            "pair_name": ["pair"],
            "active_gate": ["mam_all"],
            "direction_mode": ["overforecast_only"],
            "pair_bucket": [12.0],
            "gate_active_row": [True],
            "prior_rows": [50],
            "prior_mean_residual_c": [0.2],
            "prior_direction": ["overforecast"],
            "specialist_active": [True],
            "specialist_correction_c": [1.0],
        }
    )
    for frame in (top_0095, top_0094, diagnostics):
        frame["target_date"] = pd.to_datetime(frame["target_date"])

    out = build_analysis_frame(top_0095, top_0094, diagnostics)

    assert out["base_abs_error_c"].iloc[0] == 2.0
    assert out["candidate_abs_error_c"].iloc[0] == 1.0
    assert out["abs_error_improvement_c"].iloc[0] == 1.0
    assert out["mam_submonth"].iloc[0] == "march"
    assert out["pair_bucket_label"].iloc[0] == "bucket_12"


def test_summarize_groups_reports_candidate_minus_base_delta() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-03-01", periods=20),
            "gate_active_row": [True] * 20,
            "specialist_active": [True] * 20,
            "base_abs_error_c": [2.0] * 20,
            "candidate_abs_error_c": [1.0] * 20,
            "abs_error_improvement_c": [1.0] * 20,
            "specialist_correction_c": [0.2] * 20,
            "prior_rows": [40] * 20,
            "prior_direction": ["overforecast"] * 20,
        }
    )

    summary = summarize_groups(frame, ["prior_direction"])

    assert summary["delta_mae_candidate_minus_base"].iloc[0] == -1.0
    assert summary["status"].iloc[0] == "stable_improving"
