from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_router_simplification_or_archive_refresh_decision import (
    best_balanced_router,
    decision_branch,
    inclusive_calendar_days,
    press_archive_gap_summary,
    unique_target_day_coverage,
)


def test_inclusive_calendar_days_counts_both_endpoints() -> None:
    assert inclusive_calendar_days("2020-01-01", "2020-01-01") == 1
    assert inclusive_calendar_days("2020-01-01", "2020-01-03") == 3


def test_unique_target_day_coverage_reports_missing_calendar_days() -> None:
    predictions = pd.DataFrame({"target_date": ["2020-01-01", "2020-01-03", "2020-01-03"]})

    coverage = unique_target_day_coverage(predictions)

    assert coverage["unique_target_days"] == 2
    assert coverage["expected_calendar_days"] == 3
    assert coverage["missing_calendar_days"] == 1
    assert coverage["continuity_ratio"] == 2 / 3


def test_press_archive_gap_summary_counts_raw_and_zero_years() -> None:
    summary = press_archive_gap_summary(
        {
            "candidate_first_year": 2000,
            "candidate_last_year": 2004,
            "raw_detail_years": [2000, 2001],
            "zero_raw_detail_years": [2002, 2003, 2004],
            "scoreable_forecast_day_rows": 123,
        }
    )

    assert summary["candidate_year_count"] == 5
    assert summary["raw_detail_year_count"] == 2
    assert summary["raw_detail_year_coverage_ratio"] == 0.4
    assert summary["zero_raw_detail_year_count"] == 3


def test_best_balanced_router_prefers_all_segment_full_mae_over_late_only() -> None:
    robustness = pd.DataFrame(
        [
            {
                "candidate_id": "late_only",
                "mae": 0.99,
                "late_eval_mae": 0.97,
                "segments_scored": 10,
                "segments_beating_anchor": 8,
                "worst_delta_vs_anchor": 0.01,
                "source_mae_spread": 0.02,
            },
            {
                "candidate_id": "balanced",
                "mae": 0.98,
                "late_eval_mae": 0.975,
                "segments_scored": 10,
                "segments_beating_anchor": 10,
                "worst_delta_vs_anchor": -0.001,
                "source_mae_spread": 0.01,
            },
        ]
    )

    selected = best_balanced_router(robustness)

    assert selected["candidate_id"] == "balanced"


def test_decision_branch_prefers_archive_refresh_for_low_continuity_tiny_gain() -> None:
    decision, reason = decision_branch(
        continuity_ratio=0.30,
        scored_rows=2670,
        robust_full_mae_gain_vs_0041=0.0002,
        robust_late_mae_gain_vs_0041_best_late=-0.0001,
        robust_candidate_beats_all_segments=True,
    )

    assert decision == "archive_refresh_first_freeze_router_benchmark"
    assert "marginal" in reason


def test_decision_branch_allows_router_first_when_continuous_and_stable() -> None:
    decision, _reason = decision_branch(
        continuity_ratio=0.95,
        scored_rows=9000,
        robust_full_mae_gain_vs_0041=0.003,
        robust_late_mae_gain_vs_0041_best_late=0.003,
        robust_candidate_beats_all_segments=True,
    )

    assert decision == "simplify_router_first"
