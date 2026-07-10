from __future__ import annotations

import math

import pandas as pd

from scripts.run_hkg_t24_0103_current_rss_continuation_without_blocked_sources import (
    archive_gap_audit,
    build_slice_scoreboard,
    score_subset,
)


def sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-05"]),
            "forecast_source_family": ["rss_archive", "rss_archive", "press_archive"],
            "season": ["DJF", "DJF", "DJF"],
            "frame_segment": ["current"] * 3,
            "era_bucket": ["rss_2021", "rss_2021", "press_2021"],
            "target_tmax_c": [20.0, 22.0, 24.0],
            "forecast_max_c": [21.0, 20.0, 23.0],
            "candidate_prediction_c": [20.5, 21.0, 23.5],
            "specialist_active": [True, False, True],
            "specialist_correction_c": [-0.5, 0.0, 0.5],
        }
    )


def test_score_subset_reports_candidate_delta_vs_official() -> None:
    scored = score_subset(sample_predictions(), slice_type="overall", slice_value="all")

    assert scored["n"] == 3
    assert math.isclose(float(scored["official_mae"]), 4.0 / 3.0)
    assert math.isclose(float(scored["candidate_mae"]), 2.0 / 3.0)
    assert math.isclose(float(scored["delta_mae_vs_official"]), -2.0 / 3.0)
    assert scored["active_correction_rows"] == 2


def test_archive_gap_audit_detects_missing_days_between_scored_dates() -> None:
    gaps = archive_gap_audit(sample_predictions())

    assert len(gaps) == 1
    assert gaps.iloc[0]["gap_start"] == "2021-01-03"
    assert gaps.iloc[0]["gap_end"] == "2021-01-04"
    assert gaps.iloc[0]["missing_days"] == 2


def test_build_slice_scoreboard_adds_rss_year_slice() -> None:
    scoreboard = build_slice_scoreboard(sample_predictions())

    rss_year = scoreboard[
        scoreboard["slice_type"].eq("rss_year") & scoreboard["slice_value"].eq("2021")
    ]

    assert len(rss_year) == 1
    assert int(rss_year.iloc[0]["n"]) == 2
