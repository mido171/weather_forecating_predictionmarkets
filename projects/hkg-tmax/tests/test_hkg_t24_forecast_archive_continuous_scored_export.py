from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_forecast_archive_continuous_scored_export import (
    CONFIRMATION_START,
    contiguous_date_segments,
    continuity_summary,
    gap_rows_from_segments,
    season_from_month,
)


def test_contiguous_date_segments_splits_gaps() -> None:
    segments = contiguous_date_segments(
        pd.Series(["2005-01-01", "2005-01-02", "2005-01-05", "2005-01-06", "2005-01-10"])
    )

    assert segments["first_date"].to_list() == ["2005-01-01", "2005-01-05", "2005-01-10"]
    assert segments["last_date"].to_list() == ["2005-01-02", "2005-01-06", "2005-01-10"]
    assert segments["observed_days"].to_list() == [2, 2, 1]

    gaps = gap_rows_from_segments(segments)
    assert gaps["missing_start"].to_list() == ["2005-01-03", "2005-01-07"]
    assert gaps["missing_end"].to_list() == ["2005-01-04", "2005-01-09"]
    assert gaps["missing_days"].to_list() == [2, 3]


def test_continuity_summary_counts_unique_target_dates_by_source() -> None:
    frame = pd.DataFrame(
        {
            "forecast_source_family": ["press", "press", "press", "rss"],
            "target_date": ["2005-01-01", "2005-01-01", "2005-01-03", "2021-01-01"],
        }
    )

    summary = continuity_summary(frame)

    press = summary[summary["forecast_source_family"].eq("press")].iloc[0]
    rss = summary[summary["forecast_source_family"].eq("rss")].iloc[0]
    assert int(press["observed_target_days"]) == 2
    assert int(press["span_days"]) == 3
    assert int(press["missing_days_inside_span"]) == 1
    assert float(press["continuity_ratio"]) == 2 / 3
    assert int(rss["segment_count"]) == 1


def test_confirmation_start_constant_remains_sealed_period_boundary() -> None:
    assert str(CONFIRMATION_START.date()) == "2024-01-01"


def test_season_from_month_handles_standard_hko_seasons() -> None:
    assert season_from_month(1) == "DJF"
    assert season_from_month("4") == "MAM"
    assert season_from_month(8) == "JJA"
    assert season_from_month(11) == "SON"
    assert season_from_month(None) == ""
