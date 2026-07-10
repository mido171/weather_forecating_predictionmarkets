from __future__ import annotations

import math

import pandas as pd

from hkg_tmax.research.evidence import compute_date_gaps, make_row_hash, score_predictions


def test_row_hash_is_stable_and_sensitive_to_key_parts() -> None:
    first = make_row_hash("official", "2023-01-01", "rss_archive", "v1")
    second = make_row_hash("official", "2023-01-01", "rss_archive", "v1")
    changed = make_row_hash("official", "2023-01-01", "press_archive", "v1")

    assert first == second
    assert first != changed
    assert len(first) == 64


def test_score_predictions_reports_standard_error_metrics() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "target_tmax_c": [20.0, 22.0, 24.0],
            "prediction_c": [21.0, 21.0, 27.0],
        }
    )

    scored = score_predictions(frame, "prediction_c")

    assert scored["n"] == 3
    assert math.isclose(float(scored["mae"]), 5.0 / 3.0)
    assert math.isclose(float(scored["bias"]), 1.0)
    assert scored["first_date"] == "2023-01-01"
    assert scored["last_date"] == "2023-01-03"


def test_compute_date_gaps_finds_calendar_holes() -> None:
    gaps = compute_date_gaps(
        pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-05"]),
        "F-SAMPLE",
    )

    assert len(gaps) == 1
    assert gaps.iloc[0]["frame_id"] == "F-SAMPLE"
    assert gaps.iloc[0]["gap_start"] == "2023-01-03"
    assert gaps.iloc[0]["gap_end"] == "2023-01-04"
    assert int(gaps.iloc[0]["missing_days"]) == 2
