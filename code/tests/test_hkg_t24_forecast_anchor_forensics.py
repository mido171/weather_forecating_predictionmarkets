from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_forecast_anchor_forensics import (
    aggregate_error_by,
    bucket_series,
    safe_corr,
    source_score,
)


def test_bucket_series_uses_insufficient_guard_for_tiny_inputs() -> None:
    labels = bucket_series(pd.Series([1.0, 2.0, 3.0]), 5, "q")

    assert labels.to_list() == ["insufficient", "insufficient", "insufficient"]


def test_aggregate_error_by_reports_official_bias_and_mae() -> None:
    frame = pd.DataFrame(
        {
            "forecast_source_family": ["rss", "rss", "press"],
            "season": ["JJA", "JJA", "DJF"],
            "target_date": pd.to_datetime(["2021-01-01", "2021-01-02", "2001-01-01"]),
            "official_error_c": [1.0, -3.0, 2.0],
        }
    )

    table = aggregate_error_by(frame, ["forecast_source_family", "season"])
    rss = table[table["forecast_source_family"].eq("rss")].iloc[0]

    assert float(rss["mae"]) == 2.0
    assert float(rss["bias"]) == -1.0


def test_source_score_scores_each_forecast_source() -> None:
    frame = pd.DataFrame(
        {
            "forecast_source_family": ["rss", "rss", "press"],
            "target_date": pd.to_datetime(["2021-01-01", "2021-01-02", "2001-01-01"]),
            "target_tmax_c": [20.0, 22.0, 25.0],
            "forecast_max_c": [21.0, 20.0, 26.0],
        }
    )

    table = source_score(frame)

    assert set(table["forecast_source_family"]) == {"rss", "press"}
    assert float(table.loc[table["forecast_source_family"].eq("rss"), "mae"].iloc[0]) == 1.5


def test_safe_corr_allows_binary_response_for_underprediction_diagnostics() -> None:
    corr = safe_corr(
        pd.Series(range(300), dtype=float),
        pd.Series([0, 1] * 150, dtype=float),
        min_rows=100,
    )

    assert pd.notna(corr)
