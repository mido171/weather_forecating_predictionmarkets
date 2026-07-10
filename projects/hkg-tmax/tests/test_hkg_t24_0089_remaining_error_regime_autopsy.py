from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_0089_remaining_error_regime_autopsy import (
    error_regime_summary,
    high_low_feature_contrasts,
    specialist_action,
)


def make_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
            "forecast_source_family": ["press_archive", "press_archive", "rss_archive", "rss_archive"],
            "season": ["DJF", "DJF", "DJF", "DJF"],
            "frame_segment": ["old", "old", "new", "new"],
            "era_bucket": ["a", "a", "b", "b"],
            "candidate_error_c": [0.2, -0.4, 2.0, -2.4],
            "candidate_abs_error_c": [0.2, 0.4, 2.0, 2.4],
            "raw_abs_error_c": [0.3, 0.5, 2.2, 2.6],
        }
    )


def test_error_regime_summary_ranks_high_mae_groups() -> None:
    summary = error_regime_summary(make_prediction_frame())
    source_rows = summary[summary["grouping"].eq("forecast_source_family")]

    top_source = source_rows.sort_values("mae", ascending=False).iloc[0]

    assert top_source["group_value"] == "rss_archive"
    assert top_source["mae"] == 2.2


def test_high_low_feature_contrasts_detects_tail_separator() -> None:
    frame = pd.DataFrame(
        {
            "candidate_abs_error_c": [float(value) for value in range(20)],
            "candidate_error_c": [float(value) for value in range(20)],
            "remaining_improvement_vs_raw_c": [0.0] * 20,
            "isd_tail_separator": [float(value) for value in range(20)],
            "flat_feature": [1.0] * 20,
        }
    )

    contrasts = high_low_feature_contrasts(
        frame,
        ["isd_tail_separator", "flat_feature"],
        high_quantile=0.75,
        low_quantile=0.25,
        min_feature_rows=10,
        min_tail_rows=2,
    )

    assert contrasts.iloc[0]["feature"] == "isd_tail_separator"
    assert contrasts.iloc[0]["family"] == "isd_station_network"
    assert contrasts.iloc[0]["standardized_high_low_diff"] > 1.0
    assert contrasts.iloc[0]["corr_abs_error"] > 0.99


def test_specialist_action_is_family_specific() -> None:
    action = specialist_action("ua_layer_925_850_ceiling_minus_isd_temp_c", "upper_air")

    assert "boundary-layer ceiling specialist" in action
