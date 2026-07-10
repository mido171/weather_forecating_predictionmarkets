from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_0095_mam_error_direction_split_lab import DirectionSplitSpec
from scripts.run_hkg_t24_0098_source_submonth_stable_cell_specialist import (
    SourceSubmonthCellSets,
    apply_source_submonth_specialist,
    cell_policy_allows,
    load_source_submonth_cell_sets,
    source_submonth_key,
)


def base_spec() -> DirectionSplitSpec:
    return DirectionSplitSpec(
        candidate_id="test",
        pair_name="feature_a__x__feature_b",
        feature_a="feature_a",
        feature_b="feature_b",
        group_a="target_memory",
        group_b="upper_air_ceiling",
        active_gate="mam_all",
        direction_mode="overforecast_only",
        min_history=1,
        direction_threshold_c=0.0,
        shrink_rows=0.0,
        correction_cap_c=10.0,
    )


def test_source_submonth_key_labels_mam_month_and_source_direction() -> None:
    assert source_submonth_key("rss_archive", pd.Timestamp("2023-03-15"), "overforecast") == (
        "rss_archive",
        "march",
        "overforecast",
    )


def test_load_source_submonth_cell_sets_splits_stable_and_damaging() -> None:
    frame = pd.DataFrame(
        [
            {
                "forecast_source_family": "rss_archive",
                "mam_submonth": "march",
                "prior_direction": "overforecast",
                "status": "stable_improving",
            },
            {
                "forecast_source_family": "press_archive",
                "mam_submonth": "may",
                "prior_direction": "underforecast",
                "status": "damaging",
            },
            {
                "forecast_source_family": "press_archive",
                "mam_submonth": "april",
                "prior_direction": "inactive",
                "status": "stable_improving",
            },
        ]
    )

    cells = load_source_submonth_cell_sets(frame)

    assert cells.stable == {("rss_archive", "march", "overforecast")}
    assert cells.damaging == {("press_archive", "may", "underforecast")}


def test_cell_policy_allows_expected_boolean_logic() -> None:
    assert cell_policy_allows("bucket_only", bucket_allowed=True, source_submonth_allowed=False)
    assert not cell_policy_allows("bucket_only", bucket_allowed=False, source_submonth_allowed=True)
    assert cell_policy_allows("source_submonth_only", bucket_allowed=False, source_submonth_allowed=True)
    assert cell_policy_allows("bucket_or_source_submonth", bucket_allowed=True, source_submonth_allowed=False)
    assert cell_policy_allows("bucket_or_source_submonth", bucket_allowed=False, source_submonth_allowed=True)
    assert cell_policy_allows("bucket_and_source_submonth", bucket_allowed=True, source_submonth_allowed=True)
    assert not cell_policy_allows("bucket_and_source_submonth", bucket_allowed=True, source_submonth_allowed=False)
    with pytest.raises(ValueError, match="Unsupported 0098 cell policy"):
        cell_policy_allows("bad", bucket_allowed=True, source_submonth_allowed=True)


def test_source_submonth_specialist_uses_prior_rows_and_stable_source_month_only() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-05-01", "2020-05-02", "2020-05-03"]),
            "forecast_source_family": ["press_archive", "press_archive", "rss_archive"],
            "season": ["MAM", "MAM", "MAM"],
            "frame_segment": ["current_0081_frame"] * 3,
            "era_bucket": ["a"] * 3,
            "target_tmax_c": [18.0, 19.0, 18.0],
            "forecast_max_c": [20.0, 21.0, 21.0],
            "candidate_prediction_c": [20.0, 21.0, 21.0],
            "base_residual_c": [2.0, 2.0, 3.0],
            "feature_a__x__feature_b__bucket": [0.0, 0.0, 0.0],
        }
    )

    prediction, diagnostics = apply_source_submonth_specialist(
        frame,
        base_spec(),
        cell_policy="source_submonth_only",
        stable_bucket_cells=set(),
        damaging_bucket_cells=set(),
        source_submonth_cells=SourceSubmonthCellSets(
            stable={("press_archive", "may", "overforecast")},
            damaging=set(),
        ),
    )

    assert prediction.tolist() == [20.0, 19.0, 21.0]
    assert diagnostics["prior_rows"].tolist() == [0, 1, 2]
    assert diagnostics["source_submonth_stable_allowed"].tolist() == [False, True, False]
    assert diagnostics["specialist_active"].tolist() == [False, True, False]


def test_source_submonth_specialist_damaging_source_cell_blocks_correction() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-05-01", "2020-05-02"]),
            "forecast_source_family": ["press_archive", "press_archive"],
            "season": ["MAM", "MAM"],
            "frame_segment": ["current_0081_frame", "current_0081_frame"],
            "era_bucket": ["a", "a"],
            "target_tmax_c": [18.0, 19.0],
            "forecast_max_c": [20.0, 21.0],
            "candidate_prediction_c": [20.0, 21.0],
            "base_residual_c": [2.0, 2.0],
            "feature_a__x__feature_b__bucket": [0.0, 0.0],
        }
    )

    prediction, diagnostics = apply_source_submonth_specialist(
        frame,
        base_spec(),
        cell_policy="source_submonth_only",
        stable_bucket_cells=set(),
        damaging_bucket_cells=set(),
        source_submonth_cells=SourceSubmonthCellSets(
            stable={("press_archive", "may", "overforecast")},
            damaging={("press_archive", "may", "overforecast")},
        ),
    )

    assert prediction.tolist() == [20.0, 21.0]
    assert diagnostics["source_submonth_stable_allowed"].tolist() == [False, True]
    assert diagnostics["source_submonth_damaging_blocked"].tolist() == [False, True]
    assert diagnostics["specialist_active"].tolist() == [False, False]
