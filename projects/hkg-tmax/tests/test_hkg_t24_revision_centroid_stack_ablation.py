from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_revision_centroid_stack_ablation import (
    expert_groups,
    strict_past_expert_stack,
)


def test_strict_past_stack_excludes_current_label_from_selection() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "forecast_source_family": ["press", "press"],
            "target_tmax_c": [10.0, 100.0],
            "official_raw": [10.0, 10.0],
            "expert_a": [20.0, 100.0],
            "expert_b": [10.0, 10.0],
        }
    )

    predictions = strict_past_expert_stack(
        frame,
        experts=["expert_a", "expert_b"],
        mode="best",
        same_source=False,
        min_history=1,
    )

    assert predictions.loc[1, "selected_expert"] == "expert_b"
    assert predictions.loc[1, "expert_prediction_c"] == 10.0


def test_strict_past_stack_excludes_same_date_other_source_label() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "forecast_source_family": ["press", "press", "rss"],
            "target_tmax_c": [10.0, 100.0, 100.0],
            "official_raw": [10.0, 10.0, 10.0],
            "expert_a": [20.0, 100.0, 100.0],
            "expert_b": [10.0, 10.0, 10.0],
        }
    )

    predictions = strict_past_expert_stack(
        frame,
        experts=["expert_a", "expert_b"],
        mode="best",
        same_source=False,
        min_history=1,
    )

    assert predictions.loc[2, "selected_expert"] == "expert_b"
    assert predictions.loc[2, "expert_prediction_c"] == 10.0


def test_strict_past_stack_same_source_isolates_prior_history() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "forecast_source_family": ["press", "rss", "rss"],
            "target_tmax_c": [10.0, 10.0, 10.0],
            "official_raw": [10.0, 10.0, 10.0],
            "expert_a": [20.0, 10.0, 10.0],
            "expert_b": [10.0, 20.0, 20.0],
        }
    )

    predictions = strict_past_expert_stack(
        frame,
        experts=["expert_a", "expert_b"],
        mode="best",
        same_source=True,
        min_history=1,
    )

    assert predictions.loc[2, "selected_expert"] == "expert_a"
    assert predictions.loc[2, "expert_prediction_c"] == 10.0


def test_positive_lift_stack_falls_back_when_no_expert_beats_official_prior() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "forecast_source_family": ["press", "press"],
            "target_tmax_c": [10.0, 100.0],
            "official_raw": [10.0, 10.0],
            "expert_a": [20.0, 100.0],
        }
    )

    predictions = strict_past_expert_stack(
        frame,
        experts=["expert_a"],
        mode="positive_lift_top3",
        same_source=False,
        min_history=1,
    )

    assert predictions.loc[1, "selected_expert"] == "official_raw_fallback"
    assert predictions.loc[1, "expert_prediction_c"] == 10.0


def test_expert_groups_include_official_in_each_ablation() -> None:
    mapping = pd.DataFrame(
        {
            "expert_id": [
                "official_raw",
                "s1",
                "c1",
                "r1",
                "b33",
                "0034_prior_blend_01_cluster_centroid_blend_inverse_mae_all_prior",
                "b35",
            ],
            "source_group": [
                "official",
                "0033_smooth_specialist",
                "0034_centroid_specialist",
                "0035_revision_specialist",
                "0033_prior_blend",
                "0034_prior_blend",
                "0035_prior_blend",
            ],
        }
    )

    groups = expert_groups(mapping)

    assert all(group[0] == "official_raw" for group in groups.values())
    assert groups["specialists_all"] == ["official_raw", "s1", "c1", "r1"]
    assert groups["current_0034_blend_only"] == [
        "official_raw",
        "0034_prior_blend_01_cluster_centroid_blend_inverse_mae_all_prior",
    ]
