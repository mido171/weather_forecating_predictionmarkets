from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_hkg_t24_0101_stable_mam_cell_feature_specialists import (
    FeatureSpecialistSpec,
    apply_feature_specialist,
    assign_bucket_from_edges,
    fixed_quantile_edges,
    gate_mask_for_scope,
    select_candidate_features,
)


def test_fixed_quantile_edges_uses_enough_history_and_unique_edges() -> None:
    values = pd.Series(range(120))

    edges = fixed_quantile_edges(values, bucket_count=3)

    assert len(edges) == 2
    assert edges[0] < edges[1]


def test_fixed_quantile_edges_rejects_tiny_or_flat_history() -> None:
    assert fixed_quantile_edges(pd.Series([1.0, 2.0, 3.0]), bucket_count=3) == ()
    assert fixed_quantile_edges(pd.Series([1.0] * 100), bucket_count=3) == ()


def test_assign_bucket_from_edges_labels_finite_values_and_missing() -> None:
    buckets = assign_bucket_from_edges(pd.Series([0.0, 2.0, 5.0, np.nan]), (1.0, 4.0))

    assert buckets.tolist() == [0, 1, 2, -1]


def test_gate_mask_for_scope_uses_requested_boolean_column() -> None:
    frame = pd.DataFrame(
        {
            "agreement_row": [True, False, True],
            "specialist_active_row": [False, True, True],
        }
    )

    assert gate_mask_for_scope(frame, "agreement").tolist() == [True, False, True]
    assert gate_mask_for_scope(frame, "specialist_active").tolist() == [False, True, True]
    with pytest.raises(ValueError, match="Unsupported 0101 gate scope"):
        gate_mask_for_scope(frame, "bad")


def test_select_candidate_features_limits_to_future_allowed_target_memory_and_station() -> None:
    atlas = pd.DataFrame(
        [
            {
                "feature": "target_roll14_std_lag7_c",
                "family": "target_memory",
                "diagnostic_score": 3.0,
                "allowed_for_future_walkforward": True,
            },
            {
                "feature": "isd_dew_point_mean_c_change_1d",
                "family": "isd_station_network",
                "diagnostic_score": 2.0,
                "allowed_for_future_walkforward": True,
            },
            {
                "feature": "igra_hgt_1000hpa_m",
                "family": "upper_air",
                "diagnostic_score": 4.0,
                "allowed_for_future_walkforward": False,
            },
        ]
    )

    selected = select_candidate_features(atlas)

    assert selected["feature"].tolist() == [
        "target_roll14_std_lag7_c",
        "isd_dew_point_mean_c_change_1d",
    ]


def test_apply_feature_specialist_updates_after_current_date_only() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-05-01", "2020-05-02", "2020-05-03"]),
            "forecast_source_family": ["rss_archive", "rss_archive", "rss_archive"],
            "season": ["MAM", "MAM", "MAM"],
            "frame_segment": ["current_0081_frame"] * 3,
            "era_bucket": ["rss"] * 3,
            "agreement_row": [True, True, True],
            "specialist_active_row": [True, True, True],
            "target_tmax_c": [18.0, 19.0, 20.0],
            "best_0099_prediction_c": [20.0, 21.0, 22.0],
            "best_0099_error_c": [2.0, 2.0, 2.0],
            "feature_a": [0.5, 0.5, 0.5],
        }
    )
    spec = FeatureSpecialistSpec(
        candidate_id="test",
        feature="feature_a",
        family="target_memory",
        bucket_count=2,
        bin_edges=(1.0,),
        gate_scope="agreement",
        min_history=1,
        shrink_rows=0.0,
        correction_cap_c=10.0,
        min_abs_prior_mean_c=0.0,
    )

    prediction, diagnostics = apply_feature_specialist(frame, spec)

    assert prediction.tolist() == [20.0, 19.0, 20.0]
    assert diagnostics["prior_rows"].tolist() == [0, 1, 2]
    assert diagnostics["specialist_active"].tolist() == [False, True, True]
