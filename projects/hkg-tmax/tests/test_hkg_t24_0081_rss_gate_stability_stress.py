from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_0081_rss_gate_stability_stress import (
    RssGateStressSpec,
    apply_rss_gate,
    rss_start_mask,
)


def make_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["base", "base", "base", "base"],
            "target_date": pd.to_datetime(["2004-01-01", "2021-06-01", "2022-01-01", "2023-01-01"]),
            "current_target_tmax_c": [20.0, 21.0, 22.0, 23.0],
            "forecast_source_family": ["press_archive", "rss_archive", "rss_archive", "rss_archive"],
            "fold_id": ["fold_2000_2005", "fold_2021", "fold_2022", "fold_2023"],
            "row_index": [0, 1, 2, 3],
            "m0075_prediction_c": [19.0, 20.0, 21.0, 22.0],
            "m0078_prediction_c": [19.5, 20.5, 21.5, 22.5],
            "candidate_prediction_c": [19.7, 20.7, 21.7, 22.7],
            "guard_active": [True, True, True, True],
            "selected_families": ["a;b", "a;b", "a;b", "a;b"],
            "selected_candidates": ["x;y", "x;y", "x;y", "x;y"],
        }
    )


def test_rss_start_mask_excludes_press_and_pre_start_rss() -> None:
    mask = rss_start_mask(make_predictions(), "2022-01-01")

    assert mask.tolist() == [False, False, True, True]


def test_apply_rss_gate_falls_back_to_0078_before_start() -> None:
    spec = RssGateStressSpec(
        candidate_id="test",
        base_0079_candidate_id="base",
        rss_start_date="2022-01-01",
        min_changed_rows=1,
        min_changed_years=1,
        min_changed_rows_per_year=1,
    )

    out = apply_rss_gate(make_predictions(), spec)

    assert out["candidate_prediction_c"].tolist() == [19.5, 20.5, 21.7, 22.7]
    assert out["changed_from_0078"].tolist() == [False, False, True, True]
