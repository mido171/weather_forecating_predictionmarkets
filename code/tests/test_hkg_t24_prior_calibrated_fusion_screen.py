from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_prior_calibrated_fusion_screen import (
    WEIGHT_GRID,
    FusionSpec,
    apply_fold_transfer_spec,
    apply_prior_weight_spec,
    blend_prediction,
    fixed_weight_stability,
    select_prior_weight,
)


def test_blend_prediction_applies_station_weight() -> None:
    frame = pd.DataFrame(
        {
            "official_family_prediction_c": [20.0],
            "station_family_prediction_c": [24.0],
        }
    )

    assert blend_prediction(frame, 0.25).tolist() == [21.0]


def test_prior_weight_selector_excludes_current_row() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "target_tmax_c": [24.0, 30.0],
            "official_family_prediction_c": [20.0, 20.0],
            "station_family_prediction_c": [24.0, 30.0],
            "forecast_source_family": ["rss", "rss"],
            "family_disagreement_c": [4.0, 10.0],
            "abs_family_disagreement_c": [4.0, 10.0],
            "active_member_count": [1, 1],
            "fold_id": ["fold", "fold"],
        }
    )
    spec = FusionSpec("prior", "prior_best_weight", 0.0, "global", 1, 0.0, 0.0)

    out = apply_prior_weight_spec(frame, spec)

    assert out["station_weight"].iloc[0] == 0.0
    assert out["station_weight"].iloc[1] == max(WEIGHT_GRID)


def test_select_prior_weight_softmax_stays_inside_grid_range() -> None:
    abs_sums = np.linspace(10.0, 1.0, len(WEIGHT_GRID))
    weight = select_prior_weight(abs_sums, 10, FusionSpec("soft", "prior_soft_weight", 0.0, "global", 1, 0.0, 0.05))

    assert 0.0 <= weight <= max(WEIGHT_GRID)


def test_fold_transfer_uses_only_earlier_folds() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2021-01-01"]),
            "target_tmax_c": [24.0, 24.0, 30.0],
            "official_family_prediction_c": [20.0, 20.0, 20.0],
            "station_family_prediction_c": [24.0, 24.0, 30.0],
            "forecast_source_family": ["press", "press", "rss"],
            "fold_id": ["fold1", "fold1", "fold2"],
        }
    )
    spec = FusionSpec("fold", "fold_prior_best_weight", 0.0, "global", 2, 0.0, 0.0)

    out = apply_fold_transfer_spec(frame, spec)

    assert out["station_weight"].iloc[0] == 0.0
    assert out["station_weight"].iloc[2] == max(WEIGHT_GRID)
    assert out["prior_count"].iloc[2] == 2


def test_fixed_weight_stability_ranks_best_weight_by_group() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "target_tmax_c": [24.0, 20.0],
            "official_family_prediction_c": [20.0, 20.0],
            "station_family_prediction_c": [24.0, 24.0],
            "forecast_source_family": ["a", "a"],
            "fold_id": ["fold", "fold"],
        }
    )

    stability = fixed_weight_stability(frame)
    best = stability[stability["group_name"].eq("all") & stability["rank_in_group"].eq(1.0)].iloc[0]

    assert best["station_weight"] in WEIGHT_GRID
