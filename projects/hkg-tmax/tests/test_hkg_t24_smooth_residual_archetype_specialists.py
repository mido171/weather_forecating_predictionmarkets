from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_residual_failure_cluster_discovery import ArchetypeCondition
from scripts.run_hkg_t24_smooth_residual_archetype_specialists import (
    SmoothArchetypeSpec,
    build_smooth_archetype_specs,
    past_only_smooth_archetype_predictions,
    smooth_half_life_residual_correction,
)


def test_smooth_half_life_residual_correction_weights_recent_neighbors_more() -> None:
    x_prior = np.array([[1.0], [1.0]])
    residual_prior = np.array([10.0, 0.0])
    ages = np.array([1000.0, 1.0])
    current = np.array([1.0])

    no_decay = smooth_half_life_residual_correction(
        x_prior,
        residual_prior,
        ages,
        current,
        k_neighbors=2,
        shrinkage=0.0,
        correction_clip_c=20.0,
        half_life_days=None,
        min_local_mae_improvement_c=0.0,
    )
    with_decay = smooth_half_life_residual_correction(
        x_prior,
        residual_prior,
        ages,
        current,
        k_neighbors=2,
        shrinkage=0.0,
        correction_clip_c=20.0,
        half_life_days=10.0,
        min_local_mae_improvement_c=0.0,
    )

    assert no_decay.correction == 5.0
    assert 0.0 <= with_decay.correction < 1.0


def test_smooth_do_no_harm_gate_blocks_non_improving_local_correction() -> None:
    result = smooth_half_life_residual_correction(
        np.array([[1.0], [1.0], [1.0], [1.0]]),
        np.array([1.0, -1.0, 1.0, -1.0]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.0]),
        k_neighbors=4,
        shrinkage=0.0,
        correction_clip_c=20.0,
        half_life_days=None,
        min_local_mae_improvement_c=0.01,
    )

    assert result.gate_passed is False
    assert result.correction == 0.0


def test_past_only_smooth_archetype_predictions_excludes_current_label() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "forecast_source_family": ["press", "press"],
            "primary_regime": ["default", "default"],
            "target_tmax_c": [10.0, 100.0],
            "official_raw": [10.0, 10.0],
            "prediction_0018_c": [10.0, 10.0],
            "feature": [1.0, 2.0],
        }
    )
    spec = SmoothArchetypeSpec(
        family_name="guard",
        anchor_col="prediction_0018_c",
        conditions=(ArchetypeCondition("feature", "high", 0.50),),
        features=("feature",),
        k_neighbors=1,
        same_source=False,
        half_life_days=None,
        min_history=1,
        min_match_rows=1,
        shrinkage=0.0,
        correction_clip_c=100.0,
    )

    predictions = past_only_smooth_archetype_predictions(frame, spec)

    assert predictions.loc[1, "candidate_prediction_c"] == 10.0


def test_past_only_smooth_archetype_predictions_same_source_isolates_history() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "forecast_source_family": ["press", "rss", "rss"],
            "primary_regime": ["default", "default", "default"],
            "target_tmax_c": [10.0, 30.0, 10.0],
            "official_raw": [10.0, 10.0, 10.0],
            "prediction_0018_c": [10.0, 10.0, 10.0],
            "feature": [1.0, 1.0, 1.0],
        }
    )
    spec = SmoothArchetypeSpec(
        family_name="same_source",
        anchor_col="prediction_0018_c",
        conditions=(ArchetypeCondition("feature", "flag"),),
        features=("feature",),
        k_neighbors=1,
        same_source=True,
        half_life_days=None,
        min_history=1,
        min_match_rows=1,
        shrinkage=0.0,
        correction_clip_c=100.0,
    )

    predictions = past_only_smooth_archetype_predictions(frame, spec)

    assert bool(predictions.loc[2, "do_no_harm_gate_passed"]) is True
    assert predictions.loc[2, "candidate_prediction_c"] == 30.0


def test_build_smooth_archetype_specs_filters_missing_families() -> None:
    rows = 350
    frame = pd.DataFrame(
        {
            "prediction_0018_c": [25.0] * rows,
            "prediction_0026_c": [25.0] * rows,
            "isd_dew_point_mean_c_change_1d": [index % 20 for index in range(rows)],
            "isd_temp_dewpoint_spread_mean_c": [20 - (index % 20) for index in range(rows)],
            "rh_min_pct": [50 + (index % 30) for index in range(rows)],
            "rh_max_pct": [70 + (index % 20) for index in range(rows)],
            "forecast_max_c": [25 + (index % 8) for index in range(rows)],
            "forecast_min_c": [20 + (index % 5) for index in range(rows)],
            "forecast_range_c": [5.0] * rows,
            "month": [(index % 12) + 1 for index in range(rows)],
            "monsoon_phase_code": [index % 4 for index in range(rows)],
            "text_any_rain": [index % 2 for index in range(rows)],
        }
    )

    specs = build_smooth_archetype_specs(frame)

    assert specs
    assert {spec.family_name for spec in specs} == {"moisture_surge"}
