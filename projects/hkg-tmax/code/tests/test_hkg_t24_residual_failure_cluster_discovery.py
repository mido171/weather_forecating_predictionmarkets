from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_residual_failure_cluster_discovery import (
    ArchetypeCondition,
    ArchetypeSpec,
    build_archetype_specs,
    condition_prior_mask,
    past_only_archetype_predictions,
    simple_kmeans,
)


def test_simple_kmeans_deterministically_separates_two_groups() -> None:
    matrix = pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2, 9.8, 10.0, 10.1],
            "y": [0.0, 0.2, 0.1, 10.2, 10.0, 9.9],
        }
    ).to_numpy()

    labels = simple_kmeans(matrix, 2)

    assert labels.tolist() == simple_kmeans(matrix, 2).tolist()
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[-1]


def test_condition_prior_mask_uses_prior_threshold_and_current_match() -> None:
    ordered = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    base_prior = pd.Series([True, True, True, True, False]).to_numpy()
    current = ordered.iloc[4]

    current_match, prior_mask, threshold = condition_prior_mask(
        ordered,
        base_prior,
        current,
        ArchetypeCondition("feature", "high", 0.75),
    )

    assert current_match is True
    assert round(threshold, 2) == 3.25
    assert prior_mask.tolist() == [False, False, False, True, False]


def test_past_only_archetype_predictions_excludes_current_label() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "forecast_source_family": ["press", "press"],
            "primary_regime": ["default", "default"],
            "target_tmax_c": [10.0, 100.0],
            "official_raw": [10.0, 10.0],
            "feature": [1.0, 2.0],
        }
    )
    spec = ArchetypeSpec(
        name="current_label_guard",
        anchor_col="official_raw",
        conditions=(ArchetypeCondition("feature", "high", 0.50),),
        same_source=False,
        min_history=1,
        min_match_rows=1,
        shrinkage=0.0,
        correction_clip_c=100.0,
    )

    predictions = past_only_archetype_predictions(frame, spec)

    assert bool(predictions.loc[1, "used_archetype_correction"]) is True
    assert predictions.loc[1, "candidate_prediction_c"] == 10.0


def test_past_only_archetype_predictions_same_source_isolates_history() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "forecast_source_family": ["press", "rss", "rss"],
            "primary_regime": ["default", "default", "default"],
            "target_tmax_c": [10.0, 30.0, 10.0],
            "official_raw": [10.0, 10.0, 10.0],
            "anchor": [10.0, 10.0, 10.0],
            "feature": [1.0, 1.0, 1.0],
        }
    )
    spec = ArchetypeSpec(
        name="same_source_guard",
        anchor_col="anchor",
        conditions=(ArchetypeCondition("feature", "flag"),),
        same_source=True,
        min_history=1,
        min_match_rows=1,
        shrinkage=0.0,
        correction_clip_c=100.0,
    )

    predictions = past_only_archetype_predictions(frame, spec)

    assert bool(predictions.loc[2, "used_archetype_correction"]) is True
    assert predictions.loc[2, "candidate_prediction_c"] == 30.0


def test_build_archetype_specs_filters_missing_features() -> None:
    rows = 700
    frame = pd.DataFrame(
        {
            "official_raw": [25.0] * rows,
            "prediction_0018_c": [25.0] * rows,
            "prediction_0026_c": [25.0] * rows,
            "forecast_max_c": [20.0 + (index % 20) for index in range(rows)],
            "text_any_rain": [index % 2 for index in range(rows)],
        }
    )

    specs = build_archetype_specs(frame)

    assert specs
    assert {spec.name for spec in specs} == {"rain_hot_forecast"}
