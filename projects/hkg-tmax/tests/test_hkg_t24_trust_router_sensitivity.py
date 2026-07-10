from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_trust_router_sensitivity import (
    build_sensitivity_specs,
    family_variant_map,
    robustness_summary,
    segment_scoreboard,
    sensitivity_candidate_id,
    specs_from_catalog,
)


def family_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "family_name": [
                "official_raw",
                "anchor_0038_c",
                "hard_0039_best_c",
                "smooth_0040_01",
                "smooth_0040_02",
                "smooth_0040_03",
                "smooth_0040_04",
            ],
            "source_experiment": ["official", "0038", "0039", "0040", "0040", "0040", "0040"],
        }
    )


def meta_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "meta_feature": [
                "meta_source_family",
                "meta_text_signal_state",
                "meta_forecast_range_change_sign",
                "meta_hard_0039_active",
                "meta_smooth_0040_01_active",
                "meta_smooth_0040_02_active",
                "meta_smooth_late_full_disagreement",
                "meta_anchor_smooth_disagreement",
                "meta_hard_smooth_disagreement",
            ]
        }
    )


def test_family_variant_map_keeps_anchor_and_expected_core() -> None:
    variants = family_variant_map(family_catalog())

    assert variants["core"] == ("anchor_0038_c", "hard_0039_best_c", "smooth_0040_01", "smooth_0040_02")
    assert "anchor_0038_c" in variants["expanded"]
    assert all("anchor_0038_c" in families for families in variants.values())


def test_build_sensitivity_specs_has_unique_candidate_ids_and_threshold_variants() -> None:
    catalog = build_sensitivity_specs(family_catalog(), meta_catalog())

    assert catalog["candidate_id"].is_unique
    assert {"context", "routing_mode", "history_threshold", "family_inclusion"}.issubset(
        set(catalog["variant_kind"])
    )
    assert (catalog["min_global_history"] == 500).any()
    assert (catalog["family_group"] == "anchor_hard").any()


def test_specs_from_catalog_round_trips_global_feature_set() -> None:
    catalog = build_sensitivity_specs(family_catalog(), meta_catalog())
    global_row = catalog[catalog["feature_set"].eq("global")].head(1)

    spec = specs_from_catalog(global_row)[0]

    assert spec.stack_spec.feature_names == ()
    assert sensitivity_candidate_id(spec) == str(global_row.iloc[0]["candidate_id"])


def test_segment_scoreboard_excludes_2024_and_scores_source_segments() -> None:
    predictions = pd.DataFrame(
        {
            "candidate_id": ["c1", "c1", "c1", "c1"],
            "target_date": pd.to_datetime(["2000-01-01", "2000-01-02", "2021-04-14", "2021-04-15"]),
            "forecast_source_family": ["press_archive", "press_archive", "rss_archive", "rss_archive"],
            "target_tmax_c": [10.0, 11.0, 20.0, 21.0],
            "official_raw": [9.0, 10.0, 19.0, 20.0],
            "anchor_0038_c": [9.0, 10.0, 19.0, 20.0],
            "candidate_prediction_c": [10.0, 11.0, 20.0, 21.0],
            "variant_kind": ["test"] * 4,
            "comparison_group": ["case"] * 4,
        }
    )

    scores = segment_scoreboard(predictions)

    assert scores.empty


def test_segment_scoreboard_scores_large_segments_and_robustness() -> None:
    rows = []
    for idx in range(60):
        rows.append(
            {
                "candidate_id": "c1",
                "target_date": pd.Timestamp("2021-04-14") + pd.Timedelta(days=idx),
                "forecast_source_family": "rss_archive",
                "target_tmax_c": 20.0,
                "official_raw": 18.0,
                "anchor_0038_c": 19.0,
                "candidate_prediction_c": 20.0,
                "variant_kind": "test",
                "comparison_group": "case",
            }
        )
    predictions = pd.DataFrame(rows)
    segments = segment_scoreboard(predictions)
    scoreboard = pd.DataFrame(
        {
            "candidate_id": ["c1"],
            "variant_kind": ["test"],
            "comparison_group": ["case"],
            "feature_set": ["source"],
            "mode": ["positive_lift"],
            "same_source": [True],
            "family_group": ["core"],
            "family_count": [4],
            "min_global_history": [160],
            "min_bucket_history": [45],
            "mae": [0.0],
            "rmse": [0.0],
            "delta_vs_anchor": [-1.0],
            "late_eval_mae": [0.0],
            "late_eval_delta_vs_anchor": [-1.0],
        }
    )

    robust = robustness_summary(scoreboard, segments)

    assert "late_eval_actual" in set(segments["segment"])
    assert robust.loc[0, "segments_beating_anchor"] >= 1
