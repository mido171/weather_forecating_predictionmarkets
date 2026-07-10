from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_station_feature_bucket_residual_mining import (
    CandidateSpec,
    CorrectionSpec,
    TermSpec,
    compute_prior_active_correction,
    mask_for_candidate,
    promotion_gate,
    term_specs,
)


def test_candidate_mask_uses_declared_bucket_terms_and_season() -> None:
    frame = pd.DataFrame(
        {
            "season": ["DJF", "MAM", "DJF"],
            "bucket__feature_a": ["low", "low", "high"],
            "bucket__feature_b": ["mid", "high", "mid"],
            "target_tmax_c": [20.0, 22.0, 21.0],
        }
    )
    terms = {
        "a_low_djf": TermSpec(
            "a_low_djf",
            "season_feature_bucket",
            "feature_a",
            "low",
            "DJF",
            "station_attribute",
            "450110-99999",
            "feature a",
        ),
        "b_mid": TermSpec(
            "b_mid",
            "feature_bucket",
            "feature_b",
            "mid",
            "",
            "station_attribute",
            "450110-99999",
            "feature b",
        ),
    }
    candidate = CandidateSpec(
        "candidate",
        "pair_feature_bucket",
        ("a_low_djf", "b_mid"),
        CorrectionSpec("expanding", "expanding", None, 1, 0.0, 2.0),
    )

    mask = mask_for_candidate(frame, candidate, terms)

    assert mask.tolist() == [True, False, False]


def test_prior_active_correction_excludes_current_active_row() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "reference_residual_to_add_c": [1.0, 3.0, 90.0],
        }
    )
    active = pd.Series([True, True, True])
    correction = CorrectionSpec("expanding", "expanding", None, 1, 0.0, 100.0)

    corrections, prior_rows, raw_means = compute_prior_active_correction(frame, active, correction)

    assert corrections.tolist() == [0.0, 1.0, 2.0]
    assert prior_rows.tolist() == [0, 1, 2]
    assert raw_means[0] != raw_means[0]


def test_promotion_gate_blocks_large_fold_damage() -> None:
    row = {
        "active_n": 200,
        "active_delta_mae_vs_reference": -0.05,
        "delta_mae_vs_reference": -0.01,
        "fold_delta_max": 0.02,
        "active_correction_share": 0.9,
    }

    assert promotion_gate(row) is False

    row["fold_delta_max"] = 0.005

    assert promotion_gate(row) is True


def test_term_specs_treat_csv_blank_season_as_no_season_gate() -> None:
    terms = term_specs(
        pd.DataFrame(
            {
                "term_id": ["term"],
                "term_type": ["feature_bucket"],
                "feature_id": ["feature_a"],
                "bucket_value": ["low"],
                "season": [float("nan")],
                "source_family": ["station_attribute"],
                "station_ids": ["450110-99999"],
                "display_name": ["feature a"],
            }
        )
    )

    assert terms["term"].season == ""
