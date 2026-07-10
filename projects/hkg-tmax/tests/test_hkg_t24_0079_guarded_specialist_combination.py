from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_0079_guarded_specialist_combination import (
    GuardedCombinationSpec,
    apply_guarded_combination,
    best_family_decisions,
    combine_family_decisions,
    decisions_agree,
)


def make_spec(**overrides: object) -> GuardedCombinationSpec:
    values = {
        "candidate_id": "test_guard",
        "pool_mode": "full_positive",
        "fallback_mode": "m0075",
        "combine_mode": "mean",
        "combo_weight": 1.0,
        "min_independent_families": 2,
        "min_prior_lift_c": 0.0,
        "min_abs_correction_c": 0.0,
        "require_same_sign": True,
        "correction_cap_c": 0.20,
    }
    values.update(overrides)
    return GuardedCombinationSpec(**values)  # type: ignore[arg-type]


def make_pool_meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["isd_a", "marine_b"],
            "new_champion_gate_passed": [True, True],
            "mae": [0.94, 0.95],
            "delta_mae_vs_0075": [-0.01, -0.01],
            "active_delta_mae_vs_0075": [-0.02, -0.03],
            "late_delta_mae_vs_0075": [-0.01, 0.0],
        }
    )


def make_specialist_rows(*, second_correction: float = 0.10, second_lift: float = 0.02) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["isd_a", "marine_b"],
            "row_index": [0, 0],
            "predicate_active_and_eligible": [True, True],
            "selected_prior_lift_c": [0.01, second_lift],
            "specialist_correction_c": [0.20, second_correction],
            "feature_family": ["regional_isd_station", "hko_daily_climate"],
        }
    )


def test_guard_requires_two_independent_families() -> None:
    rows = make_specialist_rows().iloc[[0]].copy()
    decisions = best_family_decisions(rows, make_pool_meta(), make_spec())

    assert not decisions_agree(decisions, make_spec())
    assert combine_family_decisions(decisions, make_spec()) == 0.0


def test_guard_blocks_opposite_sign_corrections() -> None:
    rows = make_specialist_rows(second_correction=-0.10)
    decisions = best_family_decisions(rows, make_pool_meta(), make_spec())

    assert len(decisions) == 2
    assert not decisions_agree(decisions, make_spec())
    assert combine_family_decisions(decisions, make_spec()) == 0.0


def test_guard_filters_low_prior_lift() -> None:
    rows = make_specialist_rows(second_lift=0.001)
    decisions = best_family_decisions(rows, make_pool_meta(), make_spec(min_prior_lift_c=0.005))

    assert len(decisions) == 1
    assert not decisions_agree(decisions, make_spec(min_prior_lift_c=0.005))


def test_apply_guarded_combination_uses_combo_when_two_families_agree() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2021-01-01"]),
            "current_target_tmax_c": [25.0],
            "forecast_source_family": ["rss_archive"],
            "m0075_prediction_c": [20.0],
        }
    )
    top_predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "candidate_id": ["champ"],
                    "row_index": [0],
                    "target_date": pd.to_datetime(["2021-01-01"]),
                    "candidate_prediction_c": [20.05],
                    "predicate_active_and_eligible": [False],
                    "selected_prior_lift_c": [0.0],
                    "specialist_correction_c": [0.0],
                    "feature_family": ["regional_isd_station"],
                }
            ),
            make_specialist_rows(),
        ],
        ignore_index=True,
    )
    scoreboard = pd.concat(
        [
            make_pool_meta(),
            pd.DataFrame(
                {
                    "candidate_id": ["champ"],
                    "new_champion_gate_passed": [True],
                    "mae": [0.94],
                    "delta_mae_vs_0075": [-0.01],
                    "active_delta_mae_vs_0075": [-0.01],
                    "late_delta_mae_vs_0075": [0.0],
                }
            ),
        ],
        ignore_index=True,
    )

    out = apply_guarded_combination(frame, top_predictions, scoreboard, make_spec(), "champ")

    assert out.loc[0, "guard_active"]
    assert out.loc[0, "selected_family_count"] == 2
    assert out.loc[0, "candidate_prediction_c"] == 20.15


def test_apply_guarded_combination_falls_back_when_guard_disagrees() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2021-01-01"]),
            "current_target_tmax_c": [25.0],
            "forecast_source_family": ["rss_archive"],
            "m0075_prediction_c": [20.0],
        }
    )
    top_predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "candidate_id": ["champ"],
                    "row_index": [0],
                    "target_date": pd.to_datetime(["2021-01-01"]),
                    "candidate_prediction_c": [20.05],
                    "predicate_active_and_eligible": [False],
                    "selected_prior_lift_c": [0.0],
                    "specialist_correction_c": [0.0],
                    "feature_family": ["regional_isd_station"],
                }
            ),
            make_specialist_rows(second_correction=-0.10),
        ],
        ignore_index=True,
    )
    scoreboard = make_pool_meta()

    out = apply_guarded_combination(
        frame,
        top_predictions,
        scoreboard,
        make_spec(fallback_mode="m0078"),
        "champ",
    )

    assert not out.loc[0, "guard_active"]
    assert out.loc[0, "candidate_prediction_c"] == 20.05
