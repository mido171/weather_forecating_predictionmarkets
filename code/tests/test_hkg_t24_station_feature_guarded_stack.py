from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_station_feature_guarded_stack import (
    StackSpec,
    correction_for_stack,
    promotion_gate,
    select_member_candidates,
)


def test_active_mean_uses_only_active_members() -> None:
    corrections = np.array([[1.0, 3.0], [2.0, 8.0], [5.0, 0.0]])
    active = np.array([[True, True], [True, False], [False, False]])
    spec = StackSpec("mean", ("a", "b"), "active_mean", 1.0, 10.0, 1)

    out, active_count = correction_for_stack(spec, ["a", "b"], corrections, active)

    assert out.tolist() == [2.0, 2.0, 0.0]
    assert active_count.tolist() == [2, 1, 0]


def test_rank_first_uses_first_active_member_by_rank() -> None:
    corrections = np.array([[1.0, 3.0], [0.0, 8.0], [5.0, 7.0]])
    active = np.array([[True, True], [False, True], [True, True]])
    spec = StackSpec("first", ("a", "b"), "rank_first", 1.0, 10.0, 1)

    out, _active_count = correction_for_stack(spec, ["a", "b"], corrections, active)

    assert out.tolist() == [1.0, 8.0, 5.0]


def test_promotion_gate_blocks_stack_that_does_not_beat_best_singleton() -> None:
    row = {
        "delta_mae_vs_0064": -0.01,
        "delta_mae_vs_best_0065": 0.0,
        "fold_delta_max_vs_0064": 0.0,
        "fold_delta_max_vs_best_0065": 0.0,
        "active_n": 500,
    }

    assert promotion_gate(row) is False

    row["delta_mae_vs_best_0065"] = -0.001

    assert promotion_gate(row) is True


def test_select_member_candidates_prefers_promoted_unique_ids() -> None:
    scoreboard = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "candidate_type": ["feature_bucket", "pair_feature_bucket", "season_feature_bucket"],
            "promotion_gate_passed": [True, False, True],
            "delta_mae_vs_reference": [-0.02, -0.5, -0.01],
            "active_delta_mae_vs_reference": [-0.03, -0.9, -0.02],
            "fold_delta_max": [0.0, 0.0, 0.0],
            "source_families": ["x", "y", "z"],
        }
    )

    selected = select_member_candidates(scoreboard)

    assert selected == ["a", "c"]
