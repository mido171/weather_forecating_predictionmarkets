from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_nonlinear_local_residual_fusion_lab import DELTA_GRID
from scripts.run_hkg_t24_sparse_specialist_delta_stack import (
    SparseStackSpec,
    combine_diagnostic_deltas,
    combine_prior_decisions,
    diagnostic_cell_map,
    leakage_audit,
    select_prior_cell_decision,
)


def make_spec(**overrides: object) -> SparseStackSpec:
    values = {
        "candidate_id": "test",
        "mode": "causal_prior_sparse_stack",
        "candidate_class": "causal_sparse_specialist_stack",
        "group_modes": ("source_signeddiff_range",),
        "min_history": 2,
        "min_prior_lift_c": 0.05,
        "min_abs_delta": 0.03,
        "combine_mode": "best_lift",
        "shrink": 1.0,
        "max_abs_delta": 0.18,
        "diagnostic_top_n": 0,
        "diagnostic_min_cell_n": 0,
        "diagnostic_max_active_delta_mae": 0.0,
    }
    values.update(overrides)
    return SparseStackSpec(**values)  # type: ignore[arg-type]


def test_select_prior_cell_decision_requires_history_and_lift() -> None:
    spec = make_spec(min_history=3, min_prior_lift_c=0.05)
    abs_sums = np.ones(len(DELTA_GRID), dtype=float) * 10.0

    assert (
        select_prior_cell_decision(
            group_mode="g",
            group_key="k",
            count=2,
            abs_sums=abs_sums,
            spec=spec,
        )
        is None
    )

    zero_index = int(np.argmin(np.abs(np.array(DELTA_GRID))))
    best_index = list(DELTA_GRID).index(-0.18)
    abs_sums[zero_index] = 9.0
    abs_sums[best_index] = 6.0
    decision = select_prior_cell_decision(
        group_mode="g",
        group_key="k",
        count=3,
        abs_sums=abs_sums,
        spec=spec,
    )

    assert decision is not None
    assert decision.best_delta == -0.18
    assert decision.prior_lift_c == 1.0


def test_combine_prior_decisions_respects_agreement_mode() -> None:
    spec = make_spec(combine_mode="agreement_mean")
    neg = select_prior_cell_decision(
        group_mode="g",
        group_key="a",
        count=3,
        abs_sums=np.array([1.0 if value == -0.18 else 2.0 for value in DELTA_GRID]),
        spec=make_spec(min_prior_lift_c=0.0),
    )
    pos = select_prior_cell_decision(
        group_mode="g",
        group_key="b",
        count=3,
        abs_sums=np.array([1.0 if value == 0.18 else 2.0 for value in DELTA_GRID]),
        spec=make_spec(min_prior_lift_c=0.0),
    )

    assert neg is not None
    assert pos is not None
    assert combine_prior_decisions([neg, pos], spec) == 0.0


def test_diagnostic_cell_map_selects_only_eligible_cells() -> None:
    cells = pd.DataFrame(
        {
            "group_mode": ["source_signeddiff_range", "source_signeddiff_range", "source_weather_range"],
            "group_key": ["a", "b", "c"],
            "n": [100, 10, 100],
            "best_fixed_delta": [-0.18, 0.18, -0.12],
            "active_delta_mae": [-0.02, -0.50, -0.03],
            "base_0069_mae": [1.0, 1.0, 1.0],
            "best_fixed_delta_mae": [0.9, 0.5, 0.8],
        }
    )
    spec = make_spec(
        group_modes=("source_signeddiff_range",),
        diagnostic_top_n=5,
        diagnostic_min_cell_n=60,
        diagnostic_max_active_delta_mae=-0.01,
    )

    out = diagnostic_cell_map(cells, spec)

    assert out == {("source_signeddiff_range", "a"): -0.18}


def test_combine_diagnostic_deltas_clips_and_uses_largest_abs_delta() -> None:
    spec = make_spec(combine_mode="best_lift", shrink=1.0, max_abs_delta=0.12)

    assert combine_diagnostic_deltas([-0.18, 0.05], spec) == -0.12


def test_leakage_audit_blocks_diagnostic_deployable_status() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "base_0069_prediction_c": [20.0, 21.0],
        }
    )
    scoreboard = pd.DataFrame(
        {
            "candidate_class": ["diagnostic_cell_atlas_stack", "causal_sparse_specialist_stack"],
            "deployable_gate_passed": [False, True],
            "delta_mae_vs_0069": [-0.01, -0.01],
            "fold_delta_max_vs_0069": [0.0, 0.0],
            "late_delta_mae_vs_0069": [0.0, 0.0],
        }
    )

    audit = leakage_audit(frame, scoreboard)

    assert audit["passed"].astype(bool).all()
