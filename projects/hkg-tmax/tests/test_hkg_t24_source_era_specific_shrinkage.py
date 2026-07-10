from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_nonlinear_local_residual_fusion_lab import DELTA_GRID
from scripts.run_hkg_t24_source_era_specific_shrinkage import (
    SourceEraSpec,
    combine_expert_decisions,
    expert_names_for_row,
    leakage_audit,
    select_expert_decision,
)


def make_spec(**overrides: object) -> SourceEraSpec:
    values = {
        "candidate_id": "test",
        "mode": "causal_source_era_shrinkage",
        "candidate_class": "causal_source_era_shrinkage",
        "expert_set": "all",
        "min_total_history": 3,
        "min_fold_history": 2,
        "min_total_lift_c": 0.01,
        "min_fold_lift_c": 0.0,
        "support_shrink": 10.0,
        "lift_scale_c": 0.05,
        "base_shrink": 1.0,
        "max_abs_delta": 0.12,
        "combine_mode": "best_min_lift",
    }
    values.update(overrides)
    return SourceEraSpec(**values)  # type: ignore[arg-type]


def rss_warm_row() -> pd.Series:
    return pd.Series(
        {
            "forecast_source_family": "rss_archive",
            "signeddiff_bucket": "station_warmer_ge_1c",
            "forecast_range_bucket": "range_le_3c",
            "weather_bucket": "weather_other",
            "active_count_bucket": "station_stack_inactive",
            "month": 5,
            "season": "MAM",
        }
    )


def test_expert_names_route_rss_warm_tight_may_cell() -> None:
    names = expert_names_for_row(rss_warm_row(), "all")

    assert "rss_warm_tight_range" in names
    assert "rss_warm_tight_range_mam" in names
    assert "rss_warm_tight_range_may" in names
    assert "rss_warm_tight_range_inactive" in names


def test_select_expert_decision_requires_fold_support() -> None:
    spec = make_spec(min_total_history=3, min_fold_history=3)
    abs_sums = np.ones(len(DELTA_GRID), dtype=float) * 10.0

    assert (
        select_expert_decision(
            expert_name="rss",
            total_count=3,
            total_abs_sums=abs_sums,
            fold_count=2,
            fold_abs_sums=abs_sums,
            spec=spec,
        )
        is None
    )


def test_select_expert_decision_requires_total_and_fold_delta_sign_agreement() -> None:
    spec = make_spec(min_total_history=3, min_fold_history=3, min_total_lift_c=0.0, min_fold_lift_c=0.0)
    total = np.array([1.0 if value == -0.18 else 2.0 for value in DELTA_GRID])
    fold = np.array([1.0 if value == 0.18 else 2.0 for value in DELTA_GRID])

    assert (
        select_expert_decision(
            expert_name="rss",
            total_count=3,
            total_abs_sums=total,
            fold_count=3,
            fold_abs_sums=fold,
            spec=spec,
        )
        is None
    )


def test_combine_expert_decisions_blocks_mixed_signs_in_agreement_mode() -> None:
    spec = make_spec(min_total_lift_c=0.0, min_fold_lift_c=0.0)
    neg_sums = np.array([1.0 if value == -0.18 else 2.0 for value in DELTA_GRID])
    pos_sums = np.array([1.0 if value == 0.18 else 2.0 for value in DELTA_GRID])
    neg = select_expert_decision(
        expert_name="neg",
        total_count=10,
        total_abs_sums=neg_sums,
        fold_count=10,
        fold_abs_sums=neg_sums,
        spec=spec,
    )
    pos = select_expert_decision(
        expert_name="pos",
        total_count=10,
        total_abs_sums=pos_sums,
        fold_count=10,
        fold_abs_sums=pos_sums,
        spec=spec,
    )

    assert neg is not None
    assert pos is not None
    assert combine_expert_decisions([neg, pos], make_spec(combine_mode="agreement_mean")) == 0.0


def test_leakage_audit_keeps_fixed_rules_non_deployable() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "base_0069_prediction_c": [20.0, 21.0],
        }
    )
    scoreboard = pd.DataFrame(
        {
            "candidate_class": ["diagnostic_source_era_fixed", "causal_source_era_shrinkage"],
            "deployable_gate_passed": [False, True],
            "delta_mae_vs_0069": [-0.01, -0.01],
            "fold_delta_max_vs_0069": [0.0, 0.0],
            "late_delta_mae_vs_0069": [0.0, 0.0],
        }
    )

    audit = leakage_audit(frame, scoreboard)

    assert audit["passed"].astype(bool).all()
