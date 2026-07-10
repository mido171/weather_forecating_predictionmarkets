from __future__ import annotations

from datetime import date, timedelta

import pytest
from hkg_t24.features.matrix_builder import FeatureMatrixRow
from hkg_t24.features.online_state import ResidualObservation, build_online_state, ewma_alpha
from hkg_t24.models.experts import generate_expert_oof_predictions
from hkg_t24.models.folds import FoldSpec
from hkg_t24.models.model_selection import CandidateMetric, select_model_candidate
from hkg_t24.models.oof import check_oof_integrity
from hkg_t24.timeutils import snapshot_id


def test_online_state_excludes_current_target_residual() -> None:
    target_date = date(2023, 6, 10)
    observations = [
        ResidualObservation(
            target_date_hkt=target_date - timedelta(days=2),
            source_key="official_raw",
            state_scope="global",
            prediction_tmax_c=30.0,
            target_tmax_c=31.0,
        ),
        ResidualObservation(
            target_date_hkt=target_date,
            source_key="official_raw",
            state_scope="global",
            prediction_tmax_c=10.0,
            target_tmax_c=40.0,
        ),
    ]

    state = build_online_state(
        target_date_hkt=target_date,
        source_key="official_raw",
        state_scope="global",
        observations=observations,
    )

    assert ewma_alpha(20) == pytest.approx(1.0 - 2 ** (-1 / 20))
    assert state.n_prior_rows == 1
    assert state.warmup_status == "COLD_START"
    assert state.features["online__official_raw__global__ewma_bias_h20_c"] == pytest.approx(1.0)
    assert state.features["online__official_raw__global__underforecast_streak_days"] == 1
    assert state.features["online__official_raw__global__correction_bias_h20_capped_c"] == pytest.approx(1 / 41)


def test_model_selection_tie_breaking_prefers_more_rows_then_simpler() -> None:
    selected = select_model_candidate(
        [
            CandidateMetric("candidate_b", "E1_OFFICIAL_RESIDUAL", 0.50, 0.60, 100, 2),
            CandidateMetric("candidate_a", "E1_OFFICIAL_RESIDUAL", 0.50, 0.60, 120, 3),
            CandidateMetric("candidate_c", "E1_OFFICIAL_RESIDUAL", 0.50, 0.60, 120, 1),
        ],
        required_improvement_c=0.01,
        promoted_weight_cap=0.8,
    )

    assert selected.selected_candidate_id == "candidate_c"
    assert selected.promoted
    assert selected.router_weight_cap == pytest.approx(0.8)


def _matrix_rows() -> list[FeatureMatrixRow]:
    start = date(2021, 1, 1)
    rows: list[FeatureMatrixRow] = []
    for offset in range(20):
        target = start + timedelta(days=offset)
        official = 25.0 + (offset % 5)
        target_label = official + 0.2
        features = {
            "official__forecast_max_c": official,
            "target__lag2_tmax_c": official - 0.1,
            "gfs__center__tmax_c": official - 0.3,
            "gefsens__center__tmax_p50_c": official - 0.4,
        }
        rows.append(
            FeatureMatrixRow(
                target_date_hkt=target,
                cutoff_id="H24N",
                snapshot_id=snapshot_id(target),
                feature_scope="strict",
                schema_version="test_schema",
                features=features,
                target_tmax_c=target_label,
            )
        )
    return rows


def _matrix_rows_without_official() -> list[FeatureMatrixRow]:
    rows = _matrix_rows()
    return [
        FeatureMatrixRow(
            target_date_hkt=row.target_date_hkt,
            cutoff_id=row.cutoff_id,
            snapshot_id=row.snapshot_id,
            feature_scope=row.feature_scope,
            schema_version=row.schema_version,
            features={key: value for key, value in row.features.items() if not key.startswith("official__")},
            target_tmax_c=row.target_tmax_c,
        )
        for row in rows
    ]


def test_expert_oof_predictions_are_chronological_and_placeholders_have_zero_shadow_weight() -> None:
    fold = FoldSpec(
        "synthetic_fold",
        date(2021, 1, 1),
        date(2021, 1, 10),
        date(2021, 1, 11),
        date(2021, 1, 20),
    )

    predictions = generate_expert_oof_predictions(_matrix_rows(), [fold], e1_promoted=False)
    integrity = check_oof_integrity(predictions)

    assert integrity.passed
    e0 = [row for row in predictions if row.expert_id == "E0_OFFICIAL_RAW_ANCHOR"]
    assert e0
    assert all(row.prediction_tmax_c == row.raw_anchor_tmax_c for row in e0)
    e1 = [row for row in predictions if row.expert_id == "E1_OFFICIAL_RESIDUAL" and row.prediction_status == "active"]
    assert e1
    assert all(abs(row.prediction_residual_c or 0.0) <= 0.7 for row in e1)
    assert all(row.router_weight_cap == 0.0 for row in e1)
    shadow = [row for row in predictions if row.expert_scope == "live_shadow"]
    assert shadow
    assert all(row.router_weight_cap == 0.0 for row in shadow)
    e10 = [row for row in predictions if row.expert_id == "E10_DIAGNOSTIC_PROXY"]
    assert e10 and all(row.prediction_status == "placeholder" for row in e10)


def test_e0_e1_remain_placeholders_when_strict_official_anchor_is_unavailable() -> None:
    fold = FoldSpec(
        "synthetic_fold",
        date(2021, 1, 1),
        date(2021, 1, 10),
        date(2021, 1, 11),
        date(2021, 1, 20),
    )

    predictions = generate_expert_oof_predictions(_matrix_rows_without_official(), [fold])
    e0 = [row for row in predictions if row.expert_id == "E0_OFFICIAL_RAW_ANCHOR"]
    e1 = [row for row in predictions if row.expert_id == "E1_OFFICIAL_RESIDUAL"]
    e2 = [row for row in predictions if row.expert_id == "E2_TARGET_MEMORY"]

    assert e0 and all(row.prediction_status == "placeholder" for row in e0)
    assert all(row.placeholder_reason == "NO_ELIGIBLE_ROWS_FOR_DATE" for row in e0)
    assert e1 and all(row.prediction_status == "placeholder" for row in e1)
    assert all(row.placeholder_reason == "INSUFFICIENT_HISTORY" for row in e1)
    assert e2 and any(row.prediction_status == "active" for row in e2)
