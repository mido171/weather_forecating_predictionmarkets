from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest
from hkg_t24.db.ddl import FOUNDATION_SQL
from hkg_t24.features.matrix_builder import FeatureMatrixRow
from hkg_t24.models.distribution import threshold_probability_keys, train_distribution_from_inputs
from hkg_t24.models.final_formula import SystemPrediction, assemble_pre_distribution_predictions
from hkg_t24.models.router import (
    synthetic_router_inputs,
    train_router_from_inputs,
    train_router_suite_from_inputs,
)
from hkg_t24.models.specialists import SPECIALIST_SPECS, _prior_score, train_specialists_from_inputs
from hkg_t24.models.static_weights import apply_caps_and_masks, optimize_static_weights
from hkg_t24.models.system_replay import _system_replay_coverage_rows


def test_static_weights_respect_caps_masks_and_sum_to_one() -> None:
    expert_ids = ("E0_OFFICIAL_RAW_ANCHOR", "E1_OFFICIAL_RESIDUAL", "E2_TARGET_MEMORY")
    optimized = optimize_static_weights(
        expert_ids=expert_ids,
        prediction_matrix=[
            {
                "E0_OFFICIAL_RAW_ANCHOR": 30.0,
                "E1_OFFICIAL_RESIDUAL": 30.1,
                "E2_TARGET_MEMORY": 29.8,
            },
            {
                "E0_OFFICIAL_RAW_ANCHOR": 31.0,
                "E1_OFFICIAL_RESIDUAL": 31.1,
                "E2_TARGET_MEMORY": 30.9,
            },
        ],
        labels=[30.1, 31.1],
        caps={
            "E0_OFFICIAL_RAW_ANCHOR": 0.8,
            "E1_OFFICIAL_RESIDUAL": 0.8,
            "E2_TARGET_MEMORY": 0.4,
        },
    )
    assert sum(optimized.weights.values()) == pytest.approx(1.0)
    assert optimized.weights["E2_TARGET_MEMORY"] <= 0.4 + 1e-9

    masked = apply_caps_and_masks(
        optimized.weights,
        expert_ids=expert_ids,
        caps={
            "E0_OFFICIAL_RAW_ANCHOR": 0.8,
            "E1_OFFICIAL_RESIDUAL": 0.8,
            "E2_TARGET_MEMORY": 0.4,
        },
        availability={
            "E0_OFFICIAL_RAW_ANCHOR": True,
            "E1_OFFICIAL_RESIDUAL": False,
            "E2_TARGET_MEMORY": True,
        },
        promoted={
            "E0_OFFICIAL_RAW_ANCHOR": True,
            "E1_OFFICIAL_RESIDUAL": True,
            "E2_TARGET_MEMORY": True,
        },
    )
    assert masked.weights["E1_OFFICIAL_RESIDUAL"] == 0.0
    assert sum(masked.weights.values()) == pytest.approx(1.0)


def test_router_refuses_strict_shadow_expert_contamination() -> None:
    rows, predictions = synthetic_router_inputs(days=90)
    contaminated = list(predictions)
    shadow = next(prediction for prediction in contaminated if prediction.expert_id == "E6_IFS_OPER_SHADOW")
    contaminated.append(replace(shadow, expert_scope="strict", prediction_status="active", prediction_tmax_c=31.0))

    with pytest.raises(ValueError, match="proxy/shadow"):
        train_router_from_inputs(rows=rows, predictions=contaminated, router_id="R0")


def test_r0_demotes_when_strict_e0_official_anchor_is_unavailable() -> None:
    rows, predictions = synthetic_router_inputs(days=90)
    without_e0 = [
        replace(
            prediction,
            prediction_tmax_c=None,
            raw_anchor_tmax_c=None,
            prediction_status="placeholder",
            placeholder_reason="NO_ELIGIBLE_ROWS_FOR_DATE",
            router_weight_cap=0.0,
        )
        if prediction.expert_id == "E0_OFFICIAL_RAW_ANCHOR"
        else prediction
        for prediction in predictions
    ]

    result = train_router_from_inputs(rows=rows, predictions=without_e0, router_id="R0")

    assert result.promotion_status == "demoted"
    assert result.demotion_reason == "STRICT_E0_OFFICIAL_ANCHOR_UNAVAILABLE"
    assert all(all(weight == 0.0 for weight in prediction.final_weights.values()) for prediction in result.predictions)


def test_system_replay_coverage_rows_distinguish_strict_inputs_and_fallbacks() -> None:
    first = date(2021, 1, 1)
    second = date(2021, 1, 2)
    rows = [
        FeatureMatrixRow(
            target_date_hkt=first,
            cutoff_id="H24N",
            snapshot_id="s1",
            feature_scope="strict",
            schema_version="strict",
            features={"target__lag365_tmax_c": 28.0, "gfs__center__tmax_c": 29.0},
            target_tmax_c=30.0,
        ),
        FeatureMatrixRow(
            target_date_hkt=second,
            cutoff_id="H24N",
            snapshot_id="s2",
            feature_scope="strict",
            schema_version="strict",
            features={"official__forecast_max_c": 31.0, "target__clim30_mean_c": 27.0},
            target_tmax_c=31.0,
        ),
    ]
    predictions = [
        SystemPrediction(
            target_date_hkt=first,
            cutoff_id="H24N",
            snapshot_id="s1",
            system_version="system",
            router_selected=None,
            router_selection_reason="router_unavailable_use_fallback_expert",
            base_forecast_c=28.5,
            specialist_total_correction_c=0.0,
            final_pre_distribution_c=28.5,
            final_point_tmax_c=28.5,
            p10_c=None,
            p25_c=None,
            p50_c=None,
            p75_c=None,
            p90_c=None,
            expected_abs_error_c=None,
            threshold_probabilities={},
            confidence_state="MEDIUM",
            no_trade_flag=False,
            distribution_status="not_trained",
            quantile_monotonic_repair=False,
            component_jsonb={"fallback_expert": "E2_TARGET_MEMORY"},
            leakage_status="passed",
        ),
        SystemPrediction(
            target_date_hkt=second,
            cutoff_id="H24N",
            snapshot_id="s2",
            system_version="system",
            router_selected=None,
            router_selection_reason="router_unavailable_use_fallback_expert",
            base_forecast_c=None,
            specialist_total_correction_c=0.0,
            final_pre_distribution_c=None,
            final_point_tmax_c=None,
            p10_c=None,
            p25_c=None,
            p50_c=None,
            p75_c=None,
            p90_c=None,
            expected_abs_error_c=None,
            threshold_probabilities={},
            confidence_state="LOW",
            no_trade_flag=True,
            distribution_status="failed_closed",
            quantile_monotonic_repair=False,
            component_jsonb={"fallback_expert": None},
            leakage_status="failed_closed",
        ),
    ]

    coverage = {category: count for category, count, _ in _system_replay_coverage_rows(predictions, rows)}

    assert coverage["strict_h24n_matrix_rows"] == 2
    assert coverage["official_anchor_available_rows"] == 1
    assert coverage["official_anchor_unavailable_rows"] == 1
    assert coverage["target_memory_feature_available_rows"] == 2
    assert coverage["target_memory_fallback_rows"] == 1
    assert coverage["nwp_backed_rows"] == 1
    assert coverage["no_forecast_rows"] == 1


def test_specialist_prior_missing_components_are_neutral_until_missing_weight_exceeds_gate() -> None:
    spec = next(item for item in SPECIALIST_SPECS if item.specialist_id == "S1_MARINE_SUPPRESSION")
    prior, available = _prior_score(spec, {}, {})
    assert not available
    assert prior is None

    partial_features = {
        "gfs__center__onshore_easterly_component_mps": 1.0,
        "gfs__spatial__inland_nw_minus_marine_s_tmax_c": 0.8,
        "gfs__spatial__inland_nw_minus_center_tmax_c": 0.3,
        "gfs__center__dewpoint_change_proxy_c": 0.1,
        "gfs__center__low_cloud_pct_mean": 40.0,
        "gfs__center__shortwave_w_m2_mean": 500.0,
    }
    prior, available = _prior_score(spec, partial_features, {key: [0.0, 1.0] for key in partial_features})
    assert available
    assert prior is not None
    assert 0.0 <= prior <= 1.0


def test_jira003_synthetic_pipeline_outputs_distribution_and_specialist_rows() -> None:
    rows, expert_predictions = synthetic_router_inputs(days=180)
    router_results = train_router_suite_from_inputs(rows, expert_predictions)
    assert {result.router_id for result in router_results} == {
        "R0_OFFICIAL_LONG_HISTORY",
        "R1_CORE_GFS_GEFS",
    }
    for result in router_results:
        for prediction in result.predictions:
            weight_sum = sum(prediction.final_weights.values())
            assert weight_sum == pytest.approx(0.0 if result.promotion_status == "demoted" else 1.0)
            assert all(
                prediction.final_weights.get(expert_id, 0.0) == 0.0
                for expert_id in ("E6_IFS_OPER_SHADOW", "E7_IFS_ENS_SHADOW", "E8_AI_NWP_SHADOW")
            )

    selected_router = router_results[1].predictions if router_results[1].promotion_status == "promoted" else router_results[0].predictions
    specialists = train_specialists_from_inputs(rows, selected_router)
    assert {result.specialist_id for result in specialists} == {spec.specialist_id for spec in SPECIALIST_SPECS}
    assert all(len(result.predictions) == len(rows) for result in specialists)

    pre_distribution = assemble_pre_distribution_predictions(
        rows=rows,
        expert_predictions=expert_predictions,
        router_results=router_results,
        specialist_results=specialists,
    )
    distribution = train_distribution_from_inputs(pre_distribution, rows, force_empirical=True)
    assert distribution.distribution_status == "demoted_empirical_fallback"
    assert distribution.threshold_key_count == 41
    for prediction in distribution.updated_predictions:
        assert len(prediction.threshold_probabilities) == 41
        assert prediction.p10_c is not None
        assert prediction.p10_c <= prediction.p25_c <= prediction.p50_c <= prediction.p75_c <= prediction.p90_c
        assert isinstance(prediction.no_trade_flag, bool)


def test_distribution_threshold_keys_are_exact_contract_shape() -> None:
    keys = threshold_probability_keys()
    assert len(keys) == 41
    assert keys[0] == "prob_tmax_ge_20_0"
    assert keys[-1] == "prob_tmax_ge_40_0"
    assert "prob_tmax_ge_32_5" in keys


def test_jira003_schema_objects_exist() -> None:
    for expected in (
        "CREATE TABLE IF NOT EXISTS model_router.router_prediction",
        "CREATE TABLE IF NOT EXISTS model_router.specialist_prediction",
        "CREATE TABLE IF NOT EXISTS model_oof.system_prediction",
        "CREATE TABLE IF NOT EXISTS model_router.router_scoreboard",
        "threshold_probabilities_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb",
        "ALTER TABLE model_eval.system_prediction_component",
    ):
        assert expected in FOUNDATION_SQL
