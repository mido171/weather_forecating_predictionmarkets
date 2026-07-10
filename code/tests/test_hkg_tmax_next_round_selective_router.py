from __future__ import annotations

import pandas as pd
import pytest

from hkg_tmax.data.anchor_provenance_audit import build_anchor_provenance_audit
from hkg_tmax.evaluation.no_harm_reporting import no_harm_audit
from hkg_tmax.features.leakage_guards import next_round_leakage_audit_payload
from hkg_tmax.modeling.selective_router import (
    apply_selective_router,
    build_router_labels,
    fit_router_models,
    select_router_thresholds,
)
from hkg_tmax.modeling.tail_specialist import apply_tail_overlay


def _router_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "cutoff_profile": ["tminus1_2359", "tminus1_2359"],
            "stage": ["rolling_validation", "rolling_validation"],
            "fold_id": ["fold1", "fold2"],
            "y_true_c": [31.0, 29.0],
            "anchor_forecast_max_c": [30.0, 30.0],
            "candidate_resid_ensemble_c": [0.50, -0.50],
            "candidate_resid_lgbm_a3_c": [0.40, -0.40],
            "candidate_resid_lgbm_pruned_full_c": [0.50, -0.50],
            "candidate_resid_catboost_c": [0.45, -0.45],
            "candidate_resid_linear_c": [0.35, -0.35],
            "router_apply_probability": [0.90, 0.90],
            "router_sign_probability": [0.90, 0.90],
            "router_expected_benefit_c": [0.20, 0.20],
            "official_max_c": [30.0, 30.0],
        }
    )


def test_router_oof_predictions_do_not_use_same_inner_fold_labels() -> None:
    matrix = pd.DataFrame({"target_date": pd.to_datetime(["2020-01-01"]), "cutoff_at_utc": [pd.Timestamp("2020-01-01", tz="UTC")]})
    lineage = pd.DataFrame()
    ok = pd.DataFrame({"fold_id": ["fold1"], "router_training_fold_ids": ["fold2|fold3"]})
    audit = next_round_leakage_audit_payload(matrix, lineage, feature_names=["official_max_c"], router_predictions=ok)
    assert audit["status"] == "pass"
    bad = pd.DataFrame({"fold_id": ["fold1"], "router_training_fold_ids": ["fold1|fold2"]})
    audit_bad = next_round_leakage_audit_payload(matrix, lineage, feature_names=["official_max_c"], router_predictions=bad)
    check = [row for row in audit_bad["checks"] if row["check_name"] == "router_oof_predictions_are_inner_fold_only"][0]
    assert check["status"] == "fail"


def test_router_labels_are_evaluation_only_not_feature_columns() -> None:
    frame = build_router_labels(_router_frame())
    with pytest.raises(ValueError, match="evaluation-only"):
        fit_router_models(frame, ["official_max_c", "benefit_c"], seed=1)


def test_selective_router_abstains_to_zero_when_thresholds_not_met() -> None:
    frame = _router_frame()
    frame["router_apply_probability"] = 0.10
    out = apply_selective_router(
        frame,
        pd.DataFrame(),
        {},
        {"threshold_benefit": 0.0, "threshold_apply": 0.52, "threshold_sign": 0.56, "positive_cap_c": 0.5, "negative_cap_c": 0.35, "hard_abs_cap_c": 0.75},
    )
    assert out["router_applied_flag"].tolist() == [0, 0]
    assert out["residual_prediction_c"].tolist() == [0.0, 0.0]


def test_asymmetric_caps_are_applied_correctly() -> None:
    frame = _router_frame()
    frame["candidate_resid_ensemble_c"] = [0.80, -0.80]
    out = apply_selective_router(
        frame,
        pd.DataFrame(),
        {},
        {"threshold_benefit": 0.0, "threshold_apply": 0.52, "threshold_sign": 0.56, "positive_cap_c": 0.50, "negative_cap_c": 0.35, "hard_abs_cap_c": 0.75},
    )
    assert out["residual_prediction_c"].round(3).tolist() == [0.50, -0.35]


def test_tail_overlay_requires_probability_and_sign_thresholds() -> None:
    base = apply_selective_router(
        _router_frame(),
        pd.DataFrame(),
        {},
        {"threshold_benefit": 0.0, "threshold_apply": 0.52, "threshold_sign": 0.56, "positive_cap_c": 0.50, "negative_cap_c": 0.35, "hard_abs_cap_c": 0.75},
    )
    base["tail150_probability"] = [0.59, 0.80]
    base["tail_sign_probability"] = [0.90, 0.50]
    base["tail_residual_prediction_c"] = [0.80, -0.80]
    base["predicted_tail_benefit_c"] = [0.80, 0.80]
    out = apply_tail_overlay(
        base,
        base,
        {},
        {"tail150_probability": 0.60, "tail_sign_probability": 0.62, "min_abs_tail_correction_c": 0.25, "hard_abs_cap_c": 1.0},
    )
    assert out["tail_overlay_applied_flag"].tolist() == [0, 0]


def test_no_harm_audit_flags_monthly_degradation() -> None:
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    raw = pd.DataFrame(
        {
            "target_date": dates,
            "cutoff_profile": "tminus1_2359",
            "model_id": "A0_raw_official",
            "stage": "rolling_validation",
            "month": 1,
            "y_true_c": 10.0,
            "anchor_forecast_max_c": 10.0,
            "prediction_c": 10.0,
        }
    )
    candidate = raw.copy()
    candidate["model_id"] = "C2_selective_router"
    candidate["prediction_c"] = 10.10
    candidate["router_applied_flag"] = 1
    audit = no_harm_audit(candidate, raw_predictions=raw, config={"router": {"no_harm_guardrails": {"max_monthly_worse_vs_raw_c": 0.015, "min_apply_rate": 0.0, "max_apply_rate": 1.0}}})
    monthly = [row for row in audit["checks"] if row["check_name"] == "no_month_n_ge_100_worse_than_raw_by_more_than_guardrail"][0]
    assert monthly["status"] == "fail"


def test_anchor_provenance_reports_tminus1_1500_missing_reason() -> None:
    targets = pd.DataFrame({"target_date": [pd.Timestamp("2020-01-02")], "y_true_c": [20.0]})
    forecasts = pd.DataFrame(
        {
            "target_date": [pd.Timestamp("2020-01-02")],
            "source": ["info_gov"],
            "product_type": ["local"],
            "row_quality_status": ["usable_local_minmax"],
            "target_issue_lead_days": [1],
            "forecast_min_c": [17.0],
            "forecast_max_c": [21.0],
            "issue_at_utc": [pd.Timestamp("2020-01-01 08:15:00", tz="UTC")],
            "issue_at_hkt": [pd.Timestamp("2020-01-01 16:15:00")],
            "parse_status": ["parsed"],
        }
    )
    audit = build_anchor_provenance_audit(targets, forecasts, ["tminus1_1500"])
    assert audit.iloc[0]["strict_selected_anchor_status"] == "no_eligible_anchor"
    assert audit.iloc[0]["reason_no_anchor"] == "next_strict_eligible_after_cutoff"


def test_sealed_rows_not_used_for_router_threshold_selection() -> None:
    frame = pd.concat(
        [
            _router_frame(),
            _router_frame().assign(stage="sealed_confirmation", target_date=pd.to_datetime(["2024-01-01", "2024-01-02"])),
        ],
        ignore_index=True,
    )
    thresholds = select_router_thresholds(frame, {"router": {"threshold_benefit_grid": [0.0], "threshold_apply_grid": [0.52], "threshold_sign_grid": [0.56]}})
    assert thresholds["sealed_rows_used_for_selection"] is False
    assert thresholds["selection_stage"] == "rolling_validation_or_router_inner_oof_only"


def test_raw_error_decile_not_allowed_as_feature() -> None:
    audit = next_round_leakage_audit_payload(
        pd.DataFrame({"target_date": pd.to_datetime(["2020-01-01"])}),
        pd.DataFrame(),
        feature_names=["official_max_c", "raw_error_decile"],
    )
    check = [row for row in audit["checks"] if row["check_name"] == "posthoc_raw_error_slices_not_used_as_features"][0]
    assert check["status"] == "fail"


def test_prediction_rows_include_apply_and_tail_diagnostics() -> None:
    base = apply_selective_router(
        _router_frame(),
        pd.DataFrame(),
        {},
        {"threshold_benefit": 0.0, "threshold_apply": 0.52, "threshold_sign": 0.56, "positive_cap_c": 0.50, "negative_cap_c": 0.35, "hard_abs_cap_c": 0.75},
    )
    base["tail150_probability"] = 0.80
    base["tail_sign_probability"] = 0.90
    base["tail_residual_prediction_c"] = 0.80
    base["predicted_tail_benefit_c"] = 0.80
    out = apply_tail_overlay(base, base, {}, {"tail150_probability": 0.60, "tail_sign_probability": 0.62, "min_abs_tail_correction_c": 0.25, "hard_abs_cap_c": 1.0})
    for column in ["router_apply_probability", "router_applied_flag", "tail150_probability", "tail_overlay_applied_flag", "final_correction_source"]:
        assert column in out.columns
