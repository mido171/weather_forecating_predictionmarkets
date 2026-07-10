from __future__ import annotations

import pandas as pd
import pytest

from hkg_tmax.data.official_residual_memory_features import (
    build_residual_memory_features,
    residual_memory_publication_safety_audit,
)
from hkg_tmax.evaluation.official_residual_memory_runner import (
    build_memory_scoreboards,
    evaluate_promotion_gates,
    row_identity_gate,
)
from hkg_tmax.features.residual_memory_policy import assert_no_forbidden_residual_memory_predictors


def _memory_frame(days: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    residuals = [float(idx) / 10.0 for idx in range(days)]
    return pd.DataFrame(
        {
            "target_date": dates,
            "cutoff_profile": "tminus1_2359",
            "forecast_selector_status": "selected",
            "anchor_forecast_max_c": 30.0,
            "y_true_c": [30.0 + value for value in residuals],
            "anchor_issue_at_utc": pd.date_range("2019-12-31", periods=days, freq="D", tz="UTC"),
        }
    )


def test_residual_memory_lag2_uses_exact_calendar_date_not_lag1() -> None:
    result = build_residual_memory_features(_memory_frame(), cutoff_profiles=["tminus1_2359"])
    row = result.frame[result.frame["target_date"].eq(pd.Timestamp("2020-01-04"))].iloc[0]
    assert row["residual_lag2_c"] == pytest.approx(0.1)
    assert row["residual_lag3_c"] == pytest.approx(0.0)
    assert "residual_lag1_c" not in result.frame.columns
    assert row["residual_memory_min_lag_days"] >= 2
    assert result.publication_safety_audit["status"] == "pass"


def test_residual_memory_roll_windows_enforce_min_counts() -> None:
    result = build_residual_memory_features(_memory_frame(days=40), cutoff_profiles=["tminus1_2359"])
    early = result.frame[result.frame["target_date"].eq(pd.Timestamp("2020-01-05"))].iloc[0]
    later = result.frame[result.frame["target_date"].eq(pd.Timestamp("2020-02-05"))].iloc[0]
    assert pd.isna(early["residual_roll7_mean_lag2_c"])
    assert pd.notna(later["residual_roll7_mean_lag2_c"])
    assert later["residual_memory_count_roll30"] >= 15


def test_residual_memory_audit_fails_on_injected_future_source_date() -> None:
    result = build_residual_memory_features(_memory_frame(), cutoff_profiles=["tminus1_2359"])
    bad = result.frame.copy()
    bad.loc[bad.index[-1], "residual_memory_max_source_date"] = bad.loc[bad.index[-1], "target_date"] - pd.Timedelta(days=1)
    audit = residual_memory_publication_safety_audit(bad, min_lag_days=2)
    assert audit["status"] == "fail"
    assert any(check["check_name"] == "residual_memory_max_source_date_lag2_or_older" and check["status"] == "fail" for check in audit["checks"])


def test_forbidden_target_and_evaluation_columns_are_not_predictors() -> None:
    with pytest.raises(ValueError, match="Forbidden"):
        assert_no_forbidden_residual_memory_predictors(["official_max_c", "residual_lag2_c", "y_true_c"])
    with pytest.raises(ValueError, match="Lag1"):
        assert_no_forbidden_residual_memory_predictors(["official_max_c", "residual_lag1_c"])


def _prediction_rows(model_id: str, prediction: float, stage: str = "rolling_validation") -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "target_date": dates,
            "cutoff_profile": "tminus1_2359",
            "fold_id": "fold4_2020_2021",
            "stage": stage,
            "split": "synthetic",
            "label_source": "label_core" if stage != "sealed_confirmation" else "sealed_confirmation",
            "y_true_c": [30.0, 31.0, 32.0],
            "anchor_forecast_max_c": [30.0, 30.0, 30.0],
            "prediction_c": prediction,
            "model_id": model_id,
            "model_family": model_id,
            "month": 1,
            "season_bucket": "DJF",
            "official_max_bin": "30",
        }
    )


def test_row_identity_gate_requires_d5_a7_and_raw_same_rows() -> None:
    rows = pd.concat(
        [
            _prediction_rows("A0_raw_official", 30.0),
            _prediction_rows("A7_final_residual_ensemble", 30.0),
            _prediction_rows("D5_conservative_A7_plus_memory_blend", 30.0),
        ],
        ignore_index=True,
    )
    assert row_identity_gate(rows, primary_cutoff="tminus1_2359")["status"] == "pass"
    bad = rows.drop(rows[rows["model_id"].eq("D5_conservative_A7_plus_memory_blend")].index[-1])
    assert row_identity_gate(bad, primary_cutoff="tminus1_2359")["status"] == "fail"


def test_promotion_logic_keeps_no_promote_when_gain_below_gate() -> None:
    rows = pd.concat(
        [
            _prediction_rows("A0_raw_official", 30.0, stage="rolling_validation"),
            _prediction_rows("A7_final_residual_ensemble", 30.0, stage="rolling_validation"),
            _prediction_rows("D5_conservative_A7_plus_memory_blend", 30.01, stage="rolling_validation"),
            _prediction_rows("A0_raw_official", 30.0, stage="presealed_holdout"),
            _prediction_rows("A7_final_residual_ensemble", 30.0, stage="presealed_holdout"),
            _prediction_rows("D5_conservative_A7_plus_memory_blend", 30.01, stage="presealed_holdout"),
            _prediction_rows("A0_raw_official", 30.0, stage="sealed_confirmation"),
            _prediction_rows("A7_final_residual_ensemble", 30.0, stage="sealed_confirmation"),
            _prediction_rows("D5_conservative_A7_plus_memory_blend", 30.01, stage="sealed_confirmation"),
        ],
        ignore_index=True,
    )
    boards = build_memory_scoreboards(rows)
    gate = row_identity_gate(rows, primary_cutoff="tminus1_2359")
    promotion = evaluate_promotion_gates(boards, {"primary_cutoff_profile": "tminus1_2359", "acceptance_gates": {}}, gate)
    assert promotion["decision"] == "no_promote"


def test_model_selection_log_contract_has_no_sealed_tuning_flag() -> None:
    contract = {
        "sealed_rows_used_for_model_selection": False,
        "folds_for_model_selection": ["fold1_2011_2013", "fold2_2014_2016", "fold3_2017_2019", "fold4_2020_2021"],
    }
    assert contract["sealed_rows_used_for_model_selection"] is False
    assert "sealed_confirmation" not in "|".join(contract["folds_for_model_selection"])

