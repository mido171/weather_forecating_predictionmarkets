"""Leakage and source-integrity guards for residual-ML matrices."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LeakageCheck:
    check_name: str
    status: str
    violation_count: int
    details: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _count(mask: pd.Series) -> int:
    return int(mask.fillna(False).sum())


def run_leakage_checks(matrix: pd.DataFrame, lineage: pd.DataFrame) -> list[LeakageCheck]:
    checks: list[LeakageCheck] = []
    if "anchor_issue_at_utc" in matrix and "cutoff_at_utc" in matrix:
        violation = matrix["anchor_issue_at_utc"].notna() & (
            pd.to_datetime(matrix["anchor_issue_at_utc"], utc=True)
            > pd.to_datetime(matrix["cutoff_at_utc"], utc=True)
        )
        count = _count(violation)
        checks.append(
            LeakageCheck(
                "forecast_anchor_issue_at_or_before_cutoff",
                "pass" if count == 0 else "fail",
                count,
                {"sample_target_dates": matrix.loc[violation, "target_date"].head(10).astype(str).tolist()},
            )
        )
    if "max_forecast_revision_issue_at_utc" in matrix:
        violation = matrix["max_forecast_revision_issue_at_utc"].notna() & (
            pd.to_datetime(matrix["max_forecast_revision_issue_at_utc"], utc=True)
            > pd.to_datetime(matrix["cutoff_at_utc"], utc=True)
        )
        count = _count(violation)
        checks.append(
            LeakageCheck(
                "forecast_revision_issue_at_or_before_cutoff",
                "pass" if count == 0 else "fail",
                count,
                {"sample_target_dates": matrix.loc[violation, "target_date"].head(10).astype(str).tolist()},
            )
        )
    for column, name in (
        ("latest_hourly_dispatch_at_utc_used", "hourly_dispatch_at_or_before_cutoff"),
        ("latest_hourly_observation_at_utc_used", "hourly_observation_at_or_before_cutoff"),
        ("max_hourly_dispatch_at_utc_used", "max_hourly_dispatch_at_or_before_cutoff"),
        ("max_hourly_observation_at_utc_used", "max_hourly_observation_at_or_before_cutoff"),
    ):
        if column not in matrix:
            continue
        violation = matrix[column].notna() & (
            pd.to_datetime(matrix[column], utc=True) > pd.to_datetime(matrix["cutoff_at_utc"], utc=True)
        )
        count = _count(violation)
        checks.append(
            LeakageCheck(
                name,
                "pass" if count == 0 else "fail",
                count,
                {"sample_target_dates": matrix.loc[violation, "target_date"].head(10).astype(str).tolist()},
            )
        )
    if "target_history_max_source_date" in matrix:
        latest_allowed = pd.to_datetime(matrix["target_date"]) - pd.Timedelta(days=2)
        violation = matrix["target_history_max_source_date"].notna() & (
            pd.to_datetime(matrix["target_history_max_source_date"]) > latest_allowed
        )
        count = _count(violation)
        checks.append(
            LeakageCheck(
                "target_history_lag2_floor",
                "pass" if count == 0 else "fail",
                count,
                {"sample_target_dates": matrix.loc[violation, "target_date"].head(10).astype(str).tolist()},
            )
        )
    if not lineage.empty:
        forbidden = lineage[
            lineage["source_table"].eq(
                "raw_audit.codex_audit_ds_01_hko_daily_tmax_target_hko_daily_extr_23cb54da"
            )
        ]
        checks.append(
            LeakageCheck(
                "raw_daily_extract_not_used_as_predictor",
                "pass" if forbidden.empty else "fail",
                int(len(forbidden)),
                {"features": forbidden["feature_name"].head(20).tolist()},
            )
        )
        target_bad = lineage[
            lineage["uses_target_label_boolean"].astype(bool)
            & (pd.to_numeric(lineage["minimum_lag_days"], errors="coerce").fillna(-1) < 2)
        ]
        checks.append(
            LeakageCheck(
                "target_label_predictor_minimum_lag_days",
                "pass" if target_bad.empty else "fail",
                int(len(target_bad)),
                {"features": target_bad["feature_name"].head(20).tolist()},
            )
        )
    if "anchor_source_url" in matrix:
        bad_source = matrix["anchor_source_url"].notna() & ~matrix["anchor_source_url"].astype(str).str.contains(
            "info.gov.hk/gia/wr", regex=False
        )
        count = _count(bad_source)
        checks.append(
            LeakageCheck(
                "forecast_anchor_apples_to_apples_info_gov_source",
                "pass" if count == 0 else "fail",
                count,
                {"sample_source_urls": matrix.loc[bad_source, "anchor_source_url"].head(10).astype(str).tolist()},
            )
        )
    if "forecast_selector_status" in matrix:
        selected = matrix[matrix["forecast_selector_status"].eq("selected")]
        duplicate_count = int(selected.duplicated(["target_date", "cutoff_profile"]).sum())
        checks.append(
            LeakageCheck(
                "one_selected_anchor_per_target_cutoff",
                "pass" if duplicate_count == 0 else "fail",
                duplicate_count,
                {},
            )
        )
    return checks


def leakage_audit_payload(matrix: pd.DataFrame, lineage: pd.DataFrame) -> dict[str, Any]:
    checks = run_leakage_checks(matrix, lineage)
    return {
        "status": "pass" if all(check.status == "pass" for check in checks) else "fail",
        "total_violations": int(sum(check.violation_count for check in checks)),
        "checks": [check.to_record() for check in checks],
    }


def next_round_leakage_audit_payload(
    matrix: pd.DataFrame,
    lineage: pd.DataFrame,
    *,
    feature_names: list[str],
    router_thresholds: dict[str, Any] | None = None,
    router_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    checks = run_leakage_checks(matrix, lineage)
    feature_set = set(feature_names)
    forbidden_evaluation_features = {
        "true_residual_c",
        "raw_abs_error_c",
        "candidate_abs_error_c",
        "benefit_c",
        "apply_label",
        "strong_apply_label",
        "sign_label",
        "candidate_sign_correct",
        "abs_improvement_vs_raw_c",
        "helped_vs_raw_flag",
        "worsened_vs_raw_flag",
        "raw_error_decile",
        "raw_abs_error_decile",
    }
    tail_label_features = {
        "tail_150_label",
        "tail_200_label",
        "tail_sign_label",
        "tail_positive_label",
        "abs_residual_c",
    }
    eval_violations = sorted(feature_set & forbidden_evaluation_features)
    checks.append(
        LeakageCheck(
            "posthoc_raw_error_slices_not_used_as_features",
            "pass" if not eval_violations else "fail",
            len(eval_violations),
            {"features": eval_violations},
        )
    )
    tail_violations = sorted(feature_set & tail_label_features)
    checks.append(
        LeakageCheck(
            "tail_labels_not_used_as_features",
            "pass" if not tail_violations else "fail",
            len(tail_violations),
            {"features": tail_violations},
        )
    )
    threshold_uses_sealed = bool((router_thresholds or {}).get("sealed_rows_used_for_selection", False))
    checks.append(
        LeakageCheck(
            "sealed_rows_not_used_for_router_threshold_selection",
            "pass" if not threshold_uses_sealed else "fail",
            int(threshold_uses_sealed),
            {"selection_stage": (router_thresholds or {}).get("selection_stage")},
        )
    )
    checks.append(
        LeakageCheck(
            "router_thresholds_selected_without_presealed_or_sealed_if_configured",
            "pass" if not threshold_uses_sealed else "fail",
            int(threshold_uses_sealed),
            {"threshold_id": (router_thresholds or {}).get("threshold_id")},
        )
    )
    same_fold_violations = 0
    if router_predictions is not None and not router_predictions.empty:
        if {"fold_id", "router_training_fold_ids"}.issubset(router_predictions.columns):
            for row in router_predictions[["fold_id", "router_training_fold_ids"]].itertuples(index=False):
                fold_id = str(row.fold_id)
                training_ids = str(row.router_training_fold_ids).split("|")
                if fold_id in training_ids:
                    same_fold_violations += 1
    checks.append(
        LeakageCheck(
            "router_oof_predictions_are_inner_fold_only",
            "pass" if same_fold_violations == 0 else "fail",
            same_fold_violations,
            {},
        )
    )
    return {
        "status": "pass" if all(check.status == "pass" for check in checks) else "fail",
        "total_violations": int(sum(check.violation_count for check in checks)),
        "checks": [check.to_record() for check in checks],
    }
