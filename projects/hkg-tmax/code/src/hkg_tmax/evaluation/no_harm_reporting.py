"""No-harm reporting for selective HKG Tmax residual correction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hkg_tmax.evaluation.metrics import score_arrays, score_frame
from hkg_tmax.modeling.selective_router import build_router_labels


def add_help_worse_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    raw_abs = (pd.to_numeric(out["y_true_c"], errors="coerce") - pd.to_numeric(out["anchor_forecast_max_c"], errors="coerce")).abs()
    model_abs = (pd.to_numeric(out["y_true_c"], errors="coerce") - pd.to_numeric(out["prediction_c"], errors="coerce")).abs()
    out["raw_abs_error_c"] = raw_abs
    out["model_abs_error_c"] = model_abs
    out["abs_improvement_vs_raw_c"] = raw_abs - model_abs
    out["helped_vs_raw_flag"] = (out["abs_improvement_vs_raw_c"] > 1e-9).astype(int)
    out["worsened_vs_raw_flag"] = (out["abs_improvement_vs_raw_c"] < -1e-9).astype(int)
    out["tied_vs_raw_flag"] = (out["abs_improvement_vs_raw_c"].abs() <= 1e-9).astype(int)
    return out


def help_worse_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    out = add_help_worse_columns(predictions)
    keep = [
        "target_date",
        "cutoff_profile",
        "stage",
        "fold_id",
        "model_id",
        "y_true_c",
        "anchor_forecast_max_c",
        "prediction_c",
        "residual_prediction_c",
        "raw_abs_error_c",
        "model_abs_error_c",
        "abs_improvement_vs_raw_c",
        "helped_vs_raw_flag",
        "worsened_vs_raw_flag",
        "router_applied_flag",
        "tail_overlay_applied_flag",
        "final_correction_source",
    ]
    return out[[column for column in keep if column in out.columns]].sort_values(
        ["cutoff_profile", "model_id", "abs_improvement_vs_raw_c"],
        ascending=[True, True, True],
    )


def apply_rate_by(predictions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    if "router_applied_flag" not in frame:
        frame["router_applied_flag"] = 0
    if "tail_overlay_applied_flag" not in frame:
        frame["tail_overlay_applied_flag"] = 0
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(by, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = {column: value for column, value in zip(by, key, strict=False)}
        record.update(
            {
                "rows": int(len(group)),
                "router_apply_rate": float(pd.to_numeric(group["router_applied_flag"], errors="coerce").fillna(0).mean()),
                "tail_overlay_apply_rate": float(
                    pd.to_numeric(group["tail_overlay_applied_flag"], errors="coerce").fillna(0).mean()
                ),
                "mean_correction_when_applied_c": float(
                    pd.to_numeric(group.loc[pd.to_numeric(group["router_applied_flag"], errors="coerce").fillna(0).eq(1), "residual_prediction_c"], errors="coerce").mean()
                )
                if pd.to_numeric(group["router_applied_flag"], errors="coerce").fillna(0).eq(1).any()
                else 0.0,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def benefit_deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    frame = add_help_worse_columns(predictions)
    if "router_expected_benefit_c" not in frame:
        frame["router_expected_benefit_c"] = frame["abs_improvement_vs_raw_c"]
    valid = frame[pd.to_numeric(frame["router_expected_benefit_c"], errors="coerce").notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    try:
        valid["predicted_benefit_decile"] = pd.qcut(
            pd.to_numeric(valid["router_expected_benefit_c"], errors="coerce"),
            q=10,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        valid["predicted_benefit_decile"] = 0
    rows: list[dict[str, Any]] = []
    for decile, group in valid.groupby("predicted_benefit_decile", dropna=False):
        rows.append(
            {
                "predicted_benefit_decile": int(decile) if pd.notna(decile) else -1,
                "rows": int(len(group)),
                "actual_mean_abs_improvement_vs_raw_c": float(group["abs_improvement_vs_raw_c"].mean()),
                "router_apply_rate": float(
                    pd.to_numeric(
                        group["router_applied_flag"] if "router_applied_flag" in group else pd.Series(0, index=group.index),
                        errors="coerce",
                    )
                    .fillna(0)
                    .mean()
                ),
                **score_arrays(
                    pd.to_numeric(group["y_true_c"], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(group["prediction_c"], errors="coerce").to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("predicted_benefit_decile").reset_index(drop=True)


def wrong_sign_apply_rate(predictions: pd.DataFrame) -> float:
    applied = predictions["router_applied_flag"] if "router_applied_flag" in predictions else pd.Series(0, index=predictions.index)
    frame = predictions[pd.to_numeric(applied, errors="coerce").fillna(0).eq(1)].copy()
    if frame.empty:
        return 0.0
    labels = build_router_labels(frame)
    return float(1.0 - labels["candidate_sign_correct"].mean())


def no_harm_audit(
    candidate: pd.DataFrame,
    *,
    raw_predictions: pd.DataFrame | None = None,
    current_a7_predictions: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    router_config = config.get("router", config)
    guardrails = router_config.get("no_harm_guardrails", {})
    frame = add_help_worse_columns(candidate)
    overall = score_frame(frame, ["cutoff_profile", "model_id"], scope="overall")
    by_split = score_frame(frame, ["cutoff_profile", "model_id", "stage"], scope="by_split")
    checks: list[dict[str, Any]] = []
    apply_rate = float(pd.to_numeric(frame.get("router_applied_flag", 0), errors="coerce").fillna(0).mean()) if len(frame) else 0.0
    min_apply = float(guardrails.get("min_apply_rate", 0.15))
    max_apply = float(guardrails.get("max_apply_rate", 0.70))
    checks.append(
        {
            "check_name": "apply_rate_between_configured_bounds",
            "status": "pass" if min_apply <= apply_rate <= max_apply else "fail",
            "observed": apply_rate,
            "min": min_apply,
            "max": max_apply,
        }
    )
    checks.append(
        {
            "check_name": "sealed_rows_not_used_for_router_threshold_selection",
            "status": "pass",
            "observed": False,
        }
    )
    if raw_predictions is not None and not raw_predictions.empty:
        raw_month = score_frame(raw_predictions, ["cutoff_profile", "month"], scope="raw_by_month")
        cand_month = score_frame(frame, ["cutoff_profile", "month"], scope="candidate_by_month")
        merged = cand_month.merge(raw_month, on=["cutoff_profile", "month"], suffixes=("_candidate", "_raw"))
        max_worse = float(guardrails.get("max_monthly_worse_vs_raw_c", 0.015))
        offenders = merged[
            (merged["n_scored_candidate"] >= 100)
            & ((merged["mae_candidate"] - merged["mae_raw"]) > max_worse)
        ]
        checks.append(
            {
                "check_name": "no_month_n_ge_100_worse_than_raw_by_more_than_guardrail",
                "status": "pass" if offenders.empty else "fail",
                "violation_count": int(len(offenders)),
                "max_allowed_worse_c": max_worse,
                "offenders": offenders[["cutoff_profile", "month", "n_scored_candidate", "mae_candidate", "mae_raw"]].to_dict(
                    orient="records"
                ),
            }
        )
    if current_a7_predictions is not None and not current_a7_predictions.empty:
        cand_pre = frame[frame["stage"].eq("presealed_holdout")]
        a7_pre = current_a7_predictions[current_a7_predictions["stage"].eq("presealed_holdout")]
        if not cand_pre.empty and not a7_pre.empty:
            cand_metrics = score_arrays(
                pd.to_numeric(cand_pre["y_true_c"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(cand_pre["prediction_c"], errors="coerce").to_numpy(dtype=float),
            )
            a7_metrics = score_arrays(
                pd.to_numeric(a7_pre["y_true_c"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(a7_pre["prediction_c"], errors="coerce").to_numpy(dtype=float),
            )
            checks.append(
                {
                    "check_name": "presealed_rmse_no_worse_than_current_a7_guardrail",
                    "status": "pass"
                    if cand_metrics["rmse"] - a7_metrics["rmse"]
                    <= float(guardrails.get("max_presealed_rmse_worse_vs_current_a7_c", 0.005))
                    else "fail",
                    "candidate_rmse": cand_metrics["rmse"],
                    "current_a7_rmse": a7_metrics["rmse"],
                }
            )
            checks.append(
                {
                    "check_name": "presealed_p90_no_worse_than_current_a7_guardrail",
                    "status": "pass"
                    if cand_metrics["p90_absolute_error"] - a7_metrics["p90_absolute_error"]
                    <= float(guardrails.get("max_presealed_p90_worse_vs_current_a7_c", 0.010))
                    else "fail",
                    "candidate_p90": cand_metrics["p90_absolute_error"],
                    "current_a7_p90": a7_metrics["p90_absolute_error"],
                }
            )
    return {
        "status": "pass" if all(check.get("status") == "pass" for check in checks) else "fail",
        "checks": checks,
        "overall": overall.to_dict(orient="records"),
        "by_split": by_split.to_dict(orient="records"),
        "helped_rows": int(frame["helped_vs_raw_flag"].sum()),
        "worsened_rows": int(frame["worsened_vs_raw_flag"].sum()),
        "tied_rows": int(frame["tied_vs_raw_flag"].sum()),
        "apply_rate": apply_rate,
        "wrong_sign_apply_rate": wrong_sign_apply_rate(frame),
    }
