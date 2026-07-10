"""Reporting helpers for residual-ML research artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_missingness_report(matrix: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in matrix.groupby("split", dropna=False):
        for feature in feature_names:
            if feature not in group:
                continue
            rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "missing_pct": float(group[feature].isna().mean() * 100.0),
                    "non_null_count": int(group[feature].notna().sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "split"]).reset_index(drop=True)


def row_count_audit(
    *,
    targets: pd.DataFrame,
    forecasts: pd.DataFrame,
    hourly: pd.DataFrame,
    matrix: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "target_rows_total": int(len(targets)),
        "target_rows_by_source": targets["label_source"].value_counts(dropna=False).to_dict(),
        "forecast_rows_strict_eligible_total": int(len(forecasts)),
        "forecast_target_dates_with_strict_rows": int(forecasts["target_date"].nunique()) if not forecasts.empty else 0,
        "hourly_rows_loaded_total": int(len(hourly)),
        "joined_rows_total": int(len(matrix)),
        "joined_rows_with_anchor": int(matrix["forecast_selector_status"].eq("selected").sum()),
        "joined_rows_without_anchor": int((~matrix["forecast_selector_status"].eq("selected")).sum()),
        "scored_prediction_rows": int(len(predictions)),
    }
    audit["joined_rows_by_cutoff"] = matrix.groupby("cutoff_profile").size().astype(int).to_dict()
    audit["anchor_rows_by_cutoff"] = matrix[matrix["forecast_selector_status"].eq("selected")].groupby("cutoff_profile").size().astype(int).to_dict()
    if not predictions.empty:
        audit["prediction_rows_by_model"] = predictions.groupby("model_id").size().astype(int).to_dict()
        audit["prediction_rows_by_stage"] = predictions.groupby("stage").size().astype(int).to_dict()
    return audit


def source_eligibility_audit(matrix: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "target_date",
        "cutoff_profile",
        "cutoff_at_hkt",
        "cutoff_at_utc",
        "forecast_selector_status",
        "selected_forecast_source_url",
        "selected_forecast_issue_at_hkt",
        "selected_forecast_issue_at_utc",
        "eligible_forecast_count",
        "latest_hourly_dispatch_at_hkt_used",
        "latest_hourly_observation_at_hkt_used",
        "target_history_max_source_date",
    ]
    return matrix[[col for col in cols if col in matrix.columns]].copy()


def artifact_manifest(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + display.to_csv(index=False).strip() + "\n```"


def final_model_card(
    *,
    promotion: dict[str, Any],
    scoreboard: pd.DataFrame,
    leakage_status: str,
    feature_count: int,
    catboost_status: str,
) -> str:
    primary_rows = scoreboard[
        scoreboard["cutoff_profile"].eq("tminus1_2359")
        & scoreboard["model_id"].isin(["A0_raw_official", "A7_final_residual_ensemble"])
    ].copy()
    return f"""# HKG Tmax Residual ML Strategy Model Card

Primary benchmark cutoff: T-1 23:59 HKT.
Primary baseline: latest eligible Info.gov LOCAL WEATHER FORECAST max before cutoff.
Primary target: HKO Daily Extract Absolute Daily Max (deg. C).
Primary model target: residual versus official forecast max.
Primary metric: MAE, with RMSE and p90 absolute error as guardrails.
Sealed confirmation rows were not used for model selection.
No post-cutoff forecast or hourly observation was used.
Raw Daily Extract payload rows were not used as predictors.

## Decision

Outcome: `{promotion.get("outcome")}`.

Primary MAE improvement versus raw official at T-1 23:59 HKT: `{promotion.get("mae_improvement")}` C.

If the MAE improvement is below `0.035 C`, this run is classified as `no_promote_cosmetic` and does not claim a meaningful ML edge.

## Implementation Summary

- Feature count in schema: `{feature_count}`.
- Leakage audit status: `{leakage_status}`.
- CatBoost status: `{catboost_status}`.
- Sealed mode reported here: `sealed_blind_mode`; online sealed lag replay was not enabled because row-level Daily Extract publication availability for lagged sealed labels was not proven in the supplied docs.

## Primary Score Rows

{markdown_table(primary_rows, max_rows=20)}

## Notes

The final residual ensemble includes an explicit zero-correction option and a validation-fitted shrinkage scalar. If no non-zero correction passes the no-harm guardrails, the ensemble shrinks back to the raw official forecast.
"""


def residual_diagnostics(predictions: pd.DataFrame, promotion: dict[str, Any]) -> str:
    primary = predictions[
        predictions["cutoff_profile"].eq("tminus1_2359")
        & predictions["model_id"].isin(["A0_raw_official", "A7_final_residual_ensemble"])
    ].copy()
    return f"""# Residual Error Diagnostics

## Promotion Result

`{json.dumps(promotion, indent=2, default=str)}`

## Primary Comparison

{markdown_table(primary.groupby(["model_id", "stage"]).size().reset_index(name="rows"), max_rows=50)}

The raw official top-error decile is used only for post-hoc diagnostics and is not included as a feature.
"""


def next_round_model_card(
    *,
    summary: dict[str, Any],
    scoreboard: pd.DataFrame,
    no_harm_audit: dict[str, Any],
    leakage_audit: dict[str, Any],
    router_thresholds: dict[str, Any],
    feature_count: int,
) -> str:
    primary = scoreboard[
        scoreboard.get("cutoff_profile", pd.Series(dtype=str)).astype(str).eq("tminus1_2359")
    ].copy() if not scoreboard.empty else pd.DataFrame()
    key_rows = primary[
        primary.get("model_id", pd.Series(dtype=str)).isin(
            [
                "A0_raw_official",
                "A7_final_residual_ensemble",
                "C1_pruned_residual_ensemble",
                "C2_selective_router",
                "C3_tail_overlay_router",
            ]
        )
    ] if not primary.empty else pd.DataFrame()
    promotion = summary.get("promotion", {})
    return f"""# HKG Tmax Residual ML Next Round Model Card

Primary cutoff: T-1 23:59 HKT.
Primary target: HKO Daily Extract Absolute Daily Max (deg. C).
Primary baseline: strict Info.gov local lead-1 official forecast max.
Primary next-round question: should residual correction be applied selectively instead of on every row?

## Decision

Outcome: `{promotion.get("decision", "unknown")}`.

Reason: `{promotion.get("reason", "not recorded")}`.

Leakage audit status: `{leakage_audit.get("status")}`.
No-harm audit status: `{no_harm_audit.get("status")}`.
Selected raw feature count: `{feature_count}`.

## Selected Router Threshold

Threshold id: `{router_thresholds.get("threshold_id")}`.

- Expected-benefit threshold: `{router_thresholds.get("threshold_benefit")}`
- Apply-probability threshold: `{router_thresholds.get("threshold_apply")}`
- Sign-probability threshold: `{router_thresholds.get("threshold_sign")}`
- Positive cap: `{router_thresholds.get("positive_cap_c")}` C
- Negative cap: `{router_thresholds.get("negative_cap_c")}` C
- Hard absolute cap: `{router_thresholds.get("hard_abs_cap_c")}` C

## Primary Score Rows

{markdown_table(key_rows, max_rows=30)}

## Notes

Sealed confirmation rows were not used for threshold selection, model selection, feature selection, calibration, or hyperparameter tuning. Raw official-error slices and helped/worsened labels are evaluation-only columns and are not model features.
"""


def next_round_summary_payload(
    *,
    generated_at_utc: str,
    config: dict[str, Any],
    feature_policy: dict[str, Any],
    scoreboard: pd.DataFrame,
    no_harm_audit: dict[str, Any],
    leakage_audit: dict[str, Any],
    router_thresholds: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    primary = scoreboard[scoreboard["cutoff_profile"].eq("tminus1_2359")] if not scoreboard.empty else pd.DataFrame()

    def _mae(model_id: str) -> float | None:
        row = primary[primary["model_id"].eq(model_id)]
        if row.empty:
            return None
        return float(row.iloc[0]["mae"])

    raw_mae = _mae("A0_raw_official")
    a7_mae = _mae("A7_final_residual_ensemble")
    c2_mae = _mae("C2_selective_router")
    c3_mae = _mae("C3_tail_overlay_router")
    decision = "no_promote"
    reason = "missing primary score rows"
    if raw_mae is not None and a7_mae is not None and c2_mae is not None:
        c2_vs_raw = raw_mae - c2_mae
        c2_vs_a7 = a7_mae - c2_mae
        if leakage_audit.get("status") != "pass":
            reason = "leakage audit failed"
        elif no_harm_audit.get("status") != "pass":
            reason = "no-harm audit failed"
        elif c2_vs_raw >= 0.045 and c2_vs_a7 >= 0.010:
            decision = "promote_router_candidate"
            reason = "router passed primary MAE and no-harm gates"
        else:
            reason = f"router edge too small: c2_vs_raw={c2_vs_raw}, c2_vs_a7={c2_vs_a7}"
    return {
        "generated_at_utc": generated_at_utc,
        "config": config,
        "feature_policy": feature_policy,
        "primary_mae": {
            "A0_raw_official": raw_mae,
            "A7_final_residual_ensemble": a7_mae,
            "C2_selective_router": c2_mae,
            "C3_tail_overlay_router": c3_mae,
        },
        "promotion": {"decision": decision, "reason": reason},
        "leakage_status": leakage_audit.get("status"),
        "no_harm_status": no_harm_audit.get("status"),
        "router_threshold_id": router_thresholds.get("threshold_id"),
        "output_dir": str(output_dir),
    }
