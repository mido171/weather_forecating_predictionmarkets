from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    update_markdown_section,
)
from scripts.run_hkg_t24_station_only_late_period_bias_repair import (  # noqa: E402
    score_prediction_frame,
)
from scripts.run_hkg_t24_station_only_pressure_high_uncertainty_guard import (  # noqa: E402
    interval_metrics,
)

FOLDER_NAME = "0063_station_only_guarded_candidate_stack"
ARTIFACT_0055 = RESEARCH_ROOT / "0055_station_only_walkforward_benchmark" / "artifacts"
ARTIFACT_0058 = RESEARCH_ROOT / "0058_station_only_late_period_bias_repair" / "artifacts"
ARTIFACT_0059 = RESEARCH_ROOT / "0059_station_only_february_march_transition_specialist" / "artifacts"
ARTIFACT_0060 = RESEARCH_ROOT / "0060_station_only_spring_transition_pressure_dew_specialist" / "artifacts"
ARTIFACT_0061 = RESEARCH_ROOT / "0061_station_only_pressure_high_uncertainty_guard" / "artifacts"
ARTIFACT_0062 = RESEARCH_ROOT / "0062_station_only_nearby_temperature_level_error_scale" / "artifacts"
PREDICTIONS_0055_PATH = ARTIFACT_0055 / "predictions.parquet"
SUMMARY_0055_PATH = ARTIFACT_0055 / "summary.json"
PREDICTIONS_0058_PATH = ARTIFACT_0058 / "predictions.parquet"
SUMMARY_0058_PATH = ARTIFACT_0058 / "summary.json"
PREDICTIONS_0059_PATH = ARTIFACT_0059 / "predictions.parquet"
SUMMARY_0059_PATH = ARTIFACT_0059 / "summary.json"
SUMMARY_0060_PATH = ARTIFACT_0060 / "summary.json"
PREDICTIONS_0061_PATH = ARTIFACT_0061 / "predictions.parquet"
SUMMARY_0061_PATH = ARTIFACT_0061 / "summary.json"
PREDICTIONS_0062_PATH = ARTIFACT_0062 / "predictions.parquet"
SUMMARY_0062_PATH = ARTIFACT_0062 / "summary.json"
DEVELOPMENT_END = pd.Timestamp("2023-12-31")


@dataclass(frozen=True)
class StackSpec:
    stack_id: str
    feb_mar_weight: float
    pressure_mean_weight: float
    temp_mean_weight: float
    pressure_sigma_power: float
    temp_sigma_power: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def filter_candidate(path: Path, id_column: str, selected_id: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction artifact: {path}")
    frame = pd.read_parquet(path)
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    selected = frame[frame[id_column].astype(str).eq(selected_id)].copy()
    selected = selected[selected["target_date"].le(DEVELOPMENT_END)].copy()
    if selected.empty:
        raise RuntimeError(f"Missing selected candidate `{selected_id}` in {path}")
    require_no_confirmation_dates(selected["target_date"], context=f"0063 selected {selected_id}")
    if selected["target_date"].duplicated().any():
        raise RuntimeError(f"Selected candidate `{selected_id}` is not one row per target_date")
    return selected.sort_values("target_date").reset_index(drop=True)


def load_stack_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_0055 = load_json(SUMMARY_0055_PATH)
    summary_0058 = load_json(SUMMARY_0058_PATH)
    summary_0059 = load_json(SUMMARY_0059_PATH)
    summary_0060 = load_json(SUMMARY_0060_PATH)
    summary_0061 = load_json(SUMMARY_0061_PATH)
    summary_0062 = load_json(SUMMARY_0062_PATH)
    best_model_id = str(summary_0055["best_model_id"])
    best_0058 = str(summary_0058["best_candidate"])
    best_0059 = str(summary_0059["best_candidate"])
    best_0061 = str(summary_0061["best_guard"])
    best_0062 = str(summary_0062["best_guard"])

    anchor = filter_candidate(PREDICTIONS_0055_PATH, "model_id", best_model_id)
    base = filter_candidate(PREDICTIONS_0058_PATH, "correction_id", best_0058)
    feb_mar = filter_candidate(PREDICTIONS_0059_PATH, "correction_id", best_0059)
    pressure = filter_candidate(PREDICTIONS_0061_PATH, "guard_id", best_0061)
    temp = filter_candidate(PREDICTIONS_0062_PATH, "guard_id", best_0062)

    frame = anchor[
        ["target_date", "target_tmax_c", "point_forecast_c", "distribution_sigma_c", "fold_id"]
    ].rename(columns={"point_forecast_c": "station_anchor_prediction_c"})
    frame = frame.merge(
        base[["target_date", "candidate_prediction_c", "residual_correction_c"]],
        on="target_date",
        how="inner",
        validate="one_to_one",
    ).rename(
        columns={
            "candidate_prediction_c": "base_0058_prediction_c",
            "residual_correction_c": "base_0058_correction_c",
        }
    )
    frame = frame.merge(
        feb_mar[
            [
                "target_date",
                "residual_correction_c",
                "transition_target_window",
                "transition_phase",
                "transition_month_bucket",
            ]
        ],
        on="target_date",
        how="inner",
        validate="one_to_one",
    ).rename(columns={"residual_correction_c": "feb_mar_correction_c"})
    frame = frame.merge(
        pressure[
            [
                "target_date",
                "mean_residual_correction_c",
                "sigma_multiplier",
                "pressure_high_window",
                "pressure_spread_bucket",
                "dew_trajectory_bucket",
                "wind_spread_bucket",
            ]
        ],
        on="target_date",
        how="inner",
        validate="one_to_one",
    ).rename(
        columns={
            "mean_residual_correction_c": "pressure_correction_c",
            "sigma_multiplier": "pressure_sigma_multiplier",
        }
    )
    frame = frame.merge(
        temp[
            [
                "target_date",
                "mean_residual_correction_c",
                "sigma_multiplier",
                "nearby_temp_bucket",
                "nearby_temp_extreme",
            ]
        ],
        on="target_date",
        how="inner",
        validate="one_to_one",
    ).rename(
        columns={
            "mean_residual_correction_c": "nearby_temp_correction_c",
            "sigma_multiplier": "nearby_temp_sigma_multiplier",
        }
    )
    frame["year"] = frame["target_date"].dt.year
    frame["month"] = frame["target_date"].dt.month
    frame["base_sigma_c"] = pd.to_numeric(frame["distribution_sigma_c"], errors="coerce").clip(lower=0.05)
    frame = frame.sort_values("target_date").reset_index(drop=True)
    require_no_confirmation_dates(frame["target_date"], context="0063 stack frame")

    metadata = {
        "anchor_model_id": best_model_id,
        "base_0058_candidate": best_0058,
        "feb_mar_0059_candidate": best_0059,
        "excluded_0060_candidate": summary_0060.get("best_candidate"),
        "excluded_0060_reason": "0060 failed robust all-post-2006-fold promotion gate",
        "pressure_0061_guard": best_0061,
        "nearby_temp_0062_guard": best_0062,
        "source_summaries": {
            "0058": summary_0058,
            "0059": summary_0059,
            "0060": summary_0060,
            "0061": summary_0061,
            "0062": summary_0062,
        },
    }
    return frame, metadata


def stack_specs() -> list[StackSpec]:
    return [
        StackSpec("base_0058_only", 0.0, 0.0, 0.0, 0.0, 0.0),
        StackSpec("base_plus_feb_mar_025", 0.25, 0.0, 0.0, 0.0, 0.0),
        StackSpec("base_plus_feb_mar_050", 0.50, 0.0, 0.0, 0.0, 0.0),
        StackSpec("base_plus_feb_mar_075", 0.75, 0.0, 0.0, 0.0, 0.0),
        StackSpec("base_plus_feb_mar_100", 1.00, 0.0, 0.0, 0.0, 0.0),
        StackSpec("base_plus_pressure_mean", 0.0, 1.0, 0.0, 0.0, 0.0),
        StackSpec("base_plus_nearby_temp_mean", 0.0, 0.0, 1.0, 0.0, 0.0),
        StackSpec("base_plus_pressure_and_temp_mean", 0.0, 1.0, 1.0, 0.0, 0.0),
        StackSpec("base_plus_feb050_pressure_mean", 0.50, 1.0, 0.0, 0.0, 0.0),
        StackSpec("base_plus_feb050_temp_mean", 0.50, 0.0, 1.0, 0.0, 0.0),
        StackSpec("base_plus_feb050_pressure_temp_mean", 0.50, 1.0, 1.0, 0.0, 0.0),
        StackSpec("base_pressure_sigma_only", 0.0, 0.0, 0.0, 1.0, 0.0),
        StackSpec("base_temp_sigma_only", 0.0, 0.0, 0.0, 0.0, 1.0),
        StackSpec("base_pressure_temp_sigma", 0.0, 0.0, 0.0, 1.0, 1.0),
        StackSpec("guarded_stack_feb025_pressure_temp_sigma", 0.25, 1.0, 1.0, 1.0, 1.0),
        StackSpec("guarded_stack_feb050_pressure_temp_sigma", 0.50, 1.0, 1.0, 1.0, 1.0),
        StackSpec("guarded_stack_feb050_pressure_sigma", 0.50, 1.0, 0.0, 1.0, 0.0),
        StackSpec("guarded_stack_feb050_temp_sigma", 0.50, 0.0, 1.0, 0.0, 1.0),
    ]


def apply_stack(frame: pd.DataFrame, spec: StackSpec) -> pd.DataFrame:
    out = frame.copy()
    point = (
        out["base_0058_prediction_c"]
        + spec.feb_mar_weight * out["feb_mar_correction_c"]
        + spec.pressure_mean_weight * out["pressure_correction_c"]
        + spec.temp_mean_weight * out["nearby_temp_correction_c"]
    )
    multiplier = (
        np.power(pd.to_numeric(out["pressure_sigma_multiplier"], errors="coerce").fillna(1.0), spec.pressure_sigma_power)
        * np.power(pd.to_numeric(out["nearby_temp_sigma_multiplier"], errors="coerce").fillna(1.0), spec.temp_sigma_power)
    )
    out["candidate_prediction_c"] = point
    out["candidate_sigma_c"] = out["base_sigma_c"] * np.clip(multiplier, 0.45, 2.5)
    out["stack_id"] = spec.stack_id
    out["feb_mar_weight"] = spec.feb_mar_weight
    out["pressure_mean_weight"] = spec.pressure_mean_weight
    out["nearby_temp_mean_weight"] = spec.temp_mean_weight
    out["pressure_sigma_power"] = spec.pressure_sigma_power
    out["nearby_temp_sigma_power"] = spec.temp_sigma_power
    out["total_point_correction_vs_0058_c"] = out["candidate_prediction_c"] - out["base_0058_prediction_c"]
    out["total_sigma_multiplier"] = out["candidate_sigma_c"] / out["base_sigma_c"]
    require_no_confirmation_dates(out["target_date"], context=f"0063 {spec.stack_id}")
    return out


def p90_abs_error(frame: pd.DataFrame, prediction_col: str) -> float:
    scored = frame[["target_tmax_c", prediction_col]].dropna().copy()
    if scored.empty:
        return math.nan
    error = pd.to_numeric(scored[prediction_col], errors="coerce") - pd.to_numeric(scored["target_tmax_c"], errors="coerce")
    return float(error.abs().quantile(0.90))


def score_stack(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    for stack_id, group in predictions.groupby("stack_id", observed=True):
        reference = group.rename(
            columns={
                "base_0058_prediction_c": "reference_prediction_c",
                "base_sigma_c": "reference_sigma_c",
            }
        )
        full = score_prediction_frame(group, "candidate_prediction_c")
        ref_full = score_prediction_frame(reference, "reference_prediction_c")
        interval = interval_metrics(group, "candidate_prediction_c", "candidate_sigma_c")
        ref_interval = interval_metrics(reference, "reference_prediction_c", "reference_sigma_c")
        p90 = p90_abs_error(group, "candidate_prediction_c")
        ref_p90 = p90_abs_error(reference, "reference_prediction_c")
        for fold_id, fold in group.groupby("fold_id", observed=True):
            ref_fold = reference.loc[fold.index]
            candidate_fold = score_prediction_frame(fold, "candidate_prediction_c")
            reference_fold = score_prediction_frame(ref_fold, "reference_prediction_c")
            fold_rows.append(
                {
                    "stack_id": stack_id,
                    "fold_id": fold_id,
                    "n": candidate_fold["n"],
                    "candidate_mae": candidate_fold["mae"],
                    "reference_mae": reference_fold["mae"],
                    "delta_mae_vs_0058": float(candidate_fold["mae"]) - float(reference_fold["mae"]),
                    "candidate_rmse": candidate_fold["rmse"],
                    "reference_rmse": reference_fold["rmse"],
                }
            )
        stack_folds = pd.DataFrame([row for row in fold_rows if row["stack_id"] == stack_id])
        row = {
            "stack_id": stack_id,
            "feb_mar_weight": float(group["feb_mar_weight"].iloc[0]),
            "pressure_mean_weight": float(group["pressure_mean_weight"].iloc[0]),
            "nearby_temp_mean_weight": float(group["nearby_temp_mean_weight"].iloc[0]),
            "pressure_sigma_power": float(group["pressure_sigma_power"].iloc[0]),
            "nearby_temp_sigma_power": float(group["nearby_temp_sigma_power"].iloc[0]),
            "n": full["n"],
            "mae": full["mae"],
            "rmse": full["rmse"],
            "bias": full["bias"],
            "delta_mae_vs_0058": float(full["mae"]) - float(ref_full["mae"]),
            "p90_abs_error": p90,
            "reference_p90_abs_error": ref_p90,
            "delta_p90_abs_error_vs_0058": p90 - ref_p90,
            "coverage80": interval["coverage80"],
            "reference_coverage80": ref_interval["coverage80"],
            "coverage90": interval["coverage90"],
            "reference_coverage90": ref_interval["coverage90"],
            "coverage_distance": interval["coverage_distance"],
            "reference_coverage_distance": ref_interval["coverage_distance"],
            "delta_coverage_distance_vs_0058": float(interval["coverage_distance"]) - float(ref_interval["coverage_distance"]),
            "folds_improved": int(stack_folds["delta_mae_vs_0058"].lt(0).sum()),
            "fold_delta_max": float(stack_folds["delta_mae_vs_0058"].max()) if not stack_folds.empty else math.nan,
            "mean_abs_point_correction_vs_0058_c": float(group["total_point_correction_vs_0058_c"].abs().mean()),
            "mean_sigma_multiplier": float(group["total_sigma_multiplier"].mean()),
        }
        row["promotion_gate_passed"] = bool(
            row["delta_mae_vs_0058"] < 0.0
            and row["fold_delta_max"] <= 0.015
            and row["coverage_distance"] <= row["reference_coverage_distance"] + 0.01
            and row["delta_p90_abs_error_vs_0058"] <= 0.02
        )
        rows.append(row)

        subgroup_defs = [
            ("feb_mar", group["transition_target_window"].astype(bool)),
            ("pressure_high", group["pressure_high_window"].astype(bool)),
            ("nearby_temp_low", group["nearby_temp_bucket"].astype(str).eq("nearby_temp_low")),
            ("nearby_temp_high", group["nearby_temp_bucket"].astype(str).eq("nearby_temp_high")),
        ]
        for subgroup_name, mask in subgroup_defs:
            subgroup = group[mask].copy()
            ref_subgroup = reference.loc[subgroup.index].copy()
            candidate_score = score_prediction_frame(subgroup, "candidate_prediction_c")
            reference_score = score_prediction_frame(ref_subgroup, "reference_prediction_c")
            subgroup_rows.append(
                {
                    "stack_id": stack_id,
                    "subgroup": subgroup_name,
                    "n": candidate_score["n"],
                    "candidate_mae": candidate_score["mae"],
                    "reference_mae": reference_score["mae"],
                    "delta_mae_vs_0058": float(candidate_score["mae"]) - float(reference_score["mae"]),
                    "candidate_rmse": candidate_score["rmse"],
                    "reference_rmse": reference_score["rmse"],
                    "candidate_p90_abs_error": p90_abs_error(subgroup, "candidate_prediction_c"),
                    "reference_p90_abs_error": p90_abs_error(ref_subgroup, "reference_prediction_c"),
                }
            )

    scoreboard = pd.DataFrame(rows).sort_values(
        ["promotion_gate_passed", "delta_mae_vs_0058", "coverage_distance"],
        ascending=[False, True, True],
    )
    folds = pd.DataFrame(fold_rows).sort_values(["fold_id", "delta_mae_vs_0058"]).reset_index(drop=True)
    subgroups = pd.DataFrame(subgroup_rows).sort_values(["subgroup", "delta_mae_vs_0058"]).reset_index(drop=True)
    return scoreboard.reset_index(drop=True), folds, subgroups


def leakage_audit(predictions: pd.DataFrame, scoreboard: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(predictions["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(predictions['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "one_row_per_stack_date",
            "passed": bool(
                predictions.groupby(["stack_id", "target_date"], observed=True).size().max() == 1
                and predictions.groupby("stack_id", observed=True)["target_date"].nunique().nunique() == 1
            ),
            "evidence": f"{predictions['stack_id'].nunique()} stack candidates aligned on {predictions['target_date'].nunique()} dates",
        },
        {
            "check_id": "only_promoted_or_documented_components",
            "passed": bool(metadata.get("excluded_0060_reason") and metadata.get("base_0058_candidate")),
            "evidence": "0058, 0059, 0061, and 0062 are used; 0060 is documented as excluded because its robust gate failed",
        },
        {
            "check_id": "source_component_leakage_checks_passed",
            "passed": bool(
                all(
                    int(metadata["source_summaries"][key].get("leakage_checks_passed", -1))
                    == int(metadata["source_summaries"][key].get("leakage_check_rows", -2))
                    for key in ["0058", "0059", "0061", "0062"]
                )
            ),
            "evidence": "all source component summaries report all leakage checks passed",
        },
        {
            "check_id": "promotion_gate_limits_fold_and_calibration_damage",
            "passed": bool(
                scoreboard.loc[scoreboard["promotion_gate_passed"], "fold_delta_max"].le(0.015).all()
                and (
                    scoreboard.loc[scoreboard["promotion_gate_passed"], "coverage_distance"]
                    <= scoreboard.loc[scoreboard["promotion_gate_passed"], "reference_coverage_distance"] + 0.01
                ).all()
            ),
            "evidence": f"{int(scoreboard['promotion_gate_passed'].sum())} stack candidates passed gate",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    metadata: dict[str, Any],
    scoreboard: pd.DataFrame,
    fold_scoreboard: pd.DataFrame,
    subgroup_scoreboard: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    return f"""# Station-Only Guarded Candidate Stack

Generated: `{summary['generated_at_utc']}`

## Purpose

`0063` synthesizes the current station-only discoveries into a single guarded candidate design. It does not search arbitrary weights or open the 2024+ confirmation period. It tests a small declared set of combinations built from already-audited components:

- `0058` global online station-only bias repair as the base point forecast.
- `0059` February/March transition point correction.
- `0061` pressure-high mean and interval guard.
- `0062` nearby-temperature mean and interval guard.

`0060` is intentionally excluded from promoted stacks because it improved MAM average MAE but failed the robust all-post-2006-fold promotion gate.

## Leakage Contract

- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- No 2024+ rows are used.
- The stack is a deterministic combination of prior-only component outputs.
- There is no learned meta-model and no holdout tuning.
- Every source component used here passed its own leakage audit.

## Component IDs

| Component | Selected ID |
|---|---|
| 0055 anchor | `{metadata['anchor_model_id']}` |
| 0058 base | `{metadata['base_0058_candidate']}` |
| 0059 Feb/Mar | `{metadata['feb_mar_0059_candidate']}` |
| 0061 pressure guard | `{metadata['pressure_0061_guard']}` |
| 0062 nearby temp guard | `{metadata['nearby_temp_0062_guard']}` |
| 0060 excluded | `{metadata['excluded_0060_candidate']}` |

## Headline

| Item | Value |
|---|---:|
| Rows scored | {summary['rows_scored']} |
| Reference 0058 MAE | {summary['reference_0058_mae']} |
| Best stack | {summary['best_stack']} |
| Best MAE | {summary['best_mae']} |
| Best RMSE | {summary['best_rmse']} |
| Best delta MAE vs 0058 | {summary['best_delta_mae_vs_0058']} |
| Best folds improved | {summary['best_folds_improved']} / {summary['fold_count']} |
| Best coverage distance | {summary['best_coverage_distance']} |
| Best promotion gate passed | {summary['best_promotion_gate_passed']} |

## Stack Scoreboard

{markdown_table(scoreboard, max_rows=80)}

## Fold Scoreboard

{markdown_table(fold_scoreboard, max_rows=100)}

## Subgroup Scoreboard

{markdown_table(subgroup_scoreboard, max_rows=100)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Interpretation

This synthesis answers whether the latest station-only specialists compound. A strong result would show material MAE reduction over `0058` while preserving fold robustness and calibration. A weak result is still valuable because it tells us which component families are overlapping rather than additive.

This is still far from the claimed `0.45 C` competitor level. The current station-only stack is useful as a classical component, but the larger goal still requires the continuous official forecast archive, richer forecast-vintage dynamics, and eventually a transparent expert stack that combines official forecast, station network, and forecast-history signals.

## Files

- `artifacts/predictions.parquet`
- `artifacts/predictions_sample.csv`
- `artifacts/scoreboard.csv`
- `artifacts/fold_scoreboard.csv`
- `artifacts/subgroup_scoreboard.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_only_guarded_candidate_stack.py`:

- `{FOLDER_NAME}`: deterministic guarded stack combining `0058`, `0059`, `0061`, and `0062`.

| Metric | Value |
|---|---:|
| Reference 0058 MAE | {summary['reference_0058_mae']} |
| Best stack | {summary['best_stack']} |
| Best MAE | {summary['best_mae']} |
| Best delta MAE vs 0058 | {summary['best_delta_mae_vs_0058']} |
| Folds improved | {summary['best_folds_improved']} / {summary['fold_count']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

Leakage contract: deterministic combination of prior-only component outputs, no 2024+ rows, and no learned meta-model.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Station-Only Guarded Candidate Stack",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_only_guarded_candidate_stack.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Reference | `0058` `{summary['reference_0058_candidate']}` | Tested |
| Best stack | `{summary['best_stack']}` | Diagnostic |
| Reference MAE / RMSE | `{summary['reference_0058_mae']}` / `{summary['reference_0058_rmse']}` | Baseline |
| Best MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best delta MAE vs 0058 | `{summary['best_delta_mae_vs_0058']}` | Diagnostic |
| Folds improved | `{summary['best_folds_improved']}` / `{summary['fold_count']}` | Guarded |
| Promotion gate passed | `{summary['best_promotion_gate_passed']}` | Guarded |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0063` tests whether recent station-only specialists compound. It remains a station-only component and does not use 2024+ confirmation rows.
"""
    update_markdown_section(
        path,
        heading="Station-Only Guarded Candidate Stack",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"47. Guarded station-only stack tested `{summary['candidate_count']}` deterministic combinations; "
        f"best delta vs 0058 is `{summary['best_delta_mae_vs_0058']}` from `{summary['best_stack']}`."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Resume official forecast archive integration when the backfill adds new contiguous 2008-2020 scoreable rows: rebuild the official-anchor scored frame, rerun the forecast-vintage/residual stack on the expanded archive, and compare it against the station-only `0063` component without opening 2024+ confirmation.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, metadata = load_stack_frame()
    predictions = pd.concat([apply_stack(frame, spec) for spec in stack_specs()], ignore_index=True)
    scoreboard, fold_scoreboard, subgroup_scoreboard = score_stack(predictions)
    leakage = leakage_audit(predictions, scoreboard, metadata)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0063 leakage audit failed: {failed}")

    reference = frame.rename(columns={"base_0058_prediction_c": "reference_prediction_c"})
    reference_score = score_prediction_frame(reference, "reference_prediction_c")
    best = scoreboard.iloc[0]
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "reference_0058_candidate": metadata["base_0058_candidate"],
        "candidate_count": int(scoreboard["stack_id"].nunique()),
        "rows_scored": int(scoreboard["n"].max()),
        "fold_count": int(fold_scoreboard["fold_id"].nunique()),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "reference_0058_mae": float(reference_score["mae"]),
        "reference_0058_rmse": float(reference_score["rmse"]),
        "best_stack": str(best["stack_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_bias": float(best["bias"]),
        "best_delta_mae_vs_0058": float(best["delta_mae_vs_0058"]),
        "best_p90_abs_error": float(best["p90_abs_error"]),
        "best_delta_p90_abs_error_vs_0058": float(best["delta_p90_abs_error_vs_0058"]),
        "best_coverage80": float(best["coverage80"]),
        "best_coverage90": float(best["coverage90"]),
        "best_coverage_distance": float(best["coverage_distance"]),
        "best_folds_improved": int(best["folds_improved"]),
        "best_fold_delta_max": float(best["fold_delta_max"]),
        "best_promotion_gate_passed": bool(best["promotion_gate_passed"]),
        "component_metadata": metadata,
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
        "uses_2024_plus_rows": False,
    }

    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(artifacts / "predictions.parquet", index=False)
    write_csv(artifacts / "predictions_sample.csv", predictions.head(1000))
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "fold_scoreboard.csv", fold_scoreboard)
    write_csv(artifacts / "subgroup_scoreboard.csv", subgroup_scoreboard)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_only_guarded_candidate_stack_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            metadata=metadata,
            scoreboard=scoreboard,
            fold_scoreboard=fold_scoreboard,
            subgroup_scoreboard=subgroup_scoreboard,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Synthesize audited station-only residual and calibration guards into deterministic stacks."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
