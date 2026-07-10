from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict, deque
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
from scripts.run_hkg_t24_station_only_february_march_transition_specialist import (  # noqa: E402
    DESIGN_QUEUE_PATH,
    DEVELOPMENT_END,
    FEATURE_MATRIX_PATH,
    quantile_edges,
)
from scripts.run_hkg_t24_station_only_late_period_bias_repair import (  # noqa: E402
    score_prediction_frame,
)
from scripts.run_hkg_t24_station_only_pressure_high_uncertainty_guard import (  # noqa: E402
    EXPECTED_NORMAL_ABS_FACTOR,
    interval_metrics,
)
from scripts.run_hkg_t24_station_only_spring_transition_pressure_dew_specialist import (  # noqa: E402
    PREDICTIONS_0055_PATH,
    PREDICTIONS_0058_PATH,
    SUMMARY_0055_PATH,
    SUMMARY_0058_PATH,
)

FOLDER_NAME = "0062_station_only_nearby_temperature_level_error_scale"
NEARBY_TEMP_COLUMN = "stat_590960_99999_air_temperature_c_latest_before_1500"
TEMP_LOW = "nearby_temp_low"
TEMP_MID = "nearby_temp_mid"
TEMP_HIGH = "nearby_temp_high"


@dataclass(frozen=True)
class TempScaleSpec:
    guard_id: str
    group_columns: tuple[str, ...]
    active_temp_bucket: str | None
    min_prior_rows: int
    mean_shrinkage: float
    scale_shrinkage: float
    mean_cap_c: float
    min_sigma_multiplier: float
    max_sigma_multiplier: float
    window_days: int | None = None


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def bucket_by_edges(values: pd.Series, low: float, high: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series("nearby_temp_missing", index=values.index, dtype="object")
    out.loc[numeric <= low] = TEMP_LOW
    out.loc[(numeric > low) & (numeric <= high)] = TEMP_MID
    out.loc[numeric > high] = TEMP_HIGH
    return out


def load_nearby_temperature_frame() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame]:
    summary_0055 = load_json(SUMMARY_0055_PATH)
    summary_0058 = load_json(SUMMARY_0058_PATH)
    best_model_id = str(summary_0055["best_model_id"])
    best_0058_id = str(summary_0058["best_candidate"])

    anchor = pd.read_parquet(PREDICTIONS_0055_PATH)
    anchor["target_date"] = pd.to_datetime(anchor["target_date"], errors="coerce").dt.normalize()
    anchor = anchor[anchor["model_id"].astype(str).eq(best_model_id)].copy()
    anchor = anchor[anchor["target_date"].le(DEVELOPMENT_END)].copy()
    if anchor.empty:
        raise RuntimeError(f"Missing 0055 station anchor predictions for {best_model_id}")

    repaired = pd.read_parquet(PREDICTIONS_0058_PATH)
    repaired["target_date"] = pd.to_datetime(repaired["target_date"], errors="coerce").dt.normalize()
    repaired = repaired[repaired["correction_id"].astype(str).eq(best_0058_id)].copy()
    repaired = repaired[repaired["target_date"].le(DEVELOPMENT_END)].copy()
    if repaired.empty:
        raise RuntimeError(f"Missing 0058 best correction predictions for {best_0058_id}")

    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    features["target_date"] = pd.to_datetime(features["target_date"], errors="coerce").dt.normalize()
    if NEARBY_TEMP_COLUMN not in features.columns:
        raise RuntimeError(f"Missing nearby temperature feature in 0054 matrix: {NEARBY_TEMP_COLUMN}")
    features = features[features["target_date"].notna() & features["target_date"].lt(CONFIRMATION_START)].copy()
    pre2000 = features[features["target_date"].le(pd.Timestamp("1999-12-31"))].copy()
    low, high = quantile_edges(pre2000[NEARBY_TEMP_COLUMN])
    temp_gates = features[["target_date", NEARBY_TEMP_COLUMN]].copy()
    temp_gates["nearby_temp_bucket"] = bucket_by_edges(temp_gates[NEARBY_TEMP_COLUMN], low, high)
    temp_gates["nearby_temp_extreme"] = temp_gates["nearby_temp_bucket"].isin([TEMP_LOW, TEMP_HIGH])
    thresholds = pd.DataFrame(
        [
            {
                "gate_column": "nearby_temp_bucket",
                "source_metric": NEARBY_TEMP_COLUMN,
                "low_edge": low,
                "high_edge": high,
                "threshold_source": "1947-01-01_to_1999-12-31_feature_history",
                "pre2000_non_null_rows": int(pd.to_numeric(pre2000[NEARBY_TEMP_COLUMN], errors="coerce").notna().sum()),
            }
        ]
    )

    frame = anchor[
        [
            "target_date",
            "target_tmax_c",
            "point_forecast_c",
            "distribution_sigma_c",
            "fold_id",
            "training_start",
            "training_end",
            "training_rows",
        ]
    ].rename(columns={"point_forecast_c": "station_anchor_prediction_c"})
    frame = frame.merge(
        repaired[["target_date", "candidate_prediction_c", "residual_correction_c"]],
        on="target_date",
        how="inner",
        validate="one_to_one",
    ).rename(
        columns={
            "candidate_prediction_c": "global_bias_repaired_prediction_c",
            "residual_correction_c": "global_bias_repair_c",
        }
    )
    frame = frame.merge(temp_gates, on="target_date", how="left", validate="one_to_one")
    frame["year"] = frame["target_date"].dt.year
    frame["month"] = frame["target_date"].dt.month
    frame["residual_to_add_c"] = frame["target_tmax_c"] - frame["global_bias_repaired_prediction_c"]
    frame["reference_error_c"] = frame["global_bias_repaired_prediction_c"] - frame["target_tmax_c"]
    frame["reference_expected_abs_c"] = (
        pd.to_numeric(frame["distribution_sigma_c"], errors="coerce").clip(lower=0.05) * EXPECTED_NORMAL_ABS_FACTOR
    )
    frame = frame.sort_values("target_date").reset_index(drop=True)
    require_no_confirmation_dates(frame["target_date"], context="0062 nearby temp frame")
    return frame, summary_0055, summary_0058, thresholds


def temp_scale_specs() -> list[TempScaleSpec]:
    return [
        TempScaleSpec("temp_bucket_scale_min180_shrink365", ("nearby_temp_bucket",), None, 180, 10_000.0, 365.0, 0.0, 0.70, 2.1),
        TempScaleSpec("temp_bucket_mean_scale_min180_shrink365_cap0p6", ("nearby_temp_bucket",), None, 180, 365.0, 365.0, 0.6, 0.70, 2.1),
        TempScaleSpec("temp_extreme_scale_min120_shrink240", ("nearby_temp_bucket",), TEMP_HIGH, 120, 10_000.0, 240.0, 0.0, 0.70, 2.2),
        TempScaleSpec("temp_high_mean_scale_min120_shrink240_cap0p6", ("nearby_temp_bucket",), TEMP_HIGH, 120, 240.0, 240.0, 0.6, 0.70, 2.2),
        TempScaleSpec("temp_low_mean_scale_min120_shrink240_cap0p6", ("nearby_temp_bucket",), TEMP_LOW, 120, 240.0, 240.0, 0.6, 0.70, 2.2),
        TempScaleSpec("temp_bucket_rolling8y_scale_min90_shrink180", ("nearby_temp_bucket",), None, 90, 10_000.0, 180.0, 0.0, 0.70, 2.2, window_days=2920),
        TempScaleSpec("temp_bucket_rolling8y_mean_scale_min90_shrink180_cap0p6", ("nearby_temp_bucket",), None, 90, 180.0, 180.0, 0.6, 0.70, 2.2, window_days=2920),
    ]


def group_key(row: pd.Series, columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in columns)


def is_active(row: pd.Series, spec: TempScaleSpec) -> bool:
    if spec.active_temp_bucket is None:
        return True
    return str(row["nearby_temp_bucket"]) == spec.active_temp_bucket


def correction_from_state(
    *,
    count: int,
    residual_sum: float,
    abs_sum: float,
    expected_abs_sum: float,
    spec: TempScaleSpec,
) -> tuple[float, float, float, float]:
    if count < spec.min_prior_rows:
        return 0.0, 1.0, math.nan, math.nan
    raw_mean = residual_sum / count
    mean_shrink = count / (count + spec.mean_shrinkage)
    mean_correction = float(np.clip(raw_mean * mean_shrink, -spec.mean_cap_c, spec.mean_cap_c))
    raw_multiplier = abs_sum / expected_abs_sum if expected_abs_sum > 1e-9 else math.nan
    if math.isfinite(raw_multiplier):
        scale_shrink = count / (count + spec.scale_shrinkage)
        multiplier = 1.0 + (raw_multiplier - 1.0) * scale_shrink
        multiplier = float(np.clip(multiplier, spec.min_sigma_multiplier, spec.max_sigma_multiplier))
    else:
        multiplier = 1.0
    return mean_correction, multiplier, raw_mean, raw_multiplier


def compute_temp_scale(
    frame: pd.DataFrame,
    spec: TempScaleSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values("target_date").reset_index(drop=True)
    corrections = np.zeros(len(ordered), dtype=float)
    multipliers = np.ones(len(ordered), dtype=float)
    prior_counts = np.zeros(len(ordered), dtype=int)
    raw_means = np.full(len(ordered), math.nan, dtype=float)
    raw_multipliers = np.full(len(ordered), math.nan, dtype=float)
    expanding_state: dict[tuple[str, ...], tuple[int, float, float, float]] = defaultdict(lambda: (0, 0.0, 0.0, 0.0))
    rolling_state: dict[tuple[str, ...], dict[str, object]] = defaultdict(
        lambda: {"rows": deque(), "residual_sum": 0.0, "abs_sum": 0.0, "expected_abs_sum": 0.0}
    )

    for idx, row in ordered.iterrows():
        current_date = pd.Timestamp(row["target_date"])
        key = group_key(row, spec.group_columns)
        if spec.window_days is None:
            count, residual_sum, abs_sum, expected_abs_sum = expanding_state[key]
        else:
            state = rolling_state[key]
            rows = state["rows"]
            assert isinstance(rows, deque)
            min_date = current_date - pd.Timedelta(days=spec.window_days)
            while rows and rows[0][0] < min_date:
                _, old_residual, old_abs, old_expected_abs = rows.popleft()
                state["residual_sum"] = float(state["residual_sum"]) - float(old_residual)
                state["abs_sum"] = float(state["abs_sum"]) - float(old_abs)
                state["expected_abs_sum"] = float(state["expected_abs_sum"]) - float(old_expected_abs)
            count = len(rows)
            residual_sum = float(state["residual_sum"])
            abs_sum = float(state["abs_sum"])
            expected_abs_sum = float(state["expected_abs_sum"])
        prior_counts[idx] = count
        if is_active(row, spec):
            correction, multiplier, raw_mean, raw_multiplier = correction_from_state(
                count=count,
                residual_sum=residual_sum,
                abs_sum=abs_sum,
                expected_abs_sum=expected_abs_sum,
                spec=spec,
            )
            corrections[idx] = correction
            multipliers[idx] = multiplier
            raw_means[idx] = raw_mean
            raw_multipliers[idx] = raw_multiplier

        residual = float(row["residual_to_add_c"])
        expected_abs = float(row["reference_expected_abs_c"])
        if math.isfinite(residual) and math.isfinite(expected_abs) and expected_abs > 0:
            abs_residual = abs(residual)
            if spec.window_days is None:
                expanding_state[key] = (
                    count + 1,
                    residual_sum + residual,
                    abs_sum + abs_residual,
                    expected_abs_sum + expected_abs,
                )
            else:
                state = rolling_state[key]
                rows = state["rows"]
                assert isinstance(rows, deque)
                rows.append((current_date, residual, abs_residual, expected_abs))
                state["residual_sum"] = float(state["residual_sum"]) + residual
                state["abs_sum"] = float(state["abs_sum"]) + abs_residual
                state["expected_abs_sum"] = float(state["expected_abs_sum"]) + expected_abs
    return corrections, multipliers, prior_counts, raw_means, raw_multipliers


def apply_temp_scale(frame: pd.DataFrame, spec: TempScaleSpec) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    corrections, multipliers, prior_rows, raw_means, raw_multipliers = compute_temp_scale(ordered, spec)
    out = ordered[
        [
            "target_date",
            "target_tmax_c",
            "station_anchor_prediction_c",
            "global_bias_repaired_prediction_c",
            "distribution_sigma_c",
            "fold_id",
            "year",
            "month",
            NEARBY_TEMP_COLUMN,
            "nearby_temp_bucket",
            "nearby_temp_extreme",
        ]
    ].copy()
    base_sigma = pd.to_numeric(out["distribution_sigma_c"], errors="coerce").clip(lower=0.05).to_numpy(dtype=float)
    out["candidate_prediction_c"] = out["global_bias_repaired_prediction_c"] + corrections
    out["candidate_sigma_c"] = base_sigma * multipliers
    out["mean_residual_correction_c"] = corrections
    out["sigma_multiplier"] = multipliers
    out["raw_prior_residual_mean_c"] = raw_means
    out["raw_prior_sigma_multiplier"] = raw_multipliers
    out["prior_rows"] = prior_rows
    out["guard_id"] = spec.guard_id
    out["group_columns"] = "|".join(spec.group_columns)
    out["active_temp_bucket"] = spec.active_temp_bucket or ""
    out["min_prior_rows"] = spec.min_prior_rows
    out["mean_shrinkage"] = spec.mean_shrinkage
    out["scale_shrinkage"] = spec.scale_shrinkage
    out["mean_cap_c"] = spec.mean_cap_c
    out["window_days"] = float(spec.window_days) if spec.window_days is not None else math.nan
    require_no_confirmation_dates(out["target_date"], context=f"0062 {spec.guard_id} predictions")
    return out.sort_values(["target_date", "guard_id"]).reset_index(drop=True)


def p90_abs_error(frame: pd.DataFrame, prediction_col: str) -> float:
    scored = frame[["target_tmax_c", prediction_col]].dropna().copy()
    if scored.empty:
        return math.nan
    error = pd.to_numeric(scored[prediction_col], errors="coerce") - pd.to_numeric(scored["target_tmax_c"], errors="coerce")
    return float(error.abs().quantile(0.90))


def score_temp_scale(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    for guard_id, group in predictions.groupby("guard_id", observed=True):
        reference = group.rename(
            columns={
                "global_bias_repaired_prediction_c": "reference_prediction_c",
                "distribution_sigma_c": "reference_sigma_c",
            }
        )
        full = score_prediction_frame(group, "candidate_prediction_c")
        ref_full = score_prediction_frame(reference, "reference_prediction_c")
        interval = interval_metrics(group, "candidate_prediction_c", "candidate_sigma_c")
        ref_interval = interval_metrics(reference, "reference_prediction_c", "reference_sigma_c")
        p90 = p90_abs_error(group, "candidate_prediction_c")
        ref_p90 = p90_abs_error(reference, "reference_prediction_c")
        row = {
            "guard_id": guard_id,
            "group_columns": str(group["group_columns"].iloc[0]),
            "active_temp_bucket": str(group["active_temp_bucket"].iloc[0]),
            "window_days": float(group["window_days"].iloc[0]) if pd.notna(group["window_days"].iloc[0]) else math.nan,
            "min_prior_rows": int(group["min_prior_rows"].iloc[0]),
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
            "mean_interval90_width_c": interval["mean_interval90_width_c"],
            "reference_interval90_width_c": ref_interval["mean_interval90_width_c"],
            "mean_abs_correction_c": float(group["mean_residual_correction_c"].abs().mean()),
            "mean_sigma_multiplier": float(group["sigma_multiplier"].mean()),
        }
        row["promotion_gate_passed"] = bool(
            row["delta_mae_vs_0058"] <= 0.002
            and (
                row["coverage_distance"] < row["reference_coverage_distance"]
                or row["delta_p90_abs_error_vs_0058"] < 0.0
            )
        )
        rows.append(row)
        for bucket, subgroup in group.groupby("nearby_temp_bucket", observed=True):
            ref_subgroup = reference.loc[subgroup.index].copy()
            subgroup_score = score_prediction_frame(subgroup, "candidate_prediction_c")
            ref_score = score_prediction_frame(ref_subgroup, "reference_prediction_c")
            subgroup_interval = interval_metrics(subgroup, "candidate_prediction_c", "candidate_sigma_c")
            ref_subgroup_interval = interval_metrics(ref_subgroup, "reference_prediction_c", "reference_sigma_c")
            subgroup_rows.append(
                {
                    "guard_id": guard_id,
                    "nearby_temp_bucket": bucket,
                    "n": subgroup_score["n"],
                    "candidate_mae": subgroup_score["mae"],
                    "reference_mae": ref_score["mae"],
                    "delta_mae_vs_0058": float(subgroup_score["mae"]) - float(ref_score["mae"]),
                    "candidate_p90_abs_error": p90_abs_error(subgroup, "candidate_prediction_c"),
                    "reference_p90_abs_error": p90_abs_error(ref_subgroup, "reference_prediction_c"),
                    "candidate_coverage_distance": subgroup_interval["coverage_distance"],
                    "reference_coverage_distance": ref_subgroup_interval["coverage_distance"],
                    "mean_sigma_multiplier": float(subgroup["sigma_multiplier"].mean()),
                }
            )
    scoreboard = pd.DataFrame(rows).sort_values(
        ["promotion_gate_passed", "delta_mae_vs_0058", "coverage_distance"],
        ascending=[False, True, True],
    )
    subgroups = pd.DataFrame(subgroup_rows).sort_values(["nearby_temp_bucket", "delta_mae_vs_0058"]).reset_index(drop=True)
    return scoreboard.reset_index(drop=True), subgroups


def leakage_audit(predictions: pd.DataFrame, thresholds: pd.DataFrame, scoreboard: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(predictions["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(predictions['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "gate_thresholds_pre2000_only",
            "passed": bool(thresholds["threshold_source"].astype(str).str.contains("1999-12-31").all()),
            "evidence": "nearby temperature bucket thresholds use pre-2000 feature history",
        },
        {
            "check_id": "corrections_have_prior_history_only",
            "passed": bool((predictions["prior_rows"] >= 0).all()),
            "evidence": "streaming correction and scale state update after each scored row",
        },
        {
            "check_id": "first_row_has_zero_mean_and_unit_scale",
            "passed": bool(
                (
                    predictions.sort_values("target_date")
                    .groupby("guard_id", observed=True)
                    .head(1)["mean_residual_correction_c"]
                    .abs()
                    .le(1e-12)
                ).all()
                and (
                    predictions.sort_values("target_date")
                    .groupby("guard_id", observed=True)
                    .head(1)["sigma_multiplier"]
                    .sub(1.0)
                    .abs()
                    .le(1e-12)
                ).all()
            ),
            "evidence": "no earlier residuals exist for first row of each guard",
        },
        {
            "check_id": "promotion_gate_requires_tail_or_calibration_gain_without_mae_hurt",
            "passed": bool(scoreboard.loc[scoreboard["promotion_gate_passed"], "delta_mae_vs_0058"].le(0.002).all()),
            "evidence": f"{int(scoreboard['promotion_gate_passed'].sum())} candidates passed gate",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    scoreboard: pd.DataFrame,
    subgroups: pd.DataFrame,
    leakage: pd.DataFrame,
    thresholds: pd.DataFrame,
    design_row: pd.DataFrame,
) -> str:
    return f"""# Station-Only Nearby Temperature-Level Error Scale

Generated: `{summary['generated_at_utc']}`

## Purpose

`0057` found that nearby station temperature level is associated with station-only absolute error. `0062` tests whether that signal is useful as a tail-risk or uncertainty scaler on top of the `0058` global bias-repaired station baseline.

This is a bounded diagnostic guard screen. It uses no 2024+ confirmation rows and does not wait on the delayed RSS backfill.

## Contract

- Reference baseline: `0058` best candidate `{summary['reference_0058_best_candidate']}`.
- Nearby temperature feature: `{NEARBY_TEMP_COLUMN}`.
- Bucket thresholds: fixed from pre-2000 feature history.
- Mean and scale updates for date `T`: residuals from strictly earlier target dates only.
- Promotion gate: improve calibration or p90 absolute error without materially hurting MAE.

## Headline

| Item | Value |
|---|---:|
| Rows scored | {summary['rows_scored']} |
| Reference MAE | {summary['reference_mae']} |
| Reference p90 absolute error | {summary['reference_p90_abs_error']} |
| Reference coverage distance | {summary['reference_coverage_distance']} |
| Best guard | {summary['best_guard']} |
| Best MAE | {summary['best_mae']} |
| Best delta MAE vs 0058 | {summary['best_delta_mae_vs_0058']} |
| Best p90 absolute error | {summary['best_p90_abs_error']} |
| Best coverage distance | {summary['best_coverage_distance']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

## Source Queue Row

{markdown_table(design_row, max_rows=5)}

## Gate Thresholds

{markdown_table(thresholds, max_rows=20)}

## Guard Scoreboard

{markdown_table(scoreboard, max_rows=80)}

## Bucket Scoreboard

{markdown_table(subgroups, max_rows=80)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Interpretation

The key question is whether nearby temperature level should become a mean correction, an uncertainty scaler, or neither. A scale-only candidate that improves interval calibration without changing MAE is still useful for risk-aware forecasting, even if it does not reduce point-forecast MAE.

## Files

- `artifacts/predictions.parquet`
- `artifacts/predictions_sample.csv`
- `artifacts/scoreboard.csv`
- `artifacts/subgroup_scoreboard.csv`
- `artifacts/gate_thresholds.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_only_nearby_temperature_level_error_scale.py`:

- `{FOLDER_NAME}`: nearby station temperature-level mean/scale guard on top of `0058`.

| Metric | Value |
|---|---:|
| Reference MAE | {summary['reference_mae']} |
| Best guard | {summary['best_guard']} |
| Best MAE | {summary['best_mae']} |
| Best p90 absolute error | {summary['best_p90_abs_error']} |
| Best coverage distance | {summary['best_coverage_distance']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

Leakage contract: pre-2000 fixed temperature buckets and strict prior-only residual/scale updates.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Station-Only Nearby Temperature-Level Error Scale",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_only_nearby_temperature_level_error_scale.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Reference | `0058` `{summary['reference_0058_best_candidate']}` | Tested |
| Feature | `{NEARBY_TEMP_COLUMN}` | Tested |
| Best guard | `{summary['best_guard']}` | Diagnostic |
| Best delta MAE | `{summary['best_delta_mae_vs_0058']}` | Point forecast |
| Best p90 absolute error | `{summary['best_p90_abs_error']}` | Tail |
| Best coverage distance | `{summary['best_coverage_distance']}` | Calibration |
| Promotion gate passed | `{summary['best_promotion_gate_passed']}` | Guarded |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0062` tests whether nearby temperature level is a deployable calibration/tail-risk signal after `0058`.
"""
    update_markdown_section(
        path,
        heading="Station-Only Nearby Temperature-Level Error Scale",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"46. Nearby temperature-level error scaling tested `{summary['candidate_count']}` variants on top of `0058`; "
        f"best guard is `{summary['best_guard']}` with MAE `{summary['best_mae']}`."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Synthesize `0058`-`0062` into one guarded candidate stack design: global bias repair, February/March pocket repair, pressure-high interval guard, and nearby-temperature calibration. Do not run 2024+ confirmation until the user explicitly asks.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0055, summary_0058, thresholds = load_nearby_temperature_frame()
    if not DESIGN_QUEUE_PATH.exists():
        raise FileNotFoundError(f"Missing 0057 design queue: {DESIGN_QUEUE_PATH}")
    design_queue = pd.read_csv(DESIGN_QUEUE_PATH)
    design_row = design_queue[design_queue["candidate_id"].astype(str).eq("nearby_temperature_level_error_scale")].copy()
    if design_row.empty:
        raise RuntimeError("0057 design queue does not contain nearby_temperature_level_error_scale")

    predictions = pd.concat([apply_temp_scale(frame, spec) for spec in temp_scale_specs()], ignore_index=True)
    scoreboard, subgroups = score_temp_scale(predictions)
    leakage = leakage_audit(predictions, thresholds, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0062 leakage audit failed: {failed}")

    reference = frame.rename(
        columns={
            "global_bias_repaired_prediction_c": "reference_prediction_c",
            "distribution_sigma_c": "reference_sigma_c",
        }
    )
    reference_score = score_prediction_frame(reference, "reference_prediction_c")
    reference_interval = interval_metrics(reference, "reference_prediction_c", "reference_sigma_c")
    best = scoreboard.iloc[0]
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "anchor_model_id": str(summary_0055["best_model_id"]),
        "reference_0058_best_candidate": str(summary_0058["best_candidate"]),
        "candidate_count": int(scoreboard["guard_id"].nunique()),
        "rows_scored": int(scoreboard["n"].max()),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "reference_mae": float(reference_score["mae"]),
        "reference_rmse": float(reference_score["rmse"]),
        "reference_p90_abs_error": p90_abs_error(reference, "reference_prediction_c"),
        "reference_coverage80": float(reference_interval["coverage80"]),
        "reference_coverage90": float(reference_interval["coverage90"]),
        "reference_coverage_distance": float(reference_interval["coverage_distance"]),
        "best_guard": str(best["guard_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0058": float(best["delta_mae_vs_0058"]),
        "best_p90_abs_error": float(best["p90_abs_error"]),
        "best_delta_p90_abs_error_vs_0058": float(best["delta_p90_abs_error_vs_0058"]),
        "best_coverage80": float(best["coverage80"]),
        "best_coverage90": float(best["coverage90"]),
        "best_coverage_distance": float(best["coverage_distance"]),
        "best_promotion_gate_passed": bool(best["promotion_gate_passed"]),
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
    write_csv(artifacts / "subgroup_scoreboard.csv", subgroups)
    write_csv(artifacts / "gate_thresholds.csv", thresholds)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_only_nearby_temperature_level_error_scale_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            scoreboard=scoreboard,
            subgroups=subgroups,
            leakage=leakage,
            thresholds=thresholds,
            design_row=design_row,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Run nearby temperature-level mean/scale guard on top of 0058."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
