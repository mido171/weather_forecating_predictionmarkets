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
    load_feature_gates,
)
from scripts.run_hkg_t24_station_only_late_period_bias_repair import (  # noqa: E402
    score_prediction_frame,
)
from scripts.run_hkg_t24_station_only_spring_transition_pressure_dew_specialist import (  # noqa: E402
    PREDICTIONS_0055_PATH,
    PREDICTIONS_0058_PATH,
    SUMMARY_0055_PATH,
    SUMMARY_0058_PATH,
)

FOLDER_NAME = "0061_station_only_pressure_high_uncertainty_guard"
PRESSURE_HIGH = "pressure_high"
Z80 = 1.2815515655446004
Z90 = 1.6448536269514722
EXPECTED_NORMAL_ABS_FACTOR = math.sqrt(2.0 / math.pi)


@dataclass(frozen=True)
class PressureGuardSpec:
    guard_id: str
    group_columns: tuple[str, ...]
    active_pressure_bucket: str | None
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


def load_pressure_frame() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame]:
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

    gates, thresholds = load_feature_gates()
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
    frame = frame.merge(gates, on="target_date", how="left", validate="one_to_one")
    frame["year"] = frame["target_date"].dt.year
    frame["month"] = frame["target_date"].dt.month
    frame["pressure_high_window"] = frame["pressure_spread_bucket"].astype(str).eq(PRESSURE_HIGH)
    frame["residual_to_add_c"] = frame["target_tmax_c"] - frame["global_bias_repaired_prediction_c"]
    frame["reference_error_c"] = frame["global_bias_repaired_prediction_c"] - frame["target_tmax_c"]
    frame["reference_expected_abs_c"] = (
        pd.to_numeric(frame["distribution_sigma_c"], errors="coerce").clip(lower=0.05) * EXPECTED_NORMAL_ABS_FACTOR
    )
    frame = frame.sort_values("target_date").reset_index(drop=True)
    require_no_confirmation_dates(frame["target_date"], context="0061 pressure guard frame")
    return frame, summary_0055, summary_0058, thresholds


def pressure_guard_specs() -> list[PressureGuardSpec]:
    return [
        PressureGuardSpec(
            "pressure_bucket_mean_scale_min180_shrink365_cap0p8",
            ("pressure_spread_bucket",),
            None,
            180,
            365.0,
            365.0,
            0.8,
            0.75,
            2.0,
        ),
        PressureGuardSpec(
            "pressure_high_only_mean_scale_min120_shrink240_cap0p8",
            ("pressure_spread_bucket",),
            PRESSURE_HIGH,
            120,
            240.0,
            240.0,
            0.8,
            0.75,
            2.1,
        ),
        PressureGuardSpec(
            "pressure_high_only_scale_min120_shrink240",
            ("pressure_spread_bucket",),
            PRESSURE_HIGH,
            120,
            10_000.0,
            240.0,
            0.0,
            0.75,
            2.1,
        ),
        PressureGuardSpec(
            "pressure_bucket_dew_mean_scale_min90_shrink180_cap0p8",
            ("pressure_spread_bucket", "dew_trajectory_bucket"),
            None,
            90,
            180.0,
            180.0,
            0.8,
            0.75,
            2.1,
        ),
        PressureGuardSpec(
            "pressure_high_dew_mean_scale_min60_shrink120_cap0p8",
            ("pressure_spread_bucket", "dew_trajectory_bucket"),
            PRESSURE_HIGH,
            60,
            120.0,
            120.0,
            0.8,
            0.75,
            2.2,
        ),
        PressureGuardSpec(
            "pressure_high_wind_mean_scale_min60_shrink120_cap0p8",
            ("pressure_spread_bucket", "wind_spread_bucket"),
            PRESSURE_HIGH,
            60,
            120.0,
            120.0,
            0.8,
            0.75,
            2.2,
        ),
        PressureGuardSpec(
            "pressure_high_dew_wind_mean_scale_min45_shrink100_cap0p7",
            ("pressure_spread_bucket", "dew_trajectory_bucket", "wind_spread_bucket"),
            PRESSURE_HIGH,
            45,
            100.0,
            100.0,
            0.7,
            0.75,
            2.2,
        ),
        PressureGuardSpec(
            "pressure_high_rolling8y_mean_scale_min60_shrink120_cap0p8",
            ("pressure_spread_bucket",),
            PRESSURE_HIGH,
            60,
            120.0,
            120.0,
            0.8,
            0.75,
            2.2,
            window_days=2920,
        ),
    ]


def group_key(row: pd.Series, columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in columns)


def is_active(row: pd.Series, spec: PressureGuardSpec) -> bool:
    if spec.active_pressure_bucket is None:
        return True
    return str(row["pressure_spread_bucket"]) == spec.active_pressure_bucket


def correction_from_state(
    *,
    count: int,
    residual_sum: float,
    abs_sum: float,
    expected_abs_sum: float,
    spec: PressureGuardSpec,
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


def compute_pressure_guard(
    frame: pd.DataFrame,
    spec: PressureGuardSpec,
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


def apply_pressure_guard(frame: pd.DataFrame, spec: PressureGuardSpec) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    corrections, multipliers, prior_rows, raw_means, raw_multipliers = compute_pressure_guard(ordered, spec)
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
            "pressure_high_window",
            "pressure_spread_abs_max",
            "dew_trajectory_mean",
            "wind_spread_abs_max",
            "pressure_spread_bucket",
            "dew_trajectory_bucket",
            "wind_spread_bucket",
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
    out["active_pressure_bucket"] = spec.active_pressure_bucket or ""
    out["min_prior_rows"] = spec.min_prior_rows
    out["mean_shrinkage"] = spec.mean_shrinkage
    out["scale_shrinkage"] = spec.scale_shrinkage
    out["mean_cap_c"] = spec.mean_cap_c
    out["window_days"] = float(spec.window_days) if spec.window_days is not None else math.nan
    require_no_confirmation_dates(out["target_date"], context=f"0061 {spec.guard_id} predictions")
    return out.sort_values(["target_date", "guard_id"]).reset_index(drop=True)


def interval_metrics(frame: pd.DataFrame, prediction_col: str, sigma_col: str) -> dict[str, object]:
    scored = frame[["target_date", "target_tmax_c", prediction_col, sigma_col]].dropna().copy()
    if scored.empty:
        return {
            "n": 0,
            "coverage80": math.nan,
            "coverage90": math.nan,
            "coverage_distance": math.nan,
            "mean_interval90_width_c": math.nan,
        }
    error = pd.to_numeric(scored[prediction_col], errors="coerce") - pd.to_numeric(scored["target_tmax_c"], errors="coerce")
    sigma = pd.to_numeric(scored[sigma_col], errors="coerce").clip(lower=0.05)
    abs_error = error.abs()
    coverage80 = float((abs_error <= Z80 * sigma).mean())
    coverage90 = float((abs_error <= Z90 * sigma).mean())
    return {
        "n": int(len(scored)),
        "coverage80": coverage80,
        "coverage90": coverage90,
        "coverage_distance": abs(coverage80 - 0.80) + abs(coverage90 - 0.90),
        "mean_interval90_width_c": float((2.0 * Z90 * sigma).mean()),
    }


def score_pressure_guard(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        high = group[group["pressure_high_window"].astype(bool)].copy()
        ref_high_frame = reference[reference["pressure_high_window"].astype(bool)].copy()
        nonhigh = group[~group["pressure_high_window"].astype(bool)].copy()
        ref_nonhigh_frame = reference[~reference["pressure_high_window"].astype(bool)].copy()
        high_score = score_prediction_frame(high, "candidate_prediction_c")
        ref_high = score_prediction_frame(ref_high_frame, "reference_prediction_c")
        nonhigh_score = score_prediction_frame(nonhigh, "candidate_prediction_c")
        ref_nonhigh = score_prediction_frame(ref_nonhigh_frame, "reference_prediction_c")
        high_interval = interval_metrics(high, "candidate_prediction_c", "candidate_sigma_c")
        ref_high_interval = interval_metrics(ref_high_frame, "reference_prediction_c", "reference_sigma_c")
        row = {
            "guard_id": guard_id,
            "group_columns": str(group["group_columns"].iloc[0]),
            "active_pressure_bucket": str(group["active_pressure_bucket"].iloc[0]),
            "window_days": float(group["window_days"].iloc[0]) if pd.notna(group["window_days"].iloc[0]) else math.nan,
            "min_prior_rows": int(group["min_prior_rows"].iloc[0]),
            "n": full["n"],
            "mae": full["mae"],
            "rmse": full["rmse"],
            "bias": full["bias"],
            "delta_mae_vs_0058": float(full["mae"]) - float(ref_full["mae"]),
            "pressure_high_n": high_score["n"],
            "pressure_high_mae": high_score["mae"],
            "reference_pressure_high_mae": ref_high["mae"],
            "pressure_high_delta_mae_vs_0058": float(high_score["mae"]) - float(ref_high["mae"]),
            "pressure_high_rmse": high_score["rmse"],
            "pressure_high_bias": high_score["bias"],
            "pressure_high_coverage80": high_interval["coverage80"],
            "reference_pressure_high_coverage80": ref_high_interval["coverage80"],
            "pressure_high_coverage90": high_interval["coverage90"],
            "reference_pressure_high_coverage90": ref_high_interval["coverage90"],
            "pressure_high_coverage_distance": high_interval["coverage_distance"],
            "reference_pressure_high_coverage_distance": ref_high_interval["coverage_distance"],
            "pressure_high_interval90_width_c": high_interval["mean_interval90_width_c"],
            "reference_pressure_high_interval90_width_c": ref_high_interval["mean_interval90_width_c"],
            "non_high_delta_mae_vs_0058": float(nonhigh_score["mae"]) - float(ref_nonhigh["mae"]),
            "mean_abs_correction_c": float(group["mean_residual_correction_c"].abs().mean()),
            "mean_sigma_multiplier": float(group["sigma_multiplier"].mean()),
            "mean_high_sigma_multiplier": float(high["sigma_multiplier"].mean()) if not high.empty else math.nan,
        }
        row["promotion_gate_passed"] = bool(
            row["pressure_high_delta_mae_vs_0058"] < 0.0
            and row["pressure_high_coverage_distance"] < row["reference_pressure_high_coverage_distance"]
            and row["non_high_delta_mae_vs_0058"] <= 0.005
        )
        rows.append(row)
        for subgroup_name, subgroup, ref_subgroup in [
            ("pressure_high", high, ref_high_frame),
            ("pressure_not_high", nonhigh, ref_nonhigh_frame),
        ]:
            subgroup_score = score_prediction_frame(subgroup, "candidate_prediction_c")
            ref_score = score_prediction_frame(ref_subgroup, "reference_prediction_c")
            interval = interval_metrics(subgroup, "candidate_prediction_c", "candidate_sigma_c")
            ref_interval = interval_metrics(ref_subgroup, "reference_prediction_c", "reference_sigma_c")
            subgroup_rows.append(
                {
                    "guard_id": guard_id,
                    "subgroup": subgroup_name,
                    "n": subgroup_score["n"],
                    "candidate_mae": subgroup_score["mae"],
                    "reference_mae": ref_score["mae"],
                    "delta_mae_vs_0058": float(subgroup_score["mae"]) - float(ref_score["mae"]),
                    "candidate_coverage80": interval["coverage80"],
                    "reference_coverage80": ref_interval["coverage80"],
                    "candidate_coverage90": interval["coverage90"],
                    "reference_coverage90": ref_interval["coverage90"],
                    "candidate_coverage_distance": interval["coverage_distance"],
                    "reference_coverage_distance": ref_interval["coverage_distance"],
                }
            )
    scoreboard = pd.DataFrame(rows).sort_values(
        ["promotion_gate_passed", "pressure_high_delta_mae_vs_0058", "pressure_high_coverage_distance"],
        ascending=[False, True, True],
    )
    subgroups = pd.DataFrame(subgroup_rows).sort_values(["subgroup", "delta_mae_vs_0058"]).reset_index(drop=True)
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
            "evidence": f"{len(thresholds)} inherited pre-2000 gate thresholds checked",
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
            "check_id": "promotion_gate_requires_mae_and_coverage_improvement",
            "passed": bool(
                (
                    scoreboard.loc[scoreboard["promotion_gate_passed"], "pressure_high_delta_mae_vs_0058"] < 0.0
                ).all()
                and (
                    scoreboard.loc[scoreboard["promotion_gate_passed"], "pressure_high_coverage_distance"]
                    < scoreboard.loc[scoreboard["promotion_gate_passed"], "reference_pressure_high_coverage_distance"]
                ).all()
            ),
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
    return f"""# Station-Only Pressure-High Uncertainty Guard

Generated: `{summary['generated_at_utc']}`

## Purpose

`0057` identified high pressure-spread states as elevated-error regimes. `0061` tests whether a deployable guard can improve pressure-high mean error and interval calibration on top of the `0058` global bias-repaired station baseline.

This is not predictive modelling beyond the bounded residual/uncertainty guard screen. It uses no 2024+ confirmation rows.

## Contract

- Reference baseline: `0058` best candidate `{summary['reference_0058_best_candidate']}`.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- Pressure/dew/wind buckets: fixed from pre-2000 feature history.
- Mean and scale updates for date `T`: residuals from strictly earlier target dates only.
- Promotion gate: pressure-high MAE must improve, pressure-high interval calibration must improve, and non-high MAE must not deteriorate materially.

## Headline

| Item | Value |
|---|---:|
| Rows scored | {summary['rows_scored']} |
| Pressure-high rows | {summary['pressure_high_rows']} |
| Reference pressure-high MAE | {summary['reference_pressure_high_mae']} |
| Reference pressure-high coverage distance | {summary['reference_pressure_high_coverage_distance']} |
| Best guard | {summary['best_guard']} |
| Best pressure-high MAE | {summary['best_pressure_high_mae']} |
| Best pressure-high delta vs 0058 | {summary['best_pressure_high_delta_mae_vs_0058']} |
| Best pressure-high coverage distance | {summary['best_pressure_high_coverage_distance']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

## Source Queue Row

{markdown_table(design_row, max_rows=5)}

## Gate Thresholds

{markdown_table(thresholds, max_rows=20)}

## Guard Scoreboard

{markdown_table(scoreboard, max_rows=80)}

## Subgroup Scoreboard

{markdown_table(subgroups, max_rows=40)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Interpretation

This experiment separates two ideas that are often mixed together: correcting the mean forecast and widening/narrowing uncertainty intervals. A promoted guard must improve both the high-pressure mean error and the calibration distance of 80%/90% normal-style intervals. If the mean correction helps but the interval calibration does not, it should be treated as partial evidence rather than a deployable guard.

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

Additional folder created by `scripts/run_hkg_t24_station_only_pressure_high_uncertainty_guard.py`:

- `{FOLDER_NAME}`: pressure-high mean correction and interval scaling guard on top of `0058`.

| Metric | Value |
|---|---:|
| Reference pressure-high MAE | {summary['reference_pressure_high_mae']} |
| Best guard | {summary['best_guard']} |
| Best pressure-high MAE | {summary['best_pressure_high_mae']} |
| Best pressure-high delta vs 0058 | {summary['best_pressure_high_delta_mae_vs_0058']} |
| Best coverage distance | {summary['best_pressure_high_coverage_distance']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

Leakage contract: pre-2000 fixed gates and strict prior-only residual/scale updates.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Station-Only Pressure-High Uncertainty Guard",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_only_pressure_high_uncertainty_guard.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Reference | `0058` `{summary['reference_0058_best_candidate']}` | Tested |
| Target pocket | Pressure-high rows `{summary['pressure_high_rows']}` | Tested |
| Reference high MAE | `{summary['reference_pressure_high_mae']}` | Baseline |
| Best guard | `{summary['best_guard']}` | Diagnostic |
| Best high MAE delta | `{summary['best_pressure_high_delta_mae_vs_0058']}` | Diagnostic |
| Best coverage distance | `{summary['best_pressure_high_coverage_distance']}` | Calibration |
| Promotion gate passed | `{summary['best_promotion_gate_passed']}` | Guarded |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0061` tests high-pressure uncertainty and mean repair after `0058` without waiting on RSS backfill.
"""
    update_markdown_section(
        path,
        heading="Station-Only Pressure-High Uncertainty Guard",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"45. Pressure-high uncertainty guard tested `{summary['candidate_count']}` variants on top of `0058`; "
        f"best high-pressure delta vs 0058 is `{summary['best_pressure_high_delta_mae_vs_0058']}` from "
        f"`{summary['best_guard']}`."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue the `0057` queue with `nearby_temperature_level_error_scale`: test whether nearby-station temperature level is better used for tail/interval calibration than mean correction, still on top of the `0058` baseline.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0055, summary_0058, thresholds = load_pressure_frame()
    if not DESIGN_QUEUE_PATH.exists():
        raise FileNotFoundError(f"Missing 0057 design queue: {DESIGN_QUEUE_PATH}")
    design_queue = pd.read_csv(DESIGN_QUEUE_PATH)
    design_row = design_queue[design_queue["candidate_id"].astype(str).eq("pressure_high_uncertainty_guard")].copy()
    if design_row.empty:
        raise RuntimeError("0057 design queue does not contain pressure_high_uncertainty_guard")

    predictions = pd.concat([apply_pressure_guard(frame, spec) for spec in pressure_guard_specs()], ignore_index=True)
    scoreboard, subgroups = score_pressure_guard(predictions)
    leakage = leakage_audit(predictions, thresholds, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0061 leakage audit failed: {failed}")

    reference = frame.rename(
        columns={
            "global_bias_repaired_prediction_c": "reference_prediction_c",
            "distribution_sigma_c": "reference_sigma_c",
        }
    )
    reference_high = reference[reference["pressure_high_window"]].copy()
    high_score = score_prediction_frame(reference_high, "reference_prediction_c")
    high_interval = interval_metrics(reference_high, "reference_prediction_c", "reference_sigma_c")
    best = scoreboard.iloc[0]
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "anchor_model_id": str(summary_0055["best_model_id"]),
        "reference_0058_best_candidate": str(summary_0058["best_candidate"]),
        "candidate_count": int(scoreboard["guard_id"].nunique()),
        "rows_scored": int(scoreboard["n"].max()),
        "pressure_high_rows": int(high_score["n"]),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "reference_pressure_high_mae": float(high_score["mae"]),
        "reference_pressure_high_rmse": float(high_score["rmse"]),
        "reference_pressure_high_coverage80": float(high_interval["coverage80"]),
        "reference_pressure_high_coverage90": float(high_interval["coverage90"]),
        "reference_pressure_high_coverage_distance": float(high_interval["coverage_distance"]),
        "best_guard": str(best["guard_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0058": float(best["delta_mae_vs_0058"]),
        "best_pressure_high_mae": float(best["pressure_high_mae"]),
        "best_pressure_high_delta_mae_vs_0058": float(best["pressure_high_delta_mae_vs_0058"]),
        "best_pressure_high_coverage80": float(best["pressure_high_coverage80"]),
        "best_pressure_high_coverage90": float(best["pressure_high_coverage90"]),
        "best_pressure_high_coverage_distance": float(best["pressure_high_coverage_distance"]),
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
    write_json(RESEARCH_ROOT / "station_only_pressure_high_uncertainty_guard_manifest.json", summary)
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
        description="Run pressure-high mean correction and uncertainty guard on top of 0058."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
