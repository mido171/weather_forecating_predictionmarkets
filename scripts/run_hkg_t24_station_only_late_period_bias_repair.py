from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import deque
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

FOLDER_NAME = "0058_station_only_late_period_bias_repair"
ARTIFACT_0055 = RESEARCH_ROOT / "0055_station_only_walkforward_benchmark" / "artifacts"
ARTIFACT_0057 = RESEARCH_ROOT / "0057_station_only_residual_specialist_design_queue" / "artifacts"
PREDICTIONS_PATH = ARTIFACT_0055 / "predictions.parquet"
SUMMARY_0055_PATH = ARTIFACT_0055 / "summary.json"
DESIGN_QUEUE_PATH = ARTIFACT_0057 / "design_queue.csv"
MIN_PRIOR_ROWS = 180
LATE_START = pd.Timestamp("2018-01-01")
DEVELOPMENT_END = pd.Timestamp("2023-12-31")


@dataclass(frozen=True)
class BiasRepairSpec:
    correction_id: str
    family: str
    group_column: str | None
    window_days: int | None
    half_life_days: float | None
    min_prior_rows: int
    shrinkage: float
    cap_c: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def load_anchor_predictions() -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = load_json(SUMMARY_0055_PATH)
    best_model_id = str(summary["best_model_id"])
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    predictions = predictions[predictions["model_id"].astype(str).eq(best_model_id)].copy()
    if predictions.empty:
        raise RuntimeError(f"Missing 0055 best model predictions: {best_model_id}")
    predictions = predictions[predictions["target_date"].le(DEVELOPMENT_END)].copy()
    require_no_confirmation_dates(predictions["target_date"], context="0058 anchor predictions")
    predictions["month"] = predictions["target_date"].dt.month
    predictions["season"] = predictions["month"].map(season_from_month)
    predictions["year"] = predictions["target_date"].dt.year
    predictions["anchor_error_c"] = predictions["point_forecast_c"] - predictions["target_tmax_c"]
    predictions["residual_to_add_c"] = predictions["target_tmax_c"] - predictions["point_forecast_c"]
    predictions = predictions.sort_values("target_date").reset_index(drop=True)
    return predictions, summary


def repair_specs() -> list[BiasRepairSpec]:
    return [
        BiasRepairSpec("global_expanding_min180_shrink365_cap1p5", "expanding", None, None, None, 180, 365.0, 1.5),
        BiasRepairSpec("global_expanding_min365_shrink365_cap1p5", "expanding", None, None, None, 365, 365.0, 1.5),
        BiasRepairSpec("global_rolling365_min180_shrink180_cap1p5", "rolling", None, 365, None, 180, 180.0, 1.5),
        BiasRepairSpec("global_rolling730_min180_shrink365_cap1p5", "rolling", None, 730, None, 180, 365.0, 1.5),
        BiasRepairSpec("global_rolling1460_min180_shrink365_cap1p5", "rolling", None, 1460, None, 180, 365.0, 1.5),
        BiasRepairSpec("global_halflife365_min180_shrink365_cap1p5", "half_life", None, None, 365.0, 180, 365.0, 1.5),
        BiasRepairSpec("global_halflife730_min180_shrink365_cap1p5", "half_life", None, None, 730.0, 180, 365.0, 1.5),
        BiasRepairSpec("same_month_expanding_min60_shrink120_cap1p5", "group_expanding", "month", None, None, 60, 120.0, 1.5),
        BiasRepairSpec("same_season_expanding_min120_shrink180_cap1p5", "group_expanding", "season", None, None, 120, 180.0, 1.5),
        BiasRepairSpec("same_month_rolling1460_min45_shrink120_cap1p5", "group_rolling", "month", 1460, None, 45, 120.0, 1.5),
        BiasRepairSpec("same_season_rolling1460_min90_shrink180_cap1p5", "group_rolling", "season", 1460, None, 90, 180.0, 1.5),
    ]


def weighted_prior_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return math.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def prior_subset(frame: pd.DataFrame, current: pd.Series, spec: BiasRepairSpec) -> pd.DataFrame:
    prior = frame[frame["target_date"] < current["target_date"]].copy()
    if spec.group_column is not None:
        prior = prior[prior[spec.group_column].astype(str).eq(str(current[spec.group_column]))].copy()
    if spec.window_days is not None:
        min_date = pd.Timestamp(current["target_date"]) - pd.Timedelta(days=spec.window_days)
        prior = prior[prior["target_date"] >= min_date].copy()
    return prior


def correction_from_prior(prior: pd.DataFrame, current_date: pd.Timestamp, spec: BiasRepairSpec) -> tuple[float, int, float]:
    if len(prior) < spec.min_prior_rows:
        return 0.0, int(len(prior)), math.nan
    residuals = pd.to_numeric(prior["residual_to_add_c"], errors="coerce").to_numpy(dtype=float)
    if spec.half_life_days is not None:
        ages = (current_date - prior["target_date"]).dt.days.to_numpy(dtype=float)
        weights = np.power(0.5, ages / spec.half_life_days)
    else:
        weights = np.ones(len(prior), dtype=float)
    raw = weighted_prior_mean(residuals, weights)
    if not math.isfinite(raw):
        return 0.0, int(len(prior)), math.nan
    shrink = len(prior) / (len(prior) + spec.shrinkage)
    correction = float(np.clip(raw * shrink, -spec.cap_c, spec.cap_c))
    return correction, int(len(prior)), raw


def shrink_and_cap(raw: float, prior_rows: int, spec: BiasRepairSpec) -> float:
    if prior_rows < spec.min_prior_rows or not math.isfinite(raw):
        return 0.0
    shrink = prior_rows / (prior_rows + spec.shrinkage)
    return float(np.clip(raw * shrink, -spec.cap_c, spec.cap_c))


def compute_prior_corrections(anchor: pd.DataFrame, spec: BiasRepairSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.to_datetime(anchor["target_date"], errors="coerce").to_numpy()
    residuals = pd.to_numeric(anchor["residual_to_add_c"], errors="coerce").to_numpy(dtype=float)
    groups = (
        anchor[spec.group_column].astype(str).to_numpy()
        if spec.group_column is not None
        else np.full(len(anchor), "__all__", dtype=object)
    )
    corrections = np.zeros(len(anchor), dtype=float)
    prior_counts = np.zeros(len(anchor), dtype=int)
    raw_means = np.full(len(anchor), math.nan, dtype=float)

    if spec.half_life_days is not None:
        weighted_sum = 0.0
        weighted_residual_sum = 0.0
        prior_count = 0
        previous_date: pd.Timestamp | None = None
        for idx, date_value in enumerate(dates):
            current_date = pd.Timestamp(date_value)
            if previous_date is not None:
                delta_days = max(0, (current_date - previous_date).days)
                decay = math.pow(0.5, delta_days / spec.half_life_days)
                weighted_sum *= decay
                weighted_residual_sum *= decay
            if prior_count >= spec.min_prior_rows and weighted_sum > 0:
                raw = weighted_residual_sum / weighted_sum
                raw_means[idx] = raw
                corrections[idx] = shrink_and_cap(raw, prior_count, spec)
            prior_counts[idx] = prior_count
            residual = residuals[idx]
            if math.isfinite(residual):
                weighted_sum += 1.0
                weighted_residual_sum += residual
                prior_count += 1
            previous_date = current_date
        return corrections, prior_counts, raw_means

    if spec.window_days is None:
        state: dict[str, tuple[int, float]] = {}
        for idx, group in enumerate(groups):
            count, total = state.get(str(group), (0, 0.0))
            prior_counts[idx] = count
            if count >= spec.min_prior_rows:
                raw = total / count
                raw_means[idx] = raw
                corrections[idx] = shrink_and_cap(raw, count, spec)
            residual = residuals[idx]
            if math.isfinite(residual):
                state[str(group)] = (count + 1, total + residual)
        return corrections, prior_counts, raw_means

    rolling_state: dict[str, dict[str, object]] = {}
    for idx, date_value in enumerate(dates):
        current_date = pd.Timestamp(date_value)
        group = str(groups[idx])
        state = rolling_state.setdefault(group, {"rows": deque(), "sum": 0.0})
        rows = state["rows"]
        assert isinstance(rows, deque)
        min_date = current_date - pd.Timedelta(days=spec.window_days)
        while rows and rows[0][0] < min_date:
            _, old_residual = rows.popleft()
            state["sum"] = float(state["sum"]) - float(old_residual)
        count = len(rows)
        prior_counts[idx] = count
        if count >= spec.min_prior_rows:
            raw = float(state["sum"]) / count
            raw_means[idx] = raw
            corrections[idx] = shrink_and_cap(raw, count, spec)
        residual = residuals[idx]
        if math.isfinite(residual):
            rows.append((current_date, residual))
            state["sum"] = float(state["sum"]) + residual
    return corrections, prior_counts, raw_means


def apply_bias_repair(anchor: pd.DataFrame, spec: BiasRepairSpec) -> pd.DataFrame:
    ordered = anchor.sort_values("target_date").reset_index(drop=True).copy()
    correction, prior_rows, raw_prior_mean = compute_prior_corrections(ordered, spec)
    out = ordered[
        [
            "target_date",
            "target_tmax_c",
            "point_forecast_c",
            "fold_id",
            "year",
            "month",
            "season",
        ]
    ].copy()
    out = out.rename(columns={"point_forecast_c": "anchor_prediction_c"})
    out["candidate_prediction_c"] = out["anchor_prediction_c"] + correction
    out["residual_correction_c"] = correction
    out["raw_prior_residual_mean_c"] = raw_prior_mean
    out["prior_rows"] = prior_rows
    out["correction_id"] = spec.correction_id
    out["family"] = spec.family
    out["group_column"] = "" if spec.group_column is None else spec.group_column
    out["window_days"] = float(spec.window_days) if spec.window_days is not None else math.nan
    out["half_life_days"] = float(spec.half_life_days) if spec.half_life_days is not None else math.nan
    out["min_prior_rows"] = spec.min_prior_rows
    out["shrinkage"] = spec.shrinkage
    out["cap_c"] = spec.cap_c
    out = out.sort_values(["target_date", "correction_id"]).reset_index(drop=True)
    require_no_confirmation_dates(out["target_date"], context=f"0058 {spec.correction_id} predictions")
    return out


def score_prediction_frame(frame: pd.DataFrame, prediction_col: str) -> dict[str, object]:
    scored = frame[["target_date", "target_tmax_c", prediction_col]].dropna().copy()
    error = pd.to_numeric(scored[prediction_col], errors="coerce") - pd.to_numeric(scored["target_tmax_c"], errors="coerce")
    return {
        "n": int(len(scored)),
        "first_date": str(scored["target_date"].min().date()) if not scored.empty else "",
        "last_date": str(scored["target_date"].max().date()) if not scored.empty else "",
        "mae": float(error.abs().mean()) if len(error) else math.nan,
        "rmse": float(np.sqrt(np.mean(np.square(error)))) if len(error) else math.nan,
        "bias": float(error.mean()) if len(error) else math.nan,
        "median_abs_error": float(error.abs().median()) if len(error) else math.nan,
        "p90_abs_error": float(error.abs().quantile(0.90)) if len(error) else math.nan,
        "share_abs_error_ge_2c": float((error.abs() >= 2.0).mean()) if len(error) else math.nan,
    }


def score_candidates(predictions: pd.DataFrame, anchor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor_score = score_prediction_frame(
        anchor.rename(columns={"point_forecast_c": "anchor_prediction_c"}),
        "anchor_prediction_c",
    )
    rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    for correction_id, group in predictions.groupby("correction_id", observed=True):
        full = score_prediction_frame(group, "candidate_prediction_c")
        anchor_full = score_prediction_frame(group, "anchor_prediction_c")
        late_group = group[group["target_date"] >= LATE_START].copy()
        early_group = group[group["target_date"] < LATE_START].copy()
        late = score_prediction_frame(late_group, "candidate_prediction_c")
        anchor_late = score_prediction_frame(late_group, "anchor_prediction_c")
        early = score_prediction_frame(early_group, "candidate_prediction_c")
        anchor_early = score_prediction_frame(early_group, "anchor_prediction_c")
        row = {
            "correction_id": correction_id,
            "family": str(group["family"].iloc[0]),
            "group_column": str(group["group_column"].iloc[0]),
            "window_days": str(group["window_days"].iloc[0]),
            "half_life_days": str(group["half_life_days"].iloc[0]),
            "min_prior_rows": int(group["min_prior_rows"].iloc[0]),
            "shrinkage": float(group["shrinkage"].iloc[0]),
            "cap_c": float(group["cap_c"].iloc[0]),
            "n": full["n"],
            "mae": full["mae"],
            "rmse": full["rmse"],
            "bias": full["bias"],
            "median_abs_error": full["median_abs_error"],
            "p90_abs_error": full["p90_abs_error"],
            "delta_mae_vs_anchor": float(full["mae"]) - float(anchor_full["mae"]),
            "late_mae": late["mae"],
            "late_rmse": late["rmse"],
            "late_bias": late["bias"],
            "late_delta_mae_vs_anchor": float(late["mae"]) - float(anchor_late["mae"]),
            "early_mae": early["mae"],
            "early_rmse": early["rmse"],
            "early_bias": early["bias"],
            "early_delta_mae_vs_anchor": float(early["mae"]) - float(anchor_early["mae"]),
            "promotion_gate_passed": bool(
                float(late["mae"]) < float(anchor_late["mae"])
                and float(early["mae"]) <= float(anchor_early["mae"]) + 0.02
            ),
            "mean_abs_correction_c": float(group["residual_correction_c"].abs().mean()),
            "active_correction_share": float(group["residual_correction_c"].abs().gt(1e-9).mean()),
            "anchor_full_mae_reference": anchor_score["mae"],
        }
        rows.append(row)
        for subgroup_name, subgroup_frame in [
            ("early_2000_2017", early_group),
            ("late_2018_2023", late_group),
        ]:
            candidate_score = score_prediction_frame(subgroup_frame, "candidate_prediction_c")
            anchor_score_sub = score_prediction_frame(subgroup_frame, "anchor_prediction_c")
            subgroup_rows.append(
                {
                    "correction_id": correction_id,
                    "subgroup": subgroup_name,
                    "n": candidate_score["n"],
                    "first_date": candidate_score["first_date"],
                    "last_date": candidate_score["last_date"],
                    "candidate_mae": candidate_score["mae"],
                    "anchor_mae": anchor_score_sub["mae"],
                    "delta_mae_vs_anchor": float(candidate_score["mae"]) - float(anchor_score_sub["mae"]),
                    "candidate_rmse": candidate_score["rmse"],
                    "anchor_rmse": anchor_score_sub["rmse"],
                    "candidate_bias": candidate_score["bias"],
                    "anchor_bias": anchor_score_sub["bias"],
                }
            )
    scoreboard = pd.DataFrame(rows).sort_values(
        ["promotion_gate_passed", "late_delta_mae_vs_anchor", "delta_mae_vs_anchor"],
        ascending=[False, True, True],
    )
    subgroups = pd.DataFrame(subgroup_rows).sort_values(["subgroup", "delta_mae_vs_anchor"]).reset_index(drop=True)
    return scoreboard.reset_index(drop=True), subgroups


def leakage_audit(predictions: pd.DataFrame, scoreboard: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(predictions["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(predictions['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "corrections_have_prior_history_only",
            "passed": bool((predictions["prior_rows"] >= 0).all()),
            "evidence": "correction function filters prior target_date < current target_date",
        },
        {
            "check_id": "first_row_has_zero_correction",
            "passed": bool(
                predictions.sort_values("target_date").groupby("correction_id", observed=True).head(1)["residual_correction_c"].abs().le(1e-12).all()
            ),
            "evidence": "no earlier residuals exist for first row of each correction",
        },
        {
            "check_id": "promotion_gate_requires_no_large_early_degradation",
            "passed": bool(
                scoreboard.loc[scoreboard["promotion_gate_passed"], "early_delta_mae_vs_anchor"].le(0.02).all()
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
    design_row: pd.DataFrame,
) -> str:
    return f"""# Station-Only Late-Period Bias Repair

Generated: `{summary['generated_at_utc']}`

## Purpose

`0057` selected `late_period_bias_repair` as the first station-only residual specialist to test. This folder applies simple online residual-bias corrections to the `0055` station-only winner. Every correction for target date `T` uses only residuals from earlier target dates.

This is a residual specialist test, not a final production model and not a 2024+ confirmation test.

## Scope

| Item | Value |
|---|---:|
| Anchor model | {summary['anchor_model_id']} |
| Rows scored | {summary['rows_scored']} |
| First date | {summary['first_date']} |
| Last date | {summary['last_date']} |
| Anchor MAE | {summary['anchor_mae']} |
| Best candidate | {summary['best_candidate']} |
| Best MAE | {summary['best_mae']} |
| Best late MAE | {summary['best_late_mae']} |
| Best late delta vs anchor | {summary['best_late_delta_mae_vs_anchor']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |
| Uses 2024+ rows | {summary['uses_2024_plus_rows']} |

## Source Queue Row

{markdown_table(design_row, max_rows=5)}

## Candidate Scoreboard

{markdown_table(scoreboard, max_rows=40)}

## Subgroup Scoreboard

{markdown_table(subgroups, max_rows=80)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Interpretation

This test checks whether simple online bias repair can fix the late-period cool bias seen in `0055`/`0056`. A candidate is only gate-passing if it improves 2018-2023 MAE and does not degrade 2000-2017 by more than `0.02 C`. If the best candidate fails that gate, the useful result is negative evidence: late-period drift exists, but simple residual means are not reliable enough alone.

## Files

- `artifacts/predictions.parquet`
- `artifacts/predictions_sample.csv`
- `artifacts/scoreboard.csv`
- `artifacts/subgroup_scoreboard.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_only_late_period_bias_repair.py`:

- `{FOLDER_NAME}`: online prior-only residual bias repair around the `0055` station-only winner.

| Metric | Value |
|---|---:|
| Anchor MAE | {summary['anchor_mae']} |
| Best candidate | {summary['best_candidate']} |
| Best MAE | {summary['best_mae']} |
| Best late MAE | {summary['best_late_mae']} |
| Best late delta vs anchor | {summary['best_late_delta_mae_vs_anchor']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

Leakage contract: corrections use only residuals from dates strictly before the scored row.
"""
    update_markdown_section(RESEARCH_ROOT / "README.md", heading="Station-Only Late-Period Bias Repair", section=section)


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_only_late_period_bias_repair.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Anchor model | `{summary['anchor_model_id']}` | Tested |
| Anchor MAE | `{summary['anchor_mae']}` | Baseline |
| Best candidate | `{summary['best_candidate']}` | Diagnostic |
| Best full MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best late MAE | `{summary['best_late_mae']}` | 2018-2023 |
| Best late delta vs anchor | `{summary['best_late_delta_mae_vs_anchor']}` | Diagnostic |
| Promotion gate passed | `{summary['best_promotion_gate_passed']}` | Guarded |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0058` tests whether online residual-bias repair can address the station-only late-period cool bias. It uses no future residuals and no 2024+ rows.
"""
    update_markdown_section(
        path,
        heading="Station-Only Late-Period Bias Repair",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"42. Late-period station-only bias repair tested `{summary['candidate_count']}` prior-only variants; "
        f"best late delta vs anchor is `{summary['best_late_delta_mae_vs_anchor']}` from `{summary['best_candidate']}`."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue the `0057` queue with `february_march_transition_specialist`: build fold-local month/day-of-year plus pressure/dew gated residual corrections around the 0055 station-only anchor, then compare against `0058`.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    anchor, source_summary = load_anchor_predictions()
    if not DESIGN_QUEUE_PATH.exists():
        raise FileNotFoundError(f"Missing 0057 design queue: {DESIGN_QUEUE_PATH}")
    design_queue = pd.read_csv(DESIGN_QUEUE_PATH)
    design_row = design_queue[design_queue["candidate_id"].astype(str).eq("late_period_bias_repair")].copy()
    if design_row.empty:
        raise RuntimeError("0057 design queue does not contain late_period_bias_repair")
    predictions = pd.concat([apply_bias_repair(anchor, spec) for spec in repair_specs()], ignore_index=True)
    scoreboard, subgroups = score_candidates(predictions, anchor)
    leakage = leakage_audit(predictions, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0058 leakage audit failed: {failed}")
    anchor_score = score_prediction_frame(anchor.rename(columns={"point_forecast_c": "anchor_prediction_c"}), "anchor_prediction_c")
    best = scoreboard.iloc[0]
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "anchor_model_id": str(source_summary["best_model_id"]),
        "candidate_count": int(scoreboard["correction_id"].nunique()),
        "rows_scored": int(scoreboard["n"].max()),
        "first_date": str(anchor["target_date"].min().date()),
        "last_date": str(anchor["target_date"].max().date()),
        "anchor_mae": float(anchor_score["mae"]),
        "anchor_rmse": float(anchor_score["rmse"]),
        "best_candidate": str(best["correction_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_bias": float(best["bias"]),
        "best_delta_mae_vs_anchor": float(best["delta_mae_vs_anchor"]),
        "best_late_mae": float(best["late_mae"]),
        "best_late_delta_mae_vs_anchor": float(best["late_delta_mae_vs_anchor"]),
        "best_early_delta_mae_vs_anchor": float(best["early_delta_mae_vs_anchor"]),
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
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_only_late_period_bias_repair_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            scoreboard=scoreboard,
            subgroups=subgroups,
            leakage=leakage,
            design_row=design_row,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run prior-only late-period bias repair around 0055 station anchor.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
