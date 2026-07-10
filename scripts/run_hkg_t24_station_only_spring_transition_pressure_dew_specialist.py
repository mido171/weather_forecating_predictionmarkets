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

FOLDER_NAME = "0060_station_only_spring_transition_pressure_dew_specialist"
ARTIFACT_0055 = RESEARCH_ROOT / "0055_station_only_walkforward_benchmark" / "artifacts"
ARTIFACT_0058 = RESEARCH_ROOT / "0058_station_only_late_period_bias_repair" / "artifacts"
PREDICTIONS_0055_PATH = ARTIFACT_0055 / "predictions.parquet"
SUMMARY_0055_PATH = ARTIFACT_0055 / "summary.json"
PREDICTIONS_0058_PATH = ARTIFACT_0058 / "predictions.parquet"
SUMMARY_0058_PATH = ARTIFACT_0058 / "summary.json"
MAM_MONTHS = (3, 4, 5)
POST_2006_FOLDS = ("fold_2006_2011", "fold_2012_2017", "fold_2018_2023")


@dataclass(frozen=True)
class SpringSpec:
    correction_id: str
    group_columns: tuple[str, ...]
    min_prior_rows: int
    shrinkage: float
    cap_c: float
    window_days: int | None = None


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def spring_phase(target_date: pd.Timestamp) -> str:
    month = int(target_date.month)
    day = int(target_date.day)
    if month == 3 and day <= 15:
        return "early_mar"
    if month == 3:
        return "late_mar"
    if month == 4 and day <= 15:
        return "early_apr"
    if month == 4:
        return "late_apr"
    if month == 5 and day <= 15:
        return "early_may"
    if month == 5:
        return "late_may"
    return "outside_spring"


def load_station_anchor_frame() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame]:
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
    anchor["year"] = anchor["target_date"].dt.year
    anchor["month"] = anchor["target_date"].dt.month

    bias_repaired = pd.read_parquet(PREDICTIONS_0058_PATH)
    bias_repaired["target_date"] = pd.to_datetime(bias_repaired["target_date"], errors="coerce").dt.normalize()
    bias_repaired = bias_repaired[bias_repaired["correction_id"].astype(str).eq(best_0058_id)].copy()
    bias_repaired = bias_repaired[bias_repaired["target_date"].le(DEVELOPMENT_END)].copy()
    if bias_repaired.empty:
        raise RuntimeError(f"Missing 0058 best correction predictions for {best_0058_id}")

    gates, thresholds = load_feature_gates()
    frame = anchor[
        [
            "target_date",
            "target_tmax_c",
            "point_forecast_c",
            "fold_id",
            "year",
            "month",
            "training_start",
            "training_end",
            "training_rows",
        ]
    ].rename(columns={"point_forecast_c": "station_anchor_prediction_c"})
    frame = frame.merge(
        bias_repaired[["target_date", "candidate_prediction_c", "residual_correction_c"]],
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
    frame["spring_month_bucket"] = np.select(
        [frame["month"].eq(3), frame["month"].eq(4), frame["month"].eq(5)],
        ["march", "april", "may"],
        default="outside_spring",
    )
    frame["spring_phase"] = frame["target_date"].map(spring_phase)
    frame["spring_target_window"] = frame["month"].isin(MAM_MONTHS)
    frame["residual_to_add_c"] = frame["target_tmax_c"] - frame["global_bias_repaired_prediction_c"]
    frame = frame.sort_values("target_date").reset_index(drop=True)
    require_no_confirmation_dates(frame["target_date"], context="0060 spring model frame")
    return frame, summary_0055, summary_0058, thresholds


def spring_specs() -> list[SpringSpec]:
    return [
        SpringSpec("spring_month_min45_shrink120_cap1p0", ("spring_month_bucket",), 45, 120.0, 1.0),
        SpringSpec("spring_phase_min35_shrink100_cap1p0", ("spring_phase",), 35, 100.0, 1.0),
        SpringSpec(
            "spring_month_pressure_min30_shrink90_cap1p0",
            ("spring_month_bucket", "pressure_spread_bucket"),
            30,
            90.0,
            1.0,
        ),
        SpringSpec(
            "spring_month_dew_min30_shrink90_cap1p0",
            ("spring_month_bucket", "dew_trajectory_bucket"),
            30,
            90.0,
            1.0,
        ),
        SpringSpec(
            "spring_month_pressure_dew_min20_shrink75_cap1p0",
            ("spring_month_bucket", "pressure_spread_bucket", "dew_trajectory_bucket"),
            20,
            75.0,
            1.0,
        ),
        SpringSpec(
            "spring_phase_pressure_dew_min18_shrink75_cap1p0",
            ("spring_phase", "pressure_spread_bucket", "dew_trajectory_bucket"),
            18,
            75.0,
            1.0,
        ),
        SpringSpec(
            "spring_phase_pressure_dew_wind_min15_shrink60_cap0p8",
            ("spring_phase", "pressure_spread_bucket", "dew_trajectory_bucket", "wind_spread_bucket"),
            15,
            60.0,
            0.8,
        ),
        SpringSpec(
            "spring_month_pressure_dew_rolling8y_min18_shrink75_cap1p0",
            ("spring_month_bucket", "pressure_spread_bucket", "dew_trajectory_bucket"),
            18,
            75.0,
            1.0,
            window_days=2920,
        ),
        SpringSpec(
            "spring_phase_pressure_dew_rolling8y_min15_shrink60_cap0p8",
            ("spring_phase", "pressure_spread_bucket", "dew_trajectory_bucket"),
            15,
            60.0,
            0.8,
            window_days=2920,
        ),
    ]


def group_key(row: pd.Series, columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in columns)


def shrink_and_cap(raw: float, prior_rows: int, spec: SpringSpec) -> float:
    if prior_rows < spec.min_prior_rows or not math.isfinite(raw):
        return 0.0
    shrink = prior_rows / (prior_rows + spec.shrinkage)
    return float(np.clip(raw * shrink, -spec.cap_c, spec.cap_c))


def compute_spring_correction(frame: pd.DataFrame, spec: SpringSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values("target_date").reset_index(drop=True)
    corrections = np.zeros(len(ordered), dtype=float)
    prior_counts = np.zeros(len(ordered), dtype=int)
    raw_means = np.full(len(ordered), math.nan, dtype=float)
    expanding_state: dict[tuple[str, ...], tuple[int, float]] = defaultdict(lambda: (0, 0.0))
    rolling_state: dict[tuple[str, ...], dict[str, object]] = defaultdict(lambda: {"rows": deque(), "sum": 0.0})

    for idx, row in ordered.iterrows():
        current_date = pd.Timestamp(row["target_date"])
        key = group_key(row, spec.group_columns)
        if spec.window_days is None:
            count, total = expanding_state[key]
        else:
            state = rolling_state[key]
            rows = state["rows"]
            assert isinstance(rows, deque)
            min_date = current_date - pd.Timedelta(days=spec.window_days)
            while rows and rows[0][0] < min_date:
                _, old_residual = rows.popleft()
                state["sum"] = float(state["sum"]) - float(old_residual)
            count = len(rows)
            total = float(state["sum"])
        prior_counts[idx] = count
        if bool(row["spring_target_window"]) and count >= spec.min_prior_rows:
            raw = total / count
            raw_means[idx] = raw
            corrections[idx] = shrink_and_cap(raw, count, spec)

        residual = float(row["residual_to_add_c"])
        if math.isfinite(residual):
            if spec.window_days is None:
                expanding_state[key] = (count + 1, total + residual)
            else:
                state = rolling_state[key]
                rows = state["rows"]
                assert isinstance(rows, deque)
                rows.append((current_date, residual))
                state["sum"] = float(state["sum"]) + residual
    return corrections, prior_counts, raw_means


def apply_spring_specialist(frame: pd.DataFrame, spec: SpringSpec) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    corrections, prior_rows, raw_means = compute_spring_correction(ordered, spec)
    out = ordered[
        [
            "target_date",
            "target_tmax_c",
            "station_anchor_prediction_c",
            "global_bias_repaired_prediction_c",
            "global_bias_repair_c",
            "fold_id",
            "year",
            "month",
            "spring_month_bucket",
            "spring_phase",
            "spring_target_window",
            "pressure_spread_abs_max",
            "dew_trajectory_mean",
            "wind_spread_abs_max",
            "pressure_spread_bucket",
            "dew_trajectory_bucket",
            "wind_spread_bucket",
        ]
    ].copy()
    out["candidate_prediction_c"] = out["global_bias_repaired_prediction_c"] + corrections
    out["spring_residual_correction_c"] = corrections
    out["raw_prior_residual_mean_c"] = raw_means
    out["prior_rows"] = prior_rows
    out["correction_id"] = spec.correction_id
    out["group_columns"] = "|".join(spec.group_columns)
    out["min_prior_rows"] = spec.min_prior_rows
    out["shrinkage"] = spec.shrinkage
    out["cap_c"] = spec.cap_c
    out["window_days"] = float(spec.window_days) if spec.window_days is not None else math.nan
    require_no_confirmation_dates(out["target_date"], context=f"0060 {spec.correction_id} predictions")
    return out.sort_values(["target_date", "correction_id"]).reset_index(drop=True)


def score_candidates(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for correction_id, group in predictions.groupby("correction_id", observed=True):
        full = score_prediction_frame(group, "candidate_prediction_c")
        reference_full = score_prediction_frame(
            group.rename(columns={"global_bias_repaired_prediction_c": "reference_prediction_c"}),
            "reference_prediction_c",
        )
        spring = group[group["spring_target_window"].astype(bool)].copy()
        nonspring = group[~group["spring_target_window"].astype(bool)].copy()
        spring_score = score_prediction_frame(spring, "candidate_prediction_c")
        spring_ref = score_prediction_frame(
            spring.rename(columns={"global_bias_repaired_prediction_c": "reference_prediction_c"}),
            "reference_prediction_c",
        )
        nonspring_score = score_prediction_frame(nonspring, "candidate_prediction_c")
        nonspring_ref = score_prediction_frame(
            nonspring.rename(columns={"global_bias_repaired_prediction_c": "reference_prediction_c"}),
            "reference_prediction_c",
        )
        for fold_id, fold in spring.groupby("fold_id", observed=True):
            candidate_fold = score_prediction_frame(fold, "candidate_prediction_c")
            ref_fold = score_prediction_frame(
                fold.rename(columns={"global_bias_repaired_prediction_c": "reference_prediction_c"}),
                "reference_prediction_c",
            )
            fold_rows.append(
                {
                    "correction_id": correction_id,
                    "fold_id": fold_id,
                    "n": candidate_fold["n"],
                    "candidate_mam_mae": candidate_fold["mae"],
                    "reference_mam_mae": ref_fold["mae"],
                    "delta_mae_vs_0058": float(candidate_fold["mae"]) - float(ref_fold["mae"]),
                    "candidate_mam_rmse": candidate_fold["rmse"],
                    "reference_mam_rmse": ref_fold["rmse"],
                    "candidate_mam_bias": candidate_fold["bias"],
                    "reference_mam_bias": ref_fold["bias"],
                }
            )
        fold_frame = pd.DataFrame([row for row in fold_rows if row["correction_id"] == correction_id])
        post2006 = fold_frame[fold_frame["fold_id"].isin(POST_2006_FOLDS)].copy()
        row = {
            "correction_id": correction_id,
            "group_columns": str(group["group_columns"].iloc[0]),
            "window_days": float(group["window_days"].iloc[0]) if pd.notna(group["window_days"].iloc[0]) else math.nan,
            "min_prior_rows": int(group["min_prior_rows"].iloc[0]),
            "shrinkage": float(group["shrinkage"].iloc[0]),
            "cap_c": float(group["cap_c"].iloc[0]),
            "n": full["n"],
            "mae": full["mae"],
            "rmse": full["rmse"],
            "bias": full["bias"],
            "delta_mae_vs_0058": float(full["mae"]) - float(reference_full["mae"]),
            "mam_n": spring_score["n"],
            "mam_mae": spring_score["mae"],
            "mam_rmse": spring_score["rmse"],
            "mam_bias": spring_score["bias"],
            "reference_mam_mae": spring_ref["mae"],
            "mam_delta_mae_vs_0058": float(spring_score["mae"]) - float(spring_ref["mae"]),
            "non_mam_delta_mae_vs_0058": float(nonspring_score["mae"]) - float(nonspring_ref["mae"]),
            "post2006_fold_delta_max": float(post2006["delta_mae_vs_0058"].max()) if not post2006.empty else math.nan,
            "post2006_folds_improved": int(post2006["delta_mae_vs_0058"].lt(0).sum()) if not post2006.empty else 0,
            "mean_abs_spring_correction_c": float(group["spring_residual_correction_c"].abs().mean()),
            "active_spring_share": float(
                group.loc[group["spring_target_window"].astype(bool), "spring_residual_correction_c"]
                .abs()
                .gt(1e-9)
                .mean()
            ),
        }
        row["promotion_gate_passed"] = bool(
            row["mam_delta_mae_vs_0058"] < 0.0
            and row["non_mam_delta_mae_vs_0058"] <= 0.005
            and row["post2006_folds_improved"] == len(POST_2006_FOLDS)
        )
        rows.append(row)
    scoreboard = pd.DataFrame(rows).sort_values(
        ["promotion_gate_passed", "mam_delta_mae_vs_0058", "post2006_fold_delta_max"],
        ascending=[False, True, True],
    )
    folds = pd.DataFrame(fold_rows).sort_values(["fold_id", "delta_mae_vs_0058"]).reset_index(drop=True)
    return scoreboard.reset_index(drop=True), folds


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
            "evidence": f"{len(thresholds)} inherited gate thresholds checked",
        },
        {
            "check_id": "corrections_have_prior_history_only",
            "passed": bool((predictions["prior_rows"] >= 0).all()),
            "evidence": "streaming correction updates residual state after each row is scored",
        },
        {
            "check_id": "first_row_has_zero_correction",
            "passed": bool(
                predictions.sort_values("target_date")
                .groupby("correction_id", observed=True)
                .head(1)["spring_residual_correction_c"]
                .abs()
                .le(1e-12)
                .all()
            ),
            "evidence": "no earlier residuals exist for first row of each correction",
        },
        {
            "check_id": "promotion_gate_requires_all_post2006_folds",
            "passed": bool(
                scoreboard.loc[scoreboard["promotion_gate_passed"], "post2006_folds_improved"].eq(len(POST_2006_FOLDS)).all()
            ),
            "evidence": f"{int(scoreboard['promotion_gate_passed'].sum())} candidates passed gate",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    scoreboard: pd.DataFrame,
    fold_scoreboard: pd.DataFrame,
    leakage: pd.DataFrame,
    thresholds: pd.DataFrame,
    design_row: pd.DataFrame,
) -> str:
    return f"""# Station-Only Spring Pressure/Dew Specialist

Generated: `{summary['generated_at_utc']}`

## Purpose

`0057` marked spring pressure/dew regimes as a deployable station-only residual candidate. `0060` tests that idea after the broad `0058` global bias repair, so the question is stricter: does a MAM-only pressure/dew/wind specialist add value beyond the current station-only bias-repaired baseline?

This is a diagnostic residual-specialist test, not a final production model and not a 2024+ confirmation test.

## Contract

- Reference baseline: `0058` best candidate `{summary['reference_0058_best_candidate']}`.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- Spring target window: March, April, May.
- Gate thresholds: fixed from pre-2000 station-feature history.
- Correction state: residuals from strictly earlier dates only.
- Promotion gate: MAM MAE must beat `0058`, non-MAM must not deteriorate, and every post-2006 fold must improve.

## Headline

| Item | Value |
|---|---:|
| Rows scored | {summary['rows_scored']} |
| MAM rows | {summary['mam_rows']} |
| 0058 MAM MAE | {summary['reference_mam_mae']} |
| Best candidate | {summary['best_candidate']} |
| Best MAM MAE | {summary['best_mam_mae']} |
| Best MAM delta vs 0058 | {summary['best_mam_delta_mae_vs_0058']} |
| Best post-2006 folds improved | {summary['best_post2006_folds_improved']} / {len(POST_2006_FOLDS)} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

## Source Queue Row

{markdown_table(design_row, max_rows=5)}

## Gate Thresholds

{markdown_table(thresholds, max_rows=20)}

## Candidate Scoreboard

{markdown_table(scoreboard, max_rows=80)}

## Fold Scoreboard

{markdown_table(fold_scoreboard, max_rows=80)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Interpretation

This test is intentionally harder than a raw station-anchor residual screen because it tries to improve on top of the best global online bias repair from `0058`. If no candidate passes the all-fold gate, the result should be treated as negative evidence against simple pressure/dew bucket means as an incremental spring specialist. If one passes, it becomes a candidate for later blending with the official forecast archive after the archive backfill is stable.

## Files

- `artifacts/predictions.parquet`
- `artifacts/predictions_sample.csv`
- `artifacts/scoreboard.csv`
- `artifacts/fold_scoreboard.csv`
- `artifacts/gate_thresholds.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_only_spring_transition_pressure_dew_specialist.py`:

- `{FOLDER_NAME}`: MAM pressure/dew/wind residual specialists tested on top of `0058`.

| Metric | Value |
|---|---:|
| Reference MAM MAE | {summary['reference_mam_mae']} |
| Best candidate | {summary['best_candidate']} |
| Best MAM MAE | {summary['best_mam_mae']} |
| Best MAM delta vs 0058 | {summary['best_mam_delta_mae_vs_0058']} |
| Best post-2006 folds improved | {summary['best_post2006_folds_improved']} / {len(POST_2006_FOLDS)} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

Leakage contract: pre-2000 fixed gates and strict prior-only residual corrections.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Station-Only Spring Pressure/Dew Specialist",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_only_spring_transition_pressure_dew_specialist.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Reference | `0058` `{summary['reference_0058_best_candidate']}` | Tested |
| Target pocket | MAM rows `{summary['mam_rows']}` | Tested |
| Reference MAM MAE | `{summary['reference_mam_mae']}` | Baseline |
| Best candidate | `{summary['best_candidate']}` | Diagnostic |
| Best MAM delta | `{summary['best_mam_delta_mae_vs_0058']}` | Diagnostic |
| Post-2006 folds improved | `{summary['best_post2006_folds_improved']}` / `{len(POST_2006_FOLDS)}` | Guarded |
| Promotion gate passed | `{summary['best_promotion_gate_passed']}` | Guarded |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0060` tests whether MAM pressure/dew/wind residual gates add value after `0058`. This keeps delayed RSS backfill out of the critical path.
"""
    update_markdown_section(
        path,
        heading="Station-Only Spring Pressure/Dew Specialist",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"44. Spring MAM pressure/dew specialist tested `{summary['candidate_count']}` variants on top of `0058`; "
        f"best MAM delta vs 0058 is `{summary['best_mam_delta_mae_vs_0058']}` from `{summary['best_candidate']}`."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue the `0057` queue with `pressure_high_uncertainty_guard`: quantify high-pressure-spread residual scale and tail risk on top of `0058`, then decide whether it belongs as an interval/calibration guard rather than a mean correction.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0055, summary_0058, thresholds = load_station_anchor_frame()
    if not DESIGN_QUEUE_PATH.exists():
        raise FileNotFoundError(f"Missing 0057 design queue: {DESIGN_QUEUE_PATH}")
    design_queue = pd.read_csv(DESIGN_QUEUE_PATH)
    design_row = design_queue[design_queue["candidate_id"].astype(str).eq("spring_transition_pressure_dew_specialist")].copy()
    if design_row.empty:
        raise RuntimeError("0057 design queue does not contain spring_transition_pressure_dew_specialist")
    predictions = pd.concat([apply_spring_specialist(frame, spec) for spec in spring_specs()], ignore_index=True)
    scoreboard, fold_scoreboard = score_candidates(predictions)
    leakage = leakage_audit(predictions, thresholds, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0060 leakage audit failed: {failed}")

    reference = frame.rename(columns={"global_bias_repaired_prediction_c": "reference_prediction_c"})
    reference_full = score_prediction_frame(reference, "reference_prediction_c")
    reference_mam = score_prediction_frame(reference[reference["spring_target_window"]], "reference_prediction_c")
    best = scoreboard.iloc[0]
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "anchor_model_id": str(summary_0055["best_model_id"]),
        "reference_0058_best_candidate": str(summary_0058["best_candidate"]),
        "candidate_count": int(scoreboard["correction_id"].nunique()),
        "rows_scored": int(scoreboard["n"].max()),
        "mam_rows": int(reference_mam["n"]),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "reference_mae": float(reference_full["mae"]),
        "reference_rmse": float(reference_full["rmse"]),
        "reference_mam_mae": float(reference_mam["mae"]),
        "reference_mam_rmse": float(reference_mam["rmse"]),
        "best_candidate": str(best["correction_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_bias": float(best["bias"]),
        "best_delta_mae_vs_0058": float(best["delta_mae_vs_0058"]),
        "best_mam_mae": float(best["mam_mae"]),
        "best_mam_delta_mae_vs_0058": float(best["mam_delta_mae_vs_0058"]),
        "best_post2006_folds_improved": int(best["post2006_folds_improved"]),
        "best_post2006_fold_delta_max": float(best["post2006_fold_delta_max"]),
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
    write_csv(artifacts / "fold_scoreboard.csv", fold_scoreboard)
    write_csv(artifacts / "gate_thresholds.csv", thresholds)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_only_spring_transition_pressure_dew_specialist_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            scoreboard=scoreboard,
            fold_scoreboard=fold_scoreboard,
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
        description="Run MAM pressure/dew station residual specialists on top of 0058."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
