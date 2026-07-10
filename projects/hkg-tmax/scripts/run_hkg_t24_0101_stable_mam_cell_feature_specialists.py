from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_0086_guarded_long_history_residual_specialists import ResidualState
from scripts.run_hkg_t24_0095_mam_error_direction_split_lab import BASE_ID, evaluation_masks_0095, safe_token, short_hash
from scripts.run_hkg_t24_0097_stable_directional_cell_specialist import score_with_0094_delta
from scripts.run_hkg_t24_0100_stable_mam_cell_feature_atlas import (
    INPUT_0099_DIAGNOSTICS_PATH,
    INPUT_0099_SUMMARY_PATH,
    INPUT_0099_TOP_PATH,
    build_eval_frame,
    load_feature_matrix,
)
from scripts.run_hkg_t24_beastmode_signal_discovery import (
    CONFIRMATION_START,
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import update_markdown_section

FOLDER_NAME = "0101_stable_mam_cell_feature_specialists"
CANDIDATE_CLASS = "0101_stable_mam_cell_feature_specialists"
INPUT_0100_ATLAS_PATH = RESEARCH_ROOT / "0100_stable_mam_cell_feature_atlas" / "artifacts" / "feature_atlas.csv"
TARGET_MEMORY_FEATURE_LIMIT = 12
STATION_FEATURE_LIMIT = 8
BUCKET_COUNTS = (3, 5)
GATE_SCOPES = ("agreement", "specialist_active")
MIN_HISTORY_GRID = (12, 20)
CORRECTION_CAP_GRID = (0.15, 0.25)
SHRINK_ROWS = 30.0
MIN_ABS_PRIOR_MEAN_C = 0.03
EVAL_START = pd.Timestamp("2000-01-01")


@dataclass(frozen=True)
class FeatureSpecialistSpec:
    candidate_id: str
    feature: str
    family: str
    bucket_count: int
    bin_edges: tuple[float, ...]
    gate_scope: str
    min_history: int
    shrink_rows: float
    correction_cap_c: float
    min_abs_prior_mean_c: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def float_token(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def fixed_quantile_edges(values: pd.Series, *, bucket_count: int) -> tuple[float, ...]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < bucket_count * 20 or clean.nunique(dropna=True) < bucket_count:
        return ()
    quantiles = [i / bucket_count for i in range(1, bucket_count)]
    raw_edges = np.quantile(clean.to_numpy(dtype=float), quantiles)
    edges = sorted({float(edge) for edge in raw_edges if math.isfinite(float(edge))})
    return tuple(edges)


def assign_bucket_from_edges(values: pd.Series, edges: tuple[float, ...]) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    buckets = np.full(len(numeric), -1, dtype=int)
    finite = np.isfinite(numeric)
    if edges:
        buckets[finite] = np.searchsorted(np.asarray(edges, dtype=float), numeric[finite], side="right")
    elif finite.any():
        buckets[finite] = 0
    return buckets


def load_0099_and_0100_inputs() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [
        path
        for path in (INPUT_0099_SUMMARY_PATH, INPUT_0099_TOP_PATH, INPUT_0099_DIAGNOSTICS_PATH, INPUT_0100_ATLAS_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"0101 requires 0099 and 0100 artifacts first: {missing}")
    summary_0099 = json.loads(INPUT_0099_SUMMARY_PATH.read_text(encoding="utf-8"))
    top_0099 = pd.read_csv(INPUT_0099_TOP_PATH)
    diagnostics_0099 = pd.read_csv(INPUT_0099_DIAGNOSTICS_PATH)
    atlas = pd.read_csv(INPUT_0100_ATLAS_PATH)
    for frame, context in ((top_0099, "0101 0099 top"), (diagnostics_0099, "0101 0099 diagnostics")):
        frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
        frame.dropna(subset=["target_date"], inplace=True)
        frame.sort_values("target_date", inplace=True)
        require_no_confirmation_dates(frame["target_date"], context=context)
    return summary_0099, top_0099, diagnostics_0099, atlas, load_feature_matrix()


def select_candidate_features(atlas: pd.DataFrame) -> pd.DataFrame:
    allowed = atlas[atlas["allowed_for_future_walkforward"].astype(bool)].copy()
    allowed = allowed[allowed["family"].isin(["target_memory", "isd_station_network"])].copy()
    allowed["diagnostic_score"] = pd.to_numeric(allowed["diagnostic_score"], errors="coerce")
    selected = pd.concat(
        [
            allowed[allowed["family"].eq("target_memory")]
            .sort_values(["diagnostic_score", "feature"], ascending=[False, True])
            .head(TARGET_MEMORY_FEATURE_LIMIT),
            allowed[allowed["family"].eq("isd_station_network")]
            .sort_values(["diagnostic_score", "feature"], ascending=[False, True])
            .head(STATION_FEATURE_LIMIT),
        ],
        ignore_index=True,
    )
    return selected.drop_duplicates(subset=["feature"]).reset_index(drop=True)


def make_specs(selected_features: pd.DataFrame, full_features: pd.DataFrame) -> tuple[list[FeatureSpecialistSpec], pd.DataFrame]:
    prehistory = full_features[full_features["target_date"] < EVAL_START].copy()
    specs: list[FeatureSpecialistSpec] = []
    rows: list[dict[str, object]] = []
    for feature_row in selected_features.to_dict("records"):
        feature = str(feature_row["feature"])
        family = str(feature_row["family"])
        if feature not in full_features.columns:
            continue
        for bucket_count in BUCKET_COUNTS:
            edges = fixed_quantile_edges(prehistory[feature], bucket_count=bucket_count)
            if len(edges) != bucket_count - 1:
                rows.append(
                    {
                        "feature": feature,
                        "family": family,
                        "bucket_count": bucket_count,
                        "status": "skipped_insufficient_pre2000_edges",
                        "pre2000_non_null_rows": int(pd.to_numeric(prehistory[feature], errors="coerce").notna().sum()),
                        "bin_edges": ";".join(f"{edge:.6g}" for edge in edges),
                    }
                )
                continue
            for gate_scope in GATE_SCOPES:
                for min_history in MIN_HISTORY_GRID:
                    for cap in CORRECTION_CAP_GRID:
                        candidate_id = (
                            f"featcell_{short_hash(feature, family, str(bucket_count), gate_scope, str(min_history), str(cap))}_"
                            f"{safe_token(feature, max_len=28)}_q{bucket_count}_{gate_scope}_m{min_history}_c{float_token(cap)}"
                        )
                        specs.append(
                            FeatureSpecialistSpec(
                                candidate_id=candidate_id,
                                feature=feature,
                                family=family,
                                bucket_count=bucket_count,
                                bin_edges=edges,
                                gate_scope=gate_scope,
                                min_history=min_history,
                                shrink_rows=SHRINK_ROWS,
                                correction_cap_c=cap,
                                min_abs_prior_mean_c=MIN_ABS_PRIOR_MEAN_C,
                            )
                        )
            rows.append(
                {
                    "feature": feature,
                    "family": family,
                    "bucket_count": bucket_count,
                    "status": "usable",
                    "pre2000_non_null_rows": int(pd.to_numeric(prehistory[feature], errors="coerce").notna().sum()),
                    "bin_edges": ";".join(f"{edge:.6g}" for edge in edges),
                }
            )
    return specs, pd.DataFrame(rows)


def gate_mask_for_scope(frame: pd.DataFrame, gate_scope: str) -> np.ndarray:
    if gate_scope == "agreement":
        return frame["agreement_row"].astype(bool).to_numpy(dtype=bool)
    if gate_scope == "specialist_active":
        return frame["specialist_active_row"].astype(bool).to_numpy(dtype=bool)
    raise ValueError(f"Unsupported 0101 gate scope: {gate_scope}")


def apply_feature_specialist(frame: pd.DataFrame, spec: FeatureSpecialistSpec) -> tuple[np.ndarray, pd.DataFrame]:
    base = frame["best_0099_prediction_c"].to_numpy(dtype=float)
    residual = frame["best_0099_error_c"].to_numpy(dtype=float)
    buckets = assign_bucket_from_edges(frame[spec.feature], spec.bin_edges)
    gate_active = gate_mask_for_scope(frame, spec.gate_scope)
    predictions = base.copy()
    active = np.zeros(len(frame), dtype=bool)
    prior_rows = np.zeros(len(frame), dtype=int)
    prior_mean = np.full(len(frame), np.nan, dtype=float)
    corrections = np.zeros(len(frame), dtype=float)
    states: dict[int, ResidualState] = {}

    for _, date_group in frame.groupby("target_date", sort=True, observed=True):
        pending_updates: list[tuple[int, float]] = []
        for idx in date_group.index:
            row_idx = int(idx)
            bucket = int(buckets[row_idx])
            if not gate_active[row_idx] or bucket < 0:
                continue
            state = states.setdefault(bucket, ResidualState())
            prior_rows[row_idx] = state.count
            mean_residual = state.mean()
            prior_mean[row_idx] = mean_residual if state.count > 0 else math.nan
            if state.count >= spec.min_history and abs(mean_residual) >= spec.min_abs_prior_mean_c:
                shrink = state.count / (state.count + spec.shrink_rows)
                correction = float(np.clip(mean_residual * shrink, -spec.correction_cap_c, spec.correction_cap_c))
                predictions[row_idx] = base[row_idx] - correction
                corrections[row_idx] = correction
                active[row_idx] = abs(correction) > 1e-12
            if math.isfinite(residual[row_idx]):
                pending_updates.append((bucket, float(residual[row_idx])))
        for bucket, residual_value in pending_updates:
            states[bucket].update(residual_value)

    diagnostics = frame[
        [
            "target_date",
            "forecast_source_family",
            "season",
            "frame_segment",
            "era_bucket",
            "agreement_row",
            "specialist_active_row",
        ]
    ].copy()
    diagnostics["candidate_id"] = spec.candidate_id
    diagnostics["feature"] = spec.feature
    diagnostics["family"] = spec.family
    diagnostics["gate_scope"] = spec.gate_scope
    diagnostics["feature_bucket"] = buckets
    diagnostics["gate_active_row"] = gate_active
    diagnostics["prior_rows"] = prior_rows
    diagnostics["prior_mean_residual_c"] = prior_mean
    diagnostics["specialist_active"] = active
    diagnostics["specialist_correction_c"] = corrections
    diagnostics["candidate_prediction_c"] = predictions
    diagnostics["candidate_error_c"] = predictions - frame["target_tmax_c"].to_numpy(dtype=float)
    return predictions, diagnostics


def score_against_0099(
    frame: pd.DataFrame,
    *,
    candidate_id: str,
    candidate_class: str,
    prediction: np.ndarray,
    mask_map: dict[str, np.ndarray],
    input_0099_prediction: np.ndarray,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row = score_with_0094_delta(
        frame,
        candidate_id=candidate_id,
        candidate_class=candidate_class,
        prediction=prediction,
        mask_map=mask_map,
        extra=extra,
    )
    target = frame["target_tmax_c"].to_numpy(dtype=float)
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    from scripts.run_hkg_t24_0090_guarded_specialists_from_error_autopsy import score_arrays

    candidate_score = score_arrays(target=target, prediction=prediction, dates=dates)
    input_score = score_arrays(target=target, prediction=input_0099_prediction, dates=dates)
    row["delta_mae_vs_0099"] = float(candidate_score["mae"]) - float(input_score["mae"])
    delta_keys = ["delta_mae_vs_0099"]
    for prefix, mask in mask_map.items():
        scored = score_arrays(target=target[mask], prediction=prediction[mask], dates=dates[mask], prefix=prefix)
        input_scored = score_arrays(target=target[mask], prediction=input_0099_prediction[mask], dates=dates[mask], prefix=prefix)
        key = f"{prefix}delta_mae_vs_0099"
        row[key] = float(scored[f"{prefix}mae"]) - float(input_scored[f"{prefix}mae"])
        delta_keys.append(key)
    row["hardened_vs_0099_gate_passed"] = (
        float(row["delta_mae_vs_0099"]) < 0.0
        and all(float(row[key]) <= 0.0 for key in delta_keys[1:])
    )
    return row


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    summary_0099, top_0099, diagnostics_0099, atlas, full_features = load_0099_and_0100_inputs()
    eval_frame = build_eval_frame(full_features, top_0099, diagnostics_0099)
    selected_features = select_candidate_features(atlas)
    specs, bin_audit = make_specs(selected_features, full_features)
    score_frame = eval_frame.copy()
    score_frame["candidate_prediction_c"] = score_frame["base_0094_prediction_c"]
    mask_map = evaluation_masks_0095(score_frame)
    input_0099_prediction = score_frame["best_0099_prediction_c"].to_numpy(dtype=float)
    raw_prediction = score_frame["forecast_max_c"].to_numpy(dtype=float)
    base_0094_prediction = score_frame["base_0094_prediction_c"].to_numpy(dtype=float)
    rows = [
        score_against_0099(
            score_frame,
            candidate_id="official_raw",
            candidate_class="official_raw",
            prediction=raw_prediction,
            mask_map=mask_map,
            input_0099_prediction=input_0099_prediction,
        ),
        score_against_0099(
            score_frame,
            candidate_id=BASE_ID,
            candidate_class="0094_base",
            prediction=base_0094_prediction,
            mask_map=mask_map,
            input_0099_prediction=input_0099_prediction,
        ),
        score_against_0099(
            score_frame,
            candidate_id=str(summary_0099["best_0099_candidate"]),
            candidate_class="0099_mam_cell_policy_sensitivity",
            prediction=input_0099_prediction,
            mask_map=mask_map,
            input_0099_prediction=input_0099_prediction,
            extra={"source_experiment": "0099"},
        ),
    ]
    definitions: list[dict[str, object]] = []
    predictions_by_id: dict[str, np.ndarray] = {str(summary_0099["best_0099_candidate"]): input_0099_prediction}
    diagnostics_by_id: dict[str, pd.DataFrame] = {}
    for spec in specs:
        prediction, diagnostics = apply_feature_specialist(score_frame, spec)
        predictions_by_id[spec.candidate_id] = prediction
        diagnostics_by_id[spec.candidate_id] = diagnostics
        definitions.append(
            {
                "candidate_id": spec.candidate_id,
                "candidate_class": CANDIDATE_CLASS,
                "feature": spec.feature,
                "family": spec.family,
                "bucket_count": spec.bucket_count,
                "bin_edges": ";".join(f"{edge:.6g}" for edge in spec.bin_edges),
                "gate_scope": spec.gate_scope,
                "min_history": spec.min_history,
                "shrink_rows": spec.shrink_rows,
                "correction_cap_c": spec.correction_cap_c,
                "min_abs_prior_mean_c": spec.min_abs_prior_mean_c,
            }
        )
        rows.append(
            score_against_0099(
                score_frame,
                candidate_id=spec.candidate_id,
                candidate_class=CANDIDATE_CLASS,
                prediction=prediction,
                mask_map=mask_map,
                input_0099_prediction=input_0099_prediction,
                extra={
                    "feature": spec.feature,
                    "family": spec.family,
                    "bucket_count": spec.bucket_count,
                    "gate_scope": spec.gate_scope,
                    "min_history": spec.min_history,
                    "shrink_rows": spec.shrink_rows,
                    "correction_cap_c": spec.correction_cap_c,
                    "min_abs_prior_mean_c": spec.min_abs_prior_mean_c,
                    "specialist_active_rows": int(diagnostics["specialist_active"].sum()),
                    "mean_active_correction_c": float(
                        diagnostics.loc[diagnostics["specialist_active"].astype(bool), "specialist_correction_c"].mean()
                    )
                    if diagnostics["specialist_active"].any()
                    else 0.0,
                },
            )
        )

    scoreboard = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    candidate_definitions = pd.DataFrame(definitions)
    candidate_rows = scoreboard[scoreboard["candidate_class"].eq(CANDIDATE_CLASS)].copy()
    eligible = candidate_rows[
        candidate_rows["hardened_vs_0099_gate_passed"].astype(bool)
        & (pd.to_numeric(candidate_rows["delta_mae_vs_0099"], errors="coerce") < 0.0)
    ].copy()
    if eligible.empty:
        best_row = scoreboard[scoreboard["candidate_class"].eq("0099_mam_cell_policy_sensitivity")].iloc[0]
        best_prediction = input_0099_prediction
    else:
        best_row = eligible.sort_values(["mae", "rmse"]).iloc[0]
        best_prediction = predictions_by_id[str(best_row["candidate_id"])]
    top_predictions = score_frame[
        [
            "target_date",
            "forecast_source_family",
            "target_tmax_c",
            "forecast_max_c",
            "season",
            "frame_segment",
            "era_bucket",
        ]
    ].copy()
    top_predictions["candidate_id"] = str(best_row["candidate_id"])
    top_predictions["candidate_prediction_c"] = best_prediction
    top_predictions["candidate_error_c"] = top_predictions["candidate_prediction_c"] - top_predictions["target_tmax_c"]
    raw_best = candidate_rows.sort_values(["mae", "rmse"]).iloc[0] if not candidate_rows.empty else best_row
    best_diagnostics = diagnostics_by_id.get(str(raw_best["candidate_id"]), pd.DataFrame())
    dates = pd.to_datetime(score_frame["target_date"], errors="coerce")
    summary = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "rows": int(len(score_frame)),
        "first_target_date": dates.min().date().isoformat(),
        "last_target_date": dates.max().date().isoformat(),
        "input_0099_candidate": summary_0099["best_0099_candidate"],
        "input_0099_mae": float(summary_0099["best_0099_mae"]),
        "input_0099_rmse": float(summary_0099["best_0099_rmse"]),
        "selected_feature_count": int(len(selected_features)),
        "candidate_count": int(len(candidate_rows)),
        "hardened_vs_0099_count": int(candidate_rows["hardened_vs_0099_gate_passed"].astype(bool).sum())
        if not candidate_rows.empty
        else 0,
        "best_raw_0101_candidate": str(raw_best["candidate_id"]),
        "best_raw_0101_mae": float(raw_best["mae"]),
        "best_raw_0101_rmse": float(raw_best["rmse"]),
        "best_raw_0101_delta_mae_vs_0099": float(raw_best["delta_mae_vs_0099"]),
        "best_candidate": str(best_row["candidate_id"]),
        "best_candidate_class": str(best_row["candidate_class"]),
        "best_mae": float(best_row["mae"]),
        "best_rmse": float(best_row["rmse"]),
        "best_delta_mae_vs_0099": float(best_row["delta_mae_vs_0099"]),
        "new_0101_champion": bool(str(best_row["candidate_class"]) == CANDIDATE_CLASS),
        "agreement_rows": int(score_frame["agreement_row"].sum()),
        "specialist_active_rows": int(score_frame["specialist_active_row"].sum()),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "stable_mam_cell_feature_specialists_complete",
        "next_recommended_task": (
            "Run 0102_timestamp_proof_unlock_queue: attach issue/available-at proof for the high-scoring "
            "0100 upper-air and daily marine features, then rerun 0101 with newly eligible families only if "
            "the timestamp audit passes."
        ),
    }
    require_no_confirmation_dates(top_predictions["target_date"], context="0101 top predictions")
    return scoreboard, candidate_definitions, bin_audit, best_diagnostics, top_predictions, summary


def build_readme(
    *,
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    candidate_definitions: pd.DataFrame,
    bin_audit: pd.DataFrame,
    best_diagnostics: pd.DataFrame,
) -> str:
    candidate_cols = [
        "candidate_id",
        "candidate_class",
        "mae",
        "rmse",
        "delta_mae_vs_0099",
        "hardened_vs_0099_gate_passed",
        "feature",
        "family",
        "bucket_count",
        "gate_scope",
        "min_history",
        "correction_cap_c",
        "specialist_active_rows",
    ]
    return f"""# 0101 Stable MAM Cell Feature Specialists

Generated: `{summary['generated_at_utc']}`

## Purpose

`0100` ranked long-history features inside the stable MAM agreement cell. `0101` tests the future-allowed part of that atlas: lagged target-memory features and pre-cutoff ISD station-network features. Upper-air, HKO daily climate, and daily marine features are deliberately excluded because their timestamp/publication proof is not attached yet.

Each candidate uses fixed feature buckets learned from pre-2000 history only. During the 2000-2023 evaluation period, each bucket stores only prior residuals from earlier target dates. The current target date updates the residual state only after it has been scored.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Input 0099 candidate | `{summary['input_0099_candidate']}` |
| Input 0099 MAE | `{summary['input_0099_mae']}` |
| Input 0099 RMSE | `{summary['input_0099_rmse']}` |
| Selected features | `{summary['selected_feature_count']}` |
| Candidates tested | `{summary['candidate_count']}` |
| Hardened vs 0099 count | `{summary['hardened_vs_0099_count']}` |
| Best raw 0101 candidate | `{summary['best_raw_0101_candidate']}` |
| Best raw 0101 MAE | `{summary['best_raw_0101_mae']}` |
| Best raw 0101 RMSE | `{summary['best_raw_0101_rmse']}` |
| Best raw 0101 delta vs 0099 | `{summary['best_raw_0101_delta_mae_vs_0099']}` |
| Overall selected best | `{summary['best_candidate']}` |
| Overall selected best class | `{summary['best_candidate_class']}` |
| Overall selected MAE | `{summary['best_mae']}` |
| Overall selected RMSE | `{summary['best_rmse']}` |
| New 0101 champion | `{summary['new_0101_champion']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Interpretation

If `new_0101_champion` is false, the result is still useful: it means the high-scoring 0100 future-allowed features explain residual structure but do not yet convert into a hardened improvement over the tiny 0099 policy under this conservative fold-local rule.

## Scoreboard

{markdown_table(scoreboard[candidate_cols].head(40), max_rows=40)}

## Candidate Definition Sample

{markdown_table(candidate_definitions.head(40), max_rows=40)}

## Pre-2000 Bin Audit

{markdown_table(bin_audit.head(60), max_rows=60)}

## Best Raw 0101 Diagnostics Sample

{markdown_table(best_diagnostics[best_diagnostics['gate_active_row'].astype(bool)].head(60), max_rows=60) if not best_diagnostics.empty else '_No diagnostics._'}

## Leakage Controls

All target rows are before `{summary['confirmation_start']}`. Feature bucket edges are computed only from feature rows before `2000-01-01`. Residual states are updated after each target date is scored. Only 0100 `allowed_for_future_walkforward=True` features from `target_memory` and `isd_station_network` are eligible. Diagnostic-only upper-air, HKO daily climate, and daily marine features are excluded.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(summary: dict[str, object], scoreboard: pd.DataFrame) -> None:
    display_cols = [
        "candidate_id",
        "candidate_class",
        "mae",
        "rmse",
        "delta_mae_vs_0099",
        "hardened_vs_0099_gate_passed",
        "feature",
        "family",
        "gate_scope",
        "min_history",
    ]
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0101_stable_mam_cell_feature_specialists.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Input 0099 MAE/RMSE | `{summary['input_0099_mae']}` / `{summary['input_0099_rmse']}` | Baseline |
| Selected future-allowed features | `{summary['selected_feature_count']}` | Target-memory + station only |
| Candidates tested | `{summary['candidate_count']}` | Pre-2000 fixed buckets |
| Hardened vs 0099 | `{summary['hardened_vs_0099_count']}` | Promotion gate |
| Best raw 0101 MAE | `{summary['best_raw_0101_mae']}` | Screen result |
| Best raw delta vs 0099 | `{summary['best_raw_0101_delta_mae_vs_0099']}` | Screen result |
| Overall selected best | `{summary['best_candidate']}` | `{summary['best_candidate_class']}` |
| New 0101 champion | `{summary['new_0101_champion']}` | Gate result |
| Leakage | `0` 2024+ rows | PASS |

Top rows:

{markdown_table(scoreboard[display_cols].head(15), max_rows=15)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0101 Stable MAM Cell Feature Specialists",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=str(summary["next_recommended_task"]),
    )


def run() -> dict[str, object]:
    scoreboard, candidate_definitions, bin_audit, best_diagnostics, top_predictions, summary = build_outputs()
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "candidate_definitions.csv", candidate_definitions)
    write_csv(artifacts / "bin_audit.csv", bin_audit)
    write_csv(artifacts / "best_candidate_diagnostics.csv", best_diagnostics)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "stable_mam_cell_feature_specialists_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            scoreboard=scoreboard,
            candidate_definitions=candidate_definitions,
            bin_audit=bin_audit,
            best_diagnostics=best_diagnostics,
        ),
    )
    update_milestones(summary, scoreboard)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-summary", action="store_true", help="Print JSON summary after writing artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run()
    if args.print_summary:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
