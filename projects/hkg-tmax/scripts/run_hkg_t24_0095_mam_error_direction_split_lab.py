from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_0085_long_history_feature_station_residual_bridge import FEATURE_MATRIX_PATH
from scripts.run_hkg_t24_0086_guarded_long_history_residual_specialists import (
    ResidualState,
    assign_bucket,
    pre2000_thresholds,
)
from scripts.run_hkg_t24_0090_guarded_specialists_from_error_autopsy import evaluation_masks, score_candidate
from scripts.run_hkg_t24_0094_expanded_high_error_interaction_lab import active_mask_for_gate
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

FOLDER_NAME = "0095_mam_error_direction_split_lab"
BASE_ID = "0094_expanded_high_error_interaction_base"
INPUT_0094_TOP_PATH = RESEARCH_ROOT / "0094_expanded_high_error_interaction_lab" / "artifacts" / "top_predictions.csv"
INPUT_0094_SUMMARY_PATH = RESEARCH_ROOT / "0094_expanded_high_error_interaction_lab" / "artifacts" / "summary.json"
INPUT_0094_PAIRS_PATH = RESEARCH_ROOT / "0094_expanded_high_error_interaction_lab" / "artifacts" / "pair_definitions.csv"
INPUT_0094_SCOREBOARD_PATH = RESEARCH_ROOT / "0094_expanded_high_error_interaction_lab" / "artifacts" / "scoreboard.csv"

TOP_PAIR_COUNT = 6
ACTIVE_GATES = ("mam_all", "mam_new_frame", "mam_press_archive")
MIN_HISTORY_VALUES = (40, 80)
DIRECTION_THRESHOLDS_C = (0.0, 0.05, 0.10)
CORRECTION_CAPS_C = (0.25, 0.35)
DIRECTION_MODES = ("bidirectional", "overforecast_only", "underforecast_only")
SHRINK_ROWS = 80.0
DEFAULT_CHUNK_SIZE = 54


@dataclass(frozen=True)
class DirectionSplitSpec:
    candidate_id: str
    pair_name: str
    feature_a: str
    feature_b: str
    group_a: str
    group_b: str
    active_gate: str
    direction_mode: str
    min_history: int
    direction_threshold_c: float
    shrink_rows: float
    correction_cap_c: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def short_hash(*values: str) -> str:
    return hashlib.sha1("||".join(values).encode()).hexdigest()[:8]


def safe_token(value: str, *, max_len: int = 34) -> str:
    return (
        value.replace("_", "-")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
        .replace(".", "p")[:max_len]
    )


def prior_direction(mean_residual_c: float, threshold_c: float) -> str:
    if mean_residual_c >= threshold_c:
        return "overforecast"
    if mean_residual_c <= -threshold_c:
        return "underforecast"
    return "neutral"


def mode_allows_direction(mode: str, direction: str) -> bool:
    if mode == "bidirectional":
        return direction in {"overforecast", "underforecast"}
    if mode == "overforecast_only":
        return direction == "overforecast"
    if mode == "underforecast_only":
        return direction == "underforecast"
    raise ValueError(f"Unsupported 0095 direction mode: {mode}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    missing = [
        path
        for path in (
            FEATURE_MATRIX_PATH,
            INPUT_0094_TOP_PATH,
            INPUT_0094_SUMMARY_PATH,
            INPUT_0094_PAIRS_PATH,
            INPUT_0094_SCOREBOARD_PATH,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"0095 requires 0094 artifacts first: {missing}")

    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    features = features.loc[:, ~features.columns.duplicated()].copy()
    features["target_date"] = pd.to_datetime(features["target_date"], errors="coerce").dt.normalize()
    features = features[features["target_date"].notna() & (features["target_date"] < CONFIRMATION_START)].copy()

    base = pd.read_csv(INPUT_0094_TOP_PATH)
    base["target_date"] = pd.to_datetime(base["target_date"], errors="coerce").dt.normalize()
    base = base[base["target_date"].notna() & (base["target_date"] < CONFIRMATION_START)].copy()
    for column in ("target_tmax_c", "forecast_max_c", "candidate_prediction_c"):
        base[column] = pd.to_numeric(base[column], errors="coerce")
    base = base[base[["target_tmax_c", "forecast_max_c", "candidate_prediction_c"]].notna().all(axis=1)].copy()
    for column in ("forecast_source_family", "season", "frame_segment", "era_bucket"):
        base[column] = base[column].astype(str)

    pairs = pd.read_csv(INPUT_0094_PAIRS_PATH)
    scoreboard = pd.read_csv(INPUT_0094_SCOREBOARD_PATH)
    summary_0094 = json.loads(INPUT_0094_SUMMARY_PATH.read_text(encoding="utf-8"))
    require_no_confirmation_dates(features["target_date"], context="0095 feature matrix")
    require_no_confirmation_dates(base["target_date"], context="0095 0094 base predictions")
    return features, base, pairs, scoreboard, summary_0094


def load_light_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    missing = [
        path for path in (INPUT_0094_SUMMARY_PATH, INPUT_0094_PAIRS_PATH, INPUT_0094_SCOREBOARD_PATH) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"0095 requires 0094 lightweight artifacts first: {missing}")
    pairs = pd.read_csv(INPUT_0094_PAIRS_PATH)
    scoreboard = pd.read_csv(INPUT_0094_SCOREBOARD_PATH)
    summary_0094 = json.loads(INPUT_0094_SUMMARY_PATH.read_text(encoding="utf-8"))
    return pairs, scoreboard, summary_0094


def select_strong_pairs(pairs: pd.DataFrame, scoreboard: pd.DataFrame) -> pd.DataFrame:
    candidate_rows = scoreboard[
        scoreboard["candidate_class"].eq("0094_expanded_high_error_interaction")
        & scoreboard["hardened_gate_passed"].astype(bool)
        & (pd.to_numeric(scoreboard["delta_mae_vs_0093_base"], errors="coerce") < 0.0)
    ].copy()
    pair_order: list[str] = []
    for _, row in candidate_rows.sort_values(["mae", "rmse"]).iterrows():
        pair_name = str(row["pair_name"])
        if pair_name not in pair_order:
            pair_order.append(pair_name)
        if len(pair_order) >= TOP_PAIR_COUNT:
            break
    if len(pair_order) < TOP_PAIR_COUNT:
        for _, row in pairs.sort_values(["pair_priority", "mam_new_frame_valid_rows"], ascending=False).iterrows():
            pair_name = str(row["pair_name"])
            if pair_name not in pair_order:
                pair_order.append(pair_name)
            if len(pair_order) >= TOP_PAIR_COUNT:
                break
    order = {pair_name: idx for idx, pair_name in enumerate(pair_order)}
    selected = pairs[pairs["pair_name"].isin(pair_order)].copy()
    selected["selection_rank"] = selected["pair_name"].map(order).astype(int) + 1
    return selected.sort_values("selection_rank").reset_index(drop=True)


def build_working_frame(
    features: pd.DataFrame,
    base: pd.DataFrame,
    selected_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_names = sorted(
        set(selected_pairs["feature_a"].astype(str).tolist()) | set(selected_pairs["feature_b"].astype(str).tolist())
    )
    missing_features = [feature for feature in feature_names if feature not in features.columns]
    if missing_features:
        raise RuntimeError(f"0095 selected pair features missing from feature matrix: {missing_features}")
    joined = base.merge(features[["target_date", *feature_names]], on="target_date", how="left")
    joined["base_residual_c"] = joined["candidate_prediction_c"] - joined["target_tmax_c"]

    threshold_rows: list[dict[str, object]] = []
    for feature in feature_names:
        thresholds = pre2000_thresholds(features, feature)
        if thresholds is None:
            joined[f"{feature}__bucket"] = np.nan
            continue
        joined[f"{feature}__bucket"] = assign_bucket(joined[feature], thresholds)
        threshold_rows.append(
            {
                "feature": feature,
                "thresholds": ";".join(f"{value:.6g}" for value in thresholds),
                "threshold_count": int(len(thresholds)),
            }
        )

    for row in selected_pairs.itertuples(index=False):
        feature_a = str(row.feature_a)
        feature_b = str(row.feature_b)
        pair_name = str(row.pair_name)
        a_bucket = joined[f"{feature_a}__bucket"]
        b_bucket = joined[f"{feature_b}__bucket"]
        pair_bucket = pd.Series(np.nan, index=joined.index, dtype="float64")
        valid = a_bucket.notna() & b_bucket.notna()
        pair_bucket.loc[valid] = (a_bucket.loc[valid].astype(int) * 10) + b_bucket.loc[valid].astype(int)
        joined[f"{pair_name}__bucket"] = pair_bucket
    require_no_confirmation_dates(joined["target_date"], context="0095 working frame")
    return joined.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True), pd.DataFrame(threshold_rows)


def evaluation_masks_0095(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    masks = evaluation_masks(frame)
    for gate in ACTIVE_GATES:
        masks[f"{gate}_"] = active_mask_for_gate(frame, gate)
    return masks


def make_specs(selected_pairs: pd.DataFrame) -> list[DirectionSplitSpec]:
    specs: list[DirectionSplitSpec] = []
    for row in selected_pairs.itertuples(index=False):
        feature_a = str(row.feature_a)
        feature_b = str(row.feature_b)
        token = f"{safe_token(feature_a)}__x__{safe_token(feature_b)}"
        pair_name = str(row.pair_name)
        for active_gate in ACTIVE_GATES:
            for min_history in MIN_HISTORY_VALUES:
                for threshold in DIRECTION_THRESHOLDS_C:
                    for cap in CORRECTION_CAPS_C:
                        for mode in DIRECTION_MODES:
                            specs.append(
                                DirectionSplitSpec(
                                    candidate_id=(
                                        f"dirsplit_{short_hash(pair_name, active_gate, mode, str(min_history), str(threshold), str(cap))}_"
                                        f"{token}_{active_gate}_{mode}_m{min_history}_t{str(threshold).replace('.', 'p')}_"
                                        f"cap{str(cap).replace('.', 'p')}"
                                    ),
                                    pair_name=pair_name,
                                    feature_a=feature_a,
                                    feature_b=feature_b,
                                    group_a=str(row.group_a),
                                    group_b=str(row.group_b),
                                    active_gate=active_gate,
                                    direction_mode=mode,
                                    min_history=min_history,
                                    direction_threshold_c=threshold,
                                    shrink_rows=SHRINK_ROWS,
                                    correction_cap_c=cap,
                                )
                            )
    return specs


def apply_direction_split(
    frame: pd.DataFrame,
    spec: DirectionSplitSpec,
    *,
    include_diagnostics: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    base = frame["candidate_prediction_c"].to_numpy(dtype=float)
    residual = frame["base_residual_c"].to_numpy(dtype=float)
    pair_bucket = frame[f"{spec.pair_name}__bucket"].to_numpy(dtype=float)
    gate_active = active_mask_for_gate(frame, spec.active_gate)
    predictions = base.copy()
    active = np.zeros(len(frame), dtype=bool)
    prior_rows = np.zeros(len(frame), dtype=int)
    prior_mean = np.full(len(frame), np.nan, dtype=float)
    corrections = np.zeros(len(frame), dtype=float)
    direction_codes = np.full(len(frame), "inactive", dtype=object)
    states: dict[tuple[str, int], ResidualState] = {}

    for _, date_group in frame.groupby("target_date", sort=True, observed=True):
        pending_updates: list[tuple[tuple[str, int], float]] = []
        for idx in date_group.index:
            row_idx = int(idx)
            if not gate_active[row_idx] or not math.isfinite(pair_bucket[row_idx]):
                continue
            key = (spec.pair_name, int(pair_bucket[row_idx]))
            state = states.setdefault(key, ResidualState())
            prior_rows[row_idx] = state.count
            mean_residual = state.mean()
            prior_mean[row_idx] = mean_residual if state.count > 0 else math.nan
            direction = prior_direction(mean_residual, spec.direction_threshold_c) if state.count > 0 else "neutral"
            direction_codes[row_idx] = direction
            if state.count >= spec.min_history and mode_allows_direction(spec.direction_mode, direction):
                shrink = state.count / (state.count + spec.shrink_rows)
                correction = float(np.clip(mean_residual * shrink, -spec.correction_cap_c, spec.correction_cap_c))
                predictions[row_idx] = base[row_idx] - correction
                corrections[row_idx] = correction
                active[row_idx] = abs(correction) > 1e-12
            pending_updates.append((key, residual[row_idx]))
        for key, residual_value in pending_updates:
            states[key].update(residual_value)

    if not include_diagnostics:
        return predictions, pd.DataFrame()
    diagnostics = frame[["target_date", "forecast_source_family", "season", "frame_segment", "era_bucket"]].copy()
    diagnostics["candidate_id"] = spec.candidate_id
    diagnostics["pair_name"] = spec.pair_name
    diagnostics["active_gate"] = spec.active_gate
    diagnostics["direction_mode"] = spec.direction_mode
    diagnostics["pair_bucket"] = pair_bucket
    diagnostics["gate_active_row"] = gate_active
    diagnostics["prior_rows"] = prior_rows
    diagnostics["prior_mean_residual_c"] = prior_mean
    diagnostics["prior_direction"] = direction_codes
    diagnostics["specialist_active"] = active
    diagnostics["specialist_correction_c"] = corrections
    return predictions, diagnostics


def spec_definition(spec: DirectionSplitSpec) -> dict[str, object]:
    return {
        "candidate_id": spec.candidate_id,
        "pair_name": spec.pair_name,
        "feature_a": spec.feature_a,
        "feature_b": spec.feature_b,
        "group_a": spec.group_a,
        "group_b": spec.group_b,
        "active_gate": spec.active_gate,
        "direction_mode": spec.direction_mode,
        "min_history": spec.min_history,
        "direction_threshold_c": spec.direction_threshold_c,
        "shrink_rows": spec.shrink_rows,
        "correction_cap_c": spec.correction_cap_c,
    }


def score_specs(
    *,
    frame: pd.DataFrame,
    specs: list[DirectionSplitSpec],
    mask_map: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, DirectionSplitSpec]]:
    rows: list[dict[str, object]] = []
    specs_by_id: dict[str, DirectionSplitSpec] = {}
    for spec in specs:
        prediction, _diagnostics = apply_direction_split(frame, spec, include_diagnostics=False)
        scored = score_candidate(
            frame,
            candidate_id=spec.candidate_id,
            candidate_class="0095_mam_error_direction_split",
            prediction=prediction,
            mask_map=mask_map,
            extra=spec_definition(spec),
        )
        scored["delta_mae_vs_0094_base"] = float(scored["delta_mae_vs_0088_base"])
        rows.append(scored)
        specs_by_id[spec.candidate_id] = spec
    return rows, specs_by_id


def run_worker(start: int, stop: int, output: Path) -> dict[str, object]:
    if not output.is_absolute():
        output = REPO_ROOT / output
    output = output.resolve()
    features, base, pairs, previous_scoreboard, _summary_0094 = load_inputs()
    selected_pairs = select_strong_pairs(pairs, previous_scoreboard)
    frame, _thresholds = build_working_frame(features, base, selected_pairs)
    mask_map = evaluation_masks_0095(frame)
    specs = make_specs(selected_pairs)[start:stop]
    rows, _specs_by_id = score_specs(frame=frame, specs=specs, mask_map=mask_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return {
        "worker_start": start,
        "worker_stop": stop,
        "worker_rows": len(rows),
        "worker_output": str(output),
    }


def score_specs_in_chunks(specs: list[DirectionSplitSpec], *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[dict[str, object]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    worker_dir = PROJECT_PATHS.run_root / "tmp" / "0095_worker_chunks"
    rows: list[dict[str, object]] = []
    for start in range(0, len(specs), chunk_size):
        stop = min(start + chunk_size, len(specs))
        expected_rows = stop - start
        output = worker_dir / f"s_{start:04d}_{stop:04d}.csv"
        if output.exists():
            chunk = pd.read_csv(output)
            if len(chunk) == expected_rows:
                rows.extend(chunk.to_dict("records"))
                continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-start",
            str(start),
            "--worker-stop",
            str(stop),
            "--worker-output",
            str(output),
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
            text=True,
            timeout=240,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "0095 direction-split worker failed "
                f"for rows {start}:{stop} with exit code {completed.returncode}\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        chunk = pd.read_csv(output)
        if len(chunk) != expected_rows:
            raise RuntimeError(
                f"0095 direction-split worker wrote {len(chunk)} rows for {start}:{stop}; expected {expected_rows}"
            )
        rows.extend(chunk.to_dict("records"))
        gc.collect()
        time.sleep(1)
    return rows


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    pairs, previous_scoreboard, summary_0094 = load_light_inputs()
    selected_pairs = select_strong_pairs(pairs, previous_scoreboard)
    specs = make_specs(selected_pairs)
    scored_rows = score_specs_in_chunks(specs)
    gc.collect()

    features, base, _pairs, _previous_scoreboard, _summary_0094 = load_inputs()
    selected_pairs = select_strong_pairs(_pairs, _previous_scoreboard)
    frame, thresholds = build_working_frame(features, base, selected_pairs)
    mask_map = evaluation_masks_0095(frame)
    raw_prediction = frame["forecast_max_c"].to_numpy(dtype=float)
    base_prediction = frame["candidate_prediction_c"].to_numpy(dtype=float)
    rows = [
        score_candidate(
            frame,
            candidate_id="official_raw",
            candidate_class="official_raw",
            prediction=raw_prediction,
            mask_map=mask_map,
        ),
        score_candidate(
            frame,
            candidate_id=BASE_ID,
            candidate_class="0094_base",
            prediction=base_prediction,
            mask_map=mask_map,
        ),
    ]
    rows[0]["delta_mae_vs_0094_base"] = float(rows[0]["delta_mae_vs_0088_base"])
    rows[1]["delta_mae_vs_0094_base"] = 0.0

    specs_by_id = {spec.candidate_id: spec for spec in specs}
    rows.extend(scored_rows)
    scoreboard = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    candidates = scoreboard[scoreboard["candidate_class"].eq("0095_mam_error_direction_split")].copy()
    hardened = candidates[
        candidates["hardened_gate_passed"].astype(bool) & (candidates["delta_mae_vs_0094_base"].astype(float) < 0.0)
    ].copy()
    if hardened.empty:
        best_id = BASE_ID
        best_prediction = base_prediction
        diagnostics = pd.DataFrame()
    else:
        best_row = hardened.sort_values(["mae", "rmse"]).iloc[0]
        best_id = str(best_row["candidate_id"])
        best_spec = specs_by_id[best_id]
        best_prediction, diagnostics = apply_direction_split(frame, best_spec, include_diagnostics=True)

    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    best_score = scoreboard[scoreboard["candidate_id"].eq(best_id)].iloc[0]
    top_predictions = frame[
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
    top_predictions["candidate_id"] = best_id
    top_predictions["candidate_prediction_c"] = best_prediction
    top_predictions["candidate_error_c"] = top_predictions["candidate_prediction_c"] - top_predictions["target_tmax_c"]
    summary = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "rows": int(len(frame)),
        "first_target_date": dates.min().date().isoformat(),
        "last_target_date": dates.max().date().isoformat(),
        "selected_pair_count": int(len(selected_pairs)),
        "usable_feature_count": int(len(thresholds)),
        "candidate_count": int(len(scoreboard)),
        "direction_split_candidate_count": int(len(candidates)),
        "hardened_direction_split_candidate_count": int(len(hardened)),
        "input_0094_best_candidate": summary_0094["best_candidate"],
        "input_0094_best_mae": float(summary_0094["best_mae"]),
        "input_0094_best_rmse": float(summary_0094["best_rmse"]),
        "best_candidate": best_id,
        "best_mae": float(best_score["mae"]),
        "best_rmse": float(best_score["rmse"]),
        "best_delta_mae_vs_0094_base": float(best_score["delta_mae_vs_0094_base"]),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "active_gates": list(ACTIVE_GATES),
        "direction_modes": list(DIRECTION_MODES),
        "direction_thresholds_c": list(DIRECTION_THRESHOLDS_C),
        "correction_caps_c": list(CORRECTION_CAPS_C),
        "min_history_values": list(MIN_HISTORY_VALUES),
        "status": "mam_error_direction_split_lab_complete",
        "next_recommended_task": (
            "Run 0096_directional_cell_failure_audit: analyze where 0095 direction-split candidates helped or "
            "failed by prior-direction, pair bucket, source family, and MAM sub-month, then design the next "
            "bounded specialist from only the stable improving cells."
        ),
    }
    require_no_confirmation_dates(top_predictions["target_date"], context="0095 top predictions")
    return scoreboard, selected_pairs, thresholds, diagnostics, top_predictions, summary


def summarize_scoreboard(scoreboard: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = scoreboard[scoreboard["candidate_class"].eq("0095_mam_error_direction_split")].copy()
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    by_mode = (
        candidates.groupby(["direction_mode", "active_gate"], observed=True)
        .agg(
            candidate_count=("candidate_id", "count"),
            hardened_count=("hardened_gate_passed", lambda values: int(pd.Series(values).astype(bool).sum())),
            best_mae=("mae", "min"),
            median_mae=("mae", "median"),
            best_delta_vs_0094=("delta_mae_vs_0094_base", "min"),
        )
        .reset_index()
        .sort_values(["best_delta_vs_0094", "best_mae"])
    )
    by_pair = (
        candidates.groupby(["pair_name", "direction_mode"], observed=True)
        .agg(
            candidate_count=("candidate_id", "count"),
            hardened_count=("hardened_gate_passed", lambda values: int(pd.Series(values).astype(bool).sum())),
            best_mae=("mae", "min"),
            best_delta_vs_0094=("delta_mae_vs_0094_base", "min"),
        )
        .reset_index()
        .sort_values(["best_delta_vs_0094", "best_mae"])
    )
    return by_mode, by_pair


def build_readme(
    *,
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    thresholds: pd.DataFrame,
    mode_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> str:
    return f"""# 0095 MAM Error-Direction Split Lab

Generated: `{summary['generated_at_utc']}`

## Purpose

`0094` improved the persistent spring error regime with a MAM interaction between long target heat memory and regional morning-to-midday station warming. `0095` asks whether the next gain comes from splitting those spring cells by prior error direction.

The key leakage rule is strict: the current row's true error is never used to choose a direction. For each pair bucket, the candidate sees only residuals from earlier target dates inside the same active MAM gate. If the prior mean residual is positive, the baseline has historically overforecast that cell; if it is negative, it has historically underforecast that cell.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Input 0094 best | `{summary['input_0094_best_candidate']}` |
| Input 0094 MAE | `{summary['input_0094_best_mae']}` |
| Direction candidates | `{summary['direction_split_candidate_count']}` |
| Hardened direction candidates | `{summary['hardened_direction_split_candidate_count']}` |
| Best candidate | `{summary['best_candidate']}` |
| Best MAE | `{summary['best_mae']}` |
| Best RMSE | `{summary['best_rmse']}` |
| Delta vs 0094 base | `{summary['best_delta_mae_vs_0094_base']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Plain-English Finding

This experiment tests whether the model should correct only historically too-warm cells, only historically too-cold cells, or both. A positive prior residual means the forecast was too high in comparable prior rows, so subtracting the correction cools the prediction. A negative prior residual means the forecast was too low, so subtracting a negative correction warms the prediction.

Promotion remains strict. A candidate must beat the 0094 champion on full pre-2024 MAE and avoid tracked source, frame, season, and MAM-gate regressions. If no direction split is promoted, the 0094 champion remains the baseline.

## Selected 0094 Pairs

{markdown_table(selected_pairs, max_rows=40)}

## Feature Thresholds

{markdown_table(thresholds, max_rows=40)}

## Scoreboard

{markdown_table(scoreboard.head(30), max_rows=30)}

## Mode Summary

{markdown_table(mode_summary, max_rows=40)}

## Pair Summary

{markdown_table(pair_summary.head(40), max_rows=40)}

## Leakage Controls

All predictions are scored only on rows before `{summary['confirmation_start']}`. Feature buckets use pre-2000 thresholds. Direction states are updated after each target date is scored, never before. The active gates are MAM-only, so this experiment does not opportunistically adjust unrelated seasons.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    mode_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0095_mam_error_direction_split_lab.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| MAM prior-direction candidates | `{summary['direction_split_candidate_count']}` over `{summary['rows']}` rows | Pre-2024 only |
| Hardened direction candidates | `{summary['hardened_direction_split_candidate_count']}` | Strict gate |
| Input 0094 best MAE | `{summary['input_0094_best_mae']}` | Baseline |
| Best 0095 candidate | `{summary['best_candidate']}` | `{summary['best_mae']}` MAE |
| Delta vs 0094 base | `{summary['best_delta_mae_vs_0094_base']}` | Promotion value |
| Leakage | `0` 2024+ rows | PASS |

Top 0095 candidates:

{markdown_table(scoreboard.head(8), max_rows=8)}

Mode summary:

{markdown_table(mode_summary.head(12), max_rows=12)}

Pair summary:

{markdown_table(pair_summary.head(12), max_rows=12)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0095 MAM Error-Direction Split Lab",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=str(summary["next_recommended_task"]),
    )


def run() -> dict[str, object]:
    scoreboard, selected_pairs, thresholds, diagnostics, top_predictions, summary = build_outputs()
    mode_summary, pair_summary = summarize_scoreboard(scoreboard)
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "selected_pairs.csv", selected_pairs)
    write_csv(artifacts / "feature_thresholds.csv", thresholds)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "mode_summary.csv", mode_summary)
    write_csv(artifacts / "pair_summary.csv", pair_summary)
    write_csv(artifacts / "best_gate_diagnostics.csv", diagnostics)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "mam_error_direction_split_lab_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            scoreboard=scoreboard,
            selected_pairs=selected_pairs,
            thresholds=thresholds,
            mode_summary=mode_summary,
            pair_summary=pair_summary,
        ),
    )
    update_milestones(summary, scoreboard, mode_summary, pair_summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MAM prior-error-direction split specialists against the 0094 HKG Tmax baseline."
    )
    parser.add_argument("--worker-start", type=int, default=None)
    parser.add_argument("--worker-stop", type=int, default=None)
    parser.add_argument("--worker-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker_args = (args.worker_start, args.worker_stop, args.worker_output)
    if any(value is not None for value in worker_args):
        if args.worker_start is None or args.worker_stop is None or args.worker_output is None:
            raise SystemExit("--worker-start, --worker-stop, and --worker-output must be provided together")
        print(json.dumps(run_worker(args.worker_start, args.worker_stop, args.worker_output), indent=2))
        return
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
