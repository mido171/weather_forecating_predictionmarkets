from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_0085_long_history_feature_station_residual_bridge import FEATURE_MATRIX_PATH
from scripts.run_hkg_t24_0086_guarded_long_history_residual_specialists import (
    ResidualState,
    assign_bucket,
    pre2000_thresholds,
)
from scripts.run_hkg_t24_0090_guarded_specialists_from_error_autopsy import (
    evaluation_masks,
    score_candidate,
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
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (
    classify_feature_family,
    station_ids_in_feature,
    update_markdown_section,
)

FOLDER_NAME = "0094_expanded_high_error_interaction_lab"
BASE_ID = "0093_guarded_champion_base"
INPUT_0093_TOP_PATH = RESEARCH_ROOT / "0093_guarded_champion_sensitivity_check" / "artifacts" / "top_predictions.csv"
INPUT_0093_SUMMARY_PATH = RESEARCH_ROOT / "0093_guarded_champion_sensitivity_check" / "artifacts" / "summary.json"
INPUT_0089_CONTRASTS_PATH = (
    RESEARCH_ROOT / "0089_remaining_error_regime_autopsy" / "artifacts" / "high_low_feature_contrasts.csv"
)
MAX_FEATURES_PER_GROUP = 4
MAX_PAIRS = 16
ACTIVE_GATES = ("mam_new_frame", "mam_press_archive", "mam_all")
MIN_HISTORY_VALUES = (40, 80)
SHRINK_ROWS = 80.0
CORRECTION_CAP_C = 0.35
DESIRED_GROUPS = ("target_memory", "isd_station_network", "upper_air_ceiling", "marine_proxy")


@dataclass(frozen=True)
class InteractionSpec:
    candidate_id: str
    pair_name: str
    feature_a: str
    feature_b: str
    group_a: str
    group_b: str
    active_gate: str
    min_history: int
    shrink_rows: float
    correction_cap_c: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def safe_token(value: str, *, max_len: int = 34) -> str:
    return (
        value.replace("_", "-")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
        .replace(".", "p")[:max_len]
    )


def pair_hash(feature_a: str, feature_b: str) -> str:
    return hashlib.sha1(f"{feature_a}||{feature_b}".encode()).hexdigest()[:8]


def interaction_group(feature: str, base_family: str | None = None) -> str:
    name = feature.lower()
    family = base_family or classify_feature_family(feature)
    if any(token in name for token in ("sea_temperature", "north_point", "waglan", "marine")):
        return "marine_proxy"
    if family == "upper_air" and any(token in name for token in ("ceiling", "inversion", "925", "850", "lower")):
        return "upper_air_ceiling"
    if family in {"target_memory", "isd_station_network"}:
        return family
    return family


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    missing = [
        path
        for path in (FEATURE_MATRIX_PATH, INPUT_0093_TOP_PATH, INPUT_0093_SUMMARY_PATH, INPUT_0089_CONTRASTS_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"0094 requires 0089 and 0093 artifacts first: {missing}")
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    features = features.loc[:, ~features.columns.duplicated()].copy()
    features["target_date"] = pd.to_datetime(features["target_date"], errors="coerce").dt.normalize()
    features = features[features["target_date"].notna() & (features["target_date"] < CONFIRMATION_START)].copy()

    base = pd.read_csv(INPUT_0093_TOP_PATH)
    base["target_date"] = pd.to_datetime(base["target_date"], errors="coerce").dt.normalize()
    base = base[base["target_date"].notna() & (base["target_date"] < CONFIRMATION_START)].copy()
    for column in ("target_tmax_c", "forecast_max_c", "candidate_prediction_c"):
        base[column] = pd.to_numeric(base[column], errors="coerce")
    base = base[base[["target_tmax_c", "forecast_max_c", "candidate_prediction_c"]].notna().all(axis=1)].copy()
    for column in ("forecast_source_family", "season", "frame_segment", "era_bucket"):
        base[column] = base[column].astype(str)

    contrasts = pd.read_csv(INPUT_0089_CONTRASTS_PATH)
    summary_0093 = json.loads(INPUT_0093_SUMMARY_PATH.read_text(encoding="utf-8"))
    require_no_confirmation_dates(features["target_date"], context="0094 feature matrix")
    require_no_confirmation_dates(base["target_date"], context="0094 0093 base predictions")
    return features, base, contrasts, summary_0093


def select_features(contrasts: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    available = contrasts[contrasts["feature"].isin(features.columns)].copy()
    available = available[~available["feature"].isin({"target_tmax_c"})].copy()
    available["interaction_group"] = [
        interaction_group(str(row.feature), str(row.family)) for row in available.itertuples(index=False)
    ]
    selected_rows: list[pd.Series] = []
    used: set[str] = set()
    for group_name in DESIRED_GROUPS:
        group = available[available["interaction_group"].eq(group_name)].sort_values(
            "contrast_priority",
            ascending=False,
        )
        for _, row in group.head(MAX_FEATURES_PER_GROUP).iterrows():
            feature = str(row["feature"])
            if feature in used:
                continue
            selected_rows.append(row)
            used.add(feature)
    if not selected_rows:
        return pd.DataFrame(columns=[*available.columns])
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def build_working_frame(
    features: pd.DataFrame,
    base: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_names = selected["feature"].astype(str).tolist()
    joined = base.merge(features[["target_date", *feature_names]], on="target_date", how="left")
    joined["base_residual_c"] = joined["candidate_prediction_c"] - joined["target_tmax_c"]

    threshold_rows: list[dict[str, object]] = []
    selected_lookup = selected.set_index("feature").to_dict("index")
    usable_features: list[str] = []
    for feature in feature_names:
        thresholds = pre2000_thresholds(features, feature)
        if thresholds is None:
            joined[f"{feature}__bucket"] = np.nan
            continue
        joined[f"{feature}__bucket"] = assign_bucket(joined[feature], thresholds)
        usable_features.append(feature)
        selected_row = selected_lookup.get(feature, {})
        threshold_rows.append(
            {
                "feature": feature,
                "family": selected_row.get("family", classify_feature_family(feature)),
                "interaction_group": selected_row.get("interaction_group", interaction_group(feature)),
                "station_ids": station_ids_in_feature(feature),
                "contrast_priority": selected_row.get("contrast_priority", math.nan),
                "thresholds": ";".join(f"{value:.6g}" for value in thresholds),
                "threshold_count": int(len(thresholds)),
            }
        )

    pair_rows: list[dict[str, object]] = []
    for feature_a, feature_b in combinations(usable_features, 2):
        row_a = selected_lookup.get(feature_a, {})
        row_b = selected_lookup.get(feature_b, {})
        group_a = str(row_a.get("interaction_group", interaction_group(feature_a)))
        group_b = str(row_b.get("interaction_group", interaction_group(feature_b)))
        if group_a == group_b:
            continue
        pair_name = f"{feature_a}__x__{feature_b}"
        a_bucket = joined[f"{feature_a}__bucket"]
        b_bucket = joined[f"{feature_b}__bucket"]
        pair_bucket = pd.Series(np.nan, index=joined.index, dtype="float64")
        valid = a_bucket.notna() & b_bucket.notna()
        pair_bucket.loc[valid] = (a_bucket.loc[valid].astype(int) * 10) + b_bucket.loc[valid].astype(int)
        joined[f"{pair_name}__bucket"] = pair_bucket
        priority_a = float(row_a.get("contrast_priority", 0.0) or 0.0)
        priority_b = float(row_b.get("contrast_priority", 0.0) or 0.0)
        pair_rows.append(
            {
                "pair_name": pair_name,
                "feature_a": feature_a,
                "feature_b": feature_b,
                "group_a": group_a,
                "group_b": group_b,
                "station_ids_a": station_ids_in_feature(feature_a),
                "station_ids_b": station_ids_in_feature(feature_b),
                "pair_priority": priority_a + priority_b,
                "valid_rows": int(pair_bucket.notna().sum()),
                "mam_new_frame_valid_rows": int((pair_bucket.notna() & active_mask_for_gate(joined, "mam_new_frame")).sum()),
            }
        )
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs = pairs.sort_values(["pair_priority", "mam_new_frame_valid_rows"], ascending=False).head(MAX_PAIRS)
    return (
        joined.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True),
        pd.DataFrame(threshold_rows),
        pairs.reset_index(drop=True),
    )


def active_mask_for_gate(frame: pd.DataFrame, gate: str) -> np.ndarray:
    season = frame["season"].astype(str)
    source = frame["forecast_source_family"].astype(str)
    segment = frame["frame_segment"].astype(str)
    mam = season.eq("MAM")
    if gate == "mam_new_frame":
        return (mam & segment.eq("newly_available_official_frame")).to_numpy(dtype=bool)
    if gate == "mam_press_archive":
        return (mam & source.eq("press_archive")).to_numpy(dtype=bool)
    if gate == "mam_all":
        return mam.to_numpy(dtype=bool)
    raise ValueError(f"Unsupported 0094 active gate: {gate}")


def evaluation_masks_0094(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    masks = evaluation_masks(frame)
    for gate in ACTIVE_GATES:
        masks[f"{gate}_"] = active_mask_for_gate(frame, gate)
    return masks


def make_specs(pairs: pd.DataFrame) -> list[InteractionSpec]:
    specs: list[InteractionSpec] = []
    for row in pairs.itertuples(index=False):
        for gate in ACTIVE_GATES:
            for min_history in MIN_HISTORY_VALUES:
                feature_a = str(row.feature_a)
                feature_b = str(row.feature_b)
                token = f"{safe_token(feature_a)}__x__{safe_token(feature_b)}"
                specs.append(
                    InteractionSpec(
                        candidate_id=(
                            f"mamint_{pair_hash(feature_a, feature_b)}_{token}_{gate}_m{min_history}"
                        ),
                        pair_name=str(row.pair_name),
                        feature_a=feature_a,
                        feature_b=feature_b,
                        group_a=str(row.group_a),
                        group_b=str(row.group_b),
                        active_gate=gate,
                        min_history=min_history,
                        shrink_rows=SHRINK_ROWS,
                        correction_cap_c=CORRECTION_CAP_C,
                    )
                )
    return specs


def apply_targeted_interaction(
    frame: pd.DataFrame,
    spec: InteractionSpec,
    *,
    include_diagnostics: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    base = frame["candidate_prediction_c"].to_numpy(dtype=float)
    residual = frame["base_residual_c"].to_numpy(dtype=float)
    pair_bucket = frame[f"{spec.pair_name}__bucket"].to_numpy(dtype=float)
    active_gate = active_mask_for_gate(frame, spec.active_gate)
    predictions = base.copy()
    active = np.zeros(len(frame), dtype=bool)
    prior_rows = np.zeros(len(frame), dtype=int)
    corrections = np.zeros(len(frame), dtype=float)
    states: dict[tuple[object, ...], ResidualState] = {}

    for _, date_group in frame.groupby("target_date", sort=True, observed=True):
        pending_updates: list[tuple[tuple[object, ...], float]] = []
        for idx in date_group.index:
            row_idx = int(idx)
            if not active_gate[row_idx] or not math.isfinite(pair_bucket[row_idx]):
                continue
            key = (spec.pair_name, int(pair_bucket[row_idx]))
            state = states.setdefault(key, ResidualState())
            prior_rows[row_idx] = state.count
            if state.count >= spec.min_history:
                shrink = state.count / (state.count + spec.shrink_rows)
                correction = float(np.clip(state.mean() * shrink, -spec.correction_cap_c, spec.correction_cap_c))
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
    diagnostics["feature_a"] = spec.feature_a
    diagnostics["feature_b"] = spec.feature_b
    diagnostics["group_a"] = spec.group_a
    diagnostics["group_b"] = spec.group_b
    diagnostics["active_gate"] = spec.active_gate
    diagnostics["pair_bucket"] = pair_bucket
    diagnostics["gate_active_row"] = active_gate
    diagnostics["prior_rows"] = prior_rows
    diagnostics["specialist_active"] = active
    diagnostics["specialist_correction_c"] = corrections
    return predictions, diagnostics


def spec_definition(spec: InteractionSpec) -> dict[str, object]:
    return {
        "candidate_id": spec.candidate_id,
        "pair_name": spec.pair_name,
        "feature_a": spec.feature_a,
        "feature_b": spec.feature_b,
        "group_a": spec.group_a,
        "group_b": spec.group_b,
        "active_gate": spec.active_gate,
        "min_history": spec.min_history,
        "shrink_rows": spec.shrink_rows,
        "correction_cap_c": spec.correction_cap_c,
    }


def score_specs(
    *,
    frame: pd.DataFrame,
    specs: list[InteractionSpec],
    mask_map: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, InteractionSpec]]:
    rows: list[dict[str, object]] = []
    specs_by_id: dict[str, InteractionSpec] = {}
    for spec in specs:
        prediction, _diagnostics = apply_targeted_interaction(frame, spec, include_diagnostics=False)
        scored = score_candidate(
            frame,
            candidate_id=spec.candidate_id,
            candidate_class="0094_expanded_high_error_interaction",
            prediction=prediction,
            mask_map=mask_map,
            extra=spec_definition(spec),
        )
        scored["delta_mae_vs_0093_base"] = float(scored["delta_mae_vs_0088_base"])
        rows.append(scored)
        specs_by_id[spec.candidate_id] = spec
    return rows, specs_by_id


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    features, base, contrasts, summary_0093 = load_inputs()
    selected = select_features(contrasts, features)
    frame, thresholds, pairs = build_working_frame(features, base, selected)
    if pairs.empty:
        raise RuntimeError("0094 found no usable cross-family interaction pairs")
    mask_map = evaluation_masks_0094(frame)
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
            candidate_class="0093_base",
            prediction=base_prediction,
            mask_map=mask_map,
        ),
    ]
    rows[0]["delta_mae_vs_0093_base"] = float(rows[0]["delta_mae_vs_0088_base"])
    rows[1]["delta_mae_vs_0093_base"] = 0.0
    specs = make_specs(pairs)
    scored_rows, specs_by_id = score_specs(frame=frame, specs=specs, mask_map=mask_map)
    rows.extend(scored_rows)

    scoreboard = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    sensitivity = scoreboard[scoreboard["candidate_class"].eq("0094_expanded_high_error_interaction")].copy()
    hardened = sensitivity[
        sensitivity["hardened_gate_passed"].astype(bool) & (sensitivity["delta_mae_vs_0093_base"].astype(float) < 0.0)
    ].copy()
    if hardened.empty:
        best_id = BASE_ID
        best_prediction = base_prediction
        diagnostics = pd.DataFrame()
    else:
        best_row = hardened.sort_values(["mae", "rmse"]).iloc[0]
        best_id = str(best_row["candidate_id"])
        best_spec = specs_by_id[best_id]
        best_prediction, diagnostics = apply_targeted_interaction(frame, best_spec, include_diagnostics=True)

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
        "selected_feature_count": int(len(selected)),
        "usable_feature_count": int(len(thresholds)),
        "pair_count": int(len(pairs)),
        "candidate_count": int(len(scoreboard)),
        "interaction_candidate_count": int(len(sensitivity)),
        "hardened_interaction_candidate_count": int(len(hardened)),
        "input_0093_best_candidate": summary_0093["best_candidate"],
        "input_0093_best_mae": float(summary_0093["best_mae"]),
        "input_0093_best_rmse": float(summary_0093["best_rmse"]),
        "best_candidate": best_id,
        "best_mae": float(best_score["mae"]),
        "best_rmse": float(best_score["rmse"]),
        "best_delta_mae_vs_0093_base": float(best_score["delta_mae_vs_0093_base"]),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "active_gates": list(ACTIVE_GATES),
        "min_history_values": list(MIN_HISTORY_VALUES),
        "status": "expanded_high_error_interaction_lab_complete",
        "next_recommended_task": (
            "Run 0095_mam_error_direction_split_lab: split the persistent MAM error regime into underforecast "
            "and overforecast sub-regimes, then test asymmetric guarded residual corrections using the strongest "
            "0094 pair definitions and the current 0093/0094 champion baseline."
        ),
    }
    require_no_confirmation_dates(top_predictions["target_date"], context="0094 top predictions")
    return scoreboard, selected, thresholds, pairs, diagnostics, top_predictions, summary


def summarize_scoreboard(scoreboard: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = scoreboard[scoreboard["candidate_class"].eq("0094_expanded_high_error_interaction")].copy()
    if interactions.empty:
        return pd.DataFrame(), pd.DataFrame()
    by_gate = (
        interactions.groupby("active_gate", observed=True)
        .agg(
            candidate_count=("candidate_id", "count"),
            hardened_count=("hardened_gate_passed", lambda values: int(pd.Series(values).astype(bool).sum())),
            best_mae=("mae", "min"),
            median_mae=("mae", "median"),
            best_delta_vs_0093=("delta_mae_vs_0093_base", "min"),
        )
        .reset_index()
        .sort_values(["best_delta_vs_0093", "best_mae"])
    )
    by_group_pair = (
        interactions.groupby(["group_a", "group_b"], observed=True)
        .agg(
            candidate_count=("candidate_id", "count"),
            hardened_count=("hardened_gate_passed", lambda values: int(pd.Series(values).astype(bool).sum())),
            best_mae=("mae", "min"),
            best_delta_vs_0093=("delta_mae_vs_0093_base", "min"),
        )
        .reset_index()
        .sort_values(["best_delta_vs_0093", "best_mae"])
    )
    return by_gate, by_group_pair


def build_readme(
    *,
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    selected: pd.DataFrame,
    thresholds: pd.DataFrame,
    pairs: pd.DataFrame,
    gate_summary: pd.DataFrame,
    group_pair_summary: pd.DataFrame,
) -> str:
    return f"""# 0094 Expanded High-Error Interaction Lab

Generated: `{summary['generated_at_utc']}`

## Purpose

`0089` showed that the worst remaining regime is MAM inside the newly available official frame. `0094` attacks that exact failure mode without using 2024+ confirmation rows and without replacing the current 0093 champion unless an interaction candidate improves the full pre-2024 score and passes the hardened no-regression gate.

The lab pairs high-priority long-history features across four signal groups:

- `target_memory`: lagged target heat-state and persistence/reversal features.
- `isd_station_network`: regional station and intraday station-network features.
- `upper_air_ceiling`: upper-air ceiling, inversion, and lower-troposphere features.
- `marine_proxy`: sea-temperature and Waglan/North Point marine-adjacent features from HKO daily climate archives.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Input 0093 best | `{summary['input_0093_best_candidate']}` |
| Input 0093 MAE | `{summary['input_0093_best_mae']}` |
| Interaction candidates | `{summary['interaction_candidate_count']}` |
| Hardened interaction candidates | `{summary['hardened_interaction_candidate_count']}` |
| Best candidate | `{summary['best_candidate']}` |
| Best MAE | `{summary['best_mae']}` |
| Best RMSE | `{summary['best_rmse']}` |
| Delta vs 0093 base | `{summary['best_delta_mae_vs_0093_base']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Plain-English Finding

This run asks whether the stubborn spring/new-frame errors need a two-signal explanation rather than another one-feature correction. Every candidate is active only inside a MAM gate: either `mam_new_frame`, `mam_press_archive`, or all MAM rows. Outside the active gate, the prediction remains the 0093 champion prediction. Inside the gate, the candidate may apply a small residual correction after enough prior rows exist for the same pair bucket.

Promotion is intentionally strict. A candidate must improve the total MAE versus 0093 and avoid worsening every tracked source, frame, season, and MAM-gate slice. This means the experiment can return a negative result even if some local spring cells look attractive.

## Selected Features

{markdown_table(selected, max_rows=80)}

## Usable Thresholded Features

{markdown_table(thresholds, max_rows=80)}

## Pair Definitions

{markdown_table(pairs, max_rows=80)}

## Scoreboard

{markdown_table(scoreboard.head(30), max_rows=30)}

## Gate Summary

{markdown_table(gate_summary, max_rows=20)}

## Group-Pair Summary

{markdown_table(group_pair_summary, max_rows=30)}

## Leakage Controls

All rows are before `{summary['confirmation_start']}`. Buckets are created from pre-2000 thresholds. Residual states are updated only after each target date has been scored, so the current target residual cannot influence its own prediction. The active-gate design also prevents broad opportunistic corrections outside the high-error spring regime being studied.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    gate_summary: pd.DataFrame,
    group_pair_summary: pd.DataFrame,
) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0094_expanded_high_error_interaction_lab.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| MAM high-error interaction candidates | `{summary['interaction_candidate_count']}` over `{summary['rows']}` rows | Pre-2024 only |
| Hardened interaction candidates | `{summary['hardened_interaction_candidate_count']}` | Strict gate |
| Input 0093 best MAE | `{summary['input_0093_best_mae']}` | Baseline |
| Best 0094 candidate | `{summary['best_candidate']}` | `{summary['best_mae']}` MAE |
| Delta vs 0093 base | `{summary['best_delta_mae_vs_0093_base']}` | Promotion value |
| Leakage | `0` 2024+ rows | PASS |

Top 0094 candidates:

{markdown_table(scoreboard.head(8), max_rows=8)}

Gate summary:

{markdown_table(gate_summary, max_rows=12)}

Group-pair summary:

{markdown_table(group_pair_summary.head(12), max_rows=12)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0094 Expanded High-Error Interaction Lab",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=str(summary["next_recommended_task"]),
    )


def run() -> dict[str, object]:
    scoreboard, selected, thresholds, pairs, diagnostics, top_predictions, summary = build_outputs()
    gate_summary, group_pair_summary = summarize_scoreboard(scoreboard)
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "selected_features.csv", selected)
    write_csv(artifacts / "feature_thresholds.csv", thresholds)
    write_csv(artifacts / "pair_definitions.csv", pairs)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "gate_summary.csv", gate_summary)
    write_csv(artifacts / "group_pair_summary.csv", group_pair_summary)
    write_csv(artifacts / "best_gate_diagnostics.csv", diagnostics)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "expanded_high_error_interaction_lab_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            scoreboard=scoreboard,
            selected=selected,
            thresholds=thresholds,
            pairs=pairs,
            gate_summary=gate_summary,
            group_pair_summary=group_pair_summary,
        ),
    )
    update_milestones(summary, scoreboard, gate_summary, group_pair_summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Run targeted MAM high-error interaction specialists against the 0093 HKG Tmax baseline."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
