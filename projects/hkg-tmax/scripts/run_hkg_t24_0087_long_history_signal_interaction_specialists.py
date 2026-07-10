from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
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

from scripts.run_hkg_t24_0083_expanded_frame_official_anchor_replay import date_text, score_arrays
from scripts.run_hkg_t24_0085_long_history_feature_station_residual_bridge import (
    FEATURE_MATRIX_PATH,
    classify_feature_family,
    station_ids_in_feature,
)
from scripts.run_hkg_t24_0086_guarded_long_history_residual_specialists import (
    ResidualState,
    assign_bucket,
    pre2000_thresholds,
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

FOLDER_NAME = "0087_long_history_signal_interaction_specialists"
INPUT_0086_TOP_PATH = (
    RESEARCH_ROOT / "0086_guarded_long_history_residual_specialists" / "artifacts" / "top_predictions.csv"
)
INPUT_0085_RANKINGS_PATH = (
    RESEARCH_ROOT / "0085_long_history_feature_station_residual_bridge" / "artifacts" / "feature_residual_rankings.csv"
)
BASE_ID = "0086_best_guarded_long_history_specialist"
MAX_FEATURES = 6
MAX_PAIRS = 15


@dataclass(frozen=True)
class InteractionSpec:
    candidate_id: str
    feature_a: str
    feature_b: str
    context_mode: str
    min_history: int
    shrink_rows: float
    correction_cap_c: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def safe_token(value: str) -> str:
    return (
        value.replace("_", "-")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
        .replace(".", "p")[:52]
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [path for path in (FEATURE_MATRIX_PATH, INPUT_0086_TOP_PATH, INPUT_0085_RANKINGS_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"0087 requires feature/0085/0086 artifacts first: {missing}")
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    features = features.loc[:, ~features.columns.duplicated()].copy()
    features["target_date"] = pd.to_datetime(features["target_date"], errors="coerce").dt.normalize()
    features = features[features["target_date"].notna() & (features["target_date"] < CONFIRMATION_START)].copy()
    base = pd.read_csv(INPUT_0086_TOP_PATH)
    base["target_date"] = pd.to_datetime(base["target_date"], errors="coerce").dt.normalize()
    base = base[base["target_date"].notna() & (base["target_date"] < CONFIRMATION_START)].copy()
    rankings = pd.read_csv(INPUT_0085_RANKINGS_PATH)
    require_no_confirmation_dates(features["target_date"], context="0087 feature matrix")
    require_no_confirmation_dates(base["target_date"], context="0087 0086 base predictions")
    return features, base, rankings


def select_interaction_features(rankings: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    available = rankings[rankings["feature"].isin(features.columns)].copy()
    priority = [
        "isd_station_network",
        "target_memory",
        "upper_air",
        "hko_daily_climate",
        "calendar_climatology",
    ]
    rows: list[pd.Series] = []
    used: set[str] = set()
    for family in priority:
        group = available[available["family"].eq(family)].sort_values("priority_score", ascending=False)
        for _, row in group.head(2).iterrows():
            feature = str(row["feature"])
            if feature not in used:
                rows.append(row)
                used.add(feature)
            if len(rows) >= MAX_FEATURES:
                break
        if len(rows) >= MAX_FEATURES:
            break
    if len(rows) < MAX_FEATURES:
        for _, row in available.sort_values("priority_score", ascending=False).iterrows():
            feature = str(row["feature"])
            if feature not in used:
                rows.append(row)
                used.add(feature)
            if len(rows) >= MAX_FEATURES:
                break
    return pd.DataFrame(rows).reset_index(drop=True)


def build_working_frame(
    features: pd.DataFrame,
    base: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_names = selected["feature"].astype(str).tolist()
    joined = base.merge(features[["target_date", *feature_names]], on="target_date", how="left")
    for col in ("target_tmax_c", "forecast_max_c", "candidate_prediction_c"):
        joined[col] = pd.to_numeric(joined[col], errors="coerce")
    joined = joined[joined[["target_tmax_c", "forecast_max_c", "candidate_prediction_c"]].notna().all(axis=1)].copy()
    joined["base_residual_c"] = joined["candidate_prediction_c"] - joined["target_tmax_c"]
    joined["forecast_source_family"] = joined["forecast_source_family"].astype(str)
    joined["season"] = joined["season"].astype(str)
    joined["frame_segment"] = joined["frame_segment"].astype(str)
    joined["era_bucket"] = joined["era_bucket"].astype(str)

    threshold_rows: list[dict[str, object]] = []
    usable_features: list[str] = []
    for feature in feature_names:
        thresholds = pre2000_thresholds(features, feature)
        if thresholds is None:
            joined[f"{feature}__bucket"] = np.nan
            continue
        usable_features.append(feature)
        joined[f"{feature}__bucket"] = assign_bucket(joined[feature], thresholds)
        threshold_rows.append(
            {
                "feature": feature,
                "family": classify_feature_family(feature),
                "station_ids": station_ids_in_feature(feature),
                "thresholds": ";".join(f"{value:.6g}" for value in thresholds),
                "threshold_count": int(len(thresholds)),
            }
        )

    pair_rows: list[dict[str, object]] = []
    for feature_a, feature_b in list(combinations(usable_features, 2))[:MAX_PAIRS]:
        pair_name = f"{feature_a}__x__{feature_b}"
        a_bucket = joined[f"{feature_a}__bucket"]
        b_bucket = joined[f"{feature_b}__bucket"]
        pair_bucket = pd.Series(np.nan, index=joined.index, dtype="float64")
        valid = a_bucket.notna() & b_bucket.notna()
        pair_bucket.loc[valid] = (a_bucket.loc[valid].astype(int) * 10) + b_bucket.loc[valid].astype(int)
        joined[f"{pair_name}__bucket"] = pair_bucket
        pair_rows.append(
            {
                "pair_name": pair_name,
                "feature_a": feature_a,
                "feature_b": feature_b,
                "family_a": classify_feature_family(feature_a),
                "family_b": classify_feature_family(feature_b),
                "station_ids_a": station_ids_in_feature(feature_a),
                "station_ids_b": station_ids_in_feature(feature_b),
                "valid_rows": int(pair_bucket.notna().sum()),
            }
        )
    return (
        joined.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True),
        pd.DataFrame(threshold_rows),
        pd.DataFrame(pair_rows),
    )


def context_key(row: pd.Series, spec: InteractionSpec) -> tuple[object, ...] | None:
    pair_name = f"{spec.feature_a}__x__{spec.feature_b}"
    bucket = row.get(f"{pair_name}__bucket")
    if pd.isna(bucket):
        return None
    base_key: tuple[object, ...] = (pair_name, int(bucket))
    if spec.context_mode == "interaction":
        return base_key
    if spec.context_mode == "source_interaction":
        return (*base_key, row["forecast_source_family"])
    if spec.context_mode == "source_frame_interaction":
        return (*base_key, row["forecast_source_family"], row["frame_segment"])
    raise ValueError(f"Unsupported 0087 interaction context: {spec.context_mode}")


def make_specs(pairs: pd.DataFrame) -> list[InteractionSpec]:
    specs: list[InteractionSpec] = []
    for row in pairs.itertuples(index=False):
        for context_mode in ("interaction", "source_interaction"):
            token = f"{safe_token(row.feature_a)}__x__{safe_token(row.feature_b)}"
            specs.append(
                InteractionSpec(
                    candidate_id=f"interaction_{token}_{context_mode}_m60",
                    feature_a=str(row.feature_a),
                    feature_b=str(row.feature_b),
                    context_mode=context_mode,
                    min_history=60,
                    shrink_rows=120.0,
                    correction_cap_c=0.55,
                )
            )
    return specs


def apply_interaction_specialist(frame: pd.DataFrame, spec: InteractionSpec) -> tuple[np.ndarray, pd.DataFrame]:
    base = frame["candidate_prediction_c"].to_numpy(dtype=float)
    residual = frame["base_residual_c"].to_numpy(dtype=float)
    predictions = base.copy()
    active = np.zeros(len(frame), dtype=bool)
    prior_rows = np.zeros(len(frame), dtype=int)
    corrections = np.zeros(len(frame), dtype=float)
    states: dict[tuple[object, ...], ResidualState] = {}

    for _, date_group in frame.groupby("target_date", sort=True, observed=True):
        pending_updates: list[tuple[tuple[object, ...], float]] = []
        for idx, row in date_group.iterrows():
            key = context_key(row, spec)
            if key is None:
                continue
            state = states.setdefault(key, ResidualState())
            row_idx = int(idx)
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
    diagnostics = frame[["target_date", "forecast_source_family", "season", "frame_segment", "era_bucket"]].copy()
    diagnostics["candidate_id"] = spec.candidate_id
    diagnostics["feature_a"] = spec.feature_a
    diagnostics["feature_b"] = spec.feature_b
    diagnostics["context_mode"] = spec.context_mode
    diagnostics["prior_rows"] = prior_rows
    diagnostics["specialist_active"] = active
    diagnostics["specialist_correction_c"] = corrections
    return predictions, diagnostics


def masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    source = frame["forecast_source_family"].astype(str)
    segment = frame["frame_segment"].astype(str)
    return {
        "old_frame_": segment.eq("current_0081_frame").to_numpy(dtype=bool),
        "newly_available_": segment.eq("newly_available_official_frame").to_numpy(dtype=bool),
        "press_": source.eq("press_archive").to_numpy(dtype=bool),
        "rss_": source.eq("rss_archive").to_numpy(dtype=bool),
    }


def score_candidate(
    frame: pd.DataFrame,
    *,
    candidate_id: str,
    candidate_class: str,
    prediction: np.ndarray,
    mask_map: dict[str, np.ndarray],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    target = frame["target_tmax_c"].to_numpy(dtype=float)
    base = frame["candidate_prediction_c"].to_numpy(dtype=float)
    raw = frame["forecast_max_c"].to_numpy(dtype=float)
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "candidate_class": candidate_class,
        **score_arrays(target=target, prediction=prediction, dates=dates),
    }
    base_score = score_arrays(target=target, prediction=base, dates=dates)
    raw_score = score_arrays(target=target, prediction=raw, dates=dates)
    row["delta_mae_vs_0086_base"] = float(row["mae"]) - float(base_score["mae"])
    row["delta_mae_vs_official_raw"] = float(row["mae"]) - float(raw_score["mae"])
    for prefix, mask in mask_map.items():
        score = score_arrays(target=target[mask], prediction=prediction[mask], dates=dates[mask], prefix=prefix)
        base_segment = score_arrays(target=target[mask], prediction=base[mask], dates=dates[mask], prefix=prefix)
        row.update(score)
        row[f"{prefix}delta_mae_vs_0086_base"] = float(score[f"{prefix}mae"]) - float(
            base_segment[f"{prefix}mae"]
        )
    if extra:
        row.update(extra)
    row["hardened_gate_passed"] = (
        float(row["delta_mae_vs_0086_base"]) < 0.0
        and float(row["old_frame_delta_mae_vs_0086_base"]) <= 0.0
        and float(row["newly_available_delta_mae_vs_0086_base"]) <= 0.0
        and float(row["press_delta_mae_vs_0086_base"]) <= 0.0
        and float(row["rss_delta_mae_vs_0086_base"]) <= 0.0
    )
    return row


def build_outputs(frame: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mask_map = masks(frame)
    base = frame["candidate_prediction_c"].to_numpy(dtype=float)
    raw = frame["forecast_max_c"].to_numpy(dtype=float)
    rows = [
        score_candidate(
            frame,
            candidate_id="official_raw",
            candidate_class="official_raw",
            prediction=raw,
            mask_map=mask_map,
        ),
        score_candidate(
            frame,
            candidate_id=BASE_ID,
            candidate_class="0086_base",
            prediction=base,
            mask_map=mask_map,
        ),
    ]
    definitions: list[dict[str, object]] = []
    predictions = {"official_raw": raw, BASE_ID: base}
    specs_by_id: dict[str, InteractionSpec] = {}
    for spec in make_specs(pairs):
        prediction, _diagnostics = apply_interaction_specialist(frame, spec)
        predictions[spec.candidate_id] = prediction
        specs_by_id[spec.candidate_id] = spec
        definitions.append(
            {
                "candidate_id": spec.candidate_id,
                "candidate_class": "long_history_interaction_specialist",
                "feature_a": spec.feature_a,
                "feature_b": spec.feature_b,
                "family_a": classify_feature_family(spec.feature_a),
                "family_b": classify_feature_family(spec.feature_b),
                "context_mode": spec.context_mode,
                "min_history": spec.min_history,
                "shrink_rows": spec.shrink_rows,
                "correction_cap_c": spec.correction_cap_c,
            }
        )
        rows.append(
            score_candidate(
                frame,
                candidate_id=spec.candidate_id,
                candidate_class="long_history_interaction_specialist",
                prediction=prediction,
                mask_map=mask_map,
                extra={
                    "feature_a": spec.feature_a,
                    "feature_b": spec.feature_b,
                    "family_a": classify_feature_family(spec.feature_a),
                    "family_b": classify_feature_family(spec.feature_b),
                    "context_mode": spec.context_mode,
                    "min_history": spec.min_history,
                },
            )
        )
    scoreboard = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    hardened = scoreboard[scoreboard["hardened_gate_passed"].astype(bool)].copy()
    if not hardened.empty:
        best_id = str(hardened.sort_values(["mae", "rmse"]).iloc[0]["candidate_id"])
    else:
        best_id = BASE_ID
    best_prediction = predictions[best_id]
    if best_id in specs_by_id:
        _prediction, diagnostics = apply_interaction_specialist(frame, specs_by_id[best_id])
    else:
        diagnostics = pd.DataFrame()
    top_predictions = frame[
        [
            "target_date",
            "forecast_source_family",
            "target_tmax_c",
            "forecast_max_c",
            "candidate_prediction_c",
            "season",
            "frame_segment",
            "era_bucket",
        ]
    ].copy()
    top_predictions["candidate_id"] = best_id
    top_predictions["candidate_prediction_c"] = best_prediction
    top_predictions["candidate_error_c"] = best_prediction - top_predictions["target_tmax_c"]
    return scoreboard, pd.DataFrame(definitions), diagnostics, top_predictions


def build_summary(
    *,
    generated_at: str,
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    pairs: pd.DataFrame,
    scoreboard: pd.DataFrame,
) -> dict[str, object]:
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    base = scoreboard[scoreboard["candidate_id"].eq(BASE_ID)].iloc[0].to_dict()
    hardened = scoreboard[scoreboard["hardened_gate_passed"].astype(bool)].copy()
    if hardened.empty:
        best = base
        best_hardened = {}
    else:
        best = hardened.sort_values(["mae", "rmse"]).iloc[0].to_dict()
        best_hardened = best
    return {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "rows": int(len(frame)),
        "first_target_date": date_text(dates.min()),
        "last_target_date": date_text(dates.max()),
        "selected_feature_count": int(len(selected)),
        "interaction_pair_count": int(len(pairs)),
        "candidate_count": int(len(scoreboard)),
        "hardened_candidate_count": int(len(hardened)),
        "best_candidate": str(best["candidate_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "base_0086_mae": float(base["mae"]),
        "base_0086_rmse": float(base["rmse"]),
        "best_delta_mae_vs_0086_base": float(best["delta_mae_vs_0086_base"]),
        "best_hardened_candidate": best_hardened.get("candidate_id", ""),
        "best_hardened_feature_a": best_hardened.get("feature_a", ""),
        "best_hardened_feature_b": best_hardened.get("feature_b", ""),
        "best_hardened_context_mode": best_hardened.get("context_mode", ""),
        "best_hardened_delta_mae_vs_0086_base": best_hardened.get("delta_mae_vs_0086_base"),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "long_history_interaction_specialist_screen_complete",
        "next_recommended_task": (
            "Run 0088 to stack only hardened-passing single-feature and interaction specialists with prior-only "
            "source/frame gates, then compare against 0086 and the older 0081 partial-frame champion. Keep 2024+ sealed."
        ),
    }


def build_readme(
    *,
    generated_at: str,
    summary: dict[str, object],
    selected: pd.DataFrame,
    thresholds: pd.DataFrame,
    pairs: pd.DataFrame,
    scoreboard: pd.DataFrame,
) -> str:
    return f"""# 0087 Long-History Signal Interaction Specialists

Generated: `{generated_at}`

## Purpose

`0086` proved that the strongest single long-history residual signal, `isd_morning_to_midday_temp_rise_c`, can improve the expanded official-anchor frame. `0087` tests the next question: do paired regimes among the top long-history signals carry additional residual information?

This screen mines interactions among ISD morning warming, target-memory, upper-air ceiling/thermal structure, dewpoint spread, sea-temperature, and seasonal climatology signals. It is still conservative: every feature bucket is defined from historical data before 2000, and every residual correction is learned only from earlier official forecast target dates.

## Inputs

- `0086` best guarded prediction frame from `2000-01-02` to `2023-12-31`.
- `0085` ranked long-history residual features.
- Long-history feature matrix at `{FEATURE_MATRIX_PATH}`.
- No 2024+ confirmation rows.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Selected features | `{summary['selected_feature_count']}` |
| Interaction pairs | `{summary['interaction_pair_count']}` |
| Candidate count | `{summary['candidate_count']}` |
| Hardened passing candidates | `{summary['hardened_candidate_count']}` |
| Best candidate | `{summary['best_candidate']}` |
| Best MAE | `{summary['best_mae']}` |
| Best RMSE | `{summary['best_rmse']}` |
| Base 0086 MAE | `{summary['base_0086_mae']}` |
| Delta vs 0086 base | `{summary['best_delta_mae_vs_0086_base']}` |
| Best hardened feature A | `{summary['best_hardened_feature_a']}` |
| Best hardened feature B | `{summary['best_hardened_feature_b']}` |
| Best hardened context | `{summary['best_hardened_context_mode']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Selected Features

{markdown_table(selected, max_rows=20)}

## Pre-2000 Feature Thresholds

{markdown_table(thresholds, max_rows=20)}

## Interaction Pairs

{markdown_table(pairs, max_rows=30)}

## Top Candidates

{markdown_table(scoreboard.head(20), max_rows=20)}

## Interpretation

This screen is deliberately strict. A candidate must improve the full expanded frame and avoid regressions on old-frame, newly available press-frame, press-source, and RSS-source slices to count as hardened. If an interaction candidate wins, it becomes a real candidate for stacking. If the unchanged `0086` base remains best, then the current pair buckets are diagnostic but not yet deployable as direct residual corrections.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(summary: dict[str, object], scoreboard: pd.DataFrame) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0087_long_history_signal_interaction_specialists.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Interaction specialist screen | `{summary['candidate_count']}` candidates over `{summary['rows']}` rows | Pre-2024 only |
| Interaction pairs | `{summary['interaction_pair_count']}` | Top 0085 signal pairs |
| Hardened passing candidates | `{summary['hardened_candidate_count']}` | Source/frame-gated |
| Best candidate | `{summary['best_candidate']}` | `{summary['best_mae']}` MAE |
| Delta vs 0086 base | `{summary['best_delta_mae_vs_0086_base']}` | Interaction value |
| Leakage | `0` 2024+ rows | PASS |

Top 0087 candidates:

{markdown_table(scoreboard.head(8), max_rows=8)}

Interpretation: `0087` tests whether paired long-history regimes add deployable residual signal beyond the `0086` morning-warming specialist.
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0087 Long-History Signal Interaction Specialists",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=(
            "Implement `0088_prior_gated_specialist_stack`: stack the hardened-passing 0086 single-feature "
            "specialists and any hardened-passing 0087 interaction specialists using prior-only source/frame gates. "
            "Keep 2024+ sealed."
        ),
    )


def run() -> dict[str, object]:
    generated_at = now_utc()
    features, base, rankings = load_inputs()
    selected = select_interaction_features(rankings, features)
    frame, thresholds, pairs = build_working_frame(features, base, selected)
    scoreboard, definitions, diagnostics, top_predictions = build_outputs(frame, pairs)
    summary = build_summary(
        generated_at=generated_at,
        frame=frame,
        selected=selected,
        pairs=pairs,
        scoreboard=scoreboard,
    )
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "selected_features.csv", selected)
    write_csv(artifacts / "feature_thresholds.csv", thresholds)
    write_csv(artifacts / "interaction_pairs.csv", pairs)
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "best_gate_diagnostics.csv", diagnostics)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "long_history_signal_interaction_specialists_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            generated_at=generated_at,
            summary=summary,
            selected=selected,
            thresholds=thresholds,
            pairs=pairs,
            scoreboard=scoreboard,
        ),
    )
    update_milestones(summary, scoreboard)
    require_no_confirmation_dates(top_predictions["target_date"], context="0087 top predictions")
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Mine guarded interaction specialists from top long-history HKG Tmax residual signals."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
