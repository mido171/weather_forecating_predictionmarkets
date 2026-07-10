from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
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

FOLDER_NAME = "0086_guarded_long_history_residual_specialists"
INPUT_0084_TOP_PATH = (
    RESEARCH_ROOT / "0084_expanded_frame_hardened_official_specialists" / "artifacts" / "top_predictions.csv"
)
INPUT_0085_RANKINGS_PATH = (
    RESEARCH_ROOT / "0085_long_history_feature_station_residual_bridge" / "artifacts" / "feature_residual_rankings.csv"
)
BASE_ID = "0084_best_official_anchor"
MIN_PRE2000_ROWS = 3650
FEATURES_PER_FAMILY = 2
MAX_FEATURES = 10


@dataclass(frozen=True)
class SpecialistSpec:
    candidate_id: str
    feature: str
    context_mode: str
    min_history: int
    shrink_rows: float
    correction_cap_c: float


@dataclass
class ResidualState:
    residual_sum: float = 0.0
    count: int = 0

    def mean(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.residual_sum / self.count

    def update(self, residual: float) -> None:
        self.residual_sum += float(residual)
        self.count += 1


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [path for path in (FEATURE_MATRIX_PATH, INPUT_0084_TOP_PATH, INPUT_0085_RANKINGS_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"0086 requires feature/0084/0085 artifacts first: {missing}")
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    features = features.loc[:, ~features.columns.duplicated()].copy()
    features["target_date"] = pd.to_datetime(features["target_date"], errors="coerce").dt.normalize()
    features = features[features["target_date"].notna() & (features["target_date"] < CONFIRMATION_START)].copy()
    residuals = pd.read_csv(INPUT_0084_TOP_PATH)
    residuals["target_date"] = pd.to_datetime(residuals["target_date"], errors="coerce").dt.normalize()
    residuals = residuals[residuals["target_date"].notna() & (residuals["target_date"] < CONFIRMATION_START)].copy()
    rankings = pd.read_csv(INPUT_0085_RANKINGS_PATH)
    require_no_confirmation_dates(features["target_date"], context="0086 feature matrix")
    require_no_confirmation_dates(residuals["target_date"], context="0086 0084 residual input")
    return features, residuals, rankings


def select_features(rankings: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    available = rankings[rankings["feature"].isin(features.columns)].copy()
    for _, group in available.groupby("family", observed=True):
        rows.append(group.sort_values("priority_score", ascending=False).head(FEATURES_PER_FAMILY))
    selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    selected = selected.sort_values("priority_score", ascending=False).head(MAX_FEATURES).reset_index(drop=True)
    return selected


def pre2000_thresholds(features: pd.DataFrame, feature: str) -> np.ndarray | None:
    train = features[
        (features["target_date"] >= pd.Timestamp("1949-01-01"))
        & (features["target_date"] <= pd.Timestamp("1999-12-31"))
    ]
    values = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < MIN_PRE2000_ROWS or values.nunique(dropna=True) < 5:
        return None
    thresholds = values.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy(dtype=float)
    thresholds = np.unique(thresholds[np.isfinite(thresholds)])
    if len(thresholds) < 2:
        return None
    return thresholds


def assign_bucket(values: pd.Series, thresholds: np.ndarray) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    bucket = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = numeric.notna()
    bucket.loc[valid] = np.searchsorted(thresholds, numeric.loc[valid].to_numpy(dtype=float), side="right")
    return bucket


def build_working_frame(features: pd.DataFrame, residuals: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = ["target_date", *selected["feature"].tolist()]
    joined = residuals.merge(features[keep], on="target_date", how="left")
    joined["target_tmax_c"] = pd.to_numeric(joined["target_tmax_c"], errors="coerce")
    joined["forecast_max_c"] = pd.to_numeric(joined["forecast_max_c"], errors="coerce")
    joined["candidate_prediction_c"] = pd.to_numeric(joined["candidate_prediction_c"], errors="coerce")
    joined = joined[joined[["target_tmax_c", "forecast_max_c", "candidate_prediction_c"]].notna().all(axis=1)].copy()
    joined["base_residual_c"] = joined["candidate_prediction_c"] - joined["target_tmax_c"]
    joined["forecast_source_family"] = joined["forecast_source_family"].astype(str)
    joined["season"] = joined["season"].astype(str)
    joined["frame_segment"] = joined["frame_segment"].astype(str)
    joined["era_bucket"] = joined["era_bucket"].astype(str)
    threshold_rows: list[dict[str, object]] = []
    for feature in selected["feature"].tolist():
        thresholds = pre2000_thresholds(features, feature)
        if thresholds is None:
            joined[f"{feature}__bucket"] = np.nan
            continue
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
    return joined.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True), pd.DataFrame(threshold_rows)


def context_key(row: pd.Series, spec: SpecialistSpec) -> tuple[object, ...] | None:
    bucket = row.get(f"{spec.feature}__bucket")
    if pd.isna(bucket):
        return None
    if spec.context_mode == "feature":
        return (spec.feature, int(bucket))
    if spec.context_mode == "source_feature":
        return (spec.feature, int(bucket), row["forecast_source_family"])
    if spec.context_mode == "source_season_feature":
        return (spec.feature, int(bucket), row["forecast_source_family"], row["season"])
    if spec.context_mode == "source_frame_feature":
        return (spec.feature, int(bucket), row["forecast_source_family"], row["frame_segment"])
    raise ValueError(f"Unsupported 0086 context mode: {spec.context_mode}")


def make_specs(selected: pd.DataFrame, thresholds: pd.DataFrame) -> list[SpecialistSpec]:
    features_with_thresholds = set(thresholds["feature"].tolist())
    specs: list[SpecialistSpec] = []
    for feature in selected["feature"].tolist():
        if feature not in features_with_thresholds:
            continue
        safe_feature = (
            feature.replace("_", "-")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "-")
            .replace(".", "p")[:80]
        )
        for context_mode in ("feature", "source_feature", "source_frame_feature"):
            min_history = 90
            specs.append(
                SpecialistSpec(
                    candidate_id=f"specialist_{safe_feature}_{context_mode}_m{min_history}",
                    feature=feature,
                    context_mode=context_mode,
                    min_history=min_history,
                    shrink_rows=90.0,
                    correction_cap_c=0.75,
                )
            )
    return specs


def apply_specialist(frame: pd.DataFrame, spec: SpecialistSpec) -> tuple[np.ndarray, pd.DataFrame]:
    base = frame["candidate_prediction_c"].to_numpy(dtype=float)
    residual = frame["base_residual_c"].to_numpy(dtype=float)
    predictions = base.copy()
    active = np.zeros(len(frame), dtype=bool)
    prior_rows = np.zeros(len(frame), dtype=int)
    correction_values = np.zeros(len(frame), dtype=float)
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
                correction_values[row_idx] = correction
                active[row_idx] = abs(correction) > 1e-12
            pending_updates.append((key, residual[row_idx]))
        for key, residual_value in pending_updates:
            states[key].update(residual_value)
    diagnostics = frame[["target_date", "forecast_source_family", "season", "frame_segment", "era_bucket"]].copy()
    diagnostics["candidate_id"] = spec.candidate_id
    diagnostics["feature"] = spec.feature
    diagnostics["context_mode"] = spec.context_mode
    diagnostics["prior_rows"] = prior_rows
    diagnostics["specialist_active"] = active
    diagnostics["specialist_correction_c"] = correction_values
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
    row["delta_mae_vs_0084_base"] = float(row["mae"]) - float(base_score["mae"])
    row["delta_mae_vs_official_raw"] = float(row["mae"]) - float(raw_score["mae"])
    for prefix, mask in mask_map.items():
        score = score_arrays(target=target[mask], prediction=prediction[mask], dates=dates[mask], prefix=prefix)
        base_segment = score_arrays(target=target[mask], prediction=base[mask], dates=dates[mask], prefix=prefix)
        row.update(score)
        row[f"{prefix}delta_mae_vs_0084_base"] = float(score[f"{prefix}mae"]) - float(
            base_segment[f"{prefix}mae"]
        )
    if extra:
        row.update(extra)
    row["hardened_gate_passed"] = (
        float(row["delta_mae_vs_0084_base"]) < 0.0
        and float(row["old_frame_delta_mae_vs_0084_base"]) <= 0.0
        and float(row["newly_available_delta_mae_vs_0084_base"]) <= 0.0
        and float(row["press_delta_mae_vs_0084_base"]) <= 0.0
        and float(row["rss_delta_mae_vs_0084_base"]) <= 0.0
    )
    return row


def build_outputs(frame: pd.DataFrame, selected: pd.DataFrame, thresholds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            candidate_class="0084_base",
            prediction=base,
            mask_map=mask_map,
        ),
    ]
    definitions: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {"official_raw": raw, BASE_ID: base}
    specs_by_id: dict[str, SpecialistSpec] = {}
    for spec in make_specs(selected, thresholds):
        prediction, _diagnostics = apply_specialist(frame, spec)
        predictions[spec.candidate_id] = prediction
        specs_by_id[spec.candidate_id] = spec
        definitions.append(
            {
                "candidate_id": spec.candidate_id,
                "candidate_class": "guarded_long_history_residual_specialist",
                "feature": spec.feature,
                "family": classify_feature_family(spec.feature),
                "station_ids": station_ids_in_feature(spec.feature),
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
                candidate_class="guarded_long_history_residual_specialist",
                prediction=prediction,
                mask_map=mask_map,
                extra={
                    "feature": spec.feature,
                    "family": classify_feature_family(spec.feature),
                    "station_ids": station_ids_in_feature(spec.feature),
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
        _prediction, diagnostics = apply_specialist(frame, specs_by_id[best_id])
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
    thresholds: pd.DataFrame,
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
        "threshold_feature_count": int(len(thresholds)),
        "candidate_count": int(len(scoreboard)),
        "hardened_candidate_count": int(len(hardened)),
        "best_candidate": str(best["candidate_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "base_0084_mae": float(base["mae"]),
        "base_0084_rmse": float(base["rmse"]),
        "best_delta_mae_vs_0084_base": float(best["delta_mae_vs_0084_base"]),
        "best_hardened_candidate": best_hardened.get("candidate_id", ""),
        "best_hardened_feature": best_hardened.get("feature", ""),
        "best_hardened_context_mode": best_hardened.get("context_mode", ""),
        "best_hardened_delta_mae_vs_0084_base": best_hardened.get("delta_mae_vs_0084_base"),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "guarded_long_history_specialist_screen_complete",
        "next_recommended_task": (
            "Run 0087 to combine only hardened-passing long-history specialists with the official-anchor base, "
            "or, if 0086 has no hardened pass, deepen feature engineering around the top 0085 residual signals "
            "before attempting another specialist stack. Keep 2024+ sealed."
        ),
    }


def build_readme(
    *,
    generated_at: str,
    summary: dict[str, object],
    selected: pd.DataFrame,
    thresholds: pd.DataFrame,
    scoreboard: pd.DataFrame,
) -> str:
    return f"""# 0086 Guarded Long-History Residual Specialists

Generated: `{generated_at}`

## Purpose

`0085` ranked long-history features by residual information gain. `0086` turns those signals into actual guarded specialists. This is the first screen in this continuation that uses long-history station/weather attributes directly to modify official-anchor predictions on the expanded official forecast frame.

The design is intentionally conservative. Feature buckets are defined from `1949-1999` history only, not from the evaluation rows. Each specialist then learns a residual correction only from earlier official forecast target dates inside its feature/source/season/frame context. Rows with the same target date are predicted before that date's residuals are added to state.

## Inputs

- `0084` best official-anchor predictions on `2000-01-02` to `2023-12-31`.
- `0085` top residual feature ranking.
- Long-history feature matrix at `{FEATURE_MATRIX_PATH}`.
- No 2024+ confirmation rows.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Selected features | `{summary['selected_feature_count']}` |
| Features with pre-2000 thresholds | `{summary['threshold_feature_count']}` |
| Candidate count | `{summary['candidate_count']}` |
| Hardened passing candidates | `{summary['hardened_candidate_count']}` |
| Best candidate | `{summary['best_candidate']}` |
| Best MAE | `{summary['best_mae']}` |
| Best RMSE | `{summary['best_rmse']}` |
| Base 0084 MAE | `{summary['base_0084_mae']}` |
| Delta vs 0084 base | `{summary['best_delta_mae_vs_0084_base']}` |
| Best hardened feature | `{summary['best_hardened_feature']}` |
| Best hardened context | `{summary['best_hardened_context_mode']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Selected Features

{markdown_table(selected, max_rows=20)}

## Pre-2000 Bucket Thresholds

{markdown_table(thresholds, max_rows=20)}

## Top Candidates

{markdown_table(scoreboard.head(20), max_rows=20)}

## Interpretation

This screen answers whether the strongest long-history feature signals can immediately become deployable-style local residual specialists. A candidate only counts as hardened if it improves full-frame MAE and does not worsen the old frame, newly available press frame, press source, or RSS source versus the `0084` base.

If no candidate passes, that is still useful: it means the feature correlations from `0085` are diagnostic and need richer interaction design before they should be trusted as direct residual corrections. If a candidate passes, it becomes the next component to stack with the official-anchor base.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(summary: dict[str, object], scoreboard: pd.DataFrame) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0086_guarded_long_history_residual_specialists.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Guarded long-history specialist screen | `{summary['candidate_count']}` candidates over `{summary['rows']}` rows | Pre-2024 only |
| Selected features | `{summary['selected_feature_count']}` | From 0085 residual ranking |
| Hardened passing candidates | `{summary['hardened_candidate_count']}` | Acceptance-gated |
| Best candidate | `{summary['best_candidate']}` | `{summary['best_mae']}` MAE |
| Delta vs 0084 base | `{summary['best_delta_mae_vs_0084_base']}` | Specialist value |
| Leakage | `0` 2024+ rows | PASS |

Top 0086 candidates:

{markdown_table(scoreboard.head(8), max_rows=8)}

Interpretation: `0086` is the first guarded residual-specialist conversion from the 0085 long-history feature ranking. Passing candidates must improve full MAE without source/frame regressions.
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0086 Guarded Long-History Residual Specialists",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=(
            "Implement `0087_long_history_signal_interaction_specialists`: if no 0086 specialist passed all hardening "
            "gates, mine interactions among the top 0085 ISD warming, target-memory, upper-air ceiling/thermal, "
            "dewpoint-spread, and Waglan sea-temperature signals before building the next specialist stack. Keep 2024+ sealed."
        ),
    )


def run() -> dict[str, object]:
    generated_at = now_utc()
    features, residuals, rankings = load_inputs()
    selected = select_features(rankings, features)
    frame, thresholds = build_working_frame(features, residuals, selected)
    scoreboard, definitions, diagnostics, top_predictions = build_outputs(frame, selected, thresholds)
    summary = build_summary(
        generated_at=generated_at,
        frame=frame,
        selected=selected,
        thresholds=thresholds,
        scoreboard=scoreboard,
    )
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "selected_features.csv", selected)
    write_csv(artifacts / "bucket_thresholds.csv", thresholds)
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "best_gate_diagnostics.csv", diagnostics)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "guarded_long_history_residual_specialists_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            generated_at=generated_at,
            summary=summary,
            selected=selected,
            thresholds=thresholds,
            scoreboard=scoreboard,
        ),
    )
    update_milestones(summary, scoreboard)
    require_no_confirmation_dates(top_predictions["target_date"], context="0086 top predictions")
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Convert 0085 long-history residual signals into guarded past-only specialists."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
