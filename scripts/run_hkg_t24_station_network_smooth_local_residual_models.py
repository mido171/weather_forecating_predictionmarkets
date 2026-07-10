from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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
    score_prediction_frame,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_station_network_forecast_residual_interaction_mining import (  # noqa: E402
    FOLDER_NAME as STATION_NETWORK_0039_FOLDER,
)
from scripts.run_hkg_t24_station_network_forecast_residual_interaction_mining import (  # noqa: E402
    LATE_EVAL_START,
    build_analysis_frame,
)

FOLDER_NAME = "0040_station_network_smooth_residuals"
SELECTED_FEATURES_PATH = (
    RESEARCH_ROOT / STATION_NETWORK_0039_FOLDER / "artifacts" / "selected_features.csv"
)
SCREEN_STAGE = "stage1_high_signal_bounded"
TOP_FEATURES = 12
PAIR_FEATURES = 4
K_NEIGHBORS = (60,)
HALF_LIFE_OPTIONS: tuple[float | None, ...] = (None,)
MIN_HISTORY = 180
MIN_MATCH_ROWS = 45
SHRINKAGE = 100.0
CORRECTION_CLIP_C = 1.8


@dataclass(frozen=True)
class SmoothStationSpec:
    rank: int
    feature: str
    feature_label: str
    extra_features: tuple[str, ...]
    state_cols: tuple[str, ...]
    same_source: bool
    k_neighbors: int
    half_life_days: float | None
    min_history: int = MIN_HISTORY
    min_match_rows: int = MIN_MATCH_ROWS
    shrinkage: float = SHRINKAGE
    correction_clip_c: float = CORRECTION_CLIP_C
    min_local_mae_improvement_c: float = 0.0


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 110) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return math.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def finite_numeric_matrix(frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    columns: list[np.ndarray] = []
    for feature in features:
        if feature not in frame.columns:
            columns.append(np.full(len(frame), np.nan, dtype=float))
        else:
            columns.append(pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float))
    return np.column_stack(columns) if columns else np.empty((len(frame), 0), dtype=float)


def prior_scaled_distances(prior_features: np.ndarray, current_features: np.ndarray) -> np.ndarray:
    if prior_features.size == 0 or len(current_features) == 0:
        return np.full(len(prior_features), math.nan)
    q25 = np.nanpercentile(prior_features, 25, axis=0)
    q75 = np.nanpercentile(prior_features, 75, axis=0)
    scale = q75 - q25
    std = np.nanstd(prior_features, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    normalized = (prior_features - current_features) / scale
    return np.sqrt(np.nanmean(np.square(normalized), axis=1))


def smooth_residual_correction(
    *,
    prior_features: np.ndarray,
    prior_residuals: np.ndarray,
    prior_age_days: np.ndarray,
    current_features: np.ndarray,
    k_neighbors: int,
    half_life_days: float | None,
    shrinkage: float,
    correction_clip_c: float,
    min_local_mae_improvement_c: float,
) -> tuple[float, int, float, float, float, bool]:
    distances = prior_scaled_distances(prior_features, current_features)
    valid = np.isfinite(distances) & np.isfinite(prior_residuals)
    if int(valid.sum()) < k_neighbors:
        return 0.0, 0, math.nan, math.nan, math.nan, False
    valid_distances = distances[valid]
    valid_residuals = prior_residuals[valid]
    valid_ages = prior_age_days[valid]
    k = min(k_neighbors, len(valid_distances))
    selected = np.argpartition(valid_distances, k - 1)[:k]
    selected_distances = valid_distances[selected]
    selected_residuals = valid_residuals[selected]
    selected_ages = valid_ages[selected]
    positive_distances = selected_distances[selected_distances > 1e-9]
    scale = float(np.nanmedian(positive_distances)) if len(positive_distances) else math.nan
    if not np.isfinite(scale) or scale <= 1e-9:
        weights = np.ones(len(selected_distances), dtype=float)
    else:
        weights = np.exp(-selected_distances / scale)
    if half_life_days is not None and half_life_days > 0:
        weights = weights * np.power(0.5, selected_ages / float(half_life_days))
    raw_correction = weighted_mean(selected_residuals, weights)
    if not np.isfinite(raw_correction):
        return 0.0, 0, math.nan, math.nan, math.nan, False
    shrink = len(selected_residuals) / (len(selected_residuals) + shrinkage)
    correction = float(np.clip(raw_correction * shrink, -correction_clip_c, correction_clip_c))
    local_anchor_mae = weighted_mean(np.abs(selected_residuals), weights)
    local_corrected_mae = weighted_mean(np.abs(selected_residuals - correction), weights)
    gate_passed = bool(
        np.isfinite(local_anchor_mae)
        and np.isfinite(local_corrected_mae)
        and local_corrected_mae <= local_anchor_mae - min_local_mae_improvement_c
    )
    if not gate_passed:
        correction = 0.0
    return (
        correction,
        int(len(selected_residuals)),
        float(np.mean(selected_distances)),
        float(local_anchor_mae),
        float(local_corrected_mae),
        gate_passed,
    )


def load_selected_features() -> pd.DataFrame:
    if not SELECTED_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing 0039 selected feature catalog: {SELECTED_FEATURES_PATH}")
    selected = pd.read_csv(SELECTED_FEATURES_PATH)
    if "feature" not in selected.columns or "rank" not in selected.columns:
        raise ValueError(f"Selected feature catalog lacks required columns: {SELECTED_FEATURES_PATH}")
    return selected.sort_values("rank").reset_index(drop=True)


def available_selected_features(frame: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in selected.head(TOP_FEATURES).itertuples(index=False):
        feature = str(row.feature)
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if int(values.notna().sum()) < MIN_HISTORY or values.nunique(dropna=True) <= 2:
            continue
        rows.append(
            {
                "rank": int(row.rank),
                "feature": feature,
                "feature_label": f"f{int(row.rank):02d}_{slug(feature, limit=55)}",
                "family": str(getattr(row, "family", "")),
                "interaction_priority": float(getattr(row, "interaction_priority", math.nan)),
            }
        )
    return pd.DataFrame(rows)


def state_sets(frame: pd.DataFrame) -> tuple[tuple[str, ...], ...]:
    candidates = (
        (),
        ("meta_text_signal_state",),
        ("meta_forecast_range_change_sign",),
        ("meta_forecast_vs_prior7_sign",),
    )
    out: list[tuple[str, ...]] = []
    for cols in candidates:
        filtered = tuple(col for col in cols if col in frame.columns)
        if filtered not in out:
            out.append(filtered)
    return tuple(out)


def extra_feature_sets(frame: pd.DataFrame, *, include_pairs: bool) -> tuple[tuple[str, ...], ...]:
    if not include_pairs:
        return ((),)
    candidates = (
        (),
        ("forecast_max_vs_prior7_mean_source_c",),
        ("forecast_max_change_1_source_c",),
    )
    out: list[tuple[str, ...]] = []
    for cols in candidates:
        filtered = tuple(col for col in cols if col in frame.columns)
        if filtered not in out:
            out.append(filtered)
    return tuple(out)


def build_specs(frame: pd.DataFrame, feature_catalog: pd.DataFrame) -> list[SmoothStationSpec]:
    specs: list[SmoothStationSpec] = []
    seen: set[tuple[object, ...]] = set()
    for row in feature_catalog.itertuples(index=False):
        include_pairs = int(row.rank) <= PAIR_FEATURES
        for extra_features in extra_feature_sets(frame, include_pairs=include_pairs):
            for states in state_sets(frame):
                for same_source in (False, True):
                    for k_neighbors in K_NEIGHBORS:
                        for half_life_days in HALF_LIFE_OPTIONS:
                            key = (
                                str(row.feature),
                                extra_features,
                                states,
                                same_source,
                                k_neighbors,
                                half_life_days,
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            specs.append(
                                SmoothStationSpec(
                                    rank=int(row.rank),
                                    feature=str(row.feature),
                                    feature_label=str(row.feature_label),
                                    extra_features=extra_features,
                                    state_cols=states,
                                    same_source=same_source,
                                    k_neighbors=k_neighbors,
                                    half_life_days=half_life_days,
                                )
                            )
    return specs


def candidate_id(spec: SmoothStationSpec) -> str:
    states = "feature_only" if not spec.state_cols else "_".join(col.replace("meta_", "") for col in spec.state_cols)
    extras = "solo" if not spec.extra_features else "plus_" + "_".join(slug(col, limit=30) for col in spec.extra_features)
    source = "same_source" if spec.same_source else "all_prior"
    half_life = "hl_none" if spec.half_life_days is None else f"hl_{int(spec.half_life_days)}d"
    return slug(f"smooth_station_{spec.feature_label}_{extras}_{states}_{source}_k{spec.k_neighbors}_{half_life}")


def past_only_smooth_predictions(frame: pd.DataFrame, spec: SmoothStationSpec) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").dt.normalize()
    date_array = dates.to_numpy(dtype="datetime64[ns]")
    source_array = ordered["forecast_source_family"].astype(str).to_numpy()
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    anchor = pd.to_numeric(ordered["anchor_0038_c"], errors="coerce").to_numpy(dtype=float)
    official = pd.to_numeric(ordered["official_raw"], errors="coerce").to_numpy(dtype=float)
    residual = target - anchor
    features = (spec.feature, *spec.extra_features)
    feature_matrix = finite_numeric_matrix(ordered, features)
    state_values = {col: ordered[col].fillna("missing").astype(str).to_numpy() for col in spec.state_cols}

    predictions: list[float] = []
    corrections: list[float] = []
    prior_rows: list[int] = []
    matched_rows: list[int] = []
    neighbor_rows: list[int] = []
    mean_distances: list[float] = []
    local_anchor_maes: list[float] = []
    local_corrected_maes: list[float] = []
    gate_passed: list[bool] = []

    for index, target_date in enumerate(date_array):
        if not np.isfinite(anchor[index]) or not np.isfinite(feature_matrix[index]).all():
            predictions.append(float(anchor[index]) if np.isfinite(anchor[index]) else math.nan)
            corrections.append(0.0)
            prior_rows.append(0)
            matched_rows.append(0)
            neighbor_rows.append(0)
            mean_distances.append(math.nan)
            local_anchor_maes.append(math.nan)
            local_corrected_maes.append(math.nan)
            gate_passed.append(False)
            continue

        prior_limit = int(np.searchsorted(date_array, target_date, side="left"))
        base_prior = np.arange(len(ordered)) < prior_limit
        if spec.same_source:
            base_prior &= source_array == source_array[index]
        base_prior &= np.isfinite(residual) & np.isfinite(anchor) & np.isfinite(target)
        base_prior &= np.isfinite(feature_matrix).all(axis=1)
        for values in state_values.values():
            current_state = values[index]
            if current_state == "missing":
                base_prior &= False
            else:
                base_prior &= values == current_state

        prior_count = int(base_prior.sum())
        if prior_count < spec.min_history or prior_count < spec.min_match_rows:
            predictions.append(float(anchor[index]))
            corrections.append(0.0)
            prior_rows.append(prior_count)
            matched_rows.append(prior_count)
            neighbor_rows.append(0)
            mean_distances.append(math.nan)
            local_anchor_maes.append(math.nan)
            local_corrected_maes.append(math.nan)
            gate_passed.append(False)
            continue

        prior_index = np.flatnonzero(base_prior)
        age_days = (dates.iloc[index] - dates.iloc[prior_index]).dt.days.to_numpy(dtype=float)
        correction, neighbors, mean_distance, local_anchor_mae, local_corrected_mae, passed = smooth_residual_correction(
            prior_features=feature_matrix[prior_index],
            prior_residuals=residual[prior_index],
            prior_age_days=age_days,
            current_features=feature_matrix[index],
            k_neighbors=spec.k_neighbors,
            half_life_days=spec.half_life_days,
            shrinkage=spec.shrinkage,
            correction_clip_c=spec.correction_clip_c,
            min_local_mae_improvement_c=spec.min_local_mae_improvement_c,
        )
        predictions.append(float(anchor[index] + correction))
        corrections.append(correction)
        prior_rows.append(prior_count)
        matched_rows.append(prior_count)
        neighbor_rows.append(neighbors)
        mean_distances.append(mean_distance)
        local_anchor_maes.append(local_anchor_mae)
        local_corrected_maes.append(local_corrected_mae)
        gate_passed.append(passed)

    out = ordered[["target_date", "forecast_source_family", "target_tmax_c"]].copy()
    out["official_raw"] = official
    out["anchor_0038_c"] = anchor
    out["candidate_prediction_c"] = predictions
    out["residual_correction_c"] = corrections
    out["prior_rows"] = prior_rows
    out["matched_rows"] = matched_rows
    out["neighbor_rows"] = neighbor_rows
    out["mean_neighbor_distance"] = mean_distances
    out["local_anchor_mae"] = local_anchor_maes
    out["local_corrected_mae"] = local_corrected_maes
    out["do_no_harm_gate_passed"] = gate_passed
    out["candidate_id"] = candidate_id(spec)
    out["feature"] = spec.feature
    out["extra_features"] = ",".join(spec.extra_features)
    out["state_cols"] = ",".join(spec.state_cols)
    out["same_source"] = spec.same_source
    out["k_neighbors"] = spec.k_neighbors
    out["half_life_days"] = -1 if spec.half_life_days is None else float(spec.half_life_days)
    return out


def score_candidate(predictions: pd.DataFrame, spec: SmoothStationSpec) -> dict[str, object]:
    score = score_prediction_frame(predictions.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    anchor = score_prediction_frame(predictions.rename(columns={"anchor_0038_c": "prediction"}), "prediction")
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "prediction"}), "prediction")
    late = predictions[pd.to_datetime(predictions["target_date"]) >= LATE_EVAL_START].copy()
    late_score = score_prediction_frame(late.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    late_anchor = score_prediction_frame(late.rename(columns={"anchor_0038_c": "prediction"}), "prediction")
    corrected = pd.to_numeric(predictions["residual_correction_c"], errors="coerce").abs() > 1e-12
    return {
        "candidate_id": candidate_id(spec),
        "feature_rank": spec.rank,
        "feature": spec.feature,
        "extra_features": ",".join(spec.extra_features),
        "state_cols": ",".join(spec.state_cols),
        "same_source": spec.same_source,
        "k_neighbors": spec.k_neighbors,
        "half_life_days": -1 if spec.half_life_days is None else float(spec.half_life_days),
        **score,
        "anchor_same_rows_mae": anchor["mae"],
        "official_same_rows_mae": official["mae"],
        "delta_vs_anchor": float(score["mae"] - anchor["mae"]),
        "delta_vs_official": float(score["mae"] - official["mae"]),
        "late_eval_n": int(late_score["n"]),
        "late_eval_first_date": str(late_score["first_date"]),
        "late_eval_last_date": str(late_score["last_date"]),
        "late_eval_mae": float(late_score["mae"]),
        "late_eval_rmse": float(late_score["rmse"]),
        "late_eval_anchor_mae": float(late_anchor["mae"]),
        "late_eval_delta_vs_anchor": float(late_score["mae"] - late_anchor["mae"]),
        "corrected_rows": int(corrected.sum()),
        "gate_passed_rows": int(predictions["do_no_harm_gate_passed"].sum()),
        "mean_abs_correction_c": float(pd.to_numeric(predictions["residual_correction_c"], errors="coerce").abs().mean()),
    }


def run_smooth_screen(frame: pd.DataFrame, specs: list[SmoothStationSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs:
        predictions = past_only_smooth_predictions(frame, spec)
        score_rows.append(score_candidate(predictions, spec))
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(score_rows)
    if not scoreboard.empty:
        scoreboard = scoreboard.sort_values(["late_eval_mae", "mae", "rmse"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return scoreboard, predictions


def baseline_comparison(scoreboard: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for system, column in (("0040_anchor_0038", "anchor_0038_c"), ("official_raw", "official_raw")):
        score = score_prediction_frame(frame.rename(columns={column: "prediction"}), "prediction")
        late = frame[pd.to_datetime(frame["target_date"]) >= LATE_EVAL_START].copy()
        late_score = score_prediction_frame(late.rename(columns={column: "prediction"}), "prediction")
        rows.append(
            {
                "system": system,
                "candidate_id": column,
                **score,
                "late_eval_mae": late_score["mae"],
                "late_eval_rmse": late_score["rmse"],
            }
        )
    if not scoreboard.empty:
        best_late = scoreboard.iloc[0]
        best_full = scoreboard.sort_values(["mae", "late_eval_mae", "rmse"]).iloc[0]
        for system, best in (
            ("0040_best_late_smooth_station_network", best_late),
            ("0040_best_full_smooth_station_network", best_full),
        ):
            rows.append(
                {
                    "system": system,
                    "candidate_id": str(best["candidate_id"]),
                    "n": int(best["n"]),
                    "first_date": str(best["first_date"]),
                    "last_date": str(best["last_date"]),
                    "mae": float(best["mae"]),
                    "rmse": float(best["rmse"]),
                    "bias": float(best["bias"]),
                    "median_abs_error": float(best["median_abs_error"]),
                    "late_eval_mae": float(best["late_eval_mae"]),
                    "late_eval_rmse": float(best["late_eval_rmse"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["late_eval_mae", "mae"]).reset_index(drop=True)


def write_readme(
    *,
    folder: Path,
    manifest: dict[str, object],
    feature_catalog: pd.DataFrame,
    scoreboard: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best_late = scoreboard.iloc[0] if not scoreboard.empty else None
    best_full = scoreboard.sort_values(["mae", "late_eval_mae", "rmse"]).iloc[0] if not scoreboard.empty else None
    best_text = "No scoreable smooth station-network candidate was produced."
    if best_late is not None and best_full is not None:
        best_text = (
            f"Best actual late-window smooth candidate: `{best_late['candidate_id']}` with MAE "
            f"`{best_late['late_eval_mae']:.4f}` versus anchor `{best_late['late_eval_anchor_mae']:.4f}` "
            f"(delta `{best_late['late_eval_delta_vs_anchor']:.4f}`), and full MAE `{best_late['mae']:.4f}`.\n\n"
            f"Best full-window smooth candidate: `{best_full['candidate_id']}` with full MAE "
            f"`{best_full['mae']:.4f}` versus anchor `{best_full['anchor_same_rows_mae']:.4f}` "
            f"(delta `{best_full['delta_vs_anchor']:.4f}`), and actual late-window MAE `{best_full['late_eval_mae']:.4f}`."
        )
    readme = f"""# Station-Network Smooth Local Residual Models

Generated: `{manifest['generated_at_utc']}`

## Purpose

`0039` found that hard station/network residual buckets can improve the current best forecast-history anchor, especially pressure-change crossed with official text state. This run tests a smoother version: for each selected station/network feature, find prior rows with similar continuous feature values, optionally inside the same forecast/text state, and apply a shrinked residual correction only when the local prior neighborhood says the correction would have helped.

This is a bounded stage-1 smooth screen. The first expanded 800-candidate grid exceeded the local execution timeout before writing artifacts, so this version keeps the strongest pre-2018 feature ranks and one robust neighbor setting. It is intentionally staged rather than silently claiming the full expanded grid finished.

## Data Window

Rows used: `{manifest['official_rows']}` scored forecast rows.

Full date range: `{manifest['first_target_date']}` to `{manifest['last_target_date']}`.

Configured late evaluation start: `{manifest['late_eval_start']}`.

Actual late evaluation date range: `{manifest['late_eval_first_target_date']}` to `{manifest['late_eval_last_target_date']}`.

Late evaluation rows: `{manifest['late_eval_rows']}`.

## Leakage Contract

- All scored rows are earlier than `{CONFIRMATION_START.date()}`.
- Feature selection comes from the 0039 pre-2018 selected-feature artifact.
- For each target row, neighbors are restricted to `target_date < current target_date`.
- Same-source variants additionally restrict prior rows to the current forecast source family.
- Optional state-gated variants require exact prior/current state matches using pre-cutoff forecast/text state columns.
- Distance scaling is computed from the current row's eligible prior rows only.
- The do-no-harm gate uses only the selected prior neighbors and suppresses corrections that would not have improved local prior MAE.
- 2024+ confirmation rows are not loaded or scored.

## Main Result

{best_text}

## Baseline Comparison

{markdown_table(comparison, max_rows=10)}

## Feature Catalog

{markdown_table(feature_catalog, max_rows=30)}

## Candidate Scoreboard

{markdown_table(scoreboard.head(60), max_rows=60)}

## Interpretation

This experiment tests whether the 0039 station/network signal is smoother than a simple bucket lookup. If the best smooth candidate improves late-window and full-window MAE, the station/network residual channel is worth promoting into the next stack. If it trails 0039, the hard pressure/text state correction remains the better simple deployable form until the forecast archive becomes continuous.
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Station-Network Smooth Local Residual Models\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_network_smooth_local_residual_models.py`:

- `{FOLDER_NAME}`: smooth prior-neighbor station/network residual specialists around the current 0038/0039 anchor.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Selected continuous features | {manifest['selected_features']} |
| Screen stage | {manifest['screen_stage']} |
| Smooth candidates | {manifest['smooth_candidates']} |
| Best late candidate full MAE | {manifest['best_late_candidate_full_mae']} |
| Best late candidate actual late-window MAE | {manifest['best_late_eval_mae']} |
| Best late candidate actual late-window delta vs anchor | {manifest['best_late_eval_delta_vs_anchor']} |
| Best full candidate full MAE | {manifest['best_full_mae']} |
| Best full candidate full delta vs anchor | {manifest['best_full_delta_vs_anchor']} |

Leakage contract: features are selected from the 0039 pre-2018 artifact; each smooth correction uses only prior target dates and prior-only local scaling; all scored rows are before `{CONFIRMATION_START.date()}`.
"""
    write_text(index_path, text)


def update_milestones(manifest: dict[str, object]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Milestones\n"
    section_marker = "\n## Station-Network Smooth Local Residual Models\n"
    blockers_marker = "\n## Current Blockers And Gaps\n"
    next_marker = "\n## Exact Next Recommended Codex Task\n"
    if section_marker in existing:
        base, rest = existing.split(section_marker, 1)
        suffix = f"{blockers_marker}{rest.split(blockers_marker, 1)[1]}" if blockers_marker in rest else ""
    elif blockers_marker in existing:
        base, rest = existing.split(blockers_marker, 1)
        suffix = f"{blockers_marker}{rest}"
    else:
        base = existing.rstrip()
        suffix = ""

    section = f"""{base.rstrip()}
{section_marker}
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_network_smooth_local_residual_models.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Rows / candidates | Strongest current finding | Status |
|---|---:|---|---|
| Smooth station/network residual specialists | `{manifest['official_rows']}` rows; `{manifest['selected_features']}` selected features; `{manifest['smooth_candidates']}` candidates | Best late candidate `{manifest['best_late_candidate']}`: actual late MAE `{manifest['best_late_eval_mae']}`, late delta `{manifest['best_late_eval_delta_vs_anchor']}`, full MAE `{manifest['best_late_candidate_full_mae']}`. Best full candidate `{manifest['best_full_candidate']}`: full MAE `{manifest['best_full_mae']}`, full delta `{manifest['best_full_delta_vs_anchor']}`, actual late MAE `{manifest['best_full_candidate_late_eval_mae']}` | Audited |
| Actual late window | `{manifest['late_eval_rows']}` rows | `{manifest['late_eval_first_target_date']}` to `{manifest['late_eval_last_target_date']}` because the stable scored archive is still non-contiguous | Explicit |
| Leakage guards | prior-neighbor scaling, exact state gates, do-no-harm gate | Zero 2024+ scored rows; every correction uses only `target_date < current target_date` | Guarded |

Interpretation: `0040` checks whether the station/network signal from `0039` benefits from smooth local residual modelling rather than hard bucket corrections. The result should be compared against both the `0038` anchor and the `0039` hard-bucket result before promotion.
"""
    if suffix:
        if next_marker in suffix:
            before_next, _after_next = suffix.split(next_marker, 1)
            next_task = f"""{next_marker}

Compare `0039` hard station-network interactions and `0040` smooth local residual specialists. If `0040` improves both full-window and actual late-window MAE, implement `0041_station_network_forecast_stack`: stack the best 0037/0038 forecast-history trust anchor, 0039 hard residual correction, and 0040 smooth specialists with strict prior-only trust weights. If `0040` does not improve, promote the best 0039 hard pressure/text interaction into the stack and focus next on refreshing the non-contiguous 2005-2026 forecast archive.
"""
            suffix = before_next.rstrip() + "\n\n" + next_task
        section += suffix
    write_text(path, section)


def write_outputs(
    *,
    frame: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    comparison = baseline_comparison(scoreboard, frame)
    write_csv(artifacts / "feature_catalog.csv", feature_catalog)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "predictions.csv", predictions)
    write_csv(artifacts / "baseline_comparison.csv", comparison)
    top_ids = scoreboard.head(8)["candidate_id"].to_list() if not scoreboard.empty else []
    write_csv(
        artifacts / "top_predictions.csv",
        predictions[predictions["candidate_id"].isin(top_ids)].copy() if top_ids else predictions.head(0),
    )

    best_late = scoreboard.iloc[0] if not scoreboard.empty else None
    best_full = scoreboard.sort_values(["mae", "late_eval_mae", "rmse"]).iloc[0] if not scoreboard.empty else None
    late_mask = pd.to_datetime(frame["target_date"]) >= LATE_EVAL_START
    late_frame = frame[late_mask].copy()
    late_first = "" if late_frame.empty else str(pd.to_datetime(late_frame["target_date"]).min().date())
    late_last = "" if late_frame.empty else str(pd.to_datetime(late_frame["target_date"]).max().date())
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "screen_stage": SCREEN_STAGE,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "late_eval_start": str(LATE_EVAL_START.date()),
        "late_eval_first_target_date": late_first,
        "late_eval_last_target_date": late_last,
        "late_eval_rows": int(late_mask.sum()),
        "selected_features": int(len(feature_catalog)),
        "smooth_candidates": int(len(scoreboard)),
        "best_late_candidate": "" if best_late is None else str(best_late["candidate_id"]),
        "best_late_candidate_full_mae": None if best_late is None else float(best_late["mae"]),
        "best_late_eval_mae": None if best_late is None else float(best_late["late_eval_mae"]),
        "best_late_eval_delta_vs_anchor": None
        if best_late is None
        else float(best_late["late_eval_delta_vs_anchor"]),
        "best_full_candidate": "" if best_full is None else str(best_full["candidate_id"]),
        "best_full_mae": None if best_full is None else float(best_full["mae"]),
        "best_full_delta_vs_anchor": None if best_full is None else float(best_full["delta_vs_anchor"]),
        "best_full_candidate_late_eval_mae": None if best_full is None else float(best_full["late_eval_mae"]),
        "anchor_full_mae": float(comparison.loc[comparison["system"].eq("0040_anchor_0038"), "mae"].iloc[0]),
        "anchor_late_eval_mae": float(
            comparison.loc[comparison["system"].eq("0040_anchor_0038"), "late_eval_mae"].iloc[0]
        ),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "station_network_smooth_residuals_manifest.json", manifest)
    write_readme(
        folder=folder,
        manifest=manifest,
        feature_catalog=feature_catalog,
        scoreboard=scoreboard,
        comparison=comparison,
    )
    update_master_index(manifest)
    update_milestones(manifest)
    return manifest


def run() -> dict[str, object]:
    frame, _derived_catalog = build_analysis_frame()
    require_no_confirmation_dates(frame["target_date"], context="0040 smooth station-network frame")
    selected = load_selected_features()
    feature_catalog = available_selected_features(frame, selected)
    specs = build_specs(frame, feature_catalog)
    scoreboard, predictions = run_smooth_screen(frame, specs)
    require_no_confirmation_dates(predictions["target_date"], context="0040 smooth station-network predictions")
    return write_outputs(
        frame=frame,
        feature_catalog=feature_catalog,
        scoreboard=scoreboard,
        predictions=predictions,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 station-network smooth local residual models.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
