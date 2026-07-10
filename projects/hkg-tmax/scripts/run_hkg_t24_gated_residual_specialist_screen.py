from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
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
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    FEATURE_MATRIX_PATH,
    TRAIN_END,
    apply_tertile_bins,
    load_feature_matrix,
    quantile_edges_from_train,
    update_markdown_section,
)
from scripts.run_hkg_t24_station_contribution_atlas import (  # noqa: E402
    load_station_day_features,
    load_target,
)

DATASETS_ROOT = PROJECT_PATHS.data_root / "datasets"
OFFICIAL_SCORED_PATH = (
    DATASETS_ROOT
    / "05_hko_historical_rss_forecasts"
    / "hko_official_t15_scored_pre2024.parquet"
)
PHYSICAL_INTERACTIONS_PATH = (
    RESEARCH_ROOT
    / "0046_long_history_cross_family_interaction_atlas"
    / "artifacts"
    / "physical_interactions.csv"
)
STATION_ATTRIBUTE_ATLAS_PATH = (
    RESEARCH_ROOT / "0047_station_contribution_atlas" / "artifacts" / "station_attribute_atlas.csv"
)
PAIR_SPREAD_ATLAS_PATH = (
    RESEARCH_ROOT / "0047_station_contribution_atlas" / "artifacts" / "pair_spread_atlas.csv"
)
FOLDER_NAME = "0048_gated_residual_specialist_screen"
LATE_EVAL_START = pd.Timestamp("2021-04-14")
MIN_HISTORY_OPTIONS = (20, 40)
SHRINK_K_OPTIONS = (20.0,)
CAP_OPTIONS = (0.8, 1.2)
CONTEXT_OPTIONS = ("base", "season")
SAME_SOURCE_OPTIONS = (False, True)
TOP_PHYSICAL_DEFAULT = 6
TOP_STATION_ATTRIBUTES_DEFAULT = 4
TOP_STATION_PAIRS_DEFAULT = 4


@dataclass(frozen=True)
class GateSpec:
    gate_id: str
    family: str
    description: str
    feature_columns: tuple[str, ...]
    source_artifact: str


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    gate_id: str
    family: str
    context_mode: str
    same_source: bool
    min_history: int
    shrink_k: float
    correction_cap_c: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return cleaned[:limit]


def parse_edges(value: object) -> tuple[float, float] | None:
    if value is None or pd.isna(value):
        return None
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list | tuple) or len(parsed) != 2:
        return None
    try:
        left = float(parsed[0])
        right = float(parsed[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(left) or not math.isfinite(right) or left >= right:
        return None
    return left, right


def load_official_frame() -> pd.DataFrame:
    if not OFFICIAL_SCORED_PATH.exists():
        raise FileNotFoundError(f"Missing official scored frame: {OFFICIAL_SCORED_PATH}")
    official = pd.read_parquet(OFFICIAL_SCORED_PATH).copy()
    official["target_date"] = pd.to_datetime(official["target_date"], errors="coerce").dt.normalize()
    official = official[official["target_date"].notna() & (official["target_date"] < CONFIRMATION_START)].copy()
    require_no_confirmation_dates(official["target_date"], context="0048 official scored frame")
    required = {"target_date", "forecast_source_family", "forecast_max_c", "target_tmax_c"}
    missing = required.difference(official.columns)
    if missing:
        raise ValueError(f"Official scored frame missing required columns: {sorted(missing)}")
    official["forecast_max_c"] = pd.to_numeric(official["forecast_max_c"], errors="coerce")
    official["target_tmax_c"] = pd.to_numeric(official["target_tmax_c"], errors="coerce")
    official = official.dropna(subset=["forecast_max_c", "target_tmax_c"]).copy()
    official["official_error_c"] = official["forecast_max_c"] - official["target_tmax_c"]
    official["residual_to_add_c"] = official["target_tmax_c"] - official["forecast_max_c"]
    official["anchor_prediction_c"] = official["forecast_max_c"]
    official["month"] = official["target_date"].dt.month
    official["season"] = ((official["month"] % 12) // 3).astype(int)
    return official.sort_values("target_date").reset_index(drop=True)


def load_wide_feature_frame(required_features: set[str]) -> pd.DataFrame:
    wide = load_feature_matrix(FEATURE_MATRIX_PATH)
    keep = ["target_date", *sorted(feature for feature in required_features if feature in wide.columns)]
    return wide[keep].drop_duplicates("target_date", keep="last").copy()


def station_pair_feature_name(attribute: str, station_a: str, station_b: str) -> str:
    return "pair__" + slug(f"{attribute}__{station_a}_minus_{station_b}", limit=150)


def station_attribute_feature_name(station_id: str, attribute: str) -> str:
    return "station__" + slug(f"{station_id}__{attribute}", limit=150)


def build_station_feature_frame(
    pair_rows: pd.DataFrame,
    station_attribute_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]], list[GateSpec]]:
    target = load_target()
    station_frame = load_station_day_features(target)
    feature_parts: list[pd.DataFrame] = []
    edge_lookup: dict[str, tuple[float, float]] = {}
    specs: list[GateSpec] = []

    for row in pair_rows.itertuples(index=False):
        attribute = str(row.attribute)
        station_a = str(row.station_a)
        station_b = str(row.station_b)
        feature_name = station_pair_feature_name(attribute, station_a, station_b)
        pivot = station_frame.pivot_table(
            index="target_date",
            columns="station_id",
            values=attribute,
            aggfunc="last",
        )
        if station_a not in pivot.columns or station_b not in pivot.columns:
            continue
        series = (pivot[station_a] - pivot[station_b]).rename(feature_name).reset_index()
        edges = quantile_edges_from_train(series.loc[series["target_date"] <= TRAIN_END, feature_name])
        if edges is None:
            continue
        edge_lookup[feature_name] = edges
        feature_parts.append(series)
        specs.append(
            GateSpec(
                gate_id=f"station_pair_{slug(feature_name, limit=80)}",
                family="station_pair_spread",
                description=f"{attribute}: {station_a} minus {station_b}",
                feature_columns=(feature_name,),
                source_artifact="0047 pair_spread_atlas.csv",
            )
        )

    for row in station_attribute_rows.itertuples(index=False):
        station_id = str(row.station_id)
        attribute = str(row.attribute)
        feature_name = station_attribute_feature_name(station_id, attribute)
        subset = station_frame[station_frame["station_id"].eq(station_id)][["target_date", attribute]].copy()
        if subset.empty:
            continue
        subset = subset.rename(columns={attribute: feature_name}).drop_duplicates("target_date", keep="last")
        edges = quantile_edges_from_train(subset.loc[subset["target_date"] <= TRAIN_END, feature_name])
        if edges is None:
            continue
        edge_lookup[feature_name] = edges
        feature_parts.append(subset)
        specs.append(
            GateSpec(
                gate_id=f"station_attribute_{slug(feature_name, limit=80)}",
                family="station_attribute",
                description=f"{station_id}: {attribute}",
                feature_columns=(feature_name,),
                source_artifact="0047 station_attribute_atlas.csv",
            )
        )

    if not feature_parts:
        return pd.DataFrame({"target_date": pd.Series(dtype="datetime64[ns]")}), edge_lookup, specs

    merged = feature_parts[0]
    for part in feature_parts[1:]:
        merged = merged.merge(part, on="target_date", how="outer")
    merged["target_date"] = pd.to_datetime(merged["target_date"], errors="coerce").dt.normalize()
    merged = merged[merged["target_date"].notna() & (merged["target_date"] < CONFIRMATION_START)]
    return merged.drop_duplicates("target_date", keep="last"), edge_lookup, specs


def load_gate_specs(
    *,
    top_physical: int,
    top_station_attributes: int,
    top_station_pairs: int,
) -> tuple[list[GateSpec], pd.DataFrame, dict[str, tuple[float, float]]]:
    physical = pd.read_csv(PHYSICAL_INTERACTIONS_PATH).head(top_physical)
    station_attributes = pd.read_csv(STATION_ATTRIBUTE_ATLAS_PATH).head(top_station_attributes)
    station_pairs = pd.read_csv(PAIR_SPREAD_ATLAS_PATH).head(top_station_pairs)

    required_wide_features: set[str] = set()
    edge_lookup: dict[str, tuple[float, float]] = {}
    specs: list[GateSpec] = []
    for row in physical.itertuples(index=False):
        feature_a = str(row.feature_a)
        feature_b = str(row.feature_b)
        edges_a = parse_edges(row.train_edges_a)
        edges_b = parse_edges(row.train_edges_b)
        if edges_a is None or edges_b is None:
            continue
        required_wide_features.update([feature_a, feature_b])
        edge_lookup[feature_a] = edges_a
        edge_lookup[feature_b] = edges_b
        specs.append(
            GateSpec(
                gate_id=f"physical_{slug(feature_a + '_x_' + feature_b, limit=90)}",
                family="physical_cross_family",
                description=f"{feature_a} x {feature_b}",
                feature_columns=(feature_a, feature_b),
                source_artifact="0046 physical_interactions.csv",
            )
        )

    wide_features = load_wide_feature_frame(required_wide_features)
    station_features, station_edges, station_specs = build_station_feature_frame(
        station_pairs,
        station_attributes,
    )
    edge_lookup.update(station_edges)
    specs.extend(station_specs)

    features = wide_features.merge(station_features, on="target_date", how="outer") if not station_features.empty else wide_features
    features = features.drop_duplicates("target_date", keep="last")
    return specs, features, edge_lookup


def bin_state(values: pd.Series, edges: tuple[float, float]) -> pd.Series:
    return apply_tertile_bins(values, edges).astype("object")


def build_gate_states(
    official_with_features: pd.DataFrame,
    specs: list[GateSpec],
    edge_lookup: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = official_with_features.copy()
    diagnostics: list[dict[str, object]] = []
    for spec in specs:
        state_parts: list[pd.Series] = []
        usable = True
        for feature in spec.feature_columns:
            if feature not in frame.columns or feature not in edge_lookup:
                usable = False
                break
            part = bin_state(pd.to_numeric(frame[feature], errors="coerce"), edge_lookup[feature])
            state_parts.append(part.rename(feature))
        state_col = f"gate_state__{spec.gate_id}"
        if not usable or not state_parts:
            frame[state_col] = np.nan
            diagnostics.append(
                {
                    "gate_id": spec.gate_id,
                    "family": spec.family,
                    "description": spec.description,
                    "usable": False,
                    "non_null_rows": 0,
                    "unique_states": 0,
                }
            )
            continue
        state_frame = pd.concat(state_parts, axis=1)
        missing_state = state_frame.isna().any(axis=1)
        state = state_frame.fillna("__missing__").astype(str).agg("|".join, axis=1)
        state[missing_state] = np.nan
        frame[state_col] = state
        diagnostics.append(
            {
                "gate_id": spec.gate_id,
                "family": spec.family,
                "description": spec.description,
                "usable": True,
                "non_null_rows": int(state.notna().sum()),
                "unique_states": int(state.nunique(dropna=True)),
                "feature_columns": ",".join(spec.feature_columns),
                "source_artifact": spec.source_artifact,
            }
        )
    return frame, pd.DataFrame(diagnostics)


def context_state(base_state: pd.Series, frame: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "base":
        return base_state
    if mode == "season":
        return base_state.astype("object") + "|season=" + frame["season"].astype(str)
    if mode == "month":
        return base_state.astype("object") + "|month=" + frame["month"].astype(str)
    raise ValueError(f"Unknown context mode: {mode}")


def past_only_residual_correction(
    frame: pd.DataFrame,
    *,
    state: pd.Series,
    same_source: bool,
    min_history: int,
    shrink_k: float,
    correction_cap_c: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    work = frame.copy()
    work["_gate_state"] = state.to_numpy()
    ordered = work.sort_values("target_date").reset_index(drop=True)
    residual = ordered["residual_to_add_c"].astype(float)
    dates = ordered["target_date"]
    states = ordered["_gate_state"]
    sources = ordered["forecast_source_family"].astype(str)
    corrections: list[float] = []
    prior_cell_rows: list[int] = []
    prior_global_rows: list[int] = []
    global_stats: dict[str, list[float]] = {}
    cell_stats: dict[tuple[str, str], list[float]] = {}
    for _target_date, group in ordered.groupby(dates, sort=True):
        group_corrections: list[float] = []
        group_cell_rows: list[int] = []
        group_global_rows: list[int] = []
        for index in group.index:
            current_state = states.iloc[index]
            source_key = str(sources.iloc[index]) if same_source else "__all_sources__"
            global_n, global_sum = global_stats.get(source_key, [0.0, 0.0])
            group_global_rows.append(int(global_n))
            if pd.isna(current_state) or global_n < min_history:
                group_corrections.append(0.0)
                group_cell_rows.append(0)
                continue
            cell_key = (source_key, str(current_state))
            cell_n, cell_sum = cell_stats.get(cell_key, [0.0, 0.0])
            group_cell_rows.append(int(cell_n))
            if cell_n < min_history:
                group_corrections.append(0.0)
                continue
            cell_mean = cell_sum / cell_n
            global_mean = global_sum / global_n
            weight = cell_n / (cell_n + shrink_k)
            correction = weight * cell_mean + (1.0 - weight) * global_mean
            group_corrections.append(float(np.clip(correction, -correction_cap_c, correction_cap_c)))
        corrections.extend(group_corrections)
        prior_cell_rows.extend(group_cell_rows)
        prior_global_rows.extend(group_global_rows)

        for index in group.index:
            value = residual.iloc[index]
            current_state = states.iloc[index]
            if pd.isna(value):
                continue
            source_key = str(sources.iloc[index]) if same_source else "__all_sources__"
            global_n, global_sum = global_stats.get(source_key, [0.0, 0.0])
            global_stats[source_key] = [global_n + 1.0, global_sum + float(value)]
            if pd.isna(current_state):
                continue
            cell_key = (source_key, str(current_state))
            cell_n, cell_sum = cell_stats.get(cell_key, [0.0, 0.0])
            cell_stats[cell_key] = [cell_n + 1.0, cell_sum + float(value)]
    return (
        pd.Series(corrections, index=ordered.index),
        pd.Series(prior_cell_rows, index=ordered.index),
        pd.Series(prior_global_rows, index=ordered.index),
    )


def make_candidate_specs(gate_specs: list[GateSpec]) -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    for gate in gate_specs:
        for context_mode, same_source, min_history, shrink_k, cap in itertools.product(
            CONTEXT_OPTIONS,
            SAME_SOURCE_OPTIONS,
            MIN_HISTORY_OPTIONS,
            SHRINK_K_OPTIONS,
            CAP_OPTIONS,
        ):
            candidate_id = (
                f"{gate.gate_id}__ctx_{context_mode}"
                f"__same_source_{int(same_source)}"
                f"__min_{min_history}"
                f"__shrink_{int(shrink_k)}"
                f"__cap_{str(cap).replace('.', 'p')}"
            )
            candidates.append(
                CandidateSpec(
                    candidate_id=candidate_id,
                    gate_id=gate.gate_id,
                    family=gate.family,
                    context_mode=context_mode,
                    same_source=same_source,
                    min_history=min_history,
                    shrink_k=shrink_k,
                    correction_cap_c=cap,
                )
            )
    return candidates


def score_segments(frame: pd.DataFrame, prediction_col: str) -> dict[str, object]:
    full = score_prediction_frame(frame, prediction_col)
    late = score_prediction_frame(frame[frame["target_date"] >= LATE_EVAL_START], prediction_col)
    press = score_prediction_frame(frame[frame["forecast_source_family"].eq("press_archive")], prediction_col)
    rss = score_prediction_frame(frame[frame["forecast_source_family"].eq("rss_archive")], prediction_col)
    return {
        "full_mae": full["mae"],
        "full_rmse": full["rmse"],
        "full_bias": full["bias"],
        "full_n": full["n"],
        "late_mae": late["mae"],
        "late_rmse": late["rmse"],
        "late_n": late["n"],
        "press_mae": press["mae"],
        "press_n": press["n"],
        "rss_mae": rss["mae"],
        "rss_n": rss["n"],
    }


def evaluate_candidates(
    frame: pd.DataFrame,
    gate_specs: list[GateSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchor_scores = score_segments(frame, "anchor_prediction_c")
    candidate_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    candidates = make_candidate_specs(gate_specs)
    for candidate in candidates:
        state_col = f"gate_state__{candidate.gate_id}"
        if state_col not in frame.columns or frame[state_col].notna().sum() == 0:
            continue
        state = context_state(frame[state_col], frame, candidate.context_mode)
        correction, cell_rows, global_rows = past_only_residual_correction(
            frame,
            state=state,
            same_source=candidate.same_source,
            min_history=candidate.min_history,
            shrink_k=candidate.shrink_k,
            correction_cap_c=candidate.correction_cap_c,
        )
        pred_col = "candidate_prediction_c"
        evaluated = frame.sort_values("target_date").reset_index(drop=True).copy()
        evaluated["residual_correction_c"] = correction
        evaluated["prior_cell_rows"] = cell_rows
        evaluated["prior_global_rows"] = global_rows
        evaluated[pred_col] = evaluated["forecast_max_c"] + evaluated["residual_correction_c"]
        scores = score_segments(evaluated, pred_col)
        candidate_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "gate_id": candidate.gate_id,
                "family": candidate.family,
                "context_mode": candidate.context_mode,
                "same_source": candidate.same_source,
                "min_history": candidate.min_history,
                "shrink_k": candidate.shrink_k,
                "correction_cap_c": candidate.correction_cap_c,
                **scores,
                "delta_full_mae_vs_anchor": float(scores["full_mae"] - anchor_scores["full_mae"]),
                "delta_late_mae_vs_anchor": float(scores["late_mae"] - anchor_scores["late_mae"]),
                "rows_with_nonzero_correction": int((evaluated["residual_correction_c"].abs() > 1e-12).sum()),
                "mean_abs_correction_c": float(evaluated["residual_correction_c"].abs().mean()),
            }
        )
        if len(prediction_parts) < 20:
            prediction_parts.append(
                evaluated[
                    [
                        "target_date",
                        "forecast_source_family",
                        "target_tmax_c",
                        "forecast_max_c",
                        "candidate_prediction_c",
                        "residual_correction_c",
                        "prior_cell_rows",
                        "prior_global_rows",
                    ]
                ].assign(candidate_id=candidate.candidate_id)
            )

    scoreboard = pd.DataFrame(candidate_rows)
    if not scoreboard.empty:
        scoreboard = scoreboard.sort_values(
            ["delta_late_mae_vs_anchor", "delta_full_mae_vs_anchor"],
            ascending=[True, True],
        ).reset_index(drop=True)
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    anchor_frame = pd.DataFrame([{"candidate_id": "official_anchor_raw", **anchor_scores}])
    return scoreboard, predictions, anchor_frame


def build_readme(
    *,
    generated_at: str,
    summary: dict[str, Any],
    gate_diagnostics: pd.DataFrame,
    anchor_scores: pd.DataFrame,
    scoreboard: pd.DataFrame,
) -> str:
    gate_display = gate_diagnostics[
        ["gate_id", "family", "usable", "non_null_rows", "unique_states", "description"]
    ].head(40)
    top_display = scoreboard[
        [
            "candidate_id",
            "family",
            "context_mode",
            "same_source",
            "min_history",
            "full_mae",
            "late_mae",
            "delta_full_mae_vs_anchor",
            "delta_late_mae_vs_anchor",
            "rows_with_nonzero_correction",
        ]
    ].head(25) if not scoreboard.empty else pd.DataFrame()

    return f"""# Gated Residual Specialist Screen

Generated: `{generated_at}`

## Purpose

This insight folder tests whether the best signals from `0046` and `0047` can become leakage-safe residual specialists around the official forecast anchor. It is a diagnostic screen, not a production model. The official forecast archive is still non-contiguous, so the result is useful for ranking candidate gates but not enough for final promotion.

## Leakage Control

- All official rows with `target_date >= 2024-01-01` are excluded.
- Gate definitions come from prior discovery artifacts: `0046` physical interactions and `0047` station contribution rankings.
- For each target date, the residual correction uses only official forecast residuals from strictly earlier target dates.
- Same-source variants use only earlier residuals from the same official source family.
- Station features are T-1 latest-before-15:00 HKT features as defined in `0047`.
- The run does not train on or score the sealed 2024-2026 confirmation period.

## Dataset Scope

| Item | Value |
|---|---:|
| Official scored rows | {summary["official_rows"]} |
| Official target range | {summary["official_first_date"]} to {summary["official_last_date"]} |
| Gate specs | {summary["gate_spec_count"]} |
| Candidate variants | {summary["candidate_count"]} |
| Best full MAE | {summary["best_full_mae"]} |
| Best late MAE | {summary["best_late_mae"]} |
| Anchor full MAE | {summary["anchor_full_mae"]} |
| Anchor late MAE | {summary["anchor_late_mae"]} |
| Uses 2024+ rows | {summary["uses_2024_plus_rows"]} |

## Anchor Scores

{markdown_table(anchor_scores, max_rows=10)}

## Gate Diagnostics

{markdown_table(gate_display, max_rows=40)}

## Candidate Leaders

{markdown_table(top_display, max_rows=25)}

## Main Finding

This screen is the first bridge from the feature-atlas work into deployable-style residual logic. It checks whether the strongest station pressure/wind and upper-air gates can safely alter the official forecast using only prior residual evidence. The important result is not just the top MAE. The important result is whether any gate improves both the full non-contiguous frame and the late RSS frame without relying on future data.

If the best candidate improves only one segment, it remains a mechanism clue, not a champion. If it improves both segments, it becomes a candidate for a more serious fold-local stack after the official forecast archive is made continuous.

## What This Does Not Prove

This does not solve the 2008-2026 official press raw-detail gap. It does not use a continuous 2000-2026 forecast archive. It does not touch 2024+ confirmation rows. It should be rerun after the archive backfill is complete and after the official scored export is regenerated.

## Artifact Files

- `artifacts/gate_diagnostics.csv`
- `artifacts/candidate_scoreboard.csv`
- `artifacts/sample_candidate_predictions.csv`
- `artifacts/anchor_scores.csv`
- `artifacts/summary.json`
"""


def milestone_section(*, summary: dict[str, Any], scoreboard: pd.DataFrame) -> str:
    top_display = scoreboard[
        [
            "candidate_id",
            "family",
            "context_mode",
            "same_source",
            "min_history",
            "full_mae",
            "late_mae",
            "delta_full_mae_vs_anchor",
            "delta_late_mae_vs_anchor",
        ]
    ].head(10) if not scoreboard.empty else pd.DataFrame()
    return f"""Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_gated_residual_specialist_screen.py
```

New folder: `research/data_analysis/0048_gated_residual_specialist_screen`.

| Area | Evidence | Status |
|---|---|---|
| Official frame | `{summary["official_rows"]}` rows, `{summary["official_first_date"]}` to `{summary["official_last_date"]}` | Non-contiguous |
| Gates tested | `{summary["gate_spec_count"]}` gates from `0046`/`0047`; `{summary["candidate_count"]}` variants | Diagnostic |
| Leakage guard | each correction uses only target dates strictly before the scored row; zero 2024+ rows | Guarded |
| Anchor score | full MAE `{summary["anchor_full_mae"]}`, late MAE `{summary["anchor_late_mae"]}` | Baseline |
| Best candidate | full MAE `{summary["best_full_mae"]}`, late MAE `{summary["best_late_mae"]}` | Not production eligible |

Top gated residual candidates:

{markdown_table(top_display, max_rows=10)}

Interpretation: `0048` converts the best station/upper-air diagnostic signals into prior-only residual corrections around the official forecast anchor. It is useful as a gate-ranking and mechanism test. It should not be treated as a final champion until the 2008-2026 forecast archive gap is closed and the full official-anchor chain is rerun on a continuous frame.
"""


def run(
    *,
    output_root: Path = RESEARCH_ROOT,
    top_physical: int = TOP_PHYSICAL_DEFAULT,
    top_station_attributes: int = TOP_STATION_ATTRIBUTES_DEFAULT,
    top_station_pairs: int = TOP_STATION_PAIRS_DEFAULT,
) -> dict[str, Any]:
    generated_at = now_utc()
    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"

    official = load_official_frame()
    gate_specs, features, edge_lookup = load_gate_specs(
        top_physical=top_physical,
        top_station_attributes=top_station_attributes,
        top_station_pairs=top_station_pairs,
    )
    official_with_features = official.merge(features, on="target_date", how="left")
    official_with_features, gate_diagnostics = build_gate_states(
        official_with_features,
        gate_specs,
        edge_lookup,
    )
    usable_gates = [
        gate
        for gate in gate_specs
        if f"gate_state__{gate.gate_id}" in official_with_features.columns
        and official_with_features[f"gate_state__{gate.gate_id}"].notna().sum() > 0
    ]
    scoreboard, predictions, anchor_scores = evaluate_candidates(official_with_features, usable_gates)

    best_full = scoreboard.sort_values("delta_full_mae_vs_anchor").iloc[0] if not scoreboard.empty else {}
    best_late = scoreboard.sort_values("delta_late_mae_vs_anchor").iloc[0] if not scoreboard.empty else {}
    anchor_full_mae = float(anchor_scores.iloc[0]["full_mae"])
    anchor_late_mae = float(anchor_scores.iloc[0]["late_mae"])
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "official_rows": int(len(official)),
        "official_first_date": str(official["target_date"].min().date()),
        "official_last_date": str(official["target_date"].max().date()),
        "gate_spec_count": int(len(gate_specs)),
        "usable_gate_count": int(len(usable_gates)),
        "candidate_count": int(len(scoreboard)),
        "anchor_full_mae": anchor_full_mae,
        "anchor_late_mae": anchor_late_mae,
        "best_full_mae": float(best_full["full_mae"]) if len(best_full) else math.nan,
        "best_late_mae": float(best_late["late_mae"]) if len(best_late) else math.nan,
        "best_full_candidate": best_full.to_dict() if len(best_full) else {},
        "best_late_candidate": best_late.to_dict() if len(best_late) else {},
        "uses_2024_plus_rows": False,
        "leakage_guard": {
            "confirmation_start": str(CONFIRMATION_START.date()),
            "prior_residual_rule": "target_date < current target_date",
            "same_source_variants": True,
            "station_feature_timing": "T-1 latest_before_1500_hkt",
        },
    }

    write_csv(artifacts / "gate_diagnostics.csv", gate_diagnostics)
    write_csv(artifacts / "candidate_scoreboard.csv", scoreboard)
    write_csv(artifacts / "sample_candidate_predictions.csv", predictions)
    write_csv(artifacts / "anchor_scores.csv", anchor_scores)
    write_json(artifacts / "summary.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            generated_at=generated_at,
            summary=summary,
            gate_diagnostics=gate_diagnostics,
            anchor_scores=anchor_scores,
            scoreboard=scoreboard,
        ),
    )
    manifest = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "summary_path": str(artifacts / "summary.json"),
        "readme_path": str(folder / "README.md"),
        "official_rows": summary["official_rows"],
        "gate_spec_count": summary["gate_spec_count"],
        "candidate_count": summary["candidate_count"],
        "anchor_full_mae": summary["anchor_full_mae"],
        "anchor_late_mae": summary["anchor_late_mae"],
        "best_full_mae": summary["best_full_mae"],
        "best_late_mae": summary["best_late_mae"],
        "uses_2024_plus_rows": False,
    }
    write_json(output_root / "gated_residual_specialist_screen_manifest.json", manifest)
    update_markdown_section(
        output_root / "README.md",
        heading="0048 Gated Residual Specialist Screen",
        section=(
            f"Generated `{generated_at}`. See `{FOLDER_NAME}`. "
            f"Tested `{summary['candidate_count']}` prior-only residual specialist variants "
            f"from `{summary['usable_gate_count']}` usable station/upper-air gates."
        ),
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Gated Residual Specialist Screen",
        section=milestone_section(summary=summary, scoreboard=scoreboard),
        insert_before="## Current Blockers And Gaps",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen gated residual specialists against official anchor.")
    parser.add_argument("--output-root", type=Path, default=RESEARCH_ROOT)
    parser.add_argument("--top-physical", type=int, default=TOP_PHYSICAL_DEFAULT)
    parser.add_argument("--top-station-attributes", type=int, default=TOP_STATION_ATTRIBUTES_DEFAULT)
    parser.add_argument("--top-station-pairs", type=int, default=TOP_STATION_PAIRS_DEFAULT)
    args = parser.parse_args()
    summary = run(
        output_root=args.output_root,
        top_physical=args.top_physical,
        top_station_attributes=args.top_station_attributes,
        top_station_pairs=args.top_station_pairs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
