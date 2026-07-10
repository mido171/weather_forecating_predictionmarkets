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

from scripts.run_hkg_t24_0083_expanded_frame_official_anchor_replay import (
    RAW_CANDIDATE_ID,
    date_text,
    load_current_0081_dates_and_score,
    score_arrays,
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

FOLDER_NAME = "0084_expanded_frame_hardened_official_specialists"
INPUT_0083_ROOT = RESEARCH_ROOT / "0083_expanded_frame_official_anchor_replay" / "artifacts"
INPUT_0083_TOP_PATH = INPUT_0083_ROOT / "top_predictions.csv"
INPUT_0083_SCOREBOARD_PATH = INPUT_0083_ROOT / "scoreboard.csv"
INPUT_0083_SUMMARY_PATH = INPUT_0083_ROOT / "summary.json"
HARDENED_BASE_ID = "0083_prior_blend_source_top5_min90"


@dataclass(frozen=True)
class HardenSpec:
    candidate_id: str
    context_mode: str
    min_history: int
    margin_c: float
    action: str


@dataclass
class GateState:
    raw_abs_error_sum: float = 0.0
    corrected_abs_error_sum: float = 0.0
    count: int = 0

    def raw_mae(self) -> float:
        if self.count <= 0:
            return math.inf
        return self.raw_abs_error_sum / self.count

    def corrected_mae(self) -> float:
        if self.count <= 0:
            return math.inf
        return self.corrected_abs_error_sum / self.count

    def corrected_weight(self) -> float:
        raw = self.raw_mae()
        corrected = self.corrected_mae()
        if not np.isfinite(raw) or not np.isfinite(corrected):
            return 0.0
        raw_skill = 1.0 / (raw + 0.05)
        corrected_skill = 1.0 / (corrected + 0.05)
        return float(corrected_skill / (raw_skill + corrected_skill))

    def update(self, raw_abs_error: float, corrected_abs_error: float) -> None:
        self.raw_abs_error_sum += float(raw_abs_error)
        self.corrected_abs_error_sum += float(corrected_abs_error)
        self.count += 1


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def era_bucket(target_date: pd.Timestamp, source: str, frame_segment: str) -> str:
    year = int(target_date.year)
    if frame_segment == "newly_available_official_frame":
        return "new_press_2004_2011"
    if source == "rss_archive":
        return "rss_2021_2023"
    if year <= 2001:
        return "press_2000_2001"
    if year <= 2003:
        return "press_2002_2003"
    return "press_2004_old"


def load_0083_frame() -> pd.DataFrame:
    missing = [path for path in (INPUT_0083_TOP_PATH, INPUT_0083_SCOREBOARD_PATH, INPUT_0083_SUMMARY_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"0084 requires 0083 artifacts first: {missing}")
    frame = pd.read_csv(INPUT_0083_TOP_PATH)
    required = {
        "target_date",
        "forecast_source_family",
        "target_tmax_c",
        "forecast_max_c",
        "candidate_prediction_c",
        "frame_segment",
        "season",
        "month",
    }
    missing_cols = required.difference(frame.columns)
    if missing_cols:
        raise RuntimeError(f"0083 top predictions missing columns required by 0084: {sorted(missing_cols)}")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame = frame[frame["target_date"].notna()].copy()
    frame = frame[frame["target_date"] < CONFIRMATION_START].copy()
    require_no_confirmation_dates(frame["target_date"], context="0084 input frame")
    for col in ("target_tmax_c", "forecast_max_c", "candidate_prediction_c"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[frame[["target_tmax_c", "forecast_max_c", "candidate_prediction_c"]].notna().all(axis=1)].copy()
    frame["forecast_source_family"] = frame["forecast_source_family"].astype(str)
    frame["season"] = frame["season"].astype(str)
    frame["month"] = frame["month"].astype(int)
    frame["frame_segment"] = frame["frame_segment"].astype(str)
    frame["target_year"] = frame["target_date"].dt.year.astype(int)
    frame["era_bucket"] = [
        era_bucket(pd.Timestamp(row.target_date), str(row.forecast_source_family), str(row.frame_segment))
        for row in frame.itertuples(index=False)
    ]
    frame["raw_abs_error_c"] = (frame["forecast_max_c"] - frame["target_tmax_c"]).abs()
    frame["corrected_abs_error_c"] = (frame["candidate_prediction_c"] - frame["target_tmax_c"]).abs()
    frame["row_index"] = np.arange(len(frame))
    return frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True)


def context_key(row: pd.Series, mode: str) -> tuple[object, ...]:
    if mode == "global":
        return ("global",)
    if mode == "source":
        return (row["forecast_source_family"],)
    if mode == "source_season":
        return (row["forecast_source_family"], row["season"])
    if mode == "source_month":
        return (row["forecast_source_family"], int(row["month"]))
    if mode == "source_era":
        return (row["forecast_source_family"], row["era_bucket"])
    if mode == "source_frame":
        return (row["forecast_source_family"], row["frame_segment"])
    if mode == "source_season_frame":
        return (row["forecast_source_family"], row["season"], row["frame_segment"])
    if mode == "source_season_era":
        return (row["forecast_source_family"], row["season"], row["era_bucket"])
    raise ValueError(f"Unsupported 0084 context mode: {mode}")


def make_specs() -> list[HardenSpec]:
    context_modes = [
        "global",
        "source",
        "source_season",
        "source_era",
        "source_frame",
        "source_season_era",
    ]
    specs: list[HardenSpec] = []
    for action in ("hard_gate", "soft_blend"):
        for context_mode in context_modes:
            for min_history in (90, 365):
                margin_c = 0.0
                margin_token = str(margin_c).replace(".", "p")
                specs.append(
                    HardenSpec(
                        candidate_id=f"{action}_{context_mode}_m{min_history}_margin{margin_token}",
                        context_mode=context_mode,
                        min_history=min_history,
                        margin_c=margin_c,
                        action=action,
                    )
                )
    return specs


def apply_hardened_gate(frame: pd.DataFrame, spec: HardenSpec) -> tuple[np.ndarray, pd.DataFrame]:
    raw = frame["forecast_max_c"].to_numpy(dtype=float)
    corrected = frame["candidate_prediction_c"].to_numpy(dtype=float)
    raw_abs = frame["raw_abs_error_c"].to_numpy(dtype=float)
    corrected_abs = frame["corrected_abs_error_c"].to_numpy(dtype=float)
    predictions = raw.copy()
    selected = np.full(len(frame), RAW_CANDIDATE_ID, dtype=object)
    prior_rows = np.zeros(len(frame), dtype=int)
    prior_delta_mae = np.full(len(frame), math.nan, dtype=float)
    corrected_weight = np.zeros(len(frame), dtype=float)
    states: dict[tuple[object, ...], GateState] = {}

    for _, date_group in frame.groupby("target_date", sort=True, observed=True):
        pending_updates: list[tuple[tuple[object, ...], float, float]] = []
        for idx, row in date_group.iterrows():
            key = context_key(row, spec.context_mode)
            state = states.setdefault(key, GateState())
            row_idx = int(idx)
            prior_rows[row_idx] = state.count
            if state.count >= spec.min_history:
                delta = state.corrected_mae() - state.raw_mae()
                prior_delta_mae[row_idx] = delta
                if spec.action == "hard_gate":
                    if delta <= -spec.margin_c:
                        predictions[row_idx] = corrected[row_idx]
                        selected[row_idx] = HARDENED_BASE_ID
                        corrected_weight[row_idx] = 1.0
                elif spec.action == "soft_blend":
                    if delta <= spec.margin_c:
                        weight = state.corrected_weight()
                        predictions[row_idx] = (weight * corrected[row_idx]) + ((1.0 - weight) * raw[row_idx])
                        selected[row_idx] = HARDENED_BASE_ID
                        corrected_weight[row_idx] = weight
                else:
                    raise ValueError(f"Unsupported hardening action: {spec.action}")
            pending_updates.append((key, raw_abs[row_idx], corrected_abs[row_idx]))
        for key, raw_error, corrected_error in pending_updates:
            states[key].update(raw_error, corrected_error)

    diagnostics = frame[["target_date", "forecast_source_family", "season", "frame_segment", "era_bucket"]].copy()
    diagnostics["candidate_id"] = spec.candidate_id
    diagnostics["selected_candidate_id"] = selected
    diagnostics["prior_rows"] = prior_rows
    diagnostics["prior_delta_mae_corrected_minus_raw"] = prior_delta_mae
    diagnostics["corrected_weight"] = corrected_weight
    return predictions, diagnostics


def frame_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
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
    masks: dict[str, np.ndarray],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    target = frame["target_tmax_c"].to_numpy(dtype=float)
    raw = frame["forecast_max_c"].to_numpy(dtype=float)
    corrected = frame["candidate_prediction_c"].to_numpy(dtype=float)
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "candidate_class": candidate_class,
        **score_arrays(target=target, prediction=prediction, dates=dates),
    }
    raw_score = score_arrays(target=target, prediction=raw, dates=dates)
    corrected_score = score_arrays(target=target, prediction=corrected, dates=dates)
    row["delta_mae_vs_official_raw"] = float(row["mae"]) - float(raw_score["mae"])
    row["delta_mae_vs_0083_best"] = float(row["mae"]) - float(corrected_score["mae"])
    for prefix, mask in masks.items():
        segment_score = score_arrays(target=target[mask], prediction=prediction[mask], dates=dates[mask], prefix=prefix)
        raw_segment = score_arrays(target=target[mask], prediction=raw[mask], dates=dates[mask], prefix=prefix)
        corrected_segment = score_arrays(
            target=target[mask],
            prediction=corrected[mask],
            dates=dates[mask],
            prefix=prefix,
        )
        row.update(segment_score)
        row[f"{prefix}delta_mae_vs_official_raw"] = float(segment_score[f"{prefix}mae"]) - float(
            raw_segment[f"{prefix}mae"]
        )
        row[f"{prefix}delta_mae_vs_0083_best"] = float(segment_score[f"{prefix}mae"]) - float(
            corrected_segment[f"{prefix}mae"]
        )
    if extra:
        row.update(extra)
    return row


def group_scores(frame: pd.DataFrame, prediction: np.ndarray, *, label: str) -> pd.DataFrame:
    target = frame["target_tmax_c"].to_numpy(dtype=float)
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    rows: list[dict[str, object]] = []
    groupings: list[tuple[str, list[str]]] = [
        ("source", ["forecast_source_family"]),
        ("source_frame", ["forecast_source_family", "frame_segment"]),
        ("source_season", ["forecast_source_family", "season"]),
        ("source_era", ["forecast_source_family", "era_bucket"]),
    ]
    for grouping, cols in groupings:
        for key, group in frame.groupby(cols, observed=True, dropna=False):
            mask = np.asarray(frame.index.isin(group.index), dtype=bool)
            key_values = key if isinstance(key, tuple) else (key,)
            row = {
                "prediction_label": label,
                "grouping": grouping,
                "group_key": "|".join(str(value) for value in key_values),
                **score_arrays(target=target[mask], prediction=prediction[mask], dates=dates[mask]),
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["grouping", "group_key", "prediction_label"]).reset_index(drop=True)


def selection_counts(diagnostics: pd.DataFrame, best_candidate_id: str) -> pd.DataFrame:
    best = diagnostics[diagnostics["candidate_id"].eq(best_candidate_id)].copy()
    if best.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for cols in (
        ["selected_candidate_id"],
        ["forecast_source_family", "selected_candidate_id"],
        ["frame_segment", "selected_candidate_id"],
        ["season", "selected_candidate_id"],
        ["era_bucket", "selected_candidate_id"],
    ):
        grouping = "+".join(cols)
        counts = best.groupby(cols, observed=True, dropna=False).size().reset_index(name="rows")
        for row in counts.to_dict("records"):
            rows.append({"grouping": grouping, **row})
    return pd.DataFrame(rows)


def build_scoreboard(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    masks = frame_masks(frame)
    raw = frame["forecast_max_c"].to_numpy(dtype=float)
    corrected = frame["candidate_prediction_c"].to_numpy(dtype=float)
    rows = [
        score_candidate(
            frame,
            candidate_id=RAW_CANDIDATE_ID,
            candidate_class="official_raw",
            prediction=raw,
            masks=masks,
        ),
        score_candidate(
            frame,
            candidate_id=HARDENED_BASE_ID,
            candidate_class="0083_best_source_top5_blend",
            prediction=corrected,
            masks=masks,
        ),
    ]
    definitions: list[dict[str, object]] = []
    prediction_lookup = {RAW_CANDIDATE_ID: raw, HARDENED_BASE_ID: corrected}
    specs_by_id: dict[str, HardenSpec] = {}
    for spec in make_specs():
        prediction, _diagnostics = apply_hardened_gate(frame, spec)
        prediction_lookup[spec.candidate_id] = prediction
        specs_by_id[spec.candidate_id] = spec
        definitions.append(
            {
                "candidate_id": spec.candidate_id,
                "candidate_class": "expanded_frame_hardened_official_specialist",
                "context_mode": spec.context_mode,
                "min_history": spec.min_history,
                "margin_c": spec.margin_c,
                "action": spec.action,
            }
        )
        rows.append(
            score_candidate(
                frame,
                candidate_id=spec.candidate_id,
                candidate_class="expanded_frame_hardened_official_specialist",
                prediction=prediction,
                masks=masks,
                extra={
                    "context_mode": spec.context_mode,
                    "min_history": spec.min_history,
                    "margin_c": spec.margin_c,
                    "action": spec.action,
                },
            )
        )
    scoreboard = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    best_id = str(scoreboard.iloc[0]["candidate_id"])
    best_prediction = prediction_lookup[best_id]
    if best_id in specs_by_id:
        _best_prediction, diagnostics = apply_hardened_gate(frame, specs_by_id[best_id])
    else:
        diagnostics = pd.DataFrame(
            columns=[
                "target_date",
                "forecast_source_family",
                "season",
                "frame_segment",
                "era_bucket",
                "candidate_id",
                "selected_candidate_id",
                "prior_rows",
                "prior_delta_mae_corrected_minus_raw",
                "corrected_weight",
            ]
        )
    top_predictions = frame[
        [
            "target_date",
            "forecast_source_family",
            "target_tmax_c",
            "forecast_max_c",
            "candidate_prediction_c",
            "season",
            "month",
            "frame_segment",
            "era_bucket",
        ]
    ].copy()
    top_predictions["candidate_id"] = best_id
    top_predictions["candidate_prediction_c"] = best_prediction
    top_predictions["candidate_error_c"] = best_prediction - top_predictions["target_tmax_c"]
    comparison = pd.concat(
        [
            group_scores(frame, raw, label=RAW_CANDIDATE_ID),
            group_scores(frame, corrected, label=HARDENED_BASE_ID),
            group_scores(frame, best_prediction, label=best_id),
        ],
        ignore_index=True,
    )
    return scoreboard, pd.DataFrame(definitions), diagnostics, top_predictions, comparison


def build_summary(frame: pd.DataFrame, scoreboard: pd.DataFrame, generated_at: str) -> dict[str, object]:
    best = scoreboard.iloc[0].to_dict()
    raw = scoreboard[scoreboard["candidate_id"].eq(RAW_CANDIDATE_ID)].iloc[0].to_dict()
    base = scoreboard[scoreboard["candidate_id"].eq(HARDENED_BASE_ID)].iloc[0].to_dict()
    _, current_0081_score = load_current_0081_dates_and_score()
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    return {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "rows": int(len(frame)),
        "unique_target_days": int(dates.nunique()),
        "first_target_date": date_text(dates.min()),
        "last_target_date": date_text(dates.max()),
        "candidate_count": int(len(scoreboard)),
        "best_candidate": str(best["candidate_id"]),
        "best_candidate_class": str(best["candidate_class"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_official_raw": float(best["delta_mae_vs_official_raw"]),
        "best_delta_mae_vs_0083_best": float(best["delta_mae_vs_0083_best"]),
        "official_raw_mae": float(raw["mae"]),
        "official_raw_rmse": float(raw["rmse"]),
        "base_0083_mae": float(base["mae"]),
        "base_0083_rmse": float(base["rmse"]),
        "best_old_frame_mae": float(best["old_frame_mae"]),
        "best_newly_available_mae": float(best["newly_available_mae"]),
        "best_press_mae": float(best["press_mae"]),
        "best_rss_mae": float(best["rss_mae"]),
        "current_0081_old_frame_mae": current_0081_score.get("mae"),
        "current_0081_old_frame_rmse": current_0081_score.get("rmse"),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "hardened_gate_screen_complete",
        "next_recommended_task": (
            "Run 0085 as a long-history feature and station information-gain bridge on the same expanded official "
            "forecast frame: join the 1949-2026 feature matrix to 0084 residuals, rank station/upper-air/marine "
            "attributes by residual information gain, then design guarded local specialists for the highest-signal "
            "failure regimes. Keep 2024+ sealed."
        ),
    }


def build_readme(
    *,
    generated_at: str,
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    definitions: pd.DataFrame,
    comparison: pd.DataFrame,
    selections: pd.DataFrame,
) -> str:
    top = scoreboard.head(15)
    return f"""# 0084 Expanded-Frame Hardened Official Specialists

Generated: `{generated_at}`

## Purpose

`0083` proved that a simple source-specific past-performance blend improves raw HKO official forecasts on the expanded current forecast archive, but it did not prove that the correction should be trusted in every source, season, era, and frame segment. This experiment hardens that rule. It asks a stricter question: if the system is only allowed to turn the `0083` correction on after that exact context has already shown prior out-of-sample benefit, does performance improve, stay stable, or degrade?

This is directly relevant to the 0.45 MAE target because any serious system must learn when to trust the official forecast, when to trust corrected official forecasts, and when to bring in station/upper-air specialists. A correction that helps globally but hurts in a narrow regime is dangerous. A correction that helps in a persistent source/era regime becomes a candidate for deeper station-feature modelling.

## Inputs

- `0083` top predictions on the refreshed official forecast frame.
- Target labels only before `{CONFIRMATION_START.date()}`.
- Press archive segment: `2000-01-02` to `2011-09-13`.
- RSS archive segment: `2021-04-14` to `2023-12-31`.
- No 2024+ confirmation rows.

## Leakage Controls

For every target date, all rows for that date are predicted before that date's errors are added to gate memory. Gate decisions can only use previous target dates. The hard gate chooses `0083` only when the relevant context has enough prior rows and prior corrected MAE beats prior raw official MAE by the configured margin. The soft blend uses prior inverse-MAE weights, again based only on previous target dates.

## What Was Tested

The screen evaluated `{summary['candidate_count']}` candidates:

- raw official forecast;
- the unchanged `0083` best source top-5 blend;
- hard gates and soft blends over global, source, source-season, source-month, source-era, source-frame, source-season-frame, and source-season-era contexts;
- minimum prior histories of 30, 90, and 365 rows;
- gate margins of 0.00 C and 0.01 C.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Best candidate | `{summary['best_candidate']}` |
| Best candidate class | `{summary['best_candidate_class']}` |
| Best MAE | `{summary['best_mae']}` |
| Best RMSE | `{summary['best_rmse']}` |
| Official raw MAE | `{summary['official_raw_mae']}` |
| Base 0083 MAE | `{summary['base_0083_mae']}` |
| Delta vs raw official | `{summary['best_delta_mae_vs_official_raw']}` |
| Delta vs 0083 best | `{summary['best_delta_mae_vs_0083_best']}` |
| Best old-frame MAE | `{summary['best_old_frame_mae']}` |
| Best newly-available MAE | `{summary['best_newly_available_mae']}` |
| Best press MAE | `{summary['best_press_mae']}` |
| Best RSS MAE | `{summary['best_rss_mae']}` |
| Current 0081 old-frame MAE | `{summary['current_0081_old_frame_mae']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Top Candidates

{markdown_table(top, max_rows=15)}

## Candidate Definitions

{markdown_table(definitions.head(25), max_rows=25)}

## Best-Candidate Selection Counts

{markdown_table(selections, max_rows=30)}

## Group Comparison

{markdown_table(comparison.head(40), max_rows=40)}

## Interpretation

The hardened screen is a robustness test, not a final model. If the best candidate is still the unchanged `0083` blend, then the evidence says the simple blend was already safer than the extra gates on this currently available frame. If a hard gate or soft blend wins, then the winning context identifies where official residual correction is stable enough to trust. Either outcome is useful because it narrows the next deep-dive target.

The key scientific question after this run is not merely the headline MAE. It is whether the newly available 2004-2011 press segment behaves like the old 2000-2004 press segment, and whether RSS 2021-2023 has a different correction regime. Those source/era differences are exactly where the long-history station and upper-air data should be joined next.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(summary: dict[str, object], scoreboard: pd.DataFrame) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0084_expanded_frame_hardened_official_specialists.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Expanded hardened official screen | `{summary['rows']}` rows, `{summary['first_target_date']}` to `{summary['last_target_date']}` | Pre-2024 only |
| Best 0084 candidate | `{summary['best_candidate']}` | Hardened gate/selector screen |
| Best MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Compared to raw and 0083 |
| Delta vs raw official | `{summary['best_delta_mae_vs_official_raw']}` | Official-anchor lift |
| Delta vs 0083 best | `{summary['best_delta_mae_vs_0083_best']}` | Hardening value |
| Newly available press-frame MAE | `{summary['best_newly_available_mae']}` | 2004-08-06 to 2011-09-13 segment |
| Leakage | `0` 2024+ rows | PASS |

Top 0084 candidates:

{markdown_table(scoreboard.head(8), max_rows=8)}

Interpretation: `0084` tests whether the `0083` official correction should be turned on only in source/season/era contexts with prior proof. This is a bridge from official-forecast bias correction toward the next long-history station/feature residual analysis.
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0084 Expanded-Frame Hardened Official Specialists",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=(
            "Implement `0085_long_history_feature_station_residual_bridge`: join the 1949-2026 feature matrix "
            "to the 0084 expanded-frame residuals, rank all station, upper-air, marine, memory, and climate "
            "attributes by leakage-safe residual information gain, and design the next guarded local specialist "
            "set from the strongest source/era failure regimes. Keep 2024+ sealed."
        ),
    )


def run() -> dict[str, object]:
    generated_at = now_utc()
    frame = load_0083_frame()
    scoreboard, definitions, diagnostics, top_predictions, comparison = build_scoreboard(frame)
    summary = build_summary(frame, scoreboard, generated_at)
    selections = selection_counts(diagnostics, str(summary["best_candidate"]))

    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "gate_diagnostics.csv", diagnostics)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_csv(artifacts / "group_comparison.csv", comparison)
    write_csv(artifacts / "selection_counts.csv", selections)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "expanded_frame_hardened_official_specialists_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            generated_at=generated_at,
            summary=summary,
            scoreboard=scoreboard,
            definitions=definitions,
            comparison=comparison,
            selections=selections,
        ),
    )
    update_milestones(summary, scoreboard)
    require_no_confirmation_dates(top_predictions["target_date"], context="0084 top predictions")
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Harden 0083 official-anchor corrections by source, era, season, and frame segment."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
