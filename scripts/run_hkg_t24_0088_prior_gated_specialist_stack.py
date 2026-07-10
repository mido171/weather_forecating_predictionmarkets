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

FOLDER_NAME = "0088_prior_gated_specialist_stack"
INPUT_0086_TOP_PATH = (
    RESEARCH_ROOT / "0086_guarded_long_history_residual_specialists" / "artifacts" / "top_predictions.csv"
)
INPUT_0087_TOP_PATH = (
    RESEARCH_ROOT / "0087_long_history_signal_interaction_specialists" / "artifacts" / "top_predictions.csv"
)
INPUT_0087_SUMMARY_PATH = (
    RESEARCH_ROOT / "0087_long_history_signal_interaction_specialists" / "artifacts" / "summary.json"
)
EPSILON_MAE = 0.05


@dataclass(frozen=True)
class StackSpec:
    candidate_id: str
    mode: str
    context_mode: str
    min_history: int
    blend_top_k: int | None


@dataclass
class SelectorState:
    sum_abs_errors: np.ndarray
    count: int = 0

    def update(self, errors: np.ndarray) -> None:
        self.sum_abs_errors += errors
        self.count += 1

    def prior_mae(self) -> np.ndarray:
        if self.count <= 0:
            return np.full_like(self.sum_abs_errors, np.inf, dtype=float)
        return self.sum_abs_errors / self.count


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_stack_frame() -> tuple[pd.DataFrame, str]:
    missing = [path for path in (INPUT_0086_TOP_PATH, INPUT_0087_TOP_PATH, INPUT_0087_SUMMARY_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"0088 requires 0086 and 0087 artifacts first: {missing}")
    frame_0086 = pd.read_csv(INPUT_0086_TOP_PATH)
    frame_0087 = pd.read_csv(INPUT_0087_TOP_PATH)
    summary_0087 = json.loads(INPUT_0087_SUMMARY_PATH.read_text(encoding="utf-8"))
    for frame in (frame_0086, frame_0087):
        frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
        frame.dropna(subset=["target_date"], inplace=True)
        frame.drop(frame[frame["target_date"] >= CONFIRMATION_START].index, inplace=True)
        require_no_confirmation_dates(frame["target_date"], context="0088 stack input")
    keep = [
        "target_date",
        "forecast_source_family",
        "target_tmax_c",
        "forecast_max_c",
        "candidate_prediction_c",
        "season",
        "frame_segment",
        "era_bucket",
    ]
    left = frame_0086[keep].rename(columns={"candidate_prediction_c": "prediction_0086_c"})
    right = frame_0087[["target_date", "forecast_source_family", "candidate_prediction_c"]].rename(
        columns={"candidate_prediction_c": "prediction_0087_c"}
    )
    joined = left.merge(right, on=["target_date", "forecast_source_family"], how="inner")
    for col in ("target_tmax_c", "forecast_max_c", "prediction_0086_c", "prediction_0087_c"):
        joined[col] = pd.to_numeric(joined[col], errors="coerce")
    joined = joined[joined[["target_tmax_c", "forecast_max_c", "prediction_0086_c", "prediction_0087_c"]].notna().all(axis=1)].copy()
    joined["forecast_source_family"] = joined["forecast_source_family"].astype(str)
    joined["season"] = joined["season"].astype(str)
    joined["frame_segment"] = joined["frame_segment"].astype(str)
    joined["era_bucket"] = joined["era_bucket"].astype(str)
    best_0087 = str(summary_0087.get("best_candidate", "0087_interaction"))
    return joined.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True), best_0087


def make_specs() -> list[StackSpec]:
    specs: list[StackSpec] = []
    for mode in ("selector", "blend"):
        for context_mode in ("global", "source", "source_frame", "source_season"):
            for min_history in (60, 180):
                blend_top_k = 2 if mode == "blend" else None
                specs.append(
                    StackSpec(
                        candidate_id=f"stack_{mode}_{context_mode}_m{min_history}",
                        mode=mode,
                        context_mode=context_mode,
                        min_history=min_history,
                        blend_top_k=blend_top_k,
                    )
                )
    return specs


def context_key(row: pd.Series, mode: str) -> tuple[object, ...]:
    if mode == "global":
        return ("global",)
    if mode == "source":
        return (row["forecast_source_family"],)
    if mode == "source_frame":
        return (row["forecast_source_family"], row["frame_segment"])
    if mode == "source_season":
        return (row["forecast_source_family"], row["season"])
    raise ValueError(f"Unsupported 0088 context mode: {mode}")


def apply_stack(frame: pd.DataFrame, spec: StackSpec, candidate_ids: list[str], matrix: np.ndarray) -> tuple[np.ndarray, list[str]]:
    target = frame["target_tmax_c"].to_numpy(dtype=float)
    base_idx = candidate_ids.index("0086_base")
    predictions = matrix[:, base_idx].copy()
    selected = ["0086_base" for _ in range(len(frame))]
    states: dict[tuple[object, ...], SelectorState] = {}

    for _, date_group in frame.groupby("target_date", sort=True, observed=True):
        pending: list[tuple[tuple[object, ...], np.ndarray]] = []
        for idx, row in date_group.iterrows():
            key = context_key(row, spec.context_mode)
            state = states.setdefault(key, SelectorState(sum_abs_errors=np.zeros(len(candidate_ids), dtype=float)))
            row_idx = int(idx)
            if state.count >= spec.min_history:
                prior_mae = state.prior_mae()
                order = np.argsort(prior_mae)
                if spec.mode == "selector":
                    chosen = int(order[0])
                    predictions[row_idx] = matrix[row_idx, chosen]
                    selected[row_idx] = candidate_ids[chosen]
                elif spec.mode == "blend":
                    chosen = order[: int(spec.blend_top_k or 2)]
                    weights = 1.0 / (prior_mae[chosen] + EPSILON_MAE)
                    weights = weights / weights.sum()
                    predictions[row_idx] = float(np.dot(matrix[row_idx, chosen], weights))
                    selected[row_idx] = "blend:" + ";".join(candidate_ids[int(pos)] for pos in chosen)
                else:
                    raise ValueError(f"Unsupported stack mode: {spec.mode}")
            errors = np.abs(matrix[row_idx, :] - target[row_idx])
            pending.append((key, errors))
        for key, errors in pending:
            states[key].update(errors)
    return predictions, selected


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
    base = frame["prediction_0087_c"].to_numpy(dtype=float)
    raw = frame["forecast_max_c"].to_numpy(dtype=float)
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "candidate_class": candidate_class,
        **score_arrays(target=target, prediction=prediction, dates=dates),
    }
    base_score = score_arrays(target=target, prediction=base, dates=dates)
    raw_score = score_arrays(target=target, prediction=raw, dates=dates)
    row["delta_mae_vs_0087_base"] = float(row["mae"]) - float(base_score["mae"])
    row["delta_mae_vs_official_raw"] = float(row["mae"]) - float(raw_score["mae"])
    for prefix, mask in mask_map.items():
        score = score_arrays(target=target[mask], prediction=prediction[mask], dates=dates[mask], prefix=prefix)
        base_segment = score_arrays(target=target[mask], prediction=base[mask], dates=dates[mask], prefix=prefix)
        row.update(score)
        row[f"{prefix}delta_mae_vs_0087_base"] = float(score[f"{prefix}mae"]) - float(
            base_segment[f"{prefix}mae"]
        )
    if extra:
        row.update(extra)
    row["hardened_gate_passed"] = (
        float(row["delta_mae_vs_0087_base"]) < 0.0
        and float(row["old_frame_delta_mae_vs_0087_base"]) <= 0.0
        and float(row["newly_available_delta_mae_vs_0087_base"]) <= 0.0
        and float(row["press_delta_mae_vs_0087_base"]) <= 0.0
        and float(row["rss_delta_mae_vs_0087_base"]) <= 0.0
    )
    return row


def build_outputs(frame: pd.DataFrame, best_0087_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_ids = ["official_raw", "0086_base", best_0087_id]
    matrix = np.column_stack(
        [
            frame["forecast_max_c"].to_numpy(dtype=float),
            frame["prediction_0086_c"].to_numpy(dtype=float),
            frame["prediction_0087_c"].to_numpy(dtype=float),
        ]
    )
    mask_map = masks(frame)
    rows = [
        score_candidate(
            frame,
            candidate_id="official_raw",
            candidate_class="official_raw",
            prediction=matrix[:, 0],
            mask_map=mask_map,
        ),
        score_candidate(
            frame,
            candidate_id="0086_base",
            candidate_class="0086_base",
            prediction=matrix[:, 1],
            mask_map=mask_map,
        ),
        score_candidate(
            frame,
            candidate_id=best_0087_id,
            candidate_class="0087_base",
            prediction=matrix[:, 2],
            mask_map=mask_map,
        ),
    ]
    definitions: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {
        "official_raw": matrix[:, 0],
        "0086_base": matrix[:, 1],
        best_0087_id: matrix[:, 2],
    }
    selected_lookup: dict[str, list[str]] = {}
    for spec in make_specs():
        prediction, selected = apply_stack(frame, spec, candidate_ids, matrix)
        predictions[spec.candidate_id] = prediction
        selected_lookup[spec.candidate_id] = selected
        definitions.append(
            {
                "candidate_id": spec.candidate_id,
                "candidate_class": "prior_gated_specialist_stack",
                "mode": spec.mode,
                "context_mode": spec.context_mode,
                "min_history": spec.min_history,
                "blend_top_k": spec.blend_top_k or "",
            }
        )
        rows.append(
            score_candidate(
                frame,
                candidate_id=spec.candidate_id,
                candidate_class="prior_gated_specialist_stack",
                prediction=prediction,
                mask_map=mask_map,
                extra={
                    "mode": spec.mode,
                    "context_mode": spec.context_mode,
                    "min_history": spec.min_history,
                    "blend_top_k": spec.blend_top_k or "",
                },
            )
        )
    scoreboard = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    hardened = scoreboard[scoreboard["hardened_gate_passed"].astype(bool)].copy()
    if not hardened.empty:
        best_id = str(hardened.sort_values(["mae", "rmse"]).iloc[0]["candidate_id"])
    else:
        best_id = best_0087_id
    best_prediction = predictions[best_id]
    selection = pd.DataFrame(
        {
            "target_date": frame["target_date"],
            "forecast_source_family": frame["forecast_source_family"],
            "frame_segment": frame["frame_segment"],
            "candidate_id": best_id,
            "selected_candidate_id": selected_lookup.get(best_id, [best_id for _ in range(len(frame))]),
        }
    )
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
    top_predictions["candidate_error_c"] = best_prediction - top_predictions["target_tmax_c"]
    return scoreboard, pd.DataFrame(definitions), selection, top_predictions


def selection_counts(selection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cols in (
        ["selected_candidate_id"],
        ["forecast_source_family", "selected_candidate_id"],
        ["frame_segment", "selected_candidate_id"],
    ):
        grouping = "+".join(cols)
        grouped = selection.groupby(cols, observed=True, dropna=False).size().reset_index(name="rows")
        for row in grouped.to_dict("records"):
            rows.append({"grouping": grouping, **row})
    return pd.DataFrame(rows)


def build_summary(
    *,
    generated_at: str,
    frame: pd.DataFrame,
    best_0087_id: str,
    scoreboard: pd.DataFrame,
) -> dict[str, object]:
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    base = scoreboard[scoreboard["candidate_id"].eq(best_0087_id)].iloc[0].to_dict()
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
        "candidate_count": int(len(scoreboard)),
        "hardened_candidate_count": int(len(hardened)),
        "base_0087_candidate": best_0087_id,
        "base_0087_mae": float(base["mae"]),
        "base_0087_rmse": float(base["rmse"]),
        "best_candidate": str(best["candidate_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0087_base": float(best["delta_mae_vs_0087_base"]),
        "best_hardened_candidate": best_hardened.get("candidate_id", ""),
        "best_hardened_delta_mae_vs_0087_base": best_hardened.get("delta_mae_vs_0087_base"),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "prior_gated_specialist_stack_complete",
        "next_recommended_task": (
            "Run 0089 to inspect remaining high-error regimes of the 0088/0087 champion, then mine station-specific "
            "and source-era interactions targeted only at those failures. Keep 2024+ sealed."
        ),
    }


def build_readme(
    *,
    generated_at: str,
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    definitions: pd.DataFrame,
    counts: pd.DataFrame,
) -> str:
    return f"""# 0088 Prior-Gated Specialist Stack

Generated: `{generated_at}`

## Purpose

`0086` found a deployable-style single-feature specialist and `0087` found a hardened interaction specialist. `0088` tests whether a simple prior-only stack can route between raw official, `0086`, and `0087` predictions based only on past errors.

This is not a final model. It is a guardrail experiment: if prior performance can safely decide when to trust the interaction specialist, it may improve stability. If the unchanged `0087` specialist remains best, then the interaction rule is already more useful than the added router on the current frame.

## Inputs

- `0086` best top predictions.
- `0087` best top predictions.
- No 2024+ confirmation rows.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Candidate count | `{summary['candidate_count']}` |
| Hardened passing candidates | `{summary['hardened_candidate_count']}` |
| Base 0087 candidate | `{summary['base_0087_candidate']}` |
| Base 0087 MAE | `{summary['base_0087_mae']}` |
| Best candidate | `{summary['best_candidate']}` |
| Best MAE | `{summary['best_mae']}` |
| Best RMSE | `{summary['best_rmse']}` |
| Delta vs 0087 base | `{summary['best_delta_mae_vs_0087_base']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Candidate Definitions

{markdown_table(definitions, max_rows=30)}

## Top Candidates

{markdown_table(scoreboard.head(20), max_rows=20)}

## Best Selection Counts

{markdown_table(counts, max_rows=30)}

## Interpretation

The key acceptance check is not only full-frame MAE. A stack candidate only counts as hardened if it improves the full frame and does not worsen old-frame, newly available press-frame, press-source, or RSS-source slices versus the `0087` base.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(summary: dict[str, object], scoreboard: pd.DataFrame) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0088_prior_gated_specialist_stack.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Prior-gated specialist stack | `{summary['candidate_count']}` candidates over `{summary['rows']}` rows | Pre-2024 only |
| Hardened passing candidates | `{summary['hardened_candidate_count']}` | Source/frame-gated |
| Base 0087 MAE | `{summary['base_0087_mae']}` | Benchmark |
| Best candidate | `{summary['best_candidate']}` | `{summary['best_mae']}` MAE |
| Delta vs 0087 base | `{summary['best_delta_mae_vs_0087_base']}` | Stack value |
| Leakage | `0` 2024+ rows | PASS |

Top 0088 candidates:

{markdown_table(scoreboard.head(8), max_rows=8)}

Interpretation: `0088` tests whether prior-only routing adds value over the 0087 interaction champion.
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0088 Prior-Gated Specialist Stack",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=(
            "Implement `0089_remaining_error_regime_autopsy`: inspect the worst remaining 0088/0087 errors by "
            "source, era, season, station features, upper-air features, and forecast-range behavior, then design the "
            "next targeted specialist set. Keep 2024+ sealed."
        ),
    )


def run() -> dict[str, object]:
    generated_at = now_utc()
    frame, best_0087_id = load_stack_frame()
    scoreboard, definitions, selection, top_predictions = build_outputs(frame, best_0087_id)
    counts = selection_counts(selection)
    summary = build_summary(
        generated_at=generated_at,
        frame=frame,
        best_0087_id=best_0087_id,
        scoreboard=scoreboard,
    )
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "selection_trace.csv", selection)
    write_csv(artifacts / "selection_counts.csv", counts)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "prior_gated_specialist_stack_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            generated_at=generated_at,
            summary=summary,
            scoreboard=scoreboard,
            definitions=definitions,
            counts=counts,
        ),
    )
    update_milestones(summary, scoreboard)
    require_no_confirmation_dates(top_predictions["target_date"], context="0088 top predictions")
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Stack 0086 and 0087 HKG Tmax specialists with prior-only gates."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
