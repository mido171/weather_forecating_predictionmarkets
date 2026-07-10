from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_0095_mam_error_direction_split_lab import (
    BASE_ID,
    DirectionSplitSpec,
    evaluation_masks_0095,
    safe_token,
    short_hash,
    spec_definition,
)
from scripts.run_hkg_t24_0097_stable_directional_cell_specialist import align_prediction, score_with_0094_delta
from scripts.run_hkg_t24_0098_source_submonth_stable_cell_specialist import (
    CANDIDATE_CLASS as INPUT_0098_CANDIDATE_CLASS,
    apply_source_submonth_specialist,
    load_inputs as load_0098_inputs,
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

FOLDER_NAME = "0099_mam_cell_policy_sensitivity"
INPUT_0098_SUMMARY_PATH = RESEARCH_ROOT / "0098_source_submonth_stable_cell_specialist" / "artifacts" / "summary.json"
INPUT_0098_TOP_PATH = RESEARCH_ROOT / "0098_source_submonth_stable_cell_specialist" / "artifacts" / "top_predictions.csv"
CANDIDATE_CLASS = "0099_mam_cell_policy_sensitivity"
BEST_0098_POLICY = "bucket_and_source_submonth"
MIN_HISTORY_GRID = (60, 80, 100)
DIRECTION_THRESHOLD_GRID = (0.05, 0.10, 0.15)
CORRECTION_CAP_GRID = (0.20, 0.25, 0.30)


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def float_token(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def make_sensitivity_specs(base_spec: DirectionSplitSpec, *, policy: str = BEST_0098_POLICY) -> list[DirectionSplitSpec]:
    specs: list[DirectionSplitSpec] = []
    for min_history in MIN_HISTORY_GRID:
        for threshold in DIRECTION_THRESHOLD_GRID:
            for cap in CORRECTION_CAP_GRID:
                candidate_id = (
                    f"mampol_{short_hash(base_spec.pair_name, policy, str(min_history), str(threshold), str(cap))}_"
                    f"m{min_history}_t{float_token(threshold)}_c{float_token(cap)}"
                )
                specs.append(
                    DirectionSplitSpec(
                        candidate_id=candidate_id,
                        pair_name=base_spec.pair_name,
                        feature_a=base_spec.feature_a,
                        feature_b=base_spec.feature_b,
                        group_a=base_spec.group_a,
                        group_b=base_spec.group_b,
                        active_gate=base_spec.active_gate,
                        direction_mode=base_spec.direction_mode,
                        min_history=min_history,
                        direction_threshold_c=threshold,
                        shrink_rows=base_spec.shrink_rows,
                        correction_cap_c=cap,
                    )
                )
    return specs


def summarize_sensitivity(candidates: pd.DataFrame, *, input_0095_mae: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_column in ("min_history", "direction_threshold_c", "correction_cap_c"):
        for value, group in candidates.groupby(group_column, observed=True):
            mae = pd.to_numeric(group["mae"], errors="coerce")
            rows.append(
                {
                    "group_dimension": group_column,
                    "group_value": value,
                    "candidate_count": int(len(group)),
                    "best_mae": float(mae.min()),
                    "median_mae": float(mae.median()),
                    "worst_mae": float(mae.max()),
                    "best_delta_vs_0095": float(mae.min() - input_0095_mae),
                    "median_delta_vs_0095": float(mae.median() - input_0095_mae),
                    "improves_vs_0095_count": int((mae < input_0095_mae).sum()),
                    "hardened_count": int(group["hardened_gate_passed"].astype(bool).sum()),
                    "mean_active_rows": float(pd.to_numeric(group["specialist_active_rows"], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["group_dimension", "best_mae"]).reset_index(drop=True)


def build_outputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    (
        frame,
        summary_0095,
        _summary_0096,
        _bucket_summary,
        _source_submonth_summary,
        top_0095,
        base_spec,
        stable_bucket_cells,
        damaging_bucket_cells,
        source_submonth_cells,
    ) = load_0098_inputs()
    missing = [path for path in (INPUT_0098_SUMMARY_PATH, INPUT_0098_TOP_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"0099 requires 0098 artifacts first: {missing}")
    summary_0098 = json.loads(INPUT_0098_SUMMARY_PATH.read_text(encoding="utf-8"))
    top_0098 = pd.read_csv(INPUT_0098_TOP_PATH)
    top_0098["target_date"] = pd.to_datetime(top_0098["target_date"], errors="coerce").dt.normalize()
    top_0098 = top_0098[top_0098["target_date"].notna() & (top_0098["target_date"] < CONFIRMATION_START)].copy()
    require_no_confirmation_dates(top_0098["target_date"], context="0099 0098 top predictions")

    mask_map = evaluation_masks_0095(frame)
    raw_prediction = frame["forecast_max_c"].to_numpy(dtype=float)
    base_prediction = frame["candidate_prediction_c"].to_numpy(dtype=float)
    input_0095_prediction = align_prediction(frame, top_0095, "candidate_prediction_c")
    input_0098_prediction = align_prediction(frame, top_0098, "candidate_prediction_c")
    rows = [
        score_with_0094_delta(
            frame,
            candidate_id="official_raw",
            candidate_class="official_raw",
            prediction=raw_prediction,
            mask_map=mask_map,
        ),
        score_with_0094_delta(
            frame,
            candidate_id=BASE_ID,
            candidate_class="0094_base",
            prediction=base_prediction,
            mask_map=mask_map,
        ),
        score_with_0094_delta(
            frame,
            candidate_id=str(summary_0095["best_candidate"]),
            candidate_class="0095_mam_error_direction_split",
            prediction=input_0095_prediction,
            mask_map=mask_map,
            extra={"source_experiment": "0095"},
        ),
        score_with_0094_delta(
            frame,
            candidate_id=str(summary_0098["best_0098_candidate"]),
            candidate_class=INPUT_0098_CANDIDATE_CLASS,
            prediction=input_0098_prediction,
            mask_map=mask_map,
            extra={"source_experiment": "0098", "cell_policy": summary_0098["best_0098_policy"]},
        ),
    ]
    predictions_by_id: dict[str, np.ndarray] = {
        str(summary_0095["best_candidate"]): input_0095_prediction,
        str(summary_0098["best_0098_candidate"]): input_0098_prediction,
    }
    diagnostics_by_id: dict[str, pd.DataFrame] = {}
    definitions: list[dict[str, object]] = []
    for spec in make_sensitivity_specs(base_spec, policy=BEST_0098_POLICY):
        prediction, diagnostics = apply_source_submonth_specialist(
            frame,
            spec,
            cell_policy=BEST_0098_POLICY,
            stable_bucket_cells=stable_bucket_cells,
            damaging_bucket_cells=damaging_bucket_cells,
            source_submonth_cells=source_submonth_cells,
            include_diagnostics=True,
        )
        predictions_by_id[spec.candidate_id] = prediction
        diagnostics_by_id[spec.candidate_id] = diagnostics
        definitions.append(
            {
                **spec_definition(spec),
                "candidate_id": spec.candidate_id,
                "candidate_class": CANDIDATE_CLASS,
                "cell_policy": BEST_0098_POLICY,
                "input_0098_best_candidate": summary_0098["best_0098_candidate"],
                "policy_token": safe_token(BEST_0098_POLICY),
            }
        )
        rows.append(
            score_with_0094_delta(
                frame,
                candidate_id=spec.candidate_id,
                candidate_class=CANDIDATE_CLASS,
                prediction=prediction,
                mask_map=mask_map,
                extra={
                    **spec_definition(spec),
                    "cell_policy": BEST_0098_POLICY,
                    "specialist_active_rows": int(diagnostics["specialist_active"].sum()),
                    "bucket_stable_allowed_rows": int(diagnostics["bucket_stable_allowed"].sum()),
                    "source_submonth_stable_allowed_rows": int(
                        diagnostics["source_submonth_stable_allowed"].sum()
                    ),
                    "cell_policy_allowed_rows": int(diagnostics["cell_policy_allowed"].sum()),
                    "bucket_damaging_blocked_rows": int(diagnostics["bucket_damaging_blocked"].sum()),
                    "source_submonth_damaging_blocked_rows": int(
                        diagnostics["source_submonth_damaging_blocked"].sum()
                    ),
                },
            )
        )

    scoreboard = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    candidates = scoreboard[scoreboard["candidate_class"].eq(CANDIDATE_CLASS)].copy()
    robustness = summarize_sensitivity(candidates, input_0095_mae=float(summary_0095["best_mae"]))
    best_0099 = candidates.sort_values(["mae", "rmse"]).iloc[0]
    eligible = scoreboard[
        scoreboard["candidate_class"].isin(["0095_mam_error_direction_split", INPUT_0098_CANDIDATE_CLASS, CANDIDATE_CLASS])
        & scoreboard["hardened_gate_passed"].astype(bool)
        & (pd.to_numeric(scoreboard["delta_mae_vs_0094_base"], errors="coerce") < 0.0)
    ].copy()
    if eligible.empty:
        best_row = scoreboard[scoreboard["candidate_id"].eq(BASE_ID)].iloc[0]
        best_prediction = base_prediction
    else:
        best_row = eligible.sort_values(["mae", "rmse"]).iloc[0]
        best_prediction = predictions_by_id.get(str(best_row["candidate_id"]), input_0098_prediction)

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
    top_predictions["candidate_id"] = str(best_row["candidate_id"])
    top_predictions["candidate_prediction_c"] = best_prediction
    top_predictions["candidate_error_c"] = top_predictions["candidate_prediction_c"] - top_predictions["target_tmax_c"]
    best_diagnostics = diagnostics_by_id.get(str(best_0099["candidate_id"]), pd.DataFrame())
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    candidate_mae = pd.to_numeric(candidates["mae"], errors="coerce")
    summary = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "rows": int(len(frame)),
        "first_target_date": dates.min().date().isoformat(),
        "last_target_date": dates.max().date().isoformat(),
        "input_0095_best_candidate": summary_0095["best_candidate"],
        "input_0095_best_mae": float(summary_0095["best_mae"]),
        "input_0095_best_rmse": float(summary_0095["best_rmse"]),
        "input_0098_best_candidate": summary_0098["best_0098_candidate"],
        "input_0098_best_mae": float(summary_0098["best_0098_mae"]),
        "input_0098_best_rmse": float(summary_0098["best_0098_rmse"]),
        "cell_policy": BEST_0098_POLICY,
        "candidate_count": int(len(candidates)),
        "min_history_grid": list(MIN_HISTORY_GRID),
        "direction_threshold_grid": list(DIRECTION_THRESHOLD_GRID),
        "correction_cap_grid": list(CORRECTION_CAP_GRID),
        "improves_vs_0095_count": int((candidate_mae < float(summary_0095["best_mae"])).sum()),
        "improves_vs_0098_count": int((candidate_mae < float(summary_0098["best_0098_mae"])).sum()),
        "hardened_candidate_count": int(candidates["hardened_gate_passed"].astype(bool).sum()),
        "best_0099_candidate": str(best_0099["candidate_id"]),
        "best_0099_mae": float(best_0099["mae"]),
        "best_0099_rmse": float(best_0099["rmse"]),
        "best_0099_delta_mae_vs_0095": float(best_0099["mae"]) - float(summary_0095["best_mae"]),
        "best_0099_delta_mae_vs_0098": float(best_0099["mae"]) - float(summary_0098["best_0098_mae"]),
        "best_0099_delta_mae_vs_0094_base": float(best_0099["delta_mae_vs_0094_base"]),
        "best_0099_min_history": int(best_0099["min_history"]),
        "best_0099_direction_threshold_c": float(best_0099["direction_threshold_c"]),
        "best_0099_correction_cap_c": float(best_0099["correction_cap_c"]),
        "best_0099_specialist_active_rows": int(best_0099["specialist_active_rows"]),
        "median_candidate_mae": float(candidate_mae.median()),
        "worst_candidate_mae": float(candidate_mae.max()),
        "best_candidate": str(best_row["candidate_id"]),
        "best_candidate_class": str(best_row["candidate_class"]),
        "best_mae": float(best_row["mae"]),
        "best_rmse": float(best_row["rmse"]),
        "best_delta_mae_vs_0094_base": float(best_row["delta_mae_vs_0094_base"]),
        "new_0099_champion": bool(str(best_row["candidate_class"]) == CANDIDATE_CLASS),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "mam_cell_policy_sensitivity_complete",
        "next_recommended_task": (
            "Run 0100_stable_mam_cell_feature_atlas: inside the confirmed MAM bucket/source/submonth agreement "
            "cells, rank all leakage-eligible long-history station, upper-air, marine, and target-memory features "
            "for residual separation and information gain with 2024+ rows still sealed."
        ),
    }
    require_no_confirmation_dates(top_predictions["target_date"], context="0099 top predictions")
    return scoreboard, pd.DataFrame(definitions), robustness, best_diagnostics, top_predictions, summary


def build_readme(
    *,
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    definitions: pd.DataFrame,
    robustness: pd.DataFrame,
    best_diagnostics: pd.DataFrame,
) -> str:
    return f"""# 0099 MAM Cell Policy Sensitivity

Generated: `{summary['generated_at_utc']}`

## Purpose

`0098` found a tiny new pre-2024 champion by requiring both the stable bucket-direction cell and the stable source/submonth/direction cell to agree before applying the MAM residual correction. `0099` checks whether that result survives nearby settings instead of being one fragile point.

The sensitivity grid keeps the same policy, same cells, and same point-in-time residual update rule, but varies:

- minimum prior rows: `{summary['min_history_grid']}`
- prior direction threshold: `{summary['direction_threshold_grid']}`
- correction cap: `{summary['correction_cap_grid']}`

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Input 0095 MAE | `{summary['input_0095_best_mae']}` |
| Input 0098 MAE | `{summary['input_0098_best_mae']}` |
| Candidate count | `{summary['candidate_count']}` |
| Improves vs 0095 count | `{summary['improves_vs_0095_count']}` |
| Improves vs 0098 count | `{summary['improves_vs_0098_count']}` |
| Hardened candidate count | `{summary['hardened_candidate_count']}` |
| Best 0099 candidate | `{summary['best_0099_candidate']}` |
| Best 0099 MAE | `{summary['best_0099_mae']}` |
| Best 0099 RMSE | `{summary['best_0099_rmse']}` |
| Best 0099 delta vs 0098 | `{summary['best_0099_delta_mae_vs_0098']}` |
| Best min history | `{summary['best_0099_min_history']}` |
| Best direction threshold C | `{summary['best_0099_direction_threshold_c']}` |
| Best correction cap C | `{summary['best_0099_correction_cap_c']}` |
| Best active rows | `{summary['best_0099_specialist_active_rows']}` |
| New 0099 champion | `{summary['new_0099_champion']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Plain-English Finding

This experiment asks whether the `0098` rule is stable when we make it a little stricter or looser. If many neighboring settings improve on `0095`, the idea is more credible. If only one exact setting works, the idea is fragile.

The result should be treated as a sensitivity screen, not final confirmation. The 2024+ period remains sealed.

## Scoreboard

{markdown_table(scoreboard.head(35), max_rows=35)}

## Candidate Definitions

{markdown_table(definitions.head(35), max_rows=35)}

## Robustness Summary

{markdown_table(robustness, max_rows=40)}

## Best Diagnostics Sample

{markdown_table(best_diagnostics[best_diagnostics['gate_active_row']].head(50), max_rows=50) if not best_diagnostics.empty else '_No diagnostics._'}

## Leakage Controls

All rows are before `{summary['confirmation_start']}`. The current row's target error is never used to choose a correction. Pair residual states are updated after the current target date is scored. The stable cells come from the documented `0096` audit and the best policy comes from documented `0098`; no 2024+ confirmation rows are opened.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(summary: dict[str, object], scoreboard: pd.DataFrame, robustness: pd.DataFrame) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0099_mam_cell_policy_sensitivity.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Sensitivity candidates | `{summary['candidate_count']}` | Bounded grid |
| Improves vs 0095 | `{summary['improves_vs_0095_count']}` | Robustness signal |
| Improves vs 0098 | `{summary['improves_vs_0098_count']}` | Incremental signal |
| Hardened candidates | `{summary['hardened_candidate_count']}` | No tracked slice regression |
| Best 0099 MAE | `{summary['best_0099_mae']}` | Candidate |
| Best 0099 RMSE | `{summary['best_0099_rmse']}` | Candidate |
| Delta vs 0098 | `{summary['best_0099_delta_mae_vs_0098']}` | Promotion value |
| Overall best after 0099 | `{summary['best_candidate']}` | `{summary['best_candidate_class']}` |
| Leakage | `0` 2024+ rows | PASS |

Top rows:

{markdown_table(scoreboard.head(10), max_rows=10)}

Robustness summary:

{markdown_table(robustness, max_rows=20)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0099 MAM Cell Policy Sensitivity",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=str(summary["next_recommended_task"]),
    )


def run() -> dict[str, object]:
    scoreboard, definitions, robustness, best_diagnostics, top_predictions, summary = build_outputs()
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "robustness_summary.csv", robustness)
    write_csv(artifacts / "best_gate_diagnostics.csv", best_diagnostics)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "mam_cell_policy_sensitivity_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            scoreboard=scoreboard,
            definitions=definitions,
            robustness=robustness,
            best_diagnostics=best_diagnostics,
        ),
    )
    update_milestones(summary, scoreboard, robustness)
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
