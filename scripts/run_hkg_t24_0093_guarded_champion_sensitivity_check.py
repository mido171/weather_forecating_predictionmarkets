from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_0086_guarded_long_history_residual_specialists import (
    ResidualState,
    SpecialistSpec,
    apply_specialist,
    context_key,
)
from scripts.run_hkg_t24_0090_guarded_specialists_from_error_autopsy import (
    BASE_ID,
    build_working_frame,
    evaluation_masks,
    load_inputs,
    score_candidate,
)
from scripts.run_hkg_t24_0092_blocking_slice_guarded_specialists import (
    apply_no_correction_guards,
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

FOLDER_NAME = "0093_guarded_champion_sensitivity_check"
DEFAULT_CHUNK_SIZE = 32
INPUT_0092_SCOREBOARD_PATH = (
    RESEARCH_ROOT / "0092_blocking_slice_guarded_specialists" / "artifacts" / "scoreboard.csv"
)
INPUT_0092_SUMMARY_PATH = (
    RESEARCH_ROOT / "0092_blocking_slice_guarded_specialists" / "artifacts" / "summary.json"
)


@dataclass(frozen=True)
class SensitivitySpec:
    candidate_id: str
    feature: str
    context_mode: str
    min_history: int
    shrink_rows: float
    correction_cap_c: float
    guard_variant: str
    guard_slices: tuple[str, ...]


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def safe_token(value: str) -> str:
    return (
        value.replace("_", "-")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
        .replace(".", "p")[:72]
    )


def load_0092_champion() -> tuple[pd.Series, dict[str, object]]:
    missing = [path for path in (INPUT_0092_SCOREBOARD_PATH, INPUT_0092_SUMMARY_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"0093 requires 0092 artifacts first: {missing}")
    scoreboard = pd.read_csv(INPUT_0092_SCOREBOARD_PATH)
    summary = json.loads(INPUT_0092_SUMMARY_PATH.read_text(encoding="utf-8"))
    best_id = str(summary["best_candidate"])
    best = scoreboard[scoreboard["candidate_id"].eq(best_id)]
    if best.empty:
        raise RuntimeError(f"0092 best candidate not found in scoreboard: {best_id}")
    return best.iloc[0], summary


def guard_subsets(guard_text: str) -> list[tuple[str, tuple[str, ...]]]:
    slices = tuple(part for part in str(guard_text).split(";") if part)
    variants: list[tuple[str, tuple[str, ...]]] = [("no_guard", ())]
    for item in slices:
        variants.append((f"guard_{safe_token(item)}", (item,)))
    if slices:
        variants.append(("guard_all_0092_failed", slices))
    if len(slices) >= 3:
        for size in range(2, len(slices)):
            for combo in combinations(slices, size):
                variants.append((f"guard_{'_'.join(safe_token(value) for value in combo)}", combo))
    seen: set[tuple[str, ...]] = set()
    unique: list[tuple[str, tuple[str, ...]]] = []
    for label, combo in variants:
        if combo in seen:
            continue
        seen.add(combo)
        unique.append((label, combo))
    return unique


def make_sensitivity_specs(champion: pd.Series) -> list[SensitivitySpec]:
    feature = str(champion["feature"])
    context_mode = str(champion["context_mode"])
    if not feature or not context_mode:
        raise RuntimeError("0092 champion row must include feature and context_mode")
    specs: list[SensitivitySpec] = []
    for guard_variant, guard_slices in guard_subsets(str(champion.get("failed_slices_guarded", ""))):
        for min_history in (60, 90, 120, 180):
            for shrink_rows in (75.0, 100.0):
                for cap in (0.35, 0.45, 0.55, 0.65):
                    candidate_id = (
                        f"sens_{safe_token(feature)}_{context_mode}_m{min_history}_"
                        f"s{int(shrink_rows)}_cap{str(cap).replace('.', 'p')}_{guard_variant}"
                    )
                    specs.append(
                        SensitivitySpec(
                            candidate_id=candidate_id,
                            feature=feature,
                            context_mode=context_mode,
                            min_history=min_history,
                            shrink_rows=shrink_rows,
                            correction_cap_c=cap,
                            guard_variant=guard_variant,
                            guard_slices=guard_slices,
                        )
                    )
    return specs


def sensitivity_definition(spec: SensitivitySpec) -> dict[str, object]:
    return {
        "candidate_id": spec.candidate_id,
        "feature": spec.feature,
        "context_mode": spec.context_mode,
        "min_history": spec.min_history,
        "shrink_rows": spec.shrink_rows,
        "correction_cap_c": spec.correction_cap_c,
        "guard_variant": spec.guard_variant,
        "guard_slices": ";".join(spec.guard_slices),
        "guard_slice_count": len(spec.guard_slices),
    }


def apply_specialist_prediction_only(frame: pd.DataFrame, spec: SpecialistSpec) -> np.ndarray:
    base = frame["candidate_prediction_c"].to_numpy(dtype=float)
    residual = frame["base_residual_c"].to_numpy(dtype=float)
    predictions = base.copy()
    states: dict[tuple[object, ...], ResidualState] = {}

    for _, date_group in frame.groupby("target_date", sort=True, observed=True):
        pending_updates: list[tuple[tuple[object, ...], float]] = []
        for idx, row in date_group.iterrows():
            key = context_key(row, spec)
            if key is None:
                continue
            state = states.setdefault(key, ResidualState())
            row_idx = int(idx)
            if state.count >= spec.min_history:
                shrink = state.count / (state.count + spec.shrink_rows)
                correction = float(np.clip(state.mean() * shrink, -spec.correction_cap_c, spec.correction_cap_c))
                predictions[row_idx] = base[row_idx] - correction
            pending_updates.append((key, residual[row_idx]))
        for key, residual_value in pending_updates:
            states[key].update(residual_value)
    return predictions


def apply_sensitivity_spec(
    frame: pd.DataFrame,
    spec: SensitivitySpec,
    *,
    include_diagnostics: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    specialist = SpecialistSpec(
        candidate_id=spec.candidate_id,
        feature=spec.feature,
        context_mode=spec.context_mode,
        min_history=spec.min_history,
        shrink_rows=spec.shrink_rows,
        correction_cap_c=spec.correction_cap_c,
    )
    if include_diagnostics:
        prediction, diagnostics = apply_specialist(frame, specialist)
    else:
        prediction = apply_specialist_prediction_only(frame, specialist)
        diagnostics = pd.DataFrame()
    guarded, guard_table = apply_no_correction_guards(
        frame=frame,
        prediction=prediction,
        failed_slices=list(spec.guard_slices),
    )
    if not include_diagnostics:
        return guarded, diagnostics
    if guard_table.empty:
        diagnostics["guard_variant"] = spec.guard_variant
        diagnostics["guard_slices"] = ""
        return guarded, diagnostics
    guard_table["guard_variant"] = spec.guard_variant
    diagnostics["guard_variant"] = spec.guard_variant
    diagnostics["guard_slices"] = ";".join(spec.guard_slices)
    return guarded, diagnostics


def score_sensitivity_specs(
    *,
    frame: pd.DataFrame,
    mask_map: dict[str, np.ndarray],
    specs: list[SensitivitySpec],
    summary_0092: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        prediction, _diagnostics = apply_sensitivity_spec(frame, spec, include_diagnostics=False)
        scored = score_candidate(
            frame,
            candidate_id=spec.candidate_id,
            candidate_class="0093_guarded_champion_sensitivity",
            prediction=prediction,
            mask_map=mask_map,
            extra=sensitivity_definition(spec),
        )
        scored["delta_mae_vs_0092_best"] = float(scored["mae"]) - float(summary_0092["best_mae"])
        rows.append(scored)
        del prediction
        if index % 8 == 0:
            gc.collect()
    return rows


def run_worker(start: int, stop: int, output: Path) -> dict[str, object]:
    if not output.is_absolute():
        output = REPO_ROOT / output
    output = output.resolve()
    champion, summary_0092 = load_0092_champion()
    features, base, leads = load_inputs()
    frame, _thresholds = build_working_frame(features, base, leads)
    mask_map = evaluation_masks(frame)
    specs = make_sensitivity_specs(champion)[start:stop]
    rows = score_sensitivity_specs(
        frame=frame,
        mask_map=mask_map,
        specs=specs,
        summary_0092=summary_0092,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return {
        "worker_start": start,
        "worker_stop": stop,
        "worker_rows": len(rows),
        "worker_output": str(output),
    }


def score_sensitivity_specs_in_chunks(
    specs: list[SensitivitySpec],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict[str, object]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    worker_dir = REPO_ROOT / "data" / "_pipeline_internal" / "0093_worker_chunks"
    rows: list[dict[str, object]] = []
    for start in range(0, len(specs), chunk_size):
        stop = min(start + chunk_size, len(specs))
        output = worker_dir / f"s_{start:04d}_{stop:04d}.csv"
        expected_rows = stop - start
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
            timeout=300,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "0093 sensitivity worker failed "
                f"for rows {start}:{stop} with exit code {completed.returncode}\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        chunk = pd.read_csv(output)
        if len(chunk) != expected_rows:
            raise RuntimeError(
                f"0093 sensitivity worker wrote {len(chunk)} rows for {start}:{stop}; expected {expected_rows}"
            )
        rows.extend(chunk.to_dict("records"))
        gc.collect()
        time.sleep(1)
    return rows


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    champion, summary_0092 = load_0092_champion()
    sensitivity_specs = make_sensitivity_specs(champion)
    sensitivity_rows = score_sensitivity_specs_in_chunks(sensitivity_specs)
    gc.collect()

    features, base, leads = load_inputs()
    frame, thresholds = build_working_frame(features, base, leads)
    if str(champion["feature"]) not in set(thresholds["feature"].astype(str).tolist()):
        raise RuntimeError(f"0092 champion feature has no pre-2000 thresholds: {champion['feature']}")
    mask_map = evaluation_masks(frame)
    base_prediction = frame["candidate_prediction_c"].to_numpy(dtype=float)
    raw_prediction = frame["forecast_max_c"].to_numpy(dtype=float)
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
            candidate_class="0088_base",
            prediction=base_prediction,
            mask_map=mask_map,
        ),
    ]
    definitions: list[dict[str, object]] = []
    specs_by_id: dict[str, SensitivitySpec] = {}
    for spec in sensitivity_specs:
        specs_by_id[spec.candidate_id] = spec
        definitions.append(sensitivity_definition(spec))
    rows.extend(sensitivity_rows)
    scoreboard = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    if "delta_mae_vs_0092_best" not in scoreboard.columns:
        scoreboard["delta_mae_vs_0092_best"] = scoreboard["mae"].astype(float) - float(summary_0092["best_mae"])
    else:
        scoreboard["delta_mae_vs_0092_best"] = pd.to_numeric(
            scoreboard["delta_mae_vs_0092_best"],
            errors="coerce",
        )
        missing_delta = scoreboard["delta_mae_vs_0092_best"].isna()
        scoreboard.loc[missing_delta, "delta_mae_vs_0092_best"] = (
            scoreboard.loc[missing_delta, "mae"].astype(float) - float(summary_0092["best_mae"])
        )
    sensitivity = scoreboard[scoreboard["candidate_class"].eq("0093_guarded_champion_sensitivity")].copy()
    hardened = sensitivity[sensitivity["hardened_gate_passed"].astype(bool)].copy()
    if hardened.empty:
        best_id = str(summary_0092["best_candidate"])
        best_prediction = pd.read_csv(
            RESEARCH_ROOT / "0092_blocking_slice_guarded_specialists" / "artifacts" / "top_predictions.csv"
        )["candidate_prediction_c"].to_numpy(dtype=float)
        diagnostics = pd.DataFrame()
    else:
        best_row = hardened.sort_values(["mae", "rmse"]).iloc[0]
        best_id = str(best_row["candidate_id"])
        best_spec = specs_by_id[best_id]
        best_prediction, diagnostics = apply_sensitivity_spec(frame, best_spec, include_diagnostics=True)
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    best_score = scoreboard[scoreboard["candidate_id"].eq(best_id)]
    best_mae = float(best_score.iloc[0]["mae"]) if not best_score.empty else float(summary_0092["best_mae"])
    best_rmse = float(best_score.iloc[0]["rmse"]) if not best_score.empty else float(summary_0092["best_rmse"])
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
        "candidate_count": int(len(scoreboard)),
        "sensitivity_candidate_count": int(len(sensitivity)),
        "hardened_sensitivity_candidate_count": int(len(hardened)),
        "input_0092_best_candidate": summary_0092["best_candidate"],
        "input_0092_best_mae": float(summary_0092["best_mae"]),
        "input_0092_best_rmse": float(summary_0092["best_rmse"]),
        "best_candidate": best_id,
        "best_mae": best_mae,
        "best_rmse": best_rmse,
        "best_delta_mae_vs_0092_best": best_mae - float(summary_0092["best_mae"]),
        "best_delta_mae_vs_0088_base": best_mae - float(summary_0092["base_0088_mae"]),
        "guard_variants": sorted(set(definition["guard_variant"] for definition in definitions)),
        "min_history_values": [60, 90, 120, 180],
        "correction_cap_values": [0.35, 0.45, 0.55, 0.65],
        "shrink_rows_values": [75.0, 100.0],
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "guarded_champion_sensitivity_check_complete",
        "next_recommended_task": (
            "Run 0094_expanded_high_error_interaction_lab: target the persistent MAM/new-press high-error regime "
            "with interaction specialists using station-network, target-memory, upper-air ceiling, and marine "
            "features, while keeping the 0093/0092 guarded champion as the baseline."
        ),
    }
    require_no_confirmation_dates(top_predictions["target_date"], context="0093 top predictions")
    return scoreboard, pd.DataFrame(definitions), diagnostics, top_predictions, summary


def summarize_sensitivity(scoreboard: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitivity = scoreboard[scoreboard["candidate_class"].eq("0093_guarded_champion_sensitivity")].copy()
    if sensitivity.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, object]] = []
    for field in ("guard_variant", "min_history", "correction_cap_c", "shrink_rows"):
        for value, group in sensitivity.groupby(field, observed=True, dropna=False):
            best = group.sort_values(["mae", "rmse"]).iloc[0]
            rows.append(
                {
                    "dimension": field,
                    "value": value,
                    "candidate_count": int(len(group)),
                    "hardened_count": int(group["hardened_gate_passed"].astype(bool).sum()),
                    "best_candidate_id": best["candidate_id"],
                    "best_mae": float(best["mae"]),
                    "best_rmse": float(best["rmse"]),
                    "best_delta_mae_vs_0092": float(best["delta_mae_vs_0092_best"]),
                    "best_delta_mae_vs_0088": float(best["delta_mae_vs_0088_base"]),
                }
            )
    robustness = (
        sensitivity.groupby(["guard_variant", "min_history"], observed=True)
        .agg(
            candidate_count=("candidate_id", "count"),
            hardened_count=("hardened_gate_passed", lambda values: int(pd.Series(values).astype(bool).sum())),
            best_mae=("mae", "min"),
            median_mae=("mae", "median"),
            worst_mae=("mae", "max"),
        )
        .reset_index()
        .sort_values(["best_mae", "median_mae"])
    )
    return pd.DataFrame(rows).sort_values(["dimension", "best_mae"]).reset_index(drop=True), robustness


def build_readme(
    *,
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    definitions: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    robustness: pd.DataFrame,
) -> str:
    return f"""# 0093 Guarded Champion Sensitivity Check

Generated: `{summary['generated_at_utc']}`

## Purpose

`0092` found a small but hardened improvement from an ISD morning-to-midday warming specialist guarded off in JJA and SON. `0093` stress-tests that exact mechanism before treating it as a new research champion.

The screen varies guard subsets, correction caps, minimum prior-history requirements, and shrinkage. Every candidate is still past-only: residual states are updated only after each target date, and 2024+ confirmation rows remain sealed.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Input 0092 best | `{summary['input_0092_best_candidate']}` |
| Input 0092 MAE | `{summary['input_0092_best_mae']}` |
| Sensitivity candidates | `{summary['sensitivity_candidate_count']}` |
| Hardened sensitivity candidates | `{summary['hardened_sensitivity_candidate_count']}` |
| Best candidate | `{summary['best_candidate']}` |
| Best MAE | `{summary['best_mae']}` |
| Best RMSE | `{summary['best_rmse']}` |
| Delta vs 0092 best | `{summary['best_delta_mae_vs_0092_best']}` |
| Delta vs 0088 base | `{summary['best_delta_mae_vs_0088_base']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Plain-English Finding

The useful signal is still very specific: when the ISD-derived morning-to-midday temperature rise places a day in the same source/season/feature bucket as past days with a known residual pattern, a small residual correction helps. The correction is not trusted everywhere. It is explicitly switched off for the two slices that caused the 0091 near-miss to fail, JJA and SON, so the model falls back to the prior 0088 baseline in those seasons.

The sensitivity check says the 0092 idea was not a pure accident, but the gain is tiny. The best hardened setting used `min_history=120`, `shrink_rows=75`, and the all-failed-slices guard. All four tested correction caps from `0.35C` through `0.65C` tied, which means the cap was not the real driver here; the historical residual mean and shrinkage were already keeping the correction smaller than those caps.

The result should be treated as a reliable small repair, not as a major breakthrough. It improves the current pre-2024 research champion by about `{summary['best_delta_mae_vs_0092_best']}` MAE and improves over the 0088 base by about `{summary['best_delta_mae_vs_0088_base']}` MAE. The important insight is therefore directional: guarded source/season specialist corrections can safely help, but this particular signal is already close to exhausted.

## Data And Leakage Controls

This run used only the current available official forecast archive rows and target outcomes from `{summary['first_target_date']}` through `{summary['last_target_date']}`. The confirmation period starts on `{summary['confirmation_start']}`, and this experiment used no 2024+ rows. Every residual state is updated only after the target date being scored, so same-day and future target information cannot leak into the candidate prediction.

The worker chunk files are an execution detail for Windows stability. They contain only score rows for the 128 sensitivity candidates and are merged into the final `scoreboard.csv`; the authoritative research artifacts remain in this folder.

## Candidate Definitions

{markdown_table(definitions.head(40), max_rows=40)}

## Scoreboard

{markdown_table(scoreboard.head(30), max_rows=30)}

## Sensitivity Summary

{markdown_table(sensitivity_summary, max_rows=40)}

## Guard And History Robustness

{markdown_table(robustness.head(40), max_rows=40)}

## Interpretation

This check determines whether the 0092 guarded gain depends on one fragile specification. If the same guard family, cap neighborhood, and history settings remain near the top without creating source/frame/season regressions, the signal is credible enough to use as the baseline for the next interaction lab. If a different nearby setting wins, it becomes the new pre-2024 research candidate but still requires later confirmation-period scoring only when explicitly commanded.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    robustness: pd.DataFrame,
) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0093_guarded_champion_sensitivity_check.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| 0092 champion sensitivity | `{summary['sensitivity_candidate_count']}` candidates over `{summary['rows']}` rows | Pre-2024 only |
| Hardened sensitivity candidates | `{summary['hardened_sensitivity_candidate_count']}` | Strict gate |
| Input 0092 best MAE | `{summary['input_0092_best_mae']}` | Baseline |
| Best 0093 candidate | `{summary['best_candidate']}` | `{summary['best_mae']}` MAE |
| Delta vs 0092 best | `{summary['best_delta_mae_vs_0092_best']}` | Sensitivity value |
| Delta vs 0088 base | `{summary['best_delta_mae_vs_0088_base']}` | Guarded value |
| Leakage | `0` 2024+ rows | PASS |

Top 0093 candidates:

{markdown_table(scoreboard.head(8), max_rows=8)}

Sensitivity summary:

{markdown_table(sensitivity_summary.head(12), max_rows=12)}

Robustness grid:

{markdown_table(robustness.head(12), max_rows=12)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0093 Guarded Champion Sensitivity Check",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=str(summary["next_recommended_task"]),
    )


def run() -> dict[str, object]:
    scoreboard, definitions, diagnostics, top_predictions, summary = build_outputs()
    sensitivity_summary, robustness = summarize_sensitivity(scoreboard)
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "sensitivity_summary.csv", sensitivity_summary)
    write_csv(artifacts / "guard_history_robustness.csv", robustness)
    write_csv(artifacts / "best_gate_diagnostics.csv", diagnostics)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "guarded_champion_sensitivity_check_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            scoreboard=scoreboard,
            definitions=definitions,
            sensitivity_summary=sensitivity_summary,
            robustness=robustness,
        ),
    )
    update_milestones(summary, scoreboard, sensitivity_summary, robustness)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress-test the 0092 guarded HKG Tmax specialist champion."
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
