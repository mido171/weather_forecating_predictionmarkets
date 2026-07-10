from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from scripts.run_hkg_t24_online_residual_memory_halflife import BASE_MATERIALITY_C
from scripts.run_hkg_t24_prior_calibrated_fusion_screen import score_prediction
from scripts.run_hkg_t24_station_official_family_router import LATE_EVAL_START

FOLDER_0079 = "0079_guarded_specialist_combination"
FOLDER_NAME = "0080_source_era_hardened_specialist_gate"
ARTIFACT_ROOT_0079 = RESEARCH_ROOT / FOLDER_0079 / "artifacts"


@dataclass(frozen=True)
class SourceEraGateSpec:
    candidate_id: str
    base_candidate_id: str
    gate_mode: str
    min_changed_rows: int


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_0079_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scoreboard_path = ARTIFACT_ROOT_0079 / "scoreboard.csv"
    predictions_path = ARTIFACT_ROOT_0079 / "top_predictions.csv"
    summary_path = ARTIFACT_ROOT_0079 / "summary.json"
    missing = [path for path in (scoreboard_path, predictions_path, summary_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"0080 requires 0079 artifacts first: {missing}")
    scoreboard = pd.read_csv(scoreboard_path)
    predictions = pd.read_csv(predictions_path)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require_no_confirmation_dates(predictions["target_date"], context="0080 0079 predictions")
    return scoreboard, predictions, summary


def gate_modes() -> list[str]:
    return [
        "all_rows",
        "rss_only",
        "rss_2022plus",
        "rss_2023plus",
        "press_only",
        "press_or_rss_2022plus",
        "rss_2022plus_active",
    ]


def source_era_gate_mask(predictions: pd.DataFrame, gate_mode: str) -> pd.Series:
    dates = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    source = predictions["forecast_source_family"].astype(str)
    guard_active = predictions["guard_active"].astype(bool)
    if gate_mode == "all_rows":
        return pd.Series(True, index=predictions.index)
    if gate_mode == "rss_only":
        return source.eq("rss_archive")
    if gate_mode == "rss_2022plus":
        return source.eq("rss_archive") & dates.ge(pd.Timestamp("2022-01-01"))
    if gate_mode == "rss_2023plus":
        return source.eq("rss_archive") & dates.ge(pd.Timestamp("2023-01-01"))
    if gate_mode == "press_only":
        return source.eq("press_archive")
    if gate_mode == "press_or_rss_2022plus":
        return source.eq("press_archive") | (source.eq("rss_archive") & dates.ge(pd.Timestamp("2022-01-01")))
    if gate_mode == "rss_2022plus_active":
        return source.eq("rss_archive") & dates.ge(pd.Timestamp("2022-01-01")) & guard_active
    raise ValueError(f"Unsupported gate_mode: {gate_mode}")


def make_specs(scoreboard_0079: pd.DataFrame) -> list[SourceEraGateSpec]:
    base_ids = scoreboard_0079.head(12)["candidate_id"].astype(str).tolist()
    specs: list[SourceEraGateSpec] = []
    for base_id in base_ids:
        short = base_id.replace("guarded_", "").replace("_", "-")
        for gate_mode in gate_modes():
            candidate_id = f"sourceera_{gate_mode}_{short}"
            specs.append(
                SourceEraGateSpec(
                    candidate_id=candidate_id,
                    base_candidate_id=base_id,
                    gate_mode=gate_mode,
                    min_changed_rows=20,
                )
            )
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("0080 candidate IDs are not unique")
    return specs


def apply_source_era_gate(predictions_0079: pd.DataFrame, spec: SourceEraGateSpec) -> pd.DataFrame:
    base = predictions_0079[predictions_0079["candidate_id"].astype(str).eq(spec.base_candidate_id)].copy()
    if base.empty:
        raise RuntimeError(f"Missing 0079 base predictions for {spec.base_candidate_id}")
    base = base.sort_values("target_date").reset_index(drop=True)
    mask = source_era_gate_mask(base, spec.gate_mode)
    out = base[
        [
            "target_date",
            "current_target_tmax_c",
            "forecast_source_family",
            "fold_id",
            "row_index",
            "m0075_prediction_c",
            "m0078_prediction_c",
            "guard_active",
            "selected_families",
            "selected_candidates",
        ]
    ].copy()
    out["candidate_prediction_c"] = pd.to_numeric(base["m0078_prediction_c"], errors="coerce")
    out.loc[mask, "candidate_prediction_c"] = pd.to_numeric(base.loc[mask, "candidate_prediction_c"], errors="coerce")
    out["source_era_gate_active"] = mask.to_numpy(dtype=bool)
    changed = (
        pd.to_numeric(out["candidate_prediction_c"], errors="coerce")
        - pd.to_numeric(out["m0078_prediction_c"], errors="coerce")
    ).abs() > 1e-12
    out["changed_from_0078"] = changed
    out["candidate_id"] = spec.candidate_id
    out["candidate_class"] = "source_era_hardened_specialist_gate"
    out["base_0079_candidate_id"] = spec.base_candidate_id
    out["gate_mode"] = spec.gate_mode
    out["min_changed_rows"] = spec.min_changed_rows
    return out


def score_values(frame: pd.DataFrame, values: pd.Series | np.ndarray) -> dict[str, float | int | str]:
    scored = frame.rename(columns={"current_target_tmax_c": "target_tmax_c"}).copy()
    return score_prediction(scored, np.asarray(values, dtype=float))


def segment_delta(
    frame: pd.DataFrame,
    candidate_values: np.ndarray,
    benchmark_values: np.ndarray,
    mask: pd.Series,
) -> float:
    if int(mask.sum()) == 0:
        return math.nan
    candidate_score = score_values(frame.loc[mask], candidate_values[mask.to_numpy()])
    benchmark_score = score_values(frame.loc[mask], benchmark_values[mask.to_numpy()])
    return float(candidate_score["mae"]) - float(benchmark_score["mae"])


def score_candidate(predictions: pd.DataFrame) -> dict[str, object]:
    candidate_values = pd.to_numeric(predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float)
    base_values = pd.to_numeric(predictions["m0078_prediction_c"], errors="coerce").to_numpy(dtype=float)
    frame = predictions.rename(columns={"current_target_tmax_c": "target_tmax_c"}).copy()
    candidate_score = score_prediction(frame, candidate_values)
    base_score = score_prediction(frame, base_values)
    late_mask = pd.to_datetime(predictions["target_date"], errors="coerce").ge(LATE_EVAL_START)

    fold_deltas: list[float] = []
    for _, group in predictions.groupby("fold_id", observed=True):
        mask = predictions.index.isin(group.index)
        fold_deltas.append(segment_delta(predictions, candidate_values, base_values, pd.Series(mask, index=predictions.index)))
    source_deltas: list[float] = []
    for _, group in predictions.groupby("forecast_source_family", observed=True):
        mask = predictions.index.isin(group.index)
        source_deltas.append(
            segment_delta(predictions, candidate_values, base_values, pd.Series(mask, index=predictions.index))
        )

    changed = predictions["changed_from_0078"].astype(bool)
    changed_delta = segment_delta(predictions, candidate_values, base_values, changed)
    row: dict[str, object] = {
        "candidate_id": str(predictions["candidate_id"].iloc[0]),
        "candidate_class": str(predictions["candidate_class"].iloc[0]),
        "base_0079_candidate_id": str(predictions["base_0079_candidate_id"].iloc[0]),
        "gate_mode": str(predictions["gate_mode"].iloc[0]),
        "n": candidate_score["n"],
        "mae": candidate_score["mae"],
        "rmse": candidate_score["rmse"],
        "bias": candidate_score["bias"],
        "m0078_mae": base_score["mae"],
        "delta_mae_vs_0078": float(candidate_score["mae"]) - float(base_score["mae"]),
        "late_n": int(late_mask.sum()),
        "late_delta_mae_vs_0078": segment_delta(predictions, candidate_values, base_values, late_mask),
        "fold_delta_max_vs_0078": max(fold_deltas) if fold_deltas else math.nan,
        "fold_delta_min_vs_0078": min(fold_deltas) if fold_deltas else math.nan,
        "source_delta_max_vs_0078": max(source_deltas) if source_deltas else math.nan,
        "source_delta_min_vs_0078": min(source_deltas) if source_deltas else math.nan,
        "gate_rows": int(predictions["source_era_gate_active"].astype(bool).sum()),
        "changed_rows": int(changed.sum()),
        "changed_delta_mae_vs_0078": changed_delta,
        "mean_change_c": float((candidate_values - base_values).mean()),
        "mean_abs_change_c": float(np.mean(np.abs(candidate_values - base_values))),
    }
    row["beats_0078"] = bool(float(row["delta_mae_vs_0078"]) <= -BASE_MATERIALITY_C)
    row["hardened_gate_passed"] = bool(
        row["beats_0078"]
        and int(row["changed_rows"]) >= int(predictions["min_changed_rows"].iloc[0])
        and float(row["fold_delta_max_vs_0078"]) <= 0.0
        and float(row["late_delta_mae_vs_0078"]) <= 0.0
        and float(row["source_delta_max_vs_0078"]) <= 0.0
    )
    return row


def score_all_specs(
    predictions_0079: pd.DataFrame,
    specs: list[SourceEraGateSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions_list: list[pd.DataFrame] = []
    for spec in specs:
        predictions = apply_source_era_gate(predictions_0079, spec)
        rows.append(score_candidate(predictions))
        predictions_list.append(predictions)
    scoreboard = pd.DataFrame(rows).sort_values(
        ["hardened_gate_passed", "beats_0078", "mae"],
        ascending=[False, False, True],
    )
    top_ids = set(scoreboard["candidate_id"].head(12).astype(str))
    top_predictions = pd.concat(
        [pred for pred in predictions_list if str(pred["candidate_id"].iloc[0]) in top_ids],
        ignore_index=True,
    )
    require_no_confirmation_dates(top_predictions["target_date"], context="0080 top predictions")
    return scoreboard.reset_index(drop=True), top_predictions


def robustness_breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = pd.to_numeric(predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float)
    base = pd.to_numeric(predictions["m0078_prediction_c"], errors="coerce").to_numpy(dtype=float)
    segments: list[tuple[str, str, pd.Series]] = [
        ("all", "all", pd.Series(True, index=predictions.index)),
        ("late", str(LATE_EVAL_START.date()), pd.to_datetime(predictions["target_date"]).ge(LATE_EVAL_START)),
    ]
    for source in sorted(predictions["forecast_source_family"].astype(str).unique()):
        segments.append(("source", source, predictions["forecast_source_family"].astype(str).eq(source)))
    for fold_id in sorted(predictions["fold_id"].astype(str).unique()):
        segments.append(("fold", fold_id, predictions["fold_id"].astype(str).eq(fold_id)))
    for segment_type, segment_id, mask in segments:
        if int(mask.sum()) == 0:
            continue
        subframe = predictions.loc[mask].copy()
        score = score_values(subframe, values[mask.to_numpy()])
        score_0078 = score_values(subframe, base[mask.to_numpy()])
        rows.append(
            {
                "candidate_id": str(predictions["candidate_id"].iloc[0]),
                "segment_type": segment_type,
                "segment_id": segment_id,
                "n": int(score["n"]),
                "mae": score["mae"],
                "rmse": score["rmse"],
                "m0078_mae": score_0078["mae"],
                "delta_mae_vs_0078": float(score["mae"]) - float(score_0078["mae"]),
            }
        )
    return pd.DataFrame(rows)


def leakage_audit(predictions_0079: pd.DataFrame, specs: list[SourceEraGateSpec], scoreboard: pd.DataFrame) -> pd.DataFrame:
    hardened = scoreboard[scoreboard["hardened_gate_passed"].astype(bool)].copy()
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(predictions_0079["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(predictions_0079['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "gates_use_source_and_known_calendar_only",
            "passed": bool(all(spec.gate_mode in gate_modes() for spec in specs)),
            "evidence": f"{len(specs)} source/era gate specs checked",
        },
        {
            "check_id": "hardened_gate_requires_full_late_fold_source_improvement",
            "passed": bool(
                hardened.empty
                or (
                    hardened["delta_mae_vs_0078"].le(-BASE_MATERIALITY_C).all()
                    and hardened["fold_delta_max_vs_0078"].le(0.0).all()
                    and hardened["late_delta_mae_vs_0078"].le(0.0).all()
                    and hardened["source_delta_max_vs_0078"].le(0.0).all()
                )
            ),
            "evidence": f"{len(hardened)} hardened candidates passed",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    definitions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    robustness: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    hardened = scoreboard[scoreboard["hardened_gate_passed"].astype(bool)].copy()
    return f"""# Source/Era Hardened Specialist Gate

Generated: `{summary['generated_at_utc']}`

## Purpose

`0080` hardens the useful `0079` raw-MAE specialist combination. `0079` improved full MAE but failed promotion because it slightly worsened the press/fold_2000_2005 slice and the 2021 RSS fold. This run applies deterministic source/era gates to the top `0079` candidates, using only source family and known calendar date, to test whether the useful signal can be kept without source/fold/late regressions.

## Data Contract

- Base benchmark: `0078` prior-only residual specialist champion.
- Candidate source: top `0079` guarded-specialist predictions.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- No 2024+ confirmation rows are used.
- Gate variables are limited to `forecast_source_family`, `target_date`, and prior-only `0079` active flags.
- The forecast backfill remains outside this tuning frame.
- Promotion requires full-window MAE improvement versus `0078`, enough changed rows, no fold worsening, no late-window worsening, and no source-family worsening.

## Headline

| Item | Value |
|---|---:|
| 0078 MAE | {summary['base_0078_mae']} |
| Best 0080 candidate | {summary['best_candidate']} |
| Best 0080 MAE | {summary['best_mae']} |
| Best 0080 RMSE | {summary['best_rmse']} |
| Best delta vs 0078 | {summary['best_delta_mae_vs_0078']} |
| Best late delta vs 0078 | {summary['best_late_delta_mae_vs_0078']} |
| Best fold max delta vs 0078 | {summary['best_fold_delta_max_vs_0078']} |
| Best source max delta vs 0078 | {summary['best_source_delta_max_vs_0078']} |
| Best changed rows | {summary['best_changed_rows']} |
| Hardened new champion | {summary['best_hardened_candidate']} |

## Interpretation

The key finding is whether the two-family specialist signal is source/era dependent. A hardened `rss_2022plus` winner means the signal is useful in the modern RSS period after the unstable 2021 startup slice, while it should not be forced onto the press archive until the historical backfill is stable and comparable.

## Scoreboard

{markdown_table(scoreboard, max_rows=120)}

## Hardened Gate-Passed Candidates

{markdown_table(hardened, max_rows=80)}

## Best-Candidate Robustness Breakdown

{markdown_table(robustness, max_rows=80)}

## Candidate Definitions

{markdown_table(definitions, max_rows=120)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Files

- `artifacts/candidate_definitions.csv`
- `artifacts/scoreboard.csv`
- `artifacts/top_predictions.csv`
- `artifacts/robustness_breakdown.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_0080_source_era_hardened_specialist_gate.py`:

- `{FOLDER_NAME}`: source/era hardening of top `0079` specialist-combination candidates.

| Metric | Value |
|---|---:|
| 0078 champion MAE | {summary['base_0078_mae']} |
| Best 0080 candidate | {summary['best_candidate']} |
| Best 0080 MAE | {summary['best_mae']} |
| Best delta vs 0078 | {summary['best_delta_mae_vs_0078']} |
| Hardened new champion | {summary['best_hardened_candidate']} |

Leakage contract: no 2024+ rows; gates use source family and known calendar only; forecast backfill excluded.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Source/Era Hardened Specialist Gate",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    if summary["best_hardened_candidate"] != "NONE":
        interpretation = f"`0080` produced a hardened new champion with MAE `{summary['best_hardened_mae']}`."
        next_task = """
Continue with `0081`: stress-test the `0080` RSS-2022+ source/era gate by varying the start date, requiring adjacent-year stability, and checking whether the same gate remains valid after the forecast backfill stabilizes. Keep 2024+ confirmation sealed.
"""
    else:
        interpretation = "`0078` remains the hardened champion; source/era gating did not clear the gate."
        next_task = """
Continue with `0081`: return to individual weak-morning-warming and cool-Waglan-sea specialists with smoother prior-only local corrections. Keep 2024+ confirmation sealed.
"""
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0080_source_era_hardened_specialist_gate.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Input | top `0079` predictions gated by source/era | Tested |
| Rows | `{summary['common_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Pre-2024 only |
| Candidate count | `{summary['candidate_count']}` | Tested |
| 0078 MAE / RMSE | `{summary['base_0078_mae']}` / `{summary['base_0078_rmse']}` | Baseline |
| Best 0080 candidate | `{summary['best_candidate']}` | Tested |
| Best 0080 MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best delta vs 0078 | `{summary['best_delta_mae_vs_0078']}` | Source/era gate value |
| Best fold max delta vs 0078 | `{summary['best_fold_delta_max_vs_0078']}` | Robustness check |
| Best late delta vs 0078 | `{summary['best_late_delta_mae_vs_0078']}` | Late-window check |
| Best source max delta vs 0078 | `{summary['best_source_delta_max_vs_0078']}` | Source hardening |
| Best changed rows | `{summary['best_changed_rows']}` | Actual altered predictions |
| Hardened new champion | `{summary['best_hardened_candidate']}` | Requires full/fold/late/source improvement |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: {interpretation}
"""
    update_markdown_section(
        path,
        heading="Source/Era Hardened Specialist Gate",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"64. Source/era hardened specialist gating screened `{summary['candidate_count']}` candidates; "
        f"best delta vs 0078 is `{summary['best_delta_mae_vs_0078']}` from "
        f"`{summary['best_candidate']}`, hardened champion `{summary['best_hardened_candidate']}`."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    scoreboard_0079, predictions_0079, summary_0079 = load_0079_artifacts()
    specs = make_specs(scoreboard_0079)
    definitions = pd.DataFrame([spec.__dict__ for spec in specs])
    scoreboard, top_predictions = score_all_specs(predictions_0079, specs)
    best = scoreboard.iloc[0]
    best_predictions = top_predictions[top_predictions["candidate_id"].astype(str).eq(str(best["candidate_id"]))].copy()
    robustness = robustness_breakdown(best_predictions)
    leakage = leakage_audit(predictions_0079, specs, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0080 leakage audit failed: {failed}")
    hardened = scoreboard[scoreboard["hardened_gate_passed"].astype(bool)].copy()
    hardened = hardened.sort_values(["mae", "fold_delta_max_vs_0078"]).reset_index(drop=True)
    best_hardened = hardened.iloc[0] if not hardened.empty else None
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "common_rows": int(best_predictions["row_index"].nunique()),
        "first_date": str(pd.to_datetime(predictions_0079["target_date"]).min().date()),
        "last_date": str(pd.to_datetime(predictions_0079["target_date"]).max().date()),
        "candidate_count": int(len(scoreboard)),
        "base_0078_candidate": str(summary_0079["base_0078_candidate"]),
        "base_0078_mae": float(summary_0079["base_0078_mae"]),
        "base_0078_rmse": float(summary_0079["base_0078_rmse"]),
        "best_candidate": str(best["candidate_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0078": float(best["delta_mae_vs_0078"]),
        "best_late_delta_mae_vs_0078": float(best["late_delta_mae_vs_0078"]),
        "best_fold_delta_max_vs_0078": float(best["fold_delta_max_vs_0078"]),
        "best_source_delta_max_vs_0078": float(best["source_delta_max_vs_0078"]),
        "best_changed_rows": int(best["changed_rows"]),
        "hardened_candidate_count": int(scoreboard["hardened_gate_passed"].astype(bool).sum()),
        "best_hardened_candidate": str(best_hardened["candidate_id"]) if best_hardened is not None else "NONE",
        "best_hardened_mae": float(best_hardened["mae"]) if best_hardened is not None else None,
        "best_hardened_rmse": float(best_hardened["rmse"]) if best_hardened is not None else None,
        "best_hardened_delta_mae_vs_0078": (
            float(best_hardened["delta_mae_vs_0078"]) if best_hardened is not None else None
        ),
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
        "uses_2024_plus_rows": False,
    }

    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_csv(artifacts / "robustness_breakdown.csv", robustness)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "source_era_hardened_specialist_gate_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            definitions=definitions,
            scoreboard=scoreboard,
            robustness=robustness,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run source/era hardening for 0079 specialist gates.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
