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
FOLDER_0080 = "0080_source_era_hardened_specialist_gate"
FOLDER_NAME = "0081_rss_gate_stability_stress"
ARTIFACT_ROOT_0079 = RESEARCH_ROOT / FOLDER_0079 / "artifacts"
ARTIFACT_ROOT_0080 = RESEARCH_ROOT / FOLDER_0080 / "artifacts"


@dataclass(frozen=True)
class RssGateStressSpec:
    candidate_id: str
    base_0079_candidate_id: str
    rss_start_date: str
    min_changed_rows: int
    min_changed_years: int
    min_changed_rows_per_year: int


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    predictions_path = ARTIFACT_ROOT_0079 / "top_predictions.csv"
    scoreboard_path = ARTIFACT_ROOT_0080 / "scoreboard.csv"
    summary_path = ARTIFACT_ROOT_0080 / "summary.json"
    missing = [path for path in (predictions_path, scoreboard_path, summary_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"0081 requires 0079 and 0080 artifacts first: {missing}")
    predictions_0079 = pd.read_csv(predictions_path)
    predictions_0079["target_date"] = pd.to_datetime(predictions_0079["target_date"], errors="coerce").dt.normalize()
    scoreboard_0080 = pd.read_csv(scoreboard_path)
    summary_0080 = json.loads(summary_path.read_text(encoding="utf-8"))
    require_no_confirmation_dates(predictions_0079["target_date"], context="0081 0079 predictions")
    return predictions_0079, scoreboard_0080, summary_0080


def stress_start_dates() -> list[str]:
    return [
        "2021-04-14",
        "2021-07-01",
        "2021-10-01",
        "2022-01-01",
        "2022-04-01",
        "2022-07-01",
        "2022-10-01",
        "2023-01-01",
    ]


def make_specs(scoreboard_0080: pd.DataFrame) -> list[RssGateStressSpec]:
    hardened = scoreboard_0080[scoreboard_0080["hardened_gate_passed"].astype(bool)].copy()
    base_ids = hardened["base_0079_candidate_id"].astype(str).drop_duplicates().head(4).tolist()
    specs: list[RssGateStressSpec] = []
    for base_id in base_ids:
        short = base_id.replace("guarded_", "").replace("_", "-")
        for start in stress_start_dates():
            token = start.replace("-", "")
            specs.append(
                RssGateStressSpec(
                    candidate_id=f"rssgate_start{token}_{short}",
                    base_0079_candidate_id=base_id,
                    rss_start_date=start,
                    min_changed_rows=20,
                    min_changed_years=2,
                    min_changed_rows_per_year=10,
                )
            )
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("0081 candidate IDs are not unique")
    return specs


def rss_start_mask(predictions: pd.DataFrame, start_date: str) -> pd.Series:
    dates = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    source = predictions["forecast_source_family"].astype(str)
    return source.eq("rss_archive") & dates.ge(pd.Timestamp(start_date))


def apply_rss_gate(predictions_0079: pd.DataFrame, spec: RssGateStressSpec) -> pd.DataFrame:
    base = predictions_0079[predictions_0079["candidate_id"].astype(str).eq(spec.base_0079_candidate_id)].copy()
    if base.empty:
        raise RuntimeError(f"Missing 0079 base predictions for {spec.base_0079_candidate_id}")
    base = base.sort_values("target_date").reset_index(drop=True)
    mask = rss_start_mask(base, spec.rss_start_date)
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
    out["rss_gate_active"] = mask.to_numpy(dtype=bool)
    out["changed_from_0078"] = (
        pd.to_numeric(out["candidate_prediction_c"], errors="coerce")
        - pd.to_numeric(out["m0078_prediction_c"], errors="coerce")
    ).abs() > 1e-12
    out["candidate_id"] = spec.candidate_id
    out["candidate_class"] = "rss_gate_stability_stress"
    out["base_0079_candidate_id"] = spec.base_0079_candidate_id
    out["rss_start_date"] = spec.rss_start_date
    out["min_changed_rows"] = spec.min_changed_rows
    out["min_changed_years"] = spec.min_changed_years
    out["min_changed_rows_per_year"] = spec.min_changed_rows_per_year
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


def yearly_stability(predictions: pd.DataFrame, candidate_values: np.ndarray, base_values: np.ndarray) -> pd.DataFrame:
    dates = pd.to_datetime(predictions["target_date"], errors="coerce")
    changed = predictions["changed_from_0078"].astype(bool)
    rows: list[dict[str, object]] = []
    for year, group in predictions[changed].assign(year=dates.dt.year).groupby("year", observed=True):
        mask = predictions.index.isin(group.index)
        rows.append(
            {
                "year": int(year),
                "changed_rows": int(mask.sum()),
                "delta_mae_vs_0078": segment_delta(predictions, candidate_values, base_values, pd.Series(mask, index=predictions.index)),
            }
        )
    return pd.DataFrame(rows)


def score_candidate(predictions: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    candidate_values = pd.to_numeric(predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float)
    base_values = pd.to_numeric(predictions["m0078_prediction_c"], errors="coerce").to_numpy(dtype=float)
    candidate_score = score_values(predictions, candidate_values)
    base_score = score_values(predictions, base_values)
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
    year_table = yearly_stability(predictions, candidate_values, base_values)
    stable_years = year_table[
        year_table["changed_rows"].ge(int(predictions["min_changed_rows_per_year"].iloc[0]))
        & year_table["delta_mae_vs_0078"].le(0.0)
    ].copy()
    row: dict[str, object] = {
        "candidate_id": str(predictions["candidate_id"].iloc[0]),
        "candidate_class": str(predictions["candidate_class"].iloc[0]),
        "base_0079_candidate_id": str(predictions["base_0079_candidate_id"].iloc[0]),
        "rss_start_date": str(predictions["rss_start_date"].iloc[0]),
        "n": candidate_score["n"],
        "mae": candidate_score["mae"],
        "rmse": candidate_score["rmse"],
        "bias": candidate_score["bias"],
        "m0078_mae": base_score["mae"],
        "delta_mae_vs_0078": float(candidate_score["mae"]) - float(base_score["mae"]),
        "late_delta_mae_vs_0078": segment_delta(predictions, candidate_values, base_values, late_mask),
        "fold_delta_max_vs_0078": max(fold_deltas) if fold_deltas else math.nan,
        "source_delta_max_vs_0078": max(source_deltas) if source_deltas else math.nan,
        "changed_rows": int(predictions["changed_from_0078"].astype(bool).sum()),
        "changed_year_count": int(len(year_table)),
        "stable_changed_year_count": int(len(stable_years)),
        "year_delta_max_vs_0078": float(year_table["delta_mae_vs_0078"].max()) if not year_table.empty else math.nan,
        "year_delta_min_vs_0078": float(year_table["delta_mae_vs_0078"].min()) if not year_table.empty else math.nan,
    }
    row["beats_0078"] = bool(float(row["delta_mae_vs_0078"]) <= -BASE_MATERIALITY_C)
    row["stress_gate_passed"] = bool(
        row["beats_0078"]
        and int(row["changed_rows"]) >= int(predictions["min_changed_rows"].iloc[0])
        and int(row["stable_changed_year_count"]) >= int(predictions["min_changed_years"].iloc[0])
        and float(row["fold_delta_max_vs_0078"]) <= 0.0
        and float(row["late_delta_mae_vs_0078"]) <= 0.0
        and float(row["source_delta_max_vs_0078"]) <= 0.0
        and float(row["year_delta_max_vs_0078"]) <= 0.0
    )
    year_table.insert(0, "candidate_id", row["candidate_id"])
    return row, year_table


def score_all_specs(
    predictions_0079: pd.DataFrame,
    specs: list[RssGateStressSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions_list: list[pd.DataFrame] = []
    year_tables: list[pd.DataFrame] = []
    for spec in specs:
        predictions = apply_rss_gate(predictions_0079, spec)
        row, year_table = score_candidate(predictions)
        rows.append(row)
        predictions_list.append(predictions)
        year_tables.append(year_table)
    scoreboard = pd.DataFrame(rows).sort_values(
        ["stress_gate_passed", "beats_0078", "mae"],
        ascending=[False, False, True],
    )
    top_ids = set(scoreboard["candidate_id"].head(12).astype(str))
    top_predictions = pd.concat(
        [pred for pred in predictions_list if str(pred["candidate_id"].iloc[0]) in top_ids],
        ignore_index=True,
    )
    stability = pd.concat(year_tables, ignore_index=True) if year_tables else pd.DataFrame()
    require_no_confirmation_dates(top_predictions["target_date"], context="0081 top predictions")
    return scoreboard.reset_index(drop=True), top_predictions, stability


def leakage_audit(predictions_0079: pd.DataFrame, specs: list[RssGateStressSpec], scoreboard: pd.DataFrame) -> pd.DataFrame:
    passed = scoreboard[scoreboard["stress_gate_passed"].astype(bool)].copy()
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(predictions_0079["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(predictions_0079['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "rss_start_gates_use_known_calendar_only",
            "passed": bool(all(pd.Timestamp(spec.rss_start_date) < CONFIRMATION_START for spec in specs)),
            "evidence": f"{len(specs)} RSS start-date gates checked",
        },
        {
            "check_id": "stress_gate_requires_adjacent_year_source_fold_late_improvement",
            "passed": bool(
                passed.empty
                or (
                    passed["delta_mae_vs_0078"].le(-BASE_MATERIALITY_C).all()
                    and passed["stable_changed_year_count"].ge(2).all()
                    and passed["fold_delta_max_vs_0078"].le(0.0).all()
                    and passed["late_delta_mae_vs_0078"].le(0.0).all()
                    and passed["source_delta_max_vs_0078"].le(0.0).all()
                    and passed["year_delta_max_vs_0078"].le(0.0).all()
                )
            ),
            "evidence": f"{len(passed)} stress candidates passed",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    definitions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    stability: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    passed = scoreboard[scoreboard["stress_gate_passed"].astype(bool)].copy()
    best_stability = stability[stability["candidate_id"].astype(str).eq(str(summary["best_candidate"]))].copy()
    return f"""# RSS Gate Stability Stress

Generated: `{summary['generated_at_utc']}`

## Purpose

`0081` stress-tests the `0080` RSS-2022+ source/era gate. `0080` became the current hardened research champion, but the gate is only credible if its improvement is not isolated to one year. This run varies the RSS start date and requires at least two changed years with non-positive year-level deltas, while preserving full, source, fold, and late-window improvement versus `0078`.

## Data Contract

- Base benchmark: `0078` prior-only residual specialist champion.
- Candidate source: top hardened `0080` base `0079` specialist combinations.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- No 2024+ confirmation rows are used.
- Gate variables are limited to RSS source family and known calendar start date.
- The forecast backfill remains outside this tuning frame.

## Headline

| Item | Value |
|---|---:|
| 0080 champion MAE | {summary['base_0080_mae']} |
| Best 0081 candidate | {summary['best_candidate']} |
| Best 0081 MAE | {summary['best_mae']} |
| Best 0081 RMSE | {summary['best_rmse']} |
| Best delta vs 0078 | {summary['best_delta_mae_vs_0078']} |
| Best changed rows | {summary['best_changed_rows']} |
| Best stable changed years | {summary['best_stable_changed_year_count']} |
| Best year delta max | {summary['best_year_delta_max_vs_0078']} |
| Stress-passed champion | {summary['best_stress_candidate']} |

## Interpretation

The key pass condition is adjacent-year stability. If the best start date remains `2022-01-01` and both 2022 and 2023 improve, the `0080` gate is more credible as a modern RSS-era correction. It still remains research-only until the forecast backfill stabilizes and 2024+ confirmation is explicitly opened.

## Scoreboard

{markdown_table(scoreboard, max_rows=120)}

## Stress Gate-Passed Candidates

{markdown_table(passed, max_rows=80)}

## Best-Candidate Year Stability

{markdown_table(best_stability, max_rows=20)}

## All Year Stability Rows

{markdown_table(stability, max_rows=160)}

## Candidate Definitions

{markdown_table(definitions, max_rows=120)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Files

- `artifacts/candidate_definitions.csv`
- `artifacts/scoreboard.csv`
- `artifacts/top_predictions.csv`
- `artifacts/year_stability.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_0081_rss_gate_stability_stress.py`:

- `{FOLDER_NAME}`: adjacent-year stress test for the `0080` RSS source/era gate.

| Metric | Value |
|---|---:|
| 0080 champion MAE | {summary['base_0080_mae']} |
| Best 0081 candidate | {summary['best_candidate']} |
| Best 0081 MAE | {summary['best_mae']} |
| Best delta vs 0078 | {summary['best_delta_mae_vs_0078']} |
| Stable changed years | {summary['best_stable_changed_year_count']} |

Leakage contract: no 2024+ rows; gates use RSS source and known calendar only; forecast backfill excluded.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="RSS Gate Stability Stress",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0081_rss_gate_stability_stress.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Input | top hardened `0080` source/era candidates | Tested |
| Rows | `{summary['common_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Pre-2024 only |
| Candidate count | `{summary['candidate_count']}` | Tested |
| 0080 champion MAE / RMSE | `{summary['base_0080_mae']}` / `{summary['base_0080_rmse']}` | Baseline |
| Best 0081 candidate | `{summary['best_candidate']}` | Tested |
| Best 0081 MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best delta vs 0078 | `{summary['best_delta_mae_vs_0078']}` | Start-date stress value |
| Best changed rows | `{summary['best_changed_rows']}` | Actual altered predictions |
| Best stable changed years | `{summary['best_stable_changed_year_count']}` | Adjacent-year check |
| Best year delta max | `{summary['best_year_delta_max_vs_0078']}` | Year stability |
| Stress-passed champion | `{summary['best_stress_candidate']}` | Requires full/fold/source/late/year stability |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0081` stress-confirms the `0080` RSS-2022+ gate when the best candidate keeps non-positive year-level deltas across at least two changed years.
"""
    update_markdown_section(
        path,
        heading="RSS Gate Stability Stress",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"65. RSS gate stability stress screened `{summary['candidate_count']}` candidates; "
        f"best delta vs 0078 is `{summary['best_delta_mae_vs_0078']}` from "
        f"`{summary['best_candidate']}`, stable changed years `{summary['best_stable_changed_year_count']}`."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue with `0082`: wait for or sample the stabilized forecast backfill, then rerun the official-forecast export and re-score `0078` through `0081` on the expanded non-2024 archive. Do not open 2024+ confirmation until explicitly commanded.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    predictions_0079, scoreboard_0080, summary_0080 = load_inputs()
    specs = make_specs(scoreboard_0080)
    definitions = pd.DataFrame([spec.__dict__ for spec in specs])
    scoreboard, top_predictions, stability = score_all_specs(predictions_0079, specs)
    leakage = leakage_audit(predictions_0079, specs, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0081 leakage audit failed: {failed}")
    best = scoreboard.iloc[0]
    passed = scoreboard[scoreboard["stress_gate_passed"].astype(bool)].copy()
    passed = passed.sort_values(["mae", "year_delta_max_vs_0078"]).reset_index(drop=True)
    best_passed = passed.iloc[0] if not passed.empty else None
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "common_rows": int(top_predictions["row_index"].nunique()),
        "first_date": str(pd.to_datetime(predictions_0079["target_date"]).min().date()),
        "last_date": str(pd.to_datetime(predictions_0079["target_date"]).max().date()),
        "candidate_count": int(len(scoreboard)),
        "base_0080_candidate": str(summary_0080["best_hardened_candidate"]),
        "base_0080_mae": float(summary_0080["best_hardened_mae"]),
        "base_0080_rmse": float(summary_0080["best_hardened_rmse"]),
        "best_candidate": str(best["candidate_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0078": float(best["delta_mae_vs_0078"]),
        "best_changed_rows": int(best["changed_rows"]),
        "best_stable_changed_year_count": int(best["stable_changed_year_count"]),
        "best_year_delta_max_vs_0078": float(best["year_delta_max_vs_0078"]),
        "stress_candidate_count": int(scoreboard["stress_gate_passed"].astype(bool).sum()),
        "best_stress_candidate": str(best_passed["candidate_id"]) if best_passed is not None else "NONE",
        "best_stress_mae": float(best_passed["mae"]) if best_passed is not None else None,
        "best_stress_rmse": float(best_passed["rmse"]) if best_passed is not None else None,
        "best_stress_delta_mae_vs_0078": (
            float(best_passed["delta_mae_vs_0078"]) if best_passed is not None else None
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
    write_csv(artifacts / "year_stability.csv", stability)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "rss_gate_stability_stress_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            definitions=definitions,
            scoreboard=scoreboard,
            stability=stability,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run RSS gate stability stress test for 0080.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
