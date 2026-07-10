from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_cell_robustness_smooth_shrinkage import (  # noqa: E402
    ensure_calendar_columns,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    update_markdown_section,
)
from scripts.run_hkg_t24_nonlinear_local_residual_fusion_lab import (  # noqa: E402
    build_feature_frame,
)
from scripts.run_hkg_t24_online_residual_memory_halflife import (  # noqa: E402
    BASE_MATERIALITY_C,
    OnlineMemorySpec,
    context_diagnostics,
    leakage_audit,
    score_all_specs,
)
from scripts.run_hkg_t24_prior_calibrated_fusion_screen import score_prediction  # noqa: E402

FOLDER_NAME = "0075_online_residual_memory_refinement"


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def refined_online_memory_specs() -> list[OnlineMemorySpec]:
    specs: list[OnlineMemorySpec] = []
    for context_set in ("behavior", "seasonal_behavior", "all"):
        for halflife_rows in (20.0, 45.0, 90.0):
            for min_history in (10, 20):
                for correction_cap in (0.12, 0.20):
                    candidate_id = (
                        f"causal_onmem_refine_{context_set}_h{int(halflife_rows)}_"
                        f"n{min_history}_cap{correction_cap}_lift_weighted"
                    ).replace(".", "p")
                    specs.append(
                        OnlineMemorySpec(
                            candidate_id=candidate_id,
                            context_set=context_set,
                            min_history=min_history,
                            min_perf_history=min_history,
                            halflife_rows=halflife_rows,
                            support_shrink=60.0,
                            min_prior_lift_c=0.0,
                            correction_cap_c=correction_cap,
                            combine_mode="lift_weighted",
                            max_contexts=4,
                        )
                    )
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("0075 candidate IDs are not unique")
    return specs


def build_readme(
    *,
    summary: dict[str, Any],
    definitions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    diagnostics: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    deployable = scoreboard[scoreboard["deployable_gate_passed"].astype(bool)].copy()
    return f"""# Online Residual Memory Refinement

Generated: `{summary['generated_at_utc']}`

## Purpose

`0075` is a focused sensitivity pass around the `0074` winner. It keeps the same causal online residual-memory mechanism, but narrows the search to behavior, seasonal-behavior, and all-context families with shorter half-lives and smaller support thresholds.

## Data Contract

- Base prediction: `0069` best deployable prediction.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- No 2024+ rows are used.
- Each row reads prior context state only, then updates state after scoring.
- This is a deterministic residual-memory correction, not predictive ML training.

## Headline

| Item | Value |
|---|---:|
| Base 0069 MAE | {summary['base_0069_mae']} |
| 0074 champion MAE | {summary['base_0074_best_mae']} |
| Best 0075 candidate | {summary['best_candidate']} |
| Best 0075 MAE | {summary['best_mae']} |
| Best 0075 RMSE | {summary['best_rmse']} |
| Best delta vs 0069 | {summary['best_delta_mae_vs_0069']} |
| Best delta vs 0074 | {summary['best_delta_mae_vs_0074']} |
| Best fold max delta vs 0069 | {summary['best_fold_delta_max_vs_0069']} |
| Best deployable candidate | {summary['best_deployable_candidate']} |
| Best deployable MAE | {summary['best_deployable_mae']} |

## Interpretation

This run tests whether `0074` was under-tuned. It is promotion-eligible only if it clears the same material full-sample, fold, and late-window checks versus `0069`; the delta versus `0074` is reported separately so we do not confuse refinement with a new mechanism.

## Scoreboard

{markdown_table(scoreboard, max_rows=120)}

## Deployable Candidates

{markdown_table(deployable, max_rows=80)}

## Best-Candidate Context Diagnostics

{markdown_table(diagnostics, max_rows=120)}

## Candidate Definitions

{markdown_table(definitions, max_rows=80)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Files

- `artifacts/candidate_definitions.csv`
- `artifacts/scoreboard.csv`
- `artifacts/top_predictions.csv`
- `artifacts/best_context_diagnostics.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_online_residual_memory_refinement.py`:

- `{FOLDER_NAME}`: focused refinement around the `0074` causal online residual-memory winner.

| Metric | Value |
|---|---:|
| Base 0069 MAE | {summary['base_0069_mae']} |
| 0074 champion MAE | {summary['base_0074_best_mae']} |
| Best 0075 candidate | {summary['best_candidate']} |
| Best 0075 MAE | {summary['best_mae']} |
| Best delta vs 0069 | {summary['best_delta_mae_vs_0069']} |
| Best delta vs 0074 | {summary['best_delta_mae_vs_0074']} |
| Best deployable candidate | {summary['best_deployable_candidate']} |

Leakage contract: no 2024+ rows; all residual-memory states are prior-only at prediction time.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Online Residual Memory Refinement",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_online_residual_memory_refinement.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Input | `0069` base predictions plus refined `0074` online-memory settings | Tested |
| Common rows | `{summary['common_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Non-contiguous official archive |
| Candidate count | `{summary['candidate_count']}` | Tested |
| Base 0069 MAE / RMSE | `{summary['base_0069_mae']}` / `{summary['base_0069_rmse']}` | Baseline |
| 0074 champion MAE | `{summary['base_0074_best_mae']}` | Prior online-memory champion |
| Best 0075 candidate | `{summary['best_candidate']}` | Tested |
| Best 0075 MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best delta vs 0069 | `{summary['best_delta_mae_vs_0069']}` | Online-memory value |
| Best delta vs 0074 | `{summary['best_delta_mae_vs_0074']}` | Refinement value |
| Best fold max delta vs 0069 | `{summary['best_fold_delta_max_vs_0069']}` | Robustness check |
| Best late delta vs 0069 | `{summary['best_late_delta_mae_vs_0069']}` | Late-window check |
| Gate-passed deployable candidate | `{summary['best_deployable_candidate']}` | Requires material full, fold, and late improvement |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0075` checks whether the `0074` half-life memory result improves with shorter memory and smaller history thresholds.
"""
    update_markdown_section(
        path,
        heading="Online Residual Memory Refinement",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"59. Online residual-memory refinement screened `{summary['candidate_count']}` candidates; "
        f"best MAE is `{summary['best_mae']}` with delta vs 0074 of "
        f"`{summary['best_delta_mae_vs_0074']}`."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue while the forecast backfill runs: either harden `0075` with a no-regret ensemble against `0069`/`0074`, or switch to explicit station/upper-air residual specialists for the largest remaining `0075` error clusters.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0069 = build_feature_frame()
    frame = ensure_calendar_columns(frame).sort_values("target_date").reset_index(drop=True)
    require_no_confirmation_dates(frame["target_date"], context="0075 input frame")
    base_score = score_prediction(
        frame,
        pd.to_numeric(frame["base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    specs = refined_online_memory_specs()
    definitions = pd.DataFrame([spec.__dict__ for spec in specs])
    scoreboard, top_predictions = score_all_specs(frame, specs, base_0069_mae=float(base_score["mae"]))
    best = scoreboard.iloc[0]
    best_predictions = top_predictions[top_predictions["candidate_id"].eq(best["candidate_id"])].copy()
    diagnostics = context_diagnostics(frame, best_predictions)
    leakage = leakage_audit(frame, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0075 leakage audit failed: {failed}")

    summary_0074_path = RESEARCH_ROOT / "0074_online_residual_memory_halflife" / "artifacts" / "summary.json"
    summary_0074 = json.loads(summary_0074_path.read_text(encoding="utf-8"))
    base_0074_best_mae = float(summary_0074["best_mae"])
    deployable_pool = scoreboard[scoreboard["deployable_gate_passed"].astype(bool)].copy()
    deployable_pool = deployable_pool.sort_values(["mae", "fold_delta_max_vs_0069"]).reset_index(drop=True)
    best_deployable = deployable_pool.iloc[0] if not deployable_pool.empty else None

    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "common_rows": int(len(frame)),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "candidate_count": int(len(scoreboard)),
        "deployable_candidate_count": int(scoreboard["deployable_gate_passed"].astype(bool).sum()),
        "base_0069_candidate": str(summary_0069["best_deployable_candidate"]),
        "base_0069_mae": float(base_score["mae"]),
        "base_0069_rmse": float(base_score["rmse"]),
        "base_0074_best_mae": base_0074_best_mae,
        "best_candidate": str(best["candidate_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0069": float(best["delta_mae_vs_0069"]),
        "best_delta_mae_vs_0074": float(best["mae"]) - base_0074_best_mae,
        "best_late_delta_mae_vs_0069": float(best["late_delta_mae_vs_0069"]),
        "best_fold_delta_max_vs_0069": float(best["fold_delta_max_vs_0069"]),
        "best_active_rows": int(best["active_rows"]),
        "best_mean_abs_correction_c": float(best["mean_abs_correction_c"]),
        "best_deployable_candidate": str(best_deployable["candidate_id"]) if best_deployable is not None else "NONE",
        "best_deployable_mae": float(best_deployable["mae"]) if best_deployable is not None else None,
        "best_deployable_rmse": float(best_deployable["rmse"]) if best_deployable is not None else None,
        "best_deployable_delta_mae_vs_0069": (
            float(best_deployable["delta_mae_vs_0069"]) if best_deployable is not None else None
        ),
        "diagnostic_context_rows": int(len(diagnostics)),
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
        "uses_2024_plus_rows": False,
        "promotion_materiality_c": BASE_MATERIALITY_C,
    }

    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_csv(artifacts / "best_context_diagnostics.csv", diagnostics)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "online_residual_memory_refinement_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            definitions=definitions,
            scoreboard=scoreboard,
            diagnostics=diagnostics,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run focused online residual-memory refinement.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
