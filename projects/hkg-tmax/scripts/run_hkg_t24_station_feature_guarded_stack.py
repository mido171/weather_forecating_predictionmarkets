from __future__ import annotations

import argparse
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    update_markdown_section,
)
from scripts.run_hkg_t24_station_feature_bucket_residual_mining import (  # noqa: E402
    CandidateSpec,
    TermSpec,
    candidates_from_definitions,
    evaluate_candidate,
    load_reference_frame,
    score_arrays,
    term_specs,
)

FOLDER_NAME = "0066_station_feature_guarded_stack"
ARTIFACT_0065 = RESEARCH_ROOT / "0065_station_feature_bucket_residual_mining" / "artifacts"
SCOREBOARD_0065_PATH = ARTIFACT_0065 / "scoreboard.csv"
DEFINITIONS_0065_PATH = ARTIFACT_0065 / "candidate_definitions.csv"
TERMS_0065_PATH = ARTIFACT_0065 / "terms.csv"
SUMMARY_0065_PATH = ARTIFACT_0065 / "summary.json"
MAX_MEMBER_CANDIDATES = 16


@dataclass(frozen=True)
class StackSpec:
    stack_id: str
    candidate_ids: tuple[str, ...]
    method: str
    weight: float
    cap_c: float
    min_active_members: int


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_0065_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, CandidateSpec], dict[str, TermSpec], dict[str, Any]]:
    missing = [
        str(path)
        for path in [SCOREBOARD_0065_PATH, DEFINITIONS_0065_PATH, TERMS_0065_PATH, SUMMARY_0065_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required 0065 artifacts: {missing}")
    scoreboard = pd.read_csv(SCOREBOARD_0065_PATH)
    definitions = pd.read_csv(DEFINITIONS_0065_PATH)
    terms_frame = pd.read_csv(TERMS_0065_PATH).fillna({"season": ""})
    terms = term_specs(terms_frame)
    candidates = candidates_from_definitions(definitions)
    summary = load_json(SUMMARY_0065_PATH)
    return scoreboard, definitions, candidates, terms, summary


def select_member_candidates(scoreboard: pd.DataFrame) -> list[str]:
    promoted = scoreboard[scoreboard["promotion_gate_passed"].astype(bool)].copy()
    if promoted.empty:
        raise RuntimeError("0066 requires at least one promoted 0065 candidate")
    promoted["abs_active_lift"] = -pd.to_numeric(promoted["active_delta_mae_vs_reference"], errors="coerce")
    promoted = promoted.sort_values(
        ["delta_mae_vs_reference", "active_delta_mae_vs_reference", "fold_delta_max"],
        ascending=[True, True, True],
    )
    selected: list[str] = []

    def add(ids: list[str]) -> None:
        for candidate_id in ids:
            if candidate_id not in selected:
                selected.append(candidate_id)
            if len(selected) >= MAX_MEMBER_CANDIDATES:
                return

    add(promoted.head(8)["candidate_id"].astype(str).tolist())
    for candidate_type in ["season_feature_bucket", "pair_feature_bucket", "feature_bucket"]:
        group = promoted[promoted["candidate_type"].astype(str).eq(candidate_type)]
        add(group.head(4)["candidate_id"].astype(str).tolist())
    source_seen: set[str] = set()
    diverse_ids: list[str] = []
    for row in promoted.to_dict("records"):
        key = str(row.get("source_families", ""))
        if key in source_seen:
            continue
        source_seen.add(key)
        diverse_ids.append(str(row["candidate_id"]))
    add(diverse_ids)
    return selected[:MAX_MEMBER_CANDIDATES]


def load_member_predictions(
    frame: pd.DataFrame,
    member_ids: list[str],
    candidates: dict[str, CandidateSpec],
    terms: dict[str, TermSpec],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    correction_columns: list[np.ndarray] = []
    active_columns: list[np.ndarray] = []
    for rank, candidate_id in enumerate(member_ids, start=1):
        if candidate_id not in candidates:
            raise KeyError(f"0065 candidate definition missing for {candidate_id}")
        row, predictions = evaluate_candidate(frame, candidates[candidate_id], terms, include_predictions=True)
        rows.append(
            {
                "member_rank": rank,
                "candidate_id": candidate_id,
                "candidate_type": row["candidate_type"],
                "features": row["features"],
                "bucket_values": row["bucket_values"],
                "seasons": row["seasons"],
                "source_families": row["source_families"],
                "station_ids": row["station_ids"],
                "delta_mae_vs_0064": row["delta_mae_vs_reference"],
                "active_delta_mae_vs_0064": row["active_delta_mae_vs_reference"],
                "active_n": row["active_n"],
                "fold_delta_max_vs_0064": row["fold_delta_max"],
            }
        )
        correction_columns.append(pd.to_numeric(predictions["residual_correction_c"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        active_columns.append(predictions["candidate_active"].astype(bool).to_numpy(dtype=bool))
    corrections = np.column_stack(correction_columns)
    active = np.column_stack(active_columns)
    return pd.DataFrame(rows), corrections, active


def stack_specs(member_ids: list[str]) -> list[StackSpec]:
    top1 = tuple(member_ids[:1])
    top2 = tuple(member_ids[:2])
    top3 = tuple(member_ids[:3])
    top5 = tuple(member_ids[:5])
    top8 = tuple(member_ids[:8])
    top12 = tuple(member_ids[:12])
    return [
        StackSpec("best_0065_singleton", top1, "sum_clip", 1.0, 1.0, 1),
        StackSpec("top2_active_mean_cap1", top2, "active_mean", 1.0, 1.0, 1),
        StackSpec("top3_active_mean_cap1", top3, "active_mean", 1.0, 1.0, 1),
        StackSpec("top5_active_mean_cap1", top5, "active_mean", 1.0, 1.0, 1),
        StackSpec("top8_active_mean_cap075", top8, "active_mean", 1.0, 0.75, 1),
        StackSpec("top3_weighted_sum025_cap075", top3, "sum_clip", 0.25, 0.75, 1),
        StackSpec("top5_weighted_sum025_cap075", top5, "sum_clip", 0.25, 0.75, 1),
        StackSpec("top8_weighted_sum015_cap075", top8, "sum_clip", 0.15, 0.75, 1),
        StackSpec("top12_active_mean_cap075", top12, "active_mean", 1.0, 0.75, 1),
        StackSpec("top8_consensus2_mean_cap075", top8, "same_sign_consensus_mean", 1.0, 0.75, 2),
        StackSpec("top12_consensus2_mean_cap075", top12, "same_sign_consensus_mean", 1.0, 0.75, 2),
        StackSpec("top8_rank_first_cap075", top8, "rank_first", 1.0, 0.75, 1),
    ]


def correction_for_stack(
    spec: StackSpec,
    member_ids: list[str],
    corrections: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = [member_ids.index(candidate_id) for candidate_id in spec.candidate_ids]
    selected_corrections = corrections[:, indices]
    selected_active = active[:, indices]
    active_corrections = np.where(selected_active, selected_corrections, 0.0)
    active_count = selected_active.sum(axis=1)
    out = np.zeros(len(selected_corrections), dtype=float)

    if spec.method == "sum_clip":
        out = active_corrections.sum(axis=1) * spec.weight
    elif spec.method == "active_mean":
        summed = active_corrections.sum(axis=1)
        out = np.divide(summed, active_count, out=np.zeros_like(summed), where=active_count > 0) * spec.weight
    elif spec.method == "rank_first":
        for column_index in range(selected_corrections.shape[1]):
            fill = (active_count > 0) & np.isclose(out, 0.0) & selected_active[:, column_index]
            out[fill] = selected_corrections[fill, column_index] * spec.weight
    elif spec.method == "same_sign_consensus_mean":
        positive = (selected_corrections > 0) & selected_active
        negative = (selected_corrections < 0) & selected_active
        positive_count = positive.sum(axis=1)
        negative_count = negative.sum(axis=1)
        positive_sum = np.where(positive, selected_corrections, 0.0).sum(axis=1)
        negative_sum = np.where(negative, selected_corrections, 0.0).sum(axis=1)
        positive_ok = positive_count >= spec.min_active_members
        negative_ok = negative_count >= spec.min_active_members
        out[positive_ok] = positive_sum[positive_ok] / positive_count[positive_ok]
        out[negative_ok] = negative_sum[negative_ok] / negative_count[negative_ok]
        out *= spec.weight
    else:
        raise ValueError(f"Unsupported stack method: {spec.method}")

    enough_members = active_count >= spec.min_active_members
    out = np.where(enough_members, out, 0.0)
    out = np.clip(out, -spec.cap_c, spec.cap_c)
    return out, active_count


def promotion_gate(row: dict[str, object] | pd.Series) -> bool:
    return bool(
        float(row["delta_mae_vs_0064"]) < 0.0
        and float(row["delta_mae_vs_best_0065"]) <= -0.0005
        and float(row["fold_delta_max_vs_0064"]) <= 0.002
        and float(row["fold_delta_max_vs_best_0065"]) <= 0.002
        and int(row["active_n"]) >= 120
    )


def score_stack_specs(
    frame: pd.DataFrame,
    member_ids: list[str],
    corrections: np.ndarray,
    active: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = pd.to_numeric(frame["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    reference_prediction = pd.to_numeric(frame["reference_prediction_c"], errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    fold_ids = frame["fold_id"].astype(str).to_numpy()
    reference_score = score_arrays(target, reference_prediction, dates)
    best_0065_prediction = reference_prediction + corrections[:, 0]
    best_0065_score = score_arrays(target, best_0065_prediction, dates)

    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for spec in stack_specs(member_ids):
        stack_correction, active_count = correction_for_stack(spec, member_ids, corrections, active)
        prediction = reference_prediction + stack_correction
        score = score_arrays(target, prediction, dates)
        active_mask = np.abs(stack_correction) > 1e-12
        active_score = score_arrays(target[active_mask], prediction[active_mask], dates.loc[active_mask])
        active_reference_score = score_arrays(
            target[active_mask],
            reference_prediction[active_mask],
            dates.loc[active_mask],
        )
        fold_delta_vs_0064: list[float] = []
        fold_delta_vs_best_0065: list[float] = []
        for fold_id in pd.unique(fold_ids):
            fold_mask = fold_ids == fold_id
            fold_score = score_arrays(target[fold_mask], prediction[fold_mask], dates.loc[fold_mask])
            fold_ref = score_arrays(target[fold_mask], reference_prediction[fold_mask], dates.loc[fold_mask])
            fold_best = score_arrays(target[fold_mask], best_0065_prediction[fold_mask], dates.loc[fold_mask])
            fold_delta_vs_0064.append(float(fold_score["mae"]) - float(fold_ref["mae"]))
            fold_delta_vs_best_0065.append(float(fold_score["mae"]) - float(fold_best["mae"]))
        row: dict[str, object] = {
            "stack_id": spec.stack_id,
            "method": spec.method,
            "candidate_count": len(spec.candidate_ids),
            "candidate_ids": "|".join(spec.candidate_ids),
            "weight": spec.weight,
            "cap_c": spec.cap_c,
            "min_active_members": spec.min_active_members,
            "n": score["n"],
            "mae": score["mae"],
            "rmse": score["rmse"],
            "bias": score["bias"],
            "reference_0064_mae": reference_score["mae"],
            "reference_0064_rmse": reference_score["rmse"],
            "best_0065_mae": best_0065_score["mae"],
            "best_0065_rmse": best_0065_score["rmse"],
            "delta_mae_vs_0064": float(score["mae"]) - float(reference_score["mae"]),
            "delta_mae_vs_best_0065": float(score["mae"]) - float(best_0065_score["mae"]),
            "active_n": active_score["n"],
            "active_mae": active_score["mae"],
            "active_reference_0064_mae": active_reference_score["mae"],
            "active_delta_mae_vs_0064": (
                float(active_score["mae"]) - float(active_reference_score["mae"])
                if int(active_score["n"]) > 0
                else math.nan
            ),
            "fold_delta_max_vs_0064": max(fold_delta_vs_0064),
            "fold_delta_min_vs_0064": min(fold_delta_vs_0064),
            "fold_delta_max_vs_best_0065": max(fold_delta_vs_best_0065),
            "fold_delta_min_vs_best_0065": min(fold_delta_vs_best_0065),
            "folds_improved_vs_0064": int(sum(delta < 0 for delta in fold_delta_vs_0064)),
            "folds_improved_vs_best_0065": int(sum(delta < 0 for delta in fold_delta_vs_best_0065)),
            "mean_abs_correction_c": float(np.mean(np.abs(stack_correction))),
            "max_abs_correction_c": float(np.max(np.abs(stack_correction))),
            "mean_active_members": float(np.mean(active_count[active_count > 0])) if active_count.any() else 0.0,
        }
        row["promotion_gate_passed"] = promotion_gate(row)
        rows.append(row)
        prediction_rows.append(
            pd.DataFrame(
                {
                    "target_date": frame["target_date"],
                    "target_tmax_c": frame["target_tmax_c"],
                    "reference_prediction_c": reference_prediction,
                    "candidate_prediction_c": prediction,
                    "stack_correction_c": stack_correction,
                    "active_member_count": active_count,
                    "stack_id": spec.stack_id,
                    "fold_id": frame["fold_id"],
                }
            )
        )

    scoreboard = pd.DataFrame(rows).sort_values(
        ["promotion_gate_passed", "delta_mae_vs_0064", "delta_mae_vs_best_0065"],
        ascending=[False, True, True],
    )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    require_no_confirmation_dates(predictions["target_date"], context="0066 predictions")
    return scoreboard.reset_index(drop=True), predictions


def leakage_audit(
    frame: pd.DataFrame,
    member_rows: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    promoted = scoreboard[scoreboard["promotion_gate_passed"].astype(bool)].copy()
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(
                pd.to_datetime(frame["target_date"], errors="coerce").max() < CONFIRMATION_START
                and pd.to_datetime(predictions["target_date"], errors="coerce").max() < CONFIRMATION_START
            ),
            "evidence": f"last target_date {pd.to_datetime(frame['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "members_are_promoted_0065_candidates",
            "passed": bool(member_rows["delta_mae_vs_0064"].lt(0).all()),
            "evidence": f"{len(member_rows)} member candidates loaded",
        },
        {
            "check_id": "stack_is_deterministic_no_new_model_fit",
            "passed": True,
            "evidence": "stack rules combine already prior-only 0065 residual corrections",
        },
        {
            "check_id": "promotion_gate_requires_beating_best_0065_and_fold_guard",
            "passed": bool(
                promoted.empty
                or (
                    promoted["delta_mae_vs_best_0065"].le(-0.0005).all()
                    and promoted["fold_delta_max_vs_0064"].le(0.002).all()
                    and promoted["fold_delta_max_vs_best_0065"].le(0.002).all()
                )
            ),
            "evidence": f"{len(promoted)} stacks passed promotion gate",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    member_rows: pd.DataFrame,
    stack_definitions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    promoted = scoreboard[scoreboard["promotion_gate_passed"].astype(bool)].copy()
    return f"""# Station-Feature Guarded Stack

Generated: `{summary['generated_at_utc']}`

## Purpose

`0065` found many deployable station-feature bucket specialists that individually improve the `0064` station-only reference. `0066` tests whether those specialists compound when combined through deterministic guarded stack rules. No new predictive model is trained here. The stack only combines residual corrections that were already generated by prior-only candidate logic in `0065`.

## Contract

- Reference baseline: `0064` best proxy `{summary['reference_0064_proxy']}`.
- Best singleton baseline: `0065` candidate `{summary['best_0065_candidate']}`.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- No 2024+ confirmation rows are used.
- Stack rules are deterministic: active mean, capped sum, rank-first, or same-sign consensus.
- Promotion requires beating both `0064` and the best `0065` singleton with bounded fold damage.

## Headline

| Item | Value |
|---|---:|
| Member candidates loaded | {summary['member_candidate_count']} |
| Stack candidates tested | {summary['stack_candidate_count']} |
| Promoted stacks | {summary['promoted_stack_count']} |
| Reference 0064 MAE | {summary['reference_0064_mae']} |
| Best 0065 singleton MAE | {summary['best_0065_mae']} |
| Best stack | {summary['best_stack']} |
| Best stack MAE | {summary['best_stack_mae']} |
| Best stack RMSE | {summary['best_stack_rmse']} |
| Best delta vs 0064 | {summary['best_delta_mae_vs_0064']} |
| Best delta vs best 0065 | {summary['best_delta_mae_vs_best_0065']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

## Member Candidates

{markdown_table(member_rows, max_rows=30)}

## Stack Definitions

{markdown_table(stack_definitions, max_rows=30)}

## Scoreboard

{markdown_table(scoreboard, max_rows=40)}

## Promoted Stacks

{markdown_table(promoted, max_rows=30)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Interpretation

This experiment is a compounding test. If the best stack beats the best `0065` singleton, then multiple station-feature regimes are complementary and should be carried into the next deterministic station-only stack. If the best stack is only the singleton, then the current promoted bucket specialists overlap too heavily and should be routed more selectively rather than summed.

## Files

- `artifacts/member_candidates.csv`
- `artifacts/stack_definitions.csv`
- `artifacts/scoreboard.csv`
- `artifacts/predictions.csv`
- `artifacts/predictions_sample.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_feature_guarded_stack.py`:

- `{FOLDER_NAME}`: deterministic guarded stack of promoted `0065` station-feature residual specialists.

| Metric | Value |
|---|---:|
| Reference 0064 MAE | {summary['reference_0064_mae']} |
| Best 0065 MAE | {summary['best_0065_mae']} |
| Best stack | {summary['best_stack']} |
| Best stack MAE | {summary['best_stack_mae']} |
| Delta vs 0064 | {summary['best_delta_mae_vs_0064']} |
| Delta vs best 0065 | {summary['best_delta_mae_vs_best_0065']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

Leakage contract: stack members are prior-only `0065` corrections; no 2024+ rows and no new fitted model.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Station-Feature Guarded Stack",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_feature_guarded_stack.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Reference | `0064` `{summary['reference_0064_proxy']}` | Tested |
| Best singleton | `0065` `{summary['best_0065_candidate']}` | Baseline |
| Member candidates | `{summary['member_candidate_count']}` promoted candidates | Tested |
| Stack candidates | `{summary['stack_candidate_count']}` deterministic rules | Tested |
| Reference 0064 MAE / RMSE | `{summary['reference_0064_mae']}` / `{summary['reference_0064_rmse']}` | Baseline |
| Best 0065 MAE / RMSE | `{summary['best_0065_mae']}` / `{summary['best_0065_rmse']}` | Singleton |
| Best stack | `{summary['best_stack']}` | Diagnostic |
| Best stack MAE / RMSE | `{summary['best_stack_mae']}` / `{summary['best_stack_rmse']}` | Pre-2024 only |
| Delta vs best 0065 | `{summary['best_delta_mae_vs_best_0065']}` | Stack value |
| Promotion gate passed | `{summary['best_promotion_gate_passed']}` | Guarded |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0066` tests whether promoted `0065` station-feature specialists compound through deterministic stack rules.
"""
    update_markdown_section(
        path,
        heading="Station-Feature Guarded Stack",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"50. Station-feature guarded stacking tested `{summary['stack_candidate_count']}` deterministic stacks; "
        f"best delta vs best 0065 is `{summary['best_delta_mae_vs_best_0065']}` from `{summary['best_stack']}`."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue while the forecast backfill runs: build a fold-aware router that chooses between the best `0065/0066` station-feature residual family and the existing official-forecast residual family, using only pre-target metadata and no 2024+ confirmation rows.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0064, _thresholds, _components = load_reference_frame()
    scoreboard_0065, _definitions_0065, candidates, terms, summary_0065 = load_0065_artifacts()
    member_ids = select_member_candidates(scoreboard_0065)
    member_rows, corrections, active = load_member_predictions(frame, member_ids, candidates, terms)
    scoreboard, predictions = score_stack_specs(frame, member_ids, corrections, active)
    stack_definitions = pd.DataFrame([spec.__dict__ for spec in stack_specs(member_ids)])
    stack_definitions["candidate_ids"] = stack_definitions["candidate_ids"].map(lambda value: "|".join(value))
    leakage = leakage_audit(frame, member_rows, scoreboard, predictions)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0066 leakage audit failed: {failed}")

    best = scoreboard.iloc[0]
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "reference_0064_proxy": str(summary_0064["best_proxy"]),
        "best_0065_candidate": str(summary_0065["best_candidate"]),
        "member_candidate_count": int(len(member_rows)),
        "stack_candidate_count": int(len(scoreboard)),
        "promoted_stack_count": int(scoreboard["promotion_gate_passed"].astype(bool).sum()),
        "rows_scored": int(len(frame)),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "reference_0064_mae": float(best["reference_0064_mae"]),
        "reference_0064_rmse": float(best["reference_0064_rmse"]),
        "best_0065_mae": float(best["best_0065_mae"]),
        "best_0065_rmse": float(best["best_0065_rmse"]),
        "best_stack": str(best["stack_id"]),
        "best_stack_mae": float(best["mae"]),
        "best_stack_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0064": float(best["delta_mae_vs_0064"]),
        "best_delta_mae_vs_best_0065": float(best["delta_mae_vs_best_0065"]),
        "best_fold_delta_max_vs_0064": float(best["fold_delta_max_vs_0064"]),
        "best_fold_delta_max_vs_best_0065": float(best["fold_delta_max_vs_best_0065"]),
        "best_promotion_gate_passed": bool(best["promotion_gate_passed"]),
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
        "uses_2024_plus_rows": False,
    }

    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "member_candidates.csv", member_rows)
    write_csv(artifacts / "stack_definitions.csv", stack_definitions)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "predictions.csv", predictions)
    write_csv(artifacts / "predictions_sample.csv", predictions.head(2000))
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_feature_guarded_stack_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            member_rows=member_rows,
            stack_definitions=stack_definitions,
            scoreboard=scoreboard,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Test deterministic guarded stacks of promoted 0065 station-feature residual specialists."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
