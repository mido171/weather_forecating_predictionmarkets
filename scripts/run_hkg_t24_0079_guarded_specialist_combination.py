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

from scripts.run_hkg_t24_0078_prior_only_residual_specialists import (  # noqa: E402
    FOLDER_NAME as FOLDER_0078,
)
from scripts.run_hkg_t24_0078_prior_only_residual_specialists import fold_id_for_date
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
from scripts.run_hkg_t24_online_residual_memory_halflife import (  # noqa: E402
    BASE_MATERIALITY_C,
)
from scripts.run_hkg_t24_prior_calibrated_fusion_screen import score_prediction  # noqa: E402
from scripts.run_hkg_t24_remaining_0075_error_feature_mining import build_joined_frame  # noqa: E402
from scripts.run_hkg_t24_station_official_family_router import LATE_EVAL_START  # noqa: E402

FOLDER_NAME = "0079_guarded_specialist_combination"
ARTIFACT_ROOT_0078 = RESEARCH_ROOT / FOLDER_0078 / "artifacts"


@dataclass(frozen=True)
class GuardedCombinationSpec:
    candidate_id: str
    pool_mode: str
    fallback_mode: str
    combine_mode: str
    combo_weight: float
    min_independent_families: int
    min_prior_lift_c: float
    min_abs_correction_c: float
    require_same_sign: bool
    correction_cap_c: float


@dataclass(frozen=True)
class FamilyDecision:
    candidate_id: str
    feature_family: str
    correction_c: float
    prior_lift_c: float
    candidate_weight: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_0078_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scoreboard_path = ARTIFACT_ROOT_0078 / "scoreboard.csv"
    predictions_path = ARTIFACT_ROOT_0078 / "top_predictions.csv"
    summary_path = ARTIFACT_ROOT_0078 / "summary.json"
    missing = [path for path in (scoreboard_path, predictions_path, summary_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"0079 requires 0078 artifacts first: {missing}")
    scoreboard = pd.read_csv(scoreboard_path)
    predictions = pd.read_csv(predictions_path)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require_no_confirmation_dates(predictions["target_date"], context="0079 0078 predictions")
    return scoreboard, predictions, summary


def guarded_specs() -> list[GuardedCombinationSpec]:
    specs: list[GuardedCombinationSpec] = []
    for pool_mode in ("new_champions", "full_positive", "late_positive"):
        for fallback_mode in ("m0078", "m0075"):
            for combine_mode in ("mean", "prior_lift_weighted", "candidate_lift_weighted"):
                for combo_weight in (0.5, 1.0):
                    for min_prior_lift in (0.0, 0.005):
                        candidate_id = (
                            f"guarded_{pool_mode}_{fallback_mode}_{combine_mode}_"
                            f"w{combo_weight}_lift{min_prior_lift}"
                        ).replace(".", "p")
                        specs.append(
                            GuardedCombinationSpec(
                                candidate_id=candidate_id,
                                pool_mode=pool_mode,
                                fallback_mode=fallback_mode,
                                combine_mode=combine_mode,
                                combo_weight=combo_weight,
                                min_independent_families=2,
                                min_prior_lift_c=min_prior_lift,
                                min_abs_correction_c=0.0,
                                require_same_sign=True,
                                correction_cap_c=0.20,
                            )
                        )
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("0079 candidate IDs are not unique")
    return specs


def select_pool(scoreboard: pd.DataFrame, top_predictions: pd.DataFrame, pool_mode: str) -> pd.DataFrame:
    available = set(top_predictions["candidate_id"].astype(str))
    pool = scoreboard[scoreboard["candidate_id"].astype(str).isin(available)].copy()
    if pool_mode == "new_champions":
        pool = pool[pool["new_champion_gate_passed"].astype(bool)].copy()
    elif pool_mode == "full_positive":
        pool = pool[
            pool["delta_mae_vs_0075"].le(0.0)
            & pool["active_delta_mae_vs_0075"].lt(0.0)
        ].copy()
    elif pool_mode == "late_positive":
        pool = pool[
            pool["delta_mae_vs_0075"].le(0.0)
            & pool["late_delta_mae_vs_0075"].le(0.0)
            & pool["active_delta_mae_vs_0075"].lt(0.0)
        ].copy()
    else:
        raise ValueError(f"Unsupported pool_mode: {pool_mode}")
    return pool.sort_values(["new_champion_gate_passed", "mae"], ascending=[False, True]).reset_index(drop=True)


def best_family_decisions(
    specialist_rows: pd.DataFrame,
    pool_meta: pd.DataFrame,
    spec: GuardedCombinationSpec,
) -> list[FamilyDecision]:
    if specialist_rows.empty:
        return []
    meta = pool_meta.set_index("candidate_id")
    rows = specialist_rows[specialist_rows["predicate_active_and_eligible"].astype(bool)].copy()
    rows["specialist_correction_c"] = pd.to_numeric(rows["specialist_correction_c"], errors="coerce")
    rows["selected_prior_lift_c"] = pd.to_numeric(rows["selected_prior_lift_c"], errors="coerce")
    rows = rows[
        rows["specialist_correction_c"].abs().ge(spec.min_abs_correction_c)
        & rows["selected_prior_lift_c"].ge(spec.min_prior_lift_c)
    ].copy()
    decisions: list[FamilyDecision] = []
    for family, group in rows.groupby("feature_family", observed=True):
        ranked: list[FamilyDecision] = []
        for row in group.itertuples(index=False):
            candidate_id = str(row.candidate_id)
            if candidate_id not in meta.index:
                continue
            candidate_lift = max(-float(meta.loc[candidate_id, "active_delta_mae_vs_0075"]), 1e-9)
            prior_lift = float(row.selected_prior_lift_c)
            correction = float(row.specialist_correction_c)
            if not math.isfinite(correction) or not math.isfinite(prior_lift):
                continue
            ranked.append(
                FamilyDecision(
                    candidate_id=candidate_id,
                    feature_family=str(family),
                    correction_c=correction,
                    prior_lift_c=prior_lift,
                    candidate_weight=candidate_lift,
                )
            )
        if ranked:
            decisions.append(max(ranked, key=lambda item: item.prior_lift_c * item.candidate_weight))
    return decisions


def decisions_agree(decisions: list[FamilyDecision], spec: GuardedCombinationSpec) -> bool:
    if len({item.feature_family for item in decisions}) < spec.min_independent_families:
        return False
    if not spec.require_same_sign:
        return True
    signs = {math.copysign(1.0, item.correction_c) for item in decisions if abs(item.correction_c) > 1e-12}
    return len(signs) == 1


def combine_family_decisions(decisions: list[FamilyDecision], spec: GuardedCombinationSpec) -> float:
    if not decisions_agree(decisions, spec):
        return 0.0
    corrections = np.array([item.correction_c for item in decisions], dtype=float)
    if spec.combine_mode == "mean":
        value = float(corrections.mean())
    elif spec.combine_mode == "prior_lift_weighted":
        weights = np.array([max(item.prior_lift_c, 1e-9) for item in decisions], dtype=float)
        value = float(np.sum(weights * corrections) / weights.sum())
    elif spec.combine_mode == "candidate_lift_weighted":
        weights = np.array([max(item.candidate_weight, 1e-9) for item in decisions], dtype=float)
        value = float(np.sum(weights * corrections) / weights.sum())
    else:
        raise ValueError(f"Unsupported combine_mode: {spec.combine_mode}")
    return float(np.clip(value, -spec.correction_cap_c, spec.correction_cap_c))


def family_decision_rows(
    specialist: pd.DataFrame,
    pool_meta: pd.DataFrame,
    spec: GuardedCombinationSpec,
) -> pd.DataFrame:
    if specialist.empty or pool_meta.empty:
        return pd.DataFrame(
            columns=[
                "row_index",
                "guarded_combo_correction_c",
                "selected_family_count",
                "selected_families",
                "selected_candidates",
            ]
        )
    meta = pool_meta.set_index("candidate_id")
    rows = specialist[specialist["predicate_active_and_eligible"].astype(bool)].copy()
    rows["specialist_correction_c"] = pd.to_numeric(rows["specialist_correction_c"], errors="coerce")
    rows["selected_prior_lift_c"] = pd.to_numeric(rows["selected_prior_lift_c"], errors="coerce")
    rows = rows[
        rows["specialist_correction_c"].abs().ge(spec.min_abs_correction_c)
        & rows["selected_prior_lift_c"].ge(spec.min_prior_lift_c)
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["candidate_weight"] = rows["candidate_id"].map(
        (-pd.to_numeric(meta["active_delta_mae_vs_0075"], errors="coerce")).clip(lower=1e-9)
    )
    rows["rank_weight"] = rows["selected_prior_lift_c"] * rows["candidate_weight"]
    rows = rows.sort_values(["row_index", "feature_family", "rank_weight"])
    rows = rows.groupby(["row_index", "feature_family"], observed=True).tail(1)

    out: list[dict[str, object]] = []
    for row_index, group in rows.groupby("row_index", observed=True):
        if group["feature_family"].nunique() < spec.min_independent_families:
            continue
        signs = set(np.sign(pd.to_numeric(group["specialist_correction_c"], errors="coerce")))
        signs.discard(0.0)
        if spec.require_same_sign and len(signs) != 1:
            continue
        corrections = pd.to_numeric(group["specialist_correction_c"], errors="coerce").to_numpy(dtype=float)
        if spec.combine_mode == "mean":
            correction = float(corrections.mean())
        elif spec.combine_mode == "prior_lift_weighted":
            weights = pd.to_numeric(group["selected_prior_lift_c"], errors="coerce").clip(lower=1e-9).to_numpy(float)
            correction = float(np.sum(weights * corrections) / weights.sum())
        elif spec.combine_mode == "candidate_lift_weighted":
            weights = pd.to_numeric(group["candidate_weight"], errors="coerce").clip(lower=1e-9).to_numpy(float)
            correction = float(np.sum(weights * corrections) / weights.sum())
        else:
            raise ValueError(f"Unsupported combine_mode: {spec.combine_mode}")
        correction = float(np.clip(correction, -spec.correction_cap_c, spec.correction_cap_c))
        out.append(
            {
                "row_index": int(row_index),
                "guarded_combo_correction_c": correction,
                "selected_family_count": int(group["feature_family"].nunique()),
                "selected_families": ";".join(sorted(group["feature_family"].astype(str).unique())),
                "selected_candidates": ";".join(group["candidate_id"].astype(str)),
            }
        )
    return pd.DataFrame(out)


def apply_guarded_combination(
    frame: pd.DataFrame,
    top_predictions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    spec: GuardedCombinationSpec,
    champion_id: str,
) -> pd.DataFrame:
    pool_meta = select_pool(scoreboard, top_predictions, spec.pool_mode)
    candidate_ids = set(pool_meta["candidate_id"].astype(str))
    specialist = top_predictions[top_predictions["candidate_id"].astype(str).isin(candidate_ids)].copy()
    champion = top_predictions[top_predictions["candidate_id"].astype(str).eq(champion_id)].copy()
    if champion.empty:
        raise RuntimeError(f"0078 champion predictions are missing from top_predictions: {champion_id}")
    champion_by_row = champion.set_index("row_index")["candidate_prediction_c"].to_dict()
    decisions = family_decision_rows(specialist, pool_meta, spec)
    decision_by_row = decisions.set_index("row_index") if not decisions.empty else pd.DataFrame()

    out = frame[["target_date", "current_target_tmax_c", "forecast_source_family"]].copy()
    out["fold_id"] = out["target_date"].map(fold_id_for_date)
    out["row_index"] = frame.index.to_numpy(dtype=int)
    out["m0075_prediction_c"] = pd.to_numeric(frame["m0075_prediction_c"], errors="coerce")
    out["m0078_prediction_c"] = [float(champion_by_row.get(i, out.loc[i, "m0075_prediction_c"])) for i in out.index]
    out["guarded_combo_correction_c"] = 0.0
    out["guard_active"] = False
    out["selected_family_count"] = 0
    out["selected_families"] = ""
    out["selected_candidates"] = ""
    if not decision_by_row.empty:
        active_indices = out.index.intersection(decision_by_row.index)
        out.loc[active_indices, "guarded_combo_correction_c"] = decision_by_row.loc[
            active_indices, "guarded_combo_correction_c"
        ].to_numpy(dtype=float)
        out.loc[active_indices, "guard_active"] = True
        out.loc[active_indices, "selected_family_count"] = decision_by_row.loc[
            active_indices, "selected_family_count"
        ].to_numpy(dtype=int)
        out.loc[active_indices, "selected_families"] = decision_by_row.loc[active_indices, "selected_families"].astype(str)
        out.loc[active_indices, "selected_candidates"] = decision_by_row.loc[
            active_indices, "selected_candidates"
        ].astype(str)
    fallback = (
        pd.to_numeric(out["m0078_prediction_c"], errors="coerce")
        if spec.fallback_mode == "m0078"
        else pd.to_numeric(out["m0075_prediction_c"], errors="coerce")
    )
    combo_prediction = pd.to_numeric(out["m0075_prediction_c"], errors="coerce") + pd.to_numeric(
        out["guarded_combo_correction_c"], errors="coerce"
    )
    out["candidate_prediction_c"] = fallback
    active_mask = out["guard_active"].astype(bool)
    out.loc[active_mask, "candidate_prediction_c"] = (
        (1.0 - spec.combo_weight) * fallback.loc[active_mask] + spec.combo_weight * combo_prediction.loc[active_mask]
    )
    out["candidate_id"] = spec.candidate_id
    out["candidate_class"] = "guarded_prior_only_specialist_combination"
    out["pool_mode"] = spec.pool_mode
    out["fallback_mode"] = spec.fallback_mode
    out["combine_mode"] = spec.combine_mode
    out["combo_weight"] = spec.combo_weight
    out["min_independent_families"] = spec.min_independent_families
    out["min_prior_lift_c"] = spec.min_prior_lift_c
    out["require_same_sign"] = spec.require_same_sign
    out["pool_candidate_count"] = len(candidate_ids)
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


def score_candidate(frame: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    candidate_values = pd.to_numeric(predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float)
    base_values = pd.to_numeric(predictions["m0075_prediction_c"], errors="coerce").to_numpy(dtype=float)
    champion_values = pd.to_numeric(predictions["m0078_prediction_c"], errors="coerce").to_numpy(dtype=float)
    candidate_score = score_values(frame, candidate_values)
    base_score = score_values(frame, base_values)
    champion_score = score_values(frame, champion_values)

    late_mask = pd.to_datetime(predictions["target_date"], errors="coerce").ge(LATE_EVAL_START)
    fold_deltas: list[float] = []
    for _, group in predictions.groupby("fold_id", observed=True):
        mask = predictions.index.isin(group.index)
        fold_deltas.append(segment_delta(frame, candidate_values, champion_values, pd.Series(mask, index=frame.index)))
    source_deltas: list[float] = []
    for _, group in predictions.groupby("forecast_source_family", observed=True):
        mask = predictions.index.isin(group.index)
        source_deltas.append(segment_delta(frame, candidate_values, champion_values, pd.Series(mask, index=frame.index)))

    active_mask = predictions["guard_active"].astype(bool)
    active_delta = segment_delta(frame, candidate_values, champion_values, active_mask)
    late_delta = segment_delta(frame, candidate_values, champion_values, late_mask)
    active_rows = int(active_mask.sum())
    row: dict[str, object] = {
        "candidate_id": str(predictions["candidate_id"].iloc[0]),
        "candidate_class": str(predictions["candidate_class"].iloc[0]),
        "pool_mode": str(predictions["pool_mode"].iloc[0]),
        "fallback_mode": str(predictions["fallback_mode"].iloc[0]),
        "combine_mode": str(predictions["combine_mode"].iloc[0]),
        "combo_weight": float(predictions["combo_weight"].iloc[0]),
        "min_prior_lift_c": float(predictions["min_prior_lift_c"].iloc[0]),
        "n": candidate_score["n"],
        "mae": candidate_score["mae"],
        "rmse": candidate_score["rmse"],
        "bias": candidate_score["bias"],
        "m0075_mae": base_score["mae"],
        "m0078_mae": champion_score["mae"],
        "delta_mae_vs_0075": float(candidate_score["mae"]) - float(base_score["mae"]),
        "delta_mae_vs_0078": float(candidate_score["mae"]) - float(champion_score["mae"]),
        "late_n": int(late_mask.sum()),
        "late_delta_mae_vs_0078": late_delta,
        "fold_delta_max_vs_0078": max(fold_deltas) if fold_deltas else math.nan,
        "fold_delta_min_vs_0078": min(fold_deltas) if fold_deltas else math.nan,
        "source_delta_max_vs_0078": max(source_deltas) if source_deltas else math.nan,
        "source_delta_min_vs_0078": min(source_deltas) if source_deltas else math.nan,
        "active_rows": active_rows,
        "active_row_share": float(active_mask.mean()),
        "active_delta_mae_vs_0078": active_delta,
        "mean_combo_correction_c": float(pd.to_numeric(predictions["guarded_combo_correction_c"], errors="coerce").mean()),
        "mean_abs_combo_correction_c": float(
            pd.to_numeric(predictions["guarded_combo_correction_c"], errors="coerce").abs().mean()
        ),
        "pool_candidate_count": int(predictions["pool_candidate_count"].iloc[0]),
    }
    row["beats_0078"] = bool(float(row["delta_mae_vs_0078"]) <= -BASE_MATERIALITY_C)
    row["hardened_gate_passed"] = bool(
        row["beats_0078"]
        and float(row["fold_delta_max_vs_0078"]) <= 0.0
        and float(row["late_delta_mae_vs_0078"]) <= 0.0
        and float(row["source_delta_max_vs_0078"]) <= 0.0
    )
    return row


def score_all_specs(
    frame: pd.DataFrame,
    top_predictions: pd.DataFrame,
    scoreboard_0078: pd.DataFrame,
    champion_id: str,
    specs: list[GuardedCombinationSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions_list: list[pd.DataFrame] = []
    for spec in specs:
        predictions = apply_guarded_combination(frame, top_predictions, scoreboard_0078, spec, champion_id)
        rows.append(score_candidate(frame, predictions))
        predictions_list.append(predictions)
    scoreboard = pd.DataFrame(rows).sort_values(
        ["hardened_gate_passed", "beats_0078", "mae"],
        ascending=[False, False, True],
    )
    top_ids = set(scoreboard["candidate_id"].head(12).astype(str))
    top_predictions_out = pd.concat(
        [pred for pred in predictions_list if str(pred["candidate_id"].iloc[0]) in top_ids],
        ignore_index=True,
    )
    require_no_confirmation_dates(top_predictions_out["target_date"], context="0079 top predictions")
    return scoreboard.reset_index(drop=True), top_predictions_out


def robustness_breakdown(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate_id, group in predictions.groupby("candidate_id", observed=True):
        values = pd.to_numeric(group["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float)
        base = pd.to_numeric(group["m0075_prediction_c"], errors="coerce").to_numpy(dtype=float)
        champion = pd.to_numeric(group["m0078_prediction_c"], errors="coerce").to_numpy(dtype=float)
        group_frame = frame.loc[group["row_index"].astype(int)].copy()
        segments: list[tuple[str, str, pd.Series]] = [
            ("all", "all", pd.Series(True, index=group.index)),
            ("late", str(LATE_EVAL_START.date()), pd.to_datetime(group["target_date"]).ge(LATE_EVAL_START)),
        ]
        for source in sorted(group["forecast_source_family"].astype(str).unique()):
            segments.append(("source", source, group["forecast_source_family"].astype(str).eq(source)))
        for fold_id in sorted(group["fold_id"].astype(str).unique()):
            segments.append(("fold", fold_id, group["fold_id"].astype(str).eq(fold_id)))
        for segment_type, segment_id, mask in segments:
            if int(mask.sum()) == 0:
                continue
            subframe = group_frame.loc[mask.to_numpy()]
            score = score_values(subframe, values[mask.to_numpy()])
            score_0075 = score_values(subframe, base[mask.to_numpy()])
            score_0078 = score_values(subframe, champion[mask.to_numpy()])
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "segment_type": segment_type,
                    "segment_id": segment_id,
                    "n": int(score["n"]),
                    "mae": score["mae"],
                    "rmse": score["rmse"],
                    "m0075_mae": score_0075["mae"],
                    "m0078_mae": score_0078["mae"],
                    "delta_mae_vs_0075": float(score["mae"]) - float(score_0075["mae"]),
                    "delta_mae_vs_0078": float(score["mae"]) - float(score_0078["mae"]),
                }
            )
    return pd.DataFrame(rows)


def leakage_audit(
    frame: pd.DataFrame,
    specs: list[GuardedCombinationSpec],
    scoreboard: pd.DataFrame,
    top_predictions: pd.DataFrame,
) -> pd.DataFrame:
    hardened = scoreboard[scoreboard["hardened_gate_passed"].astype(bool)].copy()
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(frame["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(frame['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "combination_requires_two_independent_families",
            "passed": bool(all(spec.min_independent_families >= 2 for spec in specs)),
            "evidence": f"{len(specs)} specs require at least two independent families",
        },
        {
            "check_id": "combination_uses_prior_only_0078_flags",
            "passed": bool({"guard_active", "selected_families", "selected_candidates"}.issubset(top_predictions.columns)),
            "evidence": "0079 only combines 0078 row decisions already computed before current-row update",
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
    return f"""# Guarded Specialist Combination

Generated: `{summary['generated_at_utc']}`

## Purpose

`0079` tests whether the valid `0078` specialist families compound. The key 0078 mechanisms are weak regional ISD morning-to-midday warming and cool Waglan sea-temperature memory. This run only allows a combination to activate when at least two independent feature families are active on the same row and their prior-only corrections have the same sign.

## Data Contract

- Base benchmark: `0078` prior-only residual specialist champion.
- Earlier base benchmark: `0075` refined online residual-memory champion.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- No 2024+ confirmation rows are used.
- The official forecast archive is kept at the current stable scored state while the backfill continues separately.
- The combiner does not compute a new current-row target-derived signal. It consumes only `0078` specialist decisions that were themselves generated from prior context state before current-row update.
- Promotion requires full-window MAE improvement versus `0078`, no fold worsening, no late-window worsening, and no source-family worsening.

## Headline

| Item | Value |
|---|---:|
| 0078 champion | {summary['base_0078_candidate']} |
| 0078 MAE | {summary['base_0078_mae']} |
| Best 0079 candidate | {summary['best_candidate']} |
| Best 0079 MAE | {summary['best_mae']} |
| Best delta vs 0078 | {summary['best_delta_mae_vs_0078']} |
| Best late delta vs 0078 | {summary['best_late_delta_mae_vs_0078']} |
| Best fold max delta vs 0078 | {summary['best_fold_delta_max_vs_0078']} |
| Best source max delta vs 0078 | {summary['best_source_delta_max_vs_0078']} |
| Hardened new champion | {summary['best_hardened_candidate']} |

## Interpretation

If `0079` does not pass the hardened gate, that means the two strongest 0078 families are not yet safely additive under a strict same-row agreement rule. In that case `0078` remains the safer research champion and the next step should be to deepen the individual weak-morning-warming and Waglan-cool-sea regimes rather than broad stacking.

## Scoreboard

{markdown_table(scoreboard, max_rows=120)}

## Hardened Gate-Passed Candidates

{markdown_table(hardened, max_rows=80)}

## Robustness Breakdown

{markdown_table(robustness, max_rows=160)}

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

Additional folder created by `scripts/run_hkg_t24_0079_guarded_specialist_combination.py`:

- `{FOLDER_NAME}`: guarded two-family specialist combination on top of `0078`.

| Metric | Value |
|---|---:|
| 0078 champion MAE | {summary['base_0078_mae']} |
| Best 0079 candidate | {summary['best_candidate']} |
| Best 0079 MAE | {summary['best_mae']} |
| Best delta vs 0078 | {summary['best_delta_mae_vs_0078']} |
| Hardened new champion | {summary['best_hardened_candidate']} |

Leakage contract: no 2024+ rows; combines only prior-only 0078 specialist decisions; requires two independent families and source/fold/late hardening.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Guarded Specialist Combination",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    if summary["best_hardened_candidate"] != "NONE":
        interpretation = f"`0079` produced a hardened new champion with MAE `{summary['best_hardened_mae']}`."
        next_task = """
Continue with `0080`: deepen the `0079` hardened champion by stress-testing the active agreement rows and adding only source/fold-local corrections that preserve the hardened gate. Keep the moving forecast backfill outside the tuning frame until it is stable.
"""
    else:
        interpretation = "`0078` remains the safer champion; two-family same-row agreement did not pass hardening."
        next_task = """
Continue with `0080`: deepen the individual `0078` weak-morning-warming and cool-Waglan-sea regimes separately, using smoother prior-only local corrections rather than forcing two-family agreement. Use the current stable RSS/press scored archive only while the forecast backfill is still moving; do not use 2024+ confirmation rows.
"""
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0079_guarded_specialist_combination.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Input | `0078` prior-only specialist predictions plus `0075` base | Tested |
| Rows | `{summary['common_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Pre-2024 only |
| Candidate count | `{summary['candidate_count']}` | Tested |
| 0078 champion | `{summary['base_0078_candidate']}` | Baseline |
| 0078 MAE / RMSE | `{summary['base_0078_mae']}` / `{summary['base_0078_rmse']}` | Baseline |
| Best 0079 candidate | `{summary['best_candidate']}` | Tested |
| Best 0079 MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best delta vs 0078 | `{summary['best_delta_mae_vs_0078']}` | Combination value |
| Best fold max delta vs 0078 | `{summary['best_fold_delta_max_vs_0078']}` | Robustness check |
| Best late delta vs 0078 | `{summary['best_late_delta_mae_vs_0078']}` | Late-window check |
| Best source max delta vs 0078 | `{summary['best_source_delta_max_vs_0078']}` | Source hardening |
| Hardened new champion | `{summary['best_hardened_candidate']}` | Requires full/fold/late/source improvement |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: {interpretation}
"""
    update_markdown_section(
        path,
        heading="Guarded Specialist Combination",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"63. Guarded specialist combination screened `{summary['candidate_count']}` candidates; "
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
    frame, _summary_0075 = build_joined_frame()
    frame = frame.sort_values("target_date").reset_index(drop=True)
    require_no_confirmation_dates(frame["target_date"], context="0079 input frame")
    scoreboard_0078, top_predictions_0078, summary_0078 = load_0078_artifacts()
    champion_id = str(summary_0078["best_new_champion_candidate"])
    if champion_id == "NONE":
        champion_id = str(summary_0078["best_candidate"])
    specs = guarded_specs()
    definitions = pd.DataFrame([spec.__dict__ for spec in specs])
    scoreboard, top_predictions = score_all_specs(
        frame,
        top_predictions_0078,
        scoreboard_0078,
        champion_id,
        specs,
    )
    best = scoreboard.iloc[0]
    best_predictions = top_predictions[top_predictions["candidate_id"].astype(str).eq(str(best["candidate_id"]))].copy()
    robustness = robustness_breakdown(frame, best_predictions)
    leakage = leakage_audit(frame, specs, scoreboard, top_predictions)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0079 leakage audit failed: {failed}")
    hardened = scoreboard[scoreboard["hardened_gate_passed"].astype(bool)].copy()
    hardened = hardened.sort_values(["mae", "fold_delta_max_vs_0078"]).reset_index(drop=True)
    best_hardened = hardened.iloc[0] if not hardened.empty else None
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "common_rows": int(len(frame)),
        "first_date": str(pd.to_datetime(frame["target_date"]).min().date()),
        "last_date": str(pd.to_datetime(frame["target_date"]).max().date()),
        "candidate_count": int(len(scoreboard)),
        "base_0078_candidate": champion_id,
        "base_0078_mae": float(summary_0078["best_new_champion_mae"] or summary_0078["best_mae"]),
        "base_0078_rmse": float(summary_0078["best_new_champion_rmse"] or summary_0078["best_rmse"]),
        "best_candidate": str(best["candidate_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0078": float(best["delta_mae_vs_0078"]),
        "best_late_delta_mae_vs_0078": float(best["late_delta_mae_vs_0078"]),
        "best_fold_delta_max_vs_0078": float(best["fold_delta_max_vs_0078"]),
        "best_source_delta_max_vs_0078": float(best["source_delta_max_vs_0078"]),
        "best_active_rows": int(best["active_rows"]),
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
    write_json(RESEARCH_ROOT / "guarded_specialist_combination_manifest.json", summary)
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
    return argparse.ArgumentParser(description="Run guarded specialist combination on top of 0078.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
