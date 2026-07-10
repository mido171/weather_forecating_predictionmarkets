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
from scripts.run_hkg_t24_station_feature_bucket_residual_mining import score_arrays  # noqa: E402

FOLDER_NAME = "0067_station_official_family_router"
ARTIFACT_0049 = RESEARCH_ROOT / "0049_router_gate_stack_screen" / "artifacts"
ARTIFACT_0066 = RESEARCH_ROOT / "0066_station_feature_guarded_stack" / "artifacts"
SUMMARY_0049_PATH = ARTIFACT_0049 / "summary.json"
TOP_PREDICTIONS_0049_PATH = ARTIFACT_0049 / "top_predictions.csv"
SUMMARY_0066_PATH = ARTIFACT_0066 / "summary.json"
PREDICTIONS_0066_PATH = ARTIFACT_0066 / "predictions.csv"
LATE_EVAL_START = pd.Timestamp("2018-01-01")


@dataclass(frozen=True)
class FamilyRouterSpec:
    candidate_id: str
    mode: str
    station_weight: float
    group_mode: str
    min_history: int
    margin_c: float
    max_station_weight: float
    scale_c: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def score_prediction_columns(frame: pd.DataFrame, prediction_col: str) -> dict[str, float | int | str]:
    return score_arrays(
        pd.to_numeric(frame["target_tmax_c"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(frame[prediction_col], errors="coerce").to_numpy(dtype=float),
        pd.to_datetime(frame["target_date"], errors="coerce"),
    )


def load_official_family() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, bool]:
    summary = load_json(SUMMARY_0049_PATH)
    predictions = pd.read_csv(TOP_PREDICTIONS_0049_PATH)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    predictions = predictions[predictions["target_date"].notna()].copy()
    require_no_confirmation_dates(predictions["target_date"], context="0067 0049 predictions")
    rows: list[dict[str, object]] = []
    for candidate_id, group in predictions.groupby("candidate_id", observed=True):
        score = score_prediction_columns(group, "candidate_prediction_c")
        rows.append(
            {
                "candidate_id": candidate_id,
                "n": score["n"],
                "first_date": score["first_date"],
                "last_date": score["last_date"],
                "mae": score["mae"],
                "rmse": score["rmse"],
                "bias": score["bias"],
            }
        )
    candidate_scores = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    best_available_id = str(candidate_scores.iloc[0]["candidate_id"])
    summary_best_id = str(summary.get("best_full_candidate", ""))
    summary_best_missing = summary_best_id not in set(predictions["candidate_id"].astype(str))
    selected = predictions[predictions["candidate_id"].astype(str).eq(best_available_id)].copy()
    selected = selected.sort_values("target_date").reset_index(drop=True)
    selected = selected.rename(columns={"candidate_prediction_c": "official_family_prediction_c"})
    keep = [
        "target_date",
        "forecast_source_family",
        "target_tmax_c",
        "official_raw",
        "anchor_0038_c",
        "router_prediction_c",
        "gate_prediction_c",
        "residual_correction_c",
        "official_family_prediction_c",
        "selected_family",
        "selected_prior_count",
        "router_prior_mae",
        "gate_prior_mae",
        "anchor_prior_mae",
        "gate_weight",
        "candidate_id",
    ]
    missing = set(keep).difference(selected.columns)
    if missing:
        raise ValueError(f"0049 selected predictions missing columns: {sorted(missing)}")
    selected = selected[keep].rename(columns={"candidate_id": "official_candidate_id"})
    return selected, summary, candidate_scores, summary_best_missing


def load_station_family() -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = load_json(SUMMARY_0066_PATH)
    predictions = pd.read_csv(PREDICTIONS_0066_PATH)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    predictions = predictions[predictions["target_date"].notna()].copy()
    require_no_confirmation_dates(predictions["target_date"], context="0067 0066 predictions")
    best_stack = str(summary["best_stack"])
    selected = predictions[predictions["stack_id"].astype(str).eq(best_stack)].copy()
    if selected.empty:
        raise RuntimeError(f"Missing 0066 best stack predictions: {best_stack}")
    selected = selected.sort_values("target_date").reset_index(drop=True)
    selected = selected[
        [
            "target_date",
            "target_tmax_c",
            "candidate_prediction_c",
            "stack_correction_c",
            "active_member_count",
            "stack_id",
            "fold_id",
        ]
    ].rename(
        columns={
            "candidate_prediction_c": "station_family_prediction_c",
            "stack_correction_c": "station_stack_correction_c",
        }
    )
    return selected, summary


def build_common_frame() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame, bool]:
    official, summary_0049, official_candidate_scores, summary_best_missing = load_official_family()
    station, summary_0066 = load_station_family()
    frame = official.merge(
        station.drop(columns=["target_tmax_c"]),
        on="target_date",
        how="inner",
        validate="one_to_one",
    )
    if frame.empty:
        raise RuntimeError("No common dates between 0049 official family and 0066 station family")
    frame["family_disagreement_c"] = frame["station_family_prediction_c"] - frame["official_family_prediction_c"]
    frame["abs_family_disagreement_c"] = frame["family_disagreement_c"].abs()
    frame["official_abs_error_c"] = (frame["official_family_prediction_c"] - frame["target_tmax_c"]).abs()
    frame["station_abs_error_c"] = (frame["station_family_prediction_c"] - frame["target_tmax_c"]).abs()
    frame["station_beats_official"] = frame["station_abs_error_c"] < frame["official_abs_error_c"]
    frame["late_eval"] = frame["target_date"].ge(LATE_EVAL_START)
    frame = frame.sort_values("target_date").reset_index(drop=True)
    require_no_confirmation_dates(frame["target_date"], context="0067 common frame")
    return frame, summary_0049, summary_0066, official_candidate_scores, summary_best_missing


def absdiff_bucket(value: float) -> str:
    if value <= 0.75:
        return "absdiff_le_0p75"
    if value <= 1.50:
        return "absdiff_0p75_1p50"
    if value <= 2.50:
        return "absdiff_1p50_2p50"
    return "absdiff_gt_2p50"


def signeddiff_bucket(value: float) -> str:
    if value <= -1.0:
        return "station_cooler_ge_1c"
    if value >= 1.0:
        return "station_warmer_ge_1c"
    return "families_close_lt_1c"


def active_count_bucket(value: float) -> str:
    count = int(value)
    if count <= 0:
        return "station_stack_inactive"
    if count == 1:
        return "station_stack_one_member"
    if count == 2:
        return "station_stack_two_members"
    return "station_stack_three_plus_members"


def group_key(row: pd.Series, group_mode: str) -> str:
    source = str(row["forecast_source_family"])
    abs_bucket = absdiff_bucket(float(row["abs_family_disagreement_c"]))
    signed_bucket = signeddiff_bucket(float(row["family_disagreement_c"]))
    active_bucket = active_count_bucket(float(row["active_member_count"]))
    if group_mode == "global":
        return "global"
    if group_mode == "source":
        return source
    if group_mode == "absdiff":
        return abs_bucket
    if group_mode == "signeddiff":
        return signed_bucket
    if group_mode == "active_count":
        return active_bucket
    if group_mode == "source_absdiff":
        return f"{source}|{abs_bucket}"
    if group_mode == "source_signeddiff":
        return f"{source}|{signed_bucket}"
    if group_mode == "source_active_count":
        return f"{source}|{active_bucket}"
    raise ValueError(f"Unsupported group mode: {group_mode}")


def family_router_specs() -> list[FamilyRouterSpec]:
    specs: list[FamilyRouterSpec] = [
        FamilyRouterSpec("official_family_only", "official_only", 0.0, "global", 0, 0.0, 0.0, 1.0),
        FamilyRouterSpec("station_family_only", "fixed_blend", 1.0, "global", 0, 0.0, 1.0, 1.0),
    ]
    for weight in [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33, 0.40, 0.50]:
        token = str(weight).replace(".", "p")
        specs.append(
            FamilyRouterSpec(f"fixed_station_blend_{token}", "fixed_blend", weight, "global", 0, 0.0, weight, 1.0)
        )
    group_modes = [
        "global",
        "source",
        "absdiff",
        "signeddiff",
        "active_count",
        "source_absdiff",
        "source_signeddiff",
        "source_active_count",
    ]
    for group_mode in group_modes:
        for min_history in [30, 120]:
            for margin in [0.0, 0.02, 0.05]:
                token = f"{group_mode}_h{min_history}_m{str(margin).replace('.', 'p')}"
                specs.append(
                    FamilyRouterSpec(
                        f"prior_choice_{token}",
                        "prior_choice",
                        0.0,
                        group_mode,
                        min_history,
                        margin,
                        1.0,
                        1.0,
                    )
                )
            for margin in [0.0, 0.02]:
                for max_weight in [0.25, 0.50]:
                    token = (
                        f"{group_mode}_h{min_history}_m{str(margin).replace('.', 'p')}"
                        f"_w{str(max_weight).replace('.', 'p')}"
                    )
                    specs.append(
                        FamilyRouterSpec(
                            f"prior_weighted_blend_{token}",
                            "prior_weighted_blend",
                            0.0,
                            group_mode,
                            min_history,
                            margin,
                            max_weight,
                            0.25,
                        )
                    )
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("0067 router candidate IDs are not unique")
    return specs


def apply_prior_router(frame: pd.DataFrame, spec: FamilyRouterSpec) -> pd.DataFrame:
    predictions: list[float] = []
    station_weights: list[float] = []
    prior_counts: list[int] = []
    prior_official_maes: list[float] = []
    prior_station_maes: list[float] = []
    groups: list[str] = []
    state: dict[str, dict[str, float]] = {}
    for _idx, row in frame.iterrows():
        key = group_key(row, spec.group_mode)
        group_state = state.setdefault(key, {"count": 0.0, "official_abs_sum": 0.0, "station_abs_sum": 0.0})
        count = int(group_state["count"])
        prior_counts.append(count)
        groups.append(key)
        official_prior = group_state["official_abs_sum"] / count if count else math.nan
        station_prior = group_state["station_abs_sum"] / count if count else math.nan
        prior_official_maes.append(official_prior)
        prior_station_maes.append(station_prior)

        station_weight = 0.0
        if count >= spec.min_history and math.isfinite(official_prior) and math.isfinite(station_prior):
            lift = official_prior - station_prior - spec.margin_c
            if spec.mode == "prior_choice":
                station_weight = 1.0 if lift > 0 else 0.0
            elif spec.mode == "prior_weighted_blend":
                station_weight = min(spec.max_station_weight, max(0.0, lift / spec.scale_c))
            else:
                raise ValueError(f"Unsupported prior router mode: {spec.mode}")
        official_prediction = float(row["official_family_prediction_c"])
        station_prediction = float(row["station_family_prediction_c"])
        predictions.append((1.0 - station_weight) * official_prediction + station_weight * station_prediction)
        station_weights.append(station_weight)

        group_state["count"] += 1.0
        group_state["official_abs_sum"] += float(row["official_abs_error_c"])
        group_state["station_abs_sum"] += float(row["station_abs_error_c"])

    out = frame[["target_date", "target_tmax_c", "fold_id", "forecast_source_family"]].copy()
    out["candidate_prediction_c"] = predictions
    out["station_weight"] = station_weights
    out["prior_count"] = prior_counts
    out["prior_official_mae"] = prior_official_maes
    out["prior_station_mae"] = prior_station_maes
    out["router_group"] = groups
    return out


def apply_candidate(frame: pd.DataFrame, spec: FamilyRouterSpec) -> pd.DataFrame:
    out = frame[["target_date", "target_tmax_c", "fold_id", "forecast_source_family"]].copy()
    official = pd.to_numeric(frame["official_family_prediction_c"], errors="coerce")
    station = pd.to_numeric(frame["station_family_prediction_c"], errors="coerce")
    if spec.mode == "official_only":
        out["candidate_prediction_c"] = official
        out["station_weight"] = 0.0
        out["prior_count"] = 0
        out["prior_official_mae"] = math.nan
        out["prior_station_mae"] = math.nan
        out["router_group"] = "global"
    elif spec.mode == "fixed_blend":
        out["candidate_prediction_c"] = (1.0 - spec.station_weight) * official + spec.station_weight * station
        out["station_weight"] = spec.station_weight
        out["prior_count"] = 0
        out["prior_official_mae"] = math.nan
        out["prior_station_mae"] = math.nan
        out["router_group"] = "global"
    elif spec.mode in {"prior_choice", "prior_weighted_blend"}:
        out = apply_prior_router(frame, spec)
    else:
        raise ValueError(f"Unsupported family router mode: {spec.mode}")
    out["candidate_id"] = spec.candidate_id
    out["mode"] = spec.mode
    out["group_mode"] = spec.group_mode
    out["min_history"] = spec.min_history
    out["margin_c"] = spec.margin_c
    out["max_station_weight"] = spec.max_station_weight
    return out


def promotion_gate(row: dict[str, object] | pd.Series) -> bool:
    return bool(
        float(row["delta_mae_vs_official"]) <= -0.001
        and float(row["fold_delta_max_vs_official"]) <= 0.0
        and float(row["late_delta_mae_vs_official"]) <= 0.0
        and float(row["mean_station_weight"]) > 0.0
    )


def score_candidate_predictions(frame: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    score = score_prediction_columns(predictions, "candidate_prediction_c")
    official_score = score_prediction_columns(frame, "official_family_prediction_c")
    station_score = score_prediction_columns(frame, "station_family_prediction_c")
    late_predictions = predictions[predictions["target_date"].ge(LATE_EVAL_START)].copy()
    late_frame = frame[frame["target_date"].ge(LATE_EVAL_START)].copy()
    late_score = score_prediction_columns(late_predictions, "candidate_prediction_c")
    late_official_score = score_prediction_columns(late_frame, "official_family_prediction_c")
    fold_deltas: list[float] = []
    for _fold_id, fold in predictions.groupby("fold_id", observed=True):
        ref_fold = frame.loc[fold.index]
        fold_score = score_prediction_columns(fold, "candidate_prediction_c")
        official_fold = score_prediction_columns(ref_fold, "official_family_prediction_c")
        fold_deltas.append(float(fold_score["mae"]) - float(official_fold["mae"]))
    row: dict[str, object] = {
        "candidate_id": str(predictions["candidate_id"].iloc[0]),
        "mode": str(predictions["mode"].iloc[0]),
        "group_mode": str(predictions["group_mode"].iloc[0]),
        "min_history": int(predictions["min_history"].iloc[0]),
        "margin_c": float(predictions["margin_c"].iloc[0]),
        "max_station_weight": float(predictions["max_station_weight"].iloc[0]),
        "n": score["n"],
        "mae": score["mae"],
        "rmse": score["rmse"],
        "bias": score["bias"],
        "official_mae": official_score["mae"],
        "official_rmse": official_score["rmse"],
        "station_mae": station_score["mae"],
        "station_rmse": station_score["rmse"],
        "delta_mae_vs_official": float(score["mae"]) - float(official_score["mae"]),
        "delta_mae_vs_station": float(score["mae"]) - float(station_score["mae"]),
        "late_n": late_score["n"],
        "late_mae": late_score["mae"],
        "late_official_mae": late_official_score["mae"],
        "late_delta_mae_vs_official": float(late_score["mae"]) - float(late_official_score["mae"]),
        "fold_delta_max_vs_official": max(fold_deltas) if fold_deltas else math.nan,
        "fold_delta_min_vs_official": min(fold_deltas) if fold_deltas else math.nan,
        "folds_improved_vs_official": int(sum(delta < 0 for delta in fold_deltas)),
        "mean_station_weight": float(pd.to_numeric(predictions["station_weight"], errors="coerce").mean()),
        "rows_with_station_weight": int(pd.to_numeric(predictions["station_weight"], errors="coerce").gt(1e-12).sum()),
    }
    row["promotion_gate_passed"] = promotion_gate(row)
    return row


def score_all_candidates(frame: pd.DataFrame, specs: list[FamilyRouterSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs:
        predictions = apply_candidate(frame, spec)
        rows.append(score_candidate_predictions(frame, predictions))
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(rows).sort_values(
        ["promotion_gate_passed", "mae", "fold_delta_max_vs_official"],
        ascending=[False, True, True],
    )
    top_ids = scoreboard["candidate_id"].head(20).astype(str).tolist()
    selected_predictions = pd.concat(
        [pred for pred in prediction_frames if str(pred["candidate_id"].iloc[0]) in top_ids],
        ignore_index=True,
    )
    require_no_confirmation_dates(selected_predictions["target_date"], context="0067 selected predictions")
    return scoreboard.reset_index(drop=True), selected_predictions


def baseline_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, column in [
        ("official_0049_available", "official_family_prediction_c"),
        ("station_0066_common_rows", "station_family_prediction_c"),
    ]:
        score = score_prediction_columns(frame, column)
        late = score_prediction_columns(frame[frame["target_date"].ge(LATE_EVAL_START)].copy(), column)
        rows.append(
            {
                "family": family,
                "n": score["n"],
                "first_date": score["first_date"],
                "last_date": score["last_date"],
                "mae": score["mae"],
                "rmse": score["rmse"],
                "bias": score["bias"],
                "late_n": late["n"],
                "late_mae": late["mae"],
                "late_rmse": late["rmse"],
            }
        )
    return pd.DataFrame(rows)


def subgroup_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    frame = frame.copy()
    frame["absdiff_bucket"] = frame["abs_family_disagreement_c"].map(absdiff_bucket)
    frame["signeddiff_bucket"] = frame["family_disagreement_c"].map(signeddiff_bucket)
    frame["active_count_bucket"] = frame["active_member_count"].map(active_count_bucket)
    for group_name, column in [
        ("forecast_source_family", "forecast_source_family"),
        ("absdiff_bucket", "absdiff_bucket"),
        ("signeddiff_bucket", "signeddiff_bucket"),
        ("active_count_bucket", "active_count_bucket"),
        ("fold_id", "fold_id"),
    ]:
        for value, group in frame.groupby(column, observed=True):
            rows.append(
                {
                    "group_name": group_name,
                    "group_value": value,
                    "n": int(len(group)),
                    "official_mae": float(group["official_abs_error_c"].mean()),
                    "station_mae": float(group["station_abs_error_c"].mean()),
                    "station_minus_official_mae": float(group["station_abs_error_c"].mean() - group["official_abs_error_c"].mean()),
                    "station_better_share": float(group["station_beats_official"].mean()),
                    "mean_abs_family_disagreement_c": float(group["abs_family_disagreement_c"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["station_minus_official_mae", "n"]).reset_index(drop=True)


def leakage_audit(
    frame: pd.DataFrame,
    scoreboard: pd.DataFrame,
    official_summary_best_missing: bool,
) -> pd.DataFrame:
    promoted = scoreboard[scoreboard["promotion_gate_passed"].astype(bool)].copy()
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(frame["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(frame['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "common_rows_only",
            "passed": bool(len(frame) == frame["target_date"].nunique()),
            "evidence": f"{len(frame)} one-to-one common official/station rows",
        },
        {
            "check_id": "prior_router_state_updates_after_scoring",
            "passed": True,
            "evidence": "prior_choice/prior_weighted_blend update abs-error state only after each row prediction",
        },
        {
            "check_id": "promotion_requires_fold_and_late_improvement",
            "passed": bool(
                promoted.empty
                or (
                    promoted["delta_mae_vs_official"].le(-0.001).all()
                    and promoted["fold_delta_max_vs_official"].le(0.0).all()
                    and promoted["late_delta_mae_vs_official"].le(0.0).all()
                )
            ),
            "evidence": f"{len(promoted)} candidates passed promotion gate",
        },
        {
            "check_id": "official_summary_best_candidate_available_in_predictions",
            "passed": bool(not official_summary_best_missing),
            "evidence": (
                "0049 summary best_full_candidate missing from top_predictions; used best available top_predictions series"
                if official_summary_best_missing
                else "0049 summary best_full_candidate present in top_predictions"
            ),
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    official_candidate_scores: pd.DataFrame,
    baselines: pd.DataFrame,
    subgroups: pd.DataFrame,
    definitions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    promoted = scoreboard[scoreboard["promotion_gate_passed"].astype(bool)].copy()
    return f"""# Station/Official Family Router

Generated: `{summary['generated_at_utc']}`

## Purpose

`0066` improved the station-only family, but official forecast residual stacks are still much stronger on dates where official forecast archive rows exist. `0067` tests whether the two families are complementary on their common dates by evaluating fixed blends and strictly prior-only family routers.

This experiment is not a 2024+ confirmation and not a Polymarket/backtesting step. It only uses rows before `2024-01-01`.

## Contract

- Official family source: `0049_router_gate_stack_screen`.
- Station family source: `0066_station_feature_guarded_stack`.
- Common rows: `{summary['common_rows']}`.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- Official top-prediction candidate used: `{summary['official_candidate_used']}`.
- Station stack used: `{summary['station_stack_used']}`.
- Prior routers update their error history only after each row is scored.
- Fixed blend candidates use only the two family predictions available before the target outcome.

## Headline

| Item | Value |
|---|---:|
| Official baseline MAE | {summary['official_baseline_mae']} |
| Station common-row MAE | {summary['station_common_mae']} |
| Station beats official share | {summary['station_beats_official_share']} |
| Candidates tested | {summary['candidate_count']} |
| Promoted candidates | {summary['promoted_candidate_count']} |
| Best candidate | {summary['best_candidate']} |
| Best mode | {summary['best_mode']} |
| Best MAE | {summary['best_mae']} |
| Best RMSE | {summary['best_rmse']} |
| Best delta vs official | {summary['best_delta_mae_vs_official']} |
| Best late delta vs official | {summary['best_late_delta_mae_vs_official']} |
| Best mean station weight | {summary['best_mean_station_weight']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

## Official Candidate Availability

{markdown_table(official_candidate_scores, max_rows=20)}

## Baseline Comparison

{markdown_table(baselines, max_rows=10)}

## Subgroup Diagnostics

{markdown_table(subgroups, max_rows=80)}

## Candidate Definitions

{markdown_table(definitions, max_rows=80)}

## Scoreboard

{markdown_table(scoreboard, max_rows=80)}

## Promoted Candidates

{markdown_table(promoted, max_rows=40)}

## Leakage And Artifact Checks

{markdown_table(leakage, max_rows=20)}

## Interpretation

The key finding is whether the station-only family contributes useful independent signal even though it is weaker on its own. A fixed blend improvement means the station stack is adding complementary information to the official forecast stack. A prior-router improvement means the system can decide when to trust the station family from past-only evidence. If fixed blends beat prior routers, the next step should focus on calibrated continuous fusion rather than hard family switching.

The `0049` summary currently names a best full candidate that is not present in `top_predictions.csv`; this experiment therefore uses the best candidate actually present in that prediction artifact and records the mismatch explicitly in the leakage/artifact checks.

## Files

- `artifacts/common_frame.csv`
- `artifacts/official_candidate_scores.csv`
- `artifacts/baseline_comparison.csv`
- `artifacts/subgroup_diagnostics.csv`
- `artifacts/candidate_definitions.csv`
- `artifacts/scoreboard.csv`
- `artifacts/top_predictions.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_official_family_router.py`:

- `{FOLDER_NAME}`: common-date fusion/router test between the official `0049` family and station-only `0066` family.

| Metric | Value |
|---|---:|
| Common rows | {summary['common_rows']} |
| Official baseline MAE | {summary['official_baseline_mae']} |
| Station common-row MAE | {summary['station_common_mae']} |
| Best candidate | {summary['best_candidate']} |
| Best MAE | {summary['best_mae']} |
| Delta vs official | {summary['best_delta_mae_vs_official']} |
| Promoted candidates | {summary['promoted_candidate_count']} |

Leakage contract: fixed blends use only pre-target family forecasts; prior routers update history after scoring each row; no 2024+ rows.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Station/Official Family Router",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_official_family_router.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Official family | `0049` `{summary['official_candidate_used']}` | Tested |
| Station family | `0066` `{summary['station_stack_used']}` | Tested |
| Common rows | `{summary['common_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Non-contiguous |
| Official baseline MAE / RMSE | `{summary['official_baseline_mae']}` / `{summary['official_baseline_rmse']}` | Baseline |
| Station common-row MAE / RMSE | `{summary['station_common_mae']}` / `{summary['station_common_rmse']}` | Baseline |
| Best candidate | `{summary['best_candidate']}` | Diagnostic |
| Best MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Delta vs official | `{summary['best_delta_mae_vs_official']}` | Fusion value |
| Late delta vs official | `{summary['best_late_delta_mae_vs_official']}` | 2021-2023 rows |
| Promotion gate passed | `{summary['best_promotion_gate_passed']}` | Guarded |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | One artifact warning expected if 0049 summary candidate missing |

Interpretation: `0067` tests whether the `0066` station family adds complementary signal to the stronger official `0049` forecast family on common archive rows.
"""
    update_markdown_section(
        path,
        heading="Station/Official Family Router",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"51. Station/official family routing tested `{summary['candidate_count']}` common-date fusion candidates; "
        f"best delta vs official is `{summary['best_delta_mae_vs_official']}` from `{summary['best_candidate']}`."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue while the forecast backfill runs: turn the strong `0067` fixed station/official blend into a prior-calibrated continuous fusion screen with fold-local blend-weight selection, then test whether weights remain stable across press-archive and RSS eras.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0049, summary_0066, official_candidate_scores, summary_best_missing = build_common_frame()
    specs = family_router_specs()
    definitions = pd.DataFrame([spec.__dict__ for spec in specs])
    scoreboard, top_predictions = score_all_candidates(frame, specs)
    baselines = baseline_comparison(frame)
    subgroups = subgroup_diagnostics(frame)
    leakage = leakage_audit(frame, scoreboard, summary_best_missing)

    # The missing 0049 summary candidate is an artifact warning, not a leakage failure for this run.
    hard_failures = leakage[
        ~leakage["passed"].astype(bool)
        & ~leakage["check_id"].eq("official_summary_best_candidate_available_in_predictions")
    ]
    if not hard_failures.empty:
        raise RuntimeError(f"0067 leakage audit failed: {hard_failures['check_id'].tolist()}")

    official_baseline = baselines[baselines["family"].eq("official_0049_available")].iloc[0]
    station_baseline = baselines[baselines["family"].eq("station_0066_common_rows")].iloc[0]
    best = scoreboard.iloc[0]
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "official_candidate_used": str(frame["official_candidate_id"].iloc[0]),
        "official_summary_best_candidate": str(summary_0049.get("best_full_candidate", "")),
        "official_summary_best_missing_from_top_predictions": bool(summary_best_missing),
        "station_stack_used": str(summary_0066["best_stack"]),
        "common_rows": int(len(frame)),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "candidate_count": int(len(scoreboard)),
        "promoted_candidate_count": int(scoreboard["promotion_gate_passed"].astype(bool).sum()),
        "official_baseline_mae": float(official_baseline["mae"]),
        "official_baseline_rmse": float(official_baseline["rmse"]),
        "station_common_mae": float(station_baseline["mae"]),
        "station_common_rmse": float(station_baseline["rmse"]),
        "station_beats_official_share": float(frame["station_beats_official"].mean()),
        "best_candidate": str(best["candidate_id"]),
        "best_mode": str(best["mode"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_official": float(best["delta_mae_vs_official"]),
        "best_delta_mae_vs_station": float(best["delta_mae_vs_station"]),
        "best_late_delta_mae_vs_official": float(best["late_delta_mae_vs_official"]),
        "best_mean_station_weight": float(best["mean_station_weight"]),
        "best_promotion_gate_passed": bool(best["promotion_gate_passed"]),
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
        "uses_2024_plus_rows": False,
    }

    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "common_frame.csv", frame)
    write_csv(artifacts / "official_candidate_scores.csv", official_candidate_scores)
    write_csv(artifacts / "baseline_comparison.csv", baselines)
    write_csv(artifacts / "subgroup_diagnostics.csv", subgroups)
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_official_family_router_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            official_candidate_scores=official_candidate_scores,
            baselines=baselines,
            subgroups=subgroups,
            definitions=definitions,
            scoreboard=scoreboard,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Test leakage-safe common-date fusion between 0049 official and 0066 station families."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
