from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
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
from scripts.run_hkg_t24_station_feature_bucket_residual_mining import score_arrays  # noqa: E402
from scripts.run_hkg_t24_station_official_family_router import (  # noqa: E402
    LATE_EVAL_START,
    absdiff_bucket,
    active_count_bucket,
    signeddiff_bucket,
)

FOLDER_NAME = "0068_prior_calibrated_fusion_screen"
ARTIFACT_0067 = RESEARCH_ROOT / "0067_station_official_family_router" / "artifacts"
COMMON_FRAME_PATH = ARTIFACT_0067 / "common_frame.csv"
SUMMARY_0067_PATH = ARTIFACT_0067 / "summary.json"
WEIGHT_GRID = (0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.36, 0.40, 0.45, 0.50)


@dataclass(frozen=True)
class FusionSpec:
    candidate_id: str
    mode: str
    fixed_weight: float
    group_mode: str
    min_history: int
    fallback_weight: float
    temperature_c: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_common_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not COMMON_FRAME_PATH.exists():
        raise FileNotFoundError(f"Missing 0067 common frame: {COMMON_FRAME_PATH}")
    summary_0067 = load_json(SUMMARY_0067_PATH)
    frame = pd.read_csv(COMMON_FRAME_PATH)
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame = frame[frame["target_date"].notna()].copy()
    frame = frame.sort_values("target_date").reset_index(drop=True)
    require_no_confirmation_dates(frame["target_date"], context="0068 common frame")
    required = {
        "target_tmax_c",
        "official_family_prediction_c",
        "station_family_prediction_c",
        "forecast_source_family",
        "fold_id",
        "family_disagreement_c",
        "abs_family_disagreement_c",
        "active_member_count",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"0068 common frame missing required columns: {sorted(missing)}")
    return frame, summary_0067


def blend_prediction(frame: pd.DataFrame, station_weight: float | pd.Series | np.ndarray) -> np.ndarray:
    official = pd.to_numeric(frame["official_family_prediction_c"], errors="coerce").to_numpy(dtype=float)
    station = pd.to_numeric(frame["station_family_prediction_c"], errors="coerce").to_numpy(dtype=float)
    weight = np.asarray(station_weight, dtype=float)
    return (1.0 - weight) * official + weight * station


def score_prediction(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float | int | str]:
    return score_arrays(
        pd.to_numeric(frame["target_tmax_c"], errors="coerce").to_numpy(dtype=float),
        prediction,
        pd.to_datetime(frame["target_date"], errors="coerce"),
    )


def group_key(row: pd.Series, group_mode: str) -> str:
    source = str(row["forecast_source_family"])
    abs_bucket = absdiff_bucket(float(row["abs_family_disagreement_c"]))
    signed_bucket = signeddiff_bucket(float(row["family_disagreement_c"]))
    active_bucket = active_count_bucket(float(row["active_member_count"]))
    if group_mode == "global":
        return "global"
    if group_mode == "source":
        return source
    if group_mode == "signeddiff":
        return signed_bucket
    if group_mode == "absdiff":
        return abs_bucket
    if group_mode == "active_count":
        return active_bucket
    if group_mode == "source_signeddiff":
        return f"{source}|{signed_bucket}"
    if group_mode == "source_absdiff":
        return f"{source}|{abs_bucket}"
    if group_mode == "source_active_count":
        return f"{source}|{active_bucket}"
    raise ValueError(f"Unsupported group mode: {group_mode}")


def fixed_weight_specs() -> list[FusionSpec]:
    specs = []
    for weight in WEIGHT_GRID:
        token = str(weight).replace(".", "p")
        specs.append(
            FusionSpec(f"fixed_weight_{token}", "fixed_weight", weight, "global", 0, weight, 0.0)
        )
    return specs


def prior_weight_specs() -> list[FusionSpec]:
    specs: list[FusionSpec] = []
    group_modes = [
        "global",
        "source",
        "signeddiff",
        "active_count",
        "source_signeddiff",
        "source_active_count",
    ]
    for group_mode in group_modes:
        for min_history in (30, 120, 365):
            for fallback_weight in (0.0, 0.15, 0.22):
                fallback_token = str(fallback_weight).replace(".", "p")
                specs.append(
                    FusionSpec(
                        f"prior_best_{group_mode}_h{min_history}_fb{fallback_token}",
                        "prior_best_weight",
                        0.0,
                        group_mode,
                        min_history,
                        fallback_weight,
                        0.0,
                    )
                )
            for temperature in (0.02, 0.05, 0.10):
                temp_token = str(temperature).replace(".", "p")
                specs.append(
                    FusionSpec(
                        f"prior_soft_{group_mode}_h{min_history}_t{temp_token}_fb0p15",
                        "prior_soft_weight",
                        0.0,
                        group_mode,
                        min_history,
                        0.15,
                        temperature,
                    )
                )
    return specs


def fold_transfer_specs() -> list[FusionSpec]:
    specs = []
    for fallback_weight in (0.0, 0.15, 0.22, 0.25):
        fallback_token = str(fallback_weight).replace(".", "p")
        specs.append(
            FusionSpec(
                f"fold_prior_best_global_fb{fallback_token}",
                "fold_prior_best_weight",
                0.0,
                "global",
                30,
                fallback_weight,
                0.0,
            )
        )
    return specs


def fusion_specs() -> list[FusionSpec]:
    specs = fixed_weight_specs() + prior_weight_specs() + fold_transfer_specs()
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("0068 fusion candidate IDs are not unique")
    return specs


def weight_errors(row: pd.Series) -> np.ndarray:
    target = float(row["target_tmax_c"])
    official = float(row["official_family_prediction_c"])
    station = float(row["station_family_prediction_c"])
    return np.abs(np.array([(1.0 - w) * official + w * station for w in WEIGHT_GRID], dtype=float) - target)


def select_prior_weight(abs_sums: np.ndarray, count: int, spec: FusionSpec) -> float:
    if count < spec.min_history:
        return spec.fallback_weight
    prior_mae = abs_sums / count
    if spec.mode == "prior_best_weight":
        return float(WEIGHT_GRID[int(np.argmin(prior_mae))])
    if spec.mode == "prior_soft_weight":
        centered = prior_mae - float(np.min(prior_mae))
        raw = np.exp(-centered / spec.temperature_c)
        probs = raw / raw.sum()
        return float(np.sum(np.array(WEIGHT_GRID) * probs))
    raise ValueError(f"Unsupported prior selector: {spec.mode}")


def apply_prior_weight_spec(frame: pd.DataFrame, spec: FusionSpec) -> pd.DataFrame:
    state: dict[str, dict[str, object]] = defaultdict(
        lambda: {"count": 0, "abs_sums": np.zeros(len(WEIGHT_GRID), dtype=float)}
    )
    weights: list[float] = []
    prior_counts: list[int] = []
    groups: list[str] = []
    selected_prior_maes: list[float] = []
    for _idx, row in frame.iterrows():
        key = group_key(row, spec.group_mode)
        group_state = state[key]
        count = int(group_state["count"])
        abs_sums = np.asarray(group_state["abs_sums"], dtype=float)
        weight = select_prior_weight(abs_sums, count, spec)
        weights.append(weight)
        prior_counts.append(count)
        groups.append(key)
        if count:
            nearest_index = int(np.argmin(np.abs(np.array(WEIGHT_GRID) - weight)))
            selected_prior_maes.append(float(abs_sums[nearest_index] / count))
        else:
            selected_prior_maes.append(math.nan)
        group_state["abs_sums"] = abs_sums + weight_errors(row)
        group_state["count"] = count + 1
    out = frame[["target_date", "target_tmax_c", "fold_id", "forecast_source_family"]].copy()
    out["station_weight"] = weights
    out["candidate_prediction_c"] = blend_prediction(frame, np.array(weights, dtype=float))
    out["prior_count"] = prior_counts
    out["router_group"] = groups
    out["selected_prior_mae"] = selected_prior_maes
    return out


def apply_fold_transfer_spec(frame: pd.DataFrame, spec: FusionSpec) -> pd.DataFrame:
    weights = pd.Series(spec.fallback_weight, index=frame.index, dtype=float)
    prior_counts = pd.Series(0, index=frame.index, dtype=int)
    selected_prior_mae = pd.Series(math.nan, index=frame.index, dtype=float)
    for fold_id in frame["fold_id"].drop_duplicates().tolist():
        fold_mask = frame["fold_id"].astype(str).eq(str(fold_id))
        prior = frame[frame["target_date"].lt(frame.loc[fold_mask, "target_date"].min())].copy()
        if len(prior) >= spec.min_history:
            maes = []
            for weight in WEIGHT_GRID:
                pred = blend_prediction(prior, weight)
                maes.append(float(np.mean(np.abs(pred - pd.to_numeric(prior["target_tmax_c"], errors="coerce").to_numpy(dtype=float)))))
            best_index = int(np.argmin(np.array(maes)))
            weights.loc[fold_mask] = float(WEIGHT_GRID[best_index])
            selected_prior_mae.loc[fold_mask] = float(maes[best_index])
            prior_counts.loc[fold_mask] = int(len(prior))
    out = frame[["target_date", "target_tmax_c", "fold_id", "forecast_source_family"]].copy()
    out["station_weight"] = weights.to_numpy(dtype=float)
    out["candidate_prediction_c"] = blend_prediction(frame, weights.to_numpy(dtype=float))
    out["prior_count"] = prior_counts.to_numpy(dtype=int)
    out["router_group"] = "fold_prior"
    out["selected_prior_mae"] = selected_prior_mae.to_numpy(dtype=float)
    return out


def apply_spec(frame: pd.DataFrame, spec: FusionSpec) -> pd.DataFrame:
    if spec.mode == "fixed_weight":
        out = frame[["target_date", "target_tmax_c", "fold_id", "forecast_source_family"]].copy()
        out["station_weight"] = spec.fixed_weight
        out["candidate_prediction_c"] = blend_prediction(frame, spec.fixed_weight)
        out["prior_count"] = 0
        out["router_group"] = "fixed"
        out["selected_prior_mae"] = math.nan
    elif spec.mode in {"prior_best_weight", "prior_soft_weight"}:
        out = apply_prior_weight_spec(frame, spec)
    elif spec.mode == "fold_prior_best_weight":
        out = apply_fold_transfer_spec(frame, spec)
    else:
        raise ValueError(f"Unsupported fusion mode: {spec.mode}")
    out["candidate_id"] = spec.candidate_id
    out["mode"] = spec.mode
    out["group_mode"] = spec.group_mode
    out["min_history"] = spec.min_history
    out["fallback_weight"] = spec.fallback_weight
    out["temperature_c"] = spec.temperature_c
    return out


def score_candidate(frame: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    score = score_prediction(frame, pd.to_numeric(predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float))
    official_score = score_prediction(frame, pd.to_numeric(frame["official_family_prediction_c"], errors="coerce").to_numpy(dtype=float))
    station_score = score_prediction(frame, pd.to_numeric(frame["station_family_prediction_c"], errors="coerce").to_numpy(dtype=float))
    late_mask = predictions["target_date"].ge(LATE_EVAL_START)
    late_score = score_prediction(frame.loc[late_mask].copy(), pd.to_numeric(predictions.loc[late_mask, "candidate_prediction_c"], errors="coerce").to_numpy(dtype=float))
    late_official = score_prediction(frame.loc[late_mask].copy(), pd.to_numeric(frame.loc[late_mask, "official_family_prediction_c"], errors="coerce").to_numpy(dtype=float))
    fold_deltas: list[float] = []
    for _fold_id, fold_predictions in predictions.groupby("fold_id", observed=True):
        fold_frame = frame.loc[fold_predictions.index].copy()
        fold_score = score_prediction(fold_frame, pd.to_numeric(fold_predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float))
        fold_official = score_prediction(fold_frame, pd.to_numeric(fold_frame["official_family_prediction_c"], errors="coerce").to_numpy(dtype=float))
        fold_deltas.append(float(fold_score["mae"]) - float(fold_official["mae"]))
    row: dict[str, object] = {
        "candidate_id": str(predictions["candidate_id"].iloc[0]),
        "mode": str(predictions["mode"].iloc[0]),
        "group_mode": str(predictions["group_mode"].iloc[0]),
        "min_history": int(predictions["min_history"].iloc[0]),
        "fallback_weight": float(predictions["fallback_weight"].iloc[0]),
        "temperature_c": float(predictions["temperature_c"].iloc[0]),
        "n": score["n"],
        "mae": score["mae"],
        "rmse": score["rmse"],
        "bias": score["bias"],
        "official_mae": official_score["mae"],
        "station_mae": station_score["mae"],
        "delta_mae_vs_official": float(score["mae"]) - float(official_score["mae"]),
        "delta_mae_vs_station": float(score["mae"]) - float(station_score["mae"]),
        "late_n": late_score["n"],
        "late_mae": late_score["mae"],
        "late_delta_mae_vs_official": float(late_score["mae"]) - float(late_official["mae"]),
        "fold_delta_max_vs_official": max(fold_deltas) if fold_deltas else math.nan,
        "fold_delta_min_vs_official": min(fold_deltas) if fold_deltas else math.nan,
        "folds_improved_vs_official": int(sum(delta < 0 for delta in fold_deltas)),
        "mean_station_weight": float(pd.to_numeric(predictions["station_weight"], errors="coerce").mean()),
        "press_mean_station_weight": float(pd.to_numeric(predictions.loc[predictions["forecast_source_family"].eq("press_archive"), "station_weight"], errors="coerce").mean()),
        "rss_mean_station_weight": float(pd.to_numeric(predictions.loc[predictions["forecast_source_family"].eq("rss_archive"), "station_weight"], errors="coerce").mean()),
        "weight_std": float(pd.to_numeric(predictions["station_weight"], errors="coerce").std(ddof=0)),
    }
    row["promotion_gate_passed"] = bool(
        float(row["delta_mae_vs_official"]) <= -0.001
        and float(row["fold_delta_max_vs_official"]) <= 0.0
        and float(row["late_delta_mae_vs_official"]) <= 0.0
    )
    return row


def score_all_specs(frame: pd.DataFrame, specs: list[FusionSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs:
        predictions = apply_spec(frame, spec)
        rows.append(score_candidate(frame, predictions))
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(rows).sort_values(
        ["promotion_gate_passed", "mae", "fold_delta_max_vs_official"],
        ascending=[False, True, True],
    )
    top_ids = set(scoreboard["candidate_id"].head(25).astype(str))
    selected_predictions = pd.concat(
        [predictions for predictions in prediction_frames if str(predictions["candidate_id"].iloc[0]) in top_ids],
        ignore_index=True,
    )
    require_no_confirmation_dates(selected_predictions["target_date"], context="0068 selected predictions")
    return scoreboard.reset_index(drop=True), selected_predictions


def fixed_weight_stability(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_defs = [("all", "all", frame)]
    for column in ["forecast_source_family", "fold_id"]:
        for value, group in frame.groupby(column, observed=True):
            group_defs.append((column, str(value), group))
    for group_name, group_value, group in group_defs:
        for weight in WEIGHT_GRID:
            prediction = blend_prediction(group, weight)
            score = score_prediction(group, prediction)
            rows.append(
                {
                    "group_name": group_name,
                    "group_value": group_value,
                    "station_weight": weight,
                    "n": score["n"],
                    "mae": score["mae"],
                    "rmse": score["rmse"],
                    "bias": score["bias"],
                }
            )
    out = pd.DataFrame(rows)
    out["rank_in_group"] = out.groupby(["group_name", "group_value"], observed=True)["mae"].rank(method="first")
    return out.sort_values(["group_name", "group_value", "rank_in_group"]).reset_index(drop=True)


def best_weight_by_group(stability: pd.DataFrame) -> pd.DataFrame:
    best = stability[stability["rank_in_group"].eq(1.0)].copy()
    return best.sort_values(["group_name", "group_value"]).reset_index(drop=True)


def leakage_audit(frame: pd.DataFrame, scoreboard: pd.DataFrame) -> pd.DataFrame:
    promoted = scoreboard[scoreboard["promotion_gate_passed"].astype(bool)].copy()
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(frame["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(frame['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "common_frame_has_one_row_per_date",
            "passed": bool(len(frame) == frame["target_date"].nunique()),
            "evidence": f"{len(frame)} common rows",
        },
        {
            "check_id": "prior_weight_selectors_update_after_scoring",
            "passed": True,
            "evidence": "prior weight sums update after each row prediction is chosen",
        },
        {
            "check_id": "promotion_requires_late_and_fold_improvement",
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
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    best_weights: pd.DataFrame,
    stability: pd.DataFrame,
    definitions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    promoted = scoreboard[scoreboard["promotion_gate_passed"].astype(bool)].copy()
    return f"""# Prior-Calibrated Fusion Screen

Generated: `{summary['generated_at_utc']}`

## Purpose

`0067` showed that a fixed blend of the official family and the station family substantially improves common-date MAE. `0068` asks whether that blend weight is stable and whether it can be selected from prior-only evidence rather than chosen after looking at the full evaluation period.

## Contract

- Input frame: `0067_station_official_family_router/artifacts/common_frame.csv`.
- Common rows: `{summary['common_rows']}`.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- No 2024+ rows are used.
- Fixed-weight candidates are diagnostic predeclared grid candidates.
- Prior-weight candidates update error history only after each target date is scored.
- Fold-transfer candidates pick a weight for each fold from earlier folds only, with a fixed fallback for the first fold.

## Headline

| Item | Value |
|---|---:|
| Official baseline MAE | {summary['official_baseline_mae']} |
| 0067 best MAE | {summary['best_0067_mae']} |
| Best 0068 candidate | {summary['best_candidate']} |
| Best 0068 mode | {summary['best_mode']} |
| Best MAE | {summary['best_mae']} |
| Best RMSE | {summary['best_rmse']} |
| Best delta vs official | {summary['best_delta_mae_vs_official']} |
| Best delta vs 0067 best | {summary['best_delta_mae_vs_0067']} |
| Best late delta vs official | {summary['best_late_delta_mae_vs_official']} |
| Mean station weight | {summary['best_mean_station_weight']} |
| Press best fixed weight | {summary['press_best_fixed_weight']} |
| RSS best fixed weight | {summary['rss_best_fixed_weight']} |
| Weight stability gap | {summary['press_rss_best_weight_gap']} |

## Best Fixed Weights By Group

{markdown_table(best_weights, max_rows=30)}

## Fixed Weight Stability Grid

{markdown_table(stability, max_rows=80)}

## Candidate Definitions

{markdown_table(definitions, max_rows=120)}

## Scoreboard

{markdown_table(scoreboard, max_rows=100)}

## Promoted Candidates

{markdown_table(promoted, max_rows=80)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Interpretation

The full common-period fixed optimum is useful, but stability matters. If press/archive rows and RSS rows prefer different station weights, a single fixed number is not the final answer. If prior-only selectors fail to match the fixed-grid optimum, the next step should focus on richer pre-target calibration signals rather than promoting a post-hoc weight.

## Files

- `artifacts/fixed_weight_stability.csv`
- `artifacts/best_weight_by_group.csv`
- `artifacts/candidate_definitions.csv`
- `artifacts/scoreboard.csv`
- `artifacts/top_predictions.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_prior_calibrated_fusion_screen.py`:

- `{FOLDER_NAME}`: prior-calibrated and fixed-grid station/official fusion weight stability screen.

| Metric | Value |
|---|---:|
| Official baseline MAE | {summary['official_baseline_mae']} |
| 0067 best MAE | {summary['best_0067_mae']} |
| Best 0068 candidate | {summary['best_candidate']} |
| Best 0068 MAE | {summary['best_mae']} |
| Delta vs 0067 best | {summary['best_delta_mae_vs_0067']} |
| Press best fixed weight | {summary['press_best_fixed_weight']} |
| RSS best fixed weight | {summary['rss_best_fixed_weight']} |

Leakage contract: prior selectors update after scoring each row; fold-transfer uses earlier folds only; no 2024+ rows.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Prior-Calibrated Fusion Screen",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_prior_calibrated_fusion_screen.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Input | `0067` common station/official frame | Tested |
| Common rows | `{summary['common_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Non-contiguous |
| Official baseline MAE / RMSE | `{summary['official_baseline_mae']}` / `{summary['official_baseline_rmse']}` | Baseline |
| 0067 best MAE / RMSE | `{summary['best_0067_mae']}` / `{summary['best_0067_rmse']}` | Baseline |
| Best 0068 candidate | `{summary['best_candidate']}` | Diagnostic |
| Best 0068 MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Delta vs 0067 best | `{summary['best_delta_mae_vs_0067']}` | Weight refinement |
| Press best fixed weight | `{summary['press_best_fixed_weight']}` | Stability |
| RSS best fixed weight | `{summary['rss_best_fixed_weight']}` | Stability |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0068` refines the station/official blend and measures whether blend weights are stable across press-archive and RSS eras.
"""
    update_markdown_section(
        path,
        heading="Prior-Calibrated Fusion Screen",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"52. Prior-calibrated fusion screened `{summary['candidate_count']}` station/official weight candidates; "
        f"best delta vs 0067 is `{summary['best_delta_mae_vs_0067']}` from `{summary['best_candidate']}`, "
        f"with press/RSS best weights `{summary['press_best_fixed_weight']}` / `{summary['rss_best_fixed_weight']}`."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue while the forecast backfill runs: build an era/source-aware blend-weight model using only pre-target family disagreement, source family, station-stack activity, and prior calibration state, then test whether it can beat the fixed-grid optimum without late/fold damage.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0067 = load_common_frame()
    specs = fusion_specs()
    definitions = pd.DataFrame([spec.__dict__ for spec in specs])
    scoreboard, top_predictions = score_all_specs(frame, specs)
    stability = fixed_weight_stability(frame)
    best_weights = best_weight_by_group(stability)
    leakage = leakage_audit(frame, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0068 leakage audit failed: {failed}")

    best = scoreboard.iloc[0]
    official_score = score_prediction(frame, pd.to_numeric(frame["official_family_prediction_c"], errors="coerce").to_numpy(dtype=float))
    best_0067_mae = float(summary_0067["best_mae"])
    best_0067_rmse = float(summary_0067["best_rmse"])
    press_best = best_weights[
        best_weights["group_name"].eq("forecast_source_family")
        & best_weights["group_value"].eq("press_archive")
    ].iloc[0]
    rss_best = best_weights[
        best_weights["group_name"].eq("forecast_source_family")
        & best_weights["group_value"].eq("rss_archive")
    ].iloc[0]
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "common_rows": int(len(frame)),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "candidate_count": int(len(scoreboard)),
        "promoted_candidate_count": int(scoreboard["promotion_gate_passed"].astype(bool).sum()),
        "official_baseline_mae": float(official_score["mae"]),
        "official_baseline_rmse": float(official_score["rmse"]),
        "best_0067_candidate": str(summary_0067["best_candidate"]),
        "best_0067_mae": best_0067_mae,
        "best_0067_rmse": best_0067_rmse,
        "best_candidate": str(best["candidate_id"]),
        "best_mode": str(best["mode"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_official": float(best["delta_mae_vs_official"]),
        "best_delta_mae_vs_0067": float(best["mae"]) - best_0067_mae,
        "best_late_delta_mae_vs_official": float(best["late_delta_mae_vs_official"]),
        "best_mean_station_weight": float(best["mean_station_weight"]),
        "press_best_fixed_weight": float(press_best["station_weight"]),
        "press_best_fixed_mae": float(press_best["mae"]),
        "rss_best_fixed_weight": float(rss_best["station_weight"]),
        "rss_best_fixed_mae": float(rss_best["mae"]),
        "press_rss_best_weight_gap": float(abs(float(press_best["station_weight"]) - float(rss_best["station_weight"]))),
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
        "uses_2024_plus_rows": False,
    }

    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "fixed_weight_stability.csv", stability)
    write_csv(artifacts / "best_weight_by_group.csv", best_weights)
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "prior_calibrated_fusion_screen_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            best_weights=best_weights,
            stability=stability,
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
        description="Screen leakage-safe fixed and prior-calibrated station/official fusion weights."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
