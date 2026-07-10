from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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
    score_prediction_frame,
    write_csv,
    write_json,
    write_text,
)

FOLDER_NAME = "0036_revision_centroid_stack_ablation"
MIN_HISTORY = 160
TOP_SPECIALISTS_PER_FAMILY = 16


@dataclass(frozen=True)
class ExpertSource:
    source_group: str
    folder: str
    scoreboard_file: str
    predictions_file: str
    prediction_col: str
    top_n: int


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 140) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def expert_id_for(source_group: str, rank: int, candidate_id: str) -> str:
    prefix = slug(source_group, limit=18)
    return f"{prefix}_{rank:02d}_{slug(candidate_id, limit=50)}"


def source_artifact_dir(folder: str) -> Path:
    return RESEARCH_ROOT / folder / "artifacts"


def expert_sources() -> tuple[ExpertSource, ...]:
    return (
        ExpertSource(
            source_group="0033_smooth_specialist",
            folder="0033_smooth_residual_archetype_specialists",
            scoreboard_file="smooth_scoreboard.csv",
            predictions_file="top_smooth_predictions.csv",
            prediction_col="candidate_prediction_c",
            top_n=TOP_SPECIALISTS_PER_FAMILY,
        ),
        ExpertSource(
            source_group="0034_centroid_specialist",
            folder="0034_cluster_centroid_soft_gating",
            scoreboard_file="centroid_scoreboard.csv",
            predictions_file="top_centroid_predictions.csv",
            prediction_col="candidate_prediction_c",
            top_n=TOP_SPECIALISTS_PER_FAMILY,
        ),
        ExpertSource(
            source_group="0035_revision_specialist",
            folder="0035_forecast_revision_momentum_deep_dive",
            scoreboard_file="revision_scoreboard.csv",
            predictions_file="top_revision_predictions.csv",
            prediction_col="candidate_prediction_c",
            top_n=TOP_SPECIALISTS_PER_FAMILY,
        ),
        ExpertSource(
            source_group="0033_prior_blend",
            folder="0033_smooth_residual_archetype_specialists",
            scoreboard_file="blend_scoreboard.csv",
            predictions_file="blend_predictions.csv",
            prediction_col="expert_prediction_c",
            top_n=4,
        ),
        ExpertSource(
            source_group="0034_prior_blend",
            folder="0034_cluster_centroid_soft_gating",
            scoreboard_file="blend_scoreboard.csv",
            predictions_file="blend_predictions.csv",
            prediction_col="expert_prediction_c",
            top_n=4,
        ),
        ExpertSource(
            source_group="0035_prior_blend",
            folder="0035_forecast_revision_momentum_deep_dive",
            scoreboard_file="blend_scoreboard.csv",
            predictions_file="blend_predictions.csv",
            prediction_col="expert_prediction_c",
            top_n=4,
        ),
    )


def load_base_rows() -> pd.DataFrame:
    path = (
        RESEARCH_ROOT
        / "0035_forecast_revision_momentum_deep_dive"
        / "artifacts"
        / "top_revision_predictions.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing base 0035 predictions artifact: {path}")
    base = pd.read_csv(
        path,
        usecols=["target_date", "forecast_source_family", "target_tmax_c", "official_raw"],
    )
    base["target_date"] = pd.to_datetime(base["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(base["target_date"], context="0036 base rows")
    base = (
        base.dropna(subset=["target_date", "forecast_source_family", "target_tmax_c", "official_raw"])
        .drop_duplicates(["target_date", "forecast_source_family"], keep="last")
        .sort_values(["target_date", "forecast_source_family"])
        .reset_index(drop=True)
    )
    return base


def normalized_scoreboard(path: Path, top_n: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing scoreboard artifact: {path}")
    scoreboard = pd.read_csv(path)
    if "candidate_id" not in scoreboard.columns:
        raise ValueError(f"{path} missing candidate_id")
    sort_cols = [col for col in ("mae", "rmse") if col in scoreboard.columns]
    if sort_cols:
        scoreboard = scoreboard.sort_values(sort_cols).reset_index(drop=True)
    return scoreboard.head(top_n).copy()


def load_source_experts(source: ExpertSource) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = source_artifact_dir(source.folder)
    scoreboard = normalized_scoreboard(directory / source.scoreboard_file, source.top_n)
    predictions_path = directory / source.predictions_file
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing prediction artifact: {predictions_path}")
    usecols = ["target_date", "forecast_source_family", "candidate_id", source.prediction_col]
    predictions = pd.read_csv(predictions_path, usecols=usecols)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(predictions["target_date"], context=f"{source.source_group} predictions")

    mapping_rows: list[dict[str, object]] = []
    long_rows: list[pd.DataFrame] = []
    for rank, row in enumerate(scoreboard.itertuples(index=False), start=1):
        candidate_id = str(row.candidate_id)
        expert_id = expert_id_for(source.source_group, rank, candidate_id)
        mapping = {
            "expert_id": expert_id,
            "source_group": source.source_group,
            "source_folder": source.folder,
            "candidate_id": candidate_id,
            "source_rank": rank,
        }
        for col in (
            "family_name",
            "anchor_col",
            "mode",
            "same_source",
            "mae",
            "rmse",
            "delta_vs_official_same_rows",
            "delta_vs_anchor_same_rows",
            "corrected_rows",
            "active_rows",
            "n",
        ):
            if col in scoreboard.columns:
                mapping[col] = getattr(row, col)
        mapping_rows.append(mapping)

        subset = predictions[predictions["candidate_id"].eq(candidate_id)][
            ["target_date", "forecast_source_family", source.prediction_col]
        ].copy()
        subset = subset.rename(columns={source.prediction_col: "expert_prediction_c"})
        subset["expert_id"] = expert_id
        long_rows.append(subset)

    long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    mapping = pd.DataFrame(mapping_rows)
    return long, mapping


def build_stack_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = load_base_rows()
    longs: list[pd.DataFrame] = []
    mappings: list[pd.DataFrame] = [
        pd.DataFrame(
            [
                {
                    "expert_id": "official_raw",
                    "source_group": "official",
                    "source_folder": "",
                    "candidate_id": "official_raw",
                    "source_rank": 0,
                    "mae": math.nan,
                    "rmse": math.nan,
                    "delta_vs_official_same_rows": 0.0,
                }
            ]
        )
    ]
    for source in expert_sources():
        long, mapping = load_source_experts(source)
        if not long.empty:
            longs.append(long)
        if not mapping.empty:
            mappings.append(mapping)

    frame = base.copy()
    if longs:
        all_long = pd.concat(longs, ignore_index=True)
        wide = (
            all_long.pivot_table(
                index=["target_date", "forecast_source_family"],
                columns="expert_id",
                values="expert_prediction_c",
                aggfunc="last",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        frame = frame.merge(wide, on=["target_date", "forecast_source_family"], how="left")
    mapping = pd.concat(mappings, ignore_index=True)
    return frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True), mapping


def expert_groups(mapping: pd.DataFrame) -> dict[str, list[str]]:
    group_ids = {
        group: mapping.loc[mapping["source_group"].eq(group), "expert_id"].astype(str).to_list()
        for group in mapping["source_group"].dropna().unique()
    }
    official = ["official_raw"]
    smooth = group_ids.get("0033_smooth_specialist", [])
    centroid = group_ids.get("0034_centroid_specialist", [])
    revision = group_ids.get("0035_revision_specialist", [])
    prior_blends = (
        group_ids.get("0033_prior_blend", [])
        + group_ids.get("0034_prior_blend", [])
        + group_ids.get("0035_prior_blend", [])
    )
    best_centroid_blend = [
        expert
        for expert in group_ids.get("0034_prior_blend", [])
        if "cluster_centroid_blend_inverse_mae_all_prior" in expert
    ]
    return {
        "official_only": official,
        "0033_smooth_top": official + smooth,
        "0034_centroid_top": official + centroid,
        "0035_revision_top": official + revision,
        "0033_0034_smooth_centroid": official + smooth + centroid,
        "0033_0035_smooth_revision": official + smooth + revision,
        "0034_0035_centroid_revision": official + centroid + revision,
        "specialists_all": official + smooth + centroid + revision,
        "prior_blends_only": official + prior_blends,
        "all_experts_with_prior_blends": official + smooth + centroid + revision + prior_blends,
        "current_0034_blend_only": official + best_centroid_blend,
    }


def prior_error_stats(
    *,
    values: np.ndarray,
    target: np.ndarray,
    prior_mask: np.ndarray,
) -> tuple[int, float]:
    valid = prior_mask & np.isfinite(values) & np.isfinite(target)
    count = int(valid.sum())
    if count == 0:
        return 0, math.nan
    return count, float(np.abs(values[valid] - target[valid]).mean())


def strict_past_expert_stack(
    frame: pd.DataFrame,
    *,
    experts: list[str],
    mode: str,
    same_source: bool,
    min_history: int = MIN_HISTORY,
    top_k: int = 3,
) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").dt.normalize().to_numpy(dtype="datetime64[ns]")
    sources = ordered["forecast_source_family"].astype(str).to_numpy()
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    official = pd.to_numeric(ordered["official_raw"], errors="coerce").to_numpy(dtype=float)
    expert_values = {
        expert: pd.to_numeric(ordered[expert], errors="coerce").to_numpy(dtype=float)
        for expert in experts
        if expert in ordered.columns
    }
    if "official_raw" not in expert_values:
        expert_values["official_raw"] = official

    predictions: list[float] = []
    selected: list[str] = []
    eligible_counts: list[int] = []
    prior_official_maes: list[float] = []
    selected_prior_maes: list[float] = []
    min_prior_maes: list[float] = []

    for index, target_date in enumerate(dates):
        prior_mask = dates < target_date
        if same_source:
            prior_mask &= sources == sources[index]
        official_count, official_mae = prior_error_stats(values=official, target=target, prior_mask=prior_mask)
        if official_count < min_history:
            official_mae = math.nan

        current_available = [
            expert
            for expert, values in expert_values.items()
            if index < len(values) and np.isfinite(values[index])
        ]
        scored: list[tuple[str, int, float]] = []
        for expert in current_available:
            count, mae = prior_error_stats(values=expert_values[expert], target=target, prior_mask=prior_mask)
            if count >= min_history and np.isfinite(mae):
                scored.append((expert, count, mae))

        if not scored:
            predictions.append(float(official[index]) if np.isfinite(official[index]) else math.nan)
            selected.append("official_raw_fallback")
            eligible_counts.append(0)
            prior_official_maes.append(official_mae)
            selected_prior_maes.append(math.nan)
            min_prior_maes.append(math.nan)
            continue

        scored = sorted(scored, key=lambda item: (item[2], item[0]))
        eligible_counts.append(len(scored))
        prior_official_maes.append(official_mae)
        min_prior_maes.append(scored[0][2])

        if mode == "best":
            chosen = scored[0]
            predictions.append(float(expert_values[chosen[0]][index]))
            selected.append(chosen[0])
            selected_prior_maes.append(chosen[2])
        elif mode == "inverse_mae":
            weights = np.array([1.0 / max(item[2], 1e-6) for item in scored], dtype=float)
            values = np.array([float(expert_values[item[0]][index]) for item in scored], dtype=float)
            weights = weights / weights.sum()
            predictions.append(float(np.dot(weights, values)))
            selected.append("inverse_mae_blend")
            selected_prior_maes.append(scored[0][2])
        elif mode == "top3_inverse_mae":
            chosen_rows = scored[:top_k]
            weights = np.array([1.0 / max(item[2], 1e-6) for item in chosen_rows], dtype=float)
            values = np.array([float(expert_values[item[0]][index]) for item in chosen_rows], dtype=float)
            weights = weights / weights.sum()
            predictions.append(float(np.dot(weights, values)))
            selected.append("top3_inverse_mae_blend")
            selected_prior_maes.append(chosen_rows[0][2])
        elif mode == "positive_lift_top3":
            if not np.isfinite(official_mae):
                lifted: list[tuple[str, int, float, float]] = []
            else:
                lifted = [
                    (expert, count, mae, official_mae - mae)
                    for expert, count, mae in scored
                    if expert != "official_raw" and mae < official_mae
                ]
            if not lifted:
                predictions.append(float(official[index]) if np.isfinite(official[index]) else math.nan)
                selected.append("official_raw_fallback")
                selected_prior_maes.append(math.nan)
                continue
            lifted = sorted(lifted, key=lambda item: (-item[3], item[2], item[0]))[:top_k]
            weights = np.array([max(item[3], 1e-9) for item in lifted], dtype=float)
            values = np.array([float(expert_values[item[0]][index]) for item in lifted], dtype=float)
            weights = weights / weights.sum()
            predictions.append(float(np.dot(weights, values)))
            selected.append("positive_lift_top3_blend")
            selected_prior_maes.append(min(item[2] for item in lifted))
        else:
            raise ValueError(f"Unknown stack mode: {mode}")

    out = ordered[["target_date", "forecast_source_family", "target_tmax_c", "official_raw"]].copy()
    out["expert_prediction_c"] = predictions
    out["selected_expert"] = selected
    out["eligible_expert_count"] = eligible_counts
    out["mode"] = mode
    out["same_source"] = same_source
    out["prior_official_mae"] = prior_official_maes
    out["selected_prior_mae"] = selected_prior_maes
    out["min_prior_mae"] = min_prior_maes
    return out


def score_stack_candidate(
    predictions: pd.DataFrame,
    *,
    candidate_id: str,
    ablation: str,
    mode: str,
    same_source: bool,
    expert_count: int,
) -> dict[str, object]:
    candidate = score_prediction_frame(
        predictions.rename(columns={"expert_prediction_c": "prediction"}),
        "prediction",
    )
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "prediction"}), "prediction")
    selected = predictions["selected_expert"].astype(str)
    return {
        "candidate_id": candidate_id,
        "ablation": ablation,
        "mode": mode,
        "same_source": same_source,
        "expert_count": expert_count,
        **candidate,
        "official_same_rows_mae": official["mae"],
        "official_same_rows_rmse": official["rmse"],
        "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
        "official_fallback_rows": int(selected.eq("official_raw_fallback").sum()),
        "official_selected_rows": int(selected.isin(["official_raw", "official_raw_fallback"]).sum()),
        "mean_eligible_experts": float(predictions["eligible_expert_count"].mean()),
        "mean_selected_prior_mae": float(pd.to_numeric(predictions["selected_prior_mae"], errors="coerce").mean()),
    }


def run_stack_screen(
    frame: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = expert_groups(mapping)
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for ablation, experts in groups.items():
        unique_experts = list(dict.fromkeys(expert for expert in experts if expert in frame.columns))
        if ablation == "official_only":
            modes = ("best",)
            same_source_options = (False,)
        else:
            modes = ("best", "inverse_mae", "top3_inverse_mae", "positive_lift_top3")
            same_source_options = (False, True)
        for mode in modes:
            for same_source in same_source_options:
                candidate_id = f"stack_{slug(ablation)}_{mode}_{'same_source' if same_source else 'all_prior'}"
                predictions = strict_past_expert_stack(
                    frame,
                    experts=unique_experts,
                    mode=mode,
                    same_source=same_source,
                    min_history=MIN_HISTORY,
                )
                predictions["candidate_id"] = candidate_id
                predictions["ablation"] = ablation
                score_rows.append(
                    score_stack_candidate(
                        predictions,
                        candidate_id=candidate_id,
                        ablation=ablation,
                        mode=mode,
                        same_source=same_source,
                        expert_count=len(unique_experts),
                    )
                )
                prediction_frames.append(predictions)

    scoreboard = pd.DataFrame(score_rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    ablation = (
        scoreboard.groupby("ablation", observed=True)
        .agg(
            candidates=("candidate_id", "count"),
            expert_count=("expert_count", "max"),
            best_mae=("mae", "min"),
            best_rmse=("rmse", "min"),
            best_delta_vs_official=("delta_vs_official_same_rows", "min"),
            min_official_selected_rows=("official_selected_rows", "min"),
            max_mean_eligible_experts=("mean_eligible_experts", "max"),
        )
        .reset_index()
        .sort_values(["best_mae", "best_rmse"])
    )
    best_by_ablation = scoreboard.sort_values(["mae", "rmse"]).drop_duplicates("ablation", keep="first")[
        ["ablation", "candidate_id"]
    ]
    ablation = ablation.merge(best_by_ablation, on="ablation", how="left")
    selection_counts = (
        predictions.groupby(["candidate_id", "ablation", "selected_expert"], dropna=False, observed=True)
        .agg(rows=("target_date", "count"))
        .reset_index()
        .sort_values(["candidate_id", "rows"], ascending=[True, False])
    )
    return scoreboard, predictions, ablation, selection_counts


def prior_baseline_comparison(scoreboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prior_path = RESEARCH_ROOT / "0035_forecast_revision_momentum_deep_dive" / "artifacts" / "baseline_comparison.csv"
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        for row in prior.itertuples(index=False):
            rows.append(
                {
                    "system": str(row.system),
                    "candidate_id": str(row.candidate_id),
                    "mae": float(row.mae),
                    "rmse": float(row.rmse),
                    "delta_vs_official": float(row.delta_vs_official),
                    "n": int(row.n),
                    "first_date": str(row.first_date),
                    "last_date": str(row.last_date),
                }
            )
    if not scoreboard.empty:
        best = scoreboard.iloc[0]
        rows.append(
            {
                "system": "0036_best_revision_centroid_stack",
                "candidate_id": str(best["candidate_id"]),
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "delta_vs_official": float(best["delta_vs_official_same_rows"]),
                "n": int(best["n"]),
                "first_date": str(best["first_date"]),
                "last_date": str(best["last_date"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)


def write_readme(
    *,
    folder: Path,
    manifest: dict[str, object],
    scoreboard: pd.DataFrame,
    ablation: pd.DataFrame,
    selection_counts: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best = scoreboard.iloc[0] if not scoreboard.empty else None
    if best is None:
        best_text = "No scoreable 0036 stack was produced."
    else:
        best_text = (
            f"Best 0036 stack: `{best['candidate_id']}` with MAE `{best['mae']:.4f}`, "
            f"RMSE `{best['rmse']:.4f}`, and official delta "
            f"`{best['delta_vs_official_same_rows']:.4f}`."
        )
    readme = f"""# Revision-Centroid Stack Ablation

Generated: `{manifest['generated_at_utc']}`

## Purpose

`0033` found targeted smooth forecast-jump specialists, `0034` found complementary failure-neighbor centroid blend signal, and `0035` decomposed revision/range-widening subfamilies. This run tests whether those three families compound when stacked with strict fold-local prior performance selection and explicit ablation.

## Data Window

Rows used: `{manifest['official_rows']}` official forecast/target rows.

Date range: `{manifest['first_target_date']}` to `{manifest['last_target_date']}`.

Source counts: `{manifest['source_counts']}`.

## Leakage Contract

- All scored rows are earlier than `{CONFIRMATION_START.date()}`.
- Stack selection and stack weights use only rows with `target_date < current target_date`.
- Same-date rows from another source family are explicitly excluded from prior performance history.
- Same-source variants additionally restrict prior history to the current source family.
- 2024+ confirmation rows are not loaded or scored.
- Prior blend experts from 0033/0034/0035 are treated as already generated prior-only artifacts, then selected again through this stricter stack gate.

## Main Result

{best_text}

## Baseline Comparison

{markdown_table(comparison, max_rows=20)}

## Ablation Summary

{markdown_table(ablation, max_rows=30)}

## Stack Scoreboard

{markdown_table(scoreboard.head(40), max_rows=40)}

## Selection Counts

{markdown_table(selection_counts.head(50), max_rows=50)}

## Interpretation

This is an explicit compounding test. If the best stack beats `0034`, then the recent forecast-jump, centroid, and revision subfamilies contain complementary deployable information. If it does not, the evidence says the families are mostly overlapping under the current non-contiguous forecast archive, and the next high-value step is to promote the growing 2005+ official forecast backfill before adding more second-stage stack complexity.
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Revision-Centroid Stack Ablation\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_revision_centroid_stack_ablation.py`:

- `{FOLDER_NAME}`: strict date-prior stack and ablation over 0033 smooth forecast-jump specialists, 0034 centroid specialists/blends, and 0035 revision-momentum specialists/blends.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Expert columns | {manifest['expert_columns']} |
| Stack candidates | {manifest['stack_candidates']} |
| Best stack MAE | {manifest['best_stack_mae']} |
| Best stack RMSE | {manifest['best_stack_rmse']} |
| Best stack delta vs official | {manifest['best_stack_delta_vs_official']} |
| Current overall best MAE after 0036 | {manifest['current_overall_best_mae']} |

Leakage contract: all scored rows are before `{CONFIRMATION_START.date()}`; expert selection and weights use only `target_date < current target_date`, excluding same-date rows even across source families.
"""
    write_text(index_path, text)


def write_outputs(
    *,
    frame: pd.DataFrame,
    mapping: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
    ablation: pd.DataFrame,
    selection_counts: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    comparison = prior_baseline_comparison(scoreboard)

    write_csv(artifacts / "expert_mapping.csv", mapping)
    write_csv(artifacts / "stack_scoreboard.csv", scoreboard)
    write_csv(artifacts / "stack_predictions.csv", predictions)
    write_csv(artifacts / "ablation_summary.csv", ablation)
    write_csv(artifacts / "selection_counts.csv", selection_counts)
    write_csv(artifacts / "baseline_comparison.csv", comparison)
    top_ids = scoreboard.head(8)["candidate_id"].to_list() if not scoreboard.empty else []
    write_csv(
        artifacts / "top_stack_predictions.csv",
        predictions[predictions["candidate_id"].isin(top_ids)].copy() if top_ids else predictions.head(0),
    )

    best = scoreboard.iloc[0] if not scoreboard.empty else None
    overall_best = comparison.iloc[0] if not comparison.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "source_counts": {str(k): int(v) for k, v in frame["forecast_source_family"].value_counts().to_dict().items()},
        "expert_columns": int(len([col for col in frame.columns if col not in {"target_date", "forecast_source_family", "target_tmax_c", "official_raw"}])),
        "stack_candidates": int(len(scoreboard)),
        "best_stack": "" if best is None else str(best["candidate_id"]),
        "best_stack_mae": None if best is None else float(best["mae"]),
        "best_stack_rmse": None if best is None else float(best["rmse"]),
        "best_stack_delta_vs_official": None if best is None else float(best["delta_vs_official_same_rows"]),
        "current_overall_best": "" if overall_best is None else str(overall_best["system"]),
        "current_overall_best_mae": None if overall_best is None else float(overall_best["mae"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "revision_centroid_stack_ablation_manifest.json", manifest)
    write_readme(
        folder=folder,
        manifest=manifest,
        scoreboard=scoreboard,
        ablation=ablation,
        selection_counts=selection_counts,
        comparison=comparison,
    )
    update_master_index(manifest)
    return manifest


def run() -> dict[str, object]:
    frame, mapping = build_stack_frame()
    require_no_confirmation_dates(frame["target_date"], context="0036 stack frame")
    scoreboard, predictions, ablation, selection_counts = run_stack_screen(frame, mapping)
    require_no_confirmation_dates(predictions["target_date"], context="0036 stack predictions")
    return write_outputs(
        frame=frame,
        mapping=mapping,
        scoreboard=scoreboard,
        predictions=predictions,
        ablation=ablation,
        selection_counts=selection_counts,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 revision-centroid stack ablation.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
