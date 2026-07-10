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
from scripts.run_hkg_t24_prior_calibrated_fusion_screen import (  # noqa: E402
    WEIGHT_GRID,
    blend_prediction,
    load_common_frame,
    score_prediction,
    weight_errors,
)
from scripts.run_hkg_t24_station_official_family_router import (  # noqa: E402
    LATE_EVAL_START,
    absdiff_bucket,
    active_count_bucket,
    signeddiff_bucket,
)

FOLDER_NAME = "0069_era_source_aware_fusion_model"
ARTIFACT_0068 = RESEARCH_ROOT / "0068_prior_calibrated_fusion_screen" / "artifacts"
SUMMARY_0068_PATH = ARTIFACT_0068 / "summary.json"


@dataclass(frozen=True)
class EraSourceFusionSpec:
    candidate_id: str
    mode: str
    candidate_class: str
    primary_group_mode: str
    secondary_group_mode: str
    min_history: int
    fallback_weight: float
    fallback_mode: str
    temperature_c: float
    primary_alpha: float
    secondary_alpha: float
    global_alpha: float
    press_weight: float
    rss_weight: float
    tilt_mode: str
    tilt_step: float
    cap_low: float
    cap_high: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def source_base_weight(row: pd.Series, spec: EraSourceFusionSpec) -> float:
    source = str(row.get("forecast_source_family", ""))
    if source == "press_archive":
        return spec.press_weight
    if source == "rss_archive":
        return spec.rss_weight
    return spec.fallback_weight


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
    if group_mode == "source_signeddiff_active":
        return f"{source}|{signed_bucket}|{active_bucket}"
    if group_mode == "source_absdiff_active":
        return f"{source}|{abs_bucket}|{active_bucket}"
    raise ValueError(f"Unsupported group mode: {group_mode}")


def tilt_adjustment(row: pd.Series, tilt_mode: str, tilt_step: float) -> float:
    if tilt_mode == "none":
        return 0.0

    absdiff = float(row["abs_family_disagreement_c"])
    signed_bucket = signeddiff_bucket(float(row["family_disagreement_c"]))
    active_bucket = active_count_bucket(float(row["active_member_count"]))
    active_good = active_bucket in {
        "station_stack_two_members",
        "station_stack_three_plus_members",
    }
    inactive = active_bucket == "station_stack_inactive"

    if tilt_mode == "active_more":
        if active_good:
            return tilt_step
        if inactive:
            return -tilt_step
        return 0.0
    if tilt_mode == "large_diff_less":
        if absdiff > 2.5:
            return -tilt_step
        if absdiff <= 0.75:
            return tilt_step / 2.0
        return 0.0
    if tilt_mode == "station_warmer_less":
        if signed_bucket == "station_warmer_ge_1c":
            return -tilt_step
        if signed_bucket == "station_cooler_ge_1c":
            return tilt_step
        return 0.0
    if tilt_mode == "station_cooler_less":
        if signed_bucket == "station_cooler_ge_1c":
            return -tilt_step
        if signed_bucket == "station_warmer_ge_1c":
            return tilt_step
        return 0.0
    if tilt_mode == "activity_diff_guard":
        if active_good and absdiff <= 1.5:
            return tilt_step
        if inactive or absdiff > 2.5:
            return -tilt_step
        return 0.0
    raise ValueError(f"Unsupported tilt mode: {tilt_mode}")


def clipped_weight(value: float, spec: EraSourceFusionSpec) -> float:
    return float(np.clip(value, spec.cap_low, spec.cap_high))


def fallback_for_row(row: pd.Series, spec: EraSourceFusionSpec) -> float:
    if spec.fallback_mode == "constant":
        return spec.fallback_weight
    if spec.fallback_mode == "source_map":
        return clipped_weight(source_base_weight(row, spec), spec)
    raise ValueError(f"Unsupported fallback mode: {spec.fallback_mode}")


def select_prior_weight(
    *,
    abs_sums: np.ndarray,
    count: int,
    mode: str,
    min_history: int,
    fallback_weight: float,
    temperature_c: float,
) -> float:
    if count < min_history:
        return fallback_weight
    prior_mae = abs_sums / count
    if mode == "prior_best_weight":
        return float(WEIGHT_GRID[int(np.argmin(prior_mae))])
    if mode == "prior_soft_weight":
        centered = prior_mae - float(np.min(prior_mae))
        raw = np.exp(-centered / temperature_c)
        probs = raw / raw.sum()
        return float(np.sum(np.array(WEIGHT_GRID) * probs))
    raise ValueError(f"Unsupported prior selector mode: {mode}")


def source_fixed_specs() -> list[EraSourceFusionSpec]:
    specs: list[EraSourceFusionSpec] = []
    for press_weight in (0.22, 0.25, 0.28, 0.30):
        for rss_weight in (0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25):
            candidate_id = (
                f"diagnostic_source_fixed_press{press_weight}_rss{rss_weight}"
                .replace(".", "p")
            )
            specs.append(
                EraSourceFusionSpec(
                    candidate_id=candidate_id,
                    mode="source_fixed_weight",
                    candidate_class="diagnostic_source_map",
                    primary_group_mode="source",
                    secondary_group_mode="global",
                    min_history=0,
                    fallback_weight=0.15,
                    fallback_mode="source_map",
                    temperature_c=0.0,
                    primary_alpha=0.0,
                    secondary_alpha=0.0,
                    global_alpha=0.0,
                    press_weight=press_weight,
                    rss_weight=rss_weight,
                    tilt_mode="none",
                    tilt_step=0.0,
                    cap_low=0.0,
                    cap_high=0.50,
                )
            )
    return specs


def source_tilt_specs() -> list[EraSourceFusionSpec]:
    specs: list[EraSourceFusionSpec] = []
    base_pairs = [(0.28, 0.15), (0.25, 0.15), (0.28, 0.18), (0.22, 0.15)]
    tilt_modes = [
        "active_more",
        "large_diff_less",
        "station_warmer_less",
        "station_cooler_less",
        "activity_diff_guard",
    ]
    for press_weight, rss_weight in base_pairs:
        for tilt_mode in tilt_modes:
            for tilt_step in (0.03, 0.05, 0.08):
                candidate_id = (
                    f"diagnostic_source_tilt_press{press_weight}_rss{rss_weight}"
                    f"_{tilt_mode}_s{tilt_step}"
                ).replace(".", "p")
                specs.append(
                    EraSourceFusionSpec(
                        candidate_id=candidate_id,
                        mode="source_fixed_weight",
                        candidate_class="diagnostic_source_tilt",
                        primary_group_mode="source",
                        secondary_group_mode="global",
                        min_history=0,
                        fallback_weight=0.15,
                        fallback_mode="source_map",
                        temperature_c=0.0,
                        primary_alpha=0.0,
                        secondary_alpha=0.0,
                        global_alpha=0.0,
                        press_weight=press_weight,
                        rss_weight=rss_weight,
                        tilt_mode=tilt_mode,
                        tilt_step=tilt_step,
                        cap_low=0.0,
                        cap_high=0.50,
                    )
                )
    return specs


def prior_mixture_specs() -> list[EraSourceFusionSpec]:
    specs: list[EraSourceFusionSpec] = []
    primary_groups = [
        "source",
        "source_signeddiff",
        "source_active_count",
        "source_signeddiff_active",
    ]
    fallback_options = [
        ("constant", 0.15, 0.22, 0.15),
        ("constant", 0.22, 0.22, 0.15),
        ("source_map", 0.15, 0.28, 0.15),
    ]
    alpha_options = [
        (1.00, 0.00, 0.00),
        (0.75, 0.25, 0.00),
    ]
    for group_mode in primary_groups:
        for min_history in (30, 120):
            for fallback_mode, fallback_weight, press_weight, rss_weight in fallback_options:
                for primary_alpha, secondary_alpha, global_alpha in alpha_options:
                    for mode, temperature in [
                        ("prior_best_weight", 0.0),
                        ("prior_soft_weight", 0.02),
                    ]:
                        selector = "best" if mode == "prior_best_weight" else f"soft{temperature}"
                        candidate_class = (
                            "causal_prior_selector"
                            if fallback_mode == "constant"
                            else "diagnostic_source_fallback_prior"
                        )
                        candidate_id = (
                            f"{candidate_class}_{selector}_{group_mode}_h{min_history}"
                            f"_fb{fallback_mode}{fallback_weight}_a"
                            f"{primary_alpha}_{secondary_alpha}_{global_alpha}"
                        ).replace(".", "p")
                        specs.append(
                            EraSourceFusionSpec(
                                candidate_id=candidate_id,
                                mode=mode,
                                candidate_class=candidate_class,
                                primary_group_mode=group_mode,
                                secondary_group_mode="source",
                                min_history=min_history,
                                fallback_weight=fallback_weight,
                                fallback_mode=fallback_mode,
                                temperature_c=temperature,
                                primary_alpha=primary_alpha,
                                secondary_alpha=secondary_alpha,
                                global_alpha=global_alpha,
                                press_weight=press_weight,
                                rss_weight=rss_weight,
                                tilt_mode="none",
                                tilt_step=0.0,
                                cap_low=0.0,
                                cap_high=0.50,
                            )
                        )
    return specs


def fusion_specs() -> list[EraSourceFusionSpec]:
    specs = source_fixed_specs() + source_tilt_specs() + prior_mixture_specs()
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("0069 fusion candidate IDs are not unique")
    return specs


def apply_source_fixed_spec(frame: pd.DataFrame, spec: EraSourceFusionSpec) -> pd.DataFrame:
    weights = []
    router_groups = []
    for _idx, row in frame.iterrows():
        weight = source_base_weight(row, spec) + tilt_adjustment(
            row,
            spec.tilt_mode,
            spec.tilt_step,
        )
        weights.append(clipped_weight(weight, spec))
        router_groups.append(group_key(row, spec.primary_group_mode))
    out = frame[["target_date", "target_tmax_c", "fold_id", "forecast_source_family"]].copy()
    out["station_weight"] = np.array(weights, dtype=float)
    out["candidate_prediction_c"] = blend_prediction(frame, out["station_weight"].to_numpy(dtype=float))
    out["primary_prior_count"] = 0
    out["secondary_prior_count"] = 0
    out["global_prior_count"] = 0
    out["router_group"] = router_groups
    return out


def state_key(group_mode: str, key: str) -> str:
    return f"{group_mode}::{key}"


def apply_prior_mixture_spec(frame: pd.DataFrame, spec: EraSourceFusionSpec) -> pd.DataFrame:
    state: dict[str, dict[str, object]] = defaultdict(
        lambda: {"count": 0, "abs_sums": np.zeros(len(WEIGHT_GRID), dtype=float)}
    )
    weights: list[float] = []
    primary_counts: list[int] = []
    secondary_counts: list[int] = []
    global_counts: list[int] = []
    router_groups: list[str] = []

    for _idx, row in frame.iterrows():
        fallback = fallback_for_row(row, spec)
        keys = {
            "primary": state_key(spec.primary_group_mode, group_key(row, spec.primary_group_mode)),
            "secondary": state_key(spec.secondary_group_mode, group_key(row, spec.secondary_group_mode)),
            "global": state_key("global", "global"),
        }
        selected: dict[str, float] = {}
        counts: dict[str, int] = {}
        for name, key in keys.items():
            group_state = state[key]
            count = int(group_state["count"])
            abs_sums = np.asarray(group_state["abs_sums"], dtype=float)
            selected[name] = select_prior_weight(
                abs_sums=abs_sums,
                count=count,
                mode=spec.mode,
                min_history=spec.min_history,
                fallback_weight=fallback,
                temperature_c=spec.temperature_c,
            )
            counts[name] = count
        weight = (
            spec.primary_alpha * selected["primary"]
            + spec.secondary_alpha * selected["secondary"]
            + spec.global_alpha * selected["global"]
        )
        weights.append(clipped_weight(weight, spec))
        primary_counts.append(counts["primary"])
        secondary_counts.append(counts["secondary"])
        global_counts.append(counts["global"])
        router_groups.append(keys["primary"])

        errors = weight_errors(row)
        for key in set(keys.values()):
            group_state = state[key]
            group_state["abs_sums"] = np.asarray(group_state["abs_sums"], dtype=float) + errors
            group_state["count"] = int(group_state["count"]) + 1

    out = frame[["target_date", "target_tmax_c", "fold_id", "forecast_source_family"]].copy()
    out["station_weight"] = np.array(weights, dtype=float)
    out["candidate_prediction_c"] = blend_prediction(frame, out["station_weight"].to_numpy(dtype=float))
    out["primary_prior_count"] = primary_counts
    out["secondary_prior_count"] = secondary_counts
    out["global_prior_count"] = global_counts
    out["router_group"] = router_groups
    return out


def apply_spec(frame: pd.DataFrame, spec: EraSourceFusionSpec) -> pd.DataFrame:
    if spec.mode == "source_fixed_weight":
        out = apply_source_fixed_spec(frame, spec)
    elif spec.mode in {"prior_best_weight", "prior_soft_weight"}:
        out = apply_prior_mixture_spec(frame, spec)
    else:
        raise ValueError(f"Unsupported 0069 mode: {spec.mode}")
    out["candidate_id"] = spec.candidate_id
    out["mode"] = spec.mode
    out["candidate_class"] = spec.candidate_class
    out["primary_group_mode"] = spec.primary_group_mode
    out["secondary_group_mode"] = spec.secondary_group_mode
    out["min_history"] = spec.min_history
    out["fallback_weight"] = spec.fallback_weight
    out["fallback_mode"] = spec.fallback_mode
    out["temperature_c"] = spec.temperature_c
    return out


def score_candidate(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    best_0068_mae: float,
) -> dict[str, object]:
    pred_values = pd.to_numeric(predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float)
    score = score_prediction(frame, pred_values)
    official_score = score_prediction(
        frame,
        pd.to_numeric(frame["official_family_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    station_score = score_prediction(
        frame,
        pd.to_numeric(frame["station_family_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    late_mask = predictions["target_date"].ge(LATE_EVAL_START)
    late_score = score_prediction(frame.loc[late_mask].copy(), pred_values[late_mask.to_numpy()])
    late_official = score_prediction(
        frame.loc[late_mask].copy(),
        pd.to_numeric(
            frame.loc[late_mask, "official_family_prediction_c"],
            errors="coerce",
        ).to_numpy(dtype=float),
    )
    fold_deltas: list[float] = []
    source_deltas: list[float] = []
    for _fold_id, fold_predictions in predictions.groupby("fold_id", observed=True):
        fold_frame = frame.loc[fold_predictions.index].copy()
        fold_score = score_prediction(
            fold_frame,
            pd.to_numeric(fold_predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        fold_official = score_prediction(
            fold_frame,
            pd.to_numeric(fold_frame["official_family_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        fold_deltas.append(float(fold_score["mae"]) - float(fold_official["mae"]))
    for _source, source_predictions in predictions.groupby("forecast_source_family", observed=True):
        source_frame = frame.loc[source_predictions.index].copy()
        source_score = score_prediction(
            source_frame,
            pd.to_numeric(source_predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        source_official = score_prediction(
            source_frame,
            pd.to_numeric(source_frame["official_family_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        source_deltas.append(float(source_score["mae"]) - float(source_official["mae"]))

    row: dict[str, object] = {
        "candidate_id": str(predictions["candidate_id"].iloc[0]),
        "mode": str(predictions["mode"].iloc[0]),
        "candidate_class": str(predictions["candidate_class"].iloc[0]),
        "primary_group_mode": str(predictions["primary_group_mode"].iloc[0]),
        "secondary_group_mode": str(predictions["secondary_group_mode"].iloc[0]),
        "min_history": int(predictions["min_history"].iloc[0]),
        "fallback_weight": float(predictions["fallback_weight"].iloc[0]),
        "fallback_mode": str(predictions["fallback_mode"].iloc[0]),
        "temperature_c": float(predictions["temperature_c"].iloc[0]),
        "n": score["n"],
        "mae": score["mae"],
        "rmse": score["rmse"],
        "bias": score["bias"],
        "official_mae": official_score["mae"],
        "station_mae": station_score["mae"],
        "delta_mae_vs_official": float(score["mae"]) - float(official_score["mae"]),
        "delta_mae_vs_station": float(score["mae"]) - float(station_score["mae"]),
        "delta_mae_vs_0068": float(score["mae"]) - best_0068_mae,
        "late_n": late_score["n"],
        "late_mae": late_score["mae"],
        "late_delta_mae_vs_official": float(late_score["mae"]) - float(late_official["mae"]),
        "fold_delta_max_vs_official": max(fold_deltas) if fold_deltas else math.nan,
        "fold_delta_min_vs_official": min(fold_deltas) if fold_deltas else math.nan,
        "source_delta_max_vs_official": max(source_deltas) if source_deltas else math.nan,
        "source_delta_min_vs_official": min(source_deltas) if source_deltas else math.nan,
        "folds_improved_vs_official": int(sum(delta < 0 for delta in fold_deltas)),
        "sources_improved_vs_official": int(sum(delta < 0 for delta in source_deltas)),
        "mean_station_weight": float(pd.to_numeric(predictions["station_weight"], errors="coerce").mean()),
        "press_mean_station_weight": float(
            pd.to_numeric(
                predictions.loc[
                    predictions["forecast_source_family"].eq("press_archive"),
                    "station_weight",
                ],
                errors="coerce",
            ).mean()
        ),
        "rss_mean_station_weight": float(
            pd.to_numeric(
                predictions.loc[
                    predictions["forecast_source_family"].eq("rss_archive"),
                    "station_weight",
                ],
                errors="coerce",
            ).mean()
        ),
        "weight_std": float(pd.to_numeric(predictions["station_weight"], errors="coerce").std(ddof=0)),
    }
    row["promotion_gate_passed"] = bool(
        float(row["delta_mae_vs_official"]) <= -0.001
        and float(row["fold_delta_max_vs_official"]) <= 0.0
        and float(row["late_delta_mae_vs_official"]) <= 0.0
    )
    row["beats_0068_best"] = bool(float(row["delta_mae_vs_0068"]) <= -0.0005)
    row["deployable_gate_passed"] = bool(
        row["promotion_gate_passed"]
        and row["beats_0068_best"]
        and str(row["candidate_class"]) == "causal_prior_selector"
    )
    return row


def score_all_specs(
    frame: pd.DataFrame,
    specs: list[EraSourceFusionSpec],
    *,
    best_0068_mae: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs:
        predictions = apply_spec(frame, spec)
        rows.append(score_candidate(frame, predictions, best_0068_mae=best_0068_mae))
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(rows).sort_values(
        [
            "deployable_gate_passed",
            "beats_0068_best",
            "promotion_gate_passed",
            "mae",
            "fold_delta_max_vs_official",
        ],
        ascending=[False, False, False, True, True],
    )
    top_ids = set(scoreboard["candidate_id"].head(30).astype(str))
    selected_predictions = pd.concat(
        [
            predictions
            for predictions in prediction_frames
            if str(predictions["candidate_id"].iloc[0]) in top_ids
        ],
        ignore_index=True,
    )
    require_no_confirmation_dates(selected_predictions["target_date"], context="0069 selected predictions")
    return scoreboard.reset_index(drop=True), selected_predictions


def per_source_scoreboard(frame: pd.DataFrame, top_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate_id, predictions in top_predictions.groupby("candidate_id", observed=True):
        for source, source_predictions in predictions.groupby("forecast_source_family", observed=True):
            source_frame = source_predictions[
                ["target_date", "candidate_prediction_c", "station_weight"]
            ].merge(
                frame[
                    [
                        "target_date",
                        "target_tmax_c",
                        "official_family_prediction_c",
                    ]
                ],
                on="target_date",
                how="inner",
                validate="one_to_one",
            )
            candidate_score = score_prediction(
                source_frame,
                pd.to_numeric(
                    source_frame["candidate_prediction_c"],
                    errors="coerce",
                ).to_numpy(dtype=float),
            )
            official_score = score_prediction(
                source_frame,
                pd.to_numeric(
                    source_frame["official_family_prediction_c"],
                    errors="coerce",
                ).to_numpy(dtype=float),
            )
            rows.append(
                {
                    "candidate_id": str(candidate_id),
                    "forecast_source_family": str(source),
                    "n": candidate_score["n"],
                    "mae": candidate_score["mae"],
                    "rmse": candidate_score["rmse"],
                    "official_mae": official_score["mae"],
                    "delta_mae_vs_official": float(candidate_score["mae"])
                    - float(official_score["mae"]),
                    "mean_station_weight": float(
                        pd.to_numeric(
                            source_predictions["station_weight"],
                            errors="coerce",
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["candidate_id", "forecast_source_family"],
    ).reset_index(drop=True)


def leakage_audit(
    frame: pd.DataFrame,
    scoreboard: pd.DataFrame,
    summary_0068: dict[str, Any],
) -> pd.DataFrame:
    deployable = scoreboard[scoreboard["deployable_gate_passed"].astype(bool)].copy()
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
            "check_id": "source_and_fold_confounding_documented",
            "passed": bool(
                frame.groupby("forecast_source_family", observed=True)["fold_id"].nunique().max() == 1
            ),
            "evidence": "current common frame has press_archive in 2000-2005 and rss_archive in 2018-2023",
        },
        {
            "check_id": "prior_selectors_update_after_scoring",
            "passed": True,
            "evidence": "online prior states update only after each row prediction is chosen",
        },
        {
            "check_id": "diagnostic_source_maps_not_marked_deployable",
            "passed": bool(
                scoreboard.loc[
                    scoreboard["candidate_class"].ne("causal_prior_selector"),
                    "deployable_gate_passed",
                ].eq(False).all()
            ),
            "evidence": "source-specific fixed maps are reported as diagnostics, not deployable champions",
        },
        {
            "check_id": "deployable_gate_requires_beating_0068",
            "passed": bool(
                deployable.empty
                or deployable["mae"].le(float(summary_0068["best_mae"]) - 0.0005).all()
            ),
            "evidence": f"{len(deployable)} deployable candidates beat 0068 by at least 0.0005 C MAE",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    definitions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    source_scores: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    deployable = scoreboard[scoreboard["deployable_gate_passed"].astype(bool)].copy()
    diagnostic = scoreboard[scoreboard["candidate_class"].ne("causal_prior_selector")].head(20).copy()
    return f"""# Era/Source-Aware Fusion Model

Generated: `{summary['generated_at_utc']}`

## Purpose

`0068` proved that the best station/official blend weight is not stable across the currently scoreable official forecast archive. Press/archive rows prefer about `0.28` station weight, while RSS rows prefer about `0.15`. `0069` tests that finding more explicitly.

This is not a Polymarket, trading, machine-learning, or 2024+ confirmation experiment. It is a leakage-safe forecast-system diagnostic using the currently available scoreable official forecast rows while the longer RSS/forecast backfill continues outside this run.

## Data Contract

- Input frame: `0067_station_official_family_router/artifacts/common_frame.csv`.
- Current common rows: `{summary['common_rows']}`.
- Date range: `{summary['first_date']}` to `{summary['last_date']}`.
- Current source split: press/archive rows are 2000-2005; RSS rows are 2018-2023.
- No 2024+ rows are used.
- The current official archive remains non-contiguous, so source family and era/fold are confounded in this frame.

## What Was Tested

1. Diagnostic source maps: fixed press/RSS station weights, such as press `0.28` and RSS `0.15`.
2. Diagnostic source maps with pre-target tilts from family disagreement and station-stack activity.
3. Causal online prior selectors: weights selected from past rows only, using source, disagreement, active-count, and global prior states.
4. Source-map fallback prior selectors: useful for diagnosis, but not marked deployable because their RSS fallback encodes a source-specific assumption.

## Headline

| Item | Value |
|---|---:|
| Official baseline MAE | {summary['official_baseline_mae']} |
| 0068 best MAE | {summary['best_0068_mae']} |
| Best 0069 candidate | {summary['best_candidate']} |
| Best 0069 class | {summary['best_candidate_class']} |
| Best 0069 MAE | {summary['best_mae']} |
| Best 0069 RMSE | {summary['best_rmse']} |
| Best delta vs 0068 | {summary['best_delta_mae_vs_0068']} |
| Best deployable candidate | {summary['best_deployable_candidate']} |
| Best deployable MAE | {summary['best_deployable_mae']} |
| Deployable beats 0068 | {summary['deployable_beats_0068']} |
| Candidates tested | {summary['candidate_count']} |
| Promoted candidates | {summary['promoted_candidate_count']} |
| Deployable promoted candidates | {summary['deployable_candidate_count']} |

## Interpretation

The main question is whether the source/era weight difference can be used safely. A fixed source map can score well, but it is diagnostic because source and era are confounded and the RSS side currently has only the available 2018-2023 block. A causal prior selector is more deployable because it updates from previous target dates only. If the diagnostic map beats the causal selector, the result is still valuable: it says the missing continuous official forecast backfill matters because it would let us learn the RSS-era weight without hard-coding it.

## Scoreboard

{markdown_table(scoreboard, max_rows=100)}

## Deployable Candidates

{markdown_table(deployable, max_rows=80)}

## Diagnostic Source Maps

{markdown_table(diagnostic, max_rows=40)}

## Per-Source Top-Candidate Scores

{markdown_table(source_scores, max_rows=100)}

## Candidate Definitions

{markdown_table(definitions, max_rows=120)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Files

- `artifacts/candidate_definitions.csv`
- `artifacts/scoreboard.csv`
- `artifacts/top_predictions.csv`
- `artifacts/per_source_scoreboard.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_era_source_aware_fusion_model.py`:

- `{FOLDER_NAME}`: source/era-aware official/station fusion diagnostics and causal prior-selector screen.

| Metric | Value |
|---|---:|
| Official baseline MAE | {summary['official_baseline_mae']} |
| 0068 best MAE | {summary['best_0068_mae']} |
| Best 0069 candidate | {summary['best_candidate']} |
| Best 0069 MAE | {summary['best_mae']} |
| Best 0069 class | {summary['best_candidate_class']} |
| Best deployable candidate | {summary['best_deployable_candidate']} |
| Best deployable MAE | {summary['best_deployable_mae']} |

Leakage contract: no 2024+ rows; causal prior selectors update after each row is scored; source-map fixed weights are diagnostic only.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Era/Source-Aware Fusion Model",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_era_source_aware_fusion_model.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Input | `0067` common station/official frame plus `0068` baseline | Tested |
| Common rows | `{summary['common_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Non-contiguous |
| Current source split | press/archive 2000-2005; RSS 2018-2023 | Source/era confounded |
| Official baseline MAE / RMSE | `{summary['official_baseline_mae']}` / `{summary['official_baseline_rmse']}` | Baseline |
| 0068 best MAE / RMSE | `{summary['best_0068_mae']}` / `{summary['best_0068_rmse']}` | Baseline |
| Best 0069 candidate | `{summary['best_candidate']}` | Tested |
| Best 0069 class | `{summary['best_candidate_class']}` | Diagnostic/deployable classification |
| Best 0069 MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best delta vs 0068 | `{summary['best_delta_mae_vs_0068']}` | Fusion refinement |
| Best deployable candidate | `{summary['best_deployable_candidate']}` | Prior-only selector |
| Best deployable MAE | `{summary['best_deployable_mae']}` | Pre-2024 only |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0069` confirms whether source/era-specific station weights add value and separates post-hoc source-map diagnostics from deployable prior-only weight selection.
"""
    update_markdown_section(
        path,
        heading="Era/Source-Aware Fusion Model",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"53. Era/source-aware fusion screened `{summary['candidate_count']}` candidates; "
        f"best overall delta vs 0068 is `{summary['best_delta_mae_vs_0068']}` from "
        f"`{summary['best_candidate']}`, while best deployable prior-only candidate is "
        f"`{summary['best_deployable_candidate']}` with MAE `{summary['best_deployable_mae']}`."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue while the forecast backfill runs: use the current 0069 source/era split to build a nonlinear local residual-fusion lab that tests station-weight triggers from disagreement shape, official forecast range, source metadata, and station-stack activity, while keeping diagnostic source maps separate from deployable prior-only selectors.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, _summary_0067 = load_common_frame()
    summary_0068 = load_json(SUMMARY_0068_PATH)
    best_0068_mae = float(summary_0068["best_mae"])
    best_0068_rmse = float(summary_0068["best_rmse"])
    specs = fusion_specs()
    definitions = pd.DataFrame([spec.__dict__ for spec in specs])
    scoreboard, top_predictions = score_all_specs(frame, specs, best_0068_mae=best_0068_mae)
    source_scores = per_source_scoreboard(frame, top_predictions)
    leakage = leakage_audit(frame, scoreboard, summary_0068)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0069 leakage audit failed: {failed}")

    official_score = score_prediction(
        frame,
        pd.to_numeric(frame["official_family_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    best = scoreboard.iloc[0]
    deployable = scoreboard[scoreboard["candidate_class"].eq("causal_prior_selector")].copy()
    deployable = deployable.sort_values(["mae", "fold_delta_max_vs_official"]).reset_index(drop=True)
    best_deployable = deployable.iloc[0]
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "common_rows": int(len(frame)),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "candidate_count": int(len(scoreboard)),
        "promoted_candidate_count": int(scoreboard["promotion_gate_passed"].astype(bool).sum()),
        "deployable_candidate_count": int(scoreboard["deployable_gate_passed"].astype(bool).sum()),
        "official_baseline_mae": float(official_score["mae"]),
        "official_baseline_rmse": float(official_score["rmse"]),
        "best_0068_candidate": str(summary_0068["best_candidate"]),
        "best_0068_mae": best_0068_mae,
        "best_0068_rmse": best_0068_rmse,
        "best_candidate": str(best["candidate_id"]),
        "best_candidate_class": str(best["candidate_class"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_official": float(best["delta_mae_vs_official"]),
        "best_delta_mae_vs_0068": float(best["delta_mae_vs_0068"]),
        "best_late_delta_mae_vs_official": float(best["late_delta_mae_vs_official"]),
        "best_mean_station_weight": float(best["mean_station_weight"]),
        "best_press_mean_station_weight": float(best["press_mean_station_weight"]),
        "best_rss_mean_station_weight": float(best["rss_mean_station_weight"]),
        "best_deployable_candidate": str(best_deployable["candidate_id"]),
        "best_deployable_mae": float(best_deployable["mae"]),
        "best_deployable_rmse": float(best_deployable["rmse"]),
        "best_deployable_delta_mae_vs_0068": float(best_deployable["delta_mae_vs_0068"]),
        "deployable_beats_0068": bool(float(best_deployable["delta_mae_vs_0068"]) <= -0.0005),
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
    write_csv(artifacts / "per_source_scoreboard.csv", source_scores)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "era_source_aware_fusion_model_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            definitions=definitions,
            scoreboard=scoreboard,
            source_scores=source_scores,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Screen leakage-safe era/source-aware station/official fusion weights."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
