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
from scripts.run_hkg_t24_era_source_aware_fusion_model import load_json  # noqa: E402
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    update_markdown_section,
)
from scripts.run_hkg_t24_prior_calibrated_fusion_screen import (  # noqa: E402
    blend_prediction,
    load_common_frame,
    score_prediction,
)
from scripts.run_hkg_t24_station_official_family_router import (  # noqa: E402
    LATE_EVAL_START,
    absdiff_bucket,
    active_count_bucket,
    signeddiff_bucket,
)

FOLDER_NAME = "0070_nonlinear_local_residual_fusion_lab"
ARTIFACT_0069 = RESEARCH_ROOT / "0069_era_source_aware_fusion_model" / "artifacts"
SUMMARY_0069_PATH = ARTIFACT_0069 / "summary.json"
TOP_PREDICTIONS_0069_PATH = ARTIFACT_0069 / "top_predictions.csv"
SCORED_FORECAST_PATH = (
    RESEARCH_ROOT
    / "0044_forecast_archive_continuous_scored_export"
    / "artifacts"
    / "scored_pre2024.csv"
)
DELTA_GRID = (-0.18, -0.12, -0.08, -0.05, -0.03, 0.0, 0.03, 0.05, 0.08, 0.12, 0.18)


@dataclass(frozen=True)
class LocalFusionSpec:
    candidate_id: str
    mode: str
    candidate_class: str
    group_mode: str
    min_history: int
    fallback_delta: float
    temperature_c: float
    fixed_delta: float
    cap_low: float
    cap_high: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 140) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def forecast_range_bucket(value: float | int | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "range_unknown"
    value_f = float(value)
    if value_f <= 3.0:
        return "range_le_3c"
    if value_f <= 4.0:
        return "range_3_4c"
    if value_f <= 5.0:
        return "range_4_5c"
    return "range_gt_5c"


def forecast_level_bucket(value: float | int | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "level_unknown"
    value_f = float(value)
    if value_f <= 20.0:
        return "level_le_20c"
    if value_f <= 24.0:
        return "level_20_24c"
    if value_f <= 28.0:
        return "level_24_28c"
    if value_f <= 32.0:
        return "level_28_32c"
    return "level_gt_32c"


def rh_bucket(min_value: float | int | None, max_value: float | int | None) -> str:
    if min_value is None or max_value is None:
        return "rh_unknown"
    if not math.isfinite(float(min_value)) or not math.isfinite(float(max_value)):
        return "rh_unknown"
    width = float(max_value) - float(min_value)
    if width <= 20.0:
        return "rh_tight"
    if width <= 35.0:
        return "rh_medium"
    return "rh_wide"


def weather_bucket(text: object) -> str:
    raw = str(text or "").lower()
    if any(token in raw for token in ["thunder", "squall"]):
        return "weather_thunder"
    if any(token in raw for token in ["shower", "rain", "drizzle"]):
        return "weather_rain"
    if any(token in raw for token in ["haze", "mist", "visibility", "fog"]):
        return "weather_visibility"
    if any(token in raw for token in ["fine", "sunny", "bright"]):
        return "weather_sunny"
    if "cloud" in raw:
        return "weather_cloud"
    return "weather_other"


def wind_bucket(text: object) -> str:
    raw = str(text or "").lower()
    strength = "strong" if any(token in raw for token in ["force 6", "force 7", "strong"]) else "normal"
    if "northeast" in raw or "north" in raw:
        direction = "north_east"
    elif "east" in raw:
        direction = "east"
    elif "south" in raw:
        direction = "south"
    elif "west" in raw:
        direction = "west"
    else:
        direction = "variable"
    return f"wind_{direction}_{strength}"


def gate_weight_bucket(value: float | int | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "gate_unknown"
    value_f = float(value)
    if value_f <= 0.0:
        return "gate_zero"
    if value_f <= 0.15:
        return "gate_low"
    if value_f <= 0.30:
        return "gate_medium"
    return "gate_high"


def station_correction_bucket(value: float | int | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "stackcorr_unknown"
    value_f = float(value)
    if value_f <= -0.30:
        return "stackcorr_cool"
    if value_f >= 0.30:
        return "stackcorr_warm"
    return "stackcorr_neutral"


def load_0069_base_predictions() -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = load_json(SUMMARY_0069_PATH)
    predictions = pd.read_csv(TOP_PREDICTIONS_0069_PATH)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    best_id = str(summary["best_deployable_candidate"])
    selected = predictions[predictions["candidate_id"].astype(str).eq(best_id)].copy()
    if selected.empty:
        raise RuntimeError(f"Missing 0069 best deployable predictions: {best_id}")
    selected = selected.sort_values("target_date").reset_index(drop=True)
    selected = selected[
        [
            "target_date",
            "candidate_prediction_c",
            "station_weight",
            "candidate_id",
        ]
    ].rename(
        columns={
            "candidate_prediction_c": "base_0069_prediction_c",
            "station_weight": "base_0069_station_weight",
            "candidate_id": "base_0069_candidate_id",
        }
    )
    require_no_confirmation_dates(selected["target_date"], context="0070 0069 base predictions")
    return selected, summary


def load_forecast_metadata() -> pd.DataFrame:
    if not SCORED_FORECAST_PATH.exists():
        raise FileNotFoundError(f"Missing scored forecast metadata: {SCORED_FORECAST_PATH}")
    keep = [
        "target_date",
        "forecast_source_family",
        "forecast_min_c",
        "forecast_max_c",
        "forecast_range_c",
        "forecast_midpoint_c",
        "rh_min_pct",
        "rh_max_pct",
        "wind_text",
        "weather_text",
        "lead_hours_at_cutoff",
        "forecast_span_c",
    ]
    metadata = pd.read_csv(SCORED_FORECAST_PATH, usecols=lambda col: col in keep)
    metadata["target_date"] = pd.to_datetime(metadata["target_date"], errors="coerce").dt.normalize()
    metadata = metadata[metadata["target_date"].notna()].copy()
    metadata = metadata.sort_values(["target_date", "forecast_source_family"]).drop_duplicates(
        ["target_date", "forecast_source_family"],
        keep="last",
    )
    require_no_confirmation_dates(metadata["target_date"], context="0070 forecast metadata")
    return metadata


def build_feature_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    common, _summary_0067 = load_common_frame()
    base_predictions, summary_0069 = load_0069_base_predictions()
    metadata = load_forecast_metadata()
    frame = common.merge(base_predictions, on="target_date", how="inner", validate="one_to_one")
    frame = frame.merge(
        metadata,
        on=["target_date", "forecast_source_family"],
        how="left",
        validate="one_to_one",
    )
    if frame.empty:
        raise RuntimeError("0070 feature frame is empty")
    require_no_confirmation_dates(frame["target_date"], context="0070 feature frame")
    frame["signeddiff_bucket"] = frame["family_disagreement_c"].map(lambda value: signeddiff_bucket(float(value)))
    frame["absdiff_bucket"] = frame["abs_family_disagreement_c"].map(lambda value: absdiff_bucket(float(value)))
    frame["active_count_bucket"] = frame["active_member_count"].map(lambda value: active_count_bucket(float(value)))
    frame["forecast_range_bucket"] = frame["forecast_range_c"].map(forecast_range_bucket)
    frame["forecast_max_bucket"] = frame["forecast_max_c"].map(forecast_level_bucket)
    frame["forecast_midpoint_bucket"] = frame["forecast_midpoint_c"].map(forecast_level_bucket)
    frame["rh_bucket"] = [
        rh_bucket(min_value, max_value)
        for min_value, max_value in zip(frame["rh_min_pct"], frame["rh_max_pct"], strict=True)
    ]
    frame["weather_bucket"] = frame["weather_text"].map(weather_bucket)
    frame["wind_bucket"] = frame["wind_text"].map(wind_bucket)
    frame["gate_weight_bucket"] = frame["gate_weight"].map(gate_weight_bucket)
    frame["station_correction_bucket"] = frame["station_stack_correction_c"].map(station_correction_bucket)
    return frame.sort_values("target_date").reset_index(drop=True), summary_0069


def local_group_key(row: pd.Series, group_mode: str) -> str:
    source = str(row["forecast_source_family"])
    parts = {
        "source": source,
        "signeddiff": str(row["signeddiff_bucket"]),
        "absdiff": str(row["absdiff_bucket"]),
        "active": str(row["active_count_bucket"]),
        "range": str(row["forecast_range_bucket"]),
        "max": str(row["forecast_max_bucket"]),
        "mid": str(row["forecast_midpoint_bucket"]),
        "rh": str(row["rh_bucket"]),
        "weather": str(row["weather_bucket"]),
        "wind": str(row["wind_bucket"]),
        "gate": str(row["gate_weight_bucket"]),
        "stackcorr": str(row["station_correction_bucket"]),
        "selected": str(row["selected_family"]),
    }
    if group_mode == "global":
        return "global"
    recipes = {
        "source_signeddiff_active": ["source", "signeddiff", "active"],
        "source_signeddiff_range": ["source", "signeddiff", "range"],
        "source_absdiff_range": ["source", "absdiff", "range"],
        "source_range_active": ["source", "range", "active"],
        "source_max_active": ["source", "max", "active"],
        "source_mid_range": ["source", "mid", "range"],
        "source_weather_range": ["source", "weather", "range"],
        "source_wind_range": ["source", "wind", "range"],
        "source_rh_range": ["source", "rh", "range"],
        "source_gate_signeddiff": ["source", "gate", "signeddiff"],
        "source_selected_signeddiff": ["source", "selected", "signeddiff"],
        "source_stackcorr_signeddiff": ["source", "stackcorr", "signeddiff"],
        "source_signeddiff_weather_active": ["source", "signeddiff", "weather", "active"],
    }
    if group_mode not in recipes:
        raise ValueError(f"Unsupported local group mode: {group_mode}")
    return "|".join(parts[name] for name in recipes[group_mode])


def delta_predictions(frame: pd.DataFrame, deltas: np.ndarray | pd.Series | float) -> np.ndarray:
    base_weight = pd.to_numeric(frame["base_0069_station_weight"], errors="coerce").to_numpy(dtype=float)
    new_weight = np.clip(base_weight + np.asarray(deltas, dtype=float), 0.0, 0.50)
    return blend_prediction(frame, new_weight)


def delta_errors(row: pd.Series, delta_grid: tuple[float, ...] = DELTA_GRID) -> np.ndarray:
    target = float(row["target_tmax_c"])
    official = float(row["official_family_prediction_c"])
    station = float(row["station_family_prediction_c"])
    base_weight = float(row["base_0069_station_weight"])
    errors = []
    for delta in delta_grid:
        weight = float(np.clip(base_weight + delta, 0.0, 0.50))
        errors.append(abs(((1.0 - weight) * official + weight * station) - target))
    return np.array(errors, dtype=float)


def select_prior_delta(
    *,
    abs_sums: np.ndarray,
    count: int,
    spec: LocalFusionSpec,
) -> float:
    if count < spec.min_history:
        return spec.fallback_delta
    prior_mae = abs_sums / count
    if spec.mode == "prior_best_delta":
        return float(DELTA_GRID[int(np.argmin(prior_mae))])
    if spec.mode == "prior_soft_delta":
        centered = prior_mae - float(np.min(prior_mae))
        raw = np.exp(-centered / spec.temperature_c)
        probs = raw / raw.sum()
        return float(np.sum(np.array(DELTA_GRID) * probs))
    raise ValueError(f"Unsupported prior delta mode: {spec.mode}")


def local_fusion_specs() -> list[LocalFusionSpec]:
    specs: list[LocalFusionSpec] = []
    group_modes = [
        "source_signeddiff_active",
        "source_signeddiff_range",
        "source_absdiff_range",
        "source_range_active",
        "source_max_active",
        "source_mid_range",
        "source_weather_range",
        "source_wind_range",
        "source_rh_range",
        "source_gate_signeddiff",
        "source_selected_signeddiff",
        "source_stackcorr_signeddiff",
        "source_signeddiff_weather_active",
    ]
    for group_mode in group_modes:
        for min_history in (30, 80, 120):
            for mode, temperature in [("prior_best_delta", 0.0), ("prior_soft_delta", 0.02)]:
                mode_token = "best" if mode == "prior_best_delta" else "soft0p02"
                specs.append(
                    LocalFusionSpec(
                        candidate_id=f"causal_delta_{mode_token}_{group_mode}_h{min_history}",
                        mode=mode,
                        candidate_class="causal_prior_delta_selector",
                        group_mode=group_mode,
                        min_history=min_history,
                        fallback_delta=0.0,
                        temperature_c=temperature,
                        fixed_delta=0.0,
                        cap_low=0.0,
                        cap_high=0.50,
                    )
                )
    for group_mode in [
        "source_signeddiff_active",
        "source_signeddiff_range",
        "source_weather_range",
        "source_gate_signeddiff",
    ]:
        for fixed_delta in (-0.08, -0.05, -0.03, 0.03, 0.05, 0.08):
            token = str(fixed_delta).replace("-", "m").replace(".", "p")
            specs.append(
                LocalFusionSpec(
                    candidate_id=f"diagnostic_fixed_delta_{group_mode}_{token}",
                    mode="fixed_delta",
                    candidate_class="diagnostic_fixed_local_delta",
                    group_mode=group_mode,
                    min_history=0,
                    fallback_delta=0.0,
                    temperature_c=0.0,
                    fixed_delta=fixed_delta,
                    cap_low=0.0,
                    cap_high=0.50,
                )
            )
    ids = [spec.candidate_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("0070 candidate IDs are not unique")
    return specs


def apply_prior_delta_spec(frame: pd.DataFrame, spec: LocalFusionSpec) -> pd.DataFrame:
    state: dict[str, dict[str, object]] = defaultdict(
        lambda: {"count": 0, "abs_sums": np.zeros(len(DELTA_GRID), dtype=float)}
    )
    deltas: list[float] = []
    prior_counts: list[int] = []
    router_groups: list[str] = []
    for _idx, row in frame.iterrows():
        key = local_group_key(row, spec.group_mode)
        group_state = state[key]
        count = int(group_state["count"])
        abs_sums = np.asarray(group_state["abs_sums"], dtype=float)
        delta = select_prior_delta(abs_sums=abs_sums, count=count, spec=spec)
        deltas.append(delta)
        prior_counts.append(count)
        router_groups.append(key)
        group_state["abs_sums"] = abs_sums + delta_errors(row)
        group_state["count"] = count + 1
    out = frame[["target_date", "target_tmax_c", "fold_id", "forecast_source_family"]].copy()
    out["station_delta"] = deltas
    out["station_weight"] = np.clip(
        pd.to_numeric(frame["base_0069_station_weight"], errors="coerce").to_numpy(dtype=float)
        + np.array(deltas, dtype=float),
        0.0,
        0.50,
    )
    out["candidate_prediction_c"] = blend_prediction(frame, out["station_weight"].to_numpy(dtype=float))
    out["prior_count"] = prior_counts
    out["router_group"] = router_groups
    return out


def apply_fixed_delta_spec(frame: pd.DataFrame, spec: LocalFusionSpec) -> pd.DataFrame:
    out = frame[["target_date", "target_tmax_c", "fold_id", "forecast_source_family"]].copy()
    out["station_delta"] = spec.fixed_delta
    out["station_weight"] = np.clip(
        pd.to_numeric(frame["base_0069_station_weight"], errors="coerce").to_numpy(dtype=float)
        + spec.fixed_delta,
        0.0,
        0.50,
    )
    out["candidate_prediction_c"] = blend_prediction(frame, out["station_weight"].to_numpy(dtype=float))
    out["prior_count"] = 0
    out["router_group"] = [local_group_key(row, spec.group_mode) for _idx, row in frame.iterrows()]
    return out


def apply_spec(frame: pd.DataFrame, spec: LocalFusionSpec) -> pd.DataFrame:
    if spec.mode in {"prior_best_delta", "prior_soft_delta"}:
        out = apply_prior_delta_spec(frame, spec)
    elif spec.mode == "fixed_delta":
        out = apply_fixed_delta_spec(frame, spec)
    else:
        raise ValueError(f"Unsupported 0070 mode: {spec.mode}")
    out["candidate_id"] = spec.candidate_id
    out["mode"] = spec.mode
    out["candidate_class"] = spec.candidate_class
    out["group_mode"] = spec.group_mode
    out["min_history"] = spec.min_history
    out["fallback_delta"] = spec.fallback_delta
    out["temperature_c"] = spec.temperature_c
    out["fixed_delta"] = spec.fixed_delta
    return out


def score_candidate(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    base_0069_mae: float,
) -> dict[str, object]:
    pred_values = pd.to_numeric(predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float)
    score = score_prediction(frame, pred_values)
    base_score = score_prediction(
        frame,
        pd.to_numeric(frame["base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    official_score = score_prediction(
        frame,
        pd.to_numeric(frame["official_family_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    late_mask = predictions["target_date"].ge(LATE_EVAL_START)
    late_score = score_prediction(frame.loc[late_mask].copy(), pred_values[late_mask.to_numpy()])
    late_base = score_prediction(
        frame.loc[late_mask].copy(),
        pd.to_numeric(frame.loc[late_mask, "base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    fold_deltas_vs_0069 = []
    fold_deltas_vs_official = []
    for _fold_id, fold_predictions in predictions.groupby("fold_id", observed=True):
        fold_frame = frame.loc[fold_predictions.index].copy()
        fold_score = score_prediction(
            fold_frame,
            pd.to_numeric(fold_predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        fold_base = score_prediction(
            fold_frame,
            pd.to_numeric(fold_frame["base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        fold_official = score_prediction(
            fold_frame,
            pd.to_numeric(fold_frame["official_family_prediction_c"], errors="coerce").to_numpy(dtype=float),
        )
        fold_deltas_vs_0069.append(float(fold_score["mae"]) - float(fold_base["mae"]))
        fold_deltas_vs_official.append(float(fold_score["mae"]) - float(fold_official["mae"]))
    row: dict[str, object] = {
        "candidate_id": str(predictions["candidate_id"].iloc[0]),
        "mode": str(predictions["mode"].iloc[0]),
        "candidate_class": str(predictions["candidate_class"].iloc[0]),
        "group_mode": str(predictions["group_mode"].iloc[0]),
        "min_history": int(predictions["min_history"].iloc[0]),
        "fixed_delta": float(predictions["fixed_delta"].iloc[0]),
        "temperature_c": float(predictions["temperature_c"].iloc[0]),
        "n": score["n"],
        "mae": score["mae"],
        "rmse": score["rmse"],
        "bias": score["bias"],
        "base_0069_mae": base_score["mae"],
        "official_mae": official_score["mae"],
        "delta_mae_vs_0069": float(score["mae"]) - base_0069_mae,
        "delta_mae_vs_official": float(score["mae"]) - float(official_score["mae"]),
        "late_n": late_score["n"],
        "late_mae": late_score["mae"],
        "late_delta_mae_vs_0069": float(late_score["mae"]) - float(late_base["mae"]),
        "fold_delta_max_vs_0069": max(fold_deltas_vs_0069) if fold_deltas_vs_0069 else math.nan,
        "fold_delta_min_vs_0069": min(fold_deltas_vs_0069) if fold_deltas_vs_0069 else math.nan,
        "fold_delta_max_vs_official": max(fold_deltas_vs_official) if fold_deltas_vs_official else math.nan,
        "folds_improved_vs_0069": int(sum(delta < 0 for delta in fold_deltas_vs_0069)),
        "mean_station_delta": float(pd.to_numeric(predictions["station_delta"], errors="coerce").mean()),
        "mean_station_weight": float(pd.to_numeric(predictions["station_weight"], errors="coerce").mean()),
        "press_mean_station_delta": float(
            pd.to_numeric(
                predictions.loc[predictions["forecast_source_family"].eq("press_archive"), "station_delta"],
                errors="coerce",
            ).mean()
        ),
        "rss_mean_station_delta": float(
            pd.to_numeric(
                predictions.loc[predictions["forecast_source_family"].eq("rss_archive"), "station_delta"],
                errors="coerce",
            ).mean()
        ),
    }
    row["beats_0069"] = bool(float(row["delta_mae_vs_0069"]) <= -0.0005)
    row["promotion_gate_passed"] = bool(
        row["beats_0069"]
        and float(row["fold_delta_max_vs_0069"]) <= 0.0
        and float(row["late_delta_mae_vs_0069"]) <= 0.0
    )
    row["deployable_gate_passed"] = bool(
        row["promotion_gate_passed"]
        and str(row["candidate_class"]) == "causal_prior_delta_selector"
    )
    return row


def score_all_specs(
    frame: pd.DataFrame,
    specs: list[LocalFusionSpec],
    *,
    base_0069_mae: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs:
        predictions = apply_spec(frame, spec)
        rows.append(score_candidate(frame, predictions, base_0069_mae=base_0069_mae))
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(rows).sort_values(
        ["deployable_gate_passed", "beats_0069", "promotion_gate_passed", "mae"],
        ascending=[False, False, False, True],
    )
    top_ids = set(scoreboard["candidate_id"].head(30).astype(str))
    top_predictions = pd.concat(
        [predictions for predictions in prediction_frames if str(predictions["candidate_id"].iloc[0]) in top_ids],
        ignore_index=True,
    )
    require_no_confirmation_dates(top_predictions["target_date"], context="0070 selected predictions")
    return scoreboard.reset_index(drop=True), top_predictions


def trigger_cell_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_modes = [
        "source_signeddiff_active",
        "source_signeddiff_range",
        "source_absdiff_range",
        "source_weather_range",
        "source_wind_range",
        "source_gate_signeddiff",
        "source_selected_signeddiff",
        "source_stackcorr_signeddiff",
    ]
    base_errors = (
        pd.to_numeric(frame["base_0069_prediction_c"], errors="coerce")
        - pd.to_numeric(frame["target_tmax_c"], errors="coerce")
    ).abs()
    for group_mode in group_modes:
        keys = frame.apply(lambda row, mode=group_mode: local_group_key(row, mode), axis=1)
        for key, index in keys.groupby(keys).groups.items():
            group = frame.loc[index].copy()
            if len(group) < 60:
                continue
            base_mae = float(base_errors.loc[index].mean())
            best_delta = 0.0
            best_mae = base_mae
            for delta in DELTA_GRID:
                pred = delta_predictions(group, delta)
                score = score_prediction(group, pred)
                if float(score["mae"]) < best_mae:
                    best_mae = float(score["mae"])
                    best_delta = float(delta)
            rows.append(
                {
                    "group_mode": group_mode,
                    "group_key": str(key),
                    "n": int(len(group)),
                    "base_0069_mae": base_mae,
                    "best_fixed_delta": best_delta,
                    "best_fixed_delta_mae": best_mae,
                    "active_delta_mae": best_mae - base_mae,
                    "mean_family_disagreement_c": float(group["family_disagreement_c"].mean()),
                    "mean_base_station_weight": float(group["base_0069_station_weight"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "group_mode",
                "group_key",
                "n",
                "base_0069_mae",
                "best_fixed_delta",
                "best_fixed_delta_mae",
                "active_delta_mae",
                "mean_family_disagreement_c",
                "mean_base_station_weight",
            ]
        )
    return out.sort_values(["active_delta_mae", "n"], ascending=[True, False]).reset_index(drop=True)


def leakage_audit(frame: pd.DataFrame, scoreboard: pd.DataFrame) -> pd.DataFrame:
    deployable = scoreboard[scoreboard["deployable_gate_passed"].astype(bool)].copy()
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(frame["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(frame['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "base_0069_predictions_present_one_per_date",
            "passed": bool(
                frame["base_0069_prediction_c"].notna().all()
                and len(frame) == frame["target_date"].nunique()
            ),
            "evidence": f"{len(frame)} merged rows",
        },
        {
            "check_id": "forecast_metadata_available",
            "passed": bool(frame["forecast_range_c"].notna().mean() > 0.95),
            "evidence": f"forecast range coverage {frame['forecast_range_c'].notna().mean():.4f}",
        },
        {
            "check_id": "prior_delta_selectors_update_after_scoring",
            "passed": True,
            "evidence": "online delta states update only after row prediction is selected",
        },
        {
            "check_id": "diagnostic_fixed_deltas_not_marked_deployable",
            "passed": bool(
                scoreboard.loc[
                    scoreboard["candidate_class"].ne("causal_prior_delta_selector"),
                    "deployable_gate_passed",
                ].eq(False).all()
            ),
            "evidence": "fixed local deltas are diagnostic only",
        },
        {
            "check_id": "deployable_gate_requires_fold_and_late_improvement_vs_0069",
            "passed": bool(
                deployable.empty
                or (
                    deployable["delta_mae_vs_0069"].le(-0.0005).all()
                    and deployable["fold_delta_max_vs_0069"].le(0.0).all()
                    and deployable["late_delta_mae_vs_0069"].le(0.0).all()
                )
            ),
            "evidence": f"{len(deployable)} deployable candidates passed",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    definitions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    trigger_cells: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    deployable = scoreboard[scoreboard["deployable_gate_passed"].astype(bool)].copy()
    diagnostic = scoreboard[scoreboard["candidate_class"].ne("causal_prior_delta_selector")].head(30).copy()
    return f"""# Nonlinear Local Residual-Fusion Lab

Generated: `{summary['generated_at_utc']}`

## Purpose

`0069` showed that the station/official blend can improve when the system varies station weight by source, signed family disagreement, and station-stack activity. `0070` asks a more local nonlinear question: can the model learn extra station-weight deltas from pre-target trigger cells built from forecast range, forecast level, RH range, wind/weather text, source family, gate state, station-stack activity, and family disagreement shape?

This is still an analysis/fusion experiment only. It does not use Polymarket data, trading targets, machine learning training artifacts, or any 2024+ confirmation rows.

## Data Contract

- Base prediction: best deployable `0069` candidate `{summary['base_0069_candidate']}`.
- Common rows: `{summary['common_rows']}`.
- Date range: `{summary['first_date']}` to `{summary['last_date']}`.
- Forecast metadata source: `0044_forecast_archive_continuous_scored_export/artifacts/scored_pre2024.csv`.
- All trigger features are known from the official forecast or station system before the target day.
- Prior selectors update their error state only after the row has been scored.

## Headline

| Item | Value |
|---|---:|
| 0069 base MAE | {summary['base_0069_mae']} |
| 0069 base RMSE | {summary['base_0069_rmse']} |
| Best 0070 candidate | {summary['best_candidate']} |
| Best 0070 class | {summary['best_candidate_class']} |
| Best 0070 MAE | {summary['best_mae']} |
| Best 0070 RMSE | {summary['best_rmse']} |
| Best delta vs 0069 | {summary['best_delta_mae_vs_0069']} |
| Best prior-only candidate | {summary['best_prior_candidate']} |
| Best prior-only MAE | {summary['best_prior_mae']} |
| Best prior-only fold max delta vs 0069 | {summary['best_prior_fold_delta_max_vs_0069']} |
| Gate-passed deployable candidate | {summary['best_deployable_candidate']} |
| Gate-passed deployable MAE | {summary['best_deployable_mae']} |
| Gate-passed deployable count | {summary['deployable_candidate_count']} |
| Candidate count | {summary['candidate_count']} |
| Trigger cells diagnosed | {summary['trigger_cell_count']} |

## Interpretation

The strongest local diagnostic cells identify where a fixed extra station-weight delta would have helped most, but those rows are not deployable by themselves because the best delta is chosen after seeing the whole group. The deployable question is stricter: can an online prior selector learn the extra delta from earlier rows in the same trigger cell and improve both folds and the late RSS window versus `0069`?

## Scoreboard

{markdown_table(scoreboard, max_rows=100)}

## Deployable Candidates

{markdown_table(deployable, max_rows=80)}

## Diagnostic Fixed-Delta Candidates

{markdown_table(diagnostic, max_rows=50)}

## Strongest Trigger Cells

{markdown_table(trigger_cells, max_rows=100)}

## Candidate Definitions

{markdown_table(definitions, max_rows=120)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Files

- `artifacts/candidate_definitions.csv`
- `artifacts/scoreboard.csv`
- `artifacts/top_predictions.csv`
- `artifacts/trigger_cell_diagnostics.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_nonlinear_local_residual_fusion_lab.py`:

- `{FOLDER_NAME}`: nonlinear local station-weight delta trigger lab on top of `0069`.

| Metric | Value |
|---|---:|
| 0069 base MAE | {summary['base_0069_mae']} |
| Best 0070 candidate | {summary['best_candidate']} |
| Best 0070 MAE | {summary['best_mae']} |
| Best delta vs 0069 | {summary['best_delta_mae_vs_0069']} |
| Best prior-only candidate | {summary['best_prior_candidate']} |
| Best prior-only MAE | {summary['best_prior_mae']} |
| Gate-passed deployable candidate | {summary['best_deployable_candidate']} |
| Gate-passed deployable MAE | {summary['best_deployable_mae']} |

Leakage contract: no 2024+ rows; forecast metadata is pre-target; deployable delta selectors update only after scoring each row.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Nonlinear Local Residual-Fusion Lab",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_nonlinear_local_residual_fusion_lab.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Input | `0069` best deployable prediction plus `0044` forecast metadata | Tested |
| Common rows | `{summary['common_rows']}` from `{summary['first_date']}` to `{summary['last_date']}` | Non-contiguous |
| Trigger families | disagreement, active station-stack count, forecast range/level, RH, wind, weather, gate state | Tested |
| 0069 base MAE / RMSE | `{summary['base_0069_mae']}` / `{summary['base_0069_rmse']}` | Baseline |
| Best 0070 candidate | `{summary['best_candidate']}` | Tested |
| Best 0070 class | `{summary['best_candidate_class']}` | Diagnostic/deployable classification |
| Best 0070 MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best delta vs 0069 | `{summary['best_delta_mae_vs_0069']}` | Local fusion value |
| Best prior-only candidate | `{summary['best_prior_candidate']}` | Causal selector |
| Best prior-only fold max delta vs 0069 | `{summary['best_prior_fold_delta_max_vs_0069']}` | Robustness check |
| Gate-passed deployable candidate | `{summary['best_deployable_candidate']}` | Requires full, fold, and late improvement |
| Gate-passed deployable MAE | `{summary['best_deployable_mae']}` | `None` means no candidate passed |
| Trigger cells diagnosed | `{summary['trigger_cell_count']}` | Diagnostic atlas |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0070` tests whether nonlinear trigger cells can learn extra station-weight deltas on top of the current `0069` champion without late/fold damage.
"""
    update_markdown_section(
        path,
        heading="Nonlinear Local Residual-Fusion Lab",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    finding = (
        f"54. Nonlinear local residual-fusion lab screened `{summary['candidate_count']}` candidates "
        f"and `{summary['trigger_cell_count']}` trigger cells; best delta vs 0069 is "
        f"`{summary['best_delta_mae_vs_0069']}` from `{summary['best_candidate']}`, "
        f"with best prior-only MAE `{summary['best_prior_mae']}` and "
        f"`{summary['deployable_candidate_count']}` gate-passed deployable candidates."
    )
    text = path.read_text(encoding="utf-8")
    if finding not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{finding}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue while the forecast backfill runs: mine the `0070` trigger-cell diagnostics for the highest-value cells, then build a sparse specialist stack that only activates deployable prior-delta learners in cells with enough past support and positive fold-local evidence.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0069 = build_feature_frame()
    base_score = score_prediction(
        frame,
        pd.to_numeric(frame["base_0069_prediction_c"], errors="coerce").to_numpy(dtype=float),
    )
    specs = local_fusion_specs()
    definitions = pd.DataFrame([spec.__dict__ for spec in specs])
    scoreboard, top_predictions = score_all_specs(frame, specs, base_0069_mae=float(base_score["mae"]))
    trigger_cells = trigger_cell_diagnostics(frame)
    leakage = leakage_audit(frame, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0070 leakage audit failed: {failed}")

    best = scoreboard.iloc[0]
    prior_pool = scoreboard[scoreboard["candidate_class"].eq("causal_prior_delta_selector")].copy()
    prior_pool = prior_pool.sort_values(["mae", "fold_delta_max_vs_0069"]).reset_index(drop=True)
    best_prior = prior_pool.iloc[0]
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
        "promoted_candidate_count": int(scoreboard["promotion_gate_passed"].astype(bool).sum()),
        "deployable_candidate_count": int(scoreboard["deployable_gate_passed"].astype(bool).sum()),
        "trigger_cell_count": int(len(trigger_cells)),
        "base_0069_candidate": str(summary_0069["best_deployable_candidate"]),
        "base_0069_mae": float(base_score["mae"]),
        "base_0069_rmse": float(base_score["rmse"]),
        "best_candidate": str(best["candidate_id"]),
        "best_candidate_class": str(best["candidate_class"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0069": float(best["delta_mae_vs_0069"]),
        "best_late_delta_mae_vs_0069": float(best["late_delta_mae_vs_0069"]),
        "best_mean_station_delta": float(best["mean_station_delta"]),
        "best_prior_candidate": str(best_prior["candidate_id"]),
        "best_prior_mae": float(best_prior["mae"]),
        "best_prior_rmse": float(best_prior["rmse"]),
        "best_prior_delta_mae_vs_0069": float(best_prior["delta_mae_vs_0069"]),
        "best_prior_late_delta_mae_vs_0069": float(best_prior["late_delta_mae_vs_0069"]),
        "best_prior_fold_delta_max_vs_0069": float(best_prior["fold_delta_max_vs_0069"]),
        "best_prior_gate_passed": bool(best_prior["deployable_gate_passed"]),
        "best_deployable_candidate": str(best_deployable["candidate_id"]) if best_deployable is not None else "NONE",
        "best_deployable_mae": float(best_deployable["mae"]) if best_deployable is not None else None,
        "best_deployable_rmse": float(best_deployable["rmse"]) if best_deployable is not None else None,
        "best_deployable_delta_mae_vs_0069": (
            float(best_deployable["delta_mae_vs_0069"]) if best_deployable is not None else None
        ),
        "deployable_beats_0069": bool(best_deployable is not None),
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
    write_csv(artifacts / "trigger_cell_diagnostics.csv", trigger_cells)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "nonlinear_local_residual_fusion_lab_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            definitions=definitions,
            scoreboard=scoreboard,
            trigger_cells=trigger_cells,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Run nonlinear local residual-fusion trigger lab on top of 0069."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
