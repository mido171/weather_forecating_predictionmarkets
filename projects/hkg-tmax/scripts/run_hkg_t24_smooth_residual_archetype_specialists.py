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
from scripts.run_hkg_t24_official_anchor_expert_blend_screen import (  # noqa: E402
    past_only_expert_blend,
)
from scripts.run_hkg_t24_residual_failure_cluster_discovery import (  # noqa: E402
    ArchetypeCondition,
    available_numeric_features,
    build_failure_frame,
    condition_prior_mask,
)

FOLDER_NAME = "0033_smooth_residual_archetype_specialists"
MIN_HISTORY = 160
TOP_BLEND_EXPERTS = 14


@dataclass(frozen=True)
class SmoothArchetypeFamily:
    name: str
    conditions: tuple[ArchetypeCondition, ...]
    features: tuple[str, ...]


@dataclass(frozen=True)
class SmoothArchetypeSpec:
    family_name: str
    anchor_col: str
    conditions: tuple[ArchetypeCondition, ...]
    features: tuple[str, ...]
    k_neighbors: int
    same_source: bool
    half_life_days: float | None
    min_history: int = MIN_HISTORY
    min_match_rows: int = 35
    shrinkage: float = 80.0
    correction_clip_c: float = 2.0
    min_local_mae_improvement_c: float = 0.0


@dataclass(frozen=True)
class SmoothCorrectionResult:
    correction: float
    rows_used: int
    mean_distance: float
    local_anchor_mae: float
    local_corrected_mae: float
    gate_passed: bool


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 150) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def candidate_id_for_spec(spec: SmoothArchetypeSpec) -> str:
    source = "same_source" if spec.same_source else "all_prior"
    half_life = "hl_none" if spec.half_life_days is None else f"hl_{int(spec.half_life_days)}d"
    return slug(f"smooth_archetype_{spec.anchor_col}_{spec.family_name}_k{spec.k_neighbors}_{source}_{half_life}")


def finite_numeric_frame(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return math.nan
    return float(np.average(values[valid], weights=weights[valid]))


def smooth_half_life_residual_correction(
    x_prior: np.ndarray,
    residual_prior: np.ndarray,
    age_days_prior: np.ndarray,
    x_current: np.ndarray,
    *,
    k_neighbors: int,
    shrinkage: float,
    correction_clip_c: float,
    half_life_days: float | None,
    min_local_mae_improvement_c: float,
) -> SmoothCorrectionResult:
    if x_prior.ndim != 2 or len(x_prior) != len(residual_prior) or len(x_prior) != len(age_days_prior):
        raise ValueError("prior feature, residual, and age arrays are incompatible")
    if len(x_prior) == 0 or not np.isfinite(x_current).all():
        return SmoothCorrectionResult(0.0, 0, math.nan, math.nan, math.nan, False)

    means = np.nanmean(x_prior, axis=0)
    stds = np.nanstd(x_prior, axis=0)
    stds = np.where((stds <= 1e-9) | ~np.isfinite(stds), 1.0, stds)
    prior_scaled = (x_prior - means) / stds
    current_scaled = (x_current - means) / stds
    distances = np.sqrt(np.nanmean(np.square(prior_scaled - current_scaled), axis=1))
    valid = np.isfinite(distances) & np.isfinite(residual_prior) & np.isfinite(age_days_prior)
    if not valid.any():
        return SmoothCorrectionResult(0.0, 0, math.nan, math.nan, math.nan, False)

    valid_distances = distances[valid]
    valid_residuals = residual_prior[valid]
    valid_ages = age_days_prior[valid]
    k = min(int(k_neighbors), len(valid_distances))
    if k <= 0:
        return SmoothCorrectionResult(0.0, 0, math.nan, math.nan, math.nan, False)

    order = np.argpartition(valid_distances, k - 1)[:k]
    selected_distances = valid_distances[order]
    selected_residuals = valid_residuals[order]
    selected_ages = valid_ages[order]

    positive_distances = selected_distances[selected_distances > 1e-9]
    scale = float(np.nanmedian(positive_distances)) if len(positive_distances) else math.nan
    if not np.isfinite(scale) or scale <= 1e-9:
        weights = np.ones(len(selected_distances), dtype=float)
    else:
        weights = np.exp(-selected_distances / scale)
    if half_life_days is not None and half_life_days > 0:
        weights = weights * np.power(0.5, selected_ages / float(half_life_days))

    raw = weighted_mean(selected_residuals, weights)
    if not np.isfinite(raw):
        return SmoothCorrectionResult(0.0, 0, math.nan, math.nan, math.nan, False)
    shrink = len(selected_residuals) / (len(selected_residuals) + float(shrinkage))
    correction = float(np.clip(raw * shrink, -correction_clip_c, correction_clip_c))

    local_anchor_mae = weighted_mean(np.abs(selected_residuals), weights)
    local_corrected_mae = weighted_mean(np.abs(selected_residuals - correction), weights)
    gate_passed = bool(
        np.isfinite(local_anchor_mae)
        and np.isfinite(local_corrected_mae)
        and local_corrected_mae <= local_anchor_mae - min_local_mae_improvement_c
    )
    if not gate_passed:
        correction = 0.0
    return SmoothCorrectionResult(
        correction=correction,
        rows_used=int(len(selected_residuals)),
        mean_distance=float(np.mean(selected_distances)),
        local_anchor_mae=float(local_anchor_mae),
        local_corrected_mae=float(local_corrected_mae),
        gate_passed=gate_passed,
    )


def current_gate_and_prior_mask(
    ordered: pd.DataFrame,
    base_prior: np.ndarray,
    current: pd.Series,
    conditions: tuple[ArchetypeCondition, ...],
) -> tuple[bool, np.ndarray, str]:
    match_mask = base_prior.copy()
    threshold_parts: list[str] = []
    current_matches = True
    for condition in conditions:
        current_match, condition_mask, threshold = condition_prior_mask(ordered, base_prior, current, condition)
        current_matches = current_matches and current_match
        match_mask &= condition_mask
        threshold_parts.append(f"{condition.feature}:{condition.direction}:{threshold:.4f}")
    return bool(current_matches), match_mask, ";".join(threshold_parts)


def past_only_smooth_archetype_predictions(frame: pd.DataFrame, spec: SmoothArchetypeSpec) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    date_series = pd.to_datetime(ordered["target_date"], errors="coerce").dt.normalize()
    sources = ordered["forecast_source_family"].astype(str).to_numpy()
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    official = pd.to_numeric(ordered["official_raw"], errors="coerce").to_numpy(dtype=float)
    anchor = pd.to_numeric(ordered[spec.anchor_col], errors="coerce").to_numpy(dtype=float)
    residual = target - anchor
    feature_matrix = finite_numeric_frame(ordered, spec.features).to_numpy(dtype=float)

    predictions: list[float] = []
    corrections: list[float] = []
    prior_rows_used: list[int] = []
    matched_rows: list[int] = []
    neighbor_rows: list[int] = []
    current_matches: list[bool] = []
    gate_passed: list[bool] = []
    thresholds: list[str] = []
    mean_distances: list[float] = []
    local_anchor_maes: list[float] = []
    local_corrected_maes: list[float] = []

    for index, target_date in enumerate(dates):
        if not np.isfinite(anchor[index]) or not np.isfinite(feature_matrix[index]).all():
            predictions.append(float(anchor[index]) if np.isfinite(anchor[index]) else math.nan)
            corrections.append(0.0)
            prior_rows_used.append(0)
            matched_rows.append(0)
            neighbor_rows.append(0)
            current_matches.append(False)
            gate_passed.append(False)
            thresholds.append("")
            mean_distances.append(math.nan)
            local_anchor_maes.append(math.nan)
            local_corrected_maes.append(math.nan)
            continue

        prior_limit = int(np.searchsorted(dates, target_date, side="left"))
        base_prior = np.arange(len(ordered)) < prior_limit
        if spec.same_source:
            base_prior &= sources == sources[index]
        base_prior &= np.isfinite(residual) & np.isfinite(anchor) & np.isfinite(target)
        base_prior &= np.isfinite(feature_matrix).all(axis=1)
        prior_rows = int(base_prior.sum())
        if prior_rows < spec.min_history:
            predictions.append(float(anchor[index]))
            corrections.append(0.0)
            prior_rows_used.append(prior_rows)
            matched_rows.append(0)
            neighbor_rows.append(0)
            current_matches.append(False)
            gate_passed.append(False)
            thresholds.append("")
            mean_distances.append(math.nan)
            local_anchor_maes.append(math.nan)
            local_corrected_maes.append(math.nan)
            continue

        current_match, match_mask, threshold_text = current_gate_and_prior_mask(
            ordered, base_prior, ordered.iloc[index], spec.conditions
        )
        matched = int(match_mask.sum()) if current_match else 0
        if not current_match or matched < max(spec.min_match_rows, spec.k_neighbors):
            predictions.append(float(anchor[index]))
            corrections.append(0.0)
            prior_rows_used.append(prior_rows)
            matched_rows.append(matched)
            neighbor_rows.append(0)
            current_matches.append(bool(current_match))
            gate_passed.append(False)
            thresholds.append(threshold_text)
            mean_distances.append(math.nan)
            local_anchor_maes.append(math.nan)
            local_corrected_maes.append(math.nan)
            continue

        prior_index = np.flatnonzero(match_mask)
        age_days = (date_series.iloc[index] - date_series.iloc[prior_index]).dt.days.to_numpy(dtype=float)
        result = smooth_half_life_residual_correction(
            feature_matrix[prior_index],
            residual[prior_index],
            age_days,
            feature_matrix[index],
            k_neighbors=spec.k_neighbors,
            shrinkage=spec.shrinkage,
            correction_clip_c=spec.correction_clip_c,
            half_life_days=spec.half_life_days,
            min_local_mae_improvement_c=spec.min_local_mae_improvement_c,
        )
        predictions.append(float(anchor[index] + result.correction))
        corrections.append(result.correction)
        prior_rows_used.append(prior_rows)
        matched_rows.append(matched)
        neighbor_rows.append(result.rows_used)
        current_matches.append(True)
        gate_passed.append(result.gate_passed)
        thresholds.append(threshold_text)
        mean_distances.append(result.mean_distance)
        local_anchor_maes.append(result.local_anchor_mae)
        local_corrected_maes.append(result.local_corrected_mae)

    out = ordered[["target_date", "forecast_source_family", "primary_regime", "target_tmax_c"]].copy()
    out["official_raw"] = official
    out["anchor_prediction_c"] = anchor
    out["candidate_prediction_c"] = predictions
    out["residual_correction_c"] = corrections
    out["prior_rows_used"] = prior_rows_used
    out["matched_archetype_rows"] = matched_rows
    out["neighbor_rows_used"] = neighbor_rows
    out["current_archetype_match"] = current_matches
    out["do_no_harm_gate_passed"] = gate_passed
    out["condition_thresholds"] = thresholds
    out["mean_neighbor_distance"] = mean_distances
    out["local_anchor_mae"] = local_anchor_maes
    out["local_corrected_mae"] = local_corrected_maes
    return out


def family_templates() -> dict[str, SmoothArchetypeFamily]:
    return {
        "moisture_surge": SmoothArchetypeFamily(
            name="moisture_surge",
            conditions=(
                ArchetypeCondition("isd_dew_point_mean_c_change_1d", "high", 0.75),
                ArchetypeCondition("isd_temp_dewpoint_spread_mean_c", "low", 0.35),
            ),
            features=(
                "isd_dew_point_mean_c_change_1d",
                "isd_temp_dewpoint_spread_mean_c",
                "isd_dewpoint_midday_minus_temp_c",
                "rh_min_pct",
                "rh_max_pct",
                "ua_mse_1000_850_mean_kj_kg",
                "ua_theta_e_1000_850_mean_k",
                "isd_pressure_mean_hpa_roll7_mean",
                "forecast_max_c",
                "forecast_min_c",
                "forecast_range_c",
                "month",
                "monsoon_phase_code",
                "text_any_rain",
            ),
        ),
        "humid_thunder": SmoothArchetypeFamily(
            name="humid_thunder",
            conditions=(
                ArchetypeCondition("text_thunder", "flag"),
                ArchetypeCondition("rh_min_pct", "high", 0.70),
            ),
            features=(
                "text_thunder",
                "text_showers",
                "text_hot",
                "text_very_hot",
                "text_keyword_count",
                "rh_min_pct",
                "rh_max_pct",
                "forecast_max_c",
                "forecast_range_c",
                "ua_mse_1000_850_mean_kj_kg",
                "isd_dew_point_mean_c_change_1d",
                "month",
                "monsoon_phase_code",
            ),
        ),
        "forecast_jump_up": SmoothArchetypeFamily(
            name="forecast_jump_up",
            conditions=(
                ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.75),
                ArchetypeCondition("forecast_max_vs_prior7_mean_source_c", "high", 0.65),
            ),
            features=(
                "forecast_max_c",
                "forecast_min_c",
                "forecast_range_c",
                "forecast_max_change_1_source_c",
                "forecast_max_prior7_std_source_c",
                "forecast_max_vs_prior7_mean_source_c",
                "pressure_plane_slope_magnitude_hpa_per_deg",
                "isd_pressure_mean_hpa_change_1d",
                "isd_wind_speed_mean_mps",
                "month",
                "monsoon_phase_code",
            ),
        ),
        "station_gradient_heat": SmoothArchetypeFamily(
            name="station_gradient_heat",
            conditions=(
                ArchetypeCondition("abs_north_south_temp_gradient_c", "high", 0.75),
                ArchetypeCondition("forecast_max_c", "high", 0.65),
            ),
            features=(
                "abs_north_south_temp_gradient_c",
                "isd_north_south_temp_gradient_c",
                "isd_east_west_temp_gradient_c",
                "thermal_590870_minus_596730_c",
                "thermal_590960_minus_596730_c",
                "isd_graph_total_variation_c2",
                "isd_graph_laplacian_mode_1",
                "isd_graph_laplacian_mode_3",
                "forecast_max_c",
                "forecast_range_c",
                "isd_wind_speed_mean_mps",
                "month",
                "monsoon_phase_code",
            ),
        ),
        "dry_high_range": SmoothArchetypeFamily(
            name="dry_high_range",
            conditions=(
                ArchetypeCondition("forecast_range_c", "high", 0.70),
                ArchetypeCondition("isd_temp_dewpoint_spread_mean_c", "high", 0.70),
            ),
            features=(
                "forecast_max_c",
                "forecast_min_c",
                "forecast_range_c",
                "isd_temp_dewpoint_spread_mean_c",
                "isd_dewpoint_midday_minus_temp_c",
                "rh_min_pct",
                "ua_dewpoint_925hpa_c",
                "ua_mse_1000_850_mean_kj_kg",
                "ua_theta_e_1000_850_mean_k",
                "month",
                "monsoon_phase_code",
            ),
        ),
        "marine_onshore_wind": SmoothArchetypeFamily(
            name="marine_onshore_wind",
            conditions=(
                ArchetypeCondition("isd_onshore_easterly_proxy_mps", "high", 0.75),
                ArchetypeCondition("isd_wind_speed_mean_mps", "high", 0.65),
            ),
            features=(
                "isd_onshore_easterly_proxy_mps",
                "isd_northerly_proxy_mps",
                "isd_wind_speed_mean_mps",
                "isd_wind_speed_max_mps",
                "pressure_plane_slope_magnitude_hpa_per_deg",
                "isd_pressure_plane_lat_slope_hpa_per_deg",
                "isd_pressure_plane_lon_slope_hpa_per_deg",
                "slp_590870_minus_596730_hpa",
                "forecast_max_vs_prior7_mean_source_c",
                "forecast_max_c",
                "month",
                "monsoon_phase_code",
            ),
        ),
        "cloudy_high_forecast": SmoothArchetypeFamily(
            name="cloudy_high_forecast",
            conditions=(
                ArchetypeCondition("text_cloud", "flag"),
                ArchetypeCondition("forecast_max_c", "high", 0.70),
            ),
            features=(
                "text_cloud",
                "text_sunny_or_fine",
                "text_hot",
                "forecast_max_c",
                "forecast_min_c",
                "forecast_range_c",
                "daily_hong_kong_observatory_mean_cloud_amount_lag7_roll7",
                "rh_min_pct",
                "isd_temp_dewpoint_spread_mean_c",
                "thermal_590960_minus_596730_c",
                "isd_graph_laplacian_mode_3",
                "month",
                "monsoon_phase_code",
            ),
        ),
        "pressure_slope_windy": SmoothArchetypeFamily(
            name="pressure_slope_windy",
            conditions=(
                ArchetypeCondition("pressure_plane_slope_magnitude_hpa_per_deg", "high", 0.75),
                ArchetypeCondition("isd_wind_speed_mean_mps", "high", 0.65),
            ),
            features=(
                "pressure_plane_slope_magnitude_hpa_per_deg",
                "isd_pressure_plane_lat_slope_hpa_per_deg",
                "isd_pressure_plane_lon_slope_hpa_per_deg",
                "slp_590960_minus_596730_hpa",
                "slp_590870_minus_596730_hpa",
                "isd_pressure_tendency_morning_midday_hpa",
                "isd_wind_speed_mean_mps",
                "isd_wind_speed_max_mps",
                "isd_onshore_easterly_proxy_mps",
                "forecast_max_vs_prior7_mean_source_c",
                "month",
                "monsoon_phase_code",
            ),
        ),
    }


def build_smooth_archetype_specs(frame: pd.DataFrame) -> list[SmoothArchetypeSpec]:
    specs: list[SmoothArchetypeSpec] = []
    for family in family_templates().values():
        condition_features = tuple(condition.feature for condition in family.conditions)
        available_conditions = available_numeric_features(frame, condition_features, min_non_null=300)
        if set(available_conditions) != set(condition_features):
            continue
        features = available_numeric_features(frame, family.features, min_non_null=300)
        if len(features) < 4:
            continue
        for anchor_col in ("prediction_0018_c", "prediction_0026_c"):
            if anchor_col not in frame.columns:
                continue
            for same_source in (False, True):
                for k_neighbors in (40, 80):
                    for half_life_days in (None, 730.0):
                        specs.append(
                            SmoothArchetypeSpec(
                                family_name=family.name,
                                anchor_col=anchor_col,
                                conditions=family.conditions,
                                features=features,
                                k_neighbors=k_neighbors,
                                same_source=same_source,
                                half_life_days=half_life_days,
                            )
                        )
    return specs


def score_smooth_candidate(
    predictions: pd.DataFrame,
    spec: SmoothArchetypeSpec,
    candidate_id: str,
) -> dict[str, object]:
    candidate = score_prediction_frame(predictions.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    anchor = score_prediction_frame(predictions.rename(columns={"anchor_prediction_c": "prediction"}), "prediction")
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "prediction"}), "prediction")
    corrected = predictions["do_no_harm_gate_passed"].astype(bool)
    active = predictions["current_archetype_match"].astype(bool)
    return {
        "candidate_id": candidate_id,
        "family_name": spec.family_name,
        "anchor_col": spec.anchor_col,
        "features": ",".join(spec.features),
        "feature_count": len(spec.features),
        "k_neighbors": spec.k_neighbors,
        "same_source": spec.same_source,
        "half_life_days": "" if spec.half_life_days is None else spec.half_life_days,
        "conditions": ";".join(
            f"{condition.feature}:{condition.direction}:{condition.quantile}" for condition in spec.conditions
        ),
        **candidate,
        "anchor_same_rows_mae": anchor["mae"],
        "anchor_same_rows_rmse": anchor["rmse"],
        "delta_vs_anchor_same_rows": float(candidate["mae"] - anchor["mae"]),
        "official_same_rows_mae": official["mae"],
        "official_same_rows_rmse": official["rmse"],
        "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
        "active_rows": int(active.sum()),
        "corrected_rows": int(corrected.sum()),
        "fallback_rows": int((~corrected).sum()),
        "mean_matched_archetype_rows": float(predictions.loc[corrected, "matched_archetype_rows"].mean())
        if corrected.any()
        else 0.0,
        "mean_neighbor_rows_used": float(predictions.loc[corrected, "neighbor_rows_used"].mean())
        if corrected.any()
        else 0.0,
        "mean_abs_correction_c": float(predictions.loc[corrected, "residual_correction_c"].abs().mean())
        if corrected.any()
        else 0.0,
        "mean_local_anchor_mae": float(predictions.loc[corrected, "local_anchor_mae"].mean())
        if corrected.any()
        else math.nan,
        "mean_local_corrected_mae": float(predictions.loc[corrected, "local_corrected_mae"].mean())
        if corrected.any()
        else math.nan,
    }


def run_smooth_archetype_screen(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    spec_rows: list[dict[str, object]] = []
    for spec in build_smooth_archetype_specs(frame):
        candidate_id = candidate_id_for_spec(spec)
        predictions = past_only_smooth_archetype_predictions(frame, spec)
        predictions["candidate_id"] = candidate_id
        predictions["family_name"] = spec.family_name
        predictions["anchor_col"] = spec.anchor_col
        predictions["same_source"] = spec.same_source
        predictions["half_life_days"] = "" if spec.half_life_days is None else spec.half_life_days
        score_rows.append(score_smooth_candidate(predictions, spec, candidate_id))
        prediction_frames.append(predictions)
        spec_rows.append(
            {
                "candidate_id": candidate_id,
                "family_name": spec.family_name,
                "anchor_col": spec.anchor_col,
                "same_source": spec.same_source,
                "k_neighbors": spec.k_neighbors,
                "half_life_days": "" if spec.half_life_days is None else spec.half_life_days,
                "conditions": ";".join(
                    f"{condition.feature}:{condition.direction}:{condition.quantile}" for condition in spec.conditions
                ),
                "features": ",".join(spec.features),
            }
        )
    scoreboard = pd.DataFrame(score_rows)
    if not scoreboard.empty:
        scoreboard = scoreboard.sort_values(["mae", "rmse"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    specs = pd.DataFrame(spec_rows)
    return scoreboard, predictions, specs


def feature_family_summary(scoreboard: pd.DataFrame) -> pd.DataFrame:
    if scoreboard.empty:
        return pd.DataFrame()
    return (
        scoreboard.groupby(["family_name", "anchor_col"], observed=True)
        .agg(
            candidates=("candidate_id", "count"),
            best_mae=("mae", "min"),
            best_rmse=("rmse", "min"),
            best_delta_vs_anchor=("delta_vs_anchor_same_rows", "min"),
            best_delta_vs_official=("delta_vs_official_same_rows", "min"),
            max_corrected_rows=("corrected_rows", "max"),
            median_corrected_rows=("corrected_rows", "median"),
        )
        .reset_index()
        .sort_values(["best_mae", "best_rmse"])
    )


def build_blend_frame(
    frame: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = frame[["target_date", "forecast_source_family", "target_tmax_c", "official_raw"]].copy()
    if scoreboard.empty or predictions.empty:
        return base, pd.DataFrame()
    top_ids = scoreboard.head(TOP_BLEND_EXPERTS)["candidate_id"].to_list()
    mapping = scoreboard[scoreboard["candidate_id"].isin(top_ids)][
        [
            "candidate_id",
            "family_name",
            "anchor_col",
            "features",
            "feature_count",
            "k_neighbors",
            "same_source",
            "half_life_days",
            "mae",
            "rmse",
            "delta_vs_official_same_rows",
            "delta_vs_anchor_same_rows",
            "corrected_rows",
        ]
    ].copy()
    mapping["expert_id"] = [
        f"sarc_{rank:02d}_{slug(row.candidate_id, limit=42)}"
        for rank, row in enumerate(mapping.itertuples(index=False), start=1)
    ]
    long = predictions[predictions["candidate_id"].isin(top_ids)][
        ["target_date", "candidate_id", "candidate_prediction_c"]
    ].copy()
    long = long.merge(mapping[["candidate_id", "expert_id"]], on="candidate_id", how="inner")
    wide = (
        long.pivot_table(index="target_date", columns="expert_id", values="candidate_prediction_c", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return base.merge(wide, on="target_date", how="left").sort_values("target_date").reset_index(drop=True), mapping


def run_blend_screen(
    frame: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blend_frame, mapping = build_blend_frame(frame, scoreboard, predictions)
    if mapping.empty:
        return pd.DataFrame(), pd.DataFrame(), mapping
    experts = ["official_raw", *[column for column in blend_frame.columns if column.startswith("sarc_")]]
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for mode in ("best", "inverse_mae"):
        for same_source in (False, True):
            candidate_id = f"smooth_archetype_blend_{mode}_{'same_source' if same_source else 'all_prior'}"
            blend_predictions = past_only_expert_blend(
                blend_frame,
                experts=experts,
                mode=mode,
                same_source=same_source,
                min_history=MIN_HISTORY,
            )
            blend_predictions["candidate_id"] = candidate_id
            candidate = score_prediction_frame(
                blend_predictions.rename(columns={"expert_prediction_c": "prediction"}), "prediction"
            )
            official = score_prediction_frame(blend_predictions.rename(columns={"official_raw": "prediction"}), "prediction")
            score_rows.append(
                {
                    "candidate_id": candidate_id,
                    "mode": mode,
                    "same_source": same_source,
                    **candidate,
                    "official_same_rows_mae": official["mae"],
                    "official_same_rows_rmse": official["rmse"],
                    "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
                    "fallback_rows": int(blend_predictions["selected_expert"].eq("official_raw_fallback").sum()),
                }
            )
            prediction_frames.append(blend_predictions)
    blend_scoreboard = pd.DataFrame(score_rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    return blend_scoreboard, pd.concat(prediction_frames, ignore_index=True), mapping


def prior_baseline_comparison(
    frame: pd.DataFrame,
    smooth_scoreboard: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prior_paths = [
        (
            "0032_best_archetype",
            RESEARCH_ROOT / "0032_residual_failure_cluster_discovery" / "artifacts" / "archetype_scoreboard.csv",
        ),
        (
            "0032_best_archetype_blend",
            RESEARCH_ROOT / "0032_residual_failure_cluster_discovery" / "artifacts" / "blend_scoreboard.csv",
        ),
        (
            "0031_best_selector",
            RESEARCH_ROOT / "0031_regime_gated_specialist_selector" / "artifacts" / "selector_scoreboard.csv",
        ),
        (
            "0030_best_local",
            RESEARCH_ROOT / "0030_multi_signal_local_residual_lab" / "artifacts" / "local_scoreboard.csv",
        ),
    ]
    for system, path in prior_paths:
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if table.empty:
            continue
        best = table.sort_values(["mae", "rmse"]).iloc[0]
        rows.append(
            {
                "system": system,
                "candidate_id": str(best["candidate_id"]),
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "delta_vs_official": float(best.get("delta_vs_official_same_rows", math.nan)),
                "n": int(best.get("n", 0)),
                "first_date": str(best.get("first_date", "")),
                "last_date": str(best.get("last_date", "")),
            }
        )
    for system, col in [
        ("official_raw", "official_raw"),
        ("0018_official_expert_blend", "prediction_0018_c"),
        ("0026_pressure_gradient_blend", "prediction_0026_c"),
    ]:
        score = score_prediction_frame(frame.rename(columns={col: "prediction"}), "prediction")
        official = score_prediction_frame(frame.rename(columns={"official_raw": "prediction"}), "prediction")
        rows.append(
            {
                "system": system,
                "candidate_id": system,
                "mae": score["mae"],
                "rmse": score["rmse"],
                "delta_vs_official": float(score["mae"] - official["mae"]),
                "n": score["n"],
                "first_date": score["first_date"],
                "last_date": score["last_date"],
            }
        )
    if not smooth_scoreboard.empty:
        best = smooth_scoreboard.iloc[0]
        rows.append(
            {
                "system": "0033_best_smooth_archetype",
                "candidate_id": str(best["candidate_id"]),
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "delta_vs_official": float(best["delta_vs_official_same_rows"]),
                "n": int(best["n"]),
                "first_date": str(best["first_date"]),
                "last_date": str(best["last_date"]),
            }
        )
    if not blend_scoreboard.empty:
        best = blend_scoreboard.iloc[0]
        rows.append(
            {
                "system": "0033_best_smooth_blend",
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
    summary: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best_single = scoreboard.iloc[0] if not scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    best_single_text = "No smooth archetype specialist was scoreable."
    if best_single is not None:
        best_single_text = (
            f"Best smooth archetype: `{best_single['candidate_id']}` with MAE `{best_single['mae']:.4f}`, "
            f"RMSE `{best_single['rmse']:.4f}`, official delta "
            f"`{best_single['delta_vs_official_same_rows']:.4f}`, and anchor delta "
            f"`{best_single['delta_vs_anchor_same_rows']:.4f}`."
        )
    best_blend_text = "No smooth archetype blend was scoreable."
    if best_blend is not None:
        best_blend_text = (
            f"Best smooth archetype blend: `{best_blend['candidate_id']}` with MAE `{best_blend['mae']:.4f}`, "
            f"RMSE `{best_blend['rmse']:.4f}`, and official delta "
            f"`{best_blend['delta_vs_official_same_rows']:.4f}`."
        )

    readme = f"""# Smooth Residual Archetype Specialists

Generated: `{manifest['generated_at_utc']}`

## Purpose

`0032` proved that hard residual archetypes can beat the prior `0018` champion, but the lift was very small. This insight tests whether the same failure families improve when converted into smoother specialists:

- prior-quantile archetype gates define whether today's setup belongs to a failure family;
- nearest historical neighbors are selected inside that archetype using only pre-cutoff features;
- feature scaling is recomputed inside the prior slice only;
- optional half-life weighting emphasizes more recent historical analogs;
- a do-no-harm gate applies the correction only when the same local prior neighborhood would have improved the anchor forecast.

## Data Window

Rows used: `{manifest['official_rows']}` official forecast/target rows.

Date range: `{manifest['first_target_date']}` to `{manifest['last_target_date']}`.

Source counts: `{manifest['source_counts']}`.

## Leakage Contract

- All scored rows are earlier than `{CONFIRMATION_START.date()}`.
- Each target date uses only rows with strictly earlier target dates.
- Archetype thresholds are calculated from the prior slice only.
- Feature scaling, nearest-neighbor distances, residual averages, half-life weights, and do-no-harm checks are all fold-local.
- Same-source variants restrict prior history to the same official forecast source family.
- 2024+ confirmation labels are not loaded or scored.

## Main Results

{best_single_text}

{best_blend_text}

## Baseline Comparison

{markdown_table(comparison, max_rows=20)}

## Family Summary

{markdown_table(summary, max_rows=30)}

## Smooth Archetype Scoreboard

{markdown_table(scoreboard.head(30), max_rows=30)}

## Blend Scoreboard

{markdown_table(blend_scoreboard, max_rows=20)}

## Interpretation

This run directly tests whether the hard `0032` archetype finding is improved by local smoothing and time weighting. A new champion would justify promoting the corresponding family into a richer residual architecture. If the best smooth specialist does not beat `0032`, the negative result means the current archetype gates are useful for diagnosis but not yet precise enough for smooth correction; the next move should use the cluster centroids themselves as soft features rather than only the manually named archetype gates.
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Smooth Residual Archetype Specialists\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_smooth_residual_archetype_specialists.py`:

- `{FOLDER_NAME}`: smooth prior-only residual specialists built from the strongest `0032` failure archetypes, with nearest-neighbor residual correction, optional half-life decay, and fold-local do-no-harm gates.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Smooth candidates | {manifest['smooth_candidates']} |
| Blend candidates | {manifest['blend_candidates']} |
| Best smooth MAE | {manifest['best_smooth_mae']} |
| Best smooth RMSE | {manifest['best_smooth_rmse']} |
| Best smooth delta vs official | {manifest['best_smooth_delta_vs_official']} |
| Best smooth delta vs anchor | {manifest['best_smooth_delta_vs_anchor']} |
| Best blend MAE | {manifest['best_blend_mae']} |
| Best blend RMSE | {manifest['best_blend_rmse']} |
| Best blend delta vs official | {manifest['best_blend_delta_vs_official']} |

Leakage contract: all scored rows are before `{CONFIRMATION_START.date()}`; thresholds, scaling, neighbor selection, half-life weighting, local do-no-harm checks, and blend weights all use strictly prior target dates.
"""
    write_text(index_path, text)


def write_outputs(
    *,
    frame: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
    specs: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
    blend_predictions: pd.DataFrame,
    blend_mapping: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    summary = feature_family_summary(scoreboard)
    comparison = prior_baseline_comparison(frame, scoreboard, blend_scoreboard)
    top_ids = set(scoreboard.head(40)["candidate_id"].to_list()) if not scoreboard.empty else set()

    write_csv(artifacts / "smooth_specs.csv", specs)
    write_csv(artifacts / "smooth_scoreboard.csv", scoreboard)
    write_csv(artifacts / "family_summary.csv", summary)
    write_csv(
        artifacts / "top_smooth_predictions.csv",
        predictions[predictions["candidate_id"].isin(top_ids)].copy() if not predictions.empty else predictions,
    )
    write_csv(artifacts / "blend_scoreboard.csv", blend_scoreboard)
    write_csv(artifacts / "blend_predictions.csv", blend_predictions)
    write_csv(artifacts / "blend_mapping.csv", blend_mapping)
    write_csv(artifacts / "baseline_comparison.csv", comparison)

    best_single = scoreboard.iloc[0] if not scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "source_counts": {str(k): int(v) for k, v in frame["forecast_source_family"].value_counts().to_dict().items()},
        "smooth_candidates": int(len(scoreboard)),
        "blend_candidates": int(len(blend_scoreboard)),
        "best_smooth": "" if best_single is None else str(best_single["candidate_id"]),
        "best_smooth_mae": None if best_single is None else float(best_single["mae"]),
        "best_smooth_rmse": None if best_single is None else float(best_single["rmse"]),
        "best_smooth_delta_vs_official": None
        if best_single is None
        else float(best_single["delta_vs_official_same_rows"]),
        "best_smooth_delta_vs_anchor": None if best_single is None else float(best_single["delta_vs_anchor_same_rows"]),
        "best_blend": "" if best_blend is None else str(best_blend["candidate_id"]),
        "best_blend_mae": None if best_blend is None else float(best_blend["mae"]),
        "best_blend_rmse": None if best_blend is None else float(best_blend["rmse"]),
        "best_blend_delta_vs_official": None
        if best_blend is None
        else float(best_blend["delta_vs_official_same_rows"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "smooth_residual_archetype_specialists_manifest.json", manifest)
    write_readme(
        folder=folder,
        manifest=manifest,
        scoreboard=scoreboard,
        summary=summary,
        blend_scoreboard=blend_scoreboard,
        comparison=comparison,
    )
    update_master_index(manifest)
    return manifest


def run() -> dict[str, object]:
    frame, _prior_systems = build_failure_frame()
    require_no_confirmation_dates(frame["target_date"], context="smooth residual archetype specialists")
    scoreboard, predictions, specs = run_smooth_archetype_screen(frame)
    blend_scoreboard, blend_predictions, blend_mapping = run_blend_screen(frame, scoreboard, predictions)
    return write_outputs(
        frame=frame,
        scoreboard=scoreboard,
        predictions=predictions,
        specs=specs,
        blend_scoreboard=blend_scoreboard,
        blend_predictions=blend_predictions,
        blend_mapping=blend_mapping,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 smooth residual archetype specialists.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
