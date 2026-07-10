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
from scripts.run_hkg_t24_multi_signal_local_residual_lab import (  # noqa: E402
    build_multisignal_frame,
)
from scripts.run_hkg_t24_official_anchor_expert_blend_screen import (  # noqa: E402
    past_only_expert_blend,
)
from scripts.run_hkg_t24_regime_gated_specialist_selector import (  # noqa: E402
    add_regimes,
)

FOLDER_NAME = "0032_residual_failure_cluster_discovery"
MIN_NON_NULL = 650
TOP_BLEND_ARCHETYPES = 12


@dataclass(frozen=True)
class ArchetypeCondition:
    feature: str
    direction: str
    quantile: float = 0.75


@dataclass(frozen=True)
class ArchetypeSpec:
    name: str
    anchor_col: str
    conditions: tuple[ArchetypeCondition, ...]
    same_source: bool
    min_history: int = 160
    min_match_rows: int = 25
    shrinkage: float = 80.0
    correction_clip_c: float = 2.5


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 150) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def archetype_id(spec: ArchetypeSpec) -> str:
    source = "same_source" if spec.same_source else "all_prior"
    return slug(f"archetype_{spec.anchor_col}_{spec.name}_{source}")


def available_numeric_features(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    min_non_null: int = MIN_NON_NULL,
) -> tuple[str, ...]:
    out: list[str] = []
    for feature in candidates:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if int(values.notna().sum()) >= min_non_null and values.nunique(dropna=True) > 1:
            out.append(feature)
    return tuple(out)


def load_best_predictions(
    *,
    scoreboard_path: Path,
    predictions_path: Path,
    output_col: str,
    context: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not scoreboard_path.exists():
        raise FileNotFoundError(f"Missing {context} scoreboard: {scoreboard_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing {context} predictions: {predictions_path}")

    scoreboard = pd.read_csv(scoreboard_path)
    if scoreboard.empty or "candidate_id" not in scoreboard.columns:
        raise ValueError(f"{context} scoreboard is empty or missing candidate_id")
    best = scoreboard.sort_values(["mae", "rmse"], na_position="last").iloc[0]
    best_id = str(best["candidate_id"])

    predictions = pd.read_csv(predictions_path)
    required = {"target_date", "forecast_source_family", "target_tmax_c", "candidate_id", "expert_prediction_c"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"{context} predictions missing columns: {sorted(missing)}")
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(predictions["target_date"], context=f"{context} prediction load")

    selected = predictions[predictions["candidate_id"].eq(best_id)].copy()
    if selected.empty:
        raise ValueError(f"{context} prediction table has no rows for best candidate {best_id}")
    selected = selected[
        ["target_date", "forecast_source_family", "target_tmax_c", "expert_prediction_c", "candidate_id"]
    ].copy()
    selected = selected.rename(columns={"expert_prediction_c": output_col})
    meta = {
        "candidate_id": best_id,
        "mae": float(best["mae"]),
        "rmse": float(best["rmse"]),
        "delta_vs_official": float(best.get("delta_vs_official_same_rows", math.nan)),
    }
    return selected, meta


def build_failure_frame() -> tuple[pd.DataFrame, dict[str, object]]:
    frame = add_regimes(build_multisignal_frame())
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(frame["target_date"], context="residual failure cluster base frame")
    frame["official_raw"] = pd.to_numeric(frame["forecast_max_c"], errors="coerce")
    if "isd_north_south_temp_gradient_c" in frame.columns:
        frame["abs_north_south_temp_gradient_c"] = pd.to_numeric(
            frame["isd_north_south_temp_gradient_c"], errors="coerce"
        ).abs()

    prior_0018, meta_0018 = load_best_predictions(
        scoreboard_path=RESEARCH_ROOT
        / "0018_past_only_official_expert_blend_screen"
        / "artifacts"
        / "scoreboard.csv",
        predictions_path=RESEARCH_ROOT
        / "0018_past_only_official_expert_blend_screen"
        / "artifacts"
        / "predictions.csv",
        output_col="prediction_0018_c",
        context="0018 official expert blend",
    )
    prior_0026, meta_0026 = load_best_predictions(
        scoreboard_path=RESEARCH_ROOT / "0026_pressure_gradient_experts" / "artifacts" / "blend_scoreboard.csv",
        predictions_path=RESEARCH_ROOT / "0026_pressure_gradient_experts" / "artifacts" / "blend_predictions.csv",
        output_col="prediction_0026_c",
        context="0026 pressure gradient blend",
    )

    merge_keys = ["target_date", "forecast_source_family", "target_tmax_c"]
    frame = frame.merge(prior_0018[merge_keys + ["prediction_0018_c"]], on=merge_keys, how="left", validate="one_to_one")
    frame = frame.merge(prior_0026[merge_keys + ["prediction_0026_c"]], on=merge_keys, how="left", validate="one_to_one")
    for col in ("official_raw", "prediction_0018_c", "prediction_0026_c"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame[f"{col}_error_c"] = frame[col] - pd.to_numeric(frame["target_tmax_c"], errors="coerce")
        frame[f"{col}_abs_error_c"] = frame[f"{col}_error_c"].abs()
    return frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True), {
        "0018": meta_0018,
        "0026": meta_0026,
    }


def diagnostic_feature_candidates() -> tuple[str, ...]:
    return (
        "forecast_max_c",
        "forecast_min_c",
        "forecast_range_c",
        "forecast_midpoint_c",
        "forecast_max_change_1_source_c",
        "forecast_max_prior7_std_source_c",
        "forecast_max_vs_prior7_mean_source_c",
        "month",
        "monsoon_phase_code",
        "text_any_rain",
        "text_showers",
        "text_thunder",
        "text_cloud",
        "text_sunny_or_fine",
        "text_hot",
        "text_very_hot",
        "text_humid",
        "text_mist_fog_haze",
        "text_wind",
        "text_easterly",
        "text_northerly",
        "text_southerly",
        "text_keyword_count",
        "isd_dew_point_mean_c_change_1d",
        "isd_temp_dewpoint_spread_mean_c",
        "isd_dewpoint_midday_minus_temp_c",
        "daily_hong_kong_observatory_mean_relative_humidity_lag7_roll7",
        "rh_min_pct",
        "rh_max_pct",
        "dew_590960_minus_596730_c",
        "dew_590870_minus_592780_c",
        "isd_pressure_plane_lat_slope_hpa_per_deg",
        "isd_pressure_plane_lon_slope_hpa_per_deg",
        "pressure_plane_slope_magnitude_hpa_per_deg",
        "isd_pressure_tendency_morning_midday_hpa",
        "isd_pressure_mean_hpa_change_1d",
        "isd_pressure_mean_hpa_roll7_mean",
        "slp_590960_minus_596730_hpa",
        "slp_590870_minus_596730_hpa",
        "isd_north_south_temp_gradient_c",
        "abs_north_south_temp_gradient_c",
        "isd_east_west_temp_gradient_c",
        "isd_graph_laplacian_mode_1",
        "isd_graph_laplacian_mode_3",
        "isd_graph_total_variation_c2",
        "thermal_590960_minus_596730_c",
        "thermal_590870_minus_596730_c",
        "isd_wind_speed_mean_mps",
        "isd_wind_speed_max_mps",
        "isd_onshore_easterly_proxy_mps",
        "isd_northerly_proxy_mps",
        "ua_theta_e_1000_850_mean_k",
        "ua_mse_1000_850_mean_kj_kg",
        "ua_dewpoint_925hpa_c",
        "igra_thickness_1000_500_m_change_48h",
    )


def standardize_feature_matrix(
    reference: pd.DataFrame,
    subset: pd.DataFrame,
    features: tuple[str, ...],
) -> np.ndarray:
    if not features:
        return np.empty((len(subset), 0), dtype=float)
    ref = reference.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    sub = subset.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    means = ref.mean(axis=0)
    stds = ref.std(axis=0, ddof=0).replace(0.0, np.nan).fillna(1.0)
    scaled = (sub - means) / stds
    return scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def simple_kmeans(matrix: np.ndarray, n_clusters: int, *, max_iter: int = 40) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    n_rows = matrix.shape[0]
    if n_rows == 0:
        return np.array([], dtype=int)
    n_clusters = max(1, min(int(n_clusters), n_rows))
    if n_clusters == 1:
        return np.zeros(n_rows, dtype=int)

    order = np.argsort(matrix.sum(axis=1))
    positions = np.linspace(0, n_rows - 1, n_clusters).round().astype(int)
    centers = matrix[order[positions]].copy()
    labels = np.full(n_rows, -1, dtype=int)
    for _ in range(max_iter):
        distances = np.square(matrix[:, None, :] - centers[None, :, :]).sum(axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster_id in range(n_clusters):
            member_mask = labels == cluster_id
            if member_mask.any():
                centers[cluster_id] = matrix[member_mask].mean(axis=0)
    return labels


def mode_text(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return ""
    return str(clean.mode().iloc[0])


def top_centroid_features(features: tuple[str, ...], centroid: np.ndarray, *, top_n: int = 5) -> list[str]:
    if not features or centroid.size == 0:
        return []
    order = np.argsort(np.abs(centroid))[::-1][:top_n]
    return [f"{features[index]}={centroid[index]:.2f}z" for index in order]


def diagnostic_failure_clusters(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    features = available_numeric_features(frame, diagnostic_feature_candidates())
    systems = [
        ("official_raw", "official_raw_error_c", "official_raw_abs_error_c"),
        ("0018_champion", "prediction_0018_c_error_c", "prediction_0018_c_abs_error_c"),
        ("0026_pressure", "prediction_0026_c_error_c", "prediction_0026_c_abs_error_c"),
    ]
    member_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for system, error_col, abs_col in systems:
        valid = frame[np.isfinite(pd.to_numeric(frame[abs_col], errors="coerce"))].copy()
        if valid.empty:
            continue
        threshold = max(float(valid[abs_col].quantile(0.75)), 1.0)
        failures = valid[valid[abs_col] >= threshold].copy()
        if failures.empty:
            continue
        n_clusters = min(8, max(2, len(failures) // 40))
        matrix = standardize_feature_matrix(valid, failures, features)
        labels = simple_kmeans(matrix, n_clusters)
        failures["system"] = system
        failures["cluster_id"] = labels
        failures["large_miss_threshold_c"] = threshold
        keep_cols = [
            "system",
            "cluster_id",
            "target_date",
            "forecast_source_family",
            "primary_regime",
            "target_tmax_c",
            "official_raw",
            "prediction_0018_c",
            "prediction_0026_c",
            error_col,
            abs_col,
            "large_miss_threshold_c",
        ]
        keep_cols = [col for col in keep_cols if col in failures.columns]
        member_frames.append(failures[keep_cols].copy())
        for cluster_id in sorted(failures["cluster_id"].unique()):
            mask = failures["cluster_id"].eq(cluster_id).to_numpy()
            subset = failures.loc[mask].copy()
            centroid = matrix[mask].mean(axis=0) if mask.any() else np.array([], dtype=float)
            error = pd.to_numeric(subset[error_col], errors="coerce")
            summary_rows.append(
                {
                    "system": system,
                    "cluster_id": int(cluster_id),
                    "rows": int(len(subset)),
                    "large_miss_threshold_c": threshold,
                    "first_date": str(subset["target_date"].min().date()),
                    "last_date": str(subset["target_date"].max().date()),
                    "abs_error_mean_c": float(pd.to_numeric(subset[abs_col], errors="coerce").mean()),
                    "abs_error_max_c": float(pd.to_numeric(subset[abs_col], errors="coerce").max()),
                    "error_bias_c": float(error.mean()),
                    "actual_hotter_than_prediction_rate": float((error < 0).mean()),
                    "actual_cooler_than_prediction_rate": float((error > 0).mean()),
                    "source_mode": mode_text(subset["forecast_source_family"]),
                    "primary_regime_mode": mode_text(subset["primary_regime"]),
                    "month_mode": mode_text(subset["month"]) if "month" in subset.columns else "",
                    "top_centroid_features": "; ".join(top_centroid_features(features, centroid)),
                }
            )
    members = pd.concat(member_frames, ignore_index=True) if member_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows).sort_values(["system", "abs_error_mean_c"], ascending=[True, False])
    return members, summary.reset_index(drop=True), features


def condition_prior_mask(
    ordered: pd.DataFrame,
    base_prior_mask: np.ndarray,
    current: pd.Series,
    condition: ArchetypeCondition,
) -> tuple[bool, np.ndarray, float]:
    values = pd.to_numeric(ordered[condition.feature], errors="coerce").to_numpy(dtype=float)
    current_value = pd.to_numeric(pd.Series([current[condition.feature]]), errors="coerce").iloc[0]
    current_value = float(current_value) if np.isfinite(current_value) else math.nan
    valid_prior = base_prior_mask & np.isfinite(values)
    if int(valid_prior.sum()) == 0 or not np.isfinite(current_value):
        return False, np.zeros(len(ordered), dtype=bool), math.nan

    if condition.direction == "flag":
        threshold = 0.5
        current_match = current_value >= threshold
        return bool(current_match), valid_prior & (values >= threshold), threshold

    threshold = float(np.nanquantile(values[valid_prior], condition.quantile))
    if condition.direction == "high":
        return bool(current_value >= threshold), valid_prior & (values >= threshold), threshold
    if condition.direction == "low":
        return bool(current_value <= threshold), valid_prior & (values <= threshold), threshold
    raise ValueError(f"Unsupported condition direction: {condition.direction}")


def past_only_archetype_predictions(frame: pd.DataFrame, spec: ArchetypeSpec) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    sources = ordered["forecast_source_family"].astype(str).to_numpy()
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    anchor = pd.to_numeric(ordered[spec.anchor_col], errors="coerce").to_numpy(dtype=float)
    residual = target - anchor

    predictions: list[float] = []
    corrections: list[float] = []
    prior_rows_used: list[int] = []
    matched_rows: list[int] = []
    current_matches: list[bool] = []
    used_corrections: list[bool] = []
    thresholds: list[str] = []

    for index, target_date in enumerate(dates):
        if not np.isfinite(anchor[index]):
            predictions.append(math.nan)
            corrections.append(0.0)
            prior_rows_used.append(0)
            matched_rows.append(0)
            current_matches.append(False)
            used_corrections.append(False)
            thresholds.append("")
            continue

        prior_limit = int(np.searchsorted(dates, target_date, side="left"))
        base_prior = np.arange(len(ordered)) < prior_limit
        if spec.same_source:
            base_prior &= sources == sources[index]
        base_prior &= np.isfinite(residual) & np.isfinite(anchor) & np.isfinite(target)
        prior_rows = int(base_prior.sum())
        if prior_rows < spec.min_history:
            predictions.append(float(anchor[index]))
            corrections.append(0.0)
            prior_rows_used.append(prior_rows)
            matched_rows.append(0)
            current_matches.append(False)
            used_corrections.append(False)
            thresholds.append("")
            continue

        current = ordered.iloc[index]
        match_mask = base_prior.copy()
        all_current_match = True
        threshold_parts: list[str] = []
        for condition in spec.conditions:
            current_match, condition_mask, threshold = condition_prior_mask(ordered, base_prior, current, condition)
            all_current_match = all_current_match and current_match
            match_mask &= condition_mask
            threshold_parts.append(f"{condition.feature}:{condition.direction}:{threshold:.4f}")

        matched = int(match_mask.sum()) if all_current_match else 0
        if not all_current_match or matched < spec.min_match_rows:
            predictions.append(float(anchor[index]))
            corrections.append(0.0)
            prior_rows_used.append(prior_rows)
            matched_rows.append(matched)
            current_matches.append(bool(all_current_match))
            used_corrections.append(False)
            thresholds.append(";".join(threshold_parts))
            continue

        raw_correction = float(np.nanmean(residual[match_mask]))
        shrink = matched / (matched + spec.shrinkage)
        correction = float(np.clip(raw_correction * shrink, -spec.correction_clip_c, spec.correction_clip_c))
        predictions.append(float(anchor[index] + correction))
        corrections.append(correction)
        prior_rows_used.append(prior_rows)
        matched_rows.append(matched)
        current_matches.append(True)
        used_corrections.append(True)
        thresholds.append(";".join(threshold_parts))

    out = ordered[["target_date", "forecast_source_family", "primary_regime", "target_tmax_c", "official_raw"]].copy()
    out["anchor_prediction_c"] = anchor
    out["candidate_prediction_c"] = predictions
    out["residual_correction_c"] = corrections
    out["prior_rows_used"] = prior_rows_used
    out["matched_archetype_rows"] = matched_rows
    out["current_archetype_match"] = current_matches
    out["used_archetype_correction"] = used_corrections
    out["condition_thresholds"] = thresholds
    return out


def archetype_condition_templates() -> dict[str, tuple[ArchetypeCondition, ...]]:
    return {
        "rain_hot_forecast": (
            ArchetypeCondition("text_any_rain", "flag"),
            ArchetypeCondition("forecast_max_c", "high", 0.70),
        ),
        "cloudy_high_forecast": (
            ArchetypeCondition("text_cloud", "flag"),
            ArchetypeCondition("forecast_max_c", "high", 0.70),
        ),
        "hot_sunny_high_forecast": (
            ArchetypeCondition("text_sunny_or_fine", "flag"),
            ArchetypeCondition("forecast_max_c", "high", 0.70),
        ),
        "dry_mixing_heat": (
            ArchetypeCondition("isd_temp_dewpoint_spread_mean_c", "high", 0.75),
            ArchetypeCondition("forecast_max_c", "high", 0.65),
        ),
        "moisture_surge": (
            ArchetypeCondition("isd_dew_point_mean_c_change_1d", "high", 0.75),
            ArchetypeCondition("isd_temp_dewpoint_spread_mean_c", "low", 0.35),
        ),
        "pressure_drop_warm": (
            ArchetypeCondition("isd_pressure_mean_hpa_change_1d", "low", 0.25),
            ArchetypeCondition("forecast_max_c", "high", 0.65),
        ),
        "pressure_slope_windy": (
            ArchetypeCondition("pressure_plane_slope_magnitude_hpa_per_deg", "high", 0.75),
            ArchetypeCondition("isd_wind_speed_mean_mps", "high", 0.65),
        ),
        "station_gradient_heat": (
            ArchetypeCondition("abs_north_south_temp_gradient_c", "high", 0.75),
            ArchetypeCondition("forecast_max_c", "high", 0.65),
        ),
        "upper_air_warm_moist": (
            ArchetypeCondition("ua_mse_1000_850_mean_kj_kg", "high", 0.75),
            ArchetypeCondition("igra_thickness_1000_500_m_change_48h", "high", 0.65),
        ),
        "forecast_jump_up": (
            ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.75),
            ArchetypeCondition("forecast_max_vs_prior7_mean_source_c", "high", 0.65),
        ),
        "marine_onshore_wind": (
            ArchetypeCondition("isd_onshore_easterly_proxy_mps", "high", 0.75),
            ArchetypeCondition("isd_wind_speed_mean_mps", "high", 0.65),
        ),
        "hot_cloud_conflict": (
            ArchetypeCondition("text_cloud", "flag"),
            ArchetypeCondition("text_hot", "flag"),
        ),
        "humid_thunder": (
            ArchetypeCondition("text_thunder", "flag"),
            ArchetypeCondition("rh_min_pct", "high", 0.70),
        ),
        "dry_high_range": (
            ArchetypeCondition("forecast_range_c", "high", 0.70),
            ArchetypeCondition("isd_temp_dewpoint_spread_mean_c", "high", 0.70),
        ),
    }


def build_archetype_specs(frame: pd.DataFrame) -> list[ArchetypeSpec]:
    available = set(available_numeric_features(frame, tuple(sorted(set(diagnostic_feature_candidates()) | {"abs_north_south_temp_gradient_c"}))))
    anchors = [col for col in ("official_raw", "prediction_0018_c", "prediction_0026_c") if col in frame.columns]
    specs: list[ArchetypeSpec] = []
    for name, conditions in archetype_condition_templates().items():
        if any(condition.feature not in available for condition in conditions):
            continue
        for anchor in anchors:
            for same_source in (False, True):
                specs.append(ArchetypeSpec(name=name, anchor_col=anchor, conditions=conditions, same_source=same_source))
    return specs


def score_archetype_candidate(predictions: pd.DataFrame, spec: ArchetypeSpec, candidate_id: str) -> dict[str, object]:
    candidate = score_prediction_frame(predictions.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    anchor = score_prediction_frame(predictions.rename(columns={"anchor_prediction_c": "prediction"}), "prediction")
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "prediction"}), "prediction")
    corrected = predictions["used_archetype_correction"].astype(bool)
    active = predictions["current_archetype_match"].astype(bool)
    return {
        "candidate_id": candidate_id,
        "archetype_name": spec.name,
        "anchor_col": spec.anchor_col,
        "same_source": spec.same_source,
        "conditions": ";".join(
            f"{condition.feature}:{condition.direction}:{condition.quantile}" for condition in spec.conditions
        ),
        "min_history": spec.min_history,
        "min_match_rows": spec.min_match_rows,
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
        "mean_abs_correction_c": float(predictions.loc[corrected, "residual_correction_c"].abs().mean())
        if corrected.any()
        else 0.0,
    }


def run_archetype_screen(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = build_archetype_specs(frame)
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    spec_rows: list[dict[str, object]] = []
    for spec in specs:
        candidate_id = archetype_id(spec)
        predictions = past_only_archetype_predictions(frame, spec)
        predictions["candidate_id"] = candidate_id
        predictions["archetype_name"] = spec.name
        predictions["anchor_col"] = spec.anchor_col
        predictions["same_source"] = spec.same_source
        score_rows.append(score_archetype_candidate(predictions, spec, candidate_id))
        prediction_frames.append(predictions)
        spec_rows.append(
            {
                "candidate_id": candidate_id,
                "archetype_name": spec.name,
                "anchor_col": spec.anchor_col,
                "same_source": spec.same_source,
                "conditions": ";".join(
                    f"{condition.feature}:{condition.direction}:{condition.quantile}" for condition in spec.conditions
                ),
            }
        )
    scoreboard = pd.DataFrame(score_rows)
    if not scoreboard.empty:
        scoreboard = scoreboard.sort_values(["mae", "rmse"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    specs_frame = pd.DataFrame(spec_rows)
    return scoreboard, predictions, specs_frame


def build_archetype_blend_frame(
    frame: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = frame[["target_date", "forecast_source_family", "target_tmax_c", "official_raw"]].copy()
    if scoreboard.empty or predictions.empty:
        return base, pd.DataFrame()
    top_ids = scoreboard.head(TOP_BLEND_ARCHETYPES)["candidate_id"].to_list()
    mapping = scoreboard[scoreboard["candidate_id"].isin(top_ids)][
        [
            "candidate_id",
            "archetype_name",
            "anchor_col",
            "same_source",
            "conditions",
            "mae",
            "rmse",
            "delta_vs_official_same_rows",
            "delta_vs_anchor_same_rows",
            "corrected_rows",
        ]
    ].copy()
    mapping["expert_id"] = [
        f"arc_{rank:02d}_{slug(row.candidate_id, limit=44)}"
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


def run_archetype_blends(
    frame: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blend_frame, mapping = build_archetype_blend_frame(frame, scoreboard, predictions)
    if mapping.empty:
        return pd.DataFrame(), pd.DataFrame(), mapping
    experts = ["official_raw", *[column for column in blend_frame.columns if column.startswith("arc_")]]
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for mode in ("best", "inverse_mae"):
        for same_source in (False, True):
            candidate_id = f"residual_archetype_blend_{mode}_{'same_source' if same_source else 'all_prior'}"
            blend_predictions = past_only_expert_blend(
                blend_frame,
                experts=experts,
                mode=mode,
                same_source=same_source,
                min_history=160,
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
    blend_predictions = pd.concat(prediction_frames, ignore_index=True)
    return blend_scoreboard, blend_predictions, mapping


def prior_baseline_rows(frame: pd.DataFrame) -> pd.DataFrame:
    paths = [
        (
            "0030_best_local",
            RESEARCH_ROOT / "0030_multi_signal_local_residual_lab" / "artifacts" / "local_scoreboard.csv",
        ),
        (
            "0030_best_blend",
            RESEARCH_ROOT / "0030_multi_signal_local_residual_lab" / "artifacts" / "blend_scoreboard.csv",
        ),
        (
            "0031_best_selector",
            RESEARCH_ROOT / "0031_regime_gated_specialist_selector" / "artifacts" / "selector_scoreboard.csv",
        ),
    ]
    rows: list[dict[str, object]] = []
    for system, path in paths:
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
    return pd.DataFrame(rows)


def baseline_comparison(
    frame: pd.DataFrame,
    archetype_scoreboard: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
) -> pd.DataFrame:
    rows = prior_baseline_rows(frame).to_dict("records")
    if not archetype_scoreboard.empty:
        best = archetype_scoreboard.iloc[0]
        rows.append(
            {
                "system": "0032_best_archetype",
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
                "system": "0032_best_archetype_blend",
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
    cluster_summary: pd.DataFrame,
    archetype_scoreboard: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best_archetype = archetype_scoreboard.iloc[0] if not archetype_scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    best_archetype_text = "No archetype specialist was scoreable."
    if best_archetype is not None:
        best_archetype_text = (
            f"Best archetype specialist: `{best_archetype['candidate_id']}` with MAE "
            f"`{best_archetype['mae']:.4f}`, RMSE `{best_archetype['rmse']:.4f}`, "
            f"official delta `{best_archetype['delta_vs_official_same_rows']:.4f}`, "
            f"and anchor delta `{best_archetype['delta_vs_anchor_same_rows']:.4f}`."
        )
    best_blend_text = "No archetype blend was scoreable."
    if best_blend is not None:
        best_blend_text = (
            f"Best archetype blend: `{best_blend['candidate_id']}` with MAE `{best_blend['mae']:.4f}`, "
            f"RMSE `{best_blend['rmse']:.4f}`, and official delta "
            f"`{best_blend['delta_vs_official_same_rows']:.4f}`."
        )

    readme = f"""# Residual Failure Cluster Discovery

Generated: `{manifest['generated_at_utc']}`

## Purpose

This insight moves from hand-written weather regimes to observed failure families. It asks:

1. Where do raw official, the `0018` expert blend, and the `0026` pressure blend still miss badly?
2. What pre-cutoff feature patterns describe those misses?
3. Can simple deployable residual archetypes correct any of those patterns without seeing future labels?

The diagnostic cluster layer is not itself deployable; it is an atlas of recurring miss families. The archetype layer is deployable-style: each correction uses only strictly earlier target dates and only current-day pre-cutoff features.

## Data Window

Rows used: `{manifest['official_rows']}` official forecast/target rows.

Date range: `{manifest['first_target_date']}` to `{manifest['last_target_date']}`.

Source counts: `{manifest['source_counts']}`.

Prior systems joined:

- `0018`: `{manifest['prior_systems']['0018']}`
- `0026`: `{manifest['prior_systems']['0026']}`

## Leakage Contract

- All scored target rows are earlier than `{CONFIRMATION_START.date()}`.
- 2024+ confirmation labels are rejected at load time.
- Diagnostic clusters use only pre-cutoff feature columns, not future target values as features.
- Archetype thresholds are recomputed from the fold-local prior slice only.
- Archetype residual means use only rows with strictly earlier target dates via `searchsorted(..., side="left")`.
- Same-source archetypes restrict prior history to the same official forecast source family.
- Blend weights and expert choices are estimated using only prior realized errors.

## Main Results

{best_archetype_text}

{best_blend_text}

## Baseline Comparison

{markdown_table(comparison, max_rows=20)}

## Diagnostic Failure Clusters

{markdown_table(cluster_summary.head(30), max_rows=30)}

## Archetype Scoreboard

{markdown_table(archetype_scoreboard.head(30), max_rows=30)}

## Archetype Blend Scoreboard

{markdown_table(blend_scoreboard, max_rows=20)}

## Interpretation

This run is deliberately split into diagnostics and deployable corrections. The cluster atlas exposes the dominant high-miss patterns for the current official archive and for the strongest prior systems. The archetype scores show whether those patterns can already be exploited using simple fold-local mean residual corrections. If the best archetype does not beat `0018`, the useful result is still the cluster map: the next step should turn the strongest cluster signatures into smoother, lower-sparsity specialists rather than hard two-condition gates.
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Residual Failure Cluster Discovery\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_residual_failure_cluster_discovery.py`:

- `{FOLDER_NAME}`: diagnostic failure clusters for raw official, `0018`, and `0026`, plus leakage-safe archetype residual specialists and archetype blends.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Diagnostic features | {manifest['diagnostic_feature_count']} |
| Diagnostic cluster rows | {manifest['diagnostic_cluster_rows']} |
| Archetype candidates | {manifest['archetype_candidates']} |
| Blend candidates | {manifest['blend_candidates']} |
| Best archetype MAE | {manifest['best_archetype_mae']} |
| Best archetype RMSE | {manifest['best_archetype_rmse']} |
| Best archetype delta vs official | {manifest['best_archetype_delta_vs_official']} |
| Best archetype delta vs anchor | {manifest['best_archetype_delta_vs_anchor']} |
| Best blend MAE | {manifest['best_blend_mae']} |
| Best blend RMSE | {manifest['best_blend_rmse']} |
| Best blend delta vs official | {manifest['best_blend_delta_vs_official']} |

Leakage contract: all scored rows are before `{CONFIRMATION_START.date()}`; cluster features are pre-cutoff only; archetype thresholds, residual corrections, and blend weights use strictly prior target dates.
"""
    write_text(index_path, text)


def write_outputs(
    *,
    frame: pd.DataFrame,
    prior_systems: dict[str, object],
    cluster_members: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    diagnostic_features: tuple[str, ...],
    archetype_specs: pd.DataFrame,
    archetype_scoreboard: pd.DataFrame,
    archetype_predictions: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
    blend_predictions: pd.DataFrame,
    blend_mapping: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    comparison = baseline_comparison(frame, archetype_scoreboard, blend_scoreboard)
    top_ids = set(archetype_scoreboard.head(40)["candidate_id"].to_list()) if not archetype_scoreboard.empty else set()

    write_csv(artifacts / "diagnostic_cluster_members.csv", cluster_members)
    write_csv(artifacts / "diagnostic_cluster_summary.csv", cluster_summary)
    write_csv(artifacts / "diagnostic_features.csv", pd.DataFrame({"feature": diagnostic_features}))
    write_csv(artifacts / "archetype_specs.csv", archetype_specs)
    write_csv(artifacts / "archetype_scoreboard.csv", archetype_scoreboard)
    write_csv(
        artifacts / "top_archetype_predictions.csv",
        archetype_predictions[archetype_predictions["candidate_id"].isin(top_ids)].copy()
        if not archetype_predictions.empty
        else archetype_predictions,
    )
    write_csv(artifacts / "blend_scoreboard.csv", blend_scoreboard)
    write_csv(artifacts / "blend_predictions.csv", blend_predictions)
    write_csv(artifacts / "blend_mapping.csv", blend_mapping)
    write_csv(artifacts / "baseline_comparison.csv", comparison)

    best_archetype = archetype_scoreboard.iloc[0] if not archetype_scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "source_counts": {str(k): int(v) for k, v in frame["forecast_source_family"].value_counts().to_dict().items()},
        "prior_systems": prior_systems,
        "diagnostic_feature_count": int(len(diagnostic_features)),
        "diagnostic_cluster_rows": int(len(cluster_members)),
        "diagnostic_clusters": int(len(cluster_summary)),
        "archetype_candidates": int(len(archetype_scoreboard)),
        "blend_candidates": int(len(blend_scoreboard)),
        "best_archetype": "" if best_archetype is None else str(best_archetype["candidate_id"]),
        "best_archetype_mae": None if best_archetype is None else float(best_archetype["mae"]),
        "best_archetype_rmse": None if best_archetype is None else float(best_archetype["rmse"]),
        "best_archetype_delta_vs_official": None
        if best_archetype is None
        else float(best_archetype["delta_vs_official_same_rows"]),
        "best_archetype_delta_vs_anchor": None
        if best_archetype is None
        else float(best_archetype["delta_vs_anchor_same_rows"]),
        "best_blend": "" if best_blend is None else str(best_blend["candidate_id"]),
        "best_blend_mae": None if best_blend is None else float(best_blend["mae"]),
        "best_blend_rmse": None if best_blend is None else float(best_blend["rmse"]),
        "best_blend_delta_vs_official": None
        if best_blend is None
        else float(best_blend["delta_vs_official_same_rows"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "residual_failure_cluster_discovery_manifest.json", manifest)
    write_readme(
        folder=folder,
        manifest=manifest,
        cluster_summary=cluster_summary,
        archetype_scoreboard=archetype_scoreboard,
        blend_scoreboard=blend_scoreboard,
        comparison=comparison,
    )
    update_master_index(manifest)
    return manifest


def run() -> dict[str, object]:
    frame, prior_systems = build_failure_frame()
    require_no_confirmation_dates(frame["target_date"], context="residual failure cluster run")
    cluster_members, cluster_summary, diagnostic_features = diagnostic_failure_clusters(frame)
    archetype_scoreboard, archetype_predictions, archetype_specs = run_archetype_screen(frame)
    blend_scoreboard, blend_predictions, blend_mapping = run_archetype_blends(
        frame, archetype_scoreboard, archetype_predictions
    )
    return write_outputs(
        frame=frame,
        prior_systems=prior_systems,
        cluster_members=cluster_members,
        cluster_summary=cluster_summary,
        diagnostic_features=diagnostic_features,
        archetype_specs=archetype_specs,
        archetype_scoreboard=archetype_scoreboard,
        archetype_predictions=archetype_predictions,
        blend_scoreboard=blend_scoreboard,
        blend_predictions=blend_predictions,
        blend_mapping=blend_mapping,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 residual failure cluster discovery.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
