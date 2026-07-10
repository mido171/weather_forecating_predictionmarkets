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
from scripts.run_hkg_t24_official_residual_source_text_range_dynamics import (  # noqa: E402
    TEXT_FLAG_FEATURES,
)
from scripts.run_hkg_t24_official_residual_source_text_range_dynamics import (  # noqa: E402
    build_analysis_frame as build_official_residual_frame,
)
from scripts.run_hkg_t24_pressure_gradient_experts import (  # noqa: E402
    add_pressure_gradient_features,
)
from scripts.run_hkg_t24_smooth_gated_pressure_experts import (  # noqa: E402
    smooth_residual_correction,
)

FOLDER_NAME = "0030_multi_signal_local_residual_lab"
MIN_HISTORY = 160
MIN_NON_NULL = 650
TOP_BLEND_EXPERTS = 16


@dataclass(frozen=True)
class FeatureSet:
    name: str
    groups: dict[str, tuple[str, ...]]

    @property
    def features(self) -> tuple[str, ...]:
        values: list[str] = []
        for group_features in self.groups.values():
            for feature in group_features:
                if feature not in values:
                    values.append(feature)
        return tuple(values)


@dataclass(frozen=True)
class LocalResidualSpec:
    feature_set: str
    features: tuple[str, ...]
    k_neighbors: int
    same_source: bool
    phase_conditioned: bool
    shrinkage: float = 100.0
    correction_clip_c: float = 2.5
    min_history: int = MIN_HISTORY


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 150) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def available_features(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    min_non_null: int = MIN_NON_NULL,
) -> tuple[str, ...]:
    features: list[str] = []
    for feature in candidates:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if int(values.notna().sum()) >= min_non_null and values.nunique(dropna=True) > 1:
            features.append(feature)
    return tuple(features)


def filtered_group(frame: pd.DataFrame, candidates: tuple[str, ...]) -> tuple[str, ...]:
    return available_features(frame, candidates)


def build_multisignal_frame() -> pd.DataFrame:
    frame = build_official_residual_frame()
    frame = add_pressure_gradient_features(frame)
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(frame["target_date"], context="multi-signal residual frame")
    frame["official_raw"] = pd.to_numeric(frame["forecast_max_c"], errors="coerce")
    frame["actual_minus_official_c"] = pd.to_numeric(frame["target_tmax_c"], errors="coerce") - frame["official_raw"]
    return frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True)


def build_base_feature_sets(frame: pd.DataFrame) -> dict[str, FeatureSet]:
    official_core = filtered_group(
        frame,
        (
            "forecast_max_c",
            "forecast_min_c",
            "forecast_range_c",
            "forecast_midpoint_c",
            "issue_to_cutoff_hours",
            "forecast_max_change_1_source_c",
            "forecast_midpoint_change_1_source_c",
            "forecast_max_prior7_std_source_c",
            "forecast_max_vs_prior7_mean_source_c",
            "month",
            "monsoon_phase_code",
            "source_is_press",
            "source_is_rss",
        ),
    )
    text = filtered_group(
        frame,
        tuple(
            feature
            for feature in (
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
            )
            if feature in TEXT_FLAG_FEATURES or feature == "text_keyword_count"
        ),
    )
    moisture = filtered_group(
        frame,
        (
            "isd_dew_point_mean_c_change_1d",
            "isd_temp_dewpoint_spread_mean_c",
            "isd_dewpoint_midday_minus_temp_c",
            "daily_hong_kong_observatory_mean_relative_humidity_lag7_roll7",
            "rh_min_pct",
            "rh_max_pct",
            "dew_590960_minus_596730_c",
            "dew_590870_minus_592780_c",
        ),
    )
    upper_air = filtered_group(
        frame,
        (
            "ua_theta_e_1000_850_mean_k",
            "ua_mse_1000_850_mean_kj_kg",
            "ua_mse_925_850_mean_kj_kg",
            "ua_dewpoint_925hpa_c",
            "igra_thickness_1000_500_m_change_48h",
            "ua_tendency_48h_ua_theta_1000hpa_k",
            "ua_wind_u_1000hpa_mps",
            "ua_wind_v_1000hpa_mps",
        ),
    )
    station_thermal = filtered_group(
        frame,
        (
            "isd_north_south_temp_gradient_c",
            "isd_east_west_temp_gradient_c",
            "isd_graph_laplacian_mode_1",
            "isd_graph_laplacian_mode_3",
            "isd_graph_total_variation_c2",
            "thermal_590960_minus_596730_c",
            "thermal_590870_minus_596730_c",
            "thermal_592930_minus_596730_c",
            "thermal_590960_minus_592780_c",
            "isd_morning_to_midday_temp_rise_c",
        ),
    )
    wind_marine = filtered_group(
        frame,
        (
            "isd_wind_speed_mean_mps",
            "isd_wind_speed_max_mps",
            "isd_wind_vector_speed_mps",
            "isd_onshore_easterly_proxy_mps",
            "isd_northerly_proxy_mps",
            "isd_wind_speed_mean_mps_change_1d",
            "daily_waglan_island_mean_wind_speed_lag7_roll7",
            "daily_waglan_island_prevailing_wind_direction_lag7_roll7",
        ),
    )
    cloud_rain = filtered_group(
        frame,
        (
            "daily_hong_kong_observatory_mean_cloud_amount_lag7_roll7",
            "daily_hong_kong_observatory_daily_rainfall_lag7_roll7",
        ),
    )
    pressure = filtered_group(
        frame,
        (
            "isd_pressure_plane_lat_slope_hpa_per_deg",
            "isd_pressure_plane_lon_slope_hpa_per_deg",
            "pressure_plane_slope_magnitude_hpa_per_deg",
            "isd_pressure_tendency_morning_midday_hpa",
            "isd_pressure_mean_hpa_change_1d",
            "isd_pressure_mean_hpa_roll7_mean",
            "slp_590960_minus_596730_hpa",
            "slp_590870_minus_596730_hpa",
            "slp_592780_minus_596730_hpa",
        ),
    )

    sets = {
        "official_text_weather": FeatureSet(
            "official_text_weather",
            {"official_core": official_core, "text": text, "cloud_rain": cloud_rain},
        ),
        "moisture_upper_air_heat": FeatureSet(
            "moisture_upper_air_heat",
            {"official_core": official_core, "moisture": moisture, "upper_air": upper_air},
        ),
        "station_wind_marine_network": FeatureSet(
            "station_wind_marine_network",
            {
                "official_core": official_core,
                "station_thermal": station_thermal,
                "wind_marine": wind_marine,
                "cloud_rain": cloud_rain,
            },
        ),
        "pressure_moisture_advection": FeatureSet(
            "pressure_moisture_advection",
            {
                "official_core": official_core,
                "pressure": pressure,
                "moisture": moisture,
                "upper_air": upper_air,
                "wind_marine": wind_marine,
            },
        ),
        "full_multisignal_compact": FeatureSet(
            "full_multisignal_compact",
            {
                "official_core": official_core,
                "text": text,
                "moisture": moisture,
                "upper_air": upper_air,
                "station_thermal": station_thermal,
                "wind_marine": wind_marine,
                "cloud_rain": cloud_rain,
                "pressure": pressure,
            },
        ),
    }
    return {name: feature_set for name, feature_set in sets.items() if len(feature_set.features) >= 4}


def build_feature_sets(frame: pd.DataFrame) -> dict[str, FeatureSet]:
    base_sets = build_base_feature_sets(frame)
    out = dict(base_sets)
    full = base_sets.get("full_multisignal_compact")
    if full is None:
        return out
    for ablated_group in full.groups:
        groups = {group: features for group, features in full.groups.items() if group != ablated_group}
        candidate = FeatureSet(f"full_without_{ablated_group}", groups)
        if len(candidate.features) >= 4:
            out[candidate.name] = candidate
    return out


def feature_set_inventory(feature_sets: dict[str, FeatureSet]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, feature_set in feature_sets.items():
        for group, features in feature_set.groups.items():
            rows.append(
                {
                    "feature_set": name,
                    "group": group,
                    "feature_count": len(features),
                    "features": ",".join(features),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature_set", "group"]).reset_index(drop=True)


def candidate_id_for_spec(spec: LocalResidualSpec) -> str:
    source = "same_source" if spec.same_source else "all_prior"
    phase = "phase" if spec.phase_conditioned else "all_phase"
    return slug(f"multi_signal_local_{spec.feature_set}_k{spec.k_neighbors}_{source}_{phase}")


def build_specs(feature_sets: dict[str, FeatureSet]) -> list[LocalResidualSpec]:
    specs: list[LocalResidualSpec] = []
    for name, feature_set in feature_sets.items():
        for same_source in (False, True):
            for phase_conditioned in (False, True):
                specs.append(
                    LocalResidualSpec(
                        feature_set=name,
                        features=feature_set.features,
                        k_neighbors=80,
                        same_source=same_source,
                        phase_conditioned=phase_conditioned,
                    )
                )
    return specs


def past_only_local_predictions(frame: pd.DataFrame, spec: LocalResidualSpec) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    forecasts = pd.to_numeric(ordered["forecast_max_c"], errors="coerce").to_numpy(dtype=float)
    targets = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    residuals = targets - forecasts
    feature_matrix = ordered.loc[:, list(spec.features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    sources = ordered["forecast_source_family"].astype(str).to_numpy()
    phases = ordered["monsoon_phase"].astype(str).to_numpy()

    predictions: list[float] = []
    corrections: list[float] = []
    rows_used: list[int] = []
    mean_distances: list[float] = []
    for index, target_date in enumerate(dates):
        current = feature_matrix[index]
        if not np.isfinite(forecasts[index]) or not np.isfinite(current).all():
            predictions.append(float(forecasts[index]) if np.isfinite(forecasts[index]) else math.nan)
            corrections.append(0.0)
            rows_used.append(0)
            mean_distances.append(math.nan)
            continue

        prior_mask = np.arange(len(ordered)) < int(np.searchsorted(dates, target_date, side="left"))
        if spec.same_source:
            prior_mask &= sources == sources[index]
        if spec.phase_conditioned:
            prior_mask &= phases == phases[index]
        prior_mask &= np.isfinite(residuals)
        prior_mask &= np.isfinite(feature_matrix).all(axis=1)
        prior_index = np.flatnonzero(prior_mask)
        if len(prior_index) < max(spec.min_history, spec.k_neighbors):
            predictions.append(float(forecasts[index]))
            corrections.append(0.0)
            rows_used.append(0)
            mean_distances.append(math.nan)
            continue

        correction, used, mean_distance = smooth_residual_correction(
            feature_matrix[prior_index],
            residuals[prior_index],
            current,
            k_neighbors=spec.k_neighbors,
            shrinkage=spec.shrinkage,
            correction_clip_c=spec.correction_clip_c,
        )
        predictions.append(float(forecasts[index] + correction))
        corrections.append(correction)
        rows_used.append(used)
        mean_distances.append(mean_distance)

    out = ordered[["target_date", "forecast_source_family", "target_tmax_c", "forecast_max_c"]].copy()
    out["official_raw"] = forecasts
    out["candidate_prediction_c"] = predictions
    out["residual_correction_c"] = corrections
    out["past_rows_used"] = rows_used
    out["mean_neighbor_distance"] = mean_distances
    out["feature_count"] = len(spec.features)
    return out


def score_candidate(predictions: pd.DataFrame, spec: LocalResidualSpec, candidate_id: str) -> dict[str, object]:
    candidate = score_prediction_frame(predictions.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "official_prediction"}), "official_prediction")
    corrected = predictions["past_rows_used"] > 0
    return {
        "candidate_id": candidate_id,
        "feature_set": spec.feature_set,
        "features": ",".join(spec.features),
        "feature_count": len(spec.features),
        "k_neighbors": spec.k_neighbors,
        "same_source": spec.same_source,
        "phase_conditioned": spec.phase_conditioned,
        "min_history": spec.min_history,
        "shrinkage": spec.shrinkage,
        **candidate,
        "official_same_rows_mae": official["mae"],
        "official_same_rows_rmse": official["rmse"],
        "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
        "corrected_rows": int(corrected.sum()),
        "fallback_rows": int((~corrected).sum()),
        "mean_past_rows_used": float(predictions.loc[corrected, "past_rows_used"].mean()) if corrected.any() else 0.0,
        "mean_abs_correction_c": float(predictions.loc[corrected, "residual_correction_c"].abs().mean()) if corrected.any() else 0.0,
        "mean_neighbor_distance": float(predictions.loc[corrected, "mean_neighbor_distance"].mean()) if corrected.any() else math.nan,
    }


def run_local_experts(frame: pd.DataFrame, feature_sets: dict[str, FeatureSet]) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in build_specs(feature_sets):
        predictions = past_only_local_predictions(frame, spec)
        candidate_id = candidate_id_for_spec(spec)
        predictions["candidate_id"] = candidate_id
        score_rows.append(score_candidate(predictions, spec, candidate_id))
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(score_rows)
    if not scoreboard.empty:
        scoreboard = scoreboard.sort_values(["delta_vs_official_same_rows", "mae"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return scoreboard, predictions


def feature_set_summary(scoreboard: pd.DataFrame) -> pd.DataFrame:
    if scoreboard.empty:
        return pd.DataFrame()
    return (
        scoreboard.groupby("feature_set", observed=True)
        .agg(
            candidates=("candidate_id", "count"),
            best_mae=("mae", "min"),
            best_rmse=("rmse", "min"),
            best_delta_vs_official=("delta_vs_official_same_rows", "min"),
            median_delta_vs_official=("delta_vs_official_same_rows", "median"),
            max_corrected_rows=("corrected_rows", "max"),
            max_feature_count=("feature_count", "max"),
        )
        .reset_index()
        .sort_values("best_delta_vs_official")
    )


def ablation_summary(scoreboard: pd.DataFrame) -> pd.DataFrame:
    if scoreboard.empty:
        return pd.DataFrame()
    summary = feature_set_summary(scoreboard)
    full = summary[summary["feature_set"].eq("full_multisignal_compact")]
    if full.empty:
        return summary
    full_best = float(full.iloc[0]["best_mae"])
    out = summary.copy()
    out["delta_mae_vs_full_best"] = out["best_mae"] - full_best
    return out.sort_values(["delta_mae_vs_full_best", "best_mae"]).reset_index(drop=True)


def build_blend_frame(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    scoreboard: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    official = frame[["target_date", "forecast_source_family", "target_tmax_c", "forecast_max_c"]].copy()
    official["official_raw"] = pd.to_numeric(official["forecast_max_c"], errors="coerce")
    if scoreboard.empty or predictions.empty:
        return official, pd.DataFrame()
    top_ids = scoreboard.head(TOP_BLEND_EXPERTS)["candidate_id"].to_list()
    mapping = scoreboard[scoreboard["candidate_id"].isin(top_ids)][
        [
            "candidate_id",
            "feature_set",
            "features",
            "feature_count",
            "k_neighbors",
            "same_source",
            "phase_conditioned",
            "mae",
            "delta_vs_official_same_rows",
        ]
    ].copy()
    mapping["expert_id"] = [
        f"ms_{rank:02d}_{slug(row.candidate_id, limit=40)}"
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
    return official.merge(wide, on="target_date", how="left").sort_values("target_date").reset_index(drop=True), mapping


def run_blend_screen(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    scoreboard: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blend_frame, mapping = build_blend_frame(frame, predictions, scoreboard)
    if mapping.empty:
        return pd.DataFrame(), pd.DataFrame(), mapping
    experts = ["official_raw", *[column for column in blend_frame.columns if column.startswith("ms_")]]
    score_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for mode in ("best", "inverse_mae"):
        for same_source in (False, True):
            candidate_id = f"multi_signal_blend_{mode}_{'same_source' if same_source else 'all_prior'}"
            blend_predictions = past_only_expert_blend(
                blend_frame,
                experts=experts,
                mode=mode,
                same_source=same_source,
                min_history=MIN_HISTORY,
            )
            blend_predictions["candidate_id"] = candidate_id
            candidate = score_prediction_frame(
                blend_predictions.rename(columns={"expert_prediction_c": "prediction"}),
                "prediction",
            )
            official = score_prediction_frame(
                blend_predictions.rename(columns={"official_raw": "official_prediction"}),
                "official_prediction",
            )
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
            prediction_rows.append(blend_predictions)
    blend_scoreboard = pd.DataFrame(score_rows).sort_values(["delta_vs_official_same_rows", "mae"]).reset_index(drop=True)
    return blend_scoreboard, pd.concat(prediction_rows, ignore_index=True), mapping


def best_prior_screen_rows() -> pd.DataFrame:
    paths = [
        (
            "0018_official_expert_blend",
            RESEARCH_ROOT / "0018_past_only_official_expert_blend_screen" / "artifacts" / "scoreboard.csv",
        ),
        (
            "0026_pressure_gradient_blend",
            RESEARCH_ROOT / "0026_pressure_gradient_experts" / "artifacts" / "blend_scoreboard.csv",
        ),
        (
            "0029_official_residual_blend",
            RESEARCH_ROOT / "0029_official_residual_source_text_range_dynamics" / "artifacts" / "blend_scoreboard.csv",
        ),
        (
            "0029_official_residual_bucket",
            RESEARCH_ROOT / "0029_official_residual_source_text_range_dynamics" / "artifacts" / "bucket_scoreboard.csv",
        ),
    ]
    rows: list[dict[str, object]] = []
    for family, path in paths:
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if table.empty:
            continue
        best = table.sort_values(["mae", "rmse"]).iloc[0]
        rows.append(
            {
                "system": family,
                "candidate_id": str(best["candidate_id"]),
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "delta_vs_official": float(best.get("delta_vs_official_same_rows", math.nan)),
                "n": int(best.get("n", 0)),
                "first_date": str(best.get("first_date", "")),
                "last_date": str(best.get("last_date", "")),
            }
        )
    return pd.DataFrame(rows)


def baseline_comparison(
    frame: pd.DataFrame,
    scoreboard: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
) -> pd.DataFrame:
    official = score_prediction_frame(frame.rename(columns={"official_raw": "prediction"}), "prediction")
    rows: list[dict[str, object]] = [
        {
            "system": "official_raw",
            "candidate_id": "official_raw",
            "mae": official["mae"],
            "rmse": official["rmse"],
            "delta_vs_official": 0.0,
            "n": official["n"],
            "first_date": official["first_date"],
            "last_date": official["last_date"],
        }
    ]
    prior = best_prior_screen_rows()
    if not prior.empty:
        rows.extend(prior.to_dict("records"))
    if not scoreboard.empty:
        best_local = scoreboard.iloc[0]
        rows.append(
            {
                "system": "0030_best_local",
                "candidate_id": str(best_local["candidate_id"]),
                "mae": float(best_local["mae"]),
                "rmse": float(best_local["rmse"]),
                "delta_vs_official": float(best_local["delta_vs_official_same_rows"]),
                "n": int(best_local["n"]),
                "first_date": str(best_local["first_date"]),
                "last_date": str(best_local["last_date"]),
            }
        )
    if not blend_scoreboard.empty:
        best_blend = blend_scoreboard.iloc[0]
        rows.append(
            {
                "system": "0030_best_blend",
                "candidate_id": str(best_blend["candidate_id"]),
                "mae": float(best_blend["mae"]),
                "rmse": float(best_blend["rmse"]),
                "delta_vs_official": float(best_blend["delta_vs_official_same_rows"]),
                "n": int(best_blend["n"]),
                "first_date": str(best_blend["first_date"]),
                "last_date": str(best_blend["last_date"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)


def write_outputs(
    *,
    frame: pd.DataFrame,
    feature_sets: dict[str, FeatureSet],
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
    blend_predictions: pd.DataFrame,
    blend_mapping: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    inventory = feature_set_inventory(feature_sets)
    summary = feature_set_summary(scoreboard)
    ablations = ablation_summary(scoreboard)
    comparison = baseline_comparison(frame, scoreboard, blend_scoreboard)

    write_csv(artifacts / "feature_sets.csv", inventory)
    write_csv(artifacts / "local_scoreboard.csv", scoreboard)
    write_csv(artifacts / "feature_set_summary.csv", summary)
    write_csv(artifacts / "ablation_summary.csv", ablations)
    top_ids = set(scoreboard.head(40)["candidate_id"].to_list()) if not scoreboard.empty else set()
    write_csv(artifacts / "top_local_predictions.csv", predictions[predictions["candidate_id"].isin(top_ids)].copy())
    write_csv(artifacts / "blend_scoreboard.csv", blend_scoreboard)
    write_csv(artifacts / "blend_predictions.csv", blend_predictions)
    write_csv(artifacts / "blend_mapping.csv", blend_mapping)
    write_csv(artifacts / "baseline_comparison.csv", comparison)

    best_local = scoreboard.iloc[0] if not scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "source_counts": {str(k): int(v) for k, v in frame["forecast_source_family"].value_counts().to_dict().items()},
        "feature_sets": int(len(feature_sets)),
        "local_candidates": int(len(scoreboard)),
        "blend_candidates": int(len(blend_scoreboard)),
        "best_local_candidate": "" if best_local is None else str(best_local["candidate_id"]),
        "best_local_feature_set": "" if best_local is None else str(best_local["feature_set"]),
        "best_local_mae": None if best_local is None else float(best_local["mae"]),
        "best_local_rmse": None if best_local is None else float(best_local["rmse"]),
        "best_local_delta_vs_official": None if best_local is None else float(best_local["delta_vs_official_same_rows"]),
        "best_blend_candidate": "" if best_blend is None else str(best_blend["candidate_id"]),
        "best_blend_mae": None if best_blend is None else float(best_blend["mae"]),
        "best_blend_rmse": None if best_blend is None else float(best_blend["rmse"]),
        "best_blend_delta_vs_official": None if best_blend is None else float(best_blend["delta_vs_official_same_rows"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "multi_signal_local_residual_lab_manifest.json", manifest)
    write_readme(
        folder=folder,
        manifest=manifest,
        inventory=inventory,
        summary=summary,
        ablations=ablations,
        scoreboard=scoreboard,
        blend_scoreboard=blend_scoreboard,
        comparison=comparison,
    )
    update_master_index(manifest)
    return manifest


def write_readme(
    *,
    folder: Path,
    manifest: dict[str, object],
    inventory: pd.DataFrame,
    summary: pd.DataFrame,
    ablations: pd.DataFrame,
    scoreboard: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best_local = scoreboard.iloc[0] if not scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    best_local_text = "No multi-signal local expert was scoreable."
    if best_local is not None:
        best_local_text = (
            f"Best local expert: `{best_local['candidate_id']}` with MAE `{best_local['mae']:.4f}`, "
            f"RMSE `{best_local['rmse']:.4f}`, and official delta "
            f"`{best_local['delta_vs_official_same_rows']:.4f}`."
        )
    best_blend_text = "No multi-signal blend was scoreable."
    if best_blend is not None:
        best_blend_text = (
            f"Best local-expert blend: `{best_blend['candidate_id']}` with MAE `{best_blend['mae']:.4f}`, "
            f"RMSE `{best_blend['rmse']:.4f}`, and official delta "
            f"`{best_blend['delta_vs_official_same_rows']:.4f}`."
        )

    readme = f"""# Multi-Signal Local Residual Lab

Generated: `{manifest['generated_at_utc']}`

## Purpose

This insight tests whether a broader, physically mixed residual neighbourhood can extract more information than single-channel pressure, text, or dewpoint buckets. Each candidate starts from the official HKO Tmax forecast max and applies a residual correction estimated from similar historical days.

The feature spaces combine:

- official forecast metadata and source-local forecast dynamics;
- official weather text flags;
- dewpoint, humidity, and moisture-spread signals;
- upper-air heat/moisture/thickness signals;
- station thermal gradients and graph modes;
- wind, marine, and Waglan proxies;
- cloud/rain memory;
- pressure-gradient and pressure-tendency signals.

## Data Window

Rows used: `{manifest['official_rows']}` official forecast/target rows.

Date range: `{manifest['first_target_date']}` to `{manifest['last_target_date']}`.

Source counts: `{manifest['source_counts']}`.

The scored official archive is still non-contiguous: early press archive plus RSS-era rows. The moving partial 2005 acquisition is not promoted here.

## Leakage Contract

- All scored rows are earlier than `{CONFIRMATION_START.date()}`.
- Each target date uses only rows with strictly earlier target dates.
- `searchsorted(..., side="left")` excludes same-date labels across source families.
- Same-source candidates restrict history to the same official forecast source family.
- Phase-conditioned candidates restrict history to the same monsoon phase.
- Neighbour scaling, neighbour selection, and residual averaging are all computed inside the prior-only slice.
- The blend layer estimates expert MAE/weights using only prior realized errors.
- This is a research screen, not a promoted production model.

## Main Results

{best_local_text}

{best_blend_text}

## Baseline Comparison

{markdown_table(comparison, max_rows=20)}

## Feature-Set Summary

{markdown_table(summary, max_rows=20)}

## Ablation Summary

{markdown_table(ablations, max_rows=20)}

## Top Local Experts

{markdown_table(scoreboard.head(20), max_rows=20)}

## Blend Scoreboard

{markdown_table(blend_scoreboard, max_rows=20)}

## Feature Inventory

{markdown_table(inventory.head(30), max_rows=30)}

## Interpretation

This screen is designed to answer whether the next lift is likely to come from combining physical channels into one local residual neighbourhood. If the full multi-signal sets beat their ablations, that argues for compound interactions. If the ablated or narrow sets win, the added feature families are currently diluting distance quality.

The result should be interpreted against the existing `0018` and `0026` screens. A 0030 result that improves raw official but does not beat `0018` means the signal exists but the local-neighbour form is not yet competitive. A result that beats `0018` would justify promotion into a stronger expert stack after full archive refresh and sealed-period confirmation.
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Multi-Signal Local Residual Lab\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_multi_signal_local_residual_lab.py`:

- `{FOLDER_NAME}`: prior-only local residual experts combining official metadata/text, dewpoint/moisture, upper air, station gradients, wind/marine, cloud/rain, and pressure-gradient signals.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Feature sets | {manifest['feature_sets']} |
| Local candidates | {manifest['local_candidates']} |
| Blend candidates | {manifest['blend_candidates']} |
| Best local MAE | {manifest['best_local_mae']} |
| Best local RMSE | {manifest['best_local_rmse']} |
| Best local delta vs official | {manifest['best_local_delta_vs_official']} |
| Best blend MAE | {manifest['best_blend_mae']} |
| Best blend RMSE | {manifest['best_blend_rmse']} |
| Best blend delta vs official | {manifest['best_blend_delta_vs_official']} |

Leakage contract: all scored rows are before `{CONFIRMATION_START.date()}`; local neighbours, scaling, residual correction, and blend weights use strictly prior target dates.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    frame = build_multisignal_frame()
    require_no_confirmation_dates(frame["target_date"], context="multi-signal local residual lab")
    feature_sets = build_feature_sets(frame)
    scoreboard, predictions = run_local_experts(frame, feature_sets)
    blend_scoreboard, blend_predictions, blend_mapping = run_blend_screen(frame, predictions, scoreboard)
    return write_outputs(
        frame=frame,
        feature_sets=feature_sets,
        scoreboard=scoreboard,
        predictions=predictions,
        blend_scoreboard=blend_scoreboard,
        blend_predictions=blend_predictions,
        blend_mapping=blend_mapping,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 multi-signal local residual lab.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
