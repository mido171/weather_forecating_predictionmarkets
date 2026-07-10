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
from scripts.run_hkg_t24_pressure_regime_interaction_atlas import (  # noqa: E402
    build_analysis_frame,
)

FOLDER_NAME = "0028_smooth_gated_pressure_experts"
MIN_HISTORY = 160
TOP_BLEND_EXPERTS = 14


@dataclass(frozen=True)
class SmoothExpertSpec:
    name: str
    features: tuple[str, ...]
    k_neighbors: int
    same_source: bool
    phase_conditioned: bool
    shrinkage: float = 100.0
    correction_clip_c: float = 2.5
    min_history: int = MIN_HISTORY


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 140) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def available_features(frame: pd.DataFrame, candidates: tuple[str, ...], *, min_non_null: int = 700) -> tuple[str, ...]:
    features: list[str] = []
    for feature in candidates:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if int(values.notna().sum()) >= min_non_null and values.nunique(dropna=True) > 2:
            features.append(feature)
    return tuple(features)


def build_feature_sets(frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    sets = {
        "pressure_change_calendar_upper": available_features(
            frame,
            (
                "isd_pressure_mean_hpa_change_1d",
                "isd_pressure_tendency_morning_midday_hpa",
                "month",
                "monsoon_phase_code",
                "ua_mse_1000_850_mean_kj_kg",
                "ua_theta_e_1000_850_mean_k",
                "igra_thickness_1000_500_m_change_48h",
            ),
        ),
        "pressure_wind_advection": available_features(
            frame,
            (
                "isd_pressure_plane_lat_slope_hpa_per_deg",
                "isd_pressure_plane_lon_slope_hpa_per_deg",
                "pressure_plane_slope_magnitude_hpa_per_deg",
                "isd_pressure_mean_hpa_change_1d",
                "slp_590960_minus_596730_hpa",
                "slp_590870_minus_596730_hpa",
                "ua_wind_v_1000hpa_mps",
                "ua_wind_u_1000hpa_mps",
                "isd_onshore_easterly_proxy_mps",
                "isd_northerly_proxy_mps",
                "isd_wind_speed_mean_mps",
            ),
        ),
        "pressure_humidity_cap": available_features(
            frame,
            (
                "slp_590960_minus_596730_hpa",
                "slp_590870_minus_596730_hpa",
                "isd_pressure_mean_hpa_change_1d",
                "rh_min_pct",
                "rh_max_pct",
                "isd_temp_dewpoint_spread_mean_c",
                "isd_dewpoint_midday_minus_temp_c",
                "daily_hong_kong_observatory_mean_relative_humidity_lag7_roll7",
                "dew_590960_minus_596730_c",
                "dew_590870_minus_592780_c",
            ),
        ),
        "pressure_station_graph": available_features(
            frame,
            (
                "isd_pressure_mean_hpa_roll7_mean",
                "isd_pressure_mean_hpa_change_1d",
                "slp_590960_minus_596730_hpa",
                "isd_graph_laplacian_mode_1",
                "isd_graph_laplacian_mode_3",
                "isd_graph_total_variation_c2",
                "isd_north_south_temp_gradient_c",
                "isd_east_west_temp_gradient_c",
                "thermal_590960_minus_596730_c",
                "thermal_590870_minus_596730_c",
                "thermal_592930_minus_596730_c",
            ),
        ),
        "smooth_pressure_full_context": available_features(
            frame,
            (
                "isd_pressure_mean_hpa_change_1d",
                "isd_pressure_tendency_morning_midday_hpa",
                "isd_pressure_plane_lat_slope_hpa_per_deg",
                "isd_pressure_plane_lon_slope_hpa_per_deg",
                "pressure_plane_slope_magnitude_hpa_per_deg",
                "slp_590960_minus_596730_hpa",
                "ua_mse_1000_850_mean_kj_kg",
                "ua_theta_e_1000_850_mean_k",
                "ua_wind_v_1000hpa_mps",
                "isd_temp_dewpoint_spread_mean_c",
                "isd_dewpoint_midday_minus_temp_c",
                "isd_graph_laplacian_mode_3",
                "isd_north_south_temp_gradient_c",
                "month",
                "monsoon_phase_code",
            ),
        ),
    }
    return {name: features for name, features in sets.items() if len(features) >= 3}


def smooth_residual_correction(
    x_prior: np.ndarray,
    residual_prior: np.ndarray,
    x_current: np.ndarray,
    *,
    k_neighbors: int,
    shrinkage: float,
    correction_clip_c: float,
) -> tuple[float, int, float]:
    if x_prior.ndim != 2 or len(x_prior) != len(residual_prior):
        raise ValueError("x_prior and residual_prior shapes are incompatible")
    if len(x_prior) == 0:
        return 0.0, 0, math.nan
    means = np.nanmean(x_prior, axis=0)
    stds = np.nanstd(x_prior, axis=0)
    stds = np.where((stds <= 1e-9) | ~np.isfinite(stds), 1.0, stds)
    prior_scaled = (x_prior - means) / stds
    current_scaled = (x_current - means) / stds
    distances = np.sqrt(np.nanmean(np.square(prior_scaled - current_scaled), axis=1))
    valid = np.isfinite(distances) & np.isfinite(residual_prior)
    if not valid.any():
        return 0.0, 0, math.nan
    valid_distances = distances[valid]
    valid_residuals = residual_prior[valid]
    k = min(int(k_neighbors), len(valid_distances))
    order = np.argpartition(valid_distances, k - 1)[:k]
    selected_distances = valid_distances[order]
    selected_residuals = valid_residuals[order]
    positive_distances = selected_distances[selected_distances > 1e-9]
    scale = float(np.nanmedian(positive_distances)) if len(positive_distances) else math.nan
    if not np.isfinite(scale) or scale <= 1e-9:
        weights = np.ones(len(selected_distances), dtype=float)
    else:
        weights = np.exp(-selected_distances / scale)
    raw = float(np.average(selected_residuals, weights=weights))
    weight = len(selected_residuals) / (len(selected_residuals) + float(shrinkage))
    correction = float(np.clip(raw * weight, -correction_clip_c, correction_clip_c))
    return correction, int(len(selected_residuals)), float(np.mean(selected_distances))


def candidate_id_for_spec(spec: SmoothExpertSpec) -> str:
    source = "same_source" if spec.same_source else "all_prior"
    phase = "phase" if spec.phase_conditioned else "all_phase"
    return slug(f"smooth_pressure_{spec.name}_k{spec.k_neighbors}_{source}_{phase}")


def build_specs(frame: pd.DataFrame) -> list[SmoothExpertSpec]:
    specs: list[SmoothExpertSpec] = []
    for name, features in build_feature_sets(frame).items():
        for k_neighbors in (40, 80, 160):
            for same_source in (False, True):
                for phase_conditioned in (False, True):
                    specs.append(
                        SmoothExpertSpec(
                            name=name,
                            features=features,
                            k_neighbors=k_neighbors,
                            same_source=same_source,
                            phase_conditioned=phase_conditioned,
                        )
                    )
    return specs


def past_only_smooth_predictions(frame: pd.DataFrame, spec: SmoothExpertSpec) -> pd.DataFrame:
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


def score_candidate(predictions: pd.DataFrame, spec: SmoothExpertSpec, candidate_id: str) -> dict[str, object]:
    candidate = score_prediction_frame(predictions.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "official_prediction"}), "official_prediction")
    corrected = predictions["past_rows_used"] > 0
    return {
        "candidate_id": candidate_id,
        "feature_set": spec.name,
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


def run_smooth_experts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in build_specs(frame):
        predictions = past_only_smooth_predictions(frame, spec)
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
        return scoreboard
    return (
        scoreboard.groupby("feature_set", observed=True)
        .agg(
            candidates=("candidate_id", "count"),
            best_mae=("mae", "min"),
            best_delta_vs_official=("delta_vs_official_same_rows", "min"),
            median_delta_vs_official=("delta_vs_official_same_rows", "median"),
            max_corrected_rows=("corrected_rows", "max"),
            max_feature_count=("feature_count", "max"),
        )
        .reset_index()
        .sort_values("best_delta_vs_official")
    )


def build_blend_frame(frame: pd.DataFrame, predictions: pd.DataFrame, scoreboard: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        f"sp_{rank:02d}_{slug(row.candidate_id, limit=42)}"
        for rank, row in enumerate(mapping.itertuples(index=False), start=1)
    ]
    long = predictions[predictions["candidate_id"].isin(top_ids)][["target_date", "candidate_id", "candidate_prediction_c"]].copy()
    long = long.merge(mapping[["candidate_id", "expert_id"]], on="candidate_id", how="inner")
    wide = (
        long.pivot_table(index="target_date", columns="expert_id", values="candidate_prediction_c", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return official.merge(wide, on="target_date", how="left").sort_values("target_date").reset_index(drop=True), mapping


def run_blend_screen(frame: pd.DataFrame, predictions: pd.DataFrame, scoreboard: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blend_frame, mapping = build_blend_frame(frame, predictions, scoreboard)
    if mapping.empty:
        return pd.DataFrame(), pd.DataFrame(), mapping
    experts = ["official_raw", *[column for column in blend_frame.columns if column.startswith("sp_")]]
    score_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for mode in ("best", "inverse_mae"):
        for same_source in (False, True):
            candidate_id = f"smooth_pressure_blend_{mode}_{'same_source' if same_source else 'all_prior'}"
            blend_predictions = past_only_expert_blend(
                blend_frame,
                experts=experts,
                mode=mode,
                same_source=same_source,
                min_history=MIN_HISTORY,
            )
            blend_predictions["candidate_id"] = candidate_id
            candidate = score_prediction_frame(blend_predictions.rename(columns={"expert_prediction_c": "prediction"}), "prediction")
            official = score_prediction_frame(blend_predictions.rename(columns={"official_raw": "official_prediction"}), "official_prediction")
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


def write_outputs(
    *,
    frame: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
    blend_predictions: pd.DataFrame,
    blend_mapping: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    write_csv(artifacts / "smooth_scoreboard.csv", scoreboard)
    write_csv(artifacts / "feature_set_summary.csv", feature_set_summary(scoreboard))
    top_ids = set(scoreboard.head(40)["candidate_id"].to_list()) if not scoreboard.empty else set()
    write_csv(artifacts / "top_smooth_predictions.csv", predictions[predictions["candidate_id"].isin(top_ids)].copy())
    write_csv(artifacts / "blend_scoreboard.csv", blend_scoreboard)
    write_csv(artifacts / "blend_predictions.csv", blend_predictions)
    write_csv(artifacts / "blend_mapping.csv", blend_mapping)

    best_single = scoreboard.iloc[0] if not scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "feature_sets": int(len(build_feature_sets(frame))),
        "smooth_candidates": int(len(scoreboard)),
        "blend_candidates": int(len(blend_scoreboard)),
        "best_single_candidate": "" if best_single is None else str(best_single["candidate_id"]),
        "best_single_mae": None if best_single is None else float(best_single["mae"]),
        "best_single_rmse": None if best_single is None else float(best_single["rmse"]),
        "best_single_delta_vs_official": None if best_single is None else float(best_single["delta_vs_official_same_rows"]),
        "best_blend_candidate": "" if best_blend is None else str(best_blend["candidate_id"]),
        "best_blend_mae": None if best_blend is None else float(best_blend["mae"]),
        "best_blend_rmse": None if best_blend is None else float(best_blend["rmse"]),
        "best_blend_delta_vs_official": None if best_blend is None else float(best_blend["delta_vs_official_same_rows"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "smooth_gated_pressure_experts_manifest.json", manifest)

    best_single_text = "No smooth pressure expert was scoreable."
    if best_single is not None:
        best_single_text = (
            f"Best smooth expert: `{best_single['candidate_id']}` with MAE `{best_single['mae']:.4f}`, "
            f"RMSE `{best_single['rmse']:.4f}`, and official delta "
            f"`{best_single['delta_vs_official_same_rows']:.4f}`."
        )
    best_blend_text = "No smooth blend was scoreable."
    if best_blend is not None:
        best_blend_text = (
            f"Best smooth blend: `{best_blend['candidate_id']}` with MAE `{best_blend['mae']:.4f}`, "
            f"RMSE `{best_blend['rmse']:.4f}`, and official delta "
            f"`{best_blend['delta_vs_official_same_rows']:.4f}`."
        )

    readme = f"""# Smooth Gated Pressure Experts

Generated: `{manifest['generated_at_utc']}`

## What Was Tested

This insight follows `0027_pressure_regime_interaction_atlas`. Instead of hard pressure-regime buckets, it tests smooth local residual experts around the official forecast. Each expert finds the nearest prior historical official-forecast days in a physically selected feature space, then applies a shrunk weighted residual correction.

The feature spaces combine pressure change/slopes/spreads with month and monsoon phase, upper-air MSE/theta-e, 1000 hPa wind, humidity/dew spread, and station-network thermal/graph structure.

## Leakage Contract

- All rows are earlier than `{CONFIRMATION_START.date()}`.
- For each target date, scaling means/stds are computed only from strictly prior target dates.
- Nearest neighbors are selected only from strictly prior target dates.
- Residual corrections are shrunk and clipped before being added to the official forecast.
- Same-source candidates restrict history to the same official source family.
- Phase-conditioned candidates restrict history to the same monsoon phase.
- Blend selection/weights use only prior realized expert errors.

## Main Results

{best_single_text}

{best_blend_text}

## Feature-Set Summary

{markdown_table(feature_set_summary(scoreboard), max_rows=20)}

## Top Smooth Experts

{markdown_table(scoreboard.head(20), max_rows=20)}

## Blend Scoreboard

{markdown_table(blend_scoreboard, max_rows=20)}

## Interpretation

This is a stricter test of whether pressure-regime interactions can become deployable once we avoid sparse hard cells. A positive result would justify promoting smooth gated pressure experts into the official-anchor stack. A weak result means the interaction signal is real but not yet strong enough with the current non-contiguous forecast archive and simple local-residual form.
"""
    write_text(folder / "README.md", readme)
    update_master_index(manifest)
    return manifest


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Smooth Gated Pressure Experts\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_smooth_gated_pressure_experts.py`:

- `{FOLDER_NAME}`: prior-only nearest-neighbor residual experts in pressure/regime feature spaces.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Feature sets | {manifest['feature_sets']} |
| Smooth candidates | {manifest['smooth_candidates']} |
| Blend candidates | {manifest['blend_candidates']} |
| Best single MAE | {manifest['best_single_mae']} |
| Best single RMSE | {manifest['best_single_rmse']} |
| Best single delta vs official | {manifest['best_single_delta_vs_official']} |
| Best blend MAE | {manifest['best_blend_mae']} |
| Best blend RMSE | {manifest['best_blend_rmse']} |
| Best blend delta vs official | {manifest['best_blend_delta_vs_official']} |

Leakage contract: all rows are before `{CONFIRMATION_START.date()}`; scaling, nearest-neighbor selection, corrections, and blend weights use strictly prior target dates.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    frame = build_analysis_frame()
    require_no_confirmation_dates(frame["target_date"], context="smooth gated pressure frame")
    scoreboard, predictions = run_smooth_experts(frame)
    blend_scoreboard, blend_predictions, blend_mapping = run_blend_screen(frame, predictions, scoreboard)
    return write_outputs(
        frame=frame,
        scoreboard=scoreboard,
        predictions=predictions,
        blend_scoreboard=blend_scoreboard,
        blend_predictions=blend_predictions,
        blend_mapping=blend_mapping,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 smooth gated pressure experts.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
