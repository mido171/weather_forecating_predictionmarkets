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
from scripts.run_hkg_t24_multistation_attribute_information_gain import (  # noqa: E402
    build_official_feature_frame,
)
from scripts.run_hkg_t24_official_anchor_expert_blend_screen import (  # noqa: E402
    past_only_expert_blend,
)

FOLDER_NAME = "0026_pressure_gradient_experts"
MIN_HISTORY = 120
TOP_BLEND_EXPERTS = 10

PRESSURE_PAIR_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "slp_590960_minus_596730_hpa",
        "isd_station_sea_level_pressure_hpa_590960_99999",
        "isd_station_sea_level_pressure_hpa_596730_99999",
    ),
    (
        "slp_590870_minus_596730_hpa",
        "isd_station_sea_level_pressure_hpa_590870_99999",
        "isd_station_sea_level_pressure_hpa_596730_99999",
    ),
    (
        "slp_590960_minus_595010_hpa",
        "isd_station_sea_level_pressure_hpa_590960_99999",
        "isd_station_sea_level_pressure_hpa_595010_99999",
    ),
    (
        "slp_592780_minus_596730_hpa",
        "isd_station_sea_level_pressure_hpa_592780_99999",
        "isd_station_sea_level_pressure_hpa_596730_99999",
    ),
    (
        "slp_592930_minus_595010_hpa",
        "isd_station_sea_level_pressure_hpa_592930_99999",
        "isd_station_sea_level_pressure_hpa_595010_99999",
    ),
    (
        "slp_590870_minus_595010_hpa",
        "isd_station_sea_level_pressure_hpa_590870_99999",
        "isd_station_sea_level_pressure_hpa_595010_99999",
    ),
)


@dataclass(frozen=True)
class RidgeExpertSpec:
    family: str
    name: str
    features: tuple[str, ...]
    alpha: float
    same_source: bool
    min_history: int = MIN_HISTORY
    shrinkage: float = 80.0
    correction_clip_c: float = 2.5


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 130) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def add_pressure_gradient_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for feature, left, right in PRESSURE_PAIR_SPECS:
        if left in out.columns and right in out.columns:
            out[feature] = pd.to_numeric(out[left], errors="coerce") - pd.to_numeric(out[right], errors="coerce")
    if {"isd_pressure_plane_lat_slope_hpa_per_deg", "isd_pressure_plane_lon_slope_hpa_per_deg"}.issubset(out.columns):
        lat = pd.to_numeric(out["isd_pressure_plane_lat_slope_hpa_per_deg"], errors="coerce")
        lon = pd.to_numeric(out["isd_pressure_plane_lon_slope_hpa_per_deg"], errors="coerce")
        out["pressure_plane_slope_magnitude_hpa_per_deg"] = np.sqrt(np.square(lat) + np.square(lon))
    return out


def available_features(frame: pd.DataFrame, candidates: tuple[str, ...], *, min_non_null: int = 500) -> tuple[str, ...]:
    features: list[str] = []
    for feature in candidates:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if int(values.notna().sum()) >= min_non_null and values.nunique(dropna=True) > 2:
            features.append(feature)
    return tuple(features)


def build_feature_sets(frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    pressure_plane = available_features(
        frame,
        (
            "isd_pressure_plane_lat_slope_hpa_per_deg",
            "isd_pressure_plane_lon_slope_hpa_per_deg",
            "pressure_plane_slope_magnitude_hpa_per_deg",
            "isd_pressure_tendency_morning_midday_hpa",
            "isd_pressure_mean_hpa_change_1d",
            "isd_pressure_mean_hpa_roll7_mean",
            "isd_pressure_range_hpa",
        ),
    )
    pressure_spreads = available_features(frame, tuple(feature for feature, _, _ in PRESSURE_PAIR_SPECS))
    station_pressure_context = available_features(
        frame,
        (
            "isd_pressure_mean_hpa",
            "isd_pressure_min_hpa",
            "isd_pressure_max_hpa",
            "isd_pressure_range_hpa",
            "isd_morning_pressure_mean_hpa",
            "isd_midday_pressure_mean_hpa",
            "isd_pressure_tendency_morning_midday_hpa",
        ),
    )
    advection_context = available_features(
        frame,
        (
            "isd_wind_speed_max_mps",
            "isd_wind_speed_mean_mps",
            "isd_wind_speed_mean_mps_change_1d",
            "isd_dew_point_mean_c_change_1d",
            "isd_air_temp_mean_c_change_1d",
            "ua_wind_v_1000hpa_mps",
            "ua_wind_u_1000hpa_mps",
        ),
    )
    upper_tendency = available_features(
        frame,
        (
            "ua_tendency_48h_ua_theta_1000hpa_k",
            "ua_tendency_48h_igra_temp_1000hpa_c",
            "ua_tendency_48h_igra_hgt_1000hpa_m",
            "igra_thickness_1000_500_m_change_48h",
            "ua_ridge_strength_change_24h",
        ),
    )
    sets = {
        "pressure_plane": pressure_plane,
        "station_pressure_spreads": pressure_spreads,
        "pressure_plane_spreads": tuple(dict.fromkeys((*pressure_plane, *pressure_spreads))),
        "pressure_advection": tuple(dict.fromkeys((*pressure_plane, *pressure_spreads, *advection_context))),
        "pressure_upper_tendency": tuple(dict.fromkeys((*pressure_plane, *pressure_spreads, *upper_tendency))),
        "full_pressure_gradient_context": tuple(
            dict.fromkeys((*pressure_plane, *pressure_spreads, *station_pressure_context, *advection_context, *upper_tendency))
        ),
    }
    return {name: features for name, features in sets.items() if len(features) >= 2}


def candidate_id_for_spec(spec: RidgeExpertSpec) -> str:
    source = "same_source" if spec.same_source else "all_prior"
    return slug(f"{spec.family}_{spec.name}_{source}_alpha{spec.alpha:g}_min{spec.min_history}")


def fit_ridge_residual(
    x_prior: np.ndarray,
    y_prior: np.ndarray,
    x_current: np.ndarray,
    *,
    alpha: float,
    shrinkage: float,
    correction_clip_c: float,
) -> float:
    if x_prior.ndim != 2 or len(x_prior) != len(y_prior):
        raise ValueError("x_prior and y_prior shapes are incompatible")
    means = np.nanmean(x_prior, axis=0)
    stds = np.nanstd(x_prior, axis=0)
    stds = np.where((stds <= 1e-9) | ~np.isfinite(stds), 1.0, stds)
    x_scaled = (x_prior - means) / stds
    current_scaled = (x_current - means) / stds
    y_mean = float(np.mean(y_prior))
    y_centered = y_prior - y_mean
    penalty = np.eye(x_scaled.shape[1], dtype=float) * float(alpha)
    try:
        coef = np.linalg.solve(x_scaled.T @ x_scaled + penalty, x_scaled.T @ y_centered)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(x_scaled.T @ x_scaled + penalty) @ x_scaled.T @ y_centered
    raw = y_mean + float(current_scaled @ coef)
    weight = len(y_prior) / (len(y_prior) + float(shrinkage))
    correction = raw * weight
    return float(np.clip(correction, -correction_clip_c, correction_clip_c))


def past_only_ridge_predictions(frame: pd.DataFrame, spec: RidgeExpertSpec) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    forecasts = pd.to_numeric(ordered["forecast_max_c"], errors="coerce").to_numpy(dtype=float)
    targets = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    residuals = targets - forecasts
    feature_matrix = ordered.loc[:, list(spec.features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    predictions: list[float] = []
    corrections: list[float] = []
    past_rows_used: list[int] = []
    feature_counts: list[int] = []
    for index in range(len(ordered)):
        current = feature_matrix[index]
        if not np.isfinite(current).all() or not np.isfinite(forecasts[index]):
            predictions.append(float(forecasts[index]) if np.isfinite(forecasts[index]) else math.nan)
            corrections.append(0.0)
            past_rows_used.append(0)
            feature_counts.append(len(spec.features))
            continue
        prior_mask = np.arange(len(ordered)) < int(np.searchsorted(dates, dates[index], side="left"))
        if spec.same_source:
            prior_mask &= ordered["forecast_source_family"].eq(ordered.at[index, "forecast_source_family"]).to_numpy()
        prior_mask &= np.isfinite(residuals)
        prior_mask &= np.isfinite(feature_matrix).all(axis=1)
        prior_index = np.flatnonzero(prior_mask)
        if len(prior_index) < spec.min_history:
            predictions.append(float(forecasts[index]))
            corrections.append(0.0)
            past_rows_used.append(0)
            feature_counts.append(len(spec.features))
            continue
        correction = fit_ridge_residual(
            feature_matrix[prior_index],
            residuals[prior_index],
            current,
            alpha=spec.alpha,
            shrinkage=spec.shrinkage,
            correction_clip_c=spec.correction_clip_c,
        )
        predictions.append(float(forecasts[index] + correction))
        corrections.append(correction)
        past_rows_used.append(int(len(prior_index)))
        feature_counts.append(len(spec.features))

    out = ordered[["target_date", "forecast_source_family", "target_tmax_c", "forecast_max_c"]].copy()
    out["official_raw"] = forecasts
    out["candidate_prediction_c"] = predictions
    out["residual_correction_c"] = corrections
    out["past_rows_used"] = past_rows_used
    out["feature_count"] = feature_counts
    return out


def build_specs(frame: pd.DataFrame) -> list[RidgeExpertSpec]:
    feature_sets = build_feature_sets(frame)
    specs: list[RidgeExpertSpec] = []
    for name, features in feature_sets.items():
        for alpha in (1.0, 10.0, 100.0):
            for same_source in (False, True):
                specs.append(
                    RidgeExpertSpec(
                        family="pressure_gradient",
                        name=name,
                        features=features,
                        alpha=alpha,
                        same_source=same_source,
                    )
                )
    return specs


def score_candidate(predictions: pd.DataFrame, spec: RidgeExpertSpec, candidate_id: str) -> dict[str, object]:
    candidate = score_prediction_frame(predictions.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "official_prediction"}), "official_prediction")
    corrected = predictions["past_rows_used"] > 0
    return {
        "candidate_id": candidate_id,
        "family": spec.family,
        "feature_set": spec.name,
        "features": ",".join(spec.features),
        "feature_count": len(spec.features),
        "alpha": spec.alpha,
        "same_source": spec.same_source,
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
    }


def run_pressure_experts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = build_specs(frame)
    scores: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs:
        predictions = past_only_ridge_predictions(frame, spec)
        candidate_id = candidate_id_for_spec(spec)
        predictions["candidate_id"] = candidate_id
        scores.append(score_candidate(predictions, spec, candidate_id))
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(scores).sort_values(["delta_vs_official_same_rows", "mae"]).reset_index(drop=True)
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
    top_ids = scoreboard.head(TOP_BLEND_EXPERTS)["candidate_id"].to_list()
    mapping = scoreboard[scoreboard["candidate_id"].isin(top_ids)][
        ["candidate_id", "feature_set", "features", "feature_count", "alpha", "same_source", "mae", "delta_vs_official_same_rows"]
    ].copy()
    mapping["expert_id"] = [
        f"pressure_{rank:02d}_{slug(candidate_id, limit=46)}"
        for rank, candidate_id in enumerate(mapping["candidate_id"], start=1)
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
    experts = ["official_raw", *[column for column in blend_frame.columns if column.startswith("pressure_")]]
    score_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for mode in ("best", "inverse_mae"):
        for same_source in (False, True):
            candidate_id = f"pressure_blend_{mode}_{'same_source' if same_source else 'all_prior'}"
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

    top_ids = set(scoreboard.head(40)["candidate_id"].to_list())
    write_csv(artifacts / "ridge_scoreboard.csv", scoreboard)
    write_csv(artifacts / "feature_set_summary.csv", feature_set_summary(scoreboard))
    write_csv(artifacts / "top_predictions.csv", predictions[predictions["candidate_id"].isin(top_ids)].copy())
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
        "ridge_candidates": int(len(scoreboard)),
        "blend_candidates": int(len(blend_scoreboard)),
        "best_single_candidate": "" if best_single is None else str(best_single["candidate_id"]),
        "best_single_mae": None if best_single is None else float(best_single["mae"]),
        "best_single_delta_vs_official": None if best_single is None else float(best_single["delta_vs_official_same_rows"]),
        "best_blend_candidate": "" if best_blend is None else str(best_blend["candidate_id"]),
        "best_blend_mae": None if best_blend is None else float(best_blend["mae"]),
        "best_blend_rmse": None if best_blend is None else float(best_blend["rmse"]),
        "best_blend_delta_vs_official": None if best_blend is None else float(best_blend["delta_vs_official_same_rows"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "pressure_gradient_experts_manifest.json", manifest)

    best_single_text = "No pressure-gradient ridge expert was scoreable."
    if best_single is not None:
        best_single_text = (
            f"Best single pressure-gradient expert: `{best_single['candidate_id']}` with MAE "
            f"`{best_single['mae']:.4f}` versus official `{best_single['official_same_rows_mae']:.4f}` "
            f"(delta `{best_single['delta_vs_official_same_rows']:.4f}`)."
        )
    best_blend_text = "No pressure-gradient blend was scoreable."
    if best_blend is not None:
        best_blend_text = (
            f"Best pressure-gradient blend: `{best_blend['candidate_id']}` with MAE "
            f"`{best_blend['mae']:.4f}`, RMSE `{best_blend['rmse']:.4f}`, and official delta "
            f"`{best_blend['delta_vs_official_same_rows']:.4f}`."
        )

    readme = f"""# Pressure-Gradient Official Residual Experts

Generated: `{manifest['generated_at_utc']}`

## What Was Tested

This insight turns the `0025_longhist_signal_atlas` pressure-gradient finding into deployable-style residual experts around the official HKO Tmax forecast. It tests ridge residual corrections using pressure-plane slopes, station sea-level-pressure spreads, pressure tendency, wind/advection context, and upper-air pressure/thermal tendency features.

The target for each expert is the residual `actual_tmax - official_forecast_max`; the final prediction is `official_forecast_max + predicted_residual_correction`.

## Leakage Contract

- All rows are earlier than `{CONFIRMATION_START.date()}`.
- Official forecasts are still selected by the existing pre-cutoff rule.
- For every target date, the ridge correction is fitted only on strictly earlier target dates.
- Same-date rows are excluded by `searchsorted(..., side="left")`.
- `same_source` candidates restrict history to the same official source family.
- Feature scaling, residual mean, coefficients, and blend weights are all recomputed from prior rows only.
- This is a research screen, not a production model.

## Main Results

{best_single_text}

{best_blend_text}

## Feature-Set Summary

{markdown_table(feature_set_summary(scoreboard), max_rows=20)}

## Top Ridge Experts

{markdown_table(scoreboard.head(25), max_rows=25)}

## Blend Scoreboard

{markdown_table(blend_scoreboard, max_rows=20)}

## Blend Mapping

{markdown_table(blend_mapping, max_rows=20)}

## Interpretation

The purpose is to test whether the strong long-history pressure-gradient signal from `0025` becomes useful when applied directly to official forecast residuals. A small or negative result does not invalidate the pressure-gradient signal; it means the current non-contiguous official archive and this simple linear residual form are not yet enough. A positive result would justify adding pressure-gradient specialists to the broader official-anchor expert stack after continuous forecast coverage is available.
"""
    write_text(folder / "README.md", readme)
    update_master_index(manifest)
    return manifest


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Pressure-Gradient Official Residual Experts\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_pressure_gradient_experts.py`:

- `{FOLDER_NAME}`: past-only ridge residual experts using pressure-plane slopes, station pressure spreads, pressure tendency, wind/advection, and upper-air tendency features.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Ridge candidates | {manifest['ridge_candidates']} |
| Blend candidates | {manifest['blend_candidates']} |
| Best single MAE | {manifest['best_single_mae']} |
| Best single delta vs official | {manifest['best_single_delta_vs_official']} |
| Best blend MAE | {manifest['best_blend_mae']} |
| Best blend RMSE | {manifest['best_blend_rmse']} |
| Best blend delta vs official | {manifest['best_blend_delta_vs_official']} |

Leakage contract: all rows are before `{CONFIRMATION_START.date()}`; ridge scaling, coefficients, corrections, and blend weights use strictly prior target dates.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    frame = add_pressure_gradient_features(build_official_feature_frame())
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(frame["target_date"], context="pressure-gradient official frame")
    scoreboard, predictions = run_pressure_experts(frame)
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
    return argparse.ArgumentParser(description="Run HKG T24 pressure-gradient residual experts.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
