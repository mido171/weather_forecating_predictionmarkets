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

FOLDER_NAME = "0020_regime_experts"
MIN_HISTORY = 120
MIN_MATCH_ROWS = 20
TOP_BLEND_EXPERTS = 12


@dataclass(frozen=True)
class RegimeExpertSpec:
    family: str
    name: str
    features: tuple[str, ...]
    bins: int
    season_conditioned: bool
    same_source: bool
    statistic: str
    shrinkage: float = 0.0
    min_history: int = MIN_HISTORY
    min_match_rows: int = MIN_MATCH_ROWS


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 90) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def add_composite_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    composites = {
        "forecast_minus_waglan_sea_temp_roll7_c": ("forecast_max_c", "daily_waglan_island_sea_temperature_lag7_roll7"),
        "forecast_minus_waglan_sea_temp_lag7_c": ("forecast_max_c", "daily_waglan_island_sea_temperature_lag7"),
        "forecast_minus_np_sea_temp_pm_roll7_c": ("forecast_max_c", "daily_north_point_sea_temperature_pm_lag7_roll7"),
        "spread_air_592870_minus_596730_c": (
            "isd_station_air_temperature_c_592870_99999",
            "isd_station_air_temperature_c_596730_99999",
        ),
        "spread_air_450110_minus_592870_c": (
            "isd_station_air_temperature_c_450110_99999",
            "isd_station_air_temperature_c_592870_99999",
        ),
        "spread_pressure_590960_minus_592780_hpa": (
            "isd_station_sea_level_pressure_hpa_590960_99999",
            "isd_station_sea_level_pressure_hpa_592780_99999",
        ),
        "spread_dew_450070_minus_590960_c": (
            "isd_station_dew_point_c_450070_99999",
            "isd_station_dew_point_c_590960_99999",
        ),
    }
    for target, (left, right) in composites.items():
        if left in out.columns and right in out.columns:
            out[target] = pd.to_numeric(out[left], errors="coerce") - pd.to_numeric(out[right], errors="coerce")
    return out


def available_features(frame: pd.DataFrame, candidates: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(feature for feature in candidates if feature in frame.columns)


def build_regime_specs(frame: pd.DataFrame) -> list[RegimeExpertSpec]:
    base_specs: list[tuple[str, str, tuple[str, ...]]] = [
        (
            "marine_cap",
            "waglan_sea_cap",
            (
                "daily_waglan_island_sea_temperature_lag7_roll7",
                "forecast_minus_waglan_sea_temp_roll7_c",
                "daily_waglan_island_mean_wind_speed_lag7_roll7",
            ),
        ),
        (
            "marine_cap",
            "north_point_sea_cap",
            (
                "daily_north_point_sea_temperature_pm_lag7_roll7",
                "forecast_minus_np_sea_temp_pm_roll7_c",
                "daily_waglan_island_prevailing_wind_direction_lag7_roll7",
            ),
        ),
        (
            "upper_heat",
            "surface_mse_thetae",
            ("ua_mse_1000hpa_kj_kg", "ua_theta_e_1000hpa_k", "ua_dewpoint_1000hpa_c"),
        ),
        (
            "upper_heat",
            "lower_layer_heat",
            ("ua_mse_1000_850_mean_kj_kg", "ua_theta_e_1000_850_mean_k", "ua_layer_1000_850_temp_mean_c"),
        ),
        (
            "upper_heat",
            "ridge_profile",
            ("ua_ridge_strength_raw_proxy", "ua_temperature_profile_linear_slope_c_per_hpa", "ua_dewpoint_1000hpa_c"),
        ),
        (
            "station_spread",
            "air_spread_592870_596730",
            ("spread_air_592870_minus_596730_c", "isd_graph_total_variation_c2"),
        ),
        (
            "station_spread",
            "mixed_spread_pressure_dew",
            ("spread_pressure_590960_minus_592780_hpa", "spread_dew_450070_minus_590960_c"),
        ),
        (
            "station_spread",
            "air_spread_urban_remote",
            ("spread_air_450110_minus_592870_c", "isd_north_south_temp_gradient_c"),
        ),
        (
            "transition_surface",
            "morning_rise_moisture",
            ("isd_morning_to_midday_temp_rise_c", "isd_temp_dewpoint_spread_mean_c", "isd_dewpoint_midday_minus_temp_c"),
        ),
        (
            "transition_surface",
            "official_range_surface",
            ("forecast_range_c", "forecast_midpoint_c", "isd_morning_to_midday_temp_rise_c"),
        ),
        (
            "transition_surface",
            "memory_volatility_surface",
            ("target_roll60_mean_lag7_c", "target_roll14_std_lag7_c", "isd_morning_to_midday_temp_rise_c"),
        ),
    ]

    specs: list[RegimeExpertSpec] = []
    for family, name, candidates in base_specs:
        features = available_features(frame, candidates)
        if not features:
            continue
        for bins in (3, 5):
            for season_conditioned in (False, True):
                for same_source in (False, True):
                    for statistic, shrinkage in (("mean", 0.0), ("shrunk_mean", 60.0)):
                        specs.append(
                            RegimeExpertSpec(
                                family=family,
                                name=name,
                                features=features,
                                bins=bins,
                                season_conditioned=season_conditioned,
                                same_source=same_source,
                                statistic=statistic,
                                shrinkage=shrinkage,
                            )
                        )
    return specs


def quantile_bucket_index(
    prior_values: np.ndarray,
    current_value: float,
    bins: int,
    *,
    min_match_rows: int,
) -> tuple[np.ndarray, int] | None:
    if len(prior_values) < bins * min_match_rows or len(np.unique(prior_values[~np.isnan(prior_values)])) < bins:
        return None
    edges = np.unique(np.nanquantile(prior_values, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    if len(edges) < bins - 1:
        return None
    return np.searchsorted(edges, prior_values, side="right"), int(np.searchsorted(edges, current_value, side="right"))


def matched_prior_indices(
    *,
    frame: pd.DataFrame,
    index: int,
    spec: RegimeExpertSpec,
    feature_values: dict[str, np.ndarray],
    dates: np.ndarray,
) -> tuple[np.ndarray, int]:
    target_date = dates[index]
    prior_mask = np.arange(len(frame)) < int(np.searchsorted(dates, target_date, side="left"))
    if spec.same_source:
        prior_mask &= frame["forecast_source_family"].eq(frame.at[index, "forecast_source_family"]).to_numpy()
    if spec.season_conditioned and "season" in frame.columns:
        prior_mask &= frame["season"].eq(frame.at[index, "season"]).to_numpy()

    used_features = 0
    for feature_count in range(len(spec.features), 0, -1):
        mask = prior_mask.copy()
        valid_current = True
        for feature in spec.features[:feature_count]:
            current_value = feature_values[feature][index]
            if not np.isfinite(current_value):
                valid_current = False
                break
            mask &= np.isfinite(feature_values[feature])
        if not valid_current:
            continue
        prior_index = np.flatnonzero(mask)
        if len(prior_index) < spec.min_history:
            continue

        bucket_mask = np.ones(len(prior_index), dtype=bool)
        for feature in spec.features[:feature_count]:
            prior_values = feature_values[feature][prior_index]
            bucketed = quantile_bucket_index(
                prior_values,
                feature_values[feature][index],
                spec.bins,
                min_match_rows=spec.min_match_rows,
            )
            if bucketed is None:
                bucket_mask &= False
                break
            prior_buckets, current_bucket = bucketed
            bucket_mask &= prior_buckets == current_bucket
        matched = prior_index[bucket_mask]
        if len(matched) >= spec.min_match_rows:
            used_features = feature_count
            return matched, used_features
    return np.array([], dtype=int), used_features


def correction_from_residuals(residuals: np.ndarray, matched: np.ndarray, spec: RegimeExpertSpec) -> float:
    selected = residuals[matched]
    selected = selected[np.isfinite(selected)]
    if len(selected) < spec.min_match_rows:
        return 0.0
    if spec.statistic == "mean":
        return float(np.mean(selected))
    if spec.statistic == "shrunk_mean":
        raw = float(np.mean(selected))
        weight = len(selected) / (len(selected) + spec.shrinkage)
        return float(raw * weight)
    if spec.statistic == "median":
        return float(np.median(selected))
    raise ValueError(f"Unknown statistic: {spec.statistic}")


def past_only_regime_expert_prediction(frame: pd.DataFrame, spec: RegimeExpertSpec) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    forecasts = pd.to_numeric(ordered["forecast_max_c"], errors="coerce").to_numpy(dtype=float)
    targets = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    residuals = targets - forecasts
    feature_values = {
        feature: pd.to_numeric(ordered[feature], errors="coerce").to_numpy(dtype=float)
        for feature in spec.features
    }

    predictions: list[float] = []
    corrections: list[float] = []
    rows_used: list[int] = []
    feature_dims_used: list[int] = []
    for index, forecast in enumerate(forecasts):
        if not np.isfinite(forecast):
            predictions.append(math.nan)
            corrections.append(0.0)
            rows_used.append(0)
            feature_dims_used.append(0)
            continue
        matched, used_features = matched_prior_indices(
            frame=ordered,
            index=index,
            spec=spec,
            feature_values=feature_values,
            dates=dates,
        )
        correction = correction_from_residuals(residuals, matched, spec) if len(matched) else 0.0
        predictions.append(float(forecast + correction))
        corrections.append(correction)
        rows_used.append(int(len(matched)))
        feature_dims_used.append(int(used_features))

    out = ordered[["target_date", "forecast_source_family", "target_tmax_c", "forecast_max_c"]].copy()
    out["candidate_prediction_c"] = predictions
    out["past_correction_c"] = corrections
    out["past_rows_used"] = rows_used
    out["feature_dims_used"] = feature_dims_used
    out["expert_family"] = spec.family
    out["expert_name"] = spec.name
    out["features"] = ",".join(spec.features)
    out["bins"] = spec.bins
    out["season_conditioned"] = spec.season_conditioned
    out["same_source"] = spec.same_source
    out["statistic"] = spec.statistic
    return out


def candidate_id_for_spec(spec: RegimeExpertSpec) -> str:
    return (
        f"{slug(spec.family)}_{slug(spec.name)}_q{spec.bins}"
        f"_season{int(spec.season_conditioned)}_source{int(spec.same_source)}_{slug(spec.statistic)}"
    )


def score_candidate(predictions: pd.DataFrame, *, candidate_id: str, spec: RegimeExpertSpec) -> dict[str, object]:
    candidate = score_prediction_frame(predictions, "candidate_prediction_c")
    official = score_prediction_frame(predictions, "forecast_max_c")
    return {
        "candidate_id": candidate_id,
        "expert_family": spec.family,
        "expert_name": spec.name,
        "features": ",".join(spec.features),
        "bins": spec.bins,
        "season_conditioned": spec.season_conditioned,
        "same_source": spec.same_source,
        "statistic": spec.statistic,
        **candidate,
        "official_same_rows_mae": official["mae"],
        "official_same_rows_rmse": official["rmse"],
        "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
        "corrected_rows": int((predictions["past_rows_used"] > 0).sum()),
        "fallback_rows": int((predictions["past_rows_used"] == 0).sum()),
        "mean_match_rows": float(predictions.loc[predictions["past_rows_used"] > 0, "past_rows_used"].mean())
        if (predictions["past_rows_used"] > 0).any()
        else 0.0,
    }


def run_regime_experts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = build_regime_specs(frame)
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs:
        predictions = past_only_regime_expert_prediction(frame, spec)
        candidate_id = candidate_id_for_spec(spec)
        predictions["candidate_id"] = candidate_id
        score_rows.append(score_candidate(predictions, candidate_id=candidate_id, spec=spec))
        prediction_frames.append(predictions)

    scoreboard = pd.DataFrame(score_rows).sort_values(["delta_vs_official_same_rows", "mae"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return scoreboard, predictions


def build_blend_frame(frame: pd.DataFrame, predictions: pd.DataFrame, scoreboard: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    official = frame[["target_date", "forecast_source_family", "target_tmax_c", "forecast_max_c"]].copy()
    official["official_raw"] = pd.to_numeric(official["forecast_max_c"], errors="coerce")
    top_ids = scoreboard.head(TOP_BLEND_EXPERTS)["candidate_id"].to_list()
    mapping = scoreboard[scoreboard["candidate_id"].isin(top_ids)][
        ["candidate_id", "expert_family", "expert_name", "features", "mae", "delta_vs_official_same_rows"]
    ].copy()
    mapping["expert_id"] = [
        f"regime_{rank:02d}_{slug(candidate_id, limit=52)}"
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
    experts = ["official_raw", *[col for col in blend_frame.columns if col.startswith("regime_")]]
    score_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for mode in ("best", "inverse_mae"):
        for same_source in (False, True):
            candidate_id = f"regime_blend_{mode}_{'same_source' if same_source else 'all_prior'}"
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


def family_summary(scoreboard: pd.DataFrame) -> pd.DataFrame:
    if scoreboard.empty:
        return scoreboard
    return (
        scoreboard.groupby("expert_family", observed=True)
        .agg(
            candidates=("candidate_id", "count"),
            best_mae=("mae", "min"),
            best_delta_vs_official=("delta_vs_official_same_rows", "min"),
            median_delta_vs_official=("delta_vs_official_same_rows", "median"),
            max_corrected_rows=("corrected_rows", "max"),
        )
        .reset_index()
        .sort_values("best_delta_vs_official")
    )


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

    top_candidate_ids = set(scoreboard.head(30)["candidate_id"].to_list())
    write_csv(artifacts / "regime_scoreboard.csv", scoreboard)
    write_csv(artifacts / "family_summary.csv", family_summary(scoreboard))
    write_csv(artifacts / "top_regime_predictions.csv", predictions[predictions["candidate_id"].isin(top_candidate_ids)].copy())
    write_csv(artifacts / "blend_scoreboard.csv", blend_scoreboard)
    write_csv(artifacts / "blend_predictions.csv", blend_predictions)
    write_csv(artifacts / "blend_mapping.csv", blend_mapping)

    best_regime = scoreboard.iloc[0] if not scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "regime_candidates": int(len(scoreboard)),
        "blend_candidates": int(len(blend_scoreboard)),
        "best_regime_candidate": "" if best_regime is None else str(best_regime["candidate_id"]),
        "best_regime_mae": None if best_regime is None else float(best_regime["mae"]),
        "best_regime_delta_vs_official": None if best_regime is None else float(best_regime["delta_vs_official_same_rows"]),
        "best_blend_candidate": "" if best_blend is None else str(best_blend["candidate_id"]),
        "best_blend_mae": None if best_blend is None else float(best_blend["mae"]),
        "best_blend_delta_vs_official": None if best_blend is None else float(best_blend["delta_vs_official_same_rows"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "regime_expert_factory_manifest.json", manifest)

    best_regime_text = "No regime expert was scoreable."
    if best_regime is not None:
        best_regime_text = (
            f"Best single specialist: `{best_regime['candidate_id']}` with MAE `{best_regime['mae']:.4f}` "
            f"versus same-row official MAE `{best_regime['official_same_rows_mae']:.4f}` "
            f"(delta `{best_regime['delta_vs_official_same_rows']:.4f}`)."
        )
    best_blend_text = "No regime blend was scoreable."
    if best_blend is not None:
        best_blend_text = (
            f"Best regime blend: `{best_blend['candidate_id']}` with MAE `{best_blend['mae']:.4f}` "
            f"versus same-row official MAE `{best_blend['official_same_rows_mae']:.4f}` "
            f"(delta `{best_blend['delta_vs_official_same_rows']:.4f}`)."
        )

    readme = f"""# Regime Expert Factory

Generated: `{manifest['generated_at_utc']}`

## What Was Tested

This insight converts the previous diagnostic signals into explicit specialist residual-correction experts around the official forecast anchor. The specialists cover marine cap effects, upper-air heat/moisture, station spatial spreads, and transition-season surface behavior.

## Leakage Contract

- All rows are earlier than `{CONFIRMATION_START.date()}`.
- Official forecasts are selected only when issued before the T-1 15:00 HKT cutoff.
- Each specialist estimates its residual correction from target dates strictly earlier than the current target date.
- Dynamic quantile bins are computed from prior rows only.
- Same-date rows are excluded from correction history.
- The blend layer uses only prior realized expert error to select or weight experts.
- This is a research screen, not a production forecast model.

## Main Result

{best_regime_text}

{best_blend_text}

## Family Summary

{markdown_table(family_summary(scoreboard), max_rows=20)}

## Top Single Specialists

{markdown_table(scoreboard.head(20), max_rows=20)}

## Blend Scoreboard

{markdown_table(blend_scoreboard, max_rows=20)}

## Blend Mapping

{markdown_table(blend_mapping, max_rows=20)}

## Interpretation

The specialist factory tests whether the strongest discovered mechanisms are useful when converted into past-only correction experts. A material gain here would justify building richer specialists; a modest gain means the immediate bottleneck remains continuous official forecast history and more precise regime definitions.
"""
    write_text(folder / "README.md", readme)
    update_master_index(manifest)
    return manifest


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Regime Expert Factory\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_regime_expert_factory.py`:

- `{FOLDER_NAME}`: leakage-safe specialist residual experts for marine, upper-air, station-spread, and transition-surface regimes, plus prior-performance expert blending.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Regime candidates | {manifest['regime_candidates']} |
| Blend candidates | {manifest['blend_candidates']} |
| Best regime MAE | {manifest['best_regime_mae']} |
| Best regime delta vs official | {manifest['best_regime_delta_vs_official']} |
| Best blend MAE | {manifest['best_blend_mae']} |
| Best blend delta vs official | {manifest['best_blend_delta_vs_official']} |

Leakage contract: all rows are before `{CONFIRMATION_START.date()}`; specialist corrections and blend weights use strictly prior target dates.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    frame = add_composite_features(build_official_feature_frame())
    require_no_confirmation_dates(frame["target_date"], context="regime expert official frame")
    scoreboard, predictions = run_regime_experts(frame)
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
    return argparse.ArgumentParser(description="Run HKG T24 regime expert factory.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
