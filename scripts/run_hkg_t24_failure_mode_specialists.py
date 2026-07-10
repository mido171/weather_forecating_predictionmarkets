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
from scripts.run_hkg_t24_regime_expert_factory import add_composite_features  # noqa: E402

FOLDER_NAME = "0022_failure_specialists"
MIN_HISTORY = 120
MIN_MATCH_ROWS = 20
TOP_BLEND_EXPERTS = 16


@dataclass(frozen=True)
class FeatureCondition:
    feature: str
    side: str
    quantile: float


@dataclass(frozen=True)
class FailureSpecialistSpec:
    family: str
    name: str
    conditions: tuple[FeatureCondition, ...]
    months: tuple[int, ...] = ()
    same_source: bool = False
    statistic: str = "shrunk_mean"
    shrinkage: float = 60.0
    min_history: int = MIN_HISTORY
    min_match_rows: int = MIN_MATCH_ROWS


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 90) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def condition_label(condition: FeatureCondition) -> str:
    return f"{condition.feature}:{condition.side}:q{condition.quantile:.2f}"


def candidate_id_for_spec(spec: FailureSpecialistSpec) -> str:
    months = "all_months" if not spec.months else "m" + "_".join(str(month) for month in spec.months)
    source = "same_source" if spec.same_source else "all_prior"
    conditions = "__".join(condition_label(condition) for condition in spec.conditions)
    return slug(f"{spec.family}_{spec.name}_{months}_{source}_{spec.statistic}_{conditions}", limit=150)


def side_mask(values: np.ndarray, threshold: float, side: str) -> np.ndarray:
    if side == "high":
        return values >= threshold
    if side == "low":
        return values <= threshold
    raise ValueError(f"Unsupported condition side: {side}")


def fold_local_threshold(values: np.ndarray, prior_mask: np.ndarray, condition: FeatureCondition, min_history: int) -> float:
    prior_values = values[prior_mask]
    prior_values = prior_values[np.isfinite(prior_values)]
    if len(prior_values) < min_history or len(np.unique(prior_values)) < 2:
        return math.nan
    return float(np.nanquantile(prior_values, condition.quantile))


def build_failure_specialist_specs(frame: pd.DataFrame) -> list[FailureSpecialistSpec]:
    def fc(feature: str, side: str, quantile: float) -> FeatureCondition:
        return FeatureCondition(feature=feature, side=side, quantile=quantile)

    recipes: list[tuple[str, str, tuple[FeatureCondition, ...], tuple[int, ...]]] = [
        ("humidity_failure", "high_min_rh", (fc("rh_min_pct", "high", 0.80),), ()),
        ("humidity_failure", "high_max_rh", (fc("rh_max_pct", "high", 0.80),), ()),
        (
            "humidity_failure",
            "humid_surface_low_spread",
            (fc("isd_temp_dewpoint_spread_mean_c", "low", 0.20),),
            (),
        ),
        (
            "humidity_failure",
            "dewpoint_midday_excess",
            (fc("isd_dewpoint_midday_minus_temp_c", "high", 0.80),),
            (),
        ),
        (
            "inversion_humidity_failure",
            "inversion_high_rh",
            (fc("igra_inversion_925_minus_1000_c", "high", 0.67), fc("rh_max_pct", "high", 0.67)),
            (),
        ),
        (
            "spring_transition_failure",
            "spring_high_station_spread",
            (fc("spread_air_592870_minus_596730_c", "high", 0.80),),
            (3, 4, 5),
        ),
        (
            "spring_transition_failure",
            "spring_low_station_spread",
            (fc("spread_air_592870_minus_596730_c", "low", 0.20),),
            (3, 4, 5),
        ),
        (
            "spring_transition_failure",
            "spring_humid_surface",
            (fc("isd_temp_dewpoint_spread_mean_c", "low", 0.20), fc("rh_max_pct", "high", 0.67)),
            (3, 4, 5),
        ),
        (
            "spring_transition_failure",
            "spring_cool_memory",
            (fc("target_roll120_mean_lag7_c", "low", 0.20),),
            (3, 4, 5),
        ),
        (
            "cloud_rain_suppression",
            "recent_cloud_high_rh",
            (
                fc("daily_hong_kong_observatory_mean_cloud_amount_lag7_roll7", "high", 0.80),
                fc("rh_max_pct", "high", 0.67),
            ),
            (),
        ),
        (
            "cloud_rain_suppression",
            "recent_rain_high_rh",
            (
                fc("daily_hong_kong_observatory_daily_rainfall_lag7_roll7", "high", 0.80),
                fc("rh_max_pct", "high", 0.67),
            ),
            (),
        ),
        (
            "marine_cap_failure",
            "forecast_far_above_waglan_sea",
            (fc("forecast_minus_waglan_sea_temp_roll7_c", "high", 0.80),),
            (),
        ),
        (
            "marine_cap_failure",
            "cool_waglan_sea_memory",
            (fc("daily_waglan_island_sea_temperature_lag7_roll7", "low", 0.20),),
            (),
        ),
        (
            "upper_heat_failure",
            "upper_mse_hot",
            (fc("ua_mse_1000hpa_kj_kg", "high", 0.80),),
            (),
        ),
        (
            "upper_heat_failure",
            "thetae_hot_humid",
            (fc("ua_theta_e_1000hpa_k", "high", 0.80), fc("ua_dewpoint_1000hpa_c", "high", 0.67)),
            (),
        ),
        (
            "official_uncertainty_failure",
            "wide_forecast_range",
            (fc("forecast_range_c", "high", 0.80),),
            (),
        ),
        (
            "memory_volatility_failure",
            "high_recent_volatility",
            (fc("target_roll14_std_lag7_c", "high", 0.80),),
            (),
        ),
        (
            "memory_volatility_failure",
            "spectral_change_high_humidity",
            (fc("spectral_abs_change_energy_14_lag7_c", "high", 0.80), fc("rh_max_pct", "high", 0.67)),
            (),
        ),
        (
            "morning_warmup_failure",
            "strong_morning_warmup",
            (fc("isd_morning_to_midday_temp_rise_c", "high", 0.80),),
            (),
        ),
    ]

    specs: list[FailureSpecialistSpec] = []
    for family, name, conditions, months in recipes:
        if any(condition.feature not in frame.columns for condition in conditions):
            continue
        for same_source in (False, True):
            for statistic, shrinkage in (("shrunk_mean", 60.0), ("median", 0.0)):
                specs.append(
                    FailureSpecialistSpec(
                        family=family,
                        name=name,
                        conditions=conditions,
                        months=months,
                        same_source=same_source,
                        statistic=statistic,
                        shrinkage=shrinkage,
                    )
                )
    return specs


def matched_prior_rows(
    *,
    frame: pd.DataFrame,
    index: int,
    spec: FailureSpecialistSpec,
    feature_values: dict[str, np.ndarray],
    dates: np.ndarray,
    months: np.ndarray,
) -> tuple[np.ndarray, bool, list[float]]:
    target_date = dates[index]
    prior_mask = np.arange(len(frame)) < int(np.searchsorted(dates, target_date, side="left"))
    if spec.same_source:
        prior_mask &= frame["forecast_source_family"].eq(frame.at[index, "forecast_source_family"]).to_numpy()
    if spec.months:
        if int(months[index]) not in spec.months:
            return np.array([], dtype=int), False, []
        prior_mask &= np.isin(months, np.array(spec.months, dtype=int))
    if int(prior_mask.sum()) < spec.min_history:
        return np.array([], dtype=int), False, []

    match_mask = prior_mask.copy()
    thresholds: list[float] = []
    for condition in spec.conditions:
        values = feature_values[condition.feature]
        current_value = values[index]
        if not np.isfinite(current_value):
            return np.array([], dtype=int), False, thresholds
        threshold = fold_local_threshold(values, prior_mask, condition, spec.min_history)
        if not np.isfinite(threshold) or not bool(side_mask(np.array([current_value]), threshold, condition.side)[0]):
            return np.array([], dtype=int), False, thresholds
        thresholds.append(threshold)
        match_mask &= np.isfinite(values) & side_mask(values, threshold, condition.side)

    matched = np.flatnonzero(match_mask)
    return matched, True, thresholds


def correction_from_residuals(residuals: np.ndarray, matched: np.ndarray, spec: FailureSpecialistSpec) -> float:
    selected = residuals[matched]
    selected = selected[np.isfinite(selected)]
    if len(selected) < spec.min_match_rows:
        return 0.0
    if spec.statistic == "median":
        return float(np.median(selected))
    if spec.statistic == "shrunk_mean":
        raw = float(np.mean(selected))
        weight = len(selected) / (len(selected) + spec.shrinkage)
        return float(raw * weight)
    raise ValueError(f"Unsupported statistic: {spec.statistic}")


def past_only_failure_specialist_prediction(frame: pd.DataFrame, spec: FailureSpecialistSpec) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    months = pd.to_numeric(ordered["month"], errors="coerce").fillna(0).to_numpy(dtype=int)
    forecasts = pd.to_numeric(ordered["forecast_max_c"], errors="coerce").to_numpy(dtype=float)
    targets = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    residuals = targets - forecasts
    feature_values = {
        condition.feature: pd.to_numeric(ordered[condition.feature], errors="coerce").to_numpy(dtype=float)
        for condition in spec.conditions
    }

    predictions: list[float] = []
    corrections: list[float] = []
    past_rows_used: list[int] = []
    triggered: list[bool] = []
    threshold_text: list[str] = []
    for index in range(len(ordered)):
        matched, is_triggered, thresholds = matched_prior_rows(
            frame=ordered,
            index=index,
            spec=spec,
            feature_values=feature_values,
            dates=dates,
            months=months,
        )
        triggered.append(is_triggered)
        threshold_text.append(";".join(f"{value:.6g}" for value in thresholds))
        if is_triggered and len(matched) >= spec.min_match_rows:
            correction = correction_from_residuals(residuals, matched, spec)
            predictions.append(float(forecasts[index] + correction))
            corrections.append(correction)
            past_rows_used.append(int(len(matched)))
        else:
            predictions.append(float(forecasts[index]) if np.isfinite(forecasts[index]) else math.nan)
            corrections.append(0.0)
            past_rows_used.append(0)

    out = ordered[["target_date", "forecast_source_family", "target_tmax_c", "forecast_max_c"]].copy()
    out["candidate_prediction_c"] = predictions
    out["official_raw"] = forecasts
    out["residual_correction_c"] = corrections
    out["past_rows_used"] = past_rows_used
    out["triggered"] = triggered
    out["fold_local_thresholds"] = threshold_text
    return out


def score_candidate(
    predictions: pd.DataFrame,
    *,
    candidate_id: str,
    spec: FailureSpecialistSpec,
) -> dict[str, object]:
    candidate = score_prediction_frame(predictions.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "official_prediction"}), "official_prediction")
    corrected = predictions["past_rows_used"] > 0
    return {
        "candidate_id": candidate_id,
        "expert_family": spec.family,
        "expert_name": spec.name,
        "conditions": " | ".join(condition_label(condition) for condition in spec.conditions),
        "months": "all" if not spec.months else ",".join(str(month) for month in spec.months),
        "same_source": spec.same_source,
        "statistic": spec.statistic,
        "shrinkage": spec.shrinkage,
        **candidate,
        "official_same_rows_mae": official["mae"],
        "official_same_rows_rmse": official["rmse"],
        "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
        "triggered_rows": int(predictions["triggered"].sum()),
        "corrected_rows": int(corrected.sum()),
        "fallback_rows": int((~corrected).sum()),
        "mean_match_rows": float(predictions.loc[corrected, "past_rows_used"].mean()) if corrected.any() else 0.0,
        "mean_abs_correction_c": float(predictions.loc[corrected, "residual_correction_c"].abs().mean())
        if corrected.any()
        else 0.0,
    }


def run_failure_specialists(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = build_failure_specialist_specs(frame)
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs:
        predictions = past_only_failure_specialist_prediction(frame, spec)
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
        [
            "candidate_id",
            "expert_family",
            "expert_name",
            "conditions",
            "months",
            "same_source",
            "statistic",
            "mae",
            "delta_vs_official_same_rows",
            "triggered_rows",
            "corrected_rows",
        ]
    ].copy()
    mapping["expert_id"] = [
        f"failure_{rank:02d}_{slug(candidate_id, limit=48)}"
        for rank, candidate_id in enumerate(mapping["candidate_id"], start=1)
    ]
    long = predictions[predictions["candidate_id"].isin(top_ids)][["target_date", "candidate_id", "candidate_prediction_c"]].copy()
    long = long.merge(mapping[["candidate_id", "expert_id"]], on="candidate_id", how="inner")
    wide = (
        long.pivot_table(index="target_date", columns="expert_id", values="candidate_prediction_c", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    blend_frame = official.merge(wide, on="target_date", how="left").sort_values("target_date").reset_index(drop=True)
    return blend_frame, mapping


def run_blend_screen(frame: pd.DataFrame, predictions: pd.DataFrame, scoreboard: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blend_frame, mapping = build_blend_frame(frame, predictions, scoreboard)
    experts = ["official_raw", *[column for column in blend_frame.columns if column.startswith("failure_")]]
    score_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for mode in ("best", "inverse_mae"):
        for same_source in (False, True):
            candidate_id = f"failure_blend_{mode}_{'same_source' if same_source else 'all_prior'}"
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
            max_triggered_rows=("triggered_rows", "max"),
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

    top_candidate_ids = set(scoreboard.head(40)["candidate_id"].to_list())
    write_csv(artifacts / "specialist_scoreboard.csv", scoreboard)
    write_csv(artifacts / "family_summary.csv", family_summary(scoreboard))
    write_csv(artifacts / "top_specialist_predictions.csv", predictions[predictions["candidate_id"].isin(top_candidate_ids)].copy())
    write_csv(artifacts / "blend_scoreboard.csv", blend_scoreboard)
    write_csv(artifacts / "blend_predictions.csv", blend_predictions)
    write_csv(artifacts / "blend_mapping.csv", blend_mapping)

    best_specialist = scoreboard.iloc[0] if not scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "specialist_candidates": int(len(scoreboard)),
        "blend_candidates": int(len(blend_scoreboard)),
        "best_specialist_candidate": "" if best_specialist is None else str(best_specialist["candidate_id"]),
        "best_specialist_mae": None if best_specialist is None else float(best_specialist["mae"]),
        "best_specialist_delta_vs_official": None
        if best_specialist is None
        else float(best_specialist["delta_vs_official_same_rows"]),
        "best_specialist_triggered_rows": None if best_specialist is None else int(best_specialist["triggered_rows"]),
        "best_specialist_corrected_rows": None if best_specialist is None else int(best_specialist["corrected_rows"]),
        "best_blend_candidate": "" if best_blend is None else str(best_blend["candidate_id"]),
        "best_blend_mae": None if best_blend is None else float(best_blend["mae"]),
        "best_blend_delta_vs_official": None if best_blend is None else float(best_blend["delta_vs_official_same_rows"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "failure_mode_specialists_manifest.json", manifest)

    best_specialist_text = "No specialist was scoreable."
    if best_specialist is not None:
        best_specialist_text = (
            f"Best single failure specialist: `{best_specialist['candidate_id']}` with MAE "
            f"`{best_specialist['mae']:.4f}` versus same-row official MAE "
            f"`{best_specialist['official_same_rows_mae']:.4f}` "
            f"(delta `{best_specialist['delta_vs_official_same_rows']:.4f}`). It triggered on "
            f"`{int(best_specialist['triggered_rows'])}` rows and applied corrections on "
            f"`{int(best_specialist['corrected_rows'])}` rows."
        )
    best_blend_text = "No blend was scoreable."
    if best_blend is not None:
        best_blend_text = (
            f"Best failure-specialist blend: `{best_blend['candidate_id']}` with MAE "
            f"`{best_blend['mae']:.4f}` versus same-row official MAE "
            f"`{best_blend['official_same_rows_mae']:.4f}` "
            f"(delta `{best_blend['delta_vs_official_same_rows']:.4f}`)."
        )

    readme = f"""# Fold-Local Failure-Mode Specialists

Generated: `{manifest['generated_at_utc']}`

## What Was Tested

This insight converts the diagnostic weak zones from `0021_failure_modes` into deployable-style official-forecast correction specialists. It targets high humidity, inversion plus humidity, spring transition station spreads, recent cloud/rain suppression, marine-cap gaps, upper-air heat/moisture, official forecast uncertainty, recent target volatility, and strong morning warmup regimes.

Each specialist starts from the official Tmax forecast and only changes it when the current row falls into a failure-mode trigger. The correction is the past residual of similar triggered rows: `actual_tmax - official_forecast_max`.

## Leakage Contract

- All rows are earlier than `{CONFIRMATION_START.date()}`.
- The confirmation period from `{CONFIRMATION_START.date()}` onward is not touched.
- For every target date, each trigger threshold is recomputed from strictly earlier target dates only.
- Same-date rows are excluded by construction through `searchsorted(..., side="left")`.
- If a specialist is marked `same_source`, its history is restricted to the same official-forecast source family.
- Current target labels are never used for trigger selection, thresholding, correction magnitude, or blend weights.
- Blend selection/weighting uses only prior realized expert error.

## Main Result

{best_specialist_text}

{best_blend_text}

## Family Summary

{markdown_table(family_summary(scoreboard), max_rows=20)}

## Top Single Specialists

{markdown_table(scoreboard.head(25), max_rows=25)}

## Blend Scoreboard

{markdown_table(blend_scoreboard, max_rows=20)}

## Blend Mapping

{markdown_table(blend_mapping, max_rows=20)}

## Interpretation

This screen answers whether the sharp diagnostic failure modes from `0021` can become leakage-safe, deployable-style correction rules. The important distinction is that `0021` used full pre-2024 thresholds only for diagnosis; this folder recomputes thresholds fold-locally before every prediction. Any gain here is therefore stronger evidence than the diagnostic bucket lifts, but it still runs on the current non-contiguous official archive, so it is not the final system.
"""
    write_text(folder / "README.md", readme)
    update_master_index(manifest)
    return manifest


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Fold-Local Failure-Mode Specialists\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_failure_mode_specialists.py`:

- `{FOLDER_NAME}`: deployable-style failure-mode specialists using fold-local trigger thresholds and strictly prior residual corrections.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Specialist candidates | {manifest['specialist_candidates']} |
| Blend candidates | {manifest['blend_candidates']} |
| Best specialist MAE | {manifest['best_specialist_mae']} |
| Best specialist delta vs official | {manifest['best_specialist_delta_vs_official']} |
| Best specialist triggered rows | {manifest['best_specialist_triggered_rows']} |
| Best specialist corrected rows | {manifest['best_specialist_corrected_rows']} |
| Best blend MAE | {manifest['best_blend_mae']} |
| Best blend delta vs official | {manifest['best_blend_delta_vs_official']} |

Leakage contract: all rows are before `{CONFIRMATION_START.date()}`; thresholds, residual corrections, and blend weights are estimated only from strictly earlier target dates.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    frame = add_composite_features(build_official_feature_frame())
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(frame["target_date"], context="failure specialist official frame")
    scoreboard, predictions = run_failure_specialists(frame)
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
    return argparse.ArgumentParser(description="Run HKG T24 fold-local failure-mode specialists.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
