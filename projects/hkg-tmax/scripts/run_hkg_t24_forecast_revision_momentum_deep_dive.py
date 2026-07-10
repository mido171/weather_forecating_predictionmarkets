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
)
from scripts.run_hkg_t24_smooth_residual_archetype_specialists import (  # noqa: E402
    current_gate_and_prior_mask,
    finite_numeric_frame,
    smooth_half_life_residual_correction,
)

FOLDER_NAME = "0035_forecast_revision_momentum_deep_dive"
MIN_HISTORY = 160
TOP_BLEND_EXPERTS = 16


@dataclass(frozen=True)
class RevisionMomentumFamily:
    name: str
    conditions: tuple[ArchetypeCondition, ...]
    features: tuple[str, ...]
    source_family: str | None = None


@dataclass(frozen=True)
class RevisionMomentumSpec:
    family_name: str
    anchor_col: str
    conditions: tuple[ArchetypeCondition, ...]
    features: tuple[str, ...]
    k_neighbors: int
    same_source: bool
    half_life_days: float | None
    source_family: str | None = None
    min_history: int = MIN_HISTORY
    min_match_rows: int = 35
    shrinkage: float = 80.0
    correction_clip_c: float = 2.0
    min_local_mae_improvement_c: float = 0.0


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 150) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def candidate_id_for_spec(spec: RevisionMomentumSpec) -> str:
    source = "same_source" if spec.same_source else "all_prior"
    half_life = "hl_none" if spec.half_life_days is None else f"hl_{int(spec.half_life_days)}d"
    source_family = "all_sources" if spec.source_family is None else spec.source_family
    return slug(
        f"revision_momentum_{spec.anchor_col}_{spec.family_name}_{source_family}_k{spec.k_neighbors}_{source}_{half_life}"
    )


def past_only_bias_feature(
    frame: pd.DataFrame,
    *,
    anchor_col: str,
    same_source: bool,
    lookback_rows: int,
    min_history: int,
) -> pd.Series:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True)
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    sources = ordered["forecast_source_family"].astype(str).to_numpy()
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    anchor = pd.to_numeric(ordered[anchor_col], errors="coerce").to_numpy(dtype=float)
    residual = target - anchor
    values: list[float] = []
    for index, target_date in enumerate(dates):
        prior = np.arange(len(ordered)) < int(np.searchsorted(dates, target_date, side="left"))
        if same_source:
            prior &= sources == sources[index]
        prior &= np.isfinite(residual)
        prior_index = np.flatnonzero(prior)
        if len(prior_index) < min_history:
            values.append(math.nan)
            continue
        selected = prior_index[-lookback_rows:]
        values.append(float(np.mean(residual[selected])))
    return pd.Series(values, index=ordered.index)


def add_text_source_change_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    text_flags = [
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
    ]
    for flag in text_flags:
        if flag not in out.columns:
            continue
        values = pd.to_numeric(out[flag], errors="coerce")
        lag = values.groupby(out["forecast_source_family"].astype(str), sort=False).shift(1)
        out[f"{flag}_lag1_source"] = lag
        out[f"{flag}_change_1_source"] = values - lag
        out[f"{flag}_turned_on_source"] = ((values >= 0.5) & (lag < 0.5)).astype(float)
        out[f"{flag}_turned_off_source"] = ((values < 0.5) & (lag >= 0.5)).astype(float)
    return out


def add_revision_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_text_source_change_features(frame)
    for anchor_col in ("official_raw", "prediction_0018_c", "prediction_0026_c"):
        if anchor_col not in out.columns:
            continue
        source_bias = past_only_bias_feature(
            out,
            anchor_col=anchor_col,
            same_source=True,
            lookback_rows=90,
            min_history=20,
        )
        all_bias = past_only_bias_feature(
            out,
            anchor_col=anchor_col,
            same_source=False,
            lookback_rows=180,
            min_history=40,
        )
        out[f"{anchor_col}_prior90_source_residual_mean_c"] = source_bias
        out[f"{anchor_col}_prior180_all_residual_mean_c"] = all_bias
        out[f"{anchor_col}_prior90_source_abs_bias_c"] = source_bias.abs()
        out[f"{anchor_col}_prior180_all_abs_bias_c"] = all_bias.abs()
    if "forecast_max_change_1_source_c" in out.columns:
        out["forecast_max_jump_positive_c"] = pd.to_numeric(out["forecast_max_change_1_source_c"], errors="coerce").clip(
            lower=0.0
        )
    if "forecast_range_change_1_source_c" in out.columns:
        out["forecast_range_widening_c"] = pd.to_numeric(
            out["forecast_range_change_1_source_c"], errors="coerce"
        ).clip(lower=0.0)
    return out


def safe_bin(series: pd.Series, *, bins: int = 4, prefix: str = "q") -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.nunique() < 2:
        return pd.Series(["missing"] * len(series), index=series.index)
    try:
        binned = pd.qcut(values, q=min(bins, valid.nunique()), duplicates="drop")
    except ValueError:
        return pd.Series(["missing"] * len(series), index=series.index)
    codes = binned.cat.codes
    out = pd.Series([f"{prefix}{int(code) + 1}" if code >= 0 else "missing" for code in codes], index=series.index)
    return out


def revision_diagnostic_decomposition(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["target_minus_0018_c"] = pd.to_numeric(work["target_tmax_c"], errors="coerce") - pd.to_numeric(
        work["prediction_0018_c"], errors="coerce"
    )
    work["abs_error_0018_c"] = work["target_minus_0018_c"].abs()
    work["official_error_c"] = pd.to_numeric(work["official_raw"], errors="coerce") - pd.to_numeric(
        work["target_tmax_c"], errors="coerce"
    )
    work["forecast_jump_bin"] = safe_bin(work.get("forecast_max_change_1_source_c", pd.Series(index=work.index)), prefix="jump_q")
    work["forecast_vs_prior_bin"] = safe_bin(
        work.get("forecast_max_vs_prior7_mean_source_c", pd.Series(index=work.index)), prefix="vs7_q"
    )
    work["range_bin"] = safe_bin(work.get("forecast_range_c", pd.Series(index=work.index)), prefix="range_q")
    work["range_change_bin"] = safe_bin(
        work.get("forecast_range_change_1_source_c", pd.Series(index=work.index)), prefix="range_chg_q"
    )
    work["prior_bias_bin"] = safe_bin(
        work.get("prediction_0018_c_prior90_source_residual_mean_c", pd.Series(index=work.index)), prefix="bias_q"
    )
    work["pressure_change_bin"] = safe_bin(
        work.get("isd_pressure_mean_hpa_change_1d", pd.Series(index=work.index)), prefix="pressure_q"
    )
    work["wind_bin"] = safe_bin(work.get("isd_wind_speed_mean_mps", pd.Series(index=work.index)), prefix="wind_q")
    work["station_gradient_bin"] = safe_bin(
        work.get("abs_north_south_temp_gradient_c", pd.Series(index=work.index)), prefix="grad_q"
    )
    for flag in ("text_hot", "text_cloud", "text_any_rain", "text_sunny_or_fine"):
        if flag in work.columns:
            work[f"{flag}_label"] = np.where(pd.to_numeric(work[flag], errors="coerce") >= 0.5, "yes", "no")
    for change in ("text_hot_turned_on_source", "text_cloud_turned_on_source", "text_any_rain_turned_on_source"):
        if change in work.columns:
            work[f"{change}_label"] = np.where(pd.to_numeric(work[change], errors="coerce") >= 0.5, "yes", "no")

    dimensions = [
        ("source", ["forecast_source_family"]),
        ("month", ["month"]),
        ("monsoon_phase", ["monsoon_phase"]),
        ("forecast_jump", ["forecast_jump_bin"]),
        ("forecast_vs_prior", ["forecast_vs_prior_bin"]),
        ("forecast_jump_x_range", ["forecast_jump_bin", "range_bin"]),
        ("forecast_jump_x_range_change", ["forecast_jump_bin", "range_change_bin"]),
        ("forecast_jump_x_prior_bias", ["forecast_jump_bin", "prior_bias_bin"]),
        ("forecast_jump_x_pressure", ["forecast_jump_bin", "pressure_change_bin"]),
        ("forecast_jump_x_wind", ["forecast_jump_bin", "wind_bin"]),
        ("forecast_jump_x_station_gradient", ["forecast_jump_bin", "station_gradient_bin"]),
    ]
    optional_dims = [
        ("forecast_jump_x_hot_text", ["forecast_jump_bin", "text_hot_label"]),
        ("forecast_jump_x_cloud_text", ["forecast_jump_bin", "text_cloud_label"]),
        ("forecast_jump_x_rain_text", ["forecast_jump_bin", "text_any_rain_label"]),
        ("forecast_jump_x_sunny_text", ["forecast_jump_bin", "text_sunny_or_fine_label"]),
        ("forecast_jump_x_hot_turned_on", ["forecast_jump_bin", "text_hot_turned_on_source_label"]),
        ("forecast_jump_x_cloud_turned_on", ["forecast_jump_bin", "text_cloud_turned_on_source_label"]),
        ("forecast_jump_x_rain_turned_on", ["forecast_jump_bin", "text_any_rain_turned_on_source_label"]),
    ]
    dimensions.extend((name, cols) for name, cols in optional_dims if all(col in work.columns for col in cols))

    rows: list[dict[str, object]] = []
    for name, cols in dimensions:
        grouped = work.dropna(subset=["target_minus_0018_c", *cols]).groupby(cols, observed=True, dropna=False)
        for keys, subset in grouped:
            if len(subset) < 25:
                continue
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            rows.append(
                {
                    "dimension": name,
                    "values": " | ".join(str(value) for value in key_tuple),
                    "rows": int(len(subset)),
                    "first_date": str(pd.to_datetime(subset["target_date"]).min().date()),
                    "last_date": str(pd.to_datetime(subset["target_date"]).max().date()),
                    "target_minus_0018_mean_c": float(subset["target_minus_0018_c"].mean()),
                    "target_minus_0018_abs_mean_c": float(subset["target_minus_0018_c"].abs().mean()),
                    "official_error_mean_c": float(subset["official_error_c"].mean()),
                    "forecast_jump_mean_c": float(pd.to_numeric(subset["forecast_max_change_1_source_c"], errors="coerce").mean())
                    if "forecast_max_change_1_source_c" in subset.columns
                    else math.nan,
                    "forecast_vs_prior_mean_c": float(
                        pd.to_numeric(subset["forecast_max_vs_prior7_mean_source_c"], errors="coerce").mean()
                    )
                    if "forecast_max_vs_prior7_mean_source_c" in subset.columns
                    else math.nan,
                }
            )
    decomposition = pd.DataFrame(rows)
    if not decomposition.empty:
        decomposition["abs_residual_lift_vs_global_c"] = decomposition["target_minus_0018_abs_mean_c"] - float(
            work["target_minus_0018_c"].abs().mean()
        )
        decomposition = decomposition.sort_values(
            ["abs_residual_lift_vs_global_c", "target_minus_0018_abs_mean_c"], ascending=[False, False]
        ).reset_index(drop=True)

    feature_rows: list[dict[str, object]] = []
    candidates = [
        "forecast_max_change_1_source_c",
        "forecast_max_vs_prior7_mean_source_c",
        "forecast_max_prior7_std_source_c",
        "forecast_range_c",
        "forecast_range_change_1_source_c",
        "forecast_midpoint_change_1_source_c",
        "issue_to_cutoff_change_1_source_c",
        "prediction_0018_c_prior90_source_residual_mean_c",
        "isd_pressure_mean_hpa_change_1d",
        "pressure_plane_slope_magnitude_hpa_per_deg",
        "isd_wind_speed_mean_mps",
        "isd_onshore_easterly_proxy_mps",
        "abs_north_south_temp_gradient_c",
        "text_hot",
        "text_cloud",
        "text_any_rain",
        "text_hot_turned_on_source",
        "text_cloud_turned_on_source",
        "text_any_rain_turned_on_source",
    ]
    for feature in candidates:
        if feature not in work.columns:
            continue
        pair = pd.concat(
            [pd.to_numeric(work[feature], errors="coerce"), pd.to_numeric(work["target_minus_0018_c"], errors="coerce")],
            axis=1,
        ).dropna()
        if len(pair) < 200 or pair.iloc[:, 0].nunique() < 2:
            continue
        feature_rows.append(
            {
                "feature": feature,
                "rows": int(len(pair)),
                "pearson_corr_to_target_minus_0018": float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson")),
                "spearman_corr_to_target_minus_0018": float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")),
                "feature_mean": float(pair.iloc[:, 0].mean()),
                "feature_std": float(pair.iloc[:, 0].std(ddof=0)),
            }
        )
    correlations = pd.DataFrame(feature_rows)
    if not correlations.empty:
        correlations["abs_spearman"] = correlations["spearman_corr_to_target_minus_0018"].abs()
        correlations = correlations.sort_values("abs_spearman", ascending=False).reset_index(drop=True)
    return decomposition, correlations


def revision_family_templates() -> dict[str, RevisionMomentumFamily]:
    core = (
        "forecast_max_c",
        "forecast_min_c",
        "forecast_range_c",
        "forecast_max_change_1_source_c",
        "forecast_max_prior7_std_source_c",
        "forecast_max_vs_prior7_mean_source_c",
        "forecast_min_change_1_source_c",
        "forecast_range_change_1_source_c",
        "forecast_midpoint_change_1_source_c",
        "issue_to_cutoff_change_1_source_c",
        "prediction_0018_c_prior90_source_residual_mean_c",
        "prediction_0018_c_prior180_all_residual_mean_c",
        "month",
        "monsoon_phase_code",
    )
    pressure_wind = (
        "isd_pressure_mean_hpa_change_1d",
        "pressure_plane_slope_magnitude_hpa_per_deg",
        "isd_wind_speed_mean_mps",
        "isd_wind_speed_max_mps",
        "isd_onshore_easterly_proxy_mps",
    )
    station = (
        "abs_north_south_temp_gradient_c",
        "isd_north_south_temp_gradient_c",
        "isd_east_west_temp_gradient_c",
        "thermal_590870_minus_596730_c",
        "thermal_590960_minus_596730_c",
        "isd_graph_total_variation_c2",
    )
    text = (
        "text_hot",
        "text_cloud",
        "text_any_rain",
        "text_sunny_or_fine",
        "text_hot_turned_on_source",
        "text_cloud_turned_on_source",
        "text_any_rain_turned_on_source",
        "text_keyword_count",
    )
    return {
        "jump_core": RevisionMomentumFamily(
            "jump_core",
            (
                ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.75),
                ArchetypeCondition("forecast_max_vs_prior7_mean_source_c", "high", 0.65),
            ),
            core + pressure_wind,
        ),
        "jump_wide_range": RevisionMomentumFamily(
            "jump_wide_range",
            (
                ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.75),
                ArchetypeCondition("forecast_range_c", "high", 0.70),
            ),
            core + text + pressure_wind,
        ),
        "jump_range_widening": RevisionMomentumFamily(
            "jump_range_widening",
            (
                ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.75),
                ArchetypeCondition("forecast_range_change_1_source_c", "high", 0.70),
            ),
            core + text + pressure_wind,
        ),
        "jump_after_cold_bias": RevisionMomentumFamily(
            "jump_after_cold_bias",
            (
                ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.75),
                ArchetypeCondition("prediction_0018_c_prior90_source_residual_mean_c", "high", 0.65),
            ),
            core + station + pressure_wind,
        ),
        "jump_after_hot_bias": RevisionMomentumFamily(
            "jump_after_hot_bias",
            (
                ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.75),
                ArchetypeCondition("prediction_0018_c_prior90_source_residual_mean_c", "low", 0.35),
            ),
            core + station + pressure_wind,
        ),
        "jump_pressure_drop": RevisionMomentumFamily(
            "jump_pressure_drop",
            (
                ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.75),
                ArchetypeCondition("isd_pressure_mean_hpa_change_1d", "low", 0.25),
            ),
            core + pressure_wind + station,
        ),
        "jump_pressure_windy": RevisionMomentumFamily(
            "jump_pressure_windy",
            (
                ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.75),
                ArchetypeCondition("pressure_plane_slope_magnitude_hpa_per_deg", "high", 0.75),
            ),
            core + pressure_wind + station,
        ),
        "jump_marine_wind": RevisionMomentumFamily(
            "jump_marine_wind",
            (
                ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.75),
                ArchetypeCondition("isd_onshore_easterly_proxy_mps", "high", 0.75),
            ),
            core + pressure_wind + station,
        ),
        "jump_station_gradient": RevisionMomentumFamily(
            "jump_station_gradient",
            (
                ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.75),
                ArchetypeCondition("abs_north_south_temp_gradient_c", "high", 0.75),
            ),
            core + station + pressure_wind,
        ),
        "jump_hot_text": RevisionMomentumFamily(
            "jump_hot_text",
            (
                ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.75),
                ArchetypeCondition("text_hot", "flag"),
            ),
            core + text + pressure_wind,
        ),
        "jump_cloud_conflict": RevisionMomentumFamily(
            "jump_cloud_conflict",
            (
                ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.75),
                ArchetypeCondition("text_cloud", "flag"),
            ),
            core + text + station,
        ),
        "jump_hot_turn_on": RevisionMomentumFamily(
            "jump_hot_turn_on",
            (
                ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.75),
                ArchetypeCondition("text_hot_turned_on_source", "flag"),
            ),
            core + text + pressure_wind,
        ),
    }


def build_revision_specs(frame: pd.DataFrame) -> list[RevisionMomentumSpec]:
    specs: list[RevisionMomentumSpec] = []
    for family in revision_family_templates().values():
        condition_features = tuple(condition.feature for condition in family.conditions)
        available_conditions = available_numeric_features(frame, condition_features, min_non_null=250)
        if set(available_conditions) != set(condition_features):
            continue
        features = available_numeric_features(frame, family.features, min_non_null=250)
        if len(features) < 5:
            continue
        for anchor_col in ("prediction_0018_c", "prediction_0026_c"):
            if anchor_col not in frame.columns:
                continue
            for same_source in (False, True):
                for k_neighbors in (30, 40):
                    for half_life_days in (None, 730.0):
                        specs.append(
                            RevisionMomentumSpec(
                                family_name=family.name,
                                anchor_col=anchor_col,
                                conditions=family.conditions,
                                features=features,
                                k_neighbors=k_neighbors,
                                same_source=same_source,
                                half_life_days=half_life_days,
                                source_family=family.source_family,
                            )
                        )
    return specs


def past_only_revision_predictions(frame: pd.DataFrame, spec: RevisionMomentumSpec) -> pd.DataFrame:
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
        current_source_match = spec.source_family is None or sources[index] == spec.source_family
        if not current_source_match or not np.isfinite(anchor[index]) or not np.isfinite(feature_matrix[index]).all():
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
        if spec.source_family is not None:
            base_prior &= sources == spec.source_family
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
    out["matched_revision_rows"] = matched_rows
    out["neighbor_rows_used"] = neighbor_rows
    out["current_revision_match"] = current_matches
    out["do_no_harm_gate_passed"] = gate_passed
    out["condition_thresholds"] = thresholds
    out["mean_neighbor_distance"] = mean_distances
    out["local_anchor_mae"] = local_anchor_maes
    out["local_corrected_mae"] = local_corrected_maes
    return out


def score_revision_candidate(predictions: pd.DataFrame, spec: RevisionMomentumSpec, candidate_id: str) -> dict[str, object]:
    candidate = score_prediction_frame(predictions.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    anchor = score_prediction_frame(predictions.rename(columns={"anchor_prediction_c": "prediction"}), "prediction")
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "prediction"}), "prediction")
    corrected = predictions["do_no_harm_gate_passed"].astype(bool)
    active = predictions["current_revision_match"].astype(bool)
    return {
        "candidate_id": candidate_id,
        "family_name": spec.family_name,
        "anchor_col": spec.anchor_col,
        "features": ",".join(spec.features),
        "feature_count": len(spec.features),
        "k_neighbors": spec.k_neighbors,
        "same_source": spec.same_source,
        "half_life_days": "" if spec.half_life_days is None else spec.half_life_days,
        "source_family": "" if spec.source_family is None else spec.source_family,
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
        "mean_matched_revision_rows": float(predictions.loc[corrected, "matched_revision_rows"].mean())
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


def run_revision_screen(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    spec_rows: list[dict[str, object]] = []
    for spec in build_revision_specs(frame):
        candidate_id = candidate_id_for_spec(spec)
        predictions = past_only_revision_predictions(frame, spec)
        predictions["candidate_id"] = candidate_id
        predictions["family_name"] = spec.family_name
        predictions["anchor_col"] = spec.anchor_col
        predictions["same_source"] = spec.same_source
        predictions["half_life_days"] = "" if spec.half_life_days is None else spec.half_life_days
        score_rows.append(score_revision_candidate(predictions, spec, candidate_id))
        prediction_frames.append(predictions)
        spec_rows.append(
            {
                "candidate_id": candidate_id,
                "family_name": spec.family_name,
                "anchor_col": spec.anchor_col,
                "same_source": spec.same_source,
                "k_neighbors": spec.k_neighbors,
                "half_life_days": "" if spec.half_life_days is None else spec.half_life_days,
                "source_family": "" if spec.source_family is None else spec.source_family,
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


def family_summary(scoreboard: pd.DataFrame) -> pd.DataFrame:
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
        f"rev_{rank:02d}_{slug(row.candidate_id, limit=44)}"
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
    experts = ["official_raw", *[column for column in blend_frame.columns if column.startswith("rev_")]]
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for mode in ("best", "inverse_mae"):
        for same_source in (False, True):
            candidate_id = f"revision_momentum_blend_{mode}_{'same_source' if same_source else 'all_prior'}"
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


def baseline_comparison(frame: pd.DataFrame, scoreboard: pd.DataFrame, blend_scoreboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prior_paths = [
        (
            "0034_best_cluster_centroid_blend",
            RESEARCH_ROOT / "0034_cluster_centroid_soft_gating" / "artifacts" / "blend_scoreboard.csv",
        ),
        (
            "0033_best_smooth_archetype",
            RESEARCH_ROOT / "0033_smooth_residual_archetype_specialists" / "artifacts" / "smooth_scoreboard.csv",
        ),
        (
            "0032_best_archetype",
            RESEARCH_ROOT / "0032_residual_failure_cluster_discovery" / "artifacts" / "archetype_scoreboard.csv",
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
    if not scoreboard.empty:
        best = scoreboard.iloc[0]
        rows.append(
            {
                "system": "0035_best_revision_specialist",
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
                "system": "0035_best_revision_blend",
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
    decomposition: pd.DataFrame,
    correlations: pd.DataFrame,
    scoreboard: pd.DataFrame,
    summary: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best_single = scoreboard.iloc[0] if not scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    best_single_text = "No revision specialist was scoreable."
    if best_single is not None:
        best_single_text = (
            f"Best revision specialist: `{best_single['candidate_id']}` with MAE `{best_single['mae']:.4f}`, "
            f"RMSE `{best_single['rmse']:.4f}`, official delta "
            f"`{best_single['delta_vs_official_same_rows']:.4f}`, and anchor delta "
            f"`{best_single['delta_vs_anchor_same_rows']:.4f}`."
        )
    best_blend_text = "No revision blend was scoreable."
    if best_blend is not None:
        best_blend_text = (
            f"Best revision blend: `{best_blend['candidate_id']}` with MAE `{best_blend['mae']:.4f}`, "
            f"RMSE `{best_blend['rmse']:.4f}`, and official delta "
            f"`{best_blend['delta_vs_official_same_rows']:.4f}`."
        )
    readme = f"""# Forecast Revision Momentum Deep Dive

Generated: `{manifest['generated_at_utc']}`

## Purpose

`0033` found that forecast-jump/revision momentum is the strongest manually named residual correction family so far. This insight decomposes that signal by source, month, forecast range, prior bias, text changes, pressure/wind, and station-gradient context, then tests granular fold-local specialists built around those revision subfamilies.

## Data Window

Rows used: `{manifest['official_rows']}` official forecast/target rows.

Date range: `{manifest['first_target_date']}` to `{manifest['last_target_date']}`.

Source counts: `{manifest['source_counts']}`.

## Leakage Contract

- All scored rows are earlier than `{CONFIRMATION_START.date()}`.
- Past-bias features use strictly earlier target dates; same-date labels are excluded with `searchsorted(..., side="left")`.
- Text-change features compare current official text flags to previous same-source official text flags only.
- Specialist thresholds, prior revision matches, feature scaling, nearest-neighbor residual correction, half-life weighting, do-no-harm checks, and blend weights are all fold-local.
- 2024+ confirmation labels are not loaded or scored.

## Main Results

{best_single_text}

{best_blend_text}

## Baseline Comparison

{markdown_table(comparison, max_rows=20)}

## Diagnostic Decomposition

{markdown_table(decomposition.head(30), max_rows=30)}

## Revision Feature Correlations

{markdown_table(correlations.head(25), max_rows=25)}

## Family Summary

{markdown_table(summary, max_rows=30)}

## Specialist Scoreboard

{markdown_table(scoreboard.head(30), max_rows=30)}

## Blend Scoreboard

{markdown_table(blend_scoreboard, max_rows=20)}

## Interpretation

This run separates diagnostic decomposition from deployable scoring. The decomposition tables can use realized residuals for post-hoc understanding, but the specialists and blends are strictly prior-only. A new champion means the forecast-jump signal has useful substructure beyond the broad `0033` archetype. If it fails to beat `0034`, the decomposition still identifies which revision subfamilies deserve more targeted modelling after the continuous 2005+ forecast archive is promoted.
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Forecast Revision Momentum Deep Dive\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_forecast_revision_momentum_deep_dive.py`:

- `{FOLDER_NAME}`: forecast-jump/revision decomposition, prior-only bias features, text-change features, granular revision specialists, and revision-specialist blends.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Revision candidates | {manifest['revision_candidates']} |
| Blend candidates | {manifest['blend_candidates']} |
| Best revision MAE | {manifest['best_revision_mae']} |
| Best revision RMSE | {manifest['best_revision_rmse']} |
| Best revision delta vs official | {manifest['best_revision_delta_vs_official']} |
| Best revision delta vs anchor | {manifest['best_revision_delta_vs_anchor']} |
| Best blend MAE | {manifest['best_blend_mae']} |
| Best blend RMSE | {manifest['best_blend_rmse']} |
| Best blend delta vs official | {manifest['best_blend_delta_vs_official']} |

Leakage contract: all scored rows are before `{CONFIRMATION_START.date()}`; prior-bias features, revision thresholds, feature scaling, neighbor selection, residual corrections, do-no-harm checks, and blend weights use strictly prior target dates.
"""
    write_text(index_path, text)


def write_outputs(
    *,
    frame: pd.DataFrame,
    decomposition: pd.DataFrame,
    correlations: pd.DataFrame,
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
    summary = family_summary(scoreboard)
    comparison = baseline_comparison(frame, scoreboard, blend_scoreboard)
    top_ids = set(scoreboard.head(40)["candidate_id"].to_list()) if not scoreboard.empty else set()
    write_csv(artifacts / "revision_diagnostic_decomposition.csv", decomposition)
    write_csv(artifacts / "revision_feature_correlations.csv", correlations)
    write_csv(artifacts / "revision_specs.csv", specs)
    write_csv(artifacts / "revision_scoreboard.csv", scoreboard)
    write_csv(artifacts / "family_summary.csv", summary)
    write_csv(
        artifacts / "top_revision_predictions.csv",
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
        "revision_candidates": int(len(scoreboard)),
        "blend_candidates": int(len(blend_scoreboard)),
        "best_revision": "" if best_single is None else str(best_single["candidate_id"]),
        "best_revision_mae": None if best_single is None else float(best_single["mae"]),
        "best_revision_rmse": None if best_single is None else float(best_single["rmse"]),
        "best_revision_delta_vs_official": None
        if best_single is None
        else float(best_single["delta_vs_official_same_rows"]),
        "best_revision_delta_vs_anchor": None if best_single is None else float(best_single["delta_vs_anchor_same_rows"]),
        "best_blend": "" if best_blend is None else str(best_blend["candidate_id"]),
        "best_blend_mae": None if best_blend is None else float(best_blend["mae"]),
        "best_blend_rmse": None if best_blend is None else float(best_blend["rmse"]),
        "best_blend_delta_vs_official": None
        if best_blend is None
        else float(best_blend["delta_vs_official_same_rows"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "forecast_revision_momentum_deep_dive_manifest.json", manifest)
    write_readme(
        folder=folder,
        manifest=manifest,
        decomposition=decomposition,
        correlations=correlations,
        scoreboard=scoreboard,
        summary=summary,
        blend_scoreboard=blend_scoreboard,
        comparison=comparison,
    )
    update_master_index(manifest)
    return manifest


def run() -> dict[str, object]:
    frame, _prior_systems = build_failure_frame()
    frame = add_revision_features(frame)
    require_no_confirmation_dates(frame["target_date"], context="forecast revision momentum deep dive")
    decomposition, correlations = revision_diagnostic_decomposition(frame)
    scoreboard, predictions, specs = run_revision_screen(frame)
    blend_scoreboard, blend_predictions, blend_mapping = run_blend_screen(frame, scoreboard, predictions)
    return write_outputs(
        frame=frame,
        decomposition=decomposition,
        correlations=correlations,
        scoreboard=scoreboard,
        predictions=predictions,
        specs=specs,
        blend_scoreboard=blend_scoreboard,
        blend_predictions=blend_predictions,
        blend_mapping=blend_mapping,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 forecast revision momentum deep dive.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
