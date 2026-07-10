"""Pruned feature policy for the HKG Tmax next-round residual ML run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


PRUNED_FEATURES: list[str] = [
    "official_max_c",
    "official_min_c",
    "official_range_c",
    "official_midpoint_c",
    "official_max_bin",
    "official_range_bin",
    "issue_hour_bucket",
    "issue_age_minutes",
    "eligible_forecast_count",
    "rev_count",
    "rev_path_min_c",
    "rev_path_max_c",
    "rev_path_range_c",
    "rev_path_std_c",
    "rev_latest_minus_prev_max_c",
    "rev_latest_minus_first_max_c",
    "rev_last3_slope_c_per_hour",
    "rev_last_change_age_hours",
    "fcst_flag_thunderstorm",
    "fcst_flag_showers",
    "fcst_flag_heavy_showers",
    "fcst_flag_very_hot",
    "fcst_flag_nt_higher",
    "month",
    "season_bucket",
    "doy_sin",
    "doy_cos",
    "trend_years_since_2000",
    "hko_latest_temp_c",
    "hko_latest_temp_minus_official_max_c",
    "hko_latest_temp_minus_official_min_c",
    "hko_latest_dewpoint_c",
    "hko_latest_dewpoint_depression_c",
    "hko_latest_rh_pct",
    "hko_temp_trend_3h_c",
    "hko_temp_trend_6h_c",
    "hko_temp_trend_12h_c",
    "hko_temp_mean_6h_c",
    "hko_temp_mean_12h_c",
    "hko_temp_mean_24h_c",
    "hko_rh_trend_3h_pct",
    "hko_rh_mean_6h_pct",
    "hko_rh_mean_12h_pct",
    "network_mean_trend_6h_c",
    "network_spread_mean_6h_c",
    "network_spread_max_6h_c",
    "network_latest_temp_spread_c",
    "nt_heat_ceiling_index_c",
    "urban_core_mean_minus_coastal_marine_mean_c",
    "inland_nt_mean_minus_coastal_marine_mean_c",
    "west_nw_nt_mean_minus_coastal_marine_mean_c",
    "inland_coastal_spread_6h_max_c",
    "network_latest_hko_minus_mean_c",
    "hourly_any_thunderstorm_warning_24h",
    "hourly_any_rainstorm_warning_24h",
    "hourly_warning_text_count_24h",
    "hourly_rainfall_text_count_24h",
    "target_clim_doy_30yr_median_c",
    "target_clim_doy_10yr_median_c",
    "target_modern_warming_signal_c",
    "target_lag2_minus_doy30_clim_c",
    "target_roll30_anomaly_lag2_c",
    "target_roll30_minus_doy30_clim_c",
    "target_lag_2_missing_flag",
]

CANDIDATE_META_FEATURES: list[str] = [
    "candidate_resid_lgbm_a3_c",
    "candidate_resid_lgbm_pruned_full_c",
    "candidate_resid_catboost_c",
    "candidate_resid_linear_c",
    "candidate_resid_ensemble_c",
    "candidate_abs_resid_ensemble_c",
    "candidate_resid_std_c",
    "candidate_resid_sign_agreement_count",
    "candidate_positive_correction_flag",
    "candidate_negative_correction_flag",
    "candidate_correction_magnitude_bin",
]

EVALUATION_ONLY_COLUMNS: set[str] = {
    "true_residual_c",
    "raw_abs_error_c",
    "candidate_abs_error_c",
    "benefit_c",
    "apply_label",
    "strong_apply_label",
    "sign_label",
    "candidate_sign_correct",
    "abs_improvement_vs_raw_c",
    "helped_vs_raw_flag",
    "worsened_vs_raw_flag",
    "raw_error_decile",
    "raw_abs_error_decile",
}

FEATURE_FAMILIES: dict[str, str] = {
    "official_max_c": "official_anchor",
    "official_min_c": "official_anchor",
    "official_range_c": "official_anchor",
    "official_midpoint_c": "official_anchor",
    "official_max_bin": "official_anchor",
    "official_range_bin": "official_anchor",
    "issue_hour_bucket": "official_anchor",
    "issue_age_minutes": "official_anchor",
    "eligible_forecast_count": "forecast_revision",
    "month": "calendar",
    "season_bucket": "calendar",
    "doy_sin": "calendar",
    "doy_cos": "calendar",
    "trend_years_since_2000": "calendar",
}


@dataclass(frozen=True)
class PrunedFeaturePolicyResult:
    feature_names: list[str]
    missing_features: list[str]
    family_map: dict[str, str]
    max_raw_features: int
    status: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def family_for_feature(feature_name: str) -> str:
    if feature_name in FEATURE_FAMILIES:
        return FEATURE_FAMILIES[feature_name]
    if feature_name.startswith("rev_") or feature_name.startswith("fcst_flag_"):
        return "forecast_revision"
    if feature_name.startswith("hko_"):
        return "hko_hourly_state"
    if feature_name.startswith("network_") or feature_name in {
        "nt_heat_ceiling_index_c",
        "urban_core_mean_minus_coastal_marine_mean_c",
        "inland_nt_mean_minus_coastal_marine_mean_c",
        "west_nw_nt_mean_minus_coastal_marine_mean_c",
        "inland_coastal_spread_6h_max_c",
    }:
        return "station_network"
    if feature_name.startswith("hourly_"):
        return "text_warning_regime"
    if feature_name.startswith("target_"):
        return "target_history"
    if feature_name.startswith("residual_") or feature_name.startswith("residual_memory_"):
        return "official_residual_memory"
    if feature_name.startswith("candidate_"):
        return "candidate_meta"
    return "unknown"


def validate_pruned_features(
    frame: pd.DataFrame,
    *,
    max_raw_features: int = 90,
    allow_over_max: bool = False,
) -> PrunedFeaturePolicyResult:
    missing = [feature for feature in PRUNED_FEATURES if feature not in frame.columns]
    available = [feature for feature in PRUNED_FEATURES if feature in frame.columns]
    if len(available) > max_raw_features and not allow_over_max:
        raise ValueError(
            f"Pruned feature policy selected {len(available)} raw features, above max_raw_features={max_raw_features}"
        )
    family_map = {feature: family_for_feature(feature) for feature in available}
    status = "pass" if not missing and len(available) <= max_raw_features else "warn_missing_features"
    return PrunedFeaturePolicyResult(
        feature_names=available,
        missing_features=missing,
        family_map=family_map,
        max_raw_features=max_raw_features,
        status=status,
    )


def router_feature_names(frame: pd.DataFrame, *, max_raw_features: int = 90) -> list[str]:
    result = validate_pruned_features(frame, max_raw_features=max_raw_features)
    features = result.feature_names + [feature for feature in CANDIDATE_META_FEATURES if feature in frame.columns]
    forbidden = sorted(set(features) & EVALUATION_ONLY_COLUMNS)
    if forbidden:
        raise ValueError(f"Evaluation-only columns are not allowed as router features: {forbidden}")
    return features


def feature_policy_report(frame: pd.DataFrame, *, max_raw_features: int = 90) -> pd.DataFrame:
    result = validate_pruned_features(frame, max_raw_features=max_raw_features)
    rows: list[dict[str, Any]] = []
    for feature in PRUNED_FEATURES:
        rows.append(
            {
                "feature": feature,
                "present": feature in frame.columns,
                "family": family_for_feature(feature),
                "missing_pct": float(frame[feature].isna().mean() * 100.0) if feature in frame.columns else None,
            }
        )
    out = pd.DataFrame(rows)
    out["policy_status"] = result.status
    out["max_raw_features"] = max_raw_features
    out["selected_feature_count"] = len(result.feature_names)
    return out
