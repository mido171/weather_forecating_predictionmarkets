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
    feature_family,
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

FOLDER_NAME = "0029_official_residual_source_text_range_dynamics"
MIN_DIAGNOSTIC_ROWS = 500
MIN_CELL_ROWS = 35
MIN_HISTORY = 120
MIN_MATCH_ROWS = 20
TOP_METADATA_FEATURES = 24
TOP_NONPRESSURE_FEATURES = 20
TOP_BLEND_EXPERTS = 16

TEXT_PATTERNS: dict[str, tuple[str, ...]] = {
    "text_any_rain": ("rain", "shower", "drizzle"),
    "text_showers": ("shower", "showers"),
    "text_thunder": ("thunder", "squall"),
    "text_cloud": ("cloud", "cloudy", "overcast"),
    "text_sunny_or_fine": ("sunny", "fine", "bright", "sunshine"),
    "text_hot": ("hot",),
    "text_very_hot": ("very hot", "extremely hot"),
    "text_humid": ("humid", "humidity"),
    "text_mist_fog_haze": ("mist", "fog", "haze", "visibility"),
    "text_wind": ("wind", "windy", "strong wind", "fresh"),
    "text_easterly": ("easterly", "east to northeasterly", "east"),
    "text_northerly": ("northerly", "north to northeasterly", "north"),
    "text_southerly": ("southerly", "south to southwesterly", "south"),
}
TEXT_FLAG_FEATURES = tuple(TEXT_PATTERNS.keys())
EXACT_FEATURES = {
    "month",
    "monsoon_phase_code",
    "source_is_press",
    "source_is_rss",
    *TEXT_FLAG_FEATURES,
}


@dataclass(frozen=True)
class BucketExpertSpec:
    feature: str
    bins: int
    exact_match: bool
    same_source: bool
    month_conditioned: bool
    phase_conditioned: bool
    shrinkage: float = 80.0
    correction_clip_c: float = 2.5
    min_history: int = MIN_HISTORY
    min_match_rows: int = MIN_MATCH_ROWS


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 150) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def monsoon_phase(month: int) -> str:
    if month in (11, 12, 1, 2, 3):
        return "northeast_monsoon"
    if month in (6, 7, 8, 9):
        return "southwest_monsoon"
    return "transition"


def combine_official_text(frame: pd.DataFrame) -> pd.Series:
    parts: list[pd.Series] = []
    for column in ("weather_text", "wind_text", "description_text", "title"):
        if column in frame.columns:
            parts.append(frame[column].fillna("").astype(str))
    if not parts:
        return pd.Series([""] * len(frame), index=frame.index)
    combined = parts[0].copy()
    for part in parts[1:]:
        combined = combined.str.cat(part, sep=" ")
    return combined.str.replace(r"\s+", " ", regex=True).str.strip()


def add_text_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["official_text"] = combine_official_text(out)
    lower = out["official_text"].fillna("").astype(str).str.lower()
    for feature, patterns in TEXT_PATTERNS.items():
        regex = "|".join(re.escape(pattern) for pattern in patterns)
        out[feature] = lower.str.contains(regex, regex=True, na=False).astype(float)
    out["text_keyword_count"] = out[list(TEXT_FLAG_FEATURES)].sum(axis=1)
    out["official_text_length"] = lower.str.len().astype(float)
    return out


def add_source_phase_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out["target_date"], errors="coerce")
    out["month"] = dates.dt.month
    months = pd.to_numeric(out["month"], errors="coerce")
    out["season"] = months.map(lambda value: season_name(int(value)) if np.isfinite(value) else "")
    out["monsoon_phase"] = months.map(lambda value: monsoon_phase(int(value)) if np.isfinite(value) else "")
    out["monsoon_phase_code"] = out["monsoon_phase"].map(
        {"northeast_monsoon": -1.0, "transition": 0.0, "southwest_monsoon": 1.0}
    )
    source = out.get("forecast_source_family", pd.Series([""] * len(out), index=out.index)).astype(str)
    out["source_is_press"] = source.eq("press_archive").astype(float)
    out["source_is_rss"] = source.eq("rss_archive").astype(float)
    if "issue_to_cutoff_hours" in out.columns:
        out["issue_to_cutoff_abs_hours"] = pd.to_numeric(out["issue_to_cutoff_hours"], errors="coerce").abs()
    return out


def add_context_spreads(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    spreads: dict[str, tuple[str, str]] = {
        "thermal_590960_minus_596730_c": (
            "isd_station_air_temperature_c_590960_99999",
            "isd_station_air_temperature_c_596730_99999",
        ),
        "thermal_590870_minus_596730_c": (
            "isd_station_air_temperature_c_590870_99999",
            "isd_station_air_temperature_c_596730_99999",
        ),
        "thermal_592930_minus_596730_c": (
            "isd_station_air_temperature_c_592930_99999",
            "isd_station_air_temperature_c_596730_99999",
        ),
        "thermal_590960_minus_592780_c": (
            "isd_station_air_temperature_c_590960_99999",
            "isd_station_air_temperature_c_592780_99999",
        ),
        "dew_590960_minus_596730_c": (
            "isd_station_dew_point_c_590960_99999",
            "isd_station_dew_point_c_596730_99999",
        ),
        "dew_590870_minus_592780_c": (
            "isd_station_dew_point_c_590870_99999",
            "isd_station_dew_point_c_592780_99999",
        ),
    }
    for target, (left, right) in spreads.items():
        if left in out.columns and right in out.columns:
            out[target] = pd.to_numeric(out[left], errors="coerce") - pd.to_numeric(out[right], errors="coerce")

    if {"isd_wind_u_mean_mps", "isd_wind_v_mean_mps"}.issubset(out.columns):
        u = pd.to_numeric(out["isd_wind_u_mean_mps"], errors="coerce")
        v = pd.to_numeric(out["isd_wind_v_mean_mps"], errors="coerce")
        out["isd_wind_vector_speed_mps"] = np.sqrt(np.square(u) + np.square(v))
        out["isd_onshore_easterly_proxy_mps"] = -u
        out["isd_northerly_proxy_mps"] = -v
    return out


def add_forecast_dynamics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["forecast_source_family", "target_date"]).reset_index(drop=True).copy()
    grouped = out.groupby("forecast_source_family", observed=True, sort=False)
    for column, output in (
        ("forecast_max_c", "forecast_max"),
        ("forecast_min_c", "forecast_min"),
        ("forecast_range_c", "forecast_range"),
        ("forecast_midpoint_c", "forecast_midpoint"),
        ("issue_to_cutoff_hours", "issue_to_cutoff"),
    ):
        if column not in out.columns:
            continue
        numeric = pd.to_numeric(out[column], errors="coerce")
        out[f"{output}_lag1_source_c"] = grouped[column].shift(1)
        out[f"{output}_change_1_source_c"] = numeric - pd.to_numeric(grouped[column].shift(1), errors="coerce")
        prior7 = grouped[column].transform(lambda series: pd.to_numeric(series, errors="coerce").shift(1).rolling(7, min_periods=3).mean())
        out[f"{output}_prior7_mean_source_c"] = prior7
        if column in {"forecast_max_c", "forecast_range_c"}:
            prior7_std = grouped[column].transform(
                lambda series: pd.to_numeric(series, errors="coerce").shift(1).rolling(7, min_periods=3).std()
            )
            out[f"{output}_prior7_std_source_c"] = prior7_std
            out[f"{output}_vs_prior7_mean_source_c"] = numeric - prior7
    return out.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True)


def build_analysis_frame() -> pd.DataFrame:
    frame = build_official_feature_frame()
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(frame["target_date"], context="official residual source/text/range frame")
    frame = add_source_phase_features(frame)
    frame = add_context_spreads(frame)
    frame = add_text_flags(frame)
    frame = add_forecast_dynamics(frame)
    frame["official_raw"] = pd.to_numeric(frame["forecast_max_c"], errors="coerce")
    frame["actual_minus_official_c"] = pd.to_numeric(frame["target_tmax_c"], errors="coerce") - frame["official_raw"]
    frame["official_abs_error_c"] = frame["actual_minus_official_c"].abs()
    return frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True)


def safe_corr(left: pd.Series, right: pd.Series, *, method: str = "pearson", min_rows: int = 250) -> float:
    pair = pd.concat([pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1).dropna()
    if len(pair) < min_rows or pair.iloc[:, 0].nunique() <= 2 or pair.iloc[:, 1].nunique() <= 2:
        return math.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def available_numeric_features(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    min_non_null: int = MIN_DIAGNOSTIC_ROWS,
) -> tuple[str, ...]:
    out: list[str] = []
    for feature in candidates:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if int(values.notna().sum()) >= min_non_null and values.nunique(dropna=True) > 1:
            out.append(feature)
    return tuple(out)


def metadata_features(frame: pd.DataFrame) -> tuple[str, ...]:
    return available_numeric_features(
        frame,
        (
            "forecast_max_c",
            "forecast_min_c",
            "forecast_range_c",
            "forecast_midpoint_c",
            "issue_to_cutoff_hours",
            "issue_to_cutoff_abs_hours",
            "forecast_max_change_1_source_c",
            "forecast_min_change_1_source_c",
            "forecast_range_change_1_source_c",
            "forecast_midpoint_change_1_source_c",
            "forecast_max_prior7_mean_source_c",
            "forecast_max_prior7_std_source_c",
            "forecast_max_vs_prior7_mean_source_c",
            "forecast_range_prior7_mean_source_c",
            "forecast_range_prior7_std_source_c",
            "forecast_range_vs_prior7_mean_source_c",
            "text_keyword_count",
            "official_text_length",
            "source_is_press",
            "source_is_rss",
            "month",
            "monsoon_phase_code",
            "rh_min_pct",
            "rh_max_pct",
            *TEXT_FLAG_FEATURES,
        ),
    )


def nonpressure_modifier_features(frame: pd.DataFrame) -> tuple[str, ...]:
    return available_numeric_features(
        frame,
        (
            "isd_temp_dewpoint_spread_mean_c",
            "isd_dewpoint_midday_minus_temp_c",
            "isd_dew_point_mean_c_change_1d",
            "daily_hong_kong_observatory_mean_relative_humidity_lag7_roll7",
            "daily_hong_kong_observatory_mean_cloud_amount_lag7_roll7",
            "daily_hong_kong_observatory_daily_rainfall_lag7_roll7",
            "daily_waglan_island_mean_wind_speed_lag7_roll7",
            "daily_waglan_island_prevailing_wind_direction_lag7_roll7",
            "isd_wind_speed_mean_mps",
            "isd_wind_speed_max_mps",
            "isd_wind_vector_speed_mps",
            "isd_onshore_easterly_proxy_mps",
            "isd_northerly_proxy_mps",
            "isd_wind_speed_mean_mps_change_1d",
            "ua_wind_u_1000hpa_mps",
            "ua_wind_v_1000hpa_mps",
            "ua_theta_e_1000_850_mean_k",
            "ua_mse_1000_850_mean_kj_kg",
            "ua_mse_925_850_mean_kj_kg",
            "ua_dewpoint_925hpa_c",
            "ua_tendency_48h_ua_theta_1000hpa_k",
            "igra_thickness_1000_500_m_change_48h",
            "isd_north_south_temp_gradient_c",
            "isd_east_west_temp_gradient_c",
            "isd_graph_laplacian_mode_1",
            "isd_graph_laplacian_mode_3",
            "isd_graph_total_variation_c2",
            "isd_morning_to_midday_temp_rise_c",
            "thermal_590960_minus_596730_c",
            "thermal_590870_minus_596730_c",
            "thermal_592930_minus_596730_c",
            "thermal_590960_minus_592780_c",
            "dew_590960_minus_596730_c",
            "dew_590870_minus_592780_c",
            "target_lag7_minus_roll365_c",
            "target_lag7_minus_lag14_c",
            "target_roll30_std_lag7_c",
            "target_entropy_30_lag7",
        ),
    )


def bucket_labels(values: pd.Series, bins: int, *, exact_match: bool, min_rows: int = MIN_CELL_ROWS) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if exact_match:
        return numeric.map(lambda value: f"exact_{int(value)}" if np.isfinite(value) else "missing")
    if int(numeric.notna().sum()) < bins * min_rows or numeric.nunique(dropna=True) < bins:
        return pd.Series(["insufficient"] * len(values), index=values.index)
    ranked = numeric.rank(method="first")
    labels = [f"q{index + 1}" for index in range(bins)]
    return pd.qcut(ranked, bins, labels=labels).astype(str).where(numeric.notna(), "missing")


def feature_lifts(frame: pd.DataFrame, features: tuple[str, ...], *, feature_group: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in features:
        exact = feature in EXACT_FEATURES or pd.to_numeric(frame[feature], errors="coerce").nunique(dropna=True) <= 2
        buckets = bucket_labels(frame[feature], 5, exact_match=exact)
        if buckets.eq("insufficient").all():
            continue
        work = frame[["target_date", "actual_minus_official_c", "official_abs_error_c"]].copy()
        work["bucket"] = buckets
        bucket_rows: list[dict[str, object]] = []
        for bucket, group in work.groupby("bucket", observed=True, dropna=False):
            if bucket in {"missing", "insufficient"} or len(group) < MIN_CELL_ROWS:
                continue
            bucket_rows.append(
                {
                    "feature": feature,
                    "feature_group": feature_group,
                    "family": feature_family(feature),
                    "bucket": str(bucket),
                    "rows": int(len(group)),
                    "first_date": str(group["target_date"].min().date()),
                    "last_date": str(group["target_date"].max().date()),
                    "residual_actual_minus_official_mean_c": float(group["actual_minus_official_c"].mean()),
                    "mae_c": float(group["official_abs_error_c"].mean()),
                    "hotter_than_forecast_rate": float((group["actual_minus_official_c"] > 1.0).mean()),
                    "cooler_than_forecast_rate": float((group["actual_minus_official_c"] < -1.0).mean()),
                }
            )
        if len(bucket_rows) < 2:
            continue
        residuals = [float(row["residual_actual_minus_official_mean_c"]) for row in bucket_rows]
        maes = [float(row["mae_c"]) for row in bucket_rows]
        for row in bucket_rows:
            row["residual_spread_across_feature_buckets_c"] = max(residuals) - min(residuals)
            row["mae_spread_across_feature_buckets_c"] = max(maes) - min(maes)
            row["priority"] = abs(row["residual_spread_across_feature_buckets_c"]) + 0.5 * abs(
                row["mae_spread_across_feature_buckets_c"]
            )
        rows.extend(bucket_rows)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["priority", "feature", "bucket"], ascending=[False, True, True]).reset_index(
        drop=True
    )


def feature_signal_scan(frame: pd.DataFrame, features: tuple[str, ...], *, feature_group: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        valid_dates = pd.to_datetime(frame.loc[values.notna(), "target_date"], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "feature_group": feature_group,
                "family": feature_family(feature),
                "n": int(values.notna().sum()),
                "first_date": "" if valid_dates.empty else str(valid_dates.min().date()),
                "last_date": "" if valid_dates.empty else str(valid_dates.max().date()),
                "target_pearson": safe_corr(values, frame["target_tmax_c"]),
                "target_spearman": safe_corr(values, frame["target_tmax_c"], method="spearman"),
                "residual_pearson": safe_corr(values, frame["actual_minus_official_c"]),
                "residual_spearman": safe_corr(values, frame["actual_minus_official_c"], method="spearman"),
                "abs_error_pearson": safe_corr(values, frame["official_abs_error_c"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["priority"] = (
        out[["residual_pearson", "residual_spearman"]].abs().max(axis=1).fillna(0.0) * 3.0
        + out["abs_error_pearson"].abs().fillna(0.0)
        + out[["target_pearson", "target_spearman"]].abs().max(axis=1).fillna(0.0) * 0.25
    )
    return out.sort_values(["priority", "feature"], ascending=[False, True]).reset_index(drop=True)


def diagnostic_pair_interactions(
    frame: pd.DataFrame,
    metadata: tuple[str, ...],
    modifiers: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for meta_feature in metadata:
        meta_exact = meta_feature in EXACT_FEATURES or pd.to_numeric(frame[meta_feature], errors="coerce").nunique(dropna=True) <= 2
        meta_bucket = bucket_labels(frame[meta_feature], 3, exact_match=meta_exact)
        if meta_bucket.eq("insufficient").all():
            continue
        for modifier in modifiers:
            mod_exact = modifier in EXACT_FEATURES or pd.to_numeric(frame[modifier], errors="coerce").nunique(dropna=True) <= 2
            modifier_bucket = bucket_labels(frame[modifier], 3, exact_match=mod_exact)
            if modifier_bucket.eq("insufficient").all():
                continue
            work = frame[["target_date", "actual_minus_official_c", "official_abs_error_c"]].copy()
            work["metadata_bucket"] = meta_bucket
            work["modifier_bucket"] = modifier_bucket
            cell_rows: list[dict[str, object]] = []
            for (metadata_bucket, modifier_bucket_value), group in work.groupby(
                ["metadata_bucket", "modifier_bucket"],
                observed=True,
                dropna=False,
            ):
                if (
                    metadata_bucket in {"missing", "insufficient"}
                    or modifier_bucket_value in {"missing", "insufficient"}
                    or len(group) < MIN_CELL_ROWS
                ):
                    continue
                cell_rows.append(
                    {
                        "metadata_feature": meta_feature,
                        "modifier_feature": modifier,
                        "metadata_bucket": str(metadata_bucket),
                        "modifier_bucket": str(modifier_bucket_value),
                        "rows": int(len(group)),
                        "residual_actual_minus_official_mean_c": float(group["actual_minus_official_c"].mean()),
                        "mae_c": float(group["official_abs_error_c"].mean()),
                        "hotter_than_forecast_rate": float((group["actual_minus_official_c"] > 1.0).mean()),
                        "cooler_than_forecast_rate": float((group["actual_minus_official_c"] < -1.0).mean()),
                    }
                )
            if len(cell_rows) < 4:
                continue
            residuals = [float(row["residual_actual_minus_official_mean_c"]) for row in cell_rows]
            maes = [float(row["mae_c"]) for row in cell_rows]
            for row in cell_rows:
                row["residual_spread_across_cells_c"] = max(residuals) - min(residuals)
                row["mae_spread_across_cells_c"] = max(maes) - min(maes)
                row["min_cell_rows_for_pair"] = min(int(cell["rows"]) for cell in cell_rows)
                row["interaction_priority"] = abs(row["residual_spread_across_cells_c"]) + 0.75 * abs(
                    row["mae_spread_across_cells_c"]
                )
            rows.extend(cell_rows)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["interaction_priority", "mae_c", "metadata_feature", "modifier_feature"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def candidate_id_for_spec(spec: BucketExpertSpec) -> str:
    match = "exact" if spec.exact_match else f"q{spec.bins}"
    source = "same_source" if spec.same_source else "all_prior"
    month = "month" if spec.month_conditioned else "all_month"
    phase = "phase" if spec.phase_conditioned else "all_phase"
    return slug(f"official_residual_{spec.feature}_{match}_{source}_{month}_{phase}")


def prior_bucket_match(
    prior_values: np.ndarray,
    current_value: float,
    *,
    bins: int,
    exact_match: bool,
    min_match_rows: int,
) -> tuple[np.ndarray, str] | None:
    if not np.isfinite(current_value):
        return None
    if exact_match:
        return np.isclose(prior_values, current_value, equal_nan=False), f"exact_{current_value:g}"
    valid = prior_values[np.isfinite(prior_values)]
    if len(valid) < bins * min_match_rows or len(np.unique(valid)) < bins:
        return None
    edges = np.unique(np.nanquantile(valid, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    if len(edges) < bins - 1:
        return None
    prior_buckets = np.searchsorted(edges, prior_values, side="right")
    current_bucket = int(np.searchsorted(edges, current_value, side="right"))
    return prior_buckets == current_bucket, f"q{current_bucket + 1}"


def past_only_bucket_predictions(frame: pd.DataFrame, spec: BucketExpertSpec) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    forecast = pd.to_numeric(ordered["forecast_max_c"], errors="coerce").to_numpy(dtype=float)
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    residual = target - forecast
    feature_values = pd.to_numeric(ordered[spec.feature], errors="coerce").to_numpy(dtype=float)
    sources = ordered["forecast_source_family"].astype(str).to_numpy()
    months = pd.to_numeric(ordered["month"], errors="coerce").to_numpy(dtype=float)
    phases = ordered["monsoon_phase"].astype(str).to_numpy()

    predictions: list[float] = []
    corrections: list[float] = []
    rows_used: list[int] = []
    matched_buckets: list[str] = []
    for index, target_date in enumerate(dates):
        if not np.isfinite(forecast[index]) or not np.isfinite(feature_values[index]):
            predictions.append(float(forecast[index]) if np.isfinite(forecast[index]) else math.nan)
            corrections.append(0.0)
            rows_used.append(0)
            matched_buckets.append("")
            continue

        prior_mask = np.arange(len(ordered)) < int(np.searchsorted(dates, target_date, side="left"))
        if spec.same_source:
            prior_mask &= sources == sources[index]
        if spec.month_conditioned:
            prior_mask &= months == months[index]
        if spec.phase_conditioned:
            prior_mask &= phases == phases[index]
        prior_mask &= np.isfinite(residual) & np.isfinite(feature_values)
        prior_index = np.flatnonzero(prior_mask)
        if len(prior_index) < spec.min_history:
            predictions.append(float(forecast[index]))
            corrections.append(0.0)
            rows_used.append(0)
            matched_buckets.append("")
            continue

        matched = prior_bucket_match(
            feature_values[prior_index],
            feature_values[index],
            bins=spec.bins,
            exact_match=spec.exact_match,
            min_match_rows=spec.min_match_rows,
        )
        if matched is None:
            predictions.append(float(forecast[index]))
            corrections.append(0.0)
            rows_used.append(0)
            matched_buckets.append("")
            continue
        matched_mask, matched_bucket = matched
        matched_index = prior_index[matched_mask]
        if len(matched_index) < spec.min_match_rows:
            predictions.append(float(forecast[index]))
            corrections.append(0.0)
            rows_used.append(0)
            matched_buckets.append("")
            continue

        raw_correction = float(np.nanmean(residual[matched_index]))
        weight = len(matched_index) / (len(matched_index) + float(spec.shrinkage))
        correction = float(np.clip(raw_correction * weight, -spec.correction_clip_c, spec.correction_clip_c))
        predictions.append(float(forecast[index] + correction))
        corrections.append(correction)
        rows_used.append(int(len(matched_index)))
        matched_buckets.append(matched_bucket)

    out = ordered[["target_date", "forecast_source_family", "target_tmax_c", "forecast_max_c"]].copy()
    out["official_raw"] = forecast
    out["candidate_prediction_c"] = predictions
    out["residual_correction_c"] = corrections
    out["past_rows_used"] = rows_used
    out["matched_bucket"] = matched_buckets
    out["feature"] = spec.feature
    return out


def top_modifier_features(
    modifier_lifts: pd.DataFrame,
    modifier_scan: pd.DataFrame,
    pair_interactions: pd.DataFrame,
) -> tuple[str, ...]:
    features: list[str] = []
    for table, column in (
        (pair_interactions, "modifier_feature"),
        (modifier_lifts, "feature"),
        (modifier_scan, "feature"),
    ):
        if table.empty or column not in table.columns:
            continue
        for feature in table[column].dropna().astype(str).to_list():
            if feature not in features:
                features.append(feature)
            if len(features) >= TOP_NONPRESSURE_FEATURES:
                return tuple(features)
    return tuple(features)


def top_metadata_features(
    metadata: tuple[str, ...],
    metadata_lifts: pd.DataFrame,
    metadata_scan: pd.DataFrame,
    pair_interactions: pd.DataFrame,
) -> tuple[str, ...]:
    features: list[str] = []
    for table, column in (
        (pair_interactions, "metadata_feature"),
        (metadata_lifts, "feature"),
        (metadata_scan, "feature"),
    ):
        if table.empty or column not in table.columns:
            continue
        for feature in table[column].dropna().astype(str).to_list():
            if feature in metadata and feature not in features:
                features.append(feature)
            if len(features) >= TOP_METADATA_FEATURES:
                return tuple(features)
    for feature in metadata:
        if feature not in features:
            features.append(feature)
        if len(features) >= TOP_METADATA_FEATURES:
            break
    return tuple(features)


def conditioning_options(feature: str, *, exact: bool) -> tuple[tuple[bool, bool], ...]:
    options: list[tuple[bool, bool]] = [(False, False)]
    if feature != "monsoon_phase_code":
        options.append((False, True))
    if not exact and feature != "month":
        options.append((True, False))
    return tuple(options)


def build_bucket_specs(
    frame: pd.DataFrame,
    metadata: tuple[str, ...],
    metadata_lifts: pd.DataFrame,
    metadata_scan: pd.DataFrame,
    modifier_lifts: pd.DataFrame,
    modifier_scan: pd.DataFrame,
    pair_interactions: pd.DataFrame,
) -> list[BucketExpertSpec]:
    features = list(
        dict.fromkeys(
            (
                *top_metadata_features(metadata, metadata_lifts, metadata_scan, pair_interactions),
                *top_modifier_features(modifier_lifts, modifier_scan, pair_interactions),
            )
        )
    )
    specs: list[BucketExpertSpec] = []
    for feature in features:
        unique_values = pd.to_numeric(frame[feature], errors="coerce").nunique(dropna=True)
        exact = feature in EXACT_FEATURES or unique_values <= 2
        bins_list = (2,) if exact else (3, 5)
        for bins in bins_list:
            for same_source in (False, True):
                for month_conditioned, phase_conditioned in conditioning_options(feature, exact=exact):
                    specs.append(
                        BucketExpertSpec(
                            feature=feature,
                            bins=bins,
                            exact_match=exact,
                            same_source=same_source,
                            month_conditioned=month_conditioned,
                            phase_conditioned=phase_conditioned,
                        )
                    )
    return specs


def score_bucket_candidate(
    predictions: pd.DataFrame,
    spec: BucketExpertSpec,
    candidate_id: str,
) -> dict[str, object]:
    candidate = score_prediction_frame(predictions.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "official_prediction"}), "official_prediction")
    corrected = predictions["past_rows_used"] > 0
    return {
        "candidate_id": candidate_id,
        "feature": spec.feature,
        "family": feature_family(spec.feature),
        "bins": spec.bins,
        "exact_match": spec.exact_match,
        "same_source": spec.same_source,
        "month_conditioned": spec.month_conditioned,
        "phase_conditioned": spec.phase_conditioned,
        "min_history": spec.min_history,
        "min_match_rows": spec.min_match_rows,
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


def run_bucket_experts(
    frame: pd.DataFrame,
    metadata: tuple[str, ...],
    metadata_lifts: pd.DataFrame,
    metadata_scan: pd.DataFrame,
    modifier_lifts: pd.DataFrame,
    modifier_scan: pd.DataFrame,
    pair_interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in build_bucket_specs(
        frame,
        metadata,
        metadata_lifts,
        metadata_scan,
        modifier_lifts,
        modifier_scan,
        pair_interactions,
    ):
        predictions = past_only_bucket_predictions(frame, spec)
        candidate_id = candidate_id_for_spec(spec)
        predictions["candidate_id"] = candidate_id
        score_rows.append(score_bucket_candidate(predictions, spec, candidate_id))
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(score_rows)
    if not scoreboard.empty:
        scoreboard = scoreboard.sort_values(["delta_vs_official_same_rows", "mae"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return scoreboard, predictions


def family_summary(scoreboard: pd.DataFrame) -> pd.DataFrame:
    if scoreboard.empty:
        return pd.DataFrame()
    return (
        scoreboard.groupby("family", observed=True)
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
            "feature",
            "bins",
            "exact_match",
            "same_source",
            "month_conditioned",
            "phase_conditioned",
            "mae",
            "delta_vs_official_same_rows",
        ]
    ].copy()
    mapping["expert_id"] = [
        f"or_{rank:02d}_{slug(row.candidate_id, limit=42)}"
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
    experts = ["official_raw", *[column for column in blend_frame.columns if column.startswith("or_")]]
    score_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for mode in ("best", "inverse_mae"):
        for same_source in (False, True):
            candidate_id = f"official_residual_blend_{mode}_{'same_source' if same_source else 'all_prior'}"
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


def write_outputs(
    *,
    frame: pd.DataFrame,
    metadata_scan: pd.DataFrame,
    modifier_scan: pd.DataFrame,
    metadata_lifts: pd.DataFrame,
    modifier_lifts: pd.DataFrame,
    text_lifts: pd.DataFrame,
    pair_interactions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
    blend_predictions: pd.DataFrame,
    blend_mapping: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    write_csv(artifacts / "meta_scan.csv", metadata_scan)
    write_csv(artifacts / "modifier_scan.csv", modifier_scan)
    write_csv(artifacts / "metadata_lifts.csv", metadata_lifts)
    write_csv(artifacts / "modifier_lifts.csv", modifier_lifts)
    write_csv(artifacts / "text_keyword_lifts.csv", text_lifts)
    write_csv(artifacts / "pair_diagnostics.csv", pair_interactions)
    write_csv(artifacts / "bucket_scoreboard.csv", scoreboard)
    write_csv(artifacts / "family_summary.csv", family_summary(scoreboard))
    top_ids = set(scoreboard.head(50)["candidate_id"].to_list()) if not scoreboard.empty else set()
    write_csv(artifacts / "top_bucket_predictions.csv", predictions[predictions["candidate_id"].isin(top_ids)].copy())
    write_csv(artifacts / "blend_scoreboard.csv", blend_scoreboard)
    write_csv(artifacts / "blend_predictions.csv", blend_predictions)
    write_csv(artifacts / "blend_mapping.csv", blend_mapping)

    best_bucket = scoreboard.iloc[0] if not scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "source_counts": {str(k): int(v) for k, v in frame["forecast_source_family"].value_counts().to_dict().items()},
        "metadata_features": int(len(metadata_features(frame))),
        "nonpressure_modifier_features": int(len(nonpressure_modifier_features(frame))),
        "metadata_lift_rows": int(len(metadata_lifts)),
        "text_lift_rows": int(len(text_lifts)),
        "pair_diagnostic_rows": int(len(pair_interactions)),
        "bucket_candidates": int(len(scoreboard)),
        "blend_candidates": int(len(blend_scoreboard)),
        "best_bucket_candidate": "" if best_bucket is None else str(best_bucket["candidate_id"]),
        "best_bucket_mae": None if best_bucket is None else float(best_bucket["mae"]),
        "best_bucket_rmse": None if best_bucket is None else float(best_bucket["rmse"]),
        "best_bucket_delta_vs_official": None if best_bucket is None else float(best_bucket["delta_vs_official_same_rows"]),
        "best_blend_candidate": "" if best_blend is None else str(best_blend["candidate_id"]),
        "best_blend_mae": None if best_blend is None else float(best_blend["mae"]),
        "best_blend_rmse": None if best_blend is None else float(best_blend["rmse"]),
        "best_blend_delta_vs_official": None if best_blend is None else float(best_blend["delta_vs_official_same_rows"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "official_residual_source_text_range_dynamics_manifest.json", manifest)

    write_readme(
        folder=folder,
        manifest=manifest,
        metadata_scan=metadata_scan,
        modifier_scan=modifier_scan,
        metadata_lifts=metadata_lifts,
        text_lifts=text_lifts,
        pair_interactions=pair_interactions,
        scoreboard=scoreboard,
        blend_scoreboard=blend_scoreboard,
    )
    update_master_index(manifest)
    return manifest


def write_readme(
    *,
    folder: Path,
    manifest: dict[str, object],
    metadata_scan: pd.DataFrame,
    modifier_scan: pd.DataFrame,
    metadata_lifts: pd.DataFrame,
    text_lifts: pd.DataFrame,
    pair_interactions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
) -> None:
    best_bucket = scoreboard.iloc[0] if not scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    best_bucket_text = "No official-residual bucket expert was scoreable."
    if best_bucket is not None:
        best_bucket_text = (
            f"Best bucket expert: `{best_bucket['candidate_id']}` with MAE `{best_bucket['mae']:.4f}`, "
            f"RMSE `{best_bucket['rmse']:.4f}`, and official delta "
            f"`{best_bucket['delta_vs_official_same_rows']:.4f}`."
        )
    best_blend_text = "No official-residual blend was scoreable."
    if best_blend is not None:
        best_blend_text = (
            f"Best residual blend: `{best_blend['candidate_id']}` with MAE `{best_blend['mae']:.4f}`, "
            f"RMSE `{best_blend['rmse']:.4f}`, and official delta "
            f"`{best_blend['delta_vs_official_same_rows']:.4f}`."
        )

    readme = f"""# Official Residual Source/Text/Range Dynamics

Generated: `{manifest['generated_at_utc']}`

## Purpose

This insight tests a different path from the previous pressure-only screens. Instead of asking whether pressure gradients alone can correct the official forecast, it asks whether the official forecast itself contains exploitable residual structure:

- the numeric forecast level, minimum, maximum, midpoint, and range width;
- the official source family, currently `press_archive` and `rss_archive`;
- issue age at the T-1 15:00 HKT cutoff;
- source-local forecast changes and prior 7-issue dynamics;
- official wording such as rain, showers, cloud, hot, very hot, humid, haze, wind, and wind direction;
- non-pressure context that can explain why the official anchor misses, including humidity, dewpoint spread, cloud/rain memory, wind/marine proxies, upper-air heat/moisture, and station-network thermal gradients.

The reason this matters is that an elite HKG Tmax system is unlikely to win by predicting the target from scratch. It should start from the official forecast anchor and learn when that anchor is systematically too hot or too cool. This folder is one step toward that residual-intelligence layer.

## Data Window

Rows used: `{manifest['official_rows']}` official forecast/target rows.

Date range: `{manifest['first_target_date']}` to `{manifest['last_target_date']}`.

Source counts: `{manifest['source_counts']}`.

Important limitation: the stable official forecast export is still non-contiguous. It currently contains early press-archive history plus RSS-era rows, while the active 2005+ press-detail acquisition has not yet been promoted into the scored export. That means this screen is useful for mechanism discovery and deployable-style leakage testing, but it is not yet the final 2000-2026 system.

## Leakage Contract

- All scored rows are earlier than `{CONFIRMATION_START.date()}`.
- No 2024+ target labels are loaded for tuning, scoring, feature ranking, correction fitting, or blending.
- Official forecast rows come from the existing latest-pre-cutoff selection.
- Forecast dynamics use only same-source lagged forecast values.
- Diagnostic scans and interaction tables are not promoted as deployable forecasts.
- Each bucket correction estimates quantile edges and residual means from strictly earlier target dates using `searchsorted(..., side="left")`.
- Same-source candidates restrict history to the same official source family.
- Month-conditioned and phase-conditioned candidates only use matching prior rows.
- Blend selection and inverse-MAE weights are estimated only from prior realized expert errors.

## What Was Tested

There are four layers:

1. Metadata signal scan: correlations between official forecast metadata and target/residual/absolute error.
2. Text keyword lift tables: how residuals change when official wording contains rain, cloud, hot, humid, wind, haze, or directional words.
3. Pair diagnostics: official metadata crossed with non-pressure context to discover miss regimes.
4. Deployable-style bucket experts: fold-local residual corrections using only past rows, then a prior-performance blend over the strongest experts.

## Main Results

{best_bucket_text}

{best_blend_text}

## Top Metadata Signal Scan

{markdown_table(metadata_scan.head(20), max_rows=20)}

## Top Non-Pressure Modifier Signal Scan

{markdown_table(modifier_scan.head(20), max_rows=20)}

## Top Metadata Lifts

{markdown_table(metadata_lifts.head(20), max_rows=20)}

## Text Keyword Lifts

{markdown_table(text_lifts.head(25), max_rows=25)}

## Top Pair Diagnostics

{markdown_table(pair_interactions.head(20), max_rows=20)}

## Top Deployable Bucket Experts

{markdown_table(scoreboard.head(20), max_rows=20)}

## Blend Scoreboard

{markdown_table(blend_scoreboard, max_rows=20)}

## Interpretation

This screen is deliberately broad because the path to a sub-competitive MAE requires squeezing information out of the official forecast channel, not only out of weather observations. A positive bucket or blend result can later be promoted into the composite expert stack after the full forecast archive is refreshed and ablated. A weak result is still useful: it tells us which official metadata and text signals are descriptive but not stable enough as single-feature past-only corrections.

The most important next step remains forecast-archive continuity. More 2005-2020 press rows should make source-local forecast dynamics, wording behavior, and range-width calibration much more valuable because the correction layer will have enough same-source history instead of jumping from 2004 to 2021.
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Official Residual Source/Text/Range Dynamics\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_official_residual_source_text_range_dynamics.py`:

- `{FOLDER_NAME}`: official forecast residual analysis across source family, forecast range/level, source-local forecast changes, text keyword flags, and non-pressure context.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Metadata features | {manifest['metadata_features']} |
| Non-pressure modifier features | {manifest['nonpressure_modifier_features']} |
| Text lift rows | {manifest['text_lift_rows']} |
| Pair diagnostic rows | {manifest['pair_diagnostic_rows']} |
| Bucket candidates | {manifest['bucket_candidates']} |
| Best bucket MAE | {manifest['best_bucket_mae']} |
| Best bucket RMSE | {manifest['best_bucket_rmse']} |
| Best bucket delta vs official | {manifest['best_bucket_delta_vs_official']} |
| Best blend MAE | {manifest['best_blend_mae']} |
| Best blend RMSE | {manifest['best_blend_rmse']} |
| Best blend delta vs official | {manifest['best_blend_delta_vs_official']} |

Leakage contract: all scored rows are before `{CONFIRMATION_START.date()}`; forecast dynamics are same-source prior values; bucket corrections and blend weights use strictly earlier target dates.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    frame = build_analysis_frame()
    require_no_confirmation_dates(frame["target_date"], context="official residual source/text/range analysis")
    metadata = metadata_features(frame)
    modifiers = nonpressure_modifier_features(frame)
    metadata_scan = feature_signal_scan(frame, metadata, feature_group="official_metadata")
    modifier_scan = feature_signal_scan(frame, modifiers, feature_group="nonpressure_modifier")
    metadata_lifts = feature_lifts(frame, metadata, feature_group="official_metadata")
    modifier_lifts = feature_lifts(frame, modifiers, feature_group="nonpressure_modifier")
    text_lifts = feature_lifts(frame, tuple(feature for feature in TEXT_FLAG_FEATURES if feature in frame.columns), feature_group="official_text")
    pair_interactions = diagnostic_pair_interactions(frame, metadata, modifiers)
    scoreboard, predictions = run_bucket_experts(
        frame,
        metadata,
        metadata_lifts,
        metadata_scan,
        modifier_lifts,
        modifier_scan,
        pair_interactions,
    )
    blend_scoreboard, blend_predictions, blend_mapping = run_blend_screen(frame, predictions, scoreboard)
    return write_outputs(
        frame=frame,
        metadata_scan=metadata_scan,
        modifier_scan=modifier_scan,
        metadata_lifts=metadata_lifts,
        modifier_lifts=modifier_lifts,
        text_lifts=text_lifts,
        pair_interactions=pair_interactions,
        scoreboard=scoreboard,
        predictions=predictions,
        blend_scoreboard=blend_scoreboard,
        blend_predictions=blend_predictions,
        blend_mapping=blend_mapping,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 official residual source/text/range dynamics screen.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
