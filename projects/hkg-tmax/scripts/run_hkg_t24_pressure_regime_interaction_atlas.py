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
from scripts.run_hkg_t24_pressure_gradient_experts import (  # noqa: E402
    PRESSURE_PAIR_SPECS,
    add_pressure_gradient_features,
)

FOLDER_NAME = "0027_pressure_regime_interaction_atlas"
MIN_DIAGNOSTIC_ROWS = 500
MIN_CELL_ROWS = 45
MIN_HISTORY = 120
MIN_MATCH_ROWS = 20
TOP_PAIR_EXPERT_PAIRS = 45
TOP_BLEND_EXPERTS = 14


@dataclass(frozen=True)
class PairExpertSpec:
    pressure_feature: str
    modifier_feature: str
    bins: int
    same_source: bool
    phase_conditioned: bool
    shrinkage: float = 80.0
    correction_clip_c: float = 2.5
    min_history: int = MIN_HISTORY
    min_match_rows: int = MIN_MATCH_ROWS


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 140) -> str:
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


def add_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_pressure_gradient_features(frame)
    if "target_date" in out.columns:
        dates = pd.to_datetime(out["target_date"], errors="coerce")
        out["month"] = dates.dt.month
    months = pd.to_numeric(out.get("month"), errors="coerce")
    out["season"] = months.map(lambda value: season_name(int(value)) if np.isfinite(value) else "")
    out["monsoon_phase"] = months.map(lambda value: monsoon_phase(int(value)) if np.isfinite(value) else "")
    out["monsoon_phase_code"] = out["monsoon_phase"].map(
        {"northeast_monsoon": -1.0, "transition": 0.0, "southwest_monsoon": 1.0}
    )

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


def build_analysis_frame() -> pd.DataFrame:
    frame = add_interaction_features(build_official_feature_frame())
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(frame["target_date"], context="pressure-regime interaction frame")
    frame["official_raw"] = pd.to_numeric(frame["forecast_max_c"], errors="coerce")
    frame["actual_minus_official_c"] = pd.to_numeric(frame["target_tmax_c"], errors="coerce") - frame["official_raw"]
    frame["official_abs_error_c"] = frame["actual_minus_official_c"].abs()
    return frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True)


def available_numeric_features(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    min_non_null: int = MIN_DIAGNOSTIC_ROWS,
) -> tuple[str, ...]:
    features: list[str] = []
    for feature in candidates:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if int(values.notna().sum()) >= min_non_null and values.nunique(dropna=True) > 2:
            features.append(feature)
    return tuple(features)


def pressure_features(frame: pd.DataFrame) -> tuple[str, ...]:
    return available_numeric_features(
        frame,
        (
            "isd_pressure_plane_lat_slope_hpa_per_deg",
            "isd_pressure_plane_lon_slope_hpa_per_deg",
            "pressure_plane_slope_magnitude_hpa_per_deg",
            "isd_pressure_tendency_morning_midday_hpa",
            "isd_pressure_mean_hpa_change_1d",
            "isd_pressure_range_hpa",
            "isd_pressure_mean_hpa_roll7_mean",
            *(feature for feature, _, _ in PRESSURE_PAIR_SPECS),
        ),
    )


def modifier_features(frame: pd.DataFrame) -> tuple[str, ...]:
    return available_numeric_features(
        frame,
        (
            "month",
            "monsoon_phase_code",
            "isd_wind_speed_mean_mps",
            "isd_wind_speed_max_mps",
            "isd_wind_vector_speed_mps",
            "isd_onshore_easterly_proxy_mps",
            "isd_northerly_proxy_mps",
            "isd_wind_speed_mean_mps_change_1d",
            "ua_wind_u_1000hpa_mps",
            "ua_wind_v_1000hpa_mps",
            "daily_waglan_island_mean_wind_speed_lag7_roll7",
            "daily_waglan_island_prevailing_wind_direction_lag7_roll7",
            "isd_temp_dewpoint_spread_mean_c",
            "isd_dewpoint_midday_minus_temp_c",
            "isd_dew_point_mean_c_change_1d",
            "daily_hong_kong_observatory_mean_relative_humidity_lag7_roll7",
            "rh_max_pct",
            "rh_min_pct",
            "ua_theta_e_1000_850_mean_k",
            "ua_mse_1000_850_mean_kj_kg",
            "ua_mse_925_850_mean_kj_kg",
            "ua_dewpoint_925hpa_c",
            "igra_thickness_1000_500_m_change_48h",
            "ua_tendency_48h_ua_theta_1000hpa_k",
            "isd_north_south_temp_gradient_c",
            "isd_east_west_temp_gradient_c",
            "isd_graph_laplacian_mode_1",
            "isd_graph_laplacian_mode_3",
            "isd_graph_total_variation_c2",
            "thermal_590960_minus_596730_c",
            "thermal_590870_minus_596730_c",
            "thermal_592930_minus_596730_c",
            "thermal_590960_minus_592780_c",
            "dew_590960_minus_596730_c",
            "dew_590870_minus_592780_c",
            "isd_morning_to_midday_temp_rise_c",
            "forecast_range_c",
            "daily_hong_kong_observatory_mean_cloud_amount_lag7_roll7",
            "daily_hong_kong_observatory_daily_rainfall_lag7_roll7",
        ),
    )


def quantile_bucket(values: pd.Series, bins: int, *, min_rows: int = MIN_CELL_ROWS) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if int(numeric.notna().sum()) < bins * min_rows or numeric.nunique(dropna=True) < bins:
        return pd.Series(["insufficient"] * len(values), index=values.index)
    ranked = numeric.rank(method="first")
    labels = [f"q{index + 1}" for index in range(bins)]
    return pd.qcut(ranked, bins, labels=labels).astype(str).where(numeric.notna(), "missing")


def single_feature_regime_lifts(frame: pd.DataFrame) -> pd.DataFrame:
    features = tuple(dict.fromkeys((*pressure_features(frame), *modifier_features(frame))))
    rows: list[dict[str, object]] = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        buckets = quantile_bucket(values, 5)
        if buckets.eq("insufficient").all():
            continue
        bucket_rows: list[dict[str, object]] = []
        work = frame[["target_date", "actual_minus_official_c", "official_abs_error_c"]].copy()
        work["bucket"] = buckets
        for bucket, group in work.groupby("bucket", observed=True, dropna=False):
            if bucket in {"missing", "insufficient"} or len(group) < MIN_CELL_ROWS:
                continue
            bucket_rows.append(
                {
                    "feature": feature,
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
    return pd.DataFrame(rows).sort_values(["priority", "feature", "bucket"], ascending=[False, True, True]).reset_index(drop=True)


def diagnostic_pair_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pressure in pressure_features(frame):
        p_bucket = quantile_bucket(frame[pressure], 3)
        if p_bucket.eq("insufficient").all():
            continue
        for modifier in modifier_features(frame):
            if modifier == pressure:
                continue
            m_bucket = quantile_bucket(frame[modifier], 3)
            if m_bucket.eq("insufficient").all():
                continue
            work = frame[["target_date", "actual_minus_official_c", "official_abs_error_c"]].copy()
            work["pressure_bucket"] = p_bucket
            work["modifier_bucket"] = m_bucket
            cell_rows: list[dict[str, object]] = []
            for (pressure_bucket, modifier_bucket), group in work.groupby(
                ["pressure_bucket", "modifier_bucket"],
                observed=True,
                dropna=False,
            ):
                if (
                    pressure_bucket in {"missing", "insufficient"}
                    or modifier_bucket in {"missing", "insufficient"}
                    or len(group) < MIN_CELL_ROWS
                ):
                    continue
                cell_rows.append(
                    {
                        "pressure_feature": pressure,
                        "modifier_feature": modifier,
                        "pressure_family": feature_family(pressure),
                        "modifier_family": feature_family(modifier),
                        "pressure_bucket": str(pressure_bucket),
                        "modifier_bucket": str(modifier_bucket),
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
            max_mae = max(maes)
            min_mae = min(maes)
            for row in cell_rows:
                row["residual_spread_across_cells_c"] = max(residuals) - min(residuals)
                row["mae_spread_across_cells_c"] = max_mae - min_mae
                row["min_cell_rows_for_pair"] = min(int(cell["rows"]) for cell in cell_rows)
                row["interaction_priority"] = abs(row["residual_spread_across_cells_c"]) + 0.75 * abs(
                    row["mae_spread_across_cells_c"]
                )
            rows.extend(cell_rows)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["interaction_priority", "mae_c", "pressure_feature", "modifier_feature"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def candidate_id_for_spec(spec: PairExpertSpec) -> str:
    source = "same_source" if spec.same_source else "all_prior"
    phase = "phase" if spec.phase_conditioned else "all_phase"
    return slug(
        f"pressure_interaction_{spec.pressure_feature}__x__{spec.modifier_feature}_q{spec.bins}_{source}_{phase}"
    )


def prior_bucket_indices(
    prior_values: np.ndarray,
    current_value: float,
    bins: int,
    *,
    min_match_rows: int,
) -> tuple[np.ndarray, int] | None:
    valid = prior_values[np.isfinite(prior_values)]
    if len(valid) < bins * min_match_rows or len(np.unique(valid)) < bins:
        return None
    edges = np.unique(np.nanquantile(valid, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    if len(edges) < bins - 1:
        return None
    return np.searchsorted(edges, prior_values, side="right"), int(np.searchsorted(edges, current_value, side="right"))


def pair_expert_predictions(frame: pd.DataFrame, spec: PairExpertSpec) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    forecast = pd.to_numeric(ordered["forecast_max_c"], errors="coerce").to_numpy(dtype=float)
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    residual = target - forecast
    pressure_values = pd.to_numeric(ordered[spec.pressure_feature], errors="coerce").to_numpy(dtype=float)
    modifier_values = pd.to_numeric(ordered[spec.modifier_feature], errors="coerce").to_numpy(dtype=float)
    sources = ordered["forecast_source_family"].astype(str).to_numpy()
    phases = ordered["monsoon_phase"].astype(str).to_numpy()

    predictions: list[float] = []
    corrections: list[float] = []
    rows_used: list[int] = []
    for index, target_date in enumerate(dates):
        if not np.isfinite(forecast[index]) or not np.isfinite(pressure_values[index]) or not np.isfinite(modifier_values[index]):
            predictions.append(float(forecast[index]) if np.isfinite(forecast[index]) else math.nan)
            corrections.append(0.0)
            rows_used.append(0)
            continue
        prior_mask = np.arange(len(ordered)) < int(np.searchsorted(dates, target_date, side="left"))
        if spec.same_source:
            prior_mask &= sources == sources[index]
        if spec.phase_conditioned:
            prior_mask &= phases == phases[index]
        prior_mask &= np.isfinite(residual) & np.isfinite(pressure_values) & np.isfinite(modifier_values)
        prior_index = np.flatnonzero(prior_mask)
        if len(prior_index) < spec.min_history:
            predictions.append(float(forecast[index]))
            corrections.append(0.0)
            rows_used.append(0)
            continue

        pressure_bucketed = prior_bucket_indices(
            pressure_values[prior_index],
            pressure_values[index],
            spec.bins,
            min_match_rows=spec.min_match_rows,
        )
        modifier_bucketed = prior_bucket_indices(
            modifier_values[prior_index],
            modifier_values[index],
            spec.bins,
            min_match_rows=spec.min_match_rows,
        )
        if pressure_bucketed is None or modifier_bucketed is None:
            predictions.append(float(forecast[index]))
            corrections.append(0.0)
            rows_used.append(0)
            continue
        pressure_prior_buckets, pressure_current_bucket = pressure_bucketed
        modifier_prior_buckets, modifier_current_bucket = modifier_bucketed
        matched = prior_index[
            (pressure_prior_buckets == pressure_current_bucket) & (modifier_prior_buckets == modifier_current_bucket)
        ]
        if len(matched) < spec.min_match_rows:
            predictions.append(float(forecast[index]))
            corrections.append(0.0)
            rows_used.append(0)
            continue
        raw_correction = float(np.nanmean(residual[matched]))
        weight = len(matched) / (len(matched) + float(spec.shrinkage))
        correction = float(np.clip(raw_correction * weight, -spec.correction_clip_c, spec.correction_clip_c))
        predictions.append(float(forecast[index] + correction))
        corrections.append(correction)
        rows_used.append(int(len(matched)))

    out = ordered[["target_date", "forecast_source_family", "target_tmax_c", "forecast_max_c"]].copy()
    out["official_raw"] = forecast
    out["candidate_prediction_c"] = predictions
    out["residual_correction_c"] = corrections
    out["past_rows_used"] = rows_used
    out["pressure_feature"] = spec.pressure_feature
    out["modifier_feature"] = spec.modifier_feature
    return out


def build_pair_specs(pair_interactions: pd.DataFrame) -> list[PairExpertSpec]:
    if pair_interactions.empty:
        return []
    pair_table = (
        pair_interactions.sort_values("interaction_priority", ascending=False)
        .drop_duplicates(["pressure_feature", "modifier_feature"])
        .head(TOP_PAIR_EXPERT_PAIRS)
    )
    specs: list[PairExpertSpec] = []
    for row in pair_table.itertuples(index=False):
        for bins in (3, 5):
            for same_source in (False, True):
                for phase_conditioned in (False, True):
                    specs.append(
                        PairExpertSpec(
                            pressure_feature=str(row.pressure_feature),
                            modifier_feature=str(row.modifier_feature),
                            bins=bins,
                            same_source=same_source,
                            phase_conditioned=phase_conditioned,
                        )
                    )
    return specs


def score_pair_candidate(predictions: pd.DataFrame, spec: PairExpertSpec, candidate_id: str) -> dict[str, object]:
    candidate = score_prediction_frame(predictions.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "official_prediction"}), "official_prediction")
    corrected = predictions["past_rows_used"] > 0
    return {
        "candidate_id": candidate_id,
        "pressure_feature": spec.pressure_feature,
        "modifier_feature": spec.modifier_feature,
        "pressure_family": feature_family(spec.pressure_feature),
        "modifier_family": feature_family(spec.modifier_feature),
        "bins": spec.bins,
        "same_source": spec.same_source,
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


def run_pair_experts(frame: pd.DataFrame, pair_interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = build_pair_specs(pair_interactions)
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs:
        predictions = pair_expert_predictions(frame, spec)
        candidate_id = candidate_id_for_spec(spec)
        predictions["candidate_id"] = candidate_id
        score_rows.append(score_pair_candidate(predictions, spec, candidate_id))
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(score_rows)
    if not scoreboard.empty:
        scoreboard = scoreboard.sort_values(["delta_vs_official_same_rows", "mae"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return scoreboard, predictions


def build_blend_frame(frame: pd.DataFrame, predictions: pd.DataFrame, scoreboard: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    official = frame[["target_date", "forecast_source_family", "target_tmax_c", "forecast_max_c"]].copy()
    official["official_raw"] = pd.to_numeric(official["forecast_max_c"], errors="coerce")
    if scoreboard.empty or predictions.empty:
        return official, pd.DataFrame()
    top_ids = scoreboard.head(TOP_BLEND_EXPERTS)["candidate_id"].to_list()
    mapping = scoreboard[scoreboard["candidate_id"].isin(top_ids)][
        [
            "candidate_id",
            "pressure_feature",
            "modifier_feature",
            "bins",
            "same_source",
            "phase_conditioned",
            "mae",
            "delta_vs_official_same_rows",
        ]
    ].copy()
    mapping["expert_id"] = [
        f"px_{rank:02d}_{slug(row.candidate_id, limit=42)}"
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
    experts = ["official_raw", *[column for column in blend_frame.columns if column.startswith("px_")]]
    score_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for mode in ("best", "inverse_mae"):
        for same_source in (False, True):
            candidate_id = f"pressure_interaction_blend_{mode}_{'same_source' if same_source else 'all_prior'}"
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
    single_lifts: pd.DataFrame,
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

    write_csv(artifacts / "single_feature_regime_lifts.csv", single_lifts)
    write_csv(artifacts / "diagnostic_pair_interactions.csv", pair_interactions)
    write_csv(artifacts / "pair_expert_scoreboard.csv", scoreboard)
    top_ids = set(scoreboard.head(40)["candidate_id"].to_list()) if not scoreboard.empty else set()
    write_csv(artifacts / "top_pair_expert_predictions.csv", predictions[predictions["candidate_id"].isin(top_ids)].copy())
    write_csv(artifacts / "blend_scoreboard.csv", blend_scoreboard)
    write_csv(artifacts / "blend_predictions.csv", blend_predictions)
    write_csv(artifacts / "blend_mapping.csv", blend_mapping)

    best_pair = scoreboard.iloc[0] if not scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "pressure_features": int(len(pressure_features(frame))),
        "modifier_features": int(len(modifier_features(frame))),
        "single_feature_lift_rows": int(len(single_lifts)),
        "diagnostic_pair_interaction_rows": int(len(pair_interactions)),
        "pair_expert_candidates": int(len(scoreboard)),
        "blend_candidates": int(len(blend_scoreboard)),
        "best_pair_candidate": "" if best_pair is None else str(best_pair["candidate_id"]),
        "best_pair_mae": None if best_pair is None else float(best_pair["mae"]),
        "best_pair_rmse": None if best_pair is None else float(best_pair["rmse"]),
        "best_pair_delta_vs_official": None if best_pair is None else float(best_pair["delta_vs_official_same_rows"]),
        "best_blend_candidate": "" if best_blend is None else str(best_blend["candidate_id"]),
        "best_blend_mae": None if best_blend is None else float(best_blend["mae"]),
        "best_blend_rmse": None if best_blend is None else float(best_blend["rmse"]),
        "best_blend_delta_vs_official": None if best_blend is None else float(best_blend["delta_vs_official_same_rows"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "pressure_regime_interaction_atlas_manifest.json", manifest)

    best_pair_text = "No pressure-regime pair expert was scoreable."
    if best_pair is not None:
        best_pair_text = (
            f"Best pair expert: `{best_pair['candidate_id']}` with MAE `{best_pair['mae']:.4f}`, "
            f"RMSE `{best_pair['rmse']:.4f}`, and official delta "
            f"`{best_pair['delta_vs_official_same_rows']:.4f}`."
        )
    best_blend_text = "No pressure-regime blend was scoreable."
    if best_blend is not None:
        best_blend_text = (
            f"Best pair-expert blend: `{best_blend['candidate_id']}` with MAE `{best_blend['mae']:.4f}`, "
            f"RMSE `{best_blend['rmse']:.4f}`, and official delta "
            f"`{best_blend['delta_vs_official_same_rows']:.4f}`."
        )

    readme = f"""# Pressure-Regime Interaction Atlas

Generated: `{manifest['generated_at_utc']}`

## What Was Tested

This insight tests whether pressure-gradient value is conditional on other physical regimes. It crosses pressure-plane slopes and station pressure spreads with wind, humidity/dew spread, upper-air heat content, station-network thermal gradients, cloud/rain memory, and monsoon phase.

There are two evidence layers:

1. Diagnostic interaction tables: full-sample bucket summaries that identify where official forecast residuals are large. These are mechanism discovery only.
2. Deployable-style pair experts: for each target date, the correction is estimated only from strictly earlier target dates in the matching pressure/modifier bucket.

## Leakage Contract

- All rows are earlier than `{CONFIRMATION_START.date()}`.
- The official forecast selection comes from the existing pre-cutoff official-feature frame.
- Diagnostic tables are not promoted as deployable forecasts.
- Pair-expert quantile edges are recomputed from prior rows only.
- Pair-expert residual corrections use strictly earlier target dates via `searchsorted(..., side="left")`.
- Same-source candidates restrict history to the same official source family.
- Phase-conditioned candidates restrict history to the same monsoon phase, again using only prior rows.
- Blend weights/selection use only prior realized errors.

## Main Results

{best_pair_text}

{best_blend_text}

## Top Diagnostic Pair Interactions

{markdown_table(pair_interactions.head(20), max_rows=20)}

## Top Pair Experts

{markdown_table(scoreboard.head(20), max_rows=20)}

## Blend Scoreboard

{markdown_table(blend_scoreboard, max_rows=20)}

## Interpretation

This screen is designed to test whether the pressure-gradient signal from `0025` and the pressure ridge screen from `0026` become more useful when gated by physical regimes. A strong diagnostic interaction but weak past-only pair expert means the regime is real but not yet stable enough in the current non-contiguous forecast archive. A strong past-only candidate can be promoted into the later expert stack only after broader OOF and ablation checks.
"""
    write_text(folder / "README.md", readme)
    update_master_index(manifest)
    return manifest


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Pressure-Regime Interaction Atlas\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_pressure_regime_interaction_atlas.py`:

- `{FOLDER_NAME}`: pressure-gradient interactions crossed with wind, humidity, upper-air, station-gradient, cloud/rain, and monsoon-phase context.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Pressure features | {manifest['pressure_features']} |
| Modifier features | {manifest['modifier_features']} |
| Diagnostic pair-interaction rows | {manifest['diagnostic_pair_interaction_rows']} |
| Pair expert candidates | {manifest['pair_expert_candidates']} |
| Best pair MAE | {manifest['best_pair_mae']} |
| Best pair delta vs official | {manifest['best_pair_delta_vs_official']} |
| Best blend MAE | {manifest['best_blend_mae']} |
| Best blend RMSE | {manifest['best_blend_rmse']} |
| Best blend delta vs official | {manifest['best_blend_delta_vs_official']} |

Leakage contract: all scored rows are before `{CONFIRMATION_START.date()}`; deployable pair corrections and blend weights use strictly prior target dates.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    frame = build_analysis_frame()
    single_lifts = single_feature_regime_lifts(frame)
    pair_interactions = diagnostic_pair_interactions(frame)
    scoreboard, predictions = run_pair_experts(frame, pair_interactions)
    blend_scoreboard, blend_predictions, blend_mapping = run_blend_screen(frame, predictions, scoreboard)
    return write_outputs(
        frame=frame,
        single_lifts=single_lifts,
        pair_interactions=pair_interactions,
        scoreboard=scoreboard,
        predictions=predictions,
        blend_scoreboard=blend_scoreboard,
        blend_predictions=blend_predictions,
        blend_mapping=blend_mapping,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 pressure-regime interaction atlas.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
