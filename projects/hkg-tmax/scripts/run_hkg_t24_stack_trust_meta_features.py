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
from scripts.run_hkg_t24_forecast_revision_momentum_deep_dive import (  # noqa: E402
    add_revision_features,
)
from scripts.run_hkg_t24_residual_failure_cluster_discovery import build_failure_frame  # noqa: E402

FOLDER_NAME = "0037_stack_trust_meta_features"
STACK_0036_DIR = RESEARCH_ROOT / "0036_revision_centroid_stack_ablation" / "artifacts"
MIN_GLOBAL_HISTORY = 160
MIN_BUCKET_HISTORY = 45
TRUST_FAMILIES = ("official_raw", "family_0033_smooth", "family_0034_centroid", "family_0035_revision")


@dataclass(frozen=True)
class NumericBucketSpec:
    feature_name: str
    source_col: str
    thresholds: tuple[float, ...]


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def bucket_numeric(values: pd.Series, thresholds: tuple[float, ...]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    labels: list[str] = []
    for value in numeric:
        if not np.isfinite(value):
            labels.append("missing")
            continue
        placed = False
        lower = -math.inf
        for threshold in thresholds:
            if value <= threshold:
                if math.isinf(lower):
                    labels.append(f"<= {threshold:g}")
                else:
                    labels.append(f"({lower:g}, {threshold:g}]")
                placed = True
                break
            lower = threshold
        if not placed:
            labels.append(f"> {thresholds[-1]:g}")
    return pd.Series(labels, index=values.index, dtype="object")


def bucket_binary(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.where(numeric.isna(), "missing", np.where(numeric >= 0.5, "yes", "no")),
        index=values.index,
        dtype="object",
    )


def season_label(month: object) -> str:
    try:
        value = int(month)
    except (TypeError, ValueError):
        return "missing"
    if value in (12, 1, 2):
        return "DJF"
    if value in (3, 4, 5):
        return "MAM"
    if value in (6, 7, 8):
        return "JJA"
    if value in (9, 10, 11):
        return "SON"
    return "missing"


def numeric_bucket_specs() -> tuple[NumericBucketSpec, ...]:
    return (
        NumericBucketSpec("forecast_jump_bin", "forecast_max_change_1_source_c", (-1.0, 0.0, 1.0, 2.0)),
        NumericBucketSpec("forecast_vs_prior7_bin", "forecast_max_vs_prior7_mean_source_c", (-2.0, -1.0, 1.0, 2.0)),
        NumericBucketSpec("forecast_range_bin", "forecast_range_c", (3.0, 4.0, 5.0, 6.0)),
        NumericBucketSpec("forecast_range_change_bin", "forecast_range_change_1_source_c", (-1.0, 0.0, 1.0)),
        NumericBucketSpec("pressure_change_bin", "isd_pressure_mean_hpa_change_1d", (-2.0, 0.0, 2.0)),
        NumericBucketSpec("pressure_slope_bin", "pressure_plane_slope_magnitude_hpa_per_deg", (0.5, 1.0, 1.5)),
        NumericBucketSpec("wind_speed_bin", "isd_wind_speed_mean_mps", (2.5, 4.0, 5.5)),
        NumericBucketSpec("station_gradient_bin", "abs_north_south_temp_gradient_c", (1.0, 2.0, 3.0)),
        NumericBucketSpec("dew_spread_bin", "isd_temp_dewpoint_spread_mean_c", (2.0, 4.0, 6.0)),
        NumericBucketSpec("midday_rise_bin", "isd_morning_to_midday_temp_rise_c", (1.0, 3.0, 5.0)),
        NumericBucketSpec("prior_bias_bin", "prediction_0018_c_prior90_source_residual_mean_c", (-0.2, 0.0, 0.2)),
    )


def add_meta_feature_bins(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    rows: list[dict[str, object]] = []
    out["meta_source_family"] = out["forecast_source_family"].astype(str)
    rows.append({"meta_feature": "meta_source_family", "type": "categorical", "source_col": "forecast_source_family"})
    out["meta_month"] = pd.to_numeric(out.get("month"), errors="coerce").map(lambda value: f"{int(value):02d}" if pd.notna(value) else "missing")
    rows.append({"meta_feature": "meta_month", "type": "categorical", "source_col": "month"})
    out["meta_season"] = out.get("month", pd.Series(index=out.index)).map(season_label)
    rows.append({"meta_feature": "meta_season", "type": "categorical", "source_col": "month"})
    if "monsoon_phase" in out.columns:
        out["meta_monsoon_phase"] = out["monsoon_phase"].fillna("missing").astype(str)
        rows.append({"meta_feature": "meta_monsoon_phase", "type": "categorical", "source_col": "monsoon_phase"})

    for spec in numeric_bucket_specs():
        if spec.source_col not in out.columns:
            continue
        meta_col = f"meta_{spec.feature_name}"
        out[meta_col] = bucket_numeric(out[spec.source_col], spec.thresholds)
        rows.append(
            {
                "meta_feature": meta_col,
                "type": "numeric_bucket",
                "source_col": spec.source_col,
                "thresholds": ",".join(str(value) for value in spec.thresholds),
            }
        )

    for col in (
        "text_hot",
        "text_cloud",
        "text_any_rain",
        "text_sunny_or_fine",
        "text_hot_turned_on_source",
        "text_cloud_turned_on_source",
        "text_any_rain_turned_on_source",
    ):
        if col not in out.columns:
            continue
        meta_col = f"meta_{col}"
        out[meta_col] = bucket_binary(out[col])
        rows.append({"meta_feature": meta_col, "type": "binary", "source_col": col})

    catalog = pd.DataFrame(rows)
    return out, catalog


def selected_0036_family_candidate_ids() -> dict[str, str]:
    ablation = pd.read_csv(STACK_0036_DIR / "ablation_summary.csv")
    lookup = dict(zip(ablation["ablation"].astype(str), ablation["candidate_id"].astype(str), strict=False))
    scoreboard = pd.read_csv(STACK_0036_DIR / "stack_scoreboard.csv")
    return {
        "family_0033_smooth": lookup["0033_smooth_top"],
        "family_0034_centroid": lookup["0034_centroid_top"],
        "family_0035_revision": lookup["0035_revision_top"],
        "current_0036_stack": str(scoreboard.iloc[0]["candidate_id"]),
    }


def load_family_prediction_frame() -> pd.DataFrame:
    candidate_ids = selected_0036_family_candidate_ids()
    needed = set(candidate_ids.values())
    predictions = pd.read_csv(
        STACK_0036_DIR / "stack_predictions.csv",
        usecols=[
            "target_date",
            "forecast_source_family",
            "target_tmax_c",
            "official_raw",
            "candidate_id",
            "expert_prediction_c",
        ],
    )
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(predictions["target_date"], context="0037 family predictions")
    base = (
        predictions[["target_date", "forecast_source_family", "target_tmax_c", "official_raw"]]
        .drop_duplicates(["target_date", "forecast_source_family"], keep="last")
        .sort_values(["target_date", "forecast_source_family"])
        .reset_index(drop=True)
    )
    selected = predictions[predictions["candidate_id"].isin(needed)][
        ["target_date", "forecast_source_family", "candidate_id", "expert_prediction_c"]
    ].copy()
    wide = (
        selected.pivot_table(
            index=["target_date", "forecast_source_family"],
            columns="candidate_id",
            values="expert_prediction_c",
            aggfunc="last",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    inverse = {candidate_id: family for family, candidate_id in candidate_ids.items()}
    wide = wide.rename(columns=inverse)
    return base.merge(wide, on=["target_date", "forecast_source_family"], how="left")


def load_meta_context() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, _prior_systems = build_failure_frame()
    frame = add_revision_features(frame)
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(frame["target_date"], context="0037 meta context")
    frame, _catalog = add_meta_feature_bins(frame)
    meta_cols = ["target_date", "forecast_source_family", *[col for col in frame.columns if col.startswith("meta_")]]
    return frame[meta_cols].drop_duplicates(["target_date", "forecast_source_family"], keep="last"), _catalog


def build_trust_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = load_family_prediction_frame()
    meta, catalog = load_meta_context()
    frame = predictions.merge(meta, on=["target_date", "forecast_source_family"], how="left")
    for col in TRUST_FAMILIES:
        if col not in frame.columns:
            raise ValueError(f"Missing trust family prediction column: {col}")
    frame = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True)
    return frame, catalog


def family_scoreboard(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in (*TRUST_FAMILIES, "current_0036_stack"):
        if family not in frame.columns:
            continue
        score = score_prediction_frame(frame.rename(columns={family: "prediction"}), "prediction")
        official = score_prediction_frame(frame.rename(columns={"official_raw": "prediction"}), "prediction")
        rows.append(
            {
                "family": family,
                **score,
                "delta_vs_official": float(score["mae"] - official["mae"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)


def trust_atlas(frame: pd.DataFrame, meta_features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    families = list(TRUST_FAMILIES)
    if "current_0036_stack" in frame.columns:
        families.append("current_0036_stack")
    for feature in meta_features:
        if feature not in frame.columns:
            continue
        for bucket, subset in frame.groupby(feature, observed=True, dropna=False):
            if len(subset) < 30:
                continue
            row: dict[str, object] = {
                "meta_feature": feature,
                "bucket": str(bucket),
                "rows": int(len(subset)),
                "first_date": str(pd.to_datetime(subset["target_date"]).min().date()),
                "last_date": str(pd.to_datetime(subset["target_date"]).max().date()),
            }
            maes: dict[str, float] = {}
            for family in families:
                error = pd.to_numeric(subset[family], errors="coerce") - pd.to_numeric(subset["target_tmax_c"], errors="coerce")
                mae = float(error.abs().mean())
                maes[family] = mae
                row[f"{family}_mae"] = mae
            best_family = min(maes, key=maes.get)
            row["best_family"] = best_family
            row["best_family_mae"] = maes[best_family]
            row["best_vs_official_delta"] = maes[best_family] - maes["official_raw"]
            if "current_0036_stack" in maes:
                row["best_vs_current_0036_delta"] = maes[best_family] - maes["current_0036_stack"]
            rows.append(row)
    atlas = pd.DataFrame(rows)
    if not atlas.empty:
        atlas = atlas.sort_values(["best_vs_official_delta", "rows"], ascending=[True, False]).reset_index(drop=True)
    return atlas


def trust_feature_summary(atlas: pd.DataFrame) -> pd.DataFrame:
    if atlas.empty:
        return pd.DataFrame()
    summary = (
        atlas.groupby("meta_feature", observed=True)
        .agg(
            buckets=("bucket", "count"),
            rows=("rows", "sum"),
            mean_best_delta_vs_official=("best_vs_official_delta", "mean"),
            best_bucket_delta_vs_official=("best_vs_official_delta", "min"),
            median_best_family_mae=("best_family_mae", "median"),
        )
        .reset_index()
        .sort_values(["mean_best_delta_vs_official", "best_bucket_delta_vs_official"])
    )
    wins = (
        atlas.groupby(["meta_feature", "best_family"], observed=True)
        .size()
        .rename("win_buckets")
        .reset_index()
        .sort_values(["meta_feature", "win_buckets"], ascending=[True, False])
    )
    win_text = (
        wins.groupby("meta_feature", observed=True)
        .apply(lambda group: "; ".join(f"{row.best_family}:{int(row.win_buckets)}" for row in group.itertuples(index=False)), include_groups=False)
        .rename("best_family_bucket_wins")
        .reset_index()
    )
    return summary.merge(win_text, on="meta_feature", how="left")


def prior_mae_for_mask(values: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    valid = mask & np.isfinite(values) & np.isfinite(target)
    count = int(valid.sum())
    if count == 0:
        return 0, math.nan
    return count, float(np.abs(values[valid] - target[valid]).mean())


def family_prior_estimates(
    *,
    family_values: dict[str, np.ndarray],
    target: np.ndarray,
    base_prior: np.ndarray,
    feature_arrays: dict[str, np.ndarray],
    feature_names: tuple[str, ...],
    row_index: int,
    min_bucket_history: int,
    min_global_history: int,
) -> dict[str, tuple[int, float]]:
    estimates: dict[str, tuple[int, float]] = {}
    for family, values in family_values.items():
        weighted_sum = 0.0
        weight_sum = 0.0
        total_count = 0
        global_count, global_mae = prior_mae_for_mask(values, target, base_prior)
        if global_count >= min_global_history and np.isfinite(global_mae):
            weight = 0.5 * math.sqrt(global_count)
            weighted_sum += weight * global_mae
            weight_sum += weight
            total_count += global_count
        for feature in feature_names:
            current_bucket = str(feature_arrays[feature][row_index])
            if current_bucket == "missing":
                continue
            mask = base_prior & (feature_arrays[feature] == current_bucket)
            count, mae = prior_mae_for_mask(values, target, mask)
            if count >= min_bucket_history and np.isfinite(mae):
                weight = math.sqrt(count)
                weighted_sum += weight * mae
                weight_sum += weight
                total_count += count
        if weight_sum > 0:
            estimates[family] = (total_count, weighted_sum / weight_sum)
    return estimates


def past_only_meta_trust_predictions(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    mode: str,
    same_source: bool,
    min_bucket_history: int = MIN_BUCKET_HISTORY,
    min_global_history: int = MIN_GLOBAL_HISTORY,
) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").dt.normalize().to_numpy(dtype="datetime64[ns]")
    sources = ordered["forecast_source_family"].astype(str).to_numpy()
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    family_values = {
        family: pd.to_numeric(ordered[family], errors="coerce").to_numpy(dtype=float)
        for family in TRUST_FAMILIES
    }
    feature_arrays = {feature: ordered[feature].fillna("missing").astype(str).to_numpy() for feature in feature_names}

    predictions: list[float] = []
    selected: list[str] = []
    eligible_counts: list[int] = []
    official_prior_maes: list[float] = []
    selected_prior_maes: list[float] = []
    selected_prior_counts: list[int] = []

    for index, target_date in enumerate(dates):
        base_prior = dates < target_date
        if same_source:
            base_prior &= sources == sources[index]
        estimates = family_prior_estimates(
            family_values=family_values,
            target=target,
            base_prior=base_prior,
            feature_arrays=feature_arrays,
            feature_names=feature_names,
            row_index=index,
            min_bucket_history=min_bucket_history,
            min_global_history=min_global_history,
        )
        official_count, official_mae = estimates.get("official_raw", (0, math.nan))
        scored = [
            (family, count, mae)
            for family, (count, mae) in estimates.items()
            if np.isfinite(family_values[family][index]) and count >= min_bucket_history and np.isfinite(mae)
        ]
        if not scored:
            predictions.append(float(family_values["official_raw"][index]))
            selected.append("official_raw_fallback")
            eligible_counts.append(0)
            official_prior_maes.append(official_mae)
            selected_prior_maes.append(math.nan)
            selected_prior_counts.append(0)
            continue
        scored = sorted(scored, key=lambda item: (item[2], item[0]))
        eligible_counts.append(len(scored))
        official_prior_maes.append(official_mae)
        if mode == "best":
            chosen = scored[0]
            predictions.append(float(family_values[chosen[0]][index]))
            selected.append(chosen[0])
            selected_prior_maes.append(chosen[2])
            selected_prior_counts.append(chosen[1])
        elif mode == "inverse_mae":
            weights = np.array([1.0 / max(item[2], 1e-6) for item in scored], dtype=float)
            values = np.array([float(family_values[item[0]][index]) for item in scored], dtype=float)
            weights = weights / weights.sum()
            predictions.append(float(np.dot(weights, values)))
            selected.append("inverse_mae_family_blend")
            selected_prior_maes.append(scored[0][2])
            selected_prior_counts.append(scored[0][1])
        elif mode == "positive_lift":
            lifted = []
            if np.isfinite(official_mae):
                lifted = [
                    (family, count, mae, official_mae - mae)
                    for family, count, mae in scored
                    if family != "official_raw" and mae < official_mae
                ]
            if not lifted:
                predictions.append(float(family_values["official_raw"][index]))
                selected.append("official_raw_fallback")
                selected_prior_maes.append(math.nan)
                selected_prior_counts.append(0)
            else:
                lifted = sorted(lifted, key=lambda item: (-item[3], item[2], item[0]))
                weights = np.array([max(item[3], 1e-9) for item in lifted], dtype=float)
                values = np.array([float(family_values[item[0]][index]) for item in lifted], dtype=float)
                weights = weights / weights.sum()
                predictions.append(float(np.dot(weights, values)))
                selected.append("positive_lift_family_blend")
                selected_prior_maes.append(min(item[2] for item in lifted))
                selected_prior_counts.append(max(item[1] for item in lifted))
        else:
            raise ValueError(f"Unknown meta trust mode: {mode}")

    out = ordered[["target_date", "forecast_source_family", "target_tmax_c", "official_raw"]].copy()
    out["expert_prediction_c"] = predictions
    out["selected_family"] = selected
    out["eligible_family_count"] = eligible_counts
    out["official_prior_mae"] = official_prior_maes
    out["selected_prior_mae"] = selected_prior_maes
    out["selected_prior_count"] = selected_prior_counts
    out["meta_features"] = "+".join(feature_names)
    out["mode"] = mode
    out["same_source"] = same_source
    return out


def feature_sets(meta_features: list[str]) -> dict[str, tuple[str, ...]]:
    available = set(meta_features)
    sets: dict[str, tuple[str, ...]] = {feature.replace("meta_", ""): (feature,) for feature in meta_features}
    composites = {
        "season_source": ("meta_source_family", "meta_season", "meta_month"),
        "forecast_revision": (
            "meta_source_family",
            "meta_forecast_jump_bin",
            "meta_forecast_vs_prior7_bin",
            "meta_forecast_range_bin",
            "meta_forecast_range_change_bin",
            "meta_prior_bias_bin",
        ),
        "pressure_wind_gradient": (
            "meta_pressure_change_bin",
            "meta_pressure_slope_bin",
            "meta_wind_speed_bin",
            "meta_station_gradient_bin",
        ),
        "text_weather": (
            "meta_text_hot",
            "meta_text_cloud",
            "meta_text_any_rain",
            "meta_text_sunny_or_fine",
            "meta_text_hot_turned_on_source",
            "meta_text_cloud_turned_on_source",
        ),
        "thermal_moisture": ("meta_dew_spread_bin", "meta_midday_rise_bin", "meta_station_gradient_bin"),
        "all_core": (
            "meta_source_family",
            "meta_season",
            "meta_forecast_jump_bin",
            "meta_forecast_vs_prior7_bin",
            "meta_forecast_range_bin",
            "meta_pressure_change_bin",
            "meta_station_gradient_bin",
            "meta_dew_spread_bin",
            "meta_text_hot",
            "meta_text_cloud",
            "meta_text_any_rain",
        ),
    }
    for name, cols in composites.items():
        filtered = tuple(col for col in cols if col in available)
        if len(filtered) >= 2:
            sets[name] = filtered
    return sets


def score_trust_candidate(
    predictions: pd.DataFrame,
    *,
    candidate_id: str,
    feature_set: str,
    mode: str,
    same_source: bool,
    feature_count: int,
) -> dict[str, object]:
    candidate = score_prediction_frame(predictions.rename(columns={"expert_prediction_c": "prediction"}), "prediction")
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "prediction"}), "prediction")
    selected = predictions["selected_family"].astype(str)
    return {
        "candidate_id": candidate_id,
        "feature_set": feature_set,
        "mode": mode,
        "same_source": same_source,
        "feature_count": feature_count,
        **candidate,
        "official_same_rows_mae": official["mae"],
        "official_same_rows_rmse": official["rmse"],
        "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
        "fallback_rows": int(selected.eq("official_raw_fallback").sum()),
        "official_selected_rows": int(selected.isin(["official_raw", "official_raw_fallback"]).sum()),
        "mean_eligible_families": float(predictions["eligible_family_count"].mean()),
    }


def run_trust_screen(frame: pd.DataFrame, meta_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    set_rows: list[dict[str, object]] = []
    for name, features in feature_sets(meta_features).items():
        set_rows.append({"feature_set": name, "features": ",".join(features), "feature_count": len(features)})
        for mode in ("best", "inverse_mae", "positive_lift"):
            for same_source in (False, True):
                candidate_id = f"trust_{slug(name)}_{mode}_{'same_source' if same_source else 'all_prior'}"
                predictions = past_only_meta_trust_predictions(
                    frame,
                    feature_names=features,
                    mode=mode,
                    same_source=same_source,
                )
                predictions["candidate_id"] = candidate_id
                predictions["feature_set"] = name
                score_rows.append(
                    score_trust_candidate(
                        predictions,
                        candidate_id=candidate_id,
                        feature_set=name,
                        mode=mode,
                        same_source=same_source,
                        feature_count=len(features),
                    )
                )
                prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(score_rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return scoreboard, predictions, pd.DataFrame(set_rows)


def trust_selection_counts(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    return (
        predictions.groupby(["candidate_id", "feature_set", "selected_family"], observed=True, dropna=False)
        .agg(rows=("target_date", "count"))
        .reset_index()
        .sort_values(["candidate_id", "rows"], ascending=[True, False])
    )


def baseline_comparison(scoreboard: pd.DataFrame, family_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prior_path = STACK_0036_DIR / "baseline_comparison.csv"
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        for row in prior.itertuples(index=False):
            rows.append(
                {
                    "system": str(row.system),
                    "candidate_id": str(row.candidate_id),
                    "mae": float(row.mae),
                    "rmse": float(row.rmse),
                    "delta_vs_official": float(row.delta_vs_official),
                    "n": int(row.n),
                    "first_date": str(row.first_date),
                    "last_date": str(row.last_date),
                }
            )
    for row in family_scores.itertuples(index=False):
        rows.append(
            {
                "system": f"0037_family_{row.family}",
                "candidate_id": str(row.family),
                "mae": float(row.mae),
                "rmse": float(row.rmse),
                "delta_vs_official": float(row.delta_vs_official),
                "n": int(row.n),
                "first_date": str(row.first_date),
                "last_date": str(row.last_date),
            }
        )
    if not scoreboard.empty:
        best = scoreboard.iloc[0]
        rows.append(
            {
                "system": "0037_best_stack_trust_meta",
                "candidate_id": str(best["candidate_id"]),
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "delta_vs_official": float(best["delta_vs_official_same_rows"]),
                "n": int(best["n"]),
                "first_date": str(best["first_date"]),
                "last_date": str(best["last_date"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["mae", "rmse"]).drop_duplicates("system", keep="first").reset_index(drop=True)


def write_readme(
    *,
    folder: Path,
    manifest: dict[str, object],
    family_scores: pd.DataFrame,
    atlas: pd.DataFrame,
    feature_summary: pd.DataFrame,
    scoreboard: pd.DataFrame,
    selection_counts: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best = scoreboard.iloc[0] if not scoreboard.empty else None
    best_text = "No scoreable meta-trust selector was produced."
    if best is not None:
        best_text = (
            f"Best meta-trust selector: `{best['candidate_id']}` with MAE `{best['mae']:.4f}`, "
            f"RMSE `{best['rmse']:.4f}`, and official delta "
            f"`{best['delta_vs_official_same_rows']:.4f}`."
        )
    readme = f"""# Stack Trust Meta-Features

Generated: `{manifest['generated_at_utc']}`

## Purpose

`0036` proved that 0033 smooth forecast-jump specialists and 0034 centroid specialists have complementary MAE signal. This insight asks a narrower question: under which pre-cutoff meta-feature regimes should the system trust official raw, the 0033 family, the 0034 family, or the 0035 revision family?

## Data Window

Rows used: `{manifest['official_rows']}` official forecast/target rows.

Date range: `{manifest['first_target_date']}` to `{manifest['last_target_date']}`.

Source counts: `{manifest['source_counts']}`.

## Leakage Contract

- All scored rows are earlier than `{CONFIRMATION_START.date()}`.
- Trust diagnostics are post-hoc explanatory tables; deployable selectors are scored separately.
- Deployable meta-trust selectors use only rows with `target_date < current target_date`.
- Same-source variants additionally restrict prior trust evidence to the current source family.
- Same-date rows from another source family are excluded from prior trust estimates.
- Meta-feature bins use fixed deterministic thresholds or categorical labels, not target-derived cutpoints.
- 2024+ confirmation rows are not loaded or scored.

## Main Result

{best_text}

## Baseline Comparison

{markdown_table(comparison, max_rows=25)}

## Family Prediction Scoreboard

{markdown_table(family_scores, max_rows=20)}

## Trust Feature Summary

{markdown_table(feature_summary.head(30), max_rows=30)}

## Strongest Trust Atlas Cells

{markdown_table(atlas.head(40), max_rows=40)}

## Trust Selector Scoreboard

{markdown_table(scoreboard.head(40), max_rows=40)}

## Selection Counts

{markdown_table(selection_counts.head(60), max_rows=60)}

## Interpretation

This run separates explanation from deployable trust selection. The atlas identifies where each family wins after the fact, while the selectors test whether that trust information can be used with only prior target history. If the best selector does not beat `0036`, the evidence says the current fixed meta-feature trust bins explain family behavior but are not yet strong enough to improve the already-soft 0036 inverse-MAE stack.
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Stack Trust Meta-Features\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_stack_trust_meta_features.py`:

- `{FOLDER_NAME}`: fixed-bin meta-feature trust atlas and strict prior-only selectors over official raw, 0033 smooth family, 0034 centroid family, and 0035 revision family.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Meta features | {manifest['meta_features']} |
| Trust candidates | {manifest['trust_candidates']} |
| Best trust selector MAE | {manifest['best_trust_mae']} |
| Best trust selector RMSE | {manifest['best_trust_rmse']} |
| Best trust selector delta vs official | {manifest['best_trust_delta_vs_official']} |
| Current overall best MAE after 0037 | {manifest['current_overall_best_mae']} |

Leakage contract: all scored rows are before `{CONFIRMATION_START.date()}`; deployable trust selectors use only `target_date < current target_date`, and fixed meta-feature bins do not use target-derived cutpoints.
"""
    write_text(index_path, text)


def write_outputs(
    *,
    frame: pd.DataFrame,
    catalog: pd.DataFrame,
    family_scores: pd.DataFrame,
    atlas: pd.DataFrame,
    feature_summary: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_set_catalog: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    selection_counts = trust_selection_counts(predictions)
    comparison = baseline_comparison(scoreboard, family_scores)

    write_csv(artifacts / "meta_feature_catalog.csv", catalog)
    write_csv(artifacts / "feature_set_catalog.csv", feature_set_catalog)
    write_csv(artifacts / "family_scoreboard.csv", family_scores)
    write_csv(artifacts / "trust_atlas.csv", atlas)
    write_csv(artifacts / "trust_feature_summary.csv", feature_summary)
    write_csv(artifacts / "trust_scoreboard.csv", scoreboard)
    write_csv(artifacts / "trust_predictions.csv", predictions)
    write_csv(artifacts / "selection_counts.csv", selection_counts)
    write_csv(artifacts / "baseline_comparison.csv", comparison)
    top_ids = scoreboard.head(8)["candidate_id"].to_list() if not scoreboard.empty else []
    write_csv(
        artifacts / "top_trust_predictions.csv",
        predictions[predictions["candidate_id"].isin(top_ids)].copy() if top_ids else predictions.head(0),
    )

    best = scoreboard.iloc[0] if not scoreboard.empty else None
    overall_best = comparison.iloc[0] if not comparison.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "source_counts": {str(k): int(v) for k, v in frame["forecast_source_family"].value_counts().to_dict().items()},
        "meta_features": int(len(catalog)),
        "trust_candidates": int(len(scoreboard)),
        "best_trust_selector": "" if best is None else str(best["candidate_id"]),
        "best_trust_mae": None if best is None else float(best["mae"]),
        "best_trust_rmse": None if best is None else float(best["rmse"]),
        "best_trust_delta_vs_official": None if best is None else float(best["delta_vs_official_same_rows"]),
        "current_overall_best": "" if overall_best is None else str(overall_best["system"]),
        "current_overall_best_mae": None if overall_best is None else float(overall_best["mae"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "stack_trust_meta_features_manifest.json", manifest)
    write_readme(
        folder=folder,
        manifest=manifest,
        family_scores=family_scores,
        atlas=atlas,
        feature_summary=feature_summary,
        scoreboard=scoreboard,
        selection_counts=selection_counts,
        comparison=comparison,
    )
    update_master_index(manifest)
    return manifest


def run() -> dict[str, object]:
    frame, catalog = build_trust_frame()
    require_no_confirmation_dates(frame["target_date"], context="0037 trust frame")
    meta_features = catalog["meta_feature"].astype(str).to_list()
    family_scores = family_scoreboard(frame)
    atlas = trust_atlas(frame, meta_features)
    summary = trust_feature_summary(atlas)
    scoreboard, predictions, feature_set_catalog = run_trust_screen(frame, meta_features)
    require_no_confirmation_dates(predictions["target_date"], context="0037 trust predictions")
    return write_outputs(
        frame=frame,
        catalog=catalog,
        family_scores=family_scores,
        atlas=atlas,
        feature_summary=summary,
        scoreboard=scoreboard,
        predictions=predictions,
        feature_set_catalog=feature_set_catalog,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 stack trust meta-feature analysis.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
