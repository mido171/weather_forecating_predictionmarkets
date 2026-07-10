from __future__ import annotations

import argparse
import json
import math
import sys
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
    build_failure_frame,
)
from scripts.run_hkg_t24_stack_trust_meta_features import (  # noqa: E402
    TRUST_FAMILIES,
    bucket_binary,
    bucket_numeric,
    family_scoreboard,
    load_family_prediction_frame,
    load_meta_context,
    past_only_meta_trust_predictions,
    slug,
    trust_atlas,
    trust_feature_summary,
    trust_selection_counts,
)

FOLDER_NAME = "0038_forecast_history_state_deepening"
PRIOR_COMPARISON = RESEARCH_ROOT / "0037_stack_trust_meta_features" / "artifacts" / "baseline_comparison.csv"
TEXT_FLAGS = (
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
)


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([math.nan] * len(frame), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def sign_bucket(series: pd.Series, *, near: float = 0.25, large: float = 2.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    labels: list[str] = []
    for value in values:
        if not np.isfinite(value):
            labels.append("missing")
        elif value <= -large:
            labels.append("large_down")
        elif value < -near:
            labels.append("down")
        elif value <= near:
            labels.append("flat")
        elif value < large:
            labels.append("up")
        else:
            labels.append("large_up")
    return pd.Series(labels, index=series.index)


def count_bucket(series: pd.Series, thresholds: tuple[float, ...]) -> pd.Series:
    return bucket_numeric(pd.to_numeric(series, errors="coerce"), thresholds)


def combine_labels(frame: pd.DataFrame, columns: tuple[str, ...], *, missing: str = "missing") -> pd.Series:
    labels: list[str] = []
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        parts = [str(value) if pd.notna(value) else missing for value in row]
        labels.append("|".join(parts))
    return pd.Series(labels, index=frame.index)


def same_source_unchanged_streak(
    values: pd.Series,
    sources: pd.Series,
    *,
    tolerance: float = 1e-9,
) -> pd.Series:
    streak_by_source: dict[str, int] = {}
    last_value_by_source: dict[str, float] = {}
    out: list[float] = []
    for value, source in zip(pd.to_numeric(values, errors="coerce"), sources.astype(str), strict=False):
        if not np.isfinite(value):
            streak_by_source[source] = 0
            last_value_by_source[source] = math.nan
            out.append(math.nan)
            continue
        last_value = last_value_by_source.get(source, math.nan)
        if np.isfinite(last_value) and abs(value - last_value) <= tolerance:
            streak_by_source[source] = streak_by_source.get(source, 0) + 1
        else:
            streak_by_source[source] = 1
        last_value_by_source[source] = float(value)
        out.append(float(streak_by_source[source]))
    return pd.Series(out, index=values.index)


def text_signal_state(frame: pd.DataFrame) -> pd.Series:
    hot = numeric_column(frame, "text_hot") >= 0.5
    very_hot = numeric_column(frame, "text_very_hot") >= 0.5
    rain = numeric_column(frame, "text_any_rain") >= 0.5
    thunder = numeric_column(frame, "text_thunder") >= 0.5
    cloud = numeric_column(frame, "text_cloud") >= 0.5
    sunny = numeric_column(frame, "text_sunny_or_fine") >= 0.5
    humid = numeric_column(frame, "text_humid") >= 0.5
    labels: list[str] = []
    for row in zip(very_hot, hot, thunder, rain, cloud, sunny, humid, strict=False):
        row_very_hot, row_hot, row_thunder, row_rain, row_cloud, row_sunny, row_humid = row
        if row_very_hot:
            labels.append("very_hot")
        elif row_hot and row_sunny and not row_rain:
            labels.append("hot_sunny")
        elif row_hot:
            labels.append("hot")
        elif row_thunder:
            labels.append("thunder")
        elif row_rain and row_cloud:
            labels.append("rain_cloud")
        elif row_rain:
            labels.append("rain")
        elif row_cloud:
            labels.append("cloud")
        elif row_sunny:
            labels.append("sunny")
        elif row_humid:
            labels.append("humid")
        else:
            labels.append("neutral")
    return pd.Series(labels, index=frame.index)


def add_forecast_history_state_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    catalog_rows: list[dict[str, object]] = []

    numeric_specs: tuple[tuple[str, str, tuple[float, ...]], ...] = (
        ("forecast_vs_prior7_fine_bin", "forecast_max_vs_prior7_mean_source_c", (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)),
        ("forecast_vs_prior7_abs_bin", "forecast_max_vs_prior7_abs_source_c", (0.5, 1.5, 2.5, 3.5)),
        ("forecast_jump_fine_bin", "forecast_max_change_1_source_c", (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)),
        ("forecast_jump_abs_bin", "forecast_max_change_abs_1_source_c", (0.5, 1.5, 2.5, 3.5)),
        ("forecast_range_current_bin", "forecast_range_c", (2.0, 3.0, 4.0, 5.0, 6.0)),
        ("forecast_range_change_fine_bin", "forecast_range_change_1_source_c", (-2.0, -1.0, 0.0, 1.0, 2.0)),
        ("forecast_range_change_abs_bin", "forecast_range_change_abs_1_source_c", (0.5, 1.5, 2.5)),
        ("forecast_range_vs_prior7_bin", "forecast_range_vs_prior7_mean_source_c", (-2.0, -1.0, 0.0, 1.0, 2.0)),
        ("midpoint_change_fine_bin", "forecast_midpoint_change_1_source_c", (-2.0, -1.0, 0.0, 1.0, 2.0)),
        ("midpoint_change_abs_bin", "forecast_midpoint_change_abs_1_source_c", (0.5, 1.5, 2.5)),
        ("issue_to_cutoff_change_bin", "issue_to_cutoff_change_1_source_c", (-12.0, -6.0, 0.0, 6.0, 12.0)),
        ("forecast_prior7_volatility_bin", "forecast_max_prior7_std_source_c", (0.25, 0.5, 1.0, 1.5)),
        ("prior_bias_fine_bin", "prediction_0018_c_prior90_source_residual_mean_c", (-0.5, -0.2, 0.0, 0.2, 0.5)),
    )

    out["forecast_max_vs_prior7_abs_source_c"] = numeric_column(out, "forecast_max_vs_prior7_mean_source_c").abs()
    out["forecast_max_change_abs_1_source_c"] = numeric_column(out, "forecast_max_change_1_source_c").abs()
    out["forecast_range_change_abs_1_source_c"] = numeric_column(out, "forecast_range_change_1_source_c").abs()
    out["forecast_midpoint_change_abs_1_source_c"] = numeric_column(out, "forecast_midpoint_change_1_source_c").abs()

    for feature_name, source_col, thresholds in numeric_specs:
        meta_col = f"meta_{feature_name}"
        out[meta_col] = count_bucket(numeric_column(out, source_col), thresholds)
        catalog_rows.append(
            {
                "meta_feature": meta_col,
                "type": "fixed_numeric_bucket",
                "source_col": source_col,
                "thresholds": ",".join(str(value) for value in thresholds),
            }
        )

    sign_specs = (
        ("forecast_vs_prior7_sign", "forecast_max_vs_prior7_mean_source_c", 0.25, 2.0),
        ("forecast_jump_sign", "forecast_max_change_1_source_c", 0.25, 2.0),
        ("forecast_range_change_sign", "forecast_range_change_1_source_c", 0.25, 1.5),
        ("midpoint_change_sign", "forecast_midpoint_change_1_source_c", 0.25, 1.5),
        ("issue_to_cutoff_change_sign", "issue_to_cutoff_change_1_source_c", 1.0, 6.0),
    )
    for feature_name, source_col, near, large in sign_specs:
        meta_col = f"meta_{feature_name}"
        out[meta_col] = sign_bucket(numeric_column(out, source_col), near=near, large=large)
        catalog_rows.append(
            {
                "meta_feature": meta_col,
                "type": "fixed_sign_bucket",
                "source_col": source_col,
                "thresholds": f"near={near},large={large}",
            }
        )

    out["forecast_max_unchanged_streak_source"] = same_source_unchanged_streak(
        numeric_column(out, "forecast_max_c"),
        out["forecast_source_family"],
    )
    out["meta_forecast_max_staleness_bin"] = count_bucket(
        out["forecast_max_unchanged_streak_source"],
        (1.0, 2.0, 4.0, 7.0),
    )
    catalog_rows.append(
        {
            "meta_feature": "meta_forecast_max_staleness_bin",
            "type": "same_source_streak_bucket",
            "source_col": "forecast_max_c",
            "thresholds": "1.0,2.0,4.0,7.0",
        }
    )

    out["meta_forecast_range_widening_flag"] = bucket_binary(
        (numeric_column(out, "forecast_range_change_1_source_c") >= 0.5).astype(float)
    )
    catalog_rows.append(
        {
            "meta_feature": "meta_forecast_range_widening_flag",
            "type": "binary",
            "source_col": "forecast_range_change_1_source_c >= 0.5",
            "thresholds": "",
        }
    )

    change_cols = [f"{flag}_change_1_source" for flag in TEXT_FLAGS if f"{flag}_change_1_source" in out.columns]
    turned_on_cols = [f"{flag}_turned_on_source" for flag in TEXT_FLAGS if f"{flag}_turned_on_source" in out.columns]
    turned_off_cols = [f"{flag}_turned_off_source" for flag in TEXT_FLAGS if f"{flag}_turned_off_source" in out.columns]
    if change_cols:
        out["text_turnover_count_source"] = sum(numeric_column(out, col).abs().fillna(0.0) for col in change_cols)
    else:
        out["text_turnover_count_source"] = 0.0
    if turned_on_cols:
        out["text_turned_on_count_source"] = sum(numeric_column(out, col).fillna(0.0) for col in turned_on_cols)
    else:
        out["text_turned_on_count_source"] = 0.0
    if turned_off_cols:
        out["text_turned_off_count_source"] = sum(numeric_column(out, col).fillna(0.0) for col in turned_off_cols)
    else:
        out["text_turned_off_count_source"] = 0.0

    text_count_specs = (
        ("text_turnover_bin", "text_turnover_count_source", (0.0, 1.0, 2.0, 4.0)),
        ("text_turned_on_count_bin", "text_turned_on_count_source", (0.0, 1.0, 2.0)),
        ("text_turned_off_count_bin", "text_turned_off_count_source", (0.0, 1.0, 2.0)),
    )
    for feature_name, source_col, thresholds in text_count_specs:
        meta_col = f"meta_{feature_name}"
        out[meta_col] = count_bucket(numeric_column(out, source_col), thresholds)
        catalog_rows.append(
            {
                "meta_feature": meta_col,
                "type": "text_revision_count_bucket",
                "source_col": source_col,
                "thresholds": ",".join(str(value) for value in thresholds),
            }
        )

    out["meta_text_signal_state"] = text_signal_state(out)
    catalog_rows.append(
        {
            "meta_feature": "meta_text_signal_state",
            "type": "categorical",
            "source_col": ",".join(TEXT_FLAGS),
            "thresholds": "",
        }
    )

    out["meta_forecast_history_state"] = combine_labels(
        out,
        ("meta_forecast_vs_prior7_sign", "meta_forecast_jump_sign", "meta_forecast_max_staleness_bin"),
    )
    out["meta_revision_range_state"] = combine_labels(
        out,
        ("meta_forecast_jump_sign", "meta_forecast_range_change_sign", "meta_forecast_range_widening_flag"),
    )
    out["meta_forecast_confidence_state"] = combine_labels(
        out,
        ("meta_forecast_range_current_bin", "meta_forecast_prior7_volatility_bin", "meta_forecast_max_staleness_bin"),
    )
    out["meta_forecast_text_shift_state"] = combine_labels(
        out,
        ("meta_forecast_jump_sign", "meta_text_turnover_bin", "meta_text_signal_state"),
    )
    out["meta_forecast_prior_bias_state"] = combine_labels(
        out,
        ("meta_forecast_vs_prior7_sign", "meta_forecast_jump_sign", "meta_prior_bias_fine_bin"),
    )
    for meta_col, source_col in (
        ("meta_forecast_history_state", "vs_prior7_sign+jump_sign+staleness"),
        ("meta_revision_range_state", "jump_sign+range_change_sign+range_widening"),
        ("meta_forecast_confidence_state", "range_current+prior7_volatility+staleness"),
        ("meta_forecast_text_shift_state", "jump_sign+text_turnover+text_signal_state"),
        ("meta_forecast_prior_bias_state", "vs_prior7_sign+jump_sign+prior_bias"),
    ):
        catalog_rows.append(
            {
                "meta_feature": meta_col,
                "type": "composite_state",
                "source_col": source_col,
                "thresholds": "fixed_component_states",
            }
        )

    return out, pd.DataFrame(catalog_rows)


def load_forecast_history_meta_context() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_frame, _prior_systems = build_failure_frame()
    raw_frame = add_revision_features(raw_frame)
    raw_frame["target_date"] = pd.to_datetime(raw_frame["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(raw_frame["target_date"], context="0038 raw forecast history context")
    enriched, catalog = add_forecast_history_state_features(raw_frame)
    meta_cols = [
        "target_date",
        "forecast_source_family",
        *[col for col in enriched.columns if col.startswith("meta_")],
    ]
    return enriched[meta_cols].drop_duplicates(["target_date", "forecast_source_family"], keep="last"), catalog


def build_forecast_history_trust_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = load_family_prediction_frame()
    old_meta, old_catalog = load_meta_context()
    new_meta, new_catalog = load_forecast_history_meta_context()
    frame = predictions.merge(old_meta, on=["target_date", "forecast_source_family"], how="left")
    frame = frame.merge(new_meta, on=["target_date", "forecast_source_family"], how="left")
    for family in TRUST_FAMILIES:
        if family not in frame.columns:
            raise ValueError(f"Missing trust family prediction column: {family}")
    catalog = pd.concat([old_catalog, new_catalog], ignore_index=True)
    catalog = catalog.drop_duplicates("meta_feature", keep="last").reset_index(drop=True)
    return frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True), catalog


def forecast_history_feature_sets(meta_features: list[str]) -> dict[str, tuple[str, ...]]:
    available = set(meta_features)
    sets: dict[str, tuple[str, ...]] = {}
    preferred_singles = [
        "meta_forecast_vs_prior7_bin",
        "meta_forecast_vs_prior7_fine_bin",
        "meta_forecast_vs_prior7_abs_bin",
        "meta_forecast_vs_prior7_sign",
        "meta_forecast_jump_bin",
        "meta_forecast_jump_fine_bin",
        "meta_forecast_jump_abs_bin",
        "meta_forecast_jump_sign",
        "meta_forecast_max_staleness_bin",
        "meta_forecast_range_current_bin",
        "meta_forecast_range_change_fine_bin",
        "meta_forecast_range_change_abs_bin",
        "meta_forecast_range_change_sign",
        "meta_forecast_range_widening_flag",
        "meta_forecast_range_vs_prior7_bin",
        "meta_midpoint_change_fine_bin",
        "meta_midpoint_change_abs_bin",
        "meta_midpoint_change_sign",
        "meta_issue_to_cutoff_change_bin",
        "meta_issue_to_cutoff_change_sign",
        "meta_forecast_prior7_volatility_bin",
        "meta_prior_bias_fine_bin",
        "meta_text_turnover_bin",
        "meta_text_turned_on_count_bin",
        "meta_text_turned_off_count_bin",
        "meta_text_signal_state",
        "meta_forecast_history_state",
        "meta_revision_range_state",
        "meta_forecast_confidence_state",
        "meta_forecast_text_shift_state",
        "meta_forecast_prior_bias_state",
    ]
    for feature in preferred_singles:
        if feature in available:
            sets[feature.replace("meta_", "")] = (feature,)

    composites: dict[str, tuple[str, ...]] = {
        "carryforward_0037_winner": ("meta_forecast_vs_prior7_bin",),
        "forecast_vs_prior_deep": (
            "meta_forecast_vs_prior7_fine_bin",
            "meta_forecast_vs_prior7_abs_bin",
            "meta_forecast_vs_prior7_sign",
            "meta_forecast_jump_fine_bin",
            "meta_forecast_jump_sign",
        ),
        "forecast_jump_staleness": (
            "meta_forecast_jump_sign",
            "meta_forecast_jump_abs_bin",
            "meta_forecast_max_staleness_bin",
            "meta_forecast_prior7_volatility_bin",
        ),
        "range_revision": (
            "meta_forecast_range_current_bin",
            "meta_forecast_range_change_sign",
            "meta_forecast_range_widening_flag",
            "meta_forecast_range_vs_prior7_bin",
            "meta_midpoint_change_sign",
        ),
        "issue_cadence_revision": (
            "meta_issue_to_cutoff_change_bin",
            "meta_issue_to_cutoff_change_sign",
            "meta_forecast_jump_sign",
            "meta_forecast_range_change_sign",
        ),
        "text_revision": (
            "meta_text_turnover_bin",
            "meta_text_turned_on_count_bin",
            "meta_text_turned_off_count_bin",
            "meta_text_signal_state",
            "meta_forecast_jump_sign",
        ),
        "forecast_history_core": (
            "meta_forecast_history_state",
            "meta_revision_range_state",
            "meta_forecast_confidence_state",
        ),
        "forecast_history_with_text": (
            "meta_forecast_history_state",
            "meta_forecast_text_shift_state",
            "meta_text_signal_state",
        ),
        "forecast_history_with_prior_bias": (
            "meta_forecast_history_state",
            "meta_forecast_prior_bias_state",
            "meta_prior_bias_fine_bin",
        ),
        "source_month_history": (
            "meta_source_family",
            "meta_month",
            "meta_forecast_history_state",
            "meta_revision_range_state",
        ),
        "all_forecast_history_deep": (
            "meta_forecast_vs_prior7_fine_bin",
            "meta_forecast_jump_fine_bin",
            "meta_forecast_max_staleness_bin",
            "meta_forecast_range_change_fine_bin",
            "meta_midpoint_change_fine_bin",
            "meta_text_turnover_bin",
            "meta_forecast_history_state",
            "meta_revision_range_state",
        ),
    }
    for name, columns in composites.items():
        filtered = tuple(column for column in columns if column in available)
        if filtered:
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


def run_forecast_history_trust_screen(
    frame: pd.DataFrame,
    meta_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    set_rows: list[dict[str, object]] = []
    for name, features in forecast_history_feature_sets(meta_features).items():
        set_rows.append({"feature_set": name, "features": ",".join(features), "feature_count": len(features)})
        for mode in ("best", "inverse_mae", "positive_lift"):
            for same_source in (False, True):
                candidate_id = f"trust_history_{slug(name)}_{mode}_{'same_source' if same_source else 'all_prior'}"
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


def baseline_comparison(scoreboard: pd.DataFrame, family_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if PRIOR_COMPARISON.exists():
        prior = pd.read_csv(PRIOR_COMPARISON)
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
                "system": f"0038_family_{row.family}",
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
                "system": "0038_best_forecast_history_state",
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
    best_text = "No scoreable forecast-history selector was produced."
    if best is not None:
        best_text = (
            f"Best forecast-history selector: `{best['candidate_id']}` with MAE `{best['mae']:.4f}`, "
            f"RMSE `{best['rmse']:.4f}`, and official delta "
            f"`{best['delta_vs_official_same_rows']:.4f}`."
        )
    readme = f"""# Forecast-History State Deepening

Generated: `{manifest['generated_at_utc']}`

## Purpose

`0037` proved that a single same-source trust feature, `forecast_max_vs_prior7_mean_source_c`, can decide which expert family to trust better than the broad `0036` stack. This run deepens that signal. It converts the raw forecast archive state into fixed, point-in-time meta-features: forecast-versus-prior state, forecast jump state, unchanged forecast staleness, range widening, midpoint shifts, issue-cadence changes, text-turnover state, and compact composite states.

The goal is not to train a new black-box model. The goal is to test whether these richer forecast-history states can choose among official raw, 0033 smooth specialists, 0034 centroid specialists, and 0035 revision specialists using only prior target history.

## Data Window

Rows used: `{manifest['official_rows']}` official forecast/target rows.

Date range: `{manifest['first_target_date']}` to `{manifest['last_target_date']}`.

Source counts: `{manifest['source_counts']}`.

## Leakage Contract

- All scored rows are earlier than `{CONFIRMATION_START.date()}`.
- Every deployable selector uses only rows with `target_date < current target_date`.
- Same-source selectors additionally restrict prior evidence to the current source family.
- Same-date rows from another source family are excluded from prior trust evidence.
- Buckets use fixed deterministic thresholds, categorical forecast text state, or same-source forecast-value persistence. No bucket boundary is fitted on the target.
- Past residual/bias features reused from upstream scripts are already prior-only and are not allowed to look at the current target.
- 2024+ confirmation rows are not loaded or scored.

## Main Result

{best_text}

## Baseline Comparison

{markdown_table(comparison, max_rows=30)}

## Family Prediction Scoreboard

{markdown_table(family_scores, max_rows=20)}

## Forecast-History Feature Summary

{markdown_table(feature_summary.head(40), max_rows=40)}

## Strongest Forecast-History Atlas Cells

{markdown_table(atlas.head(50), max_rows=50)}

## Forecast-History Selector Scoreboard

{markdown_table(scoreboard.head(60), max_rows=60)}

## Selection Counts

{markdown_table(selection_counts.head(80), max_rows=80)}

## Interpretation

This folder is one more insight layer in the path toward a competitive HKG Tmax system. The core question is whether HKO forecast-history behavior contains enough stable state information to route trust among specialist families. A selector beating `0037` would mean the richer state adds deployable value. A selector tying or losing to `0037` would still be useful: it says the extra state is explanatory but too sparse or redundant on the current non-contiguous forecast archive, and the next priority should move to continuous 2005-2026 forecast export plus stronger local residual models.
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Forecast-History State Deepening\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_forecast_history_state_deepening.py`:

- `{FOLDER_NAME}`: fixed forecast-history state features and strict prior-only family-trust selectors over official raw, 0033 smooth family, 0034 centroid family, and 0035 revision family.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Forecast-history meta features | {manifest['forecast_history_meta_features']} |
| Trust candidates | {manifest['trust_candidates']} |
| Best selector MAE | {manifest['best_trust_mae']} |
| Best selector RMSE | {manifest['best_trust_rmse']} |
| Best selector delta vs official | {manifest['best_trust_delta_vs_official']} |
| Current overall best MAE after 0038 | {manifest['current_overall_best_mae']} |

Leakage contract: all scored rows are before `{CONFIRMATION_START.date()}`; deployable selectors use only `target_date < current target_date`, and all new state buckets are fixed from pre-cutoff forecast archive fields.
"""
    write_text(index_path, text)


def update_milestones(manifest: dict[str, object]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Milestones\n"
    section_marker = "\n## Forecast-History State Deepening\n"
    blockers_marker = "\n## Current Blockers And Gaps\n"
    next_marker = "\n## Exact Next Recommended Codex Task\n"
    if section_marker in existing:
        base, rest = existing.split(section_marker, 1)
        if blockers_marker in rest:
            rest = rest.split(blockers_marker, 1)[1]
            suffix = f"{blockers_marker}{rest}"
        else:
            suffix = ""
    elif blockers_marker in existing:
        base, rest = existing.split(blockers_marker, 1)
        suffix = f"{blockers_marker}{rest}"
    else:
        base = existing.rstrip()
        suffix = ""

    best_line = (
        f"`{manifest['best_trust_selector']}` MAE `{manifest['best_trust_mae']}`, "
        f"RMSE `{manifest['best_trust_rmse']}`, delta vs official "
        f"`{manifest['best_trust_delta_vs_official']}`"
    )
    section = f"""{base.rstrip()}
{section_marker}
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_forecast_history_state_deepening.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Rows / candidates | Strongest current finding | Status |
|---|---:|---|---|
| Forecast-history state trust selectors | `{manifest['official_rows']}` official rows; `{manifest['forecast_history_meta_features']}` total meta features; `{manifest['trust_candidates']}` trust candidates | {best_line} | Current 0038 result |
| Best state feature set | `{manifest['best_feature_set']}` | Selected families and fallback behavior are documented in `selection_counts.csv`; full row-level predictions are in `forecast_history_predictions.csv` | Auditable |
| Baseline comparison | `{manifest['comparison_systems']}` systems | Current overall best after 0038: `{manifest['current_overall_best']}` MAE `{manifest['current_overall_best_mae']}` | Champion tracking updated |
| Leakage guards | fixed bins plus prior-only selectors | Zero 2024+ scored rows; selectors use only `target_date < current target_date` and same-source variants isolate source history | Guarded |

Interpretation: `0038` tests whether the winning 0037 forecast-vs-prior trust signal improves after adding richer forecast-history state: jump magnitude/direction, unchanged forecast staleness, forecast range widening, midpoint shifts, issue-cadence shifts, text turnover, and compact composite states. If the best 0038 candidate is not materially better than `0037`, the evidence says the current non-contiguous forecast archive already captures most stable trust value with the simpler forecast-vs-prior bin; the next lift must come from continuous 2005-2026 forecast export, stronger local residual models, or long-history station/regime interactions rather than more sparse state labels.
"""
    if suffix:
        if next_marker in suffix:
            before_next, _after_next = suffix.split(next_marker, 1)
            next_task = f"""{next_marker}

First snapshot or finish the active 2005-2026 forecast-detail backfill so the stable scored archive stops being limited to 2000-2004 and 2021-2023. Then rerun the official forecast export and repeat the 0033-0038 stack on the expanded continuous forecast archive. If continuing analysis before that boundary, implement `0039_station_network_forecast_residual_interaction_mining`: combine the 1949-2026 long-history station attributes with the 2000-2026 forecast states, mine station-gradient/forecast-bias interactions, and test strict prior-only regime specialists with zero 2024+ scoring.
"""
            suffix = before_next.rstrip() + next_task
        section += suffix
    write_text(path, section)


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

    write_csv(artifacts / "forecast_history_meta_feature_catalog.csv", catalog)
    write_csv(artifacts / "forecast_history_feature_set_catalog.csv", feature_set_catalog)
    write_csv(artifacts / "family_scoreboard.csv", family_scores)
    write_csv(artifacts / "forecast_history_trust_atlas.csv", atlas)
    write_csv(artifacts / "forecast_history_feature_summary.csv", feature_summary)
    write_csv(artifacts / "forecast_history_scoreboard.csv", scoreboard)
    write_csv(artifacts / "forecast_history_predictions.csv", predictions)
    write_csv(artifacts / "selection_counts.csv", selection_counts)
    write_csv(artifacts / "baseline_comparison.csv", comparison)
    top_ids = scoreboard.head(8)["candidate_id"].to_list() if not scoreboard.empty else []
    write_csv(
        artifacts / "top_forecast_history_predictions.csv",
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
        "forecast_history_meta_features": int(len(catalog)),
        "feature_sets": int(len(feature_set_catalog)),
        "trust_candidates": int(len(scoreboard)),
        "best_trust_selector": "" if best is None else str(best["candidate_id"]),
        "best_feature_set": "" if best is None else str(best["feature_set"]),
        "best_trust_mae": None if best is None else float(best["mae"]),
        "best_trust_rmse": None if best is None else float(best["rmse"]),
        "best_trust_delta_vs_official": None if best is None else float(best["delta_vs_official_same_rows"]),
        "current_overall_best": "" if overall_best is None else str(overall_best["system"]),
        "current_overall_best_mae": None if overall_best is None else float(overall_best["mae"]),
        "comparison_systems": int(len(comparison)),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "forecast_history_state_deepening_manifest.json", manifest)
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
    update_milestones(manifest)
    return manifest


def run() -> dict[str, object]:
    frame, catalog = build_forecast_history_trust_frame()
    require_no_confirmation_dates(frame["target_date"], context="0038 forecast-history trust frame")
    meta_features = catalog["meta_feature"].astype(str).to_list()
    family_scores = family_scoreboard(frame)
    atlas = trust_atlas(frame, meta_features)
    summary = trust_feature_summary(atlas)
    scoreboard, predictions, feature_set_catalog = run_forecast_history_trust_screen(frame, meta_features)
    require_no_confirmation_dates(predictions["target_date"], context="0038 forecast-history predictions")
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
    return argparse.ArgumentParser(description="Run HKG T24 forecast-history state deepening.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
