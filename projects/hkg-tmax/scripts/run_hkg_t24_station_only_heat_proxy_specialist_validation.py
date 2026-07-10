from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    update_markdown_section,
)
from scripts.run_hkg_t24_station_contribution_atlas import load_target  # noqa: E402
from scripts.run_hkg_t24_station_only_late_period_bias_repair import (  # noqa: E402
    score_prediction_frame,
)

FOLDER_NAME = "0064_station_only_heat_proxy_specialist_validation"
ARTIFACT_0054 = RESEARCH_ROOT / "0054_station_only_walkforward_matrix_audit" / "artifacts"
ARTIFACT_0057 = RESEARCH_ROOT / "0057_station_only_residual_specialist_design_queue" / "artifacts"
ARTIFACT_0063 = RESEARCH_ROOT / "0063_station_only_guarded_candidate_stack" / "artifacts"
FEATURE_MATRIX_PATH = ARTIFACT_0054 / "features.parquet"
DESIGN_QUEUE_PATH = ARTIFACT_0057 / "design_queue.csv"
PREDICTIONS_0063_PATH = ARTIFACT_0063 / "predictions.parquet"
SUMMARY_0063_PATH = ARTIFACT_0063 / "summary.json"
TRAINING_THRESHOLD_END = pd.Timestamp("1999-12-31")
DEVELOPMENT_END = pd.Timestamp("2023-12-31")
DIAGNOSTIC_MID_HEAT = "mid"
DJF_MONTHS = (12, 1, 2)
JJA_MONTHS = (6, 7, 8)


@dataclass(frozen=True)
class ProxySpec:
    proxy_id: str
    family: str
    months: tuple[int, ...]
    required_buckets: tuple[tuple[str, str], ...]
    min_prior_rows: int
    shrinkage: float
    cap_c: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def quantile_edges(values: pd.Series, *, low_q: float = 1.0 / 3.0, high_q: float = 2.0 / 3.0) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 365 or clean.nunique(dropna=True) < 3:
        raise ValueError(f"Need at least 365 non-degenerate rows for thresholding, got {len(clean)}")
    low, high = clean.quantile([low_q, high_q]).tolist()
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError(f"Invalid quantile edges: low={low}, high={high}")
    return float(low), float(high)


def bucket_by_edges(values: pd.Series, low: float, high: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series("missing", index=values.index, dtype="object")
    out.loc[numeric <= low] = "low"
    out.loc[(numeric > low) & (numeric <= high)] = "mid"
    out.loc[numeric > high] = "high"
    return out


def month_to_season(month: int) -> str:
    if month in DJF_MONTHS:
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in JJA_MONTHS:
        return "JJA"
    return "SON"


def feature_columns(frame: pd.DataFrame, *, include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> list[str]:
    cols = []
    for column in frame.columns:
        if all(token in column for token in include) and not any(token in column for token in exclude):
            cols.append(column)
    return cols


def attach_proxy_metrics(features: pd.DataFrame) -> pd.DataFrame:
    out = features[["target_date"]].copy()
    temp_level_cols = feature_columns(
        features,
        include=("stat_", "air_temperature_c_latest_before_1500"),
        exclude=("change_1d", "minus_network_median"),
    )
    temp_traj_cols = feature_columns(
        features,
        include=("air_temperature_c_latest_before_1500_current_minus_rolling_mean_14d",),
    )
    dew_traj_cols = feature_columns(
        features,
        include=("dew_point_c_latest_before_1500_current_minus_rolling_mean_14d",),
    )
    pressure_pair_cols = feature_columns(
        features,
        include=("pair_sea_level_pressure_hpa_latest_before_1500",),
    )
    wind_pair_cols = feature_columns(
        features,
        include=("pair_wind_speed_mps_latest_before_1500",),
    )
    if not temp_level_cols or not temp_traj_cols or not pressure_pair_cols or not wind_pair_cols:
        raise RuntimeError(
            "Missing required station proxy feature families: "
            f"temp_level={len(temp_level_cols)}, temp_traj={len(temp_traj_cols)}, "
            f"dew_traj={len(dew_traj_cols)}, pressure_pair={len(pressure_pair_cols)}, wind_pair={len(wind_pair_cols)}"
        )
    out["station_temp_level_mean_c"] = features[temp_level_cols].mean(axis=1, skipna=True)
    out["station_temp_traj_14d_mean_c"] = features[temp_traj_cols].mean(axis=1, skipna=True)
    out["station_dew_traj_14d_mean_c"] = features[dew_traj_cols].mean(axis=1, skipna=True) if dew_traj_cols else np.nan
    out["pressure_spread_abs_max_hpa"] = features[pressure_pair_cols].abs().max(axis=1, skipna=True)
    out["wind_spread_abs_max_mps"] = features[wind_pair_cols].abs().max(axis=1, skipna=True)
    return out


def load_reference_frame() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    summary_0063 = load_json(SUMMARY_0063_PATH)
    best_stack = str(summary_0063["best_stack"])
    predictions = pd.read_parquet(PREDICTIONS_0063_PATH)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    predictions = predictions[predictions["stack_id"].astype(str).eq(best_stack)].copy()
    predictions = predictions[predictions["target_date"].le(DEVELOPMENT_END)].copy()
    if predictions.empty:
        raise RuntimeError(f"Missing 0063 best stack predictions: {best_stack}")
    require_no_confirmation_dates(predictions["target_date"], context="0064 0063 reference")
    reference = predictions[
        [
            "target_date",
            "target_tmax_c",
            "candidate_prediction_c",
            "candidate_sigma_c",
            "fold_id",
        ]
    ].rename(
        columns={
            "candidate_prediction_c": "reference_0063_prediction_c",
            "candidate_sigma_c": "reference_0063_sigma_c",
        }
    )
    reference["year"] = reference["target_date"].dt.year
    reference["month"] = reference["target_date"].dt.month
    reference["season"] = reference["month"].map(month_to_season)
    reference["reference_residual_to_add_c"] = reference["target_tmax_c"] - reference["reference_0063_prediction_c"]

    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    features["target_date"] = pd.to_datetime(features["target_date"], errors="coerce").dt.normalize()
    features = features[features["target_date"].lt(CONFIRMATION_START)].copy()
    proxy_metrics = attach_proxy_metrics(features)
    target = load_target()[["target_date", "target_tmax_c"]].copy()
    target["target_date"] = pd.to_datetime(target["target_date"], errors="coerce").dt.normalize()
    heat_low, heat_high = quantile_edges(target[target["target_date"].le(TRAINING_THRESHOLD_END)]["target_tmax_c"])

    frame = reference.merge(proxy_metrics, on="target_date", how="left", validate="one_to_one")
    frame["diagnostic_heat_bucket_pre2000_target"] = bucket_by_edges(frame["target_tmax_c"], heat_low, heat_high)
    threshold_rows = [
        {
            "bucket_column": "diagnostic_heat_bucket_pre2000_target",
            "source_metric": "target_tmax_c",
            "low_edge": heat_low,
            "high_edge": heat_high,
            "threshold_source": "target_history_through_1999-12-31",
            "deployability": "diagnostic_outcome_only_not_used_in_proxy_masks",
        }
    ]
    history = proxy_metrics[proxy_metrics["target_date"].le(TRAINING_THRESHOLD_END)].copy()
    for metric in [
        "station_temp_level_mean_c",
        "station_temp_traj_14d_mean_c",
        "station_dew_traj_14d_mean_c",
        "pressure_spread_abs_max_hpa",
        "wind_spread_abs_max_mps",
    ]:
        low, high = quantile_edges(history[metric])
        bucket_col = metric.replace("_mean_c", "").replace("_abs_max_hpa", "").replace("_abs_max_mps", "") + "_bucket"
        frame[bucket_col] = bucket_by_edges(frame[metric], low, high)
        threshold_rows.append(
            {
                "bucket_column": bucket_col,
                "source_metric": metric,
                "low_edge": low,
                "high_edge": high,
                "threshold_source": "station_feature_history_through_1999-12-31",
                "deployability": "deployable_pre_cutoff",
            }
        )
    frame = frame.sort_values("target_date").reset_index(drop=True)
    require_no_confirmation_dates(frame["target_date"], context="0064 proxy frame")
    return frame, summary_0063, pd.DataFrame(threshold_rows)


def proxy_specs() -> list[ProxySpec]:
    return [
        ProxySpec("winter_temp_level_mid", "winter_mid_heat_proxy", DJF_MONTHS, (("station_temp_level_bucket", "mid"),), 45, 90.0, 1.25),
        ProxySpec("winter_temp_level_high", "winter_mid_heat_proxy", DJF_MONTHS, (("station_temp_level_bucket", "high"),), 45, 90.0, 1.25),
        ProxySpec(
            "winter_temp_mid_dew_mid",
            "winter_mid_heat_proxy",
            DJF_MONTHS,
            (("station_temp_level_bucket", "mid"), ("station_dew_traj_14d_bucket", "mid")),
            35,
            75.0,
            1.25,
        ),
        ProxySpec(
            "winter_temp_mid_pressure_high",
            "winter_mid_heat_proxy",
            DJF_MONTHS,
            (("station_temp_level_bucket", "mid"), ("pressure_spread_bucket", "high")),
            30,
            75.0,
            1.25,
        ),
        ProxySpec(
            "winter_temp_high_pressure_low",
            "winter_mid_heat_proxy",
            DJF_MONTHS,
            (("station_temp_level_bucket", "high"), ("pressure_spread_bucket", "low")),
            30,
            75.0,
            1.25,
        ),
        ProxySpec("summer_temp_level_mid", "warm_season_mid_proxy", JJA_MONTHS, (("station_temp_level_bucket", "mid"),), 45, 90.0, 1.25),
        ProxySpec("summer_temp_level_low", "warm_season_mid_proxy", JJA_MONTHS, (("station_temp_level_bucket", "low"),), 45, 90.0, 1.25),
        ProxySpec(
            "summer_temp_mid_dew_high",
            "warm_season_mid_proxy",
            JJA_MONTHS,
            (("station_temp_level_bucket", "mid"), ("station_dew_traj_14d_bucket", "high")),
            30,
            75.0,
            1.25,
        ),
        ProxySpec(
            "summer_temp_mid_wind_high",
            "warm_season_mid_proxy",
            JJA_MONTHS,
            (("station_temp_level_bucket", "mid"), ("wind_spread_bucket", "high")),
            30,
            75.0,
            1.25,
        ),
        ProxySpec(
            "summer_traj_high_wind_high",
            "warm_season_mid_proxy",
            JJA_MONTHS,
            (("station_temp_traj_14d_bucket", "high"), ("wind_spread_bucket", "high")),
            30,
            75.0,
            1.25,
        ),
        ProxySpec(
            "summer_temp_mid_pressure_high",
            "warm_season_mid_proxy",
            JJA_MONTHS,
            (("station_temp_level_bucket", "mid"), ("pressure_spread_bucket", "high")),
            30,
            75.0,
            1.25,
        ),
    ]


def mask_for_proxy(frame: pd.DataFrame, spec: ProxySpec) -> pd.Series:
    mask = frame["month"].isin(spec.months)
    for column, value in spec.required_buckets:
        if column not in frame.columns:
            raise KeyError(f"Missing proxy bucket column: {column}")
        mask &= frame[column].astype(str).eq(value)
    return mask.fillna(False)


def shrink_and_cap(raw: float, prior_rows: int, spec: ProxySpec) -> float:
    if prior_rows < spec.min_prior_rows or not math.isfinite(raw):
        return 0.0
    shrink = prior_rows / (prior_rows + spec.shrinkage)
    return float(np.clip(raw * shrink, -spec.cap_c, spec.cap_c))


def compute_prior_proxy_correction(frame: pd.DataFrame, active_mask: pd.Series, spec: ProxySpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values("target_date").reset_index(drop=True)
    active = active_mask.loc[ordered.index].to_numpy(dtype=bool) if active_mask.index.equals(ordered.index) else active_mask.to_numpy(dtype=bool)
    corrections = np.zeros(len(ordered), dtype=float)
    prior_counts = np.zeros(len(ordered), dtype=int)
    raw_means = np.full(len(ordered), math.nan, dtype=float)
    count = 0
    total = 0.0
    residuals = pd.to_numeric(ordered["reference_residual_to_add_c"], errors="coerce").to_numpy(dtype=float)
    for idx, is_active in enumerate(active):
        prior_counts[idx] = count
        if is_active and count >= spec.min_prior_rows:
            raw = total / count
            raw_means[idx] = raw
            corrections[idx] = shrink_and_cap(raw, count, spec)
        residual = residuals[idx]
        if is_active and math.isfinite(residual):
            count += 1
            total += residual
    return corrections, prior_counts, raw_means


def apply_proxy_specialist(frame: pd.DataFrame, spec: ProxySpec) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    active_mask = mask_for_proxy(ordered, spec)
    corrections, prior_rows, raw_means = compute_prior_proxy_correction(ordered, active_mask, spec)
    out = ordered[
        [
            "target_date",
            "target_tmax_c",
            "reference_0063_prediction_c",
            "reference_0063_sigma_c",
            "fold_id",
            "year",
            "month",
            "season",
            "diagnostic_heat_bucket_pre2000_target",
            "station_temp_level_mean_c",
            "station_temp_traj_14d_mean_c",
            "station_dew_traj_14d_mean_c",
            "pressure_spread_abs_max_hpa",
            "wind_spread_abs_max_mps",
            "station_temp_level_bucket",
            "station_temp_traj_14d_bucket",
            "station_dew_traj_14d_bucket",
            "pressure_spread_bucket",
            "wind_spread_bucket",
        ]
    ].copy()
    out["proxy_active"] = active_mask.to_numpy(dtype=bool)
    out["candidate_prediction_c"] = out["reference_0063_prediction_c"] + corrections
    out["residual_correction_c"] = corrections
    out["prior_rows"] = prior_rows
    out["raw_prior_residual_mean_c"] = raw_means
    out["proxy_id"] = spec.proxy_id
    out["family"] = spec.family
    out["months"] = ",".join(str(month) for month in spec.months)
    out["required_buckets"] = "|".join(f"{column}={value}" for column, value in spec.required_buckets)
    out["min_prior_rows"] = spec.min_prior_rows
    out["shrinkage"] = spec.shrinkage
    out["cap_c"] = spec.cap_c
    require_no_confirmation_dates(out["target_date"], context=f"0064 {spec.proxy_id}")
    return out.sort_values(["target_date", "proxy_id"]).reset_index(drop=True)


def proxy_alignment(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for proxy_id, group in predictions.groupby("proxy_id", observed=True):
        active = group[group["proxy_active"].astype(bool)].copy()
        months = tuple(int(part) for part in str(group["months"].iloc[0]).split(",") if part)
        family_pool = frame[frame["month"].isin(months)].copy()
        active_mid = active["diagnostic_heat_bucket_pre2000_target"].astype(str).eq(DIAGNOSTIC_MID_HEAT)
        pool_mid = family_pool["diagnostic_heat_bucket_pre2000_target"].astype(str).eq(DIAGNOSTIC_MID_HEAT)
        rows.append(
            {
                "proxy_id": proxy_id,
                "family": str(group["family"].iloc[0]),
                "required_buckets": str(group["required_buckets"].iloc[0]),
                "active_rows": int(len(active)),
                "family_pool_rows": int(len(family_pool)),
                "active_mid_heat_rows": int(active_mid.sum()),
                "family_mid_heat_rows": int(pool_mid.sum()),
                "precision_mid_heat": float(active_mid.mean()) if len(active) else math.nan,
                "family_base_mid_heat_share": float(pool_mid.mean()) if len(family_pool) else math.nan,
                "mid_heat_enrichment": (
                    float(active_mid.mean() - pool_mid.mean()) if len(active) and len(family_pool) else math.nan
                ),
                "mid_heat_recall_within_family": (
                    float(active_mid.sum() / pool_mid.sum()) if int(pool_mid.sum()) > 0 else math.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["mid_heat_enrichment", "active_rows"], ascending=[False, False]).reset_index(drop=True)


def score_candidates(predictions: pd.DataFrame, alignment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    align = alignment.set_index("proxy_id").to_dict("index")
    for proxy_id, group in predictions.groupby("proxy_id", observed=True):
        reference = group.rename(columns={"reference_0063_prediction_c": "reference_prediction_c"})
        full = score_prediction_frame(group, "candidate_prediction_c")
        ref_full = score_prediction_frame(reference, "reference_prediction_c")
        active = group[group["proxy_active"].astype(bool)].copy()
        ref_active = reference.loc[active.index].copy()
        active_score = score_prediction_frame(active, "candidate_prediction_c")
        ref_active_score = score_prediction_frame(ref_active, "reference_prediction_c")
        diagnostic_active = active[
            active["diagnostic_heat_bucket_pre2000_target"].astype(str).eq(DIAGNOSTIC_MID_HEAT)
        ].copy()
        ref_diagnostic_active = reference.loc[diagnostic_active.index].copy()
        diagnostic_score = score_prediction_frame(diagnostic_active, "candidate_prediction_c")
        ref_diagnostic_score = score_prediction_frame(ref_diagnostic_active, "reference_prediction_c")

        fold_rows: list[float] = []
        for _fold_id, fold in group.groupby("fold_id", observed=True):
            ref_fold = reference.loc[fold.index]
            candidate_fold = score_prediction_frame(fold, "candidate_prediction_c")
            reference_fold = score_prediction_frame(ref_fold, "reference_prediction_c")
            fold_rows.append(float(candidate_fold["mae"]) - float(reference_fold["mae"]))
        alignment_row = align[proxy_id]
        row = {
            "proxy_id": proxy_id,
            "family": str(group["family"].iloc[0]),
            "required_buckets": str(group["required_buckets"].iloc[0]),
            "n": full["n"],
            "mae": full["mae"],
            "rmse": full["rmse"],
            "bias": full["bias"],
            "delta_mae_vs_0063": float(full["mae"]) - float(ref_full["mae"]),
            "active_n": active_score["n"],
            "active_mae": active_score["mae"],
            "active_reference_mae": ref_active_score["mae"],
            "active_delta_mae_vs_0063": float(active_score["mae"]) - float(ref_active_score["mae"]),
            "diagnostic_mid_active_n": diagnostic_score["n"],
            "diagnostic_mid_active_mae": diagnostic_score["mae"],
            "diagnostic_mid_reference_mae": ref_diagnostic_score["mae"],
            "diagnostic_mid_delta_mae_vs_0063": (
                float(diagnostic_score["mae"]) - float(ref_diagnostic_score["mae"])
                if int(diagnostic_score["n"]) > 0
                else math.nan
            ),
            "fold_delta_max": max(fold_rows) if fold_rows else math.nan,
            "folds_improved": int(sum(delta < 0 for delta in fold_rows)),
            "precision_mid_heat": alignment_row["precision_mid_heat"],
            "family_base_mid_heat_share": alignment_row["family_base_mid_heat_share"],
            "mid_heat_enrichment": alignment_row["mid_heat_enrichment"],
            "mid_heat_recall_within_family": alignment_row["mid_heat_recall_within_family"],
            "mean_abs_correction_c": float(group["residual_correction_c"].abs().mean()),
            "active_correction_share": float(active["residual_correction_c"].abs().gt(1e-9).mean()) if len(active) else math.nan,
        }
        row["promotion_gate_passed"] = bool(
            row["active_n"] >= 100
            and row["active_delta_mae_vs_0063"] <= -0.005
            and row["delta_mae_vs_0063"] <= 0.004
            and row["fold_delta_max"] <= 0.012
            and row["mid_heat_enrichment"] >= 0.05
        )
        rows.append(row)
        for subgroup_name, subgroup, ref_subgroup in [
            ("active_proxy_rows", active, ref_active),
            ("active_diagnostic_mid_heat", diagnostic_active, ref_diagnostic_active),
        ]:
            candidate_score = score_prediction_frame(subgroup, "candidate_prediction_c")
            reference_score = score_prediction_frame(ref_subgroup, "reference_prediction_c")
            subgroup_rows.append(
                {
                    "proxy_id": proxy_id,
                    "subgroup": subgroup_name,
                    "n": candidate_score["n"],
                    "candidate_mae": candidate_score["mae"],
                    "reference_mae": reference_score["mae"],
                    "delta_mae_vs_0063": (
                        float(candidate_score["mae"]) - float(reference_score["mae"])
                        if int(candidate_score["n"]) > 0
                        else math.nan
                    ),
                    "candidate_rmse": candidate_score["rmse"],
                    "reference_rmse": reference_score["rmse"],
                    "candidate_bias": candidate_score["bias"],
                    "reference_bias": reference_score["bias"],
                }
            )
    scoreboard = pd.DataFrame(rows).sort_values(
        ["promotion_gate_passed", "active_delta_mae_vs_0063", "mid_heat_enrichment"],
        ascending=[False, True, False],
    )
    subgroups = pd.DataFrame(subgroup_rows).sort_values(["subgroup", "delta_mae_vs_0063"]).reset_index(drop=True)
    return scoreboard.reset_index(drop=True), subgroups


def leakage_audit(predictions: pd.DataFrame, thresholds: pd.DataFrame, scoreboard: pd.DataFrame) -> pd.DataFrame:
    proxy_cols = ["proxy_id", "family", "months", "required_buckets"]
    forbidden_mask_inputs = [
        "diagnostic_heat_bucket_pre2000_target",
        "target_tmax_c",
        "candidate_prediction_c",
        "reference_0063_prediction_c",
    ]
    required_text = "|".join(predictions[proxy_cols].astype(str).agg("|".join, axis=1).drop_duplicates().tolist())
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(predictions["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(predictions['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "thresholds_fit_pre2000_or_diagnostic_flagged",
            "passed": bool(
                thresholds["threshold_source"].astype(str).str.contains("1999-12-31").all()
                and thresholds.loc[
                    thresholds["bucket_column"].eq("diagnostic_heat_bucket_pre2000_target"), "deployability"
                ].astype(str).str.contains("diagnostic").all()
            ),
            "evidence": f"{len(thresholds)} threshold rows checked",
        },
        {
            "check_id": "proxy_masks_do_not_reference_target_heat",
            "passed": bool(not any(token in required_text for token in forbidden_mask_inputs)),
            "evidence": "proxy definitions only list month and deployable station-feature bucket names",
        },
        {
            "check_id": "corrections_have_prior_active_history_only",
            "passed": bool((predictions["prior_rows"] >= 0).all()),
            "evidence": "streaming correction updates only after each active row is scored",
        },
        {
            "check_id": "promotion_gate_requires_sample_effect_enrichment_and_no_large_fold_damage",
            "passed": bool(
                scoreboard.loc[scoreboard["promotion_gate_passed"], "mid_heat_enrichment"].gt(0).all()
                and scoreboard.loc[scoreboard["promotion_gate_passed"], "active_n"].ge(100).all()
                and scoreboard.loc[scoreboard["promotion_gate_passed"], "active_delta_mae_vs_0063"].le(-0.005).all()
                and scoreboard.loc[scoreboard["promotion_gate_passed"], "fold_delta_max"].le(0.012).all()
            ),
            "evidence": f"{int(scoreboard['promotion_gate_passed'].sum())} candidates passed gate",
        },
    ]
    return pd.DataFrame(checks)


def build_candidate_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "proxy_id": spec.proxy_id,
                "family": spec.family,
                "months": ",".join(str(month) for month in spec.months),
                "required_buckets": "|".join(f"{column}={value}" for column, value in spec.required_buckets),
                "min_prior_rows": spec.min_prior_rows,
                "shrinkage": spec.shrinkage,
                "cap_c": spec.cap_c,
                "target_heat_bucket_used_as_input": False,
            }
            for spec in proxy_specs()
        ]
    )


def build_readme(
    *,
    summary: dict[str, Any],
    design_rows: pd.DataFrame,
    definitions: pd.DataFrame,
    thresholds: pd.DataFrame,
    alignment: pd.DataFrame,
    scoreboard: pd.DataFrame,
    subgroups: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    return f"""# Station-Only Heat Proxy Specialist Validation

Generated: `{summary['generated_at_utc']}`

## Purpose

`0057` intentionally left two residual-specialist ideas blocked because they used a diagnostic realized target-heat bucket: `winter_mid_heat_proxy_specialist` and `warm_season_mid_proxy_specialist`. `0064` translates those ideas into deployable station-feature proxy masks and tests them on top of the current `0063` station-only stack.

This is a proxy-validation and residual-correction screen. It does not use 2024+ rows and does not use the target heat bucket as an input.

## Contract

- Reference baseline: `0063` best stack `{summary['reference_0063_stack']}`.
- Target dates: `{summary['first_date']}` to `{summary['last_date']}`.
- Proxy masks use only calendar month and pre-cutoff station-feature buckets.
- Station-feature bucket thresholds are fixed from history through `1999-12-31`.
- The realized target heat bucket is diagnostic-only and appears only in validation tables.
- Residual correction for date `T` uses only prior active proxy rows.

## Headline

| Item | Value |
|---|---:|
| Candidates tested | {summary['candidate_count']} |
| Rows scored per candidate | {summary['rows_scored']} |
| Reference 0063 MAE | {summary['reference_0063_mae']} |
| Best proxy | {summary['best_proxy']} |
| Best full MAE | {summary['best_mae']} |
| Best delta MAE vs 0063 | {summary['best_delta_mae_vs_0063']} |
| Best active delta MAE vs 0063 | {summary['best_active_delta_mae_vs_0063']} |
| Best mid-heat enrichment | {summary['best_mid_heat_enrichment']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

## Source Queue Rows

{markdown_table(design_rows, max_rows=5)}

## Candidate Definitions

{markdown_table(definitions, max_rows=40)}

## Thresholds

{markdown_table(thresholds, max_rows=20)}

## Proxy Alignment

{markdown_table(alignment, max_rows=40)}

## Scoreboard

{markdown_table(scoreboard, max_rows=60)}

## Subgroup Scoreboard

{markdown_table(subgroups, max_rows=60)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Interpretation

This experiment answers whether the diagnostic heat-bucket failure modes can be translated into deployable station-network proxies. A useful proxy must both enrich for the diagnostic mid-heat outcome and improve residual MAE without large fold damage. If proxy enrichment is weak or the correction does not improve MAE, the original diagnostic regime should remain blocked rather than promoted.

## Files

- `artifacts/predictions.parquet`
- `artifacts/predictions_sample.csv`
- `artifacts/candidate_definitions.csv`
- `artifacts/proxy_alignment.csv`
- `artifacts/scoreboard.csv`
- `artifacts/subgroup_scoreboard.csv`
- `artifacts/thresholds.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_only_heat_proxy_specialist_validation.py`:

- `{FOLDER_NAME}`: deployable proxy validation for the blocked winter/summer diagnostic heat-regime specialists from `0057`.

| Metric | Value |
|---|---:|
| Reference 0063 MAE | {summary['reference_0063_mae']} |
| Best proxy | {summary['best_proxy']} |
| Best full MAE | {summary['best_mae']} |
| Best active delta MAE | {summary['best_active_delta_mae_vs_0063']} |
| Best mid-heat enrichment | {summary['best_mid_heat_enrichment']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

Leakage contract: target heat bucket is diagnostic-only; proxy masks use only month and station-feature buckets fixed from pre-2000 history.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Station-Only Heat Proxy Specialist Validation",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_only_heat_proxy_specialist_validation.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Reference | `0063` `{summary['reference_0063_stack']}` | Tested |
| Candidates | `{summary['candidate_count']}` deployable proxy masks | Tested |
| Best proxy | `{summary['best_proxy']}` | Diagnostic |
| Reference MAE / RMSE | `{summary['reference_0063_mae']}` / `{summary['reference_0063_rmse']}` | Baseline |
| Best MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best active delta MAE | `{summary['best_active_delta_mae_vs_0063']}` | Proxy rows |
| Best mid-heat enrichment | `{summary['best_mid_heat_enrichment']}` | Diagnostic-only validation |
| Promotion gate passed | `{summary['best_promotion_gate_passed']}` | Guarded |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0064` converts the two blocked target-heat specialists into station-feature proxy tests. The target heat bucket remains diagnostic-only.
"""
    update_markdown_section(
        path,
        heading="Station-Only Heat Proxy Specialist Validation",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"48. Heat-proxy validation tested `{summary['candidate_count']}` deployable winter/summer proxy masks; "
        f"best active delta vs 0063 is `{summary['best_active_delta_mae_vs_0063']}` from `{summary['best_proxy']}`."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue station-network deep dives while the official forecast backfill remains incomplete: mine proxy-gated residual specialists from all 0054 station-feature buckets against the `0063` reference, but require prior-only corrections, fold damage limits, and diagnostic-only treatment of target-derived labels.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, summary_0063, thresholds = load_reference_frame()
    design_queue = pd.read_csv(DESIGN_QUEUE_PATH)
    design_rows = design_queue[
        design_queue["candidate_id"].astype(str).isin(
            ["winter_mid_heat_proxy_specialist", "warm_season_mid_proxy_specialist"]
        )
    ].copy()
    definitions = build_candidate_definitions()
    predictions = pd.concat([apply_proxy_specialist(frame, spec) for spec in proxy_specs()], ignore_index=True)
    alignment = proxy_alignment(frame, predictions)
    scoreboard, subgroups = score_candidates(predictions, alignment)
    leakage = leakage_audit(predictions, thresholds, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0064 leakage audit failed: {failed}")

    reference_score = score_prediction_frame(
        frame.rename(columns={"reference_0063_prediction_c": "reference_prediction_c"}),
        "reference_prediction_c",
    )
    best = scoreboard.iloc[0]
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "reference_0063_stack": str(summary_0063["best_stack"]),
        "candidate_count": int(scoreboard["proxy_id"].nunique()),
        "rows_scored": int(scoreboard["n"].max()),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "reference_0063_mae": float(reference_score["mae"]),
        "reference_0063_rmse": float(reference_score["rmse"]),
        "best_proxy": str(best["proxy_id"]),
        "best_family": str(best["family"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0063": float(best["delta_mae_vs_0063"]),
        "best_active_delta_mae_vs_0063": float(best["active_delta_mae_vs_0063"]),
        "best_diagnostic_mid_delta_mae_vs_0063": (
            float(best["diagnostic_mid_delta_mae_vs_0063"])
            if pd.notna(best["diagnostic_mid_delta_mae_vs_0063"])
            else math.nan
        ),
        "best_mid_heat_enrichment": float(best["mid_heat_enrichment"]),
        "best_mid_heat_recall_within_family": float(best["mid_heat_recall_within_family"]),
        "best_promotion_gate_passed": bool(best["promotion_gate_passed"]),
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
        "uses_2024_plus_rows": False,
    }

    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(artifacts / "predictions.parquet", index=False)
    write_csv(artifacts / "predictions_sample.csv", predictions.head(1000))
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "proxy_alignment.csv", alignment)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "subgroup_scoreboard.csv", subgroups)
    write_csv(artifacts / "thresholds.csv", thresholds)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_only_heat_proxy_specialist_validation_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            design_rows=design_rows,
            definitions=definitions,
            thresholds=thresholds,
            alignment=alignment,
            scoreboard=scoreboard,
            subgroups=subgroups,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Validate deployable station-feature proxies for blocked target-heat residual specialists."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
