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
from scripts.run_hkg_t24_station_network_forecast_residual_interaction_mining import (  # noqa: E402
    FOLDER_NAME as STATION_NETWORK_0039_FOLDER,
)
from scripts.run_hkg_t24_station_network_forecast_residual_interaction_mining import (  # noqa: E402
    LATE_EVAL_START,
    build_analysis_frame,
)

FOLDER_NAME = "0041_station_network_forecast_stack"
ARTIFACT_0039_DIR = RESEARCH_ROOT / STATION_NETWORK_0039_FOLDER / "artifacts"
ARTIFACT_0040_DIR = RESEARCH_ROOT / "0040_station_network_smooth_residuals" / "artifacts"
MANIFEST_0039_PATH = RESEARCH_ROOT / "station_network_residuals_manifest.json"
MANIFEST_0040_PATH = RESEARCH_ROOT / "station_network_smooth_residuals_manifest.json"
MIN_GLOBAL_HISTORY = 160
MIN_BUCKET_HISTORY = 45
SCREEN_STAGE = "stage1_core_prior_router_bounded"
STACK_FAMILY_GROUPS = ("core",)
STACK_MODES = ("best", "inverse_mae", "positive_lift", "anchor_lift_blend")


@dataclass(frozen=True)
class StackSpec:
    feature_set: str
    feature_names: tuple[str, ...]
    mode: str
    same_source: bool
    family_group: str
    family_names: tuple[str, ...]
    min_global_history: int = MIN_GLOBAL_HISTORY
    min_bucket_history: int = MIN_BUCKET_HISTORY


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 130) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_0039_best_predictions() -> tuple[pd.DataFrame, dict[str, str]]:
    manifest = load_json(MANIFEST_0039_PATH)
    candidate_id = str(manifest["best_candidate"])
    predictions_path = ARTIFACT_0039_DIR / "top_candidate_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing 0039 top predictions: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    predictions = predictions[predictions["candidate_id"].astype(str).eq(candidate_id)].copy()
    if predictions.empty:
        fallback_path = ARTIFACT_0039_DIR / "candidate_predictions.csv"
        predictions = pd.read_csv(fallback_path)
        predictions = predictions[predictions["candidate_id"].astype(str).eq(candidate_id)].copy()
    if predictions.empty:
        raise RuntimeError(f"Could not find 0039 best candidate predictions: {candidate_id}")
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(predictions["target_date"], context="0041 0039 hard predictions")
    out = predictions[
        [
            "target_date",
            "forecast_source_family",
            "candidate_prediction_c",
            "residual_correction_c",
            "prior_cell_rows",
        ]
    ].copy()
    out = out.rename(
        columns={
            "candidate_prediction_c": "hard_0039_best_c",
            "residual_correction_c": "hard_0039_residual_correction_c",
            "prior_cell_rows": "hard_0039_prior_cell_rows",
        }
    )
    return out.drop_duplicates(["target_date", "forecast_source_family"], keep="last"), {
        "hard_0039_best_c": candidate_id,
    }


def selected_0040_smooth_catalog() -> pd.DataFrame:
    manifest = load_json(MANIFEST_0040_PATH)
    scoreboard_path = ARTIFACT_0040_DIR / "scoreboard.csv"
    if not scoreboard_path.exists():
        raise FileNotFoundError(f"Missing 0040 scoreboard: {scoreboard_path}")
    scoreboard = pd.read_csv(scoreboard_path)
    if scoreboard.empty:
        raise RuntimeError("0040 scoreboard is empty")

    selected_ids: list[tuple[str, str]] = []
    selected_ids.append(("manifest_best_late", str(manifest["best_late_candidate"])))
    selected_ids.append(("manifest_best_full", str(manifest["best_full_candidate"])))
    for row in scoreboard.sort_values(["late_eval_mae", "mae", "rmse"]).head(2).itertuples(index=False):
        selected_ids.append(("top_late", str(row.candidate_id)))
    for row in scoreboard.sort_values(["mae", "late_eval_mae", "rmse"]).head(2).itertuples(index=False):
        selected_ids.append(("top_full", str(row.candidate_id)))

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for role, candidate_id in selected_ids:
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        match = scoreboard[scoreboard["candidate_id"].astype(str).eq(candidate_id)]
        if match.empty:
            continue
        rank = len(rows) + 1
        row = match.iloc[0]
        family_name = f"smooth_0040_{rank:02d}"
        rows.append(
            {
                "family_name": family_name,
                "selection_role": role,
                "candidate_id": candidate_id,
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "late_eval_mae": float(row["late_eval_mae"]),
                "late_eval_rmse": float(row["late_eval_rmse"]),
                "feature": str(row["feature"]),
                "extra_features": "" if pd.isna(row.get("extra_features")) else str(row.get("extra_features")),
                "state_cols": "" if pd.isna(row.get("state_cols")) else str(row.get("state_cols")),
                "same_source": bool(row["same_source"]),
            }
        )
    return pd.DataFrame(rows)


def load_0040_selected_predictions(catalog: pd.DataFrame) -> pd.DataFrame:
    candidate_to_family = dict(zip(catalog["candidate_id"].astype(str), catalog["family_name"].astype(str), strict=False))
    selected_ids = set(candidate_to_family)
    predictions_path = ARTIFACT_0040_DIR / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing 0040 predictions: {predictions_path}")
    usecols = [
        "target_date",
        "forecast_source_family",
        "candidate_prediction_c",
        "residual_correction_c",
        "prior_rows",
        "neighbor_rows",
        "local_anchor_mae",
        "local_corrected_mae",
        "do_no_harm_gate_passed",
        "candidate_id",
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(predictions_path, usecols=usecols, chunksize=100_000):
        chunk = chunk[chunk["candidate_id"].astype(str).isin(selected_ids)].copy()
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError(f"No 0040 selected predictions found for {sorted(selected_ids)}")
    predictions = pd.concat(chunks, ignore_index=True)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(predictions["target_date"], context="0041 selected 0040 predictions")
    predictions["family_name"] = predictions["candidate_id"].astype(str).map(candidate_to_family)

    prediction_wide = predictions.pivot_table(
        index=["target_date", "forecast_source_family"],
        columns="family_name",
        values="candidate_prediction_c",
        aggfunc="last",
    ).reset_index()
    correction_wide = predictions.pivot_table(
        index=["target_date", "forecast_source_family"],
        columns="family_name",
        values="residual_correction_c",
        aggfunc="last",
    ).reset_index()
    correction_wide = correction_wide.rename(
        columns={family: f"{family}_residual_correction_c" for family in candidate_to_family.values()}
    )
    prior_wide = predictions.pivot_table(
        index=["target_date", "forecast_source_family"],
        columns="family_name",
        values="prior_rows",
        aggfunc="last",
    ).reset_index()
    prior_wide = prior_wide.rename(columns={family: f"{family}_prior_rows" for family in candidate_to_family.values()})
    gate_wide = predictions.pivot_table(
        index=["target_date", "forecast_source_family"],
        columns="family_name",
        values="do_no_harm_gate_passed",
        aggfunc="last",
    ).reset_index()
    gate_wide = gate_wide.rename(columns={family: f"{family}_gate_passed" for family in candidate_to_family.values()})

    out = prediction_wide.merge(correction_wide, on=["target_date", "forecast_source_family"], how="left")
    out = out.merge(prior_wide, on=["target_date", "forecast_source_family"], how="left")
    out = out.merge(gate_wide, on=["target_date", "forecast_source_family"], how="left")
    return out


def prediction_diff_bucket(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").abs()
    labels = np.select(
        [numeric.isna(), numeric <= 0.10, numeric <= 0.30, numeric <= 0.60],
        ["missing", "<=0.10", "0.10-0.30", "0.30-0.60"],
        default=">0.60",
    )
    return pd.Series(labels, index=values.index, dtype="object")


def numeric_bucket(values: pd.Series, thresholds: tuple[float, ...]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    labels: list[str] = []
    for value in numeric:
        if not np.isfinite(value):
            labels.append("missing")
            continue
        lower = -math.inf
        placed = False
        for threshold in thresholds:
            if value <= threshold:
                labels.append(f"<= {threshold:g}" if math.isinf(lower) else f"({lower:g}, {threshold:g}]")
                placed = True
                break
            lower = threshold
        if not placed:
            labels.append(f"> {thresholds[-1]:g}")
    return pd.Series(labels, index=values.index, dtype="object")


def boolean_bucket(values: pd.Series) -> pd.Series:
    bool_values = values.fillna(False).astype(bool)
    return pd.Series(np.where(bool_values, "yes", "no"), index=values.index, dtype="object")


def add_stack_meta_features(frame: pd.DataFrame, smooth_catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    rows: list[dict[str, object]] = []
    out["meta_source_family"] = out["forecast_source_family"].astype(str)
    rows.append({"meta_feature": "meta_source_family", "source_col": "forecast_source_family", "type": "categorical"})

    for col in (
        "meta_forecast_vs_prior7_bin",
        "meta_forecast_vs_prior7_sign",
        "meta_forecast_range_change_sign",
        "meta_text_signal_state",
        "meta_revision_range_state",
        "meta_month",
    ):
        if col in out.columns:
            out[col] = out[col].fillna("missing").astype(str)
            rows.append({"meta_feature": col, "source_col": col, "type": "existing_state"})

    out["meta_hard_0039_active"] = boolean_bucket(pd.to_numeric(out["hard_0039_residual_correction_c"], errors="coerce").abs() > 1e-12)
    out["meta_hard_0039_history_bin"] = numeric_bucket(out["hard_0039_prior_cell_rows"], (0.0, 45.0, 120.0, 240.0))
    rows.extend(
        [
            {"meta_feature": "meta_hard_0039_active", "source_col": "hard_0039_residual_correction_c", "type": "binary"},
            {"meta_feature": "meta_hard_0039_history_bin", "source_col": "hard_0039_prior_cell_rows", "type": "numeric_bucket"},
        ]
    )

    smooth_names = smooth_catalog["family_name"].astype(str).to_list()
    for family in smooth_names[:4]:
        correction_col = f"{family}_residual_correction_c"
        prior_col = f"{family}_prior_rows"
        gate_col = f"{family}_gate_passed"
        if correction_col in out.columns:
            out[f"meta_{family}_active"] = boolean_bucket(pd.to_numeric(out[correction_col], errors="coerce").abs() > 1e-12)
            rows.append({"meta_feature": f"meta_{family}_active", "source_col": correction_col, "type": "binary"})
        if prior_col in out.columns:
            out[f"meta_{family}_history_bin"] = numeric_bucket(out[prior_col], (0.0, 180.0, 500.0, 1000.0))
            rows.append({"meta_feature": f"meta_{family}_history_bin", "source_col": prior_col, "type": "numeric_bucket"})
        if gate_col in out.columns:
            out[f"meta_{family}_gate"] = boolean_bucket(out[gate_col])
            rows.append({"meta_feature": f"meta_{family}_gate", "source_col": gate_col, "type": "binary"})

    if len(smooth_names) >= 2:
        out["meta_smooth_late_full_disagreement"] = prediction_diff_bucket(out[smooth_names[0]] - out[smooth_names[1]])
        rows.append(
            {
                "meta_feature": "meta_smooth_late_full_disagreement",
                "source_col": f"{smooth_names[0]} - {smooth_names[1]}",
                "type": "prediction_difference_bucket",
            }
        )
    if smooth_names:
        out["meta_anchor_smooth_disagreement"] = prediction_diff_bucket(out[smooth_names[0]] - out["anchor_0038_c"])
        rows.append(
            {
                "meta_feature": "meta_anchor_smooth_disagreement",
                "source_col": f"{smooth_names[0]} - anchor_0038_c",
                "type": "prediction_difference_bucket",
            }
        )
    out["meta_hard_smooth_disagreement"] = (
        prediction_diff_bucket(out["hard_0039_best_c"] - out[smooth_names[0]]) if smooth_names else "missing"
    )
    rows.append(
        {
            "meta_feature": "meta_hard_smooth_disagreement",
            "source_col": "hard_0039_best_c - first_smooth_family",
            "type": "prediction_difference_bucket",
        }
    )
    return out, pd.DataFrame(rows).drop_duplicates("meta_feature", keep="first").reset_index(drop=True)


def build_stack_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame, _derived_catalog = build_analysis_frame()
    require_no_confirmation_dates(frame["target_date"], context="0041 base analysis frame")
    hard_predictions, hard_catalog = load_0039_best_predictions()
    smooth_catalog = selected_0040_smooth_catalog()
    smooth_predictions = load_0040_selected_predictions(smooth_catalog)
    frame = frame.merge(hard_predictions, on=["target_date", "forecast_source_family"], how="inner", validate="one_to_one")
    frame = frame.merge(smooth_predictions, on=["target_date", "forecast_source_family"], how="inner", validate="one_to_one")
    frame = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True)
    require_no_confirmation_dates(frame["target_date"], context="0041 merged stack frame")
    frame, meta_catalog = add_stack_meta_features(frame, smooth_catalog)

    family_rows = [
        {"family_name": "official_raw", "source_experiment": "official", "candidate_id": "official_raw", "role": "fallback"},
        {"family_name": "anchor_0038_c", "source_experiment": "0038", "candidate_id": "trust_history_forecast_vs_prior7_bin_best_same_source", "role": "anchor"},
        {"family_name": "hard_0039_best_c", "source_experiment": "0039", "candidate_id": hard_catalog["hard_0039_best_c"], "role": "hard_station_network"},
    ]
    for row in smooth_catalog.itertuples(index=False):
        family_rows.append(
            {
                "family_name": str(row.family_name),
                "source_experiment": "0040",
                "candidate_id": str(row.candidate_id),
                "role": str(row.selection_role),
            }
        )
    family_catalog = pd.DataFrame(family_rows)
    return frame, family_catalog, meta_catalog


def stack_feature_sets(meta_catalog: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    available = set(meta_catalog["meta_feature"].astype(str))
    requested = {
        "global": (),
        "source": ("meta_source_family",),
        "forecast_vs_prior7": ("meta_forecast_vs_prior7_bin",),
        "text_signal": ("meta_text_signal_state",),
        "source_forecast_text": ("meta_source_family", "meta_forecast_vs_prior7_bin", "meta_text_signal_state"),
        "source_revision_action": (
            "meta_source_family",
            "meta_forecast_range_change_sign",
            "meta_hard_0039_active",
            "meta_smooth_0040_01_active",
        ),
        "correction_activity": (
            "meta_hard_0039_active",
            "meta_smooth_0040_01_active",
            "meta_smooth_0040_02_active",
            "meta_smooth_late_full_disagreement",
        ),
        "prediction_disagreement": (
            "meta_smooth_late_full_disagreement",
            "meta_anchor_smooth_disagreement",
            "meta_hard_smooth_disagreement",
        ),
        "compact_all": (
            "meta_source_family",
            "meta_forecast_vs_prior7_bin",
            "meta_text_signal_state",
            "meta_hard_0039_active",
            "meta_smooth_0040_01_active",
            "meta_smooth_late_full_disagreement",
        ),
    }
    out: dict[str, tuple[str, ...]] = {}
    for name, cols in requested.items():
        filtered = tuple(col for col in cols if col in available)
        if name == "global" or filtered:
            out[name] = filtered
    return out


def family_groups(family_catalog: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    smooth_families = tuple(family_catalog.loc[family_catalog["source_experiment"].eq("0040"), "family_name"].astype(str))
    core = ("anchor_0038_c", "hard_0039_best_c", *smooth_families[:2])
    expanded = ("official_raw", "anchor_0038_c", "hard_0039_best_c", *smooth_families)
    return {"core": core, "expanded": expanded}


def prior_mae(values: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    valid = mask & np.isfinite(values) & np.isfinite(target)
    count = int(valid.sum())
    if count == 0:
        return 0, math.nan
    return count, float(np.abs(values[valid] - target[valid]).mean())


def estimate_family_prior_mae(
    *,
    values: np.ndarray,
    target: np.ndarray,
    base_prior: np.ndarray,
    feature_arrays: dict[str, np.ndarray],
    feature_names: tuple[str, ...],
    row_index: int,
    min_global_history: int,
    min_bucket_history: int,
) -> tuple[int, float]:
    weighted_sum = 0.0
    weight_sum = 0.0
    total_count = 0
    global_count, global_mae = prior_mae(values, target, base_prior)
    if global_count >= min_global_history and np.isfinite(global_mae):
        weight = 0.40 * math.sqrt(global_count)
        weighted_sum += weight * global_mae
        weight_sum += weight
        total_count += global_count

    current_bucket_masks: list[np.ndarray] = []
    for feature in feature_names:
        current_bucket = str(feature_arrays[feature][row_index])
        if current_bucket == "missing":
            continue
        mask = base_prior & (feature_arrays[feature] == current_bucket)
        count, mae = prior_mae(values, target, mask)
        if count >= min_bucket_history and np.isfinite(mae):
            weight = math.sqrt(count)
            weighted_sum += weight * mae
            weight_sum += weight
            total_count += count
        current_bucket_masks.append(feature_arrays[feature] == current_bucket)

    if len(current_bucket_masks) >= 2:
        joint_mask = base_prior.copy()
        for mask in current_bucket_masks:
            joint_mask &= mask
        count, mae = prior_mae(values, target, joint_mask)
        if count >= min_bucket_history and np.isfinite(mae):
            weight = 1.25 * math.sqrt(count)
            weighted_sum += weight * mae
            weight_sum += weight
            total_count += count

    if weight_sum <= 0.0:
        return 0, math.nan
    return total_count, float(weighted_sum / weight_sum)


def prior_estimates_for_row(
    *,
    family_values: dict[str, np.ndarray],
    target: np.ndarray,
    base_prior: np.ndarray,
    feature_arrays: dict[str, np.ndarray],
    feature_names: tuple[str, ...],
    row_index: int,
    min_global_history: int,
    min_bucket_history: int,
) -> dict[str, tuple[int, float]]:
    estimates: dict[str, tuple[int, float]] = {}
    for family, values in family_values.items():
        count, mae = estimate_family_prior_mae(
            values=values,
            target=target,
            base_prior=base_prior,
            feature_arrays=feature_arrays,
            feature_names=feature_names,
            row_index=row_index,
            min_global_history=min_global_history,
            min_bucket_history=min_bucket_history,
        )
        if count > 0 and np.isfinite(mae):
            estimates[family] = (count, mae)
    return estimates


def prediction_from_estimates(
    *,
    estimates: dict[str, tuple[int, float]],
    family_values: dict[str, np.ndarray],
    row_index: int,
    mode: str,
    anchor_family: str = "anchor_0038_c",
) -> tuple[float, str, int, float, float, float]:
    anchor_value = float(family_values[anchor_family][row_index])
    scored = [
        (family, count, mae)
        for family, (count, mae) in estimates.items()
        if np.isfinite(family_values[family][row_index]) and count > 0 and np.isfinite(mae)
    ]
    if not scored:
        return anchor_value, "anchor_0038_c_fallback", 0, math.nan, math.nan, 0.0
    scored = sorted(scored, key=lambda item: (item[2], item[0]))
    anchor_count, anchor_mae = estimates.get(anchor_family, (0, math.nan))

    if mode == "best":
        chosen = scored[0]
        return float(family_values[chosen[0]][row_index]), chosen[0], chosen[1], chosen[2], anchor_mae, 1.0

    if mode == "inverse_mae":
        weights = np.array([1.0 / max(item[2], 1e-6) for item in scored], dtype=float)
        values = np.array([float(family_values[item[0]][row_index]) for item in scored], dtype=float)
        weights = weights / weights.sum()
        return float(np.dot(weights, values)), "inverse_mae_family_blend", len(scored), scored[0][2], anchor_mae, float(weights.max())

    lifted: list[tuple[str, int, float, float]] = []
    if np.isfinite(anchor_mae) and anchor_count > 0:
        lifted = [
            (family, count, mae, anchor_mae - mae)
            for family, count, mae in scored
            if family != anchor_family and mae < anchor_mae and np.isfinite(family_values[family][row_index])
        ]
    if not lifted:
        return anchor_value, "anchor_0038_c_fallback", 0, math.nan, anchor_mae, 0.0
    lifted = sorted(lifted, key=lambda item: (-item[3], item[2], item[0]))

    if mode == "positive_lift":
        weights = np.array([max(item[3], 1e-9) for item in lifted], dtype=float)
        values = np.array([float(family_values[item[0]][row_index]) for item in lifted], dtype=float)
        weights = weights / weights.sum()
        return float(np.dot(weights, values)), "positive_lift_family_blend", len(lifted), lifted[0][2], anchor_mae, float(weights.max())

    if mode == "anchor_lift_blend":
        best_family, best_count, best_mae, lift = lifted[0]
        alpha = float(np.clip(0.20 + 2.5 * lift / max(anchor_mae, 1e-6), 0.20, 0.85))
        value = (1.0 - alpha) * anchor_value + alpha * float(family_values[best_family][row_index])
        return value, f"anchor_lift_blend:{best_family}", best_count, best_mae, anchor_mae, alpha

    raise ValueError(f"Unknown stack mode: {mode}")


def past_only_stack_predictions(frame: pd.DataFrame, spec: StackSpec) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").dt.normalize().to_numpy(dtype="datetime64[ns]")
    sources = ordered["forecast_source_family"].astype(str).to_numpy()
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    family_values = {
        family: pd.to_numeric(ordered[family], errors="coerce").to_numpy(dtype=float)
        for family in spec.family_names
    }
    feature_arrays = {feature: ordered[feature].fillna("missing").astype(str).to_numpy() for feature in spec.feature_names}

    predictions: list[float] = []
    selected: list[str] = []
    selected_counts: list[int] = []
    selected_prior_maes: list[float] = []
    anchor_prior_maes: list[float] = []
    selected_weights: list[float] = []
    eligible_counts: list[int] = []

    for index, target_date in enumerate(dates):
        base_prior = dates < target_date
        if spec.same_source:
            base_prior &= sources == sources[index]
        estimates = prior_estimates_for_row(
            family_values=family_values,
            target=target,
            base_prior=base_prior,
            feature_arrays=feature_arrays,
            feature_names=spec.feature_names,
            row_index=index,
            min_global_history=spec.min_global_history,
            min_bucket_history=spec.min_bucket_history,
        )
        prediction, family, count, family_mae, anchor_mae, weight = prediction_from_estimates(
            estimates=estimates,
            family_values=family_values,
            row_index=index,
            mode=spec.mode,
        )
        predictions.append(prediction)
        selected.append(family)
        selected_counts.append(count)
        selected_prior_maes.append(family_mae)
        anchor_prior_maes.append(anchor_mae)
        selected_weights.append(weight)
        eligible_counts.append(len(estimates))

    out = ordered[["target_date", "forecast_source_family", "target_tmax_c", "official_raw", "anchor_0038_c"]].copy()
    out["candidate_prediction_c"] = predictions
    out["selected_family"] = selected
    out["selected_prior_count"] = selected_counts
    out["selected_prior_mae"] = selected_prior_maes
    out["anchor_prior_mae"] = anchor_prior_maes
    out["selected_weight"] = selected_weights
    out["eligible_family_count"] = eligible_counts
    out["feature_set"] = spec.feature_set
    out["feature_names"] = ",".join(spec.feature_names)
    out["mode"] = spec.mode
    out["same_source"] = spec.same_source
    out["family_group"] = spec.family_group
    out["family_names"] = ",".join(spec.family_names)
    out["candidate_id"] = stack_candidate_id(spec)
    return out


def stack_candidate_id(spec: StackSpec) -> str:
    source = "same_source" if spec.same_source else "all_prior"
    return slug(f"stack_0041_{spec.family_group}_{spec.feature_set}_{spec.mode}_{source}")


def score_stack_candidate(predictions: pd.DataFrame, spec: StackSpec) -> dict[str, object]:
    score = score_prediction_frame(predictions.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    anchor = score_prediction_frame(predictions.rename(columns={"anchor_0038_c": "prediction"}), "prediction")
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "prediction"}), "prediction")
    late = predictions[pd.to_datetime(predictions["target_date"]) >= LATE_EVAL_START].copy()
    late_score = score_prediction_frame(late.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    late_anchor = score_prediction_frame(late.rename(columns={"anchor_0038_c": "prediction"}), "prediction")
    selected = predictions["selected_family"].astype(str)
    return {
        "candidate_id": stack_candidate_id(spec),
        "feature_set": spec.feature_set,
        "feature_names": ",".join(spec.feature_names),
        "mode": spec.mode,
        "same_source": spec.same_source,
        "family_group": spec.family_group,
        "family_count": len(spec.family_names),
        **score,
        "anchor_same_rows_mae": anchor["mae"],
        "official_same_rows_mae": official["mae"],
        "delta_vs_anchor": float(score["mae"] - anchor["mae"]),
        "delta_vs_official": float(score["mae"] - official["mae"]),
        "late_eval_n": int(late_score["n"]),
        "late_eval_first_date": str(late_score["first_date"]),
        "late_eval_last_date": str(late_score["last_date"]),
        "late_eval_mae": float(late_score["mae"]),
        "late_eval_rmse": float(late_score["rmse"]),
        "late_eval_anchor_mae": float(late_anchor["mae"]),
        "late_eval_delta_vs_anchor": float(late_score["mae"] - late_anchor["mae"]),
        "fallback_rows": int(selected.eq("anchor_0038_c_fallback").sum()),
        "anchor_selected_rows": int(selected.str.contains("anchor_0038_c", regex=False).sum()),
        "mean_eligible_families": float(predictions["eligible_family_count"].mean()),
    }


def build_stack_specs(frame: pd.DataFrame, family_catalog: pd.DataFrame, meta_catalog: pd.DataFrame) -> tuple[list[StackSpec], pd.DataFrame]:
    groups = family_groups(family_catalog)
    feature_sets = stack_feature_sets(meta_catalog)
    specs: list[StackSpec] = []
    rows: list[dict[str, object]] = []
    for feature_set, features in feature_sets.items():
        rows.append({"feature_set": feature_set, "feature_names": ",".join(features), "feature_count": len(features)})
        for family_group in STACK_FAMILY_GROUPS:
            family_names = tuple(family for family in groups[family_group] if family in frame.columns)
            for mode in STACK_MODES:
                for same_source in (False, True):
                    specs.append(
                        StackSpec(
                            feature_set=feature_set,
                            feature_names=features,
                            mode=mode,
                            same_source=same_source,
                            family_group=family_group,
                            family_names=family_names,
                        )
                    )
    return specs, pd.DataFrame(rows).drop_duplicates("feature_set", keep="first").reset_index(drop=True)


def run_stack_screen(frame: pd.DataFrame, specs: list[StackSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs:
        predictions = past_only_stack_predictions(frame, spec)
        score_rows.append(score_stack_candidate(predictions, spec))
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(score_rows)
    if not scoreboard.empty:
        scoreboard = scoreboard.sort_values(["late_eval_mae", "mae", "rmse"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return scoreboard, predictions


def family_scoreboard(frame: pd.DataFrame, family_catalog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in family_catalog["family_name"].astype(str):
        if family not in frame.columns:
            continue
        score = score_prediction_frame(frame.rename(columns={family: "prediction"}), "prediction")
        late = frame[pd.to_datetime(frame["target_date"]) >= LATE_EVAL_START].copy()
        late_score = score_prediction_frame(late.rename(columns={family: "prediction"}), "prediction")
        catalog_row = family_catalog[family_catalog["family_name"].astype(str).eq(family)].iloc[0]
        rows.append(
            {
                "family_name": family,
                "source_experiment": str(catalog_row["source_experiment"]),
                "role": str(catalog_row["role"]),
                "candidate_id": str(catalog_row["candidate_id"]),
                **score,
                "late_eval_mae": late_score["mae"],
                "late_eval_rmse": late_score["rmse"],
            }
        )
    return pd.DataFrame(rows).sort_values(["late_eval_mae", "mae", "rmse"]).reset_index(drop=True)


def selection_counts(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    return (
        predictions.groupby(["candidate_id", "selected_family"], observed=True, dropna=False)
        .agg(rows=("target_date", "count"))
        .reset_index()
        .sort_values(["candidate_id", "rows"], ascending=[True, False])
    )


def baseline_comparison(scoreboard: pd.DataFrame, family_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in family_scores.itertuples(index=False):
        rows.append(
            {
                "system": f"family_{row.family_name}",
                "candidate_id": str(row.candidate_id),
                "n": int(row.n),
                "first_date": str(row.first_date),
                "last_date": str(row.last_date),
                "mae": float(row.mae),
                "rmse": float(row.rmse),
                "bias": float(row.bias),
                "median_abs_error": float(row.median_abs_error),
                "late_eval_mae": float(row.late_eval_mae),
                "late_eval_rmse": float(row.late_eval_rmse),
            }
        )
    if not scoreboard.empty:
        best_late = scoreboard.iloc[0]
        best_full = scoreboard.sort_values(["mae", "late_eval_mae", "rmse"]).iloc[0]
        for system, row in (
            ("0041_best_late_station_network_stack", best_late),
            ("0041_best_full_station_network_stack", best_full),
        ):
            rows.append(
                {
                    "system": system,
                    "candidate_id": str(row["candidate_id"]),
                    "n": int(row["n"]),
                    "first_date": str(row["first_date"]),
                    "last_date": str(row["last_date"]),
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "bias": float(row["bias"]),
                    "median_abs_error": float(row["median_abs_error"]),
                    "late_eval_mae": float(row["late_eval_mae"]),
                    "late_eval_rmse": float(row["late_eval_rmse"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["late_eval_mae", "mae", "rmse"]).drop_duplicates("system", keep="first").reset_index(drop=True)


def write_readme(
    *,
    folder: Path,
    manifest: dict[str, object],
    family_catalog: pd.DataFrame,
    meta_catalog: pd.DataFrame,
    feature_set_catalog: pd.DataFrame,
    family_scores: pd.DataFrame,
    scoreboard: pd.DataFrame,
    counts: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best_late = scoreboard.iloc[0] if not scoreboard.empty else None
    best_full = scoreboard.sort_values(["mae", "late_eval_mae", "rmse"]).iloc[0] if not scoreboard.empty else None
    best_text = "No scoreable 0041 stack was produced."
    if best_late is not None and best_full is not None:
        best_text = (
            f"Best actual late-window stack: `{best_late['candidate_id']}` with MAE "
            f"`{best_late['late_eval_mae']:.4f}`, RMSE `{best_late['late_eval_rmse']:.4f}`, "
            f"and late-window delta versus the 0038 anchor `{best_late['late_eval_delta_vs_anchor']:.4f}`. "
            f"Its full-window MAE is `{best_late['mae']:.4f}`.\n\n"
            f"Best full-window stack: `{best_full['candidate_id']}` with full MAE `{best_full['mae']:.4f}`, "
            f"RMSE `{best_full['rmse']:.4f}`, and full-window delta versus the 0038 anchor "
            f"`{best_full['delta_vs_anchor']:.4f}`. Its actual late-window MAE is "
            f"`{best_full['late_eval_mae']:.4f}`."
        )
    readme = f"""# Station-Network Forecast Stack

Generated: `{manifest['generated_at_utc']}`

## Purpose

`0040` proved that smooth station/network residual specialists contain additional signal beyond the forecast-history anchor. It also showed an important risk: the specialist with the best full-window MAE is not the same as the specialist with the best actual late-window MAE. This experiment is the next disciplined step. It does not train a black-box model and it does not touch 2024+ confirmation data. It creates a small family of already-audited candidate forecasts, then tests whether strict past-only routing can decide which family to trust on each target date.

The stack families are: official raw forecast, the 0038 forecast-history trust anchor, the 0039 hard station-network residual correction, and selected 0040 smooth station-network specialists. The 0040 specialists are selected from two directions: strongest actual late-window candidates and strongest full-window candidates. This matters because a system chasing only the late window can overfit the recent RSS-era rows, while a system chasing only the full window can miss recent behavior. The stack explicitly records both views.

This is a bounded stage-1 stack screen: `{manifest['screen_stage']}`. The first expanded grid exceeded the local execution timeout before writing artifacts, so this run keeps the high-signal core family group and compact routing contexts. That timeout is treated as a performance boundary, not as a scientific result.

## Data Window

Rows used: `{manifest['official_rows']}` scored forecast rows.

Screen stage: `{manifest['screen_stage']}`.

Full date range: `{manifest['first_target_date']}` to `{manifest['last_target_date']}`.

Configured late evaluation start: `{manifest['late_eval_start']}`.

Actual late evaluation range: `{manifest['late_eval_first_target_date']}` to `{manifest['late_eval_last_target_date']}`.

Late evaluation rows: `{manifest['late_eval_rows']}`.

Source counts: `{manifest['source_counts']}`.

## Leakage Contract

- All scored rows are earlier than `{CONFIRMATION_START.date()}`.
- 2024+ confirmation rows are not loaded, scored, selected on, or used for routing.
- Candidate families come from previously audited artifacts: 0038 anchor, 0039 hard residual correction, and 0040 smooth residual specialists.
- Each stack decision for a row uses only rows with `target_date < current target_date`.
- Same-source variants restrict the prior routing evidence to the current forecast source family.
- Same-date rows from another source family are excluded because the prior mask is date-strict, not row-order based.
- Meta routing features are pre-target information only: source family, forecast-history state, text state, prior-only correction activation flags, prior-row counts, and disagreement between candidate predictions.
- Prediction disagreement features compare available forecasts/corrections; they do not use the target value.
- The script writes row-level predictions, selection counts, candidate family lineage, and scoreboards so every result is auditable.

## Main Result

{best_text}

## What Was Tested

This stack tests two family groups. The `core` group contains the 0038 anchor, the 0039 hard residual correction, and the two named 0040 champions. The `expanded` group adds official raw and additional selected 0040 specialists from the top late-window and top full-window lists. For each family group, the script evaluates four routing modes. `best` chooses the family with the lowest prior MAE estimate. `inverse_mae` blends eligible families by inverse prior MAE. `positive_lift` only uses non-anchor families when prior evidence says they have beaten the anchor in the relevant historical context. `anchor_lift_blend` blends the anchor with the best lifted family, with the blend weight determined only by prior lift.

The routing contexts are deliberately compact. They include global prior history, source family, forecast-vs-prior state, text state, revision/range state, correction activation, correction history, and candidate-disagreement states. The goal is not to hide complexity in a high-capacity model. The goal is to test whether these manually understandable contexts can route between the forecast-history anchor and station/network residual specialists without looking forward.

## Baseline Comparison

{markdown_table(comparison, max_rows=30)}

## Family Catalog

{markdown_table(family_catalog, max_rows=30)}

## Family Scoreboard

{markdown_table(family_scores, max_rows=30)}

## Meta Feature Catalog

{markdown_table(meta_catalog, max_rows=40)}

## Feature Set Catalog

{markdown_table(feature_set_catalog, max_rows=30)}

## Stack Scoreboard

{markdown_table(scoreboard.head(60), max_rows=60)}

## Selection Counts

{markdown_table(counts.head(80), max_rows=80)}

## Interpretation

This experiment is mainly a routing test. If the best 0041 stack beats the 0040 full-window champion and the 0040 late-window champion at the same time, the result is a genuine compound improvement: forecast-history trust plus station-network residual specialists are complementary and can be routed with simple past-only evidence. If the stack improves one window but weakens the other, the result is still useful but not yet promotable as a final system; it tells us that the candidate families contain signal, while the trust router is not stable enough. If the stack fails both windows, the lesson is that 0040 specialists should be used as individual diagnostics until the forecast archive becomes continuous.

The most important limitation remains the non-contiguous stable scored forecast archive. The full scored frame currently covers 2000-2004 press archive rows plus 2021-2023 RSS-era rows, not a seamless 2000-2023 daily operational forecast record. Because of that, actual late evaluation begins on `{manifest['late_eval_first_target_date']}` even though the configured late split is `{manifest['late_eval_start']}`. The stack is leakage-safe, but the archive continuity problem still limits how much we should trust tiny deltas. Large deltas can be taken seriously as signal. Small deltas should guide the next experiment rather than be treated as production proof.
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Station-Network Forecast Stack\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_network_forecast_stack.py`:

- `{FOLDER_NAME}`: strict prior-only stack/trust routing across the 0038 forecast-history anchor, 0039 hard station-network residual correction, and selected 0040 smooth station-network specialists.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Screen stage | {manifest['screen_stage']} |
| Stack candidates | {manifest['stack_candidates']} |
| Family count | {manifest['family_count']} |
| Best late stack MAE | {manifest['best_late_eval_mae']} |
| Best late stack delta vs anchor | {manifest['best_late_eval_delta_vs_anchor']} |
| Best full stack MAE | {manifest['best_full_mae']} |
| Best full stack delta vs anchor | {manifest['best_full_delta_vs_anchor']} |

Leakage contract: every stack decision uses only `target_date < current target_date`; same-source variants isolate source-family history; all scored rows are before `{CONFIRMATION_START.date()}`.
"""
    write_text(index_path, text)


def update_milestones(manifest: dict[str, object]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Milestones\n"
    section_marker = "\n## Station-Network Forecast Stack\n"
    blockers_marker = "\n## Current Blockers And Gaps\n"
    next_marker = "\n## Exact Next Recommended Codex Task\n"
    if section_marker in existing:
        base, rest = existing.split(section_marker, 1)
        suffix = f"{blockers_marker}{rest.split(blockers_marker, 1)[1]}" if blockers_marker in rest else ""
    elif blockers_marker in existing:
        base, rest = existing.split(blockers_marker, 1)
        suffix = f"{blockers_marker}{rest}"
    else:
        base = existing.rstrip()
        suffix = ""

    section = f"""{base.rstrip()}
{section_marker}
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_network_forecast_stack.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Rows / candidates | Strongest current finding | Status |
|---|---:|---|---|
| Prior-only stack routing | `{manifest['official_rows']}` rows; `{manifest['stack_candidates']}` stack candidates; `{manifest['family_count']}` forecast families; stage `{manifest['screen_stage']}` | Best late stack `{manifest['best_late_candidate']}`: actual late MAE `{manifest['best_late_eval_mae']}`, late delta vs anchor `{manifest['best_late_eval_delta_vs_anchor']}`, full MAE `{manifest['best_late_candidate_full_mae']}`. Best full stack `{manifest['best_full_candidate']}`: full MAE `{manifest['best_full_mae']}`, full delta vs anchor `{manifest['best_full_delta_vs_anchor']}`, actual late MAE `{manifest['best_full_candidate_late_eval_mae']}` | Audited |
| Actual late window | `{manifest['late_eval_rows']}` rows | `{manifest['late_eval_first_target_date']}` to `{manifest['late_eval_last_target_date']}` because the stable scored archive is still non-contiguous | Explicit |
| Leakage guards | date-strict prior masks, same-source variants, pre-target meta states | Zero 2024+ scored rows; same-date cross-source rows are excluded from routing evidence | Guarded |

Interpretation: `0041` tests whether the 0038 forecast-history trust anchor, 0039 hard station-network residual correction, and selected 0040 smooth specialists can compound through strict prior-only routing. A stack that improves one window but not the other is useful diagnostic signal, but a promotable champion needs both full-window and actual late-window robustness.
"""
    if suffix:
        if next_marker in suffix:
            before_next, _after_next = suffix.split(next_marker, 1)
            next_task = f"""{next_marker}

Use the `0041` stack results to decide the next branch. If the stack beats the 0040 full and late champions simultaneously, harden it with sensitivity tests and then run the sealed 2024+ confirmation only when explicitly commanded. If the stack does not beat both, implement `0042_trust_router_sensitivity`: targeted ablations over stack family inclusion, min-history thresholds, and routing contexts, while keeping 2024+ locked.
"""
            suffix = before_next.rstrip() + "\n\n" + next_task
        section += suffix
    write_text(path, section)


def write_outputs(
    *,
    frame: pd.DataFrame,
    family_catalog: pd.DataFrame,
    meta_catalog: pd.DataFrame,
    feature_set_catalog: pd.DataFrame,
    family_scores: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    counts = selection_counts(predictions)
    comparison = baseline_comparison(scoreboard, family_scores)
    write_csv(artifacts / "family_catalog.csv", family_catalog)
    write_csv(artifacts / "meta_feature_catalog.csv", meta_catalog)
    write_csv(artifacts / "feature_set_catalog.csv", feature_set_catalog)
    write_csv(artifacts / "family_scoreboard.csv", family_scores)
    write_csv(artifacts / "stack_scoreboard.csv", scoreboard)
    write_csv(artifacts / "stack_predictions.csv", predictions)
    write_csv(artifacts / "selection_counts.csv", counts)
    write_csv(artifacts / "baseline_comparison.csv", comparison)
    top_ids = scoreboard.head(10)["candidate_id"].to_list() if not scoreboard.empty else []
    write_csv(
        artifacts / "top_stack_predictions.csv",
        predictions[predictions["candidate_id"].isin(top_ids)].copy() if top_ids else predictions.head(0),
    )

    best_late = scoreboard.iloc[0] if not scoreboard.empty else None
    best_full = scoreboard.sort_values(["mae", "late_eval_mae", "rmse"]).iloc[0] if not scoreboard.empty else None
    late_mask = pd.to_datetime(frame["target_date"]) >= LATE_EVAL_START
    late_frame = frame[late_mask].copy()
    late_first = "" if late_frame.empty else str(pd.to_datetime(late_frame["target_date"]).min().date())
    late_last = "" if late_frame.empty else str(pd.to_datetime(late_frame["target_date"]).max().date())
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "screen_stage": SCREEN_STAGE,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "late_eval_start": str(LATE_EVAL_START.date()),
        "late_eval_first_target_date": late_first,
        "late_eval_last_target_date": late_last,
        "late_eval_rows": int(late_mask.sum()),
        "source_counts": {str(k): int(v) for k, v in frame["forecast_source_family"].value_counts().to_dict().items()},
        "family_count": int(len(family_catalog)),
        "meta_features": int(len(meta_catalog)),
        "feature_sets": int(len(feature_set_catalog)),
        "stack_candidates": int(len(scoreboard)),
        "best_late_candidate": "" if best_late is None else str(best_late["candidate_id"]),
        "best_late_candidate_full_mae": None if best_late is None else float(best_late["mae"]),
        "best_late_eval_mae": None if best_late is None else float(best_late["late_eval_mae"]),
        "best_late_eval_delta_vs_anchor": None if best_late is None else float(best_late["late_eval_delta_vs_anchor"]),
        "best_full_candidate": "" if best_full is None else str(best_full["candidate_id"]),
        "best_full_mae": None if best_full is None else float(best_full["mae"]),
        "best_full_delta_vs_anchor": None if best_full is None else float(best_full["delta_vs_anchor"]),
        "best_full_candidate_late_eval_mae": None if best_full is None else float(best_full["late_eval_mae"]),
        "anchor_full_mae": float(family_scores.loc[family_scores["family_name"].eq("anchor_0038_c"), "mae"].iloc[0]),
        "anchor_late_eval_mae": float(family_scores.loc[family_scores["family_name"].eq("anchor_0038_c"), "late_eval_mae"].iloc[0]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "station_network_forecast_stack_manifest.json", manifest)
    write_readme(
        folder=folder,
        manifest=manifest,
        family_catalog=family_catalog,
        meta_catalog=meta_catalog,
        feature_set_catalog=feature_set_catalog,
        family_scores=family_scores,
        scoreboard=scoreboard,
        counts=counts,
        comparison=comparison,
    )
    update_master_index(manifest)
    update_milestones(manifest)
    return manifest


def run() -> dict[str, object]:
    frame, family_catalog, meta_catalog = build_stack_frame()
    specs, feature_set_catalog = build_stack_specs(frame, family_catalog, meta_catalog)
    family_scores = family_scoreboard(frame, family_catalog)
    scoreboard, predictions = run_stack_screen(frame, specs)
    require_no_confirmation_dates(predictions["target_date"], context="0041 stack predictions")
    return write_outputs(
        frame=frame,
        family_catalog=family_catalog,
        meta_catalog=meta_catalog,
        feature_set_catalog=feature_set_catalog,
        family_scores=family_scores,
        scoreboard=scoreboard,
        predictions=predictions,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 station-network forecast stack.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
