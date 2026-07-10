from __future__ import annotations

import argparse
import json
import math
import re
import sys
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
    EVAL_END,
    EVAL_START,
    TRAIN_END,
    apply_tertile_bins,
    quantile_edges_from_train,
    safe_corr,
    update_markdown_section,
)
from scripts.run_hkg_t24_station_contribution_atlas import (  # noqa: E402
    load_official_residuals,
    load_station_day_features,
    load_station_metadata,
    load_target,
)
from scripts.run_hkg_t24_station_lag_slope_information_atlas import (  # noqa: E402
    station_attribute_variants,
)

FOLDER_NAME = "0051_station_regime_interaction_atlas"
ARTIFACT_ROOT_0047 = RESEARCH_ROOT / "0047_station_contribution_atlas" / "artifacts"
ARTIFACT_ROOT_0050 = RESEARCH_ROOT / "0050_station_lag_slope_information_atlas" / "artifacts"
STATION_ATTRIBUTE_ATLAS_PATH = ARTIFACT_ROOT_0047 / "station_attribute_atlas.csv"
PAIR_SPREAD_ATLAS_PATH = ARTIFACT_ROOT_0047 / "pair_spread_atlas.csv"
TRAJECTORY_ATLAS_PATH = ARTIFACT_ROOT_0050 / "station_lag_slope_variant_atlas.csv"

TOP_TRAJECTORY_FEATURES = 16
TOP_STATION_ATTRIBUTE_FEATURES = 10
TOP_PAIR_FEATURES = 10
MIN_EVAL_CELL_ROWS = 80
MIN_TRAIN_CELL_ROWS = 80
MIN_OFFICIAL_CELL_ROWS = 20
MIN_CONDITIONAL_ROWS = 120
TOP_CELL_INTERACTIONS = 25
TOP_TIMESERIES_INTERACTIONS = 5
BIN_LABELS = ("low", "mid", "high")


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def finite_float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else math.nan
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def station_token(station_id: object) -> str:
    return slug(str(station_id).replace("-", "_"), limit=40)


def unique_feature_id(base: str, used: set[str]) -> str:
    candidate = slug(base, limit=170)
    if candidate not in used:
        used.add(candidate)
        return candidate
    index = 2
    while f"{candidate}_{index}" in used:
        index += 1
    out = f"{candidate}_{index}"
    used.add(out)
    return out


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def balanced_select(
    frame: pd.DataFrame,
    *,
    limit: int,
    group_columns: list[str],
    per_group_limit: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sorted_frame = frame.sort_values("priority_score", ascending=False, na_position="last").reset_index(drop=True)
    selected: list[pd.Series] = []
    group_counts: dict[tuple[str, ...], int] = {}
    for _, row in sorted_frame.iterrows():
        keys = tuple(str(row.get(column, "")) for column in group_columns)
        if group_counts.get(keys, 0) >= per_group_limit:
            continue
        selected.append(row)
        group_counts[keys] = group_counts.get(keys, 0) + 1
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected_ids = {int(row.name) for row in selected if row.name is not None}
        for index, row in sorted_frame.iterrows():
            if int(index) in selected_ids:
                continue
            selected.append(row)
            if len(selected) >= limit:
                break
    return pd.DataFrame(selected).reset_index(drop=True)


def selected_candidate_catalog() -> pd.DataFrame:
    trajectory = read_csv_required(TRAJECTORY_ATLAS_PATH)
    trajectory = trajectory[
        trajectory["robust_pre2000_testable"].map(truthy)
        & trajectory["stable_train_eval_sign"].map(truthy)
    ].copy()
    trajectory = balanced_select(
        trajectory,
        limit=TOP_TRAJECTORY_FEATURES,
        group_columns=["source_attribute"],
        per_group_limit=4,
    )

    station_attributes = read_csv_required(STATION_ATTRIBUTE_ATLAS_PATH)
    station_attributes = station_attributes[station_attributes["robust_pre2000_testable"].map(truthy)].copy()
    station_attributes = balanced_select(
        station_attributes,
        limit=TOP_STATION_ATTRIBUTE_FEATURES,
        group_columns=["attribute"],
        per_group_limit=2,
    )

    pairs = read_csv_required(PAIR_SPREAD_ATLAS_PATH)
    pairs = balanced_select(
        pairs,
        limit=TOP_PAIR_FEATURES,
        group_columns=["attribute"],
        per_group_limit=4,
    )

    used: set[str] = set()
    rows: list[dict[str, object]] = []
    rank = 1
    for source_frame, source_family in [
        (trajectory, "station_trajectory"),
        (station_attributes, "station_attribute"),
        (pairs, "station_pair_spread"),
    ]:
        for row in source_frame.to_dict("records"):
            if source_family == "station_trajectory":
                station_id = str(row["station_id"])
                feature_name = str(row["feature_name"])
                feature_id = unique_feature_id(
                    f"traj_{station_token(station_id)}_{feature_name}",
                    used,
                )
                rows.append(
                    {
                        "selected_rank": rank,
                        "feature_id": feature_id,
                        "source_family": source_family,
                        "station_id": station_id,
                        "station_a": "",
                        "station_b": "",
                        "station_ids": station_id,
                        "source_attribute": str(row["source_attribute"]),
                        "transform": str(row["transform"]),
                        "raw_feature_name": feature_name,
                        "display_name": f"{station_id} {feature_name}",
                        "source_priority_score": finite_float(row.get("priority_score")),
                        "source_eval_abs_corr": finite_float(row.get("abs_corr_eval_2000_2023_target_anomaly")),
                        "source_official_abs_corr": finite_float(row.get("abs_corr_official_error")),
                    }
                )
            elif source_family == "station_attribute":
                station_id = str(row["station_id"])
                attribute = str(row["attribute"])
                feature_id = unique_feature_id(
                    f"stat_{station_token(station_id)}_{attribute}",
                    used,
                )
                rows.append(
                    {
                        "selected_rank": rank,
                        "feature_id": feature_id,
                        "source_family": source_family,
                        "station_id": station_id,
                        "station_a": "",
                        "station_b": "",
                        "station_ids": station_id,
                        "source_attribute": attribute,
                        "transform": "current_or_network",
                        "raw_feature_name": attribute,
                        "display_name": f"{station_id} {attribute}",
                        "source_priority_score": finite_float(row.get("priority_score")),
                        "source_eval_abs_corr": finite_float(row.get("abs_corr_eval_2000_2023_target_anomaly")),
                        "source_official_abs_corr": finite_float(row.get("abs_corr_official_error")),
                    }
                )
            else:
                station_a = str(row["station_a"])
                station_b = str(row["station_b"])
                attribute = str(row["attribute"])
                feature_id = unique_feature_id(
                    f"pair_{attribute}_{station_token(station_a)}_minus_{station_token(station_b)}",
                    used,
                )
                rows.append(
                    {
                        "selected_rank": rank,
                        "feature_id": feature_id,
                        "source_family": source_family,
                        "station_id": "",
                        "station_a": station_a,
                        "station_b": station_b,
                        "station_ids": f"{station_a},{station_b}",
                        "source_attribute": attribute,
                        "transform": "station_a_minus_station_b",
                        "raw_feature_name": attribute,
                        "display_name": f"{station_a} minus {station_b} {attribute}",
                        "source_priority_score": finite_float(row.get("priority_score")),
                        "source_eval_abs_corr": finite_float(row.get("abs_corr_eval_2000_2023_target_anomaly")),
                        "source_official_abs_corr": finite_float(row.get("abs_corr_official_error")),
                    }
                )
            rank += 1
    return pd.DataFrame(rows)


def feature_sets_overlap(left: str, right: str) -> bool:
    left_set = {item for item in left.split(",") if item}
    right_set = {item for item in right.split(",") if item}
    return bool(left_set & right_set)


def should_score_pair(gate: pd.Series, response: pd.Series) -> bool:
    if gate["feature_id"] == response["feature_id"]:
        return False
    same_source_family = str(gate["source_family"]) == str(response["source_family"])
    same_attribute = str(gate["source_attribute"]) == str(response["source_attribute"])
    same_station_overlap = feature_sets_overlap(str(gate["station_ids"]), str(response["station_ids"]))
    return not (same_source_family and same_attribute and same_station_overlap)


def build_feature_frame(
    station_frame: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = (
        station_frame[
            [
                "target_date",
                "target_tmax_c",
                "past_doy_count",
                "past_doy_mean_tmax_c",
                "target_anomaly_vs_past_doy_c",
            ]
        ]
        .drop_duplicates("target_date")
        .sort_values("target_date")
        .reset_index(drop=True)
    )
    series_frames: list[pd.DataFrame] = [base]
    pair_pivots: dict[str, pd.DataFrame] = {}

    for row in catalog.itertuples(index=False):
        feature_id = str(row.feature_id)
        source_family = str(row.source_family)
        if source_family == "station_trajectory":
            feature_name = str(row.raw_feature_name)
            attribute = str(row.source_attribute)
            subset = (
                station_frame[station_frame["station_id"].astype(str).eq(str(row.station_id))]
                .sort_values("target_date")
                .reset_index(drop=True)
            )
            if attribute not in subset.columns:
                raise ValueError(f"Station attribute missing for trajectory variant: {attribute}")
            variants = station_attribute_variants(subset[attribute], attribute)
            if feature_name not in variants:
                raise ValueError(f"Selected trajectory variant missing from on-demand formulas: {feature_name}")
            series = pd.DataFrame(
                {
                    "target_date": subset["target_date"],
                    feature_id: variants[feature_name].reset_index(drop=True),
                }
            )
        elif source_family == "station_attribute":
            attribute = str(row.raw_feature_name)
            subset = station_frame[station_frame["station_id"].astype(str).eq(str(row.station_id))]
            if attribute not in subset.columns:
                raise ValueError(f"Station attribute missing from frame: {attribute}")
            series = subset[["target_date", attribute]].rename(columns={attribute: feature_id})
        else:
            attribute = str(row.raw_feature_name)
            if attribute not in pair_pivots:
                pair_pivots[attribute] = station_frame.pivot_table(
                    index="target_date",
                    columns="station_id",
                    values=attribute,
                    aggfunc="last",
                ).sort_index()
            pivot = pair_pivots[attribute]
            station_a = str(row.station_a)
            station_b = str(row.station_b)
            if station_a not in pivot.columns or station_b not in pivot.columns:
                raise ValueError(f"Pair stations missing from pivot: {station_a}, {station_b}")
            diff = pivot[station_a] - pivot[station_b]
            series = diff.rename(feature_id).reset_index()
        series = series.drop_duplicates("target_date", keep="last")
        series_frames.append(series)

    out = series_frames[0]
    for series in series_frames[1:]:
        out = out.merge(series, on="target_date", how="left")
    out = out.sort_values("target_date").reset_index(drop=True)
    require_no_confirmation_dates(out["target_date"], context="0051 feature frame")

    enriched_rows: list[dict[str, object]] = []
    for row in catalog.to_dict("records"):
        feature_id = str(row["feature_id"])
        present = out[["target_date", feature_id]].dropna()
        train_non_null = int((out["target_date"].le(TRAIN_END) & out[feature_id].notna()).sum())
        eval_non_null = int(
            (out["target_date"].ge(EVAL_START) & out["target_date"].le(EVAL_END) & out[feature_id].notna()).sum()
        )
        enriched = dict(row)
        enriched.update(
            {
                "non_null_rows": int(out[feature_id].notna().sum()),
                "train_non_null_rows": train_non_null,
                "eval_non_null_rows": eval_non_null,
                "first_target_date": str(present["target_date"].min().date()) if not present.empty else "",
                "last_target_date": str(present["target_date"].max().date()) if not present.empty else "",
            }
        )
        enriched_rows.append(enriched)
    return out, pd.DataFrame(enriched_rows)


def summarize_cell_outcome(
    frame: pd.DataFrame,
    *,
    gate_feature_id: str,
    response_feature_id: str,
    gate_edges: tuple[float, float],
    response_edges: tuple[float, float],
    outcome_column: str,
    min_rows: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    subset = frame[["target_date", outcome_column, gate_feature_id, response_feature_id]].copy()
    subset["gate_bin"] = apply_tertile_bins(subset[gate_feature_id], gate_edges)
    subset["response_bin"] = apply_tertile_bins(subset[response_feature_id], response_edges)
    subset = subset.dropna(subset=[outcome_column, "gate_bin", "response_bin"])
    if subset.empty:
        return {
            "rows": 0,
            "valid_cells": 0,
            "spread": math.nan,
            "high_cell": "",
            "low_cell": "",
            "high_cell_rows": 0,
            "low_cell_rows": 0,
            "high_cell_mean": math.nan,
            "low_cell_mean": math.nan,
        }, pd.DataFrame()

    cells = (
        subset.groupby(["gate_bin", "response_bin"], observed=True)[outcome_column]
        .agg(["count", "mean"])
        .reset_index()
    )
    cells["cell"] = cells["gate_bin"].astype(str) + "/" + cells["response_bin"].astype(str)
    valid_cells = cells[cells["count"] >= min_rows].copy()
    if len(valid_cells) < 2:
        return {
            "rows": int(len(subset)),
            "valid_cells": int(len(valid_cells)),
            "spread": math.nan,
            "high_cell": "",
            "low_cell": "",
            "high_cell_rows": 0,
            "low_cell_rows": 0,
            "high_cell_mean": math.nan,
            "low_cell_mean": math.nan,
        }, cells
    high = valid_cells.sort_values("mean", ascending=False).iloc[0]
    low = valid_cells.sort_values("mean", ascending=True).iloc[0]
    return {
        "rows": int(len(subset)),
        "valid_cells": int(len(valid_cells)),
        "spread": float(high["mean"] - low["mean"]),
        "high_cell": str(high["cell"]),
        "low_cell": str(low["cell"]),
        "high_cell_rows": int(high["count"]),
        "low_cell_rows": int(low["count"]),
        "high_cell_mean": float(high["mean"]),
        "low_cell_mean": float(low["mean"]),
    }, cells


def bin_codes_from_edges(series: pd.Series, edges: tuple[float, float]) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    codes = np.full(len(values), -1, dtype=np.int8)
    valid = np.isfinite(values)
    codes[valid & (values <= edges[0])] = 0
    codes[valid & (values > edges[0]) & (values <= edges[1])] = 1
    codes[valid & (values > edges[1])] = 2
    return codes


def safe_corr_arrays(x: np.ndarray, y: np.ndarray, *, min_rows: int) -> tuple[int, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    n_rows = int(valid.sum())
    if n_rows < min_rows:
        return n_rows, math.nan
    x_valid = x[valid].astype(float)
    y_valid = y[valid].astype(float)
    if len(np.unique(x_valid)) < 2 or len(np.unique(y_valid)) < 2:
        return n_rows, math.nan
    return n_rows, float(np.corrcoef(x_valid, y_valid)[0, 1])


def summarize_codes(
    outcome: np.ndarray,
    gate_codes: np.ndarray,
    response_codes: np.ndarray,
    *,
    min_rows: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    valid = np.isfinite(outcome) & (gate_codes >= 0) & (response_codes >= 0)
    if not bool(valid.any()):
        return {
            "rows": 0,
            "valid_cells": 0,
            "spread": math.nan,
            "high_cell": "",
            "low_cell": "",
            "high_cell_rows": 0,
            "low_cell_rows": 0,
            "high_cell_mean": math.nan,
            "low_cell_mean": math.nan,
        }, pd.DataFrame()
    joint = gate_codes[valid].astype(int) * 3 + response_codes[valid].astype(int)
    counts = np.bincount(joint, minlength=9).astype(int)
    sums = np.bincount(joint, weights=outcome[valid].astype(float), minlength=9)
    rows: list[dict[str, object]] = []
    for index in range(9):
        count = int(counts[index])
        if count <= 0:
            continue
        gate_index = index // 3
        response_index = index % 3
        mean = float(sums[index] / count)
        rows.append(
            {
                "gate_bin": BIN_LABELS[gate_index],
                "response_bin": BIN_LABELS[response_index],
                "count": count,
                "mean": mean,
                "cell": f"{BIN_LABELS[gate_index]}/{BIN_LABELS[response_index]}",
            }
        )
    cells = pd.DataFrame(rows)
    valid_cells = cells[cells["count"] >= min_rows].copy() if not cells.empty else pd.DataFrame()
    if len(valid_cells) < 2:
        return {
            "rows": int(valid.sum()),
            "valid_cells": int(len(valid_cells)),
            "spread": math.nan,
            "high_cell": "",
            "low_cell": "",
            "high_cell_rows": 0,
            "low_cell_rows": 0,
            "high_cell_mean": math.nan,
            "low_cell_mean": math.nan,
        }, cells
    high = valid_cells.sort_values("mean", ascending=False).iloc[0]
    low = valid_cells.sort_values("mean", ascending=True).iloc[0]
    return {
        "rows": int(valid.sum()),
        "valid_cells": int(len(valid_cells)),
        "spread": float(high["mean"] - low["mean"]),
        "high_cell": str(high["cell"]),
        "low_cell": str(low["cell"]),
        "high_cell_rows": int(high["count"]),
        "low_cell_rows": int(low["count"]),
        "high_cell_mean": float(high["mean"]),
        "low_cell_mean": float(low["mean"]),
    }, cells


def conditional_response_correlations_from_codes(
    response_values: np.ndarray,
    outcome: np.ndarray,
    gate_codes: np.ndarray,
) -> dict[str, object]:
    values: dict[str, object] = {}
    finite_corrs: list[float] = []
    for gate_index, gate_bin in enumerate(BIN_LABELS):
        mask = gate_codes == gate_index
        n_rows, corr = safe_corr_arrays(
            response_values[mask],
            outcome[mask],
            min_rows=MIN_CONDITIONAL_ROWS,
        )
        values[f"eval_response_corr_in_gate_{gate_bin}"] = corr
        values[f"eval_response_corr_rows_in_gate_{gate_bin}"] = n_rows
        if math.isfinite(corr):
            finite_corrs.append(corr)
    values["eval_response_conditional_corr_range"] = (
        float(max(finite_corrs) - min(finite_corrs)) if len(finite_corrs) >= 2 else math.nan
    )
    return values


def conditional_response_correlations(
    frame: pd.DataFrame,
    *,
    gate_feature_id: str,
    response_feature_id: str,
    gate_edges: tuple[float, float],
) -> dict[str, object]:
    subset = frame[["target_anomaly_vs_past_doy_c", gate_feature_id, response_feature_id]].copy()
    subset["gate_bin"] = apply_tertile_bins(subset[gate_feature_id], gate_edges)
    values: dict[str, object] = {}
    finite_corrs: list[float] = []
    for gate_bin in ["low", "mid", "high"]:
        cell = subset[subset["gate_bin"].eq(gate_bin)]
        n_rows, corr = safe_corr(
            cell[response_feature_id],
            cell["target_anomaly_vs_past_doy_c"],
            min_rows=MIN_CONDITIONAL_ROWS,
        )
        values[f"eval_response_corr_in_gate_{gate_bin}"] = corr
        values[f"eval_response_corr_rows_in_gate_{gate_bin}"] = n_rows
        if math.isfinite(corr):
            finite_corrs.append(corr)
    values["eval_response_conditional_corr_range"] = (
        float(max(finite_corrs) - min(finite_corrs)) if len(finite_corrs) >= 2 else math.nan
    )
    return values


def score_station_regime_interactions(
    feature_frame: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    official: pd.DataFrame,
) -> pd.DataFrame:
    train_mask = feature_frame["target_date"].le(TRAIN_END)
    eval_mask = feature_frame["target_date"].ge(EVAL_START) & feature_frame["target_date"].le(EVAL_END)
    eval_frame = feature_frame.loc[eval_mask].copy()
    train_frame = feature_frame.loc[train_mask].copy()
    official_feature_frame = (
        official.merge(feature_frame.drop(columns=["target_tmax_c"], errors="ignore"), on="target_date", how="left")
        if not official.empty
        else pd.DataFrame()
    )
    if not official_feature_frame.empty:
        require_no_confirmation_dates(official_feature_frame["target_date"], context="0051 official overlap")

    edge_cache: dict[str, tuple[float, float] | None] = {
        feature_id: quantile_edges_from_train(train_frame[feature_id])
        for feature_id in feature_catalog["feature_id"].astype(str)
    }
    train_outcome = pd.to_numeric(train_frame["target_anomaly_vs_past_doy_c"], errors="coerce").to_numpy(dtype=float)
    eval_outcome = pd.to_numeric(eval_frame["target_anomaly_vs_past_doy_c"], errors="coerce").to_numpy(dtype=float)
    official_error = (
        pd.to_numeric(official_feature_frame["official_error_c"], errors="coerce").to_numpy(dtype=float)
        if not official_feature_frame.empty
        else np.array([], dtype=float)
    )
    official_abs_error = (
        pd.to_numeric(official_feature_frame["official_abs_error_c"], errors="coerce").to_numpy(dtype=float)
        if not official_feature_frame.empty
        else np.array([], dtype=float)
    )
    train_bins: dict[str, np.ndarray] = {}
    eval_bins: dict[str, np.ndarray] = {}
    official_bins: dict[str, np.ndarray] = {}
    eval_values: dict[str, np.ndarray] = {}
    eval_corr_cache: dict[str, float] = {}
    for feature_id, edges in edge_cache.items():
        if edges is None:
            continue
        train_bins[feature_id] = bin_codes_from_edges(train_frame[feature_id], edges)
        eval_bins[feature_id] = bin_codes_from_edges(eval_frame[feature_id], edges)
        eval_values[feature_id] = pd.to_numeric(eval_frame[feature_id], errors="coerce").to_numpy(dtype=float)
        _, eval_corr_cache[feature_id] = safe_corr_arrays(
            eval_values[feature_id],
            eval_outcome,
            min_rows=MIN_CONDITIONAL_ROWS,
        )
        if not official_feature_frame.empty:
            official_bins[feature_id] = bin_codes_from_edges(official_feature_frame[feature_id], edges)
    catalog_lookup = feature_catalog.set_index("feature_id").to_dict("index")
    records: list[dict[str, object]] = []
    for gate in feature_catalog.to_dict("records"):
        gate_series = pd.Series(gate)
        gate_id = str(gate["feature_id"])
        gate_edges = edge_cache.get(gate_id)
        if gate_edges is None or gate_id not in train_bins:
            continue
        for response in feature_catalog.to_dict("records"):
            response_series = pd.Series(response)
            response_id = str(response["feature_id"])
            if not should_score_pair(gate_series, response_series):
                continue
            response_edges = edge_cache.get(response_id)
            if response_edges is None or response_id not in train_bins:
                continue

            train_summary, _ = summarize_codes(
                train_outcome,
                train_bins[gate_id],
                train_bins[response_id],
                min_rows=MIN_TRAIN_CELL_ROWS,
            )
            eval_summary, _ = summarize_codes(
                eval_outcome,
                eval_bins[gate_id],
                eval_bins[response_id],
                min_rows=MIN_EVAL_CELL_ROWS,
            )
            if not math.isfinite(float(eval_summary["spread"])):
                continue

            official_error_summary = {
                "rows": 0,
                "valid_cells": 0,
                "spread": math.nan,
                "high_cell": "",
                "low_cell": "",
            }
            official_abs_summary = {
                "spread": math.nan,
                "high_cell": "",
                "low_cell": "",
            }
            if not official_feature_frame.empty:
                official_error_summary, _ = summarize_codes(
                    official_error,
                    official_bins[gate_id],
                    official_bins[response_id],
                    min_rows=MIN_OFFICIAL_CELL_ROWS,
                )
                official_abs_summary, _ = summarize_codes(
                    official_abs_error,
                    official_bins[gate_id],
                    official_bins[response_id],
                    min_rows=MIN_OFFICIAL_CELL_ROWS,
                )

            gate_eval_corr = eval_corr_cache.get(gate_id, math.nan)
            response_eval_corr = eval_corr_cache.get(response_id, math.nan)
            conditional = conditional_response_correlations_from_codes(
                eval_values[response_id],
                eval_outcome,
                eval_bins[gate_id],
            )
            eval_spread = finite_float(eval_summary["spread"])
            train_spread = finite_float(train_summary["spread"])
            official_error_spread = finite_float(official_error_summary["spread"])
            official_abs_spread = finite_float(official_abs_summary["spread"])
            conditional_range = finite_float(conditional["eval_response_conditional_corr_range"])
            stable_warm_cell = (
                str(train_summary["high_cell"]) != ""
                and str(train_summary["high_cell"]) == str(eval_summary["high_cell"])
            )
            stable_cool_cell = (
                str(train_summary["low_cell"]) != ""
                and str(train_summary["low_cell"]) == str(eval_summary["low_cell"])
            )
            priority_score = (
                eval_spread
                + 0.35 * (abs(official_error_spread) if math.isfinite(official_error_spread) else 0.0)
                + 0.15 * (official_abs_spread if math.isfinite(official_abs_spread) else 0.0)
                + 0.25 * (conditional_range if math.isfinite(conditional_range) else 0.0)
                + 0.10 * (train_spread if math.isfinite(train_spread) else 0.0)
                + (0.05 if stable_warm_cell else 0.0)
                + (0.05 if stable_cool_cell else 0.0)
            )
            gate_meta = catalog_lookup[gate_id]
            response_meta = catalog_lookup[response_id]
            records.append(
                {
                    "gate_feature_id": gate_id,
                    "gate_source_family": gate_meta["source_family"],
                    "gate_station_ids": gate_meta["station_ids"],
                    "gate_source_attribute": gate_meta["source_attribute"],
                    "gate_transform": gate_meta["transform"],
                    "gate_display_name": gate_meta["display_name"],
                    "response_feature_id": response_id,
                    "response_source_family": response_meta["source_family"],
                    "response_station_ids": response_meta["station_ids"],
                    "response_source_attribute": response_meta["source_attribute"],
                    "response_transform": response_meta["transform"],
                    "response_display_name": response_meta["display_name"],
                    "gate_edges_pre2000": json.dumps(gate_edges),
                    "response_edges_pre2000": json.dumps(response_edges),
                    "train_rows_pre2000": train_summary["rows"],
                    "train_valid_cells": train_summary["valid_cells"],
                    "train_target_anomaly_spread_c": train_spread,
                    "train_warmest_cell": train_summary["high_cell"],
                    "train_coolest_cell": train_summary["low_cell"],
                    "eval_rows_2000_2023": eval_summary["rows"],
                    "eval_valid_cells": eval_summary["valid_cells"],
                    "eval_target_anomaly_spread_c": eval_spread,
                    "eval_warmest_cell": eval_summary["high_cell"],
                    "eval_warmest_cell_mean_anomaly_c": eval_summary["high_cell_mean"],
                    "eval_warmest_cell_rows": eval_summary["high_cell_rows"],
                    "eval_coolest_cell": eval_summary["low_cell"],
                    "eval_coolest_cell_mean_anomaly_c": eval_summary["low_cell_mean"],
                    "eval_coolest_cell_rows": eval_summary["low_cell_rows"],
                    "stable_warm_cell_train_eval": stable_warm_cell,
                    "stable_cool_cell_train_eval": stable_cool_cell,
                    "eval_gate_corr_target_anomaly": gate_eval_corr,
                    "eval_response_corr_target_anomaly": response_eval_corr,
                    "official_overlap_rows": official_error_summary["rows"],
                    "official_valid_error_cells": official_error_summary["valid_cells"],
                    "official_error_spread_c": official_error_spread,
                    "official_overforecast_cell": official_error_summary["high_cell"],
                    "official_underforecast_cell": official_error_summary["low_cell"],
                    "official_abs_error_spread_c": official_abs_spread,
                    "official_abs_error_high_cell": official_abs_summary["high_cell"],
                    "official_abs_error_low_cell": official_abs_summary["low_cell"],
                    **conditional,
                    "different_station_signal": not feature_sets_overlap(
                        str(gate_meta["station_ids"]),
                        str(response_meta["station_ids"]),
                    ),
                    "priority_score": priority_score,
                }
            )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("priority_score", ascending=False).reset_index(drop=True)


def build_joint_cell_atlas(
    interactions: pd.DataFrame,
    feature_frame: pd.DataFrame,
    official: pd.DataFrame,
) -> pd.DataFrame:
    top = interactions.head(TOP_CELL_INTERACTIONS).copy()
    if top.empty:
        return pd.DataFrame()
    official_feature_frame = (
        official.merge(feature_frame.drop(columns=["target_tmax_c"], errors="ignore"), on="target_date", how="left")
        if not official.empty
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []
    periods = [
        ("train_pre2000", feature_frame[feature_frame["target_date"].le(TRAIN_END)], "target_anomaly_vs_past_doy_c"),
        (
            "eval_2000_2023",
            feature_frame[
                feature_frame["target_date"].ge(EVAL_START) & feature_frame["target_date"].le(EVAL_END)
            ],
            "target_anomaly_vs_past_doy_c",
        ),
    ]
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        gate_id = str(row.gate_feature_id)
        response_id = str(row.response_feature_id)
        gate_edges = tuple(json.loads(str(row.gate_edges_pre2000)))
        response_edges = tuple(json.loads(str(row.response_edges_pre2000)))
        for period_name, period_frame, outcome in periods:
            _, cells = summarize_cell_outcome(
                period_frame,
                gate_feature_id=gate_id,
                response_feature_id=response_id,
                gate_edges=gate_edges,  # type: ignore[arg-type]
                response_edges=response_edges,  # type: ignore[arg-type]
                outcome_column=outcome,
                min_rows=1,
            )
            for cell in cells.itertuples(index=False):
                rows.append(
                    {
                        "interaction_rank": rank,
                        "period": period_name,
                        "gate_feature_id": gate_id,
                        "response_feature_id": response_id,
                        "gate_bin": str(cell.gate_bin),
                        "response_bin": str(cell.response_bin),
                        "cell": str(cell.cell),
                        "rows": int(cell.count),
                        "mean_target_anomaly_c": float(cell.mean),
                        "mean_official_error_c": math.nan,
                        "mean_official_abs_error_c": math.nan,
                    }
                )
        if not official_feature_frame.empty:
            for outcome in ["official_error_c", "official_abs_error_c"]:
                _, cells = summarize_cell_outcome(
                    official_feature_frame,
                    gate_feature_id=gate_id,
                    response_feature_id=response_id,
                    gate_edges=gate_edges,  # type: ignore[arg-type]
                    response_edges=response_edges,  # type: ignore[arg-type]
                    outcome_column=outcome,
                    min_rows=1,
                )
                for cell in cells.itertuples(index=False):
                    output = {
                        "interaction_rank": rank,
                        "period": "official_scored_current_archive",
                        "gate_feature_id": gate_id,
                        "response_feature_id": response_id,
                        "gate_bin": str(cell.gate_bin),
                        "response_bin": str(cell.response_bin),
                        "cell": str(cell.cell),
                        "rows": int(cell.count),
                        "mean_target_anomaly_c": math.nan,
                        "mean_official_error_c": math.nan,
                        "mean_official_abs_error_c": math.nan,
                    }
                    if outcome == "official_error_c":
                        output["mean_official_error_c"] = float(cell.mean)
                    else:
                        output["mean_official_abs_error_c"] = float(cell.mean)
                    rows.append(output)
    return pd.DataFrame(rows)


def top_interaction_timeseries(
    interactions: pd.DataFrame,
    feature_frame: pd.DataFrame,
    official: pd.DataFrame,
) -> pd.DataFrame:
    top = interactions.head(TOP_TIMESERIES_INTERACTIONS).copy()
    if top.empty:
        return pd.DataFrame()
    official_keep = official[
        ["target_date", "forecast_source_family", "forecast_max_c", "official_error_c", "official_abs_error_c"]
    ].copy() if not official.empty else pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        gate_id = str(row.gate_feature_id)
        response_id = str(row.response_feature_id)
        gate_edges = tuple(json.loads(str(row.gate_edges_pre2000)))
        response_edges = tuple(json.loads(str(row.response_edges_pre2000)))
        out = feature_frame[
            [
                "target_date",
                "target_tmax_c",
                "target_anomaly_vs_past_doy_c",
                gate_id,
                response_id,
            ]
        ].copy()
        out["interaction_rank"] = rank
        out["gate_feature_id"] = gate_id
        out["response_feature_id"] = response_id
        out["gate_bin"] = apply_tertile_bins(out[gate_id], gate_edges)  # type: ignore[arg-type]
        out["response_bin"] = apply_tertile_bins(out[response_id], response_edges)  # type: ignore[arg-type]
        out["joint_cell"] = out["gate_bin"].astype(str) + "/" + out["response_bin"].astype(str)
        out = out.rename(columns={gate_id: "gate_feature_value", response_id: "response_feature_value"})
        if not official_keep.empty:
            out = out.merge(official_keep, on="target_date", how="left")
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_readme(
    *,
    summary: dict[str, Any],
    feature_catalog: pd.DataFrame,
    interactions: pd.DataFrame,
    joint_cells: pd.DataFrame,
) -> str:
    top = summary.get("top_interaction", {})
    if isinstance(top, dict) and top:
        top_text = (
            f"Top interaction: gate `{top['gate_display_name']}` with response "
            f"`{top['response_display_name']}`. Eval 2000-2023 joint-cell target-anomaly spread is "
            f"`{top['eval_target_anomaly_spread_c']}` C, official-error spread on the current scored archive is "
            f"`{top['official_error_spread_c']}` C, and priority score is `{top['priority_score']}`."
        )
    else:
        top_text = "No scoreable interaction rows were produced."
    catalog_display = feature_catalog[
        [
            "selected_rank",
            "feature_id",
            "source_family",
            "station_ids",
            "source_attribute",
            "transform",
            "source_eval_abs_corr",
            "source_official_abs_corr",
            "train_non_null_rows",
            "eval_non_null_rows",
        ]
    ].copy()
    interaction_display = interactions[
        [
            "gate_display_name",
            "response_display_name",
            "train_target_anomaly_spread_c",
            "eval_target_anomaly_spread_c",
            "official_error_spread_c",
            "official_abs_error_spread_c",
            "eval_response_conditional_corr_range",
            "priority_score",
        ]
    ].head(80) if not interactions.empty else pd.DataFrame()
    cell_display = joint_cells.head(120) if not joint_cells.empty else pd.DataFrame()
    return f"""# Station Regime Interaction Atlas

Generated: `{summary['generated_at_utc']}`

## Purpose

This folder asks a deeper station-network question than `0047` and `0050`: not just "which single station signal correlates with HKG Tmax?", but "which station signal changes the meaning of another station signal?"

That matters because HKG maximum temperature is not controlled by one scalar value. A nearby station warming above its own 14-day baseline can mean different things depending on the pressure-gradient placement, marine wind context, dew-spread regime, or another station's relative pressure. This atlas therefore uses the strongest previously discovered station trajectory, station attribute, and station-pair spread features as gates and responses. It then maps 3-by-3 joint regimes whose thresholds are learned before 2000 and evaluated from 2000 through 2023.

## Leakage Control

- All station observations inherit the cutoff-safe `0047` rule: target date `T` uses station local date `T-1`, latest observation before 15:00 HKT.
- Trajectory features inherit the `0050` rule: lags, deltas, rolling means, departures, and slopes are computed only inside each station's ordered past series.
- Tertile thresholds for every gate and response feature are learned only from rows with target dates on or before `{TRAIN_END.date()}`.
- Evaluation target-anomaly spreads are measured only on `{EVAL_START.date()}` through `{EVAL_END.date()}`.
- Official forecast residual checks use only the currently parsed/scored pre-2024 official rows. They are diagnostic until the external forecast backfill fills the known archive gap.
- Rows on or after `{CONFIRMATION_START.date()}` are rejected.

## Dataset Scope

| Item | Value |
|---|---:|
| Feature rows | {summary['feature_rows']} |
| Feature first date | {summary['feature_first_date']} |
| Feature last date | {summary['feature_last_date']} |
| Selected station features | {summary['selected_feature_count']} |
| Interaction candidates scored | {summary['interaction_rows']} |
| Current official scored overlap rows | {summary['official_overlap_rows']} |
| Uses 2024+ rows | {summary['uses_2024_plus_rows']} |

## Main Result

{top_text}

## Selected Feature Inputs

These are the exact station features promoted into the interaction screen. They were selected from the already documented `0047` and `0050` atlases, with balancing so the screen does not collapse into one repeated attribute family.

{markdown_table(catalog_display, max_rows=80)}

## Top Interaction Rows

{markdown_table(interaction_display, max_rows=80)}

## Joint Cell Detail

The cell file records the actual low/mid/high by low/mid/high regimes for the top interactions. This is the human-readable map that explains which combinations are associated with hotter or cooler HKG target anomalies.

{markdown_table(cell_display, max_rows=120)}

## Interpretation

This is still feature discovery, not a final forecast model. A high row means that the combination of two station-derived signals creates a strong temperature-regime separation under thresholds that were fixed before 2000. If the same row also has a meaningful official-error spread, it becomes a candidate for later residual correction once the forecast archive is continuous.

The most useful rows are not necessarily the highest single-feature correlations. The best interactions are the rows where the gate feature changes the response feature's relationship with HKG Tmax. Those rows are the next feature families to test in strict walk-forward residual models after the backfill supplies a continuous official-anchor frame.

## Files

- `artifacts/feature_catalog.csv`
- `artifacts/interaction_scoreboard.csv`
- `artifacts/joint_cell_atlas.csv`
- `artifacts/top_interaction_timeseries.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    top = summary.get("top_interaction", {})
    top_gate = top.get("gate_display_name", "") if isinstance(top, dict) else ""
    top_response = top.get("response_display_name", "") if isinstance(top, dict) else ""
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_regime_interaction_atlas.py`:

- `{FOLDER_NAME}`: station trajectory, station attribute, and station-pair spread regime interaction atlas.

| Metric | Value |
|---|---:|
| Selected station features | {summary['selected_feature_count']} |
| Interaction rows | {summary['interaction_rows']} |
| Top gate | `{top_gate}` |
| Top response | `{top_response}` |
| Top eval spread | {summary['top_eval_target_anomaly_spread_c']} |
| Top official-error spread | {summary['top_official_error_spread_c']} |

Leakage contract: feature thresholds are learned on pre-2000 rows only; evaluation uses 2000-2023; official residual checks use currently available pre-2024 scored rows only.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Station Regime Interaction Atlas",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    top = summary.get("top_interaction", {})
    top_gate = top.get("gate_display_name", "") if isinstance(top, dict) else ""
    top_response = top.get("response_display_name", "") if isinstance(top, dict) else ""
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_regime_interaction_atlas.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Selected features | `{summary['selected_feature_count']}` from `0047` and `0050` | Audited |
| Interaction screen | `{summary['interaction_rows']}` gate/response rows | Diagnostic |
| Top gate | `{top_gate}` | Documented |
| Top response | `{top_response}` | Documented |
| Top eval spread | `{summary['top_eval_target_anomaly_spread_c']}` | 2000-2023 only |
| Top official-error spread | `{summary['top_official_error_spread_c']}` | Current scored archive only |
| Leakage guard | pre-2000 thresholds, 2000-2023 evaluation, zero 2024+ rows | Guarded |

Interpretation: `0051` turns the best station-trajectory and station-spread discoveries into concrete low/mid/high regime maps. It is a mechanism atlas for later residual modelling, not production MAE proof.
"""
    update_markdown_section(
        path,
        heading="Station Regime Interaction Atlas",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"35. Station regime interaction screening scored `{summary['interaction_rows']}` gate/response rows. "
        f"The top interaction is gate `{top_gate}` with response `{top_response}`, eval spread "
        f"`{summary['top_eval_target_anomaly_spread_c']}` and official-error spread "
        f"`{summary['top_official_error_spread_c']}`. This is feature-discovery evidence, not final MAE proof."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue local research while the forecast archive backfill runs: build `0052` candidate residual-feature design notes from `0046`-`0051`, explicitly separating deployable inputs, diagnostic-only inputs, and the first strict walk-forward model tests to run once the official forecast archive is continuous.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"

    target = load_target()
    station_frame = load_station_day_features(target)
    metadata = load_station_metadata()
    official = load_official_residuals()
    require_no_confirmation_dates(station_frame["target_date"], context="0051 station frame")
    if not official.empty:
        require_no_confirmation_dates(official["target_date"], context="0051 official frame")

    selected_catalog = selected_candidate_catalog()
    feature_frame, feature_catalog = build_feature_frame(station_frame, selected_catalog)

    meta_cols = [
        "station_id",
        "latitude",
        "longitude",
        "elevation_m",
        "distance_to_hko_km",
        "bearing_from_hko_deg",
        "coordinate_sanity_status",
    ]
    if not metadata.empty:
        station_meta = metadata[[col for col in meta_cols if col in metadata.columns]].copy()
        feature_catalog = feature_catalog.merge(
            station_meta.add_prefix("primary_"),
            left_on="station_id",
            right_on="primary_station_id",
            how="left",
        )

    interactions = score_station_regime_interactions(feature_frame, feature_catalog, official)
    joint_cells = build_joint_cell_atlas(interactions, feature_frame, official)
    timeseries = top_interaction_timeseries(interactions, feature_frame, official)
    top_interaction = interactions.iloc[0].to_dict() if not interactions.empty else {}
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "feature_rows": int(len(feature_frame)),
        "feature_first_date": str(feature_frame["target_date"].min().date()),
        "feature_last_date": str(feature_frame["target_date"].max().date()),
        "selected_feature_count": int(len(feature_catalog)),
        "interaction_rows": int(len(interactions)),
        "joint_cell_rows": int(len(joint_cells)),
        "top_interaction_timeseries_rows": int(len(timeseries)),
        "official_overlap_rows": int(len(official)),
        "uses_2024_plus_rows": False,
        "top_interaction": top_interaction,
        "top_eval_target_anomaly_spread_c": top_interaction.get("eval_target_anomaly_spread_c", math.nan)
        if top_interaction
        else math.nan,
        "top_official_error_spread_c": top_interaction.get("official_error_spread_c", math.nan)
        if top_interaction
        else math.nan,
        "leakage_guard": {
            "station_feature_timing": "target T uses station local date T-1 latest-before-1500 HKT",
            "threshold_training_period": f"target_date <= {TRAIN_END.date()}",
            "evaluation_period": f"{EVAL_START.date()} <= target_date <= {EVAL_END.date()}",
            "confirmation_start": str(CONFIRMATION_START.date()),
        },
    }
    summary = json_ready(summary)  # type: ignore[assignment]

    write_csv(artifacts / "feature_catalog.csv", feature_catalog)
    write_csv(artifacts / "interaction_scoreboard.csv", interactions)
    write_csv(artifacts / "joint_cell_atlas.csv", joint_cells)
    write_csv(artifacts / "top_interaction_timeseries.csv", timeseries)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_regime_interaction_atlas_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            feature_catalog=feature_catalog,
            interactions=interactions,
            joint_cells=joint_cells,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 station regime interaction atlas.").parse_args()


def main() -> None:
    parse_args()
    summary = run()
    compact = {
        key: summary[key]
        for key in [
            "generated_at_utc",
            "folder",
            "selected_feature_count",
            "interaction_rows",
            "top_eval_target_anomaly_spread_c",
            "top_official_error_spread_c",
            "uses_2024_plus_rows",
        ]
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
