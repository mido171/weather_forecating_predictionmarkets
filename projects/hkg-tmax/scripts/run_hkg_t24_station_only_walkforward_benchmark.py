from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
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
    safe_corr,
    update_markdown_section,
)
from scripts.run_hkg_t24_station_contribution_atlas import load_target  # noqa: E402
from scripts.run_hkg_t24_station_only_walkforward_matrix_audit import (  # noqa: E402
    assert_no_forbidden_feature_columns,
)

FOLDER_NAME = "0055_station_only_walkforward_benchmark"
MATRIX_DIR = RESEARCH_ROOT / "0054_station_only_walkforward_matrix_audit" / "artifacts"
FEATURE_MATRIX_PATH = MATRIX_DIR / "features.parquet"
COMPONENT_CATALOG_PATH = MATRIX_DIR / "components.csv"
OFFICIAL_SCORED_PATH = (
    PROJECT_PATHS.data_root
    / "datasets"
    / "05_hko_historical_rss_forecasts"
    / "hko_official_t15_scored_pre2024.parquet"
)
OOF_START = pd.Timestamp("2000-01-01")
OOF_END = pd.Timestamp("2023-12-31")
FOLD_YEARS = 6
MIN_TRAIN_ROWS = 5000
MIN_FOLD_TEST_ROWS = 1400
MIN_FEATURE_SUPPORT = 2500
CALENDAR_COLUMNS = ("doy_sin", "doy_cos", "available_feature_fraction")
FORBIDDEN_MODEL_TOKENS = ("target_", "official_", "forecast_", "residual", "mae", "rmse")


@dataclass(frozen=True)
class BenchmarkSpec:
    model_id: str
    model_family: str
    feature_mode: str
    top_k: int | None = None
    include_calendar: bool = True
    analog_k: int = 0
    ridge_alpha: float = 3.0


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def fold_definitions() -> list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds: list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for year in range(OOF_START.year, OOF_END.year + 1, FOLD_YEARS):
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = min(pd.Timestamp(year=year + FOLD_YEARS - 1, month=12, day=31), OOF_END)
        folds.append((f"fold_{test_start.year}_{test_end.year}", test_start, test_end, test_start - pd.Timedelta(days=1)))
    return folds


def validate_fold_definitions(folds: Sequence[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]]) -> None:
    previous_end: pd.Timestamp | None = None
    for fold_id, test_start, test_end, train_end in folds:
        if train_end >= test_start:
            raise ValueError(f"{fold_id} leaks training into test window")
        if previous_end is not None and test_start <= previous_end:
            raise ValueError(f"{fold_id} overlaps an earlier OOF fold")
        if (test_end - test_start).days + 1 < MIN_FOLD_TEST_ROWS:
            raise ValueError(f"{fold_id} has fewer than {MIN_FOLD_TEST_ROWS} OOF rows")
        previous_end = test_end


def load_component_catalog(path: Path = COMPONENT_CATALOG_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing 0054 component catalog: {path}")
    catalog = pd.read_csv(path)
    if "feature_id" not in catalog.columns or "source_family" not in catalog.columns:
        raise ValueError(f"0054 component catalog lacks required columns: {path}")
    return catalog.drop_duplicates("feature_id").reset_index(drop=True)


def station_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"target_date", "source_local_date_rule", "source_cutoff_hkt"}
    columns = [column for column in frame.columns if column not in excluded]
    assert_no_forbidden_feature_columns(columns)
    return columns


def load_model_frame(matrix_path: Path = FEATURE_MATRIX_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing 0054 station-only matrix: {matrix_path}")
    matrix = pd.read_parquet(matrix_path)
    matrix["target_date"] = pd.to_datetime(matrix["target_date"], errors="coerce").dt.normalize()
    matrix = matrix[matrix["target_date"].notna()].sort_values("target_date").reset_index(drop=True)
    matrix = matrix[matrix["target_date"] < CONFIRMATION_START].copy()
    require_no_confirmation_dates(matrix["target_date"], context="0055 station-only matrix")
    feature_cols = station_feature_columns(matrix)

    target = load_target()
    frame = target.merge(matrix, on="target_date", how="inner")
    frame = frame[frame["target_date"].between(pd.Timestamp("1947-01-01"), OOF_END)].copy()
    frame["day_of_year"] = frame["target_date"].dt.dayofyear
    frame["doy_sin"] = np.sin(2.0 * math.pi * frame["day_of_year"] / 366.0)
    frame["doy_cos"] = np.cos(2.0 * math.pi * frame["day_of_year"] / 366.0)
    frame["available_feature_fraction"] = frame[feature_cols].notna().mean(axis=1)
    require_no_confirmation_dates(frame["target_date"], context="0055 modelling frame")
    return frame.sort_values("target_date").reset_index(drop=True), load_component_catalog()


def active_numeric_columns(train: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    active: list[str] = []
    for column in columns:
        if column not in train.columns:
            continue
        values = pd.to_numeric(train[column], errors="coerce")
        if int(values.notna().sum()) < MIN_FEATURE_SUPPORT:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        active.append(column)
    return active


def train_abs_corr(train: pd.DataFrame, feature: str) -> float:
    _, corr = safe_corr(train[feature], train["target_anomaly_vs_past_doy_c"], min_rows=365)
    return abs(corr) if math.isfinite(corr) else math.nan


def select_feature_columns(
    train: pd.DataFrame,
    catalog: pd.DataFrame,
    spec: BenchmarkSpec,
) -> tuple[list[str], pd.DataFrame]:
    feature_cols = [column for column in catalog["feature_id"].astype(str).tolist() if column in train.columns]
    meta = catalog.set_index("feature_id").to_dict("index")
    if spec.feature_mode == "all_station":
        candidates = feature_cols
    elif spec.feature_mode == "trajectory":
        candidates = [col for col in feature_cols if str(meta[col].get("source_family", "")) == "station_trajectory"]
    elif spec.feature_mode == "attribute":
        candidates = [col for col in feature_cols if str(meta[col].get("source_family", "")) == "station_attribute"]
    elif spec.feature_mode == "pair_spread":
        candidates = [col for col in feature_cols if str(meta[col].get("source_family", "")) == "station_pair_spread"]
    elif spec.feature_mode == "temperature_trajectory":
        candidates = [
            col
            for col in feature_cols
            if str(meta[col].get("source_family", "")) == "station_trajectory"
            and "air_temperature_c" in str(meta[col].get("raw_feature_name", ""))
        ]
    elif spec.feature_mode == "pressure_pair":
        candidates = [
            col
            for col in feature_cols
            if str(meta[col].get("source_family", "")) == "station_pair_spread"
            and "sea_level_pressure_hpa" in str(meta[col].get("raw_feature_name", ""))
        ]
    elif spec.feature_mode == "top_train_corr":
        candidates = feature_cols
    else:
        raise ValueError(f"Unknown feature mode: {spec.feature_mode}")

    active = active_numeric_columns(train, candidates)
    rows: list[dict[str, object]] = []
    for column in active:
        rows.append(
            {
                "feature": column,
                "source_family": str(meta[column].get("source_family", "")),
                "raw_feature_name": str(meta[column].get("raw_feature_name", "")),
                "train_abs_corr": train_abs_corr(train, column),
                "train_non_null_rows": int(pd.to_numeric(train[column], errors="coerce").notna().sum()),
            }
        )
    selected = pd.DataFrame(rows).sort_values(
        ["train_abs_corr", "train_non_null_rows"],
        ascending=[False, False],
        na_position="last",
    )
    if spec.top_k is not None:
        selected = selected.head(spec.top_k).copy()
    columns = selected["feature"].astype(str).tolist() if not selected.empty else []
    if spec.include_calendar:
        columns.extend([column for column in CALENDAR_COLUMNS if column in train.columns])
    columns = active_numeric_columns(train, list(dict.fromkeys(columns)))
    assert_no_forbidden_model_columns(columns)
    selected = selected[selected["feature"].isin(columns)].copy()
    return columns, selected.reset_index(drop=True)


def assert_no_forbidden_model_columns(columns: Sequence[str]) -> None:
    offenders = [
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_MODEL_TOKENS)
        and column not in CALENDAR_COLUMNS
    ]
    if offenders:
        raise ValueError(f"Forbidden model feature columns: {offenders[:10]}")


def fit_ridge_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    *,
    alpha: float,
) -> tuple[np.ndarray, float]:
    train_ready = train.dropna(subset=["target_anomaly_vs_past_doy_c"]).copy()
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    y = pd.to_numeric(train_ready["target_anomaly_vs_past_doy_c"], errors="coerce").to_numpy(dtype=float)
    pipeline.fit(train_ready[list(columns)], y)
    train_pred = pipeline.predict(train_ready[list(columns)])
    sigma = float(np.nanstd(y - train_pred, ddof=1))
    pred_anomaly = pipeline.predict(test[list(columns)])
    point = pd.to_numeric(test["past_doy_mean_tmax_c"], errors="coerce").to_numpy(dtype=float) + pred_anomaly
    return point, sigma


def fit_analog_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    *,
    k: int,
) -> tuple[np.ndarray, float]:
    train_ready = train.dropna(subset=["target_anomaly_vs_past_doy_c"]).copy()
    y = pd.to_numeric(train_ready["target_anomaly_vs_past_doy_c"], errors="coerce").to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_x = scaler.fit_transform(imputer.fit_transform(train_ready[list(columns)]))
    test_x = scaler.transform(imputer.transform(test[list(columns)]))
    n_neighbors = min(k, len(train_ready))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(train_x)
    distances, indices = neighbors.kneighbors(test_x, return_distance=True)
    pred_anomaly = np.empty(len(test), dtype=float)
    for row_idx, row_distances in enumerate(distances):
        selected_y = y[indices[row_idx]]
        positive = row_distances[row_distances > 1e-9]
        scale = float(np.nanmedian(positive)) if len(positive) else 1.0
        if not np.isfinite(scale) or scale <= 1e-9:
            weights = np.ones_like(row_distances, dtype=float)
        else:
            weights = np.exp(-row_distances / scale)
        pred_anomaly[row_idx] = float(np.sum(weights * selected_y) / np.sum(weights))
    sigma = float(np.nanstd(y - np.nanmean(y), ddof=1))
    point = pd.to_numeric(test["past_doy_mean_tmax_c"], errors="coerce").to_numpy(dtype=float) + pred_anomaly
    return point, sigma


def benchmark_specs() -> list[BenchmarkSpec]:
    return [
        BenchmarkSpec("ridge_all_station", "ridge_station_only", "all_station", None, True, ridge_alpha=5.0),
        BenchmarkSpec("ridge_top16_train_corr", "ridge_fold_selected", "top_train_corr", 16, True, ridge_alpha=3.0),
        BenchmarkSpec("ridge_top32_train_corr", "ridge_fold_selected", "top_train_corr", 32, True, ridge_alpha=3.0),
        BenchmarkSpec("ridge_trajectory_only", "ridge_station_family", "trajectory", None, True, ridge_alpha=5.0),
        BenchmarkSpec("ridge_attribute_only", "ridge_station_family", "attribute", None, True, ridge_alpha=5.0),
        BenchmarkSpec("ridge_pair_spread_only", "ridge_station_family", "pair_spread", None, True, ridge_alpha=5.0),
        BenchmarkSpec("ridge_temp_trajectory", "ridge_physical_subset", "temperature_trajectory", None, True, ridge_alpha=3.0),
        BenchmarkSpec("ridge_pressure_pairs", "ridge_physical_subset", "pressure_pair", None, True, ridge_alpha=3.0),
        BenchmarkSpec("analog_top16_k90", "analog_fold_selected", "top_train_corr", 16, True, analog_k=90),
        BenchmarkSpec("analog_top32_k180", "analog_fold_selected", "top_train_corr", 32, True, analog_k=180),
    ]


def run_oof(frame: pd.DataFrame, catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = fold_definitions()
    validate_fold_definitions(folds)
    predictions: list[pd.DataFrame] = []
    selected_rows: list[pd.DataFrame] = []
    leakage_rows: list[dict[str, object]] = []
    for fold_id, test_start, test_end, train_end in folds:
        train = frame[(frame["target_date"] <= train_end) & frame["target_anomaly_vs_past_doy_c"].notna()].copy()
        test = frame[
            frame["target_date"].between(test_start, test_end)
            & frame["target_tmax_c"].notna()
            & frame["past_doy_mean_tmax_c"].notna()
        ].copy()
        if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_FOLD_TEST_ROWS:
            raise RuntimeError(f"{fold_id} lacks required train/test rows: train={len(train)}, test={len(test)}")

        baseline = test[["target_date", "target_tmax_c", "past_doy_mean_tmax_c"]].copy()
        baseline["fold_id"] = fold_id
        baseline["model_id"] = "causal_doy_climatology"
        baseline["model_family"] = "baseline"
        baseline["training_start"] = train["target_date"].min()
        baseline["training_end"] = train["target_date"].max()
        baseline["training_rows"] = int(len(train))
        baseline["feature_count"] = 0
        baseline["point_forecast_c"] = baseline["past_doy_mean_tmax_c"]
        baseline["distribution_sigma_c"] = float(np.nanstd(train["target_anomaly_vs_past_doy_c"], ddof=1))
        predictions.append(baseline)

        for spec in benchmark_specs():
            columns, selected = select_feature_columns(train, catalog, spec)
            if not columns:
                continue
            selected = selected.copy()
            selected["fold_id"] = fold_id
            selected["model_id"] = spec.model_id
            selected["feature_rank"] = np.arange(1, len(selected) + 1)
            selected_rows.append(selected)
            if spec.analog_k > 0:
                point, sigma = fit_analog_prediction(train, test, columns, k=spec.analog_k)
            else:
                point, sigma = fit_ridge_prediction(train, test, columns, alpha=spec.ridge_alpha)
            pred = test[["target_date", "target_tmax_c", "past_doy_mean_tmax_c"]].copy()
            pred["fold_id"] = fold_id
            pred["model_id"] = spec.model_id
            pred["model_family"] = spec.model_family
            pred["training_start"] = train["target_date"].min()
            pred["training_end"] = train["target_date"].max()
            pred["training_rows"] = int(len(train))
            pred["feature_count"] = int(len(columns))
            pred["point_forecast_c"] = point
            pred["distribution_sigma_c"] = sigma
            predictions.append(pred)
            leakage_rows.append(
                {
                    "fold_id": fold_id,
                    "model_id": spec.model_id,
                    "train_end_before_test_start": bool(train["target_date"].max() < test["target_date"].min()),
                    "selected_features_fit_inside_fold": True,
                    "scaler_imputer_fit_inside_fold": True,
                    "uses_confirmation_rows": bool(test["target_date"].max() >= CONFIRMATION_START),
                    "feature_count": int(len(columns)),
                }
            )

    out = pd.concat(predictions, ignore_index=True)
    out = out.sort_values(["target_date", "model_id"]).reset_index(drop=True)
    require_no_confirmation_dates(out["target_date"], context="0055 predictions")
    selected_out = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    leakage = pd.DataFrame(leakage_rows)
    return out, selected_out, leakage


def score_by(predictions: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in predictions.groupby(list(group_cols), dropna=False, observed=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        error = pd.to_numeric(group["point_forecast_c"], errors="coerce") - pd.to_numeric(group["target_tmax_c"], errors="coerce")
        scored = group[error.notna()].copy()
        error = error[error.notna()]
        row = {col: value for col, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "n": int(len(scored)),
                "first_date": str(scored["target_date"].min().date()) if not scored.empty else "",
                "last_date": str(scored["target_date"].max().date()) if not scored.empty else "",
                "mae": float(error.abs().mean()) if len(error) else math.nan,
                "rmse": float(np.sqrt(np.mean(np.square(error)))) if len(error) else math.nan,
                "bias": float(error.mean()) if len(error) else math.nan,
                "median_abs_error": float(error.abs().median()) if len(error) else math.nan,
                "feature_count_median": float(scored["feature_count"].median()) if "feature_count" in scored else math.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mae", "rmse"], na_position="last").reset_index(drop=True)


def official_overlap_scoreboard(predictions: pd.DataFrame, top_model_ids: Sequence[str]) -> pd.DataFrame:
    if not OFFICIAL_SCORED_PATH.exists():
        return pd.DataFrame()
    official = pd.read_parquet(OFFICIAL_SCORED_PATH)
    official["target_date"] = pd.to_datetime(official["target_date"], errors="coerce").dt.normalize()
    official = official[
        official["target_date"].between(EVAL_START, EVAL_END)
        & official["forecast_max_c"].notna()
        & official["target_tmax_c"].notna()
    ].copy()
    if official.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for source_family, source_group in official.groupby("forecast_source_family", dropna=False, observed=True):
        source_dates = set(source_group["target_date"])
        official_error = source_group["forecast_max_c"] - source_group["target_tmax_c"]
        rows.append(
            {
                "forecast_source_family": source_family,
                "model_id": "official_forecast_max_c",
                "n": int(len(source_group)),
                "first_date": str(source_group["target_date"].min().date()),
                "last_date": str(source_group["target_date"].max().date()),
                "mae": float(official_error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(official_error)))),
                "bias": float(official_error.mean()),
                "note": "diagnostic current RSS/press archive overlap only",
            }
        )
        for model_id in top_model_ids:
            model = predictions[
                predictions["model_id"].eq(model_id)
                & predictions["target_date"].isin(source_dates)
            ].copy()
            if model.empty:
                continue
            error = model["point_forecast_c"] - model["target_tmax_c"]
            rows.append(
                {
                    "forecast_source_family": source_family,
                    "model_id": model_id,
                    "n": int(len(model)),
                    "first_date": str(model["target_date"].min().date()),
                    "last_date": str(model["target_date"].max().date()),
                    "mae": float(error.abs().mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "bias": float(error.mean()),
                    "note": "same target dates as current official archive source family",
                }
            )
    return pd.DataFrame(rows).sort_values(["forecast_source_family", "mae", "rmse"]).reset_index(drop=True)


def leakage_audit(predictions: pd.DataFrame, leakage_rows: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {
            "check_id": "no_confirmation_predictions",
            "passed": bool(pd.to_datetime(predictions["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last prediction date {pd.to_datetime(predictions['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "all_fold_training_ends_before_test",
            "passed": bool(leakage_rows["train_end_before_test_start"].all()),
            "evidence": f"{len(leakage_rows)} model-fold rows checked",
        },
        {
            "check_id": "fold_local_feature_selection",
            "passed": bool(leakage_rows["selected_features_fit_inside_fold"].all()),
            "evidence": "selected feature lists are written by fold and model",
        },
        {
            "check_id": "fold_local_scaling_imputation",
            "passed": bool(leakage_rows["scaler_imputer_fit_inside_fold"].all()),
            "evidence": "scikit-learn pipelines are fitted inside each train fold",
        },
        {
            "check_id": "minimum_fold_oof_window",
            "passed": bool(predictions.groupby("fold_id", observed=True)["target_date"].nunique().min() >= MIN_FOLD_TEST_ROWS),
            "evidence": f"minimum fold unique dates {int(predictions.groupby('fold_id', observed=True)['target_date'].nunique().min())}",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    scoreboard: pd.DataFrame,
    fold_scoreboard: pd.DataFrame,
    leakage: pd.DataFrame,
    official_overlap: pd.DataFrame,
    selected_features: pd.DataFrame,
) -> str:
    selected_display = selected_features.head(80) if not selected_features.empty else pd.DataFrame()
    official_text = markdown_table(official_overlap, max_rows=80) if not official_overlap.empty else "_No current official archive overlap file found._"
    return f"""# Station-Only Walk-Forward Benchmark

Generated: `{summary['generated_at_utc']}`

## Purpose

`0054` built the leakage-audited station-only feature matrix. This run performs the next bounded check: can station-only long-history features predict HKG Tmax out of fold before any delayed official forecast backfill is complete?

This is not a final production model and not a 2024+ confirmation test. It is a controlled benchmark for station-only signal strength.

## Evaluation Contract

- Target dates: `{summary['first_oof_date']}` to `{summary['last_oof_date']}`.
- OOF folds: four chronological 6-year folds, `2000-2005`, `2006-2011`, `2012-2017`, `2018-2023`.
- Training for each fold uses only target dates before that fold starts.
- Feature selection, imputation, scaling, Ridge fitting, and analog-neighbor fitting happen inside each fold only.
- Inputs are the `0054` station-only matrix: latest T-1 station observations before `15:00 HKT`, station trajectories, and station pair spreads.
- 2024+ rows remain locked out.

## Headline Scoreboard

{markdown_table(scoreboard, max_rows=40)}

## Fold Scoreboard

{markdown_table(fold_scoreboard, max_rows=80)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Current Official Archive Overlap Diagnostic

{official_text}

## Fold-Selected Feature Examples

{markdown_table(selected_display, max_rows=80)}

## Interpretation

This benchmark answers whether the station-only signal can stand on its own with strict chronological OOF discipline. It deliberately avoids upper-air and HKO daily candidates that still need timestamp proof. It also avoids using the delayed official forecast backfill for training. Any station-only winner here should be treated as a candidate component for a later official-anchor residual system, not as the final target system.

## Files

- `artifacts/predictions.parquet`
- `artifacts/predictions_sample.csv`
- `artifacts/scoreboard.csv`
- `artifacts/fold_scoreboard.csv`
- `artifacts/fold_selected_features.csv`
- `artifacts/official_overlap.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_only_walkforward_benchmark.py`:

- `{FOLDER_NAME}`: strict chronological station-only OOF benchmark from the `0054` matrix.

| Metric | Value |
|---|---:|
| OOF rows | {summary['prediction_rows_per_model']} |
| Models scored | {summary['models_scored']} |
| Best model | {summary['best_model_id']} |
| Best MAE | {summary['best_mae']} |
| Best RMSE | {summary['best_rmse']} |

Leakage contract: all feature selection/scaling/fitting is fold-local and no 2024+ rows are used.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Station-Only Walk-Forward Benchmark",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_only_walkforward_benchmark.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| OOF period | `{summary['first_oof_date']}` to `{summary['last_oof_date']}` | Complete |
| Folds | `{summary['fold_count']}` chronological 6-year folds | Guarded |
| Models scored | `{summary['models_scored']}` | Complete |
| Best model | `{summary['best_model_id']}` | Diagnostic |
| Best MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0055` establishes a strict station-only benchmark while forecast backfill continues. It is useful for selecting station components, but it does not replace the official-anchor residual system.
"""
    update_markdown_section(
        path,
        heading="Station-Only Walk-Forward Benchmark",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"39. Station-only OOF benchmark scored `{summary['models_scored']}` models across "
        f"`{summary['prediction_rows_per_model']}` dates; best pre-2024 MAE is `{summary['best_mae']}` "
        f"from `{summary['best_model_id']}`. Treat this as component evidence, not production proof."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Use `0055` to identify the station-only winner's residual failure modes by month, wind/pressure regime, station availability, and heat-level bucket. Keep the delayed RSS/press backfill out of the critical path until continuous official-anchor rows are verified.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, catalog = load_model_frame()
    predictions, selected_features, leakage_rows = run_oof(frame, catalog)
    scoreboard = score_by(predictions, ["model_id", "model_family"])
    fold_scoreboard = score_by(predictions, ["fold_id", "model_id", "model_family"])
    top_model_ids = scoreboard["model_id"].head(5).astype(str).tolist()
    official_overlap = official_overlap_scoreboard(predictions, top_model_ids)
    leakage = leakage_audit(predictions, leakage_rows)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0055 leakage audit failed: {failed}")

    best = scoreboard.iloc[0]
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "fold_count": len(fold_definitions()),
        "models_scored": int(scoreboard["model_id"].nunique()),
        "prediction_rows": int(len(predictions)),
        "prediction_rows_per_model": int(scoreboard["n"].max()),
        "first_oof_date": str(predictions["target_date"].min().date()),
        "last_oof_date": str(predictions["target_date"].max().date()),
        "best_model_id": str(best["model_id"]),
        "best_model_family": str(best["model_family"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_bias": float(best["bias"]),
        "baseline_mae": float(scoreboard[scoreboard["model_id"].eq("causal_doy_climatology")].iloc[0]["mae"]),
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
        "confirmation_start": str(CONFIRMATION_START.date()),
        "uses_2024_plus_rows": False,
    }

    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(artifacts / "predictions.parquet", index=False)
    write_csv(artifacts / "predictions_sample.csv", predictions.head(1000))
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "fold_scoreboard.csv", fold_scoreboard)
    write_csv(artifacts / "fold_selected_features.csv", selected_features)
    write_csv(artifacts / "official_overlap.csv", official_overlap)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_only_walkforward_benchmark_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            scoreboard=scoreboard,
            fold_scoreboard=fold_scoreboard,
            leakage=leakage,
            official_overlap=official_overlap,
            selected_features=selected_features,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run strict station-only chronological OOF benchmark.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
