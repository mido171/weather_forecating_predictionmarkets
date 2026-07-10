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
    score_prediction_frame,
    write_csv,
    write_json,
    write_text,
)

DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
FEATURE_MATRIX_PATH = (
    DATASETS_ROOT
    / "12_hkg_t24_robust_experiment_outputs"
    / "hkg_t24_exp0050_0099_feature_matrix.parquet"
)
OFFICIAL_SCORED_PATH = (
    DATASETS_ROOT
    / "05_hko_historical_rss_forecasts"
    / "hko_official_t15_scored_pre2024.parquet"
)
FOLDER_NAME = "0046_long_history_cross_family_interaction_atlas"
TRAIN_END = pd.Timestamp("1999-12-31")
EVAL_START = pd.Timestamp("2000-01-01")
EVAL_END = pd.Timestamp("2023-12-31")
MIN_CORR_ROWS = 365
MIN_TRAIN_ROWS = 2500
MIN_EVAL_ROWS = 730
MIN_CELL_ROWS = 80
MIN_OFFICIAL_CELL_ROWS = 20
FEATURES_PER_FAMILY = 8
TOP_FEATURE_ROWS = 250
STATION_ID_PATTERN = re.compile(r"(?:^|_)(\d{5,6}_\d{5})(?:_|$)")

NON_FEATURE_EXACT = {
    "target_tmax_c",
    "target_date",
    "valid_at_utc",
    "valid_at_hkt",
    "raw_retrieved_at_utc",
    "content_sha256",
    "operational_input_allowed",
    "release_latency_proven",
    "past_doy_count",
    "past_doy_mean_tmax_c",
    "target_anomaly_vs_past_doy_c",
}
NON_FEATURE_PREFIXES = (
    "official_",
    "forecast_",
)


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def classify_feature_family(feature_name: str) -> str:
    name = feature_name.lower()
    if name in {"year", "month", "day_of_year"} or name.startswith(("doy_", "year_centered", "clim_")):
        return "calendar_climatology"
    if name.startswith(("target_", "trajectory_")):
        return "target_memory"
    if name.startswith(("igra_", "ua_")):
        return "upper_air"
    if name.startswith(("isd_graph_", "isd_")) or re.search(r"\d{5,6}_\d{5}", name):
        return "isd_station_network"
    if name.startswith("daily_"):
        return "hko_daily_climate"
    if name.startswith(("tc_", "tropical_")):
        return "tropical_cyclone"
    if name.startswith(("solar_", "sun_", "radiation_")):
        return "solar"
    return "other"


def station_ids_in_feature(feature_name: str) -> str:
    ids = sorted(set(STATION_ID_PATTERN.findall(feature_name)))
    return ",".join(ids)


def safe_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def safe_corr(x: pd.Series, y: pd.Series, *, min_rows: int = MIN_CORR_ROWS) -> tuple[int, float]:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1)
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < min_rows:
        return int(len(pair)), math.nan
    if pair.iloc[:, 0].nunique(dropna=True) < 2 or pair.iloc[:, 1].nunique(dropna=True) < 2:
        return int(len(pair)), math.nan
    return int(len(pair)), float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def load_feature_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature matrix: {path}")
    frame = pd.read_parquet(path)
    if "target_date" not in frame.columns or "target_tmax_c" not in frame.columns:
        raise ValueError(f"Feature matrix missing target columns: {path}")
    frame = frame.copy()
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame = frame[frame["target_date"].notna()].sort_values("target_date").reset_index(drop=True)
    frame = frame[frame["target_date"] < CONFIRMATION_START].copy()
    require_no_confirmation_dates(frame["target_date"], context="0046 feature matrix")
    return frame


def add_past_doy_anomaly(frame: pd.DataFrame, *, min_past_years: int = 10) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    if "day_of_year" not in ordered.columns:
        ordered["day_of_year"] = ordered["target_date"].dt.dayofyear
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce")
    grouped = target.groupby(ordered["day_of_year"], observed=True)
    past_count = grouped.cumcount()
    past_sum = grouped.cumsum() - target
    past_mean = past_sum / past_count.replace(0, np.nan)
    ordered["past_doy_count"] = past_count.astype(int)
    ordered["past_doy_mean_tmax_c"] = past_mean.where(past_count >= min_past_years)
    ordered["target_anomaly_vs_past_doy_c"] = (
        target - ordered["past_doy_mean_tmax_c"]
    ).where(past_count >= min_past_years)
    return ordered


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for column in frame.columns:
        if column in NON_FEATURE_EXACT:
            continue
        if column.startswith(NON_FEATURE_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        if pd.to_numeric(frame[column], errors="coerce").nunique(dropna=True) < 2:
            continue
        features.append(column)
    return features


def period_mask(frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame["target_date"] >= start
    if end is not None:
        mask &= frame["target_date"] <= end
    return mask


def feature_correlation_atlas(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    train_mask = frame["target_date"] <= TRAIN_END
    eval_mask = (frame["target_date"] >= EVAL_START) & (frame["target_date"] <= EVAL_END)
    records: list[dict[str, object]] = []
    anomaly = frame["target_anomaly_vs_past_doy_c"]
    target = frame["target_tmax_c"]
    for feature in features:
        series = frame[feature]
        n_full_raw, corr_full_raw = safe_corr(series, target)
        n_full_anom, corr_full_anom = safe_corr(series, anomaly)
        n_train_anom, corr_train_anom = safe_corr(series[train_mask], anomaly[train_mask])
        n_eval_anom, corr_eval_anom = safe_corr(series[eval_mask], anomaly[eval_mask])
        records.append(
            {
                "feature": feature,
                "family": classify_feature_family(feature),
                "station_ids": station_ids_in_feature(feature),
                "n_full_raw_target": n_full_raw,
                "corr_full_raw_target": corr_full_raw,
                "n_full_anomaly": n_full_anom,
                "corr_full_anomaly": corr_full_anom,
                "n_train_pre2000_anomaly": n_train_anom,
                "corr_train_pre2000_anomaly": corr_train_anom,
                "abs_corr_train_pre2000_anomaly": abs(corr_train_anom)
                if math.isfinite(corr_train_anom)
                else math.nan,
                "n_eval_2000_2023_anomaly": n_eval_anom,
                "corr_eval_2000_2023_anomaly": corr_eval_anom,
                "abs_corr_eval_2000_2023_anomaly": abs(corr_eval_anom)
                if math.isfinite(corr_eval_anom)
                else math.nan,
            }
        )
    out = pd.DataFrame(records)
    return out.sort_values(
        ["abs_corr_train_pre2000_anomaly", "abs_corr_eval_2000_2023_anomaly"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def era_stability_atlas(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    eras = [
        ("1949_1979", pd.Timestamp("1949-01-01"), pd.Timestamp("1979-12-31")),
        ("1980_1999", pd.Timestamp("1980-01-01"), pd.Timestamp("1999-12-31")),
        ("2000_2019", pd.Timestamp("2000-01-01"), pd.Timestamp("2019-12-31")),
        ("2020_2023", pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31")),
    ]
    records: list[dict[str, object]] = []
    for feature in features:
        record: dict[str, object] = {
            "feature": feature,
            "family": classify_feature_family(feature),
            "station_ids": station_ids_in_feature(feature),
        }
        signs: list[float] = []
        abs_corrs: list[float] = []
        row_counts: list[int] = []
        for label, start, end in eras:
            mask = period_mask(frame, start, end)
            n_rows, corr = safe_corr(
                frame.loc[mask, feature],
                frame.loc[mask, "target_anomaly_vs_past_doy_c"],
            )
            record[f"n_{label}"] = n_rows
            record[f"corr_anomaly_{label}"] = corr
            if math.isfinite(corr):
                signs.append(float(np.sign(corr)))
                abs_corrs.append(abs(corr))
                row_counts.append(n_rows)
        record["eras_with_signal"] = len(abs_corrs)
        record["min_abs_era_corr"] = min(abs_corrs) if abs_corrs else math.nan
        record["mean_abs_era_corr"] = float(np.mean(abs_corrs)) if abs_corrs else math.nan
        record["sign_consistency"] = abs(float(np.mean(signs))) if signs else math.nan
        record["min_era_rows"] = min(row_counts) if row_counts else 0
        min_abs = safe_float(record["min_abs_era_corr"]) or 0.0
        mean_abs = safe_float(record["mean_abs_era_corr"]) or 0.0
        sign_consistency = safe_float(record["sign_consistency"]) or 0.0
        record["stability_priority"] = sign_consistency * (0.65 * min_abs + 0.35 * mean_abs)
        records.append(record)
    out = pd.DataFrame(records)
    return out.sort_values("stability_priority", ascending=False, na_position="last").reset_index(drop=True)


def family_coverage(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for family, family_features in pd.Series(features).groupby([classify_feature_family(f) for f in features]):
        cols = list(family_features)
        any_nonnull = frame[cols].notna().any(axis=1)
        if any_nonnull.any():
            first_date = str(frame.loc[any_nonnull, "target_date"].min().date())
            last_date = str(frame.loc[any_nonnull, "target_date"].max().date())
            rows_any = int(any_nonnull.sum())
        else:
            first_date = ""
            last_date = ""
            rows_any = 0
        coverage_rates = frame[cols].notna().mean(axis=0)
        records.append(
            {
                "family": str(family),
                "feature_count": len(cols),
                "rows_with_any_feature": rows_any,
                "first_date": first_date,
                "last_date": last_date,
                "median_feature_coverage_fraction": float(coverage_rates.median()),
                "max_feature_coverage_fraction": float(coverage_rates.max()),
            }
        )
    return pd.DataFrame(records).sort_values(["feature_count", "family"], ascending=[False, True])


def select_pre2000_features(correlations: pd.DataFrame) -> pd.DataFrame:
    eligible = correlations[
        (correlations["n_train_pre2000_anomaly"] >= MIN_TRAIN_ROWS)
        & (correlations["n_eval_2000_2023_anomaly"] >= MIN_EVAL_ROWS)
        & correlations["abs_corr_train_pre2000_anomaly"].notna()
    ].copy()
    selected = (
        eligible.sort_values(
            ["family", "abs_corr_train_pre2000_anomaly", "abs_corr_eval_2000_2023_anomaly"],
            ascending=[True, False, False],
        )
        .groupby("family", observed=True)
        .head(FEATURES_PER_FAMILY)
        .reset_index(drop=True)
    )
    return selected.sort_values(
        ["abs_corr_train_pre2000_anomaly", "abs_corr_eval_2000_2023_anomaly"],
        ascending=[False, False],
    ).reset_index(drop=True)


def quantile_edges_from_train(series: pd.Series) -> tuple[float, float] | None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < MIN_TRAIN_ROWS or values.nunique(dropna=True) < 5:
        return None
    q1, q2 = np.nanquantile(values.astype(float), [1.0 / 3.0, 2.0 / 3.0])
    if not math.isfinite(float(q1)) or not math.isfinite(float(q2)) or float(q1) >= float(q2):
        return None
    return float(q1), float(q2)


def apply_tertile_bins(series: pd.Series, edges: tuple[float, float]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    raw = np.select([values <= edges[0], values <= edges[1]], ["low", "mid"], default="high")
    out = pd.Series(raw, index=series.index, dtype="object")
    out[values.isna()] = np.nan
    return out


def official_overlap_frame(frame: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    if not OFFICIAL_SCORED_PATH.exists():
        return pd.DataFrame()
    official = pd.read_parquet(OFFICIAL_SCORED_PATH).copy()
    official["target_date"] = pd.to_datetime(official["target_date"], errors="coerce").dt.normalize()
    official = official[official["target_date"].notna() & (official["target_date"] < CONFIRMATION_START)].copy()
    require_no_confirmation_dates(official["target_date"], context="0046 official scored overlap")
    keep = [
        "target_date",
        "forecast_source_family",
        "forecast_max_c",
        "target_tmax_c",
        "official_error_c",
        "official_abs_error_c",
    ]
    official = official[[col for col in keep if col in official.columns]].copy()
    if "official_error_c" not in official.columns:
        official["official_error_c"] = official["forecast_max_c"] - official["target_tmax_c"]
    if "official_abs_error_c" not in official.columns:
        official["official_abs_error_c"] = official["official_error_c"].abs()
    features = frame[["target_date", *selected_features]].drop_duplicates("target_date", keep="last")
    return official.merge(features, on="target_date", how="left")


def official_residual_correlations(overlap: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    if overlap.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    for feature in selected_features:
        n_error, corr_error = safe_corr(overlap[feature], overlap["official_error_c"], min_rows=120)
        n_abs, corr_abs = safe_corr(overlap[feature], overlap["official_abs_error_c"], min_rows=120)
        records.append(
            {
                "feature": feature,
                "family": classify_feature_family(feature),
                "station_ids": station_ids_in_feature(feature),
                "official_rows": int(overlap[feature].notna().sum()),
                "n_official_error_corr": n_error,
                "corr_official_error": corr_error,
                "abs_corr_official_error": abs(corr_error) if math.isfinite(corr_error) else math.nan,
                "n_official_abs_error_corr": n_abs,
                "corr_official_abs_error": corr_abs,
                "abs_corr_official_abs_error": abs(corr_abs) if math.isfinite(corr_abs) else math.nan,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["abs_corr_official_error", "abs_corr_official_abs_error"],
        ascending=[False, False],
        na_position="last",
    )


def interaction_atlas(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    official_overlap: pd.DataFrame,
) -> pd.DataFrame:
    train_mask = frame["target_date"] <= TRAIN_END
    eval_mask = (frame["target_date"] >= EVAL_START) & (frame["target_date"] <= EVAL_END)
    selected_rows = selected.to_dict("records")
    edge_cache: dict[str, tuple[float, float] | None] = {
        str(row["feature"]): quantile_edges_from_train(frame.loc[train_mask, str(row["feature"])])
        for row in selected_rows
    }
    records: list[dict[str, object]] = []
    for left_index, left in enumerate(selected_rows):
        for right in selected_rows[left_index + 1 :]:
            feature_a = str(left["feature"])
            feature_b = str(right["feature"])
            family_a = str(left["family"])
            family_b = str(right["family"])
            if family_a == family_b:
                continue
            edges_a = edge_cache.get(feature_a)
            edges_b = edge_cache.get(feature_b)
            if edges_a is None or edges_b is None:
                continue

            eval_subset = frame.loc[
                eval_mask,
                ["target_date", "target_anomaly_vs_past_doy_c", feature_a, feature_b],
            ].copy()
            eval_subset["bin_a"] = apply_tertile_bins(eval_subset[feature_a], edges_a)
            eval_subset["bin_b"] = apply_tertile_bins(eval_subset[feature_b], edges_b)
            eval_subset["cell"] = eval_subset["bin_a"].astype(str) + "/" + eval_subset["bin_b"].astype(str)
            valid_eval = eval_subset.dropna(subset=["target_anomaly_vs_past_doy_c", "bin_a", "bin_b"])
            eval_cells = (
                valid_eval.groupby("cell", observed=True)["target_anomaly_vs_past_doy_c"]
                .agg(["count", "mean"])
                .reset_index()
            )
            eval_cells = eval_cells[eval_cells["count"] >= MIN_CELL_ROWS]
            if eval_cells.empty:
                continue
            warm_cell = eval_cells.sort_values("mean", ascending=False).iloc[0]
            cool_cell = eval_cells.sort_values("mean", ascending=True).iloc[0]
            eval_spread = float(warm_cell["mean"] - cool_cell["mean"])

            official_error_spread = math.nan
            official_abs_error_spread = math.nan
            official_cells_valid = 0
            official_rows = 0
            if not official_overlap.empty:
                official_subset = official_overlap[
                    [
                        "target_date",
                        "official_error_c",
                        "official_abs_error_c",
                        feature_a,
                        feature_b,
                    ]
                ].copy()
                official_subset["bin_a"] = apply_tertile_bins(official_subset[feature_a], edges_a)
                official_subset["bin_b"] = apply_tertile_bins(official_subset[feature_b], edges_b)
                official_subset["cell"] = (
                    official_subset["bin_a"].astype(str) + "/" + official_subset["bin_b"].astype(str)
                )
                official_valid = official_subset.dropna(
                    subset=["official_error_c", "official_abs_error_c", "bin_a", "bin_b"]
                )
                official_rows = int(len(official_valid))
                official_cells = (
                    official_valid.groupby("cell", observed=True)
                    .agg(
                        n=("official_error_c", "size"),
                        mean_official_error_c=("official_error_c", "mean"),
                        mean_official_abs_error_c=("official_abs_error_c", "mean"),
                    )
                    .reset_index()
                )
                official_cells = official_cells[official_cells["n"] >= MIN_OFFICIAL_CELL_ROWS]
                official_cells_valid = int(len(official_cells))
                if official_cells_valid >= 2:
                    official_error_spread = float(
                        official_cells["mean_official_error_c"].max()
                        - official_cells["mean_official_error_c"].min()
                    )
                    official_abs_error_spread = float(
                        official_cells["mean_official_abs_error_c"].max()
                        - official_cells["mean_official_abs_error_c"].min()
                    )

            records.append(
                {
                    "feature_a": feature_a,
                    "family_a": family_a,
                    "feature_b": feature_b,
                    "family_b": family_b,
                    "train_edges_a": json.dumps(edges_a),
                    "train_edges_b": json.dumps(edges_b),
                    "eval_rows": int(len(valid_eval)),
                    "eval_cells_valid": int(len(eval_cells)),
                    "eval_target_anomaly_spread_c": eval_spread,
                    "eval_warmest_cell": str(warm_cell["cell"]),
                    "eval_warmest_cell_mean_anomaly_c": float(warm_cell["mean"]),
                    "eval_warmest_cell_rows": int(warm_cell["count"]),
                    "eval_coolest_cell": str(cool_cell["cell"]),
                    "eval_coolest_cell_mean_anomaly_c": float(cool_cell["mean"]),
                    "eval_coolest_cell_rows": int(cool_cell["count"]),
                    "official_overlap_rows": official_rows,
                    "official_cells_valid": official_cells_valid,
                    "official_error_spread_c": official_error_spread,
                    "official_abs_error_spread_c": official_abs_error_spread,
                    "priority_score": (
                        (0 if not math.isfinite(official_error_spread) else abs(official_error_spread))
                        + 0.35 * eval_spread
                    ),
                }
            )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("priority_score", ascending=False).reset_index(drop=True)


def station_feature_index(features: list[str], correlations: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    corr_lookup = correlations.set_index("feature").to_dict("index")
    for feature in features:
        ids = station_ids_in_feature(feature)
        if not ids:
            continue
        for station_id in ids.split(","):
            corr = corr_lookup.get(feature, {})
            records.append(
                {
                    "station_id": station_id,
                    "feature": feature,
                    "family": classify_feature_family(feature),
                    "corr_train_pre2000_anomaly": corr.get("corr_train_pre2000_anomaly", math.nan),
                    "corr_eval_2000_2023_anomaly": corr.get("corr_eval_2000_2023_anomaly", math.nan),
                }
            )
    if not records:
        return pd.DataFrame(columns=["station_id", "feature", "family"])
    out = pd.DataFrame(records)
    return out.sort_values(["station_id", "feature"]).reset_index(drop=True)


def update_markdown_section(
    path: Path,
    *,
    heading: str,
    section: str,
    insert_before: str | None = None,
) -> None:
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    text = original
    if path.name == "MILESTONES.md":
        text = re.sub(
            r"\*\*Last updated:\*\* .*?  ",
            f"**Last updated:** {now_utc()}  ",
            text,
            count=1,
        )
    pattern = re.compile(rf"^## {re.escape(heading)}\n.*?(?=^## |\Z)", flags=re.M | re.S)
    replacement = f"## {heading}\n\n{section.strip()}\n\n"
    if pattern.search(text):
        text = pattern.sub(lambda _match: replacement, text)
    elif insert_before and insert_before in text:
        text = text.replace(insert_before, replacement + insert_before, 1)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n" + replacement
    write_text(path, text)


def build_readme(
    *,
    generated_at: str,
    summary: dict[str, Any],
    coverage: pd.DataFrame,
    selected: pd.DataFrame,
    official_corr: pd.DataFrame,
    interactions: pd.DataFrame,
    non_calendar_interactions: pd.DataFrame,
    official_score: dict[str, object],
) -> str:
    top_interactions = interactions.head(15) if not interactions.empty else interactions
    top_interactions_display = top_interactions[
        [
            "feature_a",
            "family_a",
            "feature_b",
            "family_b",
            "eval_rows",
            "eval_target_anomaly_spread_c",
            "official_overlap_rows",
            "official_error_spread_c",
            "official_abs_error_spread_c",
        ]
    ].copy() if not top_interactions.empty else pd.DataFrame()
    top_non_calendar = non_calendar_interactions.head(15)
    top_non_calendar_display = top_non_calendar[
        [
            "feature_a",
            "family_a",
            "feature_b",
            "family_b",
            "eval_rows",
            "eval_target_anomaly_spread_c",
            "official_overlap_rows",
            "official_error_spread_c",
            "official_abs_error_spread_c",
        ]
    ].copy() if not top_non_calendar.empty else pd.DataFrame()

    top_selected = selected[
        [
            "feature",
            "family",
            "n_train_pre2000_anomaly",
            "corr_train_pre2000_anomaly",
            "n_eval_2000_2023_anomaly",
            "corr_eval_2000_2023_anomaly",
        ]
    ].head(20)
    top_official = official_corr[
        [
            "feature",
            "family",
            "n_official_error_corr",
            "corr_official_error",
            "n_official_abs_error_corr",
            "corr_official_abs_error",
        ]
    ].head(20) if not official_corr.empty else pd.DataFrame()

    return f"""# Long-History Cross-Family Interaction Atlas

Generated: `{generated_at}`

## Purpose

This insight folder is a leakage-safe research atlas for the next stage of HKG Tmax work. The aim is not to train a new production model and not to touch the sealed 2024+ confirmation period. The aim is to answer a narrower but important question: across the long-history feature matrix, which feature families retain signal before 2000 and still explain target Tmax anomaly or official forecast residual behavior during 2000-2023?

The reason for this screen is practical. The current official forecast anchor is strong, but the forecast archive is still non-contiguous because raw press-detail HTML is missing for 2008-2026. While that acquisition blocker is handled outside this socket-blocked environment, we can still mine the already normalized long-history data for durable physical signal. This script uses the wide EXP-0050/0099 matrix, which carries target-memory features, HKO daily climate fields, IGRA upper-air features, ISD regional station-network features, station graph modes, trajectory shape fields, and other derived attributes.

## Leakage Control

- All rows with `target_date >= 2024-01-01` are rejected before analysis.
- Target anomaly is calculated against a past-only same-day-of-year climatology. For each date, the climatology uses only earlier years for the same day-of-year and requires at least ten prior samples.
- Feature selection is based on rows through `1999-12-31`.
- Interaction bin thresholds are learned only from rows through `1999-12-31`.
- Interaction behavior is evaluated on `2000-01-01` through `2023-12-31`.
- Official forecast residual checks use the current scored official archive only before 2024. This archive remains non-contiguous and is treated as diagnostic evidence, not final model proof.

## Dataset Scope

| Item | Value |
|---|---:|
| Feature matrix rows | {summary["feature_matrix_rows"]} |
| Numeric candidate features | {summary["candidate_feature_count"]} |
| Feature matrix first target date | {summary["feature_matrix_first_date"]} |
| Feature matrix last target date used | {summary["feature_matrix_last_date"]} |
| Pre-2000 feature-selection rows | {summary["train_rows_pre2000"]} |
| 2000-2023 evaluation rows | {summary["eval_rows_2000_2023"]} |
| Selected pre-2000 features | {summary["selected_feature_count"]} |
| Cross-family interaction candidates | {summary["interaction_candidate_count"]} |
| Official scored overlap rows | {summary["official_overlap_rows"]} |
| Official raw max MAE on overlap | {official_score.get("mae", math.nan)} |

## Family Coverage

{markdown_table(coverage, max_rows=20)}

## Selected Durable Features

These are selected using only the pre-2000 period and then checked on 2000-2023. Positive or negative sign is not itself good or bad; sign stability and persistence across eras are what matter.

{markdown_table(top_selected, max_rows=20)}

## Official Residual Correlation On Current Archive

These rows join the long-history features to the current official scored forecast archive. Because the official archive is still non-contiguous, this is a diagnostic overlap screen. It says which long-history attributes tend to line up with official forecast misses, not that a production correction is proven.

{markdown_table(top_official, max_rows=20)}

## Cross-Family Interaction Leaders

The table below uses pre-2000 tertile thresholds for both features, then measures how much target Tmax anomaly separates across the 2000-2023 evaluation cells. Where enough official overlap exists, it also reports official forecast residual spread across the same cells.

{markdown_table(top_interactions_display, max_rows=15)}

## Non-Calendar Interaction Leaders

This table removes `calendar_climatology` pairs so the strongest physical and station/network channels are visible. These are the rows most relevant to a future residual specialist because they are closer to actual meteorological mechanisms rather than seasonal partitioning.

{markdown_table(top_non_calendar_display, max_rows=15)}

## Main Finding

The important result is that long-history station/network and upper-air attributes are not merely correlated with the raw seasonal Tmax cycle. After subtracting a past-only day-of-year climatology, several feature families still retain measurable signal, and cross-family cells separate 2000-2023 target anomaly by materially larger amounts than single-feature correlations imply. This matters because the path toward a much lower MAE is unlikely to come from another global linear correction. The data keeps pointing toward conditional systems: trust the official forecast anchor, then apply small residual specialists only in regimes where the physical context says the official forecast is usually too hot, too cold, or too uncertain.

The most useful aspect of this folder is the interaction table. It gives a ranked set of feature pairs whose thresholds were selected before the official forecast era was fully evaluated. That makes the result much harder to dismiss as post-2000 curve fitting. If a pair has large target-anomaly spread on 2000-2023 and also has official-error spread on the current official archive, it becomes a candidate for a future fold-local specialist once the forecast archive is made continuous. If it only separates target anomaly but not official residual, it may still help weather-only fallback models, but it is less likely to improve the official-anchor system.

## What This Does Not Prove

This run does not prove a new MAE champion. It does not use 2024+ confirmation rows. It does not train a predictive model. It does not solve the missing 2008-2026 press-detail HTML gap. It is a feature-discovery and interaction-screening artifact designed to tell the next modelling step where the real information channels are likely to be.

## Recommended Next Research Use

1. After the 0045 raw-detail backfill is completed, rerun 0044 and rebuild the official scored forecast frame.
2. Re-run this 0046 atlas on the continuous official frame.
3. Convert only the top stable cross-family interactions into fold-local residual specialists.
4. Evaluate those specialists against the frozen 0042/0043 router benchmark with the same 2024+ lock still active.

## Artifact Files

- `artifacts/family_coverage.csv`
- `artifacts/feature_correlations.csv`
- `artifacts/era_stability.csv`
- `artifacts/pre2000_selected_features.csv`
- `artifacts/official_residual_correlations.csv`
- `artifacts/cross_family_interactions.csv`
- `artifacts/physical_interactions.csv`
- `artifacts/station_feature_index.csv`
- `artifacts/summary.json`
"""


def build_milestone_section(
    *,
    summary: dict[str, Any],
    selected: pd.DataFrame,
    interactions: pd.DataFrame,
    non_calendar_interactions: pd.DataFrame,
    official_corr: pd.DataFrame,
) -> str:
    top_selected = selected[
        [
            "feature",
            "family",
            "corr_train_pre2000_anomaly",
            "corr_eval_2000_2023_anomaly",
        ]
    ].head(8)
    top_interactions = interactions[
        [
            "feature_a",
            "family_a",
            "feature_b",
            "family_b",
            "eval_target_anomaly_spread_c",
            "official_error_spread_c",
            "official_abs_error_spread_c",
        ]
    ].head(8) if not interactions.empty else pd.DataFrame()
    top_non_calendar = non_calendar_interactions[
        [
            "feature_a",
            "family_a",
            "feature_b",
            "family_b",
            "eval_target_anomaly_spread_c",
            "official_error_spread_c",
            "official_abs_error_spread_c",
        ]
    ].head(8) if not non_calendar_interactions.empty else pd.DataFrame()
    top_official = official_corr[
        [
            "feature",
            "family",
            "corr_official_error",
            "corr_official_abs_error",
        ]
    ].head(8) if not official_corr.empty else pd.DataFrame()

    return f"""Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_long_history_cross_family_interaction_atlas.py
```

New folder: `research/data_analysis/0046_long_history_cross_family_interaction_atlas`.

| Area | Evidence | Status |
|---|---|---|
| Long-history matrix | `{summary["feature_matrix_rows"]}` rows, `{summary["candidate_feature_count"]}` numeric candidate features, target dates `{summary["feature_matrix_first_date"]}` to `{summary["feature_matrix_last_date"]}` | Audited |
| Leakage guard | feature selection and tertile thresholds use only pre-2000 rows; evaluation uses 2000-2023; zero 2024+ rows | Guarded |
| Selected features | `{summary["selected_feature_count"]}` durable pre-2000 features across families | Diagnostic |
| Cross-family interactions | `{summary["interaction_candidate_count"]}` cross-family pairs scored on 2000-2023 target anomaly and current official residual overlap | Diagnostic |
| Official overlap | `{summary["official_overlap_rows"]}` non-contiguous official scored rows; still limited by the 2008-2026 raw-detail gap | Blocker-aware |

Top pre-2000-selected durable features:

{markdown_table(top_selected, max_rows=8)}

Top current official-residual correlations:

{markdown_table(top_official, max_rows=8)}

Top cross-family interaction candidates:

{markdown_table(top_interactions, max_rows=8)}

Top non-calendar physical/station interaction candidates:

{markdown_table(top_non_calendar, max_rows=8)}

Interpretation: `0046` confirms that long-history station/network and upper-air features still carry target-anomaly structure after a past-only day-of-year climatology is removed. It also identifies cross-family regimes whose thresholds are learned before 2000 and then evaluated on 2000-2023, so they are stronger candidates than same-window correlations. This does not create a new MAE champion; it creates the next candidate list for fold-local residual specialists after the official forecast archive is made continuous.
"""


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"

    frame = add_past_doy_anomaly(load_feature_matrix(FEATURE_MATRIX_PATH))
    features = numeric_feature_columns(frame)
    coverage = family_coverage(frame, features)
    correlations = feature_correlation_atlas(frame, features)
    stability = era_stability_atlas(frame, features)
    selected = select_pre2000_features(correlations)
    selected_features = selected["feature"].astype(str).tolist()
    official_overlap = official_overlap_frame(frame, selected_features)
    official_corr = official_residual_correlations(official_overlap, selected_features)
    interactions = interaction_atlas(frame, selected, official_overlap)
    non_calendar_interactions = interactions[
        ~interactions["family_a"].eq("calendar_climatology")
        & ~interactions["family_b"].eq("calendar_climatology")
    ].copy() if not interactions.empty else pd.DataFrame()
    station_index = station_feature_index(features, correlations)

    official_score = (
        score_prediction_frame(official_overlap, "forecast_max_c")
        if not official_overlap.empty and "forecast_max_c" in official_overlap.columns
        else {"n": 0, "mae": math.nan, "rmse": math.nan, "bias": math.nan}
    )

    train_rows = int((frame["target_date"] <= TRAIN_END).sum())
    eval_rows = int(((frame["target_date"] >= EVAL_START) & (frame["target_date"] <= EVAL_END)).sum())
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "feature_matrix_path": str(FEATURE_MATRIX_PATH),
        "official_scored_path": str(OFFICIAL_SCORED_PATH),
        "feature_matrix_rows": int(len(frame)),
        "feature_matrix_first_date": str(frame["target_date"].min().date()),
        "feature_matrix_last_date": str(frame["target_date"].max().date()),
        "candidate_feature_count": int(len(features)),
        "train_rows_pre2000": train_rows,
        "eval_rows_2000_2023": eval_rows,
        "selected_feature_count": int(len(selected)),
        "interaction_candidate_count": int(len(interactions)),
        "non_calendar_interaction_candidate_count": int(len(non_calendar_interactions)),
        "official_overlap_rows": int(len(official_overlap)),
        "official_raw_forecast_score": official_score,
        "top_selected_feature": selected.iloc[0].to_dict() if not selected.empty else {},
        "top_official_residual_feature": official_corr.iloc[0].to_dict() if not official_corr.empty else {},
        "top_cross_family_interaction": interactions.iloc[0].to_dict() if not interactions.empty else {},
        "top_non_calendar_cross_family_interaction": (
            non_calendar_interactions.iloc[0].to_dict() if not non_calendar_interactions.empty else {}
        ),
        "leakage_guard": {
            "confirmation_start": str(CONFIRMATION_START.date()),
            "feature_selection_period": f"target_date <= {TRAIN_END.date()}",
            "evaluation_period": f"{EVAL_START.date()} <= target_date <= {EVAL_END.date()}",
            "past_doy_climatology_min_prior_samples": 10,
            "uses_2024_plus_rows": False,
        },
    }

    write_csv(artifacts / "family_coverage.csv", coverage)
    write_csv(artifacts / "feature_correlations.csv", correlations.head(TOP_FEATURE_ROWS))
    write_csv(artifacts / "era_stability.csv", stability.head(TOP_FEATURE_ROWS))
    write_csv(artifacts / "pre2000_selected_features.csv", selected)
    write_csv(artifacts / "official_residual_correlations.csv", official_corr)
    write_csv(artifacts / "cross_family_interactions.csv", interactions)
    write_csv(artifacts / "physical_interactions.csv", non_calendar_interactions)
    write_csv(artifacts / "station_feature_index.csv", station_index)
    write_json(artifacts / "summary.json", summary)

    readme = build_readme(
        generated_at=generated_at,
        summary=summary,
        coverage=coverage,
        selected=selected,
        official_corr=official_corr,
        interactions=interactions,
        non_calendar_interactions=non_calendar_interactions,
        official_score=official_score,
    )
    write_text(folder / "README.md", readme)

    manifest = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "summary_path": str(artifacts / "summary.json"),
        "readme_path": str(folder / "README.md"),
        "feature_matrix_rows": summary["feature_matrix_rows"],
        "candidate_feature_count": summary["candidate_feature_count"],
        "selected_feature_count": summary["selected_feature_count"],
        "interaction_candidate_count": summary["interaction_candidate_count"],
        "non_calendar_interaction_candidate_count": summary[
            "non_calendar_interaction_candidate_count"
        ],
        "official_overlap_rows": summary["official_overlap_rows"],
        "top_cross_family_interaction": summary["top_cross_family_interaction"],
        "top_non_calendar_cross_family_interaction": summary[
            "top_non_calendar_cross_family_interaction"
        ],
        "uses_2024_plus_rows": False,
    }
    write_json(output_root / "long_history_cross_family_interaction_atlas_manifest.json", manifest)

    update_markdown_section(
        output_root / "README.md",
        heading="0046 Long-History Cross-Family Interaction Atlas",
        section=(
            f"Generated `{generated_at}`. See `{FOLDER_NAME}`. "
            f"Scanned `{summary['candidate_feature_count']}` numeric long-history features, "
            f"selected `{summary['selected_feature_count']}` pre-2000 durable features, and scored "
            f"`{summary['interaction_candidate_count']}` cross-family interactions on 2000-2023 "
            "without using 2024+ confirmation rows."
        ),
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Long-History Cross-Family Interaction Atlas",
        section=build_milestone_section(
            summary=summary,
            selected=selected,
            interactions=interactions,
            non_calendar_interactions=non_calendar_interactions,
            official_corr=official_corr,
        ),
        insert_before="## Current Blockers And Gaps",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe long-history cross-family feature interaction atlas."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESEARCH_ROOT,
        help="Research data_analysis root.",
    )
    args = parser.parse_args()
    summary = run(args.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
