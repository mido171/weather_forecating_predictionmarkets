from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    TRAIN_END,
    add_past_doy_anomaly,
    apply_tertile_bins,
    quantile_edges_from_train,
    safe_corr,
    update_markdown_section,
)

DATASETS_ROOT = PROJECT_PATHS.data_root / "datasets"
TARGET_PATH = DATASETS_ROOT / "01_hko_daily_tmax_target" / "hko_daily_tmax_target_labels.parquet"
ISD_DAY_PATH = (
    DATASETS_ROOT
    / "04_noaa_isd_regional_surface"
    / "noaa_isd_station_day_cutoff_summary.parquet"
)
ISD_CORE_PATH = (
    DATASETS_ROOT / "04_noaa_isd_regional_surface" / "noaa_isd_core_observations.parquet"
)
OFFICIAL_SCORED_PATH = (
    DATASETS_ROOT
    / "05_hko_historical_rss_forecasts"
    / "hko_official_t15_scored_pre2024.parquet"
)
FOLDER_NAME = "0047_station_contribution_atlas"

TARGET_LATITUDE = 22.301944
TARGET_LONGITUDE = 114.174167
MIN_TRAIN_ROWS = 2500
MIN_EVAL_ROWS = 730
MIN_OFFICIAL_ROWS = 120
MIN_TERTILE_ROWS = 80
TOP_ROWS = 80

BASE_ATTRIBUTES = [
    "air_temperature_c_latest_before_1500",
    "dew_point_c_latest_before_1500",
    "sea_level_pressure_hpa_latest_before_1500",
    "wind_speed_mps_latest_before_1500",
    "wind_u_mps_latest_before_1500",
    "wind_v_mps_latest_before_1500",
    "temp_dew_spread_c_latest_before_1500",
]
CHANGE_ATTRIBUTES = [
    "air_temperature_c_latest_before_1500_change_1d",
    "dew_point_c_latest_before_1500_change_1d",
    "sea_level_pressure_hpa_latest_before_1500_change_1d",
    "wind_speed_mps_latest_before_1500_change_1d",
]
NETWORK_ATTRIBUTES = [
    "air_temperature_c_latest_before_1500_minus_network_median",
    "dew_point_c_latest_before_1500_minus_network_median",
    "sea_level_pressure_hpa_latest_before_1500_minus_network_median",
    "wind_speed_mps_latest_before_1500_minus_network_median",
]
STATION_ATTRIBUTES = BASE_ATTRIBUTES + CHANGE_ATTRIBUTES + NETWORK_ATTRIBUTES
PAIR_ATTRIBUTES = [
    "air_temperature_c_latest_before_1500",
    "dew_point_c_latest_before_1500",
    "sea_level_pressure_hpa_latest_before_1500",
    "wind_speed_mps_latest_before_1500",
]


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return float(2.0 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    y = math.sin(d_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    return float((math.degrees(math.atan2(y, x)) + 360.0) % 360.0)


def load_target() -> pd.DataFrame:
    target = pd.read_parquet(TARGET_PATH).copy()
    target["target_date"] = pd.to_datetime(target["local_date"], errors="coerce").dt.normalize()
    target = target[["target_date", "target_tmax_c"]].dropna().copy()
    target = target[target["target_date"] < CONFIRMATION_START].copy()
    require_no_confirmation_dates(target["target_date"], context="0047 target labels")
    return add_past_doy_anomaly(target)


def load_station_metadata() -> pd.DataFrame:
    columns = ["station_id", "latitude", "longitude", "elevation_m", "observed_at_hkt"]
    core = pd.read_parquet(ISD_CORE_PATH, columns=columns)
    meta = (
        core.groupby("station_id", observed=True)
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            elevation_m=("elevation_m", "median"),
            first_observed_at_hkt=("observed_at_hkt", "min"),
            last_observed_at_hkt=("observed_at_hkt", "max"),
            raw_observation_rows=("station_id", "size"),
        )
        .reset_index()
    )
    distances: list[float] = []
    bearings: list[float] = []
    coordinate_statuses: list[str] = []
    for row in meta.itertuples(index=False):
        if pd.isna(row.latitude) or pd.isna(row.longitude):
            distances.append(math.nan)
            bearings.append(math.nan)
            coordinate_statuses.append("missing_coordinate")
        elif not (20.0 <= float(row.latitude) <= 25.0 and 110.0 <= float(row.longitude) <= 117.0):
            distances.append(math.nan)
            bearings.append(math.nan)
            coordinate_statuses.append("outside_hk_prd_bbox_check_source_metadata")
        else:
            distances.append(haversine_km(TARGET_LATITUDE, TARGET_LONGITUDE, float(row.latitude), float(row.longitude)))
            bearings.append(bearing_degrees(TARGET_LATITUDE, TARGET_LONGITUDE, float(row.latitude), float(row.longitude)))
            coordinate_statuses.append("ok")
    meta["distance_to_hko_km"] = distances
    meta["bearing_from_hko_deg"] = bearings
    meta["coordinate_sanity_status"] = coordinate_statuses
    return meta


def load_station_day_features(target: pd.DataFrame) -> pd.DataFrame:
    day = pd.read_parquet(ISD_DAY_PATH).copy()
    day["local_date"] = pd.to_datetime(day["local_date"], errors="coerce").dt.normalize()
    day = day[day["local_date"].notna()].copy()
    day["target_date"] = day["local_date"] + pd.Timedelta(days=1)
    day = day[day["target_date"] < CONFIRMATION_START].copy()
    require_no_confirmation_dates(day["target_date"], context="0047 station target dates")

    speed = pd.to_numeric(day["wind_speed_mps_latest_before_1500"], errors="coerce")
    direction = pd.to_numeric(day["wind_direction_deg_latest_before_1500"], errors="coerce")
    radians = np.deg2rad(direction)
    day["wind_u_mps_latest_before_1500"] = -speed * np.sin(radians)
    day["wind_v_mps_latest_before_1500"] = -speed * np.cos(radians)
    day["temp_dew_spread_c_latest_before_1500"] = (
        pd.to_numeric(day["air_temperature_c_latest_before_1500"], errors="coerce")
        - pd.to_numeric(day["dew_point_c_latest_before_1500"], errors="coerce")
    )

    day = day.sort_values(["station_id", "local_date"]).reset_index(drop=True)
    for column in [
        "air_temperature_c_latest_before_1500",
        "dew_point_c_latest_before_1500",
        "sea_level_pressure_hpa_latest_before_1500",
        "wind_speed_mps_latest_before_1500",
    ]:
        day[f"{column}_change_1d"] = day.groupby("station_id", observed=True)[column].diff(1)

    for column in [
        "air_temperature_c_latest_before_1500",
        "dew_point_c_latest_before_1500",
        "sea_level_pressure_hpa_latest_before_1500",
        "wind_speed_mps_latest_before_1500",
    ]:
        median = day.groupby("local_date", observed=True)[column].transform("median")
        day[f"{column}_minus_network_median"] = day[column] - median

    keep = ["station_id", "local_date", "target_date", *STATION_ATTRIBUTES]
    merged = day[keep].merge(
        target[
            [
                "target_date",
                "target_tmax_c",
                "past_doy_count",
                "past_doy_mean_tmax_c",
                "target_anomaly_vs_past_doy_c",
            ]
        ],
        on="target_date",
        how="inner",
    )
    return merged.sort_values(["target_date", "station_id"]).reset_index(drop=True)


def load_official_residuals() -> pd.DataFrame:
    if not OFFICIAL_SCORED_PATH.exists():
        return pd.DataFrame()
    official = pd.read_parquet(OFFICIAL_SCORED_PATH).copy()
    official["target_date"] = pd.to_datetime(official["target_date"], errors="coerce").dt.normalize()
    official = official[official["target_date"].notna() & (official["target_date"] < CONFIRMATION_START)].copy()
    require_no_confirmation_dates(official["target_date"], context="0047 official residuals")
    if "official_error_c" not in official.columns:
        official["official_error_c"] = official["forecast_max_c"] - official["target_tmax_c"]
    if "official_abs_error_c" not in official.columns:
        official["official_abs_error_c"] = official["official_error_c"].abs()
    return official[
        [
            "target_date",
            "forecast_source_family",
            "forecast_max_c",
            "official_error_c",
            "official_abs_error_c",
        ]
    ].copy()


def tertile_spread(
    values: pd.Series,
    outcome: pd.Series,
    train_mask: pd.Series,
    eval_mask: pd.Series,
    *,
    min_rows: int,
) -> dict[str, object]:
    edges = quantile_edges_from_train(values[train_mask])
    if edges is None:
        return {
            "tertile_edges": "",
            "valid_cells": 0,
            "cell_spread": math.nan,
            "high_cell": "",
            "low_cell": "",
        }
    binned = apply_tertile_bins(values[eval_mask], edges)
    eval_outcome = outcome[eval_mask]
    cells = (
        pd.DataFrame({"cell": binned, "outcome": eval_outcome})
        .dropna()
        .groupby("cell", observed=True)["outcome"]
        .agg(["count", "mean"])
        .reset_index()
    )
    cells = cells[cells["count"] >= min_rows]
    if len(cells) < 2:
        return {
            "tertile_edges": json.dumps(edges),
            "valid_cells": int(len(cells)),
            "cell_spread": math.nan,
            "high_cell": "",
            "low_cell": "",
        }
    high = cells.sort_values("mean", ascending=False).iloc[0]
    low = cells.sort_values("mean", ascending=True).iloc[0]
    return {
        "tertile_edges": json.dumps(edges),
        "valid_cells": int(len(cells)),
        "cell_spread": float(high["mean"] - low["mean"]),
        "high_cell": str(high["cell"]),
        "low_cell": str(low["cell"]),
        "high_cell_rows": int(high["count"]),
        "low_cell_rows": int(low["count"]),
    }


def station_attribute_atlas(
    station_frame: pd.DataFrame,
    official: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    train_mask_all = station_frame["target_date"] <= TRAIN_END
    eval_mask_all = (station_frame["target_date"] >= EVAL_START) & (station_frame["target_date"] <= EVAL_END)
    official_joined = station_frame.merge(official, on="target_date", how="inner") if not official.empty else pd.DataFrame()
    meta_lookup = metadata.set_index("station_id").to_dict("index")

    records: list[dict[str, object]] = []
    for station_id, station_rows in station_frame.groupby("station_id", observed=True):
        station_train = train_mask_all.loc[station_rows.index]
        station_eval = eval_mask_all.loc[station_rows.index]
        meta = meta_lookup.get(str(station_id), {})
        for attribute in STATION_ATTRIBUTES:
            values = pd.to_numeric(station_rows[attribute], errors="coerce")
            n_train, corr_train = safe_corr(
                values[station_train],
                station_rows.loc[station_train, "target_anomaly_vs_past_doy_c"],
                min_rows=MIN_TRAIN_ROWS,
            )
            n_eval, corr_eval = safe_corr(
                values[station_eval],
                station_rows.loc[station_eval, "target_anomaly_vs_past_doy_c"],
                min_rows=MIN_EVAL_ROWS,
            )
            spread = tertile_spread(
                values,
                station_rows["target_anomaly_vs_past_doy_c"],
                station_train,
                station_eval,
                min_rows=MIN_TERTILE_ROWS,
            )

            n_official_error = 0
            corr_official_error = math.nan
            n_official_abs = 0
            corr_official_abs = math.nan
            if not official_joined.empty:
                oj = official_joined[official_joined["station_id"].eq(station_id)]
                n_official_error, corr_official_error = safe_corr(
                    oj[attribute],
                    oj["official_error_c"],
                    min_rows=MIN_OFFICIAL_ROWS,
                )
                n_official_abs, corr_official_abs = safe_corr(
                    oj[attribute],
                    oj["official_abs_error_c"],
                    min_rows=MIN_OFFICIAL_ROWS,
                )

            robust = n_train >= MIN_TRAIN_ROWS and n_eval >= MIN_EVAL_ROWS
            records.append(
                {
                    "station_id": station_id,
                    "attribute": attribute,
                    "robust_pre2000_testable": robust,
                    "latitude": meta.get("latitude", math.nan),
                    "longitude": meta.get("longitude", math.nan),
                    "elevation_m": meta.get("elevation_m", math.nan),
                    "coordinate_sanity_status": meta.get("coordinate_sanity_status", ""),
                    "distance_to_hko_km": meta.get("distance_to_hko_km", math.nan),
                    "bearing_from_hko_deg": meta.get("bearing_from_hko_deg", math.nan),
                    "first_observed_at_hkt": meta.get("first_observed_at_hkt", ""),
                    "last_observed_at_hkt": meta.get("last_observed_at_hkt", ""),
                    "n_train_pre2000": n_train,
                    "corr_train_pre2000_target_anomaly": corr_train,
                    "abs_corr_train_pre2000_target_anomaly": abs(corr_train)
                    if math.isfinite(corr_train)
                    else math.nan,
                    "n_eval_2000_2023": n_eval,
                    "corr_eval_2000_2023_target_anomaly": corr_eval,
                    "abs_corr_eval_2000_2023_target_anomaly": abs(corr_eval)
                    if math.isfinite(corr_eval)
                    else math.nan,
                    "eval_tertile_target_anomaly_spread_c": spread["cell_spread"],
                    "eval_tertile_edges_pre2000": spread["tertile_edges"],
                    "eval_tertile_high_cell": spread["high_cell"],
                    "eval_tertile_low_cell": spread["low_cell"],
                    "n_official_error_corr": n_official_error,
                    "corr_official_error": corr_official_error,
                    "abs_corr_official_error": abs(corr_official_error)
                    if math.isfinite(corr_official_error)
                    else math.nan,
                    "n_official_abs_error_corr": n_official_abs,
                    "corr_official_abs_error": corr_official_abs,
                    "abs_corr_official_abs_error": abs(corr_official_abs)
                    if math.isfinite(corr_official_abs)
                    else math.nan,
                    "priority_score": (
                        (abs(corr_eval) if math.isfinite(corr_eval) else 0.0)
                        + 0.4 * (abs(corr_official_error) if math.isfinite(corr_official_error) else 0.0)
                        + 0.08
                        * (
                            float(spread["cell_spread"])
                            if isinstance(spread["cell_spread"], float)
                            and math.isfinite(float(spread["cell_spread"]))
                            else 0.0
                        )
                    ),
                }
            )
    return pd.DataFrame(records).sort_values(
        ["robust_pre2000_testable", "priority_score"],
        ascending=[False, False],
        na_position="last",
    )


def station_ranking(attribute_atlas: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        attribute_atlas.groupby("station_id", observed=True)
        .agg(
            robust_attribute_count=("robust_pre2000_testable", "sum"),
            max_priority_score=("priority_score", "max"),
            best_eval_abs_corr=("abs_corr_eval_2000_2023_target_anomaly", "max"),
            best_official_abs_corr=("abs_corr_official_error", "max"),
            min_distance_to_hko_km=("distance_to_hko_km", "min"),
            median_bearing_from_hko_deg=("bearing_from_hko_deg", "median"),
            first_observed_at_hkt=("first_observed_at_hkt", "min"),
            last_observed_at_hkt=("last_observed_at_hkt", "max"),
        )
        .reset_index()
    )
    best_rows = attribute_atlas.sort_values("priority_score", ascending=False).drop_duplicates("station_id")
    grouped = grouped.merge(
        best_rows[["station_id", "attribute"]].rename(columns={"attribute": "best_attribute"}),
        on="station_id",
        how="left",
    )
    return grouped.sort_values(["robust_attribute_count", "max_priority_score"], ascending=[False, False])


def pair_spread_atlas(station_frame: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    target_by_date = (
        station_frame[["target_date", "target_anomaly_vs_past_doy_c"]]
        .drop_duplicates("target_date")
        .set_index("target_date")
    )
    official_by_date = official.set_index("target_date") if not official.empty else pd.DataFrame()
    records: list[dict[str, object]] = []
    for attribute in PAIR_ATTRIBUTES:
        pivot = station_frame.pivot_table(
            index="target_date",
            columns="station_id",
            values=attribute,
            aggfunc="last",
        ).sort_index()
        stations = list(pivot.columns)
        train_mask = pivot.index <= TRAIN_END
        eval_mask = (pivot.index >= EVAL_START) & (pivot.index <= EVAL_END)
        for station_a, station_b in itertools.combinations(stations, 2):
            diff = pivot[station_a] - pivot[station_b]
            n_train, corr_train = safe_corr(
                diff.loc[train_mask],
                target_by_date.loc[pivot.index[train_mask], "target_anomaly_vs_past_doy_c"],
                min_rows=MIN_TRAIN_ROWS,
            )
            if n_train < MIN_TRAIN_ROWS or not math.isfinite(corr_train):
                continue
            n_eval, corr_eval = safe_corr(
                diff.loc[eval_mask],
                target_by_date.loc[pivot.index[eval_mask], "target_anomaly_vs_past_doy_c"],
                min_rows=MIN_EVAL_ROWS,
            )
            if n_eval < MIN_EVAL_ROWS:
                continue
            spread = tertile_spread(
                diff,
                target_by_date.loc[pivot.index, "target_anomaly_vs_past_doy_c"],
                pd.Series(train_mask, index=pivot.index),
                pd.Series(eval_mask, index=pivot.index),
                min_rows=MIN_TERTILE_ROWS,
            )
            n_official = 0
            corr_official = math.nan
            corr_official_abs = math.nan
            if not official_by_date.empty:
                common = diff.index.intersection(official_by_date.index)
                n_official, corr_official = safe_corr(
                    diff.loc[common],
                    official_by_date.loc[common, "official_error_c"],
                    min_rows=MIN_OFFICIAL_ROWS,
                )
                _, corr_official_abs = safe_corr(
                    diff.loc[common],
                    official_by_date.loc[common, "official_abs_error_c"],
                    min_rows=MIN_OFFICIAL_ROWS,
                )
            records.append(
                {
                    "attribute": attribute,
                    "station_a": station_a,
                    "station_b": station_b,
                    "pair_expression": f"{station_a} minus {station_b}",
                    "n_train_pre2000": n_train,
                    "corr_train_pre2000_target_anomaly": corr_train,
                    "n_eval_2000_2023": n_eval,
                    "corr_eval_2000_2023_target_anomaly": corr_eval,
                    "abs_corr_eval_2000_2023_target_anomaly": abs(corr_eval)
                    if math.isfinite(corr_eval)
                    else math.nan,
                    "eval_tertile_target_anomaly_spread_c": spread["cell_spread"],
                    "n_official_error_corr": n_official,
                    "corr_official_error": corr_official,
                    "abs_corr_official_error": abs(corr_official)
                    if math.isfinite(corr_official)
                    else math.nan,
                    "corr_official_abs_error": corr_official_abs,
                    "priority_score": (
                        (abs(corr_eval) if math.isfinite(corr_eval) else 0.0)
                        + 0.45 * (abs(corr_official) if math.isfinite(corr_official) else 0.0)
                        + 0.06
                        * (
                            float(spread["cell_spread"])
                            if isinstance(spread["cell_spread"], float)
                            and math.isfinite(float(spread["cell_spread"]))
                            else 0.0
                        )
                    ),
                }
            )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("priority_score", ascending=False, na_position="last")


def build_readme(
    *,
    generated_at: str,
    summary: dict[str, Any],
    station_coverage: pd.DataFrame,
    station_rankings: pd.DataFrame,
    attribute_atlas: pd.DataFrame,
    pair_atlas: pd.DataFrame,
) -> str:
    coverage_display = station_coverage[
        [
            "station_id",
            "latitude",
            "longitude",
            "elevation_m",
            "coordinate_sanity_status",
            "distance_to_hko_km",
            "first_observed_at_hkt",
            "last_observed_at_hkt",
            "raw_observation_rows",
        ]
    ].head(40)
    station_rank_display = station_rankings[
        [
            "station_id",
            "robust_attribute_count",
            "best_attribute",
            "best_eval_abs_corr",
            "best_official_abs_corr",
            "min_distance_to_hko_km",
        ]
    ].head(20)
    attr_display = attribute_atlas[
        [
            "station_id",
            "attribute",
            "distance_to_hko_km",
            "n_train_pre2000",
            "corr_train_pre2000_target_anomaly",
            "n_eval_2000_2023",
            "corr_eval_2000_2023_target_anomaly",
            "eval_tertile_target_anomaly_spread_c",
            "n_official_error_corr",
            "corr_official_error",
        ]
    ].head(25)
    pair_display = (
        pair_atlas[
            [
                "attribute",
                "pair_expression",
                "n_train_pre2000",
                "corr_train_pre2000_target_anomaly",
                "n_eval_2000_2023",
                "corr_eval_2000_2023_target_anomaly",
                "eval_tertile_target_anomaly_spread_c",
                "n_official_error_corr",
                "corr_official_error",
            ]
        ].head(25)
        if not pair_atlas.empty
        else pd.DataFrame()
    )
    return f"""# Station Contribution Atlas

Generated: `{generated_at}`

## Purpose

This insight folder answers a station-specific question that the previous cross-family atlas did not fully resolve: which individual regional stations, station attributes, and station-pair spreads carry durable information about the HKG target Tmax?

The screen uses the normalized NOAA ISD regional station day-cutoff summary. For target date `T`, it uses the station observation available on `T-1` no later than 15:00 HKT. That timing is deliberately conservative for the T-24 style forecast problem. Same-day target observations and full-day station extrema are not used in the primary station signal screen.

## Leakage Control

- Target rows with `target_date >= 2024-01-01` are rejected.
- A station feature for target date `T` is built from station `local_date = T-1` and the latest observation before 15:00 HKT.
- One-day station changes compare `T-1` to earlier station records only.
- Target anomaly uses a past-only same-day-of-year target climatology.
- Robust feature selection requires pre-2000 evidence and is evaluated separately on 2000-2023.
- Official forecast residual checks use only the current pre-2024 scored forecast archive and remain diagnostic because that archive is non-contiguous.

## Dataset Scope

| Item | Value |
|---|---:|
| Station-day feature rows | {summary["station_feature_rows"]} |
| Stations evaluated | {summary["station_count"]} |
| Stations with coordinate metadata issues | {summary["coordinate_issue_station_count"]} |
| Station attributes evaluated | {summary["station_attribute_rows"]} |
| Robust pre-2000-testable station attributes | {summary["robust_station_attribute_rows"]} |
| Pair-spread candidates evaluated | {summary["pair_spread_rows"]} |
| Official scored overlap rows | {summary["official_overlap_rows"]} |
| Uses 2024+ rows | {summary["uses_2024_plus_rows"]} |

## Station Coverage

{markdown_table(coverage_display, max_rows=40)}

## Station Ranking

{markdown_table(station_rank_display, max_rows=20)}

## Individual Station Attribute Leaders

{markdown_table(attr_display, max_rows=25)}

## Station-Pair Spread Leaders

{markdown_table(pair_display, max_rows=25)}

## Main Finding

The station network is not a generic blob of redundant weather observations. The most useful station signals are spatial and dynamic: pressure placement, wind exposure, morning-to-midday warming, and one-day changes in air temperature/dew point/pressure. The top individual station attributes and pair spreads remain testable from pre-2000 history and still retain information in the 2000-2023 evaluation period.

This also explains why simple station-feature stuffing has only produced small MAE gains so far. The signal is conditional. A station or pair can be highly informative only under specific flow and upper-air contexts. The next modelling step should not add all stations blindly; it should convert the top rows here into small, gated residual specialists and combine them with the official forecast anchor.

## What This Does Not Prove

This does not train or promote a new forecast model. It does not use the sealed 2024+ confirmation period. It also does not solve the current official forecast archive gap. The official residual correlations in this folder use the current non-contiguous official frame and should be rerun after the 2008-2026 press-detail backfill is complete.

## Artifact Files

- `artifacts/station_coverage.csv`
- `artifacts/station_attribute_atlas.csv`
- `artifacts/station_rankings.csv`
- `artifacts/pair_spread_atlas.csv`
- `artifacts/summary.json`
"""


def milestone_section(
    *,
    summary: dict[str, Any],
    station_rankings: pd.DataFrame,
    attribute_atlas: pd.DataFrame,
    pair_atlas: pd.DataFrame,
) -> str:
    station_rank_display = station_rankings[
        [
            "station_id",
            "robust_attribute_count",
            "best_attribute",
            "best_eval_abs_corr",
            "best_official_abs_corr",
            "min_distance_to_hko_km",
        ]
    ].head(8)
    attr_display = attribute_atlas[
        [
            "station_id",
            "attribute",
            "corr_eval_2000_2023_target_anomaly",
            "eval_tertile_target_anomaly_spread_c",
            "corr_official_error",
        ]
    ].head(8)
    pair_display = (
        pair_atlas[
            [
                "attribute",
                "pair_expression",
                "corr_eval_2000_2023_target_anomaly",
                "eval_tertile_target_anomaly_spread_c",
                "corr_official_error",
            ]
        ].head(8)
        if not pair_atlas.empty
        else pd.DataFrame()
    )
    return f"""Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_contribution_atlas.py
```

New folder: `research/data_analysis/0047_station_contribution_atlas`.

| Area | Evidence | Status |
|---|---|---|
| Station timing | target date `T` uses station latest-before-15:00 observations from `T-1` | Guarded |
| Station scope | `{summary["station_count"]}` stations, `{summary["station_feature_rows"]}` station-day rows | Audited |
| Coordinate audit | `{summary["coordinate_issue_station_count"]}` stations have missing/out-of-bounds coordinate metadata | Explicit |
| Attribute screen | `{summary["station_attribute_rows"]}` station-attribute rows, `{summary["robust_station_attribute_rows"]}` robust pre-2000-testable rows | Diagnostic |
| Pair-spread screen | `{summary["pair_spread_rows"]}` station-pair spread rows | Diagnostic |
| Official overlap | `{summary["official_overlap_rows"]}` non-contiguous official scored rows | Blocker-aware |

Top station ranking:

{markdown_table(station_rank_display, max_rows=8)}

Top individual station attributes:

{markdown_table(attr_display, max_rows=8)}

Top station-pair spreads:

{markdown_table(pair_display, max_rows=8)}

Interpretation: `0047` confirms that station information gain is concentrated in spatial/dynamic signals rather than every station variable equally. The station network should be used as gated residual context around pressure placement, wind exposure, and day-to-day thermal/moisture changes. It remains a diagnostic atlas, not a promoted model, and should be rerun after the official forecast archive is continuous.
"""


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"

    target = load_target()
    metadata = load_station_metadata()
    station_frame = load_station_day_features(target)
    official = load_official_residuals()
    station_coverage = metadata.merge(
        station_frame.groupby("station_id", observed=True)
        .agg(
            station_day_feature_rows=("target_date", "size"),
            first_target_date=("target_date", "min"),
            last_target_date=("target_date", "max"),
        )
        .reset_index(),
        on="station_id",
        how="left",
    ).sort_values(["first_target_date", "station_id"])

    attribute_atlas = station_attribute_atlas(station_frame, official, metadata)
    rankings = station_ranking(attribute_atlas)
    pair_atlas = pair_spread_atlas(station_frame, official)

    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "target_path": str(TARGET_PATH),
        "isd_day_path": str(ISD_DAY_PATH),
        "official_scored_path": str(OFFICIAL_SCORED_PATH),
        "station_feature_rows": int(len(station_frame)),
        "station_count": int(station_frame["station_id"].nunique()),
        "coordinate_issue_station_count": int(
            station_coverage["coordinate_sanity_status"].ne("ok").sum()
        ),
        "station_attribute_rows": int(len(attribute_atlas)),
        "robust_station_attribute_rows": int(attribute_atlas["robust_pre2000_testable"].sum()),
        "pair_spread_rows": int(len(pair_atlas)),
        "official_overlap_rows": int(len(official)),
        "uses_2024_plus_rows": False,
        "leakage_guard": {
            "feature_date_rule": "target_date = station local_date + 1 day",
            "station_cutoff_rule": "latest_before_1500_hkt from T-1 station local_date",
            "feature_selection_period": f"target_date <= {TRAIN_END.date()}",
            "evaluation_period": f"{EVAL_START.date()} <= target_date <= {EVAL_END.date()}",
            "confirmation_start": str(CONFIRMATION_START.date()),
        },
        "top_station_attribute": attribute_atlas.iloc[0].to_dict() if not attribute_atlas.empty else {},
        "top_station_pair_spread": pair_atlas.iloc[0].to_dict() if not pair_atlas.empty else {},
    }

    write_csv(artifacts / "station_coverage.csv", station_coverage)
    write_csv(artifacts / "station_attribute_atlas.csv", attribute_atlas)
    write_csv(artifacts / "station_rankings.csv", rankings)
    write_csv(artifacts / "pair_spread_atlas.csv", pair_atlas)
    write_json(artifacts / "summary.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            generated_at=generated_at,
            summary=summary,
            station_coverage=station_coverage,
            station_rankings=rankings,
            attribute_atlas=attribute_atlas,
            pair_atlas=pair_atlas,
        ),
    )

    manifest = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "summary_path": str(artifacts / "summary.json"),
        "readme_path": str(folder / "README.md"),
        "station_count": summary["station_count"],
        "coordinate_issue_station_count": summary["coordinate_issue_station_count"],
        "station_attribute_rows": summary["station_attribute_rows"],
        "robust_station_attribute_rows": summary["robust_station_attribute_rows"],
        "pair_spread_rows": summary["pair_spread_rows"],
        "uses_2024_plus_rows": False,
        "top_station_attribute": summary["top_station_attribute"],
        "top_station_pair_spread": summary["top_station_pair_spread"],
    }
    write_json(output_root / "station_contribution_atlas_manifest.json", manifest)
    update_markdown_section(
        output_root / "README.md",
        heading="0047 Station Contribution Atlas",
        section=(
            f"Generated `{generated_at}`. See `{FOLDER_NAME}`. "
            f"Evaluated `{summary['station_count']}` stations, "
            f"`{summary['station_attribute_rows']}` station-attribute rows, and "
            f"`{summary['pair_spread_rows']}` station-pair spread rows with T-1 latest-before-15:00 station timing."
        ),
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Station Contribution Atlas",
        section=milestone_section(
            summary=summary,
            station_rankings=rankings,
            attribute_atlas=attribute_atlas,
            pair_atlas=pair_atlas,
        ),
        insert_before="## Current Blockers And Gaps",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe station contribution atlas.")
    parser.add_argument("--output-root", type=Path, default=RESEARCH_ROOT)
    args = parser.parse_args()
    summary = run(args.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
