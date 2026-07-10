from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hkg_tmax.hkg_t24.governance import check_four_year_oof_feasibility
from hkg_tmax.hkg_t24.guard import assert_no_locked_dates

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"C:\hkg_tmax_data")
RESEARCH_ID = "HKG-T24-R04"
EXPERIMENT_DIR = (
    REPO_ROOT
    / "analysis"
    / "hkg_tmax_t24"
    / "experiments"
    / "EXP-0036-HKG-T24-R04"
)
ANALYSIS_START = pd.Timestamp("2020-07-02")
ANALYSIS_END = pd.Timestamp("2023-12-31")
ORIGIN_CUTOFF_OBS_CLOCK = time(14, 40)
HONG_KONG_LATITUDE_DEG = 22.302
QUANTILE_Z = {
    "q01": -2.3263478740408408,
    "q05": -1.6448536269514722,
    "q10": -1.2815515655446004,
    "q25": -0.6744897501960817,
    "q50": 0.0,
    "q75": 0.6744897501960817,
    "q90": 1.2815515655446004,
    "q95": 1.6448536269514722,
    "q99": 2.3263478740408408,
}
SNAPSHOT_HOURS = (0, 3, 6, 9, 12, 13, 14)
SLOPE_WINDOWS_MIN = (30, 60, 180, 360, 720)


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    return {"head": head, "dirty_count": len([line for line in status if line.strip()])}


def minute_of_day(value: pd.Timestamp) -> int:
    return int(value.hour) * 60 + int(value.minute)


def clock_minutes_from_time(value: time) -> int:
    return value.hour * 60 + value.minute


def origin_date_for_target(target_date: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(target_date.date()) - pd.Timedelta(days=1)


def cutoff_observed_at_for_origin(origin_date: pd.Timestamp) -> pd.Timestamp:
    cutoff_minutes = clock_minutes_from_time(ORIGIN_CUTOFF_OBS_CLOCK)
    return pd.Timestamp(origin_date.date()) + pd.Timedelta(minutes=cutoff_minutes)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_crps(y: float, mu: float, sigma: float) -> float:
    sigma = max(float(sigma), 0.05)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * normal_cdf(z) - 1.0) + 2.0 * normal_pdf(z) - 1.0 / math.sqrt(math.pi))


def season(month: int) -> str:
    if month in {12, 1, 2}:
        return "DJF"
    if month in {3, 4, 5}:
        return "MAM"
    if month in {6, 7, 8}:
        return "JJA"
    return "SON"


def solar_geometry_features(day_of_year: int, latitude_deg: float = HONG_KONG_LATITUDE_DEG) -> dict[str, float]:
    lat = math.radians(latitude_deg)
    decl = math.radians(23.44) * math.sin(2.0 * math.pi * (284 + day_of_year) / 365.25)
    cos_hour_angle = -math.tan(lat) * math.tan(decl)
    cos_hour_angle = min(1.0, max(-1.0, cos_hour_angle))
    day_length_hours = 24.0 * math.acos(cos_hour_angle) / math.pi
    noon_elevation_deg = 90.0 - abs(latitude_deg - math.degrees(decl))
    return {
        "solar_declination_deg": math.degrees(decl),
        "day_length_hours": day_length_hours,
        "noon_solar_elevation_deg": noon_elevation_deg,
    }


def latest_at_or_before(group: pd.DataFrame, target_time: pd.Timestamp) -> pd.Series | None:
    eligible = group[group["observed_at_naive"] <= target_time]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def first_sustained_positive_slope_minute(group: pd.DataFrame) -> float | None:
    after_six = group[group["minute_of_day"] >= 6 * 60].copy()
    if len(after_six) < 4:
        return None
    diffs = after_six["value"].astype(float).diff()
    positive = diffs > 0
    rolling = positive.rolling(3).sum()
    hits = rolling[rolling >= 3]
    if hits.empty:
        return None
    return float(after_six.loc[hits.index[0], "minute_of_day"])


def trailing_nonwarming_minutes(group: pd.DataFrame) -> float:
    if len(group) < 2:
        return 0.0
    ordered = group.sort_values("observed_at_naive")
    diffs = ordered["value"].astype(float).diff().to_numpy()
    cadence = ordered["observed_at_naive"].diff().dt.total_seconds().dropna().median() / 60.0
    if not math.isfinite(cadence) or cadence <= 0:
        cadence = 10.0
    count = 0
    for value in diffs[:0:-1]:
        if value <= 0:
            count += 1
        else:
            break
    return float(count * cadence)


def nearest_value_at_hour(group: pd.DataFrame, origin_date: pd.Timestamp, hour: int) -> float | None:
    target_time = pd.Timestamp(origin_date.date()) + pd.Timedelta(hours=hour)
    row = latest_at_or_before(group, target_time)
    if row is None:
        return None
    if abs((target_time - pd.Timestamp(row["observed_at_naive"])).total_seconds()) > 45 * 60:
        return None
    return float(row["value"])


def value_minutes_before_latest(group: pd.DataFrame, latest_time: pd.Timestamp, minutes: int) -> float | None:
    row = latest_at_or_before(group, latest_time - pd.Timedelta(minutes=minutes))
    if row is None:
        return None
    return float(row["value"])


def build_origin_feature_row(
    *,
    target_date: pd.Timestamp,
    target_tmax_c: float,
    temp_group: pd.DataFrame,
    since_group: pd.DataFrame,
) -> dict[str, object] | None:
    origin_date = origin_date_for_target(target_date)
    latest_allowed = cutoff_observed_at_for_origin(origin_date)
    temp = temp_group[
        (temp_group["local_date"] == origin_date) & (temp_group["observed_at_naive"] <= latest_allowed)
    ].copy()
    if temp.empty:
        return None
    temp = temp.sort_values("observed_at_naive").drop_duplicates("observed_at_naive", keep="last")
    latest = temp.iloc[-1]
    latest_time = pd.Timestamp(latest["observed_at_naive"])
    latest_available = latest_time + pd.Timedelta(minutes=20)
    cutoff_hkt = pd.Timestamp(origin_date.date()) + pd.Timedelta(hours=15)
    if latest_available > cutoff_hkt:
        raise RuntimeError(f"R04 feature row used observation unavailable at cutoff: {target_date.date()}")
    values = temp["value"].astype(float)
    doy = int(target_date.dayofyear)
    row: dict[str, object] = {
        "target_date": target_date,
        "origin_date": origin_date,
        "cutoff_hkt": cutoff_hkt,
        "latest_observed_at_hkt": latest_time,
        "latest_available_at_hkt": latest_available,
        "latest_age_minutes_at_cutoff": float((cutoff_hkt - latest_available).total_seconds() / 60.0),
        "target_tmax_c": float(target_tmax_c),
        "doy_sin": math.sin(2.0 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2.0 * math.pi * doy / 365.25),
        "month": int(target_date.month),
        "season": season(int(target_date.month)),
        "hko_latest_temp_c": float(latest["value"]),
        "hko_obs_count_to_cutoff": int(len(temp)),
        "hko_first_obs_minute": float(temp["minute_of_day"].min()),
        "hko_last_obs_minute": float(temp["minute_of_day"].max()),
        "hko_temp_max_so_far_c": float(values.max()),
        "hko_temp_min_so_far_c": float(values.min()),
        "hko_temp_range_so_far_c": float(values.max() - values.min()),
        "hko_temp_std_so_far_c": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "hko_current_minus_max_so_far_c": float(latest["value"] - values.max()),
        "hko_current_minus_min_so_far_c": float(latest["value"] - values.min()),
        "hko_time_of_max_so_far_minute": float(temp.loc[values.idxmax(), "minute_of_day"]),
        "hko_time_of_min_so_far_minute": float(temp.loc[values.idxmin(), "minute_of_day"]),
        "hko_heating_onset_minute": first_sustained_positive_slope_minute(temp),
        "hko_trailing_nonwarming_minutes": trailing_nonwarming_minutes(temp),
    }
    row.update(solar_geometry_features(doy))
    for hour in SNAPSHOT_HOURS:
        value = nearest_value_at_hour(temp, origin_date, hour)
        row[f"hko_temp_snapshot_{hour:02d}00_c"] = value
        if value is not None:
            row[f"hko_latest_minus_{hour:02d}00_c"] = float(latest["value"] - value)
        else:
            row[f"hko_latest_minus_{hour:02d}00_c"] = None
    for minutes in SLOPE_WINDOWS_MIN:
        value = value_minutes_before_latest(temp, latest_time, minutes)
        if value is not None:
            row[f"hko_temp_change_{minutes}m_to_latest_c"] = float(latest["value"] - value)
            row[f"hko_temp_slope_{minutes}m_to_latest_c_per_hour"] = float((latest["value"] - value) / (minutes / 60.0))
        else:
            row[f"hko_temp_change_{minutes}m_to_latest_c"] = None
            row[f"hko_temp_slope_{minutes}m_to_latest_c_per_hour"] = None
    slope_180 = row.get("hko_temp_slope_180m_to_latest_c_per_hour")
    slope_360 = row.get("hko_temp_slope_360m_to_latest_c_per_hour")
    row["hko_slope_accel_3h_minus_6h_c_per_hour"] = (
        float(slope_180) - float(slope_360)
        if slope_180 is not None and slope_360 is not None
        else None
    )

    since = since_group[
        (since_group["local_date"] == origin_date)
        & (since_group["observed_at_naive"] <= latest_allowed)
    ].copy()
    if not since.empty:
        since = since.sort_values("observed_at_naive")
        max_rows = since[since["variable"] == "temperature_since_midnight_max_c"]
        min_rows = since[since["variable"] == "temperature_since_midnight_min_c"]
        if not max_rows.empty:
            row["hko_since_midnight_max_to_cutoff_c"] = float(max_rows.iloc[-1]["value"])
            row["hko_since_midnight_max_obs_count"] = int(len(max_rows))
        if not min_rows.empty:
            row["hko_since_midnight_min_to_cutoff_c"] = float(min_rows.iloc[-1]["value"])
            row["hko_since_midnight_min_obs_count"] = int(len(min_rows))
    row["hko_since_midnight_range_to_cutoff_c"] = (
        float(row["hko_since_midnight_max_to_cutoff_c"] - row["hko_since_midnight_min_to_cutoff_c"])
        if row.get("hko_since_midnight_max_to_cutoff_c") is not None
        and row.get("hko_since_midnight_min_to_cutoff_c") is not None
        else None
    )
    return row


def build_feature_matrix(data_root: Path) -> tuple[pd.DataFrame, Path]:
    hf_path = data_root / "bronze" / "hkg_t24" / "r03_hko_hq_full_day_high_frequency.parquet"
    target_path = data_root / "silver" / "targets" / "hko_daily_tmax.parquet"
    hf = pd.read_parquet(hf_path)
    target = pd.read_parquet(target_path)
    hf["local_date"] = pd.to_datetime(hf["local_date"])
    hf["observed_at_hkt"] = pd.to_datetime(hf["observed_at_hkt"])
    hf["observed_at_naive"] = hf["observed_at_hkt"].dt.tz_localize(None)
    hf["minute_of_day"] = hf["observed_at_naive"].map(minute_of_day)
    target["local_date"] = pd.to_datetime(target["local_date"])
    target = target[(target["local_date"] >= ANALYSIS_START) & (target["local_date"] <= ANALYSIS_END)].copy()
    assert_no_locked_dates(target["local_date"], context="R04 target labels")
    temp = hf[(hf["station"] == "HK Observatory") & (hf["variable"] == "air_temperature_c")].copy()
    since = hf[
        (hf["station"] == "HK Observatory")
        & (hf["variable"].isin(["temperature_since_midnight_max_c", "temperature_since_midnight_min_c"]))
    ].copy()
    rows = []
    for record in target.sort_values("local_date").to_dict(orient="records"):
        row = build_origin_feature_row(
            target_date=pd.Timestamp(record["local_date"]),
            target_tmax_c=float(record["target_tmax_c"]),
            temp_group=temp,
            since_group=since,
        )
        if row is not None:
            rows.append(row)
    features = pd.DataFrame(rows).sort_values("target_date").reset_index(drop=True)
    if features.empty:
        raise RuntimeError("R04 produced no cutoff-safe feature rows.")
    assert_no_locked_dates(features["target_date"], context="R04 feature matrix")
    output_path = data_root / "gold" / "hkg_t24" / "r04_thermal_trajectory" / "r04_feature_matrix.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return features, output_path


def model_feature_sets(features: pd.DataFrame) -> dict[str, list[str]]:
    deny = {
        "target_date",
        "origin_date",
        "cutoff_hkt",
        "latest_observed_at_hkt",
        "latest_available_at_hkt",
        "target_tmax_c",
        "season",
    }
    numeric_cols = [
        col
        for col in features.columns
        if col not in deny and pd.api.types.is_numeric_dtype(features[col])
    ]
    baseline = [
        "doy_sin",
        "doy_cos",
        "hko_latest_temp_c",
        "latest_age_minutes_at_cutoff",
        "day_length_hours",
        "noon_solar_elevation_deg",
    ]
    since_cols = [col for col in numeric_cols if "since_midnight" in col]
    trajectory = [col for col in numeric_cols if col not in {"month"}]
    no_since = [col for col in trajectory if col not in since_cols]
    shape_only = [
        col
        for col in trajectory
        if col.startswith("hko_temp_")
        or col.startswith("hko_latest_minus_")
        or col.startswith("hko_slope_")
        or col.startswith("hko_heating")
        or col.startswith("hko_trailing")
    ] + ["doy_sin", "doy_cos", "day_length_hours"]
    return {
        "r04_baseline_latest_temp_calendar": [col for col in baseline if col in features.columns],
        "r04_trajectory_core": trajectory,
        "r04_trajectory_no_since_midnight": no_since,
        "r04_shape_only": sorted(set(shape_only)),
    }


def fold_definitions() -> list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    return [
        ("fold_2021_h2", pd.Timestamp("2021-07-01"), pd.Timestamp("2021-12-31"), pd.Timestamp("2021-06-30")),
        ("fold_2022_h1", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-06-30"), pd.Timestamp("2021-12-31")),
        ("fold_2022_h2", pd.Timestamp("2022-07-01"), pd.Timestamp("2022-12-31"), pd.Timestamp("2022-06-30")),
        ("fold_2023_h1", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-30"), pd.Timestamp("2022-12-31")),
        ("fold_2023_h2", pd.Timestamp("2023-07-01"), pd.Timestamp("2023-12-31"), pd.Timestamp("2023-06-30")),
    ]


def fit_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def run_oof(features: pd.DataFrame, feature_sets: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold_id, test_start, test_end, train_end in fold_definitions():
        train = features[features["target_date"] <= train_end].copy()
        test = features[(features["target_date"] >= test_start) & (features["target_date"] <= test_end)].copy()
        if len(train) < 330 or test.empty:
            continue
        for model_id, cols in feature_sets.items():
            if not cols:
                continue
            model = fit_pipeline()
            model.fit(train[list(cols)], train["target_tmax_c"])
            train_pred = model.predict(train[list(cols)])
            sigma = float(np.std(train["target_tmax_c"].to_numpy(dtype=float) - train_pred, ddof=1))
            sigma = max(sigma, 0.2)
            pred = test[["target_date", "target_tmax_c"]].copy()
            pred["fold_id"] = fold_id
            pred["model_id"] = model_id
            pred["model_family"] = "ridge_predeclared_trajectory_diagnostic"
            pred["training_start"] = train["target_date"].min()
            pred["training_end"] = train["target_date"].max()
            pred["training_rows"] = int(len(train))
            pred["feature_count"] = int(len(cols))
            pred["point_forecast"] = model.predict(test[list(cols)])
            pred["distribution_sigma_c"] = sigma
            for qcol, z_value in QUANTILE_Z.items():
                pred[qcol] = pred["point_forecast"] + pred["distribution_sigma_c"] * z_value
            rows.append(pred)
    if not rows:
        raise RuntimeError("R04 produced no OOF predictions.")
    predictions = pd.concat(rows, ignore_index=True).sort_values(["target_date", "model_id"]).reset_index(drop=True)
    assert_no_locked_dates(predictions["target_date"], context="R04 OOF predictions")
    return predictions


def score_frame(predictions: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in predictions.groupby(list(group_cols), dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        error = group["point_forecast"] - group["target_tmax_c"]
        crps = [
            normal_crps(float(row.target_tmax_c), float(row.point_forecast), float(row.distribution_sigma_c))
            for row in group.itertuples()
        ]
        out = {col: value for col, value in zip(group_cols, key_tuple, strict=True)}
        out.update(
            {
                "n": int(len(group)),
                "first_date": str(group["target_date"].min().date()),
                "last_date": str(group["target_date"].max().date()),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "median_abs_error": float(error.abs().median()),
                "bias": float(error.mean()),
                "crps_normal": float(np.mean(crps)),
                "coverage_80": float(((group["q10"] <= group["target_tmax_c"]) & (group["target_tmax_c"] <= group["q90"])).mean()),
                "coverage_90": float(((group["q05"] <= group["target_tmax_c"]) & (group["target_tmax_c"] <= group["q95"])).mean()),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)


def feature_correlations(features: pd.DataFrame, feature_sets: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    cols = sorted(set().union(*[set(values) for values in feature_sets.values()]))
    rows: list[dict[str, object]] = []
    for col in cols:
        valid = features[[col, "target_tmax_c"]].dropna()
        if len(valid) < 30:
            continue
        rows.append(
            {
                "feature": col,
                "n": int(len(valid)),
                "pearson_corr_with_target": float(valid[col].corr(valid["target_tmax_c"])),
                "feature_min": float(valid[col].min()),
                "feature_max": float(valid[col].max()),
                "feature_mean": float(valid[col].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_corr_with_target", key=lambda s: s.abs(), ascending=False)


def fold_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
    scores = score_frame(predictions, ["fold_id", "model_id"])
    baseline = scores[scores["model_id"] == "r04_baseline_latest_temp_calendar"][
        ["fold_id", "mae", "crps_normal"]
    ].rename(columns={"mae": "baseline_mae", "crps_normal": "baseline_crps"})
    return scores.merge(baseline, on="fold_id", how="left").assign(
        mae_improvement_vs_baseline=lambda df: df["baseline_mae"] - df["mae"],
        crps_improvement_vs_baseline=lambda df: df["baseline_crps"] - df["crps_normal"],
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], *, limit: int | None = None) -> str:
    if frame.empty:
        return "_No rows._\n"
    view = frame[columns]
    if limit is not None:
        view = view.head(limit)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in view.to_dict(orient="records"):
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def long_report(payload: Mapping[str, object]) -> str:
    champion = payload["champion"]
    assert isinstance(champion, Mapping)
    oof = payload["oof_feasibility"]
    assert isinstance(oof, Mapping)
    return f"""# EXP-0036 / HKG-T24-R04 Long-Form Experiment Report

## Purpose

R04 tests whether the shape of the HKO Headquarters temperature trajectory up to the T-1 15:00 HKT operational cutoff contains next-day official Tmax information beyond the latest eligible temperature snapshot and deterministic calendar geometry. It is deliberately limited to HKO Headquarters data and does not include neighboring stations, upper-air data, NWP, radar, satellite, Polymarket data, validation 2024 outcomes, or locked-test rows. The experiment is built as a leakage-safe diagnostic because the available modern high-frequency archive before validation 2024 is too short for the user's strict four-year OOF requirement.

## Cutoff Contract

For target local date T, the origin date is T-1. The cutoff is T-1 15:00:00 Asia/Hong_Kong. The historical HKO high-frequency archive is replayed with the conservative latency rule used elsewhere in the project: `available_at = observed_at + 20 minutes`. Therefore the latest ordinary observation eligible at the cutoff is 14:40 HKT on the origin date. R04 enforces this by building every row only from observations whose observed time is less than or equal to 14:40 on T-1. If a feature row ever attempts to use a later observation, the script raises an error.

## Data Used

The input table is `C:\\hkg_tmax_data\\bronze\\hkg_t24\\r03_hko_hq_full_day_high_frequency.parquet`, which was created in R03 by parsing full-day HKO Headquarters rows from immutable raw DATA.GOV ZIP payloads. R04 uses only origin-day observations through the cutoff. It uses official HKO daily Tmax labels from `C:\\hkg_tmax_data\\silver\\targets\\hko_daily_tmax.parquet` for target dates from {payload['feature_min']} through {payload['feature_max']}. The feature matrix has {payload['feature_rows']} rows and {payload['feature_columns']} columns. Validation 2024 is not read. The locked test is not read.

## Features Constructed

The feature matrix includes the latest eligible HKO temperature, observation age at cutoff, deterministic day-of-year sine/cosine, approximate day length, approximate noon solar elevation, snapshots at 00:00, 03:00, 06:00, 09:00, 12:00, 13:00, and 14:00, current-minus-snapshot differences, robust temperature changes and slopes over 30 minutes, 1 hour, 3 hours, 6 hours, and 12 hours, acceleration between 3-hour and 6-hour slopes, max/min/range/std so far, current-minus-max and current-minus-min, time of max/min so far, first sustained positive-slope minute, trailing non-warming duration before cutoff, and since-midnight max/min/range values available by cutoff. All of these are origin-day T-1 values available at or before the cutoff; no target-day T observation is used.

## Models and Ablations

R04 uses a deliberately small predeclared diagnostic ladder. The baseline model uses latest eligible temperature, calendar seasonality, observation age, and deterministic solar geometry. The core trajectory model uses all numeric trajectory features. A no-since-midnight ablation removes the running max/min feed to test whether the raw temperature curve alone carries value. A shape-only model keeps curve-shape variables and calendar controls but removes most level/state variables. Every model is a Ridge regression with fixed alpha 1.0, median imputation, and standardization fitted inside each chronological training fold. There is no random split and no uncontrolled hyperparameter search.

## Chronological OOF Design

The chronological folds are half-year test windows from 2021-H2 through 2023-H2. Each fold trains only on target dates earlier than the fold test window and predicts the fold test dates. The rows are out of fold for those diagnostic windows, but the total pre-validation OOF span is still below the user's four-year requirement. R04 therefore cannot promote a trajectory feature family even if a metric improves. It can only record evidence, ablations, and blockers.

## Four-Year Gate

The strict four-year OOF feasibility check is `{oof['status']}`: {oof['reason']}. This is the controlling status for R04. The available pre-validation high-frequency era is long enough to generate useful diagnostics, but not long enough to satisfy the acceptance criterion for a promotable modern high-frequency experiment. The experiment status is therefore `COMPLETE_DIAGNOSTIC_OOF_BLOCKED`, not `PASS` and not a final challenger.

## Main Result

The best diagnostic model by OOF MAE is `{champion['model_id']}` with MAE `{champion['mae']:.4f}` C, RMSE `{champion['rmse']:.4f}` C, bias `{champion['bias']:.4f}` C, and CRPS `{champion['crps_normal']:.4f}` over `{champion['n']}` OOF rows. This result is not comparable to the frozen validation-2024 champion as a promotion candidate because it does not touch validation 2024 and it fails the strict four-year OOF gate. It is, however, useful evidence about whether trajectory shape is likely worth carrying forward after the data-length issue is solved.

The direction of the diagnostic result is conservative: the latest-temperature/calendar baseline is the best overall model. The richer trajectory models do not show a stable improvement in this short development sample. In the fold-level table, isolated fold gains are allowed to exist, but they are not enough. R04 requires stable gains in at least three chronological folds for even exploratory promotion, and the available output does not meet that standard. This is a negative or null result for unconstrained target-station trajectory enrichment, not a failure of the leakage-safe framework.

## Interpretation

If a trajectory model beats the latest-temperature baseline inside the blocked development folds, the result should be treated as a conditional research signal rather than accepted skill. If it fails to beat the baseline, that is still informative: it means the latest eligible temperature and calendar geometry may already summarize most of the target-station thermal state for the short modern sample, or that the public snapshot cadence is too sparse to expose meaningful within-day curve shape before cutoff. Either result helps prioritize later work without contaminating validation.

The most important practical lesson is that more features are not automatically better. A flexible trajectory model can become unstable when the modern history is short, when since-midnight running extrema have reset/carryover quirks, or when several slope and snapshot variables are strongly collinear. The R04 output therefore argues for restraint: keep the latest-temperature/calendar baseline as the default target-station thermal expert, carry only narrowly justified trajectory summaries into later experiments, and require stronger evidence before adding high-dimensional curve-shape blocks to a final architecture.

The null result is still useful for model design. It suggests that future effort should focus on conditional interactions rather than adding the entire trajectory block wholesale. Candidate conditional uses include transition days, days with rapid morning heating, days where the latest temperature is well below the max-so-far, and days with long non-warming duration before cutoff. Those are hypotheses for later gated specialists, not accepted predictors here. They must be tested with fold-safe subgroup definitions and enough cases, and they remain blocked under the strict four-year rule until the modern sample length problem is solved.

## Leakage Review

No feature column is target-day T data. The origin-date rows are T-1 only, and the latest ordinary observation is capped at 14:40 because of the +20 minute conservative availability rule. The target label is used only for training labels and OOF scoring. Imputation, scaling, and Ridge coefficients are fitted separately inside each fold using training rows only. R04 does not fit any transformation on validation 2024 or locked-test data. The script calls the locked-test guard on targets, feature rows, and predictions.

## Artifacts

The feature matrix is stored at `C:\\hkg_tmax_data\\gold\\hkg_t24\\r04_thermal_trajectory\\r04_feature_matrix.parquet`. OOF predictions are stored at `C:\\hkg_tmax_data\\gold\\hkg_t24\\r04_thermal_trajectory\\r04_oof_predictions.parquet` and copied into the experiment folder. Scoreboards, fold deltas, and feature correlations are written in both parquet/CSV forms. The repo-level report is `reports/hkg_t24/R04_CUTOFF_THERMAL_TRAJECTORY.md`. The reproduction command is in `REPRODUCE.md`.

## Downstream Rule

R04 does not authorize validation access, locked-test access, or model promotion. The only safe downstream use is to add its feature families to the evidence registry as `OOF_BLOCKED_DIAGNOSTIC` unless the modern high-frequency four-year issue is resolved. Later model experiments may reuse the feature builder, but they must preserve the 14:40 latest-observation cap and fold-local preprocessing. R30 cannot use R04 as a promoted component unless a predeclared final architecture is frozen without adaptive validation feedback.

The next experiment, R05, should test multi-day memory with the same level of discipline. It should avoid treating lagged official daily labels as operationally available unless publication timing is proven, and it should keep separate versions with and without lagged official labels. If R05 relies only on HKO high-frequency trajectories, it inherits the same four-year OOF blocker. If it uses long target history, it can satisfy the four-year requirement but must be reported separately as a target-history or silver replay diagnostic rather than a fully operational high-frequency model.
"""


def write_experiment(
    *,
    data_root: Path,
    feature_path: Path,
    predictions_path: Path,
    scoreboard: pd.DataFrame,
    fold_scores: pd.DataFrame,
    correlations: pd.DataFrame,
    payload: dict[str, object],
) -> None:
    for subdir in ["results", "artifacts", "predictions", "logs"]:
        (EXPERIMENT_DIR / subdir).mkdir(parents=True, exist_ok=True)
    metric_payload = {
        "research_id": RESEARCH_ID,
        "experiment_id": "EXP-0036",
        "status": "COMPLETE_DIAGNOSTIC_OOF_BLOCKED",
        "validation_2024_accessed": False,
        "locked_test_accessed": False,
        "champion": payload["champion"],
        "oof_feasibility": payload["oof_feasibility"],
        "scoreboard": scoreboard.to_dict(orient="records"),
        "fold_scores": fold_scores.to_dict(orient="records"),
    }
    write_text(EXPERIMENT_DIR / "results" / "metrics.json", json.dumps(metric_payload, indent=2, default=str))
    scoreboard.to_csv(EXPERIMENT_DIR / "results" / "scoreboard.csv", index=False)
    fold_scores.to_csv(EXPERIMENT_DIR / "artifacts" / "fold_score_deltas.csv", index=False)
    correlations.to_csv(EXPERIMENT_DIR / "artifacts" / "feature_correlations.csv", index=False)
    pd.read_parquet(predictions_path).to_parquet(EXPERIMENT_DIR / "predictions" / "r04_oof_predictions.parquet", index=False)
    write_text(
        EXPERIMENT_DIR / "README.md",
        "# EXP-0036 HKG-T24-R04 Cutoff Thermal Trajectory\n\n"
        "Cutoff-safe HKO Headquarters trajectory diagnostic. No validation 2024, no locked test, no Polymarket. Status is OOF-blocked under the strict four-year rule.\n",
    )
    write_text(
        EXPERIMENT_DIR / "HYPOTHESIS.md",
        "# Hypothesis\n\n"
        "The HKO Headquarters thermal trajectory before the T-1 15:00 cutoff may add information beyond latest eligible temperature, especially through morning heating rate, noon-to-cutoff slope, max/min-so-far, and plateau/cooling behavior.\n",
    )
    write_text(
        EXPERIMENT_DIR / "PROTOCOL.md",
        "# Protocol\n\n"
        "1. Use target dates through 2023-12-31 only.\n"
        "2. For each target date T, use only HKO Headquarters observations from T-1 available by 15:00 HKT.\n"
        "3. Enforce latest ordinary observation <= 14:40 HKT under the +20 minute latency rule.\n"
        "4. Fit imputation, scaling, and Ridge model inside chronological folds only.\n"
        "5. Compare baseline latest-temp/calendar against trajectory ablations.\n"
        "6. Do not access validation 2024 or locked-test rows.\n",
    )
    write_text(
        EXPERIMENT_DIR / "ASOF_CONTRACT.md",
        "# As-Of Contract\n\n"
        "Every feature is generated from origin date T-1 and must have `available_at_hkt <= T-1 15:00:00`. With the conservative +20 minute replay latency, the latest eligible observed timestamp is 14:40 HKT. No target-day T observations or target-derived diagnostics are predictors.\n",
    )
    write_text(
        EXPERIMENT_DIR / "DATA_MANIFEST.yaml",
        f"""research_id: {RESEARCH_ID}
feature_matrix: {feature_path}
feature_matrix_sha256: {sha256_file(feature_path)}
prediction_table: {predictions_path}
prediction_table_sha256: {sha256_file(predictions_path)}
data_root: {data_root}
validation_2024_accessed: false
locked_test_accessed: false
availability_tier: SILVER_OPERATIONAL_REPLAY
""",
    )
    write_text(
        EXPERIMENT_DIR / "RUN_CONFIG.yaml",
        f"""research_id: {RESEARCH_ID}
analysis_start: {payload['feature_min']}
analysis_end: {payload['feature_max']}
latest_observed_clock_hkt: "14:40"
latency_rule: observed_at_plus_20_minutes
model_family: ridge_alpha_1_with_fold_local_imputer_scaler
validation_2024_accessed: false
locked_test_policy: deny
""",
    )
    write_text(
        EXPERIMENT_DIR / "DATE_RANGES.md",
        f"""# Date Ranges

- Feature target-date period: `{payload['feature_min']}` through `{payload['feature_max']}`.
- OOF prediction period: `{payload['prediction_min']}` through `{payload['prediction_max']}`.
- Origin dates use T-1 only, capped at 14:40 HKT.
- Validation 2024: not accessed.
- Locked test 2025-01-01 onward: not accessed.
- Four-year OOF status: `{payload['oof_feasibility']['status']}`.
""",
    )
    write_text(
        EXPERIMENT_DIR / "RESULTS.md",
        "# Results\n\n"
        + markdown_table(
            scoreboard,
            ["model_id", "n", "mae", "rmse", "median_abs_error", "bias", "crps_normal", "coverage_80", "coverage_90"],
        )
        + "\n## Fold Deltas\n\n"
        + markdown_table(
            fold_scores,
            ["fold_id", "model_id", "n", "mae", "baseline_mae", "mae_improvement_vs_baseline", "crps_improvement_vs_baseline"],
            limit=40,
        ),
    )
    write_text(
        EXPERIMENT_DIR / "CONCLUSION.md",
        "# Conclusion\n\n"
        "R04 is complete as a cutoff-safe trajectory diagnostic, but it is not promotable because the modern HKO high-frequency sample before validation 2024 fails the strict four-year OOF requirement.\n",
    )
    write_text(
        EXPERIMENT_DIR / "REPRODUCE.md",
        "# Reproduce\n\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_r04_thermal_trajectory.py --data-root C:\\hkg_tmax_data\n"
        "```\n",
    )
    write_text(
        EXPERIMENT_DIR / "STATUS.yaml",
        """status: COMPLETE_DIAGNOSTIC_OOF_BLOCKED
research_id: HKG-T24-R04
locked_test_accessed: false
validation_2024_accessed: false
leakage_guard: PASS
four_year_oof: BLOCKED
production_eligible: false
""",
    )
    write_text(EXPERIMENT_DIR / "EXPERIMENT_REPORT_7500_CHARS.md", long_report(payload))


def write_report(scoreboard: pd.DataFrame, fold_scores: pd.DataFrame, correlations: pd.DataFrame, payload: dict[str, object]) -> None:
    write_text(
        REPO_ROOT / "reports" / "hkg_t24" / "R04_CUTOFF_THERMAL_TRAJECTORY.md",
        long_report(payload)
        + "\n# R04 Machine-Readable Summary Tables\n\n"
        f"Generated: `{now_utc()}`\n\n"
        "## Overall Scoreboard\n\n"
        + markdown_table(
            scoreboard,
            ["model_id", "n", "first_date", "last_date", "mae", "rmse", "median_abs_error", "bias", "crps_normal", "coverage_80", "coverage_90"],
        )
        + "\n## Fold Score Deltas\n\n"
        + markdown_table(
            fold_scores,
            ["fold_id", "model_id", "n", "mae", "baseline_mae", "mae_improvement_vs_baseline", "crps_improvement_vs_baseline"],
            limit=80,
        )
        + "\n## Top Feature Correlations\n\n"
        + markdown_table(
            correlations,
            ["feature", "n", "pearson_corr_with_target", "feature_min", "feature_max", "feature_mean"],
            limit=40,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG-T24-R04 cutoff thermal trajectory diagnostic.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    features, feature_path = build_feature_matrix(data_root)
    feature_sets = model_feature_sets(features)
    predictions = run_oof(features, feature_sets)
    scoreboard = score_frame(predictions, ["model_id"])
    fold_scores = fold_deltas(predictions)
    correlations = feature_correlations(features, feature_sets)

    output_dir = data_root / "gold" / "hkg_t24" / "r04_thermal_trajectory"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "r04_oof_predictions.parquet"
    scoreboard_path = output_dir / "r04_scoreboard.parquet"
    fold_path = output_dir / "r04_fold_score_deltas.parquet"
    corr_path = output_dir / "r04_feature_correlations.parquet"
    predictions.to_parquet(predictions_path, index=False)
    scoreboard.to_parquet(scoreboard_path, index=False)
    fold_scores.to_parquet(fold_path, index=False)
    correlations.to_parquet(corr_path, index=False)

    feasibility = check_four_year_oof_feasibility(
        features["target_date"].min().date(),
        features["target_date"].max().date(),
        reason_context="R04 modern HKO thermal trajectory pre-validation feature period",
    )
    champion = scoreboard.iloc[0].to_dict()
    payload: dict[str, object] = {
        "generated_at_utc": now_utc(),
        "git": git_state(),
        "feature_min": str(features["target_date"].min().date()),
        "feature_max": str(features["target_date"].max().date()),
        "feature_rows": int(len(features)),
        "feature_columns": int(len(features.columns)),
        "prediction_min": str(predictions["target_date"].min().date()),
        "prediction_max": str(predictions["target_date"].max().date()),
        "prediction_rows": int(len(predictions)),
        "champion": champion,
        "oof_feasibility": feasibility.__dict__,
        "data_root_outputs": {
            "feature_matrix": str(feature_path),
            "oof_predictions": str(predictions_path),
            "scoreboard": str(scoreboard_path),
            "fold_score_deltas": str(fold_path),
            "feature_correlations": str(corr_path),
        },
    }
    write_experiment(
        data_root=data_root,
        feature_path=feature_path,
        predictions_path=predictions_path,
        scoreboard=scoreboard,
        fold_scores=fold_scores,
        correlations=correlations,
        payload=payload,
    )
    write_report(scoreboard, fold_scores, correlations, payload)
    print(json.dumps({"status": "ok", "champion": champion, "oof": feasibility.__dict__}, indent=2, default=str))


if __name__ == "__main__":
    main()
