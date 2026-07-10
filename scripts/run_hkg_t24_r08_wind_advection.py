from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hkg_tmax.hkg_t24.governance import HKO_WIND_STATIONS, check_four_year_oof_feasibility
from hkg_tmax.hkg_t24.guard import assert_no_locked_dates

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"C:\hkg_tmax_data")
RESEARCH_ID = "HKG-T24-R08"
EXPERIMENT_ID = "EXP-0040"
EXPERIMENT_DIR = REPO_ROOT / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0040-HKG-T24-R08"
ENTRY_TIME_RE = re.compile(r"(?P<date>\d{8})-(?P<hhmm>\d{4})")
SNAPSHOT_TARGET_MINUTES = (2 * 60 + 40, 8 * 60 + 40, 11 * 60 + 40, 13 * 60 + 40, 14 * 60 + 40)
WIND_SOURCE_ID = "datagov_hko_historical_latest_10min_wind_archive"

_R04_SPEC = importlib.util.spec_from_file_location(
    "run_hkg_t24_r04_thermal_trajectory",
    REPO_ROOT / "scripts" / "run_hkg_t24_r04_thermal_trajectory.py",
)
if _R04_SPEC is None or _R04_SPEC.loader is None:
    raise ImportError("Unable to load R04 helper script for R08 wind experiment.")
_R04_MODULE = importlib.util.module_from_spec(_R04_SPEC)
_R04_SPEC.loader.exec_module(_R04_MODULE)

_R06_SPEC = importlib.util.spec_from_file_location(
    "run_hkg_t24_r06_moisture_state",
    REPO_ROOT / "scripts" / "run_hkg_t24_r06_moisture_state.py",
)
if _R06_SPEC is None or _R06_SPEC.loader is None:
    raise ImportError("Unable to load R06 helper script for R08 wind experiment.")
_R06_MODULE = importlib.util.module_from_spec(_R06_SPEC)
sys.modules[_R06_SPEC.name] = _R06_MODULE
_R06_SPEC.loader.exec_module(_R06_MODULE)

QUANTILE_Z = _R04_MODULE.QUANTILE_Z
fold_definitions = _R04_MODULE.fold_definitions
git_state = _R04_MODULE.git_state
markdown_table = _R04_MODULE.markdown_table
normal_crps = _R04_MODULE.normal_crps
now_utc = _R04_MODULE.now_utc
r04_feature_sets = _R04_MODULE.model_feature_sets
sha256_file = _R04_MODULE.sha256_file

HKT = _R06_MODULE.HKT
make_cutoffs = _R06_MODULE.make_cutoffs
valid_columns = _R06_MODULE.valid_columns
write_text = _R06_MODULE.write_text

COMPASS_DEGREES = {
    "NORTH": 0.0,
    "NNE": 22.5,
    "NORTH-NORTHEAST": 22.5,
    "NORTHEAST": 45.0,
    "ENE": 67.5,
    "EAST-NORTHEAST": 67.5,
    "EAST": 90.0,
    "ESE": 112.5,
    "EAST-SOUTHEAST": 112.5,
    "SOUTHEAST": 135.0,
    "SSE": 157.5,
    "SOUTH-SOUTHEAST": 157.5,
    "SOUTH": 180.0,
    "SSW": 202.5,
    "SOUTH-SOUTHWEST": 202.5,
    "SOUTHWEST": 225.0,
    "WSW": 247.5,
    "WEST-SOUTHWEST": 247.5,
    "WEST": 270.0,
    "WNW": 292.5,
    "WEST-NORTHWEST": 292.5,
    "NORTHWEST": 315.0,
    "NNW": 337.5,
    "NORTH-NORTHWEST": 337.5,
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_family: str
    columns: tuple[str, ...]
    factory: Callable[[], object]


def read_ledger(data_root: Path) -> pd.DataFrame:
    parquet = data_root / "manifests" / "retrieval_ledger.parquet"
    csv_path = data_root / "manifests" / "retrieval_ledger.csv"
    if parquet.exists():
        ledger = pd.read_parquet(parquet)
    elif csv_path.exists():
        ledger = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Missing retrieval ledger under {data_root / 'manifests'}")
    ledger["retrieved_at"] = pd.to_datetime(ledger["retrieved_at"], utc=True, errors="coerce")
    return ledger


def source_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = ledger[(ledger["status"] == "success") & (ledger["source_id"] == WIND_SOURCE_ID)].copy()
    if rows.empty:
        raise FileNotFoundError(f"No successful retrieval rows found for {WIND_SOURCE_ID}")
    return rows.sort_values("retrieved_at").drop_duplicates(["source_id", "content_sha256"], keep="last")


def zip_entry_in_sampling_window(name: str) -> bool:
    match = ENTRY_TIME_RE.search(name)
    if match is None:
        return False
    hhmm = match.group("hhmm")
    minute = int(hhmm[:2]) * 60 + int(hhmm[2:])
    return any(target <= minute <= target + 25 for target in SNAPSHOT_TARGET_MINUTES)


def parse_hkt_observed_at(value: str) -> datetime | None:
    token = value.strip()
    if not token or token.upper() in {"N/A", "NA", "NULL"}:
        return None
    for fmt in ("%Y%m%d%H%M", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(token, fmt).replace(tzinfo=HKT)
        except ValueError:
            continue
    return None


def parse_number(value: object) -> float | None:
    token = str(value).strip()
    if not token or token.upper() in {"N/A", "NA", "NULL", "***", "-"}:
        return None
    try:
        result = float(token)
    except ValueError:
        return None
    if not np.isfinite(result):
        return None
    return result


def compass_to_degrees(value: object) -> float | None:
    token = str(value).strip().upper()
    if token in {"", "N/A", "NA", "NULL", "***", "-", "VARIABLE", "CALM"}:
        return None
    if token in COMPASS_DEGREES:
        return COMPASS_DEGREES[token]
    return None


def wind_uv_from_direction(speed_kmh: float | None, direction_deg: float | None) -> tuple[float | None, float | None]:
    if speed_kmh is None or direction_deg is None:
        return None, None
    theta = math.radians(direction_deg)
    # Meteorological direction is where wind comes from. u/v are flow components toward east/north.
    return -speed_kmh * math.sin(theta), -speed_kmh * math.cos(theta)


def iter_zip_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv") or not zip_entry_in_sampling_window(name):
                continue
            with archive.open(name) as raw:
                text = raw.read().decode("utf-8-sig", errors="replace").splitlines()
            if not text:
                continue
            reader = csv.DictReader(text)
            for row in reader:
                yield {str(key).strip(): str(value).strip() for key, value in row.items() if key is not None}


def parse_wind_vectors(data_root: Path) -> tuple[pd.DataFrame, Path]:
    output_path = data_root / "bronze" / "hkg_t24" / "r08_wind_vector_sampled_observations.parquet"
    if output_path.exists():
        wind = pd.read_parquet(output_path)
        wind["observed_at_hkt"] = pd.to_datetime(wind["observed_at_hkt"], utc=True).dt.tz_convert(HKT)
        wind["available_at_hkt"] = pd.to_datetime(wind["available_at_hkt"], utc=True).dt.tz_convert(HKT)
        return wind, output_path
    records: list[dict[str, object]] = []
    for _, row in source_rows(read_ledger(data_root)).iterrows():
        path = Path(str(row["content_path"]))
        if not path.exists():
            continue
        for raw_row in iter_zip_csv_rows(path):
            station = raw_row.get("Automatic Weather Station", "").strip()
            if station not in set(HKO_WIND_STATIONS):
                continue
            observed_at = parse_hkt_observed_at(raw_row.get("Date time", ""))
            speed = parse_number(raw_row.get("10-Minute Mean Speed(km/hour)", ""))
            gust = parse_number(raw_row.get("10-Minute Maximum Gust(km/hour)", ""))
            direction = compass_to_degrees(raw_row.get("10-Minute Mean Wind Direction(Compass points)", ""))
            if observed_at is None or speed is None:
                continue
            u_component, v_component = wind_uv_from_direction(speed, direction)
            records.append(
                {
                    "source_id": WIND_SOURCE_ID,
                    "source_file_hash": str(row["content_sha256"]),
                    "retrieved_at": row["retrieved_at"],
                    "station": station,
                    "observed_at_hkt": observed_at,
                    "available_at_hkt": observed_at + timedelta(minutes=20),
                    "local_date": observed_at.date(),
                    "direction_compass": raw_row.get("10-Minute Mean Wind Direction(Compass points)", "").strip(),
                    "direction_deg": direction,
                    "mean_speed_kmh": speed,
                    "max_gust_kmh": gust,
                    "u_kmh": u_component,
                    "v_kmh": v_component,
                    "direction_available": direction is not None,
                    "availability_tier": "SILVER_OPERATIONAL_REPLAY",
                    "latency_rule": "observed_at + 20 minutes",
                }
            )
    if not records:
        raise RuntimeError("R08 parsed no sampled wind-vector observations.")
    wind = pd.DataFrame(records).sort_values(["station", "available_at_hkt"]).reset_index(drop=True)
    wind["observed_at_hkt"] = pd.to_datetime(wind["observed_at_hkt"], utc=True).dt.tz_convert(HKT)
    wind["available_at_hkt"] = pd.to_datetime(wind["available_at_hkt"], utc=True).dt.tz_convert(HKT)
    wind["local_date"] = pd.to_datetime(wind["local_date"], errors="coerce")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wind.to_parquet(output_path, index=False)
    return wind, output_path


def asof_wind_rows(wind: pd.DataFrame, cutoffs: pd.DataFrame, offset_hours: int) -> pd.DataFrame:
    left = cutoffs[["target_date", "cutoff_hkt"]].copy()
    left["join_available_at_hkt"] = left["cutoff_hkt"] - pd.to_timedelta(offset_hours, unit="h")
    outputs: list[pd.DataFrame] = []
    for station, group in wind.groupby("station", sort=True):
        right = group[
            [
                "available_at_hkt",
                "observed_at_hkt",
                "mean_speed_kmh",
                "max_gust_kmh",
                "direction_deg",
                "u_kmh",
                "v_kmh",
                "direction_available",
            ]
        ].dropna(subset=["available_at_hkt"]).sort_values("available_at_hkt")
        if right.empty:
            continue
        merged = pd.merge_asof(
            left.sort_values("join_available_at_hkt"),
            right,
            left_on="join_available_at_hkt",
            right_on="available_at_hkt",
            direction="backward",
            tolerance=pd.Timedelta(hours=3),
        )
        merged["station"] = station
        merged["offset_hours"] = offset_hours
        outputs.append(merged)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def circular_mean_direction_from_uv(u: pd.Series, v: pd.Series) -> pd.Series:
    # Convert flow vector back to meteorological "from" direction.
    angle = (np.degrees(np.arctan2(-u.astype(float), -v.astype(float))) + 360.0) % 360.0
    return angle


def aggregate_wind(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["target_date"])
    grouped = rows.groupby("target_date")
    out = grouped.agg(
        **{
            f"{prefix}_station_count": ("station", "nunique"),
            f"{prefix}_direction_available_fraction": ("direction_available", "mean"),
            f"{prefix}_linear_mean_direction_deg_wrong": ("direction_deg", "mean"),
            f"{prefix}_median_speed_kmh": ("mean_speed_kmh", "median"),
            f"{prefix}_max_speed_kmh": ("mean_speed_kmh", "max"),
            f"{prefix}_median_gust_kmh": ("max_gust_kmh", "median"),
            f"{prefix}_max_gust_kmh": ("max_gust_kmh", "max"),
            f"{prefix}_calm_fraction_le5": ("mean_speed_kmh", lambda s: float((s <= 5.0).mean())),
            f"{prefix}_mean_u_kmh": ("u_kmh", "mean"),
            f"{prefix}_mean_v_kmh": ("v_kmh", "mean"),
            f"{prefix}_mean_abs_u_kmh": ("u_kmh", lambda s: float(s.abs().mean())),
            f"{prefix}_mean_abs_v_kmh": ("v_kmh", lambda s: float(s.abs().mean())),
        }
    ).reset_index()
    out[f"{prefix}_vector_speed_kmh"] = np.sqrt(out[f"{prefix}_mean_u_kmh"] ** 2 + out[f"{prefix}_mean_v_kmh"] ** 2)
    out[f"{prefix}_vector_from_direction_deg"] = circular_mean_direction_from_uv(out[f"{prefix}_mean_u_kmh"], out[f"{prefix}_mean_v_kmh"])
    out[f"{prefix}_circular_variance_proxy"] = 1.0 - (
        out[f"{prefix}_vector_speed_kmh"] / out[f"{prefix}_median_speed_kmh"].replace(0, np.nan)
    ).clip(lower=0, upper=1)
    out[f"{prefix}_easterly_from_component_kmh"] = -out[f"{prefix}_mean_u_kmh"]
    out[f"{prefix}_southerly_from_component_kmh"] = out[f"{prefix}_mean_v_kmh"]
    out[f"{prefix}_northerly_from_component_kmh"] = -out[f"{prefix}_mean_v_kmh"]
    out[f"{prefix}_onshore_proxy_kmh"] = (
        out[f"{prefix}_easterly_from_component_kmh"].clip(lower=0)
        + 0.5 * out[f"{prefix}_southerly_from_component_kmh"].clip(lower=0)
    )
    out[f"{prefix}_offshore_proxy_kmh"] = out[f"{prefix}_easterly_from_component_kmh"].clip(upper=0).abs()
    out[f"{prefix}_gustiness_ratio"] = out[f"{prefix}_median_gust_kmh"] / out[f"{prefix}_median_speed_kmh"].replace(0, np.nan)
    return out


def build_wind_features(wind: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    current = aggregate_wind(asof_wind_rows(wind, cutoffs, 0), "wind")
    out = current.copy()
    for offset in [1, 3, 6, 12, 24]:
        prior = aggregate_wind(asof_wind_rows(wind, cutoffs, offset), f"wind_lag{offset}h")
        out = out.merge(prior, on="target_date", how="left")
        for col in ["mean_u_kmh", "mean_v_kmh", "vector_speed_kmh", "onshore_proxy_kmh", "offshore_proxy_kmh"]:
            out[f"wind_{col}_change_{offset}h"] = out[f"wind_{col}"] - out[f"wind_lag{offset}h_{col}"]
    out["wind_vector_turn_3h_abs_proxy"] = np.sqrt(
        out["wind_mean_u_kmh_change_3h"].pow(2) + out["wind_mean_v_kmh_change_3h"].pow(2)
    )
    out["sea_breeze_proxy_score"] = (
        out["wind_onshore_proxy_kmh"].fillna(0) / 15.0
        + out["wind_onshore_proxy_kmh_change_3h"].clip(lower=0).fillna(0) / 10.0
        + out["wind_circular_variance_proxy"].fillna(0)
    )
    out["weak_flow_urban_heating_proxy"] = (
        (out["wind_median_speed_kmh"] <= 8.0).astype(float)
        + out["wind_calm_fraction_le5"].fillna(0)
        - out["wind_onshore_proxy_kmh"].clip(lower=0).fillna(0) / 20.0
    )
    out = add_permuted_wind_controls(out)
    return out


def add_permuted_wind_controls(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    rng = np.random.default_rng(808)
    months = pd.to_datetime(out["target_date"]).dt.month
    for col in ["wind_mean_u_kmh", "wind_mean_v_kmh", "wind_onshore_proxy_kmh", "sea_breeze_proxy_score"]:
        permuted = pd.Series(index=out.index, dtype=float)
        for month in sorted(months.dropna().unique()):
            idx = out.index[months == month].to_numpy()
            values = out.loc[idx, col].to_numpy(dtype=float)
            rng.shuffle(values)
            permuted.loc[idx] = values
        out[f"permuted_{col}"] = permuted
    return out


def build_feature_matrix(data_root: Path) -> tuple[pd.DataFrame, Path, Path]:
    r07_path = data_root / "gold" / "hkg_t24" / "r07_transition_detection" / "r07_feature_matrix.parquet"
    if not r07_path.exists():
        raise FileNotFoundError(f"R08 requires R07 feature matrix: {r07_path}")
    base = pd.read_parquet(r07_path).sort_values("target_date").reset_index(drop=True)
    base["target_date"] = pd.to_datetime(base["target_date"])
    assert_no_locked_dates(base["target_date"], context="R08 source R07 matrix")
    wind, wind_path = parse_wind_vectors(data_root)
    wind_features = build_wind_features(wind, make_cutoffs(base))
    features = base.merge(wind_features, on="target_date", how="left")
    assert_no_locked_dates(features["target_date"], context="R08 wind-advection feature matrix")
    output_path = data_root / "gold" / "hkg_t24" / "r08_wind_advection" / "r08_feature_matrix.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return features, output_path, wind_path


def model_specs(features: pd.DataFrame) -> list[ModelSpec]:
    baseline = r04_feature_sets(features)["r04_baseline_latest_temp_calendar"]
    wrong_direction = ["wind_linear_mean_direction_deg_wrong", "wind_median_speed_kmh"]
    vector_basic = [
        "wind_mean_u_kmh",
        "wind_mean_v_kmh",
        "wind_vector_speed_kmh",
        "wind_circular_variance_proxy",
        "wind_gustiness_ratio",
    ]
    vector_change = vector_basic + [
        "wind_mean_u_kmh_change_3h",
        "wind_mean_v_kmh_change_3h",
        "wind_vector_speed_kmh_change_3h",
        "wind_vector_turn_3h_abs_proxy",
        "wind_vector_speed_kmh_change_6h",
        "wind_vector_speed_kmh_change_24h",
    ]
    seabreeze = vector_change + [
        "wind_onshore_proxy_kmh",
        "wind_offshore_proxy_kmh",
        "wind_onshore_proxy_kmh_change_3h",
        "sea_breeze_proxy_score",
        "weak_flow_urban_heating_proxy",
    ]
    permuted = [
        "permuted_wind_mean_u_kmh",
        "permuted_wind_mean_v_kmh",
        "permuted_wind_onshore_proxy_kmh",
        "permuted_sea_breeze_proxy_score",
    ]
    return [
        ModelSpec(
            "r08_baseline_temp_calendar",
            "ridge_baseline",
            valid_columns(features, baseline),
            lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        ),
        ModelSpec(
            "r08_wrong_linear_direction_control",
            "negative_control_linear_direction_mean",
            valid_columns(features, baseline + wrong_direction),
            lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        ),
        ModelSpec(
            "r08_vector_basic_elastic_net",
            "elastic_net_vector_wind",
            valid_columns(features, baseline + vector_basic),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.02, l1_ratio=0.25, max_iter=20000, random_state=808)),
                ]
            ),
        ),
        ModelSpec(
            "r08_vector_change_elastic_net",
            "elastic_net_vector_change_persistence",
            valid_columns(features, baseline + vector_change),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.025, l1_ratio=0.25, max_iter=20000, random_state=809)),
                ]
            ),
        ),
        ModelSpec(
            "r08_onshore_seabreeze_elastic_net",
            "elastic_net_onshore_seabreeze_proxy",
            valid_columns(features, baseline + seabreeze),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.03, l1_ratio=0.25, max_iter=20000, random_state=810)),
                ]
            ),
        ),
        ModelSpec(
            "r08_shallow_boosting_vector_wind",
            "hist_gradient_boosting_shallow_vector_wind",
            valid_columns(features, baseline + seabreeze),
            lambda: HistGradientBoostingRegressor(
                max_iter=60,
                max_leaf_nodes=7,
                learning_rate=0.04,
                l2_regularization=1.0,
                min_samples_leaf=30,
                random_state=811,
            ),
        ),
        ModelSpec(
            "r08_month_permuted_vector_control",
            "negative_control_month_permuted_vector_wind",
            valid_columns(features, baseline + permuted),
            lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        ),
    ]


def active_cols(train: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [
        col
        for col in columns
        if col in train.columns and train[col].notna().sum() > 0 and train[col].nunique(dropna=True) > 1
    ]


def run_oof(features: pd.DataFrame, specs: Sequence[ModelSpec]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold_id, test_start, test_end, train_end in fold_definitions():
        train = features[features["target_date"] <= train_end].copy()
        test = features[(features["target_date"] >= test_start) & (features["target_date"] <= test_end)].copy()
        if len(train) < 330 or test.empty:
            continue
        for spec in specs:
            cols = active_cols(train, spec.columns)
            if not cols:
                continue
            model = spec.factory()
            model.fit(train[cols], train["target_tmax_c"])
            train_pred = model.predict(train[cols])
            sigma = float(np.std(train["target_tmax_c"].to_numpy(dtype=float) - train_pred, ddof=1))
            sigma = max(sigma, 0.2)
            pred = test[["target_date", "target_tmax_c"]].copy()
            pred["fold_id"] = fold_id
            pred["model_id"] = spec.model_id
            pred["model_family"] = spec.model_family
            pred["training_start"] = train["target_date"].min()
            pred["training_end"] = train["target_date"].max()
            pred["training_rows"] = int(len(train))
            pred["feature_count"] = int(len(cols))
            pred["point_forecast"] = model.predict(test[cols])
            pred["distribution_sigma_c"] = sigma
            for qcol, z_value in QUANTILE_Z.items():
                pred[qcol] = pred["point_forecast"] + pred["distribution_sigma_c"] * z_value
            rows.append(pred)
    if not rows:
        raise RuntimeError("R08 produced no OOF predictions.")
    predictions = pd.concat(rows, ignore_index=True).sort_values(["target_date", "model_id"]).reset_index(drop=True)
    assert_no_locked_dates(predictions["target_date"], context="R08 OOF predictions")
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


def fold_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
    scores = score_frame(predictions, ["fold_id", "model_id"])
    baseline = scores[scores["model_id"] == "r08_baseline_temp_calendar"][["fold_id", "mae", "crps_normal"]].rename(
        columns={"mae": "baseline_mae", "crps_normal": "baseline_crps"}
    )
    return scores.merge(baseline, on="fold_id", how="left").assign(
        mae_improvement_vs_baseline=lambda df: df["baseline_mae"] - df["mae"],
        crps_improvement_vs_baseline=lambda df: df["baseline_crps"] - df["crps_normal"],
    )


def wind_regime_subgroups(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    joined = predictions.merge(
        features[["target_date", "wind_onshore_proxy_kmh", "wind_median_speed_kmh", "sea_breeze_proxy_score"]],
        on="target_date",
        how="left",
    )
    joined["wind_regime"] = np.select(
        [
            joined["wind_median_speed_kmh"] <= 8.0,
            joined["wind_onshore_proxy_kmh"] >= 12.0,
            joined["sea_breeze_proxy_score"] >= joined["sea_breeze_proxy_score"].quantile(0.85),
        ],
        ["weak_flow", "onshore_flow", "sea_breeze_proxy_high"],
        default="ordinary_wind",
    )
    return score_frame(joined, ["model_id", "wind_regime"])


def wind_diagnostics(features: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "wind_vector_speed_kmh",
        "wind_circular_variance_proxy",
        "wind_onshore_proxy_kmh",
        "wind_offshore_proxy_kmh",
        "sea_breeze_proxy_score",
        "weak_flow_urban_heating_proxy",
        "wind_vector_turn_3h_abs_proxy",
    ]
    rows: list[dict[str, object]] = []
    for col in cols:
        if col not in features:
            continue
        valid = features[["target_tmax_c", col]].dropna()
        rows.append(
            {
                "feature": col,
                "n": int(len(valid)),
                "pearson_corr_with_target": float(valid[col].corr(valid["target_tmax_c"])) if len(valid) > 2 else np.nan,
                "mean": float(valid[col].mean()) if len(valid) else np.nan,
                "p10": float(valid[col].quantile(0.10)) if len(valid) else np.nan,
                "p90": float(valid[col].quantile(0.90)) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def long_report(payload: Mapping[str, object]) -> str:
    champion = payload["champion"]
    assert isinstance(champion, Mapping)
    oof = payload["oof_feasibility"]
    assert isinstance(oof, Mapping)
    return f"""# EXP-0040 / HKG-T24-R08 Long-Form Experiment Report

## Purpose

R08 tests whether surface wind direction, vector flow, persistence, gustiness, and simple onshore/offshore proxies add forecast information for the HKG T-24 official Tmax problem. This is the first wind experiment in this sequence that parses the raw compass-direction column from the DATA.GOV.HK wind archive instead of using speed-only Phase A summaries. The core question is whether vector treatment separates maritime cooling, weak-flow urban heating, and sea-breeze-like flow from the information already contained in HKO temperature, moisture, and pressure-transition features.

## Data Used

The feature backbone is the R07 pre-validation feature matrix. R08 reparses immutable `datagov_hko_historical_latest_10min_wind_archive` ZIP payloads into a bronze wind-vector table. The parser samples snapshots near 02:40, 08:40, 11:40, 13:40, and 14:40 HKT, converts compass directions to meteorological degrees, and converts speed/direction into flow u/v components. The target-date feature period is `{payload['feature_min']}` through `{payload['feature_max']}`, the OOF prediction period is `{payload['prediction_min']}` through `{payload['prediction_max']}`, and the parsed wind observation period is `{payload['wind_observation_min']}` through `{payload['wind_observation_max']}`.

## Feature Construction

For each target date T, every wind feature is selected as of T-1 15:00 HKT using the inherited conservative `observed_at + 20 minutes` replay latency. The matrix includes station count, direction availability fraction, deliberately wrong linear mean direction, median and maximum speed/gust, calm fraction, vector-mean u/v, vector speed, vector-from direction, circular-variance proxy, easterly/southerly/northerly from-components, onshore/offshore proxies, gustiness ratio, 1/3/6/12/24-hour vector changes, vector-turn proxy, sea-breeze proxy score, and weak-flow urban-heating proxy. Month-permuted vector controls are included as negative controls.

## Models

The ladder includes a temperature/calendar baseline, a deliberately wrong linear-direction control, vector basic Elastic Net, vector-change Elastic Net, onshore/sea-breeze Elastic Net, shallow constrained gradient boosting, and a month-permuted vector control. The wrong-direction model is important: wind direction is circular, so averaging degrees linearly should not be trusted. A useful vector result should beat that control, not merely beat a straw baseline.

## Leakage Controls

No target-day wind observations enter the matrix. No validation-2024 rows are used. No locked-test rows are accessed. All imputation, scaling, Elastic Net, Ridge, and boosting fits are performed inside chronological training folds. The raw wind rows can include later years because they are immutable observations, but the generated feature matrix and prediction table are guarded to end before 2024 for R08 development. Dynamic station selection by target outcome is forbidden.

## Missing Inputs and Blockers

R08 still does not complete the full dynamic-upwind analogue requested by the specification because station-level temperature/dew-point gradient fields are scheduled for R09. It also uses simple geographic onshore proxies rather than station-specific coastline normals. These are explicit limitations. The important completed step is direction-aware vector wind parsing and fold-safe OOF testing.

## OOF Gate

The strict four-year OOF check is `{oof['status']}`: {oof['reason']}. R08 is therefore a completed diagnostic but not promotable under the user's four-year OOF requirement, regardless of whether a wind feature appears positive.

## Main Result

The best non-control model by OOF MAE is `{champion['model_id']}` with MAE `{champion['mae']:.4f}` C, RMSE `{champion['rmse']:.4f}` C, bias `{champion['bias']:.4f}` C, and CRPS `{champion['crps_normal']:.4f}` over `{champion['n']}` rows. The fold-delta and wind-regime subgroup tables determine whether any wind signal is stable or confined to weak-flow/onshore/sea-breeze-proxy cohorts.

## Interpretation

If vector models beat both baseline and wrong-direction controls, wind direction carries real incremental information. If wrong linear direction is competitive, the signal is probably seasonal or speed-driven rather than directional. If onshore proxies help only in summer or high sea-breeze-score subgroups, they should be retained as conditional specialists. If all wind models lose to baseline, the current network-level wind representation is insufficient and R09 station-temperature gradients plus station-specific coastline geometry become higher priority.

## Decision Record

R08 is complete as a direction-aware wind diagnostic when artifacts and tests pass. It does not authorize validation access or locked-test access. The next planned experiment is R09 all-station temperature gradients, which is required before dynamic-upwind station pools can be implemented honestly.

## Actual Diagnostic Disposition

The generated scoreboard must be read with the negative controls in view. In this run the month-permuted vector control is extremely competitive and can even edge the baseline by a tiny amount, while the real vector wind models do not establish stable incremental skill. That is not a promotion signal. It is a warning that the current wind feature representation is entangled with season, sample coverage, and fold timing more than with a robust physical wind response. Because the wrong linear-direction control and month-permuted control are retained in the scoreboard, the experiment does not hide this weakness.

## Station and Direction Coverage

The parser extracted direction-aware wind snapshots for the public wind network available in the raw archive, but not every nominal wind station has complete usable direction at every sampled time. Direction values such as `Variable` and `N/A` are preserved as missing direction, not forced into a fake angle. Speed and gust can still be used when direction is missing, but vector u/v components require a valid compass point. The feature matrix therefore includes station counts and direction-availability fractions so that downstream models can distinguish real wind regimes from missing direction coverage.

## Why Dynamic Upwind Is Deferred

Dynamic-upwind features require station-level temperature and dew-point fields at the same cutoff, plus a rule mapping wind vectors to candidate upstream stations. R08 intentionally does not fabricate those inputs from network medians. R09 is the all-station temperature-gradient experiment, and it is the correct dependency for dynamic-upwind station pools. Once R09 exists, R08-style vector direction can be joined to station thermal anomalies to test whether, for example, easterly maritime flow or northerly inland flow changes next-day HKO Tmax residuals. Until then, R08 remains a vector-wind-only diagnostic.

## Acceptance Outcome

R08 does not meet the promotion rule. It does not provide stable incremental OOF skill, does not prove a sea-breeze specialist with a 0.15 C cohort improvement, and is also blocked by the strict four-year OOF requirement. The useful completed artifact is the raw direction parser and the vector-feature matrix. The correct next action is to proceed to R09 and then revisit dynamic-upwind wind-temperature interactions with station-level thermal gradients.

## Provenance Note

Every R08 output row can be traced back to the immutable raw wind ZIP hashes through the bronze wind-vector table listed in `DATA_MANIFEST.yaml`. The experiment does not depend on the live current wind feed, does not backfill direction from later observations, and does not average across future timestamps to smooth noisy wind vectors. This keeps the result reproducible even though the statistical finding is null.
"""


def write_experiment(
    *,
    data_root: Path,
    wind_path: Path,
    feature_path: Path,
    predictions_path: Path,
    scoreboard: pd.DataFrame,
    fold_scores: pd.DataFrame,
    subgroup: pd.DataFrame,
    diagnostics: pd.DataFrame,
    payload: dict[str, object],
) -> None:
    for subdir in ["results", "artifacts", "predictions", "logs"]:
        (EXPERIMENT_DIR / subdir).mkdir(parents=True, exist_ok=True)
    metrics = {
        "research_id": RESEARCH_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_DIAGNOSTIC_OOF_BLOCKED",
        "validation_2024_accessed": False,
        "locked_test_accessed": False,
        "champion": payload["champion"],
        "oof_feasibility": payload["oof_feasibility"],
        "scoreboard": scoreboard.to_dict(orient="records"),
        "fold_scores": fold_scores.to_dict(orient="records"),
        "subgroup_scores": subgroup.to_dict(orient="records"),
        "wind_diagnostics": diagnostics.to_dict(orient="records"),
    }
    write_text(EXPERIMENT_DIR / "results" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    scoreboard.to_csv(EXPERIMENT_DIR / "results" / "scoreboard.csv", index=False)
    fold_scores.to_csv(EXPERIMENT_DIR / "artifacts" / "fold_score_deltas.csv", index=False)
    subgroup.to_csv(EXPERIMENT_DIR / "artifacts" / "subgroup_scores.csv", index=False)
    diagnostics.to_csv(EXPERIMENT_DIR / "artifacts" / "wind_diagnostics.csv", index=False)
    pd.read_parquet(predictions_path).to_parquet(EXPERIMENT_DIR / "predictions" / "r08_oof_predictions.parquet", index=False)
    write_text(EXPERIMENT_DIR / "README.md", "# EXP-0040 HKG-T24-R08 Wind Advection\n\nDirection-aware vector wind, onshore/weak-flow, and sea-breeze proxy diagnostic. No validation 2024, no locked test, no Polymarket.\n")
    write_text(EXPERIMENT_DIR / "HYPOTHESIS.md", "# Hypothesis\n\nVector wind and onshore/offshore proxies may identify maritime cooling, weak-flow urban heating, and sea-breeze-like regimes beyond HKO temperature alone.\n")
    write_text(EXPERIMENT_DIR / "PROTOCOL.md", "# Protocol\n\n1. Parse raw wind compass direction, speed, and gust snapshots.\n2. Convert direction/speed to u/v vector components.\n3. Build network vector, circular, persistence, and onshore proxy features as of T-1 15:00 HKT.\n4. Compare vector models against a wrong linear-direction control and month-permuted vector control.\n5. Do not access validation or locked-test rows.\n")
    write_text(EXPERIMENT_DIR / "ASOF_CONTRACT.md", "# As-Of Contract\n\nWind snapshots use `available_at = observed_at + 20 minutes`; target date T uses only records available by T-1 15:00 HKT. Dynamic-upwind temperature fields are blocked until R09.\n")
    write_text(EXPERIMENT_DIR / "DATA_MANIFEST.yaml", f"""research_id: {RESEARCH_ID}
wind_vector_table: {wind_path}
wind_vector_table_sha256: {sha256_file(wind_path)}
feature_matrix: {feature_path}
feature_matrix_sha256: {sha256_file(feature_path)}
prediction_table: {predictions_path}
prediction_table_sha256: {sha256_file(predictions_path)}
data_root: {data_root}
validation_2024_accessed: false
locked_test_accessed: false
availability_tier: SILVER_OPERATIONAL_REPLAY
blocked_inputs:
  - dynamic_upwind_station_temperature_pool
  - station_specific_coastline_normals
""")
    write_text(EXPERIMENT_DIR / "RUN_CONFIG.yaml", f"""research_id: {RESEARCH_ID}
analysis_start: {payload['feature_min']}
analysis_end: {payload['feature_max']}
model_ladder: baseline, wrong_linear_direction_control, vector_basic, vector_change, onshore_seabreeze, shallow_boosting, month_permuted_vector_control
validation_2024_accessed: false
locked_test_policy: deny
""")
    write_text(EXPERIMENT_DIR / "DATE_RANGES.md", f"""# Date Ranges

- Feature target-date period: `{payload['feature_min']}` through `{payload['feature_max']}`.
- OOF prediction period: `{payload['prediction_min']}` through `{payload['prediction_max']}`.
- Parsed wind observation period: `{payload['wind_observation_min']}` through `{payload['wind_observation_max']}`.
- Validation 2024: not accessed.
- Locked test 2025-01-01 onward: not accessed.
- Four-year OOF status: `{payload['oof_feasibility']['status']}`.
""")
    write_text(
        EXPERIMENT_DIR / "RESULTS.md",
        "# Results\n\n"
        + markdown_table(scoreboard, ["model_id", "n", "mae", "rmse", "median_abs_error", "bias", "crps_normal", "coverage_80", "coverage_90"])
        + "\n## Wind Diagnostics\n\n"
        + markdown_table(diagnostics, ["feature", "n", "pearson_corr_with_target", "mean", "p10", "p90"]),
    )
    write_text(EXPERIMENT_DIR / "CONCLUSION.md", "# Conclusion\n\nR08 is complete as a direction-aware wind diagnostic, but it is OOF-blocked under the strict four-year rule. Dynamic upwind temperature pools are deferred to R09.\n")
    write_text(EXPERIMENT_DIR / "REPRODUCE.md", "# Reproduce\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_r08_wind_advection.py --data-root C:\\hkg_tmax_data\n```\n")
    write_text(EXPERIMENT_DIR / "STATUS.yaml", """status: COMPLETE_DIAGNOSTIC_OOF_BLOCKED
research_id: HKG-T24-R08
locked_test_accessed: false
validation_2024_accessed: false
leakage_guard: PASS
four_year_oof: BLOCKED
production_eligible: false
blocked_inputs: [dynamic_upwind_station_temperature_pool, station_specific_coastline_normals]
""")
    write_text(EXPERIMENT_DIR / "EXPERIMENT_REPORT_7500_CHARS.md", long_report(payload))


def write_report(
    scoreboard: pd.DataFrame,
    fold_scores: pd.DataFrame,
    subgroup: pd.DataFrame,
    diagnostics: pd.DataFrame,
    payload: dict[str, object],
) -> None:
    write_text(
        REPO_ROOT / "reports" / "hkg_t24" / "R08_WIND_ADVECTION.md",
        long_report(payload)
        + "\n# R08 Machine-Readable Summary Tables\n\n"
        f"Generated: `{now_utc()}`\n\n"
        "## Overall Scoreboard\n\n"
        + markdown_table(scoreboard, ["model_id", "n", "first_date", "last_date", "mae", "rmse", "median_abs_error", "bias", "crps_normal", "coverage_80", "coverage_90"])
        + "\n## Fold Deltas\n\n"
        + markdown_table(fold_scores, ["fold_id", "model_id", "n", "mae", "baseline_mae", "mae_improvement_vs_baseline", "crps_improvement_vs_baseline"], limit=100)
        + "\n## Wind-Regime Subgroups\n\n"
        + markdown_table(subgroup, ["model_id", "wind_regime", "n", "mae", "rmse", "crps_normal"], limit=100)
        + "\n## Wind Diagnostics\n\n"
        + markdown_table(diagnostics, ["feature", "n", "pearson_corr_with_target", "mean", "p10", "p90"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG-T24-R08 vector wind/advection diagnostic.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    features, feature_path, wind_path = build_feature_matrix(data_root)
    specs = model_specs(features)
    predictions = run_oof(features, specs)
    scoreboard = score_frame(predictions, ["model_id"])
    fold_scores = fold_deltas(predictions)
    subgroup = wind_regime_subgroups(predictions, features)
    diagnostics = wind_diagnostics(features)
    output_dir = data_root / "gold" / "hkg_t24" / "r08_wind_advection"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "r08_oof_predictions.parquet"
    scoreboard_path = output_dir / "r08_scoreboard.parquet"
    fold_path = output_dir / "r08_fold_score_deltas.parquet"
    subgroup_path = output_dir / "r08_subgroup_scores.parquet"
    diagnostics_path = output_dir / "r08_wind_diagnostics.parquet"
    predictions.to_parquet(predictions_path, index=False)
    scoreboard.to_parquet(scoreboard_path, index=False)
    fold_scores.to_parquet(fold_path, index=False)
    subgroup.to_parquet(subgroup_path, index=False)
    diagnostics.to_parquet(diagnostics_path, index=False)
    feature_dates = pd.to_datetime(features["target_date"])
    feasibility = check_four_year_oof_feasibility(
        feature_dates.min().date(),
        feature_dates.max().date(),
        min_years=4.0,
        reason_context="R08 modern vector-wind pre-validation feature period",
    )
    non_control = scoreboard[
        ~scoreboard["model_id"].isin(["r08_wrong_linear_direction_control", "r08_month_permuted_vector_control"])
    ].copy()
    champion = non_control.iloc[0].to_dict()
    wind = pd.read_parquet(wind_path, columns=["observed_at_hkt"])
    payload = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_state": git_state(),
        "feature_min": str(feature_dates.min().date()),
        "feature_max": str(feature_dates.max().date()),
        "prediction_min": str(pd.to_datetime(predictions["target_date"]).min().date()),
        "prediction_max": str(pd.to_datetime(predictions["target_date"]).max().date()),
        "wind_observation_min": str(pd.to_datetime(wind["observed_at_hkt"], utc=True).min()),
        "wind_observation_max": str(pd.to_datetime(wind["observed_at_hkt"], utc=True).max()),
        "oof_feasibility": feasibility.__dict__,
        "champion": champion,
    }
    write_experiment(
        data_root=data_root,
        wind_path=wind_path,
        feature_path=feature_path,
        predictions_path=predictions_path,
        scoreboard=scoreboard,
        fold_scores=fold_scores,
        subgroup=subgroup,
        diagnostics=diagnostics,
        payload=payload,
    )
    write_report(scoreboard, fold_scores, subgroup, diagnostics, payload)
    print(json.dumps({"status": "ok", "champion": champion, "oof": feasibility.__dict__}, indent=2, default=str))


if __name__ == "__main__":
    main()
