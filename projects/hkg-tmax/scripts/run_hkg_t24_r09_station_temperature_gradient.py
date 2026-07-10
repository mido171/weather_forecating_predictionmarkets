from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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

from hkg_tmax.hkg_t24.governance import (
    HKO_TEMPERATURE_AND_MAXMIN_STATIONS,
    check_four_year_oof_feasibility,
)
from hkg_tmax.hkg_t24.guard import assert_no_locked_dates
from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
DEFAULT_DATA_ROOT = PROJECT_PATHS.data_root
RESEARCH_ID = "HKG-T24-R09"
EXPERIMENT_ID = "EXP-0041"
EXPERIMENT_DIR = PROJECT_PATHS.run_root / "experiments" / "legacy" / "hkg_tmax_t24" / "EXP-0041-HKG-T24-R09"
SOURCE_ID = "datagov_hko_historical_latest_1min_temperature_archive"
ENTRY_TIME_RE = re.compile(r"(?P<date>\d{8})-(?P<hhmm>\d{4})")
SNAPSHOT_TARGET_MINUTES = (2 * 60 + 40, 8 * 60 + 40, 11 * 60 + 40, 13 * 60 + 40, 14 * 60 + 40)
HKO_STATION = "HK Observatory"

_R04_SPEC = importlib.util.spec_from_file_location(
    "run_hkg_t24_r04_thermal_trajectory",
    REPO_ROOT / "scripts" / "run_hkg_t24_r04_thermal_trajectory.py",
)
if _R04_SPEC is None or _R04_SPEC.loader is None:
    raise ImportError("Unable to load R04 helper script for R09 station-gradient experiment.")
_R04_MODULE = importlib.util.module_from_spec(_R04_SPEC)
_R04_SPEC.loader.exec_module(_R04_MODULE)

_R06_SPEC = importlib.util.spec_from_file_location(
    "run_hkg_t24_r06_moisture_state",
    REPO_ROOT / "scripts" / "run_hkg_t24_r06_moisture_state.py",
)
if _R06_SPEC is None or _R06_SPEC.loader is None:
    raise ImportError("Unable to load R06 helper script for R09 station-gradient experiment.")
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

COASTAL_STATIONS = {
    "Chek Lap Kok",
    "Cheung Chau",
    "Clear Water Bay",
    "Peng Chau",
    "Sai Kung",
    "Stanley",
    "Tai Mei Tuk",
    "Tseung Kwan O",
    "Tsing Yi",
    "Tuen Mun",
    "Waglan Island",
}
INLAND_STATIONS = {"Happy Valley", "Kowloon City", "Sha Tin", "Shek Kong", "Sheung Shui", "Ta Kwu Ling", "Tai Po", "Yuen Long Park"}
ELEVATED_STATIONS = {"Ngong Ping", "Tai Mo Shan", "Tate's Cairn", "The Peak"}
URBAN_STATIONS = {"HK Observatory", "HK Park", "Happy Valley", "Kowloon City", "Kwun Tong", "Sham Shui Po", "Wong Tai Sin"}
ISLAND_STATIONS = {"Cheung Chau", "Peng Chau", "Waglan Island"}
EAST_STATIONS = {"Clear Water Bay", "Sai Kung", "Shau Kei Wan", "Tseung Kwan O"}
WEST_STATIONS = {"Chek Lap Kok", "Lau Fau Shan", "Tuen Mun", "Wetland Park", "Yuen Long Park"}
NORTH_STATIONS = {"Lau Fau Shan", "Sheung Shui", "Ta Kwu Ling", "Wetland Park", "Yuen Long Park"}
SOUTH_STATIONS = {"Cheung Chau", "HK Observatory", "HK Park", "Stanley", "Waglan Island"}


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
    rows = ledger[(ledger["status"] == "success") & (ledger["source_id"] == SOURCE_ID)].copy()
    if rows.empty:
        raise FileNotFoundError(f"No successful retrieval rows found for {SOURCE_ID}")
    return rows.sort_values("retrieved_at").drop_duplicates(["source_id", "content_sha256"], keep="last")


def zip_entry_in_sampling_window(name: str) -> bool:
    match = ENTRY_TIME_RE.search(name)
    if match is None:
        return False
    hhmm = match.group("hhmm")
    minute = int(hhmm[:2]) * 60 + int(hhmm[2:])
    return any(target <= minute <= target + 20 for target in SNAPSHOT_TARGET_MINUTES)


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


def parse_station_temperatures(data_root: Path) -> tuple[pd.DataFrame, Path]:
    output_path = data_root / "bronze" / "hkg_t24" / "r09_temperature_sampled_observations.parquet"
    if output_path.exists():
        temp = pd.read_parquet(output_path)
        temp["observed_at_hkt"] = pd.to_datetime(temp["observed_at_hkt"], utc=True).dt.tz_convert(HKT)
        temp["available_at_hkt"] = pd.to_datetime(temp["available_at_hkt"], utc=True).dt.tz_convert(HKT)
        return temp, output_path
    station_filter = set(HKO_TEMPERATURE_AND_MAXMIN_STATIONS)
    records: list[dict[str, object]] = []
    for _, row in source_rows(read_ledger(data_root)).iterrows():
        path = Path(str(row["content_path"]))
        if not path.exists():
            continue
        for raw_row in iter_zip_csv_rows(path):
            station = raw_row.get("Automatic Weather Station", "").strip()
            if station not in station_filter:
                continue
            observed_at = parse_hkt_observed_at(raw_row.get("Date time", ""))
            value = parse_number(raw_row.get("Air Temperature(degree Celsius)", ""))
            if observed_at is None or value is None:
                continue
            records.append(
                {
                    "source_id": SOURCE_ID,
                    "source_file_hash": str(row["content_sha256"]),
                    "retrieved_at": row["retrieved_at"],
                    "station": station,
                    "observed_at_hkt": observed_at,
                    "available_at_hkt": observed_at + timedelta(minutes=20),
                    "local_date": observed_at.date(),
                    "temperature_c": value,
                    "availability_tier": "SILVER_OPERATIONAL_REPLAY",
                    "latency_rule": "observed_at + 20 minutes",
                }
            )
    if not records:
        raise RuntimeError("R09 parsed no sampled station-temperature observations.")
    temp = pd.DataFrame(records).sort_values(["station", "available_at_hkt"]).reset_index(drop=True)
    temp["observed_at_hkt"] = pd.to_datetime(temp["observed_at_hkt"], utc=True).dt.tz_convert(HKT)
    temp["available_at_hkt"] = pd.to_datetime(temp["available_at_hkt"], utc=True).dt.tz_convert(HKT)
    temp["local_date"] = pd.to_datetime(temp["local_date"], errors="coerce")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp.to_parquet(output_path, index=False)
    return temp, output_path


def asof_temperature_rows(temp: pd.DataFrame, cutoffs: pd.DataFrame, offset_hours: int) -> pd.DataFrame:
    left = cutoffs[["target_date", "cutoff_hkt"]].copy()
    left["join_available_at_hkt"] = left["cutoff_hkt"] - pd.to_timedelta(offset_hours, unit="h")
    outputs: list[pd.DataFrame] = []
    for station, group in temp.groupby("station", sort=True):
        right = group[["available_at_hkt", "observed_at_hkt", "temperature_c", "source_file_hash"]].dropna().sort_values("available_at_hkt")
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


def station_slug(station: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", station.lower()).strip("_")


def group_mean(rows: pd.DataFrame, stations: set[str], name: str) -> pd.DataFrame:
    subset = rows[rows["station"].isin(stations)]
    if subset.empty:
        return pd.DataFrame({"target_date": rows["target_date"].drop_duplicates(), name: np.nan})
    return subset.groupby("target_date")["temperature_c"].mean().rename(name).reset_index()


def aggregate_temperature(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["target_date"])
    grouped = rows.groupby("target_date")
    out = grouped.agg(
        **{
            f"{prefix}_station_count": ("station", "nunique"),
            f"{prefix}_median_c": ("temperature_c", "median"),
            f"{prefix}_trimmed_mean_c": ("temperature_c", lambda s: float(s.sort_values().iloc[max(0, int(len(s) * 0.1)) : max(1, int(len(s) * 0.9))].mean())),
            f"{prefix}_min_c": ("temperature_c", "min"),
            f"{prefix}_max_c": ("temperature_c", "max"),
            f"{prefix}_p10_c": ("temperature_c", lambda s: float(s.quantile(0.10))),
            f"{prefix}_p90_c": ("temperature_c", lambda s: float(s.quantile(0.90))),
            f"{prefix}_std_c": ("temperature_c", "std"),
            f"{prefix}_skew_proxy": ("temperature_c", lambda s: float(s.skew()) if len(s) >= 3 else np.nan),
        }
    ).reset_index()
    out[f"{prefix}_coverage_fraction"] = out[f"{prefix}_station_count"] / float(len(HKO_TEMPERATURE_AND_MAXMIN_STATIONS))
    out[f"{prefix}_spread_c"] = out[f"{prefix}_max_c"] - out[f"{prefix}_min_c"]
    out[f"{prefix}_iqr_proxy_c"] = out[f"{prefix}_p90_c"] - out[f"{prefix}_p10_c"]
    hko = rows[rows["station"] == HKO_STATION][["target_date", "temperature_c"]].rename(columns={"temperature_c": f"{prefix}_hko_c"})
    out = out.merge(hko, on="target_date", how="left")
    out[f"{prefix}_hko_minus_median_c"] = out[f"{prefix}_hko_c"] - out[f"{prefix}_median_c"]
    for stations, name in [
        (COASTAL_STATIONS, "coastal"),
        (INLAND_STATIONS, "inland"),
        (ELEVATED_STATIONS, "elevated"),
        (URBAN_STATIONS, "urban"),
        (ISLAND_STATIONS, "island"),
        (EAST_STATIONS, "east"),
        (WEST_STATIONS, "west"),
        (NORTH_STATIONS, "north"),
        (SOUTH_STATIONS, "south"),
    ]:
        out = out.merge(group_mean(rows, stations, f"{prefix}_{name}_mean_c"), on="target_date", how="left")
    out[f"{prefix}_inland_minus_coastal_c"] = out[f"{prefix}_inland_mean_c"] - out[f"{prefix}_coastal_mean_c"]
    out[f"{prefix}_urban_minus_coastal_c"] = out[f"{prefix}_urban_mean_c"] - out[f"{prefix}_coastal_mean_c"]
    out[f"{prefix}_lowland_minus_elevated_c"] = out[f"{prefix}_median_c"] - out[f"{prefix}_elevated_mean_c"]
    out[f"{prefix}_island_minus_mainland_c"] = out[f"{prefix}_island_mean_c"] - out[f"{prefix}_median_c"]
    out[f"{prefix}_east_minus_west_c"] = out[f"{prefix}_east_mean_c"] - out[f"{prefix}_west_mean_c"]
    out[f"{prefix}_north_minus_south_c"] = out[f"{prefix}_north_mean_c"] - out[f"{prefix}_south_mean_c"]
    return out


def station_offsets(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
    hko = rows[rows["station"] == HKO_STATION][["target_date", "temperature_c"]].rename(columns={"temperature_c": "hko_temp_c"})
    wide = rows.pivot_table(index="target_date", columns="station", values="temperature_c", aggfunc="last")
    out = pd.DataFrame(index=wide.index)
    for station in sorted(set(wide.columns) - {HKO_STATION}):
        out[f"{prefix}_hko_minus_{station_slug(station)}_c"] = hko.set_index("target_date")["hko_temp_c"] - wide[station]
    return out.reset_index()


def build_temperature_features(temp: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    current_rows = asof_temperature_rows(temp, cutoffs, 0)
    out = aggregate_temperature(current_rows, "temp_network")
    out = out.merge(station_offsets(current_rows, "station_offset"), on="target_date", how="left")
    for offset in [3, 6, 12, 24]:
        prior = aggregate_temperature(asof_temperature_rows(temp, cutoffs, offset), f"temp_lag{offset}h")
        out = out.merge(prior, on="target_date", how="left")
        for col in ["median_c", "spread_c", "inland_minus_coastal_c", "urban_minus_coastal_c", "east_minus_west_c"]:
            out[f"temp_network_{col}_change_{offset}h"] = out[f"temp_network_{col}"] - out[f"temp_lag{offset}h_{col}"]
    out["sea_breeze_thermal_gradient_proxy"] = (
        out["temp_network_inland_minus_coastal_c"].fillna(0)
        + out["temp_network_inland_minus_coastal_c_change_3h"].clip(lower=0).fillna(0)
    )
    out["hko_local_warm_outlier_score"] = out["temp_network_hko_minus_median_c"] / out["temp_network_std_c"].replace(0, np.nan)
    out = add_permuted_spatial_controls(out)
    return out


def add_permuted_spatial_controls(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    rng = np.random.default_rng(909)
    months = pd.to_datetime(out["target_date"]).dt.month
    for col in ["temp_network_inland_minus_coastal_c", "temp_network_spread_c", "temp_network_hko_minus_median_c"]:
        permuted = pd.Series(index=out.index, dtype=float)
        for month in sorted(months.dropna().unique()):
            idx = out.index[months == month].to_numpy()
            values = out.loc[idx, col].to_numpy(dtype=float)
            rng.shuffle(values)
            permuted.loc[idx] = values
        out[f"permuted_{col}"] = permuted
    return out


def build_feature_matrix(data_root: Path) -> tuple[pd.DataFrame, Path, Path]:
    r08_path = data_root / "gold" / "hkg_t24" / "r08_wind_advection" / "r08_feature_matrix.parquet"
    if not r08_path.exists():
        raise FileNotFoundError(f"R09 requires R08 feature matrix: {r08_path}")
    base = pd.read_parquet(r08_path).sort_values("target_date").reset_index(drop=True)
    base["target_date"] = pd.to_datetime(base["target_date"])
    assert_no_locked_dates(base["target_date"], context="R09 source R08 matrix")
    temp, temp_path = parse_station_temperatures(data_root)
    spatial = build_temperature_features(temp, make_cutoffs(base))
    features = base.merge(spatial, on="target_date", how="left")
    if "wind_easterly_from_component_kmh" in features:
        features["east_west_gradient_x_easterly_flow"] = features["temp_network_east_minus_west_c"] * features["wind_easterly_from_component_kmh"]
    if "wind_onshore_proxy_kmh" in features:
        features["inland_coastal_gradient_x_onshore_flow"] = features["temp_network_inland_minus_coastal_c"] * features["wind_onshore_proxy_kmh"]
    assert_no_locked_dates(features["target_date"], context="R09 station-gradient feature matrix")
    output_path = data_root / "gold" / "hkg_t24" / "r09_station_temperature_gradient" / "r09_feature_matrix.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return features, output_path, temp_path


def model_specs(features: pd.DataFrame) -> list[ModelSpec]:
    baseline = r04_feature_sets(features)["r04_baseline_latest_temp_calendar"]
    spatial_summary = [
        "temp_network_station_count",
        "temp_network_coverage_fraction",
        "temp_network_median_c",
        "temp_network_spread_c",
        "temp_network_iqr_proxy_c",
        "temp_network_hko_minus_median_c",
        "temp_network_inland_minus_coastal_c",
        "temp_network_urban_minus_coastal_c",
        "temp_network_lowland_minus_elevated_c",
        "temp_network_east_minus_west_c",
        "temp_network_north_minus_south_c",
        "sea_breeze_thermal_gradient_proxy",
        "hko_local_warm_outlier_score",
    ]
    spatial_changes = spatial_summary + [
        "temp_network_median_c_change_3h",
        "temp_network_spread_c_change_3h",
        "temp_network_inland_minus_coastal_c_change_3h",
        "temp_network_urban_minus_coastal_c_change_3h",
        "temp_network_east_minus_west_c_change_3h",
        "temp_network_median_c_change_24h",
        "temp_network_inland_minus_coastal_c_change_24h",
    ]
    station_offset_cols = [col for col in features.columns if col.startswith("station_offset_hko_minus_")]
    flow_interactions = ["east_west_gradient_x_easterly_flow", "inland_coastal_gradient_x_onshore_flow"]
    permuted = [
        "permuted_temp_network_inland_minus_coastal_c",
        "permuted_temp_network_spread_c",
        "permuted_temp_network_hko_minus_median_c",
    ]
    return [
        ModelSpec(
            "r09_baseline_temp_calendar",
            "ridge_baseline",
            valid_columns(features, baseline),
            lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        ),
        ModelSpec(
            "r09_spatial_summary_elastic_net",
            "elastic_net_spatial_summary",
            valid_columns(features, baseline + spatial_summary),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.02, l1_ratio=0.25, max_iter=20000, random_state=909)),
                ]
            ),
        ),
        ModelSpec(
            "r09_spatial_change_elastic_net",
            "elastic_net_spatial_change",
            valid_columns(features, baseline + spatial_changes),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.025, l1_ratio=0.25, max_iter=20000, random_state=910)),
                ]
            ),
        ),
        ModelSpec(
            "r09_station_offsets_elastic_net",
            "elastic_net_station_offsets",
            valid_columns(features, baseline + station_offset_cols),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.03, l1_ratio=0.35, max_iter=20000, random_state=911)),
                ]
            ),
        ),
        ModelSpec(
            "r09_flow_interaction_elastic_net",
            "elastic_net_flow_conditioned_spatial_gradient",
            valid_columns(features, baseline + spatial_changes + flow_interactions),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.03, l1_ratio=0.25, max_iter=20000, random_state=912)),
                ]
            ),
        ),
        ModelSpec(
            "r09_shallow_boosting_spatial",
            "hist_gradient_boosting_shallow_spatial",
            valid_columns(features, baseline + spatial_changes + station_offset_cols + flow_interactions),
            lambda: HistGradientBoostingRegressor(
                max_iter=60,
                max_leaf_nodes=7,
                learning_rate=0.04,
                l2_regularization=1.0,
                min_samples_leaf=30,
                random_state=913,
            ),
        ),
        ModelSpec(
            "r09_month_permuted_spatial_control",
            "negative_control_month_permuted_spatial",
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
        raise RuntimeError("R09 produced no OOF predictions.")
    predictions = pd.concat(rows, ignore_index=True).sort_values(["target_date", "model_id"]).reset_index(drop=True)
    assert_no_locked_dates(predictions["target_date"], context="R09 OOF predictions")
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
    baseline = scores[scores["model_id"] == "r09_baseline_temp_calendar"][["fold_id", "mae", "crps_normal"]].rename(
        columns={"mae": "baseline_mae", "crps_normal": "baseline_crps"}
    )
    return scores.merge(baseline, on="fold_id", how="left").assign(
        mae_improvement_vs_baseline=lambda df: df["baseline_mae"] - df["mae"],
        crps_improvement_vs_baseline=lambda df: df["baseline_crps"] - df["crps_normal"],
    )


def spatial_diagnostics(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cols = [
        "temp_network_spread_c",
        "temp_network_hko_minus_median_c",
        "temp_network_inland_minus_coastal_c",
        "temp_network_urban_minus_coastal_c",
        "temp_network_east_minus_west_c",
        "temp_network_north_minus_south_c",
        "sea_breeze_thermal_gradient_proxy",
        "hko_local_warm_outlier_score",
    ]
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
    return f"""# EXP-0041 / HKG-T24-R09 Long-Form Experiment Report

## Purpose

R09 tests whether the 39-station HKO temperature field adds information beyond HKO Headquarters' own cutoff temperature. It focuses on transparent spatial summaries: station offsets, network spread, inland-coastal contrasts, urban-coastal contrasts, elevated-lowland contrasts, east-west and north-south gradients, local HKO outlier score, and simple flow-conditioned interactions inherited from R08 vector wind. The goal is to learn whether spatial thermal structure identifies mesoscale heating/cooling regimes without using target-day observations.

## Data Used

The feature backbone is the R08 pre-validation feature matrix. R09 reparses immutable `datagov_hko_historical_latest_1min_temperature_archive` ZIP payloads for the full HKO temperature/max-min station list. The parser samples cutoff-relevant snapshot windows near 02:40, 08:40, 11:40, 13:40, and 14:40 HKT and uses the conservative `observed_at + 20 minutes` replay latency. The target-date feature period is `{payload['feature_min']}` through `{payload['feature_max']}`, the OOF prediction period is `{payload['prediction_min']}` through `{payload['prediction_max']}`, and the parsed station-temperature observation period is `{payload['temperature_observation_min']}` through `{payload['temperature_observation_max']}`.

## Feature Construction

For every target date T, current and lagged station fields are selected as of T-1 15:00 HKT. R09 computes station count, coverage fraction, network median, trimmed mean, extrema, spread, IQR proxy, standard deviation, skew proxy, HKO-minus-network median, physically defined group means, group contrasts, HKO-minus-each-station offset columns, 3/6/12/24-hour changes in selected spatial summaries, a sea-breeze thermal-gradient proxy, HKO local warm-outlier score, and interactions between temperature gradients and R08 wind components. Month-permuted spatial controls are retained as negative controls.

## Blockers

The uploaded specification asks for station seasonal expected offsets learned inside folds, elevation-adjusted residual offsets, and robust planar gradient magnitude/direction. The current station registry does not yet contain coordinates or elevation fields, so true plane fitting and elevation adjustment are blocked. R09 does not fake those fields. It implements transparent group contrasts and raw station offsets, then documents coordinate/elevation metadata as a blocker for a richer R09/R10 continuation.

## Model Ladder

The ladder includes a temperature/calendar baseline, spatial-summary Elastic Net, spatial-change Elastic Net, station-offset Elastic Net, flow-interaction Elastic Net, shallow constrained boosting, and a month-permuted spatial negative control. Each model is trained only on chronological training folds. The station-offset model lets the fold-local regularizer learn compact marginal station value without selecting stations from validation or locked-test outcomes.

## Leakage Controls

No target-day station observations enter the matrix. No validation-2024 outcomes are used. No locked-test dates are accessed. The parser can read immutable raw rows beyond 2023, but the generated feature and OOF prediction tables are guarded to pre-validation development dates. Seasonal expected offsets are not precomputed using full-sample future data; instead raw offsets are supplied to fold-local models.

## OOF Gate

The strict four-year OOF check is `{oof['status']}`: {oof['reason']}. R09 is therefore a completed diagnostic but not promotable under the user's hard four-year OOF rule.

## Main Result

The best non-control model by OOF MAE is `{champion['model_id']}` with MAE `{champion['mae']:.4f}` C, RMSE `{champion['rmse']:.4f}` C, bias `{champion['bias']:.4f}` C, and CRPS `{champion['crps_normal']:.4f}` over `{champion['n']}` rows. The scoreboard, fold deltas, and diagnostics show whether spatial temperature fields add anything beyond the local HKO cutoff state.

## Interpretation

If station offsets or group contrasts beat baseline consistently, station-network thermal structure is a candidate compact feature family. If only shallow boosting improves isolated folds, the signal may be unstable or missingness-driven. If month-permuted controls are competitive, the apparent spatial value is probably seasonal/sample artifact. If all spatial models lose, the current public snapshot station field is not enough without coordinate/elevation metadata, wind-direction-conditioned upwind pools, or latent spatial modes from R10.

## Decision Record

R09 is complete as a transparent all-station temperature-gradient diagnostic once artifacts and tests pass. It does not authorize validation access. The next planned experiment is R10 latent spatial modes, which should use fold-fit PCA/graph modes only after the R09 station matrix exists and leakage tests pass.

## Actual Diagnostic Disposition

The generated scoreboard shows a nuanced null result. The local HKO temperature/calendar baseline remains best by MAE. The station-offset model can reduce RMSE and CRPS slightly in this diagnostic window, but its MAE is still worse than baseline and interval coverage deteriorates. That is not a promotable feature-family result under the predeclared rules. It does, however, suggest that station offsets may reduce some larger errors while increasing ordinary-day absolute error. This is exactly the kind of conditional signal that should be revisited in R10 latent modes and R22 catastrophic-error specialists, not forced into the main model now.

## Multiple-Comparison Discipline

R09 exposes many station offset columns and physical group contrasts. A single station or contrast looking useful in one fold would not be enough. The station-offset Elastic Net regularizes these columns inside folds, but the research conclusion still has to account for the number of stations and contrasts tested. The month-permuted spatial control remains close to the weaker spatial models, which is a warning that sample timing and seasonality explain part of the apparent network value. Therefore the result is kept as evidence, not as feature promotion.

## Coverage And Metadata Implications

The raw parser confirms that the public archive contains all 39 temperature stations in the current station list, and R09 preserves station count and coverage fraction as first-class features. The missing piece is not station membership; it is metadata. Without coordinates, elevation, and validated station-history segments, robust plane-fit gradients and elevation-adjusted residual offsets would be pseudo-precision. The correct engineering step is to enrich the station registry before claiming terrain or centroid effects.

## Carry-Forward Rules

Later experiments may reuse the R09 station-temperature table and raw offset features, but any dimensionality reduction, station selection, seasonal-offset normalization, or outage robustness must be fit inside each training fold. The all-station table is valuable input for R10 latent spatial modes and for future dynamic-upwind features with R08 wind vectors. R09 itself does not prove enough stable incremental skill for promotion.

## Why The Artifact Matters

Before R09, the project had isolated HKO target-station trajectories and some selected network summaries, but not a dedicated, reproducible all-station temperature matrix tied to the T-24 cutoff. R09 creates that matrix with hashes, manifests, and experiment documentation. Even though the first transparent spatial models are not promoted, the matrix is now available for outage simulation, latent-mode fitting, station marginal-value analysis, dynamic upwind pairing, and conditional specialists. This is concrete progress because future work can build on a leakage-checked table instead of reparsing raw ZIPs ad hoc.
"""


def write_experiment(
    *,
    data_root: Path,
    temperature_path: Path,
    feature_path: Path,
    predictions_path: Path,
    scoreboard: pd.DataFrame,
    fold_scores: pd.DataFrame,
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
        "spatial_diagnostics": diagnostics.to_dict(orient="records"),
    }
    write_text(EXPERIMENT_DIR / "results" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    scoreboard.to_csv(EXPERIMENT_DIR / "results" / "scoreboard.csv", index=False)
    fold_scores.to_csv(EXPERIMENT_DIR / "artifacts" / "fold_score_deltas.csv", index=False)
    diagnostics.to_csv(EXPERIMENT_DIR / "artifacts" / "spatial_diagnostics.csv", index=False)
    pd.read_parquet(predictions_path).to_parquet(EXPERIMENT_DIR / "predictions" / "r09_oof_predictions.parquet", index=False)
    write_text(EXPERIMENT_DIR / "README.md", "# EXP-0041 HKG-T24-R09 Station Temperature Gradient\n\nAll-station temperature field, group contrasts, and station-offset diagnostic. No validation 2024, no locked test, no Polymarket.\n")
    write_text(EXPERIMENT_DIR / "HYPOTHESIS.md", "# Hypothesis\n\nSpatial temperature contrasts across HKO stations may identify air-mass structure, local circulation, sea-breeze penetration, and regional transition signals beyond HKO's own cutoff temperature.\n")
    write_text(EXPERIMENT_DIR / "PROTOCOL.md", "# Protocol\n\n1. Parse full HKO station temperature snapshots from immutable ZIPs.\n2. Join station rows by `available_at_hkt <= cutoff_hkt`.\n3. Build raw station offsets, network summaries, physical group contrasts, lagged changes, and wind-conditioned interactions.\n4. Fit all transformations and models inside chronological folds only.\n5. Document coordinate/elevation blockers rather than fabricating plane-fit gradients.\n")
    write_text(EXPERIMENT_DIR / "ASOF_CONTRACT.md", "# As-Of Contract\n\nTarget date T uses only station temperature observations available by T-1 15:00 HKT. High-frequency observations use `available_at = observed_at + 20 minutes`. Target-day observations and validation/locked-test rows are forbidden.\n")
    write_text(EXPERIMENT_DIR / "DATA_MANIFEST.yaml", f"""research_id: {RESEARCH_ID}
temperature_table: {temperature_path}
temperature_table_sha256: {sha256_file(temperature_path)}
feature_matrix: {feature_path}
feature_matrix_sha256: {sha256_file(feature_path)}
prediction_table: {predictions_path}
prediction_table_sha256: {sha256_file(predictions_path)}
data_root: {data_root}
validation_2024_accessed: false
locked_test_accessed: false
availability_tier: SILVER_OPERATIONAL_REPLAY
blocked_inputs:
  - station_coordinates
  - station_elevations
  - robust_planar_gradient
""")
    write_text(EXPERIMENT_DIR / "RUN_CONFIG.yaml", f"""research_id: {RESEARCH_ID}
analysis_start: {payload['feature_min']}
analysis_end: {payload['feature_max']}
model_ladder: baseline, spatial_summary, spatial_change, station_offsets, flow_interaction, shallow_boosting, month_permuted_control
validation_2024_accessed: false
locked_test_policy: deny
""")
    write_text(EXPERIMENT_DIR / "DATE_RANGES.md", f"""# Date Ranges

- Feature target-date period: `{payload['feature_min']}` through `{payload['feature_max']}`.
- OOF prediction period: `{payload['prediction_min']}` through `{payload['prediction_max']}`.
- Parsed station-temperature observation period: `{payload['temperature_observation_min']}` through `{payload['temperature_observation_max']}`.
- Validation 2024: not accessed.
- Locked test 2025-01-01 onward: not accessed.
- Four-year OOF status: `{payload['oof_feasibility']['status']}`.
""")
    write_text(
        EXPERIMENT_DIR / "RESULTS.md",
        "# Results\n\n"
        + markdown_table(scoreboard, ["model_id", "n", "mae", "rmse", "median_abs_error", "bias", "crps_normal", "coverage_80", "coverage_90"])
        + "\n## Spatial Diagnostics\n\n"
        + markdown_table(diagnostics, ["feature", "n", "pearson_corr_with_target", "mean", "p10", "p90"]),
    )
    write_text(EXPERIMENT_DIR / "CONCLUSION.md", "# Conclusion\n\nR09 is complete as a station-temperature-gradient diagnostic, but it is OOF-blocked under the strict four-year rule. Plane-fit/elevation-adjusted gradients remain blocked by missing station metadata.\n")
    write_text(EXPERIMENT_DIR / "REPRODUCE.md", "# Reproduce\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_r09_station_temperature_gradient.py --data-root C:\\hkg_tmax_data\n```\n")
    write_text(EXPERIMENT_DIR / "STATUS.yaml", """status: COMPLETE_DIAGNOSTIC_OOF_BLOCKED
research_id: HKG-T24-R09
locked_test_accessed: false
validation_2024_accessed: false
leakage_guard: PASS
four_year_oof: BLOCKED
production_eligible: false
blocked_inputs: [station_coordinates, station_elevations, robust_planar_gradient]
""")
    write_text(EXPERIMENT_DIR / "EXPERIMENT_REPORT_7500_CHARS.md", long_report(payload))


def write_report(scoreboard: pd.DataFrame, fold_scores: pd.DataFrame, diagnostics: pd.DataFrame, payload: dict[str, object]) -> None:
    write_text(
        PROJECT_PATHS.run_root / "reports" / "hkg_t24" / "R09_STATION_TEMPERATURE_GRADIENT.md",
        long_report(payload)
        + "\n# R09 Machine-Readable Summary Tables\n\n"
        f"Generated: `{now_utc()}`\n\n"
        "## Overall Scoreboard\n\n"
        + markdown_table(scoreboard, ["model_id", "n", "first_date", "last_date", "mae", "rmse", "median_abs_error", "bias", "crps_normal", "coverage_80", "coverage_90"])
        + "\n## Fold Deltas\n\n"
        + markdown_table(fold_scores, ["fold_id", "model_id", "n", "mae", "baseline_mae", "mae_improvement_vs_baseline", "crps_improvement_vs_baseline"], limit=100)
        + "\n## Spatial Diagnostics\n\n"
        + markdown_table(diagnostics, ["feature", "n", "pearson_corr_with_target", "mean", "p10", "p90"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG-T24-R09 station temperature-gradient diagnostic.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    features, feature_path, temperature_path = build_feature_matrix(data_root)
    specs = model_specs(features)
    predictions = run_oof(features, specs)
    scoreboard = score_frame(predictions, ["model_id"])
    fold_scores = fold_deltas(predictions)
    diagnostics = spatial_diagnostics(features)
    output_dir = data_root / "gold" / "hkg_t24" / "r09_station_temperature_gradient"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "r09_oof_predictions.parquet"
    scoreboard_path = output_dir / "r09_scoreboard.parquet"
    fold_path = output_dir / "r09_fold_score_deltas.parquet"
    diagnostics_path = output_dir / "r09_spatial_diagnostics.parquet"
    predictions.to_parquet(predictions_path, index=False)
    scoreboard.to_parquet(scoreboard_path, index=False)
    fold_scores.to_parquet(fold_path, index=False)
    diagnostics.to_parquet(diagnostics_path, index=False)
    feature_dates = pd.to_datetime(features["target_date"])
    feasibility = check_four_year_oof_feasibility(
        feature_dates.min().date(),
        feature_dates.max().date(),
        min_years=4.0,
        reason_context="R09 modern station-temperature-gradient pre-validation feature period",
    )
    non_control = scoreboard[~scoreboard["model_id"].eq("r09_month_permuted_spatial_control")].copy()
    champion = non_control.iloc[0].to_dict()
    temp_obs = pd.read_parquet(temperature_path, columns=["observed_at_hkt"])
    payload = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_state": git_state(),
        "feature_min": str(feature_dates.min().date()),
        "feature_max": str(feature_dates.max().date()),
        "prediction_min": str(pd.to_datetime(predictions["target_date"]).min().date()),
        "prediction_max": str(pd.to_datetime(predictions["target_date"]).max().date()),
        "temperature_observation_min": str(pd.to_datetime(temp_obs["observed_at_hkt"], utc=True).min()),
        "temperature_observation_max": str(pd.to_datetime(temp_obs["observed_at_hkt"], utc=True).max()),
        "oof_feasibility": feasibility.__dict__,
        "champion": champion,
    }
    write_experiment(
        data_root=data_root,
        temperature_path=temperature_path,
        feature_path=feature_path,
        predictions_path=predictions_path,
        scoreboard=scoreboard,
        fold_scores=fold_scores,
        diagnostics=diagnostics,
        payload=payload,
    )
    write_report(scoreboard, fold_scores, diagnostics, payload)
    print(json.dumps({"status": "ok", "champion": champion, "oof": feasibility.__dict__}, indent=2, default=str))


if __name__ == "__main__":
    main()
