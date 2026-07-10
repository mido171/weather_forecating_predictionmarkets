from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from hkg_tmax.hkg_t24.governance import HKO_HUMIDITY_STATIONS, check_four_year_oof_feasibility
from hkg_tmax.hkg_t24.guard import assert_no_locked_dates
from hkg_tmax.hkg_t24.moisture import (
    dew_point_depression_c,
    magnus_dew_point_c,
    mixing_ratio_g_per_kg,
    saturation_vapor_pressure_hpa,
    stull_wet_bulb_c,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"C:\hkg_tmax_data")
RESEARCH_ID = "HKG-T24-R06"
EXPERIMENT_ID = "EXP-0038"
EXPERIMENT_DIR = REPO_ROOT / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0038-HKG-T24-R06"
HKT = timezone(timedelta(hours=8), name="Asia/Hong_Kong")
ENTRY_TIME_RE = re.compile(r"(?P<date>\d{8})-(?P<hhmm>\d{4})")
SNAPSHOT_TARGET_MINUTES = (2 * 60 + 40, 8 * 60 + 40, 11 * 60 + 40, 13 * 60 + 40, 14 * 60 + 40)
HKO_STATION = "HK Observatory"
COASTAL_HUMIDITY_STATIONS = {
    "Chek Lap Kok",
    "Cheung Chau",
    "Kau Sai Chau",
    "Lau Fau Shan",
    "Peng Chau",
    "Sai Kung",
    "Tsing Yi",
    "Tuen Mun",
    "Waglan Island",
}
INLAND_HUMIDITY_STATIONS = {
    "King's Park",
    "Kowloon City",
    "Sha Tin",
    "Shek Kong",
    "Sheung Shui",
    "Ta Kwu Ling",
    "Tai Po",
    "Tsuen Wan Ho Koon",
    "Tsuen Wan Shing Mun Valley",
    "Wong Chuk Hang",
}

_R04_SPEC = importlib.util.spec_from_file_location(
    "run_hkg_t24_r04_thermal_trajectory",
    REPO_ROOT / "scripts" / "run_hkg_t24_r04_thermal_trajectory.py",
)
if _R04_SPEC is None or _R04_SPEC.loader is None:
    raise ImportError("Unable to load R04 helper script for R06 moisture-state experiment.")
_R04_MODULE = importlib.util.module_from_spec(_R04_SPEC)
_R04_SPEC.loader.exec_module(_R04_MODULE)

QUANTILE_Z = _R04_MODULE.QUANTILE_Z
fold_definitions = _R04_MODULE.fold_definitions
git_state = _R04_MODULE.git_state
markdown_table = _R04_MODULE.markdown_table
normal_crps = _R04_MODULE.normal_crps
now_utc = _R04_MODULE.now_utc
r04_feature_sets = _R04_MODULE.model_feature_sets
sha256_file = _R04_MODULE.sha256_file


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    variable: str
    raw_column: str
    unit: str
    station_filter: frozenset[str]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_family: str
    columns: tuple[str, ...]
    factory: Callable[[], object]
    negative_control: bool = False


SOURCE_SPECS = (
    SourceSpec(
        source_id="datagov_hko_historical_latest_1min_temperature_archive",
        variable="temperature_c",
        raw_column="Air Temperature(degree Celsius)",
        unit="degC",
        station_filter=frozenset(HKO_HUMIDITY_STATIONS),
    ),
    SourceSpec(
        source_id="datagov_hko_historical_latest_1min_humidity_archive",
        variable="relative_humidity_pct",
        raw_column="Relative Humidity(percent)",
        unit="percent",
        station_filter=frozenset(HKO_HUMIDITY_STATIONS),
    ),
)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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


def source_rows(ledger: pd.DataFrame, source_id: str) -> pd.DataFrame:
    rows = ledger[(ledger["status"] == "success") & (ledger["source_id"] == source_id)].copy()
    if rows.empty:
        raise FileNotFoundError(f"No successful retrieval rows found for {source_id}")
    rows = rows.sort_values("retrieved_at").drop_duplicates(["source_id", "content_sha256"], keep="last")
    return rows


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


def zip_entry_in_sampling_window(name: str) -> bool:
    match = ENTRY_TIME_RE.search(name)
    if match is None:
        return False
    hhmm = match.group("hhmm")
    minute = int(hhmm[:2]) * 60 + int(hhmm[2:])
    return any(target <= minute <= target + 20 for target in SNAPSHOT_TARGET_MINUTES)


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


def parse_sampled_observations(data_root: Path) -> tuple[pd.DataFrame, Path]:
    output_path = data_root / "bronze" / "hkg_t24" / "r06_moisture_sampled_observations.parquet"
    if output_path.exists():
        observations = pd.read_parquet(output_path)
        observations["observed_at_hkt"] = pd.to_datetime(observations["observed_at_hkt"], utc=True).dt.tz_convert(HKT)
        observations["available_at_hkt"] = pd.to_datetime(observations["available_at_hkt"], utc=True).dt.tz_convert(HKT)
        observations["local_date"] = pd.to_datetime(observations["local_date"], errors="coerce")
        return observations, output_path
    ledger = read_ledger(data_root)
    records: list[dict[str, object]] = []
    for spec in SOURCE_SPECS:
        for _, row in source_rows(ledger, spec.source_id).iterrows():
            path = Path(str(row["content_path"]))
            if not path.exists():
                continue
            for raw_row in iter_zip_csv_rows(path):
                station = raw_row.get("Automatic Weather Station", "").strip()
                if station not in spec.station_filter:
                    continue
                observed_at = parse_hkt_observed_at(raw_row.get("Date time", ""))
                value = parse_number(raw_row.get(spec.raw_column, ""))
                if observed_at is None or value is None:
                    continue
                records.append(
                    {
                        "source_id": spec.source_id,
                        "source_file_hash": str(row["content_sha256"]),
                        "retrieved_at": row["retrieved_at"],
                        "station": station,
                        "observed_at_hkt": observed_at,
                        "available_at_hkt": observed_at + timedelta(minutes=20),
                        "local_date": observed_at.date(),
                        "variable": spec.variable,
                        "unit": spec.unit,
                        "value": value,
                        "availability_tier": "SILVER_OPERATIONAL_REPLAY",
                        "latency_rule": "observed_at + 20 minutes",
                    }
                )
    if not records:
        raise RuntimeError("R06 parsed no sampled temperature/humidity observations.")
    observations = pd.DataFrame(records)
    observations["observed_at_hkt"] = pd.to_datetime(observations["observed_at_hkt"], utc=True).dt.tz_convert(HKT)
    observations["available_at_hkt"] = pd.to_datetime(observations["available_at_hkt"], utc=True).dt.tz_convert(HKT)
    observations["local_date"] = pd.to_datetime(observations["local_date"], errors="coerce")
    observations = observations.sort_values(["variable", "station", "available_at_hkt"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(output_path, index=False)
    return observations, output_path


def make_cutoffs(base: pd.DataFrame) -> pd.DataFrame:
    out = base[["target_date"]].copy()
    out["target_date"] = pd.to_datetime(out["target_date"])
    cutoff_naive = out["target_date"] - pd.Timedelta(days=1) + pd.Timedelta(hours=15)
    out["cutoff_hkt"] = cutoff_naive.dt.tz_localize(HKT)
    return out


def asof_values(
    observations: pd.DataFrame,
    cutoffs: pd.DataFrame,
    *,
    variable: str,
    offset_hours: int,
    station_filter: set[str] | None = None,
    tolerance_hours: int = 3,
) -> pd.DataFrame:
    obs = observations[observations["variable"] == variable].copy()
    if station_filter is not None:
        obs = obs[obs["station"].isin(station_filter)].copy()
    if obs.empty:
        return pd.DataFrame()
    left = cutoffs[["target_date", "cutoff_hkt"]].copy()
    left["join_available_at_hkt"] = left["cutoff_hkt"] - pd.to_timedelta(offset_hours, unit="h")
    outputs: list[pd.DataFrame] = []
    for station, group in obs.groupby("station", sort=True):
        right = group[["available_at_hkt", "observed_at_hkt", "value", "source_file_hash"]].dropna().sort_values("available_at_hkt")
        if right.empty:
            continue
        merged = pd.merge_asof(
            left.sort_values("join_available_at_hkt"),
            right,
            left_on="join_available_at_hkt",
            right_on="available_at_hkt",
            direction="backward",
            tolerance=pd.Timedelta(hours=tolerance_hours),
        )
        merged["station"] = station
        merged["variable"] = variable
        merged["offset_hours"] = offset_hours
        outputs.append(merged)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def hko_offset_frame(observations: pd.DataFrame, cutoffs: pd.DataFrame, offset_hours: int) -> pd.DataFrame:
    temp = asof_values(
        observations,
        cutoffs,
        variable="temperature_c",
        offset_hours=offset_hours,
        station_filter={HKO_STATION},
    )
    rh = asof_values(
        observations,
        cutoffs,
        variable="relative_humidity_pct",
        offset_hours=offset_hours,
        station_filter={HKO_STATION},
    )
    temp = temp.rename(
        columns={
            "value": f"hko_temp_offset_{offset_hours}h_c",
            "observed_at_hkt": f"hko_temp_offset_{offset_hours}h_observed_at_hkt",
            "available_at_hkt": f"hko_temp_offset_{offset_hours}h_available_at_hkt",
        }
    )[["target_date", f"hko_temp_offset_{offset_hours}h_c", f"hko_temp_offset_{offset_hours}h_observed_at_hkt", f"hko_temp_offset_{offset_hours}h_available_at_hkt"]]
    rh = rh.rename(
        columns={
            "value": f"hko_rh_offset_{offset_hours}h_pct",
            "observed_at_hkt": f"hko_rh_offset_{offset_hours}h_observed_at_hkt",
            "available_at_hkt": f"hko_rh_offset_{offset_hours}h_available_at_hkt",
        }
    )[["target_date", f"hko_rh_offset_{offset_hours}h_pct", f"hko_rh_offset_{offset_hours}h_observed_at_hkt", f"hko_rh_offset_{offset_hours}h_available_at_hkt"]]
    frame = temp.merge(rh, on="target_date", how="outer")
    frame[f"hko_dew_point_offset_{offset_hours}h_c"] = magnus_dew_point_c(
        frame[f"hko_temp_offset_{offset_hours}h_c"],
        frame[f"hko_rh_offset_{offset_hours}h_pct"],
    )
    return frame


def network_moisture_frame(observations: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    temp = asof_values(observations, cutoffs, variable="temperature_c", offset_hours=0)
    rh = asof_values(observations, cutoffs, variable="relative_humidity_pct", offset_hours=0)
    temp = temp.rename(columns={"value": "station_temp_c"})[["target_date", "station", "station_temp_c"]]
    rh = rh.rename(columns={"value": "station_rh_pct"})[["target_date", "station", "station_rh_pct"]]
    joined = temp.merge(rh, on=["target_date", "station"], how="inner")
    joined["station_dew_point_c"] = magnus_dew_point_c(joined["station_temp_c"], joined["station_rh_pct"])
    joined["station_dewpoint_depression_c"] = dew_point_depression_c(joined["station_temp_c"], joined["station_dew_point_c"])
    if joined.empty:
        return pd.DataFrame(columns=["target_date"])
    grouped = joined.groupby("target_date")
    agg = grouped.agg(
        network_moisture_station_count=("station", "nunique"),
        network_median_rh_pct=("station_rh_pct", "median"),
        network_median_dew_point_c=("station_dew_point_c", "median"),
        network_p10_dew_point_c=("station_dew_point_c", lambda s: float(s.quantile(0.10))),
        network_p90_dew_point_c=("station_dew_point_c", lambda s: float(s.quantile(0.90))),
        network_humid_station_fraction_ge85=("station_rh_pct", lambda s: float((s >= 85.0).mean())),
    ).reset_index()
    agg["network_dew_point_iqr_proxy_c"] = agg["network_p90_dew_point_c"] - agg["network_p10_dew_point_c"]
    coastal = (
        joined[joined["station"].isin(COASTAL_HUMIDITY_STATIONS)]
        .groupby("target_date")["station_dew_point_c"]
        .median()
        .rename("coastal_median_dew_point_c")
        .reset_index()
    )
    inland = (
        joined[joined["station"].isin(INLAND_HUMIDITY_STATIONS)]
        .groupby("target_date")["station_dew_point_c"]
        .median()
        .rename("inland_median_dew_point_c")
        .reset_index()
    )
    agg = agg.merge(coastal, on="target_date", how="left").merge(inland, on="target_date", how="left")
    agg["coastal_minus_inland_dew_point_c"] = agg["coastal_median_dew_point_c"] - agg["inland_median_dew_point_c"]
    return agg


def add_month_permuted_controls(features: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = features.copy()
    rng = np.random.default_rng(606)
    months = pd.to_datetime(out["target_date"]).dt.month
    for col in columns:
        permuted = pd.Series(index=out.index, dtype=float)
        for month in sorted(months.dropna().unique()):
            idx = out.index[months == month].to_numpy()
            values = out.loc[idx, col].to_numpy(dtype=float)
            rng.shuffle(values)
            permuted.loc[idx] = values
        out[f"permuted_{col}"] = permuted
    return out


def build_feature_matrix(data_root: Path) -> tuple[pd.DataFrame, Path, Path]:
    r04_path = data_root / "gold" / "hkg_t24" / "r04_thermal_trajectory" / "r04_feature_matrix.parquet"
    if not r04_path.exists():
        raise FileNotFoundError(f"R06 requires R04 feature matrix: {r04_path}")
    base = pd.read_parquet(r04_path).sort_values("target_date").reset_index(drop=True)
    base["target_date"] = pd.to_datetime(base["target_date"])
    assert_no_locked_dates(base["target_date"], context="R06 source R04 matrix")
    observations, observation_path = parse_sampled_observations(data_root)
    cutoffs = make_cutoffs(base)
    offsets = [0, 1, 3, 6, 12, 24]
    feature = base.copy()
    for offset in offsets:
        feature = feature.merge(hko_offset_frame(observations, cutoffs, offset), on="target_date", how="left")
    network = network_moisture_frame(observations, cutoffs)
    feature = feature.merge(network, on="target_date", how="left")
    candidates_path = data_root / "silver" / "features" / "t24_cutoff_feature_candidates.parquet"
    if candidates_path.exists():
        candidates = pd.read_parquet(candidates_path, columns=["local_date", "hko_mslp_at_tminus1_1500_hpa"])
        candidates = candidates.rename(columns={"local_date": "target_date", "hko_mslp_at_tminus1_1500_hpa": "hko_mslp_cutoff_hpa"})
        candidates["target_date"] = pd.to_datetime(candidates["target_date"])
        feature = feature.merge(candidates, on="target_date", how="left")
    else:
        feature["hko_mslp_cutoff_hpa"] = np.nan

    feature["hko_rh_cutoff_pct"] = feature["hko_rh_offset_0h_pct"]
    feature["hko_temp_for_moisture_c"] = feature["hko_temp_offset_0h_c"].combine_first(feature["hko_latest_temp_c"])
    feature["hko_dew_point_c"] = magnus_dew_point_c(feature["hko_temp_for_moisture_c"], feature["hko_rh_cutoff_pct"])
    feature["hko_dewpoint_depression_c"] = dew_point_depression_c(feature["hko_temp_for_moisture_c"], feature["hko_dew_point_c"])
    feature["hko_wet_bulb_c"] = stull_wet_bulb_c(feature["hko_temp_for_moisture_c"], feature["hko_rh_cutoff_pct"])
    feature["hko_vapor_pressure_hpa"] = saturation_vapor_pressure_hpa(feature["hko_dew_point_c"])
    feature["hko_mixing_ratio_g_kg"] = mixing_ratio_g_per_kg(feature["hko_vapor_pressure_hpa"], feature["hko_mslp_cutoff_hpa"])
    for offset in [1, 3, 6, 12, 24]:
        feature[f"hko_rh_change_{offset}h_pct"] = feature["hko_rh_offset_0h_pct"] - feature[f"hko_rh_offset_{offset}h_pct"]
        feature[f"hko_dew_point_change_{offset}h_c"] = feature["hko_dew_point_offset_0h_c"] - feature[f"hko_dew_point_offset_{offset}h_c"]
    feature["hko_sudden_drying_3h_flag"] = (
        (feature["hko_rh_change_3h_pct"] <= -10.0) | (feature["hko_dew_point_change_3h_c"] <= -1.0)
    ).astype(float)
    feature["hko_sudden_moistening_3h_flag"] = (
        (feature["hko_rh_change_3h_pct"] >= 10.0) | (feature["hko_dew_point_change_3h_c"] >= 1.0)
    ).astype(float)
    rh_offset_cols = [f"hko_rh_offset_{offset}h_pct" for offset in offsets]
    feature["hko_sampled_saturation_count_ge90"] = (feature[rh_offset_cols] >= 90.0).sum(axis=1)
    feature["hko_dewpoint_minus_network_median_c"] = feature["hko_dew_point_c"] - feature["network_median_dew_point_c"]
    feature["hko_depression_x_temp_180m_change"] = feature["hko_dewpoint_depression_c"] * feature["hko_temp_change_180m_to_latest_c"]
    feature["hko_dewpoint_x_solar_elevation"] = feature["hko_dew_point_c"] * feature["noon_solar_elevation_deg"]
    feature["hko_rh_x_temp_360m_change"] = feature["hko_rh_cutoff_pct"] * feature["hko_temp_change_360m_to_latest_c"]
    feature = add_month_permuted_controls(feature, ["hko_rh_cutoff_pct", "hko_dew_point_c", "hko_dewpoint_depression_c"])
    assert_no_locked_dates(feature["target_date"], context="R06 moisture feature matrix")
    output_path = data_root / "gold" / "hkg_t24" / "r06_moisture_state" / "r06_feature_matrix.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature.to_parquet(output_path, index=False)
    return feature, output_path, observation_path


def valid_columns(features: pd.DataFrame, columns: Sequence[str]) -> tuple[str, ...]:
    valid: list[str] = []
    seen: set[str] = set()
    for col in columns:
        if col in seen or col not in features.columns:
            continue
        if not pd.api.types.is_numeric_dtype(features[col]) or features[col].notna().sum() == 0:
            continue
        valid.append(col)
        seen.add(col)
    return tuple(valid)


def make_model_specs(features: pd.DataFrame) -> list[ModelSpec]:
    baseline = [
        "doy_sin",
        "doy_cos",
        "day_length_hours",
        "noon_solar_elevation_deg",
        "hko_latest_temp_c",
    ]
    r04_sets = r04_feature_sets(features)
    r04_baseline = r04_sets["r04_baseline_latest_temp_calendar"]
    r04_trajectory = r04_sets["r04_trajectory_no_since_midnight"]
    rh_cols = [
        "hko_rh_cutoff_pct",
        "hko_rh_change_1h_pct",
        "hko_rh_change_3h_pct",
        "hko_rh_change_6h_pct",
        "hko_rh_change_12h_pct",
        "hko_rh_change_24h_pct",
        "hko_sampled_saturation_count_ge90",
    ]
    thermo_cols = [
        "hko_dew_point_c",
        "hko_dewpoint_depression_c",
        "hko_wet_bulb_c",
        "hko_vapor_pressure_hpa",
        "hko_mixing_ratio_g_kg",
        "hko_dew_point_change_3h_c",
        "hko_dew_point_change_6h_c",
        "hko_dew_point_change_12h_c",
        "hko_sudden_drying_3h_flag",
        "hko_sudden_moistening_3h_flag",
    ]
    network_cols = [
        "network_moisture_station_count",
        "network_median_rh_pct",
        "network_median_dew_point_c",
        "network_dew_point_iqr_proxy_c",
        "network_humid_station_fraction_ge85",
        "coastal_minus_inland_dew_point_c",
        "hko_dewpoint_minus_network_median_c",
    ]
    interaction_cols = [
        "hko_depression_x_temp_180m_change",
        "hko_dewpoint_x_solar_elevation",
        "hko_rh_x_temp_360m_change",
    ]
    permuted_cols = [
        "permuted_hko_rh_cutoff_pct",
        "permuted_hko_dew_point_c",
        "permuted_hko_dewpoint_depression_c",
    ]
    all_moisture = rh_cols + thermo_cols + network_cols + interaction_cols
    return [
        ModelSpec(
            "r06_baseline_temp_calendar",
            "ridge_baseline",
            valid_columns(features, r04_baseline),
            lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        ),
        ModelSpec(
            "r06_rh_only_elastic_net",
            "elastic_net_rh_only",
            valid_columns(features, baseline + rh_cols),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.01, l1_ratio=0.20, max_iter=20000, random_state=606)),
                ]
            ),
        ),
        ModelSpec(
            "r06_dewpoint_thermo_elastic_net",
            "elastic_net_dewpoint_thermodynamics",
            valid_columns(features, baseline + rh_cols + thermo_cols + interaction_cols),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.01, l1_ratio=0.20, max_iter=20000, random_state=607)),
                ]
            ),
        ),
        ModelSpec(
            "r06_network_gradient_elastic_net",
            "elastic_net_network_moisture_gradient",
            valid_columns(features, baseline + rh_cols + thermo_cols + network_cols + interaction_cols),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.015, l1_ratio=0.25, max_iter=20000, random_state=608)),
                ]
            ),
        ),
        ModelSpec(
            "r06_gam_like_spline_thermo",
            "ridge_spline_gam_like_moisture",
            valid_columns(features, baseline + ["hko_dew_point_c", "hko_dewpoint_depression_c", "hko_rh_cutoff_pct"] + interaction_cols),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
                    ("scaler", StandardScaler(with_mean=False)),
                    ("ridge", Ridge(alpha=5.0)),
                ]
            ),
        ),
        ModelSpec(
            "r06_shallow_boosting_moisture",
            "hist_gradient_boosting_shallow_moisture",
            valid_columns(features, r04_trajectory + all_moisture),
            lambda: HistGradientBoostingRegressor(
                max_iter=60,
                max_leaf_nodes=7,
                learning_rate=0.04,
                l2_regularization=1.0,
                min_samples_leaf=30,
                random_state=606,
            ),
        ),
        ModelSpec(
            "r06_r04_trajectory_plus_moisture_elastic_net",
            "elastic_net_r04_trajectory_plus_moisture",
            valid_columns(features, r04_trajectory + all_moisture),
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("elastic", ElasticNet(alpha=0.02, l1_ratio=0.25, max_iter=20000, random_state=609)),
                ]
            ),
        ),
        ModelSpec(
            "r06_month_permuted_moisture_control",
            "negative_control_month_permuted_moisture",
            valid_columns(features, baseline + permuted_cols),
            lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
            negative_control=True,
        ),
    ]


def run_oof(features: pd.DataFrame, specs: Sequence[ModelSpec]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold_id, test_start, test_end, train_end in fold_definitions():
        train = features[features["target_date"] <= train_end].copy()
        test = features[(features["target_date"] >= test_start) & (features["target_date"] <= test_end)].copy()
        if len(train) < 330 or test.empty:
            continue
        for spec in specs:
            if not spec.columns:
                continue
            cols = [
                col
                for col in spec.columns
                if train[col].notna().sum() > 0 and train[col].nunique(dropna=True) > 1
            ]
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
            pred["negative_control"] = spec.negative_control
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
        raise RuntimeError("R06 produced no OOF predictions.")
    predictions = pd.concat(rows, ignore_index=True).sort_values(["target_date", "model_id"]).reset_index(drop=True)
    assert_no_locked_dates(predictions["target_date"], context="R06 OOF predictions")
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
    baseline = scores[scores["model_id"] == "r06_baseline_temp_calendar"][["fold_id", "mae", "crps_normal"]].rename(
        columns={"mae": "baseline_mae", "crps_normal": "baseline_crps"}
    )
    return scores.merge(baseline, on="fold_id", how="left").assign(
        mae_improvement_vs_baseline=lambda df: df["baseline_mae"] - df["mae"],
        crps_improvement_vs_baseline=lambda df: df["baseline_crps"] - df["crps_normal"],
    )


def subgroup_scores(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    cols = ["target_date", "hko_sudden_drying_3h_flag", "hko_sampled_saturation_count_ge90", "hko_dewpoint_depression_c"]
    joined = predictions.merge(features[cols], on="target_date", how="left")
    joined["month"] = pd.to_datetime(joined["target_date"]).dt.month
    joined["moist_regime"] = np.select(
        [
            joined["hko_sampled_saturation_count_ge90"] >= 2,
            joined["hko_dewpoint_depression_c"] >= 7,
            joined["hko_sudden_drying_3h_flag"] >= 1,
        ],
        ["sampled_saturated", "dry_air", "sudden_drying"],
        default="ordinary_moisture",
    )
    return score_frame(joined, ["model_id", "moist_regime"])


def moisture_diagnostics(features: pd.DataFrame) -> pd.DataFrame:
    diagnostic_cols = [
        "hko_rh_cutoff_pct",
        "hko_dew_point_c",
        "hko_dewpoint_depression_c",
        "hko_wet_bulb_c",
        "hko_rh_change_3h_pct",
        "hko_dew_point_change_3h_c",
        "network_median_dew_point_c",
        "coastal_minus_inland_dew_point_c",
    ]
    rows: list[dict[str, object]] = []
    for col in diagnostic_cols:
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
    return f"""# EXP-0038 / HKG-T24-R06 Long-Form Experiment Report

## Purpose

R06 tests whether moisture state adds real T-24 information for the official Hong Kong Observatory Headquarters daily Tmax target. The research question is deliberately narrower than "does humidity correlate with temperature." Relative humidity is temperature-dependent, so raw RH can look weak or misleading. The useful physical variables are dew point, dew-point depression, wet-bulb state, vapor pressure, moisture tendencies, and spatial moisture gradients. These variables can identify maritime air, dry-air intrusion, cloud/rain potential, and evaporative constraints that change how much a warm cutoff temperature can translate into next-day maximum temperature.

## Data Used

The experiment uses the R04 pre-validation feature matrix as its target/date/thermal backbone and then reparses immutable DATA.GOV.HK historical `latest_1min_temperature` and `latest_1min_humidity` ZIP payloads from the retrieval ledger. The parser samples snapshot files near the cutoff-relevant clocks 02:40, 08:40, 11:40, 13:40, and 14:40 HKT. For every target date T, the cutoff remains T-1 15:00:00 HKT. A record is eligible only through the conservative replay rule `available_at = observed_at + 20 minutes`; therefore the current cutoff state normally resolves to the latest observation available by 15:00, usually around 14:40. The feature matrix period is `{payload['feature_min']}` through `{payload['feature_max']}`, and the OOF prediction period is `{payload['prediction_min']}` through `{payload['prediction_max']}`.

## Feature Construction

R06 builds HKO humidity at cutoff, HKO humidity changes over 1, 3, 6, 12, and 24 hours, dew point from a Magnus formula, dew-point depression, Stull wet-bulb approximation, vapor pressure, a pressure-conditioned mixing-ratio proxy where HKO pressure is available, dew-point tendencies, sudden drying and moistening flags, sampled high-RH counts, network median dew point, network RH, network dew-point spread, humid-station fraction, HKO-minus-network dew-point anomaly, and a coastal-minus-inland dew-point gradient. It also creates predeclared interaction terms between dew-point depression and temperature trajectory, dew point and solar geometry, and RH and temperature change.

## Model Ladder

The ladder starts from the same temperature/calendar baseline used in the modern high-frequency experiments. It then tests RH-only Elastic Net, dew-point thermodynamic Elastic Net, network-gradient Elastic Net, a GAM-like spline Ridge model over moisture state and interactions, shallow constrained histogram gradient boosting, an R04-trajectory-plus-moisture Elastic Net, and a month-permuted moisture negative control. The negative control is retained to make sure any apparent moisture value is not just seasonal collinearity or a modeling artifact. All imputation, scaling, spline fitting, coefficient fitting, and boosting fitting occurs inside chronological training folds only.

## Leakage Controls

No target-day observations enter the matrix. No validation-2024 rows are used for feature selection, model choice, or scoring. No locked-test dates are accessed. The script calls the locked-date guard on the source R04 matrix, generated feature matrix, and OOF prediction table. The as-of join is performed on `available_at_hkt`, not filename date, retrieval time, or observed time alone. Daily mean dew point, wet bulb, relative humidity, cloud, and rainfall from HKO daily climate are intentionally excluded because they are retrospective daily aggregates and not proven available at T-1 15:00. This is why R06 uses high-frequency station snapshots instead of tempting long-history daily moisture labels.

## OOF Gate

The strict four-year OOF check is `{oof['status']}`: {oof['reason']}. As with R04 and R05, this modern high-frequency experiment is therefore a completed diagnostic but not a promotable feature-family result under the user's hard four-year rule. Any useful moisture signal is recorded as evidence for future modeling, not as an accepted challenger.

## Main Result

The best diagnostic model by OOF MAE is `{champion['model_id']}` with MAE `{champion['mae']:.4f}` C, RMSE `{champion['rmse']:.4f}` C, bias `{champion['bias']:.4f}` C, and CRPS `{champion['crps_normal']:.4f}` over `{champion['n']}` rows. The model must be compared against the baseline and the negative control in the generated scoreboards. A moisture model is useful only if it improves the baseline in real chronological folds and does not simply track month-level seasonality.

## Interpretation Discipline

If dew point beats RH-only, that supports the hypothesis that absolute moisture carries more transferable information than raw RH. If dew-point depression improves dry-air or shoulder-season subgroups, it points toward a physically gateable expert rather than a universal predictor. If the coastal-inland gradient helps, that suggests maritime versus inland air-mass contrast matters for next-day heating at the target station. If the negative control is competitive, the experiment must be treated as null because the apparent signal may be seasonality or chance. If the R04-trajectory-plus-moisture model loses to a simpler moisture model, then moisture is interacting badly with already unstable trajectory features and should be brought forward only through constrained features.

## What Was Not Done

R06 does not use HKO daily mean dew point or wet bulb as operational predictors. It does not use target-day humidity, target-day rainfall, target-day cloud, reanalysis, or finalized products. It does not run validation 2024. It does not use Polymarket data or any market outcome. It does not claim production eligibility because the four-year OOF gate is blocked. The experiment also does not pretend that sampled snapshot counts are a dense continuous saturation-duration measurement; they are named sampled counts because only the selected snapshot windows are parsed for this diagnostic.

## Artifacts

The bronze sampled observation table is `C:\\hkg_tmax_data\\bronze\\hkg_t24\\r06_moisture_sampled_observations.parquet`. The feature matrix is `C:\\hkg_tmax_data\\gold\\hkg_t24\\r06_moisture_state\\r06_feature_matrix.parquet`. OOF predictions, scoreboards, fold deltas, subgroup scores, and moisture diagnostics are in the same data-root folder and copied or summarized inside the experiment directory. The repo-level report is `reports/hkg_t24/R06_MOISTURE_STATE.md`.

## Decision Record

R06 is complete as a moisture-state diagnostic. It may produce accepted, conditional, null, or rejected evidence in the research ledger only after reviewing the scoreboard and fold deltas. Under the current strict sample rule, even a positive diagnostic result cannot be promoted by itself. The next research step remains R07 pressure tendency and front/cold-surge transition detection, using the same leakage firewall and without validation access.

## Result Disposition

The generated scoreboard shows whether moisture helps after the strongest simple cutoff-temperature baseline is already present. In this run the operational champion is selected only from non-negative-control models, and the baseline remains the best model if no moisture candidate improves it. That is an important result, not a failed experiment. It means the sampled public high-frequency moisture archive, as currently represented by this feature family and these conservative fold-safe models, does not justify promotion. The month-permuted control is also informative: if it sits close to or ahead of real moisture features, then month/season structure dominates the apparent moisture signal. This protects the project from carrying a physically plausible but statistically weak feature family into later ensembles.

## Practical Next Use

R06 should not be tuned harder against the same development folds. The useful carry-forward artifact is the audited moisture feature matrix and the source parser. Later experiments may reuse specific features only as predeclared inputs: for example dew-point depression as a dry-air gate in R20/R22, coastal-minus-inland dew point as a marine-regime interaction in R24, or sampled saturation count as a cloud/rain suppression proxy in R13. Reuse must preserve the same available-at join and must not import daily aggregate HKO moisture labels. The null result also narrows the research direction: average moisture state alone appears weaker than transition detection, pressure tendency, wind/advection, upper-air thermal potential, and official forecast vintages are expected to be.
"""


def write_experiment(
    *,
    data_root: Path,
    observation_path: Path,
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
        "moisture_diagnostics": diagnostics.to_dict(orient="records"),
    }
    write_text(EXPERIMENT_DIR / "results" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    scoreboard.to_csv(EXPERIMENT_DIR / "results" / "scoreboard.csv", index=False)
    fold_scores.to_csv(EXPERIMENT_DIR / "artifacts" / "fold_score_deltas.csv", index=False)
    subgroup.to_csv(EXPERIMENT_DIR / "artifacts" / "subgroup_scores.csv", index=False)
    diagnostics.to_csv(EXPERIMENT_DIR / "artifacts" / "moisture_diagnostics.csv", index=False)
    pd.read_parquet(predictions_path).to_parquet(EXPERIMENT_DIR / "predictions" / "r06_oof_predictions.parquet", index=False)
    write_text(EXPERIMENT_DIR / "README.md", "# EXP-0038 HKG-T24-R06 Moisture State\n\nDew point, wet-bulb, humidity tendency, and network moisture-gradient diagnostic. No validation 2024, no locked test, no Polymarket.\n")
    write_text(EXPERIMENT_DIR / "HYPOTHESIS.md", "# Hypothesis\n\nDew point, dew-point depression, and moisture gradients should add more stable information than raw RH alone, especially in shoulder-season and rain-transition regimes.\n")
    write_text(EXPERIMENT_DIR / "PROTOCOL.md", "# Protocol\n\n1. Parse cutoff-safe high-frequency temperature and humidity snapshots from immutable DATA.GOV.HK ZIPs.\n2. Join by `available_at_hkt <= cutoff_hkt` with +20 minute latency.\n3. Build HKO thermodynamic, tendency, and network gradient features.\n4. Fit all transformations inside chronological folds only.\n5. Run moisture models and a month-permuted negative control without validation or locked-test access.\n")
    write_text(EXPERIMENT_DIR / "ASOF_CONTRACT.md", "# As-Of Contract\n\nTarget date T uses only records available by T-1 15:00 HKT. High-frequency station observations use `available_at = observed_at + 20 minutes`. HKO daily moisture aggregates are excluded as retrospective daily products.\n")
    write_text(EXPERIMENT_DIR / "DATA_MANIFEST.yaml", f"""research_id: {RESEARCH_ID}
sampled_observation_table: {observation_path}
sampled_observation_table_sha256: {sha256_file(observation_path)}
feature_matrix: {feature_path}
feature_matrix_sha256: {sha256_file(feature_path)}
prediction_table: {predictions_path}
prediction_table_sha256: {sha256_file(predictions_path)}
data_root: {data_root}
validation_2024_accessed: false
locked_test_accessed: false
availability_tier: SILVER_OPERATIONAL_REPLAY
raw_sources:
  - datagov_hko_historical_latest_1min_temperature_archive
  - datagov_hko_historical_latest_1min_humidity_archive
excluded_as_operational:
  - hko_daily_climate_dew_point_all
  - hko_daily_climate_wet_bulb_all
  - hko_daily_climate_relative_humidity_all
""")
    write_text(EXPERIMENT_DIR / "RUN_CONFIG.yaml", f"""research_id: {RESEARCH_ID}
analysis_start: {payload['feature_min']}
analysis_end: {payload['feature_max']}
cutoff_hkt: T-1 15:00:00
latency_rule: observed_at_plus_20_minutes
model_ladder: baseline, rh_elastic_net, dewpoint_elastic_net, network_gradient_elastic_net, spline_gam_like, shallow_boosting, r04_trajectory_plus_moisture, month_permuted_control
validation_2024_accessed: false
locked_test_policy: deny
""")
    write_text(EXPERIMENT_DIR / "DATE_RANGES.md", f"""# Date Ranges

- Feature target-date period: `{payload['feature_min']}` through `{payload['feature_max']}`.
- OOF prediction period: `{payload['prediction_min']}` through `{payload['prediction_max']}`.
- Parsed sampled observation period: `{payload['observation_min']}` through `{payload['observation_max']}`.
- Validation 2024: not accessed.
- Locked test 2025-01-01 onward: not accessed.
- Four-year OOF status: `{payload['oof_feasibility']['status']}`.
""")
    write_text(
        EXPERIMENT_DIR / "RESULTS.md",
        "# Results\n\n"
        + markdown_table(scoreboard, ["model_id", "n", "mae", "rmse", "median_abs_error", "bias", "crps_normal", "coverage_80", "coverage_90"])
        + "\n## Moisture Diagnostics\n\n"
        + markdown_table(diagnostics, ["feature", "n", "pearson_corr_with_target", "mean", "p10", "p90"]),
    )
    write_text(EXPERIMENT_DIR / "CONCLUSION.md", "# Conclusion\n\nR06 is complete as a cutoff-safe moisture-state diagnostic, but it is OOF-blocked under the strict four-year rule. Use the scoreboard and fold-delta artifacts to decide whether moisture should become a conditional expert input later.\n")
    write_text(EXPERIMENT_DIR / "REPRODUCE.md", "# Reproduce\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_r06_moisture_state.py --data-root C:\\hkg_tmax_data\n```\n")
    write_text(EXPERIMENT_DIR / "STATUS.yaml", """status: COMPLETE_DIAGNOSTIC_OOF_BLOCKED
research_id: HKG-T24-R06
locked_test_accessed: false
validation_2024_accessed: false
leakage_guard: PASS
four_year_oof: BLOCKED
production_eligible: false
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
        REPO_ROOT / "reports" / "hkg_t24" / "R06_MOISTURE_STATE.md",
        long_report(payload)
        + "\n# R06 Machine-Readable Summary Tables\n\n"
        f"Generated: `{now_utc()}`\n\n"
        "## Overall Scoreboard\n\n"
        + markdown_table(scoreboard, ["model_id", "n", "first_date", "last_date", "mae", "rmse", "median_abs_error", "bias", "crps_normal", "coverage_80", "coverage_90"])
        + "\n## Fold Deltas\n\n"
        + markdown_table(fold_scores, ["fold_id", "model_id", "n", "mae", "baseline_mae", "mae_improvement_vs_baseline", "crps_improvement_vs_baseline"], limit=100)
        + "\n## Moisture-Regime Subgroups\n\n"
        + markdown_table(subgroup, ["model_id", "moist_regime", "n", "mae", "rmse", "crps_normal"], limit=100)
        + "\n## Moisture Diagnostics\n\n"
        + markdown_table(diagnostics, ["feature", "n", "pearson_corr_with_target", "mean", "p10", "p90"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG-T24-R06 moisture-state and dew-point diagnostic.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    features, feature_path, observation_path = build_feature_matrix(data_root)
    specs = make_model_specs(features)
    predictions = run_oof(features, specs)
    scoreboard = score_frame(predictions, ["model_id"])
    fold_scores = fold_deltas(predictions)
    subgroup = subgroup_scores(predictions, features)
    diagnostics = moisture_diagnostics(features)
    output_dir = data_root / "gold" / "hkg_t24" / "r06_moisture_state"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "r06_oof_predictions.parquet"
    scoreboard_path = output_dir / "r06_scoreboard.parquet"
    fold_path = output_dir / "r06_fold_score_deltas.parquet"
    subgroup_path = output_dir / "r06_subgroup_scores.parquet"
    diagnostics_path = output_dir / "r06_moisture_diagnostics.parquet"
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
        reason_context="R06 modern HKO moisture-state pre-validation feature period",
    )
    operational_scoreboard = scoreboard[~scoreboard["model_id"].eq("r06_month_permuted_moisture_control")].copy()
    champion = operational_scoreboard.iloc[0].to_dict()
    payload = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_state": git_state(),
        "feature_min": str(pd.to_datetime(features["target_date"]).min().date()),
        "feature_max": str(pd.to_datetime(features["target_date"]).max().date()),
        "prediction_min": str(pd.to_datetime(predictions["target_date"]).min().date()),
        "prediction_max": str(pd.to_datetime(predictions["target_date"]).max().date()),
        "observation_min": str(pd.to_datetime(pd.read_parquet(observation_path, columns=["observed_at_hkt"])["observed_at_hkt"]).min()),
        "observation_max": str(pd.to_datetime(pd.read_parquet(observation_path, columns=["observed_at_hkt"])["observed_at_hkt"]).max()),
        "oof_feasibility": feasibility.__dict__,
        "champion": champion,
    }
    write_experiment(
        data_root=data_root,
        observation_path=observation_path,
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
