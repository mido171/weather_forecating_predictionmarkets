from __future__ import annotations

import csv
import math
import os
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from hkg_tmax.analysis_contracts import (
    HKT,
    hko_tminus1_15_cutoff,
    validate_point_in_time_rows,
)
from hkg_tmax.hko import parse_daily_climate_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HKG_TMAX_DATA_ROOT", r"C:\hkg_tmax_data"))
LEDGER_PATH = DATA_ROOT / "manifests" / "retrieval_ledger.csv"

ANALYSIS_ROOT = REPO_ROOT / "analysis" / "hkg_tmax_t24"
FINDINGS_ROOT = ANALYSIS_ROOT / "findings"
REPORTS_ROOT = REPO_ROOT / "reports"

BRONZE_ROOT = DATA_ROOT / "bronze" / "analysis_phase_a"
SILVER_ROOT = DATA_ROOT / "silver"
METADATA_ROOT = DATA_ROOT / "metadata"

PRIMARY_CUTOFF = "T-1 15:00 HKT"


DAILY_CLIMATE_META: dict[str, dict[str, str]] = {
    "hko_daily_climate_mslp_all": {
        "variable": "mean_sea_level_pressure",
        "unit": "hPa",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_mean_temperature_all": {
        "variable": "mean_temperature",
        "unit": "degC",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_dew_point_all": {
        "variable": "mean_dew_point_temperature",
        "unit": "degC",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_wet_bulb_all": {
        "variable": "mean_wet_bulb_temperature",
        "unit": "degC",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_relative_humidity_all": {
        "variable": "mean_relative_humidity",
        "unit": "percent",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_cloud_amount_all": {
        "variable": "mean_cloud_amount",
        "unit": "percent",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_rainfall_all": {
        "variable": "daily_rainfall",
        "unit": "mm",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_maximum_temperature_all": {
        "variable": "daily_maximum_temperature",
        "unit": "degC",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_minimum_temperature_all": {
        "variable": "daily_minimum_temperature",
        "unit": "degC",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_bright_sunshine_all": {
        "variable": "bright_sunshine_duration",
        "unit": "hours",
        "station_or_domain": "King's Park",
    },
    "hko_daily_climate_global_solar_radiation_all": {
        "variable": "global_solar_radiation",
        "unit": "MJ/m2",
        "station_or_domain": "King's Park",
    },
    "hko_daily_climate_evaporation_all": {
        "variable": "evaporation",
        "unit": "mm",
        "station_or_domain": "King's Park",
    },
    "hko_daily_climate_lightning_ground_all": {
        "variable": "cloud_to_ground_lightning",
        "unit": "count",
        "station_or_domain": "Hong Kong Territory",
    },
    "hko_daily_climate_lightning_cloud_all": {
        "variable": "cloud_to_cloud_lightning",
        "unit": "count",
        "station_or_domain": "Hong Kong Territory",
    },
    "hko_daily_climate_grass_min_temperature_all": {
        "variable": "grass_minimum_temperature",
        "unit": "degC",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_prevailing_wind_direction_all": {
        "variable": "prevailing_wind_direction",
        "unit": "degree_or_compass",
        "station_or_domain": "Waglan Island",
    },
    "hko_daily_climate_mean_wind_speed_all": {
        "variable": "mean_wind_speed",
        "unit": "km/h",
        "station_or_domain": "Waglan Island",
    },
    "hko_daily_climate_sea_temp_waglan_all": {
        "variable": "sea_temperature",
        "unit": "degC",
        "station_or_domain": "Waglan Island",
    },
    "hko_daily_climate_sea_temp_np_am_all": {
        "variable": "sea_temperature_am",
        "unit": "degC",
        "station_or_domain": "North Point",
    },
    "hko_daily_climate_sea_temp_np_pm_all": {
        "variable": "sea_temperature_pm",
        "unit": "degC",
        "station_or_domain": "North Point",
    },
    "hko_daily_climate_reduced_visibility_hka_all": {
        "variable": "reduced_visibility_hours",
        "unit": "hours",
        "station_or_domain": "Hong Kong International Airport",
    },
}

DAILY_CLIMATE_ROLES: dict[str, str] = {
    "daily_maximum_temperature": "TARGET_ONLY",
    "daily_minimum_temperature": "RETROSPECTIVE_MECHANISM_ONLY",
    "mean_temperature": "RETROSPECTIVE_MECHANISM_ONLY",
    "mean_dew_point_temperature": "RETROSPECTIVE_MECHANISM_ONLY",
    "mean_wet_bulb_temperature": "RETROSPECTIVE_MECHANISM_ONLY",
    "mean_relative_humidity": "RETROSPECTIVE_MECHANISM_ONLY",
    "mean_cloud_amount": "RETROSPECTIVE_MECHANISM_ONLY",
    "daily_rainfall": "RETROSPECTIVE_MECHANISM_ONLY",
    "bright_sunshine_duration": "RETROSPECTIVE_MECHANISM_ONLY",
    "global_solar_radiation": "RETROSPECTIVE_MECHANISM_ONLY",
    "evaporation": "RETROSPECTIVE_MECHANISM_ONLY",
    "cloud_to_ground_lightning": "RETROSPECTIVE_MECHANISM_ONLY",
    "cloud_to_cloud_lightning": "RETROSPECTIVE_MECHANISM_ONLY",
    "grass_minimum_temperature": "RETROSPECTIVE_MECHANISM_ONLY",
    "prevailing_wind_direction": "RETROSPECTIVE_MECHANISM_ONLY",
    "mean_wind_speed": "RETROSPECTIVE_MECHANISM_ONLY",
    "sea_temperature": "RETROSPECTIVE_MECHANISM_ONLY",
    "sea_temperature_am": "RETROSPECTIVE_MECHANISM_ONLY",
    "sea_temperature_pm": "RETROSPECTIVE_MECHANISM_ONLY",
    "reduced_visibility_hours": "RETROSPECTIVE_MECHANISM_ONLY",
    "mean_sea_level_pressure": "RETROSPECTIVE_MECHANISM_ONLY",
}

HIGH_FREQUENCY_FAMILIES: dict[str, dict[str, object]] = {
    "datagov_hko_historical_latest_1min_temperature_archive": {
        "family": "latest_1min_temperature",
        "variables": {
            "Air Temperature(degree Celsius)": ("air_temperature_c", "degC"),
        },
    },
    "datagov_hko_historical_latest_1min_humidity_archive": {
        "family": "latest_1min_humidity",
        "variables": {
            "Relative Humidity(percent)": ("relative_humidity_pct", "percent"),
        },
    },
    "datagov_hko_historical_latest_1min_pressure_archive": {
        "family": "latest_1min_pressure",
        "variables": {
            "Mean Sea Level Pressure(hPa)": ("msl_pressure_hpa", "hPa"),
        },
    },
    "datagov_hko_historical_latest_since_midnight_maxmin_archive": {
        "family": "latest_since_midnight_maxmin",
        "variables": {
            "MaximumAir Temperature Since Midnight(degree Celsius)": (
                "temperature_since_midnight_max_c",
                "degC",
            ),
            "Minimum Air Temperature Since Midnight(degree Celsius)": (
                "temperature_since_midnight_min_c",
                "degC",
            ),
        },
    },
    "datagov_hko_historical_latest_1min_solar_archive": {
        "family": "latest_1min_solar",
        "variables": {
            "Global Solar Radiation(watt/square meter)": ("global_solar_wm2", "W/m2"),
            "Direct Solar Radiation(watt/square meter)": ("direct_solar_wm2", "W/m2"),
            "Diffuse Radiation(watt/square meter)": ("diffuse_solar_wm2", "W/m2"),
        },
    },
    "datagov_hko_historical_latest_10min_wind_archive": {
        "family": "latest_10min_wind",
        "variables": {
            "10-Minute Mean Speed(km/hour)": ("mean_wind_speed_kmh", "km/h"),
            "10-Minute Maximum Gust(km/hour)": ("max_wind_gust_kmh", "km/h"),
        },
    },
    "datagov_hko_historical_latest_15min_uvindex_archive": {
        "family": "latest_15min_uvindex",
        "variables": {
            "UV Index": ("uv_index", "index"),
        },
    },
}

HQ_STATIONS = {"HK Observatory", "Hong Kong Observatory"}
NETWORK_STATIONS = {
    "HK Observatory",
    "King's Park",
    "Waglan Island",
    "Chek Lap Kok",
    "Cheung Chau",
    "Sha Tin",
    "Ta Kwu Ling",
    "Lau Fau Shan",
    "Wong Chuk Hang",
    "Happy Valley",
    "Kai Tak",
    "Kai Tak Runway Park",
    "Tseung Kwan O",
    "Sai Kung",
    "Tai Po",
    "Sheung Shui",
    "Wetland Park",
}
SUPPORT_STATIONS = {"King's Park", "Waglan Island", "Chek Lap Kok", "Green Island", "Kai Tak"}
FEATURE_WINDOW_HOURS = {9, 12, 15}
ZIP_ENTRY_TIME_RE = re.compile(r"(?P<date>\d{8})-(?P<hhmm>\d{4})")


@dataclass
class ParseSummary:
    source_id: str
    rows: int
    unique_stations: int
    observed_start: str
    observed_end: str


def ensure_dirs() -> None:
    for path in [
        ANALYSIS_ROOT,
        FINDINGS_ROOT,
        REPORTS_ROOT,
        BRONZE_ROOT,
        SILVER_ROOT / "targets",
        SILVER_ROOT / "features",
        SILVER_ROOT / "observations",
        METADATA_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_ledger() -> pd.DataFrame:
    if not LEDGER_PATH.exists():
        raise FileNotFoundError(f"Missing retrieval ledger: {LEDGER_PATH}")
    ledger = pd.read_csv(LEDGER_PATH)
    ledger["retrieved_at"] = pd.to_datetime(ledger["retrieved_at"], utc=True, errors="coerce")
    return ledger


def latest_successes(ledger: pd.DataFrame, source_ids: Iterable[str]) -> pd.DataFrame:
    source_set = set(source_ids)
    rows = ledger[(ledger["status"] == "success") & (ledger["source_id"].isin(source_set))].copy()
    if rows.empty:
        return rows
    rows = rows.sort_values("retrieved_at").drop_duplicates(["source_id", "content_sha256"], keep="last")
    return rows


def decimal_to_float(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(value)


def parse_daily_climate(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for _, source in latest_successes(ledger, DAILY_CLIMATE_META.keys()).iterrows():
        source_id = str(source["source_id"])
        meta = DAILY_CLIMATE_META[source_id]
        content_path = Path(str(source["content_path"]))
        parsed = parse_daily_climate_csv(content_path.read_bytes())
        role = DAILY_CLIMATE_ROLES[meta["variable"]]
        for item in parsed:
            rows.append(
                {
                    "source_id": source_id,
                    "content_sha256": str(source["content_sha256"]),
                    "retrieved_at": source["retrieved_at"],
                    "station_or_domain": meta["station_or_domain"],
                    "variable": meta["variable"],
                    "unit": meta["unit"],
                    "role": role,
                    "local_date": item.local_date,
                    "year": item.year,
                    "month": item.month,
                    "day": item.day,
                    "value": decimal_to_float(item.value),
                    "value_precision": decimal_to_float(item.value_precision),
                    "completeness": item.completeness,
                    "parse_issue": item.parse_issue,
                }
            )
    daily = pd.DataFrame(rows)
    daily["local_date"] = pd.to_datetime(daily["local_date"], errors="coerce")
    daily.to_parquet(BRONZE_ROOT / "hko_daily_climate_elements.parquet", index=False)

    target = daily[
        (daily["source_id"] == "hko_daily_climate_maximum_temperature_all")
        & daily["local_date"].notna()
        & daily["value"].notna()
    ][["local_date", "value", "content_sha256", "retrieved_at"]].copy()
    target = target.rename(columns={"value": "target_tmax_c", "retrieved_at": "raw_retrieved_at"})
    target["target_station"] = "Hong Kong Observatory"
    target["target_source_id"] = "hko_daily_climate_maximum_temperature_all"
    target["target_role"] = "TARGET_ONLY"
    target = target.sort_values("local_date").drop_duplicates("local_date", keep="last")
    target.to_parquet(SILVER_ROOT / "targets" / "hko_daily_tmax.parquet", index=False)
    return daily, target


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


def parse_number(value: str) -> float | None:
    token = str(value).strip()
    if not token or token.upper() in {"N/A", "NA", "NULL", "***", "-"}:
        return None
    try:
        result = float(token)
    except ValueError:
        return None
    if math.isnan(result):
        return None
    return result


def zip_entry_in_feature_window(name: str, hours: set[int] | None = None) -> bool:
    """Limit huge historical ZIP scans to issue times near operational feature cutoffs."""

    match = ZIP_ENTRY_TIME_RE.search(name)
    if match is None:
        return False
    hhmm = match.group("hhmm")
    minute_of_day = int(hhmm[:2]) * 60 + int(hhmm[2:])
    wanted_hours = FEATURE_WINDOW_HOURS if hours is None else hours
    for hour in wanted_hours:
        target = hour * 60
        if target - 25 <= minute_of_day <= target + 45:
            return True
    return False


def iter_zip_csv_rows(
    path: Path, *, hours: set[int] | None = None, feature_window_only: bool = True
) -> Iterable[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv"):
                continue
            if feature_window_only and not zip_entry_in_feature_window(name, hours):
                continue
            with archive.open(name) as raw:
                text = raw.read().decode("utf-8-sig", errors="replace").splitlines()
            if not text:
                continue
            reader = csv.DictReader(text)
            for row in reader:
                yield {str(key).strip(): str(value).strip() for key, value in row.items() if key is not None}


def selected_stations_for_source(source_id: str) -> set[str]:
    if source_id in {
        "datagov_hko_historical_latest_1min_solar_archive",
        "datagov_hko_historical_latest_10min_wind_archive",
    }:
        return HQ_STATIONS | SUPPORT_STATIONS | NETWORK_STATIONS
    return HQ_STATIONS


def parse_selected_high_frequency(ledger: pd.DataFrame) -> tuple[pd.DataFrame, list[ParseSummary]]:
    observations: list[dict[str, object]] = []
    summaries: list[ParseSummary] = []
    source_rows = latest_successes(ledger, HIGH_FREQUENCY_FAMILIES.keys())
    for _, source in source_rows.iterrows():
        source_id = str(source["source_id"])
        family = HIGH_FREQUENCY_FAMILIES[source_id]
        variable_map = family["variables"]
        assert isinstance(variable_map, dict)
        stations = selected_stations_for_source(source_id)
        source_count = 0
        source_stations: set[str] = set()
        observed_values: list[datetime] = []
        path = Path(str(source["content_path"]))
        if not path.exists():
            continue
        for raw_row in iter_zip_csv_rows(path):
            station = raw_row.get("Automatic Weather Station", "").strip()
            if station not in stations:
                continue
            observed_at = parse_hkt_observed_at(raw_row.get("Date time", ""))
            if observed_at is None:
                continue
            for raw_column, variable_pair in variable_map.items():
                variable, unit = variable_pair
                value = parse_number(raw_row.get(raw_column, ""))
                if value is None:
                    continue
                observations.append(
                    {
                        "source_id": source_id,
                        "family": str(family["family"]),
                        "content_sha256": str(source["content_sha256"]),
                        "retrieved_at": source["retrieved_at"],
                        "station": station,
                        "observed_at_hkt": observed_at,
                        "local_date": observed_at.date(),
                        "variable": variable,
                        "unit": unit,
                        "value": value,
                        "role": "OPERATIONAL_WITH_CONSERVATIVE_LATENCY",
                        "availability_assumption": "observed_at + 20 minutes conservative historical replay latency",
                        "available_at_hkt": observed_at + timedelta(minutes=20),
                    }
                )
                source_count += 1
                source_stations.add(station)
                observed_values.append(observed_at)
        if observed_values:
            summaries.append(
                ParseSummary(
                    source_id=source_id,
                    rows=source_count,
                    unique_stations=len(source_stations),
                    observed_start=min(observed_values).isoformat(),
                    observed_end=max(observed_values).isoformat(),
                )
            )
        else:
            summaries.append(
                ParseSummary(
                    source_id=source_id,
                    rows=0,
                    unique_stations=0,
                    observed_start="",
                    observed_end="",
                )
            )

    obs = pd.DataFrame(observations)
    if not obs.empty:
        obs["observed_at_hkt"] = pd.to_datetime(obs["observed_at_hkt"], utc=True).dt.tz_convert(HKT)
        obs["available_at_hkt"] = pd.to_datetime(obs["available_at_hkt"], utc=True).dt.tz_convert(HKT)
        obs["local_date"] = pd.to_datetime(obs["local_date"], errors="coerce")
        obs.to_parquet(
            BRONZE_ROOT / "hko_high_frequency_selected_station_observations.parquet",
            index=False,
        )
    return obs, summaries


def parse_temperature_network_cutoff_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    source_rows = latest_successes(
        ledger, ["datagov_hko_historical_latest_1min_temperature_archive"]
    )
    aggregates: dict[tuple[str, object], dict[str, object]] = {}
    for _, source in source_rows.iterrows():
        path = Path(str(source["content_path"]))
        if not path.exists():
            continue
        for raw_row in iter_zip_csv_rows(path, hours={15}):
            observed_at = parse_hkt_observed_at(raw_row.get("Date time", ""))
            station = raw_row.get("Automatic Weather Station", "").strip()
            value = parse_number(raw_row.get("Air Temperature(degree Celsius)", ""))
            if observed_at is None or value is None or not station:
                continue
            key = (station, observed_at.date())
            item = aggregates.setdefault(
                key,
                {
                    "station": station,
                    "local_date": observed_at.date(),
                    "count": 0,
                    "latest_before_1500_at": None,
                    "latest_before_1500_c": None,
                },
            )
            item["count"] = int(item["count"]) + 1
            if observed_at.hour < 15 or (observed_at.hour == 15 and observed_at.minute == 0):
                previous = item["latest_before_1500_at"]
                if previous is None or observed_at > previous:
                    item["latest_before_1500_at"] = observed_at
                    item["latest_before_1500_c"] = value

    records: list[dict[str, object]] = []
    for item in aggregates.values():
        count = int(item["count"])
        records.append(
            {
                "station": item["station"],
                "local_date": item["local_date"],
                "cutoff_window_obs_count": count,
                "latest_before_1500_at_hkt": item["latest_before_1500_at"],
                "cutoff_temperature_c": item["latest_before_1500_c"],
            }
        )

    network = pd.DataFrame(records)
    if not network.empty:
        network["local_date"] = pd.to_datetime(network["local_date"], errors="coerce")
        network["latest_before_1500_at_hkt"] = pd.to_datetime(
            network["latest_before_1500_at_hkt"], utc=True, errors="coerce"
        ).dt.tz_convert(HKT)
        network.to_parquet(
            SILVER_ROOT / "observations" / "hko_station_temperature_cutoff_summary.parquet",
            index=False,
        )
    return network


def merge_asof_feature(
    cutoffs: pd.DataFrame,
    series: pd.DataFrame,
    variable: str,
    feature_name: str,
    *,
    tolerance_hours: int = 2,
    offset_hours: int = 0,
) -> pd.DataFrame:
    left = cutoffs[["local_date", "cutoff_hkt"]].copy()
    left["join_time"] = left["cutoff_hkt"] - pd.to_timedelta(offset_hours, unit="h")
    right = series[
        (series["station"].isin(HQ_STATIONS)) & (series["variable"] == variable)
    ][["observed_at_hkt", "available_at_hkt", "value"]].copy()
    if right.empty:
        left[feature_name] = pd.NA
        left[f"{feature_name}_observed_at_hkt"] = pd.NaT
        left[f"{feature_name}_available_at_hkt"] = pd.NaT
        return left.drop(columns=["join_time", "cutoff_hkt"])
    left = left.sort_values("join_time")
    right = right.sort_values("observed_at_hkt")
    if offset_hours == 0:
        right = right.sort_values("available_at_hkt")
        merged = pd.merge_asof(
            left,
            right,
            left_on="cutoff_hkt",
            right_on="available_at_hkt",
            direction="backward",
            tolerance=pd.Timedelta(hours=tolerance_hours),
        )
    else:
        merged = pd.merge_asof(
            left,
            right,
            left_on="join_time",
            right_on="observed_at_hkt",
            direction="backward",
            tolerance=pd.Timedelta(hours=tolerance_hours),
        )
        unavailable = merged["available_at_hkt"] > merged["cutoff_hkt"]
        merged.loc[unavailable, ["observed_at_hkt", "available_at_hkt", "value"]] = pd.NA
    merged = merged.rename(
        columns={
            "value": feature_name,
            "observed_at_hkt": f"{feature_name}_observed_at_hkt",
            "available_at_hkt": f"{feature_name}_available_at_hkt",
        }
    )
    return merged.drop(columns=["join_time", "cutoff_hkt"])


def build_feature_candidates(target: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    target_dates = target[["local_date", "target_tmax_c"]].copy()
    target_dates["cutoff_hkt"] = [
        hko_tminus1_15_cutoff(pd.Timestamp(day).to_pydatetime().replace(tzinfo=HKT))
        for day in target_dates["local_date"]
    ]
    target_dates["cutoff_hkt"] = pd.to_datetime(target_dates["cutoff_hkt"], utc=True).dt.tz_convert(HKT)
    features = target_dates.copy()

    for variable, name in [
        ("air_temperature_c", "hko_temp_at_tminus1_1500_c"),
        ("relative_humidity_pct", "hko_rh_at_tminus1_1500_pct"),
        ("msl_pressure_hpa", "hko_mslp_at_tminus1_1500_hpa"),
        ("temperature_since_midnight_max_c", "hko_tminus1_max_so_far_1500_c"),
        ("temperature_since_midnight_min_c", "hko_tminus1_min_so_far_1500_c"),
    ]:
        addon = merge_asof_feature(features, obs, variable, name)
        features = features.merge(addon, on="local_date", how="left")

    temp_3h = merge_asof_feature(
        target_dates, obs, "air_temperature_c", "hko_temp_tminus1_1200_c", offset_hours=3
    )
    temp_6h = merge_asof_feature(
        target_dates, obs, "air_temperature_c", "hko_temp_tminus1_0900_c", offset_hours=6
    )
    pressure_3h = merge_asof_feature(
        target_dates, obs, "msl_pressure_hpa", "hko_mslp_tminus1_1200_hpa", offset_hours=3
    )
    features = features.merge(temp_3h, on="local_date", how="left")
    features = features.merge(temp_6h, on="local_date", how="left")
    features = features.merge(pressure_3h, on="local_date", how="left")

    features["hko_temp_3h_change_to_cutoff_c"] = (
        features["hko_temp_at_tminus1_1500_c"] - features["hko_temp_tminus1_1200_c"]
    )
    features["hko_temp_6h_change_to_cutoff_c"] = (
        features["hko_temp_at_tminus1_1500_c"] - features["hko_temp_tminus1_0900_c"]
    )
    features["hko_mslp_3h_change_to_cutoff_hpa"] = (
        features["hko_mslp_at_tminus1_1500_hpa"] - features["hko_mslp_tminus1_1200_hpa"]
    )

    previous = target[["local_date", "target_tmax_c"]].copy()
    previous["local_date"] = previous["local_date"] + pd.Timedelta(days=2)
    previous = previous.rename(columns={"target_tmax_c": "hko_tminus2_official_tmax_c"})
    features = features.merge(previous, on="local_date", how="left")

    features["split_role"] = features["local_date"].apply(assign_split)
    features.to_parquet(
        SILVER_ROOT / "features" / "t24_cutoff_feature_candidates.parquet",
        index=False,
    )
    return features


def assign_split(local_date: pd.Timestamp) -> str:
    day = pd.Timestamp(local_date)
    if day < pd.Timestamp("2024-01-01"):
        return "development"
    if day < pd.Timestamp("2025-01-01"):
        return "validation_2024"
    return "locked_test_or_future_holdout"


def build_feature_registry() -> pd.DataFrame:
    rows = [
        {
            "feature_family": "official_daily_hko_tmax",
            "feature_name": "target_tmax_c",
            "role": "TARGET_ONLY",
            "eligible_at_tminus1_1500_hkt": False,
            "target_derived": True,
            "available_at_rule": "after target day completion and publication",
            "notes": "Target label only. Never a predictor.",
        },
        {
            "feature_family": "official_daily_hko_lagged_tmax",
            "feature_name": "hko_tminus2_official_tmax_c",
            "role": "PROXY_WITH_LIMITATIONS",
            "eligible_at_tminus1_1500_hkt": True,
            "target_derived": False,
            "available_at_rule": "requires empirical publication-lag proof before production",
            "notes": "Useful benchmark feature, but publication timing must be proven before production.",
        },
        {
            "feature_family": "hko_high_frequency_hq",
            "feature_name": "hko_temp_at_tminus1_1500_c",
            "role": "OPERATIONAL_WITH_CONSERVATIVE_LATENCY",
            "eligible_at_tminus1_1500_hkt": True,
            "target_derived": False,
            "available_at_rule": "observed_at + 20 minutes",
            "notes": "Use latest observation available by cutoff.",
        },
        {
            "feature_family": "hko_high_frequency_hq",
            "feature_name": "hko_rh_at_tminus1_1500_pct",
            "role": "OPERATIONAL_WITH_CONSERVATIVE_LATENCY",
            "eligible_at_tminus1_1500_hkt": True,
            "target_derived": False,
            "available_at_rule": "observed_at + 20 minutes",
            "notes": "Moisture state at cutoff.",
        },
        {
            "feature_family": "hko_high_frequency_hq",
            "feature_name": "hko_mslp_at_tminus1_1500_hpa",
            "role": "OPERATIONAL_WITH_CONSERVATIVE_LATENCY",
            "eligible_at_tminus1_1500_hkt": True,
            "target_derived": False,
            "available_at_rule": "observed_at + 20 minutes",
            "notes": "Synoptic pressure state at cutoff.",
        },
        {
            "feature_family": "hko_high_frequency_hq",
            "feature_name": "hko_tminus1_max_so_far_1500_c",
            "role": "OPERATIONAL_WITH_CONSERVATIVE_LATENCY",
            "eligible_at_tminus1_1500_hkt": True,
            "target_derived": False,
            "available_at_rule": "observed_at + 20 minutes",
            "notes": "Only valid for T-1, not target day T.",
        },
        {
            "feature_family": "official_daily_climate_same_day",
            "feature_name": "daily_rainfall/cloud/sunshine/etc_for_target_day",
            "role": "RETROSPECTIVE_MECHANISM_ONLY",
            "eligible_at_tminus1_1500_hkt": False,
            "target_derived": False,
            "available_at_rule": "after target day completion and publication",
            "notes": "Allowed for mechanism EDA only. Forbidden as operational T-24 predictors.",
        },
        {
            "feature_family": "current_nwp_only",
            "feature_name": "current_gfs_gefs_cycle_payloads",
            "role": "PROSPECTIVE_ONLY_NOT_YET_BACKTESTABLE",
            "eligible_at_tminus1_1500_hkt": False,
            "target_derived": False,
            "available_at_rule": "only current/prospective cycles acquired",
            "notes": "Needs historical cycle archive before retrospective evaluation.",
        },
        {
            "feature_family": "tc_best_track",
            "feature_name": "retrospective_best_track_intensity_position",
            "role": "RETROSPECTIVE_MECHANISM_ONLY",
            "eligible_at_tminus1_1500_hkt": False,
            "target_derived": False,
            "available_at_rule": "final best track after event",
            "notes": "Use only for mechanism analysis unless advisory vintage archive is built.",
        },
        {
            "feature_family": "static_geospatial",
            "feature_name": "station_distance_bearing_static_context",
            "role": "STATIC_DETERMINISTIC",
            "eligible_at_tminus1_1500_hkt": True,
            "target_derived": False,
            "available_at_rule": "static",
            "notes": "Eligible once station identity/history is resolved.",
        },
    ]
    registry = pd.DataFrame(rows)
    registry.to_csv(METADATA_ROOT / "feature_eligibility_registry.csv", index=False)
    registry.to_parquet(METADATA_ROOT / "feature_eligibility_registry.parquet", index=False)
    return registry


def summarize_range(series: pd.Series) -> str:
    cleaned = series.dropna()
    if cleaned.empty:
        return "none"
    return f"{cleaned.min()} to {cleaned.max()}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._\n"
    subset = df[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for _, row in subset.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def correlation_table(features: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = [
        col
        for col in features.columns
        if col.startswith("hko_")
        and features[col].dtype.kind in {"f", "i"}
        and col != "target_tmax_c"
    ]
    rows: list[dict[str, object]] = []
    for col in candidate_cols:
        for split in ["development", "validation_2024"]:
            subset = features[
                (features["split_role"] == split)
                & features[col].notna()
                & features["target_tmax_c"].notna()
            ]
            corr = subset[col].corr(subset["target_tmax_c"]) if len(subset) >= 30 else None
            rows.append(
                {
                    "feature": col,
                    "split": split,
                    "n": len(subset),
                    "pearson_r_with_target_tmax": round(float(corr), 4)
                    if corr is not None and not math.isnan(float(corr))
                    else None,
                    "role": "eda_only_not_model_selection",
                }
            )
    result = pd.DataFrame(rows).sort_values(["split", "pearson_r_with_target_tmax"], ascending=[True, False])
    result.to_parquet(SILVER_ROOT / "features" / "t24_feature_eda_correlations.parquet", index=False)
    return result


def target_anatomy(target: pd.DataFrame) -> dict[str, object]:
    series = target.dropna(subset=["target_tmax_c"]).copy()
    series["year"] = series["local_date"].dt.year
    series["month"] = series["local_date"].dt.month
    annual = (
        series.groupby("year", as_index=False)
        .agg(
            days=("target_tmax_c", "count"),
            mean_tmax_c=("target_tmax_c", "mean"),
            max_tmax_c=("target_tmax_c", "max"),
            hot_days_33c=("target_tmax_c", lambda value: int((value >= 33).sum())),
            very_hot_days_35c=("target_tmax_c", lambda value: int((value >= 35).sum())),
        )
        .round(3)
    )
    monthly = (
        series.groupby("month", as_index=False)
        .agg(
            days=("target_tmax_c", "count"),
            mean_tmax_c=("target_tmax_c", "mean"),
            p90_tmax_c=("target_tmax_c", lambda value: value.quantile(0.9)),
            p99_tmax_c=("target_tmax_c", lambda value: value.quantile(0.99)),
        )
        .round(3)
    )
    top_days = series.sort_values("target_tmax_c", ascending=False).head(25)[
        ["local_date", "target_tmax_c"]
    ]
    annual.to_parquet(SILVER_ROOT / "targets" / "hko_tmax_annual_anatomy.parquet", index=False)
    monthly.to_parquet(SILVER_ROOT / "targets" / "hko_tmax_monthly_anatomy.parquet", index=False)
    top_days.to_parquet(SILVER_ROOT / "targets" / "hko_tmax_top_days.parquet", index=False)
    return {"annual": annual, "monthly": monthly, "top_days": top_days}


def station_network_analysis(network: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if network.empty:
        empty = pd.DataFrame()
        return {"coverage": empty, "hko_offsets": empty}
    coverage = (
        network.groupby("station", as_index=False)
        .agg(
            first_date=("local_date", "min"),
            last_date=("local_date", "max"),
            days=("local_date", "count"),
            mean_obs_per_day=("cutoff_window_obs_count", "mean"),
            mean_cutoff_temperature_c=("cutoff_temperature_c", "mean"),
        )
        .sort_values(["days", "station"], ascending=[False, True])
        .round(3)
    )

    hko = network[network["station"] == "HK Observatory"][
        ["local_date", "cutoff_temperature_c"]
    ].rename(columns={"cutoff_temperature_c": "hko_cutoff_temperature_c"})
    joined = network.merge(hko, on="local_date", how="left")
    joined["cutoff_temperature_offset_vs_hko_c"] = (
        joined["cutoff_temperature_c"] - joined["hko_cutoff_temperature_c"]
    )
    offsets = (
        joined[joined["station"] != "HK Observatory"]
        .groupby("station", as_index=False)
        .agg(
            overlap_days=("cutoff_temperature_offset_vs_hko_c", "count"),
            mean_offset_c=("cutoff_temperature_offset_vs_hko_c", "mean"),
            p10_offset_c=("cutoff_temperature_offset_vs_hko_c", lambda value: value.quantile(0.1)),
            p90_offset_c=("cutoff_temperature_offset_vs_hko_c", lambda value: value.quantile(0.9)),
        )
        .sort_values("mean_offset_c", ascending=False)
        .round(3)
    )
    coverage.to_parquet(
        SILVER_ROOT / "observations" / "hko_station_temperature_network_coverage.parquet",
        index=False,
    )
    offsets.to_parquet(
        SILVER_ROOT / "observations" / "hko_station_temperature_offsets_vs_hko.parquet",
        index=False,
    )
    return {"coverage": coverage, "hko_offsets": offsets}


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def coverage_matrix(
    ledger: pd.DataFrame,
    daily: pd.DataFrame,
    target: pd.DataFrame,
    obs: pd.DataFrame,
    network: pd.DataFrame,
    summaries: list[ParseSummary],
) -> pd.DataFrame:
    source_counts = (
        ledger[ledger["status"] == "success"]
        .groupby("source_id", as_index=False)
        .agg(success_rows=("source_id", "count"), bytes=("content_length", "sum"))
    )
    summary_by_source = {item.source_id: item for item in summaries}
    rows: list[dict[str, object]] = []
    for source_id, label, role in [
        ("hko_daily_climate_maximum_temperature_all", "HKO official daily Tmax target", "TARGET_ONLY"),
        ("hko_daily_climate_mean_temperature_all", "HKO daily mean temperature", "RETROSPECTIVE_MECHANISM_ONLY"),
        ("hko_daily_climate_rainfall_all", "HKO daily rainfall", "RETROSPECTIVE_MECHANISM_ONLY"),
        ("hko_daily_climate_cloud_amount_all", "HKO daily cloud amount", "RETROSPECTIVE_MECHANISM_ONLY"),
        (
            "datagov_hko_historical_latest_1min_temperature_archive",
            "HKO historical high-frequency temperature",
            "OPERATIONAL_WITH_CONSERVATIVE_LATENCY",
        ),
        (
            "datagov_hko_historical_latest_1min_humidity_archive",
            "HKO historical high-frequency humidity",
            "OPERATIONAL_WITH_CONSERVATIVE_LATENCY",
        ),
        (
            "datagov_hko_historical_latest_1min_pressure_archive",
            "HKO historical high-frequency pressure",
            "OPERATIONAL_WITH_CONSERVATIVE_LATENCY",
        ),
        (
            "datagov_hko_historical_latest_since_midnight_maxmin_archive",
            "HKO historical since-midnight max/min",
            "OPERATIONAL_WITH_CONSERVATIVE_LATENCY",
        ),
        (
            "datagov_hko_historical_latest_10min_wind_archive",
            "HKO historical 10-minute wind network",
            "OPERATIONAL_WITH_CONSERVATIVE_LATENCY",
        ),
        (
            "datagov_hko_historical_latest_1min_solar_archive",
            "HKO historical solar support stations",
            "OPERATIONAL_WITH_CONSERVATIVE_LATENCY",
        ),
    ]:
        counts = source_counts[source_counts["source_id"] == source_id]
        parsed_rows = ""
        observed_range = ""
        if source_id.startswith("hko_daily_climate"):
            subset = daily[daily["source_id"] == source_id]
            parsed_rows = str(len(subset))
            observed_range = summarize_range(subset["local_date"].dt.date.astype(str))
        elif source_id in summary_by_source:
            item = summary_by_source[source_id]
            parsed_rows = str(item.rows)
            observed_range = f"{item.observed_start} to {item.observed_end}" if item.observed_start else "none"
        rows.append(
            {
                "source_id": source_id,
                "dataset": label,
                "raw_status": "downloaded" if not counts.empty else "missing",
                "success_rows": int(counts["success_rows"].iloc[0]) if not counts.empty else 0,
                "parsed_rows": parsed_rows,
                "date_or_observed_range": observed_range,
                "point_in_time_role": role,
            }
        )

    rows.extend(
        [
            {
                "source_id": "derived:hko_daily_tmax",
                "dataset": "Silver official HKO Tmax target table",
                "raw_status": "derived",
                "success_rows": "",
                "parsed_rows": len(target),
                "date_or_observed_range": summarize_range(target["local_date"].dt.date.astype(str)),
                "point_in_time_role": "TARGET_ONLY",
            },
            {
                "source_id": "derived:t24_cutoff_feature_candidates",
                "dataset": "Leakage-screened T-24 candidate feature table",
                "raw_status": "derived",
                "success_rows": "",
                "parsed_rows": "",
                "date_or_observed_range": "",
                "point_in_time_role": "OPERATIONAL_WITH_CONSERVATIVE_LATENCY plus documented proxies",
            },
            {
                "source_id": "derived:hko_station_temperature_cutoff_summary",
                "dataset": "All-station 15:00 cutoff temperature summaries from HKO minute archive",
                "raw_status": "derived",
                "success_rows": "",
                "parsed_rows": len(network),
                "date_or_observed_range": summarize_range(network["local_date"].dt.date.astype(str))
                if not network.empty
                else "none",
                "point_in_time_role": "EDA and station-network context",
            },
        ]
    )
    matrix = pd.DataFrame(rows).astype(str)
    matrix.to_parquet(REPORTS_ROOT / "COVERAGE_MATRIX.parquet", index=False)
    write(REPORTS_ROOT / "COVERAGE_MATRIX.md", "# Coverage Matrix\n\n" + markdown_table(matrix, list(matrix.columns)))
    return matrix


def source_timestamp_contracts() -> str:
    return f"""# Source Timestamp Contracts

Primary operational target: forecast Hong Kong Observatory daily maximum temperature for target date T.

Primary cutoff: `{PRIMARY_CUTOFF}`. A feature is usable only if the feature value and all source information needed to compute it are available no later than this cutoff.

| Source family | Source time | Issue time | Valid time | Available-at contract | T-24 role |
|---|---|---|---|---|---|
| HKO official daily Tmax | Local calendar day T | Publication after observation day | Day T | after target day completion/publication | TARGET_ONLY |
| HKO other official daily climate | Local calendar day T | Publication after observation day | Day T | after target day completion/publication | RETROSPECTIVE_MECHANISM_ONLY for same-day target analysis |
| HKO high-frequency station observations | Observation timestamp in HKT | Same feed publication event | Observation instant | observed_at + 20 minutes unless live `retrieved_at` proves earlier/later | OPERATIONAL_WITH_CONSERVATIVE_LATENCY |
| HKO since-midnight max/min | Observation timestamp in HKT | Same feed publication event | Partial day up to observation time | observed_at + 20 minutes; only T-1 or earlier is allowed at this cutoff | OPERATIONAL_WITH_CONSERVATIVE_LATENCY |
| HKO forecast/warning JSON/RSS/image current feeds | Provider issue timestamp where present, otherwise retrieval timestamp | Feed issue/retrieval | Forecast/warning validity fields | retrieved_at or parsed issue availability | OPERATIONAL_POINT_IN_TIME for archived vintages only |
| Current-only NWP subsets | Model cycle time | Model cycle issuance | Forecast lead valid time | cycle-specific release lag must be proven | PROSPECTIVE_ONLY_NOT_YET_BACKTESTABLE until historical cycles exist |
| Reanalysis/final gridded products | Analysis valid time | Final product release | Historical valid time | release lag after valid time | RETROSPECTIVE_MECHANISM_ONLY unless release lag makes it eligible |
| Static geospatial context | Static | Static | Static | always available after station metadata freeze | STATIC_DETERMINISTIC |
"""


def point_in_time_eligibility_report(registry: pd.DataFrame) -> str:
    accepted = registry[registry["eligible_at_tminus1_1500_hkt"] == True]  # noqa: E712
    rejected = registry[registry["eligible_at_tminus1_1500_hkt"] == False]  # noqa: E712
    return (
        "# Point-In-Time Eligibility\n\n"
        f"Primary cutoff: `{PRIMARY_CUTOFF}`.\n\n"
        "The registry below is generated from `metadata/feature_eligibility_registry.*`. "
        "Rows marked target-only, retrospective, or prospective-only are forbidden as operational predictors.\n\n"
        "## Eligible Or Conditionally Eligible\n\n"
        + markdown_table(
            accepted,
            [
                "feature_family",
                "feature_name",
                "role",
                "eligible_at_tminus1_1500_hkt",
                "available_at_rule",
                "notes",
            ],
        )
        + "\n## Rejected For Operational T-24 Features\n\n"
        + markdown_table(
            rejected,
            [
                "feature_family",
                "feature_name",
                "role",
                "eligible_at_tminus1_1500_hkt",
                "available_at_rule",
                "notes",
            ],
        )
    )


def data_readiness_report(
    ledger: pd.DataFrame,
    daily: pd.DataFrame,
    target: pd.DataFrame,
    obs: pd.DataFrame,
    network: pd.DataFrame,
    features: pd.DataFrame,
) -> str:
    success_count = int((ledger["status"] == "success").sum())
    failed_count = int((ledger["status"] != "success").sum())
    high_freq_min = obs["observed_at_hkt"].min() if not obs.empty else "none"
    high_freq_max = obs["observed_at_hkt"].max() if not obs.empty else "none"
    usable_features = features.dropna(subset=["hko_temp_at_tminus1_1500_c"])
    return f"""# Data Readiness

## Gate Status

Status: `PARTIAL_PASS_FOR_PHASE_A_B_EDA`.

This repository can now parse the acquired raw archive into reproducible analysis tables and produce leakage-screened candidate features for the primary HKO T-24 cutoff. It is not yet cleared for final production modelling because several source families remain current-only, credential-gated, historically unavailable, or missing empirical publication-lag proof.

## Parsed Evidence

| Item | Value |
|---|---:|
| Retrieval ledger rows | {len(ledger):,} |
| Successful retrieval rows | {success_count:,} |
| Non-success retrieval rows | {failed_count:,} |
| Parsed HKO daily climate rows | {len(daily):,} |
| Official HKO Tmax target rows | {len(target):,} |
| HKO target date range | {summarize_range(target["local_date"].dt.date.astype(str))} |
| Selected high-frequency observation rows parsed | {len(obs):,} |
| Selected high-frequency observed range | {high_freq_min} to {high_freq_max} |
| Station-network cutoff summary rows | {len(network):,} |
| T-24 candidate feature rows with HKO cutoff temperature | {len(usable_features):,} |

## Hard Gates

- No target-day values are used as T-24 predictors.
- Reanalysis/final products remain retrospective-only unless release lag is explicitly proven.
- Current-only NWP is not backtestable and is rejected for retrospective model fitting.
- Official same-day daily climate rows are target/mechanism labels, not operational predictors.
- The main usable operational archive for initial analysis is HKO high-frequency station observations from 2020/2021 onward.
"""


def quality_report(daily: pd.DataFrame, target: pd.DataFrame, obs: pd.DataFrame, features: pd.DataFrame) -> str:
    daily_missing = (
        daily.groupby(["source_id", "variable"], as_index=False)
        .agg(rows=("value", "size"), missing_values=("value", lambda value: int(value.isna().sum())))
        .sort_values(["missing_values", "source_id"], ascending=[False, True])
        .head(25)
    )
    obs_missing = pd.DataFrame()
    if not obs.empty:
        obs_missing = (
            obs.groupby(["source_id", "variable"], as_index=False)
            .agg(rows=("value", "size"), min_value=("value", "min"), max_value=("value", "max"))
            .round(3)
        )
    gaps = features[
        ["local_date", "hko_temp_at_tminus1_1500_c", "hko_rh_at_tminus1_1500_pct", "hko_mslp_at_tminus1_1500_hpa"]
    ].isna().sum()
    return (
        "# Data Quality And Anomalies\n\n"
        "## Daily Climate Missingness Top 25\n\n"
        + markdown_table(daily_missing, list(daily_missing.columns))
        + "\n## High-Frequency Parsed Value Ranges\n\n"
        + markdown_table(obs_missing, list(obs_missing.columns))
        + "\n## T-24 Feature Missingness\n\n"
        + markdown_table(gaps.reset_index().rename(columns={"index": "field", 0: "missing_rows"}), ["field", "missing_rows"])
        + "\n## Current QC Interpretation\n\n"
        "- HKO daily Tmax target has the longest official record and is fit for target anatomy.\n"
        "- High-frequency HKO station features are the strongest operationally aligned source but only cover 2020/2021 onward in the public archive acquired so far.\n"
        "- Same-day daily climate values are not usable as T-24 predictors; they are retained for mechanism analysis only.\n"
    )


def station_history_dossier(daily: pd.DataFrame, network_stats: dict[str, pd.DataFrame]) -> str:
    station_daily = (
        daily.groupby("station_or_domain", as_index=False)
        .agg(first_date=("local_date", "min"), last_date=("local_date", "max"), rows=("local_date", "count"))
        .sort_values(["rows", "station_or_domain"], ascending=[False, True])
    )
    coverage = network_stats["coverage"]
    offsets = network_stats["hko_offsets"]
    return (
        "# Station History Dossier\n\n"
        "## Official Daily Climate Domains\n\n"
        + markdown_table(station_daily, ["station_or_domain", "first_date", "last_date", "rows"])
        + "\n## High-Frequency Temperature Station Coverage\n\n"
        + markdown_table(coverage.head(50), list(coverage.columns) if not coverage.empty else [])
        + "\n## Surrounding Station Daily-Max Offset Versus HKO\n\n"
        + markdown_table(offsets.head(50), list(offsets.columns) if not offsets.empty else [])
        + "\n## Interpretation\n\n"
        "- HKO official daily target history reaches back to the nineteenth century.\n"
        "- Public high-frequency station archives currently start in 2020/2021, so modern station-network effects are available only for the recent era.\n"
        "- Surrounding station offsets should be used as physically interpretable context and candidate features only when their timestamps are eligible at the cutoff.\n"
    )


def eda_master_report(corr: pd.DataFrame) -> str:
    top_dev = corr[corr["split"] == "development"].head(20)
    top_val = corr[corr["split"] == "validation_2024"].head(20)
    return (
        "# EDA Master Report\n\n"
        "This report intentionally stops before model fitting. Correlations are screening evidence, not tuned performance claims.\n\n"
        "## Strongest Development-Period Candidate Signals\n\n"
        + markdown_table(top_dev, list(top_dev.columns))
        + "\n## 2024 Validation Stability Check\n\n"
        + markdown_table(top_val, list(top_val.columns))
        + "\n## High-Value Hypotheses\n\n"
        "- T-1 afternoon HKO temperature carries direct persistence information for next-day Tmax.\n"
        "- T-1 since-midnight max/min separates hot-airmass persistence from transient afternoon spikes.\n"
        "- Pressure tendency and humidity at cutoff may help identify synoptic regime and overnight cooling potential.\n"
        "- Station-network offsets can expose sea-breeze penetration, urban heat storage, and northwestern New Territories heating regimes.\n"
    )


def target_anatomy_report(anatomy: dict[str, object]) -> str:
    annual = anatomy["annual"]
    monthly = anatomy["monthly"]
    top_days = anatomy["top_days"]
    assert isinstance(annual, pd.DataFrame)
    assert isinstance(monthly, pd.DataFrame)
    assert isinstance(top_days, pd.DataFrame)
    return (
        "# Target Anatomy\n\n"
        "## Monthly Distribution\n\n"
        + markdown_table(monthly, list(monthly.columns))
        + "\n## Hottest Official Tmax Days\n\n"
        + markdown_table(top_days, list(top_days.columns))
        + "\n## Annual Tail Snapshot: Last 20 Complete/Partial Years\n\n"
        + markdown_table(annual.tail(20), list(annual.columns))
        + "\n## Interpretation\n\n"
        "- The target is strongly seasonal, so all evaluation must stratify by month/season and hot-tail regimes.\n"
        "- Long history is excellent for climatology, but modern high-frequency operational predictors only start in 2020/2021 in the acquired public archive.\n"
    )


def evaluation_design_report() -> str:
    return f"""# Evaluation Design

Primary forecast target: HKO daily maximum temperature for local date T.

Primary cutoff: `{PRIMARY_CUTOFF}`.

## Split Policy

| Split | Dates | Purpose |
|---|---|---|
| Development | up to 2023-12-31 | EDA, feature screening, baseline design |
| Validation 2024 | 2024-01-01 to 2024-12-31 | Stability checks and tuning guardrail |
| Locked holdout | 2025-01-01 onward | Do not use for creative iteration until formal experiment protocol is frozen |

## Rules

- Freeze target definition and timestamp contracts before fitting any model.
- Fit baselines before complex model families.
- Report persistence, climatology, seasonal climatology, and simple physically motivated baselines before ML.
- Keep Polymarket/backtesting fully out of scope.
- Treat correlations in `EDA_MASTER_REPORT.md` as hypothesis generation only.
"""


def feature_hypotheses_report(corr: pd.DataFrame) -> str:
    interesting = corr[corr["split"] == "development"].head(15)
    return (
        "# Feature Ideas And Hypotheses\n\n"
        "## Candidate Signals From Current Parsed Archive\n\n"
        + markdown_table(interesting, list(interesting.columns))
        + "\n## Next Feature Families To Engineer\n\n"
        "- Diurnal shape before cutoff: morning heating slope, noon-to-15:00 acceleration, and overnight minimum recovery.\n"
        "- Urban versus marine contrast: HKO minus King's Park, HKO minus Waglan Island, Chek Lap Kok minus HKO.\n"
        "- Moisture/heat-index regime: humidity-conditioned persistence and dew-point depression proxies.\n"
        "- Synoptic pressure tendency: pressure fall/rise over 3h, 6h, 12h before cutoff.\n"
        "- Radiation/cloud mechanism proxies from King's Park solar and HKO daily cloud/sunshine labels, with strict timestamp separation.\n"
        "- TC/advection flags from advisory vintages only after historical/live pair contract is complete.\n"
    )


def negative_results_report() -> str:
    return """# Negative And Null Results

This file records constraints and rejected shortcuts discovered during Phase A/B.

| Finding | Status | Reason |
|---|---|---|
| Same-day daily climate labels as predictors | rejected | They are observed after/during target day and leak target-day information at T-1 15:00. |
| Reanalysis fields as operational features | rejected for now | Final products have release lag and are retrospective unless product-specific availability is proven. |
| Current-only NWP snapshots for backtest | rejected for now | They are prospective from acquisition start only and cannot support retrospective performance claims. |
| Official daily Tmax as feature | rejected | It is the target. |
| Public high-frequency HKO archive before 2020/2021 | unavailable in acquired archive | Existing public DATA.GOV historical ZIPs begin in 2020/2021 for the station feeds parsed here. |
"""


def initial_domain_stub(title: str, evidence: str, blockers: str) -> str:
    return f"""# {title}

## Current Parsed Evidence

{evidence}

## Blockers Or Limits

{blockers}

## Next Work

- Add source-specific parsers only when timestamp contracts and lawful availability are clear.
- Keep all transformed rows tagged with source time, available-at, content hash, and operational role.
"""


def write_finding(name: str, title: str, content: str) -> None:
    folder = FINDINGS_ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "README.md", f"# {title}\n\n{content}")


def write_findings(anatomy: dict[str, object], corr: pd.DataFrame, network_stats: dict[str, pd.DataFrame]) -> None:
    monthly = anatomy["monthly"]
    annual = anatomy["annual"]
    offsets = network_stats["hko_offsets"]
    assert isinstance(monthly, pd.DataFrame)
    assert isinstance(annual, pd.DataFrame)
    hottest_month = monthly.sort_values("mean_tmax_c", ascending=False).head(1)
    recent = annual.tail(30)
    write_finding(
        "target_long_term_warming",
        "Target Long-Term Warming",
        "Evidence table: `C:\\hkg_tmax_data\\silver\\targets\\hko_tmax_annual_anatomy.parquet`.\n\n"
        + markdown_table(recent, list(recent.columns))
        + "\nInterpretation: use the long official target history for climatology and drift diagnostics, but do not mix nineteenth-century target behavior with modern high-frequency feature availability without explicit regime handling.\n",
    )
    write_finding(
        "seasonality_and_tail_risk",
        "Seasonality And Tail Risk",
        "The hottest target months by mean Tmax are:\n\n"
        + markdown_table(hottest_month, list(hottest_month.columns))
        + "\nOperational evaluation must stratify summer/hot-tail days separately from mild-season days.\n",
    )
    top_corr = corr[corr["split"] == "development"].head(10)
    write_finding(
        "tminus1_1500_station_state",
        "T-1 15:00 Station State",
        "Development-period correlation screening for cutoff-safe HKO station-state features:\n\n"
        + markdown_table(top_corr, list(top_corr.columns))
        + "\nThis is EDA only, not model selection. The target-day label is not used as a feature.\n",
    )
    write_finding(
        "surrounding_station_offsets",
        "Surrounding Station Offsets",
        "15:00 cutoff temperature offset evidence from the all-station high-frequency temperature archive:\n\n"
        + markdown_table(offsets.head(25), list(offsets.columns) if not offsets.empty else [])
        + "\nThese offsets are candidate indicators for sea-breeze, urban, elevation, and inland heating regimes.\n",
    )
    pressure_humidity = corr[
        corr["feature"].isin(["hko_rh_at_tminus1_1500_pct", "hko_mslp_at_tminus1_1500_hpa", "hko_mslp_3h_change_to_cutoff_hpa"])
    ]
    write_finding(
        "pressure_humidity_cutoff_state",
        "Pressure And Humidity Cutoff State",
        "Moisture and pressure screening evidence:\n\n"
        + markdown_table(pressure_humidity, list(pressure_humidity.columns))
        + "\nThese variables should be treated as regime modifiers, not standalone answers.\n",
    )
    write_finding(
        "data_gap_limits",
        "Data Gap Limits",
        "The core operational high-frequency archive starts in 2020/2021 in the acquired public DATA.GOV historical ZIPs. "
        "Long target history reaches back much further, but long history alone cannot create point-in-time high-frequency features. "
        "This is a hard separation between target climatology and operational feature backtesting.\n",
    )


def validate_feature_rows(features: pd.DataFrame) -> None:
    sample = features.dropna(subset=["hko_temp_at_tminus1_1500_c"]).head(100)
    for _, row in sample.iterrows():
        validate_point_in_time_rows(
            [
                {
                    "role": "OPERATIONAL_WITH_CONSERVATIVE_LATENCY",
                    "available_at": row["hko_temp_at_tminus1_1500_c_available_at_hkt"],
                    "target_derived": "false",
                }
            ],
            cutoff_hkt=pd.Timestamp(row["cutoff_hkt"]).to_pydatetime(),
        )


def main() -> None:
    ensure_dirs()
    ledger = read_ledger()
    daily, target = parse_daily_climate(ledger)
    obs, summaries = parse_selected_high_frequency(ledger)
    network = parse_temperature_network_cutoff_summary(ledger)
    features = build_feature_candidates(target, obs)
    registry = build_feature_registry()
    validate_feature_rows(features)

    anatomy = target_anatomy(target)
    network_stats = station_network_analysis(network)
    corr = correlation_table(features)
    matrix = coverage_matrix(ledger, daily, target, obs, network, summaries)

    write(REPORTS_ROOT / "DATA_READINESS.md", data_readiness_report(ledger, daily, target, obs, network, features))
    write(REPORTS_ROOT / "SOURCE_TIMESTAMP_CONTRACTS.md", source_timestamp_contracts())
    write(REPORTS_ROOT / "POINT_IN_TIME_ELIGIBILITY.md", point_in_time_eligibility_report(registry))
    write(REPORTS_ROOT / "STATION_HISTORY_DOSSIER.md", station_history_dossier(daily, network_stats))
    write(REPORTS_ROOT / "DATA_QUALITY_AND_ANOMALIES.md", quality_report(daily, target, obs, features))
    write(REPORTS_ROOT / "EDA_MASTER_REPORT.md", eda_master_report(corr))
    write(REPORTS_ROOT / "TARGET_ANATOMY.md", target_anatomy_report(anatomy))
    write(REPORTS_ROOT / "STATION_NETWORK_ANALYSIS.md", station_history_dossier(daily, network_stats))
    write(REPORTS_ROOT / "EVALUATION_DESIGN.md", evaluation_design_report())
    write(REPORTS_ROOT / "FEATURE_IDEAS_AND_HYPOTHESES.md", feature_hypotheses_report(corr))
    write(REPORTS_ROOT / "NEGATIVE_AND_NULL_RESULTS.md", negative_results_report())
    write(
        REPORTS_ROOT / "UPPER_AIR_AND_BOUNDARY_LAYER.md",
        initial_domain_stub(
            "Upper Air And Boundary Layer",
            "IGRA and upper-air raw archives are acquired, but this Phase A/B script does not yet parse them into cutoff-safe feature tables.",
            "Upper-air release timing and target cutoff eligibility must be proven before use in operational experiments.",
        ),
    )
    write(
        REPORTS_ROOT / "RADIATION_CLOUD_RAIN.md",
        initial_domain_stub(
            "Radiation Cloud Rain",
            "Daily HKO cloud, rainfall, sunshine, and solar labels are parsed for mechanism analysis. High-frequency solar support-station archives are parsed for selected stations.",
            "Same-day daily climate labels leak target-day information at T-1 15:00 and are not operational predictors.",
        ),
    )
    write(
        REPORTS_ROOT / "WIND_MARINE_TOPOGRAPHY.md",
        initial_domain_stub(
            "Wind Marine Topography",
            "Waglan and network wind/sea-temperature official daily series are parsed as retrospective mechanism data. Selected high-frequency wind rows are parsed where public archives exist.",
            "Historical operational advisories and marine products need per-source issue/available-at contracts before model use.",
        ),
    )
    write(
        REPORTS_ROOT / "SYNOPTIC_REGIMES.md",
        initial_domain_stub(
            "Synoptic Regimes",
            "Pressure and pressure-tendency candidate features are built from HKO high-frequency station observations.",
            "NWP and reanalysis regime products remain current-only, credential-gated, or retrospective until historical cycle availability is solved.",
        ),
    )
    write(
        REPORTS_ROOT / "FORECAST_ERROR_ANATOMY.md",
        initial_domain_stub(
            "Forecast Error Anatomy",
            "No forecast model or forecast-error analysis has been run in this phase.",
            "This report intentionally remains a placeholder until baselines are frozen and fit under a separate experiment protocol.",
        ),
    )

    write_findings(anatomy, corr, network_stats)

    status = f"""# HKG Tmax T-24 Analysis Status

Generated by `scripts/build_hkg_tmax_phase_ab_analysis.py`.

| Artifact | Rows |
|---|---:|
| Coverage matrix rows | {len(matrix):,} |
| Daily climate bronze rows | {len(daily):,} |
| Target silver rows | {len(target):,} |
| Selected high-frequency bronze rows | {len(obs):,} |
| Station-network cutoff summary rows | {len(network):,} |
| T-24 feature candidate rows | {len(features):,} |
| Feature correlation rows | {len(corr):,} |

Current gate: Phase A/B documentation and leakage-screened EDA are generated. Baseline fitting and ML are not started.
"""
    write(ANALYSIS_ROOT / "STATUS.md", status)
    print(status)


if __name__ == "__main__":
    main()
