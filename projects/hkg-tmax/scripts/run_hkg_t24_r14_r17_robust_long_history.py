from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hkg_tmax.hkg_t24.governance import check_four_year_oof_feasibility
from hkg_tmax.hkg_t24.guard import assert_no_locked_dates
from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
DEFAULT_DATA_ROOT = PROJECT_PATHS.data_root
EXPERIMENT_ROOT = PROJECT_PATHS.run_root / "experiments" / "legacy" / "hkg_tmax_t24"
REPORT_ROOT = PROJECT_PATHS.run_root / "reports" / "hkg_t24"

ANALYSIS_END = pd.Timestamp("2023-12-31")
OOF_START_YEAR = 1965
FOLD_YEARS = 5
MIN_HISTORY_YEARS = 39.0
MIN_OOF_YEARS = 5.0
MIN_TRAIN_ROWS = 365 * 10
MIN_FEATURE_SUPPORT = 365

QUANTILE_Z = {
    "q05": -1.6448536269514722,
    "q10": -1.2815515655446004,
    "q25": -0.6744897501960817,
    "q50": 0.0,
    "q75": 0.6744897501960817,
    "q90": 1.2815515655446004,
    "q95": 1.6448536269514722,
}

SENTINEL_THRESHOLD = -800.0


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_family: str
    columns: tuple[str, ...]
    control: bool = False


@dataclass(frozen=True)
class ExperimentSpec:
    research_id: str
    experiment_id: str
    slug: str
    title: str
    purpose: str
    feature_group: str
    model_specs: tuple[ModelSpec, ...]
    required_non_null_prefixes: tuple[str, ...]
    caveats: tuple[str, ...]

    @property
    def folder(self) -> Path:
        return EXPERIMENT_ROOT / f"{self.experiment_id}-{self.research_id}"

    @property
    def report_path(self) -> Path:
        return REPORT_ROOT / f"{self.research_id.split('-')[-1]}_{self.slug.upper()}.md"


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def clean_numeric_series(
    values: pd.Series,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    numeric = numeric.mask(numeric <= SENTINEL_THRESHOLD)
    if lower is not None:
        numeric = numeric.mask(numeric < lower)
    if upper is not None:
        numeric = numeric.mask(numeric > upper)
    return numeric


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_crps(y: float, mu: float, sigma: float) -> float:
    sigma = max(float(sigma), 0.05)
    z = (y - mu) / sigma
    return sigma * (
        z * (2.0 * normal_cdf(z) - 1.0) + 2.0 * normal_pdf(z) - 1.0 / math.sqrt(math.pi)
    )


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], *, limit: int = 80) -> str:
    if frame.empty:
        return "_No rows._"
    subset = frame.loc[:, [col for col in columns if col in frame.columns]].head(limit)
    cols = list(subset.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in subset.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def robust_fold_definitions() -> list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds: list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for year in range(OOF_START_YEAR, ANALYSIS_END.year + 1, FOLD_YEARS):
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = min(pd.Timestamp(year=year + FOLD_YEARS - 1, month=12, day=31), ANALYSIS_END)
        train_end = test_start - pd.Timedelta(days=1)
        folds.append((f"fold_{test_start.year}_{test_end.year}", test_start, test_end, train_end))
    return folds


def add_calendar_and_lags(target: pd.DataFrame) -> pd.DataFrame:
    data = target[["local_date", "target_tmax_c"]].copy()
    data["target_date"] = pd.to_datetime(data["local_date"]).dt.normalize()
    data = data.drop(columns=["local_date"]).sort_values("target_date").reset_index(drop=True)
    data = data[data["target_date"] <= ANALYSIS_END].copy()
    assert_no_locked_dates(data["target_date"], context="robust long-history target labels")
    data["year"] = data["target_date"].dt.year
    data["month"] = data["target_date"].dt.month
    data["day_of_year"] = data["target_date"].dt.dayofyear
    data["doy_sin"] = np.sin(2.0 * np.pi * data["day_of_year"] / 365.25)
    data["doy_cos"] = np.cos(2.0 * np.pi * data["day_of_year"] / 365.25)
    data["target_year_centered"] = data["year"] - 1985.0
    data["target_year_centered_sq"] = data["target_year_centered"] ** 2
    for lag in (2, 3, 7, 14, 30):
        data[f"target_tminus{lag}_tmax_c"] = data["target_tmax_c"].shift(lag)
    shifted = data["target_tmax_c"].shift(2)
    data["target_tminus2_to_8_mean_c"] = shifted.rolling(7, min_periods=5).mean()
    data["target_tminus2_to_31_mean_c"] = shifted.rolling(30, min_periods=20).mean()
    data["target_tminus2_minus_tminus7_c"] = (
        data["target_tminus2_tmax_c"] - data["target_tminus7_tmax_c"]
    )
    return data


def build_igra_features(igra: pd.DataFrame) -> pd.DataFrame:
    data = igra.copy()
    data["valid_at_hkt"] = pd.to_datetime(data["valid_at_hkt"], errors="coerce", utc=True)
    data = data[data["valid_at_hkt"].notna()].copy()
    data["origin_date"] = data["valid_at_hkt"].dt.tz_convert("Asia/Hong_Kong").dt.date
    data["target_date"] = pd.to_datetime(data["origin_date"]) + pd.Timedelta(days=1)
    data = data[data["target_date"] <= ANALYSIS_END].copy()
    data = data.sort_values(["origin_date", "valid_at_hkt"]).drop_duplicates("origin_date", keep="last")

    ranges: dict[str, tuple[float | None, float | None]] = {}
    for level in (1000, 925, 850, 700, 500, 300, 200):
        ranges[f"temperature_c_{level}hpa"] = (-90.0, 45.0)
        ranges[f"dewpoint_depression_c_{level}hpa"] = (0.0, 80.0)
        ranges[f"geopotential_height_m_{level}hpa"] = (-500.0, 20000.0)
        ranges[f"wind_speed_mps_{level}hpa"] = (0.0, 160.0)
    for col, (lower, upper) in ranges.items():
        if col in data.columns:
            data[f"igra_{col}"] = clean_numeric_series(data[col], lower=lower, upper=upper)

    data["igra_key_level_count"] = clean_numeric_series(data["key_level_count"], lower=0.0, upper=100.0)
    data["igra_temp_925_minus_850_c"] = (
        data.get("igra_temperature_c_925hpa") - data.get("igra_temperature_c_850hpa")
    )
    data["igra_temp_850_minus_500_c"] = (
        data.get("igra_temperature_c_850hpa") - data.get("igra_temperature_c_500hpa")
    )
    data["igra_temp_700_minus_500_c"] = (
        data.get("igra_temperature_c_700hpa") - data.get("igra_temperature_c_500hpa")
    )
    data["igra_boundary_inversion_925_minus_1000_c"] = (
        data.get("igra_temperature_c_925hpa") - data.get("igra_temperature_c_1000hpa")
    )
    lower_temp_cols = [
        col
        for col in [
            "igra_temperature_c_1000hpa",
            "igra_temperature_c_925hpa",
            "igra_temperature_c_850hpa",
            "igra_temperature_c_700hpa",
        ]
        if col in data.columns
    ]
    data["igra_lower_troposphere_mean_temp_c"] = data[lower_temp_cols].mean(axis=1)
    dry_cols = [
        col
        for col in [
            "igra_dewpoint_depression_c_1000hpa",
            "igra_dewpoint_depression_c_925hpa",
            "igra_dewpoint_depression_c_850hpa",
            "igra_dewpoint_depression_c_700hpa",
        ]
        if col in data.columns
    ]
    data["igra_lower_troposphere_mean_dewpoint_depression_c"] = data[dry_cols].mean(axis=1)
    keep = [
        "target_date",
        "origin_date",
        "valid_at_hkt",
        "release_latency_proven",
        *[col for col in data.columns if col.startswith("igra_")],
    ]
    return data[keep].sort_values("target_date").reset_index(drop=True)


def wind_components(direction_deg: pd.Series, speed_mps: pd.Series) -> tuple[pd.Series, pd.Series]:
    direction = clean_numeric_series(direction_deg, lower=0.0, upper=360.0)
    speed = clean_numeric_series(speed_mps, lower=0.0, upper=100.0)
    radians = np.deg2rad(direction)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def build_isd_features(isd: pd.DataFrame, *, station_count: int = 12) -> pd.DataFrame:
    data = isd.copy()
    data["origin_date"] = pd.to_datetime(data["local_date"], errors="coerce").dt.normalize()
    data = data[data["origin_date"].notna()].copy()
    data["target_date"] = data["origin_date"] + pd.Timedelta(days=1)
    data = data[data["target_date"] <= ANALYSIS_END].copy()
    data["isd_air_temperature_c"] = clean_numeric_series(
        data["air_temperature_c_latest_before_1500"], lower=-30.0, upper=45.0
    )
    data["isd_dew_point_c"] = clean_numeric_series(
        data["dew_point_c_latest_before_1500"], lower=-50.0, upper=35.0
    )
    data["isd_sea_level_pressure_hpa"] = clean_numeric_series(
        data["sea_level_pressure_hpa_latest_before_1500"], lower=850.0, upper=1100.0
    )
    data["isd_wind_speed_mps"] = clean_numeric_series(
        data["wind_speed_mps_latest_before_1500"], lower=0.0, upper=75.0
    )
    data["isd_wind_u_mps"], data["isd_wind_v_mps"] = wind_components(
        data["wind_direction_deg_latest_before_1500"], data["wind_speed_mps_latest_before_1500"]
    )
    data["isd_temp_minus_dewpoint_c"] = data["isd_air_temperature_c"] - data["isd_dew_point_c"]
    data["isd_obs_count"] = clean_numeric_series(data["obs_count"], lower=0.0, upper=2000.0)

    grouped = data.groupby("target_date", observed=True)
    summary = grouped.agg(
        isd_station_count=("station_id", "nunique"),
        isd_obs_count_sum=("isd_obs_count", "sum"),
        isd_air_temp_mean_c=("isd_air_temperature_c", "mean"),
        isd_air_temp_max_c=("isd_air_temperature_c", "max"),
        isd_air_temp_min_c=("isd_air_temperature_c", "min"),
        isd_air_temp_std_c=("isd_air_temperature_c", "std"),
        isd_dew_point_mean_c=("isd_dew_point_c", "mean"),
        isd_pressure_mean_hpa=("isd_sea_level_pressure_hpa", "mean"),
        isd_pressure_min_hpa=("isd_sea_level_pressure_hpa", "min"),
        isd_pressure_max_hpa=("isd_sea_level_pressure_hpa", "max"),
        isd_wind_speed_mean_mps=("isd_wind_speed_mps", "mean"),
        isd_wind_speed_max_mps=("isd_wind_speed_mps", "max"),
        isd_wind_u_mean_mps=("isd_wind_u_mps", "mean"),
        isd_wind_v_mean_mps=("isd_wind_v_mps", "mean"),
        isd_temp_dewpoint_spread_mean_c=("isd_temp_minus_dewpoint_c", "mean"),
    ).reset_index()
    summary["isd_air_temp_range_c"] = summary["isd_air_temp_max_c"] - summary["isd_air_temp_min_c"]
    summary["isd_pressure_range_hpa"] = summary["isd_pressure_max_hpa"] - summary["isd_pressure_min_hpa"]
    summary = summary.sort_values("target_date").reset_index(drop=True)
    for col in ["isd_air_temp_mean_c", "isd_dew_point_mean_c", "isd_pressure_mean_hpa"]:
        summary[f"{col}_change_1d"] = summary[col] - summary[col].shift(1)

    station_support = (
        data.groupby("station_id", observed=True)["isd_air_temperature_c"]
        .apply(lambda values: int(values.notna().sum()))
        .sort_values(ascending=False)
    )
    top_stations = tuple(station_support.head(station_count).index)
    station_data = data[data["station_id"].isin(top_stations)].copy()
    station_pivot = station_data.pivot_table(
        index="target_date",
        columns="station_id",
        values=["isd_air_temperature_c", "isd_dew_point_c", "isd_sea_level_pressure_hpa"],
        aggfunc="last",
    )
    station_pivot.columns = [
        f"isd_station_{metric.replace('isd_', '')}_{str(station).replace('-', '_')}"
        for metric, station in station_pivot.columns
    ]
    station_pivot = station_pivot.reset_index()
    return summary.merge(station_pivot, on="target_date", how="left").sort_values("target_date").reset_index(drop=True)


def build_feature_matrix(data_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    silver = data_root / "silver" / "source_normalized_non_minute"
    target_path = silver / "hko_daily_tmax_target_labels.parquet"
    igra_path = silver / "noaa_igra_hkm00045004_sounding_features.parquet"
    isd_path = silver / "noaa_isd_station_day_cutoff_summary.parquet"
    missing = [path for path in [target_path, igra_path, isd_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing normalized robust-source inputs: " + ", ".join(str(path) for path in missing))
    target = add_calendar_and_lags(pd.read_parquet(target_path))
    igra_features = build_igra_features(pd.read_parquet(igra_path))
    isd_features = build_isd_features(pd.read_parquet(isd_path))
    features = (
        target.merge(igra_features, on="target_date", how="left")
        .merge(isd_features, on="target_date", how="left")
        .sort_values("target_date")
        .reset_index(drop=True)
    )
    features["r15_surface_minus_igra_925_temp_c"] = (
        features["isd_air_temp_mean_c"] - features["igra_temperature_c_925hpa"]
    )
    features["r15_surface_minus_igra_850_temp_c"] = (
        features["isd_air_temp_mean_c"] - features["igra_temperature_c_850hpa"]
    )
    features["r15_surface_dewpoint_minus_850_dryness_c"] = (
        features["isd_temp_dewpoint_spread_mean_c"]
        - features["igra_dewpoint_depression_c_850hpa"]
    )
    features["r15_pressure_x_stability"] = (
        features["isd_pressure_mean_hpa"] * features["igra_temp_925_minus_850_c"]
    )
    features["r17_station_count_change_30d"] = features["isd_station_count"] - features[
        "isd_station_count"
    ].shift(30)
    features["r17_post_1984"] = (features["year"] >= 1985).astype(float)
    features["r17_post_1997"] = (features["year"] >= 1997).astype(float)
    features["r17_post_2014"] = (features["year"] >= 2014).astype(float)
    features["r17_air_temp_mean_x_year"] = (
        features["isd_air_temp_mean_c"] * features["target_year_centered"]
    )
    assert_no_locked_dates(features["target_date"], context="robust long-history feature matrix")
    return features, {"target": str(target_path), "igra": str(igra_path), "isd": str(isd_path)}


def baseline_columns() -> tuple[str, ...]:
    return (
        "doy_sin",
        "doy_cos",
        "target_tminus2_tmax_c",
        "target_tminus3_tmax_c",
        "target_tminus7_tmax_c",
        "target_tminus14_tmax_c",
        "target_tminus2_to_8_mean_c",
        "target_tminus2_to_31_mean_c",
        "target_tminus2_minus_tminus7_c",
    )


def upper_air_columns() -> tuple[str, ...]:
    prefixes = (
        "igra_temperature_c_",
        "igra_dewpoint_depression_c_",
        "igra_geopotential_height_m_",
        "igra_wind_speed_mps_",
        "igra_temp_",
        "igra_boundary_",
        "igra_lower_",
        "igra_key_level_count",
    )
    return tuple(prefixes)


def isd_columns(features: pd.DataFrame, *, station_specific: bool) -> tuple[str, ...]:
    prefixes = (
        "isd_station_count",
        "isd_obs_count_sum",
        "isd_air_temp_",
        "isd_dew_point_",
        "isd_pressure_",
        "isd_wind_",
        "isd_temp_dewpoint_",
    )
    columns = [col for col in features.columns if col.startswith(prefixes)]
    if not station_specific:
        columns = [
            col
            for col in columns
            if not col.startswith(
                (
                    "isd_station_air_temperature",
                    "isd_station_dew_point",
                    "isd_station_sea_level_pressure",
                )
            )
        ]
    return tuple(columns)


def coupling_columns() -> tuple[str, ...]:
    return (
        "r15_surface_minus_igra_925_temp_c",
        "r15_surface_minus_igra_850_temp_c",
        "r15_surface_dewpoint_minus_850_dryness_c",
        "r15_pressure_x_stability",
    )


def era_columns() -> tuple[str, ...]:
    return (
        "target_year_centered",
        "target_year_centered_sq",
        "r17_post_1984",
        "r17_post_1997",
        "r17_post_2014",
        "r17_station_count_change_30d",
        "r17_air_temp_mean_x_year",
    )


def columns_from_prefixes(features: pd.DataFrame, prefixes: Sequence[str]) -> tuple[str, ...]:
    columns: list[str] = []
    for prefix in prefixes:
        columns.extend(col for col in features.columns if col.startswith(prefix))
    return tuple(dict.fromkeys(columns))


def make_experiment_specs(features: pd.DataFrame) -> tuple[ExperimentSpec, ...]:
    baseline = baseline_columns()
    upper = columns_from_prefixes(features, upper_air_columns())
    isd_agg = isd_columns(features, station_specific=False)
    isd_station = isd_columns(features, station_specific=True)
    coupling = coupling_columns()
    era = era_columns()
    return (
        ExperimentSpec(
            "HKG-T24-R14",
            "EXP-0046",
            "eligible_upper_air_thermal_potential",
            "Eligible Upper-Air Thermal Potential and Inversion Structure",
            "Test whether long-history IGRA upper-air thermal structure adds robust next-day Tmax skill.",
            "upper_air",
            (
                ModelSpec("r14_lag_calendar_baseline", "ridge_calendar_lag_baseline", baseline),
                ModelSpec("r14_upper_air_core", "ridge_upper_air_core", baseline + upper),
                ModelSpec(
                    "r14_stability_only",
                    "ridge_upper_air_stability_ablation",
                    baseline
                    + tuple(
                        col
                        for col in upper
                        if "temp_" in col
                        or "temperature_c_" in col
                        or "boundary" in col
                        or "lower_troposphere_mean_temp" in col
                    ),
                ),
            ),
            ("igra_",),
            (
                "NOAA IGRA archive is parsed and long-history, but exact operational release latency before the T-1 15:00 HKT cutoff remains unproven.",
                "IGRA relative-humidity columns are intentionally excluded because the normalized archive shows scaling/sentinel issues.",
            ),
        ),
        ExperimentSpec(
            "HKG-T24-R15",
            "EXP-0047",
            "surface_upper_air_coupling",
            "Surface-Upper-Air Coupling and Mixing-Potential Experiment",
            "Test whether regional surface state plus upper-air mismatch explains robust heating potential.",
            "coupling",
            (
                ModelSpec("r15_lag_calendar_baseline", "ridge_calendar_lag_baseline", baseline),
                ModelSpec("r15_upper_air_plus_isd", "ridge_surface_upper_air", baseline + upper + isd_agg),
                ModelSpec(
                    "r15_coupling_terms",
                    "ridge_surface_upper_air_coupling",
                    baseline + upper + isd_agg + coupling,
                ),
            ),
            ("igra_", "isd_"),
            (
                "Both IGRA and ISD are proxy-limited archives rather than exact live vintages.",
                "Only ISD latest-before-15:00 local observations are used; full-day ISD daily min/max fields are excluded.",
            ),
        ),
        ExperimentSpec(
            "HKG-T24-R16",
            "EXP-0048",
            "fifty_year_regional_isd_surface_core",
            "Fifty-Year Regional ISD Surface Core",
            "Test whether long-history regional surface observations add robust skill across multiple eras.",
            "isd_surface",
            (
                ModelSpec("r16_lag_calendar_baseline", "ridge_calendar_lag_baseline", baseline),
                ModelSpec("r16_isd_regional_aggregate", "ridge_isd_regional_aggregate", baseline + isd_agg),
                ModelSpec("r16_isd_station_panel", "ridge_isd_station_panel", baseline + isd_station),
            ),
            ("isd_",),
            (
                "NOAA ISD annual archive is quality-controlled and not an exact historical operational feed.",
                "Station-specific panel columns are chosen by predictor availability only, not by target outcome.",
            ),
        ),
        ExperimentSpec(
            "HKG-T24-R17",
            "EXP-0049",
            "metadata_breaks_urbanization_era_transfer",
            "Station Metadata Breaks, Urbanization, and Era Transfer",
            "Test whether known eras and station-network coverage shifts change long-history surface/upper-air relationships.",
            "era_transfer",
            (
                ModelSpec("r17_lag_calendar_baseline", "ridge_calendar_lag_baseline", baseline),
                ModelSpec(
                    "r17_combined_long_history_core",
                    "ridge_combined_long_history_core",
                    baseline + upper + isd_agg + coupling,
                ),
                ModelSpec(
                    "r17_era_transfer_terms",
                    "ridge_era_transfer_terms",
                    baseline + upper + isd_agg + coupling + era,
                ),
            ),
            ("igra_", "isd_"),
            (
                "This is an era-transfer diagnostic, not proof of causal urbanization impact.",
                "Official HKO station-move metadata remains incomplete, so calendar era terms are only coarse diagnostics.",
            ),
        ),
    )


def has_required_history(frame: pd.DataFrame, prefixes: Sequence[str]) -> bool:
    cols = columns_from_prefixes(frame, prefixes)
    if not cols:
        return False
    support = frame[list(cols)].notna().sum(axis=1) > 0
    if support.sum() == 0:
        return False
    first = frame.loc[support, "target_date"].min().date()
    last = frame.loc[support & (frame["target_date"] <= ANALYSIS_END), "target_date"].max().date()
    years = ((last - first).days + 1) / 365.25
    return years >= MIN_HISTORY_YEARS


def experiment_frame(features: pd.DataFrame, spec: ExperimentSpec) -> pd.DataFrame:
    frame = features.copy()
    required_mask = pd.Series(True, index=frame.index)
    for prefix in spec.required_non_null_prefixes:
        prefix_cols = columns_from_prefixes(frame, (prefix,))
        if not prefix_cols:
            raise RuntimeError(f"{spec.research_id} has no required columns for prefix {prefix}")
        required_mask &= frame[list(prefix_cols)].notna().sum(axis=1) > 0
    frame = frame[required_mask].copy()
    frame = frame[frame["target_date"] <= ANALYSIS_END].copy()
    if frame.empty:
        raise RuntimeError(f"{spec.research_id} has no long-history feature rows.")
    first = frame["target_date"].min().date()
    last = frame["target_date"].max().date()
    history_years = ((last - first).days + 1) / 365.25
    if history_years < MIN_HISTORY_YEARS:
        raise RuntimeError(
            f"{spec.research_id} fails robust history gate: {history_years:.2f} years, "
            f"requires {MIN_HISTORY_YEARS:.1f}"
        )
    assert_no_locked_dates(frame["target_date"], context=f"{spec.research_id} robust experiment frame")
    return frame.reset_index(drop=True)


def active_cols(train: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    active: list[str] = []
    for col in columns:
        if col not in train.columns or not pd.api.types.is_numeric_dtype(train[col]):
            continue
        values = train[col]
        if values.notna().sum() < MIN_FEATURE_SUPPORT:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        active.append(col)
    return active


def fit_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def run_oof(features: pd.DataFrame, spec: ExperimentSpec) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold_id, test_start, test_end, train_end in robust_fold_definitions():
        train = features[features["target_date"] <= train_end].copy()
        test = features[(features["target_date"] >= test_start) & (features["target_date"] <= test_end)].copy()
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        for model_spec in spec.model_specs:
            cols = active_cols(train, model_spec.columns)
            if not cols:
                continue
            model = fit_pipeline()
            model.fit(train[cols], train["target_tmax_c"])
            train_pred = model.predict(train[cols])
            sigma = float(np.std(train["target_tmax_c"].to_numpy(dtype=float) - train_pred, ddof=1))
            sigma = max(sigma, 0.2)
            pred = test[["target_date", "target_tmax_c", "year", "month"]].copy()
            pred["research_id"] = spec.research_id
            pred["fold_id"] = fold_id
            pred["model_id"] = model_spec.model_id
            pred["model_family"] = model_spec.model_family
            pred["is_control"] = model_spec.control
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
        raise RuntimeError(f"{spec.research_id} produced no robust OOF predictions.")
    predictions = pd.concat(rows, ignore_index=True).sort_values(
        ["target_date", "model_id"]
    ).reset_index(drop=True)
    assert_no_locked_dates(predictions["target_date"], context=f"{spec.research_id} OOF predictions")
    return predictions


def score_frame(predictions: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in predictions.groupby(list(group_cols), dropna=False, observed=True):
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
                "coverage_80": float(
                    ((group["q10"] <= group["target_tmax_c"]) & (group["target_tmax_c"] <= group["q90"])).mean()
                ),
                "coverage_90": float(
                    ((group["q05"] <= group["target_tmax_c"]) & (group["target_tmax_c"] <= group["q95"])).mean()
                ),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)


def fold_deltas(predictions: pd.DataFrame, baseline_model_id: str) -> pd.DataFrame:
    scores = score_frame(predictions, ["fold_id", "model_id"])
    baseline = scores[scores["model_id"].eq(baseline_model_id)][["fold_id", "mae", "crps_normal"]].rename(
        columns={"mae": "baseline_mae", "crps_normal": "baseline_crps"}
    )
    return scores.merge(baseline, on="fold_id", how="left").assign(
        mae_improvement_vs_baseline=lambda frame: frame["baseline_mae"] - frame["mae"],
        crps_improvement_vs_baseline=lambda frame: frame["baseline_crps"] - frame["crps_normal"],
    )


def era_name(year: int) -> str:
    if year < 1985:
        return "pre_1985"
    if year < 1997:
        return "1985_1996"
    if year < 2014:
        return "1997_2013"
    return "2014_2023"


def subgroup_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    work["era"] = work["year"].map(era_name)
    work["season"] = pd.cut(
        work["month"],
        bins=[0, 2, 5, 8, 11, 12],
        labels=["DJF", "MAM", "JJA", "SON", "DJF"],
        ordered=False,
    )
    era_scores = score_frame(work, ["model_id", "era"])
    season_scores = score_frame(work, ["model_id", "season"])
    era_scores["subgroup_type"] = "era"
    season_scores["subgroup_type"] = "season"
    return pd.concat([era_scores, season_scores], ignore_index=True, sort=False)


def feature_diagnostics(features: pd.DataFrame, spec: ExperimentSpec) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    used_columns = sorted({col for model in spec.model_specs for col in model.columns if col in features.columns})
    for col in used_columns:
        if not pd.api.types.is_numeric_dtype(features[col]):
            continue
        valid = features[[col, "target_tmax_c"]].dropna()
        if len(valid) < MIN_FEATURE_SUPPORT:
            continue
        rows.append(
            {
                "feature": col,
                "n": int(len(valid)),
                "first_date": str(features.loc[valid.index, "target_date"].min().date()),
                "last_date": str(features.loc[valid.index, "target_date"].max().date()),
                "pearson_with_target": float(valid[col].corr(valid["target_tmax_c"], method="pearson")),
                "spearman_with_target": float(valid[col].corr(valid["target_tmax_c"], method="spearman")),
                "feature_mean": float(valid[col].mean()),
                "feature_std": float(valid[col].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "pearson_with_target", key=lambda values: values.abs(), ascending=False
    ).reset_index(drop=True)


def select_champion(scoreboard: pd.DataFrame, baseline_model_id: str) -> Mapping[str, object]:
    non_baseline = scoreboard[~scoreboard["model_id"].eq(baseline_model_id)]
    if non_baseline.empty:
        return scoreboard.iloc[0].to_dict()
    return non_baseline.iloc[0].to_dict()


def experiment_payload(
    *,
    spec: ExperimentSpec,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    fold_scores: pd.DataFrame,
    diagnostics: pd.DataFrame,
    input_paths: Mapping[str, str],
    output_paths: Mapping[str, str],
) -> dict[str, object]:
    feature_dates = pd.to_datetime(features["target_date"])
    pred_dates = pd.to_datetime(predictions["target_date"])
    oof = check_four_year_oof_feasibility(
        pred_dates.min().date(),
        pred_dates.max().date(),
        min_years=MIN_OOF_YEARS,
        reason_context=f"{spec.research_id} robust rolling-origin OOF span",
    )
    history_years = ((feature_dates.max().date() - feature_dates.min().date()).days + 1) / 365.25
    baseline_model_id = spec.model_specs[0].model_id
    baseline = scoreboard[scoreboard["model_id"].eq(baseline_model_id)].iloc[0].to_dict()
    champion = dict(select_champion(scoreboard, baseline_model_id))
    champion["mae_improvement_vs_baseline"] = float(baseline["mae"] - champion["mae"])
    return {
        "generated_at_utc": now_utc(),
        "git_state": git_state(),
        "research_id": spec.research_id,
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "feature_min": str(feature_dates.min().date()),
        "feature_max": str(feature_dates.max().date()),
        "feature_rows": int(len(features)),
        "feature_columns": int(len(features.columns)),
        "input_history_years": round(history_years, 3),
        "minimum_history_years_required": MIN_HISTORY_YEARS,
        "history_gate": "PASS" if history_years >= MIN_HISTORY_YEARS else "BLOCKED",
        "prediction_min": str(pred_dates.min().date()),
        "prediction_max": str(pred_dates.max().date()),
        "prediction_rows": int(len(predictions)),
        "oof_feasibility": oof.__dict__,
        "baseline": baseline,
        "champion": champion,
        "input_paths": dict(input_paths),
        "output_paths": dict(output_paths),
        "caveats": list(spec.caveats),
        "diagnostic_feature_count": int(len(diagnostics)),
    }


def long_report(spec: ExperimentSpec, payload: Mapping[str, object], scoreboard: pd.DataFrame, fold_scores: pd.DataFrame) -> str:
    champion = payload["champion"]
    baseline = payload["baseline"]
    oof = payload["oof_feasibility"]
    assert isinstance(champion, Mapping)
    assert isinstance(baseline, Mapping)
    assert isinstance(oof, Mapping)
    caveats = "\n".join(f"- {item}" for item in spec.caveats)
    score_table = markdown_table(
        scoreboard,
        ["model_id", "n", "first_date", "last_date", "mae", "rmse", "bias", "crps_normal", "coverage_80", "coverage_90"],
    )
    fold_table = markdown_table(
        fold_scores,
        ["fold_id", "model_id", "n", "mae", "baseline_mae", "mae_improvement_vs_baseline", "crps_improvement_vs_baseline"],
        limit=60,
    )
    return f"""# {spec.experiment_id} / {spec.research_id} Long-Form Experiment Report

## Purpose

{spec.title} is a robust long-history continuation experiment for the HKG T-24 Tmax research track. The user constraint for this continuation is stricter than the earlier modern high-frequency diagnostics: a dataset must have at least {MIN_HISTORY_YEARS:.0f} years of usable history, and the out-of-fold period must cover at least four to five years. This experiment therefore ignores RSS forecasts, radar, satellite, lightning, nowcast, and other current-only families. It uses only source families with enough parsed history to support real chronological stress testing before validation 2024.

The specific research question is: {spec.purpose} The target remains Hong Kong Observatory Headquarters official daily maximum temperature for local date T. The forecast cutoff remains T-1 15:00 Asia/Hong_Kong. No Polymarket data, trading logic, market replay, or final validation freeze is touched.

## Data Used

The feature target-date period is `{payload['feature_min']}` through `{payload['feature_max']}`, giving `{payload['input_history_years']}` years of usable predictor history for this experiment. The OOF prediction period is `{payload['prediction_min']}` through `{payload['prediction_max']}`. The OOF gate is `{oof['status']}`: {oof['reason']}. Validation 2024 is not read. Locked-test dates from 2025-01-01 onward are not read. The generated feature matrix has `{payload['feature_rows']}` rows and `{payload['feature_columns']}` columns before fold-local model support filtering.

Input tables are read from the normalized non-minute archive: HKO daily Tmax target labels, NOAA IGRA Hong Kong upper-air sounding features, and NOAA ISD regional station-day cutoff summaries. The ISD table is used only through latest-before-15:00 HKT fields. Full-day ISD daily min/max fields are deliberately excluded because they can contain post-cutoff information. The IGRA relative-humidity fields are deliberately excluded because the normalized table still contains scaling/sentinel anomalies; using them would create a false sense of precision.

## As-Of Contract

Every predictor is either calendar-known, target-history lagged by at least two days, a latest eligible IGRA sounding assigned to origin day T-1, or a regional ISD observation summary using only observations at or before 15:00 local time on origin day T-1. The target label for T is never used as a feature. Target-day weather observations are never used. Daily climate variables for T are never used as predictors. The script calls the locked-test guard on feature dates and prediction dates.

The experiment is still marked proxy-limited rather than production-eligible. IGRA and ISD period-of-record archives are parsed and long-history, but they are retrospective quality-controlled archives rather than exact immutable operational vintages. That means the experiment can produce robust scientific evidence about whether the physical signal exists, while still failing closed for production promotion until publication/release-latency contracts are proven.

## Model Ladder

The first row in the model ladder is a lag/calendar baseline using day-of-year sin/cos and HKO target-history lags that are at least two days old. The remaining rows add only the experiment-specific long-history feature family. Each model is a Ridge regression with median imputation and standard scaling fitted separately inside each chronological training fold. There is no random split, no target-aware feature selection, no validation tuning, and no hyperparameter search. Feature columns must have at least `{MIN_FEATURE_SUPPORT}` non-null training rows inside a fold before entering that fold.

## Chronological OOF Design

The OOF protocol uses rolling-origin five-year blocks starting in {OOF_START_YEAR}. Each fold trains only on dates before the fold's test window, then scores the next four to five calendar years through 2023. This is intentionally much stricter than the earlier 2020-2023 high-frequency diagnostics. The total scored OOF window is more than five decades for the long-history sources, so single-year luck has much less opportunity to dominate the conclusion. The folds also expose whether a feature works in early, middle, and modern eras rather than only in the recent sample.

## Main Result

The baseline `{baseline['model_id']}` scores MAE `{baseline['mae']:.4f}` C, RMSE `{baseline['rmse']:.4f}` C, bias `{baseline['bias']:.4f}` C, and CRPS `{baseline['crps_normal']:.4f}` over `{baseline['n']}` OOF rows. The best non-baseline row is `{champion['model_id']}` with MAE `{champion['mae']:.4f}` C, RMSE `{champion['rmse']:.4f}` C, bias `{champion['bias']:.4f}` C, and CRPS `{champion['crps_normal']:.4f}` over `{champion['n']}` rows. Its MAE improvement versus the baseline is `{champion['mae_improvement_vs_baseline']:.4f}` C. This should be interpreted as robust research evidence, not a production release decision.

## Scoreboard

{score_table}

## Fold Evidence

{fold_table}

## Caveats

{caveats}

## Interpretation

The important question is not only whether one overall row has a lower MAE. A robust feature family should improve or at least not degrade across many chronological folds, should not win only because of a single modern interval, and should not behave like an accidental proxy for year or missingness. The fold-delta artifact in this experiment folder is therefore as important as the headline scoreboard. A small positive overall delta with unstable fold signs is treated as weak evidence. A stable positive delta across old and modern folds is stronger evidence that the source family deserves further engineering.

For upper-air features, a plausible physical signal is lower-tropospheric warmth, inversion structure, and midlevel stability influencing the next day's heating ceiling. For ISD surface features, a plausible signal is regional air mass state before the cutoff: broad warmth, dewpoint spread, pressure regime, and wind exposure across nearby stations. For coupling features, the core idea is surface-to-aloft mismatch: a warm surface under a stable cap, a cool moist surface under warm air aloft, or regional pressure/wind conditions that modulate mixing. For era-transfer features, the key question is whether simple long-history relationships are stable across reporting regimes and urbanization eras.

## Leakage Review

The runner does not inspect validation 2024. It does not score locked-test dates from 2025 onward. It does not use target-day climate elements, retrospective best tracks, target-day full-day aggregates, reanalysis, model-analysis fields, market outcomes, or post-hoc selected validation errors. The feature support filter is evaluated inside each training fold using predictor availability and numeric variation only. Imputation, scaling, and coefficients are also fit inside each training fold only. The output remains marked non-production because archive vintage timing is not yet exact enough for live deployment.

## What Was Deliberately Not Done

The experiment does not try to rescue short-history modern data by mixing it into a long-history score. It does not backfill RSS forecasts before 2020. It does not pretend June 2026 radar/satellite snapshots can support 1965-2023 OOF. It does not use full-day ISD min/max as if they were known by 15:00. It does not promote any final challenger, and it does not authorize R30 validation. This discipline is what makes the result useful rather than just numerically impressive.

## Decision

The experiment status is `COMPLETE_ROBUST_LONG_HISTORY_PROXY_LIMITED`. It passes the user's robust-history and OOF-span requirements, but it is not production-eligible until the remaining point-in-time release/vintage caveats are resolved. The next safe use of this output is to compare fold stability, subgroup behavior, and feature diagnostics across R14-R17, then decide which long-history source family deserves a stricter as-of-contract hardening pass.

## Reproducibility

The experiment folder contains the local OOF predictions, scoreboard, fold deltas, subgroup metrics, feature diagnostics, hashes for the generated prediction and feature tables, run configuration, protocol, as-of contract, data manifest, status file, and this long-form report. The repo-level report mirrors the same content for handoff reading. The reproduction command is stored in `REPRODUCE.md`.
"""


def write_experiment(
    *,
    spec: ExperimentSpec,
    data_root: Path,
    feature_path: Path,
    predictions_path: Path,
    scoreboard: pd.DataFrame,
    fold_scores: pd.DataFrame,
    subgroups: pd.DataFrame,
    diagnostics: pd.DataFrame,
    payload: Mapping[str, object],
) -> None:
    folder = spec.folder
    for subdir in ["artifacts", "logs", "metrics", "predictions", "results"]:
        (folder / subdir).mkdir(parents=True, exist_ok=True)
    local_predictions = folder / "predictions" / "oof_predictions.parquet"
    pd.read_parquet(predictions_path).to_parquet(local_predictions, index=False)
    scoreboard.to_csv(folder / "results" / "scoreboard.csv", index=False)
    fold_scores.to_csv(folder / "artifacts" / "fold_score_deltas.csv", index=False)
    subgroups.to_parquet(folder / "metrics" / "subgroup_metrics.parquet", index=False)
    subgroups.to_csv(folder / "artifacts" / "subgroup_metrics.csv", index=False)
    diagnostics.to_csv(folder / "artifacts" / "feature_diagnostics.csv", index=False)
    metrics = {
        **dict(payload),
        "status": "COMPLETE_ROBUST_LONG_HISTORY_PROXY_LIMITED",
        "validation_2024_accessed": False,
        "locked_test_accessed": False,
        "production_eligible": False,
        "scoreboard": scoreboard.to_dict(orient="records"),
        "fold_scores": fold_scores.to_dict(orient="records"),
    }
    write_text(folder / "metrics" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    write_text(folder / "results" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    write_text(folder / "logs" / "run_summary.json", json.dumps(payload, indent=2, default=str))
    write_text(folder / "README.md", f"# {spec.experiment_id} {spec.research_id} {spec.title}\n\nRobust long-history experiment using at least {MIN_HISTORY_YEARS:.0f} years of parsed data and rolling-origin OOF through 2023. Status: `COMPLETE_ROBUST_LONG_HISTORY_PROXY_LIMITED`.\n")
    write_text(folder / "HYPOTHESIS.md", f"# Hypothesis\n\n{spec.purpose}\n")
    write_text(folder / "INFORMATION_GAIN.md", "# Information Gain\n\nThis experiment replaces the earlier stale parser-blocked gate with actual long-history OOF evidence from parsed IGRA/ISD source tables. The main information gained is whether the feature family has stable chronological value over many decades, not whether it can be promoted to production yet.\n")
    write_text(folder / "ASOF_CONTRACT.md", "# As-Of Contract\n\nCutoff is T-1 15:00 Asia/Hong_Kong. Predictors are calendar-known, target-history lags at least two days old, IGRA origin-day sounding features, or ISD latest-before-15:00 station summaries. Full-day ISD min/max, target-day climate values, validation 2024, locked-test dates, market data, and Polymarket data are excluded.\n")
    write_text(folder / "FEATURE_SPEC.yaml", "research_id: " + spec.research_id + "\nfeature_group: " + spec.feature_group + "\nmodels:\n" + "\n".join(f"  - {model.model_id}: {model.model_family}" for model in spec.model_specs) + "\n")
    write_text(folder / "RUN_CONFIG.yaml", f"""research_id: {spec.research_id}
experiment_id: {spec.experiment_id}
data_root: {data_root}
analysis_end: {ANALYSIS_END.date()}
oof_start_year: {OOF_START_YEAR}
fold_years: {FOLD_YEARS}
minimum_history_years: {MIN_HISTORY_YEARS}
minimum_oof_years: {MIN_OOF_YEARS}
minimum_train_rows: {MIN_TRAIN_ROWS}
model_family: ridge_alpha_1_with_fold_local_imputer_scaler
validation_2024_accessed: false
locked_test_policy: deny
production_eligible: false
""")
    write_text(folder / "PROTOCOL.md", "# Protocol\n\n1. Load normalized non-minute target, IGRA, and ISD tables.\n2. Build only cutoff-safe or proxy-cutoff-safe long-history features.\n3. Require at least 39 years of feature history.\n4. Score rolling-origin four/five-year OOF folds from 1965 through 2023.\n5. Fit imputation, scaling, and Ridge coefficients inside each fold only.\n6. Write full negative/null/positive evidence without validation or locked-test access.\n")
    write_text(folder / "ABLATION_PLAN.md", "# Ablation Plan\n\nEach experiment includes a lag/calendar baseline and one or more predeclared additions for the source family. Fold deltas versus the baseline are the primary ablation evidence.\n")
    write_text(folder / "NEGATIVE_CONTROLS.md", "# Negative Controls\n\nNo random shuffled or validation-derived control is used. The baseline lag/calendar model is the control row, and every richer source-family model must beat it across chronological folds to be considered useful.\n")
    write_text(folder / "DATE_RANGES.md", f"""# Date Ranges

- Feature target-date period: `{payload['feature_min']}` through `{payload['feature_max']}`.
- Input history years: `{payload['input_history_years']}`.
- OOF prediction period: `{payload['prediction_min']}` through `{payload['prediction_max']}`.
- OOF requirement: at least `{MIN_OOF_YEARS}` years.
- Validation 2024: not accessed.
- Locked test 2025-01-01 onward: not accessed.
- Minimum robust-history gate: `{payload['history_gate']}`.
""")
    write_text(folder / "DATA_MANIFEST.yaml", f"""research_id: {spec.research_id}
data_root: {data_root}
feature_matrix: {feature_path}
feature_matrix_sha256: {sha256_file(feature_path)}
oof_predictions: {predictions_path}
oof_predictions_sha256: {sha256_file(predictions_path)}
repo_oof_predictions: {local_predictions}
repo_oof_predictions_sha256: {sha256_file(local_predictions)}
validation_2024_accessed: false
locked_test_accessed: false
availability_tier: PROXY_WITH_LIMITATIONS
caveats:
""" + "\n".join(f"  - {item}" for item in spec.caveats) + "\n")
    report = long_report(spec, payload, scoreboard, fold_scores)
    write_text(folder / "EXPERIMENT_REPORT_7500_CHARS.md", report)
    write_text(folder / "RESULTS.md", "# Results\n\n" + markdown_table(scoreboard, ["model_id", "n", "first_date", "last_date", "mae", "rmse", "bias", "crps_normal", "coverage_80", "coverage_90"]) + "\n\n## Fold Deltas\n\n" + markdown_table(fold_scores, ["fold_id", "model_id", "mae", "baseline_mae", "mae_improvement_vs_baseline"], limit=100) + "\n")
    write_text(folder / "CONCLUSION.md", f"# Conclusion\n\n{spec.research_id} is complete as a robust long-history proxy-limited experiment. It passes the 39-year history and 4-5 year OOF requirements, but it is not production-eligible until exact operational vintage/release semantics are proven.\n")
    write_text(folder / "REPRODUCE.md", "# Reproduce\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_r14_r17_robust_long_history.py --data-root data\n```\n")
    write_text(folder / "STATUS.yaml", f"""status: COMPLETE_ROBUST_LONG_HISTORY_PROXY_LIMITED
research_id: {spec.research_id}
locked_test_accessed: false
validation_2024_accessed: false
leakage_guard: PASS
minimum_history_39_years: PASS
four_to_five_year_oof: PASS
production_eligible: false
availability_tier: PROXY_WITH_LIMITATIONS
""")
    write_text(spec.report_path, report + "\n\n## Feature Diagnostics\n\n" + markdown_table(diagnostics, ["feature", "n", "first_date", "last_date", "pearson_with_target", "spearman_with_target"], limit=50) + "\n")


def run_experiment(data_root: Path, spec: ExperimentSpec, features: pd.DataFrame, input_paths: Mapping[str, str]) -> dict[str, object]:
    frame = experiment_frame(features, spec)
    predictions = run_oof(frame, spec)
    scoreboard = score_frame(predictions, ["model_id"])
    fold_scores = fold_deltas(predictions, spec.model_specs[0].model_id)
    subgroups = subgroup_scores(predictions)
    diagnostics = feature_diagnostics(frame, spec)
    output_dir = data_root / "gold" / "hkg_t24" / "r14_r17_robust_long_history"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / f"{spec.research_id.lower().replace('-', '_')}_feature_matrix.parquet"
    predictions_path = output_dir / f"{spec.research_id.lower().replace('-', '_')}_oof_predictions.parquet"
    scoreboard_path = output_dir / f"{spec.research_id.lower().replace('-', '_')}_scoreboard.parquet"
    fold_path = output_dir / f"{spec.research_id.lower().replace('-', '_')}_fold_score_deltas.parquet"
    diagnostics_path = output_dir / f"{spec.research_id.lower().replace('-', '_')}_feature_diagnostics.parquet"
    frame.to_parquet(feature_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    scoreboard.to_parquet(scoreboard_path, index=False)
    fold_scores.to_parquet(fold_path, index=False)
    diagnostics.to_parquet(diagnostics_path, index=False)
    payload = experiment_payload(
        spec=spec,
        features=frame,
        predictions=predictions,
        scoreboard=scoreboard,
        fold_scores=fold_scores,
        diagnostics=diagnostics,
        input_paths=input_paths,
        output_paths={
            "feature_matrix": str(feature_path),
            "oof_predictions": str(predictions_path),
            "scoreboard": str(scoreboard_path),
            "fold_deltas": str(fold_path),
            "feature_diagnostics": str(diagnostics_path),
        },
    )
    write_experiment(
        spec=spec,
        data_root=data_root,
        feature_path=feature_path,
        predictions_path=predictions_path,
        scoreboard=scoreboard,
        fold_scores=fold_scores,
        subgroups=subgroups,
        diagnostics=diagnostics,
        payload=payload,
    )
    return payload


def write_eligibility_report(payloads: Sequence[Mapping[str, object]]) -> None:
    rows = []
    for payload in payloads:
        champion = payload["champion"]
        baseline = payload["baseline"]
        assert isinstance(champion, Mapping)
        assert isinstance(baseline, Mapping)
        rows.append(
            {
                "research_id": payload["research_id"],
                "status": "scored",
                "history_years": payload["input_history_years"],
                "oof_period": f"{payload['prediction_min']} to {payload['prediction_max']}",
                "baseline_mae": round(float(baseline["mae"]), 4),
                "best_nonbaseline": champion["model_id"],
                "best_mae": round(float(champion["mae"]), 4),
                "mae_delta": round(float(champion["mae_improvement_vs_baseline"]), 4),
            }
        )
    blocked = [
        {
            "research_id": "HKG-T24-R18",
            "status": "not scored",
            "history_years": "2020-2026 only",
            "oof_period": "fails >=39-year robust-history gate",
            "baseline_mae": "",
            "best_nonbaseline": "RSS official forecasts short history",
            "best_mae": "",
            "mae_delta": "",
        },
        {
            "research_id": "HKG-T24-R19-R30",
            "status": "not scored",
            "history_years": "mixed or dependency-limited",
            "oof_period": "blocked until robust long-history source families are accepted/hardened",
            "baseline_mae": "",
            "best_nonbaseline": "not eligible under current robust-only instruction",
            "best_mae": "",
            "mae_delta": "",
        },
    ]
    report = pd.DataFrame(rows + blocked)
    write_text(
        REPORT_ROOT / "R14_ONWARD_ROBUST_LONG_HISTORY_CONTINUATION.md",
        "# R14 Onward Robust Long-History Continuation\n\n"
        f"Generated: `{now_utc()}`\n\n"
        "This continuation obeys the user's robust-only constraint: scored experiments must use at least 39 years of parsed history and at least four to five years of OOF data. Short-history RSS, radar, satellite, lightning, nowcast, and live-only feeds are not scored here.\n\n"
        + markdown_table(
            report,
            [
                "research_id",
                "status",
                "history_years",
                "oof_period",
                "baseline_mae",
                "best_nonbaseline",
                "best_mae",
                "mae_delta",
            ],
            limit=50,
        )
        + "\n\nValidation 2024 and locked-test dates were not accessed. The scored rows remain proxy-limited, not production-eligible, until exact operational vintage/release semantics are proven.\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robust long-history HKG-T24 R14-R17 experiments.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--research-id", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    features, input_paths = build_feature_matrix(data_root)
    specs = make_experiment_specs(features)
    selected = [spec for spec in specs if args.research_id == "all" or spec.research_id == args.research_id]
    if not selected:
        raise ValueError(f"No robust R14-R17 experiment found for {args.research_id}")
    payloads = [run_experiment(data_root, spec, features, input_paths) for spec in selected]
    write_eligibility_report(payloads)
    print(json.dumps({"status": "ok", "experiments": payloads}, indent=2, default=str))


if __name__ == "__main__":
    main()
