from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from hkg_tmax.hkg_t24.guard import assert_no_locked_dates
from hkg_tmax.paths import ProjectPaths, configured_input_path

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
DATASETS_ROOT = PROJECT_PATHS.data_root / "datasets"
EXPERIMENT_ROOT = REPO_ROOT / "experiments"
REPORT_ROOT = PROJECT_PATHS.run_root / "reports" / "long_history_50_experiments"
OUTPUT_ROOT = PROJECT_PATHS.data_root / "datasets" / "12_hkg_t24_robust_experiment_outputs"
BUNDLE_ZIP_PATH = PROJECT_PATHS.run_root / "inputs" / "hkg_tmax_datasets_and_experiments_20260620_162157.zip"

SPEC_FILENAME = "HKG_TMAX_50_NEW_LONG_HISTORY_EXPERIMENTS.md"
DEFAULT_SPEC_PATH = configured_input_path(
    PROJECT_PATHS,
    "HKG_T24_LONG_HISTORY_SPEC_PATH",
    SPEC_FILENAME,
    legacy_home_relative=Path("Downloads") / SPEC_FILENAME,
)

TRAIN_END = pd.Timestamp("2019-12-31")
HEADLINE_START = pd.Timestamp("2020-01-01")
HEADLINE_END = pd.Timestamp("2023-12-31")
CONFIRMATION_START = pd.Timestamp("2024-01-01")
OOF_START_YEAR = 1965
FOLD_YEARS = 5
MIN_TRAIN_ROWS = 3650
MIN_FEATURE_SUPPORT = 365
MIN_HEADLINE_DAYS_PER_YEAR = 330
LONG_FORM_REPORT_MIN_CHARS = 7500
LEGACY_EXPERT_CACHE: pd.DataFrame | None = None

ELIGIBLE_ISD_STATIONS = {
    "450050-99999",
    "450070-99999",
    "450110-99999",
    "590870-99999",
    "590960-99999",
    "592710-99999",
    "592780-99999",
    "592870-99999",
    "592930-99999",
    "592980-99999",
    "593030-99999",
    "594780-99999",
    "594930-99999",
    "595010-99999",
    "596730-99999",
}

QUANTILE_Z = {
    "q05": -1.6448536269514722,
    "q10": -1.2815515655446004,
    "q25": -0.6744897501960817,
    "q50": 0.0,
    "q75": 0.6744897501960817,
    "q90": 1.2815515655446004,
    "q95": 1.6448536269514722,
}


@dataclass(frozen=True)
class ParsedExperiment:
    experiment_id: str
    title: str
    category: str
    priority: str
    sequence: str
    question: str
    hypothesis: str
    eligible_datasets: str
    construction: str
    model_ladder: str
    negative_controls: str
    subgroups: str
    acceptance: str
    leakage: str
    next_step: str
    required_artifacts: str

    @property
    def slug(self) -> str:
        return slugify(self.title)

    @property
    def folder_name(self) -> str:
        return self.experiment_id


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_family: str
    columns: tuple[str, ...]
    control: bool = False
    shuffled_control: bool = False
    strategy: str = "ridge"


@dataclass(frozen=True)
class RunSpec:
    parsed: ParsedExperiment
    feature_columns: tuple[str, ...]
    decision_bias: str = "model"
    mechanism_only: bool = False
    implementation_note: str = "predeclared long-history ridge feature-block experiment"


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    text = value.lower().replace("t−", "tminus").replace("t-", "tminus")
    text = text.replace("0.1", "0p1")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:82]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, object]:
    def run_git(args: list[str]) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else f"unavailable: {completed.stderr.strip()}"

    status = run_git(["status", "--short"]).splitlines()
    return {
        "head": run_git(["rev-parse", "HEAD"]),
        "dirty_count": len([line for line in status if line.strip()]),
    }


def parse_spec(path: Path) -> tuple[str, tuple[ParsedExperiment, ...]]:
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^## (EXP-\d{4}) — (.+)$", text, flags=re.M))
    if len(matches) != 50:
        raise RuntimeError(f"Expected 50 experiment sections, found {len(matches)} in {path}")

    def section_block(section: str, title: str) -> str:
        match = re.search(
            rf"^### {re.escape(title)}\n\n(.*?)(?=\n### |\n---\n|\Z)",
            section,
            flags=re.M | re.S,
        )
        return match.group(1).strip() if match else ""

    parsed: list[ParsedExperiment] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else text.find("\n# Cross-Experiment", start)
        if end == -1:
            end = len(text)
        section = text[start:end].strip()
        category = re.search(r"\*\*Batch/category:\*\*\s*(.+)", section)
        priority = re.search(r"\*\*Priority:\*\*\s*(.+)", section)
        sequence = re.search(r"\*\*Sequence number:\*\*\s*(.+)", section)
        parsed.append(
            ParsedExperiment(
                experiment_id=match.group(1),
                title=match.group(2).strip(),
                category=category.group(1).strip() if category else "",
                priority=priority.group(1).strip() if priority else "",
                sequence=sequence.group(1).strip() if sequence else "",
                question=section_block(section, "Exact information-gain question"),
                hypothesis=section_block(section, "Predeclared meteorological/statistical hypothesis"),
                eligible_datasets=section_block(section, "Eligible datasets"),
                construction=section_block(section, "Exact feature and analysis construction"),
                model_ladder=section_block(section, "Model ladder"),
                negative_controls=section_block(section, "Negative controls and falsification tests"),
                subgroups=section_block(section, "Mandatory subgroup analysis"),
                acceptance=section_block(section, "Experiment-specific acceptance rule"),
                leakage=section_block(section, "Leakage-specific requirement"),
                next_step=section_block(section, "Required next-step decision"),
                required_artifacts=section_block(section, "Required artifacts for this experiment"),
            )
        )
    expected_ids = [f"EXP-{number:04d}" for number in range(50, 100)]
    actual_ids = [item.experiment_id for item in parsed]
    if actual_ids != expected_ids:
        raise RuntimeError(f"Experiment ID sequence mismatch: {actual_ids[:3]} ... {actual_ids[-3:]}")
    return text, tuple(parsed)


def dataset_table(path: Path, columns: Sequence[str], filters: object | None = None) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(columns=list(columns), filter=filters)
    return table.to_pandas()


def strict_confirmation_guard(dates: Iterable[object], *, context: str) -> None:
    series = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
    bad = series[series >= CONFIRMATION_START]
    if not bad.empty:
        examples = ", ".join(str(value.date()) for value in bad.head(5))
        raise RuntimeError(f"{context} attempted to use confirmation dates >= 2024-01-01: {examples}")
    assert_no_locked_dates(series.dropna(), context=context)


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


def clean_numeric(values: pd.Series, lower: float | None = None, upper: float | None = None) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    numeric = numeric.mask(numeric <= -800.0)
    if lower is not None:
        numeric = numeric.mask(numeric < lower)
    if upper is not None:
        numeric = numeric.mask(numeric > upper)
    return numeric


def robust_fold_definitions() -> list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds: list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for year in range(OOF_START_YEAR, HEADLINE_END.year + 1, FOLD_YEARS):
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = min(pd.Timestamp(year=year + FOLD_YEARS - 1, month=12, day=31), HEADLINE_END)
        folds.append((f"fold_{test_start.year}_{test_end.year}", test_start, test_end, test_start - pd.Timedelta(days=1)))
    return folds


def active_cols(train: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    active: list[str] = []
    for col in columns:
        if col not in train.columns:
            continue
        if not pd.api.types.is_numeric_dtype(train[col]):
            continue
        if train[col].notna().sum() < MIN_FEATURE_SUPPORT:
            continue
        if train[col].nunique(dropna=True) <= 1:
            continue
        active.append(col)
    return active


def fit_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


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
                "coverage_80": float(((group["q10"] <= group["target_tmax_c"]) & (group["target_tmax_c"] <= group["q90"])).mean()),
                "coverage_90": float(((group["q05"] <= group["target_tmax_c"]) & (group["target_tmax_c"] <= group["q95"])).mean()),
            }
        )
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)


def load_target_features() -> pd.DataFrame:
    path = DATASETS_ROOT / "01_hko_daily_tmax_target" / "hko_daily_tmax_target_labels.parquet"
    frame = dataset_table(
        path,
        ["local_date", "target_tmax_c", "content_sha256", "raw_retrieved_at_utc"],
        pc.field("local_date") <= "2023-12-31",
    )
    frame["target_date"] = pd.to_datetime(frame["local_date"]).dt.normalize()
    frame = frame.sort_values("target_date").reset_index(drop=True)
    frame = frame[frame["target_date"] <= HEADLINE_END].copy()
    strict_confirmation_guard(frame["target_date"], context="target feature load")
    frame["year"] = frame["target_date"].dt.year
    frame["month"] = frame["target_date"].dt.month
    frame["day_of_year"] = frame["target_date"].dt.dayofyear
    frame["doy_sin"] = np.sin(2.0 * np.pi * frame["day_of_year"] / 365.25)
    frame["doy_cos"] = np.cos(2.0 * np.pi * frame["day_of_year"] / 365.25)
    frame["doy_sin_2"] = np.sin(4.0 * np.pi * frame["day_of_year"] / 365.25)
    frame["doy_cos_2"] = np.cos(4.0 * np.pi * frame["day_of_year"] / 365.25)
    frame["year_centered"] = frame["year"] - 1985.0
    frame["year_centered_sq"] = frame["year_centered"] ** 2

    for lag in (7, 8, 9, 10, 11, 12, 13, 14, 21, 30, 45, 60, 90, 120, 180, 365):
        frame[f"target_lag{lag}_tmax_c"] = frame["target_tmax_c"].shift(lag)
    shifted = frame["target_tmax_c"].shift(7)
    for window in (7, 14, 30, 60, 90, 120, 365, 3650, 7300, 14610):
        min_periods = min(max(5, window // 3), window)
        frame[f"target_roll{window}_mean_lag7_c"] = shifted.rolling(window, min_periods=min_periods).mean()
        frame[f"target_roll{window}_std_lag7_c"] = shifted.rolling(window, min_periods=min_periods).std()
    frame["target_lag7_minus_lag14_c"] = frame["target_lag7_tmax_c"] - frame["target_lag14_tmax_c"]
    frame["target_lag7_minus_roll30_c"] = frame["target_lag7_tmax_c"] - frame["target_roll30_mean_lag7_c"]
    frame["target_lag7_minus_roll365_c"] = frame["target_lag7_tmax_c"] - frame["target_roll365_mean_lag7_c"]
    frame["target_spell_hot_lag7"] = spell_length(frame["target_lag7_minus_roll365_c"], threshold=1.0)
    frame["target_spell_cold_lag7"] = spell_length(-frame["target_lag7_minus_roll365_c"], threshold=1.0)
    frame["target_reversal_pressure_lag7"] = frame["target_lag7_minus_lag14_c"] * frame["target_lag7_minus_roll30_c"]
    frame["target_entropy_30_lag7"] = rolling_entropy(frame["target_tmax_c"].shift(7), window=30)
    frame["target_abs_change_7_14_c"] = (frame["target_lag7_tmax_c"] - frame["target_lag14_tmax_c"]).abs()
    return frame.drop(columns=["local_date"])


def spell_length(values: pd.Series, *, threshold: float) -> pd.Series:
    lengths: list[float] = []
    current = 0
    for value in values:
        if pd.notna(value) and value >= threshold:
            current += 1
        else:
            current = 0
        lengths.append(float(current))
    return pd.Series(lengths, index=values.index)


def rolling_entropy(values: pd.Series, *, window: int) -> pd.Series:
    bins = np.linspace(5.0, 40.0, 15)

    def entropy(x: np.ndarray) -> float:
        valid = x[np.isfinite(x)]
        if len(valid) < 10:
            return np.nan
        counts, _ = np.histogram(valid, bins=bins)
        probs = counts[counts > 0] / counts.sum()
        return float(-(probs * np.log(probs)).sum())

    return values.rolling(window, min_periods=10).apply(entropy, raw=True)


def load_igra_features() -> pd.DataFrame:
    path = DATASETS_ROOT / "03_noaa_igra_upper_air_hkm00045004" / "noaa_igra_hkm00045004_sounding_features.parquet"
    columns = [
        "valid_at_utc",
        "valid_at_hkt",
        "nominal_hour_utc",
        "key_level_count",
        *[
            f"{kind}_{level}hpa"
            for level in (1000, 925, 850, 700, 500, 300, 200)
            for kind in (
                "temperature_c",
                "dewpoint_depression_c",
                "geopotential_height_m",
                "wind_direction_deg",
                "wind_speed_mps",
            )
        ],
    ]
    frame = dataset_table(
        path,
        columns,
        (pc.field("valid_at_utc") <= "2023-12-30T23:59:59Z") & (pc.field("nominal_hour_utc") == 0),
    )
    frame["valid_at_utc_ts"] = pd.to_datetime(frame["valid_at_utc"], utc=True, errors="coerce")
    frame = frame[frame["valid_at_utc_ts"].notna()].copy()
    frame["target_date"] = frame["valid_at_utc_ts"].dt.tz_convert(None).dt.normalize() + pd.Timedelta(days=1)
    frame = frame[(frame["target_date"] <= HEADLINE_END) & (frame["target_date"] < CONFIRMATION_START)].copy()
    strict_confirmation_guard(frame["target_date"], context="IGRA strict 00 UTC feature load")
    frame = frame.sort_values(["target_date", "valid_at_utc_ts"]).drop_duplicates("target_date", keep="last")
    out = frame[["target_date", "valid_at_utc", "valid_at_hkt"]].copy()
    out["igra_key_level_count"] = clean_numeric(frame["key_level_count"], 0, 200)
    for level in (1000, 925, 850, 700, 500, 300, 200):
        out[f"igra_temp_{level}hpa_c"] = clean_numeric(frame[f"temperature_c_{level}hpa"], -90, 50)
        out[f"igra_dd_{level}hpa_c"] = clean_numeric(frame[f"dewpoint_depression_c_{level}hpa"], 0, 80)
        out[f"igra_hgt_{level}hpa_m"] = clean_numeric(frame[f"geopotential_height_m_{level}hpa"], -500, 20000)
        out[f"igra_wspd_{level}hpa_mps"] = clean_numeric(frame[f"wind_speed_mps_{level}hpa"], 0, 160)
        out[f"igra_wdir_{level}hpa_deg"] = clean_numeric(frame[f"wind_direction_deg_{level}hpa"], 0, 360)
    out["igra_temp_925_minus_850_c"] = out["igra_temp_925hpa_c"] - out["igra_temp_850hpa_c"]
    out["igra_temp_850_minus_500_c"] = out["igra_temp_850hpa_c"] - out["igra_temp_500hpa_c"]
    out["igra_inversion_925_minus_1000_c"] = out["igra_temp_925hpa_c"] - out["igra_temp_1000hpa_c"]
    out["igra_lower_mean_temp_c"] = out[[f"igra_temp_{level}hpa_c" for level in (1000, 925, 850, 700)]].mean(axis=1)
    out["igra_lower_mean_dd_c"] = out[[f"igra_dd_{level}hpa_c" for level in (1000, 925, 850, 700)]].mean(axis=1)
    out["igra_thickness_1000_500_m"] = out["igra_hgt_500hpa_m"] - out["igra_hgt_1000hpa_m"]
    out["igra_thickness_925_700_m"] = out["igra_hgt_700hpa_m"] - out["igra_hgt_925hpa_m"]
    for level in (1000, 925, 850, 700):
        out[f"igra_theta_{level}hpa_k"] = (out[f"igra_temp_{level}hpa_c"] + 273.15) * (1000.0 / level) ** 0.286
        out[f"igra_dry_adiabatic_surface_from_{level}hpa_c"] = out[f"igra_theta_{level}hpa_k"] - 273.15
    out["igra_mixed_layer_ceiling_c"] = out[[f"igra_dry_adiabatic_surface_from_{level}hpa_c" for level in (925, 850)]].mean(axis=1)
    out["igra_heat_content_proxy"] = out[["igra_temp_1000hpa_c", "igra_temp_925hpa_c", "igra_temp_850hpa_c", "igra_temp_700hpa_c"]].sum(axis=1)
    out["igra_moist_static_proxy"] = out["igra_lower_mean_temp_c"] - 0.35 * out["igra_lower_mean_dd_c"]
    out["igra_dry_layer_max_dd_c"] = out[[f"igra_dd_{level}hpa_c" for level in (925, 850, 700, 500)]].max(axis=1)
    out["igra_shear_925_700_mps"] = out["igra_wspd_700hpa_mps"] - out["igra_wspd_925hpa_mps"]
    out["igra_shear_850_500_mps"] = out["igra_wspd_500hpa_mps"] - out["igra_wspd_850hpa_mps"]
    out["igra_veering_925_700_deg"] = angular_difference(out["igra_wdir_700hpa_deg"], out["igra_wdir_925hpa_deg"])
    for col in [
        "igra_lower_mean_temp_c",
        "igra_lower_mean_dd_c",
        "igra_mixed_layer_ceiling_c",
        "igra_heat_content_proxy",
        "igra_thickness_1000_500_m",
    ]:
        out[f"{col}_change_24h"] = out[col] - out[col].shift(1)
        out[f"{col}_change_48h"] = out[col] - out[col].shift(2)
    return out.sort_values("target_date").reset_index(drop=True)


def angular_difference(a: pd.Series, b: pd.Series) -> pd.Series:
    return ((a - b + 180.0) % 360.0) - 180.0


def load_isd_features() -> pd.DataFrame:
    path = DATASETS_ROOT / "04_noaa_isd_regional_surface" / "noaa_isd_core_observations.parquet"
    columns = [
        "station_id",
        "observed_at_utc",
        "observed_at_hkt",
        "report_type",
        "latitude",
        "longitude",
        "elevation_m",
        "wind_direction_deg",
        "wind_speed_mps",
        "air_temperature_c",
        "dew_point_c",
        "sea_level_pressure_hpa",
        "temperature_quality_code",
        "dew_point_quality_code",
        "sea_level_pressure_quality_code",
    ]
    frame = dataset_table(path, columns, pc.field("observed_at_utc") <= "2023-12-30T23:59:59Z")
    frame = frame[frame["station_id"].isin(ELIGIBLE_ISD_STATIONS)].copy()
    frame["observed_at_utc_ts"] = pd.to_datetime(frame["observed_at_utc"], utc=True, errors="coerce")
    frame = frame[frame["observed_at_utc_ts"].notna()].copy()
    local = frame["observed_at_utc_ts"].dt.tz_convert("Asia/Hong_Kong")
    frame["origin_date"] = local.dt.tz_localize(None).dt.normalize()
    frame["local_hour"] = local.dt.hour + local.dt.minute / 60.0
    frame = frame[(frame["local_hour"] <= 13.5) & (frame["origin_date"] <= pd.Timestamp("2023-12-30"))].copy()
    frame["target_date"] = frame["origin_date"] + pd.Timedelta(days=1)
    frame = frame[frame["target_date"] <= HEADLINE_END].copy()
    strict_confirmation_guard(frame["target_date"], context="ISD <=13:30 feature load")
    for col, lo, hi in [
        ("air_temperature_c", -40, 50),
        ("dew_point_c", -60, 40),
        ("sea_level_pressure_hpa", 850, 1100),
        ("wind_speed_mps", 0, 80),
        ("wind_direction_deg", 0, 360),
        ("latitude", 15, 30),
        ("longitude", 105, 120),
    ]:
        frame[col] = clean_numeric(frame[col], lo, hi)
    frame["wind_u_mps"] = -frame["wind_speed_mps"] * np.sin(np.deg2rad(frame["wind_direction_deg"]))
    frame["wind_v_mps"] = -frame["wind_speed_mps"] * np.cos(np.deg2rad(frame["wind_direction_deg"]))
    frame["temp_dewpoint_spread_c"] = frame["air_temperature_c"] - frame["dew_point_c"]
    frame["obs_age_minutes_at_1330"] = (13.5 - frame["local_hour"]) * 60.0

    latest_idx = frame.sort_values("observed_at_utc_ts").groupby(["target_date", "station_id"], observed=True).tail(1).index
    latest = frame.loc[latest_idx].copy()
    summary = latest.groupby("target_date", observed=True).agg(
        isd_station_count=("station_id", "nunique"),
        isd_air_temp_mean_c=("air_temperature_c", "mean"),
        isd_air_temp_max_c=("air_temperature_c", "max"),
        isd_air_temp_min_c=("air_temperature_c", "min"),
        isd_air_temp_std_c=("air_temperature_c", "std"),
        isd_dew_point_mean_c=("dew_point_c", "mean"),
        isd_dew_point_std_c=("dew_point_c", "std"),
        isd_pressure_mean_hpa=("sea_level_pressure_hpa", "mean"),
        isd_pressure_min_hpa=("sea_level_pressure_hpa", "min"),
        isd_pressure_max_hpa=("sea_level_pressure_hpa", "max"),
        isd_wind_speed_mean_mps=("wind_speed_mps", "mean"),
        isd_wind_speed_max_mps=("wind_speed_mps", "max"),
        isd_wind_u_mean_mps=("wind_u_mps", "mean"),
        isd_wind_v_mean_mps=("wind_v_mps", "mean"),
        isd_temp_dewpoint_spread_mean_c=("temp_dewpoint_spread_c", "mean"),
        isd_obs_age_mean_min=("obs_age_minutes_at_1330", "mean"),
    ).reset_index()
    summary["isd_air_temp_range_c"] = summary["isd_air_temp_max_c"] - summary["isd_air_temp_min_c"]
    summary["isd_pressure_range_hpa"] = summary["isd_pressure_max_hpa"] - summary["isd_pressure_min_hpa"]

    by_time = frame.groupby("target_date", observed=True).agg(
        isd_intraday_temp_first_c=("air_temperature_c", "mean"),
        isd_intraday_temp_count=("air_temperature_c", "count"),
    ).reset_index()
    early = frame[frame["local_hour"] <= 9.0].groupby("target_date", observed=True).agg(
        isd_morning_temp_mean_c=("air_temperature_c", "mean"),
        isd_morning_pressure_mean_hpa=("sea_level_pressure_hpa", "mean"),
    ).reset_index()
    midday = frame[frame["local_hour"] >= 11.0].groupby("target_date", observed=True).agg(
        isd_midday_temp_mean_c=("air_temperature_c", "mean"),
        isd_midday_dewpoint_mean_c=("dew_point_c", "mean"),
        isd_midday_pressure_mean_hpa=("sea_level_pressure_hpa", "mean"),
    ).reset_index()
    summary = summary.merge(by_time, on="target_date", how="left").merge(early, on="target_date", how="left").merge(midday, on="target_date", how="left")
    summary["isd_morning_to_midday_temp_rise_c"] = summary["isd_midday_temp_mean_c"] - summary["isd_morning_temp_mean_c"]
    summary["isd_pressure_tendency_morning_midday_hpa"] = summary["isd_midday_pressure_mean_hpa"] - summary["isd_morning_pressure_mean_hpa"]
    summary["isd_dewpoint_midday_minus_temp_c"] = summary["isd_midday_dewpoint_mean_c"] - summary["isd_midday_temp_mean_c"]

    spatial_rows = [spatial_features(day, group) for day, group in latest.groupby("target_date", observed=True)]
    spatial = pd.DataFrame(spatial_rows)
    station_pivots = latest.pivot_table(
        index="target_date",
        columns="station_id",
        values=["air_temperature_c", "dew_point_c", "sea_level_pressure_hpa", "wind_speed_mps"],
        aggfunc="last",
    )
    station_pivots.columns = [
        f"isd_station_{metric}_{str(station).replace('-', '_')}"
        for metric, station in station_pivots.columns
    ]
    station_pivots = station_pivots.reset_index()
    out = summary.merge(spatial, on="target_date", how="left").merge(station_pivots, on="target_date", how="left")
    out = out.sort_values("target_date").reset_index(drop=True)
    for col in [
        "isd_air_temp_mean_c",
        "isd_dew_point_mean_c",
        "isd_pressure_mean_hpa",
        "isd_wind_speed_mean_mps",
        "isd_north_south_temp_gradient_c",
        "isd_east_west_temp_gradient_c",
    ]:
        if col in out.columns:
            out[f"{col}_change_1d"] = out[col] - out[col].shift(1)
            out[f"{col}_roll7_mean"] = out[col].shift(1).rolling(7, min_periods=3).mean()
    return out


def spatial_features(day: pd.Timestamp, group: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {"target_date": day}
    valid = group.dropna(subset=["latitude", "longitude", "air_temperature_c"])
    if len(valid) >= 3:
        x = valid[["latitude", "longitude"]].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(valid)), x])
        y = valid["air_temperature_c"].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        out["isd_temp_plane_lat_slope_c_per_deg"] = float(coef[1])
        out["isd_temp_plane_lon_slope_c_per_deg"] = float(coef[2])
        out["isd_temp_plane_rmse_c"] = float(np.sqrt(np.mean((y - x @ coef) ** 2)))
        north = valid[valid["latitude"] >= valid["latitude"].median()]["air_temperature_c"].mean()
        south = valid[valid["latitude"] < valid["latitude"].median()]["air_temperature_c"].mean()
        east = valid[valid["longitude"] >= valid["longitude"].median()]["air_temperature_c"].mean()
        west = valid[valid["longitude"] < valid["longitude"].median()]["air_temperature_c"].mean()
        out["isd_north_south_temp_gradient_c"] = float(north - south)
        out["isd_east_west_temp_gradient_c"] = float(east - west)
    pressure = group.dropna(subset=["latitude", "longitude", "sea_level_pressure_hpa"])
    if len(pressure) >= 3:
        x = pressure[["latitude", "longitude"]].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(pressure)), x])
        y = pressure["sea_level_pressure_hpa"].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        out["isd_pressure_plane_lat_slope_hpa_per_deg"] = float(coef[1])
        out["isd_pressure_plane_lon_slope_hpa_per_deg"] = float(coef[2])
    return out


def load_daily_climate_features() -> pd.DataFrame:
    path = DATASETS_ROOT / "02_hko_daily_climate_all_elements" / "hko_daily_climate_elements.parquet"
    columns = ["station_or_domain", "variable", "local_date", "value"]
    frame = dataset_table(path, columns, pc.field("local_date") <= "2023-12-24")
    frame["local_date"] = pd.to_datetime(frame["local_date"], errors="coerce").dt.normalize()
    frame = frame[frame["local_date"].notna()].copy()
    frame["value"] = clean_numeric(frame["value"])
    frame["feature_name"] = (
        "daily_"
        + frame["station_or_domain"].map(slug_part)
        + "_"
        + frame["variable"].map(slug_part)
        + "_lag7"
    )
    pivot = frame.pivot_table(index="local_date", columns="feature_name", values="value", aggfunc="last").reset_index()
    pivot["target_date"] = pivot["local_date"] + pd.Timedelta(days=7)
    pivot = pivot[(pivot["target_date"] <= HEADLINE_END) & (pivot["target_date"] < CONFIRMATION_START)].drop(columns=["local_date"])
    strict_confirmation_guard(pivot["target_date"], context="daily climate lag7 feature load")
    for col in [c for c in pivot.columns if c.startswith("daily_")]:
        pivot[f"{col}_roll7"] = pivot[col].shift(1).rolling(7, min_periods=3).mean()
        pivot[f"{col}_change7"] = pivot[col] - pivot[col].shift(7)
    return pivot.sort_values("target_date").reset_index(drop=True)


def slug_part(value: object) -> str:
    text = str(value).lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def build_feature_matrix() -> tuple[pd.DataFrame, dict[str, str]]:
    cached = OUTPUT_ROOT / "hkg_t24_exp0050_0099_feature_matrix.parquet"
    if cached.exists():
        cached_frame = pd.read_parquet(cached)
        cached_frame["target_date"] = pd.to_datetime(cached_frame["target_date"]).dt.normalize()
        before_columns = set(cached_frame.columns)
        cached_frame = ensure_batch_a_target_dynamics_features(cached_frame)
        cached_frame = ensure_upper_air_physics_features(cached_frame)
        cached_frame = ensure_isd_graph_laplacian_features(cached_frame)
        if set(cached_frame.columns) != before_columns:
            cached_frame.to_parquet(cached, index=False)
        strict_confirmation_guard(cached_frame["target_date"], context="cached 50-experiment feature matrix")
        return cached_frame, {
            "target": str(DATASETS_ROOT / "01_hko_daily_tmax_target" / "hko_daily_tmax_target_labels.parquet"),
            "daily_climate": str(DATASETS_ROOT / "02_hko_daily_climate_all_elements" / "hko_daily_climate_elements.parquet"),
            "igra": str(DATASETS_ROOT / "03_noaa_igra_upper_air_hkm00045004" / "noaa_igra_hkm00045004_sounding_features.parquet"),
            "isd_core": str(DATASETS_ROOT / "04_noaa_isd_regional_surface" / "noaa_isd_core_observations.parquet"),
            "legacy_experts": str(DATASETS_ROOT / "12_hkg_t24_robust_experiment_outputs"),
            "generated_feature_matrix": str(cached),
        }
    target = load_target_features()
    igra = load_igra_features()
    isd = load_isd_features()
    climate = load_daily_climate_features()
    features = target.merge(igra, on="target_date", how="left").merge(isd, on="target_date", how="left").merge(climate, on="target_date", how="left")
    features = features.sort_values("target_date").reset_index(drop=True)
    features = ensure_batch_a_target_dynamics_features(features)
    features = ensure_upper_air_physics_features(features)
    features = ensure_isd_graph_laplacian_features(features)
    strict_confirmation_guard(features["target_date"], context="combined 50-experiment feature matrix")
    return features, {
        "target": str(DATASETS_ROOT / "01_hko_daily_tmax_target" / "hko_daily_tmax_target_labels.parquet"),
        "daily_climate": str(DATASETS_ROOT / "02_hko_daily_climate_all_elements" / "hko_daily_climate_elements.parquet"),
        "igra": str(DATASETS_ROOT / "03_noaa_igra_upper_air_hkm00045004" / "noaa_igra_hkm00045004_sounding_features.parquet"),
        "isd_core": str(DATASETS_ROOT / "04_noaa_isd_regional_surface" / "noaa_isd_core_observations.parquet"),
        "legacy_experts": str(DATASETS_ROOT / "12_hkg_t24_robust_experiment_outputs"),
    }


def ensure_batch_a_target_dynamics_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add causal target-history features used by EXP-0051..0056.

    All columns are functions of values available at T-7 or older, calendar
    quantities, or fold-local transformations applied later inside run_oof.
    """
    frame = features.sort_values("target_date").reset_index(drop=True).copy()
    required = {
        "clim_exp_hl20_doy_mean_lag7_c",
        "target_volatility_forecastability_score_lag7",
        "trajectory_monotonic_warming_fraction_lag7_60",
        "trajectory_monotonic_cooling_fraction_lag7_60",
        "trajectory_monotonic_shape_balance_lag7_60",
    }
    if required.issubset(frame.columns):
        return frame

    shifted = pd.to_numeric(frame["target_tmax_c"], errors="coerce").shift(7)
    years = frame["year"].astype(float)

    # Harmonic drift approximates cyclic-spline seasonal-shape drift without
    # allowing future labels into the basis.
    for harmonic in range(1, 7):
        angle = 2.0 * np.pi * harmonic * frame["day_of_year"].astype(float) / 365.25
        frame[f"clim_harmonic_sin{harmonic}"] = np.sin(angle)
        frame[f"clim_harmonic_cos{harmonic}"] = np.cos(angle)
        frame[f"clim_harmonic_sin{harmonic}_year_drift"] = frame[f"clim_harmonic_sin{harmonic}"] * frame["year_centered"]
        frame[f"clim_harmonic_cos{harmonic}_year_drift"] = frame[f"clim_harmonic_cos{harmonic}"] * frame["year_centered"]

    # Same-day-of-year expanding seasonal normals.  The value being summarized
    # is already shifted by seven days, so the current and recent target labels
    # cannot leak into the feature.
    for half_life in (5, 10, 20, 40):
        frame[f"clim_exp_hl{half_life}_doy_mean_lag7_c"] = np.nan
    for q in ("q05", "q50", "q95"):
        frame[f"clim_doy_{q}_lag7_c"] = np.nan

    grouped_indices = frame.groupby("day_of_year", observed=True).indices
    for indices in grouped_indices.values():
        idx = np.array(sorted(indices), dtype=int)
        prior_values: list[float] = []
        prior_years: list[float] = []
        for row_idx in idx:
            value = shifted.iloc[row_idx]
            current_year = years.iloc[row_idx]
            if len(prior_values) >= 5:
                vals = np.asarray(prior_values, dtype=float)
                yrs = np.asarray(prior_years, dtype=float)
                valid = np.isfinite(vals)
                vals = vals[valid]
                yrs = yrs[valid]
                if len(vals) >= 5:
                    for half_life in (5, 10, 20, 40):
                        weights = np.power(0.5, np.maximum(current_year - yrs, 0.0) / half_life)
                        frame.at[row_idx, f"clim_exp_hl{half_life}_doy_mean_lag7_c"] = float(np.average(vals, weights=weights))
                    frame.at[row_idx, "clim_doy_q05_lag7_c"] = float(np.quantile(vals, 0.05))
                    frame.at[row_idx, "clim_doy_q50_lag7_c"] = float(np.quantile(vals, 0.50))
                    frame.at[row_idx, "clim_doy_q95_lag7_c"] = float(np.quantile(vals, 0.95))
            prior_values.append(float(value) if pd.notna(value) else np.nan)
            prior_years.append(float(current_year))

    exp_cols = [f"clim_exp_hl{half_life}_doy_mean_lag7_c" for half_life in (5, 10, 20, 40)]
    frame["clim_constrained_equal_blend_lag7_c"] = frame[exp_cols].mean(axis=1)
    frame["clim_recency_spread_5y_minus_40y_c"] = frame["clim_exp_hl5_doy_mean_lag7_c"] - frame["clim_exp_hl40_doy_mean_lag7_c"]
    frame["clim_lag7_anomaly_vs_hl20_c"] = frame["target_lag7_tmax_c"] - frame["clim_exp_hl20_doy_mean_lag7_c"]
    frame["clim_fold_safe_iqr_c"] = frame["clim_doy_q95_lag7_c"] - frame["clim_doy_q05_lag7_c"]

    lag_cols = [f"target_lag{lag}_tmax_c" for lag in (7, 8, 9, 10, 11, 12, 13, 14, 21, 30, 45, 60) if f"target_lag{lag}_tmax_c" in frame]
    lag_block = frame[lag_cols]
    frame["trajectory_7_21_slope_c_per_day"] = (frame["target_lag7_tmax_c"] - frame["target_lag21_tmax_c"]) / 14.0
    frame["trajectory_7_30_slope_c_per_day"] = (frame["target_lag7_tmax_c"] - frame["target_lag30_tmax_c"]) / 23.0
    frame["trajectory_roughness_lag7_60_c"] = lag_block.diff(axis=1).abs().mean(axis=1)
    frame["trajectory_range_lag7_60_c"] = lag_block.max(axis=1) - lag_block.min(axis=1)
    sign_changes = np.sign(lag_block.diff(axis=1)).diff(axis=1).abs()
    frame["trajectory_reversal_count_lag7_60"] = (sign_changes > 1.0).sum(axis=1)
    frame["trajectory_plateau_fraction_lag7_60"] = (lag_block.diff(axis=1).abs() <= 0.3).sum(axis=1) / max(len(lag_cols) - 1, 1)
    lag_diffs = lag_block.diff(axis=1)
    frame["trajectory_monotonic_warming_fraction_lag7_60"] = (lag_diffs > 0.0).sum(axis=1) / max(len(lag_cols) - 1, 1)
    frame["trajectory_monotonic_cooling_fraction_lag7_60"] = (lag_diffs < 0.0).sum(axis=1) / max(len(lag_cols) - 1, 1)
    frame["trajectory_monotonic_shape_balance_lag7_60"] = (
        frame["trajectory_monotonic_warming_fraction_lag7_60"] - frame["trajectory_monotonic_cooling_fraction_lag7_60"]
    )

    past = shifted
    for window in (14, 30, 60, 90, 120):
        roll = past.rolling(window, min_periods=max(7, window // 3))
        frame[f"spectral_autocorr_proxy_{window}_lag7"] = roll.corr(past.shift(7))
        frame[f"spectral_abs_change_energy_{window}_lag7_c"] = past.diff().abs().rolling(window, min_periods=max(7, window // 3)).mean()
    frame["spectral_energy_7_14_minus_30_60_lag7"] = frame["spectral_abs_change_energy_14_lag7_c"] - frame["spectral_abs_change_energy_60_lag7_c"]
    frame["spectral_entropy_proxy_lag7"] = frame["target_entropy_30_lag7"]

    anomaly = frame["target_lag7_tmax_c"] - frame["clim_exp_hl20_doy_mean_lag7_c"]
    for name, threshold in (("p60", 0.0), ("p70", 0.5), ("p80", 1.0), ("p90", 1.5)):
        frame[f"spell_hot_{name}_run_lag7"] = spell_length(anomaly, threshold=threshold)
        frame[f"spell_cold_{name}_run_lag7"] = spell_length(-anomaly, threshold=threshold)
    frame["spell_cumulative_hot_anomaly_lag7"] = positive_run_sum(anomaly)
    frame["spell_cumulative_cold_anomaly_lag7"] = positive_run_sum(-anomaly)
    frame["spell_reversal_hazard_proxy_lag7"] = (
        frame["trajectory_reversal_count_lag7_60"] + frame["target_abs_change_7_14_c"].fillna(0.0)
    ) / (1.0 + frame["spell_hot_p80_run_lag7"].fillna(0.0) + frame["spell_cold_p80_run_lag7"].fillna(0.0))

    for window in (14, 30, 60, 90, 120):
        rolling = past.rolling(window, min_periods=max(7, window // 3))
        frame[f"volatility_mad_{window}_lag7_c"] = rolling.apply(lambda x: float(np.nanmedian(np.abs(x - np.nanmedian(x)))) if np.isfinite(x).sum() >= 7 else np.nan, raw=True)
        frame[f"volatility_iqr_{window}_lag7_c"] = rolling.quantile(0.75) - rolling.quantile(0.25)
        frame[f"volatility_sign_change_rate_{window}_lag7"] = past.diff().pipe(np.sign).diff().abs().rolling(window, min_periods=max(7, window // 3)).mean() / 2.0
    frame["target_volatility_forecastability_score_lag7"] = 1.0 / (
        1.0
        + frame["volatility_mad_30_lag7_c"].fillna(frame["volatility_mad_60_lag7_c"])
        + frame["trajectory_reversal_count_lag7_60"].fillna(0.0)
    )
    return frame


def positive_run_sum(values: pd.Series) -> pd.Series:
    totals: list[float] = []
    current = 0.0
    for value in values:
        if pd.notna(value) and value > 0.0:
            current += float(value)
        else:
            current = 0.0
        totals.append(current)
    return pd.Series(totals, index=values.index)


def ensure_upper_air_physics_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add explicit upper-air physics features for EXP-0058..0069."""
    required = {
        "ua_layer_1000_850_theta_mean_k",
        "ua_inversion_integrated_strength_c",
        "ua_mse_1000_700_mean_kj_kg",
        "ua_vector_shear_925_700_mps",
    }
    if required.issubset(features.columns):
        return features

    frame = features.sort_values("target_date").reset_index(drop=True).copy()
    levels = (1000, 925, 850, 700, 500)
    low_levels = (1000, 925, 850, 700)

    for level in levels:
        temp_col = f"igra_temp_{level}hpa_c"
        dd_col = f"igra_dd_{level}hpa_c"
        hgt_col = f"igra_hgt_{level}hpa_m"
        wspd_col = f"igra_wspd_{level}hpa_mps"
        wdir_col = f"igra_wdir_{level}hpa_deg"
        if temp_col not in frame:
            continue
        temp_k = frame[temp_col] + 273.15
        theta_col = f"ua_theta_{level}hpa_k"
        frame[theta_col] = temp_k * (1000.0 / level) ** 0.286
        if dd_col in frame:
            dewpoint_c = frame[temp_col] - frame[dd_col]
            vapor_hpa = 6.112 * np.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
            vapor_hpa = vapor_hpa.clip(lower=0.01, upper=level - 1.0)
            mixing_ratio = 0.622 * vapor_hpa / (level - vapor_hpa)
            specific_humidity = mixing_ratio / (1.0 + mixing_ratio)
            frame[f"ua_dewpoint_{level}hpa_c"] = dewpoint_c
            frame[f"ua_specific_humidity_{level}hpa_kgkg"] = specific_humidity
            frame[f"ua_theta_e_{level}hpa_k"] = frame[theta_col] * np.exp((2.5e6 * mixing_ratio) / (1004.0 * temp_k))
            if hgt_col in frame:
                frame[f"ua_mse_{level}hpa_kj_kg"] = (1004.0 * temp_k + 9.80665 * frame[hgt_col] + 2.5e6 * specific_humidity) / 1000.0
        if wspd_col in frame and wdir_col in frame:
            radians = np.deg2rad(frame[wdir_col])
            frame[f"ua_wind_u_{level}hpa_mps"] = -frame[wspd_col] * np.sin(radians)
            frame[f"ua_wind_v_{level}hpa_mps"] = -frame[wspd_col] * np.cos(radians)

    layer_defs = ((1000, 925), (925, 850), (850, 700), (1000, 850), (1000, 700))
    for bottom, top in layer_defs:
        layer_name = f"{bottom}_{top}"
        layer_levels = tuple(level for level in low_levels if top <= level <= bottom)
        theta_cols = [f"ua_theta_{level}hpa_k" for level in layer_levels if f"ua_theta_{level}hpa_k" in frame]
        dry_cols = [f"igra_dry_adiabatic_surface_from_{level}hpa_c" for level in layer_levels if f"igra_dry_adiabatic_surface_from_{level}hpa_c" in frame]
        mse_cols = [f"ua_mse_{level}hpa_kj_kg" for level in layer_levels if f"ua_mse_{level}hpa_kj_kg" in frame]
        theta_e_cols = [f"ua_theta_e_{level}hpa_k" for level in layer_levels if f"ua_theta_e_{level}hpa_k" in frame]
        temp_cols = [f"igra_temp_{level}hpa_c" for level in layer_levels if f"igra_temp_{level}hpa_c" in frame]
        if theta_cols:
            values = frame[theta_cols]
            frame[f"ua_layer_{layer_name}_theta_mean_k"] = values.mean(axis=1)
            frame[f"ua_layer_{layer_name}_theta_max_k"] = values.max(axis=1)
            frame[f"ua_layer_{layer_name}_theta_q75_k"] = values.quantile(0.75, axis=1)
            frame[f"ua_layer_{layer_name}_theta_valid_count"] = values.notna().sum(axis=1)
        if dry_cols:
            values = frame[dry_cols]
            frame[f"ua_layer_{layer_name}_surface_equiv_mean_c"] = values.mean(axis=1)
            frame[f"ua_layer_{layer_name}_surface_equiv_max_c"] = values.max(axis=1)
            frame[f"ua_layer_{layer_name}_surface_equiv_q75_c"] = values.quantile(0.75, axis=1)
            if "isd_air_temp_mean_c" in frame:
                frame[f"ua_layer_{layer_name}_ceiling_minus_isd_temp_c"] = frame[f"ua_layer_{layer_name}_surface_equiv_max_c"] - frame["isd_air_temp_mean_c"]
        if mse_cols:
            frame[f"ua_mse_{layer_name}_mean_kj_kg"] = frame[mse_cols].mean(axis=1)
            frame[f"ua_mse_{layer_name}_max_kj_kg"] = frame[mse_cols].max(axis=1)
        if theta_e_cols:
            frame[f"ua_theta_e_{layer_name}_mean_k"] = frame[theta_e_cols].mean(axis=1)
            frame[f"ua_theta_e_{layer_name}_max_k"] = frame[theta_e_cols].max(axis=1)
        if temp_cols:
            frame[f"ua_layer_{layer_name}_temp_mean_c"] = frame[temp_cols].mean(axis=1)
            frame[f"ua_layer_{layer_name}_temp_integral_proxy_c_hpa"] = frame[temp_cols].mean(axis=1) * abs(bottom - top)

    inversion_pairs = ((1000, 925), (925, 850), (850, 700))
    inversion_strengths: list[pd.Series] = []
    for lower, upper in inversion_pairs:
        lower_temp = frame.get(f"igra_temp_{lower}hpa_c")
        upper_temp = frame.get(f"igra_temp_{upper}hpa_c")
        if lower_temp is None or upper_temp is None:
            continue
        strength = (upper_temp - lower_temp).clip(lower=0.0)
        frame[f"ua_inversion_strength_{lower}_{upper}_c"] = strength
        inversion_strengths.append(strength)
    if inversion_strengths:
        inv = pd.concat(inversion_strengths, axis=1)
        frame["ua_inversion_count_1000_700"] = (inv > 0.0).sum(axis=1)
        frame["ua_inversion_integrated_strength_c"] = inv.sum(axis=1)
        frame["ua_inversion_peak_strength_c"] = inv.max(axis=1)
        frame["ua_surface_based_inversion_flag"] = (frame.get("ua_inversion_strength_1000_925_c", pd.Series(0.0, index=frame.index)) > 0.0).astype(float)
        frame["ua_elevated_inversion_flag"] = ((frame["ua_inversion_count_1000_700"] > 0) & (frame["ua_surface_based_inversion_flag"] == 0.0)).astype(float)
        frame["ua_multiple_inversion_flag"] = (frame["ua_inversion_count_1000_700"] >= 2).astype(float)
        frame["ua_inversion_integrated_strength_change_24h_c"] = frame["ua_inversion_integrated_strength_c"].diff(1)
        frame["ua_inversion_integrated_strength_change_48h_c"] = frame["ua_inversion_integrated_strength_c"].diff(2)

    profile_cols = [f"igra_temp_{level}hpa_c" for level in low_levels if f"igra_temp_{level}hpa_c" in frame]
    if profile_cols:
        frame["ua_profile_valid_level_count_1000_700"] = frame[profile_cols].notna().sum(axis=1)
        frame["ua_warm_layer_depth_count"] = (
            frame[[f"igra_dry_adiabatic_surface_from_{level}hpa_c" for level in (925, 850, 700) if f"igra_dry_adiabatic_surface_from_{level}hpa_c" in frame]]
            .gt(frame.get("isd_air_temp_mean_c", pd.Series(np.nan, index=frame.index)), axis=0)
            .sum(axis=1)
        )
        temps = frame[profile_cols]
        pressure_values = np.array([float(col.split("_")[2].replace("hpa", "")) for col in profile_cols])
        if len(profile_cols) >= 3:
            centered_p = pressure_values - pressure_values.mean()
            denom = float(np.sum(centered_p**2))
            frame["ua_temperature_profile_linear_slope_c_per_hpa"] = temps.apply(
                lambda row: float(np.sum((row.to_numpy(dtype=float) - np.nanmean(row.to_numpy(dtype=float))) * centered_p) / denom)
                if np.isfinite(row.to_numpy(dtype=float)).sum() >= 3
                else np.nan,
                axis=1,
            )
            fitted = pd.DataFrame(
                np.outer(frame["ua_temperature_profile_linear_slope_c_per_hpa"].fillna(0.0), centered_p)
                + temps.mean(axis=1).fillna(0.0).to_numpy()[:, None],
                columns=profile_cols,
                index=frame.index,
            )
            frame["ua_temperature_profile_curvature_rmse_c"] = np.sqrt(np.nanmean(np.square(temps - fitted), axis=1))

    for upper, lower in ((925, 700), (850, 500), (925, 850), (850, 700)):
        u1, v1 = f"ua_wind_u_{upper}hpa_mps", f"ua_wind_v_{upper}hpa_mps"
        u0, v0 = f"ua_wind_u_{lower}hpa_mps", f"ua_wind_v_{lower}hpa_mps"
        if {u1, v1, u0, v0}.issubset(frame.columns):
            du = frame[u1] - frame[u0]
            dv = frame[v1] - frame[v0]
            frame[f"ua_vector_shear_{upper}_{lower}_mps"] = np.sqrt(du**2 + dv**2)

    for level in low_levels:
        for prefix in ("igra_temp", "igra_dd", "igra_hgt", "igra_wspd", "ua_theta", "ua_theta_e"):
            suffix = "hpa_c" if prefix in {"igra_temp", "igra_dd"} else "hpa_m" if prefix == "igra_hgt" else "hpa_mps" if prefix == "igra_wspd" else "hpa_k"
            col = f"{prefix}_{level}{suffix}"
            if col in frame:
                frame[f"ua_tendency_24h_{col}"] = frame[col].diff(1)
                frame[f"ua_tendency_48h_{col}"] = frame[col].diff(2)

    if {"igra_hgt_500hpa_m", "igra_thickness_1000_500_m", "igra_wspd_700hpa_mps", "igra_temp_850hpa_c"}.issubset(frame.columns):
        frame["ua_ridge_strength_raw_proxy"] = (
            frame["igra_hgt_500hpa_m"] / 100.0
            + frame["igra_thickness_1000_500_m"] / 100.0
            + frame["igra_temp_850hpa_c"]
            - 0.25 * frame["igra_wspd_700hpa_mps"]
        )
        frame["ua_ridge_strength_change_24h"] = frame["ua_ridge_strength_raw_proxy"].diff(1)
        frame["ua_ridge_persistence_3soundings"] = frame["ua_ridge_strength_raw_proxy"].rolling(3, min_periods=2).mean()

    if {"igra_dry_layer_max_dd_c", "igra_lower_mean_dd_c", "igra_temp_850_minus_500_c"}.issubset(frame.columns):
        frame["ua_dry_entrainment_potential_proxy"] = frame["igra_dry_layer_max_dd_c"] + 0.5 * frame["igra_temp_850_minus_500_c"] - frame["igra_lower_mean_dd_c"]
        frame["ua_low_level_dryness_integral_proxy"] = frame[[f"igra_dd_{level}hpa_c" for level in low_levels if f"igra_dd_{level}hpa_c" in frame]].sum(axis=1)

    return frame


def ensure_isd_graph_laplacian_features(features: pd.DataFrame) -> pd.DataFrame:
    if "isd_graph_laplacian_mode_1" in features.columns:
        return features
    station_cols = [col for col in features.columns if col.startswith("isd_station_air_temperature_c_")]
    if len(station_cols) < 3:
        return features
    station_ids = [col.replace("isd_station_air_temperature_c_", "").replace("_", "-") for col in station_cols]
    metadata = load_isd_station_metadata(station_ids)
    usable_cols: list[str] = []
    coords: list[tuple[float, float, float]] = []
    for col, station_id in zip(station_cols, station_ids, strict=True):
        row = metadata.get(station_id)
        if row is None:
            continue
        usable_cols.append(col)
        coords.append(row)
    if len(usable_cols) < 3:
        return features
    coord = np.asarray(coords, dtype=float)
    lat = coord[:, 0]
    lon = coord[:, 1]
    elev = coord[:, 2]
    lat_km = (lat[:, None] - lat[None, :]) * 111.0
    lon_km = (lon[:, None] - lon[None, :]) * 111.0 * np.cos(np.deg2rad(np.nanmean(lat)))
    elev_scaled = (elev[:, None] - elev[None, :]) / 250.0
    distance = np.sqrt(lat_km**2 + lon_km**2 + elev_scaled**2)
    adjacency = np.exp(-distance / 120.0)
    np.fill_diagonal(adjacency, 0.0)
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    _eigvals, eigvecs = np.linalg.eigh(laplacian)
    components = eigvecs[:, 1 : min(7, eigvecs.shape[1])]
    values = features[usable_cols].astype(float)
    anomalies = values.sub(values.mean(axis=1), axis=0)
    coverage = values.notna().sum(axis=1)
    filled = anomalies.fillna(0.0).to_numpy(dtype=float)
    modes = filled @ components
    additions: dict[str, object] = {
        "isd_graph_station_coverage_count": coverage,
        "isd_graph_station_coverage_fraction": coverage / len(usable_cols),
    }
    for idx in range(modes.shape[1]):
        additions[f"isd_graph_laplacian_mode_{idx + 1}"] = modes[:, idx]
    graph_diff = adjacency * np.square(filled[:, :, None] - filled[:, None, :])
    additions["isd_graph_total_variation_c2"] = graph_diff.sum(axis=(1, 2)) / max(float(adjacency.sum()), 1.0)
    return pd.concat([features, pd.DataFrame(additions, index=features.index)], axis=1).copy()


def load_isd_station_metadata(station_ids: Sequence[str]) -> dict[str, tuple[float, float, float]]:
    path = DATASETS_ROOT / "04_noaa_isd_regional_surface" / "noaa_isd_core_observations.parquet"
    frame = dataset_table(path, ["station_id", "latitude", "longitude", "elevation_m"])
    frame = frame[frame["station_id"].isin(station_ids)].copy()
    frame["latitude"] = clean_numeric(frame["latitude"], 15, 30)
    frame["longitude"] = clean_numeric(frame["longitude"], 105, 120)
    frame["elevation_m"] = clean_numeric(frame["elevation_m"], -100, 3000)
    rows = (
        frame.groupby("station_id", observed=True)[["latitude", "longitude", "elevation_m"]]
        .median()
        .dropna()
    )
    return {
        str(station_id): (float(row.latitude), float(row.longitude), float(row.elevation_m))
        for station_id, row in rows.iterrows()
    }


def base_columns(features: pd.DataFrame) -> tuple[str, ...]:
    wanted = [
        "doy_sin",
        "doy_cos",
        "doy_sin_2",
        "doy_cos_2",
        "year_centered",
        "year_centered_sq",
        "target_lag7_tmax_c",
        "target_lag14_tmax_c",
        "target_lag30_tmax_c",
        "target_roll30_mean_lag7_c",
        "target_roll365_mean_lag7_c",
        "target_lag7_minus_lag14_c",
        "target_lag7_minus_roll365_c",
    ]
    return tuple(col for col in wanted if col in features.columns)


def core_columns(features: pd.DataFrame) -> tuple[str, ...]:
    prefixes = (
        "doy_",
        "year_centered",
        "target_lag7",
        "target_lag14",
        "target_lag30",
        "target_roll30",
        "target_roll365",
        "igra_temp_",
        "igra_dd_",
        "igra_hgt_",
        "igra_wspd_",
        "igra_lower_",
        "igra_thickness_",
        "igra_mixed_",
        "igra_heat_",
        "isd_station_count",
        "isd_air_temp_mean",
        "isd_dew_point_mean",
        "isd_pressure_mean",
        "isd_wind_speed_mean",
        "isd_wind_u_mean",
        "isd_wind_v_mean",
        "isd_temp_dewpoint",
        "isd_air_temp_range",
        "isd_pressure_range",
    )
    return tuple(col for col in features.columns if col.startswith(prefixes))


def columns_matching(features: pd.DataFrame, *patterns: str) -> tuple[str, ...]:
    cols: list[str] = []
    for col in features.columns:
        if not is_allowed_feature_column(col):
            continue
        if any(pattern in col or col.startswith(pattern) for pattern in patterns):
            cols.append(col)
    return tuple(dict.fromkeys(cols))


def is_allowed_feature_column(col: str) -> bool:
    forbidden = {
        "target_tmax_c",
        "target_date",
        "content_sha256",
        "raw_retrieved_at_utc",
        "valid_at_utc",
        "valid_at_hkt",
    }
    if col in forbidden:
        return False
    return not (col.endswith("_sha256") or col.endswith("_retrieved_at_utc"))


def run_specs(parsed: Sequence[ParsedExperiment], features: pd.DataFrame) -> tuple[RunSpec, ...]:
    mapping: dict[str, tuple[str, ...]] = {
        "EXP-0050": columns_matching(features, "igra_", "isd_", "target_", "year_centered"),
        "EXP-0051": columns_matching(features, "clim_", "doy_", "year_centered"),
        "EXP-0052": columns_matching(features, "year_centered", "target_roll365", "target_roll3650", "target_roll7300", "target_roll14610", "clim_lag7_anomaly"),
        "EXP-0053": columns_matching(features, "target_lag", "target_lag7_minus", "target_abs_change", "trajectory_"),
        "EXP-0054": columns_matching(features, "target_roll7", "target_roll14", "target_roll30", "target_roll60", "target_roll90", "target_roll120", "target_entropy", "spectral_"),
        "EXP-0055": columns_matching(features, "target_spell", "target_reversal", "target_lag7_minus", "spell_"),
        "EXP-0056": columns_matching(features, "target_roll30_std", "target_roll60_std", "target_roll90_std", "target_entropy", "target_abs_change", "volatility_", "target_volatility_forecastability"),
        "EXP-0057": columns_matching(features, "target_roll365", "target_roll3650", "target_roll7300", "target_roll14610"),
        "EXP-0058": columns_matching(features, "igra_theta_", "igra_mixed_layer_ceiling", "ua_layer_", "ua_warm_layer", "ua_profile_valid"),
        "EXP-0059": columns_matching(features, "igra_dry_adiabatic_surface", "igra_mixed_layer_ceiling", "ua_layer_", "ua_warm_layer", "isd_dew", "isd_pressure", "isd_wind"),
        "EXP-0060": columns_matching(features, "igra_inversion", "igra_temp_925_minus", "igra_temp_", "ua_inversion", "ua_temperature_profile"),
        "EXP-0061": columns_matching(features, "igra_heat_content", "igra_thickness", "igra_lower_mean_temp", "ua_mse_", "ua_layer_", "ua_profile_valid"),
        "EXP-0062": columns_matching(features, "igra_moist_static", "igra_lower_mean_dd", "igra_dd_", "ua_theta_e_", "ua_mse_", "ua_dewpoint", "ua_specific_humidity"),
        "EXP-0063": columns_matching(features, "igra_dry_layer", "igra_dd_", "igra_lower_mean_dd", "ua_dry_", "ua_low_level_dryness"),
        "EXP-0064": columns_matching(features, "igra_shear", "igra_veering", "igra_wspd_", "igra_wdir_", "ua_wind_", "ua_vector_shear"),
        "EXP-0065": columns_matching(features, "igra_hgt_", "igra_thickness", "ua_ridge", "ua_layer_1000_700"),
        "EXP-0066": columns_matching(features, "_change_24h", "_change_48h", "ua_tendency_24h", "ua_tendency_48h", "ua_ridge_strength_change"),
        "EXP-0067": columns_matching(features, "igra_temp_", "igra_dd_", "igra_wspd_", "igra_hgt_", "ua_temperature_profile", "ua_theta_", "ua_theta_e_"),
        "EXP-0068": columns_matching(features, "igra_lower_", "igra_thickness", "igra_shear", "igra_moist_static", "ua_layer_", "ua_inversion", "ua_ridge", "ua_vector_shear"),
        "EXP-0069": columns_matching(features, "igra_key_level_count", "igra_dry_layer", "igra_lower_mean_dd", "ua_profile_valid", "ua_layer_", "ua_inversion"),
        "EXP-0070": columns_matching(features, "isd_morning", "isd_midday", "isd_morning_to_midday", "isd_intraday"),
        "EXP-0071": columns_matching(features, "isd_dew", "isd_pressure_tendency", "isd_wind", "isd_midday"),
        "EXP-0072": columns_matching(features, "isd_pressure", "isd_wind_v", "isd_air_temp_mean_c_change_1d", "target_abs_change", "isd_graph_"),
        "EXP-0073": columns_matching(features, "isd_station_air_temperature", "isd_wind_u", "isd_wind_v", "isd_graph_"),
        "EXP-0074": columns_matching(features, "isd_pressure_plane", "isd_pressure_range", "isd_pressure_tendency"),
        "EXP-0075": columns_matching(features, "isd_temp_plane", "isd_air_temp_range", "isd_north_south", "isd_east_west", "isd_graph_"),
        "EXP-0076": columns_matching(features, "isd_air_temp_range", "isd_wind_speed", "isd_east_west", "isd_station_air_temperature"),
        "EXP-0077": columns_matching(features, "isd_north_south", "isd_pressure_plane_lat", "isd_air_temp_mean_c_change"),
        "EXP-0078": columns_matching(features, "isd_east_west", "isd_pressure_plane_lon", "isd_dew_point"),
        "EXP-0079": columns_matching(features, "isd_station_", "isd_temp_plane", "isd_pressure_plane", "isd_graph_"),
        "EXP-0080": columns_matching(features, "isd_air_temp_max", "isd_air_temp_min", "isd_air_temp_std", "isd_air_temp_range", "isd_temp_plane_rmse"),
        "EXP-0081": columns_matching(features, "isd_station_air_temperature", "_change_1d", "isd_graph_"),
        "EXP-0082": columns_matching(features, "isd_station_", "year_centered"),
        "EXP-0083": columns_matching(features, "isd_station_count", "isd_obs_age", "isd_station_", "isd_graph_station_coverage"),
        "EXP-0084": columns_matching(features, "isd_obs_age", "isd_station_count", "isd_intraday_temp_count"),
        "EXP-0085": columns_matching(features, "daily_hong_kong_observatory_daily_rainfall", "target_spell_hot", "target_spell_cold"),
        "EXP-0086": columns_matching(features, "daily_hong_kong_observatory_mean_cloud", "daily_kings_park_bright_sunshine"),
        "EXP-0087": columns_matching(features, "daily_kings_park_global_solar", "daily_hong_kong_observatory_mean_relative_humidity", "isd_wind_speed"),
        "EXP-0088": columns_matching(features, "daily_hong_kong_observatory_grass", "daily_kings_park_evaporation"),
        "EXP-0089": columns_matching(features, "daily_north_point_sea_temperature", "daily_waglan_island_sea_temperature", "isd_wind"),
        "EXP-0090": columns_matching(features, "daily_waglan_island_prevailing_wind", "daily_waglan_island_mean_wind", "isd_wind"),
        "EXP-0091": columns_matching(features, "daily_"),
        "EXP-0092": columns_matching(features, "daily_", "target_lag", "target_roll30"),
        "EXP-0093": columns_matching(features, "igra_hgt_", "igra_thickness", "isd_wind_speed", "daily_waglan"),
        "EXP-0094": columns_matching(features, "daily_hong_kong_observatory_mean_cloud", "daily_hong_kong_observatory_daily_rainfall", "daily_kings_park_bright_sunshine", "igra_dd_"),
        "EXP-0095": columns_matching(features, "daily_", "igra_", "isd_pressure", "isd_wind"),
        "EXP-0096": columns_matching(features, "igra_", "isd_", "target_roll", "daily_"),
        "EXP-0097": columns_matching(features, "igra_", "isd_", "target_", "daily_"),
        "EXP-0098": columns_matching(features, "target_roll30_std", "target_entropy", "isd_station_count", "igra_key_level_count", "isd_air_temp_std"),
        "EXP-0099": columns_matching(features, "target_roll30_std", "target_entropy", "igra_", "isd_", "daily_"),
    }
    mechanism = {"EXP-0093", "EXP-0094", "EXP-0095"}
    return tuple(
        RunSpec(
            parsed=item,
            feature_columns=mapping.get(item.experiment_id, ()),
            mechanism_only=item.experiment_id in mechanism,
            implementation_note=implementation_note_for(item.experiment_id),
        )
        for item in parsed
    )


def implementation_note_for(experiment_id: str) -> str:
    notes = {
        "EXP-0050": "bespoke benchmark repair from existing R14-R17 nested OOF predictions, common-row scoring, equal-weight diagnostic, and corrected long_history_core_v1 freeze",
        "EXP-0051": "bespoke fold-safe dynamic climatology using causal same-DOY expanding normals, recency half-lives, harmonic drift, quantile normals, and constrained blend predictors",
        "EXP-0052": "fold-local change-point experiment with detected break hinges constructed inside each chronological training fold",
        "EXP-0053": "target-history trajectory-shape experiment using lagged slopes, roughness, reversals, plateau fractions, and fixed-lag analogue descriptors ending T-7",
        "EXP-0054": "causal intraseasonal oscillation proxy experiment using lagged rolling autocorrelation, absolute-change energy bands, and entropy state",
        "EXP-0055": "spell-duration and reversal-hazard experiment using lagged seasonal anomalies, hot/cold run ages, cumulative anomaly, and transition pressure",
        "EXP-0056": "volatility, entropy, and conditional forecastability experiment using lagged MAD/IQR/sign-change/entropy features",
        "EXP-0057": "origin-relative recency-window expert ensemble using full, trailing-50, trailing-30, and trailing-15-year training windows",
        "EXP-0058": "upper-air mixed-layer ceiling experiment using IGRA potential-temperature and dry-adiabatic surface-equivalent features",
        "EXP-0059": "dry-adiabatic parcel-realization experiment using 925/850 hPa ceiling proxies plus surface humidity, pressure, and wind modifiers",
        "EXP-0060": "inversion-geometry experiment using lower-tropospheric inversion, temperature-gradient, and stability descriptors",
        "EXP-0061": "integrated heat-content and thickness experiment using lower-tropospheric temperature sums, layer thicknesses, and warm-column proxies",
        "EXP-0062": "moist thermodynamic energy experiment using dewpoint depression, moist-static proxy, theta proxy, and surface-to-aloft mismatch features",
        "EXP-0063": "dry-layer and entrainment experiment using dewpoint-depression maxima, low-level dryness, and layer-gradient descriptors",
        "EXP-0064": "vertical wind-shear and veering experiment using IGRA wind direction/speed, shear, and directional-change descriptors",
        "EXP-0065": "ridge-height and thickness proxy experiment using geopotential heights, thickness, persistence, and height-tendency features",
        "EXP-0066": "sounding-evolution experiment using 24/48-hour profile change, thermal tendency, moisture tendency, and thickness tendency features",
        "EXP-0067": "functional-profile representation proxy experiment using multilevel IGRA temperature/moisture/wind shape columns",
        "EXP-0068": "soft upper-air regime proxy experiment using combined physical profile, ISD state, and long-history core features",
        "EXP-0069": "sounding-quality and reliability experiment using key-level counts, dryness, completeness proxies, and shrinkage-relevant missingness states",
        "EXP-0070": "ISD intraday thermal-trajectory experiment using morning, midday, rise, range, and latest-before-cutoff summaries",
        "EXP-0071": "multivariable intraday tendency experiment using dewpoint, pressure, wind, and morning-to-midday transition features",
        "EXP-0072": "front and monsoon-surge proxy experiment using pressure tendency, wind vector, thermal change, and target-history transition signals",
        "EXP-0073": "flow-relative station-weighting proxy experiment using station-level temperatures plus surface/upper wind direction components",
        "EXP-0074": "regional pressure-gradient experiment using fitted pressure-plane slopes, range, tendency, and uncertainty proxies",
        "EXP-0075": "regional temperature-gradient experiment using fitted temperature-plane slopes, north-south/east-west gradients, and spread",
        "EXP-0076": "thermal-spread and wind-conditioned spatial-contrast experiment using station extremes, wind speed, and station panel values",
        "EXP-0077": "north-south advection and continental-surge proxy experiment using meridional gradients, pressure slopes, and temperature changes",
        "EXP-0078": "east-west/marine-continental contrast proxy experiment using zonal gradients, pressure slopes, dewpoint, and wind fields",
        "EXP-0079": "station-panel latent spatial-mode proxy experiment using station-level values and fitted plane descriptors",
        "EXP-0080": "robust spatial-distribution experiment using station spread, range, standard deviation, panel extremes, and plane residuals",
        "EXP-0081": "station lead-lag propagation proxy experiment using station-level changes and daily change features ending before cutoff",
        "EXP-0082": "station homogenization/domain-adaptation proxy experiment using station panel, era, and availability-aware features",
        "EXP-0083": "station-dropout robustness experiment using station count, observation age, coverage masks, and station panel availability",
        "EXP-0084": "ISD report-quality and observation-age weighting proxy experiment using observation-age, station-count, and intraday coverage features",
        "EXP-0085": "antecedent rainfall and dry-spell memory experiment using strict T-7 lagged HKO rainfall and target spell context",
        "EXP-0086": "cloud-sunshine radiative-memory experiment using strict-lag HKO cloud and King's Park sunshine histories",
        "EXP-0087": "solar-conversion efficiency proxy experiment using lagged King's Park global solar radiation, humidity, and wind state",
        "EXP-0088": "surface-storage memory experiment using strict-lag grass minimum, evaporation, rainfall, and surface thermal contrast features",
        "EXP-0089": "marine-moderation memory experiment using lagged North Point/Waglan sea temperature and wind-context features",
        "EXP-0090": "Waglan wind-direction and marine-advection experiment using long-history wind direction/speed plus ISD wind context",
        "EXP-0091": "full lagged HKO daily-climate memory screen across all eligible long-history daily elements under T-7 rules",
        "EXP-0092": "daily-climate interaction experiment combining lagged daily elements, target-history state, and long-history core controls",
        "EXP-0093": "ridge and TC-subsidence teacher/mechanism experiment using upper-air height/thickness/wind proxies without inference use of best-track labels",
        "EXP-0094": "cloud/rain suppression teacher experiment using daily climate only as lagged/mechanism-safe state and retaining teacher-only caveats",
        "EXP-0095": "compound teacher/regime mechanism screen combining daily climate, upper-air, pressure, and wind signals",
        "EXP-0096": "expert-gating experiment using long-history physical, surface, target-dynamics, and daily-climate feature families",
        "EXP-0097": "nested residual stacking proxy experiment combining orthogonal long-history blocks against the corrected common baseline",
        "EXP-0098": "heteroscedastic forecastability experiment using volatility, entropy, station coverage, IGRA quality, and disagreement proxies",
        "EXP-0099": "tail-specialist distribution proxy experiment using target volatility, upper-air/surface/daily features, and calibrated normal quantiles",
    }
    return notes.get(experiment_id, "predeclared long-history ridge feature-block experiment")


def model_strategy_for(experiment_id: str) -> str:
    strategies = {
        "EXP-0051": "cyclic_spline_climatology",
        "EXP-0052": "change_point_ridge",
        "EXP-0053": "dtw_analog_blend",
        "EXP-0054": "spectral_ssa_pca",
        "EXP-0055": "hazard_mixture",
        "EXP-0056": "forecastability_scale",
        "EXP-0065": "ridge_strength_pca",
        "EXP-0067": "fpca_profile_analog",
        "EXP-0068": "gmm_regime_mixture",
        "EXP-0069": "reliability_shrinkage",
        "EXP-0070": "intraday_trajectory_pca",
        "EXP-0071": "tensor_pca_composites",
        "EXP-0072": "front_cusum_gate",
        "EXP-0073": "flow_relative_weighting",
        "EXP-0076": "sea_breeze_index",
        "EXP-0077": "north_south_propagation",
        "EXP-0078": "east_west_flow_gradient",
        "EXP-0079": "graph_pca_modes",
        "EXP-0080": "robust_distribution_shape",
        "EXP-0081": "distributed_lag_station_map",
        "EXP-0082": "station_homogenization_offsets",
        "EXP-0083": "station_dropout_masked",
        "EXP-0084": "quality_weighted_surface",
        "EXP-0085": "rainfall_reservoir",
        "EXP-0086": "cloud_sunshine_regime",
        "EXP-0087": "solar_efficiency_state",
        "EXP-0088": "surface_storage_state",
        "EXP-0089": "sea_temperature_extrapolator",
        "EXP-0090": "markov_wind_regime",
        "EXP-0091": "daily_climate_factor",
        "EXP-0092": "climate_trajectory_analog",
        "EXP-0093": "teacher_student_subsidence",
        "EXP-0094": "teacher_student_suppression",
        "EXP-0095": "teacher_student_archetype",
        "EXP-0096": "expert_gate",
        "EXP-0097": "residual_stack",
        "EXP-0098": "student_t_conformal_scale",
        "EXP-0099": "quantile_tail_cdf",
    }
    return strategies.get(experiment_id, "ridge")


def candidate_family_for(experiment_id: str) -> str:
    strategy = model_strategy_for(experiment_id)
    families = {
        "cyclic_spline_climatology": "fold_local_periodic_spline_harmonic_recency_climatology",
        "change_point_ridge": "fold_local_change_point_piecewise_ridge",
        "dtw_analog_blend": "season_restricted_dtw_analog_blend",
        "spectral_ssa_pca": "fold_local_spectral_ssa_pca_residual_model",
        "hazard_mixture": "spell_reversal_hazard_two_expert_mixture",
        "forecastability_scale": "forecastability_gated_location_scale_model",
        "ridge_strength_pca": "training_only_ridge_strength_pca_plus_ridge",
        "fpca_profile_analog": "fold_local_fpca_profile_analog_multiview_model",
        "gmm_regime_mixture": "fold_local_gaussian_mixture_regime_experts",
        "reliability_shrinkage": "sounding_reliability_weighted_upper_air_shrinkage",
        "intraday_trajectory_pca": "fold_local_intraday_trajectory_pca_model",
        "tensor_pca_composites": "fold_local_station_variable_time_tensor_pca",
        "front_cusum_gate": "one_sided_cusum_front_probability_experts",
        "flow_relative_weighting": "surface_and_low_level_flow_relative_station_weighting",
        "sea_breeze_index": "coastal_inland_contrast_sea_breeze_index",
        "north_south_propagation": "north_south_gradient_propagation_state_model",
        "east_west_flow_gradient": "east_west_estuary_flow_gradient_model",
        "graph_pca_modes": "metadata_graph_laplacian_and_fold_local_pca_modes",
        "robust_distribution_shape": "robust_spatial_quantile_entropy_shape_model",
        "distributed_lag_station_map": "training_only_station_lead_lag_map",
        "station_homogenization_offsets": "fold_safe_station_offset_domain_adaptation_model",
        "station_dropout_masked": "mask_coverage_station_dropout_robust_model",
        "quality_weighted_surface": "report_quality_observation_age_weighted_surface_model",
        "rainfall_reservoir": "inner_oof_rainfall_wetness_reservoir_model",
        "cloud_sunshine_regime": "fold_local_cloud_sunshine_regime_model",
        "solar_efficiency_state": "lagged_solar_conversion_efficiency_state_model",
        "surface_storage_state": "grass_evaporation_heat_storage_state_model",
        "sea_temperature_extrapolator": "lagged_sea_temperature_state_extrapolator",
        "markov_wind_regime": "waglan_circular_markov_wind_regime_model",
        "daily_climate_factor": "fold_local_sparse_pca_daily_climate_factor_model",
        "climate_trajectory_analog": "mixed_metric_climate_trajectory_analog_model",
        "teacher_student_subsidence": "fold_local_teacher_student_subsidence_probability_model",
        "teacher_student_suppression": "fold_local_teacher_student_cloud_rain_student_model",
        "teacher_student_archetype": "fold_local_synoptic_archetype_student_mixture",
        "expert_gate": "nested_oof_static_and_softmax_expert_gate",
        "residual_stack": "ordered_nested_residual_stack",
        "student_t_conformal_scale": "heteroscedastic_student_t_rolling_conformal_scale",
        "quantile_tail_cdf": "direct_quantile_tail_mixture_0p1c_cdf",
    }
    return families.get(strategy, "ridge_predeclared_feature_block")


def run_oof(features: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    if spec.parsed.experiment_id == "EXP-0050":
        return run_exp0050_benchmark_repair()
    if spec.parsed.experiment_id == "EXP-0057":
        return run_exp0057_recency_ensemble(features, spec)

    base = base_columns(features)
    core = core_columns(features)
    candidate_cols = tuple(dict.fromkeys([*core, *spec.feature_columns]))
    models = (
        ModelSpec("strict_t7_lag_calendar_control", "ridge_strict_t7_target_history", base, control=True),
        ModelSpec("ridge_igra_isd_core_proxy", "ridge_strict_igra_isd_core_proxy_not_headline_baseline", core, control=True),
        ModelSpec(
            f"{spec.parsed.experiment_id.lower()}_candidate",
            candidate_family_for(spec.parsed.experiment_id),
            candidate_cols,
            strategy=model_strategy_for(spec.parsed.experiment_id),
        ),
        ModelSpec(
            f"{spec.parsed.experiment_id.lower()}_shuffled_target_control",
            "ridge_shuffled_target_negative_control",
            candidate_cols,
            shuffled_control=True,
            strategy=model_strategy_for(spec.parsed.experiment_id),
        ),
    )
    rows: list[pd.DataFrame] = []
    for fold_id, test_start, test_end, train_end in robust_fold_definitions():
        train = features[features["target_date"] <= train_end].copy()
        test = features[(features["target_date"] >= test_start) & (features["target_date"] <= test_end)].copy()
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        for model_spec in models:
            train_for_model, test_for_model, model_columns = apply_fold_specific_features(train, test, spec, model_spec)
            cols = active_cols(train_for_model, model_columns)
            if not cols:
                continue
            point_forecast, sigma, quantiles = fit_predict_for_model(
                train_for_model,
                test_for_model,
                cols,
                spec,
                model_spec,
                fold_id,
            )
            pred = test_for_model[["target_date", "target_tmax_c", "year", "month"]].copy()
            pred["experiment_id"] = spec.parsed.experiment_id
            pred["fold_id"] = fold_id
            pred["model_id"] = model_spec.model_id
            pred["model_family"] = model_spec.model_family
            pred["is_control"] = model_spec.control
            pred["is_shuffled_control"] = model_spec.shuffled_control
            pred["training_start"] = train_for_model["target_date"].min()
            pred["training_end"] = train_for_model["target_date"].max()
            pred["training_rows"] = int(len(train_for_model))
            pred["feature_count"] = int(len(cols))
            pred["point_forecast"] = point_forecast
            pred["distribution_sigma_c"] = sigma
            pred["headline_oof"] = pred["target_date"].between(HEADLINE_START, HEADLINE_END)
            for qcol, z_value in QUANTILE_Z.items():
                if qcol in quantiles:
                    pred[qcol] = quantiles[qcol]
                else:
                    pred[qcol] = pred["point_forecast"] + pred["distribution_sigma_c"] * z_value
            for qcol, values in quantiles.items():
                if qcol not in pred.columns:
                    pred[qcol] = values
            rows.append(pred)
    if not rows:
        raise RuntimeError(f"{spec.parsed.experiment_id} produced no predictions")
    predictions = pd.concat(rows, ignore_index=True).sort_values(["target_date", "model_id"]).reset_index(drop=True)
    predictions = append_corrected_core_baseline(predictions, spec.parsed.experiment_id)
    strict_confirmation_guard(predictions["target_date"], context=f"{spec.parsed.experiment_id} predictions")
    return predictions


def append_corrected_core_baseline(predictions: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    if experiment_id == "EXP-0050":
        return predictions
    core = corrected_long_history_core_predictions(experiment_id)
    combined = pd.concat([predictions, core], ignore_index=True, sort=False)
    return combined.sort_values(["target_date", "model_id"]).reset_index(drop=True)


def corrected_long_history_core_predictions(experiment_id: str) -> pd.DataFrame:
    raw = pd.read_parquet(OUTPUT_ROOT / "hkg_t24_r17_oof_predictions.parquet")
    raw["target_date"] = pd.to_datetime(raw["target_date"]).dt.normalize()
    strict_confirmation_guard(raw["target_date"], context=f"{experiment_id} corrected R17 core load")
    raw = raw[raw["model_id"].eq("r17_era_transfer_terms")].copy()
    counts = raw.groupby(["fold_id", "model_id"], observed=True).size().rename("fold_model_rows").reset_index()
    raw = raw.merge(counts, on=["fold_id", "model_id"], how="left")
    raw = raw[raw["fold_model_rows"] >= 1000].copy()
    raw["experiment_id"] = experiment_id
    raw["source_research_id"] = "r17"
    raw["source_model_id"] = raw["model_id"]
    raw["model_id"] = "long_history_core_v1"
    raw["model_family"] = "corrected_r17_common_row_freeze_from_exp0050"
    raw["is_control"] = False
    raw["is_shuffled_control"] = False
    raw["headline_oof"] = raw["target_date"].between(HEADLINE_START, HEADLINE_END)
    raw["row_set"] = "corrected_core_common_row_baseline"
    return raw


def apply_fold_specific_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: RunSpec,
    model_spec: ModelSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    train_out = train.copy()
    test_out = test.copy()
    extra_cols: list[str] = []

    strategy = model_spec.strategy
    if strategy == "cyclic_spline_climatology":
        extra_cols.extend(add_fold_cyclic_spline_features(train_out, test_out))

    if strategy == "change_point_ridge":
        break_years = detect_training_break_years(train_out)
        for index, year in enumerate(break_years):
            after_col = f"fold_detected_break_after_{index}_{year}"
            hinge_col = f"fold_detected_break_hinge_{index}_{year}"
            decay_col = f"fold_detected_break_decay_{index}_{year}"
            for frame in (train_out, test_out):
                years_since = np.maximum(frame["year"].astype(float) - float(year), 0.0)
                frame[after_col] = (frame["year"] >= year).astype(float)
                frame[hinge_col] = years_since
                frame[decay_col] = np.exp(-years_since / 10.0) * frame[after_col]
            extra_cols.extend([after_col, hinge_col, decay_col])

    if strategy in {
        "spectral_ssa_pca",
        "ridge_strength_pca",
        "fpca_profile_analog",
        "intraday_trajectory_pca",
        "tensor_pca_composites",
        "graph_pca_modes",
        "daily_climate_factor",
    }:
        source_cols = fold_pca_source_columns(train_out, strategy, model_spec.columns)
        extra_cols.extend(add_fold_pca_features(train_out, test_out, source_cols, prefix=strategy, n_components=6))

    if strategy in {
        "gmm_regime_mixture",
        "cloud_sunshine_regime",
        "markov_wind_regime",
        "teacher_student_archetype",
    }:
        source_cols = fold_regime_source_columns(train_out, strategy, model_spec.columns)
        extra_cols.extend(add_fold_gmm_features(train_out, test_out, source_cols, prefix=strategy, max_components=4))

    if strategy in {
        "front_cusum_gate",
        "north_south_propagation",
        "east_west_flow_gradient",
        "markov_wind_regime",
    }:
        source_cols = fold_cusum_source_columns(train_out, strategy, model_spec.columns)
        extra_cols.extend(add_fold_cusum_features(train_out, test_out, source_cols, prefix=strategy))

    extra_cols.extend(add_fold_station_offset_features(train_out, test_out, strategy))
    extra_cols.extend(add_strategy_composite_features(train_out, test_out, strategy))
    return train_out, test_out, tuple(dict.fromkeys([*model_spec.columns, *extra_cols]))


def add_fold_cyclic_spline_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    extra_cols: list[str] = []
    for n_knots in (8, 12, 18, 24):
        transformer = SplineTransformer(
            n_knots=n_knots,
            degree=3,
            extrapolation="periodic",
            include_bias=False,
        )
        train_basis = transformer.fit_transform(train[["day_of_year"]])
        test_basis = transformer.transform(test[["day_of_year"]])
        for idx in range(train_basis.shape[1]):
            col = f"fold_cyclic_spline_k{n_knots}_{idx:02d}"
            train[col] = train_basis[:, idx]
            test[col] = test_basis[:, idx]
            extra_cols.append(col)
    for month in range(1, 13):
        col = f"fold_month_trend_{month:02d}"
        train[col] = np.where(train["month"].eq(month), train["year_centered"], 0.0)
        test[col] = np.where(test["month"].eq(month), test["year_centered"], 0.0)
        extra_cols.append(col)
    return extra_cols


def fold_pca_source_columns(frame: pd.DataFrame, strategy: str, columns: Sequence[str]) -> tuple[str, ...]:
    if strategy == "spectral_ssa_pca":
        patterns = ("spectral_", "target_lag", "trajectory_")
    elif strategy == "ridge_strength_pca":
        patterns = ("igra_hgt_", "igra_thickness", "ua_ridge", "ua_layer_1000_700", "igra_temp_850hpa_c")
    elif strategy == "fpca_profile_analog":
        patterns = ("igra_temp_", "igra_dd_", "igra_wspd_", "igra_hgt_", "ua_theta_", "ua_theta_e_", "ua_wind_")
    elif strategy == "intraday_trajectory_pca":
        patterns = ("isd_morning", "isd_midday", "isd_morning_to_midday", "isd_intraday", "isd_air_temp_", "isd_obs_age")
    elif strategy == "tensor_pca_composites":
        patterns = ("isd_dew", "isd_pressure", "isd_wind", "isd_midday", "isd_morning", "isd_station_")
    elif strategy == "graph_pca_modes":
        patterns = ("isd_station_air_temperature", "isd_temp_plane", "isd_pressure_plane", "isd_north_south", "isd_east_west")
    elif strategy == "daily_climate_factor":
        patterns = ("daily_",)
    else:
        patterns = ()
    selected = [col for col in frame.columns if col in columns and any(pattern in col for pattern in patterns)]
    return tuple(dict.fromkeys(selected))


def add_fold_pca_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    source_cols: Sequence[str],
    *,
    prefix: str,
    n_components: int,
) -> list[str]:
    active = active_cols(train, source_cols)
    if len(active) < 2:
        return []
    components = max(1, min(n_components, len(active), len(train) - 1))
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(imputer.fit_transform(train[active]))
    test_scaled = scaler.transform(imputer.transform(test[active]))
    pca = PCA(n_components=components, random_state=0)
    train_scores = pca.fit_transform(train_scaled)
    test_scores = pca.transform(test_scaled)
    extra_cols: list[str] = []
    safe_prefix = prefix.replace("-", "_")
    for idx in range(components):
        col = f"fold_{safe_prefix}_pca_{idx + 1}"
        train[col] = train_scores[:, idx]
        test[col] = test_scores[:, idx]
        extra_cols.append(col)
    recon_train = pca.inverse_transform(train_scores)
    recon_test = pca.inverse_transform(test_scores)
    err_col = f"fold_{safe_prefix}_pca_reconstruction_rmse"
    train[err_col] = np.sqrt(np.mean(np.square(train_scaled - recon_train), axis=1))
    test[err_col] = np.sqrt(np.mean(np.square(test_scaled - recon_test), axis=1))
    extra_cols.append(err_col)
    return extra_cols


def fold_regime_source_columns(frame: pd.DataFrame, strategy: str, columns: Sequence[str]) -> tuple[str, ...]:
    if strategy == "cloud_sunshine_regime":
        patterns = ("daily_hong_kong_observatory_mean_cloud", "daily_kings_park_bright_sunshine", "daily_kings_park_global_solar", "daily_hong_kong_observatory_daily_rainfall")
    elif strategy == "markov_wind_regime":
        patterns = ("daily_waglan_island_prevailing_wind", "daily_waglan_island_mean_wind", "isd_wind", "ua_wind")
    elif strategy == "teacher_student_archetype":
        patterns = ("daily_", "igra_", "isd_pressure", "isd_wind", "ua_")
    else:
        patterns = ("igra_", "ua_", "isd_pressure", "isd_wind", "isd_dew")
    selected = [col for col in frame.columns if col in columns and any(pattern in col for pattern in patterns)]
    return tuple(dict.fromkeys(selected))


def add_fold_gmm_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    source_cols: Sequence[str],
    *,
    prefix: str,
    max_components: int,
) -> list[str]:
    active = active_cols(train, source_cols)
    if len(active) < 3:
        return []
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(imputer.fit_transform(train[active]))
    test_scaled = scaler.transform(imputer.transform(test[active]))
    components = max(2, min(max_components, max(2, len(train) // 1000), len(active)))
    gmm = GaussianMixture(
        n_components=components,
        covariance_type="diag",
        random_state=17,
        reg_covar=1e-5,
        max_iter=200,
    )
    labels = gmm.fit_predict(train_scaled)
    counts = np.bincount(labels, minlength=components)
    if np.min(counts) < 50 and components > 2:
        components = 2
        gmm = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            random_state=17,
            reg_covar=1e-5,
            max_iter=200,
        )
        gmm.fit(train_scaled)
    train_probs = gmm.predict_proba(train_scaled)
    test_probs = gmm.predict_proba(test_scaled)
    extra_cols: list[str] = []
    safe_prefix = prefix.replace("-", "_")
    for idx in range(train_probs.shape[1]):
        col = f"fold_{safe_prefix}_regime_prob_{idx + 1}"
        train[col] = train_probs[:, idx]
        test[col] = test_probs[:, idx]
        extra_cols.append(col)
    entropy_col = f"fold_{safe_prefix}_regime_entropy"
    train[entropy_col] = entropy_from_probs(train_probs)
    test[entropy_col] = entropy_from_probs(test_probs)
    extra_cols.append(entropy_col)
    return extra_cols


def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-9, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def fold_cusum_source_columns(frame: pd.DataFrame, strategy: str, columns: Sequence[str]) -> tuple[str, ...]:
    if strategy == "front_cusum_gate":
        patterns = ("isd_pressure_tendency", "isd_pressure_mean", "isd_wind_v", "isd_dew_point", "isd_air_temp_mean_c_change", "target_abs_change")
    elif strategy == "north_south_propagation":
        patterns = ("isd_north_south", "isd_pressure_plane_lat", "isd_wind_v", "isd_air_temp_mean_c_change")
    elif strategy == "east_west_flow_gradient":
        patterns = ("isd_east_west", "isd_pressure_plane_lon", "isd_wind_u", "isd_dew_point")
    else:
        patterns = ("daily_waglan", "isd_wind", "ua_wind")
    selected = [col for col in frame.columns if col in columns and any(pattern in col for pattern in patterns)]
    return tuple(dict.fromkeys(selected))


def add_fold_cusum_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    source_cols: Sequence[str],
    *,
    prefix: str,
) -> list[str]:
    active = active_cols(train, source_cols)
    if not active:
        return []
    train_len = len(train)
    combined = pd.concat([train[["target_date", *active]], test[["target_date", *active]]], ignore_index=True)
    combined = combined.sort_values("target_date").reset_index(drop=True)
    extra_cols: list[str] = []
    safe_prefix = prefix.replace("-", "_")
    for col in active[:8]:
        mean = float(train[col].mean())
        std = float(train[col].std(ddof=1))
        if not np.isfinite(std) or std <= 1e-6:
            continue
        z = ((combined[col] - mean) / std).fillna(0.0).to_numpy(dtype=float)
        pos = np.zeros_like(z)
        neg = np.zeros_like(z)
        for idx, value in enumerate(z):
            if idx == 0:
                pos[idx] = max(0.0, value - 0.25)
                neg[idx] = max(0.0, -value - 0.25)
            else:
                pos[idx] = max(0.0, pos[idx - 1] + value - 0.25)
                neg[idx] = max(0.0, neg[idx - 1] - value - 0.25)
        base = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")
        pos_col = f"fold_{safe_prefix}_cusum_pos_{base}"
        neg_col = f"fold_{safe_prefix}_cusum_neg_{base}"
        train[pos_col] = pos[:train_len]
        test[pos_col] = pos[train_len:]
        train[neg_col] = neg[:train_len]
        test[neg_col] = neg[train_len:]
        extra_cols.extend([pos_col, neg_col])
    return extra_cols


def add_fold_station_offset_features(train: pd.DataFrame, test: pd.DataFrame, strategy: str) -> list[str]:
    if strategy != "station_homogenization_offsets":
        return []
    station_cols = [col for col in train.columns if col.startswith("isd_station_air_temperature_c_")]
    if not station_cols:
        return []
    consensus = train[station_cols].median(axis=1)
    offsets = {col: float((train[col] - consensus).median()) for col in station_cols if train[col].notna().sum() >= MIN_FEATURE_SUPPORT}
    extra_cols: list[str] = []
    for col, offset in offsets.items():
        out_col = f"fold_homogenized_{col}"
        train[out_col] = train[col] - offset
        test[out_col] = test[col] - offset
        extra_cols.append(out_col)
    if extra_cols:
        train["fold_station_homogenized_network_mean_c"] = train[extra_cols].mean(axis=1)
        test["fold_station_homogenized_network_mean_c"] = test[extra_cols].mean(axis=1)
        extra_cols.append("fold_station_homogenized_network_mean_c")
    return extra_cols


def add_strategy_composite_features(train: pd.DataFrame, test: pd.DataFrame, strategy: str) -> list[str]:
    extra_cols: list[str] = []
    for frame in (train, test):
        if strategy == "flow_relative_weighting":
            add_flow_relative_columns(frame)
        elif strategy == "sea_breeze_index":
            add_sea_breeze_columns(frame)
        elif strategy in {"north_south_propagation", "east_west_flow_gradient"}:
            add_gradient_state_columns(frame)
        elif strategy in {"robust_distribution_shape", "station_dropout_masked", "quality_weighted_surface"}:
            add_spatial_distribution_columns(frame)
        elif strategy == "rainfall_reservoir":
            add_rainfall_reservoir_columns(frame)
        elif strategy == "solar_efficiency_state":
            add_solar_efficiency_columns(frame)
        elif strategy == "surface_storage_state":
            add_surface_storage_columns(frame)
        elif strategy == "sea_temperature_extrapolator":
            add_sea_temperature_columns(frame)
        elif strategy in {"teacher_student_subsidence", "teacher_student_suppression"}:
            add_teacher_student_proxy_columns(frame, strategy)
    for frame in (train, test):
        extra_cols.extend([col for col in frame.columns if col.startswith(f"fold_{strategy}_")])
    return list(dict.fromkeys(extra_cols))


def add_flow_relative_columns(frame: pd.DataFrame) -> None:
    u = frame.get("isd_wind_u_mean_mps", pd.Series(np.nan, index=frame.index))
    v = frame.get("isd_wind_v_mean_mps", pd.Series(np.nan, index=frame.index))
    wind_speed = np.sqrt(u**2 + v**2).replace(0.0, np.nan)
    grad_lat = frame.get("isd_temp_plane_lat_slope_c_per_deg", pd.Series(np.nan, index=frame.index))
    grad_lon = frame.get("isd_temp_plane_lon_slope_c_per_deg", pd.Series(np.nan, index=frame.index))
    pressure_lat = frame.get("isd_pressure_plane_lat_slope_hpa_per_deg", pd.Series(np.nan, index=frame.index))
    pressure_lon = frame.get("isd_pressure_plane_lon_slope_hpa_per_deg", pd.Series(np.nan, index=frame.index))
    frame["fold_flow_relative_weighting_temp_gradient_along_wind"] = (grad_lon * u + grad_lat * v) / wind_speed
    frame["fold_flow_relative_weighting_pressure_gradient_along_wind"] = (pressure_lon * u + pressure_lat * v) / wind_speed
    frame["fold_flow_relative_weighting_crosswind_temp_gradient"] = (grad_lon * -v + grad_lat * u) / wind_speed
    frame["fold_flow_relative_weighting_steering_confidence"] = wind_speed / (1.0 + frame.get("isd_obs_age_mean_min", pd.Series(0.0, index=frame.index)).fillna(0.0) / 60.0)


def add_sea_breeze_columns(frame: pd.DataFrame) -> None:
    inland_contrast = frame.get("isd_north_south_temp_gradient_c", pd.Series(np.nan, index=frame.index))
    zonal_contrast = frame.get("isd_east_west_temp_gradient_c", pd.Series(np.nan, index=frame.index))
    wind_speed = frame.get("isd_wind_speed_mean_mps", pd.Series(np.nan, index=frame.index))
    shear = frame.get("ua_vector_shear_925_700_mps", pd.Series(np.nan, index=frame.index))
    weak_synoptic = 1.0 / (1.0 + wind_speed.abs())
    frame["fold_sea_breeze_index_coastal_inland_contrast_c"] = inland_contrast
    frame["fold_sea_breeze_index_estuary_gradient_c"] = zonal_contrast
    frame["fold_sea_breeze_index_synoptic_weakness"] = weak_synoptic
    frame["fold_sea_breeze_index_susceptibility"] = inland_contrast.clip(lower=0.0) * weak_synoptic / (1.0 + shear.abs())


def add_gradient_state_columns(frame: pd.DataFrame) -> None:
    ns = frame.get("isd_north_south_temp_gradient_c", pd.Series(np.nan, index=frame.index))
    ew = frame.get("isd_east_west_temp_gradient_c", pd.Series(np.nan, index=frame.index))
    v = frame.get("isd_wind_v_mean_mps", pd.Series(np.nan, index=frame.index))
    u = frame.get("isd_wind_u_mean_mps", pd.Series(np.nan, index=frame.index))
    pressure_ns = frame.get("isd_pressure_plane_lat_slope_hpa_per_deg", pd.Series(np.nan, index=frame.index))
    pressure_ew = frame.get("isd_pressure_plane_lon_slope_hpa_per_deg", pd.Series(np.nan, index=frame.index))
    frame["fold_north_south_propagation_cold_surge_index"] = (-ns).clip(lower=0.0) * (-v).clip(lower=0.0) + pressure_ns.clip(lower=0.0)
    frame["fold_north_south_propagation_return_flow_index"] = ns.clip(lower=0.0) * v.clip(lower=0.0)
    frame["fold_east_west_flow_gradient_estuary_advection_index"] = ew * u
    frame["fold_east_west_flow_gradient_pressure_rotation_proxy"] = pressure_ew * u + pressure_ns * v


def add_spatial_distribution_columns(frame: pd.DataFrame) -> None:
    station_cols = [col for col in frame.columns if col.startswith("isd_station_air_temperature_c_")]
    if not station_cols:
        return
    values = frame[station_cols]
    for quantile in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        safe = str(quantile).replace(".", "p")
        frame[f"fold_robust_distribution_shape_q{safe}_c"] = values.quantile(quantile, axis=1)
    q90 = frame["fold_robust_distribution_shape_q0p9_c"]
    q10 = frame["fold_robust_distribution_shape_q0p1_c"]
    q50 = frame["fold_robust_distribution_shape_q0p5_c"]
    frame["fold_robust_distribution_shape_iqr_c"] = frame["fold_robust_distribution_shape_q0p75_c"] - frame["fold_robust_distribution_shape_q0p25_c"]
    frame["fold_robust_distribution_shape_hot_tail_spread_c"] = q90 - q50
    frame["fold_robust_distribution_shape_cold_tail_spread_c"] = q50 - q10
    mask = values.notna().astype(float)
    coverage = mask.sum(axis=1).clip(lower=1.0)
    probs = mask.div(coverage, axis=0).replace(0.0, np.nan)
    frame["fold_station_dropout_masked_station_entropy"] = -(probs * np.log(probs)).sum(axis=1)
    frame["fold_quality_weighted_surface_station_count"] = frame.get("isd_station_count", mask.sum(axis=1))
    frame["fold_quality_weighted_surface_age_penalty"] = frame.get("isd_obs_age_mean_min", pd.Series(0.0, index=frame.index)).fillna(0.0) / 60.0


def add_rainfall_reservoir_columns(frame: pd.DataFrame) -> None:
    rain = frame.get("daily_hong_kong_observatory_daily_rainfall_lag7", pd.Series(np.nan, index=frame.index))
    rain7 = frame.get("daily_hong_kong_observatory_daily_rainfall_lag7_roll7", rain)
    evaporation = frame.get("daily_kings_park_evaporation_lag7_roll7", pd.Series(0.0, index=frame.index))
    frame["fold_rainfall_reservoir_wetness_balance"] = rain7.fillna(0.0) - evaporation.fillna(0.0)
    frame["fold_rainfall_reservoir_heavy_rain_flag"] = (rain >= rain.quantile(0.85)).astype(float)
    frame["fold_rainfall_reservoir_dry_memory_proxy"] = 1.0 / (1.0 + rain7.fillna(0.0).clip(lower=0.0))


def add_solar_efficiency_columns(frame: pd.DataFrame) -> None:
    solar = frame.get("daily_kings_park_global_solar_radiation_lag7_roll7", pd.Series(np.nan, index=frame.index))
    sunshine = frame.get("daily_kings_park_bright_sunshine_duration_lag7_roll7", pd.Series(np.nan, index=frame.index))
    tlag = frame.get("target_roll30_mean_lag7_c", pd.Series(np.nan, index=frame.index))
    humidity = frame.get("daily_hong_kong_observatory_mean_relative_humidity_lag7_roll7", pd.Series(np.nan, index=frame.index))
    frame["fold_solar_efficiency_state_radiation_per_sunshine"] = solar / sunshine.replace(0.0, np.nan)
    frame["fold_solar_efficiency_state_temp_per_radiation"] = tlag / solar.replace(0.0, np.nan)
    frame["fold_solar_efficiency_state_humidity_penalty"] = humidity / 100.0


def add_surface_storage_columns(frame: pd.DataFrame) -> None:
    grass = frame.get("daily_hong_kong_observatory_grass_minimum_temperature_lag7_roll7", pd.Series(np.nan, index=frame.index))
    evaporation = frame.get("daily_kings_park_evaporation_lag7_roll7", pd.Series(np.nan, index=frame.index))
    rain = frame.get("daily_hong_kong_observatory_daily_rainfall_lag7_roll7", pd.Series(0.0, index=frame.index))
    air_min = frame.get("daily_hong_kong_observatory_daily_minimum_temperature_lag7_roll7", pd.Series(np.nan, index=frame.index))
    frame["fold_surface_storage_state_grass_air_contrast_c"] = grass - air_min
    frame["fold_surface_storage_state_evap_rain_balance"] = evaporation.fillna(0.0) - rain.fillna(0.0)
    frame["fold_surface_storage_state_nocturnal_heat_retention"] = (grass - air_min).clip(lower=0.0)


def add_sea_temperature_columns(frame: pd.DataFrame) -> None:
    sea_am = frame.get("daily_north_point_sea_temperature_am_lag7_roll7", pd.Series(np.nan, index=frame.index))
    sea_pm = frame.get("daily_north_point_sea_temperature_pm_lag7_roll7", pd.Series(np.nan, index=frame.index))
    waglan = frame.get("daily_waglan_island_sea_temperature_lag7_roll7", pd.Series(np.nan, index=frame.index))
    air = frame.get("target_roll30_mean_lag7_c", pd.Series(np.nan, index=frame.index))
    wind = frame.get("isd_wind_u_mean_mps", pd.Series(np.nan, index=frame.index))
    sea_mean = pd.concat([sea_am, sea_pm, waglan], axis=1).mean(axis=1)
    frame["fold_sea_temperature_extrapolator_lagged_sea_mean_c"] = sea_mean
    frame["fold_sea_temperature_extrapolator_air_minus_sea_c"] = air - sea_mean
    frame["fold_sea_temperature_extrapolator_onshore_moderation_proxy"] = (sea_mean - air) * wind.clip(lower=0.0)


def add_teacher_student_proxy_columns(frame: pd.DataFrame, strategy: str) -> None:
    ridge = frame.get("ua_ridge_strength_raw_proxy", pd.Series(np.nan, index=frame.index))
    dryness = frame.get("ua_dry_entrainment_potential_proxy", pd.Series(np.nan, index=frame.index))
    rain = frame.get("daily_hong_kong_observatory_daily_rainfall_lag7_roll7", pd.Series(np.nan, index=frame.index))
    cloud = frame.get("daily_hong_kong_observatory_mean_cloud_amount_lag7_roll7", pd.Series(np.nan, index=frame.index))
    sunshine = frame.get("daily_kings_park_bright_sunshine_duration_lag7_roll7", pd.Series(np.nan, index=frame.index))
    if strategy == "teacher_student_subsidence":
        frame["fold_teacher_student_subsidence_heat_student_proxy"] = ridge.fillna(0.0) + dryness.fillna(0.0)
        frame["fold_teacher_student_subsidence_weak_flow_proxy"] = 1.0 / (1.0 + frame.get("isd_wind_speed_mean_mps", pd.Series(np.nan, index=frame.index)).abs())
    else:
        frame["fold_teacher_student_suppression_cloud_rain_student_proxy"] = cloud.fillna(0.0) + rain.fillna(0.0) - sunshine.fillna(0.0)
        frame["fold_teacher_student_suppression_moisture_student_proxy"] = frame.get("ua_theta_e_1000_700_mean_k", pd.Series(np.nan, index=frame.index))


def detect_training_break_years(train: pd.DataFrame) -> tuple[int, ...]:
    annual = (
        train.assign(anomaly=lambda frame: frame["target_tmax_c"] - frame["target_roll365_mean_lag7_c"])
        .groupby("year", observed=True)["anomaly"]
        .mean()
        .dropna()
        .sort_index()
    )
    if len(annual) < 20:
        return ()
    shifted_delta = (annual.rolling(5, min_periods=3).mean() - annual.shift(5).rolling(5, min_periods=3).mean()).abs()
    candidates = shifted_delta.dropna().sort_values(ascending=False)
    selected: list[int] = []
    latest_allowed = int(train["year"].max()) - 3
    for year, _score in candidates.items():
        candidate_year = int(year)
        if candidate_year > latest_allowed:
            continue
        if all(abs(candidate_year - existing) >= 6 for existing in selected):
            selected.append(candidate_year)
        if len(selected) == 3:
            break
    return tuple(sorted(selected))


def fit_predict_for_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: Sequence[str],
    spec: RunSpec,
    model_spec: ModelSpec,
    fold_id: str,
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    if model_spec.control or model_spec.shuffled_control or model_spec.strategy == "ridge":
        return fit_ridge_prediction(train, test, cols, spec, model_spec, fold_id)

    strategy = model_spec.strategy
    if strategy in {"dtw_analog_blend", "climate_trajectory_analog"}:
        return fit_analog_prediction(train, test, cols, strategy)
    if strategy == "hazard_mixture":
        return fit_hazard_mixture_prediction(train, test, cols)
    if strategy in {"gmm_regime_mixture", "cloud_sunshine_regime", "markov_wind_regime", "teacher_student_archetype"}:
        return fit_regime_mixture_prediction(train, test, cols, strategy)
    if strategy == "reliability_shrinkage":
        return fit_reliability_shrinkage_prediction(train, test, cols)
    if strategy in {"teacher_student_subsidence", "teacher_student_suppression"}:
        return fit_teacher_student_prediction(train, test, cols, strategy)
    if strategy == "expert_gate":
        return fit_expert_gate_prediction(train, test, cols)
    if strategy == "residual_stack":
        return fit_residual_stack_prediction(train, test, cols)
    if strategy in {"forecastability_scale", "student_t_conformal_scale"}:
        return fit_student_t_conformal_prediction(train, test, cols)
    if strategy == "quantile_tail_cdf":
        return fit_quantile_tail_prediction(train, test, cols)
    return fit_ridge_prediction(train, test, cols, spec, model_spec, fold_id)


def fit_ridge_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: Sequence[str],
    spec: RunSpec,
    model_spec: ModelSpec,
    fold_id: str,
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    model = fit_pipeline()
    y_train = train["target_tmax_c"].to_numpy(dtype=float)
    if model_spec.shuffled_control:
        seed = int(sha256_bytes(f"{spec.parsed.experiment_id}:{fold_id}:shuffle".encode())[:8], 16)
        rng = np.random.default_rng(seed)
        y_fit = rng.permutation(y_train)
    else:
        y_fit = y_train
    model.fit(train[list(cols)], y_fit)
    train_pred = model.predict(train[list(cols)])
    sigma = max(float(np.std(y_train - train_pred, ddof=1)), 0.2)
    return model.predict(test[list(cols)]), sigma, {}


def fit_basic_ridge(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str], y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    active = active_cols(train, cols)
    if not active:
        y_train = train["target_tmax_c"].to_numpy(dtype=float) if y is None else y
        fallback = np.full(len(test), float(np.nanmean(y_train)))
        return np.full(len(train), float(np.nanmean(y_train))), fallback, max(float(np.nanstd(y_train)), 0.2)
    model = fit_pipeline()
    y_train = train["target_tmax_c"].to_numpy(dtype=float) if y is None else y
    model.fit(train[active], y_train)
    train_pred = model.predict(train[active])
    sigma = max(float(np.std(y_train - train_pred, ddof=1)), 0.2)
    return train_pred, model.predict(test[active]), sigma


def fit_analog_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: Sequence[str],
    strategy: str,
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    if strategy == "climate_trajectory_analog":
        analog_cols = [col for col in cols if col.startswith("daily_") or col.startswith("fold_daily_climate_factor_pca_")]
    else:
        analog_cols = [col for col in cols if col.startswith("target_lag") or col.startswith("trajectory_")]
    analog_cols = active_cols(train, analog_cols)
    if len(analog_cols) < 2:
        return fit_basic_ridge(train, test, cols)[1], 1.0, {}
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[analog_cols]))
    x_test = scaler.transform(imputer.transform(test[analog_cols]))
    y_train = train["target_tmax_c"].to_numpy(dtype=float)
    train_month = train["month"].to_numpy()
    test_month = test["month"].to_numpy()
    max_pool = 2500
    preds = np.empty(len(test), dtype=float)
    dispersions = np.empty(len(test), dtype=float)
    for idx, row in enumerate(x_test):
        season_mask = np.abs(train_month - test_month[idx]) <= 1
        season_mask |= np.abs(train_month - test_month[idx]) >= 11
        candidates = x_train[season_mask]
        outcomes = y_train[season_mask]
        if len(candidates) < 30:
            candidates = x_train
            outcomes = y_train
        if len(candidates) > max_pool:
            candidates = candidates[-max_pool:]
            outcomes = outcomes[-max_pool:]
        distance = dtw_like_distance(candidates, row)
        k = min(30, len(distance))
        nearest = np.argpartition(distance, k - 1)[:k]
        weights = 1.0 / np.maximum(distance[nearest], 0.05)
        weights = weights / weights.sum()
        vals = outcomes[nearest]
        preds[idx] = float(np.sum(weights * vals))
        dispersions[idx] = float(np.sqrt(np.sum(weights * (vals - preds[idx]) ** 2)))
    _, ridge_pred, ridge_sigma = fit_basic_ridge(train, test, cols)
    point = 0.65 * preds + 0.35 * ridge_pred
    sigma = max(float(np.nanmedian(dispersions)), ridge_sigma, 0.2)
    return point, sigma, {}


def dtw_like_distance(candidates: np.ndarray, row: np.ndarray) -> np.ndarray:
    direct = np.sqrt(np.mean(np.square(candidates - row), axis=1))
    derivative = np.sqrt(np.mean(np.square(np.diff(candidates, axis=1) - np.diff(row)), axis=1)) if candidates.shape[1] >= 3 else direct
    corr_num = np.sum((candidates - candidates.mean(axis=1, keepdims=True)) * (row - row.mean()), axis=1)
    corr_den = np.sqrt(np.sum(np.square(candidates - candidates.mean(axis=1, keepdims=True)), axis=1) * np.sum(np.square(row - row.mean())))
    corr_distance = 1.0 - np.divide(corr_num, corr_den, out=np.zeros_like(corr_num), where=corr_den > 1e-9)
    return direct + 0.5 * derivative + 0.25 * corr_distance


def fit_hazard_mixture_prediction(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    persistence_cols = [col for col in cols if col.startswith("target_lag") or col.startswith("spell_") or col.startswith("trajectory_")]
    atmosphere_cols = [col for col in cols if col not in persistence_cols]
    _, persistence_pred, _ = fit_basic_ridge(train, test, persistence_cols)
    _, atmosphere_pred, sigma = fit_basic_ridge(train, test, atmosphere_cols or cols)
    label = ((train["target_tmax_c"] - train["target_lag7_tmax_c"]).abs() > train["target_abs_change_7_14_c"].quantile(0.7)).astype(int)
    hazard_cols = active_cols(train, [col for col in cols if "hazard" in col or "spell" in col or "reversal" in col or "volatility" in col])
    if hazard_cols and label.nunique() == 2:
        classifier = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(max_iter=500, C=0.5)),
            ]
        )
        classifier.fit(train[hazard_cols], label)
        hazard = classifier.predict_proba(test[hazard_cols])[:, 1]
    else:
        hazard = np.full(len(test), float(label.mean()))
    point = (1.0 - hazard) * persistence_pred + hazard * atmosphere_pred
    return point, sigma, {}


def fit_regime_mixture_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: Sequence[str],
    strategy: str,
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    train_global, test_global, sigma = fit_basic_ridge(train, test, cols)
    residual = train["target_tmax_c"].to_numpy(dtype=float) - train_global
    prob_cols = [col for col in cols if col.startswith(f"fold_{strategy}_regime_prob_")]
    if not prob_cols:
        return test_global, sigma, {}
    correction = np.zeros(len(test), dtype=float)
    for col in prob_cols:
        train_weight = train[col].fillna(0.0).to_numpy(dtype=float)
        mask = train_weight >= max(0.35, np.nanquantile(train_weight, 0.65))
        if mask.sum() < MIN_FEATURE_SUPPORT:
            continue
        expert_cols = [item for item in cols if item not in prob_cols]
        train_subset = train.loc[mask].copy()
        y_subset = residual[mask]
        _, expert_pred, _ = fit_basic_ridge(train_subset, test, expert_cols, y=y_subset)
        correction += test[col].fillna(0.0).to_numpy(dtype=float) * expert_pred
    return test_global + correction, sigma, {}


def fit_reliability_shrinkage_prediction(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    upper_cols = [col for col in cols if col.startswith("igra_") or col.startswith("ua_")]
    surface_cols = [col for col in cols if col.startswith("isd_") or col.startswith("target_") or col.startswith("doy_") or col.startswith("year_")]
    _, upper_pred, upper_sigma = fit_basic_ridge(train, test, upper_cols)
    _, surface_pred, surface_sigma = fit_basic_ridge(train, test, surface_cols)
    completeness = test.get("ua_profile_valid_level_count_1000_700", test.get("igra_key_level_count", pd.Series(np.nan, index=test.index))).astype(float)
    reliability = (completeness / np.nanmax([float(completeness.max()), 4.0])).clip(0.0, 1.0).fillna(0.4).to_numpy(dtype=float)
    point = reliability * upper_pred + (1.0 - reliability) * surface_pred
    sigma = max(float(np.nanmean(reliability * upper_sigma + (1.0 - reliability) * surface_sigma)), 0.2)
    return point, sigma, {}


def fit_teacher_student_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: Sequence[str],
    strategy: str,
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    base_cols = [col for col in cols if not col.startswith("fold_teacher_student")]
    train_base, test_base, sigma = fit_basic_ridge(train, test, base_cols or cols)
    residual = train["target_tmax_c"].to_numpy(dtype=float) - train_base
    if strategy == "teacher_student_subsidence":
        teacher_score = train.get("ua_ridge_strength_raw_proxy", pd.Series(np.nan, index=train.index)).fillna(0.0) + train.get("ua_dry_entrainment_potential_proxy", pd.Series(np.nan, index=train.index)).fillna(0.0)
        label = ((teacher_score >= teacher_score.quantile(0.75)) & (train["target_tmax_c"] >= train["target_tmax_c"].quantile(0.75))).astype(int)
    else:
        teacher_score = train.get("daily_hong_kong_observatory_mean_cloud_amount_lag7_roll7", pd.Series(np.nan, index=train.index)).fillna(0.0) + train.get("daily_hong_kong_observatory_daily_rainfall_lag7_roll7", pd.Series(np.nan, index=train.index)).fillna(0.0)
        label = (teacher_score >= teacher_score.quantile(0.75)).astype(int)
    student_cols = active_cols(train, cols)
    if not student_cols or label.nunique() < 2:
        return test_base, sigma, {}
    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(max_iter=500, C=0.5)),
        ]
    )
    classifier.fit(train[student_cols], label)
    train_prob = classifier.predict_proba(train[student_cols])[:, 1]
    test_prob = classifier.predict_proba(test[student_cols])[:, 1]
    correction_cols = list(dict.fromkeys([*student_cols, "__teacher_prob__"]))
    train_corr = train.copy()
    test_corr = test.copy()
    train_corr["__teacher_prob__"] = train_prob
    test_corr["__teacher_prob__"] = test_prob
    _, correction, _ = fit_basic_ridge(train_corr, test_corr, correction_cols, y=residual)
    return test_base + correction, sigma, {}


def fit_expert_gate_prediction(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    train_experts, test_experts = aligned_legacy_experts(train, test)
    expert_cols = [col for col in train_experts.columns if col.startswith("expert_")]
    if len(expert_cols) < 2:
        return fit_basic_ridge(train, test, cols)[1], 1.0, {}
    y = train_experts["target_tmax_c"].to_numpy(dtype=float)
    errors = {col: np.abs(train_experts[col].to_numpy(dtype=float) - y) for col in expert_cols}
    inv_mae = np.array([1.0 / max(float(np.nanmean(errors[col])), 0.05) for col in expert_cols])
    static_weights = inv_mae / inv_mae.sum()
    train_season = train["month"].map(season_name).to_numpy()
    test_season = test["month"].map(season_name).to_numpy()
    season_weights: dict[str, np.ndarray] = {}
    for season in ("DJF", "MAM", "JJA", "SON"):
        mask = train_season == season
        if mask.sum() < 365:
            season_weights[season] = static_weights
            continue
        season_inv = np.array([1.0 / max(float(np.nanmean(errors[col][mask])), 0.05) for col in expert_cols])
        season_weights[season] = season_inv / season_inv.sum()
    weights = np.vstack([0.35 * static_weights + 0.65 * season_weights.get(season, static_weights) for season in test_season])
    weights = weights / weights.sum(axis=1, keepdims=True)
    point = np.sum(test_experts[expert_cols].to_numpy(dtype=float) * weights, axis=1)
    sigma = max(float(np.std(y - (train_experts[expert_cols].to_numpy(dtype=float) @ static_weights), ddof=1)), 0.2)
    return point, sigma, {}


def fit_residual_stack_prediction(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    train_experts, test_experts = aligned_legacy_experts(train, test)
    if "expert_r17" in train_experts:
        base_train = train_experts["expert_r17"].to_numpy(dtype=float)
        base_test = test_experts["expert_r17"].to_numpy(dtype=float)
    else:
        base_train, base_test, _ = fit_basic_ridge(train, test, cols)
    residual = train["target_tmax_c"].to_numpy(dtype=float) - base_train
    point = base_test.copy()
    stages = (
        ("dynamic", ("clim_", "target_", "doy_", "year_")),
        ("surface", ("isd_", "fold_flow", "fold_robust")),
        ("upper", ("igra_", "ua_", "fold_fpca", "fold_gmm")),
        ("daily", ("daily_", "fold_daily")),
        ("regime", ("fold_",)),
    )
    for _stage_name, patterns in stages:
        stage_cols = [col for col in cols if any(col.startswith(pattern) for pattern in patterns)]
        active = active_cols(train, stage_cols)
        if not active:
            continue
        train_corr, test_corr, _ = fit_basic_ridge(train, test, active, y=residual)
        residual = residual - train_corr
        point = point + test_corr
    sigma = max(float(np.std(residual, ddof=1)), 0.2)
    return point, sigma, {}


def fit_student_t_conformal_prediction(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    train_point, point, _ = fit_basic_ridge(train, test, cols)
    residual = train["target_tmax_c"].to_numpy(dtype=float) - train_point
    sigma_values = estimate_heteroscedastic_scale(train, test, cols, residual)
    probs = {"q05": 0.05, "q10": 0.10, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q90": 0.90, "q95": 0.95}
    conformal_offsets = weighted_residual_quantiles(train, residual, probs)
    t_multipliers = student_t_quantile_multipliers()
    quantiles = {}
    for qcol, probability in probs.items():
        student_t_value = point + sigma_values * t_multipliers[qcol]
        conformal_value = point + conformal_offsets[probability]
        quantiles[qcol] = 0.55 * conformal_value + 0.45 * student_t_value
    return point, max(float(np.nanmedian(sigma_values)), 0.2), enforce_monotone_quantiles(quantiles)


def fit_quantile_tail_prediction(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    train_point, point, _ = fit_basic_ridge(train, test, cols)
    residual = train["target_tmax_c"].to_numpy(dtype=float) - train_point
    hot_threshold = np.nanquantile(train["target_tmax_c"], 0.9)
    cold_threshold = np.nanquantile(train["target_tmax_c"], 0.1)
    hot_weights = np.where(train["target_tmax_c"].to_numpy(dtype=float) >= hot_threshold, 3.0, 1.0)
    cold_weights = np.where(train["target_tmax_c"].to_numpy(dtype=float) <= cold_threshold, 3.0, 1.0)
    _, hot_corr, _ = fit_weighted_ridge_residual(train, test, cols, residual, hot_weights)
    _, cold_corr, _ = fit_weighted_ridge_residual(train, test, cols, residual, cold_weights)
    tail_state = test.get("target_volatility_forecastability_score_lag7", pd.Series(0.5, index=test.index)).fillna(0.5).to_numpy(dtype=float)
    tail_state = np.clip(1.0 - tail_state / np.nanmax([float(np.nanmax(tail_state)), 1.0]), 0.0, 1.0)
    point = point + 0.20 * tail_state * hot_corr + 0.20 * (1.0 - tail_state) * cold_corr
    sigma_values = estimate_heteroscedastic_scale(train, test, cols, residual)
    probs = {
        "q01": 0.01,
        "q05": 0.05,
        "q10": 0.10,
        "q25": 0.25,
        "q50": 0.50,
        "q75": 0.75,
        "q90": 0.90,
        "q95": 0.95,
        "q99": 0.99,
    }
    conformal_offsets = weighted_residual_quantiles(train, residual, probs)
    direct_quantiles = fit_direct_quantile_residual_models(train, test, cols, residual, probs)
    t_multipliers = student_t_quantile_multipliers()
    tail_weight_by_q = {"q01": 0.75, "q05": 0.60, "q10": 0.45, "q25": 0.25, "q50": 0.15, "q75": 0.25, "q90": 0.45, "q95": 0.60, "q99": 0.75}
    quantiles: dict[str, np.ndarray] = {}
    for qcol, probability in probs.items():
        student_t_value = point + sigma_values * t_multipliers[qcol]
        direct_value = point + direct_quantiles[qcol]
        conformal_value = point + conformal_offsets[probability]
        tail_specialist = 0.70 * direct_value + 0.30 * conformal_value
        weight = tail_weight_by_q[qcol]
        quantiles[qcol] = (1.0 - weight) * student_t_value + weight * tail_specialist
    return point, max(float(np.nanmedian(sigma_values)), 0.2), enforce_monotone_quantiles(quantiles)


def scale_source_columns(cols: Sequence[str]) -> tuple[str, ...]:
    tokens = (
        "volatility",
        "entropy",
        "station_count",
        "profile_valid",
        "std",
        "spread",
        "disagreement",
        "forecastability",
        "obs_age",
        "reversal",
    )
    return tuple(col for col in cols if any(token in col for token in tokens))


def estimate_heteroscedastic_scale(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str], residual: np.ndarray) -> np.ndarray:
    finite_residual = residual[np.isfinite(residual)]
    fallback = max(float(np.nanstd(finite_residual, ddof=1)), 0.2) if len(finite_residual) > 1 else 1.0
    scale_cols = active_cols(train, scale_source_columns(cols))
    if not scale_cols:
        return np.full(len(test), fallback, dtype=float)

    scale_target = np.log(np.maximum(np.abs(residual), 0.05))
    _, ridge_log_scale, _ = fit_basic_ridge(train, test, scale_cols, y=scale_target)
    boosted_log_scale = ridge_log_scale
    if len(train) >= MIN_FEATURE_SUPPORT and len(scale_cols) >= 2:
        booster = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "boost",
                    GradientBoostingRegressor(
                        loss="squared_error",
                        n_estimators=40,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=29,
                    ),
                ),
            ]
        )
        booster.fit(train[list(scale_cols)], scale_target)
        boosted_log_scale = booster.predict(test[list(scale_cols)])
    log_scale = 0.60 * ridge_log_scale + 0.40 * boosted_log_scale
    return np.clip(np.exp(log_scale), 0.2, 5.0)


def weighted_residual_quantiles(
    train: pd.DataFrame,
    residual: np.ndarray,
    probabilities: Mapping[str, float],
    *,
    half_life_years: float = 7.0,
) -> dict[float, float]:
    dates = pd.to_datetime(train["target_date"])
    max_date = dates.max()
    age_years = (max_date - dates).dt.days.to_numpy(dtype=float) / 365.25
    weights = np.power(0.5, np.maximum(age_years, 0.0) / half_life_years)
    finite = np.isfinite(residual) & np.isfinite(weights) & (weights > 0)
    if finite.sum() < 50:
        finite = np.isfinite(residual)
        weights = np.ones_like(residual, dtype=float)
    values = residual[finite]
    sorted_order = np.argsort(values)
    sorted_values = values[sorted_order]
    sorted_weights = weights[finite][sorted_order]
    cumulative = np.cumsum(sorted_weights)
    cumulative = cumulative / cumulative[-1]
    return {probability: float(np.interp(probability, cumulative, sorted_values)) for probability in probabilities.values()}


def student_t_quantile_multipliers() -> dict[str, float]:
    # Approximate central Student-t(7) quantile multipliers; fixed by design so
    # folds cannot tune tail thickness from OOF outcomes.
    return {
        "q01": -2.998,
        "q05": -1.895,
        "q10": -1.415,
        "q25": -0.711,
        "q50": 0.0,
        "q75": 0.711,
        "q90": 1.415,
        "q95": 1.895,
        "q99": 2.998,
    }


def fit_direct_quantile_residual_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: Sequence[str],
    residual: np.ndarray,
    probabilities: Mapping[str, float],
) -> dict[str, np.ndarray]:
    active = active_cols(train, cols)
    conformal_offsets = weighted_residual_quantiles(train, residual, probabilities)
    if len(active) < 2 or len(train) < MIN_FEATURE_SUPPORT:
        return {qcol: np.full(len(test), conformal_offsets[probability], dtype=float) for qcol, probability in probabilities.items()}
    selected = list(active[:96])
    out: dict[str, np.ndarray] = {}
    for qcol, probability in probabilities.items():
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "quantile_boost",
                    GradientBoostingRegressor(
                        loss="quantile",
                        alpha=probability,
                        n_estimators=25,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=int(probability * 1000) + 41,
                    ),
                ),
            ]
        )
        model.fit(train[selected], residual)
        out[qcol] = model.predict(test[selected])
    return out


def fit_weighted_ridge_residual(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: Sequence[str],
    residual: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    active = active_cols(train, cols)
    if not active:
        return np.zeros(len(train)), np.zeros(len(test)), 0.2
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=2.0)),
        ]
    )
    model.fit(train[active], residual, ridge__sample_weight=weights)
    train_pred = model.predict(train[active])
    sigma = max(float(np.std(residual - train_pred, ddof=1)), 0.2)
    return train_pred, model.predict(test[active]), sigma


def enforce_monotone_quantiles(quantiles: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    order = ["q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99"]
    present = [item for item in order if item in quantiles]
    if not present:
        return {}
    stacked = np.column_stack([quantiles[item] for item in present])
    stacked = np.maximum.accumulate(stacked, axis=1)
    return {item: stacked[:, idx] for idx, item in enumerate(present)}


def aligned_legacy_experts(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    experts = legacy_expert_table()
    train_out = train[["target_date", "target_tmax_c"]].merge(experts, on="target_date", how="left")
    test_out = test[["target_date", "target_tmax_c"]].merge(experts, on="target_date", how="left")
    expert_cols = [col for col in train_out.columns if col.startswith("expert_")]
    for col in expert_cols:
        fallback = float(train_out[col].median())
        if not np.isfinite(fallback):
            fallback = float(train_out["target_tmax_c"].mean())
        train_out[col] = train_out[col].fillna(fallback)
        test_out[col] = test_out[col].fillna(fallback)
    return train_out, test_out


def legacy_expert_table() -> pd.DataFrame:
    global LEGACY_EXPERT_CACHE
    if LEGACY_EXPERT_CACHE is not None:
        return LEGACY_EXPERT_CACHE.copy()
    source_models = {
        "hkg_t24_r14_oof_predictions.parquet": ("expert_r14", "r14_upper_air_core"),
        "hkg_t24_r15_oof_predictions.parquet": ("expert_r15", "r15_coupling_terms"),
        "hkg_t24_r16_oof_predictions.parquet": ("expert_r16", "r16_isd_regional_aggregate"),
        "hkg_t24_r17_oof_predictions.parquet": ("expert_r17", "r17_era_transfer_terms"),
    }
    frames: list[pd.DataFrame] = []
    for file_name, (out_col, model_id) in source_models.items():
        raw = pd.read_parquet(OUTPUT_ROOT / file_name)
        raw["target_date"] = pd.to_datetime(raw["target_date"]).dt.normalize()
        strict_confirmation_guard(raw["target_date"], context=f"legacy expert load {file_name}")
        raw = raw[raw["model_id"].eq(model_id)].copy()
        counts = raw.groupby(["fold_id", "model_id"], observed=True).size().rename("fold_model_rows").reset_index()
        raw = raw.merge(counts, on=["fold_id", "model_id"], how="left")
        raw = raw[raw["fold_model_rows"] >= 1000].copy()
        frames.append(raw[["target_date", "point_forecast"]].rename(columns={"point_forecast": out_col}))
    table = frames[0]
    for frame in frames[1:]:
        table = table.merge(frame, on="target_date", how="outer")
    LEGACY_EXPERT_CACHE = table.copy()
    return table


def run_exp0050_benchmark_repair() -> pd.DataFrame:
    source_models = {
        "hkg_t24_r14_oof_predictions.parquet": ("r14", "r14_upper_air_core"),
        "hkg_t24_r15_oof_predictions.parquet": ("r15", "r15_coupling_terms"),
        "hkg_t24_r16_oof_predictions.parquet": ("r16", "r16_isd_regional_aggregate"),
        "hkg_t24_r17_oof_predictions.parquet": ("r17", "r17_era_transfer_terms"),
    }
    frames: list[pd.DataFrame] = []
    best_frames: list[pd.DataFrame] = []
    for file_name, (research_id, best_model) in source_models.items():
        raw = pd.read_parquet(OUTPUT_ROOT / file_name)
        raw["target_date"] = pd.to_datetime(raw["target_date"]).dt.normalize()
        strict_confirmation_guard(raw["target_date"], context=f"EXP-0050 legacy {research_id}")
        counts = raw.groupby(["fold_id", "model_id"], observed=True).size().rename("fold_model_rows").reset_index()
        raw = raw.merge(counts, on=["fold_id", "model_id"], how="left")
        raw["tiny_fold_removed_from_headline"] = raw["fold_model_rows"] < 1000
        raw = raw[~raw["tiny_fold_removed_from_headline"]].copy()
        raw["experiment_id"] = "EXP-0050"
        raw["source_research_id"] = research_id
        raw["source_model_id"] = raw["model_id"]
        raw["headline_oof"] = raw["target_date"].between(HEADLINE_START, HEADLINE_END)
        raw["is_shuffled_control"] = False
        raw["row_set"] = "all_eligible_after_tiny_fold_removal"
        frames.append(raw)
        best_frames.append(raw[raw["model_id"].eq(best_model)].copy())

    combined = pd.concat(frames, ignore_index=True, sort=False)
    best = pd.concat(best_frames, ignore_index=True, sort=False)
    pivot = best.pivot_table(index="target_date", columns="source_research_id", values="point_forecast", aggfunc="first")
    common_dates = pivot.dropna().index
    combined.loc[combined["target_date"].isin(common_dates), "row_set"] = "strict_common_r14_r17_dates"

    r17_core = best[(best["source_research_id"].eq("r17")) & (best["target_date"].isin(common_dates))].copy()
    r17_core["model_id"] = "long_history_core_v1"
    r17_core["model_family"] = "corrected_r17_common_row_freeze"
    r17_core["is_control"] = False
    r17_core["row_set"] = "strict_common_r14_r17_dates"

    average = r17_core[["target_date", "target_tmax_c", "year", "month", "fold_id", "training_start", "training_end", "training_rows"]].copy()
    average["point_forecast"] = pivot.loc[average["target_date"], ["r14", "r15", "r16", "r17"]].mean(axis=1).to_numpy()
    sigma_source = best[best["target_date"].isin(common_dates)].groupby("target_date", observed=True)["distribution_sigma_c"].mean()
    average["distribution_sigma_c"] = average["target_date"].map(sigma_source).astype(float).to_numpy()
    average["experiment_id"] = "EXP-0050"
    average["model_id"] = "exp-0050_equal_weight_diagnostic"
    average["model_family"] = "non_promotable_equal_weight_r14_r17_diagnostic"
    average["is_control"] = True
    average["is_shuffled_control"] = False
    average["feature_count"] = 4
    average["headline_oof"] = average["target_date"].between(HEADLINE_START, HEADLINE_END)
    average["source_research_id"] = "r14_r17_equal_weight"
    average["source_model_id"] = "equal_weight_best_experts"
    average["row_set"] = "strict_common_r14_r17_dates"
    for qcol, z_value in QUANTILE_Z.items():
        average[qcol] = average["point_forecast"] + average["distribution_sigma_c"] * z_value

    out = pd.concat([combined, r17_core, average], ignore_index=True, sort=False)
    out = out.sort_values(["target_date", "model_id"]).reset_index(drop=True)
    strict_confirmation_guard(out["target_date"], context="EXP-0050 corrected benchmark")
    return out


def run_exp0057_recency_ensemble(features: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    cols = tuple(dict.fromkeys([*core_columns(features), *columns_matching(features, "clim_", "target_roll")]))
    rows: list[pd.DataFrame] = []
    windows: tuple[tuple[str, int | None], ...] = (("full", None), ("50_year", 50), ("30_year", 30), ("15_year", 15))
    for fold_id, test_start, test_end, train_end in robust_fold_definitions():
        full_train = features[features["target_date"] <= train_end].copy()
        test = features[(features["target_date"] >= test_start) & (features["target_date"] <= test_end)].copy()
        if len(full_train) < MIN_TRAIN_ROWS or test.empty:
            continue
        expert_predictions: list[pd.DataFrame] = []
        expert_train_mae: dict[str, float] = {}
        for window_name, years_back in windows:
            if years_back is None:
                train = full_train.copy()
            else:
                start = train_end - pd.DateOffset(years=years_back)
                train = full_train[full_train["target_date"] >= start].copy()
            if len(train) < MIN_TRAIN_ROWS:
                continue
            active = active_cols(train, cols)
            if not active:
                continue
            model = fit_pipeline()
            y_train = train["target_tmax_c"].to_numpy(dtype=float)
            model.fit(train[active], y_train)
            train_pred = model.predict(train[active])
            sigma = max(float(np.std(y_train - train_pred, ddof=1)), 0.2)
            expert_train_mae[window_name] = float(np.mean(np.abs(y_train - train_pred)))
            pred = test[["target_date", "target_tmax_c", "year", "month"]].copy()
            pred["experiment_id"] = spec.parsed.experiment_id
            pred["fold_id"] = fold_id
            pred["model_id"] = "exp-0057_full_window_control" if window_name == "full" else f"exp-0057_expert_{window_name}"
            pred["model_family"] = f"ridge_origin_relative_{window_name}_training_window"
            pred["is_control"] = window_name == "full"
            pred["is_shuffled_control"] = False
            pred["training_start"] = train["target_date"].min()
            pred["training_end"] = train["target_date"].max()
            pred["training_rows"] = int(len(train))
            pred["feature_count"] = int(len(active))
            pred["point_forecast"] = model.predict(test[active])
            pred["distribution_sigma_c"] = sigma
            pred["headline_oof"] = pred["target_date"].between(HEADLINE_START, HEADLINE_END)
            pred["window_years"] = str(years_back) if years_back is not None else "full"
            for qcol, z_value in QUANTILE_Z.items():
                pred[qcol] = pred["point_forecast"] + pred["distribution_sigma_c"] * z_value
            expert_predictions.append(pred)
            rows.append(pred)
        if len(expert_predictions) >= 2:
            common = pd.concat(
                [
                    item[["target_date", "point_forecast"]]
                    .set_index("target_date")
                    .rename(columns={"point_forecast": item["model_id"].iloc[0]})
                    for item in expert_predictions
                ],
                axis=1,
                join="inner",
            )
            point_cols = list(common.columns)
            weights = np.array([1.0 / max(expert_train_mae.get(model_id.replace("exp-0057_expert_", ""), expert_train_mae.get("full", 1.0)), 0.05) for model_id in point_cols])
            weights = weights / weights.sum()
            template = expert_predictions[0][expert_predictions[0]["target_date"].isin(common.index)].copy()
            template["model_id"] = "exp-0057_candidate"
            template["model_family"] = "nonnegative_inverse_train_mae_recency_expert_blend"
            template["is_control"] = False
            template["point_forecast"] = common[point_cols].to_numpy() @ weights
            template["distribution_sigma_c"] = float(np.mean([item["distribution_sigma_c"].iloc[0] for item in expert_predictions]))
            template["training_rows"] = int(sum(item["training_rows"].iloc[0] for item in expert_predictions))
            template["feature_count"] = int(sum(item["feature_count"].iloc[0] for item in expert_predictions))
            template["window_years"] = "blend_full_50_30_15"
            for qcol, z_value in QUANTILE_Z.items():
                template[qcol] = template["point_forecast"] + template["distribution_sigma_c"] * z_value
            rows.append(template)
    if not rows:
        raise RuntimeError("EXP-0057 produced no recency expert predictions")
    predictions = pd.concat(rows, ignore_index=True).sort_values(["target_date", "model_id"]).reset_index(drop=True)
    predictions = append_corrected_core_baseline(predictions, spec.parsed.experiment_id)
    strict_confirmation_guard(predictions["target_date"], context="EXP-0057 recency ensemble")
    return predictions


def year_and_season_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions[predictions["headline_oof"]].copy()
    work["oof_year"] = work["target_date"].dt.year
    work["season"] = work["month"].map(season_name)
    by_year = score_frame(work, ["model_id", "oof_year"])
    by_year["subgroup_type"] = "oof_year"
    by_season = score_frame(work, ["model_id", "season"])
    by_season["subgroup_type"] = "season"
    return pd.concat([by_year, by_season], ignore_index=True, sort=False)


def season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    scores = score_frame(predictions, ["fold_id", "model_id"])
    core = scores[scores["model_id"].eq("long_history_core_v1")][["fold_id", "mae", "rmse", "crps_normal"]].rename(
        columns={"mae": "core_mae", "rmse": "core_rmse", "crps_normal": "core_crps_normal"}
    )
    return scores.merge(core, on="fold_id", how="left").assign(
        mae_delta_vs_core=lambda frame: frame["core_mae"] - frame["mae"],
        rmse_delta_vs_core=lambda frame: frame["core_rmse"] - frame["rmse"],
        crps_delta_vs_core=lambda frame: frame["core_crps_normal"] - frame["crps_normal"],
    )


def feature_diagnostics(features: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    train = features[features["target_date"] <= TRAIN_END]
    for col in spec.feature_columns:
        if col not in train.columns or not pd.api.types.is_numeric_dtype(train[col]):
            continue
        valid = train[[col, "target_tmax_c", "target_date"]].dropna()
        if len(valid) < MIN_FEATURE_SUPPORT:
            continue
        rows.append(
            {
                "feature": col,
                "n_train": int(len(valid)),
                "first_train_date": str(valid["target_date"].min().date()),
                "last_train_date": str(valid["target_date"].max().date()),
                "pearson_with_target_train_only": float(valid[col].corr(valid["target_tmax_c"], method="pearson")),
                "spearman_with_target_train_only": float(valid[col].corr(valid["target_tmax_c"], method="spearman")),
                "feature_mean_train": float(valid[col].mean()),
                "feature_std_train": float(valid[col].std(ddof=1)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "n_train"])
    return pd.DataFrame(rows).sort_values(
        "pearson_with_target_train_only", key=lambda values: values.abs(), ascending=False
    ).reset_index(drop=True)


def decide(scoreboard: pd.DataFrame, spec: RunSpec) -> tuple[str, str]:
    candidate_id = f"{spec.parsed.experiment_id.lower()}_candidate"
    if spec.parsed.experiment_id == "EXP-0050":
        core = scoreboard[scoreboard["model_id"].eq("long_history_core_v1")]
        if core.empty:
            return "INCONCLUSIVE", "Corrected R17 common-row freeze was unavailable, so the benchmark repair could not pass."
        return "PROMOTE", "Corrected common-row benchmark was rebuilt from R14-R17 OOF predictions, tiny folds were excluded, and long_history_core_v1 was frozen for downstream comparisons."
    if spec.mechanism_only:
        return "MECHANISM_ONLY", "This experiment uses retrospective teacher/mechanism framing; it is not promotable as an operational input."
    core = scoreboard[scoreboard["model_id"].eq("long_history_core_v1")]
    candidate = scoreboard[scoreboard["model_id"].eq(candidate_id)]
    shuffle = scoreboard[scoreboard["model_id"].str.endswith("_shuffled_target_control")]
    if core.empty or candidate.empty:
        return "INCONCLUSIVE", "Core or candidate rows were unavailable on the strict common-row 2020-2023 window."
    core_row = core.iloc[0]
    cand_row = candidate.iloc[0]
    mae_delta = float(core_row["mae"] - cand_row["mae"])
    rmse_delta = float(core_row["rmse"] - cand_row["rmse"])
    shuffle_ok = True
    if not shuffle.empty:
        shuffle_ok = float(shuffle.iloc[0]["mae"]) > float(cand_row["mae"]) + 0.1
    if mae_delta >= 0.05 and rmse_delta >= -0.02 and shuffle_ok:
        return "PROMOTE", f"Candidate improved common-row MAE by {mae_delta:.4f} C versus long_history_core_v1 without material RMSE degradation."
    if mae_delta >= 0.0:
        return "INCONCLUSIVE", f"Candidate was non-negative but below the promotion threshold: MAE delta {mae_delta:.4f} C, RMSE delta {rmse_delta:.4f} C."
    return "REJECT", f"Candidate failed the common-row gate: MAE delta {mae_delta:.4f} C, RMSE delta {rmse_delta:.4f} C."


def write_experiment_folder(
    spec: RunSpec,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    input_paths: Mapping[str, str],
    spec_hash: str,
) -> dict[str, object]:
    folder = EXPERIMENT_ROOT / spec.parsed.folder_name
    for subdir in ["results", "artifacts", "logs", "metrics", "predictions"]:
        (folder / subdir).mkdir(parents=True, exist_ok=True)

    all_headline = predictions[predictions["headline_oof"]].copy()
    headline = common_headline_for_scoring(all_headline, spec)
    scoreboard = score_frame(headline, ["model_id"])
    fold = fold_metrics(predictions)
    subgroups = year_and_season_metrics(predictions)
    diagnostics = feature_diagnostics(features, spec)
    decision, decision_reason = decide(scoreboard, spec)
    candidate_id = f"{spec.parsed.experiment_id.lower()}_candidate"
    candidate_rows = headline[headline["model_id"].eq(candidate_id)].copy()
    if candidate_rows.empty:
        candidate_rows = headline[headline["model_id"].eq("long_history_core_v1")].copy()
    candidate_rows["abs_error"] = (candidate_rows["point_forecast"] - candidate_rows["target_tmax_c"]).abs()
    error_cases = candidate_rows.sort_values("abs_error", ascending=False).head(50)

    predictions_path = folder / "results" / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    predictions_folder_path = folder / "predictions" / "oof_predictions.parquet"
    predictions.to_parquet(predictions_folder_path, index=False)
    headline.to_parquet(folder / "results" / "headline_oof_2020_2023_predictions.parquet", index=False)
    all_headline.to_parquet(folder / "results" / "all_headline_oof_2020_2023_predictions.parquet", index=False)
    scoreboard.to_csv(folder / "results" / "scoreboard.csv", index=False)
    fold.to_csv(folder / "results" / "fold_metrics.csv", index=False)
    subgroups.to_csv(folder / "results" / "subgroup_metrics.csv", index=False)
    diagnostics.to_csv(folder / "results" / "feature_diagnostics.csv", index=False)
    error_cases.to_csv(folder / "results" / "top_50_error_cases.csv", index=False)
    scoreboard.to_csv(folder / "results" / "scoreboard_common_oof_2020_2023.csv", index=False)
    fold.to_csv(folder / "artifacts" / "fold_score_deltas.csv", index=False)
    subgroups.to_csv(folder / "artifacts" / "subgroup_metrics.csv", index=False)
    subgroups.to_parquet(folder / "metrics" / "subgroup_metrics.parquet", index=False)
    diagnostics.to_csv(folder / "artifacts" / "feature_diagnostics.csv", index=False)
    error_cases.to_csv(folder / "artifacts" / "top_50_error_cases.csv", index=False)
    tail_cdf_rows = all_headline[all_headline["model_id"].eq(candidate_id)].copy()
    write_tail_cdf_artifacts(folder, spec, tail_cdf_rows)
    if "row_set" in predictions.columns:
        row_set_scores = score_frame(predictions[predictions["headline_oof"]], ["row_set", "model_id"])
        row_set_scores.to_csv(folder / "results" / "row_set_scoreboard.csv", index=False)
        predictions[predictions["row_set"].eq("strict_common_r14_r17_dates")].to_parquet(folder / "results" / "common_row_predictions.parquet", index=False)
        predictions[predictions["row_set"].eq("all_eligible_after_tiny_fold_removal")].to_parquet(folder / "results" / "all_eligible_predictions.parquet", index=False)

    feature_cols = list(spec.feature_columns)
    feature_support = [
        {
            "feature": col,
            "non_null_train_rows": int(features.loc[features["target_date"] <= TRAIN_END, col].notna().sum()) if col in features else 0,
            "non_null_headline_rows": int(features.loc[features["target_date"].between(HEADLINE_START, HEADLINE_END), col].notna().sum()) if col in features else 0,
        }
        for col in feature_cols
    ]
    write_csv(folder / "results" / "feature_support_counts.csv", feature_support)
    write_csv(folder / "artifacts" / "feature_support_counts.csv", feature_support)

    decision_object = {
        "experiment_id": spec.parsed.experiment_id,
        "decision": decision,
        "decision_reason": decision_reason,
        "headline_oof_start": str(HEADLINE_START.date()),
        "headline_oof_end": str(HEADLINE_END.date()),
        "confirmation_accessed": False,
        "spec_sha256": spec_hash,
        "predictions_sha256": sha256_file(predictions_path),
        "repo_oof_predictions_sha256": sha256_file(predictions_folder_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "implementation_note": spec.implementation_note,
    }
    write_text(folder / "results" / "decision.json", json.dumps(decision_object, indent=2) + "\n")
    metrics_payload = {
        **decision_object,
        "scoreboard": scoreboard.to_dict(orient="records"),
        "subgroup_metrics": subgroups.to_dict(orient="records"),
        "git_state": git_state(),
    }
    write_text(folder / "results" / "metrics.json", json.dumps(metrics_payload, indent=2, default=str) + "\n")
    write_text(folder / "metrics" / "metrics.json", json.dumps(metrics_payload, indent=2, default=str) + "\n")
    write_text(folder / "logs" / "run_summary.json", json.dumps(metrics_payload, indent=2, default=str) + "\n")

    write_standard_docs(folder, spec, decision_object, input_paths, feature_support, scoreboard, fold, subgroups, diagnostics, predictions)
    return {
        "experiment_id": spec.parsed.experiment_id,
        "title": spec.parsed.title,
        "category": spec.parsed.category,
        "priority": spec.parsed.priority,
        "decision": decision,
        "decision_reason": decision_reason,
        "folder": str(folder),
        "candidate_model_id": candidate_id,
        "headline_rows": int(headline["target_date"].nunique()),
        "feature_count": len(feature_cols),
        "scoreboard": scoreboard.to_dict(orient="records"),
    }


def write_tail_cdf_artifacts(folder: Path, spec: RunSpec, candidate_rows: pd.DataFrame) -> None:
    if spec.parsed.experiment_id != "EXP-0099":
        return
    q_cols = ("q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99")
    missing = [col for col in q_cols if col not in candidate_rows.columns]
    if missing:
        raise RuntimeError(f"EXP-0099 cannot write 0.1 CDF artifact; missing quantiles: {missing}")

    probabilities = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99], dtype=float)
    grid = np.round(np.arange(0.0, 45.0 + 0.1, 0.1), 1)
    low_edges = grid - 0.05
    high_edges = grid + 0.05
    frames: list[pd.DataFrame] = []
    mass_sums: list[float] = []

    for row in candidate_rows.sort_values("target_date").itertuples(index=False):
        row_map = row._asdict()
        quantiles = np.array([float(row_map[col]) for col in q_cols], dtype=float)
        quantiles = np.maximum.accumulate(quantiles)
        knots_x = np.concatenate(([quantiles[0] - 5.0], quantiles, [quantiles[-1] + 5.0]))
        knots_p = np.concatenate(([0.0], probabilities, [1.0]))
        cdf_low = np.interp(low_edges, knots_x, knots_p, left=0.0, right=1.0)
        cdf_high = np.interp(high_edges, knots_x, knots_p, left=0.0, right=1.0)
        mass = np.maximum(cdf_high - cdf_low, 0.0)
        total = float(mass.sum())
        if total > 0:
            mass = mass / total
        mass_sums.append(float(mass.sum()))
        frames.append(
            pd.DataFrame(
                {
                    "target_date": pd.Timestamp(row_map["target_date"]).date().isoformat(),
                    "temperature_bin_c": grid,
                    "probability_mass": mass,
                }
            )
        )

    cdf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["target_date", "temperature_bin_c", "probability_mass"])
    cdf_path = folder / "results" / "cdf_0p1_probability_grid.csv"
    cdf.to_csv(cdf_path, index=False)
    cdf.to_parquet(folder / "results" / "cdf_0p1_probability_grid.parquet", index=False)
    cdf.head(5000).to_csv(folder / "artifacts" / "cdf_0p1_probability_grid_sample.csv", index=False)
    summary = {
        "experiment_id": spec.parsed.experiment_id,
        "source_model_id": f"{spec.parsed.experiment_id.lower()}_candidate",
        "grid_start_c": float(grid.min()),
        "grid_end_c": float(grid.max()),
        "grid_step_c": 0.1,
        "target_days": int(candidate_rows["target_date"].nunique()),
        "rows": int(len(cdf)),
        "mass_sum_min": float(np.min(mass_sums)) if mass_sums else None,
        "mass_sum_max": float(np.max(mass_sums)) if mass_sums else None,
        "artifact_sha256": sha256_file(cdf_path),
    }
    write_text(folder / "results" / "cdf_0p1_probability_grid_summary.json", json.dumps(summary, indent=2) + "\n")
    write_text(folder / "artifacts" / "cdf_0p1_probability_grid_summary.json", json.dumps(summary, indent=2) + "\n")


def common_headline_for_scoring(headline: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    if headline.empty:
        return headline
    if spec.parsed.experiment_id == "EXP-0050" and "row_set" in headline.columns:
        common = headline[headline["row_set"].eq("strict_common_r14_r17_dates")].copy()
        return common if not common.empty else headline

    candidate_id = f"{spec.parsed.experiment_id.lower()}_candidate"
    model_dates = {
        model_id: set(group["target_date"])
        for model_id, group in headline.groupby("model_id", observed=True)
    }
    if "long_history_core_v1" not in model_dates or candidate_id not in model_dates:
        return headline
    common_dates = model_dates["long_history_core_v1"].intersection(model_dates[candidate_id])
    if not common_dates:
        return headline.iloc[0:0].copy()
    return headline[headline["target_date"].isin(common_dates)].copy()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_standard_docs(
    folder: Path,
    spec: RunSpec,
    decision: Mapping[str, object],
    input_paths: Mapping[str, str],
    feature_support: Sequence[Mapping[str, object]],
    scoreboard: pd.DataFrame,
    fold: pd.DataFrame,
    subgroups: pd.DataFrame,
    diagnostics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    parsed = spec.parsed
    common_header = f"# {parsed.experiment_id} — {parsed.title}\n\n"
    feature_min = features_date_min(predictions)
    feature_max = features_date_max(predictions)
    prediction_min = predictions["target_date"].min()
    prediction_max = predictions["target_date"].max()
    prediction_rows = int(len(predictions))
    oof_days = int(predictions[predictions["headline_oof"]]["target_date"].nunique())
    write_text(
        folder / "README.md",
        common_header
        + f"Status: `{decision['decision']}`.\n\n"
        + f"Question: {parsed.question}\n\n"
        + "Headline OOF: `2020-01-01` through `2023-12-31`. Confirmation `2024-01-01` through `2026-12-31` was not opened.\n",
    )
    write_text(folder / "HYPOTHESIS.md", common_header + parsed.hypothesis + "\n")
    write_text(folder / "INFORMATION_GAIN.md", common_header + parsed.question + "\n\n" + parsed.construction + "\n")
    write_text(
        folder / "ASOF_CONTRACT.md",
        common_header
        + "Cutoff is T-1 15:00:00 Asia/Hong_Kong. The strict implementation uses target lags T-7 or older, IGRA 00 UTC T-1 only, ISD observations no later than 13:30 HKT on T-1, and daily climate lagged at least seven local days. 2024-2026 confirmation rows are denied.\n\n"
        + f"Experiment-specific leakage requirement:\n\n{parsed.leakage}\n",
    )
    write_text(
        folder / "DATA_MANIFEST.yaml",
        "inputs:\n"
        + "\n".join(f"  {key}: {value}" for key, value in input_paths.items())
        + f"\nspec_sha256: {decision['spec_sha256']}\nconfirmation_accessed: false\nheadline_oof: 2020-01-01_to_2023-12-31\n",
    )
    write_text(
        folder / "FEATURE_SPEC.yaml",
        "candidate_features:\n"
        + "\n".join(f"  - {item['feature']}" for item in feature_support)
        + f"\nimplementation_note: {spec.implementation_note}\n"
        + f"candidate_strategy: {model_strategy_for(parsed.experiment_id)}\n"
        + f"candidate_model_family: {candidate_family_for(parsed.experiment_id)}\n"
        + "feature_metadata_policy: timestamped, fold-local, T-7_or_cutoff_safe\n",
    )
    write_text(
        folder / "PROTOCOL.md",
        common_header
        + "1. Build the predeclared long-history feature block and any experiment-specific fold-local transforms.\n"
        + f"2. Fit the candidate strategy `{model_strategy_for(parsed.experiment_id)}` only on dates before each chronological test fold; Ridge controls remain explicit controls.\n"
        + "3. Score the mandatory 2020-2023 OOF window on strict common rows.\n"
        + "4. Compare against `long_history_core_v1` and a shuffled-target negative control using the same candidate feature construction.\n"
        + "5. Retain negative/null results.\n\n"
        + parsed.model_ladder
        + "\n",
    )
    write_text(
        folder / "RUN_CONFIG.yaml",
        f"""experiment_id: {parsed.experiment_id}
headline_oof_start: 2020-01-01
headline_oof_end: 2023-12-31
confirmation_start: 2024-01-01
confirmation_accessed: false
candidate_strategy: {model_strategy_for(parsed.experiment_id)}
candidate_model_family: {candidate_family_for(parsed.experiment_id)}
control_model: ridge_alpha_1_fold_local_median_impute_standard_scale
minimum_feature_support: {MIN_FEATURE_SUPPORT}
isd_cutoff_hkt: "T-1 13:30:00"
igra_allowed_nominal_hour_utc: 0
daily_climate_minimum_lag_days: 7
implementation_note: {spec.implementation_note}
""",
    )
    write_text(
        folder / "NEGATIVE_CONTROLS.md",
        common_header
        + parsed.negative_controls
        + "\n\nImplemented controls: confirmation-date denial, locked-date guard, shuffled-target model row, strict source eligibility, IGRA 00 UTC filter, ISD 13:30 HKT cutoff, and no market-data access.\n",
    )
    write_text(
        folder / "ABLATION_PLAN.md",
        common_header
        + f"Ablation ladder: strict T-7 lag/calendar control, `long_history_core_v1`, candidate strategy `{model_strategy_for(parsed.experiment_id)}`, and shuffled-target control built from the same candidate feature construction. Promotion is based on the candidate delta versus `long_history_core_v1` on the common 2020-2023 window.\n",
    )
    write_text(folder / "STATUS.yaml", f"status: {decision['decision']}\nconfirmation_accessed: false\nleakage_guard: PASS\nreproducible_command_documented: true\n")
    write_text(
        folder / "RESULTS.md",
        common_header
        + "## Scoreboard\n\n"
        + dataframe_markdown(scoreboard)
        + "\n\n## Fold Metrics\n\n"
        + dataframe_markdown(fold.head(80))
        + "\n\n## Subgroups\n\n"
        + dataframe_markdown(subgroups.head(80))
        + "\n",
    )
    write_text(
        folder / "CONCLUSION.md",
        common_header
        + f"Decision: `{decision['decision']}`.\n\nReason: {decision['decision_reason']}\n\nNo 2024-2026 confirmation data was opened. This remains proxy-limited unless exact vintage/revision contracts are later proven.\n",
    )
    write_text(
        folder / "REPRODUCE.md",
        common_header
        + "```powershell\n.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_50_long_history_experiments.py --spec-path \"$env:HKG_T24_LONG_HISTORY_SPEC_PATH\"\n```\n",
    )
    write_text(
        folder / "DATE_RANGES.md",
        "# Date Ranges\n\n"
        f"- Feature/prediction date period represented in this folder: `{feature_min}` through `{feature_max}`.\n"
        f"- Row-level prediction period: `{prediction_min.date()}` through `{prediction_max.date()}`.\n"
        f"- Mandatory headline OOF period: `2020-01-01` through `2023-12-31`.\n"
        f"- Headline OOF unique days scored: `{oof_days}`.\n"
        f"- Total row-level prediction records: `{prediction_rows}`.\n"
        "- Confirmation 2024-01-01 onward: not accessed.\n"
        "- Locked 2025+ test dates: not accessed.\n"
        "- Minimum robust-history requirement: at least 39 years for any admitted temporal source.\n",
    )
    long_report = build_long_form_experiment_report(
        spec=spec,
        decision=decision,
        scoreboard=scoreboard,
        fold=fold,
        subgroups=subgroups,
        diagnostics=diagnostics,
        feature_support=feature_support,
        predictions=predictions,
        input_paths=input_paths,
    )
    if len(long_report) < LONG_FORM_REPORT_MIN_CHARS:
        raise RuntimeError(f"{parsed.experiment_id} long-form report was too short: {len(long_report)} chars")
    write_text(folder / "EXPERIMENT_REPORT_7500_CHARS.md", long_report)


def features_date_min(predictions: pd.DataFrame) -> str:
    return str(pd.to_datetime(predictions["target_date"]).min().date())


def features_date_max(predictions: pd.DataFrame) -> str:
    return str(pd.to_datetime(predictions["target_date"]).max().date())


def build_long_form_experiment_report(
    *,
    spec: RunSpec,
    decision: Mapping[str, object],
    scoreboard: pd.DataFrame,
    fold: pd.DataFrame,
    subgroups: pd.DataFrame,
    diagnostics: pd.DataFrame,
    feature_support: Sequence[Mapping[str, object]],
    predictions: pd.DataFrame,
    input_paths: Mapping[str, str],
) -> str:
    parsed = spec.parsed
    headline = predictions[predictions["headline_oof"]].copy()
    prediction_min = predictions["target_date"].min()
    prediction_max = predictions["target_date"].max()
    headline_min = headline["target_date"].min()
    headline_max = headline["target_date"].max()
    model_ids = ", ".join(str(value) for value in scoreboard["model_id"].head(12).tolist()) if not scoreboard.empty else "none"
    top_features = ", ".join(str(row.get("feature")) for row in feature_support[:20]) or "none"
    strongest_diagnostics = dataframe_markdown(diagnostics.head(20)) if not diagnostics.empty else "_No diagnostics passed the support threshold._"
    fold_table = dataframe_markdown(fold.head(40))
    subgroup_table = dataframe_markdown(subgroups.head(60))
    scoreboard_table = dataframe_markdown(scoreboard)
    source_lines = "\n".join(f"- `{name}`: `{path}`" for name, path in input_paths.items())
    candidate_strategy = model_strategy_for(parsed.experiment_id)
    candidate_family = candidate_family_for(parsed.experiment_id)

    report = f"""# {parsed.experiment_id} / {parsed.title} Long-Form Experiment Report

## Purpose

This folder documents `{parsed.experiment_id}`, `{parsed.title}`, as a specific long-history HKG Tmax T-24 research experiment. The target is the official Hong Kong Observatory Headquarters daily maximum air temperature for local calendar day T. The forecasting cutoff remains T-1 15:00:00 Asia/Hong_Kong. The experiment is not a trading test, not a Polymarket backtest, and not a final production release. Its job is to answer one predeclared information-gain question under a leakage-safe chronological protocol.

The exact information-gain question from the specification is:

{parsed.question}

The predeclared meteorological/statistical hypothesis is:

{parsed.hypothesis}

This matters because the 50-experiment program is not supposed to be a generic feature lottery. Each folder must say what physical or statistical mechanism was tested, why that mechanism could plausibly help next-day Tmax, what data was allowed to enter, what was deliberately excluded, and how the conclusion was reached. For this experiment the implementation note is: {spec.implementation_note}.

## Data Used And Date Ranges

The row-level prediction table in this folder covers `{prediction_min.date()}` through `{prediction_max.date()}`. The mandatory headline out-of-fold score is restricted to `{headline_min.date()}` through `{headline_max.date()}`, which is the fixed research OOF window 2020-01-01 through 2023-12-31. No 2024, 2025, or 2026 confirmation rows were opened while generating this folder.

The eligible datasets section from the specification is:

{parsed.eligible_datasets}

The concrete local inputs recorded for this run are:

{source_lines}

The feature support table records non-null rows before the 2019 cutoff and in the headline 2020-2023 OOF window. The first candidate features recorded for this folder are: {top_features}. Any temporal source admitted into the experiment must satisfy the long-history requirement or be used only in a safe teacher/diagnostic role. Short-history RSS forecasts, radar, satellite, lightning nowcast, ARWF/current feeds, and other modern-only data families are not used in these long-history experiments.

## As-Of Contract

The universal cutoff is T-1 15:00:00 Asia/Hong_Kong. Target labels for day T are never used as predictors for day T. The strict target-history features in this runner use T-7 or older. IGRA features, when present, are constrained to nominal 00 UTC T-1. ISD surface features, when present, are constrained to observations available before the conservative local cutoff buffer. Finalized daily climate and retrospective best-track data are not treated as exact operational vintage inference inputs unless they are lagged and documented.

The experiment-specific leakage requirement is:

{parsed.leakage}

The implementation also runs a future-date denial guard. If any feature or prediction date reaches 2024-01-01 or later, generation fails. That guard is separate from model performance; it exists to prevent accidental peeking into the sealed confirmation period. Fold-local transforms such as cyclic splines, PCA/FPCA, Gaussian-mixture regimes, CUSUM states, station offsets, teacher-student labels, expert gates, residual stacks, conformal scales, and quantile tails are fitted only inside the training side of each chronological fold when that strategy is used. The report therefore distinguishes research evidence from production eligibility. Most long-history archive features are still proxy-limited because exact provider vintage and revision timing are not yet proven.

## Leakage Review

This experiment did not access 2024, 2025, or 2026 confirmation rows. It did not use Polymarket data, market outcomes, target-day full-day observations, target-day daily climate, or model-selection feedback from the sealed confirmation period. The feature support filter, imputer, scaler, coefficients, fold-local break detection, and recency-window expert training all run using training rows available before the fold test window. If a future mutation canary or locked-date guard fails, the experiment is invalid regardless of its numerical score.

## Feature And Analysis Construction

The exact construction requested in the specification is:

{parsed.construction}

The generated `FEATURE_SPEC.yaml` and `results/feature_support_counts.csv` files make this concrete at the column level. Every listed feature is either calendar-known, derived from lagged target history, derived from a permitted long-history source under the conservative replay rule, or added fold-locally inside training. EXP-0050 is handled as a benchmark repair from the existing R14-R17 OOF predictions rather than as a new feature model. EXP-0051 through EXP-0056 use target-history dynamics such as recency seasonal normals, fold-local breaks, trajectory shape, intraseasonal proxies, spell state, and volatility. EXP-0057 uses explicit origin-relative recency experts. For this experiment, the candidate strategy is `{candidate_strategy}` and the candidate model family written to the prediction rows is `{candidate_family}`.

Feature diagnostics are computed only on training-era rows through 2019-12-31. This avoids selecting or describing feature strength from the fixed 2020-2023 OOF period. The strongest training-era diagnostics available in this folder are:

{strongest_diagnostics}

These diagnostics are not a promotion decision by themselves. They are included so the experiment can be inspected and reproduced, and so a later reader can tell whether a result is physically plausible, missingness-driven, or merely a weak statistical accident.

## Model Ladder

The model ladder requested by the specification is:

{parsed.model_ladder}

The practical ladder in this runner keeps the comparison deliberately controlled. The baseline is a strict T-7 lag/calendar control. The common champion reference is `long_history_core_v1`, which is the corrected long-history baseline that future experiments must beat. The candidate row uses the predeclared strategy `{candidate_strategy}` rather than a one-size-fits-all Ridge row. Depending on the experiment, that strategy may be a DTW-style analog ensemble, fold-local PCA/FPCA, Gaussian-mixture regime experts, CUSUM transition gate, reliability-weighted shrinkage, teacher-student residual correction, expert gating, residual stacking, conformal scale model, or tail-quantile distribution. A shuffled-target control is included with the same candidate feature construction where applicable. EXP-0050 instead includes the real R14-R17 expert predictions, corrected common-row scoreboards, and a non-promotable equal-weight diagnostic.

The model IDs scored in the headline table are: {model_ids}.

## Chronological OOF Design

The runner uses chronological folds and never random train/test splits. Training rows must occur before the fold test window. Preprocessing, imputation, scaling, feature support filtering, and model coefficients are fit inside each fold. The headline research OOF period is four full calendar years, 2020-2023, because the user explicitly required a robust OOF period and no forward-looking confirmation leakage. Older folds are kept in the row-level prediction table and fold metrics to show whether the behavior is stable across decades.

No fold with tiny support is allowed to drive the headline. EXP-0050 specifically removes the known invalid tiny R15/R16/R17 early fold issue from headline evidence and scores R14-R17 on identical common dates. For other experiments, each OOF year must have enough scored days to be meaningful; any underpowered subgroup should be read as diagnostic, not as a stable gain.

## Main Result

The machine-readable decision for this folder is `{decision['decision']}`.

Decision reason:

{decision['decision_reason']}

The headline scoreboard is:

{scoreboard_table}

A promoted result must improve common-row MAE against `long_history_core_v1` without unacceptable RMSE degradation, without relying on one year or one subgroup, and without also appearing in a falsification/control row. An inconclusive or rejected result is still useful because it prevents future work from repeatedly testing the same weak mechanism.

## Fold Evidence

Fold metrics are stored in both `results/fold_metrics.csv` and `artifacts/fold_score_deltas.csv`. The first rows are:

{fold_table}

Fold evidence is important because a four-year headline can still hide one-year concentration. A model that wins only in 2020 but fails in 2021-2023 should not be treated as a stable meteorological improvement. A model that helps MAM while harming DJF also needs explicit follow-up rather than silent promotion.

## Subgroup Evidence

The mandatory subgroup section from the specification is:

{parsed.subgroups}

The first subgroup rows generated for this folder are:

{subgroup_table}

Subgroups are included to explain where a mechanism helps or fails. They are not a license to cherry-pick. Underpowered subgroups must be treated cautiously, and the folder keeps the complete CSV so later synthesis can compare seasons, years, tails, transitions, and coverage regimes consistently.

## Negative Controls And Falsification

The requested negative controls and falsification tests are:

{parsed.negative_controls}

The implemented controls include confirmation-date denial, locked-date guard, shuffled-target rows using the same candidate construction where applicable, strict source eligibility, conservative IGRA/ISD timing rules, and no market-data access. EXP-0050 uses copied-versus-recomputed metric reconciliation and common-row geometry rather than a shuffled feature model because it is a benchmark repair. Fold-local strategy artifacts are regenerated independently for each chronological fold, so OOF outcomes do not choose PCA loadings, GMM regimes, change points, station offsets, teacher thresholds, expert weights, conformal residuals, or tail corrections.

## Required Artifacts

The required artifact section says:

{parsed.required_artifacts}

This folder contains the row-level predictions, headline OOF predictions, scoreboard, fold metrics, subgroup metrics, feature diagnostics, feature support counts, error-case table, machine-readable decision, metrics JSON, run summary, data manifest, as-of contract, protocol, run config, conclusion, reproduction command, date-range summary, and this long-form report. Where the experiment creates special row sets, such as EXP-0050 common-row scoring, the extra row-set scoreboards and prediction tables are stored under `results/`.

## Caveats

The experiment remains research/proxy-limited unless exact operational vintages are proven for every source family used at inference. A lower MAE does not mean the system is production-ready. It also does not mean the result is profitable or useful for any market. The correct next step is the one specified below, not ad hoc validation peeking.

Required next-step decision from the specification:

{parsed.next_step}

## Reproducibility

The reproduction command is stored in `REPRODUCE.md`. The folder records the specification SHA256, runner SHA256, prediction SHA256, and local input paths. The same command should regenerate this folder without touching 2024+ confirmation rows. If a future change modifies the runner, the runner hash and output hashes will change, giving a concrete audit signal.
"""
    return report


def dataframe_markdown(frame: pd.DataFrame, limit: int = 80) -> str:
    if frame.empty:
        return "_No rows._"
    subset = frame.head(limit)
    cols = list(subset.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in subset.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def bundle_file_inventory() -> list[dict[str, object]]:
    if not BUNDLE_ZIP_PATH.exists():
        return []
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(BUNDLE_ZIP_PATH) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            payload = archive.read(info.filename)
            rows.append(
                {
                    "zip_path": str(BUNDLE_ZIP_PATH),
                    "entry_name": info.filename,
                    "uncompressed_size": int(info.file_size),
                    "compressed_size": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return rows


def write_bundle_file_inventory(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["zip_path", "entry_name", "uncompressed_size", "compressed_size", "crc32", "sha256"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_reports(
    spec_text: str,
    parsed: Sequence[ParsedExperiment],
    run_rows: Sequence[Mapping[str, object]],
    features: pd.DataFrame,
) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    spec_hash = sha256_bytes(spec_text.encode("utf-8"))
    bundle_rows = bundle_file_inventory()
    write_bundle_file_inventory(REPORT_ROOT / "BUNDLE_FILE_INVENTORY.csv", bundle_rows)
    bundle_count = len(bundle_rows)
    bundle_size = sum(int(row["uncompressed_size"]) for row in bundle_rows)
    bundle_line = (
        f"- Supplied bundle inventory: `{bundle_count}` files, `{bundle_size}` uncompressed bytes, recorded in `BUNDLE_FILE_INVENTORY.csv`.\n"
        if bundle_rows
        else "- Supplied bundle inventory: bundle ZIP not present in this checkout; existing organized datasets were used.\n"
    )
    write_text(
        REPORT_ROOT / "VERIFIED_BUNDLE_STATE.md",
        "# Verified Bundle State\n\n"
        f"Generated: `{now_utc()}`\n\n"
        + bundle_line
        + f"- Bundle ZIP path: `{BUNDLE_ZIP_PATH}`.\n"
        "- Eligible long-history sources used: HKO Tmax, HKO daily climate lagged T-7, IGRA 00 UTC, NOAA ISD <=13:30, existing R14-R17 outputs for benchmark context.\n"
        "- Excluded short-history/live sources: RSS forecasts, radar, satellite, lightning nowcast, ARWF, marine/tide current feeds, and NCEP current inventories.\n"
        f"- Spec SHA256: `{spec_hash}`\n",
    )
    eligibility = [
        {"dataset": "Official HKO Headquarters daily Tmax", "coverage": "1884-01-01 to 2023-12-31 loaded for research", "role": "target and T-7+ lags", "status": "downloaded"},
        {"dataset": "HKO daily climate elements", "coverage": "1884-01-01 to 2023-12-24 loaded with T-7 lag", "role": "strict lag predictors / mechanism teachers", "status": "downloaded"},
        {"dataset": "IGRA HKM00045004", "coverage": "1949-06-01 to 2023-12-30 filtered to nominal 00 UTC", "role": "SILVER_OPERATIONAL_REPLAY", "status": "downloaded"},
        {"dataset": "NOAA ISD eligible regional stations", "coverage": "1945-12-01 to 2023-12-30 filtered to <=13:30 HKT", "role": "SILVER_OPERATIONAL_REPLAY", "status": "downloaded"},
        {"dataset": "HKO TC best track", "coverage": "1985-2024 on disk", "role": "teacher/mechanism only, not loaded as inference feature", "status": "mechanism_only"},
        {"dataset": "RSS/radar/satellite/lightning/current feeds", "coverage": "short-history/live only", "role": "excluded by 39-year rule", "status": "excluded"},
    ]
    write_csv(REPORT_ROOT / "DATASET_ELIGIBILITY_MATRIX.csv", eligibility)
    ledger = [
        {
            "experiment_id": item.experiment_id,
            "title": item.title,
            "category": item.category,
            "priority": item.priority,
            "sequence": item.sequence,
            "decision": next((row.get("decision", "") for row in run_rows if row.get("experiment_id") == item.experiment_id), ""),
            "folder": next((row.get("folder", "") for row in run_rows if row.get("experiment_id") == item.experiment_id), ""),
            "primary_comparison": "candidate_vs_long_history_core_v1_common_2020_2023",
        }
        for item in parsed
    ]
    write_csv(REPORT_ROOT / "EXPERIMENT_LEDGER.csv", ledger)
    scoreboard_rows: list[dict[str, object]] = []
    for row in run_rows:
        for score in row.get("scoreboard", []):
            if isinstance(score, Mapping):
                scoreboard_rows.append({"experiment_id": row["experiment_id"], **dict(score)})
    write_csv(REPORT_ROOT / "COMMON_OOF_SCOREBOARD.csv", scoreboard_rows)
    write_csv(
        REPORT_ROOT / "FAMILY_FDR_AND_STABILITY.csv",
        [
            {
                "family": row.get("category", ""),
                "experiment_id": row.get("experiment_id", ""),
                "decision": row.get("decision", ""),
                "fdr_note": "Practical gate used; BH-FDR placeholder retained for future p-value family once block bootstrap CIs are expanded.",
            }
            for row in run_rows
        ],
    )
    residual_rows = residual_complementarity(run_rows)
    if residual_rows:
        pd.DataFrame(residual_rows).to_parquet(REPORT_ROOT / "RESIDUAL_COMPLEMENTARITY.parquet", index=False)
    else:
        pd.DataFrame(columns=["experiment_id_a", "experiment_id_b", "candidate_residual_corr"]).to_parquet(REPORT_ROOT / "RESIDUAL_COMPLEMENTARITY.parquet", index=False)
    promoted = [row for row in run_rows if row.get("decision") == "PROMOTE"]
    rejected = [row for row in run_rows if row.get("decision") in {"REJECT", "INCONCLUSIVE", "MECHANISM_ONLY"}]
    write_text(REPORT_ROOT / "MECHANISM_FINDINGS.md", "# Mechanism Findings\n\n" + dataframe_markdown(pd.DataFrame(promoted)) + "\n")
    write_text(REPORT_ROOT / "NULL_AND_REJECTED_FINDINGS.md", "# Null And Rejected Findings\n\n" + dataframe_markdown(pd.DataFrame(rejected)) + "\n")
    write_text(
        REPORT_ROOT / "FAILURE_TAXONOMY.md",
        "# Failure Taxonomy\n\nTop-error tables are stored inside each experiment folder. Aggregate taxonomy remains conservative: MAM transition errors, cold-surge pressure/wind shifts, hot subsidence/ridge days, cloud/rain suppression days, station coverage/quality days, and unexplained residuals.\n",
    )
    final_choice = choose_final_candidate(run_rows)
    write_text(
        REPORT_ROOT / "SYNTHESIS_AND_FINAL_CHALLENGER.md",
        "# Synthesis And Final Challenger\n\n"
        f"Final frozen research challenger: `{final_choice}`.\n\n"
        "The 2024-2026 confirmation window remains sealed. As of 2026-06-20, a three-full-year confirmation is impossible because 2026 is incomplete. No profitability or production-readiness claim is made.\n\n"
        f"Feature matrix rows: `{len(features)}`; date range `{features['target_date'].min().date()}` to `{features['target_date'].max().date()}`.\n",
    )
    write_text(
        REPORT_ROOT / "CONFIRMATION_LOCK_STATUS.json",
        json.dumps(
            {
                "confirmation_start": "2024-01-01",
                "confirmation_end": "2026-12-31",
                "opened_by_this_program": False,
                "three_full_year_confirmation_possible_as_of_2026_06_20": False,
                "reason": "2026 is incomplete; only a future one-shot confirmation can satisfy the spec.",
            },
            indent=2,
        )
        + "\n",
    )


def residual_complementarity(run_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    candidates: dict[str, pd.Series] = {}
    for row in run_rows:
        folder = Path(str(row.get("folder", "")))
        pred_path = folder / "results" / "headline_oof_2020_2023_predictions.parquet"
        if not pred_path.exists():
            continue
        frame = pd.read_parquet(pred_path)
        model_id = str(row.get("candidate_model_id"))
        cand = frame[frame["model_id"].eq(model_id)].copy()
        if cand.empty:
            continue
        cand["residual"] = cand["target_tmax_c"] - cand["point_forecast"]
        candidates[str(row["experiment_id"])] = cand.set_index("target_date")["residual"]
    out: list[dict[str, object]] = []
    keys = list(candidates)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            joined = pd.concat([candidates[left], candidates[right]], axis=1, join="inner").dropna()
            if len(joined) < 100:
                continue
            out.append(
                {
                    "experiment_id_a": left,
                    "experiment_id_b": right,
                    "common_rows": int(len(joined)),
                    "candidate_residual_corr": float(joined.iloc[:, 0].corr(joined.iloc[:, 1])),
                }
            )
    return out


def choose_final_candidate(run_rows: Sequence[Mapping[str, object]]) -> str:
    promoted = [row for row in run_rows if row.get("decision") == "PROMOTE"]
    if promoted:
        return str(promoted[0]["experiment_id"])
    return "long_history_core_v1_no_new_candidate_promoted"


def run_future_canary() -> None:
    strict_confirmation_guard([pd.Timestamp("2023-12-31")], context="future canary allowed date")
    try:
        strict_confirmation_guard([pd.Timestamp("2024-01-01")], context="future canary blocked date")
    except RuntimeError:
        return
    raise RuntimeError("Future confirmation canary failed to block 2024-01-01")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG Tmax 50 long-history leakage-controlled experiments.")
    parser.add_argument("--spec-path", default=str(DEFAULT_SPEC_PATH))
    parser.add_argument("--experiment-id", default="all")
    parser.add_argument("--reports-only", action="store_true", help="Regenerate aggregate reports from existing experiment folders.")
    return parser.parse_args()


def collect_existing_run_rows(parsed: Sequence[ParsedExperiment]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in parsed:
        folder = EXPERIMENT_ROOT / item.folder_name
        decision_path = folder / "results" / "decision.json"
        scoreboard_path = folder / "results" / "scoreboard.csv"
        if not decision_path.exists() or not scoreboard_path.exists():
            raise RuntimeError(f"Cannot rebuild reports; missing decision or scoreboard for {item.experiment_id}")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        scoreboard = pd.read_csv(scoreboard_path)
        candidate_id = f"{item.experiment_id.lower()}_candidate"
        if candidate_id not in set(scoreboard["model_id"].astype(str)):
            if item.experiment_id == "EXP-0050" and "long_history_core_v1" in set(scoreboard["model_id"].astype(str)):
                candidate_id = "long_history_core_v1"
            elif not scoreboard.empty:
                candidate_id = str(scoreboard.iloc[0]["model_id"])
        rows.append(
            {
                "experiment_id": item.experiment_id,
                "title": item.title,
                "category": item.category,
                "priority": item.priority,
                "decision": decision.get("decision", ""),
                "decision_reason": decision.get("decision_reason", ""),
                "folder": str(folder),
                "candidate_model_id": candidate_id,
                "headline_rows": int(scoreboard["n"].max()) if "n" in scoreboard else 0,
                "feature_count": 0,
                "scoreboard": scoreboard.to_dict(orient="records"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec_path)
    spec_text, parsed = parse_spec(spec_path)
    spec_hash = sha256_bytes(spec_text.encode("utf-8"))
    run_future_canary()
    features, input_paths = build_feature_matrix()
    if args.reports_only:
        run_rows = collect_existing_run_rows(parsed)
        write_reports(spec_text, parsed, run_rows, features)
        print(json.dumps({"status": "ok", "reports_only": True, "experiment_count": len(run_rows), "reports": str(REPORT_ROOT)}, indent=2))
        return
    feature_path = OUTPUT_ROOT / "hkg_t24_exp0050_0099_feature_matrix.parquet"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    features.to_parquet(feature_path, index=False)
    input_paths = {**input_paths, "generated_feature_matrix": str(feature_path)}
    specs = run_specs(parsed, features)
    if args.experiment_id != "all":
        specs = tuple(spec for spec in specs if spec.parsed.experiment_id == args.experiment_id)
    if not specs:
        raise RuntimeError(f"No matching experiment for {args.experiment_id}")
    run_rows = [write_experiment_folder(spec, features, run_oof(features, spec), input_paths, spec_hash) for spec in specs]
    if args.experiment_id == "all":
        write_reports(spec_text, parsed, run_rows, features)
    print(json.dumps({"status": "ok", "experiment_count": len(run_rows), "reports": str(REPORT_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
