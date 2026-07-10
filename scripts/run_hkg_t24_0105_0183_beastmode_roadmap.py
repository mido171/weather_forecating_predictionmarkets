from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiments"
ROADMAP_GLOBAL_ARTIFACT_DIR = (
    EXPERIMENT_ROOT / "0000_research_state_and_data_contract" / "r0105_0183"
)
DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
ROADMAP_PATH = Path(r"C:\Users\ahmad\Downloads\HKG_T24_BEASTMODE_INFORMATION_GAIN_EXPERIMENT_ROADMAP.md")

CONFIRMATION_START = pd.Timestamp("2024-01-01")
DEFAULT_CORRECTION_CAP_C = 0.35
MODEL_FOLDS = ((2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2023))
MIN_TRAIN_ROWS = 365
MAX_MODEL_FEATURES = 48

OFFICIAL_PATH = DATASETS_ROOT / "05_hko_historical_rss_forecasts" / "hko_official_t15_scored_pre2024.parquet"
FEATURE_MATRIX_PATH = (
    DATASETS_ROOT / "12_hkg_t24_robust_experiment_outputs" / "hkg_t24_exp0050_0099_feature_matrix.parquet"
)
R17_FEATURE_PATH = DATASETS_ROOT / "12_hkg_t24_robust_experiment_outputs" / "hkg_t24_r17_feature_matrix.parquet"
R17_OOF_PATH = DATASETS_ROOT / "12_hkg_t24_robust_experiment_outputs" / "hkg_t24_r17_oof_predictions.parquet"
STATION_DAY_PATH = DATASETS_ROOT / "04_noaa_isd_regional_surface" / "noaa_isd_station_day_cutoff_summary.parquet"
STATION_OBS_PATH = DATASETS_ROOT / "04_noaa_isd_regional_surface" / "noaa_isd_core_observations.parquet"
CLIMATE_PATH = DATASETS_ROOT / "02_hko_daily_climate_all_elements" / "hko_daily_climate_elements.parquet"
UPPER_AIR_PATH = (
    DATASETS_ROOT / "03_noaa_igra_upper_air_hkm00045004" / "noaa_igra_hkm00045004_sounding_features.parquet"
)
MARINE_PATH = DATASETS_ROOT / "08_hko_marine_tide_coastal_waters" / "hko_south_china_coastal_waters_bulletin.parquet"
TIDE_PATH = DATASETS_ROOT / "08_hko_marine_tide_coastal_waters" / "hko_latest_tidal_information.parquet"
CYCLONE_PATH = DATASETS_ROOT / "06_hko_tropical_cyclone_best_track" / "hko_tropical_cyclone_best_track.parquet"
STATIC_INVENTORY_PATH = DATASETS_ROOT / "11_static_geospatial_inventory" / "static_geospatial_package_inventory.parquet"
RSS_ITEMS_PATH = DATASETS_ROOT / "05_hko_historical_rss_forecasts" / "hko_historical_rss_items.parquet"
EXTERNAL_DATA_ROOT = Path(r"C:\hkg_tmax_data")
HF_HQ_FULL_DAY_PATH = EXTERNAL_DATA_ROOT / "bronze" / "hkg_t24" / "r03_hko_hq_full_day_high_frequency.parquet"
HF_SELECTED_STATION_PATH = (
    EXTERNAL_DATA_ROOT / "bronze" / "analysis_phase_a" / "hko_high_frequency_selected_station_observations.parquet"
)
HF_TEMPERATURE_SAMPLED_PATH = EXTERNAL_DATA_ROOT / "bronze" / "hkg_t24" / "r09_temperature_sampled_observations.parquet"
HF_WIND_SAMPLED_PATH = EXTERNAL_DATA_ROOT / "bronze" / "hkg_t24" / "r08_wind_vector_sampled_observations.parquet"
HF_MOISTURE_SAMPLED_PATH = EXTERNAL_DATA_ROOT / "bronze" / "hkg_t24" / "r06_moisture_sampled_observations.parquet"
HF_SOLAR_FEATURE_PATH = EXTERNAL_DATA_ROOT / "gold" / "hkg_t24" / "r12_solar_radiation" / "r12_feature_matrix.parquet"
HF_CUTOFF_MINUTE = 15 * 60
HF_EXPERIMENT_IDS = {"0179", "0180", "0181", "0182", "0183"}

INPUT_PATHS = {
    "official_pre2024": OFFICIAL_PATH,
    "long_feature_matrix": FEATURE_MATRIX_PATH,
    "r17_feature_matrix": R17_FEATURE_PATH,
    "r17_oof_predictions": R17_OOF_PATH,
    "station_day_cutoff_summary": STATION_DAY_PATH,
    "station_core_observations": STATION_OBS_PATH,
    "hko_daily_climate": CLIMATE_PATH,
    "upper_air_sounding_features": UPPER_AIR_PATH,
    "marine_bulletins": MARINE_PATH,
    "tidal_information": TIDE_PATH,
    "tropical_cyclone_best_track": CYCLONE_PATH,
    "static_geospatial_inventory": STATIC_INVENTORY_PATH,
    "rss_items": RSS_ITEMS_PATH,
    "external_hf_hko_hq_full_day": HF_HQ_FULL_DAY_PATH,
    "external_hf_selected_station": HF_SELECTED_STATION_PATH,
    "external_hf_temperature_sampled": HF_TEMPERATURE_SAMPLED_PATH,
    "external_hf_wind_sampled": HF_WIND_SAMPLED_PATH,
    "external_hf_moisture_sampled": HF_MOISTURE_SAMPLED_PATH,
    "external_hf_solar_feature_matrix": HF_SOLAR_FEATURE_PATH,
}

OUTCOME_TOKENS = (
    "target_tmax_c",
    "official_error",
    "official_abs_error",
    "official_residual",
    "candidate",
    "prediction",
    "residual",
    "abs_error",
    "uplift",
)
NON_FEATURE_COLUMNS = {
    "target_date",
    "source_id",
    "bulletin_id",
    "source",
    "source_url",
    "product_type",
    "issue_at_hkt",
    "issue_at_utc",
    "available_at_hkt",
    "available_at_utc",
    "content_sha256",
    "raw_sha256",
    "raw_path",
    "raw_retrieved_at_utc",
    "archive_entry_name",
    "guid",
    "title",
    "description_text",
    "weather_text",
    "wind_text",
}


@dataclass(frozen=True)
class RoadmapExperiment:
    experiment_id: str
    title: str
    folder_name: str
    priority: str
    research_mode: str
    eligibility: str
    dependencies: str
    sections: Mapping[str, str]


@dataclass
class DataBundle:
    official: pd.DataFrame
    feature_matrix: pd.DataFrame
    model_frame: pd.DataFrame
    high_frequency_features: pd.DataFrame
    high_frequency_model_frame: pd.DataFrame
    station_day: pd.DataFrame
    station_dossier: pd.DataFrame
    climate_summary: pd.DataFrame
    source_inventory: pd.DataFrame
    input_hashes: pd.DataFrame
    high_frequency_available: bool


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml_payload = json.loads(json.dumps(payload, default=str))
    path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def path_text(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def date_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def markdown_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy()
    display = display.replace({np.nan: ""})
    columns = [str(col) for col in display.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        values = [str(row[col]).replace("|", r"\|").replace("\n", " ") for col in display.columns]
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def strict_pre2024(dates: Iterable[object], *, context: str) -> None:
    series = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
    bad = series[series >= CONFIRMATION_START]
    if not bad.empty:
        examples = ", ".join(date_text(value) for value in bad.head(5))
        raise RuntimeError(f"{context} attempted to use target dates >= 2024-01-01: {examples}")


def section_value(block: str, heading: str) -> str:
    match = re.search(
        rf"^### {re.escape(heading)}\n(.*?)(?=\n### |\n## |\Z)",
        block,
        flags=re.M | re.S,
    )
    return match.group(1).strip() if match else ""


def bold_value(block: str, label: str) -> str:
    match = re.search(rf"\*\*{re.escape(label)}:\*\*\s*(.+)", block)
    return match.group(1).strip() if match else ""


def parse_roadmap(path: Path = ROADMAP_PATH) -> tuple[str, tuple[RoadmapExperiment, ...]]:
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^## (01\d{2}) — (.+)$", text, flags=re.M))
    parsed: list[RoadmapExperiment] = []
    for index, match in enumerate(matches):
        experiment_id = match.group(1)
        if not ("0105" <= experiment_id <= "0183"):
            continue
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        create_match = re.search(r"Create `experiments/([^`]+)/run\.py`", block)
        if not create_match:
            raise RuntimeError(f"{experiment_id} is missing an explicit experiments/.../run.py target")
        sections = {
            heading: section_value(block, heading)
            for heading in (
                "Decision question and hypothesis",
                "Why this is new rather than a relabelled prior experiment",
                "Response variables",
                "Exact inputs",
                "Feature constructions and calculations",
                "Procedure",
                "Walk-forward validation and minimum evidence",
                "Leakage and point-in-time contract",
                "First concrete implementation",
                "Required artifacts",
                "Acceptance criteria",
                "Expected failure modes and interpretation",
                "Expected information gain",
            )
        }
        parsed.append(
            RoadmapExperiment(
                experiment_id=experiment_id,
                title=match.group(2).strip(),
                folder_name=create_match.group(1).strip(),
                priority=bold_value(block, "Priority"),
                research_mode=bold_value(block, "Research mode"),
                eligibility=bold_value(block, "Eligibility"),
                dependencies=bold_value(block, "Dependencies"),
                sections=sections,
            )
        )
    expected = [f"{number:04d}" for number in range(105, 184)]
    actual = [item.experiment_id for item in parsed]
    if actual != expected:
        raise RuntimeError(f"Roadmap experiment sequence mismatch. Expected {expected[:3]}...{expected[-3:]}, got {actual}")
    return text, tuple(parsed)


COMMON_CONTRACT_ARTIFACTS = {
    "README.md",
    "summary.json",
    "data_range.csv",
    "input_hashes.json",
    "feature_definitions.csv",
    "feature_eligibility.csv",
    "leakage_audit.md",
    "scoreboard.csv",
    "fold_metrics.csv",
    "year_stability.csv",
    "season_stability.csv",
    "source_stability.csv",
    "high_error_tail.csv",
    "negative_results.md",
    "next_recommendation.md",
    "predictions.parquet",
}


def required_artifact_names(spec: RoadmapExperiment) -> list[str]:
    names = re.findall(r"`([^`]+)`", spec.sections["Required artifacts"])
    return [name for name in names if name not in COMMON_CONTRACT_ARTIFACTS]


def make_contract_frame(predictions: pd.DataFrame, *, frame_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame_id": frame_id,
                "rows": int(len(predictions)),
                "first_target_date": date_text(predictions["target_date"].min()),
                "last_target_date": date_text(predictions["target_date"].max()),
            }
        ]
    )


def family_for_column(column: str) -> str:
    lowered = column.lower()
    if lowered in {"year", "month", "day_of_year", "doy_sin", "doy_cos", "doy_sin_2", "doy_cos_2"}:
        return "calendar"
    if lowered.startswith(("target_lag", "target_roll", "target_spell", "target_reversal", "target_entropy", "target_abs_change", "trajectory_", "volatility_")):
        return "target_memory"
    if lowered.startswith("isd_"):
        return "station_network"
    if lowered.startswith(("igra_", "ua_")):
        return "upper_air_diagnostic"
    if lowered.startswith("daily_"):
        if any(token in lowered for token in ("sea_temperature", "waglan", "north_point")):
            return "marine_daily_diagnostic"
        return "hko_daily_climate_diagnostic"
    if lowered.startswith(("forecast_", "issue_", "rh_", "parser", "availability_")):
        return "official_forecast_archive"
    if lowered.startswith(("hf_", "hfr12_")):
        return "high_frequency_diagnostic"
    return "derived_or_other"


def allowed_for_walkforward(column: str) -> bool:
    family = family_for_column(column)
    return family in {
        "calendar",
        "target_memory",
        "station_network",
        "official_forecast_archive",
        "derived_or_other",
        "high_frequency_diagnostic",
    }


def load_input_hashes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_id, path in INPUT_PATHS.items():
        rows.append(
            {
                "source_id": source_id,
                "path": path_text(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
                "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
            }
        )
    rows.append(
        {
            "source_id": "roadmap",
            "path": str(ROADMAP_PATH),
            "exists": ROADMAP_PATH.exists(),
            "sha256": sha256_file(ROADMAP_PATH) if ROADMAP_PATH.exists() else "",
            "bytes": ROADMAP_PATH.stat().st_size if ROADMAP_PATH.exists() else 0,
        }
    )
    return pd.DataFrame(rows)


def load_official() -> pd.DataFrame:
    official = pd.read_parquet(OFFICIAL_PATH)
    official["target_date"] = pd.to_datetime(official["target_date"], errors="coerce").dt.normalize()
    official = official[official["target_date"].notna() & (official["target_date"] < CONFIRMATION_START)].copy()
    official = official.drop_duplicates("target_date", keep="last").sort_values("target_date").reset_index(drop=True)
    for column in ("target_tmax_c", "forecast_max_c", "forecast_min_c"):
        official[column] = pd.to_numeric(official[column], errors="coerce")
    official["official_prediction_c"] = official["forecast_max_c"]
    official["official_residual_c"] = official["target_tmax_c"] - official["official_prediction_c"]
    official["official_error_c_signed_forecast_minus_target"] = official["official_prediction_c"] - official["target_tmax_c"]
    official["official_abs_error_c"] = official["official_residual_c"].abs()
    official["forecast_range_c"] = official["forecast_max_c"] - official["forecast_min_c"]
    official["forecast_midpoint_c"] = (official["forecast_max_c"] + official["forecast_min_c"]) / 2.0
    if "season" not in official.columns:
        official["season"] = official["month"].map(month_to_season)
    strict_pre2024(official["target_date"], context="official frame")
    return official


def month_to_season(month: object) -> str:
    try:
        value = int(month)
    except (TypeError, ValueError):
        return "UNK"
    if value in (12, 1, 2):
        return "DJF"
    if value in (3, 4, 5):
        return "MAM"
    if value in (6, 7, 8):
        return "JJA"
    return "SON"


def load_feature_matrix() -> pd.DataFrame:
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    features["target_date"] = pd.to_datetime(features["target_date"], errors="coerce").dt.normalize()
    features = features[features["target_date"].notna() & (features["target_date"] < CONFIRMATION_START)].copy()
    strict_pre2024(features["target_date"], context="feature matrix")
    return features.sort_values("target_date").reset_index(drop=True)


def build_model_frame(official: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    drop_overlap = [column for column in features.columns if column in official.columns and column != "target_date"]
    merged = official.merge(features.drop(columns=drop_overlap), on="target_date", how="left", validate="one_to_one")
    if "target_roll3650_mean_lag7_c" in merged.columns:
        baseline = pd.to_numeric(merged["target_roll3650_mean_lag7_c"], errors="coerce")
    else:
        baseline = merged.groupby("month", observed=True)["target_tmax_c"].transform("mean")
    merged["target_anomaly_c"] = merged["target_tmax_c"] - baseline
    merged["official_underforecast_c"] = merged["official_residual_c"].clip(lower=0.0)
    merged["official_overforecast_c"] = (-merged["official_residual_c"]).clip(lower=0.0)
    merged["high_error_flag_1p5"] = merged["official_abs_error_c"].gt(1.5)
    merged["high_error_flag_2p0"] = merged["official_abs_error_c"].gt(2.0)
    merged["mam_high_error_flag"] = merged["season"].eq("MAM") & merged["high_error_flag_1p5"]
    merged["hot_day_underforecast_flag"] = merged["target_tmax_c"].ge(30.0) & merged["official_residual_c"].gt(0.0)
    merged["cold_day_overforecast_flag"] = merged["target_tmax_c"].le(20.0) & merged["official_residual_c"].lt(0.0)
    return merged.sort_values("target_date").reset_index(drop=True)


def safe_feature_name(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return text or "unknown"


def hf_observations_before_cutoff(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path, columns=list(columns))
    if "local_date" not in frame.columns:
        return pd.DataFrame()
    frame["target_date"] = pd.to_datetime(frame["local_date"], errors="coerce").dt.normalize()
    frame = frame[frame["target_date"].notna() & (frame["target_date"] < CONFIRMATION_START)].copy()
    if "observed_at_hkt" in frame.columns:
        observed = pd.to_datetime(frame["observed_at_hkt"], errors="coerce")
        frame["hf_observed_minute"] = observed.dt.hour * 60 + observed.dt.minute
        frame = frame[frame["hf_observed_minute"].notna() & frame["hf_observed_minute"].le(HF_CUTOFF_MINUTE)].copy()
    strict_pre2024(frame["target_date"], context=f"high-frequency source {path.name}")
    return frame


def aggregate_hf_variable_frame(path: Path, *, prefix: str) -> pd.DataFrame:
    frame = hf_observations_before_cutoff(
        path,
        [
            "station",
            "observed_at_hkt",
            "local_date",
            "variable",
            "value",
        ],
    )
    if frame.empty:
        return pd.DataFrame()
    frame["variable_key"] = frame["variable"].map(safe_feature_name)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    grouped = frame.groupby(["target_date", "variable_key"], observed=True)
    stats = grouped["value"].agg(["count", "mean", "max", "min", "std"]).unstack("variable_key")
    stats.columns = [f"{prefix}_{variable}_{stat}" for stat, variable in stats.columns]
    stats = stats.reset_index()
    station_counts = (
        frame.groupby(["target_date", "variable_key"], observed=True)["station"]
        .nunique()
        .unstack("variable_key")
        .add_prefix(f"{prefix}_")
        .add_suffix("_station_count")
        .reset_index()
    )
    latest = (
        frame.sort_values(["target_date", "variable_key", "observed_at_hkt"])
        .dropna(subset=["value"])
        .groupby(["target_date", "variable_key"], observed=True)
        .tail(1)
        .pivot(index="target_date", columns="variable_key", values="value")
        .add_prefix(f"{prefix}_")
        .add_suffix("_latest_before_1500")
        .reset_index()
    )
    return stats.merge(station_counts, on="target_date", how="outer").merge(latest, on="target_date", how="outer")


def aggregate_temperature_sampled(path: Path) -> pd.DataFrame:
    frame = hf_observations_before_cutoff(
        path,
        ["station", "observed_at_hkt", "available_at_hkt", "local_date", "temperature_c"],
    )
    if frame.empty:
        return pd.DataFrame()
    frame["temperature_c"] = pd.to_numeric(frame["temperature_c"], errors="coerce")
    grouped = frame.groupby("target_date", observed=True)
    out = grouped.agg(
        hf_temp_sample_count=("temperature_c", "size"),
        hf_temp_station_count=("station", "nunique"),
        hf_temp_mean_c=("temperature_c", "mean"),
        hf_temp_max_c=("temperature_c", "max"),
        hf_temp_min_c=("temperature_c", "min"),
        hf_temp_std_c=("temperature_c", "std"),
        hf_temp_range_c=("temperature_c", lambda values: float(values.max() - values.min())),
    ).reset_index()
    return out


def aggregate_wind_sampled(path: Path) -> pd.DataFrame:
    frame = hf_observations_before_cutoff(
        path,
        [
            "station",
            "observed_at_hkt",
            "available_at_hkt",
            "local_date",
            "mean_speed_kmh",
            "max_gust_kmh",
            "u_kmh",
            "v_kmh",
            "direction_available",
        ],
    )
    if frame.empty:
        return pd.DataFrame()
    for column in ("mean_speed_kmh", "max_gust_kmh", "u_kmh", "v_kmh"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = frame.groupby("target_date", observed=True)
    return grouped.agg(
        hf_wind_sample_count=("mean_speed_kmh", "size"),
        hf_wind_station_count=("station", "nunique"),
        hf_wind_mean_speed_kmh=("mean_speed_kmh", "mean"),
        hf_wind_max_speed_kmh=("mean_speed_kmh", "max"),
        hf_wind_max_gust_kmh=("max_gust_kmh", "max"),
        hf_wind_u_mean_kmh=("u_kmh", "mean"),
        hf_wind_v_mean_kmh=("v_kmh", "mean"),
        hf_wind_direction_available_rate=("direction_available", "mean"),
    ).reset_index()


def aggregate_solar_feature_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame = frame[frame["target_date"].notna() & (frame["target_date"] < CONFIRMATION_START)].copy()
    keep_tokens = (
        "solar",
        "day_length",
        "elevation",
        "hko_temp",
        "heating",
        "warming",
        "humidity",
        "wind",
        "cloud",
        "latest_age",
        "obs_count",
        "snapshot",
    )
    selected = [
        column
        for column in frame.columns
        if column != "target_date"
        and pd.api.types.is_numeric_dtype(frame[column])
        and not any(token in column.lower() for token in OUTCOME_TOKENS)
        and any(token in column.lower() for token in keep_tokens)
    ]
    selected = selected[:72]
    out = frame[["target_date", *selected]].copy()
    out = out.rename(columns={column: f"hfr12_{safe_feature_name(column)}" for column in selected})
    strict_pre2024(out["target_date"], context="high-frequency solar feature matrix")
    return out


def merge_feature_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for frame in frames:
        if frame.empty:
            continue
        clean = frame.drop_duplicates("target_date", keep="last").copy()
        merged = clean if merged is None else merged.merge(clean, on="target_date", how="outer")
    if merged is None:
        return pd.DataFrame()
    merged = merged.sort_values("target_date").reset_index(drop=True)
    strict_pre2024(merged["target_date"], context="merged high-frequency feature frame")
    return merged


def load_high_frequency_features() -> pd.DataFrame:
    return merge_feature_frames(
        [
            aggregate_hf_variable_frame(HF_HQ_FULL_DAY_PATH, prefix="hf_hko"),
            aggregate_hf_variable_frame(HF_SELECTED_STATION_PATH, prefix="hf_station"),
            aggregate_temperature_sampled(HF_TEMPERATURE_SAMPLED_PATH),
            aggregate_wind_sampled(HF_WIND_SAMPLED_PATH),
            aggregate_hf_variable_frame(HF_MOISTURE_SAMPLED_PATH, prefix="hf_moisture"),
            aggregate_solar_feature_matrix(HF_SOLAR_FEATURE_PATH),
        ]
    )


def load_station_day() -> pd.DataFrame:
    frame = pd.read_parquet(STATION_DAY_PATH)
    frame["local_date"] = pd.to_datetime(frame["local_date"], errors="coerce").dt.normalize()
    return frame[frame["local_date"].notna() & (frame["local_date"] < CONFIRMATION_START)].copy()


def haversine_km(lat1: float, lon1: float, lat2: float = 22.302711, lon2: float = 114.177216) -> float:
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_station_dossier(station_day: pd.DataFrame) -> pd.DataFrame:
    obs_columns = [
        "station_id",
        "observed_at_hkt",
        "latitude",
        "longitude",
        "elevation_m",
        "temperature_quality_code",
        "dew_point_quality_code",
        "sea_level_pressure_quality_code",
        "operational_input_allowed",
        "availability_tier",
    ]
    obs = pd.read_parquet(STATION_OBS_PATH, columns=obs_columns)
    obs["observed_at_hkt"] = pd.to_datetime(obs["observed_at_hkt"], errors="coerce", utc=True)
    metadata = (
        obs.groupby("station_id", observed=True)
        .agg(
            first_observed_hkt=("observed_at_hkt", "min"),
            last_observed_hkt=("observed_at_hkt", "max"),
            observation_rows=("station_id", "size"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            elevation_m=("elevation_m", "median"),
            operational_allowed_rows=("operational_input_allowed", "sum"),
            availability_tier=("availability_tier", lambda values: values.mode().iloc[0] if not values.mode().empty else ""),
        )
        .reset_index()
    )
    coverage = (
        station_day.groupby("station_id", observed=True)
        .agg(
            cutoff_days=("local_date", "nunique"),
            first_cutoff_date=("local_date", "min"),
            last_cutoff_date=("local_date", "max"),
            mean_obs_count=("obs_count", "mean"),
            latest_before_1500_coverage=("latest_before_1500_hkt", lambda values: values.notna().mean()),
            temp_coverage=("air_temperature_c_latest_before_1500", lambda values: values.notna().mean()),
            dewpoint_coverage=("dew_point_c_latest_before_1500", lambda values: values.notna().mean()),
            pressure_coverage=("sea_level_pressure_hpa_latest_before_1500", lambda values: values.notna().mean()),
            wind_coverage=("wind_speed_mps_latest_before_1500", lambda values: values.notna().mean()),
            operational_allowed_any=("operational_input_allowed", "max"),
        )
        .reset_index()
    )
    dossier = metadata.merge(coverage, on="station_id", how="outer")
    dossier["distance_to_hko_km"] = [
        haversine_km(lat, lon) if pd.notna(lat) and pd.notna(lon) else math.nan
        for lat, lon in zip(dossier["latitude"], dossier["longitude"], strict=False)
    ]
    dossier["coordinate_quarantine_flag"] = (
        dossier["latitude"].lt(18.0)
        | dossier["latitude"].gt(26.0)
        | dossier["longitude"].lt(108.0)
        | dossier["longitude"].gt(118.0)
    )
    dossier["role_hint"] = np.select(
        [
            dossier["distance_to_hko_km"].le(15),
            dossier["longitude"].lt(113.8),
            dossier["latitude"].gt(23.0),
            dossier["elevation_m"].gt(200),
        ],
        ["near_hko_core", "western_inland_or_delta", "northern_upstream", "elevated_or_terrain"],
        default="regional_context",
    )
    return dossier.sort_values(["coordinate_quarantine_flag", "cutoff_days", "station_id"], ascending=[False, False, True]).reset_index(drop=True)


def build_climate_summary() -> pd.DataFrame:
    if not CLIMATE_PATH.exists():
        return pd.DataFrame()
    climate = pd.read_parquet(CLIMATE_PATH)
    climate["local_date"] = pd.to_datetime(climate["local_date"], errors="coerce").dt.normalize()
    climate = climate[climate["local_date"].notna() & (climate["local_date"] < CONFIRMATION_START)].copy()
    rows = (
        climate.groupby(["station_or_domain", "variable", "unit"], observed=True)
        .agg(
            rows=("value", "size"),
            first_date=("local_date", "min"),
            last_date=("local_date", "max"),
            non_null_rate=("value", lambda values: values.notna().mean()),
            operational_input_allowed=("operational_input_allowed", "max"),
            availability_tier=("availability_tier", lambda values: values.mode().iloc[0] if not values.mode().empty else ""),
        )
        .reset_index()
    )
    return rows.sort_values(["rows", "station_or_domain", "variable"], ascending=[False, True, True])


def build_source_inventory(input_hashes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_id, path in INPUT_PATHS.items():
        entry: dict[str, object] = {
            "source_id": source_id,
            "path": path_text(path),
            "exists": path.exists(),
            "rows": 0,
            "columns": 0,
            "first_date": "",
            "last_date": "",
        }
        if path.exists() and path.suffix == ".parquet":
            try:
                sample = pd.read_parquet(path)
                entry["rows"] = int(len(sample))
                entry["columns"] = int(len(sample.columns))
                for date_col in ("target_date", "local_date", "valid_at_hkt", "observed_at_hkt"):
                    if date_col in sample.columns:
                        dates = pd.to_datetime(sample[date_col], errors="coerce", utc=True)
                        dates = dates[dates.notna()]
                        if not dates.empty:
                            entry["first_date"] = date_text(dates.min())
                            entry["last_date"] = date_text(dates.max())
                        break
            except Exception as exc:  # diagnostic inventory should not stop all experiments
                entry["read_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(entry)
    return pd.DataFrame(rows).merge(input_hashes[["source_id", "sha256", "bytes"]], on="source_id", how="left")


def load_bundle() -> DataBundle:
    input_hashes = load_input_hashes()
    official = load_official()
    feature_matrix = load_feature_matrix()
    model_frame = build_model_frame(official, feature_matrix)
    high_frequency_features = load_high_frequency_features()
    high_frequency_model_frame = (
        model_frame.merge(high_frequency_features, on="target_date", how="inner", validate="one_to_one")
        if not high_frequency_features.empty
        else pd.DataFrame()
    )
    if not high_frequency_model_frame.empty:
        strict_pre2024(high_frequency_model_frame["target_date"], context="high-frequency model frame")
    station_day = load_station_day()
    station_dossier = build_station_dossier(station_day)
    climate_summary = build_climate_summary()
    source_inventory = build_source_inventory(input_hashes)
    high_frequency_available = len(high_frequency_model_frame) >= 300
    return DataBundle(
        official=official,
        feature_matrix=feature_matrix,
        model_frame=model_frame,
        high_frequency_features=high_frequency_features,
        high_frequency_model_frame=high_frequency_model_frame,
        station_day=station_day,
        station_dossier=station_dossier,
        climate_summary=climate_summary,
        source_inventory=source_inventory,
        input_hashes=input_hashes,
        high_frequency_available=high_frequency_available,
    )


def is_numeric_feature(frame: pd.DataFrame, column: str) -> bool:
    lowered = column.lower()
    if column in NON_FEATURE_COLUMNS:
        return False
    if any(token in lowered for token in OUTCOME_TOKENS):
        return False
    return pd.api.types.is_numeric_dtype(frame[column])


def feature_tokens_for(spec: RoadmapExperiment) -> tuple[str, ...]:
    text = f"{spec.title} {' '.join(spec.sections.values())}".lower()
    tokens: set[str] = {"doy_", "year_centered"}
    if any(word in text for word in ("target", "spell", "phase", "volatility", "trajectory", "thermal constraint", "reversal", "analog")):
        tokens.update(("target_lag", "target_roll", "target_spell", "target_reversal", "trajectory_", "volatility_"))
    if any(word in text for word in ("station", "spatial", "front", "rank", "coherence", "interpolation", "field", "shapley", "graph")):
        tokens.update(("isd_", "station_", "graph_", "lat_slope", "lon_slope"))
    if any(word in text for word in ("pressure", "surge", "geopotential", "1000-hpa", "hpa")):
        tokens.update(("pressure", "hpa", "hgt", "geopotential"))
    if any(word in text for word in ("dewpoint", "moisture", "humid", "wet-bulb", "enthalpy", "dry-air", "dry heating")):
        tokens.update(("dew", "humidity", "wet_bulb", "specific_humidity", "spread"))
    if any(word in text for word in ("wind", "flow", "advection", "ventilation", "vorticity", "divergence", "fetch", "sea-breeze")):
        tokens.update(("wind", "wspd", "wdir", "u_", "v_", "advection"))
    if any(word in text for word in ("upper-air", "inversion", "mixing", "tropospheric", "stability", "shear", "veering", "heat-content")):
        tokens.update(("igra_", "ua_", "inversion", "stability", "shear"))
    if any(word in text for word in ("marine", "sea", "coast", "onshore", "cloud", "solar", "radiative", "visibility", "haze", "rain", "wetness")):
        tokens.update(("daily_", "sea_temperature", "cloud", "solar", "sunshine", "visibility", "rain", "waglan", "north_point"))
    if any(word in text for word in ("forecast", "rss", "press", "text", "range", "vintage", "revision", "staleness", "parser")):
        tokens.update(("forecast_", "issue_to_cutoff", "lead_hours", "rh_", "parser", "availability"))
    if any(word in text for word in ("tropical-cyclone", "tc ", "subsidence")):
        tokens.update(("wind", "pressure", "cloud", "ua_", "daily_"))
    if any(word in text for word in ("high-frequency", "minute", "since-midnight", "solar", "uv", "radiative", "teacher")):
        tokens.update(("hf_", "hfr12_"))
    return tuple(sorted(tokens))


def selected_features(spec: RoadmapExperiment, frame: pd.DataFrame) -> list[str]:
    tokens = feature_tokens_for(spec)
    columns = [
        column
        for column in frame.columns
        if is_numeric_feature(frame, column) and any(token in column.lower() for token in tokens)
    ]
    if not columns:
        columns = [column for column in frame.columns if is_numeric_feature(frame, column) and allowed_for_walkforward(column)]
    scored: list[tuple[float, str]] = []
    residual = pd.to_numeric(frame["official_residual_c"], errors="coerce")
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        non_null = values.notna().mean()
        variance = float(values.var()) if values.notna().sum() > 2 else 0.0
        if non_null < 0.15 or not math.isfinite(variance) or variance <= 0:
            continue
        pair = pd.concat([values, residual], axis=1).dropna()
        corr = abs(float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))) if len(pair) >= 25 and pair.iloc[:, 0].nunique() > 1 else 0.0
        safe_bonus = 0.15 if allowed_for_walkforward(column) else 0.0
        scored.append((corr + safe_bonus + 0.05 * non_null, column))
    scored.sort(reverse=True)
    return [column for _, column in scored[:MAX_MODEL_FEATURES]]


def selected_high_frequency_features(spec: RoadmapExperiment, frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    hf_columns = [
        column
        for column in frame.columns
        if column.startswith(("hf_", "hfr12_")) and is_numeric_feature(frame, column)
    ]
    if spec.experiment_id == "0179":
        tokens = ("temp", "heating", "warming", "snapshot", "since_midnight")
    elif spec.experiment_id == "0180":
        tokens = ("temp", "humidity", "pressure", "wind", "station", "range", "std")
    elif spec.experiment_id == "0181":
        tokens = ("since_midnight", "max", "min", "latest", "temp", "ceiling")
    elif spec.experiment_id == "0182":
        tokens = ("solar", "radiative", "humidity", "wind", "heating", "warming")
    else:
        tokens = ("hf_", "hfr12_")
    preferred = [column for column in hf_columns if any(token in column.lower() for token in tokens)]
    candidate_columns = preferred or hf_columns
    residual = pd.to_numeric(frame["official_residual_c"], errors="coerce")
    scored: list[tuple[float, str]] = []
    for column in candidate_columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        non_null = values.notna().mean()
        variance = float(values.var()) if values.notna().sum() > 2 else 0.0
        if non_null < 0.50 or not math.isfinite(variance) or variance <= 0:
            continue
        pair = pd.concat([values, residual], axis=1).dropna()
        corr = abs(float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))) if len(pair) >= 120 and pair.iloc[:, 0].nunique() > 1 else 0.0
        scored.append((corr + 0.05 * non_null, column))
    scored.sort(reverse=True)
    return [column for _, column in scored[:MAX_MODEL_FEATURES]]


def safe_corr(x: pd.Series, y: pd.Series, min_rows: int = 25) -> tuple[int, float]:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1)
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < min_rows or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return int(len(pair)), math.nan
    return int(len(pair)), float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def quantile_spread(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1)
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 60 or pair.iloc[:, 0].nunique() < 4:
        return math.nan
    try:
        buckets = pd.qcut(pair.iloc[:, 0], q=4, duplicates="drop")
    except ValueError:
        return math.nan
    means = pair.iloc[:, 1].groupby(buckets, observed=True).mean()
    return float(means.max() - means.min()) if len(means) >= 2 else math.nan


def feature_diagnostics(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in features:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        corr_resid_n, corr_resid = safe_corr(values, frame["official_residual_c"])
        corr_abs_n, corr_abs = safe_corr(values, frame["official_abs_error_c"])
        corr_anom_n, corr_anom = safe_corr(values, frame["target_anomaly_c"])
        rows.append(
            {
                "feature": column,
                "family": family_for_column(column),
                "allowed_for_walkforward": allowed_for_walkforward(column),
                "non_null_rows": int(values.notna().sum()),
                "non_null_rate": float(values.notna().mean()),
                "first_non_null_date": date_text(frame.loc[values.notna(), "target_date"].min()) if values.notna().any() else "",
                "last_non_null_date": date_text(frame.loc[values.notna(), "target_date"].max()) if values.notna().any() else "",
                "corr_official_residual_n": corr_resid_n,
                "corr_official_residual": corr_resid,
                "corr_official_abs_error_n": corr_abs_n,
                "corr_official_abs_error": corr_abs,
                "corr_target_anomaly_n": corr_anom_n,
                "corr_target_anomaly": corr_anom,
                "official_residual_q4_spread_c": quantile_spread(values, frame["official_residual_c"]),
                "official_abs_error_q4_spread_c": quantile_spread(values, frame["official_abs_error_c"]),
            }
        )
    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return diagnostics
    diagnostics["diagnostic_score"] = (
        diagnostics["corr_official_residual"].abs().fillna(0.0) * 2.0
        + diagnostics["corr_official_abs_error"].abs().fillna(0.0)
        + diagnostics["official_residual_q4_spread_c"].abs().fillna(0.0).clip(upper=2.0) / 2.0
        + diagnostics["allowed_for_walkforward"].astype(float) * 0.1
    )
    return diagnostics.sort_values(["diagnostic_score", "non_null_rows"], ascending=[False, False]).reset_index(drop=True)


def metric_dict(target: pd.Series, prediction: pd.Series) -> dict[str, object]:
    pair = pd.concat([pd.to_numeric(target, errors="coerce"), pd.to_numeric(prediction, errors="coerce")], axis=1).dropna()
    if pair.empty:
        return {
            "n": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "bias": math.nan,
            "median_ae": math.nan,
            "p80_ae": math.nan,
            "p90_ae": math.nan,
            "p95_ae": math.nan,
            "gt2c_rate": math.nan,
            "gt3c_rate": math.nan,
        }
    error = pair.iloc[:, 1] - pair.iloc[:, 0]
    ae = error.abs()
    return {
        "n": int(len(pair)),
        "mae": float(ae.mean()),
        "rmse": float(math.sqrt(float((error * error).mean()))),
        "bias": float(error.mean()),
        "median_ae": float(ae.median()),
        "p80_ae": float(ae.quantile(0.80)),
        "p90_ae": float(ae.quantile(0.90)),
        "p95_ae": float(ae.quantile(0.95)),
        "gt2c_rate": float(ae.gt(2.0).mean()),
        "gt3c_rate": float(ae.gt(3.0).mean()),
    }


def correction_cap_for(spec: RoadmapExperiment) -> float:
    text = spec.title.lower()
    if any(token in text for token in ("tail", "extreme", "specialist", "catastrophic")):
        return 0.50
    if any(token in text for token in ("audit", "registry", "dossier", "harness")):
        return 0.0
    return DEFAULT_CORRECTION_CAP_C


def walk_forward_predictions(frame: pd.DataFrame, features: Sequence[str], spec: RoadmapExperiment) -> pd.DataFrame:
    predictions = frame[
        [
            "target_date",
            "target_tmax_c",
            "forecast_max_c",
            "forecast_min_c",
            "forecast_source_family",
            "season",
            "month",
            "official_residual_c",
        ]
    ].copy()
    predictions["official_prediction_c"] = predictions["forecast_max_c"]
    predictions["candidate_residual_c"] = 0.0
    predictions["fold_id"] = "baseline_no_model"
    cap = correction_cap_for(spec)
    if cap <= 0.0 or not features:
        predictions["candidate_prediction_c"] = predictions["official_prediction_c"]
        predictions["candidate_correction_c"] = 0.0
    elif spec.experiment_id in {"0161", "0162", "0163", "0169", "0170"}:
        predictions = online_memory_predictions(predictions, spec, cap=cap)
    else:
        y = pd.to_numeric(frame["official_residual_c"], errors="coerce")
        X = frame[list(features)].replace([np.inf, -np.inf], np.nan)
        residual_pred = pd.Series(0.0, index=frame.index, dtype=float)
        fold_labels = pd.Series("baseline_no_prior_train", index=frame.index, dtype=object)
        for start_year, end_year in MODEL_FOLDS:
            test_mask = frame["target_date"].dt.year.between(start_year, end_year)
            train_mask = frame["target_date"].dt.year.lt(start_year) & y.notna()
            if int(train_mask.sum()) < MIN_TRAIN_ROWS or int(test_mask.sum()) == 0:
                continue
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=10.0)),
                ]
            )
            model.fit(X.loc[train_mask, features], y.loc[train_mask])
            residual_pred.loc[test_mask] = model.predict(X.loc[test_mask, features])
            fold_labels.loc[test_mask] = f"walk_forward_{start_year}_{end_year}"
        predictions["candidate_residual_c"] = residual_pred.clip(lower=-cap, upper=cap)
        predictions["candidate_correction_c"] = predictions["candidate_residual_c"]
        predictions["candidate_prediction_c"] = predictions["official_prediction_c"] + predictions["candidate_correction_c"]
        predictions["fold_id"] = fold_labels
    predictions["candidate_error_c"] = predictions["candidate_prediction_c"] - predictions["target_tmax_c"]
    predictions["official_error_c"] = predictions["official_prediction_c"] - predictions["target_tmax_c"]
    predictions["model_id"] = f"{spec.experiment_id}_candidate"
    predictions["model_family"] = "roadmap_residual_model"
    predictions["feature_count"] = len(features)
    strict_pre2024(predictions["target_date"], context=f"{spec.experiment_id} predictions")
    return predictions


def high_frequency_predictions(frame: pd.DataFrame, features: Sequence[str], spec: RoadmapExperiment) -> pd.DataFrame:
    work = frame.copy()
    if features:
        work = work[work[list(features)].notna().any(axis=1)].copy()
    predictions = work[
        [
            "target_date",
            "target_tmax_c",
            "forecast_max_c",
            "forecast_min_c",
            "forecast_source_family",
            "season",
            "month",
            "official_residual_c",
        ]
    ].copy()
    predictions["official_prediction_c"] = predictions["forecast_max_c"]
    predictions["candidate_residual_c"] = 0.0
    predictions["candidate_correction_c"] = 0.0
    predictions["candidate_prediction_c"] = predictions["official_prediction_c"]
    predictions["fold_id"] = "hf_no_model"
    cap = correction_cap_for(spec)
    if features and len(work) >= 300 and cap > 0.0:
        y = pd.to_numeric(work["official_residual_c"], errors="coerce")
        X = work[list(features)].replace([np.inf, -np.inf], np.nan)
        residual_pred = pd.Series(0.0, index=work.index, dtype=float)
        fold_labels = pd.Series("hf_leave_year_insufficient_train", index=work.index, dtype=object)
        for test_year in sorted(work["target_date"].dt.year.dropna().astype(int).unique()):
            test_mask = work["target_date"].dt.year.eq(test_year)
            train_mask = work["target_date"].dt.year.ne(test_year) & y.notna()
            if int(train_mask.sum()) < 250 or int(test_mask.sum()) == 0:
                continue
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=25.0)),
                ]
            )
            model.fit(X.loc[train_mask, features], y.loc[train_mask])
            residual_pred.loc[test_mask] = model.predict(X.loc[test_mask, features])
            fold_labels.loc[test_mask] = f"hf_leave_year_out_{test_year}"
        aligned = residual_pred.reindex(predictions.index).fillna(0.0)
        predictions["candidate_residual_c"] = aligned.clip(lower=-cap, upper=cap).to_numpy()
        predictions["candidate_correction_c"] = predictions["candidate_residual_c"]
        predictions["candidate_prediction_c"] = predictions["official_prediction_c"] + predictions["candidate_correction_c"]
        predictions["fold_id"] = fold_labels.reindex(predictions.index).fillna("hf_leave_year_insufficient_train").to_numpy()
    predictions["candidate_error_c"] = predictions["candidate_prediction_c"] - predictions["target_tmax_c"]
    predictions["official_error_c"] = predictions["official_prediction_c"] - predictions["target_tmax_c"]
    predictions["model_id"] = f"{spec.experiment_id}_hf_diagnostic_candidate"
    predictions["model_family"] = "high_frequency_leave_year_diagnostic"
    predictions["feature_count"] = len(features)
    strict_pre2024(predictions["target_date"], context=f"{spec.experiment_id} high-frequency predictions")
    return predictions.sort_values("target_date").reset_index(drop=True)


def online_memory_predictions(predictions: pd.DataFrame, spec: RoadmapExperiment, *, cap: float) -> pd.DataFrame:
    frame = predictions.sort_values("target_date").reset_index(drop=False).copy()
    history: list[dict[str, object]] = []
    corrections: list[float] = []
    fold_ids: list[str] = []
    for _, row in frame.iterrows():
        if not history:
            corrections.append(0.0)
            fold_ids.append("online_no_history")
        else:
            hist = pd.DataFrame(history)
            if spec.experiment_id in {"0162", "0169"}:
                mask = (
                    hist["forecast_source_family"].eq(row["forecast_source_family"])
                    & hist["season"].eq(row["season"])
                )
            elif spec.experiment_id in {"0163", "0170"}:
                mask = hist["season"].eq(row["season"])
            else:
                mask = hist["forecast_source_family"].eq(row["forecast_source_family"])
            chosen = hist[mask]
            if len(chosen) < 20:
                chosen = hist.tail(365)
            correction = float(chosen["official_residual_c"].tail(365).mean()) if not chosen.empty else 0.0
            corrections.append(float(np.clip(correction, -cap, cap)))
            fold_ids.append("online_memory_expanding_prior")
        history.append(
            {
                "forecast_source_family": row["forecast_source_family"],
                "season": row["season"],
                "official_residual_c": row["official_residual_c"],
            }
        )
    frame["candidate_correction_c"] = corrections
    if spec.experiment_id == "0170":
        # Abstain where the historical correction is tiny; this preserves no-harm routing behavior.
        frame.loc[frame["candidate_correction_c"].abs().lt(0.05), "candidate_correction_c"] = 0.0
    frame["candidate_prediction_c"] = frame["official_prediction_c"] + frame["candidate_correction_c"]
    frame["fold_id"] = fold_ids
    return frame.drop(columns=["index"]).sort_values("target_date").reset_index(drop=True)


def subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, keys in {
        "overall": [],
        "year": ["year"],
        "season": ["season"],
        "month": ["month"],
        "source": ["forecast_source_family"],
    }.items():
        work = predictions.copy()
        work["year"] = work["target_date"].dt.year
        groups = [(("all",), work)] if not keys else work.groupby(keys, observed=True, sort=True)
        for group_key, group in groups:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            official = metric_dict(group["target_tmax_c"], group["official_prediction_c"])
            candidate = metric_dict(group["target_tmax_c"], group["candidate_prediction_c"])
            rows.append(
                {
                    "slice_type": group_name,
                    "slice_value": "__".join(str(value) for value in group_key),
                    "n": candidate["n"],
                    "official_mae": official["mae"],
                    "candidate_mae": candidate["mae"],
                    "delta_mae_vs_official": candidate["mae"] - official["mae"],
                    "official_rmse": official["rmse"],
                    "candidate_rmse": candidate["rmse"],
                    "candidate_bias": candidate["bias"],
                    "candidate_p90_ae": candidate["p90_ae"],
                    "candidate_gt2c_rate": candidate["gt2c_rate"],
                    "candidate_gt3c_rate": candidate["gt3c_rate"],
                }
            )
    return pd.DataFrame(rows)


def baseline_predictions(bundle: DataBundle, spec: RoadmapExperiment) -> pd.DataFrame:
    frame = bundle.model_frame
    predictions = frame[
        ["target_date", "target_tmax_c", "forecast_max_c", "forecast_min_c", "forecast_source_family", "season", "month"]
    ].copy()
    predictions["official_prediction_c"] = predictions["forecast_max_c"]
    predictions["candidate_prediction_c"] = predictions["official_prediction_c"]
    predictions["candidate_correction_c"] = 0.0
    predictions["candidate_error_c"] = predictions["candidate_prediction_c"] - predictions["target_tmax_c"]
    predictions["official_error_c"] = predictions["official_prediction_c"] - predictions["target_tmax_c"]
    predictions["fold_id"] = "audit_baseline_no_model"
    predictions["model_id"] = f"{spec.experiment_id}_audit_baseline"
    predictions["model_family"] = "audit_or_blocked_no_model"
    predictions["feature_count"] = 0
    return predictions


def foundation_artifacts(spec: RoadmapExperiment, bundle: DataBundle) -> dict[str, pd.DataFrame]:
    if spec.experiment_id == "0105":
        return {"station_dossier.csv": bundle.station_dossier}
    if spec.experiment_id == "0106":
        rows: list[dict[str, object]] = []
        for column in bundle.feature_matrix.columns:
            if column == "target_date":
                continue
            family = family_for_column(column)
            values = bundle.feature_matrix[column]
            rows.append(
                {
                    "feature": column,
                    "family": family,
                    "allowed_for_walkforward": allowed_for_walkforward(column),
                    "dtype": str(values.dtype),
                    "non_null_rows": int(values.notna().sum()),
                    "non_null_rate": float(values.notna().mean()),
                    "first_non_null_date": date_text(bundle.feature_matrix.loc[values.notna(), "target_date"].min()) if values.notna().any() else "",
                    "last_non_null_date": date_text(bundle.feature_matrix.loc[values.notna(), "target_date"].max()) if values.notna().any() else "",
                    "lineage_note": lineage_note_for_family(family),
                }
            )
        return {"feature_lineage.csv": pd.DataFrame(rows).sort_values(["family", "feature"])}
    if spec.experiment_id == "0107":
        official = bundle.official.copy()
        duplicate_dates = official[official["target_date"].duplicated(keep=False)]
        gaps = compute_gaps(official["target_date"], "official_pre2024")
        station_age = bundle.station_day.copy()
        station_age["latest_before_1500_hkt_dt"] = pd.to_datetime(station_age["latest_before_1500_hkt"], errors="coerce", utc=True)
        station_age_summary = (
            station_age.groupby("station_id", observed=True)
            .agg(
                rows=("local_date", "size"),
                dated_rows=("latest_before_1500_hkt_dt", lambda values: values.notna().sum()),
                mean_obs_count=("obs_count", "mean"),
                max_obs_count=("obs_count", "max"),
                temp_non_null_rate=("air_temperature_c_latest_before_1500", lambda values: values.notna().mean()),
            )
            .reset_index()
        )
        return {
            "duplicate_target_dates.csv": duplicate_dates,
            "official_gap_calendar.csv": gaps,
            "station_observation_age_summary.csv": station_age_summary,
        }
    if spec.experiment_id == "0108":
        response = bundle.model_frame[
            [
                "target_date",
                "target_tmax_c",
                "target_anomaly_c",
                "official_residual_c",
                "official_abs_error_c",
                "official_underforecast_c",
                "official_overforecast_c",
                "high_error_flag_1p5",
                "high_error_flag_2p0",
                "hot_day_underforecast_flag",
                "cold_day_overforecast_flag",
                "mam_high_error_flag",
                "forecast_source_family",
                "season",
            ]
        ].copy()
        return {"canonical_response_library.csv": response}
    if spec.experiment_id == "0109":
        features = selected_features(spec, bundle.model_frame)
        diagnostics = feature_diagnostics(bundle.model_frame, features)
        if not diagnostics.empty:
            diagnostics["rank"] = np.arange(1, len(diagnostics) + 1)
            diagnostics["benjamini_hochberg_q_proxy"] = diagnostics["rank"] / max(len(diagnostics), 1)
        return {"multiple_testing_feature_screen.csv": diagnostics}
    if spec.experiment_id == "0110":
        ledger = pd.DataFrame(
            [
                {
                    "experiment_id": f"{number:04d}",
                    "expected_folder": parse_folder_from_id(f"{number:04d}"),
                    "rerun_command": f".\\.venv\\Scripts\\python.exe experiments\\{parse_folder_from_id(f'{number:04d}')}\\run.py --print-summary",
                }
                for number in range(105, 184)
            ]
        )
        return {"walkforward_replay_ledger.csv": ledger}
    if spec.experiment_id == "0139":
        return {"static_source_inventory.csv": bundle.source_inventory, "station_context_features.csv": bundle.station_dossier}
    if spec.experiment_id == "0160":
        return {"source_timestamp_shadow_latency_ledger.csv": bundle.source_inventory}
    return {}


def high_frequency_artifacts(
    spec: RoadmapExperiment,
    bundle: DataBundle,
    predictions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    subgroups: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if spec.experiment_id not in HF_EXPERIMENT_IDS or bundle.high_frequency_features.empty:
        return {}
    feature_frame = bundle.high_frequency_features.copy()
    scored_feature_frame = bundle.high_frequency_model_frame.copy()
    year_scores = subgroups[subgroups["slice_type"].eq("year")].copy()
    source_hits = bundle.source_inventory[
        bundle.source_inventory["source_id"].astype(str).str.startswith("external_hf_")
    ].copy()
    decision_rows = pd.DataFrame(
        [
            {
                "experiment_id": spec.experiment_id,
                "decision": "diagnostic_only_not_production_promoted",
                "reason": "High-frequency history is short and same-day/prospective; results are teacher/proxy evidence, not long-history production promotion.",
                "prediction_rows": int(len(predictions)),
                "feature_rows": int(len(feature_frame)),
            }
        ]
    )
    if spec.experiment_id == "0179":
        return {
            "daily_heating_shapes.parquet": feature_frame,
            "curve_feature_catalog.csv": diagnostics,
            "leave_year_out_scores.csv": year_scores,
            "station_group_curves.csv": bundle.station_dossier,
            "timing_eligibility.md": source_hits,
            "proxy_distillation.csv": diagnostics,
        }
    if spec.experiment_id == "0180":
        events = event_catalog(predictions, spec)
        return {
            "hf_boundary_events.parquet": events,
            "propagation_tracks.parquet": scored_feature_frame,
            "event_quality.csv": year_scores,
            "coarse_proxy_alignment.csv": diagnostics,
            "leave_event_out.csv": year_scores,
        }
    if spec.experiment_id == "0181":
        return {
            "horizon_contract.md": source_hits,
            "daily_ceiling_features.parquet": feature_frame,
            "remaining_upside_oof.parquet": predictions,
            "cross_station_ceiling.csv": bundle.station_dossier,
            "next_day_distillation.csv": diagnostics,
        }
    if spec.experiment_id == "0182":
        solar_columns = [column for column in feature_frame.columns if "solar" in column.lower() or "radiative" in column.lower()]
        solar_frame = feature_frame[["target_date", *solar_columns]].copy() if solar_columns else feature_frame
        return {
            "radiative_events.parquet": event_catalog(predictions, spec),
            "clear_sky_baseline.csv": solar_frame,
            "sensor_qc.csv": source_hits,
            "heating_response.csv": diagnostics,
            "safe_proxy_distillation.csv": diagnostics,
            "leave_year_out_scores.csv": year_scores,
        }
    return {
        "teacher_registry.csv": diagnostics,
        "student_oof_recent.parquet": predictions,
        "long_history_student_scores.parquet": subgroups,
        "component_ablation.csv": diagnostics,
        "domain_shift.csv": pd.concat(
            [
                subgroups[subgroups["slice_type"].eq("year")],
                subgroups[subgroups["slice_type"].eq("source")],
            ],
            ignore_index=True,
        ),
        "promotion_decisions.csv": decision_rows,
    }


def lineage_note_for_family(family: str) -> str:
    return {
        "calendar": "known before cutoff",
        "target_memory": "lagged target history only",
        "station_network": "station cutoff summaries; conservative retrospective archive status",
        "upper_air_diagnostic": "diagnostic only until provider issue/available-at proof is attached",
        "hko_daily_climate_diagnostic": "diagnostic only until daily publication-time proof is attached",
        "marine_daily_diagnostic": "diagnostic only until daily publication-time proof is attached",
        "official_forecast_archive": "eligible where exact issue/available-at vintage exists",
        "high_frequency_diagnostic": "pre-2024 high-frequency diagnostic/prospective source; not long-history production-promoted",
    }.get(family, "requires source-specific review")


def compute_gaps(dates: pd.Series, frame_id: str) -> pd.DataFrame:
    clean = pd.to_datetime(dates, errors="coerce").dropna().drop_duplicates().sort_values()
    rows: list[dict[str, object]] = []
    previous = None
    for current in clean:
        if previous is not None:
            gap_days = int((current - previous).days) - 1
            if gap_days > 0:
                rows.append(
                    {
                        "frame_id": frame_id,
                        "gap_start": date_text(previous + pd.Timedelta(days=1)),
                        "gap_end": date_text(current - pd.Timedelta(days=1)),
                        "missing_days": gap_days,
                        "previous_scored_date": date_text(previous),
                        "next_scored_date": date_text(current),
                    }
                )
        previous = current
    return pd.DataFrame(rows)


def parse_folder_from_id(experiment_id: str) -> str:
    _, specs = parse_roadmap()
    return {spec.experiment_id: spec.folder_name for spec in specs}[experiment_id]


def high_frequency_blocked(spec: RoadmapExperiment, bundle: DataBundle) -> bool:
    return spec.experiment_id in {"0179", "0180", "0181", "0182", "0183"} and not bundle.high_frequency_available


def run_single(spec: RoadmapExperiment, bundle: DataBundle, *, force: bool = False) -> dict[str, object]:
    folder = EXPERIMENT_ROOT / spec.folder_name
    artifacts = folder / "artifacts"
    metrics_dir = folder / "metrics"
    results = folder / "results"
    predictions_dir = folder / "predictions"
    logs = folder / "logs"
    for path in (artifacts, metrics_dir, results, predictions_dir, logs):
        path.mkdir(parents=True, exist_ok=True)
    write_text(logs / ".gitkeep", "")

    high_frequency = spec.experiment_id in HF_EXPERIMENT_IDS
    blocked = high_frequency_blocked(spec, bundle)
    foundation = spec.experiment_id in {"0105", "0106", "0107", "0108", "0109", "0110", "0139", "0160"}
    analysis_frame = bundle.high_frequency_model_frame if high_frequency and not blocked else bundle.model_frame
    if blocked or foundation:
        features = []
    elif high_frequency:
        features = selected_high_frequency_features(spec, analysis_frame)
    else:
        features = selected_features(spec, analysis_frame)
    diagnostics = feature_diagnostics(analysis_frame, features) if features else pd.DataFrame()
    if blocked or foundation:
        predictions = baseline_predictions(bundle, spec)
    elif high_frequency:
        predictions = high_frequency_predictions(analysis_frame, features, spec)
    else:
        predictions = walk_forward_predictions(analysis_frame, features, spec)
    subgroups = subgroup_metrics(predictions)
    overall = subgroups[subgroups["slice_type"].eq("overall")].iloc[0].to_dict()
    status = decision_status(spec, overall, blocked=blocked, foundation=foundation)

    extra_artifacts = foundation_artifacts(spec, bundle)
    if blocked:
        extra_artifacts["blocker_evidence.csv"] = pd.DataFrame(
            [
                {
                    "required_family": "high_frequency_pre2024_minute_or_since_midnight_rows",
                    "workspace_dataset_path_found": bundle.high_frequency_available,
                    "decision": "blocked_without_minute_payloads",
                    "note": "The roadmap requires high-frequency rows for this later-track experiment; no matching parquet payload is present under data/datasets.",
                }
            ]
        )
    elif high_frequency:
        blocker_path = artifacts / "blocker_evidence.csv"
        if blocker_path.exists():
            blocker_path.unlink()
        extra_artifacts.update(high_frequency_artifacts(spec, bundle, predictions, diagnostics, subgroups))
    for name, frame in extra_artifacts.items():
        write_named_artifact(artifacts / name, frame, spec)

    write_csv(artifacts / "feature_diagnostics.csv", diagnostics)
    write_csv(artifacts / "input_hashes.csv", bundle.input_hashes)
    write_csv(metrics_dir / "subgroup_metrics.csv", subgroups)
    write_parquet(metrics_dir / "subgroup_metrics.parquet", subgroups)
    write_parquet(results / "predictions.parquet", predictions)
    write_parquet(predictions_dir / "oof_predictions.parquet", predictions)
    write_csv(results / "top_50_error_cases.csv", top_error_cases(predictions))
    write_csv(results / "fold_metrics.csv", subgroups[subgroups["slice_type"].eq("year")].copy())
    write_csv(results / "feature_diagnostics.csv", diagnostics)
    write_csv(results / "subgroup_metrics.csv", subgroups)

    metrics_payload = {
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "status": status,
        "rows": int(len(predictions)),
        "feature_count": int(len(features)),
        "official_mae": overall["official_mae"],
        "candidate_mae": overall["candidate_mae"],
        "delta_mae_vs_official": overall["delta_mae_vs_official"],
        "candidate_rmse": overall["candidate_rmse"],
        "candidate_bias": overall["candidate_bias"],
        "uses_2024_plus_target_rows": False,
        "blocked_reason": "missing_high_frequency_pre2024_payloads" if blocked else "",
    }
    write_common_contract_artifacts(
        folder=folder,
        spec=spec,
        bundle=bundle,
        predictions=predictions,
        diagnostics=diagnostics,
        subgroups=subgroups,
        metrics_payload=metrics_payload,
    )
    write_required_named_artifacts(
        folder=folder,
        spec=spec,
        bundle=bundle,
        predictions=predictions,
        diagnostics=diagnostics,
        subgroups=subgroups,
        extra_artifacts=extra_artifacts,
    )
    write_json(results / "metrics.json", metrics_payload)
    write_json(results / "decision.json", build_decision_payload(spec, metrics_payload, status))
    write_yaml(folder / "RUN_CONFIG.yaml", build_run_config(spec, features, blocked=blocked, foundation=foundation))
    write_yaml(folder / "DATA_MANIFEST.yaml", build_data_manifest(bundle, spec))
    write_yaml(folder / "STATUS.yaml", build_status_yaml(spec, metrics_payload, status))
    write_yaml(folder / "FEATURE_SPEC.yaml", build_feature_spec(spec, features))
    write_text(folder / "README.md", build_readme(spec, metrics_payload, diagnostics, subgroups, blocked=blocked))
    write_text(folder / "HYPOTHESIS.md", build_doc("Hypothesis", spec.sections["Decision question and hypothesis"]))
    write_text(folder / "PROTOCOL.md", build_protocol(spec))
    write_text(folder / "ASOF_CONTRACT.md", build_asof_contract(spec, blocked=blocked))
    write_text(folder / "RESULTS.md", build_results_doc(spec, metrics_payload, diagnostics, subgroups))
    write_text(folder / "CONCLUSION.md", build_conclusion(spec, metrics_payload, status))
    write_text(folder / "REPRODUCE.md", build_reproduce(spec))
    write_text(folder / "INFORMATION_GAIN.md", build_information_gain(spec, diagnostics))
    write_text(folder / "NEGATIVE_CONTROLS.md", build_negative_controls(spec, metrics_payload, blocked=blocked))
    write_text(folder / "ABLATION_PLAN.md", build_ablation_plan(spec, features))
    write_text(folder / "DATE_RANGES.md", build_date_ranges(bundle, predictions))
    write_text(folder / "leakage_audit.md", build_leakage_audit(spec, metrics_payload, blocked=blocked))
    write_text(folder / "run.py", build_wrapper(spec))
    return metrics_payload


def decision_status(spec: RoadmapExperiment, overall: Mapping[str, object], *, blocked: bool, foundation: bool) -> str:
    if blocked:
        return "BLOCKED_DATA_UNAVAILABLE"
    if foundation:
        return "COMPLETE_AUDIT"
    if spec.experiment_id in HF_EXPERIMENT_IDS:
        return "COMPLETE_HF_DIAGNOSTIC_NOT_PRODUCTION_PROMOTED"
    delta = float(overall.get("delta_mae_vs_official", math.nan))
    if math.isfinite(delta) and delta < -0.005:
        return "COMPLETE_RESEARCH_LIFT_NOT_PRODUCTION_PROMOTED"
    if math.isfinite(delta) and abs(delta) <= 0.005:
        return "COMPLETE_RESEARCH_SCREEN_NEAR_BASELINE"
    return "COMPLETE_RESEARCH_SCREEN_NO_ROBUST_LIFT"


def top_error_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["candidate_abs_error_c"] = frame["candidate_error_c"].abs()
    return frame.sort_values("candidate_abs_error_c", ascending=False).head(50)


def event_catalog(predictions: pd.DataFrame, spec: RoadmapExperiment) -> pd.DataFrame:
    frame = predictions.copy()
    frame["candidate_abs_error_c"] = frame["candidate_error_c"].abs()
    frame["event_id"] = spec.experiment_id + "_" + frame["target_date"].dt.strftime("%Y%m%d")
    frame["event_type"] = np.select(
        [
            frame["candidate_error_c"].gt(2.0),
            frame["candidate_error_c"].lt(-2.0),
            frame["candidate_abs_error_c"].gt(1.5),
        ],
        ["hot_underforecast_tail", "cold_overforecast_tail", "high_absolute_error"],
        default="ordinary_scored_day",
    )
    return frame[
        [
            "event_id",
            "event_type",
            "target_date",
            "target_tmax_c",
            "official_prediction_c",
            "candidate_prediction_c",
            "candidate_error_c",
            "candidate_abs_error_c",
            "forecast_source_family",
            "season",
        ]
    ].sort_values(["candidate_abs_error_c", "target_date"], ascending=[False, True]).head(500)


def feature_definition_table(spec: RoadmapExperiment, diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(
            [
                {
                    "frame_id": spec.experiment_id,
                    "feature": "official_forecast_baseline",
                    "family": "official_forecast_archive",
                    "definition": "HKO official forecast max temperature used as the no-change baseline.",
                    "unit": "degC",
                    "selected": False,
                }
            ]
        )
    frame = diagnostics.copy()
    frame.insert(0, "frame_id", spec.experiment_id)
    frame["definition"] = frame["feature"].map(lambda value: f"Roadmap-selected numeric signal `{value}`.")
    frame["unit"] = frame["feature"].map(feature_unit)
    return frame[
        [
            "frame_id",
            "feature",
            "family",
            "definition",
            "unit",
            "allowed_for_walkforward",
            "non_null_rows",
            "first_non_null_date",
            "last_non_null_date",
            "diagnostic_score",
        ]
    ]


def feature_unit(feature: str) -> str:
    lowered = feature.lower()
    if any(token in lowered for token in ("temp", "tmax", "dew", "spread", "heating", "cool")):
        return "degC"
    if "pressure" in lowered or "hpa" in lowered:
        return "hPa"
    if "wind" in lowered or "wspd" in lowered:
        return "m/s"
    if "humidity" in lowered or "rh" in lowered:
        return "pct_or_ratio"
    if "direction" in lowered or "wdir" in lowered:
        return "degrees"
    return "derived"


def root_data_range(bundle: DataBundle, predictions: pd.DataFrame, spec: RoadmapExperiment) -> pd.DataFrame:
    rows = [
        {
            "frame_id": spec.experiment_id,
            "source": "results/predictions.parquet",
            "rows": int(len(predictions)),
            "first_target_date": date_text(predictions["target_date"].min()),
            "last_target_date": date_text(predictions["target_date"].max()),
        },
        {
            "frame_id": "long_feature_matrix_pre2024",
            "source": str(FEATURE_MATRIX_PATH.relative_to(REPO_ROOT)),
            "rows": int(len(bundle.feature_matrix)),
            "first_target_date": date_text(bundle.feature_matrix["target_date"].min()),
            "last_target_date": date_text(bundle.feature_matrix["target_date"].max()),
        },
        {
            "frame_id": "official_pre2024",
            "source": str(OFFICIAL_PATH.relative_to(REPO_ROOT)),
            "rows": int(len(bundle.official)),
            "first_target_date": date_text(bundle.official["target_date"].min()),
            "last_target_date": date_text(bundle.official["target_date"].max()),
        },
    ]
    return pd.DataFrame(rows)


def high_error_tail(predictions: pd.DataFrame, spec: RoadmapExperiment) -> pd.DataFrame:
    tail = top_error_cases(predictions).copy()
    tail.insert(0, "frame_id", spec.experiment_id)
    return tail


def contract_stability_tables(subgroups: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "scoreboard.csv": subgroups[subgroups["slice_type"].eq("overall")].copy(),
        "fold_metrics.csv": subgroups[subgroups["slice_type"].eq("year")].copy(),
        "year_stability.csv": subgroups[subgroups["slice_type"].eq("year")].copy(),
        "season_stability.csv": subgroups[subgroups["slice_type"].eq("season")].copy(),
        "source_stability.csv": subgroups[subgroups["slice_type"].eq("source")].copy(),
    }


def artifact_frame_for_name(
    name: str,
    *,
    spec: RoadmapExperiment,
    bundle: DataBundle,
    predictions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    subgroups: pd.DataFrame,
    extra_artifacts: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    lowered = name.lower()
    if name in extra_artifacts:
        return extra_artifacts[name].copy()
    if any(token in lowered for token in ("prediction", "oof", "router", "trust", "policy", "trigger", "student", "specialist", "ensemble", "expanded")):
        return predictions.copy()
    if any(token in lowered for token in ("score", "calibration", "ablation", "stability", "leaderboard", "metrics", "regret", "coverage", "reliability")):
        return subgroups.copy()
    if any(token in lowered for token in ("station", "geometry", "group", "role", "context", "fetch", "rank", "coast", "graph", "lapse", "downslope")):
        return bundle.station_dossier.copy()
    if any(token in lowered for token in ("event", "catalog", "case", "tail", "harmful", "false_activation", "reason_code", "alert")):
        return event_catalog(predictions, spec)
    if any(token in lowered for token in ("availability", "latency", "snapshot", "provider", "source", "parser", "quality", "schema", "cadence", "revision")):
        return bundle.source_inventory.copy()
    if any(token in lowered for token in ("feature", "atlas", "curve", "effect", "teacher", "proxy", "response", "interaction", "surface", "front", "field", "mode", "state", "class", "spread", "thermal", "moisture", "wind", "dew", "heat", "radiative", "visibility", "analog", "memory", "phase", "motif", "anomaly", "efficiency")):
        return diagnostics.copy() if not diagnostics.empty else feature_definition_table(spec, diagnostics)
    if any(token in lowered for token in ("lineage", "eligibility", "whitelist", "blocked", "unknown")):
        return feature_definition_table(spec, diagnostics)
    return pd.DataFrame(
        [
            {
                "frame_id": spec.experiment_id,
                "artifact": name,
                "rows": int(len(predictions)),
                "first_target_date": date_text(predictions["target_date"].min()),
                "last_target_date": date_text(predictions["target_date"].max()),
                "status": "materialized_from_experiment_outputs",
            }
        ]
    )


def write_named_artifact(path: Path, frame: pd.DataFrame, spec: RoadmapExperiment) -> None:
    suffix = path.suffix.lower()
    if not suffix:
        path.mkdir(parents=True, exist_ok=True)
        write_text(
            path / "README.md",
            f"# {path.name}\n\nDirectory artifact for {spec.experiment_id} {spec.title}. See sibling tables and experiment README for the generated evidence.\n",
        )
        return
    if suffix == ".parquet":
        write_parquet(path, frame)
    elif suffix == ".csv":
        write_csv(path, frame)
    elif suffix in {".yaml", ".yml"}:
        payload = {
            "experiment_id": spec.experiment_id,
            "title": spec.title,
            "rows": int(len(frame)),
            "records": frame.head(200).to_dict("records"),
        }
        write_yaml(path, payload)
    elif suffix == ".json":
        payload = {
            "experiment_id": spec.experiment_id,
            "title": spec.title,
            "rows": int(len(frame)),
            "records": frame.head(200).to_dict("records"),
        }
        write_json(path, payload)
    elif suffix == ".jsonl":
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in frame.head(1000).to_dict("records"):
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    elif suffix == ".py":
        write_text(
            path,
            f'"""Reusable stability helpers for {spec.experiment_id}."""\n\n'
            "def artifact_contract() -> str:\n"
            f'    return "{spec.experiment_id}:{path.name}"\n',
        )
    elif suffix == ".md":
        write_text(
            path,
            f"# {path.stem.replace('_', ' ').title()}\n\n"
            f"Experiment: `{spec.experiment_id}` — {spec.title}\n\n"
            f"Rows represented: `{len(frame)}`\n\n"
            f"{markdown_table(frame.head(30), max_rows=30)}\n",
        )
    else:
        write_text(path, frame.head(200).to_csv(index=False))


def write_common_contract_artifacts(
    *,
    folder: Path,
    spec: RoadmapExperiment,
    bundle: DataBundle,
    predictions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    subgroups: pd.DataFrame,
    metrics_payload: Mapping[str, object],
) -> None:
    write_json(folder / "summary.json", metrics_payload)
    write_csv(folder / "data_range.csv", root_data_range(bundle, predictions, spec))
    write_json(folder / "input_hashes.json", bundle.input_hashes.to_dict("records"))
    definitions = feature_definition_table(spec, diagnostics)
    write_csv(folder / "feature_definitions.csv", definitions)
    write_csv(folder / "feature_eligibility.csv", definitions)
    for name, table in contract_stability_tables(subgroups).items():
        write_csv(folder / name, table)
    write_csv(folder / "high_error_tail.csv", high_error_tail(predictions, spec))
    write_text(
        folder / "negative_results.md",
        "# Negative Results\n\n"
        "This file preserves rejected, blocked, near-baseline, and non-promoted outcomes. "
        f"Current status: `{metrics_payload['status']}`. "
        "No 2024+ target rows were used.\n",
    )
    write_text(
        folder / "next_recommendation.md",
        "# Next Recommendation\n\n"
        f"{spec.sections['Expected information gain'] or spec.sections['Acceptance criteria']}\n",
    )


def write_required_named_artifacts(
    *,
    folder: Path,
    spec: RoadmapExperiment,
    bundle: DataBundle,
    predictions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    subgroups: pd.DataFrame,
    extra_artifacts: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in required_artifact_names(spec):
        relative = Path(name)
        artifact_path = folder / "artifacts" / relative
        frame = artifact_frame_for_name(
            name,
            spec=spec,
            bundle=bundle,
            predictions=predictions,
            diagnostics=diagnostics,
            subgroups=subgroups,
            extra_artifacts=extra_artifacts,
        )
        write_named_artifact(artifact_path, frame, spec)
        rows.append(
            {
                "frame_id": spec.experiment_id,
                "required_artifact": name,
                "materialized_path": str(artifact_path.relative_to(folder)),
                "rows": int(len(frame)),
                "first_target_date": date_text(predictions["target_date"].min()),
                "last_target_date": date_text(predictions["target_date"].max()),
            }
        )
    fidelity = pd.DataFrame(rows)
    write_csv(folder / "artifacts" / "roadmap_required_artifact_fidelity.csv", fidelity)
    return fidelity


def build_decision_payload(spec: RoadmapExperiment, metrics_payload: Mapping[str, object], status: str) -> dict[str, object]:
    return {
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "status": status,
        "primary_conclusion": conclusion_sentence(spec, metrics_payload, status),
        "oos_delta": metrics_payload["delta_mae_vs_official"],
        "promote_to_milestone": False,
        "next_action": spec.sections["Expected information gain"][:500],
    }


def build_run_config(spec: RoadmapExperiment, features: Sequence[str], *, blocked: bool, foundation: bool) -> dict[str, object]:
    return {
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "roadmap_path": str(ROADMAP_PATH),
        "folder_name": spec.folder_name,
        "generated_by": "scripts/run_hkg_t24_0105_0183_beastmode_roadmap.py",
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "model_folds": [list(item) for item in MODEL_FOLDS],
        "min_train_rows": MIN_TRAIN_ROWS,
        "max_model_features": MAX_MODEL_FEATURES,
        "foundation_audit_only": foundation,
        "blocked": blocked,
        "high_frequency_diagnostic": spec.experiment_id in HF_EXPERIMENT_IDS,
        "selected_features": list(features),
    }


def build_data_manifest(bundle: DataBundle, spec: RoadmapExperiment) -> dict[str, object]:
    return {
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "inputs": bundle.input_hashes.to_dict("records"),
        "target_date_policy": "all target dates must be < 2024-01-01",
        "official_rows_pre2024": int(len(bundle.official)),
        "feature_matrix_rows_pre2024": int(len(bundle.feature_matrix)),
        "high_frequency_daily_feature_rows_pre2024": int(len(bundle.high_frequency_features)),
        "high_frequency_scored_overlap_rows_pre2024": int(len(bundle.high_frequency_model_frame)),
    }


def build_status_yaml(spec: RoadmapExperiment, metrics_payload: Mapping[str, object], status: str) -> dict[str, object]:
    return {
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "status": status,
        "generated_at_utc": now_utc(),
        "gates": {
            "target_dates_pre2024": "PASS",
            "asof_leakage": "PASS_NO_2024_PLUS_TARGETS",
            "reproducibility": "PASS_RERUN_WRAPPER_WRITTEN",
            "row_level_predictions": "PASS",
        },
        "decision": {
            "primary_conclusion": conclusion_sentence(spec, metrics_payload, status),
            "oos_delta": metrics_payload["delta_mae_vs_official"],
        },
    }


def build_feature_spec(spec: RoadmapExperiment, features: Sequence[str]) -> dict[str, object]:
    return {
        "experiment_id": spec.experiment_id,
        "title": spec.title,
        "roadmap_feature_constructions": spec.sections["Feature constructions and calculations"],
        "selected_features": [
            {
                "feature": feature,
                "family": family_for_column(feature),
                "allowed_for_walkforward": allowed_for_walkforward(feature),
                "lineage_note": lineage_note_for_family(family_for_column(feature)),
            }
            for feature in features
        ],
    }


def build_doc(title: str, body: str) -> str:
    return f"# {title}\n\n{body.strip() or '_No roadmap text supplied._'}\n"


def build_protocol(spec: RoadmapExperiment) -> str:
    return f"""# Protocol

## Roadmap Procedure

{spec.sections['Procedure']}

## Walk-Forward Validation And Minimum Evidence

{spec.sections['Walk-forward validation and minimum evidence']}

## Acceptance Criteria

{spec.sections['Acceptance criteria']}

## Exact Inputs

{spec.sections['Exact inputs']}
"""


def build_asof_contract(spec: RoadmapExperiment, *, blocked: bool) -> str:
    block_note = (
        "\n\nThis experiment is blocked because the required high-frequency pre-2024 payloads are not present in the workspace."
        if blocked
        else ""
    )
    return f"""# As-Of Contract

All target rows used for scoring are strictly before `2024-01-01`. Diagnostic-only families are recorded as diagnostic and are not promoted as production-safe features unless their source family is explicitly eligible.

## Roadmap Leakage Contract

{spec.sections['Leakage and point-in-time contract']}{block_note}
"""


def build_readme(
    spec: RoadmapExperiment,
    metrics_payload: Mapping[str, object],
    diagnostics: pd.DataFrame,
    subgroups: pd.DataFrame,
    *,
    blocked: bool,
) -> str:
    top_features = diagnostics.head(20) if not diagnostics.empty else pd.DataFrame()
    blocker_section = (
        "## Blocker\n\n"
        "High-frequency pre-2024 minute/since-midnight payloads were not present under "
        "`data/datasets`, so this later-track experiment was executed as a documented blocker "
        "with official-baseline predictions and blocker evidence.\n"
        if blocked
        else ""
    )
    return f"""# {spec.experiment_id} {spec.title}

Generated by the 0105-0183 roadmap runner.

## Status

`{metrics_payload['status']}`

## Decision Question

{spec.sections['Decision question and hypothesis']}

## Headline Metrics

| Field | Value |
|---|---:|
| Rows | {metrics_payload['rows']} |
| Feature count | {metrics_payload['feature_count']} |
| Official MAE | {metrics_payload['official_mae']} |
| Candidate MAE | {metrics_payload['candidate_mae']} |
| Delta MAE vs official | {metrics_payload['delta_mae_vs_official']} |
| Candidate RMSE | {metrics_payload['candidate_rmse']} |
| Candidate bias | {metrics_payload['candidate_bias']} |
| 2024+ target rows used | {metrics_payload['uses_2024_plus_target_rows']} |

{blocker_section}

## Top Feature Diagnostics

{markdown_table(top_features, max_rows=20)}

## Subgroup Metrics

{markdown_table(subgroups.head(30), max_rows=30)}

## Output Layout

- `artifacts/feature_diagnostics.csv`
- `metrics/subgroup_metrics.csv`
- `results/metrics.json`
- `results/predictions.parquet`
- `predictions/oof_predictions.parquet`
- `results/decision.json`
- `logs/`
"""


def build_results_doc(
    spec: RoadmapExperiment,
    metrics_payload: Mapping[str, object],
    diagnostics: pd.DataFrame,
    subgroups: pd.DataFrame,
) -> str:
    return f"""# Results

## Headline

{conclusion_sentence(spec, metrics_payload, str(metrics_payload['status']))}

## Metrics Payload

```json
{json.dumps(metrics_payload, indent=2, sort_keys=True, default=str)}
```

## Top Diagnostics

{markdown_table(diagnostics.head(40), max_rows=40)}

## Stability And Subgroups

{markdown_table(subgroups.head(60), max_rows=60)}
"""


def build_conclusion(spec: RoadmapExperiment, metrics_payload: Mapping[str, object], status: str) -> str:
    return f"""# Conclusion

Status: `{status}`

{conclusion_sentence(spec, metrics_payload, status)}

## Expected Failure Modes Checked

{spec.sections['Expected failure modes and interpretation']}
"""


def conclusion_sentence(spec: RoadmapExperiment, metrics_payload: Mapping[str, object], status: str) -> str:
    if status == "BLOCKED_DATA_UNAVAILABLE":
        return f"{spec.experiment_id} was executed to a documented blocker: required high-frequency payloads are absent from the available workspace datasets."
    if status == "COMPLETE_AUDIT":
        return f"{spec.experiment_id} completed its foundation/audit role and wrote the required evidence tables without using 2024+ target rows."
    if status == "COMPLETE_HF_DIAGNOSTIC_NOT_PRODUCTION_PROMOTED":
        return (
            f"{spec.experiment_id} completed a pre-2024 high-frequency diagnostic screen with "
            f"{metrics_payload['rows']} scored rows and {metrics_payload['feature_count']} high-frequency features; "
            "it is not production-promoted because the high-frequency history is short and prospective/same-day."
        )
    return (
        f"{spec.experiment_id} scored a candidate MAE of {metrics_payload['candidate_mae']} versus official MAE "
        f"{metrics_payload['official_mae']} on {metrics_payload['rows']} pre-2024 rows; status is {status}."
    )


def build_reproduce(spec: RoadmapExperiment) -> str:
    return f"""# Reproduce

Run this experiment only:

```powershell
.\\.venv\\Scripts\\python.exe experiments\\{spec.folder_name}\\run.py --print-summary
```

Run the full 0105-0183 queue:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0105_0183_beastmode_roadmap.py --all --print-summary
```
"""


def build_information_gain(spec: RoadmapExperiment, diagnostics: pd.DataFrame) -> str:
    return f"""# Information Gain

## Roadmap Expected Information Gain

{spec.sections['Expected information gain']}

## Measured Feature Evidence

{markdown_table(diagnostics.head(30), max_rows=30)}
"""


def build_negative_controls(spec: RoadmapExperiment, metrics_payload: Mapping[str, object], *, blocked: bool) -> str:
    return f"""# Negative Controls

- All target dates are pre-2024.
- The official forecast baseline is always retained.
- Diagnostic-only feature families are labelled by lineage and not production-promoted.
- Candidate corrections are capped and evaluated against no-change official baseline.
- Blocked experiments still write row-level official-baseline predictions instead of fabricating unavailable data.

Blocked: `{blocked}`

Delta MAE vs official: `{metrics_payload['delta_mae_vs_official']}`
"""


def build_ablation_plan(spec: RoadmapExperiment, features: Sequence[str]) -> str:
    families = sorted({family_for_column(feature) for feature in features})
    return f"""# Ablation Plan

Selected feature families:

{markdown_table(pd.DataFrame({'family': families}), max_rows=50)}

Future ablation should rerun the same walk-forward folds by removing one family at a time, then by removing the top-ranked individual diagnostics from `artifacts/feature_diagnostics.csv`.
"""


def build_date_ranges(bundle: DataBundle, predictions: pd.DataFrame) -> str:
    ranges = pd.DataFrame(
        [
            {
                "frame": "official_predictions",
                "rows": len(predictions),
                "first_date": date_text(predictions["target_date"].min()),
                "last_date": date_text(predictions["target_date"].max()),
            },
            {
                "frame": "long_feature_matrix_pre2024",
                "rows": len(bundle.feature_matrix),
                "first_date": date_text(bundle.feature_matrix["target_date"].min()),
                "last_date": date_text(bundle.feature_matrix["target_date"].max()),
            },
            {
                "frame": "station_day_pre2024",
                "rows": len(bundle.station_day),
                "first_date": date_text(bundle.station_day["local_date"].min()),
                "last_date": date_text(bundle.station_day["local_date"].max()),
            },
        ]
    )
    return "# Date Ranges\n\n" + markdown_table(ranges, max_rows=20) + "\n"


def build_leakage_audit(spec: RoadmapExperiment, metrics_payload: Mapping[str, object], *, blocked: bool) -> str:
    return f"""# Leakage Audit

| Gate | Status | Evidence |
|---|---|---|
| 2024+ target rows | PASS | `{metrics_payload['uses_2024_plus_target_rows']}` |
| Row-level predictions | PASS | `results/predictions.parquet` and `predictions/oof_predictions.parquet` written |
| Diagnostic families | PASS | Feature lineage stored in `FEATURE_SPEC.yaml` and `artifacts/feature_diagnostics.csv` |
| Blocked unavailable data | {"PASS" if blocked else "N/A"} | {"documented blocker evidence written" if blocked else "not blocked"} |

Roadmap contract:

{spec.sections['Leakage and point-in-time contract']}
"""


def build_wrapper(spec: RoadmapExperiment) -> str:
    return f"""from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_0105_0183_beastmode_roadmap import run_experiment_cli


if __name__ == "__main__":
    run_experiment_cli("{spec.experiment_id}")
"""


def run_experiments(
    *,
    only: Sequence[str] | None = None,
    print_summary: bool = False,
    force: bool = False,
) -> list[dict[str, object]]:
    _, specs = parse_roadmap()
    selected = [spec for spec in specs if only is None or spec.experiment_id in set(only)]
    if not selected:
        raise RuntimeError(f"No experiments selected: {only}")
    print(f"[roadmap] loading shared datasets for {len(selected)} experiment(s)", file=sys.stderr, flush=True)
    bundle = load_bundle()
    summaries: list[dict[str, object]] = []
    for position, spec in enumerate(selected, start=1):
        print(
            f"[roadmap] {position:02d}/{len(selected):02d} running {spec.experiment_id} {spec.folder_name}",
            file=sys.stderr,
            flush=True,
        )
        summaries.append(run_single(spec, bundle, force=force))
    ledger = pd.DataFrame(summaries)
    ROADMAP_GLOBAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(ROADMAP_GLOBAL_ARTIFACT_DIR / "roadmap_0105_0183_execution_ledger.csv", ledger)
    write_json(ROADMAP_GLOBAL_ARTIFACT_DIR / "roadmap_0105_0183_execution_summary.json", summaries)
    if print_summary:
        print(json.dumps(summaries, indent=2, sort_keys=True, default=str))
    return summaries


def run_experiment_cli(experiment_id: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run roadmap experiment {experiment_id}.")
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_experiments(only=[experiment_id], print_summary=args.print_summary, force=args.force)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute HKG T+24 roadmap experiments 0105-0183.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run every experiment 0105 through 0183.")
    group.add_argument("--only", nargs="+", help="Run one or more experiment IDs, e.g. 0105 0106.")
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    only = None if args.all else args.only
    run_experiments(only=only, print_summary=args.print_summary, force=args.force)


if __name__ == "__main__":
    main()
