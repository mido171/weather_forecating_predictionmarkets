from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo


NY_ZONE = ZoneInfo("America/New_York")
UTC_ZONE = ZoneInfo("UTC")

FEATURE_CONTRACT_VERSION = "v2_regime_cloud_uv_wx_visprecip"

TARGET_STATION_ID = "KLGA:9:US"
NEIGHBOR_STATION_IDS = (
    "KJFK:9:US",
    "KEWR:9:US",
    "KTEB:9:US",
    "KHPN:9:US",
    "KISP:9:US",
    "KBDR:9:US",
    "KMMU:9:US",
)
ALL_STATION_IDS = (TARGET_STATION_ID,) + NEIGHBOR_STATION_IDS

COASTAL_STATIONS = ("KJFK:9:US", "KISP:9:US", "KBDR:9:US")
INLAND_STATIONS = ("KEWR:9:US", "KMMU:9:US")
NORTH_INTERIOR_STATIONS = ("KHPN:9:US",)
URBAN_FRINGE_STATIONS = ("KTEB:9:US",)

CUTOFF_START_HOUR = 4
CUTOFF_START_MINUTE = 0
CUTOFF_END_HOUR = 18
CUTOFF_END_MINUTE = 0
CUTOFF_STEP_MINUTES = 30

WINDOW_MINUTES = (30, 60, 120, 180, 360)
REGIME_WINDOW_MINUTES = (60, 180, 360)
WX_WINDOW_MINUTES = (180, 360)
NEIGHBOR_GRADIENT_WINDOW_MINUTES = (180, 360)

OBS_ALLOWED_COLUMNS = (
    "request_location_id",
    "valid_time_utc",
    "temp",
    "dew_pt",
    "rh",
    "pressure",
    "vis",
    "wspd",
    "wdir",
    "gust",
    "precip_hrly",
    "clds",
    "wx_phrase",
    "uv_index",
    "uv_desc",
    "wdir_cardinal",
    "feels_like",
)
OBS_OPTIONAL_COLUMNS = ("feels_like",)
BANNED_OBS_COLUMNS = {"max_temp", "min_temp", "precip_total"}

SPLIT_TRAIN_START = date(1992, 1, 1)
SPLIT_TRAIN_END = date(2021, 12, 31)
SPLIT_VAL_START = date(2022, 1, 1)
SPLIT_VAL_END = date(2023, 12, 31)
SPLIT_TEST_START = date(2024, 1, 1)
SPLIT_TEST_END = date(2025, 12, 31)

DELTA_CLASS_MAX = 60
DELTA_OBJECTIVE = "multiclass"
DELTA_USE_CLASS_WEIGHTS = True
DELTA_USE_CUTOFF_WEIGHTS = False
DELTA_CUTOFF_WEIGHT_ALPHA = 1.0

ENABLE_V2_REGIME_FEATURES = True
ENABLE_NEIGHBOR_REGIME_FEATURES = True
ENABLE_V2_VIS_PRECIP_WDIR_DYNAMICS = True
ENABLE_STATE_ANOMALY_LOOKUPS = True
KEEP_MERGE_INDEX_FEATURES = False
INCLUDE_FEELS_LIKE = False

ANALOG_FEATURE_COLUMNS = (
    "cutoff_sin",
    "cutoff_cos",
    "doy_sin",
    "doy_cos",
    "is_dst",
    "temp_now",
    "dewpoint_depression_now",
    "pressure_now",
    "wspd_now",
    "wdir_sin",
    "wdir_cos",
    "tmax_sofar",
    "temp_now_minus_tmax",
    "mins_since_tmax",
    "temp_slope_30",
    "temp_slope_60",
    "temp_slope_180",
    "dew_pt_slope_60",
    "pressure_slope_180",
    "any_precip_sofar",
    "precip_frac_sofar",
    "coastal_minus_inland_temp",
    "nbr_temp_range",
    "nbr_pressure_range",
    "temp_inland_mean_minus_klga",
    "temp_jfk_minus_klga",
)

ANALOG_FEATURE_WEIGHTS = {
    "temp_now_minus_tmax": 2.0,
    "mins_since_tmax": 2.0,
    "temp_slope_60": 2.0,
    "coastal_minus_inland_temp": 2.0,
}


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    for parent in [current] + list(current.parents):
        has_pom = (parent / "pom.xml").exists()
        has_ml = (parent / "ml").exists()
        has_models = (parent / "models").exists()
        if has_pom and has_ml and has_models:
            return parent
    raise RuntimeError("Unable to locate repository root from current path.")


def default_output_root() -> Path:
    return find_repo_root() / "artifacts" / "same_day_res_poly"


def observation_columns(*, include_feels_like: bool) -> tuple[str, ...]:
    cols = [c for c in OBS_ALLOWED_COLUMNS if c not in OBS_OPTIONAL_COLUMNS]
    if include_feels_like:
        cols.extend(OBS_OPTIONAL_COLUMNS)
    return tuple(cols)


@dataclass(frozen=True)
class SplitConfig:
    train_start: date = SPLIT_TRAIN_START
    train_end: date = SPLIT_TRAIN_END
    val_start: date = SPLIT_VAL_START
    val_end: date = SPLIT_VAL_END
    test_start: date = SPLIT_TEST_START
    test_end: date = SPLIT_TEST_END


@dataclass(frozen=True)
class PipelineConfig:
    split: SplitConfig = field(default_factory=SplitConfig)
    feature_contract_version: str = FEATURE_CONTRACT_VERSION
    target_station_id: str = TARGET_STATION_ID
    neighbor_station_ids: tuple[str, ...] = NEIGHBOR_STATION_IDS
    local_zone: ZoneInfo = NY_ZONE
    cutoff_start_hour: int = CUTOFF_START_HOUR
    cutoff_start_minute: int = CUTOFF_START_MINUTE
    cutoff_end_hour: int = CUTOFF_END_HOUR
    cutoff_end_minute: int = CUTOFF_END_MINUTE
    cutoff_step_minutes: int = CUTOFF_STEP_MINUTES
    windows_minutes: tuple[int, ...] = WINDOW_MINUTES
    regime_windows_minutes: tuple[int, ...] = REGIME_WINDOW_MINUTES
    wx_windows_minutes: tuple[int, ...] = WX_WINDOW_MINUTES
    neighbor_gradient_windows_minutes: tuple[int, ...] = NEIGHBOR_GRADIENT_WINDOW_MINUTES
    delta_class_max: int = DELTA_CLASS_MAX
    delta_objective: str = DELTA_OBJECTIVE
    delta_use_class_weights: bool = DELTA_USE_CLASS_WEIGHTS
    delta_use_cutoff_weights: bool = DELTA_USE_CUTOFF_WEIGHTS
    delta_cutoff_weight_alpha: float = DELTA_CUTOFF_WEIGHT_ALPHA
    enable_v2_regime_features: bool = ENABLE_V2_REGIME_FEATURES
    enable_neighbor_regime_features: bool = ENABLE_NEIGHBOR_REGIME_FEATURES
    enable_v2_vis_precip_wdir_dynamics: bool = ENABLE_V2_VIS_PRECIP_WDIR_DYNAMICS
    enable_state_anomaly_lookups: bool = ENABLE_STATE_ANOMALY_LOOKUPS
    keep_merge_index_features: bool = KEEP_MERGE_INDEX_FEATURES
    include_feels_like: bool = INCLUDE_FEELS_LIKE
    output_root: Path = field(default_factory=default_output_root)
    analog_feature_columns: tuple[str, ...] = ANALOG_FEATURE_COLUMNS
    analog_feature_weights: dict[str, float] = field(
        default_factory=lambda: dict(ANALOG_FEATURE_WEIGHTS)
    )
    analog_k_grid: tuple[int, ...] = (50, 100, 200)
    analog_default_k: int = 100
    analog_min_pool: int = 150
    analog_min_non_peak: int = 20
    analog_season_window_doy: int = 30

    @property
    def all_station_ids(self) -> tuple[str, ...]:
        return (self.target_station_id,) + tuple(self.neighbor_station_ids)

    @property
    def observation_columns(self) -> tuple[str, ...]:
        return observation_columns(include_feels_like=self.include_feels_like)

    @property
    def feature_store_stem(self) -> str:
        return f"klga_feature_store_{self.feature_contract_version}"
