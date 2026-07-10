from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_STATION = "KLGA"
TARGET_TZ = "America/New_York"
TRADER_TZ = "Europe/Stockholm"
UTC_TZ = "UTC"

FEATURE_SET_NAME = "klga_tmax_core"
FEATURE_VERSION = "supplemental_doc_1_v1"
FORMULA_CONTRACT_HASH = "supplemental_doc_1_plus_patch_1"

TEMP_GRID_F = tuple(range(50, 116))
PMF_SUM_TOLERANCE = 1e-8

EXIT_CONFIG_ERROR = 10
EXIT_MIGRATION_ERROR = 20
EXIT_VALIDATION_ERROR = 30
EXIT_TARGET_GRID_ERROR = 31
EXIT_DATA_CONTRACT_ERROR = 40
EXIT_TRAINING_ERROR = 50
EXIT_PREDICTION_ERROR = 60
EXIT_CALIBRATION_ERROR = 70
EXIT_EVALUATION_ERROR = 80
EXIT_REPORT_ERROR = 90

GLOBAL_RANDOM_SEED = 1729

CRITICAL_SOURCE_FAMILIES = (
    "wunderground",
    "mos_guidance",
    "gribstream",
)

STALE_SOURCE_MAX_AGE_HOURS = {
    "wunderground": 48.0,
    "station_daily_actuals": 48.0,
    "station_observations": 6.0,
    "mos_guidance": 18.0,
    "gribstream": 18.0,
}

REQUIRED_SCHEMAS = (
    "registry",
    "bronze",
    "silver",
    "gold",
    "predictions",
    "trading",
    "reports",
    "audit",
)
