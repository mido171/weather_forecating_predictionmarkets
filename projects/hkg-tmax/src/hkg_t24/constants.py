"""Binding constants for the HKG-T24 implementation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

CUTOFF_ID = "H24N"
CUTOFF_RULE_VERSION = "hkg_t24_1500hkt_v1"
TARGET_DATE_COLUMN = "target_date_hkt"
SNAPSHOT_ID_PREFIX = "H24N:"
START_TARGET_DATE = date(2000, 1, 2)
END_TARGET_DATE = date(2026, 6, 21)

STRICT_SCHEMA_VERSION = "hkg_t24_h24n_strict_v1_20260626_patch1"
PROXY_SCHEMA_VERSION = "hkg_t24_h24n_proxy_v1_20260626_patch1"
SHADOW_SCHEMA_VERSION = "hkg_t24_h24n_shadow_v1_20260626_patch1"
CODE_VERSION = "hkg-t24-003-router-specialists-distribution-v1"

DATABASE_URL_ENV = "HKG_TMAX_DATABASE_URL"
DATABASE_DSN_ENV = "HKG_TMAX_DB_DSN"
MISSING_DSN_ERROR = (
    "ERROR: Database DSN not configured. Set HKG_TMAX_DATABASE_URL or HKG_TMAX_DB_DSN. "
    "HKG_TMAX_DATABASE_URL has priority when both are present."
)
DUAL_DSN_WARNING = "Using HKG_TMAX_DATABASE_URL; HKG_TMAX_DB_DSN is present but ignored."
LIGHTGBM_ERROR = (
    "ERROR: lightgbm is required for HKG T24 first full implementation. "
    "Install project dependencies before running the pipeline."
)

MODEL_SCHEMAS = (
    "model_core",
    "model_features",
    "model_oof",
    "model_router",
    "model_validation",
    "model_live",
    "model_audit",
    "model_eval",
)

REPORT_NAMES = (
    "phase0_preflight_report.md",
    "schema_conflict_report.md",
    "source_inventory_report.md",
    "source_registry.csv",
    "schema_contract_report.md",
    "schema_migration_source_registry.md",
    "schema_migration_feature_matrix.md",
    "gribstream_source_scope_audit.csv",
    "gribstream_source_scope_audit.md",
    "snapshot_coverage_report.csv",
    "snapshot_coverage_report.md",
    "live_shadow_availability_report.csv",
    "live_shadow_availability_report.md",
    "leakage_audit_report.md",
    "jira_001_contract_coverage.md",
    "official_anchor_coverage.md",
    "online_state_audit_report.md",
    "feature_dictionary.md",
    "feature_matrix_coverage_report.md",
    "expert_oof_scoreboard.md",
    "expert_factory_report.md",
    "oof_integrity_report.md",
    "model_selection_report.md",
    "jira_002_contract_coverage.md",
    "router_report.md",
    "specialist_report.md",
    "distribution_calibration_report.md",
    "calibration_report.md",
    "system_replay_report.md",
    "jira_003_contract_coverage.md",
)

PLACEHOLDER_REASON_CODES = (
    "SOURCE_TABLE_ABSENT",
    "SOURCE_TOO_SHORT",
    "NO_ELIGIBLE_ROWS_FOR_DATE",
    "BLOCKED_SOURCE",
    "NOT_PROMOTED",
    "SEALED_LABELS_UNAVAILABLE",
    "LIVE_COLLECTOR_NOT_STARTED",
    "INSUFFICIENT_HISTORY",
)

STRICT_NWP_DATASETS = ("gfs", "gefsatmosmean", "gefsatmos")
SHADOW_NWP_DATASETS = (
    "ifsoper",
    "ifsenfo",
    "cwawrf15",
    "aifsoper",
    "aifsenfo",
    "aigfssfc",
    "graphcast",
    "fourcastnetgfs",
)
BLOCKED_DAILY_TMAX_DATASETS = ("nbmoc", "aigfspres", "aigefssfc")

DATASET_FEATURE_PREFIX = {
    "gfs": "gfs",
    "gefsatmosmean": "gefsmean",
    "gefsatmos": "gefsens",
    "ifsoper": "ifsoper",
    "ifsenfo": "ifsens",
    "cwawrf15": "cwawrf15",
    "aifsoper": "aifsoper",
    "aifsenfo": "aifsens",
    "aigfssfc": "aigfssfc",
    "aigfspres": "aigfspres",
    "aigefssfc": "aigefssfc",
    "graphcast": "graphcast",
    "fourcastnetgfs": "fourcastnet",
    "nbmoc": "nbmoc",
}

GRIBSTREAM_EXPERT_ID = {
    "gfs": "E4_GFS_MOS",
    "gefsatmosmean": "E5_GEFS_ENSEMBLE",
    "gefsatmos": "E5_GEFS_ENSEMBLE",
    "ifsoper": "E6_IFS_OPER_SHADOW",
    "ifsenfo": "E7_IFS_ENS_SHADOW",
    "cwawrf15": "E9_CWA_WRF_LIVE_SHADOW",
    "aifsoper": "E8_AI_SHADOW",
    "aifsenfo": "E8_AI_SHADOW",
    "aigfssfc": "E8_AI_SHADOW",
    "graphcast": "E10_GRAPHCAST_SHADOW",
    "fourcastnetgfs": "E10_FOURCASTNET_SHADOW",
}

CALENDAR_MODEL_FEATURE_WHITELIST = (
    "calendar__month_sin1",
    "calendar__month_cos1",
    "calendar__doy_sin1",
    "calendar__doy_cos1",
    "calendar__is_mam",
    "calendar__is_jja",
    "calendar__is_son",
    "calendar__is_djf",
    "calendar__year_index",
)

TARGET_MEMORY_FEATURE_WHITELIST = (
    "target__lag2_tmax_c",
    "target__lag3_tmax_c",
    "target__lag7_tmax_c",
    "target__lag14_tmax_c",
    "target__lag30_tmax_c",
    "target__lag60_tmax_c",
    "target__lag365_tmax_c",
    "target__roll7_mean_lag2_c",
    "target__roll14_mean_lag2_c",
    "target__roll30_mean_lag2_c",
    "target__roll60_mean_lag2_c",
    "target__roll365_mean_lag2_c",
    "target__roll7_std_lag2_c",
    "target__roll14_std_lag2_c",
    "target__roll30_std_lag2_c",
    "target__range7_lag2_c",
    "target__range14_lag2_c",
    "target__slope7_lag2_c_per_day",
    "target__slope30_lag2_c_per_day",
    "target__slope7_minus_slope30_lag2_c_per_day",
    "target__lag2_minus_roll7_c",
    "target__lag2_minus_roll30_c",
    "target__roll7_minus_roll30_c",
    "target__hot_spell_length_lag2_days",
    "target__cool_spell_length_lag2_days",
    "target__clim30_mean_c",
    "target__clim30_std_c",
    "target__lag2_minus_clim30_c",
    "target__warming_trend_10y_c_per_year",
    "target__year_index",
)

TARGET_MEMORY_MISSING_INDICATOR_FEATURES = tuple(
    f"{feature_name}__is_missing"
    for feature_name in TARGET_MEMORY_FEATURE_WHITELIST
    if feature_name != "target__year_index"
)

FORBIDDEN_FINALIZED_TARGET_TERMS = ("target__lag1_", "lag1_tmax")

OFFICIAL_FEATURE_WHITELIST = (
    "official__forecast_min_c",
    "official__forecast_max_c",
    "official__forecast_range_c",
    "official__forecast_midpoint_c",
    "official__issue_hour_hkt",
    "official__hours_before_cutoff",
    "official__revision_count",
    "official__revision_min_delta_c",
    "official__revision_max_delta_c",
    "official__revision_range_delta_c",
    "official__text_hot_flag",
    "official__text_very_hot_flag",
    "official__text_showers_flag",
    "official__text_thunderstorm_flag",
    "official__text_cloudy_flag",
    "official__text_fine_flag",
    "official__text_mist_fog_flag",
    "official__text_easterly_flag",
    "official__text_light_wind_flag",
    "official__psr_numeric_proxy",
    "official__forecast_max_minus_gefs_median_c",
    "official__forecast_max_minus_gfs_center_tmax_c",
    "official__forecast_max_minus_target_roll7_c",
)

ONLINE_SOURCE_SCOPES = (
    ("official_raw", "global"),
    ("official_raw", "source_era"),
    ("official_raw", "month"),
    ("official_raw", "season"),
    ("gfs_mos", "global"),
    ("gefs_prob_mos", "global"),
    ("r0_router", "global"),
    ("r1_router", "global"),
    ("final_system", "global"),
)

ONLINE_HALF_LIVES = (5, 10, 20, 40)

STRICT_FEATURE_PREFIXES = (
    "calendar__",
    "official__",
    "target__",
    "online__",
    "gfs__",
    "gefsmean__",
    "gefsens__",
    "router__",
)

PROXY_FEATURE_PREFIXES = (
    "calendar__",
    "official__",
    "target__",
    "online__",
    "station__",
    "climate__",
)

SHADOW_FEATURE_PREFIXES = (
    "ifsoper__",
    "ifsens__",
    "aifsoper__",
    "aifsens__",
    "aigfssfc__",
    "graphcast__",
    "fourcastnet__",
    "cwawrf15__",
    "arwf__",
)

STRICT_FORBIDDEN_FEATURE_PREFIXES = (
    "ifsoper__",
    "ifsens__",
    "aifsoper__",
    "aifsens__",
    "aigfssfc__",
    "graphcast__",
    "fourcastnet__",
    "cwawrf15__",
    "arwf__",
    "station__",
    "climate__",
    "igra__",
    "tc__",
)

EXPERT_IDS = (
    "E0_OFFICIAL_RAW_ANCHOR",
    "E1_OFFICIAL_RESIDUAL",
    "E2_TARGET_MEMORY",
    "E3_STATION_PROXY",
    "E4_GFS_MOS",
    "E5_GEFS_ENSEMBLE",
    "E6_IFS_OPER_SHADOW",
    "E7_IFS_ENS_SHADOW",
    "E8_AI_NWP_SHADOW",
    "E9_CWA_WRF_LIVE_SHADOW",
    "E10_DIAGNOSTIC_PROXY",
    "E11_ARWF_LIVE_SHADOW",
)

ROUTER_IDS = (
    "R0_OFFICIAL_LONG_HISTORY",
    "R1_CORE_GFS_GEFS",
    "R2_IFS_SHADOW_ADAPTER",
    "R3_AI_SHADOW_ADAPTER",
    "R4_LIVE_SHADOW_ADAPTER",
)

ROUTER_SHORT_IDS = {
    "R0": "R0_OFFICIAL_LONG_HISTORY",
    "R1": "R1_CORE_GFS_GEFS",
    "R2": "R2_IFS_SHADOW_ADAPTER",
    "R3": "R3_AI_SHADOW_ADAPTER",
    "R4": "R4_LIVE_SHADOW_ADAPTER",
}

ROUTER_EXPERTS = {
    "R0_OFFICIAL_LONG_HISTORY": (
        "E0_OFFICIAL_RAW_ANCHOR",
        "E1_OFFICIAL_RESIDUAL",
        "E2_TARGET_MEMORY",
    ),
    "R1_CORE_GFS_GEFS": (
        "E0_OFFICIAL_RAW_ANCHOR",
        "E1_OFFICIAL_RESIDUAL",
        "E2_TARGET_MEMORY",
        "E4_GFS_MOS",
        "E5_GEFS_ENSEMBLE",
    ),
    "R2_IFS_SHADOW_ADAPTER": ("E6_IFS_OPER_SHADOW", "E7_IFS_ENS_SHADOW"),
    "R3_AI_SHADOW_ADAPTER": ("E8_AI_NWP_SHADOW", "E10_DIAGNOSTIC_PROXY"),
    "R4_LIVE_SHADOW_ADAPTER": ("E9_CWA_WRF_LIVE_SHADOW", "E11_ARWF_LIVE_SHADOW"),
}

EXPERT_STRICT_WEIGHT_CAPS = {
    "E0_OFFICIAL_RAW_ANCHOR": 0.80,
    "E1_OFFICIAL_RESIDUAL": 0.80,
    "E2_TARGET_MEMORY": 0.40,
    "E3_STATION_PROXY": 0.0,
    "E4_GFS_MOS": 0.70,
    "E5_GEFS_ENSEMBLE": 0.70,
    "E6_IFS_OPER_SHADOW": 0.0,
    "E7_IFS_ENS_SHADOW": 0.0,
    "E8_AI_NWP_SHADOW": 0.0,
    "E9_CWA_WRF_LIVE_SHADOW": 0.0,
    "E10_DIAGNOSTIC_PROXY": 0.0,
    "E11_ARWF_LIVE_SHADOW": 0.0,
}

ROUTER_TAU_GRID = (0.25, 0.35, 0.50, 0.75, 1.00)
ROUTER_LAMBDA_GRID = (0.0, 0.25, 0.50)

SPECIALIST_IDS = (
    "S1_MARINE_SUPPRESSION",
    "S2_WEAK_WIND_HEAT",
    "S3_MAM_TRANSITION",
    "S4_CLOUD_RAIN_SUPPRESSION",
    "S5_DRY_RIDGE_HEAT",
    "S6_HIGH_ERROR_TAIL",
)

DISTRIBUTION_THRESHOLDS_C = tuple(step / 2.0 for step in range(40, 81))

ROUTER_DERIVED_FEATURE_WHITELIST = (
    "router__expert_prediction_spread_c",
    "router__missing_expert_count",
    "router__expected_abs_error_c",
)

PLACEHOLDER_EXPERT_IDS = (
    "E6_IFS_OPER_SHADOW",
    "E7_IFS_ENS_SHADOW",
    "E8_AI_NWP_SHADOW",
    "E9_CWA_WRF_LIVE_SHADOW",
    "E10_DIAGNOSTIC_PROXY",
    "E11_ARWF_LIVE_SHADOW",
)

ARWF_WARNING = (
    "WARNING: ARWF source table absent. E11_ARWF_LIVE_SHADOW will emit placeholder rows "
    "with SOURCE_TABLE_ABSENT."
)
CWA_WRF_WARNING = (
    "WARNING: cwawrf15 source absent or too short. "
    "E9_CWA_WRF_LIVE_SHADOW will emit placeholder rows."
)


@dataclass(frozen=True)
class SourceRegistryRow:
    """Final-patch source-registry row."""

    source_code: str
    source_family: str
    source_role: str
    feature_prefix: str
    strict_allowed: bool
    proxy_allowed: bool
    shadow_allowed: bool
    blocked: bool
    live_only: bool
    support_only: bool
    unit_semantics_verified: bool
    availability_grade: str
    source_time_policy: str
    min_target_date_hkt: str | None
    max_target_date_hkt: str | None
    required_source_scope: str | None
    blocker_reason: str | None
    promotion_gate: str
    notes: str


SOURCE_REGISTRY_ROWS: tuple[SourceRegistryRow, ...] = (
    SourceRegistryRow(
        "hko_target_labels",
        "target",
        "strict_core",
        "target",
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        "EXACT_VINTAGE",
        "finalized daily labels may enter only as T-2-or-older target memory",
        "2000-01-02",
        "2026-06-21",
        None,
        None,
        "always included as labels and lagged target memory only",
        "Finalized target-day value is never a same-day feature.",
    ),
    SourceRegistryRow(
        "hko_official_forecasts",
        "official",
        "strict_core",
        "official",
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        "EXACT_VINTAGE",
        "issue_at_utc must be <= operational freeze for the target date",
        "2000-01-02",
        "2026-06-21",
        None,
        None,
        "always included when eligible row exists",
        "Primary official HKO forecast anchor.",
    ),
    SourceRegistryRow(
        "calendar",
        "deterministic",
        "strict_core",
        "calendar",
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        "EXACT_VINTAGE",
        "deterministic target-date metadata known before cutoff",
        "2000-01-02",
        "2026-06-21",
        None,
        None,
        "always included",
        "Only whitelisted cyclical/month/season/year-index fields may enter models.",
    ),
    SourceRegistryRow(
        "gfs",
        "gribstream",
        "strict_core",
        "gfs",
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        "CONSERVATIVE_SCHEDULE",
        "run_time_utc + 6 hours <= formal H24N cutoff",
        "2021-03-23",
        "2026-06-23",
        "full_tactical_backfill_ok_tmax",
        None,
        "core strict expert E4",
        "Strict NWP source after full-run scope and H24N safe-row filters.",
    ),
    SourceRegistryRow(
        "gefsatmosmean",
        "gribstream",
        "strict_core",
        "gefsmean",
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        "CONSERVATIVE_SCHEDULE",
        "run_time_utc + 6 hours <= formal H24N cutoff",
        "2021-03-23",
        "2026-06-23",
        "full_tactical_backfill_ok_tmax",
        None,
        "core strict expert E5 context",
        "GEFS mean context source after audited safe-row filter.",
    ),
    SourceRegistryRow(
        "gefsatmos",
        "gribstream",
        "strict_core",
        "gefsens",
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        "CONSERVATIVE_SCHEDULE",
        "run_time_utc + 6 hours <= formal H24N cutoff",
        "2021-03-23",
        "2026-06-23",
        "full_tactical_backfill_ok_tmax",
        None,
        "core strict expert E5 ensemble",
        "HKO-center GEFS ensemble source after audited safe-row filter.",
    ),
    SourceRegistryRow(
        "ifsoper",
        "gribstream",
        "shadow_challenger",
        "ifsoper",
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        "CONSERVATIVE_SCHEDULE",
        "run_time_utc + 6 hours <= formal H24N cutoff",
        "2021-03-23",
        "2026-06-23",
        "full_tactical_backfill_ok_tmax",
        None,
        "may enter after sealed protocol",
        "Shadow challenger, excluded from strict v1 features.",
    ),
    SourceRegistryRow(
        "ifsenfo",
        "gribstream",
        "shadow_challenger",
        "ifsens",
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        "CONSERVATIVE_SCHEDULE",
        "run_time_utc + 6 hours <= formal H24N cutoff",
        "2021-03-23",
        "2026-06-23",
        "full_tactical_backfill_ok_tmax",
        None,
        "may enter after sealed protocol; member-0 tracked",
        "IFS ensemble shadow challenger.",
    ),
    SourceRegistryRow(
        "cwawrf15",
        "gribstream",
        "live_shadow",
        "cwawrf15",
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        "LIVE_FIRST_SEEN_ONLY",
        "live first-seen collection only",
        "2026-06-23",
        None,
        "full_tactical_backfill_ok_tmax",
        None,
        "prospective only until two seasonal cycles",
        "CWA WRF live shadow source; absence is warning-level for Jira 001.",
    ),
    SourceRegistryRow(
        "aifsoper",
        "gribstream",
        "shadow_challenger",
        "aifsoper",
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        "CONSERVATIVE_SCHEDULE",
        "run_time_utc + 6 hours <= formal H24N cutoff",
        None,
        None,
        "full_tactical_backfill_ok_tmax",
        None,
        "may enter after sealed protocol",
        "AI deterministic shadow source.",
    ),
    SourceRegistryRow(
        "aifsenfo",
        "gribstream",
        "shadow_challenger",
        "aifsens",
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        "CONSERVATIVE_SCHEDULE",
        "run_time_utc + 6 hours <= formal H24N cutoff",
        None,
        None,
        "full_tactical_backfill_ok_tmax",
        None,
        "may enter after sealed protocol",
        "AI ensemble shadow source.",
    ),
    SourceRegistryRow(
        "aigfssfc",
        "gribstream",
        "shadow_challenger",
        "aigfssfc",
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        "CONSERVATIVE_SCHEDULE",
        "run_time_utc + 6 hours <= formal H24N cutoff",
        None,
        None,
        "full_tactical_backfill_ok_tmax",
        None,
        "may enter after sealed protocol",
        "AI/GFS surface shadow source over short range.",
    ),
    SourceRegistryRow(
        "aigfspres",
        "gribstream",
        "support_only",
        "aigfspres",
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        "CONSERVATIVE_SCHEDULE",
        "upper-air support only, not a daily Tmax source",
        None,
        None,
        "full_tactical_backfill_ok_tmax",
        None,
        "support-only source",
        "Excluded from daily Tmax strict/proxy/shadow feature matrices.",
    ),
    SourceRegistryRow(
        "aigefssfc",
        "gribstream",
        "blocked",
        "aigefssfc",
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        "BLOCKED",
        "blocked because daily Tmax coverage is too sparse",
        None,
        None,
        "full_tactical_backfill_ok_tmax",
        "Poor usable daily Tmax candidate coverage.",
        "blocked until provider/selector fix",
        "Rows can be leakage-safe but not usable enough for daily Tmax.",
    ),
    SourceRegistryRow(
        "graphcast",
        "gribstream",
        "shadow_challenger",
        "graphcast",
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        "MODEL_RUN_TIME_PROXY_ONLY",
        "model run time proxy only",
        None,
        None,
        "full_tactical_backfill_ok_tmax",
        None,
        "may enter after sealed protocol",
        "Shadow source; availability proof remains weaker than exact first-seen.",
    ),
    SourceRegistryRow(
        "fourcastnetgfs",
        "gribstream",
        "shadow_challenger",
        "fourcastnet",
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        "MODEL_RUN_TIME_PROXY_ONLY",
        "model run time proxy only; archive ends before current period",
        None,
        None,
        "full_tactical_backfill_ok_tmax",
        None,
        "may enter after sealed protocol",
        "Shadow source available through observed archive end only.",
    ),
    SourceRegistryRow(
        "nbmoc",
        "gribstream",
        "blocked",
        "nbmoc",
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        "BLOCKED",
        "blocked empty/probe-only source",
        None,
        None,
        "full_tactical_backfill_ok_tmax",
        "No usable HKO-domain daily Tmax coverage.",
        "blocked until non-empty source proof",
        "Probe-only source excluded from all feature matrices.",
    ),
    SourceRegistryRow(
        "station_network_proxy",
        "diagnostic_station_network",
        "proxy_research",
        "station",
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        "DIAGNOSTIC_ONLY",
        "proxy research only pending operational-vintage repair",
        None,
        None,
        None,
        None,
        "research proxy only",
        "Station wind direction remains forbidden until repaired.",
    ),
    SourceRegistryRow(
        "hko_daily_climate_proxy",
        "diagnostic_physics",
        "proxy_research",
        "hko_climate",
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        "DIAGNOSTIC_ONLY",
        "finalized daily climate is not live exact-vintage",
        None,
        None,
        None,
        None,
        "research proxy only",
        "Never use finalized target-day daily climate as a strict live feature.",
    ),
    SourceRegistryRow(
        "igra_upper_air_proxy",
        "diagnostic_physics",
        "support_only",
        "igra",
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        "DIAGNOSTIC_ONLY",
        "support/proxy only pending sentinel and vintage repair",
        None,
        None,
        None,
        None,
        "support-only proxy",
        "IGRA contains known sentinel/scale issues in current inventory.",
    ),
    SourceRegistryRow(
        "tc_best_track_proxy",
        "diagnostic_regime_labels",
        "support_only",
        "tc",
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        "DIAGNOSTIC_ONLY",
        "retrospective best-track only",
        None,
        None,
        None,
        None,
        "support-only proxy",
        "Retrospective TC best-track may not be used as live strict input.",
    ),
    SourceRegistryRow(
        "arwf_live",
        "live_nwp_anchor",
        "live_shadow",
        "arwf",
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        "LIVE_FIRST_SEEN_ONLY",
        "live first-seen collection only",
        "2026-06-19",
        None,
        None,
        None,
        "live shadow after enough history",
        "ARWF absence is warning-level for Jira 001 and emits placeholders.",
    ),
)


def assert_no_forbidden_target_memory_names(feature_names: tuple[str, ...]) -> None:
    """Fail if finalized daily target-memory feature names use forbidden lag1 wording."""
    offenders = [
        name
        for name in feature_names
        if any(forbidden in name for forbidden in FORBIDDEN_FINALIZED_TARGET_TERMS)
    ]
    if offenders:
        raise ValueError(f"Forbidden finalized target-memory feature names: {', '.join(offenders)}")
