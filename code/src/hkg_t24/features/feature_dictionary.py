"""Feature dictionaries and strict/proxy/shadow scope validation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import astuple, dataclass
from pathlib import Path

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.constants import (
    CALENDAR_MODEL_FEATURE_WHITELIST,
    OFFICIAL_FEATURE_WHITELIST,
    PROXY_FEATURE_PREFIXES,
    ROUTER_DERIVED_FEATURE_WHITELIST,
    SHADOW_FEATURE_PREFIXES,
    STRICT_FEATURE_PREFIXES,
    STRICT_FORBIDDEN_FEATURE_PREFIXES,
    TARGET_MEMORY_FEATURE_WHITELIST,
    TARGET_MEMORY_MISSING_INDICATOR_FEATURES,
    assert_no_forbidden_target_memory_names,
)
from hkg_t24.features.nwp_daily import (
    HKG_NWP_LOCATIONS,
    shadow_center_feature_names,
    threshold_feature_key,
)
from hkg_t24.features.online_state import all_online_state_feature_names
from hkg_t24.utils.sql import csv_line


@dataclass(frozen=True)
class FeatureDefinition:
    feature_name: str
    feature_scope: str
    source_family: str
    unit: str
    formula: str
    strict_allowed: bool
    proxy_allowed: bool
    shadow_allowed: bool


FEATURE_DICTIONARY_HEADERS = (
    "feature_name",
    "feature_scope",
    "source_family",
    "unit",
    "formula",
    "strict_allowed",
    "proxy_allowed",
    "shadow_allowed",
)


def _defs(
    names: Iterable[str],
    *,
    feature_scope: str,
    source_family: str,
    unit: str,
    formula: str,
    strict_allowed: bool,
    proxy_allowed: bool,
    shadow_allowed: bool,
) -> list[FeatureDefinition]:
    return [
        FeatureDefinition(
            feature_name=name,
            feature_scope=feature_scope,
            source_family=source_family,
            unit=unit,
            formula=formula,
            strict_allowed=strict_allowed,
            proxy_allowed=proxy_allowed,
            shadow_allowed=shadow_allowed,
        )
        for name in names
    ]


def gfs_feature_names() -> tuple[str, ...]:
    names = [f"gfs__{location}__tmax_c" for location in HKG_NWP_LOCATIONS]
    names.extend(
        [
            "gfs__center__dewpoint_change_proxy_c",
            "gfs__center__low_cloud_pct_mean",
            "gfs__center__shortwave_w_m2_mean",
            "gfs__center__precip_mm_sum",
            "gfs__center__wind_speed_10m_mean_mps",
            "gfs__center__onshore_easterly_component_mps",
            "gfs__center__temp_dewpoint_spread_mean_c",
            "gfs__center__t850_c_mean",
            "gfs__center__z500_m_mean",
            "gfs__center__relative_humidity_700_pct_mean",
            "gfs__center__dewpoint_2m_c_mean",
            "gfs__spatial__inland_nw_minus_center_tmax_c",
            "gfs__spatial__inland_nw_minus_marine_s_tmax_c",
            "gfs__spatial__center_minus_marine_s_tmax_c",
            "gfs__spatial__local_n_minus_local_s_tmax_c",
            "gfs__spatial__local_e_minus_local_w_tmax_c",
        ]
    )
    return tuple(names)


def gefsmean_feature_names() -> tuple[str, ...]:
    return tuple(
        [f"gefsmean__{location}__tmax_c" for location in HKG_NWP_LOCATIONS]
        + [
            "gefsmean__center__pwat_kg_m2_mean",
            "gefsmean__center__onshore_east_component_mps_mean",
            "gefsmean__center__wind_speed_10m_mps_mean",
        ]
    )


def gefsens_feature_names() -> tuple[str, ...]:
    names = [
        "gefsens__center__tmax_p10_c",
        "gefsens__center__tmax_p25_c",
        "gefsens__center__tmax_p50_c",
        "gefsens__center__tmax_p75_c",
        "gefsens__center__tmax_p90_c",
        "gefsens__center__tmax_spread_p90_p10_c",
    ]
    for step in range(60, 81):
        threshold = step / 2.0
        names.append(f"gefsens__center__{threshold_feature_key(threshold)}")
    return tuple(names)


def strict_feature_definitions() -> list[FeatureDefinition]:
    target_names = TARGET_MEMORY_FEATURE_WHITELIST + TARGET_MEMORY_MISSING_INDICATOR_FEATURES
    definitions: list[FeatureDefinition] = []
    definitions.extend(
        _defs(
            CALENDAR_MODEL_FEATURE_WHITELIST,
            feature_scope="strict",
            source_family="calendar",
            unit="mixed",
            formula="deterministic target-date calendar features",
            strict_allowed=True,
            proxy_allowed=False,
            shadow_allowed=False,
        )
    )
    definitions.extend(
        _defs(
            OFFICIAL_FEATURE_WHITELIST,
            feature_scope="strict",
            source_family="official",
            unit="mixed",
            formula="latest eligible pre-freeze official HKO row and pre-freeze revision path",
            strict_allowed=True,
            proxy_allowed=False,
            shadow_allowed=False,
        )
    )
    definitions.extend(
        _defs(
            target_names,
            feature_scope="strict",
            source_family="target_memory",
            unit="mixed",
            formula="finalized target labels at T-2 or older only",
            strict_allowed=True,
            proxy_allowed=False,
            shadow_allowed=False,
        )
    )
    definitions.extend(
        _defs(
            all_online_state_feature_names(),
            feature_scope="strict",
            source_family="online_residual_state",
            unit="mixed",
            formula="online residual states replayed from prior settled dates only",
            strict_allowed=True,
            proxy_allowed=False,
            shadow_allowed=False,
        )
    )
    definitions.extend(
        _defs(
            gfs_feature_names(),
            feature_scope="strict",
            source_family="gfs",
            unit="mixed",
            formula="strict H24N GribStream safe-row deterministic GFS features",
            strict_allowed=True,
            proxy_allowed=False,
            shadow_allowed=False,
        )
    )
    definitions.extend(
        _defs(
            gefsmean_feature_names(),
            feature_scope="strict",
            source_family="gefsmean",
            unit="mixed",
            formula="strict H24N GribStream safe-row GEFS mean features",
            strict_allowed=True,
            proxy_allowed=False,
            shadow_allowed=False,
        )
    )
    definitions.extend(
        _defs(
            gefsens_feature_names(),
            feature_scope="strict",
            source_family="gefsens",
            unit="mixed",
            formula="strict H24N GribStream safe-row GEFS member features at HKO center",
            strict_allowed=True,
            proxy_allowed=False,
            shadow_allowed=False,
        )
    )
    definitions.extend(
        _defs(
            ROUTER_DERIVED_FEATURE_WHITELIST,
            feature_scope="strict",
            source_family="router_derived",
            unit="mixed",
            formula="cutoff-safe OOF expert/router diagnostics derived inside Jira003 replay",
            strict_allowed=True,
            proxy_allowed=False,
            shadow_allowed=False,
        )
    )
    validate_feature_names("strict", [definition.feature_name for definition in definitions])
    return sorted(definitions, key=lambda item: item.feature_name)


def proxy_feature_definitions() -> list[FeatureDefinition]:
    names = (
        "station__network__available_count",
        "station__network__tmax_mean_c",
        "station__network__tmax_max_c",
        "station__network__tmax_min_c",
        "station__network__tmax_range_c",
        "climate__lagged__available_count",
        "climate__lagged__last_tmax_c",
        "climate__lagged__roll7_mean_c",
        "climate__lagged__roll30_mean_c",
    )
    definitions = _defs(
        names,
        feature_scope="proxy",
        source_family="station_and_climate_proxy",
        unit="mixed",
        formula="diagnostic proxy-only station/climate features excluded from strict scope",
        strict_allowed=False,
        proxy_allowed=True,
        shadow_allowed=False,
    )
    validate_feature_names("proxy", [definition.feature_name for definition in definitions])
    return sorted(definitions, key=lambda item: item.feature_name)


def shadow_feature_definitions() -> list[FeatureDefinition]:
    definitions = _defs(
        shadow_center_feature_names(),
        feature_scope="live_shadow",
        source_family="shadow_nwp",
        unit="C",
        formula="shadow/direct placeholder center Tmax feature excluded from strict scope",
        strict_allowed=False,
        proxy_allowed=False,
        shadow_allowed=True,
    )
    validate_feature_names("live_shadow", [definition.feature_name for definition in definitions])
    return sorted(definitions, key=lambda item: item.feature_name)


def all_feature_definitions() -> list[FeatureDefinition]:
    return strict_feature_definitions() + proxy_feature_definitions() + shadow_feature_definitions()


def validate_feature_names(scope: str, feature_names: Sequence[str]) -> None:
    """Validate feature names against scope-specific leakage/prefix rules."""
    assert_no_forbidden_target_memory_names(tuple(feature_names))
    if scope == "strict":
        illegal_prefix = [
            name
            for name in feature_names
            if not any(name.startswith(prefix) for prefix in STRICT_FEATURE_PREFIXES)
        ]
        forbidden_prefix = [
            name
            for name in feature_names
            if any(name.startswith(prefix) for prefix in STRICT_FORBIDDEN_FEATURE_PREFIXES)
        ]
        offenders = sorted(set(illegal_prefix + forbidden_prefix))
        if offenders:
            raise ValueError("Strict feature names contain forbidden prefixes: " + ", ".join(offenders))
    elif scope == "proxy":
        offenders = [
            name
            for name in feature_names
            if not any(name.startswith(prefix) for prefix in PROXY_FEATURE_PREFIXES)
            or any(name.startswith(prefix) for prefix in SHADOW_FEATURE_PREFIXES)
        ]
        if offenders:
            raise ValueError("Proxy feature names contain invalid prefixes: " + ", ".join(sorted(offenders)))
    elif scope == "live_shadow":
        offenders = [
            name
            for name in feature_names
            if not any(name.startswith(prefix) for prefix in SHADOW_FEATURE_PREFIXES)
        ]
        if offenders:
            raise ValueError("Shadow feature names contain invalid prefixes: " + ", ".join(sorted(offenders)))
    else:
        raise ValueError(f"Unsupported feature scope: {scope}")


def ordered_feature_names(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Lexicographic features with missing indicators immediately after base features."""
    feature_set = set(feature_names)
    base_names = sorted(name for name in feature_set if not name.endswith("__is_missing"))
    ordered: list[str] = []
    for name in base_names:
        ordered.append(name)
        missing_name = f"{name}__is_missing"
        if missing_name in feature_set:
            ordered.append(missing_name)
    remaining = sorted(feature_set - set(ordered))
    ordered.extend(remaining)
    return tuple(ordered)


def write_feature_dictionaries(writer: ReportWriter) -> tuple[Path, Path, Path, Path]:
    """Write the three CSV dictionaries and combined Markdown dictionary report."""
    strict = strict_feature_definitions()
    proxy = proxy_feature_definitions()
    shadow = shadow_feature_definitions()
    strict_path = writer.write_csv("feature_dictionary_strict.csv", FEATURE_DICTIONARY_HEADERS, [astuple(row) for row in strict])
    proxy_path = writer.write_csv("feature_dictionary_proxy.csv", FEATURE_DICTIONARY_HEADERS, [astuple(row) for row in proxy])
    shadow_path = writer.write_csv("feature_dictionary_shadow.csv", FEATURE_DICTIONARY_HEADERS, [astuple(row) for row in shadow])
    combined_lines = [
        "# HKG-T24-002 Feature Dictionary",
        "",
        "## Scope Counts",
        "",
        f"- strict: {len(strict)}",
        f"- proxy: {len(proxy)}",
        f"- live_shadow: {len(shadow)}",
        "",
        "## Strict Prefix Contract",
        "",
        "- Strict features are limited to calendar, official, target, online, gfs, gefsmean, and gefsens prefixes.",
        "- Proxy station/climate and shadow NWP prefixes are excluded from strict feature matrices.",
        "",
        "## CSV Headers",
        "",
        csv_line(FEATURE_DICTIONARY_HEADERS),
    ]
    markdown_path = writer.paths.reports_dir / "feature_dictionary.md"
    markdown_path.write_text("\n".join(combined_lines).rstrip() + "\n", encoding="utf-8")
    return strict_path, proxy_path, shadow_path, markdown_path
