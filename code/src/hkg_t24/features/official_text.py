"""Deterministic official forecast text feature helpers."""

from __future__ import annotations

import re

from hkg_t24.constants import OFFICIAL_FEATURE_WHITELIST

TEXT_FLAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "official__text_hot_flag": ("hot",),
    "official__text_very_hot_flag": ("very hot",),
    "official__text_showers_flag": ("shower", "showers"),
    "official__text_thunderstorm_flag": ("thunderstorm", "thundery"),
    "official__text_cloudy_flag": ("cloudy", "overcast"),
    "official__text_fine_flag": ("fine", "sunny"),
    "official__text_mist_fog_flag": ("mist", "fog", "haze"),
    "official__text_easterly_flag": ("easterly", "east force", "east to"),
    "official__text_light_wind_flag": ("light wind", "light winds", "force 2", "force 3"),
}


def official_text_flags(text: str | None) -> dict[str, int]:
    """Return final Jira 002 official text flags from free text."""
    normalized = "" if text is None else text.lower()
    return {
        feature_name: int(any(pattern in normalized for pattern in patterns))
        for feature_name, patterns in TEXT_FLAG_PATTERNS.items()
    }


def psr_numeric_proxy(psr_value: object | None, text: str | None) -> float | None:
    """Map HKO probability-of-significant-rain text or numeric fields to 0..1."""
    normalized_psr = "" if psr_value is None else str(psr_value).strip().lower().replace("_", " ")
    numeric_match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*%?", normalized_psr)
    if numeric_match is not None:
        numeric_value = float(numeric_match.group(1))
        return numeric_value / 100.0 if numeric_value > 1.0 else numeric_value
    psr_map = {
        "low": 0.20,
        "medium low": 0.35,
        "medium": 0.50,
        "medium high": 0.65,
        "high": 0.80,
    }
    if normalized_psr in psr_map:
        return psr_map[normalized_psr]

    normalized_text = "" if text is None else text.lower()
    if "thunderstorm" in normalized_text or "thundery" in normalized_text or "heavy rain" in normalized_text:
        return 0.70
    if "isolated showers" in normalized_text:
        return 0.40
    if "showers" in normalized_text or "shower" in normalized_text:
        return 0.55
    if "rain" in normalized_text:
        return 0.60
    if ("fine" in normalized_text or "sunny" in normalized_text) and "rain" not in normalized_text and "shower" not in normalized_text:
        return 0.15
    return None


def official_text_feature_names() -> tuple[str, ...]:
    """Return the official text/PSR feature subset in dictionary order."""
    names = tuple(TEXT_FLAG_PATTERNS) + ("official__psr_numeric_proxy",)
    return tuple(name for name in OFFICIAL_FEATURE_WHITELIST if name in names)
