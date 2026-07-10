"""Hand-built regime flags for HKO forecast and hourly context text."""

from __future__ import annotations

import re
from collections.abc import Mapping


FORECAST_PATTERNS: Mapping[str, str] = {
    "fcst_flag_showers": r"SHOWERS?",
    "fcst_flag_heavy_showers": r"(HEAVY|HEAVIER).{0,80}SHOWERS?|SHOWERS?.{0,80}(HEAVY|HEAVIER)",
    "fcst_flag_thunderstorm": r"THUNDERSTORMS?",
    "fcst_flag_squally": r"SQUALLY",
    "fcst_flag_bright_periods": r"BRIGHT PERIODS",
    "fcst_flag_sunny": r"SUNNY INTERVALS|SUNNY PERIODS|\bSUNNY\b",
    "fcst_flag_cloudy": r"MAINLY CLOUDY|\bCLOUDY\b|OVERCAST",
    "fcst_flag_fine": r"\bFINE\b",
    "fcst_flag_very_hot": r"VERY HOT",
    "fcst_flag_hot": r"\bHOT\b|VERY HOT",
    "fcst_flag_mist_fog": r"\bMIST\b|\bFOG\b|\bHAZE\b",
    "fcst_flag_tropical_cyclone": (
        r"TROPICAL STORM|TYPHOON|TROPICAL CYCLONE|SEVERE TROPICAL STORM"
    ),
    "fcst_flag_monsoon": r"MONSOON",
    "fcst_flag_active_southerly": r"ACTIVE SOUTHERLY AIRSTREAM",
    "fcst_flag_southerly": r"SOUTHERLY|SOUTHEASTERLY|SOUTH TO SOUTHEASTERLY",
    "fcst_flag_easterly_ne": r"EASTERLY|NORTHEASTERLY|NORTH TO NORTHEASTERLY",
}

HOURLY_PATTERNS: Mapping[str, str] = {
    "hourly_any_thunderstorm_warning": r"THUNDERSTORM WARNING|THUNDERSTORMS?",
    "hourly_any_rainstorm_warning": r"RAINSTORM WARNING|RAINSTORM",
    "hourly_any_amber_rainstorm": r"AMBER RAINSTORM",
    "hourly_any_red_black_rainstorm": r"RED RAINSTORM|BLACK RAINSTORM",
    "hourly_any_very_hot_warning": r"VERY HOT WEATHER WARNING",
    "hourly_any_strong_monsoon": r"STRONG MONSOON",
    "hourly_any_tropical_cyclone_text": r"TROPICAL CYCLONE|TROPICAL STORM|TYPHOON",
    "hourly_any_lightning_text": r"LIGHTNING",
    "hourly_any_rainfall_text": r"RAINFALL|MILLIMETRES OF RAINFALL|RAIN",
}


def contains_pattern(text: object, pattern: str) -> bool:
    if text is None:
        return False
    value = str(text)
    if not value or value.lower() == "nan":
        return False
    return re.search(pattern, value, flags=re.IGNORECASE | re.DOTALL) is not None


def forecast_text_flags(text: object) -> dict[str, int]:
    flags = {name: int(contains_pattern(text, pattern)) for name, pattern in FORECAST_PATTERNS.items()}
    flags["fcst_flag_nt_higher"] = int(
        contains_pattern(text, r"NEW TERRITORIES")
        and contains_pattern(text, r"COUPLE OF DEGREES HIGHER|DEGREES HIGHER")
    )
    flags["fcst_text_present_flag"] = int(text is not None and str(text).strip() != "")
    return flags


def hourly_text_flags(text_parts: list[object], suffix: str) -> dict[str, int]:
    combined = "\n".join(str(part) for part in text_parts if part is not None and str(part) != "nan")
    flags = {
        f"{name}_{suffix}": int(contains_pattern(combined, pattern))
        for name, pattern in HOURLY_PATTERNS.items()
    }
    flags[f"hourly_text_block_present_{suffix}"] = int(bool(combined.strip()))
    return flags

