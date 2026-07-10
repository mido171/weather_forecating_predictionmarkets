from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import math
import re
from statistics import median
from typing import Iterable
from zoneinfo import ZoneInfo

from klga_tmax.constants import TARGET_TZ
from klga_tmax.providers.gribstream.models import GribStreamGoldFeature, GribStreamParsedValue


FEATURE_FAMILY = "gribstream_tmax_thin"
FEATURE_BUILD_VERSION = "TMAX_THIN_V1"
KLGA_GRID_POINT_ID = "GP_KLGA_EXACT"
ENSEMBLE_MODELS = {"gefsatmos", "ifsenfo", "aifsenfo", "aigefssfc"}
SYNOPTIC_MODELS = {
    "gefsatmosmean",
    "gefsatmos",
    "ifsoper",
    "ifsenfo",
    "aifsoper",
    "aifsenfo",
    "aigefssfc",
    "aigfssfc",
}
THRESHOLD_GRID_F = tuple(range(50, 106))
GENERIC_BUCKETS_F = (
    ("lt_60", None, 60),
    ("60_64", 60, 65),
    ("65_69", 65, 70),
    ("70_74", 70, 75),
    ("75_79", 75, 80),
    ("80_84", 80, 85),
    ("85_89", 85, 90),
    ("90_94", 90, 95),
    ("95_99", 95, 100),
    ("ge_100", 100, None),
)


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _temperature_f(value: GribStreamParsedValue) -> float | None:
    raw = _finite(value.value_canonical)
    if raw is None:
        return None
    unit = (value.unit_canonical or value.unit_original or "").lower()
    if "fahrenheit" in unit or unit in {"f", "degf", "degree_fahrenheit"}:
        return raw
    if "celsius" in unit or unit in {"c", "degc", "degree_celsius"}:
        return raw * 9.0 / 5.0 + 32.0
    if "kelvin" in unit or unit == "k" or raw > 150.0:
        return (raw - 273.15) * 9.0 / 5.0 + 32.0
    return raw * 9.0 / 5.0 + 32.0


def _wind_mph(value: GribStreamParsedValue) -> float | None:
    raw = _finite(value.value_canonical)
    if raw is None:
        return None
    unit = (value.unit_canonical or value.unit_original or "").lower()
    if unit in {"mph", "mi/h"} or "mile" in unit:
        return raw
    if unit in {"kt", "knot", "knots"}:
        return raw * 1.15078
    return raw * 2.2369362921


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _valid_label(row: GribStreamParsedValue) -> str:
    valid = row.forecasted_time_utc
    if valid.hour == 18 and valid.date() == row.target_date:
        return "valid_18z"
    if valid.hour == 0 and valid.date() != row.target_date:
        return "valid_00z_nextday"
    return f"valid_{valid.strftime('%Y%m%d_%H%M')}z"


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _meta(values: Iterable[GribStreamParsedValue]) -> dict[str, object]:
    rows = list(values)
    latest_valid = max((row.forecasted_time_utc for row in rows), default=None)
    latest_run = max((row.forecasted_at_utc for row in rows), default=None)
    max_available = max((row.effective_available_at_utc for row in rows), default=None)
    cutoff = rows[0].cutoff_utc if rows else None
    source_age_hours = (
        (cutoff - max_available).total_seconds() / 3600.0
        if cutoff is not None and max_available is not None
        else None
    )
    source_latency_minutes = (
        (max_available - latest_run).total_seconds() / 60.0
        if max_available is not None and latest_run is not None
        else None
    )
    return {
        "latest_valid": latest_valid,
        "latest_run": latest_run,
        "max_available": max_available,
        "source_age_hours": source_age_hours,
        "source_latency_minutes": source_latency_minutes,
        "trace": {
            "row_count": len(rows),
            "grid_point_ids": sorted({row.grid_point_id for row in rows}),
            "members": sorted({row.member for row in rows}),
            "selector_aliases": sorted({row.variable_alias for row in rows}),
            "raw_row_hashes_sample": [row.raw_row_hash for row in rows[:50]],
        },
    }


def _feature(
    *,
    rows: Iterable[GribStreamParsedValue],
    model_id: str,
    target_date,
    cutoff_id: str,
    cutoff_utc: datetime,
    name: str,
    value: float | None,
    unit: str | None,
    extra_trace: dict[str, object] | None = None,
) -> GribStreamGoldFeature:
    source_rows = list(rows)
    meta = _meta(source_rows)
    trace = dict(meta["trace"])
    if extra_trace:
        trace.update(extra_trace)
    rounded = None if value is None else round(float(value), 6)
    return GribStreamGoldFeature(
        target_date=target_date,
        cutoff_id=cutoff_id,
        cutoff_utc=cutoff_utc,
        model_id=model_id,
        feature_family=FEATURE_FAMILY,
        feature_name=name,
        feature_value=rounded,
        feature_unit=unit,
        feature_available=rounded is not None,
        source_latest_valid_time_utc=meta["latest_valid"],
        source_latest_run_time_utc=meta["latest_run"],
        source_age_hours=meta["source_age_hours"],
        source_latency_minutes=meta["source_latency_minutes"],
        max_source_available_at_utc=meta["max_available"],
        source_trace_json=trace,
        feature_build_version=FEATURE_BUILD_VERSION,
    )


def _klga_rows(rows: Iterable[GribStreamParsedValue]) -> list[GribStreamParsedValue]:
    values = list(rows)
    exact = [row for row in values if row.grid_point_id == KLGA_GRID_POINT_ID]
    return exact or values


def _prob_ge_from_samples(samples: list[float], threshold: float) -> float | None:
    if not samples:
        return None
    return sum(1 for value in samples if value >= threshold) / len(samples)


def _bucket_probs_from_survival(survival) -> list[tuple[str, float | None]]:
    probs: list[tuple[str, float | None]] = []
    for label, lower, upper in GENERIC_BUCKETS_F:
        if lower is None and upper is not None:
            prob = 1.0 - survival(float(upper))
        elif lower is not None and upper is None:
            prob = survival(float(lower))
        elif lower is not None and upper is not None:
            prob = survival(float(lower)) - survival(float(upper))
        else:
            prob = None
        probs.append((label, None if prob is None else max(0.0, min(1.0, prob))))
    return probs


def _build_deterministic_temperature(rows: list[GribStreamParsedValue]) -> list[GribStreamGoldFeature]:
    model_id = rows[0].model_id
    target_date = rows[0].target_date
    cutoff_id = rows[0].cutoff_id
    cutoff = rows[0].cutoff_utc
    temp_rows = [row for row in rows if row.variable_alias == "temperature_2m"]
    if not temp_rows:
        return []
    core_rows = _klga_rows(temp_rows)
    core_pairs = [(row, _temperature_f(row)) for row in core_rows]
    core_values = [value for _, value in core_pairs if value is not None]
    features: list[GribStreamGoldFeature] = []
    prefix = f"grib_{model_id}_klga_core"
    unique_valid_times = {row.forecasted_time_utc for row in core_rows}
    if model_id in SYNOPTIC_MODELS and len(unique_valid_times) == 1:
        row, temp_f = core_pairs[0]
        label = _valid_label(row)
        return [
            _feature(
                rows=core_rows,
                model_id=model_id,
                target_date=target_date,
                cutoff_id=cutoff_id,
                cutoff_utc=cutoff,
                name=f"{prefix}_{label}_tmp_2m_f",
                value=temp_f,
                unit="degF",
                extra_trace={"synoptic_split_chunk": True},
            )
        ]
    if core_values:
        max_row, max_value = max(
            ((row, value) for row, value in core_pairs if value is not None),
            key=lambda item: item[1],
        )
        features.extend(
            [
                _feature(rows=core_rows, model_id=model_id, target_date=target_date, cutoff_id=cutoff_id, cutoff_utc=cutoff, name=f"{prefix}_peak_window_max_tmp_2m_f", value=max_value, unit="degF"),
                _feature(rows=core_rows, model_id=model_id, target_date=target_date, cutoff_id=cutoff_id, cutoff_utc=cutoff, name=f"{prefix}_peak_window_mean_tmp_2m_f", value=sum(core_values) / len(core_values), unit="degF"),
                _feature(rows=core_rows, model_id=model_id, target_date=target_date, cutoff_id=cutoff_id, cutoff_utc=cutoff, name=f"{prefix}_time_of_max_local_hour", value=float(max_row.forecasted_time_utc.astimezone(ZoneInfo(TARGET_TZ)).hour), unit="local_hour"),
            ]
        )
        if model_id == "nbm":
            features.append(
                _feature(
                    rows=core_rows,
                    model_id=model_id,
                    target_date=target_date,
                    cutoff_id=cutoff_id,
                    cutoff_utc=cutoff,
                    name="grib_nbm_klga_core_tmp_2m_peak_window_max_f",
                    value=max_value,
                    unit="degF",
                    extra_trace={"fallback_reason": "native_nbm_tmax_empty_in_live_pilot"},
                )
            )
    for row, temp_f in sorted(core_pairs, key=lambda item: item[0].forecasted_time_utc):
        local_hour = row.forecasted_time_utc.astimezone(ZoneInfo(TARGET_TZ)).hour
        features.append(
            _feature(
                rows=[row],
                model_id=model_id,
                target_date=target_date,
                cutoff_id=cutoff_id,
                cutoff_utc=cutoff,
                name=f"{prefix}_tmp_2m_local_{local_hour:02d}_f",
                value=temp_f,
                unit="degF",
            )
        )
    by_grid: dict[str, list[float]] = defaultdict(list)
    grid_rows: dict[str, list[GribStreamParsedValue]] = defaultdict(list)
    for row in temp_rows:
        temp_f = _temperature_f(row)
        if temp_f is None:
            continue
        by_grid[row.grid_point_id].append(temp_f)
        grid_rows[row.grid_point_id].append(row)
    if len(by_grid) > 1 and KLGA_GRID_POINT_ID in by_grid:
        grid_peak = {grid: max(values) for grid, values in by_grid.items()}
        klga_peak = grid_peak[KLGA_GRID_POINT_ID]
        features.append(
            _feature(
                rows=temp_rows,
                model_id=model_id,
                target_date=target_date,
                cutoff_id=cutoff_id,
                cutoff_utc=cutoff,
                name=f"grib_{model_id}_tier_b_peak_spread_tmp_2m_f",
                value=max(grid_peak.values()) - min(grid_peak.values()),
                unit="degF",
            )
        )
        features.append(
            _feature(
                rows=temp_rows,
                model_id=model_id,
                target_date=target_date,
                cutoff_id=cutoff_id,
                cutoff_utc=cutoff,
                name=f"grib_{model_id}_tier_b_max_minus_klga_peak_tmp_2m_f",
                value=max(grid_peak.values()) - klga_peak,
                unit="degF",
            )
        )
        for grid_point_id, peak_value in sorted(grid_peak.items()):
            if grid_point_id == KLGA_GRID_POINT_ID:
                continue
            features.append(
                _feature(
                    rows=grid_rows[grid_point_id],
                    model_id=model_id,
                    target_date=target_date,
                    cutoff_id=cutoff_id,
                    cutoff_utc=cutoff,
                    name=f"grib_{model_id}_{_slug(grid_point_id)}_minus_klga_peak_tmp_2m_f",
                    value=peak_value - klga_peak,
                    unit="degF",
                )
            )
    return features


def _build_native_nbm(rows: list[GribStreamParsedValue]) -> list[GribStreamGoldFeature]:
    model_id = rows[0].model_id
    target_date = rows[0].target_date
    cutoff_id = rows[0].cutoff_id
    cutoff = rows[0].cutoff_utc
    features: list[GribStreamGoldFeature] = []
    aliases = {
        "tmax_2m": ("grib_nbm_klga_core_tmax_2m_f", "degF"),
        "tmax_2m_ens_stddev": ("grib_nbm_klga_core_tmax_ens_std_f", "degF"),
    }
    for alias, (feature_name, unit) in aliases.items():
        alias_rows = _klga_rows([row for row in rows if row.variable_alias == alias])
        if not alias_rows:
            continue
        value = _temperature_f(alias_rows[0])
        features.append(
            _feature(
                rows=alias_rows,
                model_id=model_id,
                target_date=target_date,
                cutoff_id=cutoff_id,
                cutoff_utc=cutoff,
                name=feature_name,
                value=value,
                unit=unit,
            )
        )
    return features or _build_deterministic_temperature(rows)


def _build_rtma(rows: list[GribStreamParsedValue]) -> list[GribStreamGoldFeature]:
    model_id = rows[0].model_id
    target_date = rows[0].target_date
    cutoff_id = rows[0].cutoff_id
    cutoff = rows[0].cutoff_utc
    features: list[GribStreamGoldFeature] = []
    alias_map = {
        "temperature_2m": ("grib_rtma_klga_core_current_tmp_2m_f", "degF", _temperature_f),
        "dew_point_2m": ("grib_rtma_klga_core_current_dewpoint_2m_f", "degF", _temperature_f),
        "wind_speed_10m": ("grib_rtma_klga_core_current_wind_speed_10m_mph", "mph", _wind_mph),
        "wind_gust": ("grib_rtma_klga_core_current_wind_gust_mph", "mph", _wind_mph),
    }
    for alias, (feature_name, unit, converter) in alias_map.items():
        alias_rows = sorted(_klga_rows([row for row in rows if row.variable_alias == alias]), key=lambda row: row.forecasted_time_utc)
        if not alias_rows:
            continue
        row = alias_rows[-1]
        features.append(
            _feature(
                rows=[row],
                model_id=model_id,
                target_date=target_date,
                cutoff_id=cutoff_id,
                cutoff_utc=cutoff,
                name=feature_name,
                value=converter(row),
                unit=unit,
            )
        )
    return features


def _build_ensemble(rows: list[GribStreamParsedValue]) -> list[GribStreamGoldFeature]:
    model_id = rows[0].model_id
    target_date = rows[0].target_date
    cutoff_id = rows[0].cutoff_id
    cutoff = rows[0].cutoff_utc
    temp_rows = _klga_rows([row for row in rows if row.variable_alias == "temperature_2m"])
    by_member: dict[str, list[tuple[GribStreamParsedValue, float]]] = defaultdict(list)
    for row in temp_rows:
        temp_f = _temperature_f(row)
        if temp_f is not None:
            by_member[row.member].append((row, temp_f))
    member_peak: dict[str, tuple[list[GribStreamParsedValue], float]] = {}
    for member, pairs in by_member.items():
        member_peak[member] = ([row for row, _ in pairs], max(value for _, value in pairs))
    samples = [value for _, value in member_peak.values()]
    if not samples:
        return []
    prefix = f"grib_{model_id}_klga_core"
    features: list[GribStreamGoldFeature] = []
    stat_rows = [row for rows_for_member, _ in member_peak.values() for row in rows_for_member]
    unique_valid_times = {row.forecasted_time_utc for row in temp_rows}
    if model_id in SYNOPTIC_MODELS and len(unique_valid_times) == 1:
        label = _valid_label(temp_rows[0])
        split_prefix = f"{prefix}_{label}"
        for member, (member_rows, value) in sorted(
            member_peak.items(),
            key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0]),
        ):
            features.append(
                _feature(
                    rows=member_rows,
                    model_id=model_id,
                    target_date=target_date,
                    cutoff_id=cutoff_id,
                    cutoff_utc=cutoff,
                    name=f"{split_prefix}_member_{member}_tmp_2m_f",
                    value=value,
                    unit="degF",
                    extra_trace={"synoptic_split_chunk": True},
                )
            )
        stats = {
            "member_count": float(len(samples)),
            "tmp_2m_mean_f": sum(samples) / len(samples),
            "tmp_2m_median_f": median(samples),
            "tmp_2m_std_f": _std(samples),
            "tmp_2m_p05_f": _percentile(samples, 0.05),
            "tmp_2m_p10_f": _percentile(samples, 0.10),
            "tmp_2m_p25_f": _percentile(samples, 0.25),
            "tmp_2m_p75_f": _percentile(samples, 0.75),
            "tmp_2m_p90_f": _percentile(samples, 0.90),
            "tmp_2m_p95_f": _percentile(samples, 0.95),
        }
        for suffix, value in stats.items():
            unit = "count" if suffix == "member_count" else "degF"
            features.append(
                _feature(
                    rows=stat_rows,
                    model_id=model_id,
                    target_date=target_date,
                    cutoff_id=cutoff_id,
                    cutoff_utc=cutoff,
                    name=f"{split_prefix}_{suffix}",
                    value=value,
                    unit=unit,
                    extra_trace={"synoptic_split_chunk": True},
                )
            )
        for threshold in THRESHOLD_GRID_F:
            features.append(
                _feature(
                    rows=stat_rows,
                    model_id=model_id,
                    target_date=target_date,
                    cutoff_id=cutoff_id,
                    cutoff_utc=cutoff,
                    name=f"{split_prefix}_prob_tmp_2m_ge_{threshold}f",
                    value=_prob_ge_from_samples(samples, float(threshold)),
                    unit="probability",
                    extra_trace={"synoptic_split_chunk": True},
                )
            )
        return features
    stats = {
        "member_count": float(len(samples)),
        "tmax_proxy_mean_f": sum(samples) / len(samples),
        "tmax_proxy_median_f": median(samples),
        "tmax_proxy_std_f": _std(samples),
        "tmax_proxy_p05_f": _percentile(samples, 0.05),
        "tmax_proxy_p10_f": _percentile(samples, 0.10),
        "tmax_proxy_p25_f": _percentile(samples, 0.25),
        "tmax_proxy_p75_f": _percentile(samples, 0.75),
        "tmax_proxy_p90_f": _percentile(samples, 0.90),
        "tmax_proxy_p95_f": _percentile(samples, 0.95),
    }
    for suffix, value in stats.items():
        unit = "count" if suffix == "member_count" else "degF"
        features.append(
            _feature(
                rows=stat_rows,
                model_id=model_id,
                target_date=target_date,
                cutoff_id=cutoff_id,
                cutoff_utc=cutoff,
                name=f"{prefix}_{suffix}",
                value=value,
                unit=unit,
            )
        )
    for member, (member_rows, value) in sorted(
        member_peak.items(),
        key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0]),
    ):
        features.append(
            _feature(
                rows=member_rows,
                model_id=model_id,
                target_date=target_date,
                cutoff_id=cutoff_id,
                cutoff_utc=cutoff,
                name=f"{prefix}_member_{member}_tmax_proxy_f",
                value=value,
                unit="degF",
            )
        )
    for threshold in THRESHOLD_GRID_F:
        features.append(
            _feature(
                rows=stat_rows,
                model_id=model_id,
                target_date=target_date,
                cutoff_id=cutoff_id,
                cutoff_utc=cutoff,
                name=f"{prefix}_prob_tmax_ge_{threshold}f",
                value=_prob_ge_from_samples(samples, float(threshold)),
                unit="probability",
            )
        )
    for label, prob in _bucket_probs_from_survival(lambda threshold: _prob_ge_from_samples(samples, threshold)):
        features.append(
            _feature(
                rows=stat_rows,
                model_id=model_id,
                target_date=target_date,
                cutoff_id=cutoff_id,
                cutoff_utc=cutoff,
                name=f"{prefix}_generic_bucket_prob_{label}",
                value=prob,
                unit="probability",
                extra_trace={"bucket_family": "generic_5f_research_buckets_not_market_contract"},
            )
        )
    return features


def _parse_percentile_alias(alias: str) -> int | None:
    match = re.fullmatch(r"tmp_max18_p(\d{2})", alias)
    return int(match.group(1)) if match else None


def _survival_from_percentiles(percentiles: dict[int, float]):
    points = sorted((temp, pct / 100.0) for pct, temp in percentiles.items())

    def survival(threshold: float) -> float | None:
        if not points:
            return None
        if threshold <= points[0][0]:
            return 1.0
        if threshold >= points[-1][0]:
            return 0.0
        for (lower_temp, lower_q), (upper_temp, upper_q) in zip(points, points[1:]):
            if lower_temp <= threshold <= upper_temp:
                if upper_temp == lower_temp:
                    cdf = upper_q
                else:
                    cdf = lower_q + (upper_q - lower_q) * ((threshold - lower_temp) / (upper_temp - lower_temp))
                return max(0.0, min(1.0, 1.0 - cdf))
        return None

    return survival


def _build_nbmqmd(rows: list[GribStreamParsedValue]) -> list[GribStreamGoldFeature]:
    model_id = rows[0].model_id
    target_date = rows[0].target_date
    cutoff_id = rows[0].cutoff_id
    cutoff = rows[0].cutoff_utc
    percentile_rows = _klga_rows([row for row in rows if _parse_percentile_alias(row.variable_alias) is not None])
    by_percentile: dict[int, tuple[GribStreamParsedValue, float]] = {}
    for row in percentile_rows:
        percentile = _parse_percentile_alias(row.variable_alias)
        temp_f = _temperature_f(row)
        if percentile is not None and temp_f is not None:
            by_percentile[percentile] = (row, temp_f)
    if not by_percentile:
        return []
    prefix = "grib_nbmqmd_klga_core"
    features: list[GribStreamGoldFeature] = []
    for percentile, (row, value) in sorted(by_percentile.items()):
        features.append(
            _feature(
                rows=[row],
                model_id=model_id,
                target_date=target_date,
                cutoff_id=cutoff_id,
                cutoff_utc=cutoff,
                name=f"{prefix}_tmp_max18_p{percentile:02d}_f",
                value=value,
                unit="degF",
            )
        )
    percentiles = {percentile: value for percentile, (_, value) in by_percentile.items()}
    p10 = percentiles.get(10)
    p50 = percentiles.get(50)
    p90 = percentiles.get(90)
    stat_rows = [row for row, _ in by_percentile.values()]
    for suffix, value in {
        "mean_proxy_f": p50,
        "spread_p90_minus_p10_f": None if p10 is None or p90 is None else p90 - p10,
        "skew_proxy_p90_p10_p50_f": None if p10 is None or p50 is None or p90 is None else p90 + p10 - 2.0 * p50,
    }.items():
        features.append(
            _feature(
                rows=stat_rows,
                model_id=model_id,
                target_date=target_date,
                cutoff_id=cutoff_id,
                cutoff_utc=cutoff,
                name=f"{prefix}_{suffix}",
                value=value,
                unit="degF",
            )
        )
    survival = _survival_from_percentiles(percentiles)
    for threshold in THRESHOLD_GRID_F:
        features.append(
            _feature(
                rows=stat_rows,
                model_id=model_id,
                target_date=target_date,
                cutoff_id=cutoff_id,
                cutoff_utc=cutoff,
                name=f"{prefix}_prob_tmax_ge_{threshold}f",
                value=survival(float(threshold)),
                unit="probability",
            )
        )
    for label, prob in _bucket_probs_from_survival(survival):
        features.append(
            _feature(
                rows=stat_rows,
                model_id=model_id,
                target_date=target_date,
                cutoff_id=cutoff_id,
                cutoff_utc=cutoff,
                name=f"{prefix}_generic_bucket_prob_{label}",
                value=prob,
                unit="probability",
                extra_trace={"bucket_family": "generic_5f_research_buckets_not_market_contract"},
            )
        )
    return features


def build_tmax_thin_gold_features(values: tuple[GribStreamParsedValue, ...]) -> tuple[GribStreamGoldFeature, ...]:
    grouped: dict[tuple[str, object, str], list[GribStreamParsedValue]] = defaultdict(list)
    for value in values:
        grouped[(value.model_id, value.target_date, value.cutoff_id)].append(value)

    features: list[GribStreamGoldFeature] = []
    for (model_id, _target_date, _cutoff_id), rows in sorted(grouped.items(), key=lambda item: item[0]):
        if not rows:
            continue
        if model_id == "nbmqmd":
            features.extend(_build_nbmqmd(rows))
        elif model_id == "rtma":
            features.extend(_build_rtma(rows))
        elif model_id == "nbm" and any(row.variable_alias == "tmax_2m" for row in rows):
            features.extend(_build_native_nbm(rows))
        elif model_id in ENSEMBLE_MODELS:
            features.extend(_build_ensemble(rows))
        else:
            features.extend(_build_deterministic_temperature(rows))
    return tuple(features)
